#!/usr/bin/env python3
"""
lm - an easy human wrapper around llama.cpp / llama-server.

Edit the two paths below and the wrapper handles the rest:
    1. MODEL_DIR
    2. LLAMA_CPP_DIR

Typical usage:
    lm list
    lm doctor
    lm run gemma4
    lm status
    lm chat gemma4
    lm logs --follow
    lm restart gemma4 --context 131072
    lm stop

No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Put your GGUF models here.
MODEL_DIR = Path("/workbook/models")

# Point this at the root of your llama.cpp checkout.
LLAMA_CPP_DIR = Path("/workspace/llama.cpp")

# The wrapper automatically finds llama-server under common llama.cpp build
# layouts (build/bin, build-*/bin, build/Release/bin, etc.).

# The wrapper keeps runtime state here automatically.
STATE_DIR = MODEL_DIR.parent / ".llama-manager"

# Default llama-server settings.
DEFAULTS: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8080,
    "threads": 24,
    "batch_threads": 24,
    "context": 524288,
    "output": 32768,
    "parallel": 1,
    "jinja": True,
    "extra_args": [],
}

CONFIG_FILE = STATE_DIR / "config.json"
PID_FILE = STATE_DIR / "server.pid"
META_FILE = STATE_DIR / "server.json"
LOG_FILE = STATE_DIR / "server.log"

SHARD_PATTERN = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


# ============================================================================
# GENERAL HELPERS
# ============================================================================


def die(message: str, code: int = 1) -> None:
    raise SystemExit(message if code else 0)


def human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def first_not_none(*values: Any) -> Any:
    """Return the first value that is not None; preserve 0/False/empty values."""
    for value in values:
        if value is not None:
            return value
    return None


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return copy.deepcopy(default)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def shell_join(command: list[str]) -> str:
    """Readable command line for logs/messages without requiring shell quoting."""
    import shlex

    return shlex.join(command)


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def server_binary() -> Path:
    """Find llama-server without requiring users to know the build folder."""
    preferred = [
        LLAMA_CPP_DIR / "build-ivybridge/bin/llama-server",
        LLAMA_CPP_DIR / "build/bin/llama-server",
        LLAMA_CPP_DIR / "build/Release/bin/llama-server",
    ]

    for candidate in preferred:
        if candidate.is_file():
            return candidate

    for candidate in sorted(LLAMA_CPP_DIR.glob("build-*/bin/llama-server")):
        if candidate.is_file():
            return candidate

    for candidate in sorted(LLAMA_CPP_DIR.glob("*/bin/llama-server")):
        if candidate.is_file():
            return candidate

    # Useful error path when nothing exists yet.
    return LLAMA_CPP_DIR / "build/bin/llama-server"


# ============================================================================
# CONFIGURATION
# ============================================================================


def default_config() -> dict[str, Any]:
    return {
        "defaults": copy.deepcopy(DEFAULTS),
        "models": {},
    }


def load_config() -> dict[str, Any]:
    ensure_state_dir()

    if not CONFIG_FILE.exists():
        config = default_config()
        write_json(CONFIG_FILE, config)
        return config

    config = read_json(CONFIG_FILE, default_config())
    if not isinstance(config, dict):
        die(f"Invalid config file: {CONFIG_FILE}")

    defaults = copy.deepcopy(DEFAULTS)
    user_defaults = config.get("defaults", {})
    if isinstance(user_defaults, dict):
        defaults.update(user_defaults)
    else:
        die("Config error: 'defaults' must be an object.")

    models = config.get("models", {})
    if not isinstance(models, dict):
        die("Config error: 'models' must be an object.")

    config["defaults"] = defaults
    config["models"] = models
    return config


# ============================================================================
# MODEL DISCOVERY
# ============================================================================


def discover_models() -> list[dict[str, Any]]:
    """Return one record per model, grouping split GGUF shards together."""
    if not MODEL_DIR.exists():
        return []

    files = sorted(MODEL_DIR.glob("*.gguf"), key=lambda p: p.name.lower())
    singles: list[dict[str, Any]] = []
    shards: dict[str, list[Path]] = {}

    for path in files:
        match = SHARD_PATTERN.match(path.name)
        if match:
            key = match.group(1)
            shards.setdefault(key, []).append(path)
        else:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            singles.append(
                {
                    "name": path.stem,
                    "file": path.name,
                    "size": size,
                    "shards": [path.name],
                }
            )

    for name, shard_files in shards.items():
        shard_files.sort(key=lambda p: p.name.lower())
        total_size = 0
        for path in shard_files:
            try:
                total_size += path.stat().st_size
            except OSError:
                pass

        singles.append(
            {
                "name": name,
                "file": shard_files[0].name,
                "size": total_size,
                "shards": [p.name for p in shard_files],
            }
        )

    return sorted(singles, key=lambda item: item["name"].lower())


def model_aliases(config: dict[str, Any], filename: str) -> list[str]:
    aliases = []
    for alias, model in config.get("models", {}).items():
        if isinstance(model, dict) and model.get("file") == filename:
            aliases.append(alias)
    return sorted(aliases, key=str.lower)


def resolve_model(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Resolve alias, exact filename, exact stem, or unique substring match."""
    registry = config.get("models", {})

    # 1. Explicit configured alias.
    configured = registry.get(name)
    if isinstance(configured, dict):
        model = dict(configured)
        model.setdefault("alias", name)
        model.setdefault("name", name)
        if not model.get("file"):
            die(f"Configured model '{name}' has no 'file'.")
        return model

    models = discover_models()
    needle = name.lower()

    # 2. Exact model filename / stem / discovered name.
    exact = []
    for item in models:
        if needle in {
            item["name"].lower(),
            item["file"].lower(),
            Path(item["file"]).stem.lower(),
        }:
            exact.append(item)

    if len(exact) == 1:
        item = exact[0]
        return {
            "alias": name,
            "name": item["name"],
            "file": item["file"],
        }

    # 3. Unique substring match.
    candidates = [
        item
        for item in models
        if needle in item["name"].lower()
        or needle in item["file"].lower()
    ]

    if len(candidates) == 1:
        item = candidates[0]
        return {
            "alias": name,
            "name": item["name"],
            "file": item["file"],
        }

    if len(candidates) > 1:
        names = ", ".join(item["name"] for item in candidates)
        die(f"Ambiguous model '{name}'. Matches: {names}")

    die(f"Model '{name}' not found. Run: lm list")
    return {}  # unreachable; keeps type checkers happy.


def model_path(model: dict[str, Any]) -> Path:
    path = Path(str(model["file"])).expanduser()
    if not path.is_absolute():
        path = MODEL_DIR / path
    return path


def mmproj_path(model: dict[str, Any]) -> Path | None:
    value = model.get("mmproj")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = MODEL_DIR / path
    return path


def cmd_list(config: dict[str, Any]) -> None:
    models = discover_models()

    print(f"Models: {MODEL_DIR}")
    print()

    if not models:
        print("No .gguf models found.")
        return

    print(f"{'#':>2}  {'MODEL':<58} {'SIZE':>10}  ALIAS")
    print("-" * 90)

    for index, item in enumerate(models, 1):
        aliases = ", ".join(model_aliases(config, item["file"])) or "-"
        print(
            f"{index:2}  "
            f"{item['name'][:58]:<58} "
            f"{human_size(item['size']):>10}  "
            f"{aliases}"
        )


# ============================================================================
# PROCESS / STATE
# ============================================================================


def read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def process_alive(process_id: int | None) -> bool:
    if not process_id or process_id <= 0:
        return False

    try:
        os.kill(process_id, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def process_command(process_id: int) -> str | None:
    """Read Linux /proc command line when available."""
    proc = Path(f"/proc/{process_id}/cmdline")
    try:
        raw = proc.read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None

    parts = [x for x in raw.split(b"\0") if x]
    return " ".join(part.decode("utf-8", errors="replace") for part in parts)


def managed_process_alive() -> bool:
    process_id = read_pid()
    if not process_alive(process_id):
        return False

    # On Linux, protect against PID reuse.
    command = process_command(process_id)
    if command is not None:
        expected = str(server_binary())
        if expected not in command:
            return False

    return True


def load_metadata() -> dict[str, Any]:
    meta = read_json(META_FILE, {})
    return meta if isinstance(meta, dict) else {}


def cleanup_stale_pid() -> None:
    process_id = read_pid()
    if process_id is not None and not managed_process_alive():
        PID_FILE.unlink(missing_ok=True)
        META_FILE.unlink(missing_ok=True)


def kill_managed_process(process_id: int, sig: int) -> None:
    """Terminate the process group created by start_new_session=True."""
    try:
        os.killpg(process_id, sig)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        os.kill(process_id, sig)


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


# ============================================================================
# SERVER ARGUMENTS
# ============================================================================


def value(args: argparse.Namespace, name: str) -> Any:
    return getattr(args, name, None)


def build_server_command(
    config: dict[str, Any],
    model: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    defaults = config["defaults"]

    host = first_not_none(value(args, "host"), model.get("host"), defaults.get("host"))
    port = first_not_none(value(args, "port"), model.get("port"), defaults.get("port"))
    threads = first_not_none(value(args, "threads"), model.get("threads"), defaults.get("threads"))
    batch_threads = first_not_none(
        value(args, "batch_threads"),
        model.get("batch_threads"),
        defaults.get("batch_threads"),
    )
    context = first_not_none(value(args, "context"), model.get("context"), defaults.get("context"))
    output = first_not_none(value(args, "output"), model.get("output"), defaults.get("output"))
    parallel = first_not_none(value(args, "parallel"), model.get("parallel"), defaults.get("parallel"))
    jinja = first_not_none(value(args, "jinja"), model.get("jinja"), defaults.get("jinja", True))

    binary = server_binary()
    path = model_path(model)

    command = [
        str(binary),
        "-m", str(path),
        "--host", str(host),
        "--port", str(port),
        "-t", str(threads),
        "-tb", str(batch_threads),
        "-c", str(context),
        "-n", str(output),
        "-np", str(parallel),
    ]

    if jinja:
        command.append("--jinja")
    else:
        command.append("--no-jinja")

    mmproj = mmproj_path(model)
    if mmproj:
        command.extend(["--mmproj", str(mmproj)])

    # Advanced llama-server options stay here rather than being mixed into lm's
    # own argparse namespace. This makes the wrapper safe from option collisions.
    extra = list(defaults.get("extra_args", []))
    extra.extend(model.get("extra_args", []))
    command.extend(str(x) for x in extra)

    effective = {
        "host": host,
        "port": int(port),
        "threads": threads,
        "batch_threads": batch_threads,
        "context": context,
        "output": output,
        "parallel": parallel,
        "jinja": bool(jinja),
    }

    return command, effective


# ============================================================================
# API
# ============================================================================


def api_host_port(config: dict[str, Any]) -> tuple[str, int]:
    meta = load_metadata()
    host = str(meta.get("host", config["defaults"]["host"]))
    port = int(meta.get("port", config["defaults"]["port"]))
    return host, port


def api_base(config: dict[str, Any]) -> str:
    host, port = api_host_port(config)
    return f"http://{host}:{port}"


def api_request(
    config: dict[str, Any],
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 10,
    accept: str | None = None,
):
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    if accept:
        headers["Accept"] = accept

    request = urllib.request.Request(
        api_base(config) + path,
        data=data,
        headers=headers,
        method=method,
    )
    return urllib.request.urlopen(request, timeout=timeout)


def api_ready(config: dict[str, Any]) -> bool:
    try:
        with api_request(config, "/health", timeout=2) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_api(config: dict[str, Any], process_id: int, timeout: int = 300) -> bool:
    deadline = time.monotonic() + timeout
    last_notice = 0.0

    while time.monotonic() < deadline:
        if not process_alive(process_id):
            return False

        if api_ready(config):
            return True

        now = time.monotonic()
        if now - last_notice >= 5:
            print("Waiting for llama-server...", flush=True)
            last_notice = now

        time.sleep(1)

    return False


def get_model_id(config: dict[str, Any]) -> str | None:
    try:
        with api_request(config, "/v1/models", timeout=10) as response:
            obj = json.loads(response.read().decode("utf-8"))
        data = obj.get("data") or []
        if data:
            return data[0].get("id")
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
    ):
        pass
    return None


# ============================================================================
# RUN / STOP / RESTART / STATUS
# ============================================================================


def cmd_run(config: dict[str, Any], name: str, args: argparse.Namespace) -> None:
    ensure_state_dir()
    cleanup_stale_pid()

    if managed_process_alive():
        process_id = read_pid()
        meta = load_metadata()
        die(
            f"Already running: {meta.get('name', 'unknown model')} "
            f"(PID {process_id}). Use: lm stop or lm restart."
        )

    model = resolve_model(config, name)
    path = model_path(model)
    binary = server_binary()

    if not MODEL_DIR.exists():
        die(f"Model directory not found: {MODEL_DIR}")

    if not path.exists():
        die(f"Model not found: {path}")

    if not binary.exists():
        die(f"llama-server not found: {binary}")

    if not os.access(binary, os.X_OK):
        die(f"llama-server is not executable: {binary}")

    command, effective = build_server_command(config, model, args)

    host = str(effective["host"])
    port = int(effective["port"])

    if port_open(host, port):
        die(
            f"Port {port} on {host} is already in use. "
            "Stop the other server or choose another --port."
        )

    started = time.strftime("%Y-%m-%d %H:%M:%S")

    with LOG_FILE.open("ab") as log:
        log.write(
            (
                f"\n=== START {started} ===\n"
                f"MODEL: {model.get('name', name)}\n"
                f"COMMAND: {shell_join(command)}\n"
            ).encode("utf-8", errors="replace")
        )
        log.flush()

        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            cwd=str(LLAMA_CPP_DIR),
        )

    PID_FILE.write_text(f"{process.pid}\n", encoding="utf-8")

    meta = {
        "pid": process.pid,
        "alias": model.get("alias", name),
        "name": model.get("name", name),
        "file": str(path.relative_to(MODEL_DIR)) if path.is_relative_to(MODEL_DIR) else str(path),
        "host": host,
        "port": port,
        "started": started,
        "command": command,
        "context": effective["context"],
        "output": effective["output"],
        "parallel": effective["parallel"],
        "jinja": effective["jinja"],
        "mmproj": str(mmproj_path(model)) if mmproj_path(model) else None,
    }
    write_json(META_FILE, meta)

    print(
        f"Started {meta['name']} (PID {process.pid}) -> "
        f"{api_base(config)}/v1"
    )


def cmd_stop(config: dict[str, Any], force: bool = False) -> None:
    del config  # Kept in signature for a consistent command interface.

    cleanup_stale_pid()
    process_id = read_pid()

    if not process_id or not managed_process_alive():
        PID_FILE.unlink(missing_ok=True)
        META_FILE.unlink(missing_ok=True)
        print("No managed llama-server is running.")
        return

    print(f"Stopping llama-server (PID {process_id})...")
    kill_managed_process(process_id, signal.SIGTERM)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and process_alive(process_id):
        time.sleep(0.2)

    if process_alive(process_id) and force:
        print("Server did not stop; sending SIGKILL...")
        kill_managed_process(process_id, signal.SIGKILL)

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and process_alive(process_id):
            time.sleep(0.1)

    if process_alive(process_id):
        print("Server is still running.")
        return

    PID_FILE.unlink(missing_ok=True)
    META_FILE.unlink(missing_ok=True)
    print("Stopped.")


def ensure_requested_model_running(
    config: dict[str, Any],
    name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Ensure exactly the requested model is active and ready."""
    cleanup_stale_pid()
    requested = resolve_model(config, name)
    requested_file = str(model_path(requested).resolve())

    if managed_process_alive():
        meta = load_metadata()
        active_file = meta.get("file")

        if active_file:
            active_path = Path(str(active_file))
            if not active_path.is_absolute():
                active_path = MODEL_DIR / active_path
            active_file = str(active_path.resolve())

        if active_file != requested_file:
            die(
                f"A different model is already running: "
                f"{meta.get('name', active_file or 'unknown')}.\n"
                f"Use: lm restart {name}"
            )

        process_id = read_pid()
        if not api_ready(config):
            print("Server is running but the model is still loading...")
            if not process_id or not wait_for_api(config, process_id, timeout=300):
                die("Server did not become ready. Run: lm logs")

        return requested

    print(f"Server is not running. Starting {requested.get('name', name)}...")
    start_args = argparse.Namespace(
        host=value(args, "host"),
        port=value(args, "port"),
        threads=value(args, "threads"),
        batch_threads=value(args, "batch_threads"),
        context=value(args, "context"),
        output=value(args, "output"),
        parallel=value(args, "parallel"),
        jinja=value(args, "jinja"),
    )
    cmd_run(config, name, start_args)

    process_id = read_pid()
    if not process_id:
        die("Server was started but no PID was recorded.")

    print("Waiting for llama-server to become ready...")
    if not wait_for_api(config, process_id, timeout=300):
        die("Server did not become ready. Run: lm logs")

    return requested


def cmd_restart(config: dict[str, Any], args: argparse.Namespace) -> None:
    cleanup_stale_pid()
    old = load_metadata()

    name = args.model or old.get("alias") or old.get("name")
    if not name:
        die("restart requires a model name on first use")

    cmd_stop(config, force=args.force)

    restart_args = argparse.Namespace(
        host=args.host,
        port=args.port,
        threads=args.threads,
        batch_threads=args.batch_threads,
        context=args.context,
        output=args.output,
        parallel=args.parallel,
        jinja=args.jinja,
    )
    cmd_run(config, name, restart_args)

    process_id = read_pid()
    if process_id:
        print("Waiting for llama-server to become ready...")
        if not wait_for_api(config, process_id, timeout=300):
            die("Server did not become ready. Run: lm logs")


def cmd_status(config: dict[str, Any]) -> None:
    cleanup_stale_pid()
    process_id = read_pid()
    meta = load_metadata()

    process_running = managed_process_alive()
    host, port = api_host_port(config)
    socket_open = port_open(host, port)

    if process_running and socket_open:
        api_state = "READY" if api_ready(config) else "LOADING"
    elif process_running:
        api_state = "STARTING"
    else:
        api_state = "DOWN"

    print(f"process : {'RUNNING' if process_running else 'STOPPED'}")
    print(f"pid     : {process_id or '-'}")
    print(f"socket  : {'OPEN' if socket_open else 'CLOSED'}")
    print(f"api     : {api_state}")
    print(f"endpoint: {api_base(config)}/v1")

    if meta and process_running:
        print(f"model   : {meta.get('name', '-')}")
        print(f"file    : {meta.get('file', '-')}")
        print(f"context : {meta.get('context', '-')}")
        print(f"output  : {meta.get('output', '-')}")
        print(f"parallel: {meta.get('parallel', '-')}")


def cmd_doctor(config: dict[str, Any]) -> None:
    binary = server_binary()

    print("llama.cpp wrapper doctor")
    print("=" * 72)
    print(f"model dir   : {MODEL_DIR}")
    print(f"             {'OK' if MODEL_DIR.exists() else 'MISSING'}")
    print(f"llama.cpp   : {LLAMA_CPP_DIR}")
    print(f"             {'OK' if LLAMA_CPP_DIR.exists() else 'MISSING'}")
    print(f"server      : {binary}")
    print(f"             {'OK' if binary.exists() else 'MISSING'}")
    print(f"state dir   : {STATE_DIR}")
    print(f"config      : {CONFIG_FILE}")

    if MODEL_DIR.exists():
        models = discover_models()
        print(f"models      : {len(models)}")

    if binary.exists():
        try:
            result = subprocess.run(
                [str(binary), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(LLAMA_CPP_DIR),
            )
            version = (result.stdout or result.stderr).strip().splitlines()
            if version:
                print(f"version     : {version[0]}")
        except (OSError, subprocess.SubprocessError):
            pass

    print(f"python      : {sys.version.split()[0]}")
    print(f"config file : {CONFIG_FILE}")
    print()
    print("Advanced llama-server flags belong in config.json under:")
    print("  defaults.extra_args")
    print("  models.<alias>.extra_args")


def cmd_logs(follow: bool, lines: int) -> None:
    if not LOG_FILE.exists():
        print(f"No log file yet: {LOG_FILE}")
        return

    if not follow:
        from collections import deque

        with LOG_FILE.open("r", encoding="utf-8", errors="replace") as handle:
            tail = deque(handle, maxlen=max(1, lines))
        print("".join(tail), end="")
        return

    print(f"Following {LOG_FILE} (Ctrl-C to stop)")
    with LOG_FILE.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if line:
                print(line, end="")
                continue
            time.sleep(0.25)


def cmd_config() -> None:
    print(CONFIG_FILE)
    if CONFIG_FILE.exists():
        print()
        print(CONFIG_FILE.read_text(encoding="utf-8"))
    else:
        print("Config will be created automatically on first run.")


# ============================================================================
# CHAT
# ============================================================================


def chat_request(
    config: dict[str, Any],
    payload: dict[str, Any],
    timeout: int = 3600,
):
    return api_request(
        config,
        "/v1/chat/completions",
        method="POST",
        payload=payload,
        timeout=timeout,
        accept="text/event-stream",
    )


def save_history(path: Path, model: dict[str, Any], model_id: str, history: list[dict[str, Any]]) -> None:
    path = path.expanduser()
    if not path.name:
        die("Usage: /save FILE")

    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "model": model.get("name"),
            "model_id": model_id,
            "messages": history,
        },
    )
    print(f"Saved {path}")


def cmd_chat(config: dict[str, Any], name: str, args: argparse.Namespace) -> None:
    requested = ensure_requested_model_running(config, name, args)
    model_id = get_model_id(config) or str(model_path(requested))
    output = first_not_none(args.output, requested.get("output"), config["defaults"]["output"])

    history: list[dict[str, str]] = []

    print()
    print(f"llama chat :: {requested.get('name', name)}")
    print(f"API       :: {api_base(config)}/v1")
    print(f"Model ID  :: {model_id}")
    print("Commands  :: /help /clear /model /status /save FILE /quit")
    print("Ctrl-D or /quit exits.")
    print()

    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user.strip():
            continue

        command = user.strip()

        if command in {"/quit", "/exit", "/q"}:
            break

        if command == "/help":
            print("/help                 Show commands")
            print("/clear                Clear conversation history")
            print("/model                Show current model")
            print("/status               Show server status")
            print("/save FILE            Save conversation as JSON")
            print("/quit                 Exit chat")
            continue

        if command == "/clear":
            history.clear()
            print("Conversation cleared.")
            continue

        if command == "/model":
            print(requested.get("name", name))
            print(model_id)
            continue

        if command == "/status":
            cmd_status(config)
            continue

        if command.startswith("/save "):
            save_history(
                Path(command[6:].strip()),
                requested,
                model_id,
                history,
            )
            continue

        history.append({"role": "user", "content": user})

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": history,
            "max_tokens": output,
            "stream": True,
        }

        if args.temperature is not None:
            payload["temperature"] = args.temperature
        if args.top_p is not None:
            payload["top_p"] = args.top_p
        if args.top_k is not None:
            payload["top_k"] = args.top_k
        if args.seed is not None:
            payload["seed"] = args.seed

        print("Assistant: ", end="", flush=True)
        full_content: list[str] = []
        reasoning: list[str] = []

        try:
            with chat_request(config, payload) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = obj.get("choices") or []
                    if not choices:
                        continue

                    delta = choices[0].get("delta") or {}
                    reasoning_content = delta.get("reasoning_content")
                    content = delta.get("content")

                    if reasoning_content:
                        reasoning.append(reasoning_content)

                    if content:
                        print(content, end="", flush=True)
                        full_content.append(content)

            print()

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"\n[chat error] HTTP {exc.code}: {detail}")
            history.pop()
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"\n[chat error] {exc}")
            history.pop()
            continue

        answer = "".join(full_content)

        if answer:
            history.append({"role": "assistant", "content": answer})
        elif reasoning:
            print(
                "[No final answer was emitted; output may have been exhausted "
                "during reasoning.]"
            )


# ============================================================================
# ARGPARSE
# ============================================================================


def add_server_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--batch-threads", dest="batch_threads", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--output", type=int, help="Server max output tokens (-n)")
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--jinja", dest="jinja", action="store_true", default=None)
    parser.add_argument("--no-jinja", dest="jinja", action="store_false")


def add_common_run_options(parser: argparse.ArgumentParser) -> None:
    add_server_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lm",
        description="Easy human wrapper around llama.cpp / llama-server.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List GGUF models")
    sub.add_parser("status", help="Show server/process/API status")
    sub.add_parser("doctor", help="Check paths, model directory, and server")
    sub.add_parser("config", help="Show the wrapper configuration file")

    run = sub.add_parser("run", help="Start a model")
    run.add_argument("model")
    add_common_run_options(run)

    stop = sub.add_parser("stop", help="Stop the managed server")
    stop.add_argument("--force", action="store_true")

    restart = sub.add_parser("restart", help="Restart the selected/current model")
    restart.add_argument("model", nargs="?")
    restart.add_argument("--force", action="store_true")
    add_common_run_options(restart)

    logs = sub.add_parser("logs", help="Show server logs")
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--lines", type=int, default=200)

    chat = sub.add_parser("chat", help="Interactive OpenAI-compatible chat")
    chat.add_argument("model")
    chat.add_argument("--output", type=int, help="Max tokens per response")
    chat.add_argument("--context", type=int, help="Context size when auto-starting")
    chat.add_argument("--host")
    chat.add_argument("--port", type=int)
    chat.add_argument("--threads", type=int)
    chat.add_argument("--batch-threads", dest="batch_threads", type=int)
    chat.add_argument("--parallel", type=int)
    chat.add_argument("--jinja", dest="jinja", action="store_true", default=None)
    chat.add_argument("--no-jinja", dest="jinja", action="store_false")
    chat.add_argument("--temperature", type=float)
    chat.add_argument("--top-p", dest="top_p", type=float)
    chat.add_argument("--top-k", dest="top_k", type=int)
    chat.add_argument("--seed", type=int)

    return parser


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config()

    commands = {
        "list": lambda: cmd_list(config),
        "status": lambda: cmd_status(config),
        "doctor": lambda: cmd_doctor(config),
        "config": cmd_config,
        "run": lambda: cmd_run(config, args.model, args),
        "stop": lambda: cmd_stop(config, args.force),
        "restart": lambda: cmd_restart(config, args),
        "logs": lambda: cmd_logs(args.follow, args.lines),
        "chat": lambda: cmd_chat(config, args.model, args),
    }

    commands[args.cmd]()


if __name__ == "__main__":
    main()
