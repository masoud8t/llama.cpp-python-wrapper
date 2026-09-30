# `lm` — Easy Human Wrapper for `llama.cpp`

`lm` is a small, dependency-free command-line wrapper around [`llama.cpp`](https://github.com/ggml-org/llama.cpp) and `llama-server`.

The goal is simple:

> **Make a powerful local llama.cpp installation easy to operate without requiring you to remember long commands, build paths, GGUF filenames, process IDs, or API details.**

You point the script at:

1. your **GGUF model directory**
2. your **llama.cpp directory**

Then `lm` handles the rest.

---

## What it does

`lm` provides a human-friendly interface for the most common llama.cpp server tasks:

- discover GGUF models automatically
- recognize split/sharded GGUF models
- use short model aliases
- start `llama-server`
- stop it cleanly
- restart it with new settings
- detect stale PID files
- protect against PID reuse on Linux when `/proc` is available
- detect port conflicts before starting
- wait until the API is actually ready
- show process, socket, and API status separately
- display the active model and runtime settings
- follow server logs
- run an interactive terminal chat
- save chat history as JSON
- pass advanced llama-server arguments through configuration
- work with model-specific settings without rewriting the Python script

No third-party Python packages are required.

---

# 1. Quick start

## Edit the two paths

Open `lm.py` and change these values near the top:

```python
MODEL_DIR = Path("/workbook/models")
LLAMA_CPP_DIR = Path("/workspace/llama.cpp")
```

For example:

```python
MODEL_DIR = Path("/home/masoud/models")
LLAMA_CPP_DIR = Path("/home/masoud/llama.cpp")
```

`MODEL_DIR` should contain your `.gguf` model files.

`LLAMA_CPP_DIR` should point to the root of your llama.cpp checkout.

The wrapper searches common build locations automatically, including:

```text
build/bin/llama-server
build-*/bin/llama-server
build/Release/bin/llama-server
*/bin/llama-server
```

The script also checks the Ivy Bridge layout first:

```text
build-ivybridge/bin/llama-server
```

---

## Make it executable

On Linux:

```bash
chmod +x lm.py
```

You can then run it directly:

```bash
./lm.py list
```

Or install/symlink it as `lm`:

```bash
ln -s "$(pwd)/lm.py" ~/.local/bin/lm
```

Then:

```bash
lm list
```

Make sure `~/.local/bin` is in your `PATH`.

---

# 2. First commands to run

After configuring the paths:

```bash
lm doctor
```

This checks the important pieces of the installation.

Then:

```bash
lm list
```

You should see the discovered GGUF models.

Then start one:

```bash
lm run gemma4
```

Check it:

```bash
lm status
```

Start a terminal chat:

```bash
lm chat gemma4
```

Stop the server when finished:

```bash
lm stop
```

That's the basic workflow.

---

# 3. Command reference

## `lm list`

List GGUF models found in `MODEL_DIR`.

```bash
lm list
```

Example:

```text
Models: /workbook/models

 #  MODEL                                                        SIZE  ALIAS
------------------------------------------------------------------------------------------
 1  gemma-4-e2b-it-Q4_K_M.gguf                                 6.2 GiB  gemma4
 2  nemotron-super-49b-v1-Q4_K_M                                 29.4 GiB  nemotron
```

The wrapper automatically groups files that look like split GGUF shards:

```text
model-00001-of-00005.gguf
model-00002-of-00005.gguf
model-00003-of-00005.gguf
...
```

They appear as one model in the list.

---

## `lm doctor`

Check the local installation.

```bash
lm doctor
```

It reports:

- model directory
- llama.cpp directory
- detected `llama-server`
- state directory
- configuration path
- discovered model count
- llama-server version when it can be queried
- Python version

It also reminds you where advanced llama-server arguments belong.

---

## `lm config`

Show the wrapper configuration file and its current contents.

```bash
lm config
```

The configuration is created automatically the first time the wrapper runs.

---

## `lm run MODEL`

Start a model server.

```bash
lm run gemma4
```

You can override common settings for one run:

```bash
lm run gemma4 --context 131072
```

Other supported runtime options include:

```text
--host
--port
--threads
--batch-threads
--context
--output
--parallel
--jinja
--no-jinja
```

Example:

```bash
lm run gemma4 \
    --context 131072 \
    --threads 24 \
    --batch-threads 24 \
    --port 8080
```

The wrapper checks for an already managed server and checks whether the requested port is already occupied before starting.

---

## `lm status`

Show the state of the managed server.

```bash
lm status
```

Typical output:

```text
process : RUNNING
pid     : 12345
socket  : OPEN
api     : READY
endpoint: http://127.0.0.1:8080/v1
model   : Gemma 4 E2B
file    : gemma-4-e2b-it-Q4_K_M.gguf
context : 131072
output  : 32768
parallel: 1
```

The important distinction is that these are separate states:

```text
process : RUNNING
socket  : OPEN
api     : LOADING
```

A running process and open port do **not** automatically mean the model is ready for requests. `lm` waits for the API health check to succeed.

Possible API states include:

- `DOWN`
- `STARTING`
- `LOADING`
- `READY`

---

## `lm chat MODEL`

Start the requested model if necessary, wait for it to become ready, then open an interactive chat.

```bash
lm chat gemma4
```

If a different model is already running, `lm chat` does not silently replace it. Instead it tells you to explicitly restart with the requested model.

### Chat commands

Inside the chat:

```text
/help
/clear
/model
/status
/save FILE
/quit
```

Examples:

```text
/save ~/chats/test.json
```

```text
/clear
```

Exit with:

```text
/quit
```

or `Ctrl-D`.

### Generation options

You can pass common generation parameters when starting chat:

```bash
lm chat gemma4 --temperature 0.7
```

Supported chat options include:

```text
--output
--context
--host
--port
--threads
--batch-threads
--parallel
--jinja
--no-jinja
--temperature
--top-p
--top-k
--seed
```

`lm` uses the OpenAI-compatible chat endpoint exposed by the server and streams the response to the terminal.

---

## `lm stop`

Stop the managed llama-server cleanly:

```bash
lm stop
```

If the process does not stop normally, use:

```bash
lm stop --force
```

The forced path escalates from normal termination to `SIGKILL` when necessary.

---

## `lm restart [MODEL]`

Restart the current model:

```bash
lm restart
```

Or explicitly select a model:

```bash
lm restart gemma4
```

You can also change runtime settings:

```bash
lm restart gemma4 --context 131072
```

All common server options supported by `run` are also available for `restart`.

Use `--force` when a normal stop does not finish:

```bash
lm restart gemma4 --force
```

---

## `lm logs`

Show the last 200 lines of the server log:

```bash
lm logs
```

Choose the number of lines:

```bash
lm logs --lines 500
```

Follow the log live:

```bash
lm logs --follow
```

Press `Ctrl-C` to stop following.

---

# 4. Model names and aliases

The wrapper can resolve a model in several ways.

### Configured alias

You can define:

```json
"models": {
  "gemma4": {
    "file": "gemma-4-e2b-it-Q4_K_M.gguf"
  }
}
```

Then:

```bash
lm run gemma4
```

### Exact filename or model name

You can also use the actual model filename or discovered model name.

### Unique substring

A unique substring is enough.

For example, when only one model matches:

```bash
lm run gemma
```

If more than one model matches, the wrapper refuses to guess and reports the ambiguity.

This is intentional.

> **A wrapper should make easy things easier without making ambiguous things dangerous.**

---

# 5. Configuration

The wrapper automatically creates:

```text
STATE_DIR/config.json
```

With the default paths used in the script, that is:

```text
/workbook/.llama-manager/config.json
```

A typical configuration looks like this:

```json
{
  "defaults": {
    "host": "127.0.0.1",
    "port": 8080,
    "threads": 24,
    "batch_threads": 24,
    "context": 524288,
    "output": 32768,
    "parallel": 1,
    "jinja": true,
    "extra_args": []
  },
  "models": {
    "gemma4": {
      "file": "gemma-4-e2b-it-Q4_K_M.gguf",
      "context": 131072
    },
    "nemotron": {
      "file": "nemotron-model-00001-of-00005.gguf",
      "context": 524288
    }
  }
}
```

## Configuration precedence

For the common runtime options, the effective value is selected in this order:

```text
command line
    ↓
model configuration
    ↓
global defaults
```

So this:

```bash
lm run gemma4 --context 128000
```

overrides a model-specific or global default context for that run.

---

# 6. Advanced llama-server arguments

The wrapper intentionally does **not** forward arbitrary unknown command-line options.

This prevents a common class of mistakes where an option intended for `lm` accidentally leaks into the `llama-server` command.

Instead, put advanced llama-server arguments in `config.json`.

### Global arguments

```json
"defaults": {
  "extra_args": [
    "--some-llama-option",
    "123"
  ]
}
```

### Per-model arguments

```json
"models": {
  "gemma4": {
    "file": "gemma-4-e2b-it-Q4_K_M.gguf",
    "extra_args": [
      "--some-model-specific-option"
    ]
  }
}
```

Model-specific arguments are appended after global arguments.

This keeps the wrapper's command namespace separate from llama.cpp's command namespace.

---

# 7. Vision / multimodal models

A model can optionally define an `mmproj` file:

```json
"models": {
  "vision-model": {
    "file": "model.gguf",
    "mmproj": "mmproj-model-f16.gguf"
  }
}
```

The wrapper converts that into the corresponding `llama-server` option automatically.

The path may be:

- relative to `MODEL_DIR`
- absolute

---

# 8. Runtime state

`lm` keeps its runtime state inside `STATE_DIR`.

The default is:

```text
MODEL_DIR/../.llama-manager/
```

Typical files are:

```text
.llama-manager/
├── config.json
├── server.pid
├── server.json
└── server.log
```

### `config.json`

Persistent wrapper configuration.

### `server.pid`

PID of the managed llama-server process.

### `server.json`

Metadata about the currently managed server:

- model
- model file
- PID
- host
- port
- context
- output limit
- parallel slots
- Jinja setting
- mmproj path
- startup command

### `server.log`

Combined stdout/stderr output from llama-server, including the command used to start it.

---

# 9. Safety and process handling

The wrapper is deliberately conservative in a few places.

## It does not silently switch models

Suppose `gemma4` is running and you execute:

```bash
lm chat nemotron
```

`lm` will not kill Gemma and replace it automatically.

Instead, it asks for an explicit operation:

```bash
lm restart nemotron
```

This avoids accidental model switching.

## It checks for stale PID state

A PID file can survive a crash.

`lm` checks whether the recorded process is still alive and cleans stale state when appropriate.

On Linux, when `/proc` is available, it also checks the process command line to reduce the risk of treating an unrelated process with a reused PID as the managed llama-server.

## It starts llama-server in its own session

The server is launched with a separate process session so the wrapper can terminate the managed process group cleanly.

---

# 10. Typical workflows

## Start and chat

```bash
lm list
lm run gemma4
lm status
lm chat gemma4
```

## Chat with automatic startup

You do not need to run `lm run` first:

```bash
lm chat gemma4
```

`lm` will start the model when necessary and wait until the API is ready.

## Change context size

```bash
lm restart gemma4 --context 131072
```

Or start directly:

```bash
lm run gemma4 --context 131072
```

## Investigate a failed startup

```bash
lm status
lm logs
```

For a live startup log:

```bash
lm logs --follow
```

## Full health check

```bash
lm doctor
lm list
lm status
```

---

# 11. Troubleshooting

## `llama-server not found`

Run:

```bash
lm doctor
```

Then verify that `LLAMA_CPP_DIR` points to the **root of the llama.cpp checkout**, not directly to `llama-server`.

Example:

```text
/workspace/llama.cpp
├── build-ivybridge/
├── examples/
├── common/
└── ...
```

The wrapper searches for:

```text
build-ivybridge/bin/llama-server
build/bin/llama-server
build/Release/bin/llama-server
build-*/bin/llama-server
*/bin/llama-server
```

## `Model not found`

Run:

```bash
lm list
```

Check `MODEL_DIR` and make sure the model has a `.gguf` extension.

## `Ambiguous model`

Use a more specific model name or configure an explicit alias in `config.json`.

Example:

```json
"models": {
  "my-model": {
    "file": "the-exact-file-name.gguf"
  }
}
```

Then:

```bash
lm run my-model
```

## Port already in use

Check:

```bash
lm status
```

Or choose another port:

```bash
lm run gemma4 --port 8081
```

## Server says `LOADING`

This normally means the process exists and the socket is reachable, but the model is not ready yet.

Check:

```bash
lm status
```

and:

```bash
lm logs --follow
```

`lm chat` automatically waits for the API to become ready when it starts or attaches to the requested model.

## Server failed to start

Run:

```bash
lm logs
```

Then:

```bash
lm doctor
```

The startup command is also recorded in `server.log` and `server.json`, which makes debugging easier.

---

# 12. Design philosophy

`lm` is intentionally **not** a replacement for llama.cpp.

It is a thin human interface on top of it.

The philosophy is:

```text
llama.cpp does the heavy work.

lm removes operational friction.
```

Instead of remembering something like:

```bash
/path/to/llama-server \
    -m /long/path/to/model.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    -t 24 \
    -tb 24 \
    -c 524288 \
    -n 32768 \
    -np 1 \
    --jinja
```

you use:

```bash
lm run my-model
```

And instead of manually checking whether a process is alive, a port is open, and whether the model has finished loading, you use:

```bash
lm status
```

The intent is to make local inference feel more like using a normal service than operating a collection of low-level binaries and flags.

---

# 13. Requirements

- Linux or another POSIX-like environment with the usual process/signal behavior
- Python 3.10+ recommended
- a working `llama-server` build from llama.cpp
- one or more `.gguf` models

No external Python dependencies are required.

The wrapper uses only Python standard-library modules.

---

# 14. Project layout

A simple setup can look like:

```text
/workspace/
├── llama.cpp/
│   ├── build-ivybridge/
│   │   └── bin/
│   │       └── llama-server
│   └── ...
│
/workbook/
├── models/
│   ├── gemma-4-e2b-it-Q4_K_M.gguf
│   ├── another-model-00001-of-00004.gguf
│   ├── another-model-00002-of-00004.gguf
│   └── ...
│
├── .llama-manager/
│   ├── config.json
│   ├── server.pid
│   ├── server.json
│   └── server.log
│
└── lm.py
```

---

# 15. Minimal cheat sheet

```bash
# Discover everything
lm doctor
lm list

# Run a model
lm run gemma4

# Check it
lm status

# Chat
lm chat gemma4

# Change context and restart
lm restart gemma4 --context 131072

# Logs
lm logs
lm logs --follow

# Stop
lm stop

# Force stop if necessary
lm stop --force
```

---

# 16. The idea in one sentence

> **`lm` turns llama.cpp from a low-level model server you have to operate into a simple local AI service you can control with a few human-friendly commands.**
