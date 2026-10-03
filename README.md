# Local AI agent gateway

This project gives MCP clients one set of tools for consulting language models on this Mac. It currently uses Ollama for the models already downloaded here. The provider entries in `config/agents.json` also support any OpenAI-compatible local runtime, including llama.cpp and MLX when those servers are installed and running.

The architecture is: IDE or Claude → MCP gateway → local model runtime. For access away from home, Cloudflare Tunnel can route `https://mcp.<your-domain>/mcp` to the gateway on `127.0.0.1:8000`. The models and inference remain on this Mac, except that `route_local_agent` sends the task text to a cloud API to choose an agent (see Routing with Jev). The Mac must be on and online to answer requests.

## What is implemented and what is not

Implemented:

- Five MCP tools: `list_local_agents`, `local_runtime_status`, `ask_local_agent`, `compare_local_agents`, and `route_local_agent`.
- Configurable runtimes (`providers`) and named workers (`agents`) in `config/agents.json`. A worker is a model plus a system prompt and request settings.
- Jev routing with availability checks, a fixed fallback order, and a bounded number of worker attempts.
- Task-only routing by default: the context is not sent to the cloud unless `routing.jev.send_context` is turned on.
- Empty answers are rejected as worker failures, and every answer reports its `finish_reason`.
- MCP over stdio, and MCP over HTTP on localhost behind a bearer token with Host and Origin checks.
- A local run log in SQLite for `ask_local_agent`, `compare_local_agents`, and `route_local_agent`, with a localhost-only dashboard (see Observability). It stores metadata only.
- Offline tests that mock both Jev and the local runtimes.

Not implemented (future work):

- Workers have no file, shell, or browser tools of their own. They only answer the text they are given.
- No automatic task decomposition or LangGraph flow.
- The run log is local and minimal: no text capture, no automatic evaluation, no retention or pruning, no alerts, and no remote access to the dashboard.
- No automatic retry or larger token limit when an answer is cut off. The caller has to check `finish_reason`.
- No streaming responses.
- The `horizon` worker is configured, but its llama.cpp runtime is not set up here, so it is reported as unavailable. No MLX worker is configured.
- The macOS login service and the Cloudflare Tunnel are set up on this Mac by hand. Their service files and settings are not part of this repository.

## Installed deployment on this Mac

The source repository and Git history live at `$HOME/Desktop/projects/local-agents-mcp`
(the existing parent folder may appear as `Projects` on this Mac).
macOS restricts background-service access to Desktop. The `ai.localagents.mcp`
LaunchAgent therefore runs a deployed copy at
`$HOME/Library/Application Support/local-ai-agents/runtime`, with logs under
`$HOME/Library/Logs/local-ai-agents`. This runtime has no Git repository; the Desktop
checkout remains the source of truth.

After changing source, configuration, or private keys, deploy from the checkout:

```sh
python3 scripts/deploy_macos.py
claude mcp list
```

The deployment command copies the runtime files and the gateway token and optional Jev
key, synchronizes locked dependencies, and restarts the login service. Private runtime
files use owner-only permissions. The run log lives beside the runtime, in
`$HOME/Library/Application Support/local-ai-agents/data`; deployment creates that
directory if it is missing and never copies, replaces, or deletes the database. It does not call Jev or local models. The command needs
`uv` on PATH and briefly interrupts gateway availability. For a restart with no updates:

```sh
launchctl kickstart -k "gui/$(id -u)/ai.localagents.mcp"
```

Claude Code and Antigravity use the public MCP URL; relocation does not change their
remote connection settings. Keep `.local/` ignored by Git and private in both locations.

## 1. Local setup

You need [uv](https://docs.astral.sh/uv/) and Ollama running with the models named in `config/agents.json`. From this directory:

```sh
uv sync --locked
uv run python scripts/smoke_test.py
```

The smoke test lists agents and checks runtime reachability. To make it call a model too, run `RUN_GENERATION=1 uv run python scripts/smoke_test.py`.

Private values live in the `.local/` directory, which Git ignores. Nothing in it should ever be committed or pasted into chat:

- `.local/typesafe_api_key` holds the TypeSafe API key used by `route_local_agent`. The `TYPESAFE_API_KEY` environment variable is used if the file is missing or empty. Without either, routing uses the fallback order.
- `.local/mcp_token` holds the bearer token for the HTTP gateway. It is only needed for sections 3 and 4.

To create the gateway token with owner-only permissions:

```sh
mkdir -p .local && chmod 700 .local
(umask 077 && python3 -c "import secrets; print(secrets.token_urlsafe(48))" > .local/mcp_token)
```

## 2. Connect an MCP client on this Mac

The entry point is `.venv/bin/local-agents-mcp`. It uses MCP over stdio by default and needs no domain, token, or HTTP port.

### Claude Code

Register the server once for all your projects, using the absolute path to this directory:

```sh
claude mcp add --scope user local-ai-agents -- "/absolute/path/to/this/project/.venv/bin/local-agents-mcp"
```

Then run `/mcp` inside Claude Code to confirm that `local-ai-agents` is connected. Claude Code starts the server itself, so changes to the Python code take effect the next time the server is started. Changes to `config/agents.json` are read on every request.

To reach the gateway from another machine, connect over HTTP through the tunnel in section 4. Read the token from your own secure store; do not write it into a file in this repository:

```sh
claude mcp add --transport http --scope user local-ai-agents \
  "https://mcp.<your-domain>/mcp" --header "Authorization: Bearer $LOCAL_AGENTS_MCP_TOKEN"
```

### Other clients

Add a server named `local-ai-agents` to the client's MCP configuration:

```json
{
  "mcpServers": {
    "local-ai-agents": {
      "command": "/absolute/path/to/this/project/.venv/bin/local-agents-mcp"
    }
  }
}
```

For Google Antigravity, open **Settings → Customizations → Installed MCP Servers → Open MCP Config** and add that entry to the existing `mcpServers` object. Do not replace other servers in the file. Then refresh MCP servers. Antigravity may also expose **MCP Servers → Manage MCP Servers → View raw config** from the agent panel. The current global config path in Google's guide is `~/.gemini/config/mcp_config.json`.

Ask the client to call `list_local_agents` and then `ask_local_agent` with `agent="reviewer"` and a short task.

## 3. Local HTTP gateway

`scripts/serve_http.py` starts the gateway at `http://127.0.0.1:8000/mcp`. It reads the bearer token from `.local/mcp_token` and allows one public hostname, which is set in the script (`mcp.ag-mcp.xyz` here; change it for your own domain).

```sh
uv run python scripts/serve_http.py
```

Every HTTP request must include `Authorization: Bearer <your-token>`. A `401 Unauthorized` response when opening the URL in a browser is expected. The server binds only to localhost and rejects unexpected Host and Origin values.

On this Mac a macOS login service named `ai.localagents.mcp` runs that script at sign-in. The service file is not part of this repository. The local login service was tested with a valid MCP initialization and a rejected request without a token.

## 4. Domain and Cloudflare Tunnel

Cloudflare Tunnel publishes the localhost gateway on a hostname you own, without opening a port on your router. To set one up:

1. Add your domain to Cloudflare.
2. In the Cloudflare Zero Trust dashboard, create a tunnel and install the `cloudflared` connector on this Mac with the command the dashboard shows. That command contains the tunnel token. Keep it private and do not save it in this repository.
3. Add a published application route that maps `mcp.<your-domain>` to `http://127.0.0.1:8000`. Never point a route at Ollama's port 11434, which has no authentication.
4. Set the same hostname in `scripts/serve_http.py` so the gateway accepts it.

The gateway bearer token is still required on every request that arrives through the tunnel.

On this Mac the domain is `ag-mcp.xyz`, `cloudflared` runs as a system service, and the remote MCP endpoint is `https://mcp.ag-mcp.xyz/mcp`. Verified on 2026-10-02: `mcp.ag-mcp.xyz` resolves publicly, the tunnel is connected, an unauthenticated MCP request returns HTTP 401, and an authenticated MCP tool call reaches the local Gemma reviewer and returns a response.

## Calling workers

`ask_local_agent` calls one configured worker. `compare_local_agents` calls two or three sequentially to fit this Mac's memory. `route_local_agent` picks the worker for you (see the next section).

Each answer includes `finish_reason`, passed through from the runtime. `stop` means the model finished. `length` means the answer was cut off at `max_tokens`, so it may be incomplete.

An answer whose content is null, empty, or only whitespace is treated as a worker failure with the reason `empty_response`. `ask_local_agent` raises an error, `compare_local_agents` records the error for that worker, and `route_local_agent` moves on to the next-ranked worker.

Qwen 3 is a reasoning model. If its thinking uses the whole token budget, no answer text is left. The Qwen workers therefore send `reasoning_effort: "none"` through `request_overrides`. On Ollama 0.34.4, the native `think: false` flag is ignored by `/v1/chat/completions`, while `reasoning_effort: "none"` turns thinking off.

## Routing with Jev

`route_local_agent` takes a task and optional context, picks an agent, runs it, and returns the answer with a `routing` record. It checks which agents are available (runtime reachable and model listed), asks TypeSafe's Jev model to choose among those, and calls the winner.

This is the one tool that is not fully local. The task text is sent to `api.typesafe.ai` to make the choice. The context is not sent unless `routing.jev.send_context` is turned on; it goes only to the chosen local agent. The other tools never contact TypeSafe.

The question, its per-agent criteria, the Jev settings, and the fallback rules are all in the `routing` block of `config/agents.json`. The API key is read from the ignored `.local/typesafe_api_key` file on each routed request.

If Jev cannot answer (no key, timeout, API error, or a choice that is not an available agent), the first available agent in `routing.fallback.order` is used and `routing.fallback_reason` says why. If the chosen agent fails or returns an empty answer, the next-ranked one is tried, up to `routing.fallback.max_worker_attempts` agents in total, each at most once. Each failed attempt in `routing.attempts` has a `reason` (`empty_response` or `worker_error`) and the error text.

## Observability

Each call to `ask_local_agent`, `compare_local_agents`, or `route_local_agent` is recorded as a run with ordered steps. The other two tools are not recorded.

### What is stored

The database is `$HOME/Library/Application Support/local-ai-agents/data/observability.sqlite3` (owner-only; override with `LOCAL_AGENTS_OBS_DB`). The stdio server and the login service write to the same file. Set `LOCAL_AGENTS_OBS=0` to turn recording off.

| Table | Contents |
|---|---|
| `runs` | Tool, transport, process ID, project label, execution status, error kind, final agent, routing source and fallback reason, worker-fallback, truncation and empty-answer flags, task and context character counts, token limit, UTC start and end, elapsed milliseconds |
| `steps` | Kind (`availability`, `jev`, or `worker`), name, provider, model, status, reason, finish reason, truncation flag, input and output tokens, Jev confidence, UTC start and end, elapsed milliseconds |
| `evaluations` | Evaluation ID, run ID, scope, result, method, method version, UTC timestamp |

Only metadata is stored. There are no columns for task, context, answer, prompt, or error text, and no key is ever written. Every stored text value must look like a short identifier; anything else is stored as `invalid`, and an agent name that is not in the config is stored as `unknown`. Recorder log lines carry the error class only, never its message.

Jev and the local workers are separate steps, so their time and token counts are never mixed. The Jev step's time includes any retries the TypeSafe SDK makes internally.

### How a run is written

A run is inserted with the status `running` before any work starts. Each step is inserted when it starts and updated when it ends. The run is updated when the tool returns or raises. Start and end are UTC timestamps; elapsed time comes from the monotonic clock, so a clock change cannot make it negative.

If the process stops mid-call, the run and its open step keep the status `running` with no end time. Nothing marks them finished later: the gateway cannot tell a stopped process from a call that another process is still running. The dashboard shows them as running and says that no end was recorded.

| Execution status | Meaning |
|---|---|
| `running` | No end recorded |
| `ok` | The call returned an answer |
| `partial` | `compare_local_agents` returned, but at least one worker failed |
| `error` | The call raised, was cancelled, or every worker failed |

Recording is best-effort. Each write waits at most 0.25 seconds for a lock held by another process, and a run stops recording after two dropped writes. A missing, locked, corrupt, or newer-version database never changes a tool's answer; the result then has `"run_id": null`. Each tool result otherwise includes the `run_id` of its run.

A `run_id` means the run's row was written at the start of the call. It does not mean the recording is complete. If later writes are dropped, the ID is still returned so the partial trace can be found and evaluated, but that run may stay `running` with no end time and may be missing steps.

### Execution is not evaluation

A run's execution status says only whether the call completed. Whether the answer is any good is recorded separately, as rows in `evaluations`:

- Each evaluation names its **scope**, the one check it refers to, such as `arithmetic`, `coverage`, or `label_accuracy`. Any short name without spaces is accepted.
- The **result** is `pass`, `fail`, or `partial`, and applies to that scope only.
- The **method** and optional **method version** say how it was checked, for example `manual` or `checker` / `v2`.
- Evaluations are only ever added. Repeating a check of the same scope adds a row and keeps the earlier ones.
- There is no overall verdict, and nothing is derived from the execution status. A run with no evaluation has not been checked.
- All fields are short labels. There is no notes field.

Add an evaluation from the run's page in the dashboard, or from Python with `observability.add_evaluation(run_id, scope, result, method, method_version)`.

### Project labels

The gateway cannot tell which repository a call came from: remote clients all reach one HTTP process. The three tools therefore take an optional `project` argument, a short label the caller supplies, such as the repository's folder name. For stdio, `LOCAL_AGENTS_PROJECT` is used when the argument is empty. Otherwise the project is `unknown`.

The label is self-reported and display-only. It is not verified, it is not sent to Jev, and nothing in the gateway branches on it. For Claude Code to fill it in, add a line such as "pass `project` as the current repository's folder name" to the delegation rule in `~/.claude/CLAUDE.md`.

### Dashboard

```sh
uv run python -m local_agents_mcp.dashboard
```

This serves `http://127.0.0.1:8001/` until you stop it. It is a separate process from the gateway, so it has to be started by hand and a dashboard fault cannot affect model calls. From the deployed copy, run `.venv/bin/python -m local_agents_mcp.dashboard` in the runtime directory.

- **Runs** lists the latest 100 runs with execution status, the latest result for each evaluated scope, and fallback, empty-answer, and truncation flags. It can be filtered by project, tool, and status.
- **Run detail** shows Jev and worker time and tokens side by side, a step timeline, the step table, the evaluation history, and a form to add an evaluation.

The dashboard binds to `127.0.0.1` only and has no option to bind elsewhere. It refuses any request whose `Host` header is not `127.0.0.1:8001` or `localhost:8001`, and any POST whose `Origin` is not the dashboard itself. The Cloudflare Tunnel forwards to port 8000, so it does not reach port 8001; do not add a tunnel route to it. Pages use no JavaScript and no external assets, and open the database read-only.

### Schema changes and backups

The schema version is kept in SQLite's `user_version`. When a newer gateway opens an older database, it first writes a snapshot next to it with SQLite's online backup API (`observability.sqlite3.v<old version>.bak`), then applies the change in one transaction. If the change fails, the transaction is rolled back, the database is left as it was, and recording is skipped.

Version 1 kept one evaluation result and scope on each run. Version 2 moves them to the `evaluations` table: each run with a `pass`, `fail`, or `partial` result becomes one row with the method `migrated_v1` and its original timestamp, and the old columns are removed.

A database written by a newer gateway is left untouched by an older one.

### Long calls and the client timeout

Claude Code gives each request to an HTTP MCP server 60 seconds by default to produce its first response byte. This gateway returns one JSON body when the tool finishes, so a call that takes longer than that times out in the client while the gateway carries on and completes it. The run log then shows the run as `ok`, because it records what the gateway did, not whether the caller received the answer.

To allow longer calls, start the session with a higher tool timeout:

```sh
MCP_TOOL_TIMEOUT=180000 claude
```

The value is in milliseconds. It applies to every MCP server in that session and also becomes the overall limit for one tool call. This is from the Claude Code documentation as read with version 2.1.288.

Other limits sit behind the client's. The gateway waits up to `timeout_seconds` (240) for each worker attempt. Calls that arrive through the tunnel also pass through Cloudflare, whose documentation gives a default proxy read timeout of 125 seconds; that limit has not been verified for this tunnel, so do not assume a call can run for the full 180 seconds over the public URL.

### Checked on this Mac

These runs were made against the deployed gateway on 2026-10-03, over the public URL. Times and token counts are from the run log.

| Run | Call | Result recorded by the gateway | What the caller got |
|---|---|---|---|
| `0de968d2…` | `ask_local_agent` to `reviewer` | `ok`, 5.0 s, `finish_reason` `stop` | The answer |
| `12c37536…` | `route_local_agent`, 400-token limit | `ok`, 10.8 s, Jev chose `second_opinion`, truncated (`length`) | A truncated answer |
| `871df3a2…` | `route_local_agent`, 4,000-token limit | `ok`, 68.1 s, Jev chose `second_opinion`, `stop` | Nothing: the client timed out at its default limit |
| `864e2557…` | `route_local_agent`, 4,000-token limit | `ok`, 64.6 s, Jev chose `second_opinion` (confidence 0.90), `stop`, 157 tokens in and 2,569 out | Reported as received |

The last run is the routed test taken as successful for this phase. It ran longer than the 60-second default, which is consistent with a session started with the command above, but the run log does not record client settings or delivery. No evaluation was recorded for any of these runs, so none of the answers has been checked.

## The `@local` trigger in Claude Code

Claude Code can be told to delegate only when asked. This is a rule in the user's global instructions file, `~/.claude/CLAUDE.md`, not something this repository installs. With the rule in place, a message that starts with `@local` sends that subtask through `route_local_agent`; without it, the local tools are not called. For example:

```text
@local Review this function for edge cases.
```

To set it up, add a section like this to `~/.claude/CLAUDE.md`:

```markdown
# Manual local delegation

Rules for the `local-ai-agents` MCP tools (local models on this Mac).

- Trigger: `@local` at the beginning of my message is a manual delegation trigger, not a file reference.
- Without that trigger or an explicit request from me, do not call `local-ai-agents` tools.
- With `@local`, use `route_local_agent` for the requested subtask. Let Jev choose the worker.
- Send a concise task description in `task`. Put supporting code or project excerpts in `context`; include only what the worker needs.
- Use `max_tokens: 800` unless I specify another limit.
- Check the returned answer and `finish_reason`. Report the worker, the routing source, and any fallback or truncation.
- Do not automatically repeat calls or increase token limits.
- Local delegation does not independently authorize file edits.
- Privacy: when reporting a delegation, explain that worker inference is local, but routing uses the online Jev API (TypeSafe). The `task` text is sent to Jev; `context` goes only to the local worker unless `send_context` is enabled in the gateway's `config/agents.json`.
```

Because the task sentence goes to the Jev API, keep code and project excerpts in `context`.

## Tests

Run the offline tests, which mock both Jev and the local runtimes and make no network calls. They write the run log to a temporary directory, never to the real database, and the deployment test runs against a stand-in checkout and home directory with `launchctl` and `uv` mocked:

```sh
uv run python -m unittest discover -s tests
```

`RUN_ROUTING=1 uv run python scripts/smoke_test.py` makes one real, billed Jev call and one local generation.
