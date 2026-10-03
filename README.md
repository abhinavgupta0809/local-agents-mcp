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
- Offline tests that mock both Jev and the local runtimes.

Not implemented (future work):

- Workers have no file, shell, or browser tools of their own. They only answer the text they are given.
- No automatic task decomposition, LangGraph flow, or observability platform.
- No automatic retry or larger token limit when an answer is cut off. The caller has to check `finish_reason`.
- No streaming responses.
- The `horizon` worker is configured, but its llama.cpp runtime is not set up here, so it is reported as unavailable. No MLX worker is configured.
- The macOS login service and the Cloudflare Tunnel are set up on this Mac by hand. Their service files and settings are not part of this repository.

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

Run the offline tests, which mock both Jev and the local runtimes and make no network calls:

```sh
uv run python -m unittest discover -s tests
```

`RUN_ROUTING=1 uv run python scripts/smoke_test.py` makes one real, billed Jev call and one local generation.
