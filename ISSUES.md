# Issue log

Every issue found while building and testing the gateway, newest work last. Add to this file when a new issue is found; change the status line rather than deleting an entry.

Statuses: **Open** (not addressed), **Worked around** (usable, root cause remains), **Fixed** (changed in the repository), **Unverified** (suspected or untested, not confirmed).

Last updated: 2026-10-03.

## Summary

| # | Issue | Status |
|---|---|---|
| 1 | Calls longer than 60 s time out in Claude Code and the answer is lost | Worked around |
| 2 | Cloudflare may cut off tunnelled calls at about 125 s | Unverified |
| 3 | The run log shows `ok` when the caller never received the answer | Open |
| 4 | The gateway keeps running a worker after the caller has gone | Open |
| 5 | `second_opinion` writes its reasoning into the answer and burns the token budget | Open |
| 6 | The service was not loaded after a deploy | Open |
| 7 | A session's tool definition stays stale after a deploy and reconnect | Open |
| 8 | Gateway error messages can contain model output | Open |
| 9 | Run-log writes block the gateway's event loop | Open |
| 10 | The gateway access log has no timestamps | Open |
| 11 | Runs show project `unknown` unless the caller passes a label | Open |
| 12 | The `horizon` worker is always unavailable | Open |
| 13 | Smoke scripts write to the real run log | Open |
| 14 | The `sqlite3` command-line tool could not open the run log read-only | Unverified |
| 15 | Offline test output looks like real TypeSafe calls | Open |
| 16 | Parts of the observability work have not been exercised for real | Unverified |
| 17 | Token usage was dropped for empty answers and for Jev | Fixed |
| 18 | A dashboard on the gateway's port would have been public | Fixed |

## Timeouts and delivery

### 1. Calls longer than 60 s time out in Claude Code and the answer is lost

- **Status:** Worked around.
- **Seen:** run `871df3a22743402f9d752ab56d8025b6` (routed, 4,000-token limit). The worker took 67.9 s; the client returned "The operation timed out" first. The answer was never delivered and is not stored anywhere.
- **Cause:** Claude Code gives each request to an HTTP MCP server 60 seconds by default to produce its first response byte. The gateway uses `json_response=True` (`src/local_agents_mcp/http.py`), so it sends nothing until the tool finishes. The 60-second figure is from the Claude Code documentation; the exact moment of the client error was not captured.
- **Workaround:** start the session with `MCP_TOOL_TIMEOUT=180000 claude`. It applies to every MCP server in that session.
- **Possible fixes:** have the gateway stream or send progress so a first byte arrives early; or use the stdio registration, which has no per-request timer; or keep worker calls short with a lower `max_tokens`.
- **Cost:** the lost call still made one billed Jev request.

### 2. Cloudflare may cut off tunnelled calls at about 125 s

- **Status:** Unverified.
- **Basis:** Cloudflare's error 524 page gives a default proxy read timeout of 125 seconds, changeable only on Enterprise plans. Not checked against this tunnel or plan, and no call has run that long.
- **Effect if true:** a 180-second client timeout would not be the real limit over the public URL.
- **Next step:** check the tunnel's behaviour with a deliberately long local-only call, or read the tunnel settings.

### 3. The run log shows `ok` when the caller never received the answer

- **Status:** Open.
- **Seen:** run `871df3a2…` is recorded as `ok`, not truncated, although the answer was lost (issue 1).
- **Cause:** the log records gateway execution only. The gateway cannot confirm delivery.
- **Proposed:** a separate delivery field (`unknown`, `response_sent`, `not_delivered`, `confirmed`), with `confirmed` set only by the caller, plus a "longer than 60 s" risk hint in the dashboard. Not built.

### 4. The gateway keeps running a worker after the caller has gone

- **Status:** Open.
- **Seen:** in run `871df3a2…` the worker ran to completion after the client had timed out.
- **Cause:** the worker call runs in a thread that cannot be cancelled, and nothing checks for a client disconnect.
- **Effect:** wasted local compute, and on this 16 GB machine a long orphaned call delays the next one.

## Workers

### 5. `second_opinion` writes its reasoning into the answer and burns the token budget

- **Status:** Open. Cause not investigated.
- **Seen:** run `12c37536cab04ad7a8c5270391b40457` (400-token limit) returned only thinking-aloud text and stopped at `finish_reason: length` with no ranked answer. The two 4,000-token runs used about 2,600 output tokens each and took over 60 s.
- **Note:** the worker (`qwen3:4b`) is configured with `reasoning_effort: "none"`, which was meant to prevent this.
- **Effect:** truncated or slow answers, and it pushes routed calls past the 60-second client limit (issue 1).
- **Next step:** check whether `reasoning_effort` is being honoured for this model on the installed Ollama version.

### 12. The `horizon` worker is always unavailable

- **Status:** Open (known before this work).
- **Seen:** every routed call reports `horizon` as unavailable because the `llama_cpp` provider is not reachable.
- **Effect:** general, non-code tasks have no dedicated worker; Jev can only choose among the three Ollama workers.

## Deployment and client

### 6. The service was not loaded after a deploy

- **Status:** Open. Cause not confirmed.
- **Seen:** after `scripts/deploy_macos.py` ran at 17:26 on 2026-10-03, the files were copied and the service file rewritten, but `ai.localagents.mcp` was not loaded. The public URL returned HTTP 502 until `launchctl bootstrap` was run by hand.
- **Suspected cause:** `launchctl bootstrap` failing straight after the preceding `bootout`. The deploy's terminal output was not seen.
- **Next step:** capture the error on the next deploy; consider waiting for the old service to stop, or retrying the bootstrap.

### 7. A session's tool definition stays stale after a deploy and reconnect

- **Status:** Open.
- **Seen:** after the deploy and an `/mcp` reconnect, the session's definitions of `ask_local_agent` and `route_local_agent` still had no `project` parameter. The gateway accepted and stored `project` anyway.
- **Effect:** a model working from the stale definition may not pass `project`.
- **Next step:** confirm that a new Claude Code session shows the updated definition.

### 11. Runs show project `unknown` unless the caller passes a label

- **Status:** Open (needs a user action).
- **Cause:** the gateway cannot infer the caller's repository; the label is self-reported.
- **Next step:** add a line to the delegation rule in `~/.claude/CLAUDE.md` telling Claude Code to pass `project`. Not done; that file was left alone on instruction.

## Gateway code

### 8. Gateway error messages can contain model output

- **Status:** Open.
- **Where:** `_request_json` and `_chat_sync` in `src/local_agents_mcp/server.py` put the raw runtime response body into the exception message.
- **Effect:** that text is returned to the caller in `error` fields and could reach logs. The run log avoids it by storing reason codes only.

### 9. Run-log writes block the gateway's event loop

- **Status:** Open (accepted limitation for now).
- **Cause:** recorder writes are synchronous. With the database locked by another process, a call can stall the gateway for about 0.5 s before recording gives up.

### 10. The gateway access log has no timestamps

- **Status:** Open.
- **Seen:** `POST /mcp` lines in `gateway.log` could not be matched to run `871df3a2…`, so it is unknown whether the gateway noticed the client disconnect.

### 13. Smoke scripts write to the real run log

- **Status:** Open.
- **Detail:** `scripts/smoke_test.py` with `RUN_GENERATION=1` or `RUN_ROUTING=1` records runs in the real database, and `RUN_ROUTING=1` makes a billed Jev call.

### 15. Offline test output looks like real TypeSafe calls

- **Status:** Open (cosmetic).
- **Detail:** the mocked transport logs `POST https://api.typesafe.ai/...` lines during the offline suite. No request leaves the machine, but the output reads as if one did.

## Unverified

### 14. The `sqlite3` command-line tool could not open the run log read-only

- **Status:** Unverified.
- **Seen:** `sqlite3 -readonly` on the run log failed with "unable to open database file", while Python's read-only connection to the same file worked. It may have been a restriction of the tool session rather than of the file.

### 16. Parts of the observability work have not been exercised for real

- **Status:** Unverified.
- The dashboard has only been run against a sample database, not started from the deployed copy with `.venv/bin/python -m local_agents_mcp.dashboard`.
- Cancellation is tested with a mock; how a real MCP client cancellation reaches the tool is unknown.
- The version 1 to version 2 migration has only run against test databases.
- Whether a per-server `timeout` field is honoured for a server registered in user config is unknown.
- That port 8001 is unreachable through the tunnel rests on the README's statement that the tunnel forwards to port 8000.
- Delivery of run `864e2557475248e3831e5baf59003555`, the routed test taken as successful, was reported from another session and not observed in the run log.

## Fixed

### 17. Token usage was dropped for empty answers and for Jev

- **Status:** Fixed in `76f6219`.
- **Was:** `EmptyResponse` kept the finish reason and token count only inside its message, and `ask_jev` discarded the SDK's usage.
- **Now:** both are carried through and recorded on their steps.

### 18. A dashboard on the gateway's port would have been public

- **Status:** Fixed by design in `76f6219`.
- **Was:** the tunnel forwards the public hostname to port 8000, and tunnelled requests arrive from loopback, so a dashboard there would have been reachable from outside.
- **Now:** the dashboard is a separate process on `127.0.0.1:8001` with Host and Origin checks.
