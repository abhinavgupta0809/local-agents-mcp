"""A small read-mostly dashboard for the run log, served on localhost only.

Run it by hand: python -m local_agents_mcp.dashboard

It is a separate process from the gateway, on its own port, so the public tunnel
(which forwards to the gateway's port) does not reach it and a dashboard fault
cannot affect model calls. The only write it makes is adding an evaluation to a run.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from local_agents_mcp import observability


HOST = "127.0.0.1"
DEFAULT_PORT = 8001
RUN_LIMIT = 100
MAX_FORM_BYTES = 2048

RUN_PATH = re.compile(r"/runs/([0-9a-f]{32})")
EVALUATION_PATH = re.compile(r"/runs/([0-9a-f]{32})/evaluations")
FILTERS = ("project", "tool", "status")
SCOPES_SHOWN = 3
EVALUATION_ERROR = (
    "Not saved. Give a scope, a result, and a method. Scope, method, and version are "
    "short names with no spaces, such as arithmetic, manual, or v2."
)

# Colors follow one role each: step kind (blue, orange, aqua) or state (good,
# warning, critical). Text always uses the text tokens, never a data color.
STYLE = """
:root { color-scheme: dark; --page: #121211; --surface: #1a1a19; --line: #383835;
  --text: #ffffff; --text-2: #c3c2b7; --text-3: #8f8e85;
  --worker: #3987e5; --jev: #d95926; --availability: #199e70;
  --good: #0ca30c; --warning: #fab219; --critical: #d03b3b; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 48px; background: var(--page); color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 0 0 12px; }
a { color: var(--text); } p { margin: 0 0 12px; } .muted { color: var(--text-2); }
.note { color: var(--text-2); max-width: 70ch; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
  padding: 16px; margin-top: 16px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th { text-align: left; font-weight: 600; color: var(--text-2); white-space: nowrap; }
th, td { padding: 6px 12px 6px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
tr:last-child td { border-bottom: 0; } td a { white-space: nowrap; } td.num, th.num { text-align: right; }
code { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--text-2); }
.badge { white-space: nowrap; } .badge::before { content: ""; display: inline-block; width: 8px;
  height: 8px; border-radius: 50%; margin-right: 6px; background: var(--text-3); }
.badge.good::before { background: var(--good); } .badge.warning::before { background: var(--warning); }
.badge.critical::before { background: var(--critical); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; }
.tile .label { color: var(--text-2); } .tile .value { font-size: 22px; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px 24px; margin: 0; }
.facts dt { color: var(--text-2); } .facts dd { margin: 0; }
form { display: flex; flex-wrap: wrap; gap: 8px; align-items: end; }
label { display: grid; gap: 2px; color: var(--text-2); font-size: 12px; }
input, select, button { font: inherit; color: var(--text); background: var(--page);
  border: 1px solid var(--line); border-radius: 6px; padding: 5px 8px; }
button { cursor: pointer; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; color: var(--text-2); margin-bottom: 12px; }
.key { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.worker { background: var(--worker); } .jev { background: var(--jev); } .availability { background: var(--availability); }
.timeline { display: grid; grid-template-columns: minmax(120px, 220px) minmax(160px, 1fr) minmax(90px, 150px);
  gap: 8px 12px; align-items: center; min-width: 520px; }
.track { position: relative; height: 12px; border-left: 1px solid var(--line); }
.bar { position: absolute; top: 0; height: 12px; min-width: 3px; border-radius: 4px; }
.bar.open { border-radius: 4px 0 0 4px; opacity: 0.55;
  -webkit-mask-image: linear-gradient(90deg, #000 30%, transparent); mask-image: linear-gradient(90deg, #000 30%, transparent); }
.axis { display: flex; justify-content: space-between; color: var(--text-3); font-size: 12px; }
.error { color: var(--text); border-left: 3px solid var(--critical); padding-left: 10px; }
"""

EXECUTION_BADGES = {"ok": ("good", "✓ ok"), "partial": ("warning", "◐ partial"), "error": ("critical", "✕ error"),
                    "running": ("", "… running"), "skipped": ("", "– skipped")}
EVALUATION_BADGES = {"pass": ("good", "✓ pass"), "fail": ("critical", "✕ fail"), "partial": ("warning", "◐ partial")}


def text(value: Any, empty: str = "–") -> str:
    return empty if value is None or value == "" else escape(str(value))


def badge(value: Any, badges: dict[str, tuple[str, str]]) -> str:
    tone, words = badges.get(value, ("", str(value)))
    return f'<span class="badge {tone}">{escape(words)}</span>'


def duration(milliseconds: Any) -> str:
    if not isinstance(milliseconds, (int, float)):
        return "–"
    return f"{milliseconds:,.0f} ms" if milliseconds < 1000 else f"{milliseconds / 1000:,.1f} s"


def when(timestamp: Any) -> str:
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError):
        return "–"


def offset_ms(start: Any, timestamp: Any) -> float:
    """Milliseconds from a run's start to a step's start, from their UTC timestamps."""
    try:
        delta = datetime.fromisoformat(timestamp) - datetime.fromisoformat(start)
    except (TypeError, ValueError):
        return 0.0
    return max(delta.total_seconds() * 1000, 0.0)


def flags(run: sqlite3.Row) -> str:
    found = []
    if run["fallback_reason"]:
        found.append("routing fallback")
    if run["worker_fallback"]:
        found.append("worker fallback")
    if run["empty_responses"]:
        found.append(f"{run['empty_responses']} empty")
    if run["truncated"]:
        found.append("truncated")
    return escape(", ".join(found)) or "–"


def page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>"
    )


def runs_page(connection: sqlite3.Connection, query: dict[str, str]) -> str:
    conditions, parameters = [], []
    for name in FILTERS:
        value = observability.label(query.get(name))
        if value:
            conditions.append(f"{name} = ?")
            parameters.append(value)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    runs = connection.execute(
        f"SELECT * FROM runs {where} ORDER BY started_at DESC LIMIT {RUN_LIMIT}", parameters
    ).fetchall()
    # The latest result for each scope of each listed run. Scopes are never merged into one verdict.
    latest: dict[str, dict[str, str]] = {}
    for row in connection.execute(
        f"SELECT run_id, scope, result FROM evaluations WHERE run_id IN (SELECT id FROM runs {where}"
        f" ORDER BY started_at DESC LIMIT {RUN_LIMIT}) ORDER BY evaluated_at, rowid",
        parameters,
    ):
        latest.setdefault(row["run_id"], {})[row["scope"]] = row["result"]

    def evaluations(run_id: str) -> str:
        scopes = list(latest.get(run_id, {}).items())
        if not scopes:
            return '<span class="muted">none recorded</span>'
        shown = "<br>".join(f"{text(scope)} {badge(result, EVALUATION_BADGES)}" for scope, result in scopes[:SCOPES_SHOWN])
        return shown + (f'<br><span class="muted">+{len(scopes) - SCOPES_SHOWN} more</span>' if len(scopes) > SCOPES_SHOWN else "")

    def options(name: str, column: str) -> str:
        values = [row[0] for row in connection.execute(f"SELECT DISTINCT {column} FROM runs ORDER BY 1")]
        chosen = query.get(name, "")
        items = ['<option value="">all</option>'] + [
            f'<option{" selected" if value == chosen else ""}>{escape(str(value))}</option>' for value in values
        ]
        return f'<label>{escape(name.replace("_", " "))}<select name="{name}">{"".join(items)}</select></label>'

    rows = "".join(
        "<tr>"
        f'<td><a href="/runs/{escape(run["id"])}">{when(run["started_at"])}</a></td>'
        f"<td>{text(run['project'])}</td><td>{text(run['tool'])}</td>"
        f"<td>{badge(run['status'], EXECUTION_BADGES)}</td>"
        f"<td>{evaluations(run['id'])}</td>"
        f"<td>{text(run['final_agent'])}</td><td>{text(run['routing_source'])}</td>"
        f"<td>{flags(run)}</td><td class=\"num\">{duration(run['elapsed_ms'])}</td>"
        "</tr>"
        for run in runs
    )
    table = (
        "<table><thead><tr><th>Started</th><th>Project</th><th>Tool</th><th>Execution</th><th>Latest evaluation by scope</th>"
        f'<th>Agent</th><th>Routing</th><th>Flags</th><th class="num">Elapsed</th></tr></thead><tbody>{rows}</tbody></table>'
        if runs
        else '<p class="muted">No runs match.</p>'
    )
    return page(
        "Runs",
        "<h1>Runs</h1>"
        '<p class="note">Execution says whether the call completed. Evaluations say which named checks were made '
        "on the answer and how each came out. A completed call is not a correct answer, and a run with no "
        "evaluation has not been checked. Project is a label the caller supplied; it is not verified.</p>"
        f'<div class="panel"><form method="get" action="/">{options("project", "project")}{options("tool", "tool")}'
        f'{options("status", "status")}<button>Filter</button></form></div>'
        f'<div class="panel">{table}</div>'
        f'<p class="muted">Showing the latest {RUN_LIMIT} at most. Metadata only: no task, context, or answer text is stored.</p>',
    )


def timeline(run: sqlite3.Row, steps: list[sqlite3.Row]) -> str:
    """Each step as a bar on the run's own time axis. A step with no recorded end fades out."""
    placed = [(step, offset_ms(run["started_at"], step["started_at"])) for step in steps]
    ends = [offset + (step["elapsed_ms"] or 0) for step, offset in placed]
    span = max([run["elapsed_ms"] or 0, *ends, 1])
    rows = []
    for step, offset in placed:
        left = min(offset / span * 100, 99.0)
        kind = step["kind"] if step["kind"] in ("worker", "jev", "availability") else "worker"
        name = f"{text(step['kind'])} · {text(step['name'])}"
        if step["ended_at"] is None:
            style, classes, value = f"left:{left:.2f}%;right:0", f"bar open {kind}", "no end recorded"
        else:
            width = min(max((step["elapsed_ms"] or 0) / span * 100, 0.4), 100 - left)
            style, classes, value = f"left:{left:.2f}%;width:{width:.2f}%", f"bar {kind}", duration(step["elapsed_ms"])
            if step["status"] != "ok":
                value += f" · {escape(EXECUTION_BADGES.get(step['status'], ('', str(step['status'])))[1])}"
        tip = escape(f"{step['kind']} {step['name']}: starts at {duration(offset)}, {value}")
        rows.append(
            f'<div>{step["seq"]}. {name}</div>'
            f'<div class="track"><div class="{classes}" style="{style}" title="{tip}"></div></div>'
            f'<div class="muted">{value}</div>'
        )
    return (
        '<div class="legend"><span><span class="key availability"></span>availability check</span>'
        '<span><span class="key jev"></span>Jev routing (cloud)</span>'
        '<span><span class="key worker"></span>local worker</span></div>'
        f'<div class="timeline">{"".join(rows)}'
        f'<div></div><div class="axis"><span>0</span><span>{duration(span)}</span></div><div></div></div>'
    )


def run_page(connection: sqlite3.Connection, run_id: str, message: str = "") -> str | None:
    run = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        return None
    steps = connection.execute("SELECT * FROM steps WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()

    def total(kind: str, column: str) -> Any:
        values = [step[column] for step in steps if step["kind"] == kind and step[column] is not None]
        return sum(values) if values else None

    def tokens(kind: str) -> str:
        return f"{text(total(kind, 'input_tokens'))} in / {text(total(kind, 'output_tokens'))} out"

    tiles = "".join(
        f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in (
            ("Total elapsed", duration(run["elapsed_ms"])),
            ("Jev time", duration(total("jev", "elapsed_ms"))),
            ("Jev tokens", tokens("jev")),
            ("Worker time", duration(total("worker", "elapsed_ms"))),
            ("Worker tokens", tokens("worker")),
        )
    )
    facts = "".join(
        f"<div><dt>{label}</dt><dd>{value}</dd></div>"
        for label, value in (
            ("Execution", badge(run["status"], EXECUTION_BADGES) + (f" · {text(run['error_kind'])}" if run["error_kind"] else "")),
            ("Tool", text(run["tool"])),
            ("Project (self-reported)", f"{text(run['project'])} <span class=\"muted\">via {text(run['project_source'])}</span>"),
            ("Started", when(run["started_at"])),
            ("Ended", when(run["ended_at"]) if run["ended_at"] else "not recorded"),
            ("Final agent", text(run["final_agent"])),
            ("Routing source", text(run["routing_source"])),
            ("Routing fallback reason", text(run["fallback_reason"])),
            ("Flags", flags(run)),
            ("Input size", f"{text(run['task_chars'])} task chars, {text(run['context_chars'])} context chars"),
            ("Token limit", text(run["max_tokens"])),
            ("Transport, process", f"{text(run['transport'])}, pid {text(run['pid'])}"),
        )
    )
    unfinished = (
        '<p class="note">No end has been recorded for this run. The call may still be in progress, or its '
        "process may have stopped before it finished. This page does not guess which.</p>"
        if run["status"] == "running"
        else ""
    )
    step_rows = "".join(
        "<tr>"
        f"<td>{step['seq']}</td><td>{text(step['kind'])}</td><td>{text(step['name'])}</td><td>{text(step['model'])}</td>"
        f"<td>{badge(step['status'], EXECUTION_BADGES)}</td><td>{text(step['reason'])}</td>"
        f"<td>{text(step['finish_reason'])}{' (truncated)' if step['truncated'] else ''}</td>"
        f'<td class="num">{duration(step["elapsed_ms"])}</td>'
        f'<td class="num">{text(step["input_tokens"])}</td><td class="num">{text(step["output_tokens"])}</td>'
        "</tr>"
        for step in steps
    )
    history = connection.execute(
        "SELECT * FROM evaluations WHERE run_id = ? ORDER BY evaluated_at DESC, rowid DESC", (run_id,)
    ).fetchall()
    history_rows = "".join(
        "<tr>"
        f"<td>{when(item['evaluated_at'])}</td><td>{text(item['scope'])}</td>"
        f"<td>{badge(item['result'], EVALUATION_BADGES)}</td>"
        f"<td>{text(item['method'])}</td><td>{text(item['method_version'])}</td>"
        "</tr>"
        for item in history
    )
    history_table = (
        "<table><thead><tr><th>Evaluated</th><th>Scope</th><th>Result</th><th>Method</th><th>Version</th></tr></thead>"
        f"<tbody>{history_rows}</tbody></table>"
        if history
        else '<p class="muted">No evaluations recorded. This answer has not been checked.</p>'
    )
    results = "".join(f"<option>{result}</option>" for result in observability.EVAL_RESULTS)
    examples = "".join(f'<option value="{example}">' for example in observability.EVAL_SCOPE_EXAMPLES)
    notice = f'<p class="error">{escape(message)}</p>' if message else ""
    return page(
        "Run detail",
        f'<p><a href="/">← Runs</a></p><h1>Run detail</h1><p><code>{escape(run["id"])}</code></p>{unfinished}'
        f'<div class="panel"><div class="tiles">{tiles}</div></div>'
        f'<div class="panel"><h2>Execution</h2><dl class="facts">{facts}</dl></div>'
        f'<div class="panel"><h2>Step timeline</h2>{timeline(run, steps) if steps else "<p class=muted>No steps recorded.</p>"}</div>'
        '<div class="panel"><h2>Steps</h2><table><thead><tr><th>#</th><th>Kind</th><th>Name</th><th>Model</th>'
        '<th>Status</th><th>Reason</th><th>Finish reason</th><th class="num">Elapsed</th>'
        f'<th class="num">Tokens in</th><th class="num">Tokens out</th></tr></thead><tbody>{step_rows}</tbody></table></div>'
        '<div class="panel"><h2>Evaluation history</h2>'
        '<p class="note">Each row is one check of the answer, newest first. A result applies to its scope only; '
        "there is no overall verdict, and none is derived from the execution status above. Earlier rows are kept "
        "when a check is repeated.</p>"
        f'{history_table}</div><div class="panel"><h2>Add an evaluation</h2>{notice}'
        f'<form method="post" action="/runs/{escape(run["id"])}/evaluations">'
        '<label>Scope (what was checked)<input name="scope" list="scopes" maxlength="80" required></label>'
        f'<datalist id="scopes">{examples}</datalist>'
        f'<label>Result<select name="result">{results}</select></label>'
        '<label>Method<input name="method" maxlength="80" value="manual" required></label>'
        '<label>Method version (optional)<input name="method_version" maxlength="80"></label>'
        "<button>Add evaluation</button></form></div>",
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalAgentsDashboard"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def reply(self, status: int, body: str, content_type: str = "text/html; charset=utf-8", location: str = "") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        if location:
            self.send_header("Location", location)
        self.end_headers()
        self.wfile.write(data)

    def refuse(self, status: int, reason: str) -> None:
        self.reply(status, reason, "text/plain; charset=utf-8")

    def local_request(self) -> bool:
        """Reject any Host but this dashboard's own, which blocks DNS rebinding and proxied requests."""
        port = self.server.server_address[1]
        if self.headers.get("Host", "") not in (f"{HOST}:{port}", f"localhost:{port}"):
            self.refuse(403, "Forbidden: this dashboard only answers on localhost.")
            return False
        return True

    def do_GET(self) -> None:
        if not self.local_request():
            return
        url = urlsplit(self.path)
        query = {name: values[0] for name, values in parse_qs(url.query).items()}
        try:
            connection = observability.connect_readonly()
        except sqlite3.Error:
            self.reply(200, page("Runs", '<h1>Runs</h1><p class="note">No run database yet. It is created by the first recorded tool call.</p>'))
            return
        try:
            match = RUN_PATH.fullmatch(url.path)
            if url.path == "/":
                body = runs_page(connection, query)
            elif match:
                body = run_page(connection, match.group(1), EVALUATION_ERROR if "error" in query else "")
            else:
                body = None
        except sqlite3.Error as error:
            self.refuse(500, f"Could not read the run database ({type(error).__name__}).")
            return
        finally:
            connection.close()
        if body is None:
            self.refuse(404, "Not found")
        else:
            self.reply(200, body)

    def do_POST(self) -> None:
        if not self.local_request():
            return
        port = self.server.server_address[1]
        # A form on any other site carries that site's Origin, so it cannot add an evaluation.
        if self.headers.get("Origin", "") not in (f"http://{HOST}:{port}", f"http://localhost:{port}"):
            self.refuse(403, "Forbidden: cross-origin request.")
            return
        match = EVALUATION_PATH.fullmatch(urlsplit(self.path).path)
        if not match:
            self.refuse(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_FORM_BYTES:
            self.refuse(413, "Form too large")
            return
        form = {name: values[0] for name, values in parse_qs(self.rfile.read(length).decode("utf-8", "replace")).items()}
        run_id = match.group(1)
        try:
            found = observability.add_evaluation(
                run_id, form.get("scope", ""), form.get("result", ""), form.get("method", ""), form.get("method_version", "")
            )
        except ValueError:
            self.reply(303, "", location=f"/runs/{run_id}?error=1")
            return
        except sqlite3.Error as error:
            self.refuse(500, f"Could not save the evaluation ({type(error).__name__}).")
            return
        if not found:
            self.refuse(404, "Not found")
            return
        self.reply(303, "", location=f"/runs/{run_id}")


def create_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Bind to the loopback interface only. There is no option to bind elsewhere."""
    return ThreadingHTTPServer((HOST, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Localhost dashboard for the local agent run log")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = create_server(args.port)
    print(f"Run log dashboard: http://{HOST}:{args.port}/  (database: {observability.db_path()})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
