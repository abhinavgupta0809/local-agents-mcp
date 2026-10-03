"""Record tool calls as runs with nested steps in a local SQLite database.

Only metadata is stored: names, counts, timings, and reason codes. Task, context,
answer, and error text never reach this module's storage or its log lines.
Recording is best-effort: no failure here may reach the model call around it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

DB_ENV = "LOCAL_AGENTS_OBS_DB"
ENABLED_ENV = "LOCAL_AGENTS_OBS"
PROJECT_ENV = "LOCAL_AGENTS_PROJECT"
# Outside the deployed runtime directory, so a deployment never touches it.
DEFAULT_DB_PATH = Path.home() / "Library/Application Support/local-ai-agents/data/observability.sqlite3"

# How long one write may wait for another process's lock before it is dropped.
BUSY_TIMEOUT_SECONDS = 0.25
# After this many dropped writes a run stops recording, which caps the delay it can add.
MAX_WRITE_FAILURES = 2

# Set by the server entry point: "stdio" or "http".
transport = "unknown"

EVAL_RESULTS = ("pass", "fail", "partial")
# Suggestions only; any label-shaped name is accepted as a scope.
EVAL_SCOPE_EXAMPLES = ("arithmetic", "coverage", "label_accuracy")

# Stored text must look like an identifier, never like prose.
_LABEL = re.compile(r"[A-Za-z0-9._:/+@-]{1,80}")

RUN_FIELDS = frozenset(
    {
        "ended_at", "elapsed_ms", "status", "error_kind", "final_agent", "routing_source",
        "fallback_reason", "worker_fallback", "truncated", "empty_responses",
    }
)
STEP_FIELDS = frozenset(
    {
        "ended_at", "elapsed_ms", "status", "reason", "provider", "model", "finish_reason",
        "truncated", "input_tokens", "output_tokens", "confidence",
    }
)
STEP_REASONS = frozenset({"empty_response", "worker_error"})

# MIGRATIONS[n] upgrades a database from schema version n to n + 1.
MIGRATIONS: list[tuple[str, ...]] = [
    (
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            elapsed_ms INTEGER,
            tool TEXT NOT NULL,
            transport TEXT,
            pid INTEGER,
            project TEXT NOT NULL,
            project_source TEXT NOT NULL,
            status TEXT NOT NULL,
            error_kind TEXT,
            final_agent TEXT,
            routing_source TEXT,
            fallback_reason TEXT,
            worker_fallback INTEGER,
            truncated INTEGER,
            empty_responses INTEGER,
            task_chars INTEGER,
            context_chars INTEGER,
            max_tokens INTEGER,
            eval_status TEXT NOT NULL DEFAULT 'unevaluated',
            eval_scope TEXT,
            eval_updated_at TEXT
        )
        """,
        """
        CREATE TABLE steps (
            run_id TEXT NOT NULL REFERENCES runs(id),
            seq INTEGER NOT NULL,
            kind TEXT NOT NULL,
            name TEXT,
            provider TEXT,
            model TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            elapsed_ms INTEGER,
            status TEXT NOT NULL,
            reason TEXT,
            finish_reason TEXT,
            truncated INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            confidence REAL,
            PRIMARY KEY (run_id, seq)
        )
        """,
        "CREATE INDEX runs_started_at ON runs(started_at)",
    ),
    # Evaluations move from one result per run to an append-only table. Judgements
    # held in the old columns are carried over before those columns are dropped.
    (
        """
        CREATE TABLE evaluations (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id),
            scope TEXT NOT NULL,
            result TEXT NOT NULL CHECK (result IN ('pass', 'fail', 'partial')),
            method TEXT NOT NULL,
            method_version TEXT,
            evaluated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX evaluations_run ON evaluations(run_id, evaluated_at)",
        """
        INSERT INTO evaluations (id, run_id, scope, result, method, method_version, evaluated_at)
        SELECT lower(hex(randomblob(16))), id, COALESCE(eval_scope, 'unspecified'), eval_status,
               'migrated_v1', NULL, COALESCE(eval_updated_at, started_at)
        FROM runs WHERE eval_status IN ('pass', 'fail', 'partial')
        """,
        "ALTER TABLE runs DROP COLUMN eval_status",
        "ALTER TABLE runs DROP COLUMN eval_scope",
        "ALTER TABLE runs DROP COLUMN eval_updated_at",
    ),
]

# Database files this process has already created and migrated.
_ready: set[str] = set()


def enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "1").strip().lower() not in {"0", "false", "off"}


def db_path() -> Path:
    return Path(os.environ.get(DB_ENV) or DEFAULT_DB_PATH)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def label(value: Any) -> str | None:
    """Return value if it is a short identifier-like string, "invalid" if it is other text."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if _LABEL.fullmatch(text) else "invalid"


def _clean(value: Any) -> Any:
    """Reduce a value to something safe to store: a label, a number, or NULL."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return label(value)
    return None


def resolve_project(argument: str = "") -> tuple[str, str]:
    """The caller's self-reported project label and where it came from. Display-only."""
    if isinstance(argument, str) and argument.strip():
        return label(argument) or "unknown", "argument"
    from_environment = os.environ.get(PROJECT_ENV, "").strip()
    if from_environment:
        return label(from_environment) or "unknown", "environment"
    return "unknown", "none"


def _log_failure(action: str, error: BaseException) -> None:
    # The class name only: exception text can carry SQL parameters or file contents.
    logger.warning("observability %s failed: %s", action, type(error).__name__)


def _backup_before_migration(connection: sqlite3.Connection, path: Path, version: int) -> None:
    """Snapshot a database with SQLite's backup API before changing its schema."""
    final = path.with_name(f"{path.name}.v{version}.bak")
    if final.exists():
        return
    partial = path.with_name(f"{path.name}.v{version}.bak.{os.getpid()}.tmp")
    try:
        target = sqlite3.connect(partial)
        try:
            connection.backup(target)
            copied_version = target.execute("PRAGMA user_version").fetchone()[0]
        finally:
            target.close()
        os.chmod(partial, 0o600)
        # Another process may have migrated first; keep only a pre-migration snapshot.
        if copied_version == version and not final.exists():
            os.replace(partial, final)
    finally:
        partial.unlink(missing_ok=True)


def _migrate(connection: sqlite3.Connection, path: Path) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    target = len(MIGRATIONS)
    if version == target:
        return
    if version > target:
        raise RuntimeError("database schema is newer than this gateway")
    if version > 0:
        _backup_before_migration(connection, path, version)
    connection.execute("BEGIN IMMEDIATE")
    try:
        # Re-read under the write lock: another process may have migrated meanwhile.
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        for number in range(version, target):
            for statement in MIGRATIONS[number]:
                connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {target}")
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the database for writing, creating and migrating it when needed."""
    path = path or db_path()
    key = str(path)
    if key not in _ready:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
    try:
        if key not in _ready:
            os.chmod(path, 0o600)
            connection.execute("PRAGMA journal_mode = WAL")
            _migrate(connection, path)
            _ready.add(key)
    except BaseException:
        connection.close()
        raise
    return connection


def connect_readonly(path: Path | None = None) -> sqlite3.Connection:
    """Open an existing database for reading only. Raises if it does not exist."""
    path = path or db_path()
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=BUSY_TIMEOUT_SECONDS)
    connection.row_factory = sqlite3.Row
    return connection


def _execute(action: str, sql: str, parameters: tuple[Any, ...]) -> bool:
    """Run one bounded write. Returns False, after logging the error class, on any failure."""
    path = db_path()
    try:
        connection = connect(path)
        try:
            connection.execute(sql, parameters)
        finally:
            connection.close()
        return True
    except Exception as error:
        # Forget the file so the next write re-creates or re-checks it.
        _ready.discard(str(path))
        _log_failure(action, error)
        return False


def _update(table: str, allowed: frozenset[str], fields: dict[str, Any]) -> tuple[str, list[Any]]:
    names = [name for name in fields if name in allowed]
    assignments = ", ".join(f"{name} = ?" for name in names)
    return f"UPDATE {table} SET {assignments}", [_clean(fields[name]) for name in names]


def _error_reason(error: BaseException) -> str:
    reason = getattr(error, "reason", None)
    return reason if isinstance(reason, str) and reason in STEP_REASONS else "worker_error"


class Step:
    """One timed part of a run. Use as a context manager around the work it measures."""

    def __init__(self, run: "Run", seq: int, kind: str, name: str | None, fields: dict[str, Any]):
        self.run = run
        self.seq = seq
        self.kind = kind
        self.name = name
        self.fields: dict[str, Any] = dict(fields)
        self._clock = time.monotonic()
        self._closed = False

    def set(self, **fields: Any) -> None:
        self.fields.update(fields)

    def _begin(self) -> None:
        try:
            self.run._write(
                "step start",
                "INSERT INTO steps (run_id, seq, kind, name, provider, model, started_at, status)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'running')",
                (
                    self.run.id, self.seq, _clean(self.kind), _clean(self.name),
                    _clean(self.fields.get("provider")), _clean(self.fields.get("model")), utc_now(),
                ),
            )
        except Exception as error:
            _log_failure("step start", error)

    def finish(self, status: str, **fields: Any) -> None:
        try:
            if self._closed:
                return
            self._closed = True
            self.fields.update(fields)
            self.fields.update(
                status=status, ended_at=utc_now(), elapsed_ms=round((time.monotonic() - self._clock) * 1000)
            )
            if self.fields.get("reason") == "empty_response":
                self.run.empty_responses += 1
            sql, parameters = _update("steps", STEP_FIELDS, self.fields)
            self.run._write("step end", f"{sql} WHERE run_id = ? AND seq = ?", (*parameters, self.run.id, self.seq))
        except Exception as error:
            _log_failure("step end", error)

    def __enter__(self) -> "Step":
        return self

    def __exit__(self, kind: Any, error: BaseException | None, traceback: Any) -> bool:
        if error is None:
            self.finish(self.fields.get("status") or "ok")
        elif isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)):
            self.finish("error", reason="cancelled")
        else:
            self.finish(
                "error",
                reason=_error_reason(error),
                finish_reason=getattr(error, "finish_reason", None),
                truncated=getattr(error, "finish_reason", None) == "length",
                **usage_tokens(getattr(error, "usage", None)),
            )
        return False


class Run:
    """One tool call. Written as "running" at the start and updated when it ends.

    A run whose process stops first keeps the status "running": nothing here
    guesses at how another process's call ended.
    """

    def __init__(self, tool: str, *, task: str = "", context: str = "", max_tokens: int | None = None, project: str = ""):
        self.id = uuid.uuid4().hex
        self.recorded = False
        self.fields: dict[str, Any] = {}
        self.empty_responses = 0
        self._failures = 0
        self._seq = 0
        self._clock = time.monotonic()
        self._closed = False
        self._active = False
        try:
            self._active = enabled()
            name, source = resolve_project(project)
            self.recorded = self._write(
                "run start",
                "INSERT INTO runs (id, started_at, tool, transport, pid, project, project_source, status,"
                " task_chars, context_chars, max_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)",
                (
                    self.id, utc_now(), _clean(tool), _clean(transport), os.getpid(), name, source,
                    len(task) if isinstance(task, str) else None,
                    len(context) if isinstance(context, str) else None,
                    _clean(max_tokens),
                ),
            )
        except Exception as error:
            _log_failure("run start", error)

    @property
    def run_id(self) -> str | None:
        """The id to hand back to the caller, or None when the run row was never written.

        It is set by the initial insert alone and kept even if later writes are
        dropped, so that a partial trace can still be found and evaluated. It
        therefore says a run row exists, not that the recording is complete: the
        run may be left "running" and steps may be missing.
        """
        return self.id if self.recorded else None

    def _write(self, action: str, sql: str, parameters: tuple[Any, ...]) -> bool:
        if not self._active or self._failures >= MAX_WRITE_FAILURES:
            return False
        written = _execute(action, sql, parameters)
        if not written:
            self._failures += 1
        return written

    def set(self, **fields: Any) -> None:
        self.fields.update(fields)

    def step(self, kind: str, name: str | None = None, **fields: Any) -> Step:
        self._seq += 1
        step = Step(self, self._seq, kind, name, fields)
        step._begin()
        return step

    def finish(self, status: str, **fields: Any) -> None:
        try:
            if self._closed:
                return
            self._closed = True
            self.fields.update(fields)
            self.fields.update(
                status=status,
                ended_at=utc_now(),
                elapsed_ms=round((time.monotonic() - self._clock) * 1000),
                empty_responses=self.empty_responses,
            )
            sql, parameters = _update("runs", RUN_FIELDS, self.fields)
            self._write("run end", f"{sql} WHERE id = ?", (*parameters, self.id))
        except Exception as error:
            _log_failure("run end", error)

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, kind: Any, error: BaseException | None, traceback: Any) -> bool:
        if error is None:
            self.finish(self.fields.get("status") or "ok")
        elif isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)):
            self.finish("error", error_kind="cancelled")
        else:
            self.finish("error", error_kind=self.fields.get("error_kind") or _error_reason(error))
        return False


def usage_tokens(usage: Any) -> dict[str, Any]:
    """Token counts from an OpenAI-style or TypeSafe-style usage object."""
    if not isinstance(usage, dict):
        return {}
    tokens = {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
    }
    return {name: value for name, value in tokens.items() if isinstance(value, int) and not isinstance(value, bool)}


def add_evaluation(run_id: str, scope: str, result: str, method: str, method_version: str = "") -> str | None:
    """Append one judgement of a run's answer and return its id, or None if the run is unknown.

    Evaluations are only ever added, so earlier ones, including repeats of the
    same scope, are kept. Every field is a short label: there is nowhere to put
    a prompt, an answer, or a free-text note. Execution status is never touched.
    """
    if result not in EVAL_RESULTS:
        raise ValueError(f"evaluation result must be one of: {', '.join(EVAL_RESULTS)}")
    for name, value, required in (("scope", scope, True), ("method", method, True), ("method version", method_version, False)):
        value = (value or "").strip()
        if (required and not value) or (value and label(value) != value):
            raise ValueError(f"evaluation {name} must be a short name with no spaces, such as arithmetic or v2")
    evaluation_id = uuid.uuid4().hex
    connection = connect()
    try:
        cursor = connection.execute(
            "INSERT INTO evaluations (id, run_id, scope, result, method, method_version, evaluated_at)"
            " SELECT ?, id, ?, ?, ?, ?, ? FROM runs WHERE id = ?",
            (evaluation_id, scope.strip(), result, method.strip(), (method_version or "").strip() or None, utc_now(), run_id),
        )
        return evaluation_id if cursor.rowcount == 1 else None
    finally:
        connection.close()
