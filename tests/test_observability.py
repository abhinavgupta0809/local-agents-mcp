"""Offline tests for the run log and its dashboard. Jev and the local runtimes are mocked."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import http.client
import importlib.util
import io
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from local_agents_mcp import dashboard, observability, routing, server
from test_routing import FAKE_KEY, OLLAMA_MODELS, FakeJev, FakeLocal, reply


REPO = Path(__file__).resolve().parents[1]
RANKED = {"reviewer": 0.6, "second_opinion": 0.3, "coder": 0.1}


class ObservabilityCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "data" / "obs.sqlite3"
        self.key_path = Path(self.tmp.name) / "typesafe_api_key"
        self.key_path.write_text(FAKE_KEY + "\n", encoding="utf-8")
        self.local = FakeLocal()
        self.config = server._load_config()
        self.config["routing"]["jev"]["timeout_seconds"] = 0.5
        for patcher in (
            mock.patch.object(routing, "KEY_PATH", self.key_path),
            mock.patch.dict(os.environ, {observability.DB_ENV: str(self.db)}, clear=False),
            mock.patch.object(server, "_request_json", self.local),
            mock.patch.object(server, "_load_config", lambda: self.config),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in (routing.KEY_ENV, observability.PROJECT_ENV, observability.ENABLED_ENV):
            os.environ.pop(name, None)
        observability._ready.clear()
        self.addCleanup(observability._ready.clear)
        self.use_jev(FakeJev())

    def use_jev(self, jev: FakeJev) -> None:
        self.jev = jev
        patcher = mock.patch.object(routing, "_transport", jev.transport())
        patcher.start()
        self.addCleanup(patcher.stop)

    def query(self, sql: str, parameters=()) -> list[sqlite3.Row]:
        connection = sqlite3.connect(self.db)
        connection.row_factory = sqlite3.Row
        try:
            return connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

    def runs(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM runs ORDER BY started_at, rowid")

    def evaluations(self, run_id: str | None = None) -> list[sqlite3.Row]:
        if run_id is None:
            return self.query("SELECT * FROM evaluations ORDER BY evaluated_at, rowid")
        return self.query("SELECT * FROM evaluations WHERE run_id = ? ORDER BY evaluated_at, rowid", (run_id,))

    def steps(self, run_id: str | None = None) -> list[sqlite3.Row]:
        if run_id is None:
            return self.query("SELECT * FROM steps ORDER BY run_id, seq")
        return self.query("SELECT * FROM steps WHERE run_id = ? ORDER BY seq", (run_id,))

    def stored_bytes(self) -> bytes:
        """Everything on disk for the database, including its write-ahead log."""
        return b"".join(path.read_bytes() for path in sorted(self.db.parent.iterdir()) if path.is_file())

    async def route(self, task="Review this function for bugs", context="def f(): pass", **arguments):
        return await server.route_local_agent(task=task, context=context, **arguments)


class RecordingTests(ObservabilityCase):
    async def test_ask_records_a_run_with_one_worker_step(self):
        result = await server.ask_local_agent(agent="coder", task="hello", context="abc", max_tokens=50)
        (run,) = self.runs()
        self.assertEqual(result["run_id"], run["id"])
        self.assertEqual((run["tool"], run["status"], run["final_agent"]), ("ask_local_agent", "ok", "coder"))
        self.assertEqual((run["task_chars"], run["context_chars"], run["max_tokens"]), (5, 3, 50))
        (step,) = self.steps()
        self.assertEqual((step["kind"], step["name"], step["provider"], step["model"]), ("worker", "coder", "ollama", "qwen3:8b"))
        self.assertEqual((step["status"], step["finish_reason"], step["truncated"]), ("ok", "stop", 0))
        self.assertEqual((step["input_tokens"], step["output_tokens"]), (10, 5))

    async def test_times_are_utc_stamps_plus_a_monotonic_duration(self):
        await server.ask_local_agent(agent="coder", task="hello")
        for row in (*self.runs(), *self.steps()):
            self.assertTrue(row["started_at"].endswith("+00:00"))
            self.assertTrue(row["ended_at"].endswith("+00:00"))
            self.assertLessEqual(row["started_at"], row["ended_at"])
            self.assertIsInstance(row["elapsed_ms"], int)
            self.assertGreaterEqual(row["elapsed_ms"], 0)
        # The duration comes from the monotonic clock, not from the wall-clock stamps.
        with mock.patch.object(observability, "utc_now", return_value="2000-01-01T00:00:00.000+00:00"):
            with mock.patch.object(observability.time, "monotonic", side_effect=[100.0, 101.5]):
                run = observability.Run("ask_local_agent")
                run.finish("ok")
        (timed,) = self.query("SELECT * FROM runs WHERE id = ?", (run.id,))
        self.assertEqual(timed["elapsed_ms"], 1500)

    async def test_route_keeps_jev_and_worker_timing_and_usage_apart(self):
        result = await self.route()
        (run,) = self.runs()
        self.assertEqual((run["status"], run["routing_source"], run["final_agent"]), ("ok", "jev", "reviewer"))
        self.assertEqual((run["fallback_reason"], run["worker_fallback"], run["empty_responses"]), (None, 0, 0))
        availability, jev, worker = self.steps(result["run_id"])
        self.assertEqual([availability["kind"], jev["kind"], worker["kind"]], ["availability", "jev", "worker"])
        self.assertEqual((jev["status"], jev["model"], jev["confidence"]), ("ok", "jev-latest", 0.8))
        self.assertEqual((jev["input_tokens"], jev["output_tokens"]), (100, 10))
        self.assertEqual((worker["name"], worker["input_tokens"], worker["output_tokens"]), ("reviewer", 10, 5))
        for step in (availability, jev, worker):
            self.assertIsNotNone(step["elapsed_ms"])
        # The reply to the caller is unchanged apart from the run id.
        self.assertNotIn("usage", result["routing"])

    async def test_compare_records_each_worker_and_a_partial_status(self):
        self.local.failing_models = {"qwen3:8b"}
        result = await server.compare_local_agents(agents=["coder", "reviewer"], task="hello")
        (run,) = self.runs()
        self.assertEqual((run["id"], run["status"]), (result["run_id"], "partial"))
        failed, answered = self.steps()
        self.assertEqual((failed["name"], failed["status"], failed["reason"]), ("coder", "error", "worker_error"))
        self.assertEqual((answered["name"], answered["status"]), ("reviewer", "ok"))
        self.local.failing_models = set(OLLAMA_MODELS)
        await server.compare_local_agents(agents=["coder", "reviewer"], task="hello")
        self.assertEqual((self.runs()[-1]["status"], self.runs()[-1]["error_kind"]), ("error", "all_workers_failed"))

    async def test_routing_fallbacks_are_recorded_with_their_reason(self):
        self.use_jev(FakeJev(status=500))
        failed = await self.route()
        self.key_path.unlink()
        no_key = await self.route()
        self.local.models = ["qwen3:4b"]
        only = await self.route()
        cases = (
            (failed, "error", "jev_error:TypeSafeInternalServerError:500", "fallback"),
            (no_key, "skipped", "missing_key", "fallback"),
            (only, "skipped", "only_available", "only_available"),
        )
        for result, status, reason, source in cases:
            jev = self.steps(result["run_id"])[1]
            self.assertEqual((jev["kind"], jev["status"], jev["reason"]), ("jev", status, reason))
            (run,) = self.query("SELECT * FROM runs WHERE id = ?", (result["run_id"],))
            self.assertEqual((run["status"], run["routing_source"]), ("ok", source))
        self.assertEqual(self.runs()[0]["fallback_reason"], "jev_error:TypeSafeInternalServerError:500")

    async def test_empty_answer_and_worker_fallback_are_explicit(self):
        self.local.replies = {"gemma3:4b": reply("", finish_reason="length", completion_tokens=350)}
        self.use_jev(FakeJev(probabilities=RANKED))
        result = await self.route()
        (run,) = self.runs()
        self.assertEqual((run["status"], run["final_agent"]), ("ok", "second_opinion"))
        self.assertEqual((run["worker_fallback"], run["empty_responses"], run["truncated"]), (1, 1, 0))
        empty, answered = self.steps(result["run_id"])[2:]
        self.assertEqual((empty["name"], empty["status"], empty["reason"]), ("reviewer", "error", "empty_response"))
        self.assertEqual((empty["finish_reason"], empty["truncated"], empty["output_tokens"]), ("length", 1, 350))
        self.assertEqual((answered["name"], answered["status"]), ("second_opinion", "ok"))

    async def test_truncated_answer_is_flagged_but_still_a_completed_run(self):
        self.local.replies = {"gemma3:4b": reply("def f(", finish_reason="length", completion_tokens=200)}
        result = await self.route()
        (run,) = self.runs()
        self.assertEqual((run["status"], run["truncated"]), ("ok", 1))
        worker = self.steps(result["run_id"])[-1]
        self.assertEqual((worker["status"], worker["finish_reason"], worker["truncated"]), ("ok", "length", 1))

    async def test_failed_runs_record_what_failed(self):
        self.local.failing_models = set(OLLAMA_MODELS)
        with self.assertRaises(RuntimeError):
            await self.route()
        self.local.models = []
        with self.assertRaises(RuntimeError):
            await self.route()
        self.local.models, self.local.failing_models = OLLAMA_MODELS, set()
        self.local.replies = {"qwen3:8b": reply(None)}
        with self.assertRaises(server.EmptyResponse):
            await server.ask_local_agent(agent="coder", task="hello")
        all_failed, none_available, empty = self.runs()
        self.assertEqual((all_failed["status"], all_failed["error_kind"]), ("error", "all_workers_failed"))
        self.assertEqual([step["status"] for step in self.steps(all_failed["id"])], ["ok", "ok", "error", "error"])
        self.assertEqual((none_available["status"], none_available["error_kind"]), ("error", "no_available_agent"))
        self.assertEqual((empty["status"], empty["error_kind"], empty["empty_responses"]), ("error", "empty_response", 1))
        for run in (all_failed, none_available, empty):
            self.assertIsNotNone(run["ended_at"])

    async def test_project_label_is_self_reported_and_display_only(self):
        plain = await server.ask_local_agent(agent="coder", task="hello")
        labelled = await server.ask_local_agent(agent="coder", task="hello", project="local-agents-mcp")
        with mock.patch.dict(os.environ, {observability.PROJECT_ENV: "from-env"}):
            await server.ask_local_agent(agent="coder", task="hello")
            await server.ask_local_agent(agent="coder", task="hello", project="wins")
        await server.ask_local_agent(agent="coder", task="hello", project="not a label; DROP TABLE runs")
        self.assertEqual(
            [(run["project"], run["project_source"]) for run in self.runs()],
            [("unknown", "none"), ("local-agents-mcp", "argument"), ("from-env", "environment"),
             ("wins", "argument"), ("invalid", "argument")],
        )
        # The label changes nothing about the call or its answer.
        strip = lambda result: {key: value for key, value in result.items() if key != "run_id"}
        self.assertEqual(strip(plain), strip(labelled))
        self.assertEqual(self.local.chats[0], self.local.chats[1])

    async def test_project_label_is_not_sent_to_jev(self):
        await self.route(project="PROJECT-LABEL")
        self.assertNotIn("PROJECT-LABEL", self.jev.requests[0].content.decode())

    async def test_recording_can_be_switched_off(self):
        with mock.patch.dict(os.environ, {observability.ENABLED_ENV: "0"}):
            result = await server.ask_local_agent(agent="coder", task="hello")
        self.assertIsNone(result["run_id"])
        self.assertFalse(self.db.exists())


class InterruptedRunTests(ObservabilityCase):
    async def test_run_and_step_are_stored_before_the_work_finishes(self):
        seen = {}

        def slow_worker(method, url, **arguments):
            if method == "POST":
                seen["run"] = self.runs()[0]
                seen["steps"] = self.steps()
            return self.local(method, url, **arguments)

        with mock.patch.object(server, "_request_json", slow_worker):
            await self.route()
        self.assertEqual((seen["run"]["status"], seen["run"]["ended_at"], seen["run"]["elapsed_ms"]), ("running", None, None))
        self.assertEqual([step["status"] for step in seen["steps"]], ["ok", "ok", "running"])
        self.assertIsNone(seen["steps"][-1]["ended_at"])
        self.assertEqual(self.runs()[0]["status"], "ok")

    async def test_a_run_whose_process_stops_stays_visible_as_running(self):
        # A killed process never reaches its finish calls.
        interrupted = observability.Run("route_local_agent", task="hello")
        interrupted.step("availability", "providers").finish("ok")
        interrupted.step("worker", "coder")
        del interrupted

        # Later calls, a fresh process, and the dashboard must all leave it alone.
        await server.ask_local_agent(agent="coder", task="hello")
        observability._ready.clear()
        observability.connect().close()
        with running_dashboard() as port:
            status, _, listing = fetch(port, "/")
            first = self.runs()[0]
            _, _, detail = fetch(port, f"/runs/{first['id']}")
        self.assertEqual(status, 200)
        first, second = self.runs()
        self.assertEqual((first["status"], first["ended_at"], first["elapsed_ms"]), ("running", None, None))
        self.assertEqual(self.evaluations(), [])
        self.assertEqual([step["status"] for step in self.steps(first["id"])], ["ok", "running"])
        self.assertEqual(second["status"], "ok")
        self.assertIn("running", listing)
        self.assertIn("No end has been recorded", detail)
        self.assertIn("no end recorded", detail)

    async def test_a_cancelled_call_is_never_recorded_as_successful(self):
        async def cancelled(*arguments, **keywords):
            raise asyncio.CancelledError()

        with mock.patch.object(server.asyncio, "to_thread", cancelled):
            with self.assertRaises(asyncio.CancelledError):
                await server.ask_local_agent(agent="coder", task="hello")
        (run,) = self.runs()
        self.assertEqual((run["status"], run["error_kind"]), ("error", "cancelled"))
        self.assertEqual(self.steps()[0]["reason"], "cancelled")


class DatabaseFailureTests(ObservabilityCase):
    async def expected(self):
        """The three tools' answers with recording switched off."""
        with mock.patch.dict(os.environ, {observability.ENABLED_ENV: "0"}):
            return await self.call_all()

    async def call_all(self):
        self.use_jev(FakeJev())
        return (
            await server.ask_local_agent(agent="coder", task="hello"),
            await server.compare_local_agents(agents=["coder", "reviewer"], task="hello"),
            await self.route(),
        )

    async def assert_calls_unaffected(self):
        expected = await self.expected()
        with self.assertLogs(observability.logger, level="WARNING") as captured:
            self.assertEqual(await self.call_all(), expected)
        for record in captured.records:
            self.assertRegex(record.getMessage(), r"^observability [a-z ]+ failed: [A-Za-z]+$")

    async def test_unwritable_location_does_not_break_calls(self):
        blocker = Path(self.tmp.name) / "blocker"
        blocker.write_text("a file where the directory should be")
        with mock.patch.dict(os.environ, {observability.DB_ENV: str(blocker / "obs.sqlite3")}):
            await self.assert_calls_unaffected()

    async def test_corrupt_database_does_not_break_calls(self):
        self.db.parent.mkdir(parents=True)
        self.db.write_bytes(b"this is not a sqlite database" * 100)
        await self.assert_calls_unaffected()

    async def test_newer_schema_is_left_alone_and_does_not_break_calls(self):
        observability.connect().close()
        connection = sqlite3.connect(self.db)
        connection.execute("PRAGMA user_version = 99")
        connection.close()
        observability._ready.clear()
        await self.assert_calls_unaffected()
        self.assertEqual(self.query("PRAGMA user_version")[0][0], 99)
        self.assertEqual(self.runs(), [])

    async def test_a_raising_recorder_does_not_break_calls(self):
        expected = await self.expected()
        with mock.patch.object(observability, "_execute", side_effect=RuntimeError("TASK-TEXT-IN-ERROR")):
            with self.assertLogs(observability.logger, level="WARNING") as captured:
                self.assertEqual(await self.call_all(), expected)
        self.assertNotIn("TASK-TEXT-IN-ERROR", "\n".join(captured.output))

    async def test_a_locked_database_costs_a_bounded_delay(self):
        observability.connect().close()
        holder = sqlite3.connect(self.db, isolation_level=None)
        holder.execute("BEGIN EXCLUSIVE")
        try:
            started = time.monotonic()
            with self.assertLogs(observability.logger, level="WARNING") as captured:
                result = await self.route()
            waited = time.monotonic() - started
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        self.assertEqual(result["response"], "answer from gemma3:4b")
        self.assertIsNone(result["run_id"])
        # A run gives up after MAX_WRITE_FAILURES dropped writes instead of waiting on each one.
        self.assertEqual(len(captured.records), observability.MAX_WRITE_FAILURES)
        self.assertLess(waited, observability.MAX_WRITE_FAILURES * observability.BUSY_TIMEOUT_SECONDS + 1.5)
        self.assertEqual(self.runs(), [])


    async def test_run_id_survives_later_write_failures_but_the_trace_is_partial(self):
        expected = (await self.expected())[2]
        self.use_jev(FakeJev())
        real, calls = observability._execute, []

        def first_write_only(action, sql, parameters):
            calls.append(action)
            return real(action, sql, parameters) if len(calls) == 1 else False

        with mock.patch.object(observability, "_execute", first_write_only):
            result = await self.route()
        # The answer is unaffected and the id of the row that does exist is returned.
        self.assertEqual({**result, "run_id": None}, expected)
        (run,) = self.runs()
        self.assertEqual(result["run_id"], run["id"])
        # The id does not promise a complete recording: the end and the steps were lost.
        self.assertEqual((run["status"], run["ended_at"], run["elapsed_ms"]), ("running", None, None))
        self.assertEqual(self.steps(), [])
        # Recording stopped after the allowed number of dropped writes.
        self.assertEqual(calls, ["run start", "step start", "step end"])
        # The partial trace can still be found and evaluated.
        self.assertIsNotNone(observability.add_evaluation(result["run_id"], "arithmetic", "pass", "manual"))
        self.assertEqual(self.runs()[0]["status"], "running")


class SensitiveDataTests(ObservabilityCase):
    async def test_no_task_context_answer_error_or_key_text_is_stored_or_logged(self):
        secrets = {
            "task": "TASK-SENTINEL-7f3a please review",
            "context": "CONTEXT-SENTINEL-91bc def secret(): pass",
            "answer": "ANSWER-SENTINEL-55de the answer text",
            "error": "ERROR-SENTINEL-c0de",
            "agent": "AGENT SENTINEL e1e1",
            "project": "PROJECT SENTINEL 0a0a",
            "finish": "FINISH SENTINEL 4d4d",
            "key": FAKE_KEY,
        }
        self.local.replies = {
            # The gateway's own error for this reply embeds the whole runtime body.
            "gemma3:4b": reply([secrets["error"]]),
            "qwen3:4b": reply(secrets["answer"], finish_reason=secrets["finish"]),
            "qwen3:8b": reply(secrets["answer"]),
        }
        self.use_jev(FakeJev(probabilities=RANKED))
        with self.assertLogs(level="DEBUG") as captured:
            routed = await self.route(task=secrets["task"], context=secrets["context"], project=secrets["project"])
            self.assertIn(secrets["error"], routed["routing"]["attempts"][0]["error"])
            await server.compare_local_agents(
                agents=["reviewer", "coder"], task=secrets["task"], context=secrets["context"]
            )
            with self.assertRaises(ValueError):
                await server.ask_local_agent(agent=secrets["agent"], task=secrets["task"])
            self.use_jev(FakeJev(status=401))
            await self.route(task=secrets["task"], context=secrets["context"])
            # A failing write must not leak its parameters either.
            with mock.patch.object(observability, "connect", side_effect=sqlite3.OperationalError(secrets["task"])):
                await server.ask_local_agent(agent="coder", task=secrets["task"])
        self.assertEqual(len(self.runs()), 4)
        stored = self.stored_bytes()
        rows = repr([tuple(row) for row in (*self.runs(), *self.steps())])
        ours = "\n".join(record.getMessage() for record in captured.records if record.name == observability.logger.name)
        self.assertIn("OperationalError", ours)
        for name, secret in secrets.items():
            with self.subTest(name):
                for fragment in (secret, secret.split()[0]):
                    self.assertNotIn(fragment.encode(), stored)
                    self.assertNotIn(fragment, rows)
                    self.assertNotIn(fragment, ours)
        # What is kept instead: sizes and neutral placeholders.
        self.assertEqual(self.runs()[0]["task_chars"], len(secrets["task"]))
        self.assertEqual(self.runs()[0]["project"], "invalid")
        self.assertIn("unknown", [step["name"] for step in self.steps()])
        self.assertIn("invalid", [step["finish_reason"] for step in self.steps()])

    def test_schema_has_no_text_columns_for_content(self):
        observability.connect().close()
        tables = ("runs", "steps", "evaluations")
        columns = {row["name"] for table in tables for row in self.query(f"PRAGMA table_info({table})")}
        self.assertFalse(
            columns & {"task", "context", "response", "output", "prompt", "error", "message", "api_key", "note", "notes", "comment"}
        )
        self.assertEqual(
            [row["name"] for row in self.query("PRAGMA table_info(evaluations)")],
            ["id", "run_id", "scope", "result", "method", "method_version", "evaluated_at"],
        )

    def test_database_and_directory_are_private(self):
        observability.connect().close()
        self.assertEqual(self.db.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.db.parent.stat().st_mode & 0o777, 0o700)


class EvaluationTests(ObservabilityCase):
    async def test_a_successful_run_has_no_evaluation_until_one_is_added(self):
        result = await server.ask_local_agent(agent="coder", task="hello")
        self.assertEqual(self.runs()[0]["status"], "ok")
        self.assertEqual(self.evaluations(), [])
        self.assertFalse([name for name in self.runs()[0].keys() if name.startswith("eval")])

        evaluation_id = observability.add_evaluation(result["run_id"], "arithmetic", "fail", "manual")
        (stored,) = self.evaluations()
        self.assertEqual(
            (stored["id"], stored["run_id"], stored["scope"], stored["result"], stored["method"], stored["method_version"]),
            (evaluation_id, result["run_id"], "arithmetic", "fail", "manual", None),
        )
        self.assertTrue(stored["evaluated_at"].endswith("+00:00"))
        # A failed check does not rewrite how the call itself went.
        self.assertEqual(self.runs()[0]["status"], "ok")

    async def test_a_failed_run_can_hold_a_passing_check(self):
        self.local.failing_models = {"qwen3:8b"}
        with self.assertRaises(RuntimeError):
            await server.ask_local_agent(agent="coder", task="hello")
        run_id = self.runs()[0]["id"]
        observability.add_evaluation(run_id, "coverage", "pass", "manual")
        self.assertEqual(self.runs()[0]["status"], "error")
        self.assertEqual(self.evaluations()[0]["result"], "pass")

    async def test_multiple_scopes_are_kept_side_by_side(self):
        run_id = (await server.ask_local_agent(agent="coder", task="hello"))["run_id"]
        other = (await server.ask_local_agent(agent="reviewer", task="hello"))["run_id"]
        observability.add_evaluation(run_id, "arithmetic", "pass", "manual")
        observability.add_evaluation(run_id, "coverage", "fail", "checker", "v2")
        observability.add_evaluation(run_id, "label_accuracy", "partial", "checker", "v2")
        observability.add_evaluation(run_id, "my-own-check", "pass", "manual")
        observability.add_evaluation(other, "arithmetic", "fail", "manual")
        self.assertEqual(
            [(item["scope"], item["result"], item["method"], item["method_version"]) for item in self.evaluations(run_id)],
            [("arithmetic", "pass", "manual", None), ("coverage", "fail", "checker", "v2"),
             ("label_accuracy", "partial", "checker", "v2"), ("my-own-check", "pass", "manual", None)],
        )
        self.assertEqual([(item["scope"], item["result"]) for item in self.evaluations(other)], [("arithmetic", "fail")])

    async def test_repeated_evaluations_of_one_scope_are_all_preserved(self):
        run_id = (await server.ask_local_agent(agent="coder", task="hello"))["run_id"]
        ids = [
            observability.add_evaluation(run_id, "arithmetic", "fail", "checker", "v1"),
            observability.add_evaluation(run_id, "arithmetic", "pass", "checker", "v2"),
            observability.add_evaluation(run_id, "arithmetic", "pass", "checker", "v2"),
        ]
        self.assertEqual(len(set(ids)), 3)
        history = self.evaluations(run_id)
        self.assertEqual([item["id"] for item in history], ids)
        self.assertEqual([(item["result"], item["method_version"]) for item in history], [("fail", "v1"), ("pass", "v2"), ("pass", "v2")])
        self.assertEqual(sorted(item["evaluated_at"] for item in history), [item["evaluated_at"] for item in history])

    async def test_only_short_labels_are_accepted(self):
        run_id = (await server.ask_local_agent(agent="coder", task="hello"))["run_id"]
        prose = "NOTE-SENTINEL the answer leaked a customer name"
        rejected = (
            ("", "pass", "manual", ""), ("two words", "pass", "manual", ""), ("arithmetic", "correct", "manual", ""),
            ("arithmetic", "unevaluated", "manual", ""), ("arithmetic", "pass", "", ""), ("arithmetic", "pass", prose, ""),
            ("arithmetic", "pass", "manual", prose), (prose, "pass", "manual", ""),
        )
        for scope, result, method, version in rejected:
            with self.subTest(scope=scope, result=result, method=method, version=version):
                with self.assertRaises(ValueError) as raised:
                    observability.add_evaluation(run_id, scope, result, method, version)
                self.assertNotIn("SENTINEL", str(raised.exception))
        self.assertEqual(self.evaluations(), [])
        self.assertNotIn(b"NOTE-SENTINEL", self.stored_bytes())
        self.assertIsNone(observability.add_evaluation("0" * 32, "arithmetic", "pass", "manual"))
        self.assertEqual(self.evaluations(), [])

    async def test_recording_and_evaluating_do_not_touch_each_other(self):
        self.assertFalse([name for name in observability.RUN_FIELDS | observability.STEP_FIELDS if name.startswith("eval")])
        result = await self.route()
        before = [tuple(row) for row in (*self.runs(), *self.steps())]
        observability.add_evaluation(result["run_id"], "arithmetic", "pass", "manual")
        self.assertEqual([tuple(row) for row in (*self.runs(), *self.steps())], before)
        await self.route()
        self.assertEqual(len(self.evaluations()), 1)


class running_dashboard:
    """Serve the dashboard on a free localhost port for the duration of a with block."""

    def __enter__(self) -> int:
        self.server = dashboard.create_server(0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.server.server_address[1]

    def __exit__(self, *error) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def fetch(port, path, *, method="GET", host=None, origin=None, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.putrequest(method, path, skip_host=True)
        connection.putheader("Host", host or f"127.0.0.1:{port}")
        if origin:
            connection.putheader("Origin", origin)
        data = (body or "").encode()
        if method == "POST":
            connection.putheader("Content-Type", "application/x-www-form-urlencoded")
            connection.putheader("Content-Length", str(len(data)))
        connection.endheaders(data if method == "POST" else None)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read().decode()
    finally:
        connection.close()


class DashboardTests(ObservabilityCase):
    async def asyncSetUp(self):
        self.result = await self.route(project="local-agents-mcp")
        self.run_id = self.result["run_id"]
        context = running_dashboard()
        self.port = context.__enter__()
        self.addCleanup(context.__exit__)

    def evaluate(self, **arguments):
        return fetch(self.port, f"/runs/{self.run_id}/evaluations", method="POST", **arguments)

    def test_it_binds_to_the_loopback_interface_only(self):
        self.assertEqual(dashboard.HOST, "127.0.0.1")
        with running_dashboard() as port:
            pass
        server_ = dashboard.create_server(0)
        self.addCleanup(server_.server_close)
        self.assertEqual(server_.server_address[0], "127.0.0.1")
        self.assertEqual(dashboard.DEFAULT_PORT, 8001)

    def test_runs_and_run_detail_render(self):
        status, headers, listing = fetch(self.port, "/")
        self.assertEqual(status, 200)
        self.assertIn(f"/runs/{self.run_id}", listing)
        self.assertIn("local-agents-mcp", listing)
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        status, _, detail = fetch(self.port, f"/runs/{self.run_id}")
        self.assertEqual(status, 200)
        for expected in ("Step timeline", "Jev time", "Worker tokens", "100 in / 10 out", "self-reported", "Evaluation history",
                         "No evaluations recorded"):
            self.assertIn(expected, detail)
        # One bar per step, coloured by kind.
        self.assertEqual(detail.count('class="bar '), 3)
        for kind in ("availability", "jev", "worker"):
            self.assertIn(f'class="bar {kind}"', detail)
        self.assertEqual(fetch(self.port, "/runs/" + "0" * 32)[0], 404)
        self.assertEqual(fetch(self.port, "/elsewhere")[0], 404)

    def test_filters_narrow_the_list(self):
        self.assertIn(self.run_id, fetch(self.port, "/?project=local-agents-mcp&status=ok")[2])
        self.assertNotIn(self.run_id, fetch(self.port, "/?project=other")[2])
        self.assertNotIn(self.run_id, fetch(self.port, "/?status=error")[2])

    def test_other_hosts_are_refused(self):
        for host in ("mcp.ag-mcp.xyz", f"evil.example:{self.port}", "127.0.0.1", f"127.0.0.1:{self.port + 1}", ""):
            with self.subTest(host=host):
                status, _, body = fetch(self.port, "/", host=host or " ")
                self.assertEqual(status, 403)
                self.assertNotIn(self.run_id, body)
        self.assertEqual(fetch(self.port, "/", host=f"localhost:{self.port}")[0], 200)

    def test_evaluation_needs_a_same_origin_post(self):
        form = "scope=arithmetic&result=pass&method=manual"
        for origin in (None, "https://evil.example", "null", f"http://127.0.0.1:{self.port + 1}"):
            with self.subTest(origin=origin):
                self.assertEqual(self.evaluate(origin=origin, body=form)[0], 403)
        self.assertEqual(self.evaluate(origin=f"http://127.0.0.1:{self.port}", host="evil.example", body=form)[0], 403)
        self.assertEqual(self.evaluations(), [])

        status, headers, _ = self.evaluate(origin=f"http://127.0.0.1:{self.port}", body=form)
        self.assertEqual((status, headers["Location"]), (303, f"/runs/{self.run_id}"))
        (stored,) = self.evaluations()
        self.assertEqual((stored["scope"], stored["result"], stored["method"]), ("arithmetic", "pass", "manual"))
        self.assertEqual(self.runs()[0]["status"], "ok")

    def test_evaluation_history_is_shown_apart_from_execution(self):
        connection = sqlite3.connect(self.db)
        connection.execute("UPDATE runs SET status = 'error', error_kind = 'all_workers_failed'")
        connection.commit()
        connection.close()
        origin = f"http://127.0.0.1:{self.port}"
        for form in ("scope=arithmetic&result=fail&method=checker&method_version=v1",
                     "scope=arithmetic&result=pass&method=checker&method_version=v2",
                     "scope=coverage&result=partial&method=manual"):
            self.assertEqual(self.evaluate(origin=origin, body=form)[0], 303)
        detail = fetch(self.port, f"/runs/{self.run_id}")[2]
        execution, history = detail.split("<h2>Evaluation history</h2>")
        history = history.split("<h2>Add an evaluation</h2>")[0]
        # Execution is reported in its own panel and is unchanged by any evaluation.
        self.assertIn("✕ error", execution.split("<h2>Execution</h2>")[1].split("</dl>")[0])
        self.assertNotIn("✓ pass", execution)
        # All three rows are listed, newest first, including the repeated scope.
        self.assertEqual(history.count("<tr>"), 4)
        self.assertLess(history.index("coverage"), history.index("v2"))
        self.assertLess(history.index("v2"), history.index("v1"))
        self.assertIn("no overall verdict", history)
        # The list shows the latest result per scope, not a combined verdict.
        row = fetch(self.port, "/")[2].split(f'/runs/{self.run_id}"')[1].split("</tr>")[0]
        self.assertIn("✕ error", row)
        self.assertIn("arithmetic", row)
        self.assertIn("coverage", row)
        self.assertEqual((row.count("✓ pass"), row.count("◐ partial"), row.count("✕ fail")), (1, 1, 0))

    def test_an_unevaluated_run_is_shown_as_unchecked(self):
        self.assertIn("none recorded", fetch(self.port, "/")[2])
        self.assertIn("This answer has not been checked", fetch(self.port, f"/runs/{self.run_id}")[2])

    def test_incomplete_evaluation_forms_are_not_saved(self):
        origin = f"http://127.0.0.1:{self.port}"
        for form in ("scope=&result=pass&method=manual", "scope=two+words&result=pass&method=manual",
                     "scope=arithmetic&result=pass&method=", "scope=arithmetic&result=great&method=manual"):
            with self.subTest(form=form):
                status, headers, _ = self.evaluate(origin=origin, body=form)
                self.assertEqual((status, headers["Location"]), (303, f"/runs/{self.run_id}?error=1"))
        self.assertIn("Not saved", fetch(self.port, f"/runs/{self.run_id}?error=1")[2])
        self.assertEqual(self.evaluate(origin=origin, body="x" * 5000)[0], 413)
        self.assertEqual(self.evaluations(), [])

    def test_stored_values_are_escaped(self):
        connection = sqlite3.connect(self.db)
        connection.execute("UPDATE runs SET project = ?, final_agent = ?", ("<script>alert(1)</script>", '"><img src=x>'))
        connection.execute("UPDATE steps SET name = ?", ("<b>bold</b>",))
        connection.execute(
            "INSERT INTO evaluations VALUES ('e1', ?, '<b>bold</b>', 'pass', '<script>x</script>', NULL, 'not-a-time')",
            (self.run_id,),
        )
        connection.commit()
        connection.close()
        for path in ("/", f"/runs/{self.run_id}"):
            body = fetch(self.port, path)[2]
            self.assertNotIn("<script>", body)
            self.assertNotIn("<img", body)
            self.assertNotIn("<b>bold", body)
            self.assertIn("&lt;script&gt;", body)

    def test_reading_pages_does_not_change_the_database(self):
        before = [tuple(row) for row in (*self.runs(), *self.steps())]
        fetch(self.port, "/")
        fetch(self.port, f"/runs/{self.run_id}")
        self.assertEqual([tuple(row) for row in (*self.runs(), *self.steps())], before)
        with self.assertRaises(sqlite3.OperationalError):
            observability.connect_readonly().execute("DELETE FROM runs")

    def test_missing_database_shows_an_empty_page(self):
        with mock.patch.dict(os.environ, {observability.DB_ENV: str(Path(self.tmp.name) / "none.sqlite3")}):
            status, _, body = fetch(self.port, "/")
        self.assertEqual(status, 200)
        self.assertIn("No run database yet", body)


class PreservationTests(ObservabilityCase):
    def old_database(self):
        """A version 1 database, as written before evaluations had their own table."""
        with mock.patch.object(observability, "MIGRATIONS", observability.MIGRATIONS[:1]):
            first = observability.Run("ask_local_agent", task="hello")
            first.step("worker", "coder").finish("ok", input_tokens=10, output_tokens=5)
            first.finish("ok", final_agent="coder")
            second = observability.Run("route_local_agent", task="hello")
            second.finish("error", error_kind="all_workers_failed")
            third = observability.Run("ask_local_agent", task="hello")
            third.finish("ok")
            interrupted = observability.Run("ask_local_agent", task="hello")
        connection = sqlite3.connect(self.db)
        for run, status, scope, stamp in (
            (first, "pass", "arithmetic", "2026-10-03T10:00:00.000+00:00"),
            (second, "fail", "coverage", "2026-10-03T11:00:00.000+00:00"),
        ):
            connection.execute(
                "UPDATE runs SET eval_status = ?, eval_scope = ?, eval_updated_at = ? WHERE id = ?",
                (status, scope, stamp, run.id),
            )
        connection.commit()
        self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
        connection.close()
        observability._ready.clear()
        return first, second, third, interrupted

    async def test_old_evaluation_columns_migrate_into_the_table_without_loss(self):
        first, second, third, interrupted = self.old_database()
        old_runs = self.query("SELECT id, started_at, ended_at, elapsed_ms, tool, status, error_kind, final_agent FROM runs ORDER BY rowid")
        old_steps = [tuple(row) for row in self.steps()]

        # The first call made by the upgraded gateway triggers the migration.
        result = await server.ask_local_agent(agent="reviewer", task="hello")

        self.assertEqual(self.query("PRAGMA user_version")[0][0], len(observability.MIGRATIONS))
        kept = self.query("SELECT id, started_at, ended_at, elapsed_ms, tool, status, error_kind, final_agent FROM runs ORDER BY rowid")
        self.assertEqual([tuple(row) for row in kept[:4]], [tuple(row) for row in old_runs])
        self.assertEqual(kept[4]["id"], result["run_id"])
        self.assertEqual([tuple(row) for row in self.steps() if row["run_id"] != result["run_id"]], old_steps)
        self.assertEqual(self.query("SELECT status FROM runs WHERE id = ?", (interrupted.id,))[0]["status"], "running")
        self.assertFalse([name for name in self.runs()[0].keys() if name.startswith("eval")])
        self.assertEqual(
            [(item["run_id"], item["scope"], item["result"], item["method"], item["method_version"], item["evaluated_at"])
             for item in self.evaluations()],
            [(first.id, "arithmetic", "pass", "migrated_v1", None, "2026-10-03T10:00:00.000+00:00"),
             (second.id, "coverage", "fail", "migrated_v1", None, "2026-10-03T11:00:00.000+00:00")],
        )
        self.assertEqual(len({item["id"] for item in self.evaluations()}), 2)
        # Runs that were never evaluated gain no row, and a new check adds to the migrated one.
        self.assertEqual(self.evaluations(third.id), [])
        observability.add_evaluation(first.id, "arithmetic", "fail", "checker", "v2")
        self.assertEqual([item["result"] for item in self.evaluations(first.id)], ["pass", "fail"])

        # The pre-migration snapshot still holds the old columns and values.
        backup = self.db.with_name(self.db.name + ".v1.bak")
        saved = sqlite3.connect(backup)
        self.assertEqual(saved.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertEqual(saved.execute("PRAGMA user_version").fetchone()[0], 1)
        self.assertEqual(
            saved.execute("SELECT id, eval_status, eval_scope FROM runs ORDER BY rowid").fetchall(),
            [(first.id, "pass", "arithmetic"), (second.id, "fail", "coverage"),
             (third.id, "unevaluated", None), (interrupted.id, "unevaluated", None)],
        )
        self.assertEqual(saved.execute("SELECT name FROM sqlite_master WHERE name = 'evaluations'").fetchall(), [])
        saved.close()

    async def test_a_failed_migration_changes_nothing_and_does_not_break_calls(self):
        first, *_ = self.old_database()
        broken = (*observability.MIGRATIONS[1][:4], "ALTER TABLE runs DROP COLUMN no_such_column")
        with mock.patch.object(observability, "MIGRATIONS", [observability.MIGRATIONS[0], broken]):
            with self.assertLogs(observability.logger, level="WARNING"):
                result = await server.ask_local_agent(agent="reviewer", task="hello")
        self.assertEqual(result["response"], "answer from gemma3:4b")
        self.assertIsNone(result["run_id"])
        self.assertEqual(self.query("PRAGMA user_version")[0][0], 1)
        self.assertEqual(self.query("SELECT eval_status, eval_scope FROM runs WHERE id = ?", (first.id,))[0][:], ("pass", "arithmetic"))
        self.assertEqual(self.query("SELECT name FROM sqlite_master WHERE name = 'evaluations'"), [])
        # The real migration still succeeds afterwards.
        observability._ready.clear()
        await server.ask_local_agent(agent="reviewer", task="hello")
        self.assertEqual([item["scope"] for item in self.evaluations()], ["arithmetic", "coverage"])

    async def test_migration_backs_up_with_the_sqlite_backup_api_while_the_database_is_in_use(self):
        run_id = (await server.ask_local_agent(agent="coder", task="hello"))["run_id"]
        evaluation_id = observability.add_evaluation(run_id, "arithmetic", "pass", "manual")
        current = len(observability.MIGRATIONS)
        observability._ready.clear()
        upgrade = ("ALTER TABLE runs ADD COLUMN added_later INTEGER",)
        with mock.patch.object(observability, "MIGRATIONS", [*observability.MIGRATIONS, upgrade]):
            with mock.patch.object(shutil, "copy2", side_effect=AssertionError("file copy used")):
                with mock.patch.object(shutil, "copyfile", side_effect=AssertionError("file copy used")):
                    with mock.patch.object(observability, "_backup_before_migration", wraps=observability._backup_before_migration) as backed_up:
                        # A second connection is mid-use while the upgrade runs.
                        reader = observability.connect_readonly()
                        reader.execute("SELECT * FROM runs").fetchall()
                        await server.ask_local_agent(agent="reviewer", task="hello")
                        reader.close()
        self.assertEqual(backed_up.call_count, 1)
        backup = self.db.with_name(f"{self.db.name}.v{current}.bak")
        self.assertTrue(backup.is_file())
        self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
        self.assertEqual([path.name for path in self.db.parent.glob("*.tmp")], [])
        saved = sqlite3.connect(backup)
        self.assertEqual(saved.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertEqual(saved.execute("PRAGMA user_version").fetchone()[0], current)
        self.assertEqual([row[0] for row in saved.execute("SELECT id FROM runs")], [run_id])
        self.assertEqual([row[0] for row in saved.execute("SELECT id FROM evaluations")], [evaluation_id])
        saved.close()
        self.assertEqual(self.query("PRAGMA user_version")[0][0], current + 1)
        self.assertEqual([run["final_agent"] for run in self.runs()], ["coder", "reviewer"])
        self.assertEqual([item["id"] for item in self.evaluations()], [evaluation_id])
        self.assertIn("added_later", self.runs()[0].keys())

    async def test_reopening_keeps_existing_rows(self):
        await server.ask_local_agent(agent="coder", task="hello")
        observability._ready.clear()
        await server.ask_local_agent(agent="reviewer", task="hello")
        self.assertEqual(len(self.runs()), 2)
        self.assertEqual(list(self.db.parent.glob("*.bak")), [])

    def test_default_database_is_outside_the_deployed_runtime(self):
        relative = observability.DEFAULT_DB_PATH.relative_to(Path.home())
        self.assertEqual(relative, Path("Library/Application Support/local-ai-agents/data/observability.sqlite3"))
        self.assertNotIn("runtime", relative.parts)

    async def test_deployment_preserves_the_database(self):
        home = Path(self.tmp.name) / "home"
        (home / "Library/Logs").mkdir(parents=True)
        database = home / observability.DEFAULT_DB_PATH.relative_to(Path.home())
        with mock.patch.dict(os.environ, {observability.DB_ENV: str(database)}):
            run_id = (await server.ask_local_agent(agent="coder", task="hello"))["run_id"]
            observability.add_evaluation(run_id, "arithmetic", "pass", "manual")
        digest = hashlib.sha256(database.read_bytes()).hexdigest()

        # A stand-in checkout, so no real token or key is read or copied.
        checkout = Path(self.tmp.name) / "checkout"
        for folder in ("scripts", "src/local_agents_mcp", "config", ".local"):
            (checkout / folder).mkdir(parents=True)
        shutil.copy(REPO / "scripts/deploy_macos.py", checkout / "scripts/deploy_macos.py")
        for name in ("pyproject.toml", "uv.lock", "README.md", "scripts/serve_http.py", "config/agents.json",
                     "src/local_agents_mcp/server.py", ".local/mcp_token"):
            (checkout / name).write_text("placeholder\n")
        spec = importlib.util.spec_from_file_location("deploy_under_test", checkout / "scripts/deploy_macos.py")
        deploy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deploy)

        commands = []
        with mock.patch.object(Path, "home", return_value=home):
            with mock.patch.object(deploy.subprocess, "run", side_effect=lambda command, **_: commands.append(command)):
                with mock.patch.object(deploy.shutil, "which", return_value="/fake/uv"), contextlib.redirect_stdout(io.StringIO()):
                    deploy.main()
                    deploy.main()  # A redeploy over an existing managed runtime.

        runtime = home / "Library/Application Support/local-ai-agents/runtime"
        self.assertTrue((runtime / "src/local_agents_mcp/server.py").is_file())
        self.assertEqual({command[0] for command in commands}, {"launchctl", "/fake/uv"})
        self.assertEqual(hashlib.sha256(database.read_bytes()).hexdigest(), digest)
        self.assertEqual(database.parent.stat().st_mode & 0o777, 0o700)
        self.assertFalse(list(runtime.rglob("*.sqlite3")))
        connection = sqlite3.connect(database)
        self.assertEqual(connection.execute("SELECT id, status FROM runs").fetchall(), [(run_id, "ok")])
        self.assertEqual(
            connection.execute("SELECT run_id, scope, result FROM evaluations").fetchall(), [(run_id, "arithmetic", "pass")]
        )
        connection.close()


if __name__ == "__main__":
    unittest.main()
