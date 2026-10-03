"""Offline tests for route_local_agent. Jev and the local runtimes are both mocked."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx2
from mcp import Client

from local_agents_mcp import routing, server


FAKE_KEY = "test-key-not-real"
OLLAMA_MODELS = ["qwen3:8b", "gemma3:4b", "qwen3:4b"]


class FakeLocal:
    """Stands in for server._request_json: Ollama is up, llama.cpp and MLX are down."""

    def __init__(self, models=OLLAMA_MODELS, failing_models=()):
        self.models = list(models)
        self.failing_models = set(failing_models)
        # model -> the whole chat completion body to return instead of the default.
        self.replies: dict[str, dict] = {}
        self.chats: list[dict] = []

    def __call__(self, method, url, *, api_key=None, payload=None, timeout=180):
        if "11434" not in url:
            raise RuntimeError(f"Could not reach local model API at {url}: refused")
        if method == "GET":
            return {"data": [{"id": model} for model in self.models]}
        self.chats.append(payload)
        if payload["model"] in self.failing_models:
            raise RuntimeError("Local model API returned HTTP 500: boom")
        if payload["model"] in self.replies:
            return self.replies[payload["model"]]
        return reply(f"answer from {payload['model']}")


def reply(content, finish_reason="stop", completion_tokens=5, **message):
    """A chat completion body shaped like Ollama's /v1/chat/completions."""
    return {
        "choices": [{"message": {"role": "assistant", "content": content, **message}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": completion_tokens, "total_tokens": 10 + completion_tokens},
    }


class FakeJev:
    """An httpx2 transport that records requests and replies as the TypeSafe API would."""

    def __init__(self, choice="reviewer", probabilities=None, status=200, raises=None):
        self.choice = choice
        self.probabilities = probabilities
        self.status = status
        self.raises = raises
        self.requests: list[httpx2.Request] = []

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.raises:
            raise self.raises("mocked network failure", request=request)
        if self.status != 200:
            return httpx2.Response(self.status, json={"error": "mocked failure"})
        options = list(json.loads(request.content)["questions"]["worker"]["criteria"])
        probabilities = self.probabilities or {
            name: (0.9 if name == self.choice else 0.1 / max(len(options) - 1, 1)) for name in options
        }
        return httpx2.Response(
            200,
            json={
                "model": "jev-test",
                "answers": {
                    "worker": {
                        "type": "choice",
                        "choice": self.choice,
                        "confidence": 0.8,
                        "probabilities": probabilities,
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
        )

    def body(self, index=0) -> dict:
        return json.loads(self.requests[index].content)


class RouteLocalAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.key_path = Path(self.tmp.name) / "typesafe_api_key"
        self.key_path.write_text(FAKE_KEY + "\n", encoding="utf-8")
        self.local = FakeLocal()
        self.jev = FakeJev()
        self.config = server._load_config()
        # Keep retries instant; the real key file and environment are never touched.
        self.config["routing"]["jev"]["timeout_seconds"] = 0.5
        for patcher in (
            mock.patch.object(routing, "KEY_PATH", self.key_path),
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(server, "_request_json", self.local),
            mock.patch.object(server, "_load_config", lambda: self.config),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop(routing.KEY_ENV, None)
        self.use_jev(self.jev)

    def use_jev(self, jev: FakeJev) -> None:
        self.jev = jev
        patcher = mock.patch.object(routing, "_transport", jev.transport())
        patcher.start()
        self.addCleanup(patcher.stop)

    async def route(self, task="Review this function for bugs", context="def f(): pass"):
        return await server.route_local_agent(task=task, context=context)

    async def test_jev_selection_runs_the_chosen_worker(self):
        result = await self.route()
        self.assertEqual(result["agent"], "reviewer")
        self.assertEqual(result["response"], "answer from gemma3:4b")
        info = result["routing"]
        self.assertEqual((info["chosen"], info["first_choice"], info["source"]), ("reviewer", "reviewer", "jev"))
        self.assertEqual(info["confidence"], 0.8)
        self.assertAlmostEqual(sum(info["probabilities"].values()), 1.0)
        self.assertIsNone(info["fallback_reason"])
        self.assertEqual(info["attempts"], [{"agent": "reviewer", "ok": True}])

    async def test_only_the_task_reaches_jev_and_context_reaches_the_worker(self):
        await self.route(task="Review this function", context="SECRET-CONTEXT")
        self.assertEqual(len(self.jev.requests), 1)
        self.assertEqual(self.jev.body()["state"], {"task": "Review this function"})
        self.assertNotIn("SECRET-CONTEXT", self.jev.requests[0].content.decode())
        self.assertIn("SECRET-CONTEXT", self.local.chats[0]["messages"][1]["content"])

    async def test_send_context_opt_in_sends_a_capped_excerpt(self):
        self.config["routing"]["jev"].update(send_context=True, context_char_limit=5)
        await self.route(context="0123456789")
        self.assertEqual(self.jev.body()["state"]["context"], "01234")

    async def test_unavailable_workers_are_not_offered_to_jev(self):
        result = await self.route()
        self.assertEqual(sorted(self.jev.body()["questions"]["worker"]["criteria"]), ["coder", "reviewer", "second_opinion"])
        self.assertEqual(result["routing"]["available"], ["coder", "reviewer", "second_opinion"])
        self.assertIn("not reachable", result["routing"]["unavailable"]["horizon"])

    async def test_missing_model_makes_a_worker_unavailable(self):
        self.local.models = ["qwen3:8b", "qwen3:4b"]
        self.use_jev(FakeJev(choice="coder"))
        result = await self.route()
        self.assertNotIn("reviewer", self.jev.body()["questions"]["worker"]["criteria"])
        self.assertIn("does not list model", result["routing"]["unavailable"]["reviewer"])

    async def test_single_available_worker_skips_jev(self):
        self.local.models = ["qwen3:4b"]
        result = await self.route()
        self.assertEqual(self.jev.requests, [])
        self.assertEqual((result["agent"], result["routing"]["source"]), ("second_opinion", "only_available"))

    async def test_no_available_worker_raises_without_calling_jev(self):
        self.local.models = []
        with self.assertRaisesRegex(RuntimeError, "No routable local agent is available"):
            await self.route()
        self.assertEqual(self.jev.requests, [])

    async def test_missing_key_falls_back_without_calling_jev(self):
        self.key_path.unlink()
        result = await self.route()
        self.assertEqual(self.jev.requests, [])
        info = result["routing"]
        self.assertEqual((info["source"], info["fallback_reason"], info["chosen"]), ("fallback", "missing_key", "coder"))
        self.assertIsNone(info["confidence"])
        self.assertIsNone(info["probabilities"])

    async def test_empty_key_file_counts_as_missing(self):
        self.key_path.write_text("\n", encoding="utf-8")
        self.assertEqual((await self.route())["routing"]["fallback_reason"], "missing_key")

    async def test_invalid_choice_falls_back(self):
        # horizon is configured but its runtime is down, so it was never offered.
        self.use_jev(FakeJev(choice="horizon", probabilities={"horizon": 1.0}))
        info = (await self.route())["routing"]
        self.assertEqual((info["source"], info["fallback_reason"], info["chosen"]), ("fallback", "invalid_choice", "coder"))

    async def test_api_error_falls_back_after_bounded_retries(self):
        self.use_jev(FakeJev(status=500))
        info = (await self.route())["routing"]
        self.assertEqual(info["source"], "fallback")
        self.assertEqual(info["fallback_reason"], "jev_error:TypeSafeInternalServerError:500")
        self.assertEqual(info["chosen"], "coder")
        self.assertLessEqual(len(self.jev.requests), 1 + self.config["routing"]["jev"]["max_retries"])

    async def test_auth_error_falls_back_without_retry(self):
        self.use_jev(FakeJev(status=401))
        info = (await self.route())["routing"]
        self.assertEqual(info["fallback_reason"], "jev_error:TypeSafeAuthenticationError:401")
        self.assertEqual(len(self.jev.requests), 1)

    async def test_timeout_falls_back(self):
        self.use_jev(FakeJev(raises=httpx2.ConnectTimeout))
        info = (await self.route())["routing"]
        self.assertEqual((info["source"], info["fallback_reason"]), ("fallback", "jev_error:TypeSafeAPITimeoutError"))

    async def test_the_key_never_appears_in_results_or_logs(self):
        self.use_jev(FakeJev(status=401))
        with self.assertLogs(level="DEBUG") as captured:
            # assertLogs fails on silence, so emit one record of our own.
            logging.getLogger(__name__).debug("capturing")
            result = await self.route()
        self.assertEqual(self.jev.requests[0].headers["authorization"], f"Bearer {FAKE_KEY}")
        self.assertNotIn(FAKE_KEY, json.dumps(result))
        self.assertNotIn(FAKE_KEY, "\n".join(captured.output))

    async def test_failed_worker_is_retried_on_the_next_ranked_worker(self):
        self.local.failing_models = {"gemma3:4b"}
        self.use_jev(FakeJev(probabilities={"reviewer": 0.6, "second_opinion": 0.3, "coder": 0.1}))
        result = await self.route()
        info = result["routing"]
        self.assertEqual((info["first_choice"], info["chosen"]), ("reviewer", "second_opinion"))
        self.assertEqual([attempt["agent"] for attempt in info["attempts"]], ["reviewer", "second_opinion"])
        self.assertFalse(info["attempts"][0]["ok"])

    async def test_worker_attempts_stop_at_the_configured_limit(self):
        self.local.failing_models = set(OLLAMA_MODELS)
        with self.assertRaisesRegex(RuntimeError, "All 2 worker attempts failed"):
            await self.route()
        self.assertEqual(len(self.local.chats), 2)

    async def test_empty_worker_answers_fall_back_to_the_next_worker(self):
        # What qwen3 returned when thinking used the whole budget: no content, only reasoning.
        thinking_only = reply("", finish_reason="length", completion_tokens=350, reasoning="Let me think...")
        no_content_key = {"choices": [{"message": {"role": "assistant"}, "finish_reason": "stop"}], "usage": None}
        cases = {
            "empty": thinking_only,
            "whitespace": reply(" \n\t "),
            "null": reply(None),
            "missing": no_content_key,
        }
        for label, body in cases.items():
            with self.subTest(label):
                self.local.chats.clear()
                self.local.replies = {"gemma3:4b": body}
                self.use_jev(FakeJev(probabilities={"reviewer": 0.6, "second_opinion": 0.3, "coder": 0.1}))
                result = await self.route()
                info = result["routing"]
                self.assertEqual((info["first_choice"], info["chosen"]), ("reviewer", "second_opinion"))
                self.assertEqual(result["response"], "answer from qwen3:4b")
                failed, succeeded = info["attempts"]
                self.assertEqual((failed["agent"], failed["ok"], failed["reason"]), ("reviewer", False, "empty_response"))
                self.assertTrue(failed["error"].startswith("empty_response: gemma3:4b returned no answer text"))
                self.assertEqual(succeeded, {"agent": "second_opinion", "ok": True})

    async def test_empty_response_reason_reports_the_token_limit(self):
        self.local.replies = {"gemma3:4b": reply("", finish_reason="length", completion_tokens=350)}
        failed = (await self.route())["routing"]["attempts"][0]
        self.assertIn("finish_reason=length", failed["error"])
        self.assertIn("completion_tokens=350", failed["error"])

    async def test_empty_answers_stop_at_the_configured_attempt_limit(self):
        self.local.replies = {model: reply("") for model in OLLAMA_MODELS}
        with self.assertRaisesRegex(RuntimeError, "All 2 worker attempts failed.*empty_response"):
            await self.route()
        self.assertEqual(len(self.local.chats), 2)

    async def test_other_worker_failures_are_not_labelled_empty(self):
        self.local.failing_models = {"gemma3:4b"}
        failed = (await self.route())["routing"]["attempts"][0]
        self.assertEqual(failed["reason"], "worker_error")
        self.assertIn("HTTP 500", failed["error"])

    async def test_direct_and_compare_tools_reject_empty_answers(self):
        self.local.replies = {"qwen3:8b": reply(None)}
        with self.assertRaisesRegex(server.EmptyResponse, "empty_response"):
            await server.ask_local_agent(agent="coder", task="hello")
        compared = await server.compare_local_agents(agents=["coder", "reviewer"], task="hello")
        empty, answered = compared["results"]
        self.assertEqual((empty["agent"], empty["reason"]), ("coder", "empty_response"))
        self.assertNotIn("response", empty)
        self.assertEqual(answered["response"], "answer from gemma3:4b")

    async def test_non_text_content_is_an_unexpected_response(self):
        self.local.replies = {"qwen3:8b": reply(["not", "text"])}
        with self.assertRaisesRegex(RuntimeError, "Unexpected response from ollama") as raised:
            await server.ask_local_agent(agent="coder", task="hello")
        self.assertNotIsInstance(raised.exception, server.EmptyResponse)

    async def test_truncated_answer_is_returned_with_its_finish_reason(self):
        self.local.replies = {"gemma3:4b": reply("def average(values):\n    return sum(", finish_reason="length", completion_tokens=200)}
        result = await self.route()
        self.assertEqual(result["agent"], "reviewer")
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["response"], "def average(values):\n    return sum(")
        self.assertEqual(result["routing"]["attempts"], [{"agent": "reviewer", "ok": True}])

    async def test_finish_reason_is_preserved_by_every_worker_tool(self):
        self.assertEqual((await self.route())["finish_reason"], "stop")
        self.local.replies = {"qwen3:8b": reply("partial", finish_reason="length")}
        self.assertEqual((await server.ask_local_agent(agent="coder", task="hello"))["finish_reason"], "length")
        compared = await server.compare_local_agents(agents=["coder", "reviewer"], task="hello")
        self.assertEqual([item["finish_reason"] for item in compared["results"]], ["length", "stop"])

    async def test_missing_finish_reason_is_reported_as_none(self):
        self.local.replies = {"qwen3:8b": {"choices": [{"message": {"content": "hi"}}], "usage": None}}
        self.assertIsNone((await server.ask_local_agent(agent="coder", task="hello"))["finish_reason"])

    async def test_qwen_agents_disable_thinking_with_reasoning_effort(self):
        for agent in ("coder", "second_opinion"):
            await server.ask_local_agent(agent=agent, task="hello")
            sent = self.local.chats[-1]
            self.assertEqual(sent["reasoning_effort"], "none")
            # Ollama's /v1/chat/completions ignores the native `think` flag.
            self.assertNotIn("think", sent)
        await server.ask_local_agent(agent="reviewer", task="hello")
        self.assertNotIn("reasoning_effort", self.local.chats[-1])

    async def test_existing_tools_work_when_jev_and_routing_are_broken(self):
        self.use_jev(FakeJev(status=500))
        self.key_path.unlink()
        del self.config["routing"]
        with self.assertRaisesRegex(ValueError, "no routing object"):
            await self.route()
        asked = await server.ask_local_agent(agent="coder", task="hello")
        self.assertEqual(asked["response"], "answer from qwen3:8b")
        self.assertIn("coder", server.list_local_agents()["agents"])
        self.assertTrue((await server.local_runtime_status())["providers"]["ollama"]["reachable"])
        compared = await server.compare_local_agents(agents=["coder", "reviewer"], task="hello")
        self.assertEqual(len(compared["results"]), 2)

    async def test_tool_is_served_over_mcp_beside_the_existing_tools(self):
        async with Client(server.mcp) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            self.assertLessEqual(
                {"list_local_agents", "local_runtime_status", "ask_local_agent", "compare_local_agents", "route_local_agent"},
                names,
            )
            called = await client.call_tool("route_local_agent", {"task": "Review this function"})
        self.assertEqual(called.structured_content["routing"]["source"], "jev")

    def test_shipped_config_passes_routing_validation(self):
        shipped = json.loads(server.CONFIG_PATH.read_text(encoding="utf-8"))
        loaded = routing.load_routing(shipped)
        self.assertFalse(loaded["jev"]["send_context"])
        self.assertEqual(set(loaded["question"]["criteria"]), set(shipped["agents"]))


if __name__ == "__main__":
    unittest.main()
