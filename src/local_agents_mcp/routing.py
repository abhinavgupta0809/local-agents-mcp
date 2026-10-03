"""Choose a local agent for a task with Jev, falling back to a fixed order.

The question, its criteria, and the fallback rules all live in the "routing"
block of agents.json. This module only turns that block into a decision.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEY_PATH = PROJECT_ROOT / ".local" / "typesafe_api_key"
KEY_ENV = "TYPESAFE_API_KEY"

# Tests replace this with an httpx2.MockTransport so no request leaves the machine.
_transport: Any = None


class JevUnavailable(Exception):
    """Jev could not give a usable answer; the message is the fallback reason."""


def load_routing(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the routing block. Called per routed request, not at config load,
    so a bad routing block cannot break the other tools."""
    routing = config.get("routing")
    if not isinstance(routing, dict):
        raise ValueError("agents.json has no routing object")
    question = routing.get("question")
    if not isinstance(question, dict) or not question.get("instructions"):
        raise ValueError("routing.question needs instructions")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("routing.question.criteria must describe at least one agent")
    unknown = sorted(set(criteria) - set(config["agents"]))
    if unknown:
        raise ValueError(f"routing.question.criteria names unknown agents: {', '.join(unknown)}")

    fallback = routing.get("fallback")
    if not isinstance(fallback, dict) or not isinstance(fallback.get("order"), list):
        raise ValueError("routing.fallback needs an order list")
    stray = [name for name in fallback["order"] if name not in criteria]
    if stray:
        raise ValueError(f"routing.fallback.order names agents without criteria: {', '.join(stray)}")
    attempts = fallback.get("max_worker_attempts", 1)
    if not isinstance(attempts, int) or not 1 <= attempts <= len(criteria):
        raise ValueError("routing.fallback.max_worker_attempts must be between 1 and the number of routable agents")

    if not isinstance(routing.get("jev", {}), dict):
        raise ValueError("routing.jev must be an object")
    return routing


def read_api_key() -> str | None:
    """Return the TypeSafe key from the private key file, or the environment."""
    try:
        key = KEY_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        key = ""
    return key or os.environ.get(KEY_ENV, "").strip() or None


def fallback_ranking(routing: dict[str, Any], available: list[str]) -> list[str]:
    """Available agents in the configured fallback order, then any others."""
    ordered = [name for name in routing["fallback"]["order"] if name in available]
    return ordered + [name for name in available if name not in ordered]


async def ask_jev(
    routing: dict[str, Any], task: str, context: str, available: list[str]
) -> dict[str, Any]:
    """Ask Jev which available agent fits the task. Raises JevUnavailable."""
    api_key = read_api_key()
    if not api_key:
        raise JevUnavailable("missing_key")
    try:
        from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy, TypeSafeAPIError
    except ImportError as error:
        raise JevUnavailable("sdk_not_installed") from error

    jev = routing.get("jev", {})
    question = routing["question"]
    state = {"task": task[: int(jev.get("task_char_limit", 12000))]}
    if jev.get("send_context", False) and context.strip():
        state["context"] = context[: int(jev.get("context_char_limit", 8000))]
    timeout = float(jev.get("timeout_seconds", 5))
    max_retries = int(jev.get("max_retries", 1))

    # Error text is never copied into the reason: only the class and HTTP status.
    try:
        async with AsyncTypeSafeClient(
            api_key=api_key,
            model=jev.get("model", "jev-latest"),
            timeout=timeout,
            retry=RetryPolicy(max_retries=max_retries, timeout=timeout * (max_retries + 1)),
            transport=_transport,
        ) as client:
            response = await client.system_one(
                state,
                {
                    "worker": Choice(
                        instructions=question["instructions"],
                        criteria={name: question["criteria"][name] for name in available},
                    )
                },
            )
        answer = response.choices["worker"]
    except TypeSafeAPIError as error:
        raise JevUnavailable(f"jev_error:{type(error).__name__}:{error.status}") from error
    except Exception as error:  # Routing must degrade to the fallback, never abort.
        raise JevUnavailable(f"jev_error:{type(error).__name__}") from error

    if answer.choice not in available:
        raise JevUnavailable("invalid_choice")
    usage = getattr(response, "usage", None)
    return {
        "choice": answer.choice,
        "confidence": answer.confidence,
        "probabilities": dict(answer.probabilities),
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        },
    }


async def select(
    routing: dict[str, Any], task: str, context: str, available: list[str]
) -> dict[str, Any]:
    """Rank the available agents and say how the ranking was made."""
    decision: dict[str, Any] = {
        "source": "fallback",
        "confidence": None,
        "probabilities": None,
        "fallback_reason": None,
        # Token counts for the run log; the caller removes this before replying.
        "usage": None,
    }
    order = fallback_ranking(routing, available)
    if len(available) == 1:
        decision["source"] = "only_available"
    else:
        try:
            answer = await ask_jev(routing, task, context, available)
        except JevUnavailable as unavailable:
            decision["fallback_reason"] = str(unavailable)
        else:
            probabilities = answer["probabilities"]
            rest = sorted(
                (name for name in order if name != answer["choice"]),
                key=lambda name: -probabilities.get(name, 0.0),
            )
            order = [answer["choice"], *rest]
            decision.update(
                source="jev",
                confidence=answer["confidence"],
                probabilities=probabilities,
                usage=answer["usage"],
            )
    decision["ranked"] = order
    return decision
