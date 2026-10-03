"""Expose local OpenAI-compatible models as named MCP agents."""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from local_agents_mcp import routing as jev_routing


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(
    os.environ.get("LOCAL_AGENTS_CONFIG", PROJECT_ROOT / "config" / "agents.json")
)


class EmptyResponse(RuntimeError):
    """The worker answered with null, empty, or whitespace-only content."""

    reason = "empty_response"


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config.get("providers"), dict):
        raise ValueError("agents.json must contain a providers object")
    if not isinstance(config.get("agents"), dict):
        raise ValueError("agents.json must contain an agents object")
    return config


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local model API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach local model API at {url}: {error.reason}") from error


def _provider(config: dict[str, Any], provider_name: str) -> dict[str, Any]:
    try:
        provider = config["providers"][provider_name]
    except KeyError as error:
        raise ValueError(f"Unknown provider: {provider_name}") from error
    if not provider.get("base_url"):
        raise ValueError(f"Provider {provider_name!r} has no base_url")
    return provider


def _api_key(provider: dict[str, Any]) -> str | None:
    if provider.get("api_key"):
        return str(provider["api_key"])
    variable = provider.get("api_key_env")
    if variable:
        value = os.environ.get(str(variable))
        if not value:
            raise RuntimeError(f"Required environment variable {variable} is not set")
        return value
    return None


def _agent(config: dict[str, Any], agent_name: str) -> dict[str, Any]:
    try:
        return config["agents"][agent_name]
    except KeyError as error:
        available = ", ".join(sorted(config["agents"]))
        raise ValueError(f"Unknown agent {agent_name!r}. Available agents: {available}") from error


def _chat_sync(agent_name: str, task: str, context: str, max_tokens: int) -> dict[str, Any]:
    config = _load_config()
    agent = _agent(config, agent_name)
    provider_name = agent["provider"]
    provider = _provider(config, provider_name)

    user_content = task.strip()
    if context.strip():
        user_content = f"Task:\n{user_content}\n\nContext:\n{context.strip()}"

    payload = {
        "model": agent["model"],
        "messages": [
            {"role": "system", "content": agent["system_prompt"]},
            {"role": "user", "content": user_content},
        ],
        "temperature": agent.get("temperature", 0.2),
        "max_tokens": max_tokens,
        "stream": False,
    }
    payload.update(agent.get("request_overrides", {}))
    result = _request_json(
        "POST",
        f"{provider['base_url'].rstrip('/')}/chat/completions",
        api_key=_api_key(provider),
        payload=payload,
        timeout=int(provider.get("timeout_seconds", 180)),
    )
    try:
        choice = result["choices"][0]
        text = choice["message"].get("content")
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise RuntimeError(f"Unexpected response from {provider_name}: {result}") from error
    if text is not None and not isinstance(text, str):
        raise RuntimeError(f"Unexpected response from {provider_name}: {result}")
    # A reasoning model can spend the whole token budget thinking and return no answer.
    if text is None or not text.strip():
        completion_tokens = (result.get("usage") or {}).get("completion_tokens")
        raise EmptyResponse(
            f"empty_response: {agent['model']} returned no answer text "
            f"(finish_reason={finish_reason}, completion_tokens={completion_tokens})"
        )

    return {
        "agent": agent_name,
        "provider": provider_name,
        "model": agent["model"],
        "response": text,
        "finish_reason": finish_reason,
        "usage": result.get("usage"),
    }


async def _provider_status(providers: dict[str, Any]) -> dict[str, dict[str, Any]]:
    async def check(name: str, provider: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        try:
            api_key = _api_key(provider)
            response = await asyncio.to_thread(
                _request_json,
                "GET",
                f"{provider['base_url'].rstrip('/')}/models",
                api_key=api_key,
                timeout=3,
            )
            model_ids = [item.get("id") for item in response.get("data", [])]
            return name, {"reachable": True, "models": model_ids}
        except Exception as error:  # Status should report failures rather than abort.
            return name, {"reachable": False, "error": str(error)}

    return dict(await asyncio.gather(*(check(name, provider) for name, provider in providers.items())))


async def _agent_availability(
    config: dict[str, Any], names: list[str]
) -> tuple[list[str], dict[str, str]]:
    """Split agents into those whose runtime is serving their model and the rest."""
    providers = {
        config["agents"][name]["provider"]: config["providers"][config["agents"][name]["provider"]]
        for name in names
        if config["agents"][name]["provider"] in config["providers"]
    }
    status = await _provider_status(providers)
    available: list[str] = []
    unavailable: dict[str, str] = {}
    for name in names:
        agent = config["agents"][name]
        provider_name = agent["provider"]
        state = status.get(provider_name)
        if state is None:
            unavailable[name] = f"unknown provider {provider_name!r}"
        elif not state["reachable"]:
            unavailable[name] = f"provider {provider_name!r} is not reachable"
        elif agent["model"] not in state["models"] and not providers[provider_name].get("skip_model_check"):
            unavailable[name] = f"provider {provider_name!r} does not list model {agent['model']!r}"
        else:
            available.append(name)
    return available, unavailable


mcp = MCPServer(
    "Local AI Agents",
    instructions=(
        "Use these tools to consult language models running on this Mac. "
        "Call list_local_agents before choosing an unfamiliar agent. "
        "route_local_agent picks the agent for you; it sends the task text, "
        "not the context, to the TypeSafe cloud API to make that choice."
    ),
)


@mcp.tool()
def list_local_agents() -> dict[str, Any]:
    """List configured local agents, their roles, models, and runtime providers."""
    config = _load_config()
    agents = {
        name: {
            "description": value["description"],
            "provider": value["provider"],
            "model": value["model"],
        }
        for name, value in config["agents"].items()
    }
    return {"agents": agents}


@mcp.tool()
async def local_runtime_status() -> dict[str, Any]:
    """Check which configured local model runtimes are reachable."""
    config = _load_config()
    return {"providers": await _provider_status(config["providers"])}


@mcp.tool()
async def ask_local_agent(
    agent: str,
    task: str,
    context: str = "",
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """Ask one named local agent to perform a task using optional supporting context."""
    if not task.strip():
        raise ValueError("task cannot be empty")
    if not 1 <= max_tokens <= 4096:
        raise ValueError("max_tokens must be between 1 and 4096")
    return await asyncio.to_thread(_chat_sync, agent, task, context, max_tokens)


@mcp.tool()
async def compare_local_agents(
    agents: list[str],
    task: str,
    context: str = "",
    max_tokens_each: int = 800,
) -> dict[str, Any]:
    """Ask two or three local agents the same question and return their answers."""
    if not 2 <= len(agents) <= 3:
        raise ValueError("Choose two or three agents")
    if not task.strip():
        raise ValueError("task cannot be empty")
    if not 1 <= max_tokens_each <= 2048:
        raise ValueError("max_tokens_each must be between 1 and 2048")
    results: list[dict[str, Any]] = []
    # The default machine has 16 GB of unified memory. Loading several models
    # at once is likely to evict models or exhaust memory.
    for name in agents:
        try:
            results.append(await asyncio.to_thread(_chat_sync, name, task, context, max_tokens_each))
        except Exception as error:
            results.append(
                {"agent": name, "reason": getattr(error, "reason", "worker_error"), "error": str(error)}
            )
    return {"results": results}


@mcp.tool()
async def route_local_agent(
    task: str,
    context: str = "",
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """Pick the best available local agent for a task, run it, and report how it was chosen.

    Only the task text is sent to the TypeSafe cloud API for the choice; the
    context goes only to the chosen local agent.
    """
    if not task.strip():
        raise ValueError("task cannot be empty")
    if not 1 <= max_tokens <= 4096:
        raise ValueError("max_tokens must be between 1 and 4096")
    config = _load_config()
    routing = jev_routing.load_routing(config)
    available, unavailable = await _agent_availability(config, list(routing["question"]["criteria"]))
    if not available:
        detail = "; ".join(f"{name}: {reason}" for name, reason in unavailable.items())
        raise RuntimeError(f"No routable local agent is available. {detail}")

    decision = await jev_routing.select(routing, task, context, available)
    ranked = decision.pop("ranked")
    # Each agent is tried at most once, best first, up to the configured limit.
    attempts: list[dict[str, Any]] = []
    for name in ranked[: routing["fallback"]["max_worker_attempts"]]:
        try:
            result = await asyncio.to_thread(_chat_sync, name, task, context, max_tokens)
        except Exception as error:
            attempts.append(
                {
                    "agent": name,
                    "ok": False,
                    "reason": getattr(error, "reason", "worker_error"),
                    "error": str(error),
                }
            )
            continue
        attempts.append({"agent": name, "ok": True})
        return {
            **result,
            "routing": {
                "chosen": name,
                "first_choice": ranked[0],
                **decision,
                "available": available,
                "unavailable": unavailable,
                "attempts": attempts,
            },
        }
    failures = "; ".join(f"{attempt['agent']}: {attempt['error']}" for attempt in attempts)
    raise RuntimeError(f"All {len(attempts)} worker attempts failed. {failures}")


def main() -> None:
    """Run a local stdio server or a localhost HTTP endpoint for a tunnel."""
    parser = argparse.ArgumentParser(description="Local model MCP gateway")
    parser.add_argument("--http", action="store_true", help="Serve authenticated Streamable HTTP")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.http:
        import uvicorn

        from local_agents_mcp.http import create_app

        uvicorn.run(create_app(mcp, args.port), host="127.0.0.1", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
