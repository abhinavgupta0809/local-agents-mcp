"""Exercise the server through the real in-memory MCP transport."""

from __future__ import annotations

import asyncio
import json
import os

from mcp import Client

from local_agents_mcp.server import mcp


async def run() -> None:
    async with Client(mcp) as client:
        listed = await client.call_tool("list_local_agents", {})
        status = await client.call_tool("local_runtime_status", {})
        output = {
            "agents": listed.structured_content,
            "status": status.structured_content,
        }
        if os.environ.get("RUN_GENERATION") == "1":
            generated = await client.call_tool(
                "ask_local_agent",
                {
                    "agent": "reviewer",
                    "task": "Reply with exactly: local agent is working",
                    "max_tokens": 80,
                },
            )
            output["generation"] = generated.structured_content
        # Makes one billed call to the TypeSafe API.
        if os.environ.get("RUN_ROUTING") == "1":
            routed = await client.call_tool(
                "route_local_agent",
                {"task": "Reply with exactly: routing is working", "max_tokens": 80},
            )
            output["routing"] = routed.structured_content
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
