"""Verify the exact stdio command configured in Cursor."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def run() -> None:
    parameters = StdioServerParameters(
        command=str(PROJECT_ROOT / ".venv" / "bin" / "local-agents-mcp"),
    )
    async with Client(parameters) as client:
        tools = await client.list_tools()
        print(", ".join(sorted(tool.name for tool in tools.tools)))


if __name__ == "__main__":
    asyncio.run(run())
