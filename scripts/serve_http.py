"""Start the localhost HTTP gateway with its private, file-backed token."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from local_agents_mcp.server import main


ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = ROOT / ".local" / "mcp_token"


if __name__ == "__main__":
    token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise SystemExit("Gateway token file must contain at least 32 characters")
    os.environ["LOCAL_AGENTS_MCP_TOKEN"] = token
    os.environ["LOCAL_AGENTS_PUBLIC_HOST"] = "mcp.ag-mcp.xyz"
    sys.argv = [sys.argv[0], "--http", "--port", "8000"]
    main()
