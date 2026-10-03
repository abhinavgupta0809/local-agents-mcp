"""Small bearer-token gate for the MCP HTTP transport."""

from __future__ import annotations

import hmac
import os
from typing import Any

from mcp.server.transport_security import TransportSecuritySettings


class BearerGate:
    def __init__(self, app: Any, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode("utf-8")

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"authorization", b"")
        if not hmac.compare_digest(supplied, self.expected):
            await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"Unauthorized"})
            return
        await self.app(scope, receive, send)


def create_app(mcp: Any, port: int) -> BearerGate:
    token = os.environ.get("LOCAL_AGENTS_MCP_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Set LOCAL_AGENTS_MCP_TOKEN to a random token of at least 32 characters before starting HTTP")
    public_host = os.environ.get("LOCAL_AGENTS_PUBLIC_HOST", "").strip()
    if public_host and ("/" in public_host or ":" in public_host):
        raise ValueError("LOCAL_AGENTS_PUBLIC_HOST must be a hostname such as mcp.example.com")
    allowed_hosts = [f"127.0.0.1:{port}", f"localhost:{port}"]
    allowed_origins = [f"http://127.0.0.1:{port}", f"http://localhost:{port}"]
    if public_host:
        allowed_hosts.append(public_host)
        allowed_origins.append(f"https://{public_host}")
    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        max_request_body_size=1024 * 1024,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        ),
    )
    return BearerGate(app, token)
