"""Deploy the Desktop checkout to a macOS-safe background-service location.

Run explicitly after source/config/key updates; never calls model APIs.
"""
from __future__ import annotations
import os
from pathlib import Path
import plistlib
import shutil
import subprocess


def main() -> None:
    source = Path(__file__).resolve().parents[1]
    runtime = Path.home() / "Library/Application Support/local-ai-agents/runtime"
    marker = runtime / ".managed-local-agents-runtime"
    if runtime.exists() and not marker.exists():
        raise SystemExit("Unmanaged runtime directory already exists; refusing to overwrite it.")
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("Install uv and put it on PATH before deploying.")
    # Check credentials before stopping a working service.
    if not (source / ".local/mcp_token").is_file():
        raise SystemExit("Missing .local/mcp_token; no deployment performed.")
    domain = f"gui/{os.getuid()}"
    label = "ai.localagents.mcp"
    subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], capture_output=True)
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(runtime, 0o700)
    marker.touch(mode=0o600)
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(source / name, runtime / name)
    for name in ("src", "config"):
        shutil.copytree(source / name, runtime / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (runtime / "scripts").mkdir(exist_ok=True)
    shutil.copy2(source / "scripts/serve_http.py", runtime / "scripts/serve_http.py")
    private = runtime / ".local"
    private.mkdir(mode=0o700, exist_ok=True)
    os.chmod(private, 0o700)
    for name in ("mcp_token", "typesafe_api_key"):
        target = private / name
        origin = source / ".local" / name
        if origin.is_file():
            shutil.copy2(origin, target)
            os.chmod(target, 0o600)
        elif target.exists():
            target.unlink()  # Do not retain a key removed from the source installation.
    subprocess.run([uv, "sync", "--locked", "--project", str(runtime)], check=True)
    logs = Path.home() / "Library/Logs/local-ai-agents"
    logs.mkdir(mode=0o700, exist_ok=True)
    plist = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
    plist.parent.mkdir(exist_ok=True)
    settings = {
        "Label": label,
        "ProgramArguments": [str(runtime / ".venv/bin/python"), str(runtime / "scripts/serve_http.py")],
        "WorkingDirectory": str(runtime),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(logs / "gateway.log"),
        "StandardErrorPath": str(logs / "gateway.err.log"),
    }
    plist.write_bytes(plistlib.dumps(settings))
    subprocess.run(["launchctl", "bootstrap", domain, str(plist)], check=True)
    print(f"Gateway deployed to {runtime}")
    print("No model inference or TypeSafe API requests were made.")


if __name__ == "__main__":
    main()
