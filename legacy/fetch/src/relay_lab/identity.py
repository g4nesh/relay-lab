"""Persistent local Fetch identities; secret material is never part of a rollout."""

import hashlib
import json
import os
import secrets
from pathlib import Path


def identity_seed(role: str, path: Path | None = None) -> str:
    """Create one private project key, then derive distinct stable role identities."""
    if role not in ("team", "robots", "client"):
        raise ValueError("Unknown Fetch identity role")
    path = path or Path.cwd() / ".relay-lab" / "identity.json"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"version": 1, "master_seed": secrets.token_hex(32)}, stream)
            stream.write("\n")
    if path.is_symlink():
        raise ValueError("Identity file must not be a symbolic link")
    data = json.loads(path.read_text(encoding="utf-8"))
    master = data.get("master_seed", "")
    if data.get("version") != 1 or not isinstance(master, str) or len(master) != 64:
        raise ValueError("Invalid local Fetch identity file")
    try:
        bytes.fromhex(master)
    except ValueError as exc:
        raise ValueError("Invalid local Fetch identity file") from exc
    os.chmod(path, 0o600)
    return hashlib.sha256(f"relay-lab:{role}:{master}".encode()).hexdigest()
