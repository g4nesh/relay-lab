"""Explicit single-customer origin and resource limits for hosted Studio."""

from __future__ import annotations

import importlib.util
import ipaddress
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


def experiments_requested():
    """Stripped distributions always use the control plane, including local preview."""
    return (
        os.environ.get("RELAY_STUDIO_EXPERIMENTS", "1") != "0"
        and importlib.util.find_spec("relay_lab.llm") is not None
    )


@dataclass(frozen=True)
class DeploymentConfig:
    public_origin: str = ""
    bind: str = "127.0.0.1"
    port: int = 8770
    root: Path = Path(".")
    max_body_bytes: int = 128 * 1024
    max_concurrent: int = 8
    body_timeout_seconds: float = 15.0
    request_timeout_seconds: float = 120.0

    def __post_init__(self):
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("Invalid Studio port")
        if self.bind not in ("127.0.0.1", "::1", "0.0.0.0", "::"):
            raise ValueError("Bind must be an explicit loopback or wildcard address")
        if self.public_origin:
            origin = urlsplit(self.public_origin)
            if (
                origin.scheme != "https"
                or not origin.hostname
                or origin.username is not None
                or origin.password is not None
                or origin.path
                or origin.query
                or origin.fragment
                or self.public_origin != "https://" + origin.netloc
                or origin.netloc != origin.netloc.lower()
            ):
                raise ValueError("Public origin must be a lowercase HTTPS origin without a path")
            try:
                _ = origin.port
                origin.netloc.encode("ascii")
            except (ValueError, UnicodeError):
                raise ValueError("Public origin has an invalid host or port") from None
            if any(character.isspace() for character in origin.netloc):
                raise ValueError("Public origin has an invalid host")
            try:
                ipaddress.ip_address(origin.hostname)
            except ValueError:
                if len(origin.hostname) > 253 or any(
                    not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                    for label in origin.hostname.split(".")
                ):
                    raise ValueError("Public origin has an invalid host")
        elif not ipaddress.ip_address(self.bind).is_loopback:
            raise ValueError("A non-loopback bind requires RELAY_STUDIO_PUBLIC_ORIGIN=https://...")
        if type(self.max_body_bytes) is not int or not 1024 <= self.max_body_bytes <= 128 * 1024:
            raise ValueError("The hosted body limit must be between 1 and 128 KiB")
        if type(self.max_concurrent) is not int or not 1 <= self.max_concurrent <= 32:
            raise ValueError("Concurrent request limit must be between 1 and 32")
        for value in (self.body_timeout_seconds, self.request_timeout_seconds):
            if type(value) not in (int, float) or not 0 < value <= 180:
                raise ValueError("Request timeouts must be between 0 and 180 seconds")

    @property
    def hosted(self):
        return bool(self.public_origin)

    @property
    def authority(self):
        return urlsplit(self.public_origin).netloc if self.hosted else f"127.0.0.1:{self.port}"

    def guard(self, headers, port=None):
        """Never derive authority or security from client-supplied forwarded headers."""
        port = port or self.port
        origins = (
            {self.public_origin}
            if self.hosted
            else {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
        )
        authorities = {urlsplit(origin).netloc for origin in origins}
        if headers.get("Host") not in authorities:
            raise PermissionError("Request host does not match this Studio deployment")
        origin = headers.get("Origin")
        if origin and origin not in origins:
            raise PermissionError("Cross-origin requests are not allowed")

    @classmethod
    def from_env(cls):
        return cls(
            public_origin=os.environ.get("RELAY_STUDIO_PUBLIC_ORIGIN", os.environ.get("RENDER_EXTERNAL_URL", "")),
            bind=os.environ.get("RELAY_STUDIO_BIND", "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"),
            port=int(os.environ.get("PORT", os.environ.get("RELAY_STUDIO_PORT", "8770"))),
            root=Path(os.environ.get("RELAY_STUDIO_ROOT", ".")).resolve(),
            max_concurrent=int(os.environ.get("RELAY_STUDIO_MAX_CONCURRENT", "8")),
        )
