#!/usr/bin/env python3
"""Install, run and publish the pinned, private, localhost coordination backend.

No global CLI install, shell modification, cloud login, or database clearing.
The publisher credential never appears in command arguments or normal output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.10.0"
PRIVATE = ROOT / ".relay-lab"
STATE = PRIVATE / "spacetime"
CLI_ROOT = STATE / "cli"
CLI = CLI_ROOT / "bin" / VERSION / "spacetimedb-cli"
TOKEN_FILE = PRIVATE / "spacetime-token"
URL = "http://127.0.0.1:3000"
DATABASE = "relay-lab"
MODULE = ROOT / "spacetimedb"
INSTALLER_SHA256_MAC_ARM = "729d0330cae4b3a6058a2257d594c4e0c3ef3c2f6ad44b249d7788f51a5f0c4c"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("Refusing to redirect a localhost backend request")


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlink at {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def setup_directories() -> None:
    for path in (PRIVATE, STATE, CLI_ROOT):
        private_directory(path)


def run(*arguments: str, cwd: Path = ROOT) -> None:
    subprocess.run(list(arguments), cwd=cwd, check=True)


def cli_command(*arguments: str) -> list[str]:
    if not CLI.is_file():
        raise RuntimeError("Run `python3 scripts/spacetime.py install` first")
    return [str(CLI), "--root-dir", str(CLI_ROOT), *arguments]


def check_version() -> None:
    result = subprocess.run(cli_command("--version"), check=True, capture_output=True, text=True)
    if f"spacetimedb tool version {VERSION};" not in result.stdout:
        raise RuntimeError(f"The local CLI does not match pinned version {VERSION}")


def request(path: str, *, method="GET", body=None, token=None, content_type=None):
    headers = {}
    if token:
        headers["Authorization"] = "Bearer " + token
    if content_type:
        headers["Content-Type"] = content_type
    req = Request(URL + path, data=body, headers=headers, method=method)
    with build_opener(ProxyHandler({}), NoRedirect()).open(req, timeout=60) as response:
        data = response.read()
        return json.loads(data) if data else None


def read_token() -> str:
    descriptor = os.open(TOKEN_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "r") as handle:
        info = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_size > 16_384
        ):
            raise RuntimeError("The publisher token must be an owner-only regular file (0600)")
        token = handle.read(16_385).strip()
    if not token or any(character.isspace() for character in token):
        raise RuntimeError("Invalid local publisher token file")
    return token


def ensure_token() -> str:
    if TOKEN_FILE.exists() or TOKEN_FILE.is_symlink():
        return read_token()
    # Do not replace a lost owner identity for an existing database. A new
    # credential cannot recover that owner's permissions.
    try:
        request(f"/v1/database/{DATABASE}")
    except HTTPError as error:
        if error.code != 404:
            raise
    else:
        raise RuntimeError("Database exists but its publisher token is missing; restore the token")
    identity = request("/v1/identity", method="POST", body=b"")
    token = identity["token"]
    descriptor = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(token + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return token


def install() -> None:
    setup_directories()
    if not CLI.is_file():
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError(
                "This bootstrap is verified on Apple Silicon. Install the official "
                f"{VERSION} CLI into {CLI.parent} for another platform."
            )
        helper_dir = STATE / "bin"
        private_directory(helper_dir)
        helper = helper_dir / "spacetimedb-update"
        download = helper_dir / "download.tmp"
        url = (
            "https://spacetimedb-client-binaries.s3.amazonaws.com/refs/tags/"
            f"v{VERSION}/spacetimedb-update-aarch64-apple-darwin"
        )
        run("curl", "--fail", "--location", "--silent", "--show-error", url, "-o", str(download))
        if hashlib.sha256(download.read_bytes()).hexdigest() != INSTALLER_SHA256_MAC_ARM:
            download.unlink()
            raise RuntimeError("The pinned official installer checksum differs")
        download.chmod(0o700)
        download.replace(helper)
        # Do not use self-install: it installs latest and has independent
        # root-dir options. The version installer creates only this local root.
        run(
            str(helper),
            f"--root-dir={CLI_ROOT}",
            "version",
            "install",
            VERSION,
            "--use",
            "--yes",
        )
    check_version()
    run(
        "npm",
        "ci",
        "--ignore-scripts",
        "--cache",
        str(STATE / "npm-cache"),
        cwd=MODULE,
    )
    run("npm", "run", "check", cwd=MODULE)
    print(f"SpacetimeDB {VERSION} installed locally. No shell or global CLI changes.")


def start() -> None:
    setup_directories()
    check_version()
    private_directory(STATE / "data")
    command = cli_command(
        "start",
        "--listen-addr",
        "127.0.0.1:3000",
        "--data-dir",
        str(STATE / "data"),
        "--page_pool_max_size",
        "1073741824",
        "--non-interactive",
    )
    print(f"Starting SpacetimeDB {VERSION} at {URL}; data stays in {STATE / 'data'}", flush=True)
    os.execv(command[0], command)


def publish() -> None:
    setup_directories()
    check_version()
    request("/v1/ping")
    run("npm", "run", "check", cwd=MODULE)
    run(*cli_command("build", "--module-path", str(MODULE)))
    token = ensure_token()
    # Matches the pinned official CLI's host_type=Js request. clear=false is
    # explicit, and this helper provides no reset/delete path.
    result = request(
        f"/v1/database/{DATABASE}?host_type=Js&clear=false",
        method="PUT",
        body=(MODULE / "dist" / "bundle.js").read_bytes(),
        token=token,
        content_type="application/javascript",
    )
    if "Success" not in result:
        raise RuntimeError("The server did not confirm the module publication")
    print(f"Database {DATABASE} {result['Success']['op']} at {URL}; existing data preserved.")
    status()


def status() -> None:
    request("/v1/ping")
    info = request(f"/v1/database/{DATABASE}")
    result = request(
        f"/v1/database/{DATABASE}/sql",
        method="POST",
        body=b"SELECT status FROM task",
        token=read_token(),
        content_type="text/plain",
    )
    counts = Counter(row[0] for row in result[0]["rows"])
    print(
        json.dumps(
            {
                "url": URL,
                "database": DATABASE,
                "host_type": info["host_type"],
                "private_owner_access": True,
                "tasks": dict(counts),
            },
            indent=2,
        )
    )


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "start", "publish", "status"))
    args = parser.parse_args()
    try:
        {"install": install, "start": start, "publish": publish, "status": status}[args.command]()
    except HTTPError as error:
        # Never echo raw request headers or a server response that might carry
        # credentials. Reducer failures are diagnosed by the runtime client.
        print(f"Local SpacetimeDB request failed (HTTP {error.code}).", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        safe = re.sub(r"eyJ[A-Za-z0-9_.-]+", "[redacted credential]", str(error))
        print(safe, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
