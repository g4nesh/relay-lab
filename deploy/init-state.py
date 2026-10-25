"""Install an existing database owner's token from stdin, without echoing it."""

import os
import sys
from pathlib import Path

from relay_lab.spacetime import SpacetimeConfig

os.umask(0o077)
root = Path(os.environ.get("RELAY_STUDIO_ROOT", "/data")) / ".relay-lab"
root.mkdir(parents=True, exist_ok=True, mode=0o700)
token = sys.stdin.read(16385).strip()
SpacetimeConfig(
    server=os.environ["RELAY_SPACETIME_SERVER"],
    database=os.environ["RELAY_SPACETIME_DATABASE"],
    token=token,
)
descriptor = os.open(root / "spacetime-token", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as stream:
    stream.write(token + "\n")
    stream.flush()
    os.fsync(stream.fileno())
print("Private credential installed. Existing credentials are never overwritten.")
