"""Retrieve the first-manager setup code into a private local file, never stdout."""

import argparse
import os
import tempfile
from pathlib import Path

from relay_lab.spacetime import SpacetimeConfig, _read_token
from relay_lab.studio.durable import DurableState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = SpacetimeConfig(args.server, args.database, _read_token(args.token_file))
    with tempfile.TemporaryDirectory(prefix="relay-setup-") as temporary:
        state = DurableState(temporary, config)
        row = state.row()
        if not row or not row["payload"]:
            raise RuntimeError("Start the hosted app before retrieving its setup code")
        state._restore(row["payload"])
        code = Path(temporary) / "setup-code"
        if not code.is_file():
            raise RuntimeError("Setup is already complete; use the existing manager account")
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(code.read_text())
    print("First-manager code saved to the requested private file; do not commit it.")


if __name__ == "__main__":
    main()
