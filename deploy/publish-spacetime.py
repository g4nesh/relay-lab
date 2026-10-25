"""Publish an already-built module without clearing data or exposing its owner token.

Use a separate database and owner credential for each customer's Studio instance.
The operator must explicitly supply the bundle and private owner token file.
"""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from relay_lab.spacetime import SpacetimeConfig, _NoRedirect, _read_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    config = SpacetimeConfig(args.server, args.database, _read_token(args.token_file))
    if not args.bundle.is_file() or not 1 <= args.bundle.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError("Expected a compiled JavaScript module of at most 16 MiB")
    request = Request(
        f"{config.server.rstrip('/')}/v1/database/{config.database}?host_type=Js&clear=false",
        method="PUT",
        data=args.bundle.read_bytes(),
        headers={
            "Authorization": "Bearer " + config.token,
            "Content-Type": "application/javascript",
        },
    )
    try:
        with build_opener(ProxyHandler({}), _NoRedirect()).open(request, timeout=60) as response:
            result = json.loads(response.read(64 * 1024))
        if "Success" not in result:
            raise RuntimeError("The server did not confirm publication; inspect the database")
    except HTTPError as error:
        raise RuntimeError(
            f"Publication rejected (HTTP {error.code}); no data reset was requested"
        ) from None
    print("Module publication confirmed; clear=false. Check Studio /readyz before use.")


if __name__ == "__main__":
    main()
