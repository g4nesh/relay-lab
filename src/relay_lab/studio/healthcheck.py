"""Container readiness probe; prints neither credentials nor backend details."""

from urllib.request import ProxyHandler, Request, build_opener

from .deployment import DeploymentConfig


def main():
    config = DeploymentConfig.from_env()
    request = Request(f"http://127.0.0.1:{config.port}/readyz", headers={"Host": config.authority})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=11) as response:
            return 0 if response.status == 200 else 1
    except Exception:  # noqa: BLE001 - only the exit status is consumed
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
