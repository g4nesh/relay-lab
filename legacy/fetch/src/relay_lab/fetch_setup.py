"""Connect through the SDK's loopback inspector API when browser access fails."""

import getpass
import json
from pathlib import Path
from urllib.request import Request, urlopen


def connect_mailbox(port: int = 8010, token_file: Path | None = None) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    token = (
        token_file.read_text(encoding="utf-8").strip()
        if token_file
        else getpass.getpass("Agentverse mailbox-write API key (hidden): ").strip()
    )
    if not token or any(character.isspace() for character in token):
        raise ValueError("A valid Agentverse API key is required")
    request = Request(
        f"http://127.0.0.1:{port}/connect",
        data=json.dumps({"user_token": token, "agent_type": "mailbox"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
    if result.get("success") is not True:
        # Never print a server response that may reflect a credential.
        raise RuntimeError(
            "Agentverse declined mailbox registration. Check the API key's expiry/permissions "
            "and the gateway log."
        )
    print("Mailbox registered with Agentverse. Keep the gateway running for ASI:One requests.")
