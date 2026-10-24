"""The SDK/physics/feedback boundary must actually close the handoff loop."""

import json

import pytest

from relay_lab.identity import identity_seed
from relay_lab.runner import RunConfig, run_episode


def test_identities_persist_privately_and_separate_roles(tmp_path):
    path = tmp_path / "identity.json"
    team = identity_seed("team", path)
    assert identity_seed("team", path) == team
    assert identity_seed("robots", path) != team
    assert path.stat().st_mode & 0o777 == 0o600
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="symbolic"):
        identity_seed("team", alias)


def test_fetch_reference_completes_with_verified_handoff_messages(tmp_path, monkeypatch):
    pytest.importorskip("uagents")
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "episode"
    summary = run_episode(
        RunConfig(seed=7, policy="scripted", video=False, transport="fetch"), directory
    )
    assert summary["success"]
    assert summary["packed"] == summary["order"]
    assert summary["execution_events"]["transfer_ready"] == 3
    transport = summary["transport_info"]
    assert transport["network"] == "SDK local dispatcher"
    assert transport["public_registration"] is False
    assert transport["acknowledgements"] == transport["sent"] + transport["feedback_sent"]
    assert transport["rejected"] == 0
    rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    ready = set()
    for row in rows:
        packer = row["agents"]["packer"]
        for event in packer["peer_execution_events"]:
            assert event["task_id"] == summary["task_id"]
            assert "order" not in event and "packed" not in event
            if event["kind"] == "transfer_ready":
                ready.add(event["object_id"])
        if packer["instruction"]["kind"] == "pick":
            assert packer["instruction"]["object_id"] in ready


def test_fetch_disabled_does_not_bypass_the_message_channel(tmp_path, monkeypatch):
    pytest.importorskip("uagents")
    monkeypatch.chdir(tmp_path)
    summary = run_episode(
        RunConfig(seed=7, video=False, transport="fetch", communication=False), tmp_path / "off"
    )
    assert not summary["success"]
    assert summary["transport_info"]["acknowledgements"] == 0
    assert summary["messages"]["delivered"] == 0
