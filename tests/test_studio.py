"""Studio registration, information boundaries and durable-delivery failure behavior."""

import copy
import json
import struct
from http.client import HTTPResponse
from io import BytesIO
from types import SimpleNamespace

import pytest

from relay_lab.llm import Generation
from relay_lab.studio.server import StudioHandler, validate_safetensors
from relay_lab.studio.service import (
    StudioService,
    captured_report,
    example_workspace,
    validate_workspace,
)


class FakeCoordinator:
    def __init__(self):
        self.client = self
        self.messages, self.revision, self.commits = [], 0, 0
        self.corrupt_read = False

    def create(self, *args, **kwargs):
        self.initial = kwargs["state"]

    def prepare(self, revision, commands):
        assert revision == self.revision and commands == []

    def commit(self, revision, state, outcomes, *, messages):
        assert revision == self.revision and state["tick"] == revision + 1
        assert outcomes == []
        self.messages.extend(copy.deepcopy(messages))
        self.revision += 1
        self.commits += 1

    def _sql(self, query):
        assert "factory_event" in query
        return [] if self.corrupt_read else [{"payload_json": json.dumps(m)} for m in self.messages]


class FakeLLM:
    def __init__(self, text="Peer observation received."):
        self.text, self.prompts = text, []

    def generate(self, messages):
        self.prompts.append(messages)
        return Generation(self.text, 0.1, 12, 7)


@pytest.fixture
def studio(tmp_path):
    return StudioService(tmp_path, coordinator_factory=FakeCoordinator, llm=FakeLLM())


def open_session(studio, mode="mediated", direction="both"):
    workspace = {**example_workspace(), "mode": mode, "direction": direction}
    return studio.connect(workspace)["session_id"]


def report(request_id="report-1"):
    return {
        "request_id": request_id,
        "source": "manual",
        "reports": {"left": "LEFT_ONLY: obstacle", "right": "RIGHT_ONLY: clear"},
    }


@pytest.mark.parametrize(
    "change",
    [
        {"agents": []},
        {"agents": example_workspace()["agents"] * 2},
        {"mode": "automatic_alignment"},
        {"direction": "all"},
        {"objective": ""},
    ],
)
def test_invalid_teams_cannot_open_channels(change):
    with pytest.raises(ValueError):
        validate_workspace({**example_workspace(), **change}, complete=True)


def test_weight_registration_is_inert_and_roundtrips(studio):
    workspace = example_workspace()
    workspace["mode"] = "mediated"
    a = workspace["agents"][0]
    a.update(robot="drone", family="custom", rig="http://127.0.0.1:8899/rig")
    a["source"] = {"kind": "local", "ref": "/never/load/untrusted-model.bin", "revision": ""}
    saved = studio.save(workspace)
    assert studio.workspace() == saved
    connection = studio.connect(saved)
    assert connection["models_loaded"] is False
    assert connection["hardware_execution"] is False
    assert studio.session(connection["session_id"])["coordinator"].commits == 0


@pytest.mark.parametrize("mode", ["latent", "hybrid"])
def test_wrong_checkpoint_never_uses_example_latent_adapter(studio, mode):
    workspace = {**example_workspace(), "mode": mode}
    workspace["agents"][0]["source"]["revision"] = "different-weights"
    with pytest.raises(ValueError, match="trained sender/receiver"):
        studio.connect(workspace)
    assert studio.sessions == {}


@pytest.mark.parametrize("method", ["save", "connect"])
def test_missing_upload_rejected(studio, method):
    workspace = {**example_workspace(), "mode": "mediated"}
    workspace["agents"][0]["source"] = {
        "kind": "upload",
        "ref": "a" * 32 + ".safetensors",
        "revision": "",
    }
    with pytest.raises(ValueError, match="missing"):
        getattr(studio, method)(workspace)


def test_mediator_does_not_leak_peer_observations_into_sender_prompt(studio):
    sid = open_session(studio)
    result = studio.exchange(sid, report())
    prompts = [json.dumps(p) for p in studio.llm.prompts]
    assert "LEFT_ONLY" in prompts[0] and "RIGHT_ONLY" not in prompts[0]
    assert "RIGHT_ONLY" in prompts[1] and "LEFT_ONLY" not in prompts[1]
    assert len(result["messages"]) == 2 and result["readback_verified"]
    assert not result["actions_executed"]
    assert result["messages"] == studio.history(sid)["messages"]
    assert [m["recipient"] for m in result["messages"]] == ["right", "left"]


def test_captured_report_does_not_infer_clear_path_from_occlusion():
    local = {"phase": "approach", "observation": {"agent": "right", "obstacle": None}}
    assert "does not establish that the shared path is clear" in captured_report(local)
    local["observation"]["obstacle"] = {"active": True, "position": [1, 2, 3]}
    assert "ACTIVE OBSTACLE" in captured_report(local)


def test_receiver_safety_facts_come_from_committed_packet():
    seen = []

    class Policy:
        sha256 = "hash"

        def act(self, observation, phase, features, belief, **kwargs):
            seen.append(belief)
            return {}, {}

    sample = {
        a: {"observation": {}, "phase": "hold", "features": [], "extractor_id": a}
        for a in ("left", "right")
    }
    packet = {
        "recipient": "right",
        "expires_tick": 9,
        "payload": {
            "vector": [0.2] * 64,
            "safety_facts": {"obstacle_visible": 1.0},
            "adapter": {"checkpoint_sha256": "hash"},
            "exchange": 1,
            "mode": "hybrid",
        },
    }
    StudioService.predict(Policy(), sample, [packet], "hybrid")
    assert seen[0] is None
    assert seen[1]["features"] == packet["payload"]["safety_facts"]


def test_idempotence_and_sequence_survive_retries(studio):
    sid = open_session(studio)
    first = studio.exchange(sid, report())
    assert studio.exchange(sid, report()) == first
    coordinator = studio.session(sid)["coordinator"]
    assert coordinator.commits == 1 and len(studio.llm.prompts) == 2
    changed = report()
    changed["reports"]["left"] = "new data"
    with pytest.raises(ValueError, match="reused"):
        studio.exchange(sid, changed)
    second = studio.exchange(sid, report("report-2"))
    assert second["exchange"] == 2
    assert {m["sequence"] for m in second["messages"]} == {1}
    assert {m["expires_tick"] for m in second["messages"]} == {10}


def test_delivery_direction_is_enforced(studio):
    sid = open_session(studio, direction="right")
    result = studio.exchange(sid, report())
    assert len(result["messages"]) == 1
    assert result["messages"][0]["sender"] == "right"
    assert result["messages"][0]["recipient"] == "left"
    assert len(studio.llm.prompts) == 1


def test_orchestrator_receives_both_reports_and_routes_individual_instructions(studio):
    studio.llm.text = '{"left":"Hold left.","right":"Hold right."}'
    sid = open_session(studio, "orchestrated")
    result = studio.exchange(sid, report())
    assert len(studio.llm.prompts) == 1
    prompt = json.dumps(studio.llm.prompts[0])
    assert "LEFT_ONLY" in prompt and "RIGHT_ONLY" in prompt
    assert {m["recipient"]: m["payload"]["context"] for m in result["messages"]} == {
        "left": "Hold left.",
        "right": "Hold right.",
    }


def test_bad_llm_response_does_not_commit_or_fabricate_fallback(studio):
    sid = open_session(studio, "orchestrated")
    with pytest.raises(ValueError):
        studio.exchange(sid, report())
    assert studio.session(sid)["coordinator"].commits == 0
    assert studio.history(sid)["messages"] == []


def test_uncertain_readback_stops_further_commits(studio):
    sid = open_session(studio)
    coordinator = studio.session(sid)["coordinator"]
    coordinator.corrupt_read = True
    with pytest.raises(RuntimeError, match="readback"):
        studio.exchange(sid, report())
    assert studio.session(sid)["failed"]
    with pytest.raises(ValueError, match="stopped"):
        studio.exchange(sid, report("next"))
    assert coordinator.commits == 1


def test_preview_failure_does_not_replay_successful_commit(studio, monkeypatch):
    torch = pytest.importorskip("torch")
    from relay_lab.cloth import latent_adapter

    class Policy:
        def __init__(self, path):
            self.model_info = {"test": True}
            self.network = SimpleNamespace(
                receivers={a: lambda x: torch.zeros(1, 32) for a in ("left", "right")}
            )

        def encode(self, *args):
            return [0.1] * 64

    monkeypatch.setattr(latent_adapter, "LearnedLatentPolicy", Policy)
    monkeypatch.setattr(studio.artifacts, "policy", lambda mode: Policy(None))
    monkeypatch.setattr(
        studio,
        "sample",
        lambda condition: {
            a: {"features": [1.0], "extractor_id": a, "observation": {}, "phase": "hold"}
            for a in ("left", "right")
        },
    )

    def failed_preview(*args):
        raise RuntimeError("Preview failed")

    monkeypatch.setattr(studio, "predict", failed_preview)
    sid = open_session(studio, "latent")
    request = {"request_id": "sample", "source": "captured"}
    result = studio.exchange(sid, request)
    assert result["readback_verified"] and result["decisions"] == {}
    assert result["prediction_error"] == "RuntimeError: Preview failed"
    assert studio.exchange(sid, request) == result
    assert studio.session(sid)["coordinator"].commits == 1
    result = studio.exchange(sid, {**request, "request_id": "next"})
    assert result["exchange"] == 2


class MemorySocket:
    def __init__(self, incoming):
        self.incoming, self.sent = incoming, bytearray()

    def makefile(self, *args):
        return BytesIO(self.incoming)

    def sendall(self, value):
        self.sent.extend(value)


@pytest.fixture
def http(studio):
    def request(path="/", method="GET", data=None, headers=None):
        h = {"Host": "127.0.0.1:8770", "Connection": "close", **(headers or {})}
        if data is not None:
            h["Content-Length"] = str(len(data))
        lines = [f"{method} {path} HTTP/1.1", *(f"{k}: {v}" for k, v in h.items())]
        socket = MemorySocket(("\r\n".join(lines) + "\r\n\r\n").encode() + (data or b""))
        StudioHandler(
            socket,
            ("127.0.0.1", 1),
            SimpleNamespace(server_port=8770),
            service=studio,
            token="test-token",
        )
        response = HTTPResponse(MemorySocket(bytes(socket.sent)), method=method)
        response.begin()
        return response.status, dict(response.getheaders()), response.read()

    return request


def test_http_origin_token_and_static_boundary(http):
    assert http()[0] == 200
    assert http("/studio.js")[0] == 200
    assert http("/api/bootstrap", headers={"Host": "evil.example:8770"})[0] == 403
    assert http("/api/bootstrap", headers={"Origin": "https://evil.example"})[0] == 403
    assert http("/api/workspace", "POST", b"{}")[0] == 403
    assert http("/../.relay-lab/spacetime-token")[0] == 404
    headers = {"X-Relay-Token": "test-token", "Content-Type": "application/json"}
    assert http("/api/workspace", "POST", b'{"version":NaN}', headers)[0] == 400
    status, _, data = http(
        "/api/workspace", "POST", json.dumps(example_workspace()).encode(), headers
    )
    assert status == 200 and json.loads(data)["agents"][1]["family"] == "xvla"


def weights():
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.5)


def test_upload_is_local_inert_and_malformed_upload_is_cleaned(http, studio):
    headers = {"X-Relay-Token": "test-token", "X-Filename": "test.safetensors"}
    status, _, data = http("/api/upload", "POST", weights(), headers)
    metadata = json.loads(data)
    assert status == 201 and metadata["loaded"] is False and metadata["tensors"] == 1
    path = studio.uploads / metadata["asset"]
    assert path.read_bytes() == weights() and validate_safetensors(path) == 1
    assert http("/api/upload", "POST", b"not-weights-xxxxxxxx", headers)[0] == 400
    assert len(list(studio.uploads.glob("*.safetensors"))) == 1
    assert (
        http("/api/upload", "POST", weights(), {**headers, "X-Filename": "../escape.safetensors"})[
            0
        ]
        == 400
    )
