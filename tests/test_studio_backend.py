"""Artifact integrity, durable recovery and bounded Studio information flow."""

import copy
import hashlib
import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from relay_lab.cloth.communication import SAFETY_FIELDS
from relay_lab.llm import Generation
from relay_lab.studio.artifacts import BUNDLE, FILES, ArtifactRegistry
from relay_lab.studio.service import StudioService, example_workspace

ROOT = Path(__file__).resolve().parents[1]


class Journal:
    """Explicit test double; records committed bytes separately from handles."""

    def __init__(self):
        self.runs, self.batches, self.messages, self.queries = {}, {}, {}, []
        self.commits, self.acks = 0, 0

    def coordinator(self):
        journal = self

        class Coordinator:
            def __init__(self):
                self.client = self

            def create(self, run_id, worker_id, robots, jobs, resources, **kwargs):
                self.run_id, self.worker_id = run_id, worker_id
                journal.runs[run_id] = {
                    "run_id": run_id,
                    "worker_id": worker_id,
                    "revision": 0,
                    "tick": 0,
                    "phase": "ready",
                    "status": "running",
                    "state": copy.deepcopy(kwargs["state"]),
                }
                journal.messages[run_id] = []

            def snapshot(self):
                return copy.deepcopy(journal.runs[self.run_id])

            def prepare(self, revision, commands):
                assert revision == journal.runs[self.run_id]["revision"] and commands == []
                journal.batches[(self.run_id, revision)] = {
                    "prepare_json": json.dumps({"commands": [], "messages": []}),
                    "result_json": "",
                }
                journal.runs[self.run_id]["phase"] = "executing"

            def commit(self, revision, state, outcomes, *, messages):
                run = journal.runs[self.run_id]
                assert run["revision"] == revision and state["tick"] == revision + 1
                value = {"state": state, "outcomes": outcomes, "messages": messages}
                journal.batches[(self.run_id, revision)]["result_json"] = json.dumps(value)
                journal.messages[self.run_id].extend(copy.deepcopy(messages))
                run.update(
                    state=copy.deepcopy(state),
                    revision=revision + 1,
                    tick=revision + 1,
                    phase="ready",
                )
                journal.commits += 1

            def _batch(self, revision):
                return copy.deepcopy(journal.batches.get((self.run_id, revision)))

            def _mutate(self, reducer, args, reader, expected):
                assert reducer == "commit_factory_batch"
                assert journal.batches[(args[0], args[2])]["result_json"] == args[3]
                assert reader()["result_json"] == expected["result_json"]
                journal.acks += 1
                return reader()

            def _sql(self, query):
                journal.queries.append(query)
                match = re.search(r"run_id = '([^']+)'", query)
                if "factory_event" in query:
                    return [{"payload_json": json.dumps(m)} for m in journal.messages[match[1]]]
                if "SELECT state_json" in query:
                    return [{"state_json": json.dumps(journal.runs[match[1]]["state"])}]
                return [
                    {k: r[k] for k in ("run_id", "tick", "status", "phase")}
                    for r in journal.runs.values()
                ]

        return Coordinator()


class LLM:
    def __init__(self):
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return Generation("Hold position; a peer reports an obstacle.", 0.01, 20, 10)

    def close(self):
        pass


def request(request_id="observation-1"):
    return {
        "request_id": request_id,
        "source": "manual",
        "reports": {"left": "Blocked", "right": "No local obstacle"},
    }


@pytest.fixture
def services(tmp_path):
    journal = Journal()

    def make():
        return StudioService(tmp_path, coordinator_factory=journal.coordinator, llm=LLM())

    return journal, make


def connect(service, mode="mediated"):
    return service.connect({**example_workspace(), "mode": mode})["session_id"]


def test_restart_recovers_exact_request_identity_without_inference(services):
    journal, make = services
    first = make()
    sid = connect(first)
    original = first.exchange(sid, request())
    second = make()
    history = second.history(sid)
    assert history["status"] == "inspect_only" and history["resumable"]
    assert second.sessions == {} and journal.acks == 0  # GET is read-only.
    restored = second.recover(sid)
    assert restored["messages"] == original["messages"] and restored["recovered"]
    retry = second.exchange(sid, request())
    assert retry["messages"] == original["messages"]
    assert retry["generations"] == original["generations"]
    assert retry["database_ms"] is None and retry["elapsed_ms"] is None
    assert retry["preview_status"] == "not_persisted"
    assert second.llm.calls == 0 and journal.commits == 1 and journal.acks == 1
    changed = request()
    changed["reports"]["left"] = "Different observation"
    with pytest.raises(ValueError, match="reused"):
        second.exchange(sid, changed)
    next_result = second.exchange(sid, request("next"))
    assert next_result["exchange"] == 2
    assert {m["sequence"] for m in next_result["messages"]} == {1}


@pytest.mark.parametrize("change", ["prepared", "legacy", "terminal"])
def test_unsafe_sessions_remain_inspectable_but_cannot_resume(services, change):
    journal, make = services
    first = make()
    sid = connect(first)
    first.exchange(sid, request())
    if change == "prepared":
        first.session(sid)["coordinator"].prepare(1, [])
    elif change == "legacy":
        journal.runs[sid]["state"].pop("studio_protocol")
    else:
        journal.runs[sid].update(phase="terminal", status="failed")
    second = make()
    history = second.history(sid)
    assert history["status"] == "inspect_only" and not history["resumable"]
    assert len(history["messages"]) == 2
    with pytest.raises(ValueError):
        second.recover(sid)
    assert second.sessions == {} and journal.acks == 0


def test_recovery_rejects_event_batch_disagreement(services):
    journal, make = services
    service = make()
    sid = connect(service)
    service.exchange(sid, request())
    journal.messages[sid][0]["payload"]["context"] = "Corrupted"
    with pytest.raises(ValueError, match="disagree"):
        make().recover(sid)
    assert journal.acks == 0


def test_commit_readback_is_exact_tick_and_history_has_upper_bound(services):
    journal, make = services
    service = make()
    sid = connect(service)
    service.exchange(sid, request())
    assert any("created_tick = 1" in q for q in journal.queries)
    journal.messages[sid].append({"sender": "left", "payload": {"exchange": 2}})
    assert len(service.history(sid)["messages"]) == 2
    assert "created_tick <= 1" in journal.queries[-1]


def test_session_ids_cannot_enter_sql_unvalidated(services):
    journal, make = services
    service = make()
    with pytest.raises(ValueError):
        service.history("studio-x' OR TRUE")
    with pytest.raises(ValueError):
        service.recover("studio-x' OR TRUE")
    assert journal.queries == []


def test_parallel_identical_requests_commit_once(services):
    journal, make = services
    service = make()
    sid = connect(service)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(service.exchange, sid, request()) for _ in range(2)]
        assert futures[0].result() == futures[1].result()
    assert journal.commits == 1 and service.llm.calls == 2


def test_blocked_inference_does_not_block_workspace_save(services):
    _, make = services
    service = make()
    sid = connect(service)
    entered, release = threading.Event(), threading.Event()
    original = service.llm.generate

    def blocked(messages):
        entered.set()
        assert release.wait(5)
        return original(messages)

    service.llm.generate = blocked
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(service.exchange, sid, request())
        assert entered.wait(2)
        try:
            assert pool.submit(service.save, example_workspace()).result(timeout=2)["version"] == 1
        finally:
            release.set()
        pending.result(timeout=2)


def test_bundled_features_images_and_loaded_adapter_contract():
    service = StudioService(ROOT, llm=LLM())
    assert service.capabilities()["adapter"]["verified"]
    samples = service.sample("left_obstacle")
    assert all(samples[a]["row_id"].startswith("test-") for a in ("left", "right"))
    assert len(samples["left"]["features"]) == 960
    assert len(samples["right"]["features"]) == 1024
    assert service.sample_image("left_obstacle", "left").startswith(b"\x89PNG\r\n\x1a\n")
    policy = service.artifacts.policy("latent")
    assert service.artifacts.describe()["checkpoints"]["latent"]["loaded"]
    assert policy.model_info["dimension"] == 64
    assert set(service.safety_facts(samples["left"]["observation"])) == set(SAFETY_FIELDS)


@pytest.fixture
def bundle(tmp_path):
    destination = tmp_path / BUNDLE
    for name in ("files.json", *FILES):
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / BUNDLE / name, destination / name)
    return tmp_path, destination


def test_changed_artifact_invalidates_cached_policy(bundle):
    root, path = bundle
    registry = ArtifactRegistry(root)
    registry.policy("latent")
    checkpoint = path / "models/embeddings.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupted")
    assert not registry.describe()["available"]
    with pytest.raises(ValueError, match="integrity"):
        registry.policy("latent")


def test_wrong_mode_checkpoint_rejected_even_with_updated_manifest(bundle):
    root, path = bundle
    shutil.copyfile(path / "models/hybrid.pt", path / "models/embeddings.pt")
    manifest = json.loads((path / "files.json").read_text())
    manifest["models/embeddings.pt"] = hashlib.sha256(
        (path / "models/embeddings.pt").read_bytes()
    ).hexdigest()
    evaluation = json.loads((path / "models/evaluation.json").read_text())
    evaluation["modes"]["embeddings"]["checkpoint_sha256"] = manifest["models/embeddings.pt"]
    (path / "models/evaluation.json").write_text(json.dumps(evaluation))
    manifest["models/evaluation.json"] = hashlib.sha256(
        (path / "models/evaluation.json").read_bytes()
    ).hexdigest()
    (path / "files.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mode"):
        ArtifactRegistry(root).policy("latent")


@pytest.mark.parametrize("field", ["dataset", "cache", "checkpoint"])
def test_recombined_training_provenance_cannot_pass_individual_file_hashes(bundle, field):
    root, path = bundle
    evaluation_path = path / "models/evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    if field == "dataset":
        evaluation["training"]["dataset_rows_sha256"] = "0" * 64
    elif field == "cache":
        evaluation["training"]["feature_cache_sha256"]["left"] = "0" * 64
    else:
        evaluation["modes"]["embeddings"]["checkpoint_sha256"] = "0" * 64
    evaluation_path.write_text(json.dumps(evaluation))
    manifest_path = path / "files.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["models/evaluation.json"] = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ArtifactRegistry(root).verify()


def preview_packet():
    return {
        "recipient": "right",
        "expires_tick": 9,
        "payload": {
            "exchange": 1,
            "mode": "hybrid",
            "adapter": {"checkpoint_sha256": "checkpoint"},
            "vector": [0.0] * 64,
            "safety_facts": {"obstacle_visible": 1.0},
        },
    }


@pytest.mark.parametrize("change", ["checkpoint", "expired", "future", "mode"])
def test_prediction_rejects_invalid_delivered_contract(change):
    packet = preview_packet()
    if change == "checkpoint":
        packet["payload"]["adapter"]["checkpoint_sha256"] = "other"
    elif change == "expired":
        packet["expires_tick"] = 2
    elif change == "future":
        packet["payload"]["exchange"] = 3
    else:
        packet["payload"]["mode"] = "latent"
    sample = {
        a: {"observation": {}, "phase": "hold", "features": [], "extractor_id": a}
        for a in ("left", "right")
    }
    policy = SimpleNamespace(sha256="checkpoint", act=lambda *args, **kwargs: ({}, {}))
    with pytest.raises(ValueError):
        StudioService.predict(policy, sample, [packet], "hybrid", 2)


def test_missing_bundle_prevents_latent_registration_before_database(services):
    journal, make = services
    with pytest.raises(FileNotFoundError):
        connect(make(), "latent")
    assert journal.runs == {}


def test_open_session_does_not_silently_switch_trained_adapter(services, monkeypatch):
    journal, make = services
    service = make()
    policy = SimpleNamespace(model_info={"checkpoint_sha256": "first"})
    monkeypatch.setattr(service.artifacts, "policy", lambda mode: policy)
    sid = connect(service, "latent")
    policy.model_info = {"checkpoint_sha256": "second"}
    with pytest.raises(ValueError, match="pinned adapter changed"):
        service.exchange(sid, request())
    assert journal.commits == 0 and journal.batches == {}
