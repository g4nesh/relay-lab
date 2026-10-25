"""Learned communication, provenance, and episode-split contracts."""

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from relay_lab.cloth.communication import AGENTS, FEATURES, SAFETY_FIELDS
from relay_lab.cloth.features import feature_contract
from relay_lab.cloth.latent_adapter import (
    DIMENSION,
    FORMAT,
    LOCAL_DIM,
    MAX_DELTA,
    MODES,
    WIDTHS,
    DualLatentAdapter,
    LearnedLatentPolicy,
)
from relay_lab.cloth.latent_dataset import array_hash, canonical_hash
from relay_lab.cloth.latent_training import load_pairs, sha


def _batch():
    torch.manual_seed(42)
    return {
        "features": {"left": torch.randn(3, 7), "right": torch.randn(3, 9)},
        "states": {a: torch.randn(3, LOCAL_DIM) for a in AGENTS},
        "facts": {a: torch.randn(3, len(FEATURES)) for a in AGENTS},
    }


def _forward(model, batch, *, valid=None, age=None):
    return model(
        batch["features"],
        batch["states"],
        batch["facts"],
        torch.ones(3) if valid is None else valid,
        torch.zeros(3) if age is None else age,
    )


@pytest.mark.parametrize("mode", MODES)
def test_fresh_only_model_initialization_has_no_arbitrary_age_response(mode):
    model = DualLatentAdapter(mode, {"left": 7, "right": 9}).eval()
    batch = _batch()
    fresh = _forward(model, batch, age=torch.zeros(3))
    aged = _forward(model, batch, age=torch.full((3,), 0.8))
    for arm in AGENTS:
        assert torch.count_nonzero(model.heads[arm][0].weight[:, -1]) == 0
        torch.testing.assert_close(fresh[arm], aged[arm], rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
def test_missing_message_is_zero_before_and_after_receiver_bias(mode):
    model = DualLatentAdapter(mode, {"left": 7, "right": 9}).eval()
    batch = _batch()
    observed = {}
    before = model.receivers["left"].register_forward_pre_hook(
        lambda module, args: observed.update(receiver_input=args[0].detach().clone())
    )
    after = model.heads["left"].register_forward_pre_hook(
        lambda module, args: observed.update(head_input=args[0].detach().clone())
    )
    model.decode_action(
        "left",
        batch["features"]["left"],
        batch["states"]["left"],
        torch.full((3, DIMENSION), 99.0),
        batch["facts"]["right"],
        torch.zeros(3),
        torch.ones(3),
    )
    before.remove()
    after.remove()
    assert torch.count_nonzero(observed["receiver_input"]) == 0
    peer_start = 32 + LOCAL_DIM
    assert torch.count_nonzero(observed["head_input"][:, peer_start : peer_start + 32]) == 0
    facts_start = peer_start + 32
    assert (
        torch.count_nonzero(observed["head_input"][:, facts_start : facts_start + len(FEATURES)])
        == 0
    )


@pytest.mark.parametrize("mode", MODES)
def test_modes_use_only_the_declared_peer_information(mode):
    model = DualLatentAdapter(mode, {"left": 7, "right": 9}).eval()
    batch = _batch()
    expected = _forward(model, batch)["left"].detach()
    if mode in ("none", "structured_text"):
        batch["features"]["right"] = torch.randn_like(batch["features"]["right"]) * 50
    if mode in ("none", "embeddings"):
        batch["facts"]["right"] = torch.randn_like(batch["facts"]["right"]) * 50
    if mode == "hybrid":
        for index, name in enumerate(FEATURES):
            if name not in SAFETY_FIELDS:
                batch["facts"]["right"][:, index] += 100
    actual = _forward(
        model,
        batch,
        valid=torch.zeros(3) if mode == "none" else None,
        age=torch.ones(3) if mode == "none" else None,
    )["left"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
def test_peer_action_loss_trains_the_sender_only_for_latent_modes(mode):
    model = DualLatentAdapter(mode, {"left": 7, "right": 9})
    _forward(model, _batch())["left"].square().sum().backward()
    gradient = sum(
        float(p.grad.abs().sum()) for p in model.senders["right"].parameters() if p.grad is not None
    )
    assert (gradient > 0) == (mode in ("embeddings", "hybrid"))


def _observation(arm):
    return {
        "agent": arm,
        "ee_position": [-0.2 if arm == "left" else 0.2, -0.1, 0.1],
        "landmarks": {"hem": [0.1, -0.2, 0.01]},
        "grasped": False,
        "grasp_target": "none",
        "obstacle": None,
    }


def _checkpoint(tmp_path, mode="embeddings"):
    torch.manual_seed(9)
    network = DualLatentAdapter(mode)
    ids = {
        a: feature_contract("smolvla" if a == "left" else "xvla")["feature_space_id"]
        for a in AGENTS
    }
    path = tmp_path / f"{mode}.pt"
    torch.save(
        {
            "format": FORMAT,
            "mode": mode,
            "widths": WIDTHS,
            "state_dict": network.state_dict(),
            "extractor_ids": ids,
            "training": {"scope": "synthetic unit fixture", "delay_training": False},
        },
        path,
    )
    return network, LearnedLatentPolicy(path), ids


def test_checkpoint_roundtrip_preserves_learned_messages_and_provenance(tmp_path):
    network, policy, ids = _checkpoint(tmp_path)
    vector = np.linspace(-1, 1, WIDTHS["left"], dtype=np.float32)
    expected = network.encode("left", torch.from_numpy(vector)[None])[0].detach().numpy()
    actual = policy.encode("left", vector, ids["left"])
    assert actual == pytest.approx(expected)
    assert policy.model_info["checkpoint_sha256"] == sha(policy.path)
    assert policy.model_info["source_extractor_ids"] == ids
    assert policy.model_info["native_action_heads_used"] is False
    assert policy.training["delay_training"] is False
    assert not policy.network.training
    assert all(not p.requires_grad for p in policy.network.parameters())


def test_runtime_rejects_wrong_extractor_and_conservatively_holds_missing_peer(tmp_path):
    _, policy, ids = _checkpoint(tmp_path)
    observation = _observation("left")
    vector = np.zeros(WIDTHS["left"], np.float32)
    with pytest.raises(ValueError, match="extractor"):
        policy.encode("left", vector, ids["right"])
    with pytest.raises(ValueError, match="local native"):
        policy.act(observation, "lift_hem", vector, None, extractor_id=ids["right"])
    action, trace = policy.act(observation, "lift_hem", vector, None, extractor_id=ids["left"])
    assert action == {"target": observation["ee_position"], "grip": observation["grasped"]}
    assert trace["missing_message_hold"] is True


def test_runtime_clamps_motion_and_rejects_wrong_or_expired_belief(tmp_path):
    _, policy, ids = _checkpoint(tmp_path)
    with torch.no_grad():
        for parameter in policy.network.parameters():
            parameter.zero_()
        policy.network.heads["left"][-1].bias.copy_(torch.tensor([100.0, 100.0, 100.0, 2.0, -10.0]))
    observation = _observation("left")
    vector = np.zeros(WIDTHS["left"], np.float32)
    belief = {
        "recipient": "left",
        "checkpoint_sha256": policy.sha256,
        "age_ticks": 0,
        "vector": [0.0] * DIMENSION,
        "features": {},
    }
    action, trace = policy.act(observation, "lift_hem", vector, belief, extractor_id=ids["left"])
    assert np.linalg.norm(np.array(action["target"]) - observation["ee_position"]) == pytest.approx(
        MAX_DELTA
    )
    assert action["grip"] is True
    assert trace["learned_hold"] is False
    for changed in ({"recipient": "right"}, {"checkpoint_sha256": "0" * 64}, {"age_ticks": 24}):
        with pytest.raises(ValueError, match="delivered adapter belief"):
            policy.act(
                observation, "lift_hem", vector, {**belief, **changed}, extractor_id=ids["left"]
            )


def _json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def paired_cache(tmp_path):
    dataset, cache = tmp_path / "data", tmp_path / "cache"
    dataset.mkdir()
    cache.mkdir()
    rows, images = [], []
    for split, seed in (("train", 10010), ("dev", 11), ("test", 104)):
        for arm in AGENTS:
            observation = _observation(arm)
            rgb = np.full((16, 16, 3), len(rows), np.uint8)
            episode = f"{split}-{seed}"
            action = {"target": [observation["ee_position"][0], -0.1, 0.11], "grip": False}
            rows.append(
                {
                    "id": f"{episode}-clear-{arm}",
                    "agent": arm,
                    "split": split,
                    "seed": seed,
                    "episode_id": episode,
                    "snapshot_id": f"{episode}-s0",
                    "pair_id": f"{episode}-clear",
                    "physical_state_sha256": hashlib.sha256(episode.encode()).hexdigest(),
                    "counterfactual": "clear",
                    "phase": "lift_hem",
                    "instruction": "Fold the T-shirt. lift hem.",
                    "rgb_index": len(images),
                    "rgb_sha256": array_hash(rgb),
                    "observation_sha256": canonical_hash(observation),
                    "observation": observation,
                    "action": action,
                    "labels": {"nominal_action": action, "coordinated_hold": False},
                }
            )
            images.append(rgb)
    _json(dataset / "rows.json", rows)
    np.savez_compressed(dataset / "rgb.npz", rgb=np.stack(images))
    for arm in AGENTS:
        contract = feature_contract("smolvla" if arm == "left" else "xvla")
        model = {
            **contract,
            "agent": arm,
            "backbone_frozen": True,
            "native_action_head_executed": False,
            "observation_numeric_state_used": False,
        }
        indices = np.array([i for i, row in enumerate(rows) if row["agent"] == arm])
        features = np.stack([np.full(WIDTHS[arm], i + 1, np.float32) for i in indices])
        path = cache / f"{arm}.npz"
        np.savez_compressed(path, row_indices=indices, features=features)
        _json(
            path.with_suffix(".json"),
            {
                "kind": "actual-frozen-vla-feature-cache",
                "model": model,
                "rows_sha256": sha(dataset / "rows.json"),
                "rgb_sha256": sha(dataset / "rgb.npz"),
                "cache_sha256": sha(path),
                "rows": len(indices),
                "audit": [
                    {
                        **model,
                        "row_index": int(i),
                        "row_id": rows[i]["id"],
                        "vector_sha256": hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest(),
                        "rgb_sha256": hashlib.sha256(images[i].tobytes()).hexdigest(),
                        "instruction_sha256": hashlib.sha256(
                            rows[i]["instruction"].encode()
                        ).hexdigest(),
                        "tokenization": {"full_prompt_roundtrip_verified": True},
                    }
                    for i, vector in zip(indices, features)
                ],
            },
        )
    return dataset, cache


def test_pair_loader_joins_original_rows_and_frozen_features_by_whole_episode(paired_cache):
    data, sources, episodes = load_pairs(*paired_cache)
    assert set(data) == {"train", "dev", "test"}
    assert len(episodes) == 3
    for split in data.values():
        assert split["size"] == 1
        for arm in AGENTS:
            assert split["features"][arm].shape == (1, WIDTHS[arm])
            assert sources[arm]["model"]["agent"] == arm


@pytest.mark.parametrize("rewrite_digest", [False, True])
def test_cache_bytes_and_per_row_feature_audit_both_detect_tampering(paired_cache, rewrite_digest):
    dataset, cache = paired_cache
    path = cache / "left.npz"
    with np.load(path) as stored:
        indices, features = stored["row_indices"], stored["features"]
    features[0, 0] += 1
    np.savez_compressed(path, row_indices=indices, features=features)
    if rewrite_digest:
        meta = json.loads(path.with_suffix(".json").read_text())
        meta["cache_sha256"] = sha(path)
        _json(path.with_suffix(".json"), meta)
    with pytest.raises(ValueError, match="cache|audit"):
        load_pairs(dataset, cache)


def test_fractional_cache_row_indices_are_rejected(paired_cache):
    dataset, cache = paired_cache
    path = cache / "left.npz"
    with np.load(path) as stored:
        indices, features = stored["row_indices"], stored["features"]
    np.savez_compressed(path, row_indices=indices.astype(float) + 0.25, features=features)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["cache_sha256"] = sha(path)
    _json(path.with_suffix(".json"), meta)
    with pytest.raises(ValueError, match="integer array"):
        load_pairs(dataset, cache)


def test_unknown_backbone_feature_identity_is_rejected(paired_cache):
    dataset, cache = paired_cache
    path = cache / "left.json"
    meta = json.loads(path.read_text())
    meta["model"]["feature_space_id"] = "0" * 64
    _json(path, meta)
    with pytest.raises(ValueError, match="provenance"):
        load_pairs(dataset, cache)


@pytest.mark.parametrize("leak", ["episode", "renamed_seed", "snapshot"])
def test_split_leakage_and_renamed_counterfactuals_are_rejected(paired_cache, leak):
    dataset, cache = paired_cache
    rows = json.loads((dataset / "rows.json").read_text())
    if leak == "episode":
        rows[2]["episode_id"] = rows[0]["episode_id"]
    elif leak == "renamed_seed":
        rows[2]["seed"] = rows[0]["seed"]
    else:
        rows[2]["snapshot_id"] = rows[0]["snapshot_id"]
    _json(dataset / "rows.json", rows)
    with pytest.raises(ValueError, match="episode|Snapshot"):
        load_pairs(dataset, cache)
