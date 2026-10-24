"""Learned private-cue perception and matched text/vector communication tests."""

import numpy as np
import pytest

from relay_lab.factory.routing_codec import (
    DEFAULT_CHECKPOINT,
    RoutingCodec,
    RoutingObservationError,
)


@pytest.fixture(scope="module")
def codec():
    pytest.importorskip("torch")
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip("Train the original private inspection codec first")
    return RoutingCodec()


@pytest.fixture(scope="module")
def frames():
    path = DEFAULT_CHECKPOINT.with_name("verification_rgb.npz")
    if not path.is_file():
        pytest.skip("Original held-out RGB fixtures are optional")
    with np.load(path) as data:
        return data["images"].copy(), data["labels"].copy()


def test_private_rgb_is_the_only_encoder_input(codec):
    with pytest.raises(ValueError, match="private RGB"):
        codec.encode({"route": "A"})
    with pytest.raises(ValueError, match="private RGB"):
        codec.encode(np.zeros((32, 32, 3), dtype=np.uint8))


def test_learned_camera_reads_actual_heldout_crops(codec, frames):
    images, labels = frames
    for image, label in zip(images, labels, strict=True):
        encoded = codec.encode(image)
        if label == 2:
            with pytest.raises(RoutingObservationError):
                codec.decode(encoded["vector"])
        else:
            assert codec.decode(encoded["vector"])["route"] == ("A", "B")[label]


def test_route_label_is_absent_from_encoder_envelope(codec, frames):
    image = frames[0][0]
    encoded = codec.encode(image)
    assert len(encoded["vector"]) == 32
    assert abs(np.linalg.norm(encoded["vector"]) - 1) < 1e-5
    assert encoded["space_id"] == "factory-route-v1"
    assert len(encoded["checkpoint_sha256"]) == 64
    assert len(encoded["source_frame_sha256"]) == 64
    assert not ({"route", "text", "truth", "expected_slot", "label"} & set(encoded))


def test_text_and_vector_use_identical_learned_decision(codec, frames):
    import json

    images, labels = frames
    for image, label in zip(images, labels, strict=True):
        if label == 2:
            continue
        encoded = codec.encode(image)
        sender_decision = codec.decode(encoded["vector"])
        text_received = json.loads(sender_decision["text"])
        vector_received = json.loads(json.dumps(encoded["vector"]))
        receiver_decision = codec.decode(vector_received)
        assert text_received["route"] == receiver_decision["route"]
        assert sender_decision["confidence"] == receiver_decision["confidence"]
        assert sender_decision["text"] in ('{"route":"A"}', '{"route":"B"}')


def test_image_ablation_abstains(codec):
    zero = codec.encode(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(RoutingObservationError):
        codec.decode(zero["vector"])


def test_decoder_weights_are_causally_required(codec, frames):
    clone = RoutingCodec()
    vector = clone.encode(frames[0][0])["vector"]
    assert clone.decode(vector)["route"] == "A"
    with clone.torch.no_grad():
        for parameter in clone.network.route_head.parameters():
            parameter.zero_()
    with pytest.raises(RoutingObservationError):
        clone.decode(vector)


def test_corrupt_latent_rejected(codec):
    with pytest.raises(ValueError, match="normalized"):
        codec.decode([0.0] * 32)
    with pytest.raises(ValueError, match="normalized"):
        codec.decode([float("nan")] + [0.0] * 31)
    with pytest.raises(ValueError, match="normalized"):
        codec.decode([1.0] + [0.0] * 30)


def test_checkpoint_cannot_change_private_camera_contract(codec, tmp_path):
    payload = codec.torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    payload["sensor_signature"] = "0" * 64
    path = tmp_path / "bad.pt"
    codec.torch.save(payload, path)
    with pytest.raises(ValueError, match="camera contract mismatch"):
        RoutingCodec(path)


def test_received_real_latent_changes_gantry_choice_with_identical_local_inputs(codec, frames):
    """Substitute only received visual evidence; keep job, index and clock fixed."""
    from relay_lab.factory.routing_exchange import RoutingExchange

    class CommittedPacketReader:
        packet = None

        def retrieve_semantic_packet(self, *, recipient, job_id, tick):
            assert (recipient, job_id, tick) == ("gantry-01", "job-002", 100)
            return self.packet

    class InaccessibleWorld:
        def __getattribute__(self, name):
            raise AssertionError("Receiver must not inspect simulator or private camera")

    reader = CommittedPacketReader()
    exchange = RoutingExchange(
        InaccessibleWorld(), reader, "counterfactual", mode="embeddings", codec=codec
    )
    choices = []
    for label in (0, 1):
        image = frames[0][np.flatnonzero(frames[1] == label)[0]]
        encoded = codec.encode(image)
        reader.packet = {
            "vector": encoded["vector"],
            "packet_id": "humanoid-01-job-002-100",
            "observation_tick": 20,
            "emitted_tick": 100,
            "source_frame_sha256": encoded["source_frame_sha256"],
        }
        choices.append(exchange.decide("job-002", 1, 100))
    assert [decision["route"] for decision in choices] == ["A", "B"]
    assert [decision["pallet_slot"] for decision in choices] == [2, 3]
    assert choices[0]["job_id"] == choices[1]["job_id"]
    assert choices[0]["tick"] == choices[1]["tick"]
    reader.packet = None
    assert exchange.decide("job-002", 1, 100)["pallet_slot"] == 2
