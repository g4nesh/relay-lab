"""Exercise the actual original final transformer block without model weights."""

from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth.xvla_adaptation import (
    CHANNELS,
    load_adapter,
    native_block_shapes,
    save_adapter,
)
from relay_lab.cloth.xvla_final_block import cache_prefix, configure_final_gradients, forward_final
from relay_lab.cloth.xvla_training import native_forward, native_loss


def test_final_block_cache_is_exact_and_gradient_scope_preserves_frozen_prefix():
    torch = pytest.importorskip("torch")
    from lerobot.policies.xvla.action_hub import EE6DActionSpace
    from lerobot.policies.xvla.soft_transformer import SoftPromptedTransformer

    torch.manual_seed(113)
    transformer = SoftPromptedTransformer(
        hidden_size=8,
        multi_modal_input_size=8,
        depth=2,
        num_heads=2,
        num_domains=30,
        len_soft_prompts=2,
    ).eval()
    model = SimpleNamespace(transformer=transformer, action_space=EE6DActionSpace())
    batch = {
        "vlm_features": torch.randn(2, 3, 8),
        "aux_visual_inputs": torch.randn(2, 2, 8),
        "proprio": torch.zeros(2, 20),
    }
    captured = []
    handle = transformer.blocks[-1].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    with torch.no_grad():
        original = native_forward(model, batch, torch.randn(1, 30, 20))
    handle.remove()
    assert len(captured) == 1
    assert torch.equal(original, forward_final(transformer, captured[0]))
    before = {
        name: parameter.detach().clone() for name, parameter in transformer.named_parameters()
    }
    groups, handles = configure_final_gradients(transformer)
    optimizer = torch.optim.Adam(groups, lr=1e-3, weight_decay=0)
    prediction = forward_final(transformer, captured[0])
    loss, _ = native_loss(prediction, torch.tensor([[0.1, 0.2, 0.3, 1.0], [0.2, 0.3, 0.4, 0.0]]))
    loss.backward()
    assert transformer.blocks[-1].attn.qkv.weight.grad.abs().sum() > 0
    optimizer.step()
    for name, parameter in transformer.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(parameter, before[name])
    untouched = [index for index in range(20) if index not in CHANNELS]
    assert torch.equal(
        transformer.action_decoder.fc.weight[5].view(8, 20)[:, untouched],
        before["action_decoder.fc.weight"][5].view(8, 20)[:, untouched],
    )
    assert not torch.equal(
        transformer.blocks[-1].attn.qkv.weight, before["blocks.1.attn.qkv.weight"]
    )
    assert not transformer.training
    for hook in handles:
        hook.remove()


def test_only_exact_original_block_23_weights_can_be_saved(tmp_path):
    pytest.importorskip("safetensors")
    block = {name: np.zeros(shape, np.float32) for name, shape in native_block_shapes().items()}
    metadata = save_adapter(
        tmp_path / "block",
        np.zeros((1024, 20), np.float32),
        np.zeros(20, np.float32),
        dataset_sha256="a" * 64,
        training={},
        native_block=block,
    )
    assert metadata["adapted_native_blocks"] == [23]
    loaded, _, _ = load_adapter(tmp_path / "block", expected_sha256=metadata["decoder_sha256"])
    assert loaded == metadata
    block["replacement_controller.weight"] = np.zeros(1, np.float32)
    with pytest.raises(ValueError, match="final-block"):
        save_adapter(
            tmp_path / "invalid",
            np.zeros((1024, 20), np.float32),
            np.zeros(20, np.float32),
            dataset_sha256="a" * 64,
            training={},
            native_block=block,
        )
    assert not (tmp_path / "invalid").exists()


def test_prefix_cache_rejects_unbound_multimodal_projection_before_read_or_write(tmp_path):
    native = SimpleNamespace(adapter={"adapted_native_multimodal": ["vlm_proj", "soft_prompt_hub"]})
    with pytest.raises(ValueError, match="adapted VLM projection"):
        cache_prefix(native, tmp_path / "missing-input", tmp_path / "output")
    assert not (tmp_path / "output").exists()
