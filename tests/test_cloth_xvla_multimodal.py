from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth.xvla_adaptation import CHANNELS, _read_adapter_tensors, save_adapter
from relay_lab.cloth.xvla_multimodal import configure_multimodal, frozen_hash
from relay_lab.cloth.xvla_training import native_forward, native_loss


def test_native_multimodal_gradients_cross_frozen_blocks_without_updating_them():
    torch = pytest.importorskip("torch")
    from lerobot.policies.xvla.action_hub import EE6DActionSpace
    from lerobot.policies.xvla.soft_transformer import SoftPromptedTransformer

    torch.manual_seed(712)
    transformer = SoftPromptedTransformer(
        hidden_size=8,
        multi_modal_input_size=8,
        depth=2,
        num_heads=2,
        num_domains=30,
        len_soft_prompts=2,
    ).eval()
    model = SimpleNamespace(transformer=transformer, action_space=EE6DActionSpace())
    groups, hooks = configure_multimodal(transformer)
    frozen = frozen_hash(transformer)
    old_decoder = transformer.action_decoder.fc.weight.detach().clone()
    old_prompts = transformer.soft_prompt_hub.weight.detach().clone()
    optimizer = torch.optim.Adam(groups, lr=0.001)
    batch = {
        "vlm_features": torch.randn(2, 3, 8),
        "aux_visual_inputs": torch.randn(2, 2, 8),
        "proprio": torch.zeros(2, 20),
    }
    predicted = native_forward(model, batch, torch.randn(1, 30, 20))
    loss, _ = native_loss(predicted, torch.tensor([[0.1, 0.2, 0.3, 1], [0.2, 0.3, 0.4, 0]]))
    loss.backward()
    assert transformer.vlm_proj.weight.grad.abs().sum() > 0
    assert transformer.soft_prompt_hub.weight.grad[5].abs().sum() > 0
    assert transformer.norm.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None for block in transformer.blocks for parameter in block.parameters()
    )
    assert transformer.action_encoder.fc.weight.grad is None
    optimizer.step()
    assert frozen_hash(transformer) == frozen
    assert torch.equal(transformer.soft_prompt_hub.weight[:5], old_prompts[:5])
    unused = [index for index in range(20) if index not in CHANNELS]
    assert torch.equal(
        transformer.action_decoder.fc.weight[5].reshape(8, 20)[:, unused],
        old_decoder[5].reshape(8, 20)[:, unused],
    )
    for hook in hooks:
        hook.remove()


def test_only_exact_native_projection_and_own_prompt_tensors_can_be_installed(tmp_path):
    pytest.importorskip("safetensors")
    multimodal = {
        "vlm_proj.weight": np.zeros((1024, 1024), np.float32),
        "vlm_proj.bias": np.zeros(1024, np.float32),
        "soft_prompt_hub.weight": np.zeros(32768, np.float32),
    }
    norm = {"norm.weight": np.ones(1024, np.float32), "norm.bias": np.zeros(1024, np.float32)}
    result = save_adapter(
        tmp_path / "native",
        np.zeros((1024, 20), np.float32),
        np.zeros(20, np.float32),
        dataset_sha256="a" * 64,
        training={},
        native_multimodal=multimodal,
        native_norm=norm,
    )
    assert result["adapted_native_multimodal"] == ["vlm_proj", "soft_prompt_hub"]
    assert result["trainable_parameters"] == 1088516
    metadata, tensors = _read_adapter_tensors(
        tmp_path / "native", expected_sha256=result["decoder_sha256"]
    )
    assert metadata == result
    assert np.array_equal(tensors["vlm_proj.weight"], multimodal["vlm_proj.weight"])
    with pytest.raises(ValueError, match="multimodal"):
        save_adapter(
            tmp_path / "invalid",
            np.zeros((1024, 20), np.float32),
            np.zeros(20, np.float32),
            dataset_sha256="a" * 64,
            training={},
            native_multimodal={**multimodal, "replacement.weight": np.ones(1)},
        )
