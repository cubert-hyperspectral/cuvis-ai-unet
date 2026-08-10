"""Integration: a DynUNet pipeline survives save_to_file -> load_pipeline -> forward.

The manifest-loading smoke (test_plugin_manifest_loading) proves the nodes resolve
from the manifest; this proves the stronger reload contract the plugin skill calls
out — a built pipeline serializes, rebuilds from its config through a fresh
NodeRegistry, restores its weights, and reproduces its output to within a 1e-6
tolerance. It uses
only the plugin's own nodes (DynUNet + the loss nodes), so it needs neither the
augment plugin nor the running-stats normalizer, and runs CPU-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parents[1] / "plugins.yaml"


@pytest.mark.integration
def test_pipeline_save_load_forward_roundtrip(tmp_path) -> None:
    import torch
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.utils.node_registry import NodeRegistry
    from cuvis_ai_schemas.enums import ExecutionStage

    from cuvis_ai_unet.node.dynunet import DynUNet
    from cuvis_ai_unet.node.losses import CrossEntropyLoss, DiceLoss

    in_channels, num_classes = 4, 2
    net = DynUNet(
        name="DynUNet",
        mode="2d",
        in_channels=in_channels,
        num_classes=num_classes,
        features=(8, 16),
        tile_size=None,  # small synthetic frame -> direct (untiled) forward
    )
    dice = DiceLoss(name="DiceLoss")
    ce = CrossEntropyLoss(name="CrossEntropyLoss")

    pipe = CuvisPipeline("roundtrip")
    # DynUNet feeds both losses; the losses are TRAIN/VAL/TEST-gated, so at INFERENCE
    # only DynUNet executes, taking its cube from the injected batch.
    pipe.connect(
        (net.outputs.logits, dice.inputs.logits),
        (net.outputs.logits, ce.inputs.logits),
    )

    cube = torch.randn(1, 32, 40, in_channels)  # [B, H, W, C]
    batch = {"data": cube}
    before = pipe.forward(batch=batch, stage=ExecutionStage.INFERENCE)[("DynUNet", "logits")]
    assert before.shape == (1, 32, 40, num_classes)

    # Serialize, then rebuild from the config through a *fresh* registry + restore weights.
    pipe.save_to_file(str(tmp_path / "p.yaml"), save_weights=True)
    fresh = NodeRegistry()
    fresh.register_plugin(str(MANIFEST))
    restored = CuvisPipeline.load_pipeline(
        str(tmp_path / "p.yaml"),
        weights_path=str(tmp_path / "p.pt"),
        node_registry=fresh,
    )
    after = restored.forward(batch=batch, stage=ExecutionStage.INFERENCE)[("DynUNet", "logits")]

    assert after.shape == before.shape
    assert torch.allclose(before, after, atol=1e-6)  # reload reproduces the output
