"""Integration smoke: the shipped manifest loads via NodeRegistry and resolves all nodes."""

from __future__ import annotations

from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parents[1] / "plugins.yaml"


@pytest.mark.integration
def test_manifest_loads_and_registers_nodes() -> None:
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    registry = NodeRegistry()
    registry.register_plugin(str(MANIFEST))
    assert "unet" in registry.list_plugins()
    for cls_name in (
        "DynUNet",
        "DiceLoss",
        "CrossEntropyLoss",
        "OHEMCrossEntropyLoss",
        "SegMetrics",
        "SegmentationAnomalyScore",
    ):
        assert registry.get(cls_name) is not None, f"{cls_name} not resolvable from the manifest"
