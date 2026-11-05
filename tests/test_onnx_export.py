"""ONNX export of :class:`DynUNet`: the adapter, the traced graph, and the deploy folder.

``export_module`` is checked three ways, narrowest first: the adapter alone against
``forward`` (no exporter involved), the full ``torch.onnx.export`` + ONNX Runtime
parity on a tiny net, then the deploy-folder round trip through
``cuvis_ai_core.pipeline.export.export_deployment`` and back through
``cuvis_ai.node.exported_span.ExportedSpan``. The onnx extras are optional at runtime,
so the last two groups skip cleanly without them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cuvis_ai_core")

import torch  # noqa: E402
from cuvis_ai_core.node.node import Node  # noqa: E402
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag  # noqa: E402
from cuvis_ai_schemas.pipeline import PortSpec  # noqa: E402

from cuvis_ai_unet.node.dynunet import DynUNet  # noqa: E402

pytestmark = pytest.mark.unit

B, BANDS, K = 2, 6, 3


def _node(**kw) -> DynUNet:
    return DynUNet(mode="2d", in_channels=BANDS, num_classes=K, features=(8, 16), **kw).eval()


# --------------------------------------------------------------------- adapter parity


def test_node_tag_onnx() -> None:
    assert NodeTag.ONNX in DynUNet._tags


@pytest.mark.parametrize("h,w", [(16, 16), (18, 14), (24, 20)])
def test_export_module_matches_forward(h: int, w: int) -> None:
    # 18x14 is not a multiple of the net's stride grid (4, 4): the adapter must pad
    # and crop exactly like forward's direct path.
    torch.manual_seed(0)
    node = _node()
    adapter = node.export_module()
    assert adapter is not None
    adapter.eval()

    data = torch.randn(B, h, w, BANDS)
    with torch.no_grad():
        expected = node.forward(data)["logits"]
        got = adapter(data=data)["logits"]

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-6)


def test_export_module_no_adapter_when_tiled() -> None:
    node = _node(tile_size=(16, 16))
    with pytest.raises(Exception, match="tile"):
        node.export_module()


def test_export_module_shares_node_weights() -> None:
    # The exporter calls export_module() on a deep-copied node, already eval()'d
    # and discarded after tracing; the adapter must read that copy's live weights
    # rather than a frozen snapshot, so it tracks in-place parameter updates.
    node = _node()
    adapter = node.export_module()
    assert adapter is not None
    adapter.eval()

    data = torch.randn(1, 16, 16, BANDS)
    with torch.no_grad():
        before = adapter(data=data)["logits"].clone()
        for p in node.parameters():
            p.add_(1.0)
        after = adapter(data=data)["logits"]
    assert not torch.allclose(before, after)


# --------------------------------------------------------------------- graph export


def test_export_onnx_runtime_parity(tmp_path) -> None:
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    import onnx  # noqa: F401 (optional dep declares the ir; exercised via torch.onnx.export)

    torch.manual_seed(1)
    node = _node()
    adapter = node.export_module()
    assert adapter is not None
    adapter.eval()

    data = torch.randn(1, 20, 18, BANDS)
    onnx_path = tmp_path / "dynunet.onnx"
    with torch.no_grad():
        expected = adapter(data=data)["logits"]
    torch.onnx.export(
        adapter,
        (),
        str(onnx_path),
        kwargs={"data": data},
        dynamo=True,
        opset_version=18,
        input_names=["data"],
        output_names=["logits"],
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (got,) = session.run(["logits"], {"data": data.numpy()})
    assert torch.allclose(torch.from_numpy(got), expected, atol=1e-5, rtol=1e-4)


def test_tiled_config_refuses_export() -> None:
    pytest.importorskip("onnxscript")
    node = _node(tile_size=(16, 16))
    with pytest.raises(Exception, match="[Tt]il"):
        node.export_module()


# --------------------------------------------------------------------- deploy folder


class _MockSource(Node):
    """A fixed-cube source node standing in for a real data loader in the span test."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.HYPERSPECTRAL})
    EXECUTION_STAGES = frozenset({ExecutionStage.INFERENCE})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1, -1), description="A fixed BHWC cube."
        )
    }

    def __init__(self, cube: torch.Tensor, **kwargs) -> None:
        super().__init__(cube=cube, **kwargs)
        self.register_buffer("_cube", cube)
        # A real loader node's hparams are a config (a path, a selector), never the
        # data itself; drop the tensor the base class captured so the deploy yaml's
        # pydantic serialization (exercised by nodes that stay outside the span)
        # has something to serialize.
        self.hparams = {}

    def forward(self, **_: object) -> dict[str, torch.Tensor]:
        return {"cube": self._cube}


def test_deploy_folder_round_trip(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from cuvis_ai_core.pipeline.export import export_deployment
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

    torch.manual_seed(2)
    cube = torch.randn(1, 20, 18, BANDS)
    source = _MockSource(cube, name="source")
    detector = DynUNet(
        name="detector", mode="2d", in_channels=BANDS, num_classes=K, features=(8, 16)
    ).eval()

    pipe = CuvisPipeline("unet_export_roundtrip")
    pipe.connect((source.outputs.cube, detector.inputs.data))

    examples = [{"detector.data": cube}]
    deployment = export_deployment(
        pipe,
        tmp_path,
        inputs=["detector.data"],
        outputs=["detector.logits"],
        examples=examples,
        name="unet_export_roundtrip",
    )
    assert deployment.onnx_path.is_file()
    assert deployment.sidecar_path.is_file()

    import os

    from cuvis_ai.node.exported_span import ExportedSpan

    os.environ["CUVIS_AI_DEPLOY_DIR"] = str(tmp_path)
    try:
        span_node = ExportedSpan(onnx_path=deployment.onnx_path.name, name="exported_span")
        span_node.eval()
        with torch.no_grad():
            expected = detector.forward(cube)["logits"]
            port_name = deployment.sidecar["span"]["inputs"][0]["name"]
            got = span_node.forward(**{port_name: cube})
            out_name = deployment.sidecar["span"]["outputs"][0]["name"]
    finally:
        del os.environ["CUVIS_AI_DEPLOY_DIR"]

    assert torch.allclose(got[out_name], expected, atol=1e-5, rtol=1e-4)
