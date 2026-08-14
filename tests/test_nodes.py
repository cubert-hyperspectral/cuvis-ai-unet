"""Integration tests for the cuvis-ai node wrappers.

These require a cuvis-ai env (cuvis_ai_core on the path); they skip cleanly
where it is absent (e.g. a bare authoring machine).
"""

import pytest

pytest.importorskip("cuvis_ai_core")

import torch  # noqa: E402
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory  # noqa: E402
from cuvis_ai_schemas.execution import Context  # noqa: E402

import cuvis_ai_unet.node.dynunet as dynunet_module  # noqa: E402
from cuvis_ai_unet.node.dynunet import DynUNet  # noqa: E402
from cuvis_ai_unet.node.losses import (  # noqa: E402
    CrossEntropyLoss,
    DiceLoss,
    OHEMCrossEntropyLoss,
)

pytestmark = pytest.mark.unit

B, H, W, BANDS, K = 2, 24, 20, 8, 4


def _node(mode: str = "2d", num_classes: int = K, **kw) -> DynUNet:
    return DynUNet(mode=mode, in_channels=BANDS, num_classes=num_classes, features=(8, 16), **kw)


@pytest.mark.parametrize("mode", ["2d", "2p5d", "3d"])
def test_node_forward_bhwc(mode: str) -> None:
    out = _node(mode).forward(torch.randn(B, H, W, BANDS))
    assert out["logits"].shape == (B, H, W, K)
    assert out["logits"].dtype == torch.float32


@pytest.mark.parametrize("mode", ["2d", "2p5d", "3d"])
def test_node_pads_odd_sizes(mode: str) -> None:
    # The strict backbone would reject 49x41; the node grid-pads and crops back.
    node = _node(mode)
    ctx = Context(stage=ExecutionStage.TRAIN)
    out = node.forward(torch.randn(B, 49, 41, BANDS), context=ctx)
    assert out["logits"].shape == (B, 49, 41, K)


def test_node_contract() -> None:
    assert "data" in DynUNet.INPUT_SPECS
    assert "logits" in DynUNet.OUTPUT_SPECS
    assert DynUNet._category == NodeCategory.MODEL


def test_hparams_and_state_dict_reload_with_tiling() -> None:
    # Non-default tiling values: catches the serializer silent-default gotcha
    # (a forgotten super().__init__ forward would record the defaults instead).
    n1 = _node("2p5d", tile_size=(16, 16), tile_overlap=0.25, tile_gaussian=False).eval()
    hp = n1.hparams
    assert hp["mode"] == "2p5d" and hp["in_channels"] == BANDS and hp["num_classes"] == K
    assert tuple(hp["tile_size"]) == (16, 16)
    assert hp["tile_overlap"] == 0.25
    assert hp["tile_gaussian"] is False
    n2 = DynUNet(
        mode=hp["mode"],
        in_channels=hp["in_channels"],
        num_classes=hp["num_classes"],
        features=hp["features"],
        tile_size=hp["tile_size"],
        tile_overlap=hp["tile_overlap"],
        tile_gaussian=hp["tile_gaussian"],
    ).eval()
    n2.load_state_dict(n1.state_dict())
    x = torch.randn(B, H, W, BANDS)
    with torch.no_grad():
        assert torch.allclose(n1.forward(x)["logits"], n2.forward(x)["logits"], atol=1e-6)


def test_stage_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node("2d", tile_size=(16, 16))
    calls = {"n": 0}
    real = dynunet_module.sliding_window_inference

    def counting(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(dynunet_module, "sliding_window_inference", counting)
    oversized = torch.randn(1, 24, 20, BANDS)  # > (16, 16) tile

    node.train()
    node.forward(oversized, context=Context(stage=ExecutionStage.TRAIN))
    assert calls["n"] == 0  # TRAIN never tiles
    node.forward(oversized, context=Context(stage=ExecutionStage.VAL))
    assert calls["n"] == 1  # VAL + oversize tiles
    node.eval()
    node.forward(oversized)  # no context: module flag decides
    assert calls["n"] == 2
    node.train()
    node.forward(oversized)
    assert calls["n"] == 2  # no context + training mode -> direct
    node.eval()
    node.forward(torch.randn(1, 16, 16, BANDS))  # exactly tile-sized: not oversize
    assert calls["n"] == 2


def test_tile_validation_errors() -> None:
    with pytest.raises(ValueError, match="divisible"):
        # 3-stage net -> grid (4, 4); 18 % 4 != 0
        DynUNet(
            mode="2d", in_channels=BANDS, num_classes=2, features=(8, 16, 32), tile_size=(18, 16)
        )
    with pytest.raises(ValueError, match="tile_overlap"):
        _node("2d", tile_size=(16, 16), tile_overlap=1.0)


def test_bottleneck_and_depth_warnings() -> None:
    with pytest.warns(UserWarning, match="InstanceNorm"):
        DynUNet(
            mode="2d", in_channels=BANDS, num_classes=2, features=(8, 16, 32), tile_size=(8, 8)
        )  # 8 / grid 4 = 2 < 4
    with pytest.warns(UserWarning, match="spectral stride grid"):
        DynUNet(
            mode="3d", in_channels=3, num_classes=2, features=(8, 16, 32), spectral_downsample=True
        )  # depth grid 4 > 3 bands


@pytest.mark.parametrize("loss_cls", [DiceLoss, CrossEntropyLoss, OHEMCrossEntropyLoss])
def test_loss_nodes(loss_cls: type) -> None:
    node = _node()
    logits = node.forward(torch.randn(B, H, W, BANDS))["logits"]
    targets = torch.randint(0, K, (B, H, W, 1))
    ln = loss_cls()
    assert ExecutionStage.INFERENCE not in ln.execution_stages
    loss = ln.forward(logits, targets)["loss"]
    assert loss.dim() == 0
    node.zero_grad()
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in node.parameters() if p.grad is not None) > 0


def test_binary_target_single_logit() -> None:
    node = _node(num_classes=1)
    logits = node.forward(torch.randn(B, H, W, BANDS))["logits"]
    assert logits.shape == (B, H, W, 1)
    targets = (torch.rand(B, H, W, 1) > 0.5).long()
    loss = DiceLoss().forward(logits, targets)["loss"]
    assert loss.dim() == 0


def test_ohem_equals_plain_ce_when_all_pixels_kept() -> None:
    # ratio + min_kept large enough to keep every background pixel -> OHEM averages
    # over all pixels, which is exactly plain cross-entropy mean reduction.
    import torch.nn.functional as F

    torch.manual_seed(0)
    logits = torch.randn(B, H, W, K)
    targets = torch.randint(0, K, (B, H, W))
    ohem = OHEMCrossEntropyLoss(ratio=1e9, min_kept=B * H * W).forward(logits, targets)["loss"]
    plain = F.cross_entropy(logits.permute(0, 3, 1, 2), targets.long())
    assert torch.allclose(ohem, plain, atol=1e-6)


def test_ohem_upweights_hard_background() -> None:
    # Mostly-easy background with a few hard (wrong-but-confident) pixels: dropping the
    # easy majority makes the OHEM mean strictly exceed the plain CE mean.
    import torch.nn.functional as F

    torch.manual_seed(1)
    logits = torch.full((1, 8, 8, 2), 0.0)
    logits[..., 0] = 8.0  # confident background everywhere (correct)
    targets = torch.zeros(1, 8, 8, dtype=torch.long)
    # make 4 pixels hard: model confidently predicts FG but they are background
    logits[0, 0, :4, 0] = -8.0
    logits[0, 0, :4, 1] = 8.0
    ohem = OHEMCrossEntropyLoss(ratio=1.0, min_kept=4).forward(logits, targets)["loss"]
    plain = F.cross_entropy(logits.permute(0, 3, 1, 2), targets)
    assert ohem > plain


def test_ohem_no_foreground_uses_min_kept() -> None:
    node = _node()
    logits = node.forward(torch.randn(B, H, W, BANDS))["logits"]
    targets = torch.zeros(B, H, W, dtype=torch.int32)  # all background
    loss = OHEMCrossEntropyLoss(min_kept=16).forward(logits, targets)["loss"]
    assert loss.dim() == 0 and torch.isfinite(loss)
    node.zero_grad()
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in node.parameters() if p.grad is not None) > 0


def test_ohem_ignore_index_excludes_void_pixels() -> None:
    # The codebase's void convention (255) must not crash and must not contribute.
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 4, 2)
    targets = torch.zeros(1, 4, 4, dtype=torch.int64)
    targets[0, 0, :] = 255  # a void row
    targets[0, 1, 0] = 1  # one foreground pixel

    loss = OHEMCrossEntropyLoss(ratio=1e9, min_kept=64, ignore_index=255).forward(logits, targets)[
        "loss"
    ]

    # reference: keep-everything OHEM == mean CE over the VALID pixels only
    per_pixel = torch.nn.functional.cross_entropy(
        logits.permute(0, 3, 1, 2), targets, reduction="none", ignore_index=255
    )
    expected = per_pixel[targets != 255].mean()
    assert torch.allclose(loss, expected)


def test_ohem_void_pixels_never_dilute_the_kept_set() -> None:
    # Reviewer scenario: bg pool smaller than k (small crop) with void pixels around.
    # The zero-loss void pixels must not be pulled into the kept set.
    torch.manual_seed(1)
    logits = torch.randn(1, 2, 4, 3)
    targets = torch.full((1, 2, 4), 255, dtype=torch.int64)
    targets[0, 0, :2] = 0  # only two valid background pixels, everything else void

    loss = OHEMCrossEntropyLoss(min_kept=4096, ignore_index=255).forward(logits, targets)["loss"]
    per_pixel = torch.nn.functional.cross_entropy(
        logits.permute(0, 3, 1, 2), targets, reduction="none", ignore_index=255
    )
    expected = per_pixel[targets == 0].mean()  # mean over the two valid bg pixels ONLY
    assert torch.allclose(loss, expected)
    # diluted (wrong) value would average zeros in: strictly smaller
    assert loss > per_pixel.mean()


def test_ohem_default_path_unchanged_without_void_labels() -> None:
    # Regression guard: with plain {0,1} masks and the default ignore_index, the
    # loss equals the original formula (fg + top-k of ALL background pixels).
    torch.manual_seed(2)
    logits = torch.randn(2, 8, 8, 2)
    targets = (torch.rand(2, 8, 8) > 0.8).long()

    loss = OHEMCrossEntropyLoss(ratio=2.0, min_kept=8).forward(logits, targets)["loss"]

    per_pixel = torch.nn.functional.cross_entropy(
        logits.permute(0, 3, 1, 2), targets, reduction="none"
    )
    fg = targets > 0
    k = min(max(int(2.0 * max(int(fg.sum()), 1)), 8), int((~fg).sum()))
    expected = torch.cat([per_pixel[fg], torch.topk(per_pixel[~fg], k).values]).mean()
    assert torch.allclose(loss, expected)
