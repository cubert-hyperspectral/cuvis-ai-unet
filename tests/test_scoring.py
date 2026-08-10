"""Tests for the SegmentationAnomalyScore node."""

import pytest

pytest.importorskip("cuvis_ai_core")

import torch  # noqa: E402
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory  # noqa: E402

from cuvis_ai_unet.node.scoring import SegmentationAnomalyScore  # noqa: E402

pytestmark = pytest.mark.unit

B, H, W, K = 2, 16, 12, 2


def test_contract_and_stages() -> None:
    assert set(SegmentationAnomalyScore.INPUT_SPECS) == {"logits", "targets"}
    assert SegmentationAnomalyScore.INPUT_SPECS["targets"].optional is True
    assert set(SegmentationAnomalyScore.OUTPUT_SPECS) == {"scores", "anomaly_score", "targets_bool"}
    assert SegmentationAnomalyScore._category == NodeCategory.TRANSFORM
    node = SegmentationAnomalyScore()
    assert ExecutionStage.TRAIN not in node.execution_stages
    assert ExecutionStage.INFERENCE in node.execution_stages


def test_forward_without_targets() -> None:
    node = SegmentationAnomalyScore()
    out = node.forward(logits=torch.randn(B, H, W, K))
    assert out["scores"].shape == (B, H, W, 1)
    assert out["anomaly_score"].shape == (B,)
    assert "targets_bool" not in out  # no mask connected


def test_forward_with_targets_emits_bool_mask() -> None:
    node = SegmentationAnomalyScore()
    targets = torch.randint(0, K, (B, H, W), dtype=torch.int32)
    out = node.forward(logits=torch.randn(B, H, W, K), targets=targets)
    assert out["targets_bool"].dtype == torch.bool
    assert out["targets_bool"].shape == (B, H, W, 1)
    assert torch.equal(out["targets_bool"][..., 0], targets >= 1)


def test_score_is_foreground_probability() -> None:
    # Class-1 logit dominant on all pixels -> FO prob ~1 -> anomaly_score ~1.
    logits = torch.zeros(1, H, W, 2)
    logits[..., 1] = 20.0
    out = SegmentationAnomalyScore().forward(logits=logits)
    assert torch.allclose(out["anomaly_score"], torch.ones(1), atol=1e-4)
    # Background-dominant -> FO prob ~0.
    logits[...] = 0.0
    logits[..., 0] = 20.0
    out = SegmentationAnomalyScore().forward(logits=logits)
    assert torch.allclose(out["anomaly_score"], torch.zeros(1), atol=1e-4)


def test_top_frac_averages_only_the_hottest_pixels() -> None:
    # A single hot pixel among many cold ones: top_frac=1/npix isolates just it.
    npix = H * W
    logits = torch.zeros(1, H, W, 2)
    logits[..., 0] = 20.0  # background elsewhere -> FO prob ~0
    logits[0, 0, 0, 0] = 0.0
    logits[0, 0, 0, 1] = 20.0  # one hot FO pixel -> prob ~1
    node = SegmentationAnomalyScore(top_frac=1.0 / npix)
    out = node.forward(logits=logits)
    assert out["anomaly_score"].item() > 0.99  # the hottest single pixel


def test_top_frac_selects_exact_pixel_count() -> None:
    # Lock the integer-floor convention k = int((1 - top_frac) * N), mean of flat[k:]
    # (shared with cuvis_ai_rfdetr.functional.top_frac_mean). N=192; 10 hot pixels.
    h2, w2 = 16, 12
    n = h2 * w2
    logits = torch.zeros(1, h2, w2, 2)
    logits[..., 0] = 20.0  # background everywhere -> FO prob ~0
    for i in range(10):  # 10 foreground pixels -> FO prob ~1
        r, c = divmod(i, w2)
        logits[0, r, c, 0] = 0.0
        logits[0, r, c, 1] = 20.0
    s10 = SegmentationAnomalyScore(top_frac=10.0 / n).forward(logits=logits)["anomaly_score"].item()
    s20 = SegmentationAnomalyScore(top_frac=20.0 / n).forward(logits=logits)["anomaly_score"].item()
    assert abs(s10 - 1.0) < 1e-3  # k selects exactly the 10 hot pixels
    assert abs(s20 - 0.5) < 5e-3  # k selects 10 hot + 10 zero -> mean 0.5


def test_invalid_top_frac_raises() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            SegmentationAnomalyScore(top_frac=bad)


def test_rejects_execution_stages_override() -> None:
    with pytest.raises(AssertionError):
        SegmentationAnomalyScore(execution_stages={ExecutionStage.TRAIN})
