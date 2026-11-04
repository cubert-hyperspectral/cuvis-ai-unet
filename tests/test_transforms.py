"""RandomForegroundBiasedMultiCrop: N crops from every input frame, read once."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cuvis_ai_augment")

from cuvis_ai_augment.transforms.base import TRANSFORM_REGISTRY  # noqa: E402

from cuvis_ai_unet.transforms import (  # noqa: E402
    RandomForegroundBiasedCrop,
    RandomForegroundBiasedMultiCrop,
)

B, H, W, C, N, P = 2, 48, 64, 3, 4, 16


def _frames() -> torch.Tensor:
    """Frame b is filled with the value b, so a crop shows which frame it came from."""
    return torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1).expand(B, H, W, C).clone()


def _one_fg_pixel_per_frame() -> tuple[torch.Tensor, list[tuple[int, int]]]:
    mask = torch.zeros(B, H, W, dtype=torch.int64)
    pixels = [(5, 7), (40, 55)]
    for b, (y, x) in enumerate(pixels):
        mask[b, y, x] = 3
    return mask, pixels


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_shapes_and_each_crop_comes_from_its_frame():
    mask, _ = _one_fg_pixel_per_frame()
    t = RandomForegroundBiasedMultiCrop(size=(P, P), samples_per_input=N, probabilistic=True)
    cube_out, mask_out = t(_frames(), mask, _gen())
    assert cube_out.shape == (B * N, P, P, C)
    assert mask_out is not None and mask_out.shape == (B * N, P, P)
    for j in range(B * N):
        assert torch.all(cube_out[j] == j // N)  # crops of one frame sit next to each other


def test_one_sample_per_input_is_the_single_crop():
    mask, _ = _one_fg_pixel_per_frame()
    kw = {"size": (P, P), "fg_percent": 0.5, "probabilistic": True}
    single = RandomForegroundBiasedCrop(**kw)(_frames(), mask, _gen(7))
    multi = RandomForegroundBiasedMultiCrop(samples_per_input=1, **kw)(_frames(), mask, _gen(7))
    assert torch.equal(single[0], multi[0]) and torch.equal(single[1], multi[1])


def test_forced_crops_contain_their_frames_foreground():
    mask, _ = _one_fg_pixel_per_frame()
    t = RandomForegroundBiasedMultiCrop(
        size=(P, P), samples_per_input=N, fg_percent=1.0, probabilistic=True
    )
    _, mask_out = t(_frames(), mask, _gen())
    assert mask_out is not None
    assert all(bool((mask_out[j] == 3).any()) for j in range(B * N))


def test_deterministic_forcing_applies_per_frame():
    mask, _ = _one_fg_pixel_per_frame()
    t = RandomForegroundBiasedMultiCrop(
        size=(P, P), samples_per_input=N, fg_percent=0.5, prob=0.0, probabilistic=False
    )
    _, mask_out = t(_frames(), mask, _gen())
    assert mask_out is not None
    # prob=0 makes the unforced crops centre crops, which miss both pixels; the last half of every
    # frame's crops is forced onto its foreground.
    hits = [bool((mask_out[j] == 3).any()) for j in range(B * N)]
    assert hits == [False, False, True, True] * B


def test_without_a_mask():
    t = RandomForegroundBiasedMultiCrop(size=(P, P), samples_per_input=N, probabilistic=True)
    cube_out, mask_out = t(_frames(), None, _gen())
    assert cube_out.shape == (B * N, P, P, C) and mask_out is None


def test_rejects_bad_samples_per_input():
    with pytest.raises(ValueError, match="samples_per_input"):
        RandomForegroundBiasedMultiCrop(size=(P, P), samples_per_input=0)


def test_registered_and_composable():
    from cuvis_ai_augment.node.compose import AugmentationCompose

    assert TRANSFORM_REGISTRY["RandomForegroundBiasedMultiCrop"] is RandomForegroundBiasedMultiCrop
    node = AugmentationCompose(
        name="aug",
        seed=0,
        extra_transform_modules=["cuvis_ai_unet.transforms"],
        transforms=[
            {
                "type": "RandomForegroundBiasedMultiCrop",
                "size": [P, P],
                "samples_per_input": N,
                "fg_percent": 0.5,
                "probabilistic": True,
            },
            {"type": "RandomHorizontalFlip", "prob": 0.5},
        ],
    )
    mask, _ = _one_fg_pixel_per_frame()
    out = node.forward(cube=_frames(), mask=mask)
    assert out["cube"].shape == (B * N, P, P, C) and out["mask"].shape == (B * N, P, P)
