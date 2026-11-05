"""The read-once crop the lentils examples configure: cuvis-ai-augment's RandomForegroundBiasedCrop
with ``samples_per_input`` (augment 0.5.0 or later), N crops from every input frame."""

from __future__ import annotations

import inspect

import pytest
import torch

pytest.importorskip("cuvis_ai_augment")

from cuvis_ai_augment.node.compose import AugmentationCompose  # noqa: E402
from cuvis_ai_augment.transforms.spatial import RandomForegroundBiasedCrop  # noqa: E402

if "samples_per_input" not in inspect.signature(RandomForegroundBiasedCrop.__init__).parameters:
    pytest.skip("cuvis-ai-augment before 0.5.0 has no samples_per_input", allow_module_level=True)

B, H, W, C, N, P = 2, 48, 64, 3, 4, 16


def test_examples_crop_config_cuts_n_crops_from_every_frame():
    # Frame b is filled with the value b, so a crop shows which frame it came from.
    frames = torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1).expand(B, H, W, C).clone()
    mask = torch.zeros(B, H, W, dtype=torch.int64)
    mask[0, 5, 7] = 3
    mask[1, 40, 55] = 3
    node = AugmentationCompose(
        name="aug",
        seed=0,
        extra_transform_modules=["cuvis_ai_unet.transforms"],
        transforms=[
            {
                "type": "RandomForegroundBiasedCrop",
                "size": [P, P],
                "samples_per_input": N,
                "fg_percent": 0.5,
                "probabilistic": True,
            },
            {"type": "RandomHorizontalFlip", "prob": 0.5},
        ],
    )
    out = node.forward(cube=frames, mask=mask)
    assert out["cube"].shape == (B * N, P, P, C) and out["mask"].shape == (B * N, P, P)
    for j in range(B * N):
        assert torch.all(out["cube"][j] == j // N)  # the crops of one frame sit next to each other
