"""Shared fixtures: tiny synthetic BHWC cubes and integer masks matching the node ports."""

from __future__ import annotations

import pytest
import torch

# tiny-but-real geometry: B=2, H=W=16, C=6 spectral bands, K=2 classes
B, H, W, C, K = 2, 16, 16, 6, 2


@pytest.fixture()
def cube() -> torch.Tensor:
    """Seeded random BHWC float32 cube."""
    g = torch.Generator().manual_seed(0)
    return torch.rand(B, H, W, C, generator=g)


@pytest.fixture()
def mask() -> torch.Tensor:
    """Seeded random [B, H, W] int32 class mask with K classes."""
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, K, (B, H, W), generator=g, dtype=torch.int32)
