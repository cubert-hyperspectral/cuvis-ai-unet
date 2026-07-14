"""Foreground-biased crop transform for cuvis-ai-augment's AugmentationCompose.

Vendored here until the same class ships in cuvis-ai-augment
(https://github.com/cubert-hyperspectral/cuvis-ai-augment/pull/13) — load it
via the compose node's ``extra_transform_modules: [cuvis_ai_unet.transforms]``
and delete this module once that PR lands in a release.

Semantics mirror nnU-Net's verified foreground oversampling
(``oversample_foreground_percent``): a subset of every batch is forced to
contain foreground by centering the crop window on a randomly chosen pixel of
a randomly chosen foreground class; remaining samples crop exactly like
``RandomSpatialCrop``. The forced subset is either the deterministic *last*
``round(B * fg_percent)`` samples (nnU-Net's default rule) or, with
``probabilistic=True``, an independent per-sample Bernoulli draw — required
for ``batch_size 1``, where the deterministic rule rounds to zero forced
samples. ``fg_labels`` restricts which mask labels count as foreground
(default: anything ``> 0``). Samples without eligible foreground (or with no
mask connected) fall back to the uniform behaviour.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_augment.transforms.base import TRANSFORM_REGISTRY, Transform
from torch import Tensor


class RandomForegroundBiasedCrop(Transform):
    """Crop to ``(H_out, W_out)`` with nnU-Net-style foreground oversampling.

    Parameters
    ----------
    size : tuple[int, int]
        Output ``(H_out, W_out)``. Must satisfy ``H_out <= H`` and ``W_out <= W``.
    fg_percent : float
        Fraction of each batch forced to contain foreground. Deterministic
        mode forces the **last** ``round(B * fg_percent)`` samples (exactly
        nnU-Net's rule); probabilistic mode draws Bernoulli(``fg_percent``)
        per sample.
    prob : float
        For the non-forced samples: probability of a random offset; otherwise
        a deterministic centre crop (identical to ``RandomSpatialCrop``).
    fg_labels : list[int], optional
        Mask labels that count as foreground. ``None`` (default) means every
        label ``> 0``; pass an explicit list when some labels are semantically
        background (e.g. "normal object" classes).
    probabilistic : bool
        ``False`` (default): deterministic last-k selection, RNG stream
        identical to ``RandomSpatialCrop`` for foreground-free batches.
        ``True``: per-sample Bernoulli — use for small/unit batch sizes
        (``B=1`` deterministic rounds to zero forced samples) and as a
        stochastic augmentation (e.g. ``fg_percent=0.5`` for a 50/50 mix of
        object-centered and uniform crops).
    """

    def __init__(
        self,
        size: tuple[int, int] | list[int],
        fg_percent: float = 0.33,
        prob: float = 1.0,
        fg_labels: list[int] | None = None,
        probabilistic: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        size_t = tuple(int(x) for x in size)
        if len(size_t) != 2 or any(s <= 0 for s in size_t):
            raise ValueError(f"size must be a pair of positive ints, got {size!r}")
        if not 0.0 <= float(fg_percent) <= 1.0:
            raise ValueError(f"fg_percent must be in [0, 1], got {fg_percent!r}")
        self.size: tuple[int, int] = (size_t[0], size_t[1])
        self.fg_percent = float(fg_percent)
        self.fg_labels = None if fg_labels is None else [int(x) for x in fg_labels]
        if self.fg_labels is not None and len(self.fg_labels) == 0:
            raise ValueError("fg_labels must be None or a non-empty list of labels")
        self.probabilistic = bool(probabilistic)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
        wavelengths: list[float] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        del wavelengths  # wavelength-agnostic
        self._validate_shapes(cube, mask)
        B, H, W, C = cube.shape
        H_out, W_out = self.size

        if H_out > H or W_out > W:
            raise ValueError(
                f"Crop size ({H_out}, {W_out}) exceeds cube spatial dims (H={H}, W={W})."
            )

        device = cube.device
        max_top = H - H_out
        max_left = W - W_out

        # Base offsets for every sample — identical to RandomSpatialCrop.
        random_apply = self._draw_apply_mask(B, rng, device)
        if max_top > 0:
            rand_top = torch.randint(low=0, high=max_top + 1, size=(B,), generator=rng).to(
                device=device
            )
        else:
            rand_top = torch.zeros(B, dtype=torch.long, device=device)
        if max_left > 0:
            rand_left = torch.randint(low=0, high=max_left + 1, size=(B,), generator=rng).to(
                device=device
            )
        else:
            rand_left = torch.zeros(B, dtype=torch.long, device=device)
        centre_top = torch.full((B,), max_top // 2, dtype=torch.long, device=device)
        centre_left = torch.full((B,), max_left // 2, dtype=torch.long, device=device)
        top = torch.where(random_apply, rand_top, centre_top)
        left = torch.where(random_apply, rand_left, centre_left)

        # Which samples are forced to contain foreground. Drawn AFTER the base
        # offsets so deterministic mode keeps the RandomSpatialCrop RNG stream.
        if self.probabilistic:
            forced = torch.rand(B, generator=rng) < self.fg_percent
        else:
            n_forced = int(round(B * self.fg_percent))
            forced = torch.zeros(B, dtype=torch.bool)
            if n_forced:
                forced[B - n_forced :] = True

        cube_out = torch.empty((B, H_out, W_out, C), dtype=cube.dtype, device=device)
        mask_out = (
            torch.empty((B, H_out, W_out), dtype=mask.dtype, device=device)
            if mask is not None
            else None
        )
        for b in range(B):
            top_o = int(top[b].item())
            left_o = int(left[b].item())
            if mask is not None and bool(forced[b]):
                fg = self._fg_center(mask[b], rng)
                if fg is not None:
                    cy, cx = fg
                    top_o = min(max(cy - H_out // 2, 0), max_top)
                    left_o = min(max(cx - W_out // 2, 0), max_left)
            cube_out[b] = cube[b, top_o : top_o + H_out, left_o : left_o + W_out, :]
            if mask is not None and mask_out is not None:
                mask_out[b] = mask[b, top_o : top_o + H_out, left_o : left_o + W_out]
        return cube_out, mask_out

    def _fg_center(self, mask_b: Tensor, rng: torch.Generator) -> tuple[int, int] | None:
        """Random pixel of a random eligible foreground class (None if empty).

        nnU-Net's selection: pick an eligible class uniformly, then a pixel of
        that class uniformly — so rare classes are sampled as often as common
        ones. Eligible classes are ``> 0`` by default or the explicit
        ``fg_labels`` set. Works for bool and integer masks. Draws from ``rng``
        only when eligible foreground exists, keeping the RNG stream identical
        to ``RandomSpatialCrop`` for foreground-free batches.
        """
        labels = torch.unique(mask_b)
        if self.fg_labels is None:
            labels = labels[labels > 0]
        else:
            allowed = torch.tensor(self.fg_labels, dtype=labels.dtype, device=labels.device)
            labels = labels[torch.isin(labels, allowed)]
        if labels.numel() == 0:
            return None
        cls = labels[int(torch.randint(labels.numel(), (1,), generator=rng).item())]
        coords = (mask_b == cls).nonzero(as_tuple=False)  # [N, 2] (y, x)
        j = int(torch.randint(coords.shape[0], (1,), generator=rng).item())
        return int(coords[j, 0].item()), int(coords[j, 1].item())


# Guarded registration: augment's @register raises on duplicates, so skip when a
# future cuvis-ai-augment release already ships this transform under the same name.
if "RandomForegroundBiasedCrop" not in TRANSFORM_REGISTRY:
    TRANSFORM_REGISTRY["RandomForegroundBiasedCrop"] = RandomForegroundBiasedCrop
