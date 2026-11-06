"""The lentils example engine's frame-by-frame evaluation, on fake nodes and frames (no cu3s data)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ENGINE = Path(__file__).resolve().parents[1] / "examples" / "lentils" / "_engine.py"


@pytest.fixture(scope="module")
def eng():
    sharing = torch.multiprocessing.get_sharing_strategy()
    spec = importlib.util.spec_from_file_location("lentils_engine", ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    torch.multiprocessing.set_sharing_strategy(sharing)  # the engine switches it on import


class _Node:
    def __init__(self, name):
        self.name = name


class _Standardize(_Node):
    INPUT_SPECS = {"cube": None}

    def forward(self, cube):
        return {"cube": cube}


class _ZScoreNormalizer(_Node):
    """The July checkpoints' normalizer (cuvis-ai 0.20.x): ``data`` in, ``normalized`` out."""

    INPUT_SPECS = {"data": None}

    def forward(self, data):
        return {"normalized": data}


class _Net(_Node):
    """Two-class logits that predict foreground wherever band 0 of the cube is positive."""

    def __init__(self):
        super().__init__("DynUNet")
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self.weight])

    def forward(self, cube, context):
        fg = cube[..., :1]
        return {"logits": torch.cat([-fg, fg], dim=-1)}


class _Pipeline:
    def __init__(self, norm):
        self.nodes = [norm, _Net()]


def _frames():
    cube = torch.full((4, 4, 3), -1.0)
    cube[:2, :2, 0] = 1.0
    mask = torch.zeros(4, 4, dtype=torch.int64)
    mask[:2, :2] = 3  # a category id; any id >= 1 is foreground
    return [{"cube": cube, "mask": mask}, {"cube": torch.full((4, 4, 3), -1.0), "mask": mask * 0}]


@pytest.mark.parametrize("norm_cls", [_Standardize, _ZScoreNormalizer])
def test_evaluate_runs_either_normalizer(eng, monkeypatch, norm_cls):
    monkeypatch.setattr(eng, "split_dataset", lambda *args, **kwargs: _frames())
    metrics = eng.evaluate(_Pipeline(norm_cls("Norm")), "universe.csv", "splits.json")
    assert metrics["frames"] == 2 and metrics["object_frames"] == 1
    assert metrics["fg_iou"] == pytest.approx(1.0)
    assert metrics["fg_dice"] == pytest.approx(1.0)
