"""Shared engine for the lentils segmentation examples.

Builds the pipeline in code (the cuvis-ai plugin-family convention), trains it in
two phases (normalizer statistics, then gradient training of DynUNet), and
persists/restores the canonical artifact written by ``pipeline.save_to_file``
(YAML + co-located ``.pt``). The thin CLIs (`train.py`, `evaluate.py`,
`profile_pipeline.py`) are argparse fronts over these functions.

Graph (wired here, serialized into every artifact)::

    DataSource ──cube──▶ Norm ──normalized──▶ Augment ──cube──▶ DynUNet ──logits──▶ DiceLoss
        └───────mask────────────────────────▶   └──────mask───────────────────────▶ CrossEntropyLoss

Norm sits UPSTREAM of Augment so its statistics describe full frames both when
they are fitted (statistical init) and when they are applied — structural, not a
side-effect of Augment's train-only stage gate. Augment crops already-normalized
data at TRAIN and passes through at inference; for running-stats z-score the two
orders are numerically equivalent (an elementwise affine commutes with cropping).
"""

from __future__ import annotations

import inspect
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

# Full lentils frames are ~263 MB (1000x1080x61 f32); the default file-descriptor
# sharing strategy pushes these through /dev/shm and exhausts it with several
# workers. file_system sharing avoids the shm/fd ceiling.
torch.multiprocessing.set_sharing_strategy("file_system")

from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.training.config import PipelineMetadata
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.training.config import TrainingConfig
from cuvis_ai_schemas.training.optimizer import OptimizerConfig

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
UNET_MANIFEST = REPO_ROOT / "plugins.yaml"
AUGMENT_MANIFEST = HERE / "augment.yaml"

# Published reference numbers for the champion configuration (2D @ 128, AdaCLIP
# split, 20 epochs): what a reproduction run is compared against.
CHAMPION = {"2d_128": {"fg_iou": 0.7905, "fg_dice": 0.8787, "image_auroc": 0.998}}


def register_plugins(
    unet_manifest: str | Path = UNET_MANIFEST,
    augment_manifest: str | Path = AUGMENT_MANIFEST,
) -> NodeRegistry:
    """Register the unet + augment plugin manifests and return the registry.

    Needed both to build (imports resolve) and to restore artifacts
    (``load_pipeline`` resolves node classes through this registry).
    """
    registry = NodeRegistry()
    registry.register_plugin(str(unet_manifest))
    registry.register_plugin(str(augment_manifest))
    return registry


def _supports_running_stats(cls: type) -> bool:
    return "use_running_stats" in inspect.signature(cls.__init__).parameters


def build_graph(
    *,
    mode: str = "2d",
    in_channels: int = 61,
    num_classes: int = 2,
    features: tuple[int, ...] = (32, 64, 128, 256, 512),
    patch: int = 128,
    tile_overlap: float = 0.5,
    tile_gaussian: bool = True,
    tile_batch: int = 16,
    normalizer: str = "zscore",
    max_init_frames: int = 100,
    fg_percent: float = 0.5,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    dataset_crop: bool = False,
    augment_seed: int = 0,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
    with_metrics: bool = False,
    tb_dir: str | None = None,
    tb_run_name: str | None = None,
    name: str = "lentils_unet",
) -> CuvisPipeline:
    """Build the segmentation pipeline in code and return it (untrained).

    ``normalizer`` is one of ``zscore`` (dataset-level running stats, scalar),
    ``zscore-perband`` (running stats per band), or ``persample`` (classic
    per-sample z-score). The running-stats modes need the ZScoreNormalizer
    running-stats feature (pending upstream in cuvis-ai PR #53) and raise a
    clear error on an installation that predates it.
    """
    from cuvis_ai.node.data import CU3SDataNode
    from cuvis_ai.node.normalization import ZScoreNormalizer
    from cuvis_ai_augment.node.compose import AugmentationCompose

    from cuvis_ai_unet.node.dynunet import DynUNet
    from cuvis_ai_unet.node.losses import CrossEntropyLoss, DiceLoss

    if normalizer in ("zscore", "zscore-perband"):
        if not _supports_running_stats(ZScoreNormalizer):
            raise RuntimeError(
                f"normalizer={normalizer!r} needs ZScoreNormalizer running-stats "
                "support (pending in cuvis-ai PR #53). Install cuvis-ai from that "
                "branch, or use --normalizer persample."
            )
        norm_kwargs: dict[str, Any] = {
            "use_running_stats": True,
            "max_init_frames": max_init_frames,
        }
        if normalizer == "zscore-perband":
            norm_kwargs.update(per_band=True, num_channels=in_channels)
    elif normalizer == "persample":
        norm_kwargs = {}
    else:
        raise ValueError(f"unknown normalizer {normalizer!r}")

    source = CU3SDataNode(name="DataSource")
    norm = ZScoreNormalizer(name="Norm", **norm_kwargs)
    # The foreground-biased crop is either a graph transform here (default) OR done in the
    # dataloader (dataset_crop=True -> npz_multi crop_size); doing both would double-crop, so
    # when the dataloader crops we keep only the flips (which run on the already-cropped patch).
    crop_transforms = (
        []
        if dataset_crop
        else [
            {
                "type": "RandomForegroundBiasedCrop",
                "size": [patch, patch],
                "fg_percent": fg_percent,
                "probabilistic": True,
            }
        ]
    )
    augment = AugmentationCompose(
        name="Augment",
        seed=augment_seed,
        extra_transform_modules=["cuvis_ai_unet.transforms"],
        transforms=[
            *crop_transforms,
            {"type": "RandomHorizontalFlip", "prob": hflip_prob},
            {"type": "RandomVerticalFlip", "prob": vflip_prob},
        ],
    )
    net = DynUNet(
        name="DynUNet",
        mode=mode,
        in_channels=in_channels,
        num_classes=num_classes,
        features=list(features),
        tile_size=patch,
        tile_overlap=tile_overlap,
        tile_gaussian=tile_gaussian,
        tile_batch=tile_batch,
    )
    dice = DiceLoss(name="DiceLoss", weight=dice_weight)
    ce = CrossEntropyLoss(name="CrossEntropyLoss", weight=ce_weight)

    edges = [
        (source.outputs.cube, norm.inputs.data),
        (norm.outputs.normalized, augment.inputs.cube),
        (source.outputs.mask, augment.inputs.mask),
        (augment.outputs.cube, net.inputs.data),
        (net.outputs.logits, dice.inputs.logits),
        (augment.outputs.mask, dice.inputs.targets),
        (net.outputs.logits, ce.inputs.logits),
        (augment.outputs.mask, ce.inputs.targets),
    ]
    if with_metrics:
        # SegMetrics computes fg-IoU/Dice at VAL/TEST (no builtin seg-metric node exists);
        # TensorBoardMonitorNode is the ecosystem sink that streams them to TensorBoard.
        # Both are wired INTO the graph here so validation actually logs metric curves;
        # train(...) passes metric_nodes=[SegMetrics], monitors=[TensorBoard] to the trainer.
        from cuvis_ai.node.monitor import TensorBoardMonitorNode

        from cuvis_ai_unet.node.seg_metrics import SegMetrics

        seg = SegMetrics(name="SegMetrics")
        tb = TensorBoardMonitorNode(
            name="TensorBoard",
            output_dir=tb_dir or "runs/tensorboard",
            run_name=tb_run_name or name,
        )
        edges += [
            (net.outputs.logits, seg.inputs.logits),
            (augment.outputs.mask, seg.inputs.targets),
            (seg.outputs.metrics, tb.inputs.metrics),
        ]

    pipe = CuvisPipeline(name)
    pipe.connect(*edges)
    return pipe


def make_datamodule(
    universe_csv: str | Path,
    splits_json: str | Path,
    *,
    batch_size: int = 8,
    num_workers: int = 4,
    samples_per_frame: int | None = None,
    crop_size: tuple[int, int] | None = None,
    crop_fg_percent: float = 0.5,
):
    """Build the multi-npz datamodule from a universe.csv + splits.json.

    Both artifacts come from ``gen_splits.py`` (from existing NPZ, SDK-free) or
    ``convert_split_manifest`` (from cu3s + COCO): the universe.csv maps each frame's
    ``(source, index)`` identity to its ``.npz``; the splits.json is a
    ``DataSplitConfig`` of per-source selectors. ``samples_per_frame`` (N fg-biased
    crops per frame per epoch) is applied by the base datamodule to the *train*
    split only — val/test stay one crop per frame.

    ``crop_size`` (needs a dataloader release with npz_multi crop support; no released
    version has it yet, so passing it raises with instructions) makes the dataset return
    a foreground-biased ``(h, w)`` patch per train sample instead of the whole frame —
    the I/O-cheap alternative to the in-graph crop node (build the graph with
    ``dataset_crop=True`` so the crop is not applied twice). Val/test stay full-frame.

    CAVEAT (``dataset_crop``): the crop then sits *upstream* of the ``Norm`` node, so the
    Phase-1 ZScoreNormalizer fits its running stats on the cropped train patches, not on
    full frames (eval frames are normalized with those crop-fit stats). With global z-score
    this only shifts the fitted mean/std and is usually fine, but it makes the run not
    strictly comparable to an in-graph-crop run whose Norm was fit on full frames.
    """
    from cuvis_ai_core.data.splits_io import load_splits
    from cuvis_ai_dataloader.data.datamodule_npz_multi import MultiNpzDataModule

    kwargs: dict[str, Any] = {
        "universe_csv": str(universe_csv),
        "splits": load_splits(str(splits_json)),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if samples_per_frame is not None:
        kwargs["samples_per_frame"] = samples_per_frame
    if crop_size is not None:
        if "crop_size" not in inspect.signature(MultiNpzDataModule.__init__).parameters:
            raise RuntimeError(
                "--dataset-crop needs a cuvis-ai-dataloader with npz_multi crop_size "
                "support (not yet released); use the in-graph crop (the default, "
                "without --dataset-crop) instead."
            )
        kwargs["crop_size"] = tuple(crop_size)
        kwargs["crop_fg_percent"] = crop_fg_percent
    return MultiNpzDataModule(**kwargs)


class EpochLog(Callback):
    """Print a timestamped, flushed line per training epoch (loss + seconds/epoch)."""

    def on_train_epoch_start(self, trainer, pl_module):  # noqa: ANN001, D102
        self._t0 = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: ANN001, D102
        dt = time.monotonic() - getattr(self, "_t0", time.monotonic())
        metrics = {
            k: (float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else v)
            for k, v in trainer.callback_metrics.items()
        }
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] epoch {trainer.current_epoch:>3} | {dt:6.1f}s | {metrics}", flush=True)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def train(
    pipeline: CuvisPipeline,
    datamodule,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    accelerator: str = "auto",
    val_every: int = 0,
    save_best_val: bool = False,
    gradient_clip_val: float | None = None,
    out_dir: str | Path,
    run_meta: dict[str, Any] | None = None,
) -> Path:
    """Two-phase training; returns the saved artifact's YAML path.

    Phase 1 fits statistical nodes (normalizer stats, on full frames — Augment
    passes through outside TRAIN). Phase 2 gradient-trains DynUNet against the
    Dice+CE losses. ``val_every=0`` disables in-training validation (full-frame
    tiled validation is expensive, and the direct 3D forward can exhaust GPU
    memory); pass a positive N to validate every N epochs and once at the end.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes = {n.name: n for n in pipeline.nodes}

    t0 = time.monotonic()
    StatisticalTrainer(pipeline=pipeline, datamodule=datamodule).fit()
    for n in pipeline.nodes:
        if getattr(n, "requires_initial_fit", False):
            assert getattr(n, "_statistically_initialized", False), f"{n.name} not initialized"
    t_stat = time.monotonic() - t0

    pipeline.unfreeze_nodes_by_name(["DynUNet"])
    # If build_graph(with_metrics=True) added them, feed the metric node + TensorBoard sink
    # to the trainer: metric_nodes -> logged as SegMetrics/fg_iou etc. at val/test; monitors
    # -> the TensorBoard writer that also receives train/loss + val/loss aggregates.
    metric_nodes = [nodes["SegMetrics"]] if "SegMetrics" in nodes else None
    monitors = [nodes["TensorBoard"]] if "TensorBoard" in nodes else None
    if save_best_val and val_every <= 0:
        raise ValueError(
            "save_best_val requires val_every > 0 (validation must run to rank epochs)."
        )
    # best-val checkpointing: a ModelCheckpoint on val_loss, kept IN the callbacks list so the
    # trainer uses it (a passed callbacks list takes precedence over training_config.callbacks;
    # relying on the config alone would silently fall back to Lightning's default last-epoch
    # checkpoint). A divergence that sends val_loss to NaN never beats the best, so the reloaded
    # + saved model stays the best pre-divergence epoch.
    callbacks: list[Callback] = [EpochLog()]
    if save_best_val:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(out / "checkpoints"),
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                verbose=True,
            )
        )
    trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[nodes["DiceLoss"], nodes["CrossEntropyLoss"]],
        metric_nodes=metric_nodes,
        monitors=monitors,
        training_config=TrainingConfig(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=1,
            enable_progress_bar=False,
            log_every_n_steps=1,
            enable_checkpointing=save_best_val,
            gradient_clip_val=gradient_clip_val,
            # val_every=0: point past the last epoch so Lightning never validates.
            check_val_every_n_epoch=val_every if val_every > 0 else epochs + 1,
            optimizer=OptimizerConfig(name="adam", lr=lr),
        ),
        callbacks=callbacks,
    )
    t0 = time.monotonic()
    trainer.fit()
    t_grad = time.monotonic() - t0
    # With save_best_val, validate on the best checkpoint — this reloads the best-val_loss weights
    # into the pipeline so the saved artifact is the best epoch, not the last.
    if save_best_val:
        val_metrics = trainer.validate(ckpt_path="best")
    elif val_every > 0:
        val_metrics = trainer.validate()
    else:
        val_metrics = None

    artifact = out / "pipeline.yaml"
    pipeline.save_to_file(
        str(artifact),
        save_weights=True,
        metadata=PipelineMetadata(
            name=pipeline.name,
            description="Two-phase-trained lentils segmentation pipeline"
            + (" (best val_loss checkpoint)" if save_best_val else ""),
        ),
    )
    run = {
        "artifact": str(artifact),
        "git_commit": _git_commit(),
        "phase1_seconds": round(t_stat, 1),
        "phase2_seconds": round(t_grad, 1),
        "epochs": epochs,
        "lr": lr,
        "val_metrics": val_metrics,
        **(run_meta or {}),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print(f"[train] artifact saved: {artifact} (+ .pt)", flush=True)
    return artifact


def _mark_initialized(pipeline: CuvisPipeline) -> None:
    # `_statistically_initialized` is a plain attribute, not part of state_dict:
    # restoring weights restores the stats buffers but not the flag. A restored
    # artifact is by construction initialized.
    for n in pipeline.nodes:
        if getattr(n, "requires_initial_fit", False):
            n._statistically_initialized = True


def load_artifact(
    config_yaml: str | Path,
    weights: str | Path | None = None,
    *,
    device: str | torch.device | None = None,
    registry: NodeRegistry | None = None,
) -> CuvisPipeline:
    """Restore a saved pipeline artifact (YAML + name-keyed .pt) for inference."""
    registry = registry or register_plugins()
    weights = Path(weights) if weights else Path(config_yaml).with_suffix(".pt")
    device = str(device) if device else ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = CuvisPipeline.load_pipeline(
        str(config_yaml), weights_path=str(weights), device=device, node_registry=registry
    )
    _mark_initialized(pipe)
    for layer in pipe.torch_layers:
        layer.eval()
    return pipe


def load_raw_ckpt(
    config_yaml: str | Path,
    raw_ckpt: str | Path,
    *,
    device: str | torch.device | None = None,
    registry: NodeRegistry | None = None,
    allow_partial: bool = False,
) -> CuvisPipeline:
    """Restore a legacy raw ``torch_layers.state_dict()`` checkpoint (INDEX-keyed).

    Index-keyed checkpoints bind weights to the position of each node in the
    config's ``nodes:`` list; a reordered config silently misassigns them under
    ``strict=False``. Any missing/unexpected key is therefore a hard error
    unless ``allow_partial`` is set.
    """
    from cuvis_ai_core.pipeline.factory import PipelineBuilder

    registry = registry or register_plugins()
    pipe = PipelineBuilder(node_registry=registry).build_from_config(str(config_yaml))
    sd = torch.load(str(raw_ckpt), map_location="cpu")
    missing, unexpected = pipe.torch_layers.load_state_dict(sd, strict=False)
    print(
        f"[load] {Path(config_yaml).name} <- {Path(raw_ckpt).name}: "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if (missing or unexpected) and not allow_partial:
        raise RuntimeError(
            f"checkpoint/config mismatch (missing={list(missing)[:4]}, "
            f"unexpected={list(unexpected)[:4]}): index-keyed checkpoints require "
            "the exact nodes-list order they were saved under (--allow-partial to override)"
        )
    _mark_initialized(pipe)
    device = str(device) if device else ("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    for layer in pipe.torch_layers:
        layer.eval()
    return pipe


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic with average ranks (tie-safe)."""
    labels = labels.astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def split_frames(
    universe_csv: str | Path, splits_json: str | Path, split: str = "test"
) -> list[str]:
    """Unique npz paths of a split, resolved from the universe.csv + splits.json.

    Uses the datamodule's own selector resolution so the frames evaluated are
    exactly those the trainer would load (minus train ``samples_per_frame``
    multiplicity, which never touches val/test).
    """
    dm = make_datamodule(universe_csv, splits_json, batch_size=1, num_workers=0)
    dm.setup(stage=None)
    ds = {"train": dm.train_ds, "val": dm.val_ds, "test": dm.test_ds}.get(split)
    return [rec["materialized_path"] for rec in ds.rows] if ds is not None else []


def evaluate(
    pipeline: CuvisPipeline,
    universe_csv: str | Path,
    splits_json: str | Path,
    *,
    split: str = "test",
    tile_overlap: float | None = None,
    tile_batch: int | None = None,
    max_frames: int = 0,
) -> dict[str, Any]:
    """Frame-by-frame tiled evaluation: pixel fg-IoU/Dice + image-level AUROC.

    Frames are read directly (no DataLoader) and pushed through Norm + DynUNet
    at inference stage — the exact procedure behind the published champion
    numbers. Object frames contribute to the pixel metrics; all frames
    contribute to the image-level AUROCs (max fg-probability and predicted
    fg-area scores vs the frame-has-objects label).
    """
    nodes = {n.name: n for n in pipeline.nodes}
    norm, net = nodes["Norm"], nodes["DynUNet"]
    if tile_overlap is not None:
        net.tile_overlap = float(tile_overlap)
        if float(tile_overlap) == 0.0:
            net.tile_gaussian = False  # no-op at overlap 0 (each pixel in exactly one tile)
    if tile_batch is not None:
        net.tile_batch = int(tile_batch)
    device = next(iter(net.parameters())).device
    infer = Context(stage=ExecutionStage.INFERENCE, batch_idx=0, global_step=0)

    frames = split_frames(universe_csv, splits_json, split)
    if max_frames:
        frames = frames[:max_frames]

    ious, dices = [], []
    maxprob, fgarea, labels = [], [], []
    with torch.no_grad():
        for p in frames:
            z = np.load(p)
            mask = torch.from_numpy(np.asarray(z["mask"])).to(device)
            has_fg = int((mask >= 1).sum()) > 0
            cube = torch.from_numpy(z["cube"].astype("float32")).unsqueeze(0).to(device)
            normed = norm.forward(data=cube)["normalized"]
            logits = net.forward(normed, context=infer)["logits"][0]  # [H,W,K]
            prob = torch.softmax(logits.float(), dim=-1)[..., 1]  # fg prob [H,W]
            pred = logits.argmax(-1) >= 1
            maxprob.append(float(prob.max()))
            fgarea.append(int(pred.sum()))
            labels.append(1 if has_fg else 0)
            if has_fg:
                tgt = mask >= 1
                inter = float((pred & tgt).sum())
                psum, gsum = float(pred.sum()), float(tgt.sum())
                union = psum + gsum - inter
                ious.append(inter / union if union else float("nan"))
                dices.append(2 * inter / (psum + gsum) if (psum + gsum) else float("nan"))

    lab = np.array(labels)
    return {
        "split": split,
        "frames": len(frames),
        "object_frames": int(lab.sum()),
        "normal_frames": int((lab == 0).sum()),
        "fg_iou": float(np.nanmean(ious)) if ious else float("nan"),
        "fg_dice": float(np.nanmean(dices)) if dices else float("nan"),
        "image_auroc_maxprob": auroc(np.array(maxprob), lab),
        "image_auroc_area": auroc(np.array(fgarea, dtype=np.float64), lab),
    }


def profile(
    pipeline: CuvisPipeline,
    universe_csv: str | Path,
    splits_json: str | Path,
    *,
    split: str = "test",
    frames: int = 8,
    skip: int = 2,
    overlaps: tuple[float, ...] = (0.0, 0.25, 0.5),
    tile_batches: tuple[int, ...] = (1, 16),
) -> None:
    """Per-node inference timing via cuvis-ai's built-in pipeline profiler.

    The full graph runs at INFERENCE stage: the loss nodes are stage-filtered
    out, Augment is an identity passthrough (visible as ~0 ms in the table),
    and the file-reading DataSource is neutralized because cubes are injected
    via the batch. Frames are pre-loaded to the device, so node timings are
    pure compute; the one-off disk-read time is printed separately.
    """
    nodes = {n.name: n for n in pipeline.nodes}
    net = nodes["DynUNet"]
    for n in pipeline.nodes:
        if n.name == "DataSource":
            n.execution_stages = set()
    device = next(iter(net.parameters())).device

    paths = split_frames(universe_csv, splits_json, split)[:frames]
    batches, load_ms = [], []
    for p in paths:
        t0 = time.perf_counter()
        z = np.load(p)
        cube = torch.from_numpy(z["cube"].astype("float32")).unsqueeze(0)
        load_ms.append((time.perf_counter() - t0) * 1e3)
        batches.append({"cube": cube.to(device)})
    print(
        f"[prof] {len(batches)} frames on {device} | frame {tuple(batches[0]['cube'].shape)} "
        f"| mean disk-read={np.mean(load_ms):.0f} ms/frame (EXCLUDED from node timings)",
        flush=True,
    )

    single = len(overlaps) == 1 and len(tile_batches) == 1
    print(
        f"\n{'overlap':>7} {'tile_batch':>10} {'DynUNet ms':>11} {'Norm ms':>8} "
        f"{'fps':>6} {'peak GB':>8}",
        flush=True,
    )
    for ov in overlaps:
        net.tile_overlap = ov
        for tb in tile_batches:
            net.tile_batch = tb
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            pipeline.set_profiling(
                enabled=True, synchronize_cuda=True, reset=True, skip_first_n=skip
            )
            with torch.no_grad():
                for _ in range(skip):  # warmup samples (discarded by skip_first_n)
                    pipeline.forward(batch=batches[0], stage=ExecutionStage.INFERENCE)
                for b in batches:
                    pipeline.forward(batch=b, stage=ExecutionStage.INFERENCE)
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            stats = {
                s.node_name: s for s in pipeline.get_profiling_summary(ExecutionStage.INFERENCE)
            }
            dyn, nrm = stats["DynUNet"].mean_ms, stats["Norm"].mean_ms
            print(
                f"{ov:>7.2f} {tb:>10d} {dyn:>11.1f} {nrm:>8.2f} "
                f"{1e3 / (dyn + nrm):>6.2f} {peak:>8.2f}",
                flush=True,
            )
            if single:  # full per-node table when profiling a single config
                print(pipeline.format_profiling_summary(total_frames=len(batches)), flush=True)
    print("[prof] DONE", flush=True)
