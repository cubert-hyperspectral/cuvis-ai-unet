"""Evaluate a raw index-keyed diagnostic checkpoint on a splits CSV (GPU, tiled):
  - pixel-level foreground IoU + Dice (on object frames), and
  - image-level AUROC (object vs normal frame) from image scores.
No DataLoader (frames read directly) so it can run alongside other jobs.

This is the metric oracle behind the published champion numbers; `evaluate.py`
in ../lentils is the artifact-based front door with the same math. Raw
checkpoints are INDEX-keyed (torch_layers position): any missing/unexpected key
is a hard error unless --allow-partial, because a reordered config silently
misassigns weights under strict=False.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        default=os.path.join(HERE, "lentils_unet_npz_aug_adaclip2d128.yaml"),
        help="pipeline config YAML the checkpoint was saved under",
    )
    ap.add_argument("--ckpt", required=True, help="raw torch_layers.state_dict() .pt")
    ap.add_argument("--splits-csv", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--overlap", type=float, default=None, help="override tile_overlap")
    ap.add_argument("--tile-batch", type=int, default=None, help="override tile_batch")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="tolerate missing/unexpected checkpoint keys (DANGEROUS: index-keyed)",
    )
    ap.add_argument("--unet-manifest", default=os.path.join(REPO_ROOT, "plugins.yaml"))
    ap.add_argument(
        "--augment-manifest", default=os.path.join(REPO_ROOT, "examples", "lentils", "augment.yaml")
    )
    args = ap.parse_args()

    reg = NodeRegistry()
    reg.register_plugin(args.unet_manifest)
    reg.register_plugin(args.augment_manifest)
    pipe = PipelineBuilder(node_registry=reg).build_from_config(args.config)
    nodes = {n.name: n for n in pipe.nodes}

    sd = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = pipe.torch_layers.load_state_dict(sd, strict=False)
    print(
        f"[eval] {os.path.basename(args.config)} <- {os.path.basename(args.ckpt)}: missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if (missing or unexpected) and not args.allow_partial:
        raise SystemExit(
            f"[eval] FATAL: checkpoint/config mismatch (missing={list(missing)[:4]}, "
            f"unexpected={list(unexpected)[:4]}) — index-keyed checkpoints require the exact "
            "nodes-list order they were saved under (--allow-partial to override)"
        )

    norm, net = nodes["Norm"], nodes["DynUNet"]
    if args.overlap is not None:
        net.tile_overlap = float(args.overlap)
        if float(args.overlap) == 0.0:
            net.tile_gaussian = False  # no-op at overlap 0 (each pixel in exactly one tile)
        print(
            f"[eval] tile_overlap override -> {net.tile_overlap} (gaussian={net.tile_gaussian})",
            flush=True,
        )
    if args.tile_batch is not None:
        net.tile_batch = int(args.tile_batch)
        print(f"[eval] tile_batch override -> {net.tile_batch}", flush=True)
    norm._statistically_initialized = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    for n in pipe.torch_layers:
        n.eval()
    infer = Context(stage=ExecutionStage.INFERENCE, batch_idx=0, global_step=0)

    rows = list(csv.DictReader(open(args.splits_csv)))
    seen: set[str] = set()
    test = [
        r["npz_path"]
        for r in rows
        if r["split"] == args.split and not (r["npz_path"] in seen or seen.add(r["npz_path"]))
    ]

    ious, dices = [], []
    maxprob, fgarea, labels = [], [], []
    with torch.no_grad():
        for p in test:
            z = np.load(p)
            mask = torch.from_numpy(np.asarray(z["mask"])).to(device)
            has_fg = int((mask >= 1).sum()) > 0
            cube = torch.from_numpy(z["cube"].astype("float32")).unsqueeze(0).to(device)
            normed = norm.forward(data=cube)["normalized"]
            logits = net.forward(normed, context=infer)["logits"][0]  # [H,W,K]
            prob = torch.softmax(logits.float(), dim=-1)[..., 1]  # fg prob [H,W]
            pred = logits.argmax(-1) >= 1
            # image-level scores + label
            maxprob.append(float(prob.max()))
            fgarea.append(int(pred.sum()))
            labels.append(1 if has_fg else 0)
            # pixel metrics on object frames
            if has_fg:
                tgt = mask >= 1
                inter = float((pred & tgt).sum())
                psum, gsum = float(pred.sum()), float(tgt.sum())
                union = psum + gsum - inter
                ious.append(inter / union if union else float("nan"))
                dices.append(2 * inter / (psum + gsum) if (psum + gsum) else float("nan"))

    labels = np.array(labels)
    miou = float(np.nanmean(ious))
    mdice = float(np.nanmean(dices))
    au_prob = auroc(np.array(maxprob), labels)
    au_area = auroc(np.array(fgarea, dtype=np.float64), labels)
    print(
        f"[eval] frames: {len(test)} total | {int(labels.sum())} object | {int((labels == 0).sum())} normal",
        flush=True,
    )
    print(
        f"[eval] SEGMENTATION (object frames): fg-IoU={miou:.4f}  fg-Dice={mdice:.4f}", flush=True
    )
    print(
        f"[eval] IMAGE-LEVEL AUROC: max-prob={au_prob:.4f}  pred-fg-area={au_area:.4f}", flush=True
    )


if __name__ == "__main__":
    main()
