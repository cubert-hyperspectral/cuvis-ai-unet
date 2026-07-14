"""One-time: convert a raw torch_layers state_dict (.pt saved during diagnostics)
into a canonical cuvis-ai pipeline artifact (YAML + co-located, node-name-keyed .pt)
that CuvisPipeline.load_pipeline can consume directly.

Build the pipeline from its training config, load the diagnostic weights into
torch_layers, mark the normalizer initialized, then re-save with the framework's own
save_to_file — which round-trips with load_pipeline by construction. Future trainings
should just call pipeline.save_to_file and skip this shim.
"""
from __future__ import annotations

import argparse
import os

import torch

from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.utils.node_registry import NodeRegistry

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="training config YAML the checkpoint was saved under")
    ap.add_argument("--raw-ckpt", required=True, help="raw torch_layers.state_dict() .pt")
    ap.add_argument("--out", required=True, help="output artifact YAML path (weights land alongside)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="tolerate missing/unexpected checkpoint keys (DANGEROUS: index-keyed)")
    ap.add_argument("--unet-manifest", default=os.path.join(REPO_ROOT, "plugins.yaml"))
    ap.add_argument("--augment-manifest",
                    default=os.path.join(REPO_ROOT, "examples", "lentils", "augment.yaml"))
    args = ap.parse_args()

    reg = NodeRegistry()
    reg.register_plugin(args.unet_manifest)
    reg.register_plugin(args.augment_manifest)
    pipe = PipelineBuilder(node_registry=reg).build_from_config(args.config)
    sd = torch.load(args.raw_ckpt, map_location="cpu")
    missing, unexpected = pipe.torch_layers.load_state_dict(sd, strict=False)
    print(
        f"[conv] {os.path.basename(args.config)} <- {os.path.basename(args.raw_ckpt)}: "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if (missing or unexpected) and not args.allow_partial:
        raise SystemExit(
            "[conv] FATAL: checkpoint/config mismatch — index-keyed checkpoints require the "
            "exact nodes-list order they were saved under (--allow-partial to override)"
        )
    nodes = {n.name: n for n in pipe.nodes}
    nodes["Norm"]._statistically_initialized = True
    pipe.save_to_file(args.out, save_weights=True, validate_nodes=False)
    print(f"[conv] saved canonical artifact -> {args.out} (+ {os.path.splitext(args.out)[0]}.pt)", flush=True)


if __name__ == "__main__":
    main()
