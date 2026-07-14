"""One-time: convert a raw torch_layers state_dict (.pt saved during diagnostics)
into a canonical cuvis-ai pipeline artifact (YAML + co-located, node-name-keyed .pt)
that CuvisPipeline.load_pipeline can consume directly.

Build the pipeline from its training config, load the diagnostic weights into
torch_layers, mark the normalizer initialized, then re-save with the framework's own
save_to_file — which round-trips with load_pipeline by construction. Future trainings
should just call pipeline.save_to_file and skip this shim.

Env: CONV_YAML (training config), CONV_RAW (diagnostic .pt), CONV_OUT (output .yaml).
"""
from __future__ import annotations

import os

import torch

from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.utils.node_registry import NodeRegistry

HERE = os.path.dirname(os.path.abspath(__file__))
UNET = "/mnt/data/anish/cuvis-ai-unet/plugins.yaml"
AUG = os.path.join(HERE, "augment_local.yaml")
YAML = os.environ["CONV_YAML"]
RAW = os.environ["CONV_RAW"]
OUT = os.environ["CONV_OUT"]


def main() -> None:
    reg = NodeRegistry()
    reg.register_plugin(UNET)
    reg.register_plugin(AUG)
    pipe = PipelineBuilder(node_registry=reg).build_from_config(YAML)
    sd = torch.load(RAW, map_location="cpu")
    missing, unexpected = pipe.torch_layers.load_state_dict(sd, strict=False)
    print(
        f"[conv] {os.path.basename(YAML)} <- {os.path.basename(RAW)}: "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    nodes = {n.name: n for n in pipe.nodes}
    nodes["Norm"]._statistically_initialized = True
    pipe.save_to_file(OUT, save_weights=True, validate_nodes=False)
    print(f"[conv] saved canonical artifact -> {OUT} (+ {os.path.splitext(OUT)[0]}.pt)", flush=True)


if __name__ == "__main__":
    main()
