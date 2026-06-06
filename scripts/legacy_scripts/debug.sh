#!/bin/bash
source ~/.bashrc
conda activate Med3DVLM

PYTHONPATH=. python - <<'PY'


import torch
from src.model.projector.builder import build_mm_projector
from src.model.encoder.builder import build_vision_tower
import json

# load config so builders can be called (adjust path if needed)
cfg_path = "./output/Med3DVLM-Qwen-2.5-7B-pretrain/checkpoint-261180/config.json"
cfg = json.load(open(cfg_path))

proj = build_mm_projector(cfg)
vis = build_vision_tower(cfg)

print("proj n keys:", len(proj.state_dict().keys()))
print("proj sample keys:")
for i, k in enumerate(list(proj.state_dict().keys())[:120]):
    print(i, k)

print("\nvision n keys:", len(vis.state_dict().keys()))
print("vision sample keys:")
for i, k in enumerate(list(vis.state_dict().keys())[:120]):
    print(i, k)
PY

