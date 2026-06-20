"""Smoke-test the Fast-BEV deploy environment from inside the container.

Run:  docker compose -f docker/docker-compose.yml run --rm fastbev python docker/verify_env.py
Checks the full import chain and that custom CUDA ops actually execute on the GPU.
"""
import torch

print("=" * 60)
print("torch        :", torch.__version__)
print("cuda build   :", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available in container"
print("device       :", torch.cuda.get_device_name(0))
print("arch list    :", torch.cuda.get_arch_list())

import mmcv
import mmdet
import mmseg
print("mmcv         :", mmcv.__version__)
print("mmdet        :", mmdet.__version__)
print("mmseg        :", mmseg.__version__)

# mmcv compiled CUDA op (e.g. NMS) must run on sm_89 via PTX JIT
from mmcv.ops import nms
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]],
                     dtype=torch.float32, device="cuda")
scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
keep = nms(boxes, scores, 0.5)
print("mmcv NMS (cuda):", keep[1].tolist())

# this repo's mmdet3d custom ops
import mmdet3d
from mmdet3d.ops.iou3d import iou3d_cuda  # compiled .so
print("mmdet3d      :", mmdet3d.__version__, "(iou3d_cuda loaded)")

# build the Fast-BEV detector from the m0 config to confirm the model graph is valid
from mmcv import Config
from mmdet3d.models import build_model
cfg = Config.fromfile(
    "configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py")
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
n = sum(p.numel() for p in model.parameters())
print("fastbev model: built OK, params =", f"{n/1e6:.1f}M")

print("=" * 60)
print("ENVIRONMENT OK")
