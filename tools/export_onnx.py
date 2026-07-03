"""
Export FastBEVCus (deploy-mode detector) to ONNX with the custom mmdeploy ops
(project_2d_to_3d / emc) emitted as named nodes for the TensorRT plugins.

Ported from thor-fast-bev/tools/export_onnx.py (SG.SUH, 2023):
  - dropped the `mmdeploy.backend.tensorrt` import (mmcv 2.x dep); the plugin .so
    is only needed at TRT build/inference time, not for the symbolic export.
  - current-repo config/checkpoint paths.

NOTE: FastBEVCus keeps a persistent rolling `self.volume`; only the *current*
frame is a live input, so the past-3 frames are traced as a Constant (zeros) in
the graph -> the exported engine is effectively single-frame. This is expected
(see SESSION_HANDOFF "single frame baked" note); verify on the ONNX graph.

Create: 2026.07.03
"""

import sys

sys.path.append('.')

import argparse
import torch
import onnx

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmcv.cnn import fuse_conv_bn

from mmdet.apis import set_random_seed

from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4_export.py')
    parser.add_argument('--checkpoint', type=str,
                        default='work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/epoch_20.pth')
    parser.add_argument('--onnx_path', type=str, default='work_dirs/fastbev_m0_f4.onnx')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--opset', type=int, default=11)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=False)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=samples_per_gpu,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=False, shuffle=False)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = fuse_conv_bn(model)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model = MMDataParallel(model, device_ids=[0]).eval()

    tf_vec = torch.eye(4, 4, dtype=torch.float32).cuda()
    input_names = ['input_0', 'input_1', 'input_2']    # img, extrinsic, tf_vec
    output_names = ['output_0', 'output_1', 'output_2']  # cls_score, bbox_pred, dir_cls_pred

    import os
    os.makedirs(os.path.dirname(args.onnx_path) or '.', exist_ok=True)

    for data in data_loader:
        img = data['img'].data[0][0, :6, ...].unsqueeze(0).cuda()          # (1, 6, 3, 256, 704)
        ext = torch.tensor(data['img_metas'].data[0][0]['lidar2img']['extrinsic'][:6]).cuda()  # (6, 4, 4)
        with torch.no_grad():
            torch.onnx.export(model.module, (img, ext, tf_vec), args.onnx_path,
                              input_names=input_names, output_names=output_names,
                              opset_version=args.opset, verbose=args.verbose)
        break

    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)

    # Report the custom op nodes so we can confirm the plugins will bind.
    op_types = [n.op_type for n in onnx_model.graph.node]
    for want in ('project_2d_to_3d', 'emc'):
        print('  {:16s} nodes: {}'.format(want, op_types.count(want)))
    print('exported ONNX -> {}'.format(args.onnx_path))


if __name__ == '__main__':
    main()
