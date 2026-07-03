"""
TensorRT-engine accuracy verification for the FastBEVCus deploy path.

Streams the TRT engine over a temporally-contiguous val segment and scores it
against GT, side-by-side with the PyTorch deploy model (FastBEVCus, streaming)
on the *same* frames. This is the Stage-B counterpart of tools/test_deploy_seq.py
and lets us quantify the ONNX/TRT "single-frame baked volume" quirk: the engine
traces self.volume[:3] as a zeros Constant, so it never accumulates history,
whereas the PyTorch model streams it.

Decoding reuses test_img_file_deploy.get_bboxes / bbox3d2result (no mmdeploy dep).
Plugins are ctypes-loaded via tools/trt_plugin (no mmcv-2.x mmdeploy install).

Create: 2026.07.03
"""

import sys

sys.path.append('.')
sys.path.append('tools')

import argparse
import os
import numpy as np
import cv2
import torch
import tensorrt as trt

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel

from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset

import test_img_file_deploy as dep
from trt_plugin import load_fastbev_trt_plugins

# reuse the exact BEV/matching helpers from the deploy harness
from test_deploy_seq import (CLASS_NAMES, bev_corners, draw_boxes, match,
                             pred_xy_lab)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str,
                   default='configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4_export.py')
    p.add_argument('--checkpoint', type=str,
                   default='work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/epoch_20.pth')
    p.add_argument('--trt_path', type=str, default='work_dirs/fastbev_m0_f4_fp16.engine')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--num', type=int, default=20)
    p.add_argument('--score_thr', type=float, default=0.3)
    p.add_argument('--match_dist', type=float, default=2.0)
    p.add_argument('--warmup', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='work_dirs/trt_seq_vis')
    return p.parse_args()


def _torch_dtype(trt_dtype):
    return {trt.bool: torch.bool, trt.int8: torch.int8, trt.int32: torch.int32,
            trt.float16: torch.float16, trt.float32: torch.float32}[trt_dtype]


class TRTWrapper(torch.nn.Module):
    """Minimal engine runner: binds only the inputs the engine actually declares
    (tf_vec may be constant-folded away at export), outputs as torch cuda tensors."""

    def __init__(self, engine_path):
        super().__init__()
        with trt.Logger(trt.Logger.WARNING) as logger, trt.Runtime(logger) as runtime:
            with open(engine_path, 'rb') as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        names = [n for n in self.engine]
        self._input_names = [n for n in names if self.engine.binding_is_input(n)]
        self._output_names = [n for n in names if not self.engine.binding_is_input(n)]

    def forward(self, inputs):
        bindings = [None] * self.engine.num_bindings
        for name in self._input_names:
            t = inputs[name].contiguous()
            idx = self.engine.get_binding_index(name)
            self.context.set_binding_shape(idx, tuple(t.shape))
            bindings[idx] = t.data_ptr()
        outputs = {}
        for name in self._output_names:
            idx = self.engine.get_binding_index(name)
            shape = tuple(self.context.get_binding_shape(idx))
            out = torch.zeros(shape, dtype=_torch_dtype(self.engine.get_binding_dtype(idx)),
                              device='cuda')
            outputs[name] = out
            bindings[idx] = out.data_ptr()
        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return outputs


def build_deploy(cfg_path, checkpoint):
    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    m = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    ckpt = load_checkpoint(m, checkpoint, map_location='cpu')
    m = fuse_conv_bn(m)
    m.CLASSES = ckpt['meta']['CLASSES']
    return m, cfg


def decode_trt(out):
    """engine outputs (output_0/1/2) -> bbox_result dict, via the deploy decoder."""
    o0, o1, o2 = out['output_0'], out['output_1'], out['output_2']
    res = dep.get_bboxes([o0], [o1], [o2])[0]
    return dep.bbox3d2result(*res)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    load_fastbev_trt_plugins()

    # deploy (streaming) PyTorch model -- the already-validated reference
    model, cfg = build_deploy(args.config, args.checkpoint)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    model = MMDataParallel(model, device_ids=[0]).eval()

    trt_model = TRTWrapper(args.trt_path)
    print('[trt] inputs={} outputs={}'.format(trt_model._input_names, trt_model._output_names))

    tf_vec = torch.eye(4, 4, dtype=torch.float32).cuda()
    rng, size = 50.0, 800
    end = min(args.start + args.num, len(dataset))

    agg_tp = [0, 0]      # [trt, deploy]
    agg_pred = [0, 0]
    agg_gt = 0
    print('                 TRT (engine)              deploy (streaming)')
    print('idx frame  nGT  nPred  TP  recall  prec |  nPred  TP  recall  prec')
    for n, idx in enumerate(range(args.start, end)):
        data = dataset[idx]
        img = data['img'].data
        metas = data['img_metas'].data
        ext = np.asarray(metas['lidar2img']['extrinsic'])
        img6 = img[:6].unsqueeze(0).cuda()
        ext6 = torch.tensor(ext[:6], dtype=torch.float32).cuda()

        # --- TRT engine (bind whatever inputs it declares) ---
        feed = {'input_0': img6, 'input_1': ext6, 'input_2': tf_vec}
        feed = {k: feed[k] for k in trt_model._input_names}
        out = trt_model.forward(feed)
        res_t = decode_trt(out)
        boxes_t, lab_t, xy_t = pred_xy_lab(res_t, args.score_thr)

        # --- deploy streaming PyTorch (reference) ---
        with torch.no_grad():
            od = model(img6, ext6, tf_vec)
        res_d = dep.bbox3d2result(*dep.get_bboxes(od[0], od[1], od[2])[0])
        boxes_d, lab_d, xy_d = pred_xy_lab(res_d, args.score_thr)

        # --- GT ---
        ann = dataset.get_ann_info(idx)
        gt = ann['gt_bboxes_3d']
        gt_lab = np.asarray(ann['gt_labels_3d'])
        gt = gt[gt_lab >= 0]
        gt_lab = gt_lab[gt_lab >= 0]
        gt_xy = gt.gravity_center.numpy()[:, :2] if len(gt_lab) else np.zeros((0, 2))
        n_gt = len(gt_lab)

        tp_t, np_t, _ = match(xy_t, lab_t, gt_xy, gt_lab, args.match_dist)
        tp_d, np_d, _ = match(xy_d, lab_d, gt_xy, gt_lab, args.match_dist)
        rec_t = tp_t / n_gt if n_gt else 0.0
        rec_d = tp_d / n_gt if n_gt else 0.0
        pr_t = tp_t / np_t if np_t else 0.0
        pr_d = tp_d / np_d if np_d else 0.0
        warm = ' (warmup)' if n < args.warmup else ''
        print('{:3d} {:4d}  {:4d}  {:5d} {:3d}  {:5.3f}  {:5.3f} | {:5d} {:3d}  {:5.3f}  {:5.3f}{}'.format(
            idx, n, n_gt, np_t, tp_t, rec_t, pr_t, np_d, tp_d, rec_d, pr_d, warm))

        if n >= args.warmup:
            agg_tp[0] += tp_t; agg_pred[0] += np_t
            agg_tp[1] += tp_d; agg_pred[1] += np_d
            agg_gt += n_gt

        # BEV: GT white, TRT(green, thick), deploy(red, thin)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(canvas, (size // 2, size // 2), 4, (80, 80, 80), -1)
        if n_gt:
            draw_boxes(canvas, bev_corners(gt.tensor), (255, 255, 255), rng, size, 1)
        if len(lab_d):
            draw_boxes(canvas, bev_corners(boxes_d.tensor), (0, 0, 255), rng, size, 1)
        if len(lab_t):
            draw_boxes(canvas, bev_corners(boxes_t.tensor), (0, 255, 0), rng, size, 2)
        cv2.putText(canvas, 'idx {} | GT(white)={} TRT(green)={} deploy(red)={}'.format(idx, n_gt, np_t, np_d),
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(args.save_dir, 'bev_{:04d}.jpg'.format(idx)), canvas)

    print('\n=== aggregate (after {} warmup frames, dist<{}m, score>{}) ==='.format(
        args.warmup, args.match_dist, args.score_thr))
    for k, name in enumerate(['TRT(engine)   ', 'deploy(stream)']):
        R = agg_tp[k] / agg_gt if agg_gt else 0.0
        P = agg_tp[k] / agg_pred[k] if agg_pred[k] else 0.0
        print('{}  GT={}  pred={}  TP={}  recall={:.3f}  precision={:.3f}'.format(
            name, agg_gt, agg_pred[k], agg_tp[k], R, P))
    print('BEV images saved to {}'.format(args.save_dir))


if __name__ == '__main__':
    main()
