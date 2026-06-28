"""
Deploy-mode (FastBEVCus) accuracy verification on REAL nuScenes data.

FastBEVCus is a *streaming* detector: each forward() takes the current frame
(6 views) + that frame's full projection, and accumulates temporal context in
an internal rolling volume fused by the EMC op (tf_vec = identity, i.e. no
ego-motion compensation -- the deploy approximation).

This script streams the model over a temporally-contiguous segment of the val
set, decodes detections with the same post-processing as test_img_file_deploy,
and scores them against ground truth (center-distance match per class). It also
saves a top-down BEV image per frame (pred vs GT) for visual confirmation.

Unlike the base FastBEV (which fuses 4 motion-compensated frames in one forward
and is what tools/test.py evaluates), the deploy path only sees the current 6
views and relies on streamed history, so it warms up over the first few frames.

Create: 2026.06.28
"""

import sys

sys.path.append('.')
sys.path.append('tools')

import argparse
import os
import numpy as np
import cv2
import torch

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel

from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset

import test_img_file_deploy as dep  # reuse get_bboxes / bbox3d2result (no mmdeploy dep)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, default='configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4_export.py')
    parser.add_argument('--ref_config', type=str, default='configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py',
                        help='base FastBEV config used as the reference (its 4-frame path is what tools/test.py scores, mAP 0.2742)')
    parser.add_argument('--checkpoint', type=str, default='work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/epoch_20.pth')
    parser.add_argument('--start',      type=int, default=0,    help='first val-set index to stream from')
    parser.add_argument('--num',        type=int, default=20,   help='number of contiguous frames to stream')
    parser.add_argument('--score_thr',  type=float, default=0.3)
    parser.add_argument('--match_dist', type=float, default=2.0, help='BEV center-distance (m) for a TP match')
    parser.add_argument('--warmup',     type=int, default=4,    help='frames to skip in the aggregate metric (streaming warmup)')
    parser.add_argument('--save_dir',   type=str, default='work_dirs/deploy_seq_vis')
    return parser.parse_args()


# config class order (export config line 44) -- both preds and GT use this index
CLASS_NAMES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
COLORS = [(0, 255, 0), (0, 200, 200), (200, 200, 0), (0, 165, 255), (255, 0, 255),
          (255, 255, 0), (128, 0, 255), (0, 0, 255), (0, 128, 255), (200, 0, 100)]


def bev_corners(box_tensor):
    """xy of the 4 bottom corners for each box -> (N, 4, 2) in lidar frame."""
    from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
    if not isinstance(box_tensor, LiDARInstance3DBoxes):
        box_tensor = LiDARInstance3DBoxes(box_tensor, box_dim=box_tensor.shape[-1])
    corners = box_tensor.corners.numpy()          # (N, 8, 3)
    return corners[:, [0, 3, 7, 4], :2]           # bottom rectangle


def to_canvas(xy, rng, size):
    return np.round((xy + rng) / (2.0 * rng) * size).astype(np.int32)


def draw_boxes(canvas, corners, color, rng, size, thickness=1):
    for box in corners:
        pts = to_canvas(box, rng, size)
        for k in range(4):
            cv2.line(canvas, tuple(pts[k]), tuple(pts[(k + 1) % 4]), color, thickness)


def match(pred_xy, pred_lab, gt_xy, gt_lab, dist_thr):
    """Greedy nearest match, same class, center distance < dist_thr. -> (tp, n_pred, n_gt)"""
    n_pred, n_gt = len(pred_xy), len(gt_xy)
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt
    used = np.zeros(n_gt, dtype=bool)
    tp = 0
    for i in range(n_pred):
        d = np.linalg.norm(gt_xy - pred_xy[i], axis=1)
        d[gt_lab != pred_lab[i]] = 1e9
        d[used] = 1e9
        j = int(np.argmin(d))
        if d[j] < dist_thr:
            used[j] = True
            tp += 1
    return tp, n_pred, n_gt


def pred_xy_lab(res, score_thr):
    """bbox_result dict -> (boxes_kept, labels_kept, centers_xy)."""
    scores = res['scores_3d'].numpy()
    labels = res['labels_3d'].numpy()
    boxes = res['boxes_3d']
    keep = scores >= score_thr
    boxes = boxes[keep]
    lab = labels[keep]
    xy = boxes.gravity_center.numpy()[:, :2] if len(lab) else np.zeros((0, 2))
    return boxes, lab, xy


def build(cfg_path, checkpoint, test_cfg_from=None):
    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    m = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    ckpt = load_checkpoint(m, checkpoint, map_location='cpu')
    m = fuse_conv_bn(m)
    m.CLASSES = ckpt['meta']['CLASSES']
    return m, cfg


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # deploy (streaming) model
    model, cfg = build(args.config, args.checkpoint)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    model = MMDataParallel(model, device_ids=[0]).eval()

    # base (4-frame) reference model -- the path tools/test.py scores at mAP 0.2742
    base_model, _ = build(args.ref_config, args.checkpoint)
    base_model = base_model.cuda().eval()

    tf_vec = torch.eye(4, 4, dtype=torch.float32).cuda()

    rng, size = 50.0, 800  # BEV: +/-50 m -> 800 px
    end = min(args.start + args.num, len(dataset))

    # aggregates: [deploy, base]
    agg_tp = [0, 0]
    agg_pred = [0, 0]
    agg_gt = 0
    print('                 deploy (streaming)        base (4-frame ref)')
    print('idx frame  nGT  nPred  TP  recall  prec |  nPred  TP  recall  prec')
    for n, idx in enumerate(range(args.start, end)):
        data = dataset[idx]
        img = data['img'].data                                    # (24, 3, 256, 704)
        metas = data['img_metas'].data
        ext = np.asarray(metas['lidar2img']['extrinsic'])         # (24, 4, 4), full projection

        # --- deploy: streaming, current frame only (6 views) ---
        img6 = img[:6].unsqueeze(0).cuda()
        ext6 = torch.tensor(ext[:6], dtype=torch.float32).cuda()
        with torch.no_grad():
            out = model(img6, ext6, tf_vec)
        res_d = dep.bbox3d2result(*dep.get_bboxes(out[0], out[1], out[2])[0])
        boxes_d, lab_d, xy_d = pred_xy_lab(res_d, args.score_thr)

        # --- base: full 4-frame forward (reference) ---
        with torch.no_grad():
            res_b = base_model.simple_test(img.unsqueeze(0).cuda(), [metas])[0]
        boxes_b, lab_b, xy_b = pred_xy_lab(res_b, args.score_thr)

        # --- GT ---
        ann = dataset.get_ann_info(idx)
        gt = ann['gt_bboxes_3d']
        gt_lab = np.asarray(ann['gt_labels_3d'])
        gt = gt[gt_lab >= 0]
        gt_lab = gt_lab[gt_lab >= 0]
        gt_xy = gt.gravity_center.numpy()[:, :2] if len(gt_lab) else np.zeros((0, 2))
        n_gt = len(gt_lab)

        tp_d, np_d, _ = match(xy_d, lab_d, gt_xy, gt_lab, args.match_dist)
        tp_b, np_b, _ = match(xy_b, lab_b, gt_xy, gt_lab, args.match_dist)
        rec_d = tp_d / n_gt if n_gt else 0.0
        rec_b = tp_b / n_gt if n_gt else 0.0
        pr_d = tp_d / np_d if np_d else 0.0
        pr_b = tp_b / np_b if np_b else 0.0
        warm = ' (warmup)' if n < args.warmup else ''
        print('{:3d} {:4d}  {:4d}  {:5d} {:3d}  {:5.3f}  {:5.3f} | {:5d} {:3d}  {:5.3f}  {:5.3f}{}'.format(
            idx, n, n_gt, np_d, tp_d, rec_d, pr_d, np_b, tp_b, rec_b, pr_b, warm))

        if n >= args.warmup:
            agg_tp[0] += tp_d; agg_pred[0] += np_d
            agg_tp[1] += tp_b; agg_pred[1] += np_b
            agg_gt += n_gt

        # BEV: GT white, deploy(green), base(red, thin)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(canvas, (size // 2, size // 2), 4, (80, 80, 80), -1)
        if n_gt:
            draw_boxes(canvas, bev_corners(gt.tensor), (255, 255, 255), rng, size, 1)
        if len(lab_b):
            draw_boxes(canvas, bev_corners(boxes_b.tensor), (0, 0, 255), rng, size, 1)
        if len(lab_d):
            draw_boxes(canvas, bev_corners(boxes_d.tensor), (0, 255, 0), rng, size, 2)
        cv2.putText(canvas, 'idx {} | GT(white)={} deploy(green)={} base(red)={}'.format(idx, n_gt, np_d, np_b),
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(args.save_dir, 'bev_{:04d}.jpg'.format(idx)), canvas)

    print('\n=== aggregate (after {} warmup frames, dist<{}m, score>{}) ==='.format(
        args.warmup, args.match_dist, args.score_thr))
    n_frames = max(0, (end - args.start) - args.warmup)
    for k, name in enumerate(['deploy(stream)', 'base(4-frame) ']):
        R = agg_tp[k] / agg_gt if agg_gt else 0.0
        P = agg_tp[k] / agg_pred[k] if agg_pred[k] else 0.0
        print('{}  GT={}  pred={}  TP={}  recall={:.3f}  precision={:.3f}'.format(
            name, agg_gt, agg_pred[k], agg_tp[k], R, P))
    print('frames={}  | BEV images saved to {}'.format(n_frames, args.save_dir))


if __name__ == '__main__':
    main()
