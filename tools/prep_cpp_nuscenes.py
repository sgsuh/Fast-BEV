"""
Generate configs + image layout so the ported C++ inference app (deploy/cpp) can
run on nuScenes-mini.

The C++ app (apps/nuscenes.cpp) reads configs/{trt,param,calib}.yaml and, per
camera folder, streams the sorted *.jpg through TRTFastBEV::inference. Its inputs
mirror the validated Python deploy path exactly: input_0 = preprocessed 6 views,
input_1 = lidar2img 'extrinsic' (6x4x4), input_2 = identity tf_vec.

We take the projection straight from the FastBEV test pipeline (dataset[idx]'s
img_metas 'lidar2img'/'extrinsic') so it already bakes in the resize(704x396)+
crop(0,70,704,326) the C++ preprocessing reproduces. Camera calib for the C++
preprocessing (undistort->resize->crop) comes from the info pkl; dist_coef is
zeroed because FastBEV treats nuScenes cameras as pinhole (no undistortion).

Run inside the fastbev-trt/ fastbev service (needs mmdet3d):
  docker compose ... run --rm fastbev-trt bash -lc \
    'python tools/prep_cpp_nuscenes.py --start 0 --num 20'

Create: 2026.07.04
"""

import sys

sys.path.append('.')

import argparse
import os
import numpy as np

from mmcv import Config
from mmdet3d.datasets import build_dataset

# The C++ app + our validated path use the info's camera order directly.
CAM_ORDER = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def fmt_rows(rows, per=None):
    """YAML flow-style list-of-lists with full precision (repr floats)."""
    out = []
    for r in rows:
        out.append('[' + ', '.join(repr(float(v)) for v in r) + ']')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4_export.py')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--num', type=int, default=20)
    ap.add_argument('--cpp_dir', default='deploy/cpp')
    ap.add_argument('--onnx', default='/workspace/work_dirs/fastbev_m0_f4.onnx')
    ap.add_argument('--engine', default='/workspace/work_dirs/fastbev_m0_f4_cpp.engine')
    ap.add_argument('--out_root', default='/workspace/work_dirs/cpp_nuscenes')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    infos = dataset.data_infos
    n = min(args.num, len(dataset) - args.start)
    assert n > 0, 'empty frame range'

    data_root = os.path.join(args.out_root, 'data')
    save_dir = os.path.join(args.out_root, 'bev')
    for cam in CAM_ORDER:
        os.makedirs(os.path.join(data_root, cam), exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # --- extrinsic (projection) from the pipeline, static across the scene ---
    d0 = dataset[args.start]
    ext = np.asarray(d0['img_metas'].data['lidar2img']['extrinsic'])[:6]  # (6,4,4), CAM_ORDER
    assert ext.shape == (6, 4, 4), ext.shape

    info0 = infos[args.start]
    cams = info0['cams']

    cam_intrinsic, dist_coef = [], []
    s2l_rot, s2l_trans, s2e_rot, s2e_trans = [], [], [], []
    for cam in CAM_ORDER:
        c = cams[cam]
        cam_intrinsic.append(np.asarray(c['cam_intrinsic']).reshape(-1))          # 9
        dist_coef.append([0.0] * 5)                                               # pinhole
        s2l_rot.append(np.asarray(c['sensor2lidar_rotation']).reshape(-1))        # 9
        s2l_trans.append(np.asarray(c['sensor2lidar_translation']).reshape(-1))   # 3
        s2e_rot.append(np.asarray(c['sensor2ego_rotation']).reshape(-1))          # 4 (quat wxyz)
        s2e_trans.append(np.asarray(c['sensor2ego_translation']).reshape(-1))     # 3

    # --- write calib.yaml ---
    calib_path = os.path.join(args.cpp_dir, 'configs', 'calib.yaml')
    with open(calib_path, 'w') as f:
        f.write('# Auto-generated from nuScenes-mini by tools/prep_cpp_nuscenes.py\n')
        f.write('# extrinsic = FastBEV lidar2img (6x4x4, includes resize/crop), CAM_ORDER.\n')
        f.write('extrinsic: [' + ',\n            '.join(fmt_rows(ext.reshape(6, 16))) + ']\n')
        f.write('ego2global_rot: [' + ', '.join(repr(float(v)) for v in info0['ego2global_rotation']) + ']\n')
        f.write('ego2global_trans: [' + ', '.join(repr(float(v)) for v in info0['ego2global_translation']) + ']\n')
        f.write('lidar2ego_rot: [' + ', '.join(repr(float(v)) for v in info0['lidar2ego_rotation']) + ']\n')
        f.write('lidar2ego_trans: [' + ', '.join(repr(float(v)) for v in info0['lidar2ego_translation']) + ']\n')
        f.write('sensor2lidar_rot: [' + ',\n                   '.join(fmt_rows(s2l_rot)) + ']\n')
        f.write('cam_intrinsic: [' + ',\n                '.join(fmt_rows(cam_intrinsic)) + ']\n')
        f.write('dist_coef: [' + ',\n            '.join(fmt_rows(dist_coef)) + ']\n')
        f.write('sensor2ego_rot: [' + ',\n                 '.join(fmt_rows(s2e_rot)) + ']\n')
        f.write('sensor2lidar_trans: [' + ',\n                     '.join(fmt_rows(s2l_trans)) + ']\n')
        f.write('sensor2ego_trans: [' + ',\n                  '.join(fmt_rows(s2e_trans)) + ']\n')

    # --- param.yaml (nuScenes raw 1600x900; FastBEV resize/crop) ---
    param_path = os.path.join(args.cpp_dir, 'configs', 'param.yaml')
    with open(param_path, 'w') as f:
        f.write('# Auto-generated for nuScenes-mini by tools/prep_cpp_nuscenes.py\n')
        f.write('data_root: "%s"\n' % data_root)
        f.write('save_dir: "%s"\n' % save_dir)
        f.write('mean: [123.675, 116.28, 103.53]\n')
        f.write('std: [58.395, 57.12, 57.375]\n')
        f.write('model_height: 256\nmodel_width: 704\nimg_channel: 3\nn_times: 1\n')
        f.write('draw_boxes_indexes_bev: [[0, 1], [1, 2], [2, 3], [3, 0]]\n')
        f.write('color_map: [[70, 130, 180], [0, 0, 230], [135, 206, 235], [100, 149, 237], [219, 112, 147], [255, 61, 99], [240, 128, 128], [138, 43, 226], [112, 128, 144], [210, 105, 30]]\n')
        f.write('scale_factor: 4\ncanvas_size: 900\nshow_range: 50\nvis_thred: 0.3\n')
        f.write('resize_size: [704, 396]\n')
        f.write('crop_size: [0, 70, 704, 326]\n')
        f.write('fold_name: [%s]\n' % ', '.join('"%s"' % c for c in CAM_ORDER))
        f.write('raw_height: 900\nraw_width: 1600\nnum_cam: 6\n')

    # --- trt.yaml ---
    trt_path = os.path.join(args.cpp_dir, 'configs', 'trt.yaml')
    with open(trt_path, 'w') as f:
        f.write('Onnx: %s\n' % args.onnx)
        f.write('Engine: %s\n' % args.engine)
        f.write('WorkspaceSize: 512\nBatchSize: 1\nPrecision: FP16\n')
        f.write('Calibration: /workspace/work_dirs/CalibrationTablefastbev\n')
        f.write('InputNames: ["input_0", "input_1", "input_2"]\n')

    # --- lay out the contiguous scene's images as CAM/<idx>.jpg (symlinks) ---
    for k in range(n):
        info = infos[args.start + k]
        for cam in CAM_ORDER:
            src = os.path.abspath(info['cams'][cam]['data_path'])
            dst = os.path.join(data_root, cam, '%04d.jpg' % k)
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

    print('frames: %d (start=%d)' % (n, args.start))
    print('configs -> %s/configs/{calib,param,trt}.yaml' % args.cpp_dir)
    print('images  -> %s/<CAM>/*.jpg' % data_root)
    print('bev out -> %s' % save_dir)


if __name__ == '__main__':
    main()
