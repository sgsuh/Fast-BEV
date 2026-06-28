"""
Create: 2023.05.25
Author: SG.SUH
Python: 3.8.8
PyTorch: 1.10.0
"""

import sys

sys.path.append('.')

import argparse
import os
import glob
import numpy as np
import cv2
import mmcv
import torch
import numba

from PIL import Image
from pyquaternion.quaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmcv import Config
from mmcv.runner   import load_checkpoint
from mmcv.cnn      import fuse_conv_bn
from mmcv.parallel import MMDataParallel

from mmdet3d.models import build_model
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu

from mmdet.datasets.pipelines import to_tensor

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',        type = str,  default = 'configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4_export.py')
    parser.add_argument('--checkpoint',    type = str,  default = 'work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/epoch_20.pth')
    parser.add_argument('--data_root',     type = str,  default = 'data/nuscenes/samples')
    parser.add_argument('--save_dir',      type = str,  default = 'work_dirs/deploy_vis')
    parser.add_argument('--max_frames',    type = int,  default = 0)

    args = parser.parse_args()

    return args

def anchors_single_range(feature_size, anchor_range, scale, sizes = [[1.6, 3.9, 1.56]], rotations = [0, 1.5707963], device = 'cuda'):
    align_corner = False
    custom_values = [0, 0]

    if len(feature_size) == 2:
        feature_size = [1, feature_size[0], feature_size[1]]

    anchor_range = torch.tensor(anchor_range, device = device)
    z_centers = torch.linspace(anchor_range[2], anchor_range[5], feature_size[0] + 1, device = device)
    y_centers = torch.linspace(anchor_range[1], anchor_range[4], feature_size[1] + 1, device = device)
    x_centers = torch.linspace(anchor_range[0], anchor_range[3], feature_size[2] + 1, device = device)
    sizes = torch.tensor(sizes, device = device).reshape(-1, 3) * scale
    rotations = torch.tensor(rotations, device = device)

    if not align_corner:
        z_shift = (z_centers[1] - z_centers[0]) / 2
        y_shift = (y_centers[1] - y_centers[0]) / 2
        x_shift = (x_centers[1] - x_centers[0]) / 2
        z_centers += z_shift
        y_centers += y_shift
        x_centers += x_shift

    rets = torch.meshgrid(x_centers[:feature_size[2]], y_centers[:feature_size[1]], z_centers[:feature_size[0]], rotations)

    rets = list(rets)
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])

    for i in range(len(rets)):
        rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

    sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = sizes.repeat(tile_size_shape)
    rets.insert(3, sizes)

    ret = torch.cat(rets, dim = -1).permute([2, 1, 0, 3, 4, 5])

    if len(custom_values) > 0:
        custom_ndim = len(custom_values)
        custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])

        ret = torch.cat([ret, custom], dim = -1)

    return ret

def single_level_grid_anchors(featmap_size, scale, device = 'cuda'):
    ranges = [[-50, -50, -1.8, 50, 50, -1.8], [-50, -50, -1.8, 50, 50, -1.8], [-50, -50, -1.8, 50, 50, -1.8], [-50, -50, -1.8, 50, 50, -1.8]]
    sizes = [[0.866, 2.5981, 1.0], [0.5774, 1.7321, 1.0], [1.0, 1.0, 1.0], [0.4, 0.4, 1]]
    rotations = [0, 1.57]

    mr_anchors = []

    for anchor_range, anchor_size in zip(ranges, sizes):
        mr_anchors.append(anchors_single_range(featmap_size, anchor_range, scale, anchor_size, rotations, device = device))

    mr_anchors = torch.cat(mr_anchors, dim = -3)

    return mr_anchors

def grid_anchors(featmap_sizes, device = 'cuda'):
    num_levels = 1
    scales = [1]
    reshape_out = True

    multi_level_anchors = []

    for i in range(num_levels):
        anchors = single_level_grid_anchors(featmap_sizes[i], scales[i], device = device)

        if reshape_out:
            anchors = anchors.reshape(-1, anchors.size(-1))

        multi_level_anchors.append(anchors)

    return multi_level_anchors

def decode(anchors, deltas):
    cas, cts = [], []
    box_ndim = anchors.shape[-1]

    if box_ndim > 7:
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim = -1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim = -1)
    else:
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim = -1)
        xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim = -1)

    za = za + ha / 2
    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    lg = torch.exp(lt) * la
    wg = torch.exp(wt) * wa
    hg = torch.exp(ht) * ha
    rg = rt + ra
    zg = zg - hg / 2
    cgs = [t + a for t, a in zip(cts, cas)]

    return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim = -1)

@numba.jit(nopython = True)
def circle_nms(dets, thresh, post_max_size = 83):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype = np.int32)
    keep = []

    for _i in range(ndets):
        i = order[_i]

        if suppressed[i] == 1:
            continue

        keep.append(i)

        for _j in range(_i + 1, ndets):
            j = order[_j]

            if suppressed[j] == 1:
                continue

            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            if dist <= thresh:
                suppressed[j] = 1

    return keep[:post_max_size]

def xywhr2xyxyr(boxes_xywhr):
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]

    return boxes

def box3d_multiclass_scale_nms(mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores, score_thr, max_num, mlvl_dir_scores = None, mlvl_attr_scores = None, mlvl_bboxes2d = None):
    nms_type_list = ['rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'circle']
    nms_thr_list = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.2]
    nms_radius_thr_list = [4, 12, 10, 10, 12, 0.85, 0.85, 0.175, 0.175, 1]
    nms_rescale_factor = [1.0, 0.7, 0.55, 0.4, 0.7, 1.0, 1.0, 4.5, 9.0, 1.0]

    num_classes = mlvl_scores.shape[1] - 1
    bboxes = []
    scores = []
    labels = []
    dir_scores = []
    attr_scores = []
    bboxes2d = []

    for i in range(num_classes):
        cls_inds = mlvl_scores[:, i] > score_thr

        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]

        nms_func = {'rotate': nms_gpu, 'circle': circle_nms}[nms_type_list[i]]

        nms_thre = nms_thr_list[i]
        nms_radius_thre = nms_radius_thr_list[i]
        nms_target_thre = {'rotate': nms_thre, 'circle': nms_radius_thre}[nms_type_list[i]]

        nms_rescale = nms_rescale_factor[i]
        _bboxes_for_nms[:, 2:4] *= nms_rescale

        if nms_type_list[i] == 'rotate':
            _bboxes_for_nms = xywhr2xyxyr(_bboxes_for_nms)
            selected = nms_func(_bboxes_for_nms, _scores, nms_target_thre)
        else:
            _centers = _bboxes_for_nms[:, [0, 1]]
            _bboxes_for_nms = torch.cat([_centers, _scores.view(-1, 1)], dim = 1)
            selected = nms_func(_bboxes_for_nms.detach().cpu().numpy(), nms_target_thre)
            selected = torch.tensor(selected, dtype = torch.long, device = _bboxes_for_nms.device)

        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])

        cls_label = mlvl_bboxes.new_full((len(selected), ), i, dtype = torch.long)

        labels.append(cls_label)

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]

            dir_scores.append(_mlvl_dir_scores[selected])

        if mlvl_attr_scores is not None:
            _mlvl_attr_scores = mlvl_attr_scores[cls_inds]

            attr_scores.append(_mlvl_attr_scores[selected])

        if mlvl_bboxes2d is not None:
            _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]

            bboxes2d.append(_mlvl_bboxes2d[selected])

    if bboxes:
        bboxes = torch.cat(bboxes, dim = 0)
        scores = torch.cat(scores, dim = 0)
        labels = torch.cat(labels, dim = 0)

        if mlvl_dir_scores is not None:
            dir_scores = torch.cat(dir_scores, dim = 0)

        if mlvl_attr_scores is not None:
            attr_scores = torch.cat(attr_scores, dim = 0)

        if mlvl_bboxes2d is not None:
            bboxes2d = torch.cat(bboxes2d, dim = 0)

        if bboxes.shape[0] > max_num:
            _, inds = scores.sort(descending = True)
            inds = inds[:max_num]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]

            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]

            if mlvl_attr_scores is not None:
                attr_scores = attr_scores[inds]

            if mlvl_bboxes2d is not None:
                bboxes2d = bboxes2d[inds]
    else:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0, ))
        labels = mlvl_scores.new_zeros((0, ), dtype = torch.long)

        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores.new_zeros((0, ))

        if mlvl_attr_scores is not None:
            attr_scores = mlvl_scores.new_zeros((0, ))

        if mlvl_bboxes2d is not None:
            bboxes2d = mlvl_scores.new_zeros((0, 4))

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )

    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )

    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )

    return results  

def limit_period(val, offset = 0.5, period = np.pi):
    return val - torch.floor(val / period + offset) * period

def get_bboxes_single(cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
    num_classes = 10
    box_code_size = 9
    dir_offset = 0.7854
    dir_limit_offset = 0

    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_dir_scores = []

    for cls_score, bbox_pred, dir_cls_pred, anchors in zip(cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        
        dir_cls_score = torch.max(dir_cls_pred, dim = -1)[1]

        cls_score = cls_score.permute(1, 2, 0).reshape(-1, num_classes)
        
        scores = cls_score
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, box_code_size)
        
        nms_pre = 1000

        if scores.shape[0] > nms_pre:
            max_scores, _= scores.max(dim = 1)

            _, topk_inds = max_scores.topk(nms_pre)
            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]
            dir_cls_score = dir_cls_score[topk_inds]

        bboxes = decode(anchors, bbox_pred)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        mlvl_dir_scores.append(dir_cls_score)

    mlvl_bboxes = torch.cat(mlvl_bboxes)
    mlvl_scores = torch.cat(mlvl_scores)
    mlvl_dir_scores = torch.cat(mlvl_dir_scores)

    padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
    mlvl_scores = torch.cat([mlvl_scores, padding], dim = 1)

    score_thr = 0.05
    max_num = 500

    mlvl_bboxes_for_nms = LiDARInstance3DBoxes(mlvl_bboxes, box_dim = box_code_size).bev
    results = box3d_multiclass_scale_nms(mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores, score_thr, max_num, mlvl_dir_scores)

    bboxes, scores, labels, dir_scores = results

    if bboxes.shape[0] > 0:
        dir_rot = limit_period(bboxes[..., 6] - dir_offset, dir_limit_offset, np.pi)
        bboxes[..., 6] = (dir_rot + dir_offset + np.pi * dir_scores.to(bboxes.dtype))

    bboxes = LiDARInstance3DBoxes(bboxes, box_dim = box_code_size)

    return bboxes, scores, labels   

def get_bboxes(cls_scores, bbox_preds, dir_cls_preds):
    box_code_size = 9

    num_levels = len(cls_scores)
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    # featmap_sizes = [cls_scores[i].shape[1:3] for i in range(num_levels)]
    device = cls_scores[0].device
    mlvl_anchors = grid_anchors(featmap_sizes, device = device)
    mlvl_anchors = [anchor.reshape(-1, box_code_size) for anchor in mlvl_anchors]

    result_list = []

    for img_id in range(len(cls_scores[0])):
        cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
        dir_cls_pred_list = [dir_cls_preds[i][img_id].detach() for i in range(num_levels)]

        proposals = get_bboxes_single(cls_score_list, bbox_pred_list, dir_cls_pred_list, mlvl_anchors)
        
        result_list.append(proposals)

    return result_list

def bbox3d2result(bboxes, scores, labels, attrs = None):
    result_dict = dict(boxes_3d = bboxes.to('cpu'), scores_3d = scores.cpu(), labels_3d = labels.cpu())

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict

def get_lidar2global(lidar2ego_rot, lidar2ego_trans, ego2global_rot, ego2global_trans):
    lidar2ego = np.eye(4, dtype = np.float32)
    lidar2ego[:3, :3] = Quaternion(lidar2ego_rot).rotation_matrix
    lidar2ego[:3, 3] = lidar2ego_trans
    
    ego2global = np.eye(4, dtype = np.float32)
    ego2global[:3, :3] = Quaternion(ego2global_rot).rotation_matrix
    ego2global[:3, 3] = ego2global_trans

    return ego2global @ lidar2ego

def lidar2img(points_lidar, sensor2lidar_rot, sensor2lidar_trans, cam_intrinsic):
    points_lidar_homogeneous = np.concatenate([points_lidar, np.ones((points_lidar.shape[0], 1), dtype = points_lidar.dtype)], axis = 1)
    camera2lidar = np.eye(4, dtype = np.float32)
    camera2lidar[:3, :3] = sensor2lidar_rot
    camera2lidar[:3, 3] = sensor2lidar_trans
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype = bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = cam_intrinsic
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]

    return points_img, valid

def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))

    return valid

def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok = True)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = build_model(cfg.model, test_cfg = cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, args.checkpoint, map_location = 'cpu')

    model = fuse_conv_bn(model)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']
    model = MMDataParallel(model, device_ids = [0])
    model.eval()

    cam_folds = []

    cam_folds.append(os.path.join(args.data_root, 'CAM_FRONT'))
    cam_folds.append(os.path.join(args.data_root, 'CAM_FRONT_RIGHT'))
    cam_folds.append(os.path.join(args.data_root, 'CAM_FRONT_LEFT'))
    cam_folds.append(os.path.join(args.data_root, 'CAM_BACK'))
    cam_folds.append(os.path.join(args.data_root, 'CAM_BACK_LEFT'))
    cam_folds.append(os.path.join(args.data_root, 'CAM_BACK_RIGHT'))
    

    cam_file_lists = []

    for cam_fold in cam_folds:
        file_list = glob.glob(cam_fold + '/*.jpg')
        file_list.sort()

        cam_file_lists.append(file_list)

    file_length = len(cam_file_lists[0])

    mean = np.array([123.675, 116.28 , 103.53], dtype = np.float32)
    std = np.array([58.395, 57.12 , 57.375], dtype = np.float32)

    
    img_tensor = torch.zeros((1, 6, 3, 256, 704), dtype = torch.float32)

    extrinsic_arrays = []

    
    extrinsic_arrays.append(np.array([[5.5716125e+02,  3.5910556e+02,  1.0331641e+01, -1.3998796e+02],
                                      [3.6186972e+00,  1.5661983e+02, -5.5438989e+02, -2.4560089e+02],
                                      [-1.4038675e-04,  9.9982643e-01,  1.8631339e-02, -4.0834507e-01],
                                      [0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]], dtype = np.float32))
    
    extrinsic_arrays.append(np.array([[6.01755676e+02, -2.68105164e+02, -1.29301262e+01, -2.14842743e+02],
                                    [1.17733795e+02,  9.47874146e+01, -5.53909180e+02, -2.78117889e+02],
                                    [8.35612714e-01,  5.49300551e-01,  4.51244926e-03, -5.99209726e-01],
                                    [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]], dtype = np.float32))
    
    extrinsic_arrays.append(np.array([[2.5401711e+01,  6.6702332e+02,  1.6014914e+01, -9.6030174e+01],
                                    [-1.1391969e+02,  9.4671783e+01, -5.5812402e+02, -2.6040109e+02],
                                    [-8.1728321e-01,  5.7611644e-01,  1.1746306e-02, -4.9458849e-01],
                                    [0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]], dtype = np.float32))
    
    extrinsic_arrays.append(np.array([[-3.5820590e+02, -3.6269446e+02, -6.1956205e+00, -3.7566284e+02],
                                    [2.6065648e+00, -1.3930310e+02, -3.5710452e+02, -2.4595389e+02],
                                    [-5.9521976e-03, -9.9995369e-01, -7.5646620e-03, -1.0286568e+00],
                                    [0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]], dtype = np.float32))
    
    extrinsic_arrays.append(np.array([[-5.0567935e+02,  4.1414996e+02,  3.5667849e+00, -2.7428888e+02],
                                    [-1.2824323e+02, -2.8231831e+01, -5.5685260e+02, -1.9949387e+02],
                                    [-9.4828844e-01, -3.1605947e-01, -2.9247982e-02, -4.4169033e-01],
                                    [0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]], dtype = np.float32))
    
    extrinsic_arrays.append(np.array([[1.3247704e+02, -6.4422192e+02, -2.6663824e+01, -1.5711188e+02],
                                    [1.3741817e+02, -3.1693459e+01, -5.5667950e+02, -2.2775493e+02],
                                    [9.3327790e-01, -3.5861987e-01, -1.9599995e-02, -5.0429916e-01],
                                    [0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]], dtype = np.float32))

    extrinsic_arrays = np.ascontiguousarray(np.stack(extrinsic_arrays, axis = 0))
    extrinsic_tensor = torch.from_numpy(extrinsic_arrays)

    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    default_attribute = {'car': 'vehicle.parked', 'pedestrian': 'pedestrian.moving', 'trailer': 'vehicle.parked', 'truck': 'vehicle.parked', 'bus': 'vehicle.moving', 'motorcycle': 'cycle.without_rider', 
                            'construction_vehicle': 'vehicle.parked', 'bicycle': 'cycle.without_rider', 'barrier': '', 'traffic_cone': '',}
    draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
    color_map = [[70, 130, 180], [0, 0, 230], [135, 206, 235], [100, 149, 237], [219, 112, 147], [255, 61, 99], [240, 128, 128], [138, 43, 226], [112, 128, 144], [210, 105, 30]]
    scale_factor = 4
    canvas_size = 900
    show_range = 50
    vis_thred = 0.3

    ego2global_rot = [0.41589926602121, -0.00899084029923763, 0.003201053083100996, 0.9093606097543983]     # pose_record['rotation']
    ego2global_trans = [732.0614444472142, 949.0676735861334, 0.0]                                          # pose_record['translation']
                             
    lidar2ego_rot = [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]
    lidar2ego_trans = [0.943713, 0.0, 1.84023]

    sensor2lidar_rot = []

    sensor2lidar_rot.append(np.array([[ 0.57623081,  0.00248489, -0.81728323],
                                    [ 0.81701791,  0.02390796,  0.57611643],
                                    [ 0.02097116, -0.99971108,  0.01174631]], dtype = np.float32))
    sensor2lidar_rot.append(np.array([[ 9.99978663e-01,  6.53100607e-03, -1.40386752e-04],
                                    [ 1.86780058e-05,  1.86318577e-02,  9.99826412e-01],
                                    [ 6.53248803e-03, -9.99805081e-01,  1.86313382e-02]], dtype = np.float32))
    sensor2lidar_rot.append(np.array([[ 0.54921636, -0.01062185,  0.83561269],
                                    [-0.83526943,  0.02437028,  0.54930053],
                                    [-0.0261987 , -0.99964657,  0.00451245]], dtype = np.float32))
    sensor2lidar_rot.append(np.array([[-0.31678754,  0.01986692, -0.94828844],
                                    [ 0.94817002,  0.03286356, -0.31605948],
                                    [ 0.02488501, -0.99926237, -0.02924798]], dtype = np.float32))
    sensor2lidar_rot.append(np.array([[-0.9999353 ,  0.00969415, -0.0059522 ],
                                    [ 0.00602508,  0.00750674, -0.99995367],
                                    [-0.00964902, -0.99992483, -0.00756466]], dtype = np.float32))
    sensor2lidar_rot.append(np.array([[-0.3591125 , -0.00552986,  0.9332779 ],
                                    [-0.93261692,  0.04021792, -0.35861986],
                                    [-0.03555138, -0.99917563, -0.01959999]], dtype = np.float32))

    sensor2lidar_rot = np.ascontiguousarray(np.stack(sensor2lidar_rot, axis = 0))

    sensor2lidar_trans = []

    sensor2lidar_trans.append(np.array([-0.48966924,  0.17072295, -0.33765879], dtype = np.float32))
    sensor2lidar_trans.append(np.array([-0.00985256,  0.41448905, -0.3259788 ], dtype = np.float32))
    sensor2lidar_trans.append(np.array([ 0.49888152,  0.33472889, -0.33878223], dtype = np.float32))
    sensor2lidar_trans.append(np.array([-0.48295751,  0.07475999, -0.25081762], dtype = np.float32))
    sensor2lidar_trans.append(np.array([-0.00438737, -1.02649698, -0.28835009], dtype = np.float32))
    sensor2lidar_trans.append(np.array([ 0.48339909, -0.13279111, -0.28224232], dtype = np.float32))

    sensor2lidar_trans = np.ascontiguousarray(np.stack(sensor2lidar_trans, axis = 0))

    cam_intrinsic = []

    cam_intrinsic.append(np.array([[1.27259795e+03, 0.00000000e+00, 8.26615493e+02],
                                [0.00000000e+00, 1.27259795e+03, 4.79751654e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype = np.float32))
    cam_intrinsic.append(np.array([[1.26641720e+03, 0.00000000e+00, 8.16267020e+02],
                                [0.00000000e+00, 1.26641720e+03, 4.91507066e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype = np.float32))
    cam_intrinsic.append(np.array([[1.26084744e+03, 0.00000000e+00, 8.07968245e+02],
                                [0.00000000e+00, 1.26084744e+03, 4.95334427e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype = np.float32))
    cam_intrinsic.append(np.array([[1.25674148e+03, 0.00000000e+00, 7.92112574e+02],
                                [0.00000000e+00, 1.25674148e+03, 4.92775747e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype = np.float32))
    cam_intrinsic.append(np.array([[809.22099057,   0.        , 829.21960033],
                                [  0.        , 809.22099057, 481.77842385],
                                [  0.        ,   0.        ,   1.        ]], dtype = np.float32))
    cam_intrinsic.append(np.array([[1.25951374e+03, 0.00000000e+00, 8.07252905e+02],
                                [0.00000000e+00, 1.25951374e+03, 5.01195799e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype = np.float32))

    cam_intrinsic = np.ascontiguousarray(np.stack(cam_intrinsic, axis = 0))

    sensor2ego_rot = []

    sensor2ego_rot.append(np.array([0.6757265034669446, -0.6736266522251881, 0.21214015046209478, -0.21122827103904068], dtype = np.float32))
    sensor2ego_rot.append(np.array([0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755], dtype = np.float32))
    sensor2ego_rot.append(np.array([0.2060347966337182, -0.2026940577919598, 0.6824507824531167, -0.6713610884174485], dtype = np.float32))
    sensor2ego_rot.append(np.array([0.6924185592174665, -0.7031619420114925, -0.11648342771943819, 0.11203317912370753], dtype = np.float32))
    sensor2ego_rot.append(np.array([0.5037872666382278, -0.49740249788611096, -0.4941850223835201, 0.5045496097725578], dtype = np.float32))
    sensor2ego_rot.append(np.array([0.12280980120078765, -0.132400842670559, -0.7004305821388234, 0.690496031265798], dtype = np.float32))

    sensor2ego_rot = np.ascontiguousarray(np.stack(sensor2ego_rot, axis = 0))

    sensor2ego_trans = []

    sensor2ego_trans.append(np.array([1.52387798135, 0.494631336551, 1.50932822144], dtype = np.float32))
    sensor2ego_trans.append(np.array([1.70079118954, 0.0159456324149, 1.51095763913], dtype = np.float32))
    sensor2ego_trans.append(np.array([1.5508477543, -0.493404796419, 1.49574800619], dtype = np.float32))
    sensor2ego_trans.append(np.array([1.03569100218, 0.484795032713, 1.59097014818], dtype = np.float32))
    sensor2ego_trans.append(np.array([0.0283260309358, 0.00345136761476, 1.57910346144], dtype = np.float32))
    sensor2ego_trans.append(np.array([1.0148780988, -0.480568219723, 1.56239545128], dtype = np.float32))

    sensor2ego_trans = np.ascontiguousarray(np.stack(sensor2ego_trans, axis = 0))

    tf_vec = torch.eye(4, 4, dtype = torch.float32)

    if args.max_frames > 0:
        file_length = min(file_length, args.max_frames)

    for idx in range(file_length):
        imgs = []

        for cam_file_list in cam_file_lists:
            img = Image.open(cam_file_list[idx])
            img = img.resize((704, 396))
            img = img.crop((0, 70, 704, 326))

            img = np.asarray(img)
            
            imgs.append(img)
            
        imgs = [mmcv.imnormalize(img, mean, std, True) for img in imgs]
        imgs = [img.transpose(2, 0, 1) for img in imgs]
        imgs = np.ascontiguousarray(np.stack(imgs, axis = 0))
        imgs = to_tensor(imgs)

        
        img_tensor[0, :6, ...] = imgs

        with torch.no_grad():
            result = model(img_tensor.cuda(), extrinsic_tensor.cuda(), tf_vec.cuda())
        
            
            bbox_list = get_bboxes(result[0], result[1], result[2])
            bbox_results = [bbox3d2result(det_bboxes, det_scores, det_labels) for det_bboxes, det_scores, det_labels in bbox_list]

        l2g = get_lidar2global(lidar2ego_rot, lidar2ego_trans, ego2global_rot, ego2global_trans)

        for sample_id, det in enumerate(bbox_results):
            box3d = det['boxes_3d']
            scores = det['scores_3d'].numpy()
            labels = det['labels_3d'].numpy()

            box_gravity_center = box3d.gravity_center.numpy()
            box_dims = box3d.dims.numpy()
            box_yaw = box3d.yaw.numpy()

            box_yaw = -box_yaw - np.pi / 2

            velocity_all = box3d.tensor[:, 7:7 + 2]

            annos = []

            canvas = np.zeros((int(canvas_size), int(canvas_size), 3), dtype = np.uint8)
            show_img = np.zeros((900 * 2 + canvas_size * scale_factor, 1600 * 3, 3), dtype = np.uint8)
            cv_imgs = [0 for _ in range(6)]
            pos_idx = [1, 2, 0, 4, 3, 5]

            for cam_idx, cam_file_list in enumerate(cam_file_lists):
                cv_img = cv2.imread(cam_file_lists[cam_idx][idx])

                cv_imgs[pos_idx[cam_idx]] = cv_img

            for j in range(len(box3d)):
                quat = Quaternion(axis = [0, 0, 1], radians = box_yaw[j])
                velocity = (*velocity_all[j, :], 0.0)

                lidar_box = NuScenesBox(box_gravity_center[j], box_dims[j], quat, label = labels[j], score = scores[j], velocity = velocity)

                if lidar_box.score < vis_thred:
                    continue

                lidar_box.rotate(Quaternion(lidar2ego_rot))
                lidar_box.translate(np.array(lidar2ego_trans))

                lidar_box.rotate(Quaternion(ego2global_rot))
                lidar_box.translate(np.array(ego2global_trans))

                pred_box = [lidar_box.center.tolist() + lidar_box.wlh.tolist() + [-Quaternion(lidar_box.orientation.elements.tolist()).yaw_pitch_roll[0] - np.pi / 2]]
                pred_box = np.array(pred_box, dtype = np.float32)
                box = LiDARInstance3DBoxes(pred_box, origin = (0.5, 0.5, 0.0))
                corners_global = box.corners.numpy().reshape(-1, 3)
                corners_global = np.concatenate([corners_global, np.ones([corners_global.shape[0], 1])], axis = 1)
                corners_lidar = corners_global @ np.linalg.inv(l2g).T
                corners_lidar = corners_lidar[:, :3]
                corners_lidar = corners_lidar.reshape(-1, 8, 3)
                corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
                bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
                bottom_corners_bev = (bottom_corners_bev + show_range) / show_range / 2.0 * canvas_size
                bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
                center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis = 1)
                head_bev = corners_lidar[:, [0, 4], :2].mean(axis = 1)
                center_canvas = (center_bev + show_range) / show_range / 2.0 * canvas_size
                center_canvas = center_canvas.astype(np.int32)
                head_canvas = (head_bev + show_range) / show_range / 2.0 * canvas_size
                head_canvas = head_canvas.astype(np.int32)

                color = color_map[int(lidar_box.label)]

                for index in draw_boxes_indexes_bev:
                    cv2.line(canvas, tuple(bottom_corners_bev[0, index[0]]), tuple(bottom_corners_bev[0, index[1]]), color, thickness = 1)

                cv2.line(canvas, tuple(center_canvas[0]), tuple(head_canvas[0]), color, 1, lineType = 8)

                for cam_idx, cam_file_list in enumerate(cam_file_lists):
                    camera_box = lidar_box.copy()

                    camera_box.translate(-np.array(ego2global_trans))
                    camera_box.rotate(Quaternion(ego2global_rot).inverse)
                    
                    camera_box.translate(-np.array(sensor2ego_trans[cam_idx]))
                    camera_box.rotate(Quaternion(sensor2ego_rot[cam_idx]).inverse)

                    corners_camera = camera_box.corners()
                    viewpad = np.eye(4)
                    viewpad[:cam_intrinsic[cam_idx].shape[0], :cam_intrinsic[cam_idx].shape[1]] = cam_intrinsic[cam_idx]
                    nbr_points = corners_camera.shape[1]
                    corners_img = np.concatenate((corners_camera, np.ones((1, nbr_points))))
                    corners_img = np.dot(viewpad, corners_img)
                    corners_img = corners_img[:3, :]
                    corners_img = corners_img / corners_img[2:3, :].repeat(3, 0).reshape(3, nbr_points)
                    corners_img = corners_img[:2, :]

                    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < cv_imgs[cam_idx].shape[1])
                    visible = np.logical_and(visible, corners_img[1, :] < cv_imgs[cam_idx].shape[0])
                    visible = np.logical_and(visible, corners_img[1, :] > 0)
                    visible = np.logical_and(visible, corners_camera[2, :] > 1)

                    in_front = corners_camera[2, :] > 0.1

                    if not any(visible):
                        continue

                    if not all(in_front):
                        continue

                    for i in range(4):
                        cv2.line(cv_imgs[cam_idx], (int(corners_img[0][i]), int(corners_img[1][i])), (int(corners_img[0][i + 4]), int(corners_img[1][i + 4])), color = color_map[int(camera_box.label)], thickness = scale_factor)

                    selected_corners = corners_img.T[:4]

                    prev = selected_corners[-1]

                    for corner in selected_corners:
                        cv2.line(cv_imgs[cam_idx], (int(prev[0]), int(prev[1])), (int(corner[0]), int(corner[1])), color = color_map[int(camera_box.label)], thickness = scale_factor)

                        prev = corner

                    selected_corners = corners_img.T[4:]

                    prev = selected_corners[-1]

                    for corner in selected_corners:
                        cv2.line(cv_imgs[cam_idx], (int(prev[0]), int(prev[1])), (int(corner[0]), int(corner[1])), color = color_map[int(camera_box.label)], thickness = scale_factor)

                        prev = corner

                    center_bottom_forward = np.mean(corners_img.T[2:4], axis = 0)
                    center_bottom = np.mean(corners_img.T[[2, 3, 7, 6]], axis = 0)

                    cv2.line(cv_imgs[cam_idx], (int(center_bottom[0]), int(center_bottom[1])), (int(center_bottom_forward[0]), int(center_bottom_forward[1])), color = color_map[int(camera_box.label)], thickness = scale_factor)

            show_img[:900, :, :] = np.concatenate(cv_imgs[:3], axis = 1)
            img_back = np.concatenate([cv_imgs[3][:, ::-1, :], cv_imgs[4][:, ::-1, :], cv_imgs[5][:, ::-1, :]], axis = 1)
            show_img[900 + canvas_size * scale_factor:, :, :] = img_back
            show_img = cv2.resize(show_img, (int(1600 / scale_factor * 3), int(900 / scale_factor * 2 + canvas_size)))
            w_begin = int((1600 * 3 / scale_factor - canvas_size) // 2)
            show_img[int(900 / scale_factor):int(900 / scale_factor) + canvas_size, w_begin:w_begin + canvas_size, :] = canvas

            save_path = os.path.join(args.save_dir, '{:06d}.jpg'.format(idx))
            cv2.imwrite(save_path, show_img)
            print('[{:04d}/{:04d}] saved {}'.format(idx + 1, file_length, save_path))

if __name__ == '__main__':
    main()