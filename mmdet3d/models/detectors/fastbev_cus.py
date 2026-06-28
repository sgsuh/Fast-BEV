"""
Create: 2023.05.11
Author: SG.SUH
Python: 3.8.8
PyTorch: 1.10.0
"""

import torch
import math
import numpy as np

from torch import Tensor

from mmcv.runner import auto_fp16

from mmdet.models import DETECTORS

from mmseg.ops import resize

from mmdet3d.models.detectors.fastbev import FastBEV
from mmdet3d.models.detectors.fastbev import get_points
from mmdet3d.models.detectors.fastbev import backproject_inplace
from mmdet3d.ops.project_2d_to_3d     import project_2d_to_3d
from mmdet3d.ops.project_2d_to_3d     import TRTViewTransformer
from mmdet3d.ops.vt_builder           import TRTVtBuilder
from mmdet3d.ops.emc                  import TRTEmc



@DETECTORS.register_module()
class FastBEVCus(FastBEV):
    def __init__(self, backbone, neck, neck_fuse, neck_3d, bbox_head, seg_head, n_voxels, voxel_size, bbox_head_2d = None, train_cfg = None, test_cfg = None, train_cfg_2d = None, test_cfg_2d = None, pretrained = None,
                 init_cfg = None, extrinsic_noise = 0, seq_detach = False, multi_scale_id = None, multi_scale_3d_scaler = None, with_cp = False, backproject = 'inplace', style = 'v4',):
        super().__init__(backbone = backbone, 
                         neck = neck, 
                         neck_fuse = neck_fuse, 
                         neck_3d = neck_3d, 
                         bbox_head = bbox_head, 
                         seg_head = seg_head, 
                         n_voxels = n_voxels, 
                         voxel_size = voxel_size, 
                         bbox_head_2d = bbox_head_2d, 
                         train_cfg = train_cfg, 
                         test_cfg = test_cfg, 
                         train_cfg_2d = train_cfg_2d, 
                         test_cfg_2d = test_cfg_2d, 
                         pretrained = pretrained, 
                         init_cfg = init_cfg, 
                         extrinsic_noise = extrinsic_noise, 
                         seq_detach = seq_detach, 
                         multi_scale_id = multi_scale_id, 
                         multi_scale_3d_scaler = multi_scale_3d_scaler, 
                         with_cp = with_cp, 
                         backproject = backproject, 
                         style = style)
        
        self.lidar2img_origin = torch.from_numpy(np.array([0, 0, -1], dtype = np.float32)).cuda()
        self.lidar2img_intrinsic = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], dtype = np.float32)
        self.img_shape = (256, 704, 3)
        

        self.bev_size = [200, 200]
        self.num_points_in_pillar = 4
        self.neck_3d_single_input = int(self.neck_3d.fuse.in_channels // self.num_points_in_pillar)

        

        
        self.anchors = self.bbox_head.get_anchors_trt()

        self.projection = None

        
        self.volume = torch.zeros((4, 64, 450, 450, 4), dtype = torch.float32).cuda()

        

        self.n_voxels_cuda = torch.from_numpy(np.array(self.n_voxels)).int().cuda().squeeze(0)
        self.voxel_size_cuda = torch.from_numpy(np.array(self.voxel_size)).float().cuda()

    
    def compute_projection(self, lidar2img_extrinsic, stride, noise = 0):
        projection = []
        
        intrinsic = torch.tensor(self.lidar2img_intrinsic[:3, :3]).to(lidar2img_extrinsic.device)
        
        intrinsic[:2] /= stride
        
        extrinsics = map(torch.tensor, lidar2img_extrinsic)

        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
                

        projection = torch.stack(projection)

        

        return projection

    def extract_feat(self, img):
        
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.backbone(img)

        

        def _inner_forward(x):
            out = self.neck(x)

            return out
        
        mlvl_feats = _inner_forward(x)
        mlvl_feats = list(mlvl_feats)

        mlvl_feats_ = []

        for msid in self.multi_scale_id:
            fuse_feats = [mlvl_feats[msid]]

            for i in range(msid + 1, len(mlvl_feats)):
                resized_feat = resize(mlvl_feats[i], size = mlvl_feats[msid].size()[2:], mode = 'bilinear', align_corners = False)
                fuse_feats.append(resized_feat)

            fuse_feats = torch.cat(fuse_feats, dim = 1)
            fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)

            mlvl_feats_.append(fuse_feats)

        mlvl_feats = mlvl_feats_

        return mlvl_feats
        
    
    
    def raw_project(self, mlvl_feats, H, W, projection = None):
        mlvl_volumes = []

        for lvl, mlvl_feat in enumerate(mlvl_feats):
            
            stride_i = math.ceil(W / mlvl_feat.shape[-1])

            
            mlvl_feat = mlvl_feat.reshape([1, -1] + list(mlvl_feat.shape[1:]))
            mlvl_feat_split = torch.split(mlvl_feat, 6, dim = 1)

            volume_list = []

            for seq_id in range(len(mlvl_feat_split)):
                volumes = []

                feat_i = mlvl_feat_split[seq_id][0]

                height = math.ceil(self.img_shape[0] / stride_i)
                width = math.ceil(self.img_shape[1] / stride_i)

                
                projection = self.projection.to(feat_i.device)

                n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]

                points = get_points(n_voxels = torch.tensor(n_voxels), voxel_size = torch.tensor(voxel_size), origin = torch.tensor(self.lidar2img_origin)).to(feat_i.device)

                volume = backproject_inplace(feat_i[:, :, :height, :width], points, projection)

                volumes.append(volume)

                volume_list.append(torch.stack(volumes))

            mlvl_volumes.append(torch.cat(volume_list, dim = 1))

        mlvl_volumes = torch.cat(mlvl_volumes, dim = 1)

        return mlvl_volumes

        

    def simple_test(self, img, lidar2img_extrinsic, tf_vec):
    

        mlvl_feats = self.extract_feat(img)

        

        if self.projection is None:
            self.projection = self.compute_projection(lidar2img_extrinsic, math.ceil(img.shape[-1] / mlvl_feats[0].shape[-1])).cuda()

        
        mlvl_feat_split = mlvl_feats[0]
        


        
        feat_i = mlvl_feat_split
        
        temp_vol = TRTViewTransformer.apply(feat_i, self.n_voxels_cuda, self.voxel_size_cuda, self.lidar2img_origin, self.projection)
        

        last_vol = self.volume[:3, ...]
        last_vol = torch.cat((temp_vol.unsqueeze(0), last_vol), dim = 0)
        self.volume = last_vol

        

        bev_feat = TRTEmc.apply(self.volume, tf_vec)

        


        
        feature_bev = self.neck_3d(bev_feat)

        x = self.bbox_head(feature_bev)

        x[0][0] = x[0][0].sigmoid()

        

        return x

    def forward_test(self, img, lidar2img_extrinsic, tf_vec):
        return self.simple_test(img, lidar2img_extrinsic, tf_vec)
    
    

    @auto_fp16(apply_to = ('img', ))
    def forward(self, img, lidar2img_extrinsic, tf_vec):
        return self.forward_test(img, lidar2img_extrinsic, tf_vec)
    
    
    def set_cfg(self, cfg, device = 'cuda', dtype = torch.float32):
        for k, v in cfg.items():
            setattr(self, k, v)

        if self.lidar2cam is not None and not isinstance(self.lidar2cam, Tensor):
            lidar2cam = np.array(self.lidar2cam)
            self.lidar2cam = torch.from_numpy(lidar2cam).to(device).to(dtype)

        if self.cam2img is not None and not isinstance(self.cam2img, Tensor):
            cam2img = np.array(self.cam2img)
            self.cam2img = torch.from_numpy(cam2img).to(device).to(dtype)

        if self.params is not None:
            param_list = []

            for key, value_list in self.params.items():
                param_list += value_list

            self.projection = torch.tensor(self.params['projection']).reshape((-1, 4, 4))
            self.param = torch.nn.Parameter(torch.tensor(param_list))

    def trt_decoder(self, cls_score, bbox_pred, dir_cls_pred):
        bbox_pred = bbox_pred[0][0]
        dir_cls_pred = dir_cls_pred[0][0]

        if self.bbox_head.use_sigmoid_cls:
            cls_score = [score.sigmoid() for score in cls_score]
        else:
            cls_score = [score.softmax(-1) for score in cls_score]

        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.bbox_head.box_code_size)

        bboxes = self.bbox_head.decode(bbox_pred)

        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)

        return cls_score, [bboxes], [dir_cls_pred]