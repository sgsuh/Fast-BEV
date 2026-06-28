"""
Create: 2023.05.15
Author: SG.SUH
Python: 3.8.8
PyTorch: 1.10.0
"""

import torch
import numpy as np

from mmdet.models import HEADS

from mmdet3d.models.dense_heads.free_anchor3d_head import FreeAnchor3DHead
from mmdet3d.core                                  import box3d_multiclass_scale_nms
from mmdet3d.core                                  import limit_period
from mmdet3d.core.bbox.structures.lidar_box3d      import LiDARInstance3DBoxes

@HEADS.register_module()
class FreeAnchor3DHeadCus(FreeAnchor3DHead):
    def __init__(self, pre_anchor_topk = 50, bbox_thr = 0.6, gamma = 2.0, alpha = 0.5, init_cfg = None, pos_loss_weight = 1, neg_loss_weight = 1, **kwargs):
        super().__init__(pre_anchor_topk = pre_anchor_topk, 
                         bbox_thr = bbox_thr, 
                         gamma = gamma, 
                         alpha = alpha, 
                         init_cfg = init_cfg, 
                         pos_loss_weight = pos_loss_weight, 
                         neg_loss_weight = neg_loss_weight, 
                         **kwargs)
        
    
    def get_bboxes_single(self, cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors, cfg = None, rescale = False):
        cfg = self.test_cfg if cfg is None else cfg

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []

        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim = -1)[1]

            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)

            scores = cls_score.sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)

            if scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim = 1)

                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim = 1)

        score_thr = cfg.get('score_thr', 0)

        
        mlvl_bboxes_for_nms = LiDARInstance3DBoxes(mlvl_bboxes, box_dim = self.box_code_size).bev
        results = box3d_multiclass_scale_nms(mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores, score_thr, cfg.max_num, cfg, mlvl_dir_scores)

        bboxes, scores, labels, dir_scores = results

        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset, self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (dir_rot + self.dir_offset + np.pi * dir_scores.to(bboxes.dtype))

        
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim = self.box_code_size)

        return bboxes, scores, labels

    
    def get_bboxes(self, cls_scores, bbox_preds, dir_cls_preds, valid = None, cfg = None, rescale = False):
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device = device)
        mlvl_anchors = [anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors]

        result_list = []

        
        for img_id in range(len(cls_scores[0])):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            dir_cls_pred_list = [dir_cls_preds[i][img_id].detach() for i in range(num_levels)]

            
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, dir_cls_pred_list, mlvl_anchors, cfg, rescale)
            result_list.append(proposals)

        return result_list
    
    def get_anchors_trt(self, device = 'cuda'):
        featmap_sizes = [[100, 100]]
        anchor_list = self.anchor_generator.grid_anchors(featmap_sizes, device = device)
        self.anchors = torch.cat(anchor_list)

    def decode(self, deltas):
        anchors = self.anchors
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