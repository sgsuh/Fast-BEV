"""
Create: 2023.08.21
Author: SG.SUH
Python: 3.8.8
PyTorch: 1.10.0
"""

import torch

from . import emc_ext

def emc(vt_output, tf_vec, bev_feat):
    vt_output = vt_output.contiguous()
    tf_vec = tf_vec.contiguous()
    bev_feat = bev_feat.contiguous()

    emc_ext.emc(vt_output, tf_vec, bev_feat)

    return bev_feat

class TRTEmc(torch.autograd.Function):
    @staticmethod
    def symbolic(g, vt_output, tf_vec):
        return g.op('mmdeploy::emc', vt_output, tf_vec)
    
    @staticmethod
    def forward(g, vt_output, tf_vec):
        bev_feat = torch.zeros((1, 4 * 64, 200, 200, 4), dtype = torch.float32).cuda()
        bev_feat = emc(vt_output, tf_vec, bev_feat)

        return bev_feat