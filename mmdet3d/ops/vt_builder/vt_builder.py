"""
Create: 2023.08.03
Author: SG.SUH
Python: 3.8.8
PyTorch: 1.10.0
"""

import torch

from . import vt_builder_ext

def vt_builder(feature, projection, volume):
    feature = feature.contiguous()
    projection = projection.contiguous()
    volume = volume.contiguous()

    vt_builder_ext.vt_builder(feature, projection, volume)

    return volume

class TRTVtBuilder(torch.autograd.Function):
    @staticmethod
    def forward(g, feature, n_voxels, projection):
        
        volume = torch.zeros((feature.shape[1], int(200), int(200), int(n_voxels[2])), dtype = torch.float32).cuda()
        volume = vt_builder(feature, projection, volume)

        return volume