"""
Create: 2023.05.18
Author: SG.SUH
Python: 3.8.8
PyTorch: 1.10.0
"""

import torch

from . import project_2d_to_3d_ext


def project_2d_to_3d(feature, n_voxels, voxel_size, origin, projection, volume):

    feature = feature.contiguous()
    n_voxels = n_voxels.contiguous()
    voxel_size = voxel_size.contiguous()
    origin = origin.contiguous()
    projection = projection.contiguous()
    

    volume = volume.contiguous()


    project_2d_to_3d_ext.project_2d_to_3d(feature, n_voxels, voxel_size, origin, projection, volume)


    return volume


class TRTViewTransformer(torch.autograd.Function):
    @staticmethod
    def symbolic(g, feature, n_voxels, voxel_size, origin, projection):
        return g.op('mmdeploy::project_2d_to_3d', feature, n_voxels, voxel_size, origin, projection)
    
    @staticmethod
    def forward(g, feature, n_voxels, voxel_size, origin, projection):

        volume = torch.zeros((feature.shape[1], int(450), int(450), int(n_voxels[2])), dtype = torch.float32).cuda()
        


        volume = project_2d_to_3d(feature, n_voxels, voxel_size, origin, projection, volume)

        return volume