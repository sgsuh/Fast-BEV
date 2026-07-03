#pragma once

#include <cuda_runtime_api.h>
#include <vector>

void buildLutGpu(std::vector<int> n_voxels, float* voxel_size_dev, float* origin_dev, float* projection, int* lut, int n_images, int height, int width, cudaStream_t stream);
void backprojectLutGpu(float* features_bev, int* lut_dev, float* volume_dev, int n_images, int n_channels, std::vector<int> n_voxels, int height, int width, cudaStream_t stream);
void setVolumeZero(int n_size, float* out, cudaStream_t stream);
