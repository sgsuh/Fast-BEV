#include <cuda_runtime_api.h>

#include "view_transform.h"

#define BLOCK_SIZE  1024

__global__ void buildLUTKernel(const FeatureMeta meta, const float* lidar2img, int* lut_img_idx, int* lut_coord_idx, int* lut_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int zi = idx % meta.n_vz_;

    idx /= meta.n_vz_;

    int yi = idx % meta.n_vy_;

    idx /= meta.n_vy_;

    int xi = idx % meta.n_vx_;

    idx /= meta.n_vx_;

    int img = idx;
    int voxel_idx = (xi * meta.n_vy_ + yi) * meta.n_vz_ + zi;

    if(img < meta.n_img_) {
        // Get voxel center position
        float vc[3];

        vc[0] = (xi - meta.n_vx_ / 2.0f) * meta.vs_x_ + meta.origin_x_;
        vc[1] = (yi - meta.n_vy_ / 2.0f) * meta.vs_y_ + meta.origin_y_;
        vc[2] = (zi - meta.n_vz_ / 2.0f) * meta.vs_z_ + meta.origin_z_;

        // Convert to image coordinate, whose index is specified by img
        float ar[3] = {0.0f, 0.0f, 0.0f};

        for(unsigned int i = 0; i < 3; ++i) {
            for(unsigned int j = 0; j < 3; ++j) {
                ar[i] += lidar2img[(img * 3 + i) * 4 + j] * vc[j];
            }

            ar[i] += lidar2img[(img * 3 + i) * 4 + 3];
        }

        int projected_x = round(ar[0] / ar[2]);
        int projected_y = round(ar[1] / ar[2]);

        float z = ar[2];

        // Check the correspondence
        bool fit_in = (projected_x >= 0) && (projected_y >= 0) && (projected_x <= meta.width_) && (projected_y <= meta.height_) && (z > 0);

        int mapped_img_feat_idx = -1;

        

        

        if(fit_in) {
            // Set to visited only if valid mapping found
            int prev_counter = atomicAdd(&lut_counter[voxel_idx], 1);

            // Modify corresponding lut_coord_idx, lut_img_idx only if not visited
            if(prev_counter == 0) {
                int projected_feat_x = round((ar[0] / ar[2]) / meta.stride_);
                int projected_feat_y = round((ar[1] / ar[2]) / meta.stride_);

                mapped_img_feat_idx = projected_feat_y * (meta.width_ / meta.stride_) + projected_feat_x;

                lut_coord_idx[voxel_idx] = mapped_img_feat_idx;
                lut_img_idx[voxel_idx] = img;

                
            }

            
        }
    }
}



void buildLUT(const FeatureMeta meta, const float* lidar2img, int* lut_img_idx, int* lut_coord_idx) {
    size_t nrof_voxel = meta.n_vx_ * meta.n_vy_ * meta.n_vz_;
    dim3 thread_per_block(BLOCK_SIZE);
    dim3 block_per_grid((meta.n_img_ * nrof_voxel + thread_per_block.x - 1) / thread_per_block.x);

    

    // Clear LUT value
    GPU_CHECK(cudaMemset((void*)(lut_img_idx), -1, sizeof(int) * nrof_voxel));
    GPU_CHECK(cudaMemset((void*)(lut_coord_idx), -1, sizeof(int) * nrof_voxel));

    // Temporary counter that acts as flag
    int* lut_counter_dev;

    GPU_CHECK(cudaMalloc((void**)&lut_counter_dev, sizeof(int) * nrof_voxel));
    GPU_CHECK(cudaMemset((void*)lut_counter_dev, 0, sizeof(int) * nrof_voxel));    

    

    // Build look-up table
    buildLUTKernel<<<block_per_grid, thread_per_block>>>(meta, lidar2img, lut_img_idx, lut_coord_idx, lut_counter_dev);    

    return;
}

__global__ void backprojectKernel(const FeatureMeta meta, const float* img_features, float* volume_features, const int* lut_img_idx, const int* lut_coord_idx) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nrof_voxel = meta.n_vx_ * meta.n_vy_ * meta.n_vz_;

    if(voxel_idx < nrof_voxel) {
        int img_idx = lut_img_idx[voxel_idx];
        int coord_idx = lut_coord_idx[voxel_idx];

        if(coord_idx >= 0) {
            // Copy non-contiguous data in channel dimension
            for(unsigned int c = 0; c < meta.n_ch_; ++c) {
                int src_idx = (img_idx * meta.n_ch_ + c) * (meta.height_ / meta.stride_) * (meta.width_ / meta.stride_) + coord_idx;
                int dst_idx = c * nrof_voxel + voxel_idx;

                volume_features[dst_idx] = img_features[src_idx];
            }
        }
    }
}

void backproject(const FeatureMeta meta, const float* img_features, float* volume_features, int* lut_img_idx, int* lut_coord_idx) {
    size_t nrof_voxel = meta.n_vx_ * meta.n_vy_ * meta.n_vz_;
    dim3 thread_per_block(BLOCK_SIZE);
    dim3 block_per_grid((nrof_voxel + thread_per_block.x - 1) / thread_per_block.x);

    // Clear volume features
    GPU_CHECK(cudaMemset((void*)(volume_features), 0.0f, meta.n_ch_ * nrof_voxel * sizeof(float)));

    // Backproject main block
    backprojectKernel<<<block_per_grid, thread_per_block>>>(meta, img_features, (volume_features), (lut_img_idx), (lut_coord_idx));

    return;   
}