#include "trt_project_2d_to_3d_kernel.h"

#define BLOCK_SIZE 1024

__global__ void buildLutKernel(int n_x_voxels, int n_y_voxels, int n_z_voxels, float* voxel_size, float* origin, float* projection, int* lut, int n_images, int height, int width) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int zi  = idx % n_z_voxels;

	idx /= n_z_voxels;

	int yi = idx % n_y_voxels;

	idx /= n_y_voxels;

	int xi = idx % n_x_voxels;

	idx /= n_x_voxels;

	int img = idx;

	if(img < n_images && lut[(xi * n_y_voxels + yi) * n_z_voxels + zi] == -1) {
		float size_x = voxel_size[0];
		float size_y = voxel_size[1];
		float size_z = voxel_size[2];

		float ar[3];
		float pt[3];

		pt[0] = (xi - n_x_voxels / 2.0f) * size_x + origin[0];
		pt[1] = (yi - n_y_voxels / 2.0f) * size_y + origin[1];
		pt[2] = (zi - n_z_voxels / 2.0f) * size_z + origin[2];

		for(int i = 0; i < 3; ++i) {
			ar[i] = 0;

			for(int j = 0; j < 3; ++j) {
				ar[i] += projection[(img * 3 + i) * 4 + j] * pt[j];
			}

			ar[i] += projection[((img * 3) + i) * 4 + 3];
		}

		int x = round(ar[0] / ar[2]);
		int y = round(ar[1] / ar[2]);

		float z = ar[2];

		bool fit_in = (x >= 0) && (y >= 0) && (x < width) && (y < height) && (z > 0);
		int  target;

		if(fit_in) {
			target = (img * height + y) * width + x;

			int offset = (xi * n_y_voxels + yi) * n_z_voxels + zi;

			lut[offset] = target;
		} else {
			target = -1;
		}
	}
}

void buildLutGpu(std::vector<int> n_voxels, float* voxel_size_dev, float* origin_dev, float* projection, int* lut, int n_images, int height, int width, cudaStream_t stream) {
	int n_x_voxels = int(n_voxels[0]);
	int n_y_voxels = int(n_voxels[1]);
	int n_z_voxels = int(n_voxels[2]);

	size_t total_nrof_voxels = n_images * n_voxels[0] * n_voxels[1] * n_voxels[2];

	dim3 thread_per_block(BLOCK_SIZE);
	dim3 block_per_grid((total_nrof_voxels + thread_per_block.x - 1) / thread_per_block.x);

	buildLutKernel<<<block_per_grid, thread_per_block, 0, stream>>>(n_x_voxels, n_y_voxels, n_z_voxels, voxel_size_dev, origin_dev, projection, lut, n_images, height, width);
}

__global__ void backprojectLutKernel(float* features, int* lut, float* volume, size_t total_nrof_voxels, int n_channels, int n_images, int height, int width) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;

	if(offset < total_nrof_voxels) {
		int target = lut[offset];

		int img_idx   = target / (height * width);
		int coord_idx = target % (height * width);

		if(target >= 0) {
			for(int c = 0; c < n_channels; ++c) {
				int src_idx = img_idx * (n_channels * height * width) + c * height * width + coord_idx;
				int dst_idx = c * total_nrof_voxels + offset;

				volume[dst_idx] = features[src_idx];
			}
		}
	}
}

void backprojectLutGpu(float* features_bev, int* lut_dev, float* volume_dev, int n_images, int n_channels, std::vector<int> n_voxels, int height, int width, cudaStream_t stream) {
	size_t total_nrof_voxels = n_voxels[0] * n_voxels[1] * n_voxels[2];

	dim3 thread_per_block(BLOCK_SIZE);
	dim3 block_per_grid((total_nrof_voxels + thread_per_block.x - 1) / thread_per_block.x);

	backprojectLutKernel<<<block_per_grid, thread_per_block, 0, stream>>>(features_bev, lut_dev, volume_dev, total_nrof_voxels, n_channels, n_images, height, width);
}

__global__ void setVolumeZeroKernel(int n_size, float* __restrict__ out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= n_size) {
		return;
	}

	float* cur_out = out + idx;

	*cur_out = 0.0;
}

void setVolumeZero(int n_size, float* out, cudaStream_t stream) {
	dim3 thread_per_block(BLOCK_SIZE);
	dim3 block_per_grid((n_size + thread_per_block.x - 1) / thread_per_block.x);

	setVolumeZeroKernel<<<block_per_grid, thread_per_block, 0, stream>>>(n_size, out);
}
