#include "trt_emc_kernel.h"

__global__ void transformCropKernelNearest(const ReMapMeta meta, const float* tf_dst2src, const float* src_feat, float* dst_feat) {
	const int xi = blockIdx.x * blockDim.x + threadIdx.x;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y;
	const int zi = blockIdx.z * blockDim.z + threadIdx.z;

	if((xi < meta.n_vx_) && (xi >= 0) && (yi < meta.n_vy_) && (yi >= 0) && (zi < meta.n_vz_) && (zi >= 0)) {
		// Get voxel center position in dst space
		float vc[3];

		vc[0] = (xi - meta.n_vx_ / 2.0f) * meta.vs_x_ + meta.origin_x_;
		vc[1] = (yi - meta.n_vy_ / 2.0f) * meta.vs_y_ + meta.origin_y_;
		vc[2] = (zi - meta.n_vz_ / 2.0f) * meta.vs_z_ + meta.origin_z_;

		// Convert to src space
		float src_vc[3] = {0.0f, 0.0f, 0.0f};

		for(unsigned int i = 0; i < 3; ++i) {
			for(unsigned int j = 0; j < 3; ++j) {
				src_vc[i] += tf_dst2src[i * 4 + j] * vc[j];
			}

			src_vc[i] += tf_dst2src[i * 4 + 3];
		}

		// Convert to index of src
		float xi_src = (src_vc[0] - meta.origin_x_) / meta.vs_x_ + (meta.n_vx_ / 2.0f);
		float yi_src = (src_vc[1] - meta.origin_y_) / meta.vs_y_ + (meta.n_vy_ / 2.0f);
		float zi_src = (src_vc[2] - meta.origin_z_) / meta.vs_z_ + (meta.n_vz_ / 2.0f);

		const int padded_xi_src = static_cast<int>(xi_src + meta.m_vx_ + 0.5f);
		const int padded_yi_src = static_cast<int>(yi_src + meta.m_vy_ + 0.5f);
		const int padded_zi_src = static_cast<int>(zi_src + 0.5f);

		// Remap to padded area
		const int padded_n_vx = meta.n_vx_ + 2 * meta.m_vx_;
		const int padded_n_vy = meta.n_vy_ + 2 * meta.m_vy_;

		if((padded_xi_src < padded_n_vx) && (padded_xi_src >= 0) && (padded_yi_src < padded_n_vy) && (padded_yi_src >= 0) && (padded_zi_src < meta.n_vz_) && (padded_zi_src >= 0)) {
			for(unsigned int ci = 0; ci < meta.n_ch_; ++ci) {
				const int dst_idx = ((ci * meta.n_vx_ + xi) * meta.n_vy_ + yi) * meta.n_vz_ + zi;
				const int src_idx = ((ci * padded_n_vx + padded_xi_src) * padded_n_vy + padded_yi_src) * meta.n_vz_ + padded_zi_src;

				dst_feat[dst_idx] = src_feat[src_idx];
			}
		}
	}
}

void transformCrop(const ReMapMeta meta, const float* tf_dst2src_dev, const float* padded_feature_dev, float* cropped_feature_dev) {
	dim3 nthread_per_blocks(16, 16, 4);
	dim3 nblocks_per_grid(std::ceil(static_cast<float>(meta.n_vx_) / nthread_per_blocks.x), std::ceil(static_cast<float>(meta.n_vy_) / nthread_per_blocks.y), std::ceil(static_cast<float>(meta.n_vz_) / nthread_per_blocks.z));

	transformCropKernelNearest<<<nblocks_per_grid, nthread_per_blocks>>>(meta, tf_dst2src_dev, padded_feature_dev, cropped_feature_dev);

	return;
}

__global__ void simpleCropKernel(const ReMapMeta meta, const float* src_feat, float* dst_feat) {
	const int xi = blockIdx.x * blockDim.x + threadIdx.x;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y;
	const int zi = blockIdx.z * blockDim.z + threadIdx.z;

	if((xi < meta.n_vx_) && (xi >= 0) && (yi < meta.n_vy_) && (yi >= 0) && (zi < meta.n_vz_) && (zi >= 0)) {
		const int padded_xi   = meta.m_vx_ + xi;
		const int padded_yi   = meta.m_vy_ + yi;
		const int padded_n_vx = meta.n_vx_ + 2 * meta.m_vx_;
		const int padded_n_vy = meta.n_vy_ + 2 * meta.m_vy_;

		for(unsigned int ci = 0; ci < meta.n_ch_; ++ci) {
			const int dst_idx = ((ci * meta.n_vx_ + xi) * meta.n_vy_ + yi) * meta.n_vz_ + zi;
			const int src_idx = ((ci * padded_n_vx + padded_xi) * padded_n_vy + padded_yi) * meta.n_vz_ + zi;

			dst_feat[dst_idx] = src_feat[src_idx];
		}
	}
}

void simpleCrop(const ReMapMeta meta, const float* padded_feature_dev, float* cropped_feature_dev) {
	dim3 nthread_per_blocks(16, 16, 4);
	dim3 nblocks_per_grid(std::ceil(static_cast<float>(meta.n_vx_) / nthread_per_blocks.x), std::ceil(static_cast<float>(meta.n_vy_) / nthread_per_blocks.y), std::ceil(static_cast<float>(meta.n_vz_) / nthread_per_blocks.z));

	simpleCropKernel<<<nblocks_per_grid, nthread_per_blocks>>>(meta, padded_feature_dev, cropped_feature_dev);

	return;
}
