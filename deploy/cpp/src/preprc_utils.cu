#include <cstdio>
#include <iostream>

#include "preprc_utils.h"

__global__ void preprcKernelNearest(const PreprcMeta meta, const unsigned char* src, const float* map_x, const float* map_y, float* dst) {
	const int h = blockIdx.x * blockDim.x + threadIdx.x;
	const int w = blockIdx.y * blockDim.y + threadIdx.y;

	if((h < meta.img_h_) && (h >= 0) && (w < meta.img_w_) && (w >= 0)) {
		// Remap by nearest neighborhood
		const int mapped_x = static_cast<int>(map_x[h * meta.img_w_ + w] + 0.5f);
		const int mapped_y = static_cast<int>(map_y[h * meta.img_w_ + w] + 0.5f);

		const unsigned int dst_idx_b = h * meta.img_w_ + w;
		const unsigned int dst_idx_g = 1 * (meta.img_h_ * meta.img_w_) + dst_idx_b;
		const unsigned int dst_idx_r = 2 * (meta.img_h_ * meta.img_w_) + dst_idx_b;

		if((mapped_y < meta.raw_h_) && (mapped_y >= 0) && (mapped_x < meta.raw_w_) && (mapped_x >= 0)) {
			const unsigned int src_idx = (mapped_y * meta.raw_w_ + mapped_x) * 3;

			// Normalization
			dst[dst_idx_b] = (static_cast<float>(src[src_idx]) - meta.mean_b_) * meta.i_std_b_;
			dst[dst_idx_g] = (static_cast<float>(src[src_idx + 1]) - meta.mean_g_) * meta.i_std_g_;
			dst[dst_idx_r] = (static_cast<float>(src[src_idx + 2]) - meta.mean_r_) * meta.i_std_r_;
		} else {
			// Fill with zeros if there is no mapped pixel
			dst[dst_idx_b] = 0.0f;
			dst[dst_idx_g] = 0.0f;
			dst[dst_idx_r] = 0.0f;
		}
	}
}

__global__ void preprcKernelBilinear(const PreprcMeta meta, const u_char* src, const float* map_x, const float* map_y, float* dst) {
	const int h = blockIdx.x * blockDim.x + threadIdx.x;
	const int w = blockIdx.y * blockDim.y + threadIdx.y;

	if((h < meta.img_h_) && (h >= 0) && (w < meta.img_w_) && (w >= 0)) {
		// Remap by nearest neighborhood
		const float mapped_x = map_x[h * meta.img_w_ + w];
		const float mapped_y = map_y[h * meta.img_w_ + w];

		const int          x1        = static_cast<int>(mapped_x);
		const int          x2        = x1 + 1;
		const int          y1        = static_cast<int>(mapped_y);
		const int          y2        = y1 + 1;
		const unsigned int dst_idx_b = h * meta.img_w_ + w;
		const unsigned int dst_idx_g = 1 * (meta.img_h_ * meta.img_w_) + dst_idx_b;
		const unsigned int dst_idx_r = 2 * (meta.img_h_ * meta.img_w_) + dst_idx_b;

		if((y2 < meta.raw_h_) && (y1 >= 0) && (x2 < meta.raw_w_) && (x1 >= 0)) {
			const unsigned int q11_src_idx = (y1 * meta.raw_w_ + x1) * 3;
			const unsigned int q21_src_idx = (y1 * meta.raw_w_ + x2) * 3;
			const unsigned int q12_src_idx = (y2 * meta.raw_w_ + x1) * 3;
			const unsigned int q22_src_idx = (y2 * meta.raw_w_ + x2) * 3;

			float f_x_y1[3];
			float f_x_y2[3];
			float f_x_y[3];

			const float x2mx1 = static_cast<float>(x2 - x1);
			const float y2my1 = static_cast<float>(y2 - y1);

			f_x_y1[0] = (x2 - mapped_x) / x2mx1 * static_cast<float>(src[q11_src_idx + 0]) + (mapped_x - x1) / x2mx1 * static_cast<float>(src[q21_src_idx + 0]);
			f_x_y1[1] = (x2 - mapped_x) / x2mx1 * static_cast<float>(src[q11_src_idx + 1]) + (mapped_x - x1) / x2mx1 * static_cast<float>(src[q21_src_idx + 1]);
			f_x_y1[2] = (x2 - mapped_x) / x2mx1 * static_cast<float>(src[q11_src_idx + 2]) + (mapped_x - x1) / x2mx1 * static_cast<float>(src[q21_src_idx + 2]);

			f_x_y2[0] = (x2 - mapped_x) / x2mx1 * static_cast<float>(src[q12_src_idx + 0]) + (mapped_x - x1) / x2mx1 * static_cast<float>(src[q22_src_idx + 0]);
			f_x_y2[1] = (x2 - mapped_x) / x2mx1 * static_cast<float>(src[q12_src_idx + 1]) + (mapped_x - x1) / x2mx1 * static_cast<float>(src[q22_src_idx + 1]);
			f_x_y2[2] = (x2 - mapped_x) / x2mx1 * static_cast<float>(src[q12_src_idx + 2]) + (mapped_x - x1) / x2mx1 * static_cast<float>(src[q22_src_idx + 2]);

			f_x_y[0] = (y2 - mapped_y) / y2my1 * f_x_y1[0] + (mapped_y - y1) / y2my1 * f_x_y2[0];
			f_x_y[1] = (y2 - mapped_y) / y2my1 * f_x_y1[1] + (mapped_y - y1) / y2my1 * f_x_y2[1];
			f_x_y[2] = (y2 - mapped_y) / y2my1 * f_x_y1[2] + (mapped_y - y1) / y2my1 * f_x_y2[2];

			// Normalization
			dst[dst_idx_b] = (static_cast<float>(f_x_y[0]) - meta.mean_b_) * meta.i_std_b_;
			dst[dst_idx_g] = (static_cast<float>(f_x_y[1]) - meta.mean_g_) * meta.i_std_g_;
			dst[dst_idx_r] = (static_cast<float>(f_x_y[2]) - meta.mean_r_) * meta.i_std_r_;

		} else {
			// Fill with zeros if there is no mapped pixel
			dst[dst_idx_b] = 0.0f;
			dst[dst_idx_g] = 0.0f;
			dst[dst_idx_r] = 0.0f;
		}
	}
}

void preprcAllAtOnce(const PreprcMeta meta, const u_char* src, const float* map_x, const float* map_y, float* dst, const IntpType intp_type, cudaStream_t stream) {
	dim3 nthread_per_blocks(32, 32);
	dim3 nblocks_per_grid(std::ceil(static_cast<float>(meta.img_h_) / nthread_per_blocks.x), std::ceil(static_cast<float>(meta.img_w_) / nthread_per_blocks.y));

	switch(intp_type) {
	case NEAREST:
		preprcKernelNearest<<<nblocks_per_grid, nthread_per_blocks, 0, stream>>>(meta, src, map_x, map_y, dst);

		break;
	case BILINEAR:
		preprcKernelBilinear<<<nblocks_per_grid, nthread_per_blocks, 0, stream>>>(meta, src, map_x, map_y, dst);

		break;
	default:
		std::cout << "Invalid interpolation type" << intp_type << std::endl;

		break;
	}

	return;
}
