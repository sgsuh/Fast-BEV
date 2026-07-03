#include <cstdio>
#include <cuda_runtime_api.h>

#include "iou3d.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

const int              threads_per_block_nms = sizeof(unsigned long long) * 8;
__device__ const float eps                   = 1e-8;

__device__ inline float iouNormal(float const* const a, float const* const b) {
	float left    = fmaxf(a[0], b[0]);
	float right   = fminf(a[2], b[2]);
	float top     = fmaxf(a[1], b[1]);
	float bottom  = fminf(a[3], b[3]);
	float width   = fmaxf(right - left, 0.0f);
	float height  = fmaxf(bottom - top, 0.0f);
	float inter_s = width * height;
	float s_a     = (a[2] - a[0]) * (a[3] - a[1]);
	float s_b     = (b[2] - b[0]) * (b[3] - b[1]);

	return inter_s / fmaxf(s_a + s_b - inter_s, eps);
}

__global__ void nmsNormalKernel(const int boxes_num, const float nms_overlap_thresh, const float* boxes, unsigned long long* mask) {
	const int row_start = blockIdx.y;
	const int col_start = blockIdx.x;

	const int row_size = fminf(boxes_num - row_start * threads_per_block_nms, threads_per_block_nms);
	const int col_size = fminf(boxes_num - col_start * threads_per_block_nms, threads_per_block_nms);

	__shared__ float block_boxes[threads_per_block_nms * 5];

	if(threadIdx.x < col_size) {
		block_boxes[threadIdx.x * 5 + 0] = boxes[(threads_per_block_nms * col_start + threadIdx.x) * 5 + 0];
		block_boxes[threadIdx.x * 5 + 1] = boxes[(threads_per_block_nms * col_start + threadIdx.x) * 5 + 1];
		block_boxes[threadIdx.x * 5 + 2] = boxes[(threads_per_block_nms * col_start + threadIdx.x) * 5 + 2];
		block_boxes[threadIdx.x * 5 + 3] = boxes[(threads_per_block_nms * col_start + threadIdx.x) * 5 + 3];
		block_boxes[threadIdx.x * 5 + 4] = boxes[(threads_per_block_nms * col_start + threadIdx.x) * 5 + 4];
	}

	__syncthreads();

	if(threadIdx.x < row_size) {
		const int    cur_box_idx = threads_per_block_nms * row_start + threadIdx.x;
		const float* cur_box     = boxes + cur_box_idx * 5;

		int                i     = 0;
		unsigned long long t     = 0;
		int                start = 0;

		if(row_start == col_start) {
			start = threadIdx.x + 1;
		}

		for(int i = start; i < col_size; ++i) {
			if(iouNormal(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
				t |= 1ULL << i;
			}
		}

		const int col_blocks = DIVUP(boxes_num, threads_per_block_nms);

		mask[cur_box_idx * col_blocks + col_start] = t;
	}
}

void nmsNormalLauncher(const float* boxes, unsigned long long* mask, int boxes_num, float nms_overlap_thresh) {
	dim3 blocks(DIVUP(boxes_num, threads_per_block_nms), DIVUP(boxes_num, threads_per_block_nms));
	dim3 threads(threads_per_block_nms);

	nmsNormalKernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes, mask);
}

int nmsNormalGpu(std::vector<float>& boxes, std::vector<unsigned long long>& keep, float nms_overlap_thresh) {
	int boxes_num = boxes.size() / 5;

	float* boxes_data = nullptr;

	unsigned long long* keep_data = keep.data();

	const int col_blocks = DIVUP(boxes_num, threads_per_block_nms);

	cudaMalloc((void**)&boxes_data, boxes.size() * sizeof(float));
	cudaMemcpy(boxes_data, boxes.data(), boxes.size() * sizeof(float), cudaMemcpyHostToDevice);

	unsigned long long* mask_data = nullptr;

	cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long));

	nmsNormalLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

	std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

	cudaMemcpy(mask_cpu.data(), mask_data, boxes_num * col_blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	cudaFree(mask_data);
	cudaFree(boxes_data);

	std::vector<unsigned long long> remv_cpu(col_blocks);

	int num_to_keep = 0;

	for(int i = 0; i < boxes_num; ++i) {
		int nblock  = i / threads_per_block_nms;
		int inblock = i % threads_per_block_nms;

		if(!(remv_cpu[nblock] & (1ULL << inblock))) {
			keep_data[num_to_keep++] = i;

			unsigned long long* p = &mask_cpu[0] + i * col_blocks;

			for(int j = nblock; j < col_blocks; ++j) {
				remv_cpu[j] |= p[j];
			}
		}
	}

	if(cudaSuccess != cudaGetLastError()) {
		printf("Error!\n");
	}

	return num_to_keep;
}
