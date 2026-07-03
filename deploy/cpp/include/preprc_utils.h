#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

typedef enum {
	NEAREST  = 0,
	BILINEAR = 1
} IntpType;

struct PreprcMeta {
	size_t raw_w_;
	size_t raw_h_;
	size_t img_w_;
	size_t img_h_;
	u_int  n_cam_;
	float  mean_b_;
	float  mean_g_;
	float  mean_r_;
	float  i_std_b_;
	float  i_std_g_;
	float  i_std_r_;
};

void preprcAllAtOnce(const PreprcMeta meta, const u_char* src, const float* map_x, const float* map_y, float* dst, const IntpType intp_type, cudaStream_t stream);
