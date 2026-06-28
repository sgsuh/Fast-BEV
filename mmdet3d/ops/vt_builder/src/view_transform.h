#pragma once

#include <cstdio>

struct FeatureMeta {
int n_img_;
int height_;
int width_;
int n_ch_;
int stride_;

int n_vx_;
int n_vy_;
int n_vz_;

float vs_x_;
float vs_y_;
float vs_z_;

float origin_x_;
float origin_y_;
float origin_z_;
};

#define GPU_CHECK(ans)                                                                                                 \
{                                                                                                                      \
	GPUAssert((ans), __FILE__, __LINE__);                                                                              \
}
inline void GPUAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
if (code != cudaSuccess)
{
	fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort)
	exit(code);
}
}

void buildLUT(const FeatureMeta meta, const float* lidar2img, int* lut_img_idx, int* lut_coord_idx);
void backproject(const FeatureMeta meta, const float* img_features, float* volume_features, int* lut_img_idx, int* lut_coord_idx);