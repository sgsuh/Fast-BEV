#pragma once

#include <cstdio>

struct ReMapMeta {
    int n_vx_;
    int n_vy_;
    int n_vz_;
    int m_vx_;
    int m_vy_;
    int n_ch_;

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

void transformCrop(const ReMapMeta meta, const float* tf_dst2src_dev, const float* padded_feature_dev, float* cropped_feature_dev);
void simpleCrop(const ReMapMeta meta, const float* padded_feature_dev, float* cropped_feature_dev);