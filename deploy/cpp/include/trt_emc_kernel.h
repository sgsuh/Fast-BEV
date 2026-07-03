#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>

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

void transformCrop(const ReMapMeta meta, const float* tf_dst2src_dev, const float* padded_feature_dev, float* cropped_feature_dev);
void simpleCrop(const ReMapMeta meta, const float* padded_feature_dev, float* cropped_feature_dev);
