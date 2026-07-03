#pragma once

#include <map>
#include <vector>

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "preprc_utils.h"

typedef std::map<std::string, std::array<float*, 2>> MappingGPU;

class Preprocessor {
   public:
	Preprocessor(YAML::Node& param_node, YAML::Node& calib_node);
	~Preprocessor();

	void preprcMain(const std::vector<cv::Mat>& host_src, float* dst, std::vector<cudaStream_t> stream);

   private:
	PreprcMeta config_;

	// GPU memory
	unsigned char* dev_raw_img_;
	float*         dev_out_img_;
	MappingGPU     dev_mapping_;

	std::vector<std::string> cam_list_;
};
