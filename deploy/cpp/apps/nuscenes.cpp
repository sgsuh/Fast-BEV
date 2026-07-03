#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "qhull_cus.h"

#include "trt_fastbev.h"

void readFileLists(const std::string& dir_path, std::vector<std::string>& out_filelists, std::string type) {
	DIR* dir = opendir(dir_path.c_str());

	struct dirent* ptr;

	while((ptr = readdir(dir)) != NULL) {
		std::string file_path = ptr->d_name;

		if(file_path[0] == '.') {
			continue;
		}

		if(type.size() <= 0) {
			out_filelists.push_back(ptr->d_name);
		} else {
			if(file_path.size() < type.size()) {
				continue;
			}

			std::string file_type = file_path.substr(file_path.size() - type.size(), type.size());

			if(file_type == type) {
				out_filelists.push_back(ptr->d_name);
			}
		}
	}
}

bool computePairNum(std::string pair1, std::string pair2) {
	return pair1 < pair2;
}

void sortFileLists(std::vector<std::string>& file_lists) {
	if(file_lists.empty()) {
		return;
	}

	std::sort(file_lists.begin(), file_lists.end(), computePairNum);
}

int main(int argc, char** argv) {
	std::filesystem::path fold_path = std::filesystem::current_path().parent_path();

	const std::string trt_cfg_path   = fold_path.string() + "/configs/trt.yaml";
	const std::string param_cfg_path = fold_path.string() + "/configs/param.yaml";
	const std::string calib_cfg_path = fold_path.string() + "/configs/calib.yaml";

	YAML::Node param_node = YAML::LoadFile(param_cfg_path);
	YAML::Node calib_node = YAML::LoadFile(calib_cfg_path);

	const std::vector<std::string> img_fold_name = param_node["fold_name"].as<std::vector<std::string>>();

	std::vector<std::string>              img_fold_path;
	std::vector<std::vector<std::string>> img_file_path;

	img_fold_path.resize(img_fold_name.size());
	img_file_path.resize(img_fold_name.size());

	for(int i = 0; i < img_fold_name.size(); ++i) {
		img_fold_path[i] = param_node["data_root"].as<std::string>() + "/" + img_fold_name[i];

		readFileLists(img_fold_path[i], img_file_path[i], "jpg");
		sortFileLists(img_file_path[i]);
	}

	std::shared_ptr<TRTFastBEV> trt_fastbev = std::make_shared<TRTFastBEV>(param_node, calib_node);

	trt_fastbev->setConfig(trt_cfg_path.c_str());

	if(!trt_fastbev->loadEngine()) {
		trt_fastbev->buildEngine(calib_cfg_path);
	}

	for(int i = 0; i < img_file_path[0].size(); ++i) {
		std::vector<cv::Mat> imgs;

		for(int j = 0; j < img_file_path.size(); ++j) {
			imgs.emplace_back(cv::imread(img_fold_path[j] + "/" + img_file_path[j][i]));
		}

		auto start = std::chrono::steady_clock::now();
		trt_fastbev->inference(imgs);
		std::cout << "Elapsed(ms) = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << "ms" << std::endl;
	}

	return 0;
}
