#include "preprocessing.h"

const PreprcMeta configToMeta(YAML::Node& param_node) {
	PreprcMeta meta;

	meta.img_h_ = param_node["model_height"].as<int>();
	meta.img_w_ = param_node["model_width"].as<int>();
	meta.raw_h_ = param_node["raw_height"].as<int>();
	meta.raw_w_ = param_node["raw_width"].as<int>();
	meta.n_cam_ = param_node["num_cam"].as<int>();

	std::vector<float> mean = param_node["mean"].as<std::vector<float>>();
	std::vector<float> std  = param_node["std"].as<std::vector<float>>();

	meta.mean_b_ = mean[0];
	meta.mean_g_ = mean[1];
	meta.mean_r_ = mean[2];

	meta.i_std_b_ = 1.0f / std[0];
	meta.i_std_g_ = 1.0f / std[1];
	meta.i_std_r_ = 1.0f / std[2];

	return meta;
}

Preprocessor::Preprocessor(YAML::Node& param_node, YAML::Node& calib_node) {
	config_ = configToMeta(param_node);

	// Allocate GPU memories
	const unsigned int raw_full_data_size = config_.n_cam_ * 3 * config_.raw_h_ * config_.raw_w_;

	cudaMalloc((void**)&dev_raw_img_, raw_full_data_size * sizeof(unsigned char));

	cam_list_ = param_node["fold_name"].as<std::vector<std::string>>();

	for(const std::string cam_name: cam_list_) {
		float* dev_map_x;
		float* dev_map_y;

		cudaMalloc((void**)&dev_map_x, config_.img_w_ * config_.img_h_ * sizeof(float));
		cudaMalloc((void**)&dev_map_y, config_.img_w_ * config_.img_h_ * sizeof(float));

		dev_mapping_[cam_name] = {dev_map_x, dev_map_y};
	}

	std::vector<int> resize_size   = param_node["resize_size"].as<std::vector<int>>();
	int              resize_width  = resize_size[0];
	int              resize_height = resize_size[1];

	std::vector<int> crop_size = param_node["crop_size"].as<std::vector<int>>();

	std::vector<std::vector<float>> cam_intrinsic = calib_node["cam_intrinsic"].as<std::vector<std::vector<float>>>();
	std::vector<std::vector<float>> dist_coef = calib_node["dist_coef"].as<std::vector<std::vector<float>>>();

	// Initialize mapping
	for(unsigned int cam_idx = 0; cam_idx < config_.n_cam_; ++cam_idx) {
		const std::string cam_name     = cam_list_[cam_idx];
		const float       resize_ratio = static_cast<float>(resize_width) / config_.raw_w_;

		const int unpad_w = resize_ratio * config_.raw_w_;
		const int unpad_h = resize_ratio * config_.raw_h_;

		cv::Mat map_x;
		cv::Mat map_y;

		cv::Mat intrinsic_coeffs(3, 3, CV_32FC1);

		for(int h = 0; h < intrinsic_coeffs.rows; ++h) {
			for(int w = 0; w < intrinsic_coeffs.cols; ++w) {
				intrinsic_coeffs.at<float>(h, w) = cam_intrinsic[cam_idx][h * intrinsic_coeffs.cols + w];
			}
		}

		cv::Mat dist_coeffs(1, 5, CV_32FC1);

		for(int h = 0; h < dist_coeffs.rows; ++h) {
			for(int w = 0; w < dist_coeffs.cols; ++w) {
				dist_coeffs.at<float>(h, w) = dist_coef[cam_idx][h * dist_coeffs.cols + w];
			}
		}

		cv::initUndistortRectifyMap(intrinsic_coeffs, dist_coeffs, cv::Mat(), cv::getOptimalNewCameraMatrix(intrinsic_coeffs, dist_coeffs, cv::Size(config_.raw_w_, config_.raw_h_), 1, cv::Size(unpad_w, unpad_h), 0), cv::Size(unpad_w, unpad_h), 5, map_x, map_y);

		// Crop
		map_x = map_x(cv::Range(crop_size[1], crop_size[3]), cv::Range(crop_size[0], crop_size[2]));
		map_y = map_y(cv::Range(crop_size[1], crop_size[3]), cv::Range(crop_size[0], crop_size[2]));

		// Save mapping to GPU memory
		std::vector<float> temp_map_x;

		for(u_int i = 0; i < map_x.rows; ++i) {
			temp_map_x.insert(temp_map_x.end(), map_x.ptr<float>(i), map_x.ptr<float>(i) + map_x.cols * map_x.channels());
		}

		assert(temp_map_x.size() == config_.img_w_ * config_.img_h_);

		cudaMemcpy(dev_mapping_[cam_name][0], temp_map_x.data(), temp_map_x.size() * sizeof(float), cudaMemcpyHostToDevice);

		std::vector<float> temp_map_y;

		for(u_int i = 0; i < map_y.rows; ++i) {
			temp_map_y.insert(temp_map_y.end(), map_y.ptr<float>(i), map_y.ptr<float>(i) + map_y.cols * map_y.channels());
		}

		assert(temp_map_y.size() == config_.img_w_ * config_.img_h_);

		cudaMemcpy(dev_mapping_[cam_name][1], temp_map_y.data(), temp_map_y.size() * sizeof(float), cudaMemcpyHostToDevice);
	}
}

Preprocessor::~Preprocessor() {
	// Deallocating GPU memories
	std::cout << "Deallocating CUDA memory of Preprocessor" << std::endl;

	cudaFree(dev_raw_img_);
	cudaFree(dev_out_img_);

	for(const std::string cam_name: cam_list_) {
		cudaFree(dev_mapping_[cam_name][0]);
		cudaFree(dev_mapping_[cam_name][1]);
	}
}

void Preprocessor::preprcMain(const std::vector<cv::Mat>& host_src, float* dst, std::vector<cudaStream_t> stream) {
	// Directly copy cv::Mat to GPU memory

	// Initialize output buffer

	cudaMemcpyAsync(dev_raw_img_ + 0 * 3 * config_.raw_h_ * config_.raw_w_, host_src[0].data, 3 * config_.raw_h_ * config_.raw_w_ * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(dev_raw_img_ + 1 * 3 * config_.raw_h_ * config_.raw_w_, host_src[1].data, 3 * config_.raw_h_ * config_.raw_w_ * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[1]);
	cudaMemcpyAsync(dev_raw_img_ + 2 * 3 * config_.raw_h_ * config_.raw_w_, host_src[2].data, 3 * config_.raw_h_ * config_.raw_w_ * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[2]);
	cudaMemcpyAsync(dev_raw_img_ + 3 * 3 * config_.raw_h_ * config_.raw_w_, host_src[3].data, 3 * config_.raw_h_ * config_.raw_w_ * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[3]);
	cudaMemcpyAsync(dev_raw_img_ + 4 * 3 * config_.raw_h_ * config_.raw_w_, host_src[4].data, 3 * config_.raw_h_ * config_.raw_w_ * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[4]);
	cudaMemcpyAsync(dev_raw_img_ + 5 * 3 * config_.raw_h_ * config_.raw_w_, host_src[5].data, 3 * config_.raw_h_ * config_.raw_w_ * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[5]);

	cudaStreamSynchronize(stream[0]);
	cudaStreamSynchronize(stream[1]);
	cudaStreamSynchronize(stream[2]);
	cudaStreamSynchronize(stream[3]);
	cudaStreamSynchronize(stream[4]);
	cudaStreamSynchronize(stream[5]);

	preprcAllAtOnce(config_, dev_raw_img_ + 0 * 3 * config_.raw_h_ * config_.raw_w_, dev_mapping_.find(cam_list_[0])->second[0], dev_mapping_.find(cam_list_[0])->second[1], dst + 0 * 3 * config_.img_h_ * config_.img_w_, IntpType::BILINEAR, stream[0]);
	preprcAllAtOnce(config_, dev_raw_img_ + 1 * 3 * config_.raw_h_ * config_.raw_w_, dev_mapping_.find(cam_list_[1])->second[0], dev_mapping_.find(cam_list_[1])->second[1], dst + 1 * 3 * config_.img_h_ * config_.img_w_, IntpType::BILINEAR, stream[1]);
	preprcAllAtOnce(config_, dev_raw_img_ + 2 * 3 * config_.raw_h_ * config_.raw_w_, dev_mapping_.find(cam_list_[2])->second[0], dev_mapping_.find(cam_list_[2])->second[1], dst + 2 * 3 * config_.img_h_ * config_.img_w_, IntpType::BILINEAR, stream[2]);
	preprcAllAtOnce(config_, dev_raw_img_ + 3 * 3 * config_.raw_h_ * config_.raw_w_, dev_mapping_.find(cam_list_[3])->second[0], dev_mapping_.find(cam_list_[3])->second[1], dst + 3 * 3 * config_.img_h_ * config_.img_w_, IntpType::BILINEAR, stream[3]);
	preprcAllAtOnce(config_, dev_raw_img_ + 4 * 3 * config_.raw_h_ * config_.raw_w_, dev_mapping_.find(cam_list_[4])->second[0], dev_mapping_.find(cam_list_[4])->second[1], dst + 4 * 3 * config_.img_h_ * config_.img_w_, IntpType::BILINEAR, stream[4]);
	preprcAllAtOnce(config_, dev_raw_img_ + 5 * 3 * config_.raw_h_ * config_.raw_w_, dev_mapping_.find(cam_list_[5])->second[0], dev_mapping_.find(cam_list_[5])->second[1], dst + 5 * 3 * config_.img_h_ * config_.img_w_, IntpType::BILINEAR, stream[5]);

	cudaStreamSynchronize(stream[0]);
	cudaStreamSynchronize(stream[1]);
	cudaStreamSynchronize(stream[2]);
	cudaStreamSynchronize(stream[3]);
	cudaStreamSynchronize(stream[4]);
	cudaStreamSynchronize(stream[5]);

	// Resize, crop, and normalize
}
