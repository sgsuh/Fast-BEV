#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "ab3dmot.h"
#include "track_2d_object.h"
#include "trt_wrapper.h"

#include "preprocessing.h"

class TRTFastBEV: public TRTWrapper {
   public:
	TRTFastBEV(YAML::Node& param_node, YAML::Node& calib_node);
	virtual ~TRTFastBEV();

	void inference(std::vector<cv::Mat>& imgs);

   protected:
	// Headless BEV output (imshow is unavailable in the deploy container): draw
	// the decoded lidar-frame boxes top-down and write bev_<frame>.jpg to save_dir_.
	void saveBev(const std::vector<float>& box3d, const std::vector<float>& scores, const std::vector<float>& labels);

	std::string save_dir_;
	int         frame_idx_ = 0;

	void preprocess(const std::vector<cv::Mat>& imgs);
	void tensorFromImg(const std::vector<cv::Mat>& imgs);
	void postprocess(std::vector<cv::Mat>& imgs);
	void getBboxes(std::vector<float>& box3d, std::vector<float>& scores, std::vector<float>& labels);
	void gridAnchors(const std::vector<int>& featmap_sizes, std::vector<float>& mlvl_anchors);
	void singleLevelGridAnchors(const std::vector<int>& featmap_size, const int scale, std::vector<float>& anchors);
	void getBboxesSingle(const std::vector<float>& mlvl_anchors, std::vector<float>& box3d, std::vector<float>& scores, std::vector<float>& labels);
	void decode(const std::vector<float>& anchors, const std::vector<float>& deltas, std::vector<float>& bboxes, std::vector<float>& bboxes_nms);
	void box3dMulticlassScaleNms(const std::vector<float>& mlvl_bboxes, const std::vector<float>& mlvl_bboxes_for_nms, const std::vector<float>& mlvl_scores, const float score_thr, const int max_num, const std::vector<float>& mlvl_dir_scores, std::vector<float>& bboxes, std::vector<float>& scores, std::vector<float>& labels, std::vector<float>& dir_scores);
	void xywhr2xyxyr(const std::vector<float>& boxes_xywhr, std::vector<float>& boxes);
	void nmsGpu(const std::vector<float>& boxes, std::vector<std::pair<int, float>> scores, float thresh, std::vector<int>& keep);
	void circleNms(const std::vector<float>& dets, std::vector<std::pair<int, float>> scores, float thresh, std::vector<int>& keep);
	void limitPeriod(const std::vector<float>& boxes, const float dir_offset, const float offset, const float period, std::vector<float>& dir_rot);
	void getDetObjInfos(std::vector<float>& box3d, std::vector<float>& scores, std::vector<float>& labels);

	std::vector<std::vector<float>> anchorsSingleRange(const std::vector<int>& feature_size, const std::vector<float>& anchor_range, int scale, const std::vector<float>& sizes, const std::vector<float>& rotations);

	std::vector<std::vector<std::array<float, 9>>> getDetInfo(std::vector<float>& box3d, std::vector<float>& scores, std::vector<float>& labels);

	bool init_;

	Eigen::Matrix4f l2g_;

	std::vector<float> tensor_in_;
	std::vector<float> mlvl_anchors_;

	std::vector<std::vector<float>> tensor_out_;

	std::vector<cudaStream_t> stream_;

	const int openmp_num_threads_ = 4;
	const int box_code_size_      = 9;
	const int scales_             = 1;
	const int post_max_size_      = 83;
	const int num_classes_        = 10;
	const int nms_pre_            = 1000;
	const int max_num_            = 500;
	const int bev_code_size_      = 5;

	const float dir_offset_       = 0.7854f;
	const float dir_limit_offset_ = 0.0f;
	const float score_thr_        = 0.3f;

	const Eigen::MatrixXf nusc_corners_ = (Eigen::MatrixXf(3, 8) << 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f).finished();

	const std::vector<int> nms_type_list_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
	const std::vector<int> bev_idx_       = {0, 3, 7, 4};
	const std::vector<int> cam_dir_idx_   = {1, 2, 0, 4, 3, 5};

	const std::vector<float> ranges_              = {-50.0f, -50.0f, -1.8f, 50.0f, 50.0f, -1.8f};
	const std::vector<float> rotations_           = {0.0f, 1.57f};
	const std::vector<float> nms_thr_list_        = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.5f, 0.5f, 0.2f};
	const std::vector<float> nms_radius_thr_list_ = {4.0f, 12.0f, 10.0f, 10.0f, 12.0f, 0.85f, 0.85f, 0.175f, 0.175f, 1.0f};
	const std::vector<float> nms_rescale_factor_  = {1.0f, 0.7f, 0.55f, 0.4f, 0.7f, 1.0f, 1.0f, 4.5f, 9.0f, 1.0f};

	const std::vector<Eigen::Vector3f> corners_norm_ = {Eigen::Vector3f(-0.5f, -0.5f, 0.0f), Eigen::Vector3f(-0.5f, -0.5f, 1.0f), Eigen::Vector3f(-0.5f, 0.5f, 1.0f), Eigen::Vector3f(-0.5f, 0.5f, 0.0f), Eigen::Vector3f(0.5f, -0.5f, 0.0f), Eigen::Vector3f(0.5f, -0.5f, 1.0f), Eigen::Vector3f(0.5f, 0.5f, 1.0f), Eigen::Vector3f(0.5f, 0.5f, 0.0f)};

	const std::vector<std::vector<float>> sizes_ = {{0.866f, 2.5981f, 1.0f}, {0.5774f, 1.7321f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.4f, 0.4f, 1.0f}};

	// Parameter
	int model_height_;
	int model_width_;
	int img_channel_;
	int n_times_;
	int scale_factor_;
	int canvas_size_;
	int show_range_;
	int num_cam_;

	float vis_thred_;

	std::vector<int> resize_size_;
	std::vector<int> crop_size_;

	std::vector<float> mean_;
	std::vector<float> std_;

	std::vector<std::vector<int>> draw_boxes_indexes_bev_;
	std::vector<std::vector<int>> color_map_;

	// Calibration
	Eigen::Quaternionf ego2global_rot_;
	Eigen::Quaternionf lidar2ego_rot_;

	Eigen::Vector3f ego2global_trans_;
	Eigen::Vector3f lidar2ego_trans_;

	std::vector<float> extrinsic_;

	std::vector<Eigen::Matrix3f> sensor2lidar_rot_;
	std::vector<Eigen::Matrix3f> cam_intrinsic_;

	std::vector<Eigen::Quaternionf> sensor2ego_rot_;

	std::vector<Eigen::Vector3f> sensor2lidar_trans_;
	std::vector<Eigen::Vector3f> sensor2ego_trans_;

	std::vector<ObjInfo> det_objinfos_;

	TrackManagement tracker_;

	AB3DMOT track_3d_;

	// Preprocessing
	std::unique_ptr<Preprocessor> preprc_ptr_;
};
