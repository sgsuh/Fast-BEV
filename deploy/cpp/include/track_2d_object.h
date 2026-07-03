#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

#include "definition.h"
#include "kalman_filter_v2.h"

class Track2D {
   public:
	Track2D();
	virtual ~Track2D();

	unsigned char valid_flag_;
	unsigned char object_flag_;
	unsigned char track_flag_;
	unsigned char track_debug_;

	int ileft_;
	int ibottom_;
	int iwidth_;
	int iheight_;
	int count_;
	int life_time_;
	int penalty_;
	int assoc_index_;
	int num_assoc_;
	int num_cons_assoc_;
	int age_;
	int id_;

	float longitudinal_distance_;
	float lateral_distance_;
	float longitudinal_speed_;
	float longitudinal_acceleration_;
	float lateral_speed_;
	float estimated_width_;
	float width_error_cov_;
	float max_overlap_ratio_;
	float score_;
	float du_;
	float dv_;
	float predicted_center_u_;
	float predicted_center_v_;
	float predicted_width_;
	float predicted_height_;
	float predicted_left_;
	float predicted_bottom_;
	float predicted_theta_;
	float fleft_;
	float fbottom_;
	float fwidth_;
	float fheight_;
	float flength_;
	float ftheta_;
	float center_u_;
	float center_v_;
	float center_z_;
	float prev_center_u_;
	float prev_center_v_;
	float gate_left_top_u_;
	float gate_left_top_v_;
	float gate_right_bottom_u_;
	float gate_right_bottom_v_;
	float gate_width_;
	float gate_height_;

	std::array<float, 30> box_width_;
	std::array<float, 30> velocity_history_;

	KalmanFilterV2 kalman_position_u_;
	KalmanFilterV2 kalman_position_v_;
	KalmanFilterV2 kalman_size_w_;
	KalmanFilterV2 kalman_size_h_;
	KalmanFilterV2 kalman_position_theta_;

   protected:
};

class TrackManagement {
   public:
	TrackManagement();
	virtual ~TrackManagement();

	void removeOverlapTrack();
	void predictTrack();
	void track(std::vector<ObjInfo>& cand);

	const int max_life_      = 5;
	const int min_life_      = 3;
	const int max_count_     = 15;
	const int max_track_num_ = 100;
	const int max_penalty_   = 5;

	const float min_overlap_ratio_         = 0.01f;
	const float min_size_ratio_            = 0.1f;
	const float position_of_weight_        = 0.75f;
	const float position_assoc_weight_     = 1.0f - position_of_weight_;
	const float measurement_noise_assoc_   = 0.01f;
	const float measurement_noise_init_    = 0.01f;
	const float w_focal_length_            = 1350.0f;
	const float overlap_step_              = 0.05f;
	const float default_overlap_           = 0.01f;
	const float measurement_noise_n_assoc_ = 10.0f;
	const float sampling_time_             = 0.09f;
	const float costfunc_size_weight_      = 0.5f;
	const float costfunc_dx_weight_        = 1.0f;
	const float costfunc_dy_weight_        = 1.0f;
	const float remove_th_                 = 0.1f;
	const float merge_th_                  = 0.1f;
	const float measurement_noise_w_       = 0.05f;
	const float max_overlap_               = 1.0f;

	Eigen::Matrix2f ru_;
	Eigen::Matrix2f rv_;
	Eigen::Matrix2f rw_;
	Eigen::Matrix2f rh_;
	Eigen::Matrix2f rt_;

	Eigen::Matrix4f au_;
	Eigen::Matrix4f av_;
	Eigen::Matrix4f aw_;
	Eigen::Matrix4f ah_;
	Eigen::Matrix4f at_;

	Eigen::Matrix4f p_;
	Eigen::Matrix4f q_;

	Eigen::MatrixXf z_;

	std::vector<Track2D> track2d_;

   protected:
	void compareCountWidth(Track2D& track1, Track2D& track2);
	void invalidTrack(Track2D& track);
	void setTrackAreaLimit(Track2D& track);

	bool isPointInRect(ObjInfo& cand, Track2D& track);
	bool isOverlap(ObjInfo& cand, Track2D& track);

	float computeDistanceRatio(ObjInfo& cand, Track2D& track);
};
