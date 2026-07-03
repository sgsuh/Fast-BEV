#pragma once

#include <array>
#include <cmath>

#include <Eigen/Core>

enum CLASS_IDX {
	CLASS_IDX_CAR = 0,
	CLASS_IDX_TRUCK,
	CLASS_IDX_VEHICLE,
	CLASS_IDX_BUS,
	CLASS_IDX_TRAILER,
	CLASS_IDX_BARRIER,
	CLASS_IDX_MOTORCYCLE,
	CLASS_IDX_BICYCLE,
	CLASS_IDX_PEDESTRIAN,
	CLASS_IDX_TRAFFIC_CONE,
	CLASS_IDX_MAX_NUMBER
};

struct ObjInfo {
	Eigen::MatrixXf points_;
	Eigen::MatrixXf points_bev_;
	Eigen::Vector2f diff_position_;

	int class_;
	int cam_idx_;
	int id_;

	float score_;

	std::array<float, 9> position_;

	ObjInfo()
	    : id_(-1) {
	}

	float calculateVelocity(Eigen::Vector2f ego_velocity) {
		return std::sqrt(std::pow(ego_velocity(0) - diff_position_(0), 2) + std::pow(ego_velocity(1) - diff_position_(1), 2));
	}
};
