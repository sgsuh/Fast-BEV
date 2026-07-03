#pragma once

#include <array>

#include <Eigen/Dense>

class KalmanFilterV2 {
   public:
	KalmanFilterV2();
	virtual ~KalmanFilterV2();

	void init(Eigen::Matrix4f& a, Eigen::MatrixXf& h, Eigen::Matrix4f& p, Eigen::Matrix4f q, Eigen::Matrix2f r, Eigen::Vector4f x);
	void predict();
	void correct(std::array<float, 2>& z);

	float computeMahalanobisDistance(std::array<float, 2>& z);

	Eigen::Matrix4f p_;  // Error Covariance Matrix

	Eigen::Vector4f xp_;  // Predicted State Vector
	Eigen::Matrix2f r_;   // Measurement Noise Covariance Matrix
	Eigen::Vector4f x_;   // State Vector
   private:
	Eigen::Matrix4f a_;   // State Transition Matrix
	Eigen::MatrixXf h_;   // Measurement Matrix
	Eigen::Matrix4f q_;   // Process Noise Covariance Matrix
	Eigen::Matrix4f pp_;  // Predicted Noise Covariance Matrix
};
