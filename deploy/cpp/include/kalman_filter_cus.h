#pragma once

#include <Eigen/Dense>

class KalmanFilterCus {
   public:
	KalmanFilterCus(int dim_x, int dim_z, int dim_u = 0);

	void predict();
	void update(Eigen::VectorXf& z);

	Eigen::VectorXf x_;  // State
	Eigen::MatrixXf f_;  // Control Transition Matrix
	Eigen::MatrixXf h_;  // Measurement Function
	Eigen::MatrixXf p_;  // Uncertainty Covariance
	Eigen::MatrixXf q_;  // Process Uncertainty

   protected:
	int dim_x_;
	int dim_z_;
	int dim_u_;

	Eigen::MatrixXf r_;  // Measurement Uncertainty

	float alpha_sq_;  // Fading memory control

	Eigen::MatrixXf m_;  // Process-measurement cross correlation
	Eigen::VectorXf z_;

	// Gain and residual are computed during the innovation step. We save them so that in case you want to inspect them for various purpose
	Eigen::MatrixXf k_;  // Kalman gain
	Eigen::VectorXf y_;
	Eigen::MatrixXf s_;   // System uncertainty
	Eigen::MatrixXf si_;  // Inverse system uncertainty

	// Identity matrix
	Eigen::MatrixXf i_;

	// These will always be a copy of x, p after predict() is called
	Eigen::VectorXf x_prior_;
	Eigen::MatrixXf p_prior_;

	// These will always be a copy of x, p after update() is called
	Eigen::VectorXf x_post_;
	Eigen::MatrixXf p_post_;

	// Only computed only if requested via property
	float log_likelihood_;
	float likelihood_;
};
