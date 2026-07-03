#include "kalman_filter_v2.h"

KalmanFilterV2::KalmanFilterV2()
    : h_(2, 4) {
}

KalmanFilterV2::~KalmanFilterV2() {
}

void KalmanFilterV2::init(Eigen::Matrix4f& a, Eigen::MatrixXf& h, Eigen::Matrix4f& p, Eigen::Matrix4f q, Eigen::Matrix2f r, Eigen::Vector4f x) {
	a_ = a;
	h_ = h;
	p_ = p;
	q_ = q;
	r_ = r;
	x_ = x;
}

void KalmanFilterV2::predict() {
	Eigen::Matrix4f at;  // Transposed State Transition Matrix A
	Eigen::Matrix4f ap;
	Eigen::Matrix4f apat;

	// Predict State
	// xp = a * x
	xp_ = a_ * x_;

	// Predict Noise Covariance Matrix
	// pp = a * p * at + q
	at   = a_.transpose();
	ap   = a_ * p_;
	apat = ap * at;
	pp_  = apat + q_;
}

void KalmanFilterV2::correct(std::array<float, 2>& z) {
	Eigen::MatrixXf k(4, 2);  // Kalman Gain
	Eigen::MatrixXf hx(2, 1);
	Eigen::MatrixXf residual(2, 1);
	Eigen::MatrixXf kres(4, 1);

	// Compute Kalman Gain
	// k = pp * ht * inv(h * pp * ht + r)
	k = pp_ * h_.transpose() * (h_ * pp_ * h_.transpose() + r_).inverse();

	// Correct State Vector
	hx = h_ * xp_;
	residual << z[0] - hx(0, 0), z[1] - hx(1, 0);
	kres = k * residual;

	x_ = xp_ + kres;

	// Correct Error Covariance Matrix
	p_ = (Eigen::Matrix4f::Identity() - k * h_) * pp_;
}

float KalmanFilterV2::computeMahalanobisDistance(std::array<float, 2>& z) {
	float error    = z[0] - xp_(0);
	float distance = error * error / pp_(0, 0);

	error = z[1] - xp_(1);
	distance += error * error / pp_(1, 1);

	return distance;
}
