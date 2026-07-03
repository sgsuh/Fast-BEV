#include <cfloat>

#include "kalman_filter_cus.h"

KalmanFilterCus::KalmanFilterCus(int dim_x, int dim_z, int dim_u) {
    dim_x_ = dim_x;
    dim_z_ = dim_z;
    dim_u_ = dim_u;

    x_ = Eigen::VectorXf(dim_x).setZero();
    p_ = Eigen::MatrixXf(dim_x, dim_x).setIdentity();
    q_ = Eigen::MatrixXf(dim_x, dim_x).setIdentity();

    f_ = Eigen::MatrixXf(dim_x, dim_x).setIdentity();
    h_ = Eigen::MatrixXf(dim_z, dim_x).setZero();
    r_ = Eigen::MatrixXf(dim_z, dim_z).setIdentity();

    alpha_sq_ = 1.0f;

    m_ = Eigen::MatrixXf(dim_x, dim_z).setZero();
    z_ = Eigen::VectorXf(dim_z).transpose();

    k_ = Eigen::MatrixXf(dim_x, dim_z).setZero();
    y_ = Eigen::VectorXf(dim_z);
    s_ = Eigen::MatrixXf(dim_z, dim_z).setZero();
    si_ = Eigen::MatrixXf(dim_z, dim_z).setZero();

    i_ = Eigen::MatrixXf(dim_x, dim_x).setIdentity();

    x_prior_ = x_;
    p_prior_ = p_;

    x_post_ = x_;
    p_post_ = p_;

    log_likelihood_ = std::log(FLT_MIN);
    likelihood_ = FLT_MIN;
}

void KalmanFilterCus::predict() {
    // x = Fx + Bu
    x_ = f_ * x_;

    // P = FPF' + Q
    p_ = alpha_sq_ * f_ * p_ * f_.transpose() + q_;

    // Save prior
    x_prior_ = x_;
    p_prior_ = p_;
}

void KalmanFilterCus::update(Eigen::VectorXf& z) {
    // y = z - Hx
    // Error (residual) between measurement and prediction
    y_ = z - h_ * x_;

    // Common subexpression for speed
    Eigen::MatrixXf pht = p_ * h_.transpose();

    // S = HPH' + R
    // Project system uncertainty into measurement space
    s_ = h_ * pht + r_;
    si_ = s_.inverse();

    // K = PH'inv(S)
    // Map system uncertainty into kalman gain
    k_ = pht * si_;

    // x = x + Ky
    // Predict new x with residual scaled by the kalman gain
    x_ = x_ + k_ * y_;

    // P = (I - KH)P(I - KH)' + KRK'
    // This is more numerically stable and works for non-optimal K vs the equation P = (I - KH)P usually seen in the literature
    Eigen::MatrixXf i_kh = i_ - k_ * h_;

    p_ = i_kh * p_ * i_kh.transpose() + k_ * r_ * k_.transpose();

    z_ = z;
    x_post_ = x_;
    p_post_ = p_;
}
