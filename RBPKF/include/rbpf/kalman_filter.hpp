#ifndef RBPF_KALMAN_FILTER_HPP
#define RBPF_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <iostream>
#include "FilterMath.h"

namespace rbpf {

/**
 * @brief Generic reusable Kalman Filter class.
 *
 * @tparam Types Typdef struct containing dimensions and matrix types.
 */
template<typename Types>
class LinearKalmanFilter {
public:
    using LinearState = typename Types::LinearState;
    using LinearCov   = typename Types::LinearCov;
    using Observation = typename Types::Observation;
    using ObsCov      = typename Types::ObsCov;

    LinearState x;
    LinearCov   P;

    /**
     * @brief Initialize the filter state and covariance.
     */
    void initialize(const LinearState& x0, const LinearCov& P0) {
        x = x0;
        P = P0;
    }

    /**
     * @brief Predict step: x = A*x + bias, P = A*P*A^T + Q
     */
    // Fixed-size A (Nlin x Nlin, compile-time) so mat_vec_mul/gemm bind to the
    // FilterMath fixed-size fast path: stack-allocated, no runtime backend
    // dispatch, no heap-allocated intermediates. This matters because the RBPF
    // calls predict()/update() once per particle per step (thousands of times).
    void predict(const typename Types::Matrix_A& A,
                 const Eigen::Ref<const LinearState>& bias,
                 const LinearCov& Q) {
        // x_k|k-1 = A * x_{k-1|k-1} + bias
        LinearState Ax = filtermath::mat_vec_mul(A, x);
        x = Ax + bias;

        // P_k|k-1 = A * P * A^T + Q
        LinearCov AP = filtermath::gemm(A, P);
        LinearCov APAt = filtermath::gemm(AP, A.transpose());

        P = APAt + Q;

        // Ensure symmetry
        P = 0.5f * (P + P.transpose());
    }

    /**
     * @brief Update step using Joseph form for stability.
     */
    // Fixed-size H (Ny x Nlin, compile-time). All intermediates are fixed-size so
    // every gemm/mat_vec below takes the FilterMath fixed-size fast path; only the
    // final kalman_gain SPD solve remains on the dynamic path (small Nlin x Ny).
    void update(const Eigen::Ref<const Observation>& y,
                const typename Types::Matrix_H& H,
                const Eigen::Ref<const Observation>& offset,
                const ObsCov& R) {
        // Innovation: z = y - (H*x + offset)
        Observation Hx = filtermath::mat_vec_mul(H, x);
        Observation z = y - (Hx + offset);

        // Innovation covariance: S = H*P*H^T + R
        Eigen::Matrix<float, Types::Ny, Types::Nlin> HP = filtermath::gemm(H, P);
        ObsCov HPHt = filtermath::gemm(HP, H.transpose());
        ObsCov S = HPHt + R;

        // Kalman gain via SPD solve (more stable than explicit inverse)
        typename Types::CrossCov PHt = filtermath::gemm(P, H.transpose());
        Eigen::Matrix<float, Types::Nlin, Types::Ny> K = filtermath::kalman_gain(PHt, S);

        // Update state: x = x + K*z
        LinearState Kz = filtermath::mat_vec_mul(K, z);
        x = x + Kz;

        // Update covariance (Joseph form): P = (I - KH)P(I - KH)^T + KRK^T
        LinearCov I = LinearCov::Identity();

        LinearCov KH = filtermath::gemm(K, H);
        LinearCov I_KH = I - KH;

        LinearCov P_I_KH_T = filtermath::gemm(P, I_KH.transpose());
        LinearCov Term1 = filtermath::gemm(I_KH, P_I_KH_T);

        Eigen::Matrix<float, Types::Ny, Types::Nlin> RKt = filtermath::gemm(R, K.transpose());
        LinearCov Term2 = filtermath::gemm(K, RKt);

        P = Term1 + Term2;

        // Ensure symmetry
        P = 0.5f * (P + P.transpose());
    }
};

} // namespace rbpf

#endif // RBPF_KALMAN_FILTER_HPP
