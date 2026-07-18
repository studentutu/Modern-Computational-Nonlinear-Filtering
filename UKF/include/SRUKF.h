#ifndef SRUKF_H
#define SRUKF_H

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/Cholesky>
#include <iostream>
#include <stdexcept>
#include <string>
#include "StateSpaceModel.h"
#include "SigmaPoints.h"
#include "FilterMath.h"

namespace UKFCore {

/**
 * Square Root Unscented Kalman Filter (SRUKF)
 *
 * Propagates the Cholesky factor S of the covariance P = S*S^T
 * instead of P directly. This provides better numerical stability
 * and guarantees positive definiteness.
 *
 * Uses QR decomposition for covariance updates and Cholesky
 * updates for efficient square root propagation.
 */
template<int NX, int NY>
class SRUKF {
public:
    using Model = UKFModel::StateSpaceModel<NX, NY>;
    using State = typename Model::State;
    using Observation = typename Model::Observation;
    using StateMat = typename Model::StateMat;
    using ObsMat = typename Model::ObsMat;
    using CrossMat = Eigen::Matrix<float, NX, NY>;
    using SigmaPts = SigmaPoints<NX>;

    // Parameters - dimension adaptive
    float alpha = 1.0f;    // Will be adjusted based on dimension
    float beta = 2.0f;     // Optimal for Gaussian
    float kappa = -1.0f;   // Will be set in initialize()

    /**
     * Set the chi-squared innovation-gate threshold. The default of 25.0 is
     * roughly the 3-sigma value for NY = 3; callers with different observation
     * dimensions or tighter tail-cutoff requirements can override it.
     */
    void setInnovationGateChi2(float threshold) { innovation_gate_chi2_ = threshold; }

    /** Query the current chi-squared innovation-gate threshold. */
    float getInnovationGateChi2() const { return innovation_gate_chi2_; }

    /** Construct SRUKF with a reference to the nonlinear state-space model. */
    SRUKF(Model& model) : model_(model) {
        x_.setZero();
        S_.setIdentity();  // Square root of covariance
    }

    /**
     * Set initial state, compute Cholesky factor S from P0, and choose
     * dimension-adaptive sigma-point parameters (alpha, beta, kappa).
     */
    void initialize(const State& x0, const StateMat& P0) {
        x_ = x0;
        // Dimension-adaptive parameters for SRUKF
        // Low/medium dimensions: alpha=1.0 for better weak observability handling
        // High dimensions: alpha=1e-3 to prevent numerical issues
        if (kappa < 0) {
            beta = 2.0f;  // Optimal for Gaussian
            if (NX <= 5) {
                alpha = 1.0f;
                kappa = 3.0f - static_cast<float>(NX);
            } else {
                alpha = 1.0f;
                kappa = 0.0f;
            }
        }

        // Validate P0 up-front. A caller passing NaN/Inf/asymmetric/non-PSD P0 will
        // otherwise silently degrade into an identity square-root and produce garbage
        // eight steps later. Fail fast at initialize().
        if (!P0.allFinite()) {
            throw std::runtime_error(
                "SRUKF::initialize: P0 contains NaN or Inf entries");
        }
        // Symmetry check (P0 must be symmetric within a small relative tolerance).
        {
            StateMat asym = P0 - P0.transpose();
            float asym_norm = asym.norm();
            float p_norm = std::max(P0.norm(), 1e-30f);
            if (asym_norm > 1e-4f * p_norm) {
                throw std::runtime_error(
                    "SRUKF::initialize: P0 is not symmetric within relative tolerance 1e-4");
            }
        }

        // Compute Cholesky factor of P0 using accelerated Cholesky with fallback.
        // The ladder is: accelerated Cholesky -> Cholesky with relative jitter -> LDLT.
        // If all three fail, throw with the P0 condition number in the message so the
        // caller learns immediately that P0 is not PSD.
        Eigen::MatrixXf L0_dyn = filtermath::cholesky(P0);
        StateMat L0;
        bool have_factor = false;
        if (L0_dyn.size() > 0) {
            L0 = L0_dyn;
            have_factor = true;
        } else {
            float trace_over_n = std::max(P0.trace() / static_cast<float>(NX), 0.0f);
            float jitter = std::max(1e-6f, 1e-8f * trace_over_n);
            StateMat P_jitter = P0 + jitter * StateMat::Identity();
            L0_dyn = filtermath::cholesky(P_jitter);
            if (L0_dyn.size() > 0) {
                L0 = L0_dyn;
                have_factor = true;
            } else {
                Eigen::LDLT<StateMat> ldlt(P_jitter);
                if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
                    StateMat P_ldlt = ldlt.matrixL();
                    Eigen::VectorXf D_sqrt = ldlt.vectorD().cwiseSqrt();
                    L0 = P_ldlt * D_sqrt.asDiagonal();
                    have_factor = true;
                }
            }
        }
        if (!have_factor) {
            // Report condition number to diagnose caller's bad P0.
            Eigen::MatrixXf P0_dyn = P0;
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(P0_dyn);
            const auto& sv = svd.singularValues();
            float smax = sv.size() > 0 ? sv(0) : 0.0f;
            float smin = sv.size() > 0 ? sv(sv.size() - 1) : 0.0f;
            float cond = (smin > 0.0f) ? (smax / smin)
                                       : std::numeric_limits<float>::infinity();
            throw std::runtime_error(
                std::string("SRUKF::initialize: P0 is not positive definite; ") +
                "condition number = " + std::to_string(cond));
        }
        S_ = L0;
    }

    /**
     * Prediction Step using Square Root formulation
     * Returns cross-covariance P_{x_k, x_{k+1}} for smoothing
     */
    StateMat predict(float t_k, const Eigen::Ref<const State>& u_k) {
        // 1. Generate Sigma Points directly from S (no P computation needed!)
        SigmaPts sigmas;
        generate_sigma_points_from_sqrt<NX>(x_, S_, alpha, beta, kappa, sigmas);


        // 2. Propagate Sigma Points
        // NOTE: Use explicit temporary to force evaluation and avoid aliasing issues
        typename SigmaPts::SigmaMat X_pred;
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            State temp = model_.f(sigmas.X.col(i), t_k, u_k);
            X_pred.col(i) = temp;
        }

        // 3. Compute Predicted Mean (using circular mean for angular states)
        // Force evaluation with .eval() to break Eigen aliasing/expression templates
        // while preserving NEON/SVE2 vectorization (unlike raw C array workaround)
        typename SigmaPts::SigmaMat X_pred_eval = X_pred.eval();
        typename SigmaPts::Weights Wm_eval = sigmas.Wm.eval();

        State x_pred_mean = State::Zero();
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            x_pred_mean.noalias() += Wm_eval(i) * X_pred_eval.col(i);
        }


        // Apply circular mean for angular states (using raw arrays for consistency)
        for (int j = 0; j < NX; ++j) {
            if (model_.isAngularState(j)) {
                float sin_sum = 0.0f, cos_sum = 0.0f;
                for (int i = 0; i < SigmaPts::NSIG; ++i) {
                    sin_sum += sigmas.Wm(i) * std::sin(X_pred(j, i));
                    cos_sum += sigmas.Wm(i) * std::cos(X_pred(j, i));
                }
                x_pred_mean(j) = std::atan2(sin_sum, cos_sum);
            }
        }

        // 4. Compute Square Root of Predicted Covariance using QR decomposition
        StateMat Q = model_.Q(t_k);
        StateMat S_Q = StateMat::Identity() * 1e-4f;  // Initialize with safe default

        Eigen::MatrixXf L_Q = filtermath::cholesky(Q);
        if (L_Q.size() > 0) {
            S_Q = L_Q;
        } else {
            Q += 1e-6f * StateMat::Identity();
            L_Q = filtermath::cholesky(Q);
            if (L_Q.size() > 0) {
                S_Q = L_Q;
            } else {
                Eigen::LDLT<StateMat> ldlt_Q(Q);
                if (ldlt_Q.info() == Eigen::Success && ldlt_Q.isPositive()) {
                    StateMat Q_ldlt = ldlt_Q.matrixL();
                    Eigen::VectorXf D_vec = ldlt_Q.vectorD();
                    for (int i = 0; i < NX; ++i) {
                        D_vec(i) = std::sqrt(std::max(D_vec(i), 1e-10f));
                    }
                    S_Q = Q_ldlt * D_vec.asDiagonal();
                }
            }
        }

        // Build matrix for QR: each ROW is sqrt(Wc[i])*diff[i]
        // For SRUKF, we compute QR of A^T where A has columns sqrt(Wc[i])*diff[i]
        // If A^T = Q*R, then A*A^T = R^T*R, so S = R^T is the Cholesky factor
        // Note: skip i=0 since it has special weight that can be negative
        Eigen::Matrix<float, 3*NX, NX> chi_diff_T;  // Transposed form for QR
        for (int i = 1; i < SigmaPts::NSIG; ++i) {
            State diff = X_pred.col(i) - x_pred_mean;
            // Wrap angular state differences to [-π, π]
            for (int j = 0; j < NX; ++j) {
                if (model_.isAngularState(j)) {
                    while (diff(j) > M_PI) diff(j) -= 2.0f * M_PI;
                    while (diff(j) < -M_PI) diff(j) += 2.0f * M_PI;
                }
            }
            float wc_sign = (sigmas.Wc(i) >= 0) ? 1.0f : -1.0f;
            chi_diff_T.row(i-1) = (std::sqrt(std::abs(sigmas.Wc(i))) * diff * wc_sign).transpose();
        }
        // Add S_Q rows (transpose of S_Q columns)
        chi_diff_T.block(2*NX, 0, NX, NX) = S_Q.transpose();

        // QR decomposition: chi_diff_T = Q * R where R is NX × NX upper triangular
        Eigen::HouseholderQR<Eigen::Matrix<float, 3*NX, NX>> qr(chi_diff_T);
        // Extract R factor (upper triangular), transpose to get lower triangular S
        StateMat R_matrix = qr.matrixQR().template block<NX, NX>(0, 0).template triangularView<Eigen::Upper>();
        StateMat S_pred = R_matrix.transpose();

        // Handle the i=0 term with rank-1 update (cholupdate)
        State diff_0 = X_pred.col(0) - x_pred_mean;
        // Wrap angular state differences to [-π, π]
        for (int j = 0; j < NX; ++j) {
            if (model_.isAngularState(j)) {
                while (diff_0(j) > M_PI) diff_0(j) -= 2.0f * M_PI;
                while (diff_0(j) < -M_PI) diff_0(j) += 2.0f * M_PI;
            }
        }
        float wc_0 = sigmas.Wc(0);
        if (wc_0 < 0) {
            // Rank-1 downdate — use safe version with fallback
            bool ok = cholupdate_downdate_safe(S_pred, diff_0, std::sqrt(std::abs(wc_0)));
            if (!ok) {
                // Fallback: recompute S_pred from full covariance
                StateMat P_full = StateMat::Zero();
                for (int i = 0; i < SigmaPts::NSIG; ++i) {
                    State d = X_pred.col(i) - x_pred_mean;
                    for (int j2 = 0; j2 < NX; ++j2) {
                        if (model_.isAngularState(j2)) {
                            while (d(j2) > M_PI) d(j2) -= 2.0f * M_PI;
                            while (d(j2) < -M_PI) d(j2) += 2.0f * M_PI;
                        }
                    }
                    P_full += sigmas.Wc(i) * (d * d.transpose());
                }
                P_full += Q;
                P_full = 0.5f * (P_full + P_full.transpose());
                P_full += 1e-8f * StateMat::Identity();
                Eigen::MatrixXf L_fb = filtermath::cholesky(P_full);
                if (L_fb.size() > 0) {
                    S_pred = L_fb;
                }
                // else keep current S_pred as best effort
            }
        } else {
            // Rank-1 update
            cholupdate(S_pred, diff_0, std::sqrt(wc_0));
        }

        // 5. Compute Cross Covariance P_{x_k, x_{k+1}} for smoothing
        // Build weighted diff matrices for NEON GEMM: P_cross = Dx_w * Dp^T
        Eigen::Matrix<float, NX, SigmaPts::NSIG> Dx_w, Dp;
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            State diff_pred = X_pred.col(i) - x_pred_mean;
            State diff_x = sigmas.X.col(i) - x_;
            // Wrap angular state differences to [-π, π]
            for (int j = 0; j < NX; ++j) {
                if (model_.isAngularState(j)) {
                    while (diff_pred(j) > M_PI) diff_pred(j) -= 2.0f * M_PI;
                    while (diff_pred(j) < -M_PI) diff_pred(j) += 2.0f * M_PI;
                    while (diff_x(j) > M_PI) diff_x(j) -= 2.0f * M_PI;
                    while (diff_x(j) < -M_PI) diff_x(j) += 2.0f * M_PI;
                }
            }
            Dp.col(i) = diff_pred;
            Dx_w.col(i) = sigmas.Wc(i) * diff_x;
        }
        StateMat P_cross = Dx_w * Dp.transpose();

        // Update state
        x_ = x_pred_mean;
        S_ = S_pred;

        return P_cross;
    }

    /**
     * Update Step using Square Root formulation
     */
    void update(float t_k, const Observation& y_k) {
        // 1. Generate Sigma Points directly from S (no P computation needed!)
        SigmaPts sigmas;
        generate_sigma_points_from_sqrt<NX>(x_, S_, alpha, beta, kappa, sigmas);

        // 2. Propagate through measurement function
        Eigen::Matrix<float, NY, SigmaPts::NSIG> Y_pred;
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            Y_pred.col(i) = model_.h(sigmas.X.col(i), t_k);
        }

        // 3. Compute Predicted Measurement Mean
        Observation y_hat = Observation::Zero();
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            y_hat += sigmas.Wm(i) * Y_pred.col(i);
        }

        // 4. Compute Square Root of Innovation Covariance using QR
        ObsMat R = model_.R(t_k);

        // Compute innovation covariance square root
        // For numerical robustness, compute P_yy directly from all sigma points
        // rather than using QR decomposition
        constexpr int NSIG = SigmaPts::NSIG;
        ObsMat P_yy = ObsMat::Zero();

        for (int i = 0; i < NSIG; ++i) {
            Observation diff_y = Y_pred.col(i) - y_hat;
            P_yy += sigmas.Wc(i) * (diff_y * diff_y.transpose());
        }
        P_yy += R;  // Add measurement noise

        // Ensure positive definite
        P_yy = 0.5f * (P_yy + P_yy.transpose());

        // Compute square root via accelerated Cholesky
        ObsMat S_yy;
        Eigen::MatrixXf L_Pyy = filtermath::cholesky(P_yy);
        if (L_Pyy.size() > 0) {
            S_yy = L_Pyy;
        } else {
            P_yy += 1e-6f * ObsMat::Identity();
            L_Pyy = filtermath::cholesky(P_yy);
            if (L_Pyy.size() > 0) {
                S_yy = L_Pyy;
            } else {
                Eigen::LDLT<ObsMat> ldlt_Pyy(P_yy);
                if (ldlt_Pyy.info() != Eigen::Success || !ldlt_Pyy.isPositive()) {
                    S_yy = ObsMat::Zero();
                    for (int i = 0; i < NY; ++i)
                        S_yy(i,i) = std::sqrt(R(i,i));
                } else {
                    ObsMat Pyy_ldlt = ldlt_Pyy.matrixL();
                    Eigen::VectorXf D_sqrt = ldlt_Pyy.vectorD().cwiseSqrt();
                    S_yy = Pyy_ldlt * D_sqrt.asDiagonal();
                }
            }
        }

        // Check S_yy for numerical issues — full NaN/Inf check + diagonal check
        if (!S_yy.allFinite()) {
            // Recompute from R as last resort
            S_yy = ObsMat::Zero();
            for (int i = 0; i < NY; ++i)
                S_yy(i,i) = std::sqrt(R(i,i));
        } else {
            for (int i = 0; i < NY; ++i) {
                if (S_yy(i,i) < 1e-10f) {
                    S_yy(i,i) = std::sqrt(R(i,i));
                }
            }
        }

        // 5. Compute Cross Covariance Pxy using NEON GEMM
        Eigen::Matrix<float, NX, NSIG> Dx_w;
        Eigen::Matrix<float, NY, NSIG> Dy;
        for (int i = 0; i < NSIG; ++i) {
            Dy.col(i) = Y_pred.col(i) - y_hat;
            State diff_x = sigmas.X.col(i) - x_;
            // Wrap angular state differences to [-π, π]
            for (int j = 0; j < NX; ++j) {
                if (model_.isAngularState(j)) {
                    while (diff_x(j) > M_PI) diff_x(j) -= 2.0f * M_PI;
                    while (diff_x(j) < -M_PI) diff_x(j) += 2.0f * M_PI;
                }
            }
            Dx_w.col(i) = sigmas.Wc(i) * diff_x;
        }
        CrossMat Pxy = Dx_w * Dy.transpose();

        // 6. Kalman Gain: K = Pxy * P_yy^{-1}
        // Prefer triangular solve (O(N^2)) over explicit inverse (O(N^3)) since we have S_yy
        // K = Pxy * (S_yy * S_yy^T)^{-1} = Pxy * S_yy^{-T} * S_yy^{-1}
        // Solve: S_yy * temp = Pxy^T  =>  temp = S_yy^{-1} * Pxy^T
        // Then:  S_yy^T * K^T = temp  =>  K^T = S_yy^{-T} * temp
        Eigen::Matrix<float, NX, NY> K;
        Eigen::Matrix<float, NY, NX> temp_T = S_yy.template triangularView<Eigen::Lower>()
                                               .solve(Pxy.transpose());
        Eigen::Matrix<float, NY, NX> K_T = S_yy.transpose().template triangularView<Eigen::Upper>()
                                            .solve(temp_T);
        K = K_T.transpose();

        // 7. State Update with innovation gating
        Observation innovation = y_k - y_hat;

        // Gate the innovation to prevent catastrophic updates from outliers
        Eigen::Matrix<float, NY, 1> temp_innov = S_yy.template triangularView<Eigen::Lower>().solve(innovation);
        float mahal_dist_sq = temp_innov.squaredNorm();
        const float gate_threshold = innovation_gate_chi2_;

        float scale = 1.0f;
        if (mahal_dist_sq > gate_threshold) {
            // Scale down large corrections
            scale = std::sqrt(gate_threshold / mahal_dist_sq);
            if (!warned_gate_) {
                std::clog << "[SRUKF] innovation gate active (chi2="
                          << mahal_dist_sq << " > threshold="
                          << gate_threshold << "); scaling correction by "
                          << scale << ". Further gate events will be silent."
                          << std::endl;
                warned_gate_ = true;
            }
        }

        State correction = scale * (K * innovation);
        x_ = x_ + correction;

        // 8. Covariance Update using square root form
        // P = P - K*S_yy*S_yy^T*K^T = S*S^T - K*S_yy*S_yy^T*K^T
        // Use QR to compute: [S^T, (K*S_yy)^T]^T and extract updated S
        Eigen::Matrix<float, NX, NY> U = K * S_yy;

        // Try rank-1 downdates, but detect if they fail
        StateMat S_updated = S_;
        bool downdate_failed = false;
        for (int i = 0; i < NY && !downdate_failed; ++i) {
            downdate_failed = !cholupdate_downdate_safe(S_updated, U.col(i), 1.0f);
        }

        if (downdate_failed) {
            // Fallback: compute P directly and take Cholesky
            // P_new = P - K * P_yy * K^T = S*S^T - K*S_yy*S_yy^T*K^T
            StateMat P_curr = S_ * S_.transpose();
            StateMat KPyyKT = U * U.transpose();  // U = K*S_yy, so U*U^T = K*S_yy*S_yy^T*K^T
            StateMat P_new = P_curr - KPyyKT;

            // Ensure positive definiteness
            P_new = 0.5f * (P_new + P_new.transpose());
            // Add small regularization
            P_new += 1e-8f * StateMat::Identity();

            // Take Cholesky factor using accelerated Cholesky with fallback
            Eigen::MatrixXf L_new = filtermath::cholesky(P_new);
            StateMat S_new;
            if (L_new.size() > 0) {
                S_new = L_new;
            } else {
                P_new += 1e-6f * StateMat::Identity();
                L_new = filtermath::cholesky(P_new);
                if (L_new.size() > 0) {
                    S_new = L_new;
                } else {
                    Eigen::LDLT<StateMat> ldlt(P_new);
                    if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
                        StateMat L = ldlt.matrixL();
                        Eigen::VectorXf D_vec = ldlt.vectorD();
                        for (int j = 0; j < NX; ++j) {
                            D_vec(j) = std::sqrt(std::max(D_vec(j), 1e-10f));
                        }
                        S_new = L * D_vec.asDiagonal();
                    } else {
                        S_new = S_;
                        for (int i = 0; i < NX; ++i) {
                            if (S_new(i,i) < 1e-6f) S_new(i,i) = 1e-6f;
                        }
                    }
                }
            }
            S_ = S_new;
        } else {
            S_ = S_updated;
        }
    }

    // Getters
    const State& getState() const { return x_; }
    StateMat getCovariance() const { return S_ * S_.transpose(); }
    const StateMat& getSqrtCovariance() const { return S_; }

    // Setters
    void setState(const State& x) { x_ = x; }
    void setSqrtCovariance(const StateMat& S) { S_ = S; }

private:
    Model& model_;
    State x_;
    StateMat S_;  // Square root of covariance (Cholesky factor)

    // Innovation-gate configuration.
    float innovation_gate_chi2_ = 25.0f;
    bool  warned_gate_ = false;

    /**
     * Rank-1 Cholesky update: S_new such that
     * S_new * S_new^T = S * S^T + alpha^2 * v * v^T
     */
    void cholupdate(StateMat& S, const State& v, float alpha) {
        State v_scaled = alpha * v;
        constexpr float eps = 1e-10f;
        for (int k = 0; k < NX; ++k) {
            if (std::abs(S(k,k)) < eps) {
                S(k,k) = eps;  // Regularize to prevent division by zero
            }
            float r = std::sqrt(S(k,k)*S(k,k) + v_scaled(k)*v_scaled(k));
            float c = r / S(k,k);
            float s = v_scaled(k) / S(k,k);
            S(k,k) = r;
            if (k < NX - 1) {
                if (std::abs(c) > eps) {
                    S.block(k+1, k, NX-k-1, 1) = (S.block(k+1, k, NX-k-1, 1) + s * v_scaled.segment(k+1, NX-k-1)) / c;
                }
                v_scaled.segment(k+1, NX-k-1) = c * v_scaled.segment(k+1, NX-k-1) - s * S.block(k+1, k, NX-k-1, 1);
            }
        }
    }

    /**
     * Rank-1 Cholesky downdate: S_new such that
     * S_new * S_new^T = S * S^T - alpha^2 * v * v^T
     * Returns false if downdate fails (result would not be positive definite).
     */
    bool cholupdate_downdate_safe(StateMat& S, const State& v, float alpha) {
        State v_scaled = alpha * v;
        constexpr float eps = 1e-10f;
        for (int k = 0; k < NX; ++k) {
            if (std::abs(S(k,k)) < eps) {
                S(k,k) = eps;  // Regularize to prevent division by zero
            }
            float r_sq = S(k,k)*S(k,k) - v_scaled(k)*v_scaled(k);
            if (r_sq <= 0) {
                // Downdate would produce non-positive-definite matrix
                return false;
            }
            float r = std::sqrt(r_sq);
            float c = r / S(k,k);
            float s = v_scaled(k) / S(k,k);
            S(k,k) = r;
            if (k < NX - 1) {
                if (std::abs(c) > eps) {
                    S.block(k+1, k, NX-k-1, 1) = (S.block(k+1, k, NX-k-1, 1) - s * v_scaled.segment(k+1, NX-k-1)) / c;
                }
                v_scaled.segment(k+1, NX-k-1) = c * v_scaled.segment(k+1, NX-k-1) - s * S.block(k+1, k, NX-k-1, 1);
            }
        }
        return true;
    }

    /**
     * Rank-1 Cholesky downdate (legacy version, silently handles failures)
     */
    void cholupdate_downdate(StateMat& S, const State& v, float alpha) {
        State v_scaled = alpha * v;
        constexpr float eps = 1e-10f;
        for (int k = 0; k < NX; ++k) {
            if (std::abs(S(k,k)) < eps) {
                S(k,k) = eps;  // Regularize to prevent division by zero
            }
            float r_sq = S(k,k)*S(k,k) - v_scaled(k)*v_scaled(k);
            if (r_sq <= 0) {
                // Scale-adaptive jitter relative to diagonal element
                r_sq = 1e-6f * S(k,k) * S(k,k);
            }
            float r = std::sqrt(r_sq);
            float c = r / S(k,k);
            float s = v_scaled(k) / S(k,k);
            S(k,k) = r;
            if (k < NX - 1) {
                if (std::abs(c) > eps) {
                    S.block(k+1, k, NX-k-1, 1) = (S.block(k+1, k, NX-k-1, 1) - s * v_scaled.segment(k+1, NX-k-1)) / c;
                }
                v_scaled.segment(k+1, NX-k-1) = c * v_scaled.segment(k+1, NX-k-1) - s * S.block(k+1, k, NX-k-1, 1);
            }
        }
    }

    /**
     * Cholesky update for observation dimension
     */
    void cholupdate_obs(ObsMat& S, const Observation& v, float alpha) {
        Observation v_scaled = alpha * v;
        constexpr float eps = 1e-10f;
        for (int k = 0; k < NY; ++k) {
            if (std::abs(S(k,k)) < eps) {
                S(k,k) = eps;  // Regularize to prevent division by zero
            }
            float r = std::sqrt(S(k,k)*S(k,k) + v_scaled(k)*v_scaled(k));
            float c = r / S(k,k);
            float s = v_scaled(k) / S(k,k);
            S(k,k) = r;
            if (k < NY - 1) {
                if (std::abs(c) > eps) {
                    S.block(k+1, k, NY-k-1, 1) = (S.block(k+1, k, NY-k-1, 1) + s * v_scaled.segment(k+1, NY-k-1)) / c;
                }
                v_scaled.segment(k+1, NY-k-1) = c * v_scaled.segment(k+1, NY-k-1) - s * S.block(k+1, k, NY-k-1, 1);
            }
        }
    }

    /**
     * Cholesky downdate for observation dimension
     */
    void cholupdate_downdate_obs(ObsMat& S, const Observation& v, float alpha) {
        Observation v_scaled = alpha * v;
        constexpr float eps = 1e-10f;
        for (int k = 0; k < NY; ++k) {
            if (std::abs(S(k,k)) < eps) {
                S(k,k) = eps;  // Regularize to prevent division by zero
            }
            float r_sq = S(k,k)*S(k,k) - v_scaled(k)*v_scaled(k);
            if (r_sq <= 0) {
                // Scale-adaptive jitter relative to diagonal element
                r_sq = 1e-6f * S(k,k) * S(k,k);
            }
            float r = std::sqrt(r_sq);
            float c = r / S(k,k);
            float s = v_scaled(k) / S(k,k);
            S(k,k) = r;
            if (k < NY - 1) {
                if (std::abs(c) > eps) {
                    S.block(k+1, k, NY-k-1, 1) = (S.block(k+1, k, NY-k-1, 1) - s * v_scaled.segment(k+1, NY-k-1)) / c;
                }
                v_scaled.segment(k+1, NY-k-1) = c * v_scaled.segment(k+1, NY-k-1) - s * S.block(k+1, k, NY-k-1, 1);
            }
        }
    }
};

} // namespace UKFCore

#endif // SRUKF_H
