#ifndef EKF_SMOOTHER_H
#define EKF_SMOOTHER_H

// =============================================================================
// EKFSmoother -- FULL-INTERVAL (batch) Rauch-Tung-Striebel smoother for the EKF,
// with optional ITERATION.
// -----------------------------------------------------------------------------
// The full-interval counterpart of EKFFixedLag (same RTS math, using the stored
// state-transition Jacobian F for the smoothing gain), and the EKF sibling of
// UKFSmoother / SRUKFSmoother.  It keeps the ENTIRE forward trajectory and runs a
// single backward pass once all measurements are in, so every measurement informs
// every epoch -- the natural "revisit the whole capture after the pass" refinement.
//
// smooth(n_iterations>0) re-runs the forward EKF from the SMOOTHED initial
// condition (keeping the original prior covariance so the data still dominate) and
// re-smooths, re-linearising the Jacobians about a better trajectory; this is the
// iterated EKF smoother (an IEKS), meant to run opportunistically as time allows.
// =============================================================================

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "EKF.h"
#include "EKFFixedLag.h"   // reuse the FilterState history struct
#include "FilterMath.h"

class EKFSmoother {
public:
    explicit EKFSmoother(SystemModel* model) : model_(model) {}

    /** Set the prior; clears any stored trajectory/measurements. */
    void initialize(const Eigen::VectorXf& x0, const Eigen::MatrixXf& P0) {
        x0_ = x0; P0_ = P0; meas_.clear();
        reset_forward(x0_, P0_);
    }

    /** Consume one measurement (retained for iteration replay); no smoothing yet. */
    void step(const Eigen::VectorXf& y_k, const Eigen::VectorXf& u_k, float t_k) {
        meas_.push_back(Meas{t_k, y_k, u_k});
        forward_step(y_k, u_k, t_k);
    }

    /** Backward RTS over the full interval; n_iterations>0 -> iterated (IEKS). */
    void smooth(int n_iterations = 0) {
        backward_pass();
        for (int it = 0; it < n_iterations; ++it) {
            if (x_smooth_.empty()) break;
            const Eigen::VectorXf xs0 = x_smooth_.front();
            reset_forward(xs0, P0_);                 // re-linearise, keep original prior
            for (const auto& m : meas_) forward_step(m.y, m.u, m.t);
            backward_pass();
        }
    }

    int size() const { return static_cast<int>(buffer_.size()); }
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> filtered_state(int k) const {
        return {buffer_[k].x_upd, buffer_[k].P_upd};
    }
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> smoothed_state(int k) const {
        return {x_smooth_[k], P_smooth_[k]};
    }
    Eigen::VectorXf smoothed_initial() const { return x_smooth_.front(); }
    Eigen::VectorXf final_filtered_state() const { return buffer_.back().x_upd; }

private:
    struct Meas { float t; Eigen::VectorXf y; Eigen::VectorXf u; };

    SystemModel* model_;
    std::unique_ptr<EKF> ekf_;
    Eigen::VectorXf x0_;
    Eigen::MatrixXf P0_;
    std::vector<FilterState> buffer_;
    std::vector<Meas> meas_;
    std::vector<Eigen::VectorXf> x_smooth_;
    std::vector<Eigen::MatrixXf> P_smooth_;

    void reset_forward(const Eigen::VectorXf& x, const Eigen::MatrixXf& P) {
        ekf_ = std::make_unique<EKF>(model_, x, P);
        buffer_.clear();
        FilterState s;
        s.t = 0.0f; s.x_upd = x; s.P_upd = P;
        buffer_.push_back(s);
    }

    void forward_step(const Eigen::VectorXf& y_k, const Eigen::VectorXf& u_k, float t_k) {
        Eigen::MatrixXf F = ekf_->predict(u_k, t_k);
        FilterState s;
        s.t = t_k;
        s.x_pred = ekf_->getPredictedState();
        s.P_pred = ekf_->getPredictedCovariance();
        ekf_->update(y_k, t_k);
        s.x_upd = ekf_->getState();
        s.P_upd = ekf_->getCovariance();
        s.F = F;
        if (!s.x_upd.allFinite() || !s.P_upd.allFinite()) return;  // NaN guard
        buffer_.push_back(s);
    }

    // Backward RTS over the full buffer (identical math to EKFFixedLag::step).
    void backward_pass() {
        const int N = static_cast<int>(buffer_.size());
        x_smooth_.assign(N, Eigen::VectorXf());
        P_smooth_.assign(N, Eigen::MatrixXf());
        if (N == 0) return;
        x_smooth_[N-1] = buffer_[N-1].x_upd;
        P_smooth_[N-1] = buffer_[N-1].P_upd;
        for (int j = N - 2; j >= 0; --j) {
            // G_j = P_{j|j} F_{j+1}^T P_{j+1|j}^{-1}  (via SPD solve, no explicit inverse)
            Eigen::MatrixXf PFt = filtermath::gemm(buffer_[j].P_upd,
                                                   buffer_[j+1].F.transpose());
            Eigen::MatrixXf G = filtermath::kalman_gain(PFt, buffer_[j+1].P_pred);
            Eigen::VectorXf diff_x = x_smooth_[j+1] - buffer_[j+1].x_pred;
            x_smooth_[j] = buffer_[j].x_upd + filtermath::mat_vec_mul(G, diff_x);
            Eigen::MatrixXf diff_P = P_smooth_[j+1] - buffer_[j+1].P_pred;
            Eigen::MatrixXf t1 = filtermath::gemm(G, diff_P);
            P_smooth_[j] = buffer_[j].P_upd + filtermath::gemm(t1, G.transpose());
        }
    }
};

#endif // EKF_SMOOTHER_H
