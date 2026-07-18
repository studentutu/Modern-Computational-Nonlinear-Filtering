#ifndef UKF_SMOOTHER_H
#define UKF_SMOOTHER_H

// =============================================================================
// UKFSmoother -- FULL-INTERVAL (batch) unscented RTS smoother, with iteration.
// -----------------------------------------------------------------------------
// The plain-covariance counterpart of SRUKFSmoother (see that header for the
// full rationale).  Complements UnscentedFixedLagSmoother: keeps the ENTIRE
// forward trajectory and runs a single backward Rauch-Tung-Striebel pass once all
// measurements are in, so every measurement informs every epoch.  smooth(n>0)
// re-runs the forward filter from the smoothed initial condition (original prior
// kept) and re-smooths -- the iterated smoother, to run "as time is available".
// The RTS recursion is identical to UnscentedFixedLagSmoother::perform_smoothing.
// =============================================================================

#include <vector>
#include <Eigen/Dense>
#include "UKF.h"
#include "UnscentedFixedLagSmoother.h"   // reuse UKFHistoryEntry<NX>
#include "FilterMath.h"

namespace UKFCore {

template<int NX, int NY>
class UKFSmoother {
public:
    using UKFType = UKF<NX, NY>;
    using Model = typename UKFType::Model;
    using State = typename Model::State;
    using Observation = typename Model::Observation;
    using StateMat = typename Model::StateMat;

    explicit UKFSmoother(Model& model) : model_(model), ukf_(model) {}

    void initialize(const State& x0, const StateMat& P0) {
        x0_ = x0; P0_ = P0; meas_.clear();
        reset_forward(x0_, P0_);
    }

    /** Consume one measurement (stored for iteration replay); no smoothing yet. */
    void step(float t_k, const Observation& y_k, const Eigen::Ref<const State>& u_k) {
        meas_.push_back(Meas{t_k, y_k, State(u_k)});
        forward_step(t_k, y_k, u_k);
    }

    /** Backward RTS over the full interval; n_iterations>0 -> iterated smoother. */
    void smooth(int n_iterations = 0) {
        backward_pass();
        for (int it = 0; it < n_iterations; ++it) {
            if (smoothed_states_.empty()) break;
            const State xs0 = smoothed_states_.front();
            reset_forward(xs0, P0_);                 // re-linearise, keep original prior
            for (const auto& m : meas_) forward_step(m.t, m.y, m.u);
            backward_pass();
        }
    }

    int size() const { return static_cast<int>(history_.size()); }
    State  filtered_state(int k)      const { return history_[k].x_filt; }
    StateMat filtered_covariance(int k) const { return history_[k].P_filt; }
    State  smoothed_state(int k)      const { return smoothed_states_[k]; }
    StateMat smoothed_covariance(int k) const { return smoothed_covs_[k]; }
    State  smoothed_initial() const { return smoothed_states_.front(); }
    State  final_filtered_state() const { return history_.back().x_filt; }

private:
    struct Meas { float t; Observation y; State u; };
    Model& model_;
    UKFType ukf_;
    State x0_;
    StateMat P0_;
    std::vector<UKFHistoryEntry<NX>> history_;
    std::vector<Meas> meas_;
    std::vector<State> smoothed_states_;
    std::vector<StateMat> smoothed_covs_;

    void reset_forward(const State& x, const StateMat& P) {
        ukf_.initialize(x, P);
        history_.clear();
        UKFHistoryEntry<NX> e; e.x_filt = x; e.P_filt = P;
        history_.push_back(e);
    }

    void forward_step(float t_k, const Observation& y_k,
                      const Eigen::Ref<const State>& u_k) {
        if (history_.empty()) return;
        UKFHistoryEntry<NX>& last = history_.back();
        StateMat P_cross = ukf_.predict(t_k, u_k);
        last.x_pred_next = ukf_.getState();
        last.P_pred_next = ukf_.getCovariance();
        last.P_cross_next = P_cross;
        ukf_.update(t_k, y_k);
        if (!ukf_.getState().allFinite() || !ukf_.getCovariance().allFinite()) return;
        UKFHistoryEntry<NX> e; e.x_filt = ukf_.getState(); e.P_filt = ukf_.getCovariance();
        history_.push_back(e);
    }

    void backward_pass() {
        const int N = static_cast<int>(history_.size());
        smoothed_states_.assign(N, State::Zero());
        smoothed_covs_.assign(N, StateMat::Identity());
        if (N == 0) return;
        smoothed_states_[N-1] = history_.back().x_filt;
        smoothed_covs_[N-1]   = history_.back().P_filt;
        for (int j = N - 2; j >= 0; --j) {
            const auto& e = history_[j];
            StateMat G = filtermath::kalman_gain(
                Eigen::MatrixXf(e.P_cross_next), Eigen::MatrixXf(e.P_pred_next));
            State diff_x = smoothed_states_[j+1] - e.x_pred_next;
            Eigen::VectorXf dx = filtermath::mat_vec_mul(
                Eigen::MatrixXf(G), Eigen::VectorXf(diff_x));
            smoothed_states_[j] = e.x_filt + State(dx);
            StateMat diff_P = smoothed_covs_[j+1] - e.P_pred_next;
            Eigen::MatrixXf t1 = filtermath::gemm(Eigen::MatrixXf(G), Eigen::MatrixXf(diff_P));
            Eigen::MatrixXf t2 = filtermath::gemm(t1, Eigen::MatrixXf(G.transpose()));
            StateMat P_s = e.P_filt + StateMat(t2);
            smoothed_covs_[j] = 0.5f * (P_s + P_s.transpose());
        }
    }
};

} // namespace UKFCore

#endif // UKF_SMOOTHER_H
