#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "lorenz63_model.hpp"
#include "particle_fixed_lag.hpp"
#include "TestCheck.h"

using namespace PKF;

/**
 * Particle filter example: simulate the Lorenz-63 chaotic system with
 * heavy-tailed (Student-t) measurement noise, run a bootstrap particle
 * filter with fixed-lag ancestry-based smoothing, compute RMSE for
 * filtered and smoothed estimates, and export results to pkf_results.csv.
 */
int main() {
    // Parameters
    const size_t NUM_PARTICLES = 2000;
    const size_t LAG = 20; // Fixed lag
    const int STEPS = 500;

    // Model
    Lorenz63Model model;

    // Filter
    ParticleFilterFixedLag<3, 3> pf(&model, NUM_PARTICLES, LAG);
    // The filter seeds itself from std::random_device by default, so without this
    // the run is irreproducible and its RMSE cannot be asserted on (successive
    // runs drifted ~1.5%: 0.7987, 0.7874, ...). Seeding covers the serial and the
    // OpenMP per-thread RNGs; results are then exact for a given thread count.
    pf.set_seed(42);

    // RNG for simulation
    std::mt19937_64 rng(42);

    // Initialize True State
    Lorenz63Model::State x_true;
    x_true << 1.0f, 1.0f, 1.0f; // Start slightly off-center

    // Initialize Filter
    // Prior: Gaussian around initial truth with some spread
    auto prior_sampler = [&](std::mt19937_64& r) -> Lorenz63Model::State {
        return Noise::gaussian_sample<3>(x_true, 5.0f * Lorenz63Model::StateMat::Identity(), r);
    };
    pf.initialize(prior_sampler);

    // Data Storage
    struct DataPoint {
        float t;
        Lorenz63Model::State x_true;
        Lorenz63Model::Observation y;
        Lorenz63Model::State x_est_filt;
        Lorenz63Model::State x_est_smooth;
    };
    std::vector<DataPoint> results;
    results.reserve(STEPS);

    // Simulation Loop
    Lorenz63Model::State u = Lorenz63Model::State::Zero(); // No control

    std::cout << "Running Particle Filter with Fixed-Lag Smoother on Lorenz-63..." << std::endl;
    std::cout << "Particles: " << NUM_PARTICLES << ", Lag: " << LAG << std::endl;

    for (int k = 1; k <= STEPS; ++k) {
        float t = (float)k * Lorenz63Model::DT;

        // 1. Simulate Truth
        x_true = model.propagate(x_true, t, u) + model.sample_process_noise(t, rng);

        // 2. Simulate Observation
        Lorenz63Model::Observation y = model.observe(x_true, t) + model.sample_observation_noise(t, rng);

        // 3. Update Filter
        pf.step(y, t, u);

        // 4. Store Results
        results.push_back({t, x_true, y, pf.get_filtered_mean(), pf.get_smoothed_mean()});
    }

    // Export to CSV
    std::ofstream file("pkf_results.csv");
    file << "t,true_x,true_y,true_z,meas_x,meas_y,meas_z,filt_x,filt_y,filt_z,smooth_x,smooth_y,smooth_z\n";

    double rmse_filt = 0.0;
    double rmse_smooth = 0.0;
    int count_smooth = 0;

    for (int i = 0; i < STEPS; ++i) {
        const auto& d = results[i];
        file << d.t << ","
             << d.x_true(0) << "," << d.x_true(1) << "," << d.x_true(2) << ","
             << d.y(0) << "," << d.y(1) << "," << d.y(2) << ","
             << d.x_est_filt(0) << "," << d.x_est_filt(1) << "," << d.x_est_filt(2) << ","
             << d.x_est_smooth(0) << "," << d.x_est_smooth(1) << "," << d.x_est_smooth(2) << "\n";

        // Compute RMSE
        // Filter: x_k vs x_k_est
        rmse_filt += (d.x_true - d.x_est_filt).squaredNorm();

        // Smoother:
        // The value d.x_est_smooth at index i (time k) corresponds to time t_{k-L}.
        // So we should compare it to x_true at index i-L.
        int lag_idx = i - static_cast<int>(LAG);
        if (lag_idx >= 0) {
            const auto& d_lagged = results[lag_idx];
            rmse_smooth += (d_lagged.x_true - d.x_est_smooth).squaredNorm();
            count_smooth++;
        }
    }

    file.close();

    rmse_filt = std::sqrt(rmse_filt / STEPS);
    rmse_smooth = std::sqrt(rmse_smooth / count_smooth);

    std::cout << "Simulation Complete." << std::endl;
    std::cout << "Filter RMSE: " << rmse_filt << std::endl;
    std::cout << "Smoother RMSE: " << rmse_smooth << " (Lag: " << LAG << ")" << std::endl;

    // Registered as the PKF_Example CTest case, so these must gate the exit code;
    // this program previously had no non-zero exit path.
    //
    // Bounds are looser than the Kalman demos' on purpose. This is a bootstrap
    // particle filter: even fully seeded (set_seed above), the result is exact only
    // for a fixed thread count, because the OpenMP propagation draws per-thread and
    // the work split changes which draw lands on which particle. Seeded reference
    // ~0.80 filtered / ~0.60 smoothed; ceilings sit well clear of the spread
    // observed across 1-4 threads, and are here to catch gross breakage (filter
    // divergence, a dead smoother), not small numerical drift.
    NLF_CHECK(count_smooth > 0, "fixed-lag smoother produced estimates");
    NLF_CHECK(std::isfinite(rmse_filt) && std::isfinite(rmse_smooth), "RMSE is finite");
    NLF_CHECK(rmse_smooth < rmse_filt, "fixed-lag smoothing reduces RMSE");
    NLF_CHECK(rmse_filt < 1.10, "filter RMSE within absolute bound");
    NLF_CHECK(rmse_smooth < 0.85, "smoother RMSE within absolute bound");

    return 0;
}
