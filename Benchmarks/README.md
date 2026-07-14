# Comprehensive Filtering Benchmarks

This directory contains challenging benchmark problems and a comprehensive testing framework for evaluating all nonlinear filtering methods in this repository.

> **Portability contract.** The filters are portable C++20 + Eigen and build on any
> supported architecture; SIMD (NEON/SVE2) and GPU (CUDA/Vulkan) backends are
> *optional accelerators* selected at runtime, always with an Eigen fallback.
> Consequently: **accuracy and consistency results (RMSE, NEES, divergences) are
> backend-independent and reproduce on any host**, while **every timing number is
> specific to the machine that produced it and is always reported with that host
> named**. No single machine is "the" reference host.

## Overview

The default suite (`run_benchmarks`) runs **four** problems — Coupled Oscillators
(10D), Van der Pol (2D), Bearing-Only (4D), and Reentry Vehicle (6D) — exercising:
- **High dimensionality** (up to 10 states in the default suite)
- **Strong nonlinearity**
- **Partial / weak observability**
- **Discontinuous forcing**
- **Highly nonlinear observations** (spherical range/az/el)

> A 40-state chaotic **Lorenz96** model is also provided in
> `include/BenchmarkProblems.h` for high-dimensional experimentation, but it is
> **not part of the default run** (`run_benchmarks` instantiates the four problems
> above). See "Extending the Benchmarks" to add it to the runner.

## Benchmark Problems (default suite)

### 1. Coupled Oscillators (10D)
- **State dimension**: 10 (5 oscillators, position + velocity each)
- **Observation dimension**: 5 (positions only)
- **Characteristics**:
  - Nonlinear coupling between all oscillators
  - Damped dynamics
  - Nonlinear observation function: `y = pos + 0.1*sin(pos)`
- **Difficulty**: Medium-High (high dimensional, coupled nonlinear dynamics)

### 2. Van der Pol with Discontinuous Forcing (2D)
- **State dimension**: 2
- **Observation dimension**: 1
- **Characteristics**:
  - Nonlinearity parameter μ=5 (stiff oscillator)
  - Square wave forcing (discontinuous control)
  - Quadratic measurement nonlinearity
- **Difficulty**: Medium (low dimensional but highly stiff and discontinuous)

### 3. Reentry Vehicle Tracking (6D)
- **State dimension**: 6 (position + velocity in 3D)
- **Observation dimension**: 3 (range, azimuth, elevation)
- **Characteristics**:
  - Exponential atmospheric model
  - Altitude-dependent drag
  - Spherical observation coordinates
  - Gravity model
- **Difficulty**: High (highly nonlinear observations and dynamics)

### 4. Bearing-Only Tracking (4D)
- **State dimension**: 4 (2D position + velocity)
- **Observation dimension**: 1 (bearing angle only)
- **Characteristics**:
  - Weakly observable (bearing-only)
  - Moving observer platform
  - Constant velocity target
- **Difficulty**: High (observability challenges)

## Filters Tested

### 1. UKF (Unscented Kalman Filter)
- Derivative-free nonlinear filtering
- Sigma point propagation
- Standard covariance propagation

### 2. SRUKF (Square Root UKF)
- **NEW**: Square root formulation for improved numerical stability
- Propagates Cholesky factor S where P = S*S^T
- Uses QR decomposition and rank-1 updates
- Guarantees positive definiteness

### 3. UKF + Fixed-Lag Smoother
- UKF with Rauch-Tung-Striebel (RTS) backward smoothing
- Lag window of 20-30 steps (20 for Coupled Oscillators, Van der Pol and Reentry; 30 for Bearing-Only)
- Improves estimates using future measurements
- ⚠️ **In the default suite this is run on Coupled Oscillators (10D) only.** The
  other three problems benchmark `SRUKF+Smoother` without a `UKF+Smoother`
  counterpart.

### 4. SRUKF + Fixed-Lag Smoother
- SRUKF with RTS smoothing in square root form
- Enhanced numerical stability for smoothing
- Optimal for long lag windows
- Run on all four problems.

> Because `UKF+Smoother` covers only one problem, the suite emits **13 rows**, not
> 4 filters × 4 problems = 16.

## Evaluation Metrics

### Accuracy Metrics
- **RMSE (Root Mean Square Error)**: overall state RMSE across all components
  (`RMSE_Overall`)
- **Smoothed RMSE**: overall state RMSE of the smoothed estimates
  (`RMSE_Smoothed_Overall`), time-aligned to account for the smoother lag

> Per-subvector (position / velocity) RMSE is **deliberately not reported**.
> `compute_rmse_indices()` exists in `BenchmarkRunner.h` for callers who need it,
> but the runner does not call it: "which components are position" is
> model-specific and undefined for problems like `ReentryVehicle`, whose state has
> no clean position/velocity split. Reporting a per-subvector number the framework
> cannot define generically would be worse than omitting it.

> ⚠️ **Systematic RMSE bias (known, unfixed).** The benchmark loop
> (`src/run_benchmarks.cpp`) calls `predict()` for step `i` *before* `update()`
> with measurement `i`, then compares the result against `true_states[i]` — so
> every stored estimate is offset one timestep ahead of the truth it is scored
> against. This biases **every RMSE figure in this document by roughly +0.6%**.
> The bias is small, systematic, and identical across filters, so *relative*
> comparisons remain valid; absolute RMSE values are slightly pessimistic.
> Fixing it requires an API change to the smoother `step()` interface.

### Consistency Metrics
- **NEES (Normalized Estimation Error Squared)**
  - Median NEES (should be ≈ state dimension for consistent filter; robust to outliers)
  - Percentage within 95% chi-squared bounds
  - Uses LDLT solve and diagonal condition number screening for robustness

> ⚠️ **`Pct_In_Bounds` is optimistic.** The chi-squared bounds come from
> `chi2_quantile()`, a **Wilson-Hilferty approximation**, not exact quantiles. Its
> *lower* bound is biased low at small `n` — verified against exact values:
> −48% at n=2 (0.026 vs 0.051), −7% at n=4, −2.5% at n=6, −0.8% at n=10. Since the
> test counts `nees >= lower`, a too-low floor admits steps that should fail, so
> reported `Pct_In_Bounds` reads **high**, most notably for Van der Pol (n=2). The
> upper bound is accurate to <1% for every `n` in this suite.

### Performance Metrics
- **Average Step Time**: Computational cost per filter step
- **Total Execution Time**: End-to-end runtime

> ⚠️ Timing figures are **host-specific** and must never be quoted without naming
> the machine that produced them (see [Measured Results](#measured-results)). The
> timing loop also has no warmup and no repetitions, and takes its `chrono`
> readings inside the measured region, which inflates the smallest step times by
> ~5%. Treat sub-microsecond per-step numbers as indicative only.

### Convergence Metrics
- **Convergence Time**: first time the state error norm stays below an absolute
  threshold for 50 consecutive steps. Reported as **NaN / an empty CSV cell /
  "did not converge"** when that never happens — never as a stand-in number.
  - The suite passes a threshold of **1.0** at every call site
    (`compute_convergence_time()`'s own default is 0.5, but the runner overrides it).
  - ⚠️ The threshold is a fixed **absolute** error norm, so problems whose natural
    error scale exceeds 1.0 can never converge by this definition regardless of
    filter quality. That is why all Bearing-Only (~64 m scale) and all Reentry
    (~369 m scale) rows report *did not converge*: it reflects the metric's
    definition, not filter failure — NEES shows both are consistent. A
    problem-scaled threshold, as already done for divergences, would fix this.
- **Number of Divergences**: count of steps where the state error `||true − est||`
  exceeds a **problem-scaled** threshold (Bearing-Only 500 m, Reentry 5 km; 10.0 for
  the small-scale problems). With per-problem thresholds this is **0** for all 13
  rows of the suite. (Prior to v3.2.2 every problem used the fixed 10.0 default,
  which was far below Bearing-Only's ~64 m error scale and falsely reported ~176
  "divergences" for a filter that was actually consistent — NEES 99.6% in-bounds.)

## Building and Running

### Build

The filters are portable C++20 + Eigen. Every SIMD/GPU backend is an **optional
accelerator with an Eigen fallback**, and the target architecture is auto-detected,
so the default invocation works on any supported host:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make run_benchmarks
```

The OptMathKernels dependency is consumed as a **sibling source tree** via
`add_subdirectory` (`OPTMATH_DIR`, default `../OptimizedKernelsForRaspberryPi5_NvidiaCUDA`):

```bash
# Point at an existing checkout...
cmake .. -DCMAKE_BUILD_TYPE=Release -DOPTMATH_DIR=/path/to/OptimizedKernelsForRaspberryPi5_NvidiaCUDA
# ...or let the build clone it. Default is OFF: if the directory is missing,
# configuration FAILS with instructions rather than silently fetching.
cmake .. -DCMAKE_BUILD_TYPE=Release -DAUTO_CLONE_DEPS=ON
```

Backend and tuning options — all optional, all with fallbacks:

| Option | Default | Effect |
|---|---|---|
| `NLF_ENABLE_NEON` | ON on ARM | Build the ARM NEON kernels |
| `NLF_ENABLE_SVE2` | ON on ARM | Build the ARM SVE2 kernels (sources auto-skipped when the CPU lacks SVE2) |
| `NLF_ENABLE_VULKAN` | ON | Build the Vulkan compute backend (auto-skipped when no SDK is found) |
| `NLF_ENABLE_NATIVE_ARCH` | ON | Tune Release builds for the building CPU. The flag is **probed, not assumed**: `-mcpu=native` on ARM, `-march=native` on x86, `/O2` on MSVC, portable codegen on unknown toolchains. **Turn OFF for redistributable binaries.** |

```bash
# CUDA hosts — build for the detected GPU (e.g. Blackwell / RTX 50-series, SM 120, CUDA 13.x):
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DOPTMATH_CUDA_NATIVE=ON
# Hosts with no CUDA toolkit (CPU-only build):
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=""
# Maximally portable — no SIMD, no native tuning, pure Eigen:
cmake .. -DCMAKE_BUILD_TYPE=Release -DNLF_ENABLE_NEON=OFF -DNLF_ENABLE_SVE2=OFF -DNLF_ENABLE_NATIVE_ARCH=OFF
```

> **Note**: The benchmark target is compiled with `-fno-fast-math` and `EIGEN_FAST_MATH=0` to ensure numerically stable filter results. All linear algebra is routed through `FilterMath.h`, which dispatches at runtime to the best backend the machine actually provides: CUDA cuBLAS > SVE2 GEMM > NEON > Eigen. Tiers the host cannot supply are skipped, and **Eigen is always present as the fallback** — no backend is required for correct results. Compute kernels come from OptMathKernels **v0.6.3**, consumed as an **unpinned sibling checkout** (`add_subdirectory`), so the build tracks whatever revision is checked out in `OPTMATH_DIR` rather than a fixed release tag.
>
> **Backend independence (verified)**: RMSE/NEES are backend-independent; only timings are host-specific. Comparing a no-SIMD build (`-DNLF_ENABLE_NEON=OFF -DNLF_ENABLE_SVE2=OFF -DNLF_ENABLE_NATIVE_ARCH=OFF`) against the NEON build on the same aarch64 host: Coupled Oscillators and Van der Pol are **bit-identical**; Bearing-Only differs by 0.0008% (64.1723 vs 64.1728) and Reentry by 0.02% (369.117 vs 369.043). Those deltas are pure floating-point association, not behavioral differences.
>
> **Reproducibility**: the PKF/RBPF OpenMP paths seed one `mt19937_64` per thread deterministically from the base seed (`config.seed`) and use `schedule(static)`, so multithreaded runs are reproducible **for a given thread count** (verified: identical `test_rbpf_basic` output across runs at 8 threads on an 8-thread host; the guarantee is per-thread-count, not cross-machine).

### Run Benchmarks
```bash
./Benchmarks/run_benchmarks
```

This will:
1. Generate ground truth trajectories for each problem
2. Run all filters on each problem
3. Compute comprehensive metrics
4. Save results to CSV files

### Output Files

#### Summary Results
- `benchmark_results.csv`: metrics table for all filters and problems (13 rows).
  The schema is exactly:

```
Filter,Problem,RMSE_Overall,RMSE_Smoothed_Overall,Median_NEES,Pct_In_Bounds,NEES_Valid,NEES_Total,Avg_Step_Time_ms,Total_Time_ms,Convergence_Time,Num_Divergences
```

  `RMSE_Smoothed_Overall` is `0` for filter-only rows. `Convergence_Time` is an
  **empty cell** when the filter did not converge (read as NaN/missing by pandas
  and by every spreadsheet) — see [Convergence Metrics](#convergence-metrics).

#### Trajectory Files
- `coupled_osc_ukf.csv`, `coupled_osc_srukf.csv`, `coupled_osc_ukf_smooth.csv`, `coupled_osc_srukf_smooth.csv`
- `vanderpol_ukf.csv`, `vanderpol_srukf.csv`, `vanderpol_srukf_smooth.csv`
- `bearing_ukf.csv`, `bearing_srukf.csv`, `bearing_srukf_smooth.csv`
- `reentry_ukf.csv`, `reentry_srukf.csv`, `reentry_srukf_smooth.csv`

(14 files. `*_ukf_smooth.csv` exists only for `coupled_osc_`, since `UKF+Smoother`
runs on that problem alone.)

Each trajectory file contains:
- Time stamps
- True state values
- Filtered estimates
- Smoothed estimates (if applicable)

### Visualization

Use the provided Python script to generate comparison plots:

```bash
python3 ../scripts/plot_benchmarks.py .
```

This generates:
- `benchmark_rmse_comparison.png`: RMSE comparison across all methods
- `benchmark_timing_comparison.png`: Computational performance comparison
- `benchmark_convergence_comparison.png`: Convergence and stability metrics
- Individual trajectory plots for each test case

## Measured Results

**Accuracy and consistency figures below are backend-independent and reproduce on
any host** (see "Backend independence" above). **Timing figures are host-specific.**

> **Timing host**: aarch64, Cortex-A76 (4 cores, NEON; no SVE2, no CUDA), GCC,
> Release, `-mcpu=native`, `NLF_ENABLE_NEON=ON`. Dispatch resolves to the **NEON**
> tier on this machine. Reproduce on your own hardware rather than quoting these
> numbers — they will not transfer.

Full suite: **13 rows, ~1.9 s wall** on the timing host above.

| Filter | Problem | RMSE_Overall | RMSE_Smoothed | Median_NEES | Pct_In_Bounds | Avg_Step_ms | Convergence | Diverg |
|---|---|---|---|---|---|---|---|---|
| UKF | CoupledOscillators10D | 1.45666 | — | 9.892 | 94.5% | 0.0243 | 21.8404 s | 0 |
| SRUKF | CoupledOscillators10D | 1.45666 | — | 9.892 | 94.5% | 0.0251 | 21.8404 s | 0 |
| UKF+Smoother | CoupledOscillators10D | 1.45666 | 1.14767 | 9.892 | 94.5% | 0.0998 | 21.8404 s | 0 |
| SRUKF+Smoother | CoupledOscillators10D | 1.45666 | 1.14767 | 9.892 | 94.5% | 0.1469 | 21.8404 s | 0 |
| UKF | VanDerPol2D | 0.468053 | — | 1.136 | 95.9% | 0.000914 | 0.839999 s | 0 |
| SRUKF | VanDerPol2D | 0.46626 | — | 1.137 | 96.0% | 0.000733 | 0.839999 s | 0 |
| SRUKF+Smoother | VanDerPol2D | 0.46626 | 0.429704 | 1.137 | 96.0% | 0.0240 | 0.839999 s | 0 |
| UKF | BearingOnly4D | 63.8082 | — | 3.772 | 99.6% | 0.00118 | did not converge | 0 |
| SRUKF | BearingOnly4D | 64.1728 | — | 3.774 | 99.6% | 0.00136 | did not converge | 0 |
| SRUKF+Smoother | BearingOnly4D | 64.1728 | 52.0285 | 3.774 | 99.6% | 0.0445 | did not converge | 0 |
| UKF | ReentryVehicle6D | 369.043 | — | 4.996 | 95.9% | 0.00467 | did not converge | 0 |
| SRUKF | ReentryVehicle6D | 369.192 | — | 4.989 | 95.6% | 0.00490 | did not converge | 0 |
| SRUKF+Smoother | ReentryVehicle6D | 369.192 | 236.785 | 4.989 | 95.6% | 0.0575 | did not converge | 0 |

Per-problem wall time on the timing host (summed across that problem's filters):
Coupled Oscillators ~1.49 s (5001 steps), Van der Pol ~0.05 s (2000 steps),
Bearing-Only ~0.014 s (300 steps), Reentry ~0.020 s (300 steps).

### SRUKF vs UKF
- **Equivalent accuracy.** RMSE agrees to 4-6 significant figures on every
  problem, and is bit-identical on Coupled Oscillators (1.45666 for both).
- **Better conditioning**: guaranteed positive-definite covariance by construction.
- **Comparable-to-modestly-slower** execution. The only step-time difference this
  suite can actually resolve is Reentry, where SRUKF costs **~4.9%** per step
  (slower on all 5 of 5 repeat runs, with non-overlapping distributions) —
  consistent with the expected QR-decomposition overhead. The 10D problem shows a
  ~2% gap that is **not resolvable**: UKF's run-to-run spread there (±3.7%) is
  wider than the gap, and the sign flips between runs. Van der Pol and Bearing-Only
  step times (<1.4 µs) sit at `chrono` resolution. Treat all three as "no
  measurable difference" rather than quoting a percentage — the suite has no
  warmup and no repetitions, so a single run cannot support a claim this small.
  - ⚠️ **Do not expect SRUKF to be faster than UKF.** Earlier revisions of
    [COMPARISON_RESULTS.md](../COMPARISON_RESULTS.md) advertised SRUKF as "43-47%
    faster on all problems". That claim did not reproduce — the sign of the effect
    is *reversed* on this host — and has been retracted.

### Smoothers vs Filters
- **~8-36% RMSE improvement** from smoothing: Van der Pol 7.8%, Bearing-Only 18.9%,
  Coupled Oscillators 21.2%, Reentry 35.9%.
- **Better performance** on weakly observable problems (Bearing-Only 64.17 → 52.03).
- **~4x to ~33x computational cost** from the backward pass, scaling with the lag
  window and *inversely* with filter-step cost — the cheapest problems show the
  largest ratios. Measured: Coupled Oscillators 4.1x (UKF) / 5.9x (SRUKF), Reentry
  11.7x, Van der Pol 32.7x, Bearing-Only 32.7x. (The previously documented "2-3x"
  was wrong by roughly an order of magnitude.)

### Problem-Specific Insights

#### Bearing-Only Tracking
- Large steady-state range error (~64 m) from weak observability — this is the
  problem's nature, not filter divergence: NEES is 99.6% in-bounds and divergences
  are 0.
- Significant improvement from smoothing (18.9%).
- Reports *did not converge* because the convergence metric uses a fixed absolute
  0.5 threshold, far below this problem's error scale.

## Extending the Benchmarks

### Adding New Problems

1. Create a new model class in `include/BenchmarkProblems.h`:
```cpp
template<int NX, int NY>
class YourNewProblem : public UKFModel::StateSpaceModel<NX, NY> {
    // Implement f(), h(), Q(), R()
};
```

2. Add test case in `src/run_benchmarks.cpp`:
```cpp
{
    YourNewProblem<NX, NY> model;
    // Initialize and run benchmarks
}
```

### Adding New Filters

1. Implement filter following existing interface
2. Add benchmark runner function in `src/run_benchmarks.cpp`
3. Add to main loop

## Performance Tips

### For Real-Time Applications
- Prefer UKF for the cheapest step, but the margin over SRUKF is small (a few
  percent on the measured problems) — pick on numerical grounds, not speed
- Reduce lag window size for smoothers: the backward pass, not the filter, is what
  costs (4-33x the filter step in this suite)
- Consider decimating observations on high-rate sensors

### For Accuracy
- Use SRUKF for long-duration tracking
- Use smoothers when real-time is not critical
- Increase lag window for better smoothing (up to ~100 steps)

### For Numerical Stability
- Always use SRUKF for:
  - Very high-dimensional systems (>20 states)
  - Long-duration missions (>1000 steps)
  - Systems with very small process noise
- Monitor NEES: values >> state dimension indicate filter divergence

## References

1. **UKF**: Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation", 2004
2. **SRUKF**: Van der Merwe & Wan, "The Square-Root Unscented Kalman Filter", 2001
3. **RTS Smoothing**: Rauch et al., "Maximum Likelihood Estimates of Linear Dynamic Systems", 1965
4. **Lorenz96**: Lorenz, "Predictability: A Problem Partly Solved", 1996

## License

Part of the Modern Computational Nonlinear Filtering project.
