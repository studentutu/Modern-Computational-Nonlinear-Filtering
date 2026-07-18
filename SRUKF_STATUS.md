# SRUKF Implementation Status

## Current State (v3.4.0, July 2026)

> SRUKF (R32) wraps angular **observation** innovations to [−π, π], and (R33)
> exposes the normalized innovation squared (NIS) via `getLastNIS()` plus an
> innovation gate with an optional reject-outliers policy (`setRejectOutliers()`).
> See the Numerical Safety Chain below.

### All Benchmark Problems Working

| Problem | SRUKF RMSE | NEES median | In 95% bounds | Status |
|---------|-----------|-------------|---------------|--------|
| Coupled Oscillators (10D) | 1.457 | 9.89 | 94.5% | **WORKING** |
| Van der Pol (2D) | 0.466 | 1.14 | 96.0% | **WORKING** |
| Bearing-Only (4D) | 64.17 | 3.77 | 99.6% | **WORKING** |
| Reentry Vehicle (6D) | 369.2 | 4.99 | 95.6% | **WORKING** |

> These are accuracy figures, not timings: they are backend-independent and
> reproduce on any host and any acceleration tier (verified NEON vs Eigen-only —
> see DEVELOPMENT_NOTES "Backend portability"). No host qualifier needed.

All 4 benchmark problems pass with the fixed-lag smoother (table below). The
**full-interval RTS smoother** added in v3.4.0 is not part of the benchmark suite;
it is covered by the `SRUKF_Smoother` CTest — see DEVELOPMENT_NOTES v3.4.0.

10D coupled oscillators previously failed with NaN — now resolved through:
1. Dimension-adaptive parameters (alpha=1.0, kappa=3-n for NX<=5; kappa=0 for NX>5)
2. Direct P_yy computation instead of QR for S_yy
3. Safe Cholesky downdate with full-covariance fallback
4. Removed global `-ffast-math` which was corrupting NaN guards

### Smoother Results (fixed-lag)

| Problem | SRUKF+Smoother RMSE | Improvement |
|---------|-------------------|-------------|
| Coupled Oscillators (10D) | 1.148 | 21% |
| Van der Pol (2D) | 0.430 | 8% |
| Bearing-Only (4D) | 52.03 | 19% |
| Reentry Vehicle (6D) | 236.8 | 36% |

## Technical Details

### FilterMath Dispatch Layer
All SRUKF linear algebra now routes through `Common/include/FilterMath.h`:
- **GEMM**: CUDA cuBLAS → SVE2 cache-blocked → NEON → Eigen
- **Cholesky**: NEON accelerated → Eigen LLT/LDLT fallback
- **Kalman Gain**: SPD solve (O(n²)) instead of explicit inverse (O(n³))
- **Cross-platform**: Non-ARM falls through to Eigen (or CUDA if available)

Every tier above Eigen is **optional**. The dispatch decision is made at compile
time and the Eigen fallback is always present, so SRUKF is correct on any target
with a C++20 compiler and Eigen; the accelerators change speed, not answers.
(In practice SRUKF uses fixed-size Eigen types, which bind to FilterMath's
compile-time `gemm` overload and never reach the runtime dispatch at all.)

> **Note**: On CUDA hosts the CUDA backend is active for SM 75–120, including
> Blackwell RTX 50-series (verified on an RTX 5070 Ti Laptop GPU / SM 120,
> CUDA 13.1.115); on hosts without CUDA the build drops to the next tier
> automatically. Compute kernels come from the OptMathKernels sibling tree — see
> [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) for the version and provisioning
> policy, which is deliberately **not** duplicated here.

### Dimension-Adaptive Parameters
```cpp
if (NX <= 5) {
    alpha = 1.0f;
    kappa = 3.0f - static_cast<float>(NX);
} else {
    alpha = 1.0f;
    kappa = 0.0f;
}
beta = 2.0f;  // Optimal for Gaussian
```

### Numerical Safety Chain
1. Safe Cholesky downdate → returns false on failure
2. Full-covariance fallback → recomputes P from all sigma points
3. **Angular-observation innovation wrap (R32)** → for observables flagged
   `isAngularObservation(j)`, `y_k − ŷ` is wrapped to [−π, π] so an innovation
   straddling the ±π branch cut cannot inject a catastrophic ~2π correction.
4. **Innovation gating + NIS (R33)** → the normalized innovation squared
   NIS = innovᵀ (S_yy S_yyᵀ)⁻¹ innov is compared to a χ² gate (25). NIS is stored
   and exposed via `getLastNIS()`. On a gate trip the correction is scaled by
   `scale`: down-scaled by `√(gate/NIS)` by default, or **rejected** (`scale = 0`)
   when `setRejectOutliers(true)`.
   - **Consistency (fixed in the v3.2.1 audit):** the covariance downdate stays consistent with
     the gated gain `scale·K` via the Joseph partial-update form — the downdate
     column is scaled by `√(2·scale − scale²)`, i.e. the covariance shrinks by
     `(2·scale − scale²)·K·P_yy·Kᵀ`. Endpoints: `scale = 1` → full update;
     `scale = 0` (rejected outlier) → zero downdate, so a discarded measurement no
     longer shrinks the covariance into false certainty. Previously only the state
     was scaled while the covariance was downdated in full.
5. S_yy validation → `allFinite()` check + diagonal minimum enforcement
6. Multiple Cholesky fallbacks: accelerated → jitter → LDLT → diagonal

## Code Locations

- **FilterMath dispatch**: `Common/include/FilterMath.h`
- **SRUKF core**: `UKF/include/SRUKF.h` (class `SRUKF`)
- **Sigma points**: `UKF/include/SigmaPoints.h`
- **SRUKF fixed-lag smoother**: `UKF/include/SRUKFFixedLagSmoother.h`
  (class `SRUKFFixedLagSmoother`) — the variant exercised by the benchmarks
- **SRUKF full-interval RTS smoother**: `UKF/include/SRUKFSmoother.h`
  (class `SRUKFSmoother`) — batch + iterated, added v3.4.0; CTest `SRUKF_Smoother`
- **Benchmarks**: `Benchmarks/src/run_benchmarks.cpp`

## Future Enhancement Opportunities

### Potter's Square Root Filter / UD Factorization
For systems with extreme conditioning (cond(P) > 1e10), alternatives to Cholesky-based square root:
- UD factorization: U (unit upper triangular) × D (diagonal)
- More robust for very ill-conditioned problems
- Better for fixed-point arithmetic

### Adaptive Regularization
For production systems with unknown dynamics:
- Monitor condition number of innovation covariance
- Dynamically adjust diagonal loading based on trace
- Iterative refinement for Kalman gain
