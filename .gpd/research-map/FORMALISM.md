# FORMALISM.md — Bayesian Nonlinear Filtering (Modern Computational Nonlinear Filtering)

**Analysis Date:** 2026-07-18
**Focus:** theory
**Scope:** State-space Bayesian filters and smoothers implemented in the repository — EKF, UKF, SRUKF, PKF (bootstrap particle filter), RBPKF (Rao-Blackwellized particle filter), and their RTS smoothers (fixed-lag + full-interval, iterated where applicable). Test-problem physics for Van der Pol, coupled oscillators, bearings-only tracking, reentry vehicle, Lorenz-63, drag-ball, nonlinear pendulum, and CTRV.

**Convention marker.** All filters store state and covariance in **single-precision `float`** (32-bit IEEE 754). All symbols below are in SI base units unless a specific model overrides them (natural units are not used anywhere in this project — this is applied estimation, not field theory). "Dimensions" columns therefore refer to physical dimensions of the modeled quantities, not natural-unit powers.

---

## 1. Physical / Mathematical System

### 1.1 Class of problem

Discrete-time nonlinear state estimation on a hidden Markov model with additive-noise structure. The abstract system is a nonlinear state-space model:

```
Process (discrete):     x_k = f(x_{k-1}, u_k, t_k) + w_k,     w_k ~ N(0, Q(t_k))          (M1)
Observation:            y_k = h(x_k,     t_k)     + v_k,       v_k ~ N(0, R(t_k))          (M2)
Independence:           w_k ⊥ v_k, w_k ⊥ w_j (k≠j), v_k ⊥ v_j (k≠j), and both ⊥ x_0.
Prior:                  x_0 ~ N(x̂_0, P_0).                                                 (M3)
```

- `x_k ∈ ℝ^{NX}` state; `y_k ∈ ℝ^{NY}` observation; `u_k` control input; `t_k` epoch time.
- The Gaussian noise assumption is **relaxed by the PKF**, whose Lorenz-63 test problem uses **Student-t observation noise** (heavy-tailed, `nu = 3`); see §7.4.
- For the RBPKF (Rao-Blackwellized), the state factors into a nonlinear sub-state `x^{nl}` (Monte Carlo) and a conditionally linear-Gaussian sub-state `x^{lin}` (analytical Kalman filter), see §6.

**Governing abstract interfaces:**

| Interface | File | Purpose |
|-----------|------|---------|
| `UKFModel::StateSpaceModel<NX,NY>` | `Common/include/StateSpaceModel.h` (lines 14-65) | Base for UKF/SRUKF: `f`, `h`, `Q`, `R`, `isAngularState`, `isAngularObservation`. Fixed-size Eigen types compiled from template parameters. |
| `SystemModel` | `Common/include/SystemModel.h` (lines 13-56) | Base for EKF: adds Jacobians `F = ∂f/∂x`, `H = ∂h/∂x`. Dynamic-size `Eigen::VectorXf`/`MatrixXf`. |
| `PKF::StateSpaceModel<NX,NY>` | `PKF/include/state_space_model.hpp` (lines 17-80) | Base for particle filter: `propagate`, `observe`, `sample_process_noise`, `sample_observation_noise`, `observation_loglik`. Non-Gaussian likelihoods allowed. |
| `rbpf::NonlinearModel<Types>` + `rbpf::ConditionalLinearGaussianModel<Types>` | `RBPKF/include/rbpf/state_space_models.hpp` (lines 12-70) | RBPF split-interface: nonlinear proposal + linear-Gaussian conditional model. |

### 1.2 Degrees of freedom (project-wide)

Determined at compile time by `NX` (state dim) and `NY` (observation dim). No dynamical creation/deletion of degrees of freedom. PKF adds a Monte-Carlo dimension `N` (particle count); RBPKF adds `N` plus the linear sub-state dimension `Nlin`.

### 1.3 What is postulated vs. what is derived

- **Postulated:** the state-space equations (M1)-(M3), the noise covariances `Q`, `R`, and the prior `(x̂_0, P_0)`.
- **Derived:** the recursive Bayesian filter (see §2), all filter algorithms (§3-§6), and the smoothers (§8).

---

## 2. Bayes recursion (foundational)

The exact Bayes filter for (M1)-(M3) is:

```
Prediction (Chapman-Kolmogorov):
    p(x_k | Y_{k-1}) = ∫ p(x_k | x_{k-1}) p(x_{k-1} | Y_{k-1}) dx_{k-1}                    (B1)

Update (Bayes' rule):
    p(x_k | Y_k) = p(y_k | x_k) p(x_k | Y_{k-1}) / p(y_k | Y_{k-1})                        (B2)

Marginal likelihood:
    p(y_k | Y_{k-1}) = ∫ p(y_k | x_k) p(x_k | Y_{k-1}) dx_k                                (B3)
```

with `Y_k = {y_1, ..., y_k}`.

**Every filter in this project is an approximation to (B1)-(B2):**

| Filter | Approximation of (B1) | Approximation of (B2) |
|--------|----------------------|----------------------|
| EKF | First-order Taylor of `f` about x̂_{k-1|k-1}: Gaussian remains Gaussian to O(‖Δx‖) | First-order Taylor of `h` about x̂_{k|k-1} |
| UKF | Deterministic sigma-point propagation (unscented transform); Gaussian moment matching to third order for symmetric distributions | Same UT applied to `h`; Kalman-style linear-Gaussian update |
| SRUKF | Same UT, but propagates Cholesky factor `S` (with `P = S Sᵀ`) via QR + rank-1 Cholesky updates | Same, with square-root Kalman gain via triangular solve |
| PKF | Empirical sample from proposal `q ≡ p(x_k|x_{k-1})` (bootstrap) | Importance-weight update `w ∝ p(y|x)` + resampling |
| RBPKF | Particles for `x^{nl}`; conditional KF prediction for `x^{lin}` | Particles reweighted by conditional likelihood `p(y|x^{nl}, Y_{k-1})`; KF update for `x^{lin}` |

Dimensional consistency (M1)-(M2), (B1)-(B3): probability densities are dimensionless per Jacobian volume element; every argument of `p(⋅|⋅)` matches the corresponding state or observation dimensions. Verified.

---

## 3. Extended Kalman Filter (EKF)

**Locations:** `EKF/include/EKF.h` (lines 12-46), `EKF/src/EKF.cpp` (lines 24-90); Joseph-form update lines 75-89.

### 3.1 Prediction

Linearize `f` about the current posterior mean:

```
F_k       = ∂f/∂x |_{x̂_{k-1|k-1}, u_k, t_k}
x̂_{k|k-1} = f(x̂_{k-1|k-1}, u_k, t_k)                                                       (E1)
P_{k|k-1} = F_k P_{k-1|k-1} F_kᵀ + Q(t_k)                                                   (E2)
```

Implementation in `EKF.cpp` (lines 25-46): `filtermath::gemm` for the two matrix products, explicit symmetrization `P = 0.5 (P + Pᵀ)` at line 40.

### 3.2 Update (Joseph form)

Linearize `h` about the predicted mean:

```
H_k       = ∂h/∂x |_{x̂_{k|k-1}, t_k}
ν_k       = y_k − h(x̂_{k|k-1}, t_k)                          (innovation)                   (E3)
S_k       = H_k P_{k|k-1} H_kᵀ + R(t_k)                       (innovation covariance)       (E4)
K_k       = P_{k|k-1} H_kᵀ S_k^{-1}                           (Kalman gain, via SPD solve)  (E5)
x̂_{k|k}   = x̂_{k|k-1} + K_k ν_k                                                            (E6)
P_{k|k}   = (I − K_k H_k) P_{k|k-1} (I − K_k H_k)ᵀ + K_k R(t_k) K_kᵀ    (Joseph form)       (E7)
```

Implementation `EKF.cpp` lines 55-89. `K_k` computed via `filtermath::kalman_gain(P Hᵀ, S)` (SPD solve, avoids explicit inverse — O(n²) instead of O(n³)). Joseph form (E7) is used unconditionally rather than the shorter `P_{k|k} = (I − K H) P_{k|k-1}` because it stays PSD under round-off; symmetrized at line 89.

### 3.3 Regime of validity

- EKF is exact when both `f` and `h` are affine; otherwise the linearization error is O(‖P_{k|k-1}‖) in the state mean and O(‖P_{k|k-1}‖²) in the covariance.
- Breaks down when: (a) `f` or `h` has significant curvature within the credible region of `x_{k|k-1}`; (b) the Jacobians `F`, `H` are unavailable or expensive; (c) multimodal posteriors arise (EKF is unimodal by construction).

### 3.4 EKF Jacobians in this project

| Model | Analytic `F` | Analytic `H` |
|-------|--------------|--------------|
| `NonlinearOscillator` (`EKF/include/NonlinearOscillator.h` lines 56-77) | Euler-step Jacobian with `∂f₁/∂pos = −ω² cos(pos)·dt`, `∂f₁/∂vel = 1 − damping·dt` | `[1, 0]` (observes position only) |
| `BallTossModel` (`EKF/include/BallTossModel.h` lines 36-49) | Linear constant-velocity + gravity: identity plus `F(0,2)=F(1,3)=dt` | `[[1,0,0,0],[0,1,0,0]]` |

---

## 4. Unscented Kalman Filter (UKF)

**Locations:** `UKF/include/UKF.h` (lines 12-197), `UKF/include/SigmaPoints.h` (lines 12-186).

### 4.1 Merwe scaled unscented transform

**Sigma-point set** (symmetric, `2n+1` points, where `n = NX`):

```
λ = α² (n + κ) − n                                                                          (U1)
X^{(0)} = x̂
X^{(i)}    = x̂ + √(n+λ) · L_i,   i = 1..n                                                   (U2)
X^{(i+n)}  = x̂ − √(n+λ) · L_i,   i = 1..n
```

where `L` is a Cholesky factor of `P` (i.e. `P = L Lᵀ`) — computed via `filtermath::cholesky` in `SigmaPoints.h` (line 69), with an LDLT + jitter + diagonal-clamp recovery ladder (lines 76-99).

**Weights** (`SigmaPoints.h` lines 58-66):

```
W_m^{(0)} = λ / (n + λ)                          (mean weight, center point)                (U3)
W_c^{(0)} = λ / (n + λ) + (1 − α² + β)           (covariance weight, center point)          (U4)
W_m^{(i)} = W_c^{(i)} = 1 / [2(n + λ)],   i ≥ 1                                             (U5)
```

**Degenerate-spread guard** (`SigmaPoints.h` lines 49-54): if `n + λ < 0.5`, reset `κ = n/α² − n` so `n + λ = n`.

### 4.2 Tuning parameters

- **α** (spread of sigma points around mean). Small α → tight cluster (captures local curvature); large α → wider (captures global behavior). *Standard textbook default* α = 10⁻³.
- **β** (prior distribution knowledge). β = 2 is optimal for a Gaussian prior.
- **κ** (secondary scaling). Classical Julier-Uhlmann: `κ = 3 − n`.

**Project-specific choices (SRUKF, dimension-adaptive):** see `UKF/include/SRUKF.h` lines 70-79:

```
if (NX <= 5):   α = 1.0,  κ = 3 − NX,   β = 2.0
else       :   α = 1.0,  κ = 0,         β = 2.0
```

**UKF (plain) default:** `α = 10⁻³`, `β = 2.0`, `κ = 0.0` (`UKF/include/UKF.h` lines 24-26). NOTE: the UKF class initializes with the "classical" α = 10⁻³ and does *not* apply the dimension-adaptive logic. The SRUKF's dimension adaptation was the fix for the 10D coupled-oscillator NaN failure recorded in `COMPARISON_RESULTS.md` "Bug #1"; the UKF has since been re-verified to run stably on the same benchmark, so its default α is now inconsistent with the SRUKF but empirically works.

### 4.3 Prediction (moment matching through `f`)

```
X_k^{(i)|k-1} = f(X_{k-1}^{(i)|k-1}, u_k, t_k)                                              (U6)
x̂_{k|k-1}   = Σ_i W_m^{(i)} X_k^{(i)|k-1}                                                   (U7)
P_{k|k-1}   = Σ_i W_c^{(i)} (X_k^{(i)|k-1} − x̂_{k|k-1}) (X_k^{(i)|k-1} − x̂_{k|k-1})ᵀ + Q   (U8)
P_{x_{k-1}, x_k} = Σ_i W_c^{(i)} (X_{k-1}^{(i)|k-1} − x̂_{k-1|k-1}) (X_k^{(i)|k-1} − x̂_{k|k-1})ᵀ    (U9)
```

(U9) is the cross-covariance retained for RTS smoothing — see `UKF.h` lines 61-84 (`Dx_w * Dp.transpose()`).

Implementation: `filtermath::gemm(Dp_w, Dp.transpose())` at `UKF.h` line 72 does the (U8) sum as a single GEMM after building weighted-diff matrices.

### 4.4 Update (moment matching through `h`)

```
Y_k^{(i)}       = h(X_k^{(i)|k-1}, t_k)                                                     (U10)
ŷ_k             = Σ_i W_m^{(i)} Y_k^{(i)}                                                   (U11)
S_k             = Σ_i W_c^{(i)} (Y_k^{(i)} − ŷ_k) (Y_k^{(i)} − ŷ_k)ᵀ + R                    (U12)
P_{xy,k}        = Σ_i W_c^{(i)} (X_k^{(i)|k-1} − x̂_{k|k-1}) (Y_k^{(i)} − ŷ_k)ᵀ             (U13)
K_k             = P_{xy,k} S_k^{-1}                    (via SPD solve)                      (U14)
x̂_{k|k}         = x̂_{k|k-1} + K_k (y_k − ŷ_k)                                              (U15)
P_{k|k}         = P_{k|k-1} − K_k P_{xy,k}ᵀ − P_{xy,k} K_kᵀ + K_k S_k K_kᵀ  (Joseph, symm.) (U16)
```

Joseph-symmetric form (U16) is implemented at `UKF.h` lines 141-145, followed by an LLT / LLT+relative-jitter / LDLT / diagonal-clamp recovery ladder (lines 150-178). (U16) is algebraically equal to `P_{k|k-1} − K_k S_k K_kᵀ` (since `P_{xy,k} = K_k S_k`) but symmetric by construction; matches SRUKF's numerical defenses.

### 4.5 Regime of validity

- UT captures moments up to third order for Gaussian priors and any nonlinear `f`, `h`; no Jacobians required.
- Weights `W_m^{(i)}` sum to 1 (verified by construction — `SigmaPoints.h` lines 59, 62-66). `W_c^{(0)}` can be negative (that is intended); the SRUKF's rank-1 downdate machinery handles this correctly (§5.4).

---

## 5. Square-Root UKF (SRUKF)

**Locations:** `UKF/include/SRUKF.h` (lines 28-696), `UKF/include/SigmaPoints.h` `generate_sigma_points_from_sqrt` (lines 146-184).

### 5.1 Motivation

The plain UKF propagates the full covariance `P`, which can lose positive-definiteness under floating-point round-off (particularly at `float` precision on high-dimensional or ill-conditioned problems). The SRUKF instead propagates a triangular Cholesky factor `S` such that

```
P = S Sᵀ                                                                                    (S1)
```

`S` is guaranteed real-valued and lower-triangular; PSD of `P` is preserved by construction.

### 5.2 Sigma-point generation from `S`

Same formulas as (U1)-(U5) but using `S` directly in (U2) instead of decomposing `P`. See `SigmaPoints.h` lines 146-184.

### 5.3 Predicted square-root covariance via QR + rank-1 update

Given weighted, mean-subtracted sigma points `χᵢ = √|W_c^{(i)}| (X_k^{(i)|k-1} − x̂_{k|k-1})` for `i = 1..2n` and a Cholesky factor `S_Q` of the process-noise covariance `Q`, form the compound matrix

```
A = [ χ_1  χ_2  ...  χ_{2n}  S_Q ]     (NX × (3·NX))                                        (S2)
```

QR-decompose `Aᵀ = Q_QR R`; then `Aᵀ = Q_QR R` gives `A Aᵀ = Rᵀ Q_QRᵀ Q_QR R = Rᵀ R`, so `Rᵀ` is a valid Cholesky factor of `A Aᵀ`. Take `S_pred = Rᵀ` (lower triangular).

Implementation at `SRUKF.h` lines 214-233. Note the shape trick: `chi_diff_T` is stored as `(3·NX) × NX` (transposed), then `Eigen::HouseholderQR` is applied.

**Central sigma point** (`i = 0`), whose `W_c^{(0)}` can be negative, is handled by a rank-1 Cholesky update or downdate afterwards:

- If `W_c^{(0)} ≥ 0`: **rank-1 update** — S_pred ← S_pred s.t. S_pred S_predᵀ = old + W_c^{(0)} diff_0 diff_0ᵀ. Implemented in `cholupdate` (`SRUKF.h` lines 563-581).
- If `W_c^{(0)} < 0`: **rank-1 downdate** — S_pred ← S_pred s.t. S_pred S_predᵀ = old − |W_c^{(0)}| diff_0 diff_0ᵀ. Implemented in `cholupdate_downdate_safe` (lines 588-612): returns `false` when downdate would produce non-PSD result. On failure, falls back to full-covariance recomputation + fresh Cholesky (lines 247-268).

### 5.4 Square-root update step

Instead of a QR on the innovation direction, this SRUKF **computes `P_yy = Σ_i W_c^{(i)} (Y^{(i)} − ŷ) (Y^{(i)} − ŷ)ᵀ + R` directly** and takes its Cholesky (`SRUKF.h` lines 325-358). Direct computation was adopted because QR of the 1×N matrix (small NY) degenerates — see `COMPARISON_RESULTS.md` "Bug #3".

Kalman gain via **triangular solves** (avoids explicit inverse):

```
S_yy S_yyᵀ K^T = P_xyᵀ    →    K^T obtained by two triangular solves                       (S3)
```

Implementation `SRUKF.h` lines 403-408 (forward solve on lower, back solve on upper).

### 5.5 Innovation gating and NIS (R32/R33)

**Normalized innovation squared** (NIS):

```
NIS_k = ν_kᵀ S_k^{-1} ν_k = ‖S_yy^{-1} ν_k‖²                                                (S4)
```

Implementation `SRUKF.h` line 427 via triangular solve `temp_innov = S_yy^{-1} ν_k`. Under the assumption of a consistent Gaussian filter, NIS ~ χ²(NY). Default gate threshold **25.0** (roughly 3σ for NY=3); configurable via `setInnovationGateChi2()` (line 49). When `NIS_k > gate`:

```
scale = √(gate / NIS_k)     if reject_outliers_ = false  (default)
scale = 0                   if reject_outliers_ = true                                      (S5)
correction = scale · (K ν_k)
```

**Joseph-consistent covariance downdate under gated correction** (`SRUKF.h` lines 461-474): if the effective gain is `scale · K`, the Joseph partial-update reduces the covariance by `(2·scale − scale²) K P_yy Kᵀ`; the square-root downdate scales each column of `K S_yy` by `√(2·scale − scale²)`. Endpoints: `scale=1` → full update, `scale=0` → zero downdate (rejected measurement doesn't shrink covariance into false certainty).

### 5.6 Angular-observation innovation wrap (R32)

For observation indices where `model.isAngularObservation(j)` returns true, the innovation and every `(Y^{(i)} − ŷ)` are wrapped to `[−π, π]` via `std::remainder(x, 2π)` (SRUKF.h lines 379-386, 416-420). Prevents an innovation straddling the ±π branch cut from being read as ~2π instead of ~0.

Similarly for angular *states*: `isAngularState(j)` triggers `remainder(2π)` on state differences and a circular mean `atan2(Σ Wm sin, Σ Wm cos)` for the mean update (SRUKF.h lines 174-183, 218-222, 279-285).

### 5.7 SRUKF equations summary

Same (U6), (U10), (U11), (U15) as UKF. Structural changes:

- (U8) replaced by (S2) + rank-1 update: propagates `S_pred` not `P_pred`.
- (U12) computed directly then Cholesky-decomposed to `S_yy` (rather than a second QR).
- (U14) via triangular solves (S3), not SPD solve on full `S`.
- (U16) replaced by rank-`NY` sequential downdate `S_updated` ← rank-1 downdate on each column of `√(2s−s²) K S_yy` (lines 473-481); full-covariance fallback on failure (lines 483-522).

---

## 6. Rao-Blackwellized Particle Filter (RBPKF)

**Locations:** `RBPKF/include/rbpf/rbpf_core.hpp` (lines 22-387), `RBPKF/include/rbpf/kalman_filter.hpp` (lines 8-104).

### 6.1 Marginalization structure

The state is factored as `x_k = (x_k^{nl}, x_k^{lin})` where:

- `x_k^{nl}` (nonlinear sub-state) enters the model in a general nonlinear way; represented by Monte Carlo particles.
- `x_k^{lin}` (linear sub-state) evolves according to a **conditional linear-Gaussian model** given the trajectory `x_{0:k}^{nl}`:

```
x_k^{lin}  = A(x_{k-1}^{nl}) x_{k-1}^{lin} + B(x_{k-1}^{nl}) u_k + bias(x_{k-1}^{nl}) + w^{lin}
                                                                                          (R1)
y_k        = H(x_k^{nl}) x_k^{lin} + offset(x_k^{nl}) + v_k                                (R2)
```

Then the posterior factorizes exactly as:

```
p(x_k^{nl}, x_k^{lin} | Y_k) = p(x_k^{lin} | x_{0:k}^{nl}, Y_k) · p(x_{0:k}^{nl} | Y_k)     (R3)
```

The first factor is Gaussian (analytically tractable Kalman filter per particle); the second is approximated by weighted samples. This is the **Rao-Blackwellization**: variance reduction relative to a plain PF on `(x^{nl}, x^{lin})` jointly.

### 6.2 Per-particle Kalman filter (linear sub-state)

Each particle `i` carries `(x_{k,i}^{nl}, x_{k,i}^{lin}, P_{k,i}^{lin}, w_i)`. The KF steps are `LinearKalmanFilter::predict/update` (`RBPKF/include/rbpf/kalman_filter.hpp` lines 41-101):

```
predict:  x^{lin}_{k|k-1} = A x^{lin}_{k-1|k-1} + bias                                     (R4)
          P^{lin}_{k|k-1} = A P^{lin}_{k-1|k-1} Aᵀ + Q                                     (R5)
update:   S = H P^{lin}_{k|k-1} Hᵀ + R                                                     (R6)
          K = P^{lin}_{k|k-1} Hᵀ S^{-1}    (SPD solve)                                     (R7)
          x^{lin}_{k|k} = x^{lin}_{k|k-1} + K (y_k − H x^{lin}_{k|k-1} − offset)           (R8)
          P^{lin}_{k|k} = (I − KH) P^{lin}_{k|k-1} (I − KH)ᵀ + K R Kᵀ   (Joseph form)      (R9)
```

Fixed-size Eigen types are used for `A`, `H` (see `RBPKF/include/rbpf/types.hpp` lines 33-35) so each per-particle GEMM binds to the `filtermath` compile-time fast path.

### 6.3 Particle weight update

For each particle `i`, the conditional predictive likelihood of `y_k` is Gaussian with mean `H x^{lin}_{k|k-1} + offset` and covariance `S = H P^{lin}_{k|k-1} Hᵀ + R`:

```
log p(y_k | x_i^{nl}, Y_{k-1}) = − ½ [ (y_k − ŷ_i)ᵀ S_i^{-1} (y_k − ŷ_i) + log det S_i + NY · log 2π ]  (R10)
```

Implementation `rbpf_core.hpp` lines 156-191. `log det S` computed via LDLT diagonal-product with a `1e-30f` clamp (lines 170-181), not `det(S)` directly (float32 underflows silently at high condition number). Mahalanobis quadratic form via SPD solve (line 184).

### 6.4 Resampling and ancestry

Weights normalized via log-sum-exp (`rbpf_core.hpp` lines 308-339). ESS `= 1 / Σ w_i²` (line 342); resample when `ESS < resampling_threshold · N`. Systematic or stratified resampling (`RBPKF/src/resampling.cpp`, called from lines 208-212), with Kahan compensated summation for the cumulative-weight step to prevent O(N) rounding bias.

Fixed-lag smoothing via **ancestry backtracking** through a circular buffer of parent indices (`rbpf_core.hpp` lines 252-285).

---

## 7. Bootstrap Particle Filter (PKF)

**Location:** `PKF/include/particle_filter.hpp` (lines 26-437).

### 7.1 Sequential importance sampling with resampling (SIR)

Represents the posterior as a weighted point measure:

```
p̂(x_k | Y_k) = Σ_{i=1..N} w_k^{(i)} δ(x_k − X_k^{(i)})                                     (P1)
```

Bootstrap proposal: `q(x_k | x_{k-1}, y_k) = p(x_k | x_{k-1})` (the transition prior). This gives the **incremental weight update**:

```
X_k^{(i)}       ~ p(x_k | X_{k-1}^{(i)})                                                   (P2)
log w_k^{(i)}   = log w_{k-1}^{(i)} + log p(y_k | X_k^{(i)})                               (P3)
```

Log-weight formulation avoids underflow — see `particle_filter.hpp` lines 163-166.

### 7.2 Normalization (log-sum-exp)

```
w_k^{(i)}_normalized = exp( log w_k^{(i)} − LSE_i(log w_k^{(i)}) )                          (P4)
```

Implementation `particle_filter.hpp` lines 386-432. Degenerate case (all weights `−∞`): reset to uniform.

### 7.3 Effective sample size and resampling

```
ESS_k = 1 / Σ (w_k^{(i)})²                                                                 (P5)
```

Computed with log-sum-exp trick at `particle_filter.hpp` lines 177-193, clamped to `[1, N]`. Resample when `ESS_k < ρ · N` (ρ = `resampling_threshold`, default 0.5). Systematic or stratified resampling from `PKF/include/resampling.hpp`.

### 7.4 Non-Gaussian likelihoods

The PKF interface (`observation_loglik`) admits any log-density. **In the reference Lorenz-63 test model, `p(y | x)` is Student-t with `ν = 3` degrees of freedom** (`PKF/include/lorenz63_model.hpp` lines 33-34, 90-95), i.e. heavy-tailed:

```
log p(y | x) = log StudentT(y − h(x); 0, R, ν = 3)                                          (P6)
```

This is one place where the PKF materially exceeds the UKF/SRUKF: EKF/UKF/SRUKF are all Gaussian-noise filters.

---

## 8. RTS smoothing (fixed-lag and full-interval, iterated)

### 8.1 Rauch-Tung-Striebel (RTS) recursion (general)

Given filtered pairs `{(x̂_{j|j}, P_{j|j})}_{j=0..N}` and the one-step-ahead cross-covariance `P_{x_j, x_{j+1}}`, the **backward smoothing recursion** is:

```
G_j          = P_{x_j, x_{j+1}} P_{j+1|j}^{-1}          (smoothing gain, via SPD solve)     (T1)
x̂_{j|N}     = x̂_{j|j} + G_j (x̂_{j+1|N} − x̂_{j+1|j})                                       (T2)
P_{j|N}      = P_{j|j} + G_j (P_{j+1|N} − P_{j+1|j}) G_jᵀ                                   (T3)
```

Initial condition: `x̂_{N|N} = x̂_{N|N}`, `P_{N|N} = P_{N|N}`.

Dimensional consistency: `G_j` is dimensionless (state × state^{-1}); (T2), (T3) preserve state units and squared-state units respectively. Verified.

### 8.2 UKF/SRUKF fixed-lag smoother

**Locations:** `UKF/include/UnscentedFixedLagSmoother.h` (lines 30-213); `UKF/include/SRUKFFixedLagSmoother.h`.

The cross-covariance `P_{x_j, x_{j+1}}` (needed at (T1)) is the (U9) quantity computed and returned by `UKF::predict()`. The smoother keeps a `deque<UKFHistoryEntry>` sized to `lag + 2`, and on each `step()`:

1. Advance UKF one step, save `(x̂_{k+1|k}, P_{k+1|k}, P_{x_k, x_{k+1}}, x̂_{k+1|k+1}, P_{k+1|k+1})`.
2. Trim history to `lag + 2` entries.
3. Run the RTS backward pass over the buffer (T1)-(T3).

`get_smoothed_state(lag)` returns `x̂_{k-lag | k}` — the smoothed estimate `lag` steps back from the current filter time.

Implementation `perform_smoothing()` lines 167-209. Smoothed covariance symmetrized (line 207) to prevent round-off asymmetry.

### 8.3 Full-interval (batch) URTSS + iterated

**Locations:** `UKF/include/UKFSmoother.h` (lines 24-124), `UKF/include/SRUKFSmoother.h`.

Same (T1)-(T3) but stores the *entire* forward trajectory + all measurements, runs a single backward pass over the full length. `smooth(n_iterations)` optionally re-runs the forward filter starting from the smoothed initial state (with the original prior covariance) and re-smooths, re-linearising sigma points about a better trajectory:

```
Iteration 0:  forward filter → backward RTS → (x̂^{(0)}_{j|N})_{j=0..N}
Iteration l:  re-initialize x̂_{0|0} := x̂^{(l-1)}_{0|N},  P_{0|0} := P_0 (original prior)
              forward filter → backward RTS → (x̂^{(l)}_{j|N})
```

`smooth(0)` = single RTS pass; `smooth(n>0)` = iterated smoother (an IEKS in the EKF case). See `EKFSmoother.h` lines 44-53 for the EKF variant, `UKFSmoother.h` lines 47-56 for the UKF, and `SRUKFSmoother.h` for the SRUKF (square-root variant, same math on `S`-form). The original prior is retained so the data still dominate the smoothed trajectory.

### 8.4 EKF fixed-lag smoother

**Location:** `EKF/include/EKFFixedLag.h` (interface used by `EKF/include/EKFSmoother.h`). The RTS gain uses the stored Jacobian `F_{j+1}`:

```
G_j = P_{j|j} F_{j+1}ᵀ P_{j+1|j}^{-1}                                                       (T4)
```

Same (T2)-(T3). Implementation `EKFSmoother.h` lines 100-118.

### 8.5 PKF ancestry-based smoother

**Location:** `PKF/include/particle_fixed_lag.hpp` (via `particle_filter.hpp`).

Different from RTS: reconstructs each particle's trajectory `L` steps back by following parent indices from resampling. The smoothed marginal at time `k − L` is `Σ_i w_k^{(i)} X_{k−L}^{(ancestor(i))}` — see `RBPF/rbpf_core.hpp` lines 252-285 for the analogous RBPF version.

### 8.6 RBPKF fixed-lag smoother

Same ancestry-backtracking scheme on the nonlinear sub-state; the linear sub-state at each particle is Kalman-smoothed exactly (analytically) — a hybrid RTS + PF smoother.

---

## 9. Test-problem physics

### 9.1 Van der Pol oscillator (`Benchmarks/include/BenchmarkProblems.h` lines 189-253, class `VanDerPolDiscontinuous`)

**State:** `x = (x_1, x_2)ᵀ` with x_1 = position, x_2 = velocity. `NX=2, NY=1`.

**Dynamics** (line 246-249):

```
dx_1/dt = x_2
dx_2/dt = μ (1 − x_1²) x_2 − x_1 + F(t)                                                    (V1)
```

with **stiffness parameter μ = 5** (line 201) — the "large μ" regime; and **discontinuous square-wave forcing** F(t) = +1 for `mod(t,2) < 1`, −1 otherwise (line 243). RK4 integration with dt = 0.01.

**Observation:** `y = x_1 + 0.2 x_1²` (line 221). Nonlinear (quadratic) measurement.

**Noise:** `Q = diag(0.001, 0.01)`, `R = 0.2` (lines 225-234).

**Physics regime:** relaxation oscillator; slow drift interrupted by sharp limit-cycle transitions. Textbook nonlinear-dynamics benchmark for stiff ODE and non-smooth forcing.

### 9.2 Coupled oscillators (`Benchmarks/include/BenchmarkProblems.h` lines 17-111, class `CoupledOscillators`)

**State:** `x ∈ ℝ^{10}` = 5 pairs `(pos_i, vel_i)`, `NX=10, NY=5`.

**Dynamics** (lines 84-107):

```
d pos_i / dt = vel_i
d vel_i / dt = − ω² sin(pos_i) − γ vel_i + Σ_{j≠i} [κ (pos_j − pos_i) + η sin(pos_j − pos_i)]   (V2)
```

with **ω = 2.0** (natural freq), **γ = 0.1** (damping), **κ = 0.5** (linear coupling), **η = 0.3** (nonlinear coupling) — all set on lines 29-32. RK4 with dt = 0.01. Every oscillator couples to every other (all-to-all).

**Observation:** positions only, with mild nonlinearity: `y_i = pos_i + 0.1 sin(pos_i)` (line 58).

**Noise:** `Q_{ii} = 0.001` (positions), `Q_{ii} = 0.01` (velocities); `R = 0.1 I` (lines 63-74).

**Physics regime:** all-to-all coupled Kuramoto-like network with pendulum nonlinearity and damping. Numerically demanding due to 10D state, cross-coupling, and full-rank information from only 5 observations (velocities are unobserved, must be inferred).

### 9.3 Bearings-only tracking (`Benchmarks/include/BenchmarkProblems.h` lines 405-466, class `BearingOnlyTracking`)

**State:** `x = (px, py, vx, vy)ᵀ` target in 2D Cartesian, `NX=4, NY=1`.

**Dynamics** (lines 427-435): **constant-velocity** discrete-time,

```
px_{k+1} = px_k + dt · vx_k,   py_{k+1} = py_k + dt · vy_k,   vx_{k+1} = vx_k,   vy_{k+1} = vy_k    (V3)
```

with dt = 0.1 (line 416).

**Observation:** **bearing angle** from a moving observer platform:

```
observer position:  (100 cos(Ω t), 100 sin(Ω t)) with Ω = 0.1 rad/s (line 421-424)
y = atan2(py − obs_y, px − obs_x)                                                          (V4)
```

**Noise:** `Q_{22} = Q_{33} = 0.1` (velocity acceleration noise); `R = 0.01 rad²` (bearing noise). Position states have zero process noise (line 453-459).

**Physics regime:** classic weakly observable problem — a bearing gives an angle but not a range. The observer *must* maneuver (hence the circular path) to gain observability; range converges very slowly. Steady-state RMSE ~64 m in this configuration; NEES ~3.77, in-bounds 99.6%. See `COMPARISON_RESULTS.md` for prior configurations.

### 9.4 Reentry vehicle (`Benchmarks/include/BenchmarkProblems.h` lines 262-396, class `ReentryVehicle`)

**State:** `x = (x_1, x_2, x_3, vx, vy, vz)ᵀ` in Earth-centered coordinates, `NX=6, NY=3`.

**Dynamics** (`dynamics()` lines 353-395):

```
r        = ‖(x_1, x_2, x_3)‖
altitude = r − R₀                                                                          (V5)
ρ(h)     = ρ₀ exp(−altitude / H₀)                                                          (V6)  [exponential atmosphere]
v        = ‖(vx, vy, vz)‖
a_drag   = − (ρ v / (2 BC)) · v                                                            (V7)  [quadratic drag]
a_grav   = − μ_earth / r³ · (x_1, x_2, x_3)                                                (V8)  [Newtonian gravity, altitude-dep.]
dx/dt = v_vec,   dv/dt = a_drag + a_grav                                                   (V9)
```

**Physical constants** (lines 274-278):

| Symbol | Value | Meaning |
|--------|-------|---------|
| R₀ | 6.371 × 10⁶ m | Earth radius |
| H₀ | 9 000 m | Atmospheric scale height |
| ρ₀ | 1.225 kg/m³ | Sea-level density |
| BC | 500 kg/m² | Ballistic coefficient |
| μ_earth | 3.986 004 418 × 10¹⁴ m³/s² | Earth gravitational parameter |

RK4 integration with dt = 0.1 s (line 273).

**Observation:** ground-based radar at `(R₀, 0, 0)` producing `(range, azimuth, elevation)`:

```
Δ = pos − radar_pos
range     = ‖Δ‖
azimuth   = atan2(Δ_y, Δ_x)                                                                (V10)
elevation = asin( clamp(Δ_z / range, −1, 1) )
```

Argument to `asin` is clamped (line 321) to prevent NaN from floating-point overshoot.

**Noise:** `Q_{pos,pos} = 100 m²`, `Q_{vel,vel} = 1000 (m/s)²` (lines 330-334); `R = diag(10000 m², 10⁻⁴ rad², 10⁻⁴ rad²)` (lines 337-343).

**Divergence threshold:** 5 000 m position/velocity, set in `getDivergenceThreshold()` line 347 (problem-scaled — a fixed 10 m threshold was the source of the false "176 divergences" reported for bearings-only in older docs).

**Dimensional analysis** of (V7)-(V8):

- `[ρ] = kg/m³`, `[v/BC] = (m/s) / (kg/m²) = m³ / (kg·s)`
- `[ρ v / BC · v] = (kg/m³) · (m³/(kg·s)) · (m/s) = m/s²` ✓ (acceleration)
- `[μ_earth / r³] = m³/s² / m³ = s⁻²`; `[μ_earth / r³ · pos] = (1/s²) · m = m/s²` ✓

### 9.5 Drag-ball (`UKF/include/DragBallModel.h`)

**State:** `(px, py, vx, vy)ᵀ`, `NX=4, NY=2` (position observed).

**Dynamics** (lines 34-49): 2D ballistic with quadratic air drag and gravity:

```
v = √(vx² + vy²)
px_{k+1} = px_k + vx_k dt
py_{k+1} = py_k + vy_k dt                                                                  (V11)
vx_{k+1} = vx_k − C v vx_k · dt
vy_{k+1} = vy_k − C v vy_k · dt − g · dt
```

Default `C = 0.001` (drag coeff), `g = 9.81 m/s²`, `dt = 0.1`.

### 9.6 Nonlinear (pendulum) oscillator (`EKF/include/NonlinearOscillator.h`)

**State:** `(pos, vel)ᵀ`, `NX=2, NY=1`.

**Dynamics** (lines 27-46): Euler-integrated damped pendulum:

```
pos_{k+1} = pos_k + vel_k · dt
vel_{k+1} = vel_k + (−ω² sin(pos_k) − γ vel_k) · dt   (+ control u · dt on vel)             (V12)
```

with `ω² = 2.0`, `γ = 0.5`, `dt = 0.01` (lines 22-25). Observation: `y = pos` (line 49-53).

Analytic Jacobians provided at lines 56-77 (used by the EKF).

### 9.7 Lorenz-63 (`PKF/include/lorenz63_model.hpp`)

**State:** `(x, y, z)ᵀ`, `NX=NY=3`.

**Dynamics** (lines 50-59):

```
dx/dt = σ (y − x)
dy/dt = x (ρ − z) − y                                                                       (V13)
dz/dt = x y − β z
```

with **classical chaotic parameters** σ = 10, ρ = 28, β = 8/3 (lines 26-28); RK4 integration, dt = 0.01.

**Observation:** full state `y = x` (identity, line 70-73).

**Noise:** process noise is Gaussian random walk `~ N(0, dt · σ_p² I)` with σ_p = 1.0 (line 32, applied via Cholesky factor at line 38); **observation noise is Student-t with ν = 3 degrees of freedom** and scale `R = σ_o² I` with σ_o = 2 (lines 33-34, 84-88). The PKF handles this heavy-tailed likelihood exactly through `observation_loglik`.

### 9.8 CTRV (constant turn-rate + velocity) for RBPKF (`RBPKF/examples/example_rbpf_ctrv.cpp`)

**Nonlinear sub-state:** `ω` (turn rate); **Linear sub-state:** `(x, y, vx, vy)` (position + velocity); observation `y = (x, y)`. Types: `RbpfTypes<1, 4, 2>`.

**Nonlinear dynamics** (`CtrvNonlinearModel`): random walk `ω_k = ω_{k-1} + η_k, η_k ~ N(0, σ_ω²)`.

**Linear dynamics conditional on ω** (`CtrvConditionalModel::get_dynamics` lines 66-100):

```
if |ω| < 1e-5:        A = constant-velocity matrix (linear CV limit)
else:                 A = CTRV rotation matrix parameterized by (ω, dt):
                          A(0,2) = sin(ωdt)/ω,      A(0,3) = −(1−cos(ωdt))/ω
                          A(1,2) = (1−cos(ωdt))/ω,  A(1,3) = sin(ωdt)/ω                    (V14)
                          A(2,2) = cos(ωdt),        A(2,3) = −sin(ωdt)
                          A(3,2) = sin(ωdt),        A(3,3) = cos(ωdt)
```

Process noise `Q_{vx,vx} = Q_{vy,vy} = σ_a² dt`. Observation `H` picks off `(x, y)` with `R = σ_r² I`.

---

## 10. Equation catalog (project-wide)

Standardized format for downstream consistency checking. Every equation is dimensionally verified (all quantities in SI unless overridden by the specific model).

| ID | Equation | Type | Location | Dimensions | Status | Depends On | Used By |
|----|----------|------|----------|------------|--------|------------|---------|
| EQ-001 | `x_k = f(x_{k-1}, u_k, t_k) + w_k` | Defining | `Common/include/StateSpaceModel.h` (lines 31-33) | [state] | Postulated | — | EQ-010, EQ-021, EQ-030, EQ-050 |
| EQ-002 | `y_k = h(x_k, t_k) + v_k` | Defining | `Common/include/StateSpaceModel.h` (lines 39-40) | [obs] | Postulated | — | EQ-014, EQ-024, EQ-032, EQ-053 |
| EQ-003 | `p(x_k|Y_{k-1}) = ∫ p(x_k|x_{k-1}) p(x_{k-1}|Y_{k-1}) dx_{k-1}` (Chapman-Kolmogorov) | Defining | Textbook; not explicit in code | dimensionless (density) | Postulated | EQ-001 | EQ-010, EQ-021, EQ-050 |
| EQ-004 | `p(x_k|Y_k) ∝ p(y_k|x_k) p(x_k|Y_{k-1})` (Bayes) | Defining | Textbook; not explicit in code | dimensionless (density) | Postulated | EQ-002, EQ-003 | EQ-014, EQ-024, EQ-053 |
| EQ-010 | `x̂_{k|k-1} = f(x̂_{k-1|k-1}, u_k, t_k); P_{k|k-1} = F P F^T + Q` (EKF predict) | Derived (linearization of EQ-003) | `EKF/src/EKF.cpp` (lines 24-46) | [state]; [state²] | Derived | EQ-001, EQ-003 | EQ-014 |
| EQ-014 | `K = P H^T S^{-1}; x̂_{k|k} = x̂_{k|k-1} + K(y-h); P_{k|k} = (I-KH) P (I-KH)^T + K R K^T` (EKF Joseph update) | Derived | `EKF/src/EKF.cpp` (lines 55-89) | dimensionless; [state]; [state²] | Verified (Joseph symmetric) | EQ-002, EQ-010 | — |
| EQ-020 | `λ = α²(n+κ) − n` (UT scaling) | Defining | `UKF/include/SigmaPoints.h` (line 46) | dimensionless | Postulated | — | EQ-021 |
| EQ-021 | Sigma points: `X⁽⁰⁾ = x̂; X⁽ⁱ⁾ = x̂ ± √(n+λ) L_i` | Defining | `UKF/include/SigmaPoints.h` (lines 102-106) | [state] | Postulated | EQ-020 | EQ-022 |
| EQ-022 | Weights: `W_m⁽⁰⁾ = λ/(n+λ); W_c⁽⁰⁾ = λ/(n+λ) + (1-α²+β); W⁽ⁱ⁾ = 1/[2(n+λ)]` | Defining | `UKF/include/SigmaPoints.h` (lines 58-66) | dimensionless | Verified (Σ W_m = 1) | EQ-020 | EQ-023 |
| EQ-023 | `x̂ = Σ_i W_m⁽ⁱ⁾ X⁽ⁱ⁾; P = Σ_i W_c⁽ⁱ⁾ (X⁽ⁱ⁾-x̂)(X⁽ⁱ⁾-x̂)^T + Q` (UKF predict moment match) | Derived | `UKF/include/UKF.h` (lines 57-84) | [state]; [state²] | Derived | EQ-021, EQ-022 | EQ-024 |
| EQ-024 | `K = P_xy S^{-1}; P_{k|k} = P_{k|k-1} - K P_xy^T - P_xy K^T + K S K^T` (UKF Joseph update) | Derived | `UKF/include/UKF.h` (lines 124-145) | dimensionless; [state²] | Verified (Joseph symmetric) | EQ-023 | — |
| EQ-030 | `A = [√|W_c⁽ⁱ⁾| (X⁽ⁱ⁾-x̂)  S_Q];  A A^T = R_QR^T R_QR  → S_pred = R_QR^T` (SRUKF QR-based sqrt covariance) | Derived | `UKF/include/SRUKF.h` (lines 210-233) | [state] on chi_diff | Derived | EQ-021 | EQ-031 |
| EQ-031 | Rank-1 update/downdate for W_c⁽⁰⁾ term | Algorithmic | `UKF/include/SRUKF.h` (lines 235-271, `cholupdate`/`cholupdate_downdate_safe` lines 563-612) | [state] on S | Derived (Householder + Givens) | EQ-030 | EQ-032 |
| EQ-032 | Direct P_yy computation + Cholesky (not QR) for innovation covariance | Derived | `UKF/include/SRUKF.h` (lines 324-358) | [obs²] | Derived (numerical robustness — Bug #3) | EQ-021 | EQ-034 |
| EQ-033 | NIS = ‖S_yy^{-1} ν‖² ~ χ²(NY) (SRUKF gating statistic, R33) | Derived | `UKF/include/SRUKF.h` (lines 426-428) | dimensionless | Verified against χ² gate=25 | EQ-032 | EQ-034 |
| EQ-034 | Kalman gain via triangular solves + Joseph downdate scale √(2s-s²) (R33) | Derived | `UKF/include/SRUKF.h` (lines 401-408, 473-474) | dimensionless K, [state²] downdate | Verified (audit v3.2.1) | EQ-032, EQ-033 | — |
| EQ-050 | `X⁽ⁱ⁾ ~ p(x_k|x_{k-1}⁽ⁱ⁾); log w⁽ⁱ⁾ ← log w⁽ⁱ⁾ + log p(y|X⁽ⁱ⁾)` (SIR bootstrap) | Defining | `PKF/include/particle_filter.hpp` (lines 104-166) | [state]; dimensionless | Postulated | EQ-001, EQ-002 | EQ-051, EQ-053 |
| EQ-051 | LSE weight normalization: w⁽ⁱ⁾ = exp(log w⁽ⁱ⁾ − LSE) | Derived | `PKF/include/particle_filter.hpp` (lines 386-432) | dimensionless | Verified (degenerate fallback) | EQ-050 | EQ-052 |
| EQ-052 | ESS = 1 / Σ (w⁽ⁱ⁾)² (Kish); resample if ESS < ρN | Derived | `PKF/include/particle_filter.hpp` (lines 177-193) | dimensionless | Verified | EQ-051 | — |
| EQ-053 | log p(y|x) = Student-t log-density with ν=3 (Lorenz-63) | Defining | `PKF/include/lorenz63_model.hpp` (lines 90-95) | dimensionless (density) | Postulated | — | EQ-050 |
| EQ-060 | Conditional-linear KF predict: x^{lin} = A x^{lin} + bias; P^{lin} = A P^{lin} A^T + Q | Derived | `RBPKF/include/rbpf/kalman_filter.hpp` (lines 41-56) | [state]; [state²] | Derived | EQ-001 (linear part) | EQ-061 |
| EQ-061 | Conditional-linear KF Joseph update | Derived | `RBPKF/include/rbpf/kalman_filter.hpp` (lines 64-101) | [state²] | Verified | EQ-060 | EQ-062 |
| EQ-062 | Particle log-weight increment via Gaussian conditional log-lik (LDLT log-det + Mahalanobis via SPD solve) | Derived | `RBPKF/include/rbpf/rbpf_core.hpp` (lines 156-193) | dimensionless | Verified (LDLT clamp 1e-30f) | EQ-061 | EQ-063 |
| EQ-063 | RBPF resampling with Kahan-compensated cumulative sum | Algorithmic | `RBPKF/src/resampling.cpp` (called from `rbpf_core.hpp` lines 208-212) | dimensionless | Verified (audit v3.2.1) | EQ-062 | — |
| EQ-070 | RTS smoothing gain: G_j = P_{x_j, x_{j+1}} P_{j+1|j}^{-1} | Derived | `UKF/include/UnscentedFixedLagSmoother.h` (lines 191-192), `EKF/include/EKFSmoother.h` (line 111) | dimensionless | Derived | EQ-023, EQ-010 | EQ-071 |
| EQ-071 | x̂_{j|N} = x̂_{j|j} + G_j (x̂_{j+1|N} − x̂_{j+1|j});  P_{j|N} = P_{j|j} + G_j (P_{j+1|N} − P_{j+1|j}) G_j^T | Derived | `UKF/include/UnscentedFixedLagSmoother.h` (lines 194-207) | [state]; [state²] | Verified (post-smoothing symmetrization) | EQ-070 | EQ-072 |
| EQ-072 | Iterated RTS: reinit forward filter at x̂^{(l-1)}_{0|N}, retain P_0, re-smooth | Algorithmic | `UKF/include/UKFSmoother.h` (lines 47-56), `EKF/include/EKFSmoother.h` (lines 44-53) | — | Derived (IEKS analog) | EQ-071 | — |
| EQ-080 | Van der Pol: `dx_2/dt = μ(1−x_1²) x_2 − x_1 + F(t)`, μ=5, F=±1 | Defining | `Benchmarks/include/BenchmarkProblems.h` (lines 246-249) | [1/time] on RHS with x dimensionless | Postulated | — | — |
| EQ-081 | Reentry: `a = − (ρv/2BC) v − (μ_earth/r³) pos` | Defining | `Benchmarks/include/BenchmarkProblems.h` (lines 353-395) | m/s² | Postulated (verified dimensional) | — | — |
| EQ-082 | Exponential atmosphere: ρ(h) = ρ₀ exp(−h/H₀), H₀ = 9000 m | Defining | `Benchmarks/include/BenchmarkProblems.h` (line 367) | kg/m³ | Postulated | — | EQ-081 |
| EQ-083 | Bearings-only: y = atan2(py − obs_y, px − obs_x); obs = (100 cos Ωt, 100 sin Ωt) | Defining | `Benchmarks/include/BenchmarkProblems.h` (lines 420-451) | rad | Postulated | — | — |
| EQ-084 | Lorenz-63: `ẋ = σ(y−x); ẏ = x(ρ−z)−y; ż = xy − βz`; σ=10, ρ=28, β=8/3 | Defining | `PKF/include/lorenz63_model.hpp` (lines 50-59) | 1/time × state | Postulated | — | — |
| EQ-085 | CTRV rotation: A(ω,dt) with A(2,2)=cos(ωdt), A(2,3)=−sin(ωdt), etc. | Defining | `RBPKF/examples/example_rbpf_ctrv.cpp` (lines 80-94) | dimensionless (rotation) + m/s×(1/rad) mixed entries | Postulated | — | — |

**Load-bearing equations** (long "Used By" chains): EQ-001, EQ-002 (defining), EQ-021/EQ-022 (all UT-based filters depend on these). **Status = Verified** entries have been independently confirmed by either the numerical safety checks in the code (e.g. `LLT.info()`, symmetrization) or the audit-remediation regression tests (`test_ukf_numerical`, `test_srukf_initialize`, `test_rbpf_logdet`, `test_particle_const`).

---

## 11. Symmetries and structural properties

Filters do not have "symmetries" in the field-theoretic sense, but the following **invariance/structural** properties are relied upon:

| Property | What it says | Enforcement / where used |
|----------|--------------|--------------------------|
| Covariance PSD | `P = P^T ≥ 0` (equivalently: `∃ L` with `P = L L^T`) | Explicit symmetrization `P ← 0.5(P + P^T)` in every filter after every predict/update (`UKF.h:78,145`, `EKF.cpp:40,89`, `SRUKF.h:334,491`, `kalman_filter.hpp:55,100`). SRUKF uses `S`-form propagation which is PSD by construction. Multi-tier recovery ladders (LLT → jitter → LDLT → diagonal clamp) at `UKF.h:150-178`, `SRUKF.h:498-521`, `SigmaPoints.h:71-99`. |
| Weight normalization | `Σ_i W_m⁽ⁱ⁾ = 1` (UT); `Σ_i w⁽ⁱ⁾ = 1` (PF) | By construction: (U3)+(U5) sum to 1 exactly. PF: LSE normalization after every weight update (`particle_filter.hpp:386-432`). |
| Time reversibility of RTS | Given the same filtered pass, RTS produces a well-defined joint posterior identical to the Bayes marginals under linearity/Gaussianity | Cross-covariance (U9) is the key quantity: correctly captured in `UKF::predict` return value and consumed in `UnscentedFixedLagSmoother::perform_smoothing`. |
| Circular topology on angular quantities | Angles live on `S¹` not ℝ; differences must be wrapped to `[−π, π]` | Handled by `isAngularState` / `isAngularObservation` overrides + `std::remainder(x, 2π)` at all difference sites in SRUKF (§5.6). Circular mean via `atan2(Σ sin, Σ cos)`. Only SRUKF currently implements this; plain UKF does not. See CONCERNS.md flag. |
| Reproducibility (deterministic under seed) | Given a fixed base seed, the filter output is identical across runs | Per-thread RNG seeding via SplitMix64 mix of the base seed (`particle_filter.hpp:357-368`, `rbpf_core.hpp:358-370`); `schedule(static)` on OpenMP loops. Fixed as part of v3.2.1 audit. |
| ThreadSanitizer clean | No worker-vs-worker data races under `#pragma omp parallel for` | Per-particle dynamics/observation matrices declared *inside* the parallel loop body (`rbpf_core.hpp:112-131`). Verified with `-fsanitize=thread`: ~24 races → 0 in the v3.2.1 audit. |

---

## 12. Key results (current state)

Reproduced verbatim from `SRUKF_STATUS.md` (benchmark table verified backend-independent — reproduces on both x86_64 + RTX 5070 Ti and aarch64 Raspberry Pi 5):

| Problem | Filter | RMSE (unsmoothed) | Smoothed RMSE | Median NEES | % in 95% χ² bounds | Divergences |
|---------|--------|-------------------|---------------|-------------|--------------------|-------------|
| Coupled Osc 10D | UKF | 1.457 | 1.148 | 9.89 | 94.5% | 0 |
| Coupled Osc 10D | SRUKF | 1.457 | 1.148 | 9.89 | 94.5% | 0 |
| Van der Pol 2D | UKF | 0.468 | — | 1.14 | 95.9% | 0 |
| Van der Pol 2D | SRUKF | 0.467 | 0.430 | 1.14 | 96.0% | 0 |
| Bearings-only 4D | UKF | 63.49 | — | 3.71 | 99.6% | 0 |
| Bearings-only 4D | SRUKF | 63.84 | 51.68 | 3.72 | 99.6% | 0 |
| Reentry 6D | UKF | 366.9 | — | 4.99 | 95.9% | 0 |
| Reentry 6D | SRUKF | 367.1 | 236.3 | 4.99 | 95.9% | 0 |

Filter/model test regressions (single-run, seed 12345):

| Test | Filter RMSE | Smoothed RMSE | Model |
|------|-------------|---------------|-------|
| EKF | 0.060 | 0.052 | Nonlinear pendulum (2D) |
| UKF | 0.228 | 0.119 | Drag ball (4D) |
| SRUKF | 0.311 | 0.167 | Drag ball (4D) |
| PKF | 0.813 | 0.604 | Lorenz-63 (3D) |

---

## 13. Open derivations / theoretical work

None of the following are known-buggy — they are **derivations that could tighten the theoretical footing but are absent from the code and docs**:

1. **UKF plain default α = 10⁻³ vs SRUKF adaptive α = 1.0** (`UKF/include/UKF.h:24-26` vs `UKF/include/SRUKF.h:70-79`). The SRUKF adopted dimension-adaptive parameters after `COMPARISON_RESULTS.md` Bug #1; the plain UKF has never been retuned. It empirically works on the benchmark, but the disagreement is not derived — it is unexplained.

2. **Iterated-smoother convergence proof.** `smooth(n_iterations)` in `UKFSmoother.h` and `EKFSmoother.h` re-runs the forward pass from the smoothed initial state and re-smooths. This is a **Gauss-Newton fixed-point iteration on the linearization point** (equivalent to the IEKS for the EKF); there is no convergence criterion or divergence guard — the code always runs exactly `n_iterations`. Regime of validity and step-size choice are not derived anywhere in the project.

3. **NIS gate threshold (25.0) vs NY.** `SRUKF::innovation_gate_chi2_ = 25.0f` (line 556) is a "roughly 3σ for NY = 3" default. There is no automatic scaling with `NY`; a NY = 1 gate at χ²_{0.999} would be 10.83, not 25. This is documented as "call `setInnovationGateChi2()` for other NY" but the rule of thumb is not derived. See CONCERNS.md.

4. **Bootstrap PKF is only optimal for Gaussian process noise + tractable-density observation.** The Lorenz-63 test uses non-Gaussian (Student-t) *observation* noise, which is handled — but the process noise is Gaussian. For genuinely non-Gaussian *process* noise the bootstrap proposal is not optimal and an auxiliary PF (APF) or SMC² would be needed. Not currently in scope.

5. **Rao-Blackwellization requires the linear-Gaussian structure to be exact.** The CTRV example has a linear-in-state model conditional on ω, but real vehicles have process-noise structure (e.g. tire-slip nonlinearity) that violates linearity. The RBPF's variance-reduction guarantee against a plain PF does not hold in that regime — no derivation of the failure envelope.

6. **PSD recovery ladder statistics.** The recovery ladders in `UKF.h:150-178`, `SRUKF.h:98-141`, and `SigmaPoints.h:71-99` provide four sequential fallbacks (LLT → jitter → LDLT → diagonal clamp). There is no analysis of *how often* each tier fires in the benchmarks, whether the diagonal clamp (which discards off-diagonal information) ever fires on realistic problems, or what the error introduced by each tier is. This is instrumentable but not instrumented.

7. **Fixed-lag smoother lag choice.** The smoother lag is a compile-time / caller-set parameter with no automatic selection rule. Optimal lag is problem-dependent (a function of the mixing time of the smoothing gain series). Not derived.

---

## 14. Cross-references

- **Conventions and units** used by the code (float32, SI, sign conventions on innovations, state ordering `(pos, vel)` vs `(pos, vel, ...)`): see `.gpd/research-map/CONVENTIONS.md` (to be written by the methodology focus mapping).
- **Numerical validation strategy** (which limits are checked, which benchmarks are ground-truth): see `.gpd/research-map/VALIDATION.md`.
- **Known issues and open theoretical gaps**: see `.gpd/research-map/CONCERNS.md`.
- **References and prior artifacts**: see `.gpd/research-map/REFERENCES.md`.
- **Architecture and dispatch layer**: see `.gpd/research-map/ARCHITECTURE.md`, `.gpd/research-map/STRUCTURE.md`.
