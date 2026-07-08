# Development Notes

## OptMathKernels Dependency — Release Audit & Pinning Policy

**Date**: 2026-05-25

The compute backends (NEON / SVE2 / Vulkan / CUDA) live in the external
[OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)
(OptMathKernels) project, consumed via CMake `FetchContent` and dispatched to
through `Common/include/FilterMath.h` / `FilterMathGPU.h`.

### Pinning policy (the "tag/release format going forward")

OptMathKernels now publishes semantic-version release tags (`v0.5.0` … `v0.5.15`).
This project **pins to a specific tag** rather than tracking the moving `main`
branch. The pin lives in one place — `CMakeLists.txt`:

```cmake
set(OPTMATH_RELEASE_TAG "v0.5.15" CACHE STRING "Pinned OptMathKernels release tag")
FetchContent_Declare(OptimizedKernels ... GIT_TAG ${OPTMATH_RELEASE_TAG})
```

Adopting a newer kernel release is a deliberate, audited step:

1. `git -C $HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA fetch --tags`
2. Audit the upstream diff `git diff <old-tag>..<new-tag>` — pay attention to
   anything under `include/` (public API) and the backend `src/` the filters use.
3. Bump `OPTMATH_RELEASE_TAG`, reconfigure, rebuild.
4. `ctest --output-on-failure` (expect 24/24) and run the benchmark suite.
5. Update README/this file, then commit and tag the parent release.

### Audit: v0.5.13 → v0.5.15 (adopted 2026-05-25)

Previous pin tracked `main` (built at `v0.5.13`). Reviewed every commit and the
full `git diff v0.5.13..v0.5.15`:

| Change | File | Impact on this project |
|--------|------|------------------------|
| **Discrete-GPU preference in Vulkan device selection** | `src/vulkan/vulkan_backend.cpp` | Behavioral, positive. `VulkanContext::init()` now scores physical devices (discrete > integrated > virtual > CPU) and requires a compute queue, instead of blindly taking `devices[0]`. On this dual-GPU x86_64 box it now logs and selects the RTX 5070 Ti for Vulkan compute. No API change. |
| Per-source documentation coverage | `src/vulkan/vulkan_backend.cpp`, `src/platform/platform.cpp` | None — comments only. |
| x86_64 dual-GPU benchmark docs + release notes | `README.md` | None — upstream docs. |
| Version bump 0.5.13 → 0.5.15 | `CMakeLists.txt` | None. |

**Public API (`include/optmath/*.hpp`): zero changes** across v0.5.13..v0.5.15,
so every `optmath::` call site in `FilterMath.h`, `FilterMathGPU.h`, and
`particle_filter_gpu.hpp` is unaffected.

**Verification (2026-05-25):** reconfigured at `v0.5.15`, full rebuild, **24/24
CTest pass**, Vulkan tests confirm `[Vulkan] Selected GPU: NVIDIA GeForce RTX
5070 Ti Laptop GPU`, and the benchmark RMSE/NEES figures are numerically
identical to the prior run (the changed kernel path is not on the UKF/SRUKF
CUDA/Eigen benchmark path). Safe to adopt.

---

## CUDA Development Status

**Date**: 2026-05-25

**Status**: CUDA 13.x active for SM 75–120 including Blackwell (verified on
RTX 5070 Ti / SM 120, CUDA 13.1). CUDA 12.x remains supported for SM 75–90.

> Historical note: earlier revisions of this file capped support at SM 90 and
> listed Blackwell (SM 100/SM 120) as "blocked until CUDA 13+". That blocker is
> resolved — see the verification block below and the README CUDA section.

### Supported Architecture Targets

- SM 75: Turing (RTX 2080/2070/2060) — CUDA 12.x / 13.x
- SM 80/86: Ampere (RTX 3090/3080/3070, A100) — CUDA 12.x / 13.x
- SM 89: Ada Lovelace (RTX 4090/4080/4070) — CUDA 12.x / 13.x
- SM 90: Hopper (H100) — CUDA 12.x / 13.x
- **SM 100 / SM 120: Blackwell (RTX 5090/5080/5070) — CUDA 13.x only**

For Blackwell, configure with `-DCMAKE_CUDA_ARCHITECTURES=native -DOPTMATH_CUDA_NATIVE=ON`
(SM 120 is not in the default multi-arch list). On CUDA 12.x, `nvcc` rejects
`compute_100`/`compute_120` with `Unsupported gpu architecture`.

### Current Build Configuration

CUDA auto-detected and enabled. Active acceleration: CUDA (cuBLAS GEMM, GPU
particle filter) + Vulkan compute shaders + OpenMP. OptMathKernels cuSOLVER
Cholesky is available as of upstream v0.5.10 (verified on CUDA 13).

### CUDA Code (commit 397b2d9, active)

- cuBLAS GEMM for matrices >= 32x32
- GPU particle filter context
- Runtime CUDA enable/disable via `filtermath::config::set_cuda_enabled()`

---

## Build Verification (May 25, 2026)

### Ubuntu 26.04 (x86_64) — OptMathKernels v0.5.15

**System Info**:
- OS: Ubuntu 26.04 (x86_64)
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell, SM 120), driver 595.71.05
- CUDA: 13.1.115 (enabled, SM native / 120)
- Vulkan: 1.4.341 (discrete GPU auto-selected — RTX 5070 Ti)
- Eigen: 3.4.0
- OptMathKernels: pinned **v0.5.15** via `OPTMATH_RELEASE_TAG`

**Build Command**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DOPTMATH_CUDA_NATIVE=ON
make -j$(nproc)
```

**Test Results: 24/24 passing** (8 filter/benchmark + 16 OptMathKernels GPU/SIMD,
incl. `test_cuda_kernels` on the Blackwell GPU and the 4 Vulkan suites selecting
the discrete GPU). Total CTest time ≈ 5.9 s.

---

## Build Verification (April 13, 2026)

### Ubuntu 24.04 LTS (x86_64)

**System Info**:
- OS: Ubuntu 24.04 LTS (Noble Numbat)
- Kernel: 6.8.0-107-generic
- CUDA: 12.0.140 (enabled, SM 75–90)
- Vulkan: 1.3.275
- Compiler: GCC 13.3.0
- Shader compiler: glslangValidator (glslang-tools)

**Build Command**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Prerequisites**:
```bash
sudo apt install build-essential cmake libeigen3-dev
sudo apt install python3 python3-pip python3-venv
sudo apt install glslang-tools          # Required for Vulkan shader compilation
sudo apt install vulkan-tools libvulkan-dev  # Optional: Vulkan runtime
```

**Test Results** (24/24 passing):
| Test | Status | Time |
|------|--------|------|
| EKF_Test | ✓ | 0.11s |
| UKF_Test | ✓ | 0.15s |
| SRUKF_Test | ✓ | 1.58s |
| PKF_Test | ✓ | 1.68s |
| PKF_Example | ✓ | 0.28s |
| RBPF_Basic | ✓ | 0.04s |
| RBPF_CTRV | ✓ | 1.57s |
| Benchmarks | ✓ | 1.33s |
| OptimizedKernels (basic) | ✓ | 0.00s |
| OptimizedKernels (Vulkan) | ✓ 4/4 | ~0.9s each |
| OptimizedKernels (Radar) | ✓ 2/2 | 0.05s |
| OptimizedKernels (NEON) | Skipped (x86_64) | — |
| OptimizedKernels (Platform) | ✓ | 0.00s |
| OptimizedKernels (CUDA) | ✓ | 6.90s |

---

## Future Work

### When CUDA 13+ Available

1. Add SM 100 (Blackwell) back to architecture list in CMakeLists.txt
2. Verify `-expt-relaxed-constexpr` flag compatibility
3. Benchmark Blackwell vs Ampere/Ada Lovelace performance
4. Benchmark CUDA vs Vulkan particle filter performance

### cuSOLVER Integration (OptimizedKernels)

When OptimizedKernels adds cuSOLVER support:
- GPU Cholesky decomposition
- GPU triangular solve
- GPU matrix inverse

This will enable full GPU acceleration for UKF/SRUKF sigma point operations.

---

## Changelog

### Audit remediation (Jul 2026)

Repository-wide audit (correctness, build/reproducibility, docs). Verified on the
aarch64 Raspberry Pi host (CPU: NEON/SVE2/Eigen path; no CUDA) — all 8 project
CTest cases pass before and after; benchmark RMSE/NEES numerically unchanged.

**Correctness**
- **RBPF OpenMP data race (HIGH)**: `A,B,bias,Q,H,offset,R` were declared outside
  the `#pragma omp parallel for` in `rbpf_core.hpp::step()` (shared → concurrent
  writes). Moved inside the loop body (thread-private).
- **OpenMP RNG ignored the seed**: RBPF and PKF used a `std::random_device`
  `thread_local` in the parallel path, so `config.seed` / `set_seed()` had no
  effect and runs were non-reproducible. Replaced with per-thread `mt19937_64`
  seeded deterministically (SplitMix64 mix of the base seed) + `schedule(static)`.
  Reproducibility confirmed (identical output across runs at 4 threads).
- **RBPF fixed-lag ancestry off-by-one**: `initialize()` stored the initial
  snapshot at logical slot 0 but never advanced `parent_indices_cnt_`, so step 1
  clobbered it and the smoother could never reach the true initial state. The
  counter now advances so the initial state is preserved as an ancestor.
- **UKF covariance update**: switched `P = P - K S K^T` to the symmetric Joseph
  form `P - K Pxy^T - Pxy K^T + K S K^T` (identical in exact arithmetic, PSD-safe
  by construction) and replaced the single fixed regularization bump with an
  escalating, re-checked jitter loop.
- **SRUKF gate/covariance consistency**: when the innovation gate scales the state
  correction by `s < 1`, the covariance downdate is now scaled by `sqrt(2s - s^2)`
  (the Joseph-consistent reduction for a gain `sK`) instead of removing the full
  `K S_yy S_yy^T K^T`, so the reported covariance is no longer over-confident.

**Build / reproducibility**
- Dependency source is now the **public GitHub URL** by default (was a local
  `$HOME` path that broke clean clones), overridable via `-DOPTMATH_REPO`.
- Tests now use a `CHECK` macro (exits non-zero) instead of `assert()`, which was
  compiled out under `NDEBUG` — CTest was green regardless of correctness in
  Release. Added real numeric + fixed-lag-smoother assertions to the RBPF test.
- `Benchmarks/CMakeLists.txt` guards `find_package(Eigen3)` like the other subdirs
  (was aborting configure when Eigen came from FetchContent).
- Default CUDA arch list adds Blackwell SM 100/120 automatically when nvcc is
  CUDA ≥ 13 (documented host built without manual flags otherwise).
- `install()` rules added for the filter headers (`<prefix>/include/nlf/`).
- Python venv is opt-out (`-DNLF_BUILD_PYTHON_VENV=OFF`) so offline/CI C++ builds
  don't require PyPI.

**Docs**
- Removed README "Numerical Stability" issues describing the deleted GPS/INS /
  AircraftNav simulation (kept the real Eigen-aliasing fix, stripped the fictional
  application narrative). Reconciled the contradictory bearing-only RMSE tables.
  Corrected the Benchmarks/README to the four problems actually run (Lorenz96 is
  present in the header but not in the default suite). Added a host note clarifying
  that GPU test counts/timings are from the x86_64 reference host.

### Recommended follow-up: CI

No CI is committed yet. A minimal GitHub Actions workflow would need to build
CPU-only (`-DCMAKE_CUDA_COMPILER=""`), skip the venv (`-DNLF_BUILD_PYTHON_VENV=OFF`),
and either provide a Vulkan loader / `glslang` on the runner or gate the
dependency's Vulkan tests, because `ENABLE_VULKAN` is forced ON. Run
`ctest -R "EKF_Test|UKF_Test|SRUKF_Test|PKF_Test|PKF_Example|RBPF_Basic|RBPF_CTRV|Benchmarks"`
to enforce the eight project tests on every clean checkout.

### v3.2.0 (May 2026)
- Audited OptMathKernels major updates v0.5.13 → v0.5.15 (see "Release Audit" above)
- Adopted tag/release pinning: `OPTMATH_RELEASE_TAG` pins FetchContent to a
  release tag (now `v0.5.15`) instead of tracking `main`
- Picked up upstream Vulkan discrete-GPU preference (RTX 5070 Ti now selected for
  Vulkan compute on this dual-GPU host)
- Refreshed CUDA status to CUDA 13.x / Blackwell SM 120 (resolved former blocker)
- Rebuilt, 24/24 CTest pass, benchmarks rerun (RMSE/NEES unchanged), plots regenerated

### v3.1.0 (April 2026)
- Added CUDA GPU acceleration (FilterMath.h, FilterMathGPU.h, particle_filter_gpu.hpp)
- Disabled CUDA due to Ubuntu 24.04 CUDA 12.0 incompatibilities
- Updated CMakeLists.txt to exclude SM 100 (Blackwell)
- Created DEVELOPMENT_NOTES.md

### v3.0.0 (March 2026)
- FilterMath dispatch layer (SVE2 > NEON > Eigen)
- All filter code using FilterMath dispatch
- Bug fixes: -ffast-math removal, safe Cholesky downdate, RBPKF weight handling
- Full benchmark suite passing
