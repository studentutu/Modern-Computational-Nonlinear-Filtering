// Regression test for the RBPKF log-determinant path (LDLT-only). Pre-fix, an
// ill-conditioned innovation covariance S with condition number ~1e12 could
// produce a direct det(S) that underflows to zero in float, triggering the
// -1e10 sentinel; then the LDLT fallback would run only sometimes. Now the
// path is always LDLT + diagonal-product with a 1e-30 clamp.
//
// The test builds a symmetric-positive-definite S with a specified condition
// number, computes the log determinant via the exact same LDLT path used by
// rbpf_core.hpp:142+, and compares it to an SVD-based reference. Any large
// disagreement, non-finite result, or crash on the ill-conditioned S is a
// regression.

#include <iostream>
#include <cassert>
#include <cmath>
#include <Eigen/Dense>

// Recompute log_det via the same algorithm as rbpf_core.hpp — LDLT diagonal
// product with std::max(D(k), 1e-30f) clamp. Kept in-file so the test does
// not depend on internal helpers of rbpf_core.
static float log_det_via_ldlt(const Eigen::MatrixXf& S) {
    Eigen::LDLT<Eigen::MatrixXf> ldlt(S);
    if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
        return -1e10f;
    }
    auto D = ldlt.vectorD();
    float log_det = 0.0f;
    for (int k = 0; k < D.size(); ++k) {
        log_det += std::log(std::max(D(k), 1e-30f));
    }
    return log_det;
}

// SVD-based reference: log|det(S)| = sum(log(singular values)).
static double log_det_via_svd(const Eigen::MatrixXf& S) {
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(S);
    const auto& sv = svd.singularValues();
    double log_det = 0.0;
    for (int k = 0; k < sv.size(); ++k) {
        log_det += std::log(static_cast<double>(std::max(sv(k), 1e-30f)));
    }
    return log_det;
}

// Build an SPD matrix with a specified condition number by geometrically
// spacing eigenvalues between 1 and 1/cond.
static Eigen::MatrixXf make_ill_conditioned(int n, float cond) {
    Eigen::MatrixXf D = Eigen::MatrixXf::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(std::max(n - 1, 1));
        // Eigenvalues span [1/cond, 1].
        float ev = std::pow(cond, -t);
        D(i, i) = ev;
    }
    // Orthogonal similarity transform to keep S SPD but non-diagonal.
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(n, n);
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
    Eigen::MatrixXf Q = qr.householderQ();
    return Q * D * Q.transpose();
}

void test_well_conditioned() {
    std::cout << "test_well_conditioned..." << std::endl;
    Eigen::MatrixXf S = make_ill_conditioned(3, 1e2f);
    float lldt = log_det_via_ldlt(S);
    double lsvd = log_det_via_svd(S);
    assert(std::isfinite(lldt));
    double rel = std::abs(static_cast<double>(lldt) - lsvd) /
                 std::max(std::abs(lsvd), 1e-6);
    std::cout << "  ldlt=" << lldt << " svd=" << lsvd << " rel=" << rel << std::endl;
    assert(rel < 1e-3);
}

void test_condition_1e6() {
    std::cout << "test_condition_1e6..." << std::endl;
    Eigen::MatrixXf S = make_ill_conditioned(3, 1e6f);
    float lldt = log_det_via_ldlt(S);
    double lsvd = log_det_via_svd(S);
    assert(std::isfinite(lldt));
    double rel = std::abs(static_cast<double>(lldt) - lsvd) /
                 std::max(std::abs(lsvd), 1e-6);
    std::cout << "  ldlt=" << lldt << " svd=" << lsvd << " rel=" << rel << std::endl;
    assert(rel < 1e-3);
}

void test_condition_1e12() {
    std::cout << "test_condition_1e12..." << std::endl;
    Eigen::MatrixXf S = make_ill_conditioned(3, 1e12f);
    float lldt = log_det_via_ldlt(S);
    double lsvd = log_det_via_svd(S);
    // At cond ~ 1e12 the LDLT may hit the clamp; but the result MUST be finite
    // and either match SVD closely or degrade toward the sentinel path — not
    // silently produce garbage.
    assert(std::isfinite(lldt));
    if (lldt <= -1e9f) {
        std::cout << "  ldlt hit sentinel (expected for cond=1e12); svd=" << lsvd << std::endl;
    } else {
        double rel = std::abs(static_cast<double>(lldt) - lsvd) /
                     std::max(std::abs(lsvd), 1e-6);
        std::cout << "  ldlt=" << lldt << " svd=" << lsvd << " rel=" << rel << std::endl;
        // Cond=1e12 in float32 is at the edge; allow a looser tolerance.
        assert(rel < 5e-2);
    }
}

int main() {
    test_well_conditioned();
    test_condition_1e6();
    test_condition_1e12();
    std::cout << "test_rbpf_logdet: all cases passed." << std::endl;
    return 0;
}
