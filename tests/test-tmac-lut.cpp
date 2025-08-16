// Unit test for T-MAC LUT construction (lut_ctor_int8_g4)
// Focus: verify basic invariants of QLUT / scales / biases produced by lut_ctor_int8_g4
// This intentionally does NOT depend on full ggml tensor graph execution.

#include <cstdio>
#include <cassert>
#include <vector>
#include <cmath>

#include "ggml.h"
#include "ggml-cpu.h"

// Internal T-MAC header (guarded by GGML_USE_TMAC in build). We include via relative path.
#include "ggml-cpu/tmac/lut_ctor.h"

static void fill_pattern(std::vector<tmac_float_type> & data) {
    for (size_t i = 0; i < data.size(); ++i) {
        // Simple deterministic pattern with both positive and negative values
        int mod = static_cast<int>(i % 11);
        int sign = (i % 2) ? 1 : -1;
        data[i] = (tmac_float_type) (sign * (0.5f + 0.1f * mod));
    }
}

static int run_case(int bits, int K, int act_group_size) {
    printf("[tmac-lut] bits=%d K=%d act_group_size=%d\n", bits, K, act_group_size);

    // Preconditions similar to those expected by current implementation
    assert(K % act_group_size == 0 || K == act_group_size); // simple grouping for this test
    assert(act_group_size % 32 == 0);
    assert(bits == 2 || bits == 4); // present implementation paths

    const int groups = K / act_group_size; // usually 1 for these small tests
    std::vector<tmac_float_type> B(K);
    fill_pattern(B);

    std::vector<tmac_float_type> LUT_Scales(groups ? groups : 1, (tmac_float_type)0);
    std::vector<tmac_float_type> LUT_Biases(groups ? groups : 1, (tmac_float_type)0);
    // QLUT size: act_group_size * 4 int8 entries per activation group (mirroring implementation)
    std::vector<int8_t> QLUT((size_t)act_group_size * 4 * (groups ? groups : 1), (int8_t)0);

    tmac_kernel_config cfg{};
    cfg.bits = bits;
    cfg.act_group_size = act_group_size;
    cfg.has_scale = true;
    cfg.has_zero_point = false;
    cfg.one_scale = false;
    cfg.g = 4;
    cfg.ngroups_per_elem = 8 / cfg.g;
    cfg.kfactor = act_group_size / cfg.g; // not used directly in ctor but keep consistent
    cfg.bm = bits * 256; // arbitrary positive multiple
    cfg.q_group_size = 0; // not used in this path
    cfg.actk = act_group_size / cfg.g;
    cfg.chunk_n = 8;
    cfg.simd_n_in = 16; // typical defaults; not used directly here
    cfg.simd_n_out = 8;

    lut_ctor_int8_g4(B.data(), LUT_Scales.data(), LUT_Biases.data(), QLUT.data(), K, &cfg);

    // Invariants / sanity checks
    // 1. Scale must be > 0
    for (auto s : LUT_Scales) {
        if (!(s > 0)) {
            fprintf(stderr, "Scale not positive: %f\n", (double) s);
            return 1;
        }
    }
    // 2. Quantized LUT values within int8 bounds
    for (auto v : QLUT) {
        if (v < -127 || v > 127) {
            fprintf(stderr, "QLUT value out of expected range [-127,127]: %d\n", (int) v);
            return 2;
        }
    }
    // 3. Bias finite (not NaN/inf) & magnitude reasonable vs sum(|B|)
    double abs_sum_B = 0.0;
    for (auto x : B) abs_sum_B += std::fabs((double)x);
    for (auto b : LUT_Biases) {
        if (!std::isfinite((double)b)) {
            fprintf(stderr, "Bias not finite: %f\n", (double)b);
            return 3;
        }
        // loose bound: |bias| should not exceed total absolute sum * 2 (generous)
        if (std::fabs((double)b) > abs_sum_B * 2.0 + 1e-3) {
            fprintf(stderr, "Bias magnitude suspicious: bias=%f abs_sum=%f\n", (double)b, abs_sum_B);
            return 4;
        }
    }
    // 4. Simple reproducibility: invoking ctor again with same inputs yields identical buffers
    std::vector<tmac_float_type> LUT_Scales2 = LUT_Scales;
    std::vector<tmac_float_type> LUT_Biases2 = LUT_Biases;
    std::vector<int8_t> QLUT2 = QLUT;
    // zero then recompute
    std::fill(LUT_Scales2.begin(), LUT_Scales2.end(), (tmac_float_type)0);
    std::fill(LUT_Biases2.begin(), LUT_Biases2.end(), (tmac_float_type)0);
    std::fill(QLUT2.begin(), QLUT2.end(), 0);
    lut_ctor_int8_g4(B.data(), LUT_Scales2.data(), LUT_Biases2.data(), QLUT2.data(), K, &cfg);
    if (QLUT2 != QLUT) {
        fprintf(stderr, "QLUT not deterministic across runs\n");
        return 5;
    }
    // Accept tiny float diffs for scales/biases
    for (size_t i = 0; i < LUT_Scales.size(); ++i) {
        if (std::fabs((double)LUT_Scales[i] - (double)LUT_Scales2[i]) > 1e-7) {
            fprintf(stderr, "Scale mismatch rerun idx=%zu orig=%f new=%f\n", i, (double)LUT_Scales[i], (double)LUT_Scales2[i]);
            return 6;
        }
        if (std::fabs((double)LUT_Biases[i] - (double)LUT_Biases2[i]) > 1e-5) {
            fprintf(stderr, "Bias mismatch rerun idx=%zu orig=%f new=%f\n", i, (double)LUT_Biases[i], (double)LUT_Biases2[i]);
            return 7;
        }
    }

    printf("[tmac-lut] PASS (bits=%d) scale=%f bias=%f first_q=%d\n", bits,
           (double)LUT_Scales[0], (double)LUT_Biases[0], (int)QLUT[0]);
    return 0;
}

int main() {
#ifndef GGML_USE_TMAC
    printf("[tmac-lut] Skipped: built without GGML_TMAC\n");
    return 0;
#else
    // Single group K == act_group_size
    const int K = 64;
    const int act_group_size = 64;
    int rc = 0;
    rc = run_case(2, K, act_group_size); if (rc) return rc;
    rc = run_case(4, K, act_group_size); if (rc) return rc;
    return 0;
#endif
}
