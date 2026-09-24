// Shared pieces of the Qwen3-VL vision entries (`qwen_vision_stem`,
// `qwen_vision_block`, `qwen_vision_merger`). The counterpart of
// `metal/common/vision.h`. The projections run on the projection library's
// GEMM (`projection_gemm_accumulate` over dense weight rows, 64 x 64 tiles)
// with the vision epilogues below; the layer norm, the 2D rotary embedding
// and the full (non-causal) attention are the vision-specific launches.
//
// Every dense operand is bound canonically (row-major, unit innermost
// stride). The residual stream (stem output, block input and output, merger
// input) and the merger output are F32; the published intermediates are the
// activation A (bf16 or f16, passed as `act`: the stem has no A); bias and
// norm vectors are any dense element. Values are rounded to A exactly where
// the portable bodies (`vision.seismic`) publish them.
//
// The attention reads its operands as f16 (there is no bf16 cooperative
// matrix): the rotation launch publishes the A-rounded rotated queries and
// keys, and the values, as f16 rows [3][H][padded rows][W], zero past the
// last row, so every attention tile reads whole contiguous rows.
//
// This file is independent of any entry ABI.
#include "common/projection.glsl"
#include "common/rotary.glsl"
#include "common/flash.glsl"
#include "common/precise.glsl"

#define VISION_TILE_M 64u
#define VISION_TILE_N 64u
// 32 x 32 per subgroup: 128 invocations.
#define VISION_SUB 32u
// Output columns per attention pass (`common/flash.glsl`).
#define VISION_WINDOW FLASH_WINDOW
// Shared bytes of a vision GEMM launch: (TM + TN) * 80.
#define VISION_GEMM_SHARED 10240u
// Query rows of one attention workgroup (16 per subgroup).
#define VISION_ATTEND_ROWS 64u
#define VISION_INF uintBitsToFloat(0x7f800000u)

// ---------------------------------------------------------------------------
// Scalar functions.

float vision_gelu_tanh(float value) {
    const float argument = 0.7978845608028654 * (value + 0.044715 * value * value * value);
    const float hyperbolic = seismic_div_rn(2.0, 1.0 + precise_exp(-2.0 * argument)) - 1.0;
    return 0.5 * value * (1.0 + hyperbolic);
}

// erf (the complementary form's Chebyshev fit, Numerical Recipes `erfcc`,
// |error| < 1.2e-7).
float vision_erf(float x) {
    const float z = abs(x);
    const float t = seismic_div_rn(1.0, 1.0 + 0.5 * z);
    const float r = t * precise_exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196
        + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
        + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0.0 ? 1.0 - r : r - 1.0;
}

float vision_gelu_erf(float value) {
    return 0.5 * value * (1.0 + vision_erf(value * 0.7071067811865476));
}

// ---------------------------------------------------------------------------
// GEMM epilogues. `vision_put(out, m, n, acc)` receives the F32 product of
// output (m, n); the projection is acc + bias[n].
//   Bias          y = round_A(projection)
//   BiasGeluTanh  y = round_A(gelu_tanh(round_A(projection)))
//   BiasGeluErf   y = round_A(gelu_erf(round_A(projection)))
//   BiasF32       y = projection, plus residual[m, n] when bound (F32)
//   Stem          y = (partial[m, n] + acc + bias[n]) + blended, F32, with
//                 blended the four table corners times their coefficients
//                 summed in order

#define VISION_BIAS 0
#define VISION_BIAS_GELU_TANH 1
#define VISION_BIAS_GELU_ERF 2
#define VISION_BIAS_F32 3
#define VISION_STEM 4

struct vision_epilogue {
    int kind;
    int act;                // Bias, BiasGelu*: ELEMENT_* of A
    uint64_t y;
    uint64_t y0;            // row stride of y, in elements
    uint64_t bias;
    int bias_kind;
    uint64_t residual;      // BiasF32: F32 rows or 0; Stem: the F32 partial
    uint64_t residual0;     // their row stride, in elements
    uint64_t table;         // Stem: the position table [L, columns]
    int table_kind;
    uint64_t indices;       // Stem: [M, 4] i32
    uint64_t coefficients;  // Stem: [M, 4] F32
};

// Bias or BiasGelu*, published in A.
vision_epilogue vision_bias(const int kind, const int act, uint64_t y, uint64_t y0, uint64_t bias,
    const int bias_kind) {
    return vision_epilogue(kind, act, y, y0, bias, bias_kind, 0ul, 0ul, 0ul, ELEMENT_F32, 0ul, 0ul);
}

// The F32 projection plus the F32 `residual` rows (0: none).
vision_epilogue vision_bias_f32(uint64_t y, uint64_t y0, uint64_t bias, const int bias_kind, uint64_t residual,
    uint64_t residual0) {
    return vision_epilogue(VISION_BIAS_F32, ELEMENT_F32, y, y0, bias, bias_kind, residual, residual0, 0ul,
        ELEMENT_F32, 0ul, 0ul);
}

vision_epilogue vision_stem(uint64_t y, uint64_t y0, uint64_t bias, const int bias_kind, uint64_t partial,
    uint64_t columns, uint64_t table, const int table_kind, uint64_t indices, uint64_t coefficients) {
    return vision_epilogue(VISION_STEM, ELEMENT_F32, y, y0, bias, bias_kind, partial, columns, table, table_kind,
        indices, coefficients);
}

void vision_put(vision_epilogue out_, uint m, uint n, float value) {
    const int act = out_.act;
    const float bias = element_at(out_.bias_kind, out_.bias, n);
    const uint64_t at = uint64_t(m) * out_.y0 + n;
    if (out_.kind == VISION_BIAS) {
        element_put(act, out_.y, at, value + bias);
    } else if (out_.kind == VISION_BIAS_GELU_TANH) {
        element_put(act, out_.y, at, vision_gelu_tanh(element_round(act, value + bias)));
    } else if (out_.kind == VISION_BIAS_GELU_ERF) {
        element_put(act, out_.y, at, vision_gelu_erf(element_round(act, value + bias)));
    } else if (out_.kind == VISION_BIAS_F32) {
        const float projected = value + bias;
        element_f32_put(out_.y + at * 4ul, out_.residual == 0ul ? projected
            : element_f32_at(out_.residual + (uint64_t(m) * out_.residual0 + n) * 4ul) + projected);
    } else {
        const float projected = element_f32_at(out_.residual + (uint64_t(m) * out_.residual0 + n) * 4ul) + value + bias;
        float blended = 0.0;
        [[unroll]] for (uint i = 0u; i < 4u; ++i) {
            const int index = element_i32_at(out_.indices + (uint64_t(m) * 4ul + i) * 4ul);
            blended += element_at(out_.table_kind, out_.table, uint64_t(index) * out_.residual0 + n)
                * element_f32_at(out_.coefficients + (uint64_t(m) * 4ul + i) * 4ul);
        }
        element_f32_put(out_.y + at * 4ul, projected + blended);
    }
}

// One 64 x 64 GEMM tile of a vision projection: output rows tm_index * 64 ..,
// weight rows tn_index * 64 .. of `w` (N = rows, K = k).
void vision_gemm(const int wk, projection_prologue in_, vision_epilogue out_, projection_weights w, uint m_rows,
    uint rows, uint k, uint tm_index, uint tn_index) {
    const uint first = tn_index * VISION_TILE_N;
    projection_gemm_acc acc;
    projection_gemm_accumulate(wk, wk, false, VISION_TILE_M, VISION_TILE_N, VISION_SUB, VISION_SUB, in_, w, w, first,
        rows, tm_index * VISION_TILE_M, m_rows, k, 0u, (k + PROJECTION_GEMM_K - 1u) / PROJECTION_GEMM_K, m_rows, acc);
    [[unroll]] for (uint q = 0u; q < projection_gemm_pairs(VISION_SUB, VISION_SUB); ++q) {
        const vec2 c = projection_gemm_pair(acc, VISION_SUB, q);
        const uint m = tm_index * VISION_TILE_M + projection_gemm_pair_row(VISION_TILE_N, VISION_SUB, VISION_SUB, q);
        const uint n = first + projection_gemm_pair_column(VISION_TILE_N, VISION_SUB, q);
        if (m < m_rows && n < rows)
            vision_put(out_, m, n, c.x);
        if (m < m_rows && n + 1u < rows)
            vision_put(out_, m, n + 1u, c.y);
    }
}

// Dense weight rows of an [N, K] row-major matrix (`stride` elements per row
// of `bytes` each).
projection_weights vision_weights(uint64_t base, uint64_t stride, const uint bytes, uint k) {
    return projection_weights(base, packets_rows16(stride * uint64_t(bytes), 0ul, 0ul, 0ul, 0ul), k, 0ul);
}

// ---------------------------------------------------------------------------
// Layer norm of one F32 row of `width` values by the whole workgroup: two-pass
// centered F32 statistics, out = round_A(centered * inverse * weight + bias).
// Shared: one float per subgroup at float 0.
void vision_layer_norm(const int act, uint64_t x, uint64_t out_, uint64_t weight, const int weight_kind, uint64_t bias,
    const int bias_kind, uint width, float epsilon) {
    float sum = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < width; i += gl_WorkGroupSize.x)
        sum += element_f32_at(x + uint64_t(i) * 4ul);
    const float mean = seismic_div_rn(reduce_group_sum(sum, 0u), float(width));
    float squares = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < width; i += gl_WorkGroupSize.x) {
        const float centered = element_f32_at(x + uint64_t(i) * 4ul) - mean;
        squares = seismic_fma_rn(centered, centered, squares);
    }
    const float inverse = inversesqrt(seismic_div_rn(reduce_group_sum(squares, 0u), float(width)) + epsilon);
    for (uint i = gl_LocalInvocationIndex; i < width; i += gl_WorkGroupSize.x)
        element_put(act, out_, i, (element_f32_at(x + uint64_t(i) * 4ul) - mean) * inverse
            * element_at(weight_kind, weight, i) + element_at(bias_kind, bias, i));
}

// ---------------------------------------------------------------------------
// 2D rotary embedding of one head row of width w = 4P (a multiple of 64, at
// most FLASH_MAX_W) held by one subgroup (lane l owns columns [l E, l E + E),
// E = w / 32): column i < 2P pairs with i + 2P (16 lanes away); pair
// p = i % 2P turns by coordinates[p / P] * 10000^(-(p % P) / P). Rotated
// values are rounded to A, then published as f16 at `target`; `rotated`
// false copies the row (the values). The whole subgroup calls it.
void vision_rotate(const int act, uint64_t head_row, uint64_t coordinates, const uint w, bool rotated, uint64_t target) {
    const uint lane = gl_SubgroupInvocationID;
    const uint e = w / 32u;
    const uint p = w / 4u;
    float x[FLASH_MAX_W / 32u];
    [[unroll]] for (uint i = 0u; i < FLASH_MAX_W / 32u; ++i)
        if (i < e)
            x[i] = element_at(act, head_row, lane * e + i);
    [[unroll]] for (uint i = 0u; i < FLASH_MAX_W / 32u; ++i) {
        if (i < e) {
            const float partner = subgroupShuffleXor(x[i], 16u);
            if (rotated) {
                const uint column = lane * e + i;
                const uint pair = column % (2u * p);
                const float frequency = precise_exp(seismic_div_rn(-9.210340371976184 * float(pair % p), float(p)));
                const float angle = float(element_i32_at(coordinates + uint64_t(pair / p) * 4ul)) * frequency;
                float c;
                const float s = rotary_sincos(angle, c);
                x[i] = element_round(act, column < 2u * p ? x[i] * c - partner * s : x[i] * c + partner * s);
            }
            element_put(ELEMENT_F16, target, lane * e + i, x[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Full attention of one head over every row: workgroup (tile of
// VISION_ATTEND_ROWS query rows, head), 16 query rows per subgroup, keys
// streamed in FLASH_KEYS-row tiles, one online-softmax pass of
// `common/flash.glsl` per VISION_WINDOW output columns.
// `operands` are the f16 rows [3][heads][padded][w]; the output row r of
// head h is A at out_[r * heads * w + h * w ..], rounded from F32.
// Shared: the K/V tile (FLASH_KEYS x (w + 8) f16), then
// FLASH_SCRATCH_FLOATS floats per subgroup.
void vision_attend(const int act, uint64_t operands, uint64_t out_, uint rows, uint padded, uint heads, const uint w, uint tile,
    uint head) {
    const uint lane = gl_SubgroupInvocationID;
    const uint64_t plane = uint64_t(heads) * padded * w * 2ul;   // bytes of one of Q, K, V
    const uint64_t head_rows = uint64_t(head) * padded * w * 2ul;
    const uint64_t keys = operands + plane + head_rows;
    const uint64_t values = operands + 2ul * plane + head_rows;
    const uint block_row = tile * VISION_ATTEND_ROWS + 16u * gl_SubgroupID;
    const uint64_t block_queries = operands + head_rows + uint64_t(block_row) * w * 2ul;
    const uint scratch = (FLASH_KEYS * flash_pitch(w)) / 2u + gl_SubgroupID * FLASH_SCRATCH_FLOATS;
    const float scale = inversesqrt(float(w)) * 1.4426950408889634;

    const uint64_t width = uint64_t(heads) * w;
    flash_output o;
    // One online-softmax pass per output window.
    const uint windows = (w + VISION_WINDOW - 1u) / VISION_WINDOW;
    for (uint pass = 0u; pass < windows; ++pass) {
        const uint column0 = pass * VISION_WINDOW;
        flash_output_clear(w, VISION_WINDOW, o);
        flash_softmax softmax = flash_softmax_start();
        for (uint first = 0u; first < rows; first += FLASH_KEYS) {
            barrier();
            flash_stage(ELEMENT_F16, keys, uint64_t(w), 0ul, int(first), int(rows), w, 0u);
            barrier();
            flash_scores(block_queries, uint64_t(w), w, 0u, scratch);
            float s[16];
            flash_lane_scores(scratch, s);
            [[unroll]] for (uint j = 0u; j < 16u; ++j) {
                s[j] *= scale;
                if (first + 16u * (lane / 16u) + j >= rows)
                    s[j] = -VISION_INF;
            }
            const float alpha = flash_online(softmax, s);
            const uint p_half = flash_publish_probabilities(scratch, s);
            flash_rescale(scratch, alpha, w, VISION_WINDOW, o);
            barrier();
            flash_stage(ELEMENT_F16, values, uint64_t(w), 0ul, int(first), int(rows), w, 0u);
            barrier();
            flash_accumulate(p_half, 0u, w, VISION_WINDOW, column0, o);
        }
        const float denominator = flash_denominator(softmax);

        barrier();
        // Unrolled, so every fragment index is a constant.
        [[unroll]] for (uint q = 0u; q < VISION_WINDOW / 2u; ++q) {
            if (q >= flash_window(w, VISION_WINDOW) / 2u)
                break;
            const float value = flash_output_value(o, scratch, q);
            const uint r = flash_output_row(q);
            const float row_denominator = subgroupShuffle(denominator, r);
            const uint row = block_row + r;
            if (row < rows)
                element_put(act, out_,
                    uint64_t(row) * width + uint64_t(head) * w + flash_output_column(w, VISION_WINDOW, column0, q),
                    seismic_div_rn(value, row_denominator));
        }
    }
}
