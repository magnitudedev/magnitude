// Shared pieces of the vision tower entries (`vision_*` in vision.seismic).
// The counterpart of `metal/lib/vision/vision.h`. The projections run on the
// projection library's GEMM (`projection_gemm_accumulate` over dense weight
// rows, 64 x 64 tiles) with the vision epilogues below; the row norm, the 2D
// rotary embedding and the full (non-causal) attention are the
// vision-specific launches.
//
// Every dense operand is bound canonically (row-major, unit innermost
// stride). The residual stream (stem output, block input and output, merger
// input) and the image features are F32; the published intermediates are
// the activation A (bf16 or f16, passed as `act`: the stem has no A); bias
// and norm vectors are any dense element. Values are rounded to A exactly
// where the portable bodies (`vision.seismic`) publish them.
//
// The attention reads its operands as f16 (there is no bf16 cooperative
// matrix): the prepare launch publishes the A-rounded rotated queries and
// keys, and the values, as f16 rows [3][H][padded rows][W], zero past the
// last row, so every attention tile reads whole contiguous rows.
//
// This file is independent of any entry ABI.
#include "../projection/projection.glsl"
#include "../core/rotary.glsl"
#include "../attention/flash.glsl"
#include "../core/precise.glsl"

#define VISION_TILE_M 64u
#define VISION_TILE_N 64u
// 32 x 32 per subgroup: 128 invocations.
#define VISION_SUB 32u
// Output columns per attention pass (`lib/attention/flash.glsl`).
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

float vision_gelu_quick(float value) {
    return seismic_div_rn(value, 1.0 + precise_exp(-1.702 * value));
}

// ---------------------------------------------------------------------------
// GEMM epilogues. `vision_put(out, m, n, acc)` receives the F32 product of
// output (m, n).
//   Linear  the projection is acc, plus bias[n] when `biased`, clamped to
//           [minimum, maximum] when `clamped`; with `activation` nonzero
//           y = Y(act(round_A(projection))); with a gate (A rows, 0: none)
//           y = Y(round_A(projection) · gate[m, n]); otherwise
//           y = Y(projection), plus residual[m, n] (F32) when bound
//   Stem    y = ((partial[m, n] when bound) + acc [+ bias[n]]) + blended,
//           F32, with blended the four table corners times their
//           coefficients summed in order

#define VISION_LINEAR 0
#define VISION_STEM 1

struct vision_epilogue {
    int kind;
    int act;                // Linear: ELEMENT_* of A
    int y_kind;             // ELEMENT_* of y
    int activation;         // Linear: 0, 1 tanh GELU, 2 erf GELU, 3 quick GELU
    uint64_t y;
    uint64_t y0;            // row stride of y, in elements
    bool biased;
    uint64_t bias;
    int bias_kind;
    uint64_t residual;      // Linear: F32 rows or 0; Stem: the F32 partial or 0
    uint64_t residual0;     // their row stride, in elements
    uint64_t gate;          // Linear: A rows or 0
    uint64_t gate0;         // their row stride, in elements
    bool clamped;           // Linear
    float minimum;
    float maximum;
    uint64_t table;         // Stem: the position table [L, columns]
    int table_kind;
    uint64_t indices;       // Stem: [M, 4] i32
    uint64_t coefficients;  // Stem: [M, 4] F32
};

vision_epilogue vision_linear_epilogue(const int act, const int y_kind, int activation, uint64_t y, uint64_t y0,
    bool biased, uint64_t bias, const int bias_kind, uint64_t residual, uint64_t residual0, uint64_t gate,
    uint64_t gate0, bool clamped, float minimum, float maximum) {
    return vision_epilogue(VISION_LINEAR, act, y_kind, activation, y, y0, biased, bias, bias_kind, residual,
        residual0, gate, gate0, clamped, minimum, maximum, 0ul, ELEMENT_F32, 0ul, 0ul);
}

vision_epilogue vision_stem(uint64_t y, uint64_t y0, bool biased, uint64_t bias, const int bias_kind,
    uint64_t partial, uint64_t columns, uint64_t table, const int table_kind, uint64_t indices,
    uint64_t coefficients) {
    return vision_epilogue(VISION_STEM, ELEMENT_F32, ELEMENT_F32, 0, y, y0, biased, bias, bias_kind, partial,
        columns, 0ul, 0ul, false, 0.0, 0.0, table, table_kind, indices, coefficients);
}

float vision_activate(int code, float value) {
    return code == 1 ? vision_gelu_tanh(value) : code == 2 ? vision_gelu_erf(value) : vision_gelu_quick(value);
}

void vision_put(vision_epilogue out_, uint m, uint n, float value) {
    const uint64_t at = uint64_t(m) * out_.y0 + n;
    if (out_.kind == VISION_LINEAR) {
        float projected = value;
        if (out_.biased)
            projected = value + element_at(out_.bias_kind, out_.bias, n);
        if (out_.clamped)
            projected = min(max(projected, out_.minimum), out_.maximum);
        float published = projected;
        if (out_.activation != 0)
            published = vision_activate(out_.activation, element_round(out_.act, projected));
        else if (out_.gate != 0ul)
            published = element_round(out_.act, projected)
                * element_at(out_.act, out_.gate, uint64_t(m) * out_.gate0 + n);
        else if (out_.residual != 0ul)
            published = element_f32_at(out_.residual + (uint64_t(m) * out_.residual0 + n) * 4ul) + projected;
        element_put(out_.y_kind, out_.y, at, published);
    } else {
        float projected = value;
        if (out_.residual != 0ul)
            projected = element_f32_at(out_.residual + (uint64_t(m) * out_.residual0 + n) * 4ul) + value;
        if (out_.biased)
            projected = projected + element_at(out_.bias_kind, out_.bias, n);
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
// Norm of one F32 row of `width` values by the whole workgroup: two-pass F32
// statistics, centered (a layer norm) or not (the mean taken as 0, a
// root-mean-square norm), out[i * stride] = Y(centered * inverse [* weight]
// [+ bias]). Shared: one float per subgroup at float 0.
void vision_row_norm(const int y_kind, uint64_t x, uint64_t out_, uint64_t stride, const bool weighted, uint64_t weight,
    const int weight_kind, const bool biased, uint64_t bias, const int bias_kind, uint width, bool centered,
    float epsilon) {
    float mean = 0.0;
    if (centered) {
        float sum = 0.0;
        for (uint i = gl_LocalInvocationIndex; i < width; i += gl_WorkGroupSize.x)
            sum += element_f32_at(x + uint64_t(i) * 4ul);
        mean = seismic_div_rn(reduce_group_sum(sum, 0u), float(width));
    }
    float squares = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < width; i += gl_WorkGroupSize.x) {
        const float value = element_f32_at(x + uint64_t(i) * 4ul) - mean;
        squares = seismic_fma_rn(value, value, squares);
    }
    const float inverse = inversesqrt(seismic_div_rn(reduce_group_sum(squares, 0u), float(width)) + epsilon);
    // One expression per form, as the layer norm has always been written.
    for (uint i = gl_LocalInvocationIndex; i < width; i += gl_WorkGroupSize.x) {
        const float centered = element_f32_at(x + uint64_t(i) * 4ul) - mean;
        float value;
        if (weighted && biased)
            value = centered * inverse * element_at(weight_kind, weight, i) + element_at(bias_kind, bias, i);
        else if (weighted)
            value = centered * inverse * element_at(weight_kind, weight, i);
        else if (biased)
            value = centered * inverse + element_at(bias_kind, bias, i);
        else
            value = centered * inverse;
        element_put(y_kind, out_, uint64_t(i) * stride, value);
    }
}

// ---------------------------------------------------------------------------
// One attention operand head row of width w = 4P, prepared by one subgroup
// (lane l owns columns l, l + 32, ...): RMS-normalized (`head_norm`:
// x · rsqrt(Σx² / w + epsilon) · norm[i], `norm` 0: unit weights) when
// `normed`, then, when `rotated`, the 2D rotary embedding: column i < 2P
// pairs with i + 2P; pair p = i % 2P turns by coordinates[p / P] ·
// base^(-(p % P) / P), `log_base` = ln(base). Values are rounded to A, then
// published as f16 at `target`, zero from column w to wp.
float vision_head_value(const int act, uint64_t source, bool normed, uint64_t norm, float inverse, uint i) {
    const float x = element_at(act, source, i);
    return normed ? (norm != 0ul ? x * inverse * element_f32_at(norm + uint64_t(i) * 4ul) : x * inverse) : x;
}

void vision_prepare_head(const int act, uint64_t source, uint64_t target, bool normed, uint64_t norm, float epsilon,
    bool rotated, uint64_t coordinates, float log_base, const uint w, const uint wp) {
    const uint lane = SEISMIC_LANE;
    const uint p = w / 4u;
    float inverse = 1.0;
    if (normed) {
        float squares = 0.0;
        for (uint i = lane; i < w; i += 32u) {
            const float x = element_at(act, source, i);
            squares += x * x;
        }
        inverse = inversesqrt(seismic_div_rn(seismic_subgroup_sum_f32(squares), float(w)) + epsilon);
    }
    for (uint i = lane; i < w; i += 32u) {
        float x = vision_head_value(act, source, normed, norm, inverse, i);
        if (rotated) {
            const uint pair = i % (2u * p);
            const float frequency = precise_exp(seismic_div_rn(-log_base * float(pair % p), float(p)));
            const float angle = float(element_i32_at(coordinates + uint64_t(pair / p) * 4ul)) * frequency;
            float c;
            const float s = rotary_sincos(angle, c);
            const float partner = vision_head_value(act, source, normed, norm, inverse, i < 2u * p ? i + 2u * p : i - 2u * p);
            x = i < 2u * p ? x * c - partner * s : x * c + partner * s;
        }
        element_put(ELEMENT_F16, target, i, element_round(act, x));
    }
    for (uint i = w + lane; i < wp; i += 32u)
        element_put(ELEMENT_F16, target, i, 0.0);
}

// ---------------------------------------------------------------------------
// Non-causal attention of one head: workgroup (tile of VISION_ATTEND_ROWS
// query rows, head), 16 query rows per subgroup, keys streamed in
// FLASH_KEYS-row tiles, one online-softmax pass of `lib/attention/flash.glsl`
// per VISION_WINDOW output columns. `operands` are the f16 rows
// [3][heads][padded][wp] (zero past the head width w); the output row r of
// head h is A at out_[r * heads * w + h * w ..], rounded from F32. Scores
// are scaled by `score_scale`. Without `spans` (0) every row attends to every
// row; with them row r attends to rows [spans[2r], spans[2r + 1]), the
// workgroup walking the union of its rows' spans. Shared: the K/V tile
// (FLASH_KEYS x (wp + 8) f16), then FLASH_SCRATCH_FLOATS floats per subgroup.
void vision_attend(const int act, uint64_t operands, uint64_t out_, uint rows, uint padded, uint heads, const uint w,
    const uint wp, float score_scale, uint64_t spans, uint tile, uint head) {
    const uint lane = SEISMIC_LANE;
    const uint64_t plane = uint64_t(heads) * padded * wp * 2ul;   // bytes of one of Q, K, V
    const uint64_t head_rows = uint64_t(head) * padded * wp * 2ul;
    const uint64_t keys = operands + plane + head_rows;
    const uint64_t values = operands + 2ul * plane + head_rows;
    const uint block_row = tile * VISION_ATTEND_ROWS + 16u * SEISMIC_SUBGROUP;
    const uint64_t block_queries = operands + head_rows + uint64_t(block_row) * wp * 2ul;
    const uint scratch = (FLASH_KEYS * flash_pitch(wp)) / 2u + SEISMIC_SUBGROUP * FLASH_SCRATCH_FLOATS;
    const float scale = score_scale * 1.4426950408889634;

    // The keys of this lane's row, and the keys the workgroup walks.
    uint row_first = 0u, row_end = rows, walk_first = 0u, walk_end = rows;
    if (spans != 0ul) {
        const uint span_row = min(block_row + lane % 16u, rows - 1u);
        row_first = uint(element_i32_at(spans + uint64_t(span_row) * 8ul));
        row_end = uint(element_i32_at(spans + uint64_t(span_row) * 8ul + 4ul));
        walk_first = rows;
        walk_end = 0u;
        for (uint r = tile * VISION_ATTEND_ROWS; r < min(tile * VISION_ATTEND_ROWS + VISION_ATTEND_ROWS, rows); ++r) {
            walk_first = min(walk_first, uint(element_i32_at(spans + uint64_t(r) * 8ul)));
            walk_end = max(walk_end, uint(element_i32_at(spans + uint64_t(r) * 8ul + 4ul)));
        }
    }

    const uint64_t width = uint64_t(heads) * w;
    flash_output o;
    // One online-softmax pass per output window.
    const uint windows = (wp + VISION_WINDOW - 1u) / VISION_WINDOW;
    for (uint pass = 0u; pass < windows; ++pass) {
        const uint column0 = pass * VISION_WINDOW;
        flash_output_clear(wp, VISION_WINDOW, o);
        flash_softmax softmax = flash_softmax_start();
        for (uint first = walk_first; first < walk_end; first += FLASH_KEYS) {
            barrier();
            flash_stage(ELEMENT_F16, keys, uint64_t(wp), 0ul, int(first), int(rows), wp, 0u);
            barrier();
            flash_scores(block_queries, uint64_t(wp), wp, 0u, scratch);
            float s[16];
            flash_lane_scores(scratch, s);
            [[unroll]] for (uint j = 0u; j < 16u; ++j) {
                const uint key = first + 16u * (lane / 16u) + j;
                s[j] *= scale;
                if (key >= rows || key < row_first || key >= row_end)
                    s[j] = -VISION_INF;
            }
            const float alpha = flash_online(softmax, s);
            const uint p_half = flash_publish_probabilities(scratch, s);
            flash_rescale(scratch, alpha, wp, VISION_WINDOW, o);
            barrier();
            flash_stage(ELEMENT_F16, values, uint64_t(wp), 0ul, int(first), int(rows), wp, 0u);
            barrier();
            flash_accumulate(p_half, 0u, wp, VISION_WINDOW, column0, o);
        }
        const float denominator = flash_denominator(softmax);

        barrier();
        // Unrolled, so every fragment index is a constant.
        [[unroll]] for (uint q = 0u; q < VISION_WINDOW / 2u; ++q) {
            if (q >= flash_window(wp, VISION_WINDOW) / 2u)
                break;
            const float value = flash_output_value(o, scratch, q);
            const uint r = flash_output_row(q);
            const float row_denominator = seismic_shuffle(denominator, r);
            const uint row = block_row + r;
            const uint column = flash_output_column(wp, VISION_WINDOW, column0, q);
            if (row < rows && column < w)
                element_put(act, out_, uint64_t(row) * width + uint64_t(head) * w + column,
                    seismic_div_rn(value, row_denominator));
        }
    }
}
