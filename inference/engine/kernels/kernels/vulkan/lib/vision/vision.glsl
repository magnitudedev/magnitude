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

// The projections' GEMM shape (`projection.glsl`): 64 x 64 tiles of 32 x 32
// sub-tiles, 128 invocations.
#define VISION_TILE_M 64u
#define VISION_TILE_N 64u
#define PROJECTION_GEMM_TM VISION_TILE_M
#define PROJECTION_GEMM_TN VISION_TILE_N
#define PROJECTION_GEMM_WM 32u
#define PROJECTION_GEMM_WN 32u
#include "../projection/projection.glsl"
#include "../projection/gemm.glsl"
#include "../core/rotary.glsl"
#include "../core/norm.glsl"
#include "../attention/flash.glsl"
#include "../core/precise.glsl"

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
//           y = Y(act(projection)); with a gate (A rows, 0: none)
//           y = Y(projection · gate[m, n]), rounded once; otherwise
//           y = Y(projection), plus residual[m, n] (F32) when bound
//   Stem    y = ((partial[m, n] when bound) + acc [+ bias[n]]) + blended,
//           F32, with blended the four table corners times their
//           coefficients summed in order (zero without a table)

#define VISION_LINEAR 0
#define VISION_STEM 1

struct vision_epilogue {
    int kind;
    int act;                // Linear: ELEMENT_* of A
    int activation;         // Linear: 0, 1 tanh GELU, 2 erf GELU, 3 quick GELU, 4 squared ReLU
    Out y;
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
    uint64_t table;         // Stem: the position table [L, columns], or 0
    int table_kind;
    uint64_t indices;       // Stem: [M, 4] i32
    uint64_t coefficients;  // Stem: [M, 4] F32
};

vision_epilogue vision_linear_epilogue(const int act, int activation, Out y, uint64_t y0,
    bool biased, uint64_t bias, const int bias_kind, uint64_t residual, uint64_t residual0, uint64_t gate,
    uint64_t gate0, bool clamped, float minimum, float maximum) {
    return vision_epilogue(VISION_LINEAR, act, activation, y, y0, biased, bias, bias_kind, residual,
        residual0, gate, gate0, clamped, minimum, maximum, 0ul, ELEMENT_F32, 0ul, 0ul);
}

vision_epilogue vision_stem(Out y, uint64_t y0, bool biased, uint64_t bias, const int bias_kind,
    uint64_t partial, uint64_t columns, uint64_t table, const int table_kind, uint64_t indices,
    uint64_t coefficients) {
    return vision_epilogue(VISION_STEM, ELEMENT_F32, 0, y, y0, biased, bias, bias_kind, partial,
        columns, 0ul, 0ul, false, 0.0, 0.0, table, table_kind, indices, coefficients);
}

float vision_relu_squared(float value) {
    const float positive = max(value, 0.0);
    return positive * positive;
}

float vision_silu(float value) {
    return seismic_div_rn(value, 1.0 + precise_exp(-value));
}

float vision_activate(int code, float value) {
    return code == 1 ? vision_gelu_tanh(value)
         : code == 2 ? vision_gelu_erf(value)
         : code == 3 ? vision_gelu_quick(value)
         : code == 4 ? vision_relu_squared(value)
                     : vision_silu(value);
}

vec2 vision_load2(const int kind, uint64_t base, uint64_t index) {
    if (kind == ELEMENT_F32) {
        const uint64_t address = base + index * 4ul;
        if ((address & 7ul) == 0ul)
            return uintBitsToFloat(element_uvec2_at(address));
    } else {
        const uint64_t address = base + index * 2ul;
        if ((address & 3ul) == 0ul)
            return element_unpack2(kind, element_u32_at(address));
    }
    return vec2(element_at(kind, base, index), element_at(kind, base, index + 1ul));
}

vec4 vision_load4(const int kind, uint64_t base, uint64_t index) {
    if (kind == ELEMENT_F32) {
        const uint64_t address = base + index * 4ul;
        if ((address & 15ul) == 0ul)
            return element_vec4_at(address);
    } else {
        const uint64_t address = base + index * 2ul;
        if ((address & 7ul) == 0ul) {
            const uvec2 raw = element_uvec2_at(address);
            return vec4(element_unpack2(kind, raw.x), element_unpack2(kind, raw.y));
        }
    }
    return vec4(element_at(kind, base, index), element_at(kind, base, index + 1ul),
        element_at(kind, base, index + 2ul), element_at(kind, base, index + 3ul));
}

ivec4 vision_load_ivec4(uint64_t address) {
    if ((address & 15ul) == 0ul)
        return ivec4(element_uvec4_at(address));
    return ivec4(element_i32_at(address), element_i32_at(address + 4ul),
        element_i32_at(address + 8ul), element_i32_at(address + 12ul));
}

vec4 vision_load_vec4(uint64_t address) {
    if ((address & 15ul) == 0ul)
        return element_vec4_at(address);
    return vec4(element_f32_at(address), element_f32_at(address + 4ul),
        element_f32_at(address + 8ul), element_f32_at(address + 12ul));
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
            published = vision_activate(out_.activation, projected);
        else if (out_.gate != 0ul)
            published = projected
                * element_at(out_.act, out_.gate, uint64_t(m) * out_.gate0 + n);
        else if (out_.residual != 0ul)
            published = element_f32_at(out_.residual + (uint64_t(m) * out_.residual0 + n) * 4ul) + projected;
        out_put(out_.y, at, published);
    } else {
        float projected = value;
        if (out_.residual != 0ul)
            projected = element_f32_at(out_.residual + (uint64_t(m) * out_.residual0 + n) * 4ul) + value;
        if (out_.biased)
            projected = projected + element_at(out_.bias_kind, out_.bias, n);
        float blended = 0.0;
        if (out_.table != 0ul) {
            const ivec4 idx = vision_load_ivec4(out_.indices + uint64_t(m) * 16ul);
            const vec4 coef = vision_load_vec4(out_.coefficients + uint64_t(m) * 16ul);
            [[unroll]] for (uint i = 0u; i < 4u; ++i)
                blended += element_at(out_.table_kind, out_.table, uint64_t(idx[i]) * out_.residual0 + n) * coef[i];
        }
        out_put(out_.y, at, projected + blended);
    }
}

void vision_put2(vision_epilogue out_, uint m, uint n, float first, float second) {
    const uint64_t at = uint64_t(m) * out_.y0 + n;
    if (out_.kind == VISION_LINEAR) {
        vec2 projected = vec2(first, second);
        if (out_.biased)
            projected = projected + vision_load2(out_.bias_kind, out_.bias, uint64_t(n));
        if (out_.clamped)
            projected = min(max(projected, vec2(out_.minimum)), vec2(out_.maximum));
        vec2 published = projected;
        if (out_.activation != 0) {
            published.x = vision_activate(out_.activation, projected.x);
            published.y = vision_activate(out_.activation, projected.y);
        } else if (out_.gate != 0ul) {
            published = projected * vision_load2(out_.act, out_.gate, uint64_t(m) * out_.gate0 + n);
        } else if (out_.residual != 0ul) {
            published = vision_load2(ELEMENT_F32, out_.residual, uint64_t(m) * out_.residual0 + n) + projected;
        }
        out_put2(out_.y, at, published.x, published.y);
    } else {
        vec2 projected = vec2(first, second);
        if (out_.residual != 0ul)
            projected = vision_load2(ELEMENT_F32, out_.residual, uint64_t(m) * out_.residual0 + n) + projected;
        if (out_.biased)
            projected = projected + vision_load2(out_.bias_kind, out_.bias, uint64_t(n));
        vec2 blended = vec2(0.0);
        if (out_.table != 0ul) {
            const ivec4 idx = vision_load_ivec4(out_.indices + uint64_t(m) * 16ul);
            const vec4 coef = vision_load_vec4(out_.coefficients + uint64_t(m) * 16ul);
            [[unroll]] for (uint i = 0u; i < 4u; ++i) {
                const vec2 t = vision_load2(out_.table_kind, out_.table, uint64_t(idx[i]) * out_.residual0 + n);
                blended.x += t.x * coef[i];
                blended.y += t.y * coef[i];
            }
        }
        out_put2(out_.y, at, projected.x + blended.x, projected.y + blended.y);
    }
}

// One 64 x 64 GEMM tile of a vision projection: output rows tm_index * 64 ..,
// weight rows tn_index * 64 .. of `w` (N = rows, K = k).
void vision_gemm(const int wk, projection_prologue in_, vision_epilogue out_, projection_weights w, uint m_rows,
    uint rows, uint k, uint tm_index, uint tn_index) {
    const uint first = tn_index * VISION_TILE_N;
    projection_gemm_accumulate(wk, wk, false, in_, w, w, first, rows, tm_index * VISION_TILE_M, m_rows, k, 0u,
        (k + PROJECTION_GEMM_K - 1u) / PROJECTION_GEMM_K, m_rows);
    [[unroll]] for (uint q = 0u; q < PROJECTION_GEMM_PAIRS; ++q) {
        const vec2 c = projection_gemm_pair(q);
        const uint m = tm_index * VISION_TILE_M + projection_gemm_pair_row(q);
        const uint n = first + projection_gemm_pair_column(q);
        if (m < m_rows && n + 1u < rows)
            vision_put2(out_, m, n, c.x, c.y);
        else if (m < m_rows && n < rows)
            vision_put(out_, m, n, c.x);
    }
}

// Dense weight rows of a row-major matrix of `rows` (its slot's,
// <seismic/packets.glsl>).
projection_weights vision_weights(uint64_t base, packets_rows16 rows, uint k) {
    return projection_weights(base, rows, k, 0ul, 1ul, 0u);
}

void vision_norm_publish4(Out out_, uint64_t stride, const bool weighted, uint64_t weight, const int weight_kind,
    const bool biased, uint64_t bias, const int bias_kind, uint i, vec4 xv, float mean, float inverse,
    int activation) {
    const vec4 wv = weighted ? vision_load4(weight_kind, weight, uint64_t(i)) : vec4(0.0);
    const vec4 bv = biased ? vision_load4(bias_kind, bias, uint64_t(i)) : vec4(0.0);
    vec4 yv;
    yv.x = norm_apply(xv.x - mean, inverse, weighted, wv.x, biased, bv.x);
    yv.y = norm_apply(xv.y - mean, inverse, weighted, wv.y, biased, bv.y);
    yv.z = norm_apply(xv.z - mean, inverse, weighted, wv.z, biased, bv.z);
    yv.w = norm_apply(xv.w - mean, inverse, weighted, wv.w, biased, bv.w);
    if (activation != 0) {
        yv.x = vision_activate(activation, yv.x);
        yv.y = vision_activate(activation, yv.y);
        yv.z = vision_activate(activation, yv.z);
        yv.w = vision_activate(activation, yv.w);
    }
    if (stride == 1ul) {
        out_put2(out_, uint64_t(i), yv.x, yv.y);
        out_put2(out_, uint64_t(i + 2u), yv.z, yv.w);
    } else {
        out_put(out_, uint64_t(i) * stride, yv.x);
        out_put(out_, uint64_t(i + 1u) * stride, yv.y);
        out_put(out_, uint64_t(i + 2u) * stride, yv.z);
        out_put(out_, uint64_t(i + 3u) * stride, yv.w);
    }
}

void vision_norm_publish1(Out out_, uint64_t stride, const bool weighted, uint64_t weight, const int weight_kind,
    const bool biased, uint64_t bias, const int bias_kind, uint i, float xv, float mean, float inverse,
    int activation) {
    const float w = weighted ? element_at(weight_kind, weight, i) : 0.0;
    const float b = biased ? element_at(bias_kind, bias, i) : 0.0;
    const float normalized = norm_apply(xv - mean, inverse, weighted, w, biased, b);
    out_put(out_, uint64_t(i) * stride, activation != 0 ? vision_activate(activation, normalized) : normalized);
}

// ---------------------------------------------------------------------------
// Norm of one F32 row of `width` values by the whole workgroup: `norm_*`'s
// two-pass statistics, centered (a layer norm) or not (the mean taken as 0, a
// root-mean-square norm), out[i * stride] = Y(centered * inverse [* weight]
// [+ bias]), the statistics spread by `SEISMIC_THREAD`, as their group sums
// need. Shared: one float per subgroup at float 0.
void vision_row_norm(uint64_t x, Out out_, uint64_t stride, const bool weighted, uint64_t weight,
    const int weight_kind, const bool biased, uint64_t bias, const int bias_kind, uint width, bool centered,
    float epsilon, int activation) {
    const uint tid = SEISMIC_THREAD;
    if ((x & 15ul) == 0ul && (width & 3u) == 0u) {
        const uint vecs = width / 4u;
        vec4 cached[2];
        float mean = 0.0;
        float squares = 0.0;
        if (centered) {
            float sum = 0.0;
            [[unroll]] for (uint slot = 0u; slot < 2u; ++slot) {
                const uint v = tid + slot * gl_WorkGroupSize.x;
                if (v < vecs) {
                    const vec4 xv = element_vec4_at(x + uint64_t(v) * 16ul);
                    cached[slot] = xv;
                    sum += (xv.x + xv.y) + (xv.z + xv.w);
                }
            }
            seismic_strided_from(uint, v, 2u * gl_WorkGroupSize.x, tid, vecs, gl_WorkGroupSize.x) {
                const vec4 xv = element_vec4_at(x + uint64_t(v) * 16ul);
                sum += (xv.x + xv.y) + (xv.z + xv.w);
            }
            mean = norm_mean(reduce_group_sum(sum, 0u), width);
            [[unroll]] for (uint slot = 0u; slot < 2u; ++slot) {
                const uint v = tid + slot * gl_WorkGroupSize.x;
                if (v < vecs) {
                    const vec4 cv = cached[slot] - mean;
                    squares = norm_square(cv.x, squares);
                    squares = norm_square(cv.y, squares);
                    squares = norm_square(cv.z, squares);
                    squares = norm_square(cv.w, squares);
                }
            }
        } else {
            [[unroll]] for (uint slot = 0u; slot < 2u; ++slot) {
                const uint v = tid + slot * gl_WorkGroupSize.x;
                if (v < vecs) {
                    const vec4 xv = element_vec4_at(x + uint64_t(v) * 16ul);
                    cached[slot] = xv;
                    squares = norm_square(xv.x, squares);
                    squares = norm_square(xv.y, squares);
                    squares = norm_square(xv.z, squares);
                    squares = norm_square(xv.w, squares);
                }
            }
        }
        seismic_strided_from(uint, v, 2u * gl_WorkGroupSize.x, tid, vecs, gl_WorkGroupSize.x) {
            const vec4 cv = element_vec4_at(x + uint64_t(v) * 16ul) - mean;
            squares = norm_square(cv.x, squares);
            squares = norm_square(cv.y, squares);
            squares = norm_square(cv.z, squares);
            squares = norm_square(cv.w, squares);
        }
        const float inverse = norm_inverse(reduce_group_sum(squares, 0u), width, epsilon);
        [[unroll]] for (uint slot = 0u; slot < 2u; ++slot) {
            const uint v = tid + slot * gl_WorkGroupSize.x;
            if (v < vecs)
                vision_norm_publish4(out_, stride, weighted, weight, weight_kind, biased, bias, bias_kind,
                    v * 4u, cached[slot], mean, inverse, activation);
        }
        for (uint v = tid + 2u * gl_WorkGroupSize.x; v < vecs; v += gl_WorkGroupSize.x)
            vision_norm_publish4(out_, stride, weighted, weight, weight_kind, biased, bias, bias_kind,
                v * 4u, element_vec4_at(x + uint64_t(v) * 16ul), mean, inverse, activation);
        return;
    }

    float cached[8];
    float mean = 0.0;
    float squares = 0.0;
    if (centered) {
        float sum = 0.0;
        [[unroll]] for (uint slot = 0u; slot < 8u; ++slot) {
            const uint i = tid + slot * gl_WorkGroupSize.x;
            if (i < width) {
                const float xv = element_f32_at(x + uint64_t(i) * 4ul);
                cached[slot] = xv;
                sum += xv;
            }
        }
        seismic_strided_from(uint, i, 8u * gl_WorkGroupSize.x, tid, width, gl_WorkGroupSize.x)
            sum += element_f32_at(x + uint64_t(i) * 4ul);
        mean = norm_mean(reduce_group_sum(sum, 0u), width);
        [[unroll]] for (uint slot = 0u; slot < 8u; ++slot) {
            const uint i = tid + slot * gl_WorkGroupSize.x;
            if (i < width)
                squares = norm_square(cached[slot] - mean, squares);
        }
    } else {
        [[unroll]] for (uint slot = 0u; slot < 8u; ++slot) {
            const uint i = tid + slot * gl_WorkGroupSize.x;
            if (i < width) {
                const float xv = element_f32_at(x + uint64_t(i) * 4ul);
                cached[slot] = xv;
                squares = norm_square(xv, squares);
            }
        }
    }
    seismic_strided_from(uint, i, 8u * gl_WorkGroupSize.x, tid, width, gl_WorkGroupSize.x)
        squares = norm_square(element_f32_at(x + uint64_t(i) * 4ul) - mean, squares);
    const float inverse = norm_inverse(reduce_group_sum(squares, 0u), width, epsilon);
    [[unroll]] for (uint slot = 0u; slot < 8u; ++slot) {
        const uint i = tid + slot * gl_WorkGroupSize.x;
        if (i < width)
            vision_norm_publish1(out_, stride, weighted, weight, weight_kind, biased, bias, bias_kind,
                i, cached[slot], mean, inverse, activation);
    }
    for (uint i = tid + 8u * gl_WorkGroupSize.x; i < width; i += gl_WorkGroupSize.x)
        vision_norm_publish1(out_, stride, weighted, weight, weight_kind, biased, bias, bias_kind,
            i, element_f32_at(x + uint64_t(i) * 4ul), mean, inverse, activation);
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

void vision_prepare_head(const int act, uint64_t source, uint64_t target, bool live, bool normed, uint64_t norm,
    float epsilon, bool rotated, uint64_t coordinates, float log_base, const uint w, const uint wp) {
    const uint lane = SEISMIC_LANE;
    const uint p = w / 4u;
    float inverse = 1.0;
    // The norm's sum is taken wherever a hardware subgroup holds a normed
    // row (its two logical subgroups on 64 lanes may hold rows of different
    // planes); a row without one keeps 1.
    bool summing;
    if (seismic_steer(summing, normed)) {
        float squares = 0.0;
        seismic_strided(uint, i, lane, w, 32u) {
            const float x = element_at(act, source, i);
            squares += x * x;
        }
        const float total = seismic_subgroup_sum_f32(squares);
        if (summing)
            inverse = inversesqrt(seismic_div_rn(total, float(w)) + epsilon);
    }
    if (!live)
        return;
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
// [heads + 2 * (heads / group)][padded][wp]: the query, then key, then value
// head rows (zero past the head width w), query head h reading key and value
// head h / group; the output row r of
// head h is A at out_[r * heads * w + h * w ..], rounded from F32. Scores
// are scaled by `score_scale`. Without `spans` (0) every row attends to every
// row; with them row r attends to rows [spans[2r], spans[2r + 1]), the
// workgroup walking the union of its rows' spans. Shared: the K/V tile
// (FLASH_KEYS x (wp + 8) f16), then FLASH_SCRATCH_FLOATS floats per subgroup.
void vision_attend(const int act, uint64_t operands, Out out_, uint rows, uint padded, uint heads,
    const uint group, const uint w, const uint wp, float score_scale, uint64_t spans, uint tile, uint head) {
    const uint lane = SEISMIC_LANE;
    const uint64_t head_bytes = uint64_t(padded) * wp * 2ul;   // bytes of one head's rows
    const uint64_t head_rows = uint64_t(head) * head_bytes;
    const uint64_t keys = operands + uint64_t(heads + head / group) * head_bytes;
    const uint64_t values = keys + uint64_t(heads / group) * head_bytes;
    const uint block_row = tile * VISION_ATTEND_ROWS + 16u * SEISMIC_SUBGROUP;
    const uint64_t block_queries = operands + head_rows + uint64_t(block_row) * wp * 2ul;
    const uint scratch = (FLASH_KEYS * flash_pitch(wp)) / 2u + SEISMIC_SUBGROUP * FLASH_SCRATCH_FLOATS;
    const float scale = score_scale * 1.4426950408889634;

    // The keys of this lane's row, and the keys the workgroup walks. Every
    // row in [tile * VISION_ATTEND_ROWS, min(tile * VISION_ATTEND_ROWS +
    // VISION_ATTEND_ROWS, rows)) is held by at least one lane, so reducing
    // the lanes' own bounds yields the tile's union with zero extra loads.
    uint row_first = 0u, row_end = rows, walk_first = 0u, walk_end = rows;
    if (spans != 0ul) {
        const uint span_row = min(block_row + lane % 16u, rows - 1u);
        const uint64_t span_addr = spans + uint64_t(span_row) * 8ul;
        if ((span_addr & 7ul) == 0ul) {
            const uvec2 span = element_uvec2_at(span_addr);
            row_first = span.x;
            row_end = span.y;
        } else {
            row_first = uint(element_i32_at(span_addr));
            row_end = uint(element_i32_at(span_addr + 4ul));
        }
        const uint sg_first = seismic_redux_min_u32(row_first);
        const uint sg_end = seismic_redux_max_u32(row_end);
        if (lane == 0u) {
            seismic_shared_u32[SEISMIC_SUBGROUP] = sg_first;
            seismic_shared_u32[SEISMIC_SUBGROUPS + SEISMIC_SUBGROUP] = sg_end;
        }
        barrier();
        walk_first = seismic_shared_u32[0u];
        walk_end = seismic_shared_u32[SEISMIC_SUBGROUPS];
        for (uint s = 1u; s < SEISMIC_SUBGROUPS; ++s) {
            walk_first = min(walk_first, seismic_shared_u32[s]);
            walk_end = max(walk_end, seismic_shared_u32[SEISMIC_SUBGROUPS + s]);
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
                out_put(out_, uint64_t(row) * width + uint64_t(head) * w + column,
                    seismic_div_rn(value, row_denominator));
        }
    }
}
