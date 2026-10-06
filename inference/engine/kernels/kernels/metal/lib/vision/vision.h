// Shared pieces of the vision tower entries (`vision_*` in vision.seismic).
// The projections run on the projection library's GEMM (`projection::gemm`
// over dense weight rows) with the vision epilogues below; the row norm, the
// 2D rotary embedding and the full (non-causal) attention are the
// vision-specific launches.
//
// Numerics (`vision.seismic`): the residual stream is F32; the published
// intermediates (normalized rows, projections feeding a GEMM or the
// attention, rotated queries and keys, attention output, activations) are
// rounded to the activation element A (`element::Bf16`, `element::F16`);
// bias and norm vectors are any dense element.
//
// Every dense operand is bound canonically (row-major, unit innermost
// stride). This file is independent of any entry ABI.

#include "../core/activation.h"
#include "../projection/projection.h"
#include "../core/reduce.h"

namespace vision {

// The projection packet of a dense weight representation kind (0 = f32,
// 1 = bf16, 2 = f16).
template <int KIND> struct dense_packet;
template <> struct dense_packet<0> { typedef packets::Dense<element::F32> type; };
template <> struct dense_packet<1> { typedef packets::Dense<element::Bf16> type; };
template <> struct dense_packet<2> { typedef packets::Dense<element::F16> type; };

// GEMM tiles of every vision projection.
constant constexpr uint TILE_M = 64;
constant constexpr uint TILE_N = 64;

// Rows of dense weight `base` ([N, K] row-major, `stride` elements per row).
template <typename W>
inline projection::Weights<W> weight(device const uchar *base, ulong stride, uint bytes, uint k) {
    return projection::Weights<W>{base, packets::Rows16{stride * bytes, 0, 0, 0, 0}, k, nullptr};
}

// ---------------------------------------------------------------------------
// Scalar functions.

inline float gelu_tanh(float value) {
    const float argument = 0.7978845608028654f * (value + 0.044715f * value * value * value);
    const float hyperbolic = 2.0f / (1.0f + metal::precise::exp(-2.0f * argument)) - 1.0f;
    return 0.5f * value * (1.0f + hyperbolic);
}

// erf (Numerical Recipes `erfcc`, |error| < 1.2e-7).
inline float erf(float x) {
    const float z = metal::abs(x);
    const float t = 1.0f / (1.0f + 0.5f * z);
    const float r = t * metal::precise::exp(-z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f
        + t * (0.09678418f + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f
        + t * (-0.82215223f + t * 0.17087277f)))))))));
    return x >= 0.0f ? 1.0f - r : r - 1.0f;
}

inline float gelu_erf(float value) {
    return 0.5f * value * (1.0f + erf(value * 0.7071067811865476f));
}

inline float gelu_quick(float value) {
    return value / (1.0f + metal::precise::exp(-1.702f * value));
}

// The activation of `vision_linear`: 1 tanh GELU, 2 erf GELU, 3 quick GELU.
inline float activate(int code, float value) {
    return code == 1 ? gelu_tanh(value) : code == 2 ? gelu_erf(value) : gelu_quick(value);
}

// ---------------------------------------------------------------------------
// The `vision_linear` epilogue. `store(m, n, acc)` receives the F32 product
// of output (m, n); the projection is acc, plus bias[n] when BIAS, clamped to
// [minimum, maximum] when CLAMP. With `activation` nonzero
// y = Y(act(round_A(projection))); with GATE y = Y(round_A(projection) ·
// gate[m, n]); otherwise y = Y(projection), plus the F32 residual row when
// RESIDUAL.
template <typename A, typename B, typename Y, bool BIAS, bool RESIDUAL, bool GATE, bool CLAMP>
struct output_linear {
    device uchar *y;
    ulong stride;
    device const uchar *bias;
    device const float *residual;
    ulong residual_stride;
    device const uchar *gate;
    ulong gate_stride;
    float minimum;
    float maximum;
    int activation;
    void store(uint m, uint n, float value) const {
        float projected = value;
        if (BIAS)
            projected = value + element::at<B>(bias, n);
        if (CLAMP)
            projected = metal::min(metal::max(projected, minimum), maximum);
        float published = projected;
        if (activation != 0)
            published = activate(activation, A::round(projected));
        else if (GATE)
            published = A::round(projected) * element::at<A>(gate, ulong(m) * gate_stride + n);
        else if (RESIDUAL)
            published = residual[ulong(m) * residual_stride + n] + projected;
        element::put<Y>(y, ulong(m) * stride + n, published);
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// ---------------------------------------------------------------------------
// Norm of one F32 row of `width` values (THREADS threads): two-pass F32
// statistics, centered (a layer norm) or not (the mean taken as 0, a
// root-mean-square norm), out = Y(centered * inverse [* weight] [+ bias]).
// `partials` holds THREADS / 32 floats.
template <uint THREADS, typename Y, typename WN, typename BN, bool WEIGHT, bool BIAS>
inline void norm(device const float *row, device uchar *out, ulong stride, device const uchar *weight,
    device const uchar *bias, uint width, bool centered, float epsilon, threadgroup float *partials,
    uint thread_index, uint sg, uint lane) {
    float mean = 0.0f;
    if (centered) {
        float sum = 0.0f;
        for (uint i = thread_index; i < width; i += THREADS)
            sum += row[i];
        mean = reduce::group_sum<THREADS / 32>(sum, partials, sg, lane) / float(width);
    }
    float squares = 0.0f;
    for (uint i = thread_index; i < width; i += THREADS) {
        const float value = row[i] - mean;
        squares = metal::fma(value, value, squares);
    }
    const float inverse =
        metal::rsqrt(reduce::group_sum<THREADS / 32>(squares, partials, sg, lane) / float(width) + epsilon);
    // One expression per form: the compiler contracts a product and a sum
    // only within an expression.
    for (uint i = thread_index; i < width; i += THREADS) {
        float value;
        if (WEIGHT && BIAS)
            value = (row[i] - mean) * inverse * element::at<WN>(weight, i) + element::at<BN>(bias, i);
        else if (WEIGHT)
            value = (row[i] - mean) * inverse * element::at<WN>(weight, i);
        else if (BIAS)
            value = (row[i] - mean) * inverse + element::at<BN>(bias, i);
        else
            value = (row[i] - mean) * inverse;
        element::put<Y>(out, ulong(i) * stride, value);
    }
}

// ---------------------------------------------------------------------------
// One attention operand head row of width W = 4P, prepared by one simdgroup
// (lane l owns columns l, l + 32, ...): RMS-normalized with `norm` (`head_norm`:
// x · rsqrt(Σx² / W + epsilon) · norm[i]) when `normed` (`norm` null: unit
// weights), then, when `rotated`, the 2D rotary embedding: column i < 2P
// pairs with i + 2P; pair p = i % 2P turns by coordinates[p / P] ·
// base^(-(p % P) / P), `log_base` = ln(base). Written rounded to A at
// `target`, zero from column W to WP.
template <typename A, uint W, uint WP>
inline void prepare_head(device const typename A::storage *source, device typename A::storage *target,
    bool normed, device const float *norm, float epsilon, bool rotated, device const int *coordinates,
    float log_base, uint lane) {
    constexpr uint P = W / 4;
    float inverse = 1.0f;
    if (normed) {
        float squares = 0.0f;
        for (uint i = lane; i < W; i += 32) {
            const float x = A::load(source[i]);
            squares += x * x;
        }
        for (ushort offset = 16; offset > 0; offset /= 2)
            squares += simd_shuffle_xor(squares, offset);
        inverse = metal::rsqrt(squares / float(W) + epsilon);
    }
    auto value = [&](uint i) {
        const float x = A::load(source[i]);
        return normed ? (norm ? x * inverse * norm[i] : x * inverse) : x;
    };
    for (uint i = lane; i < W; i += 32) {
        const float x = value(i);
        float published = x;
        if (rotated) {
            const uint pair = i % (2 * P);
            const float frequency = metal::precise::exp(-log_base * float(pair % P) / float(P));
            const float angle = float(coordinates[pair / P]) * frequency;
            const float c = metal::precise::cos(angle), s = metal::precise::sin(angle);
            const float partner = value(i < 2 * P ? i + 2 * P : i - 2 * P);
            published = i < 2 * P ? x * c - partner * s : x * c + partner * s;
        }
        target[i] = A::store(published);
    }
    for (uint i = W + lane; i < WP; i += 32)
        target[i] = A::store(0.0f);
}

// ---------------------------------------------------------------------------
// Non-causal attention. A threadgroup owns ATTEND_ROWS query rows of one head
// (simdgroup s: rows 8s .. 8s + 7 of the tile) and walks the keys of its rows'
// spans in tiles of ATTEND_KEYS: K and V staged in threadgroup memory, scores =
// Q K^T on the matrix units (Q's fragments held in registers), scaled into the
// exp2 domain in F32, the online softmax, and the F32 output accumulated from
// probabilities rounded to half. `qkv` is the [rows, 3, H, WP] operand rows
// (queries and keys prepared, zero past the head width W); rows past `rows` of
// the last query tile are read (the buffer holds whole tiles) but never
// stored. Without `spans` every row attends to every row; with them row r
// attends to rows [spans[2r], spans[2r + 1]), the tile walking the union of
// its rows' spans (`bounds`, two words of threadgroup memory).
constant constexpr uint ATTEND_ROWS = 64;
constant constexpr uint ATTEND_KEYS = 32;

template <uint W>
constexpr uint attend_pitch() { return W + 8; }

// Copies rows [first, first + ATTEND_KEYS) of the key or value part at
// column `column` of the projection into `staged`; rows at or past `rows`
// are zero.
template <typename S, uint W, uint THREADS>
inline void attend_stage(threadgroup S *staged, device const S *qkv, ulong row_stride, ulong column,
    uint first, uint rows, uint thread_index) {
    constexpr uint PIECES = W / 8;
    for (uint item = thread_index; item < ATTEND_KEYS * PIECES; item += THREADS) {
        const uint k = item / PIECES;
        const uint c = (item % PIECES) * 8;
        uint4 bits = uint4(0);
        if (first + k < rows)
            bits = *reinterpret_cast<device const uint4 *>(qkv + ulong(first + k) * row_stride + column + c);
        *reinterpret_cast<threadgroup uint4 *>(staged + k * attend_pitch<W>() + c) = bits;
    }
}

template <typename S, uint W, uint WP>
inline void attend(device const S *qkv, device S *out, uint rows, uint heads, float scale,
    device const int *spans, threadgroup S *keys, threadgroup S *values, threadgroup uint *bounds, uint tile,
    uint head, uint thread_index, uint simd, uint lane) {
    constexpr uint THREADS = ATTEND_ROWS * 4;
    constexpr uint PITCH = attend_pitch<WP>();
    constexpr uint DB = WP / 8;
    constexpr uint KB = ATTEND_KEYS / 8;
    const ulong width = ulong(heads) * WP;
    const ulong row_stride = 3 * width;
    const float scale2 = scale * 1.4426950408889634f;

    // This lane's fragment coordinates: row fm, columns fn and fn + 1.
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + ((lane / 2) % 4);
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const uint first_row = tile * ATTEND_ROWS + simd * 8;

    // This lane's row's keys, and the keys the tile walks.
    uint row_first = 0, row_end = rows, walk_first = 0, walk_end = rows;
    if (spans) {
        const uint span_row = metal::min(first_row + fm, rows - 1);
        row_first = uint(spans[span_row * 2]);
        row_end = uint(spans[span_row * 2 + 1]);
        if (thread_index == 0) {
            uint low = rows, high = 0;
            for (uint r = tile * ATTEND_ROWS; r < metal::min(tile * ATTEND_ROWS + ATTEND_ROWS, rows); ++r) {
                low = metal::min(low, uint(spans[r * 2]));
                high = metal::max(high, uint(spans[r * 2 + 1]));
            }
            bounds[0] = low;
            bounds[1] = high;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        walk_first = bounds[0];
        walk_end = bounds[1];
    }

    simdgroup_matrix<S, 8, 8> q[DB];
    for (uint d = 0; d < DB; ++d)
        simdgroup_load(q[d], qkv + ulong(first_row) * row_stride + ulong(head) * WP + d * 8, row_stride);
    simdgroup_matrix<float, 8, 8> output[DB];
    for (uint d = 0; d < DB; ++d)
        output[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float maximum = -INFINITY;
    float denominator = 0.0f;

    for (uint first = walk_first; first < walk_end; first += ATTEND_KEYS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        attend_stage<S, WP, THREADS>(keys, qkv, row_stride, width + ulong(head) * WP, first, rows, thread_index);
        attend_stage<S, WP, THREADS>(values, qkv, row_stride, 2 * width + ulong(head) * WP, first, rows,
            thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> scores[KB];
        for (uint j = 0; j < KB; ++j)
            scores[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d = 0; d < DB; ++d) {
            for (uint j = 0; j < KB; ++j) {
                simdgroup_matrix<S, 8, 8> k;
                simdgroup_load(k, keys + j * 8 * PITCH + d * 8, PITCH, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(scores[j], q[d], k, scores[j]);
            }
        }
        const bool whole = first + ATTEND_KEYS <= rows;
        float tile_maximum = -INFINITY;
        for (uint j = 0; j < KB; ++j) {
            for (uint e = 0; e < 2; ++e) {
                const uint key = first + j * 8 + fn + e;
                float s = scores[j].thread_elements()[e] * scale2;
                if ((!whole && key >= rows) || key < row_first || key >= row_end)
                    s = -INFINITY;
                scores[j].thread_elements()[e] = s;
                tile_maximum = metal::max(tile_maximum, s);
            }
        }
        tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(1)));
        tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(8)));
        const float next = metal::max(maximum, tile_maximum);
        // A row that has seen no key of its span yet keeps its (empty) state.
        const bool seen = next > -INFINITY;
        const float carry = seen ? metal::fast::exp2(maximum - next) : 1.0f;
        simdgroup_matrix<half, 8, 8> probabilities[KB];
        float tile_sum = 0.0f;
        for (uint j = 0; j < KB; ++j) {
            for (uint e = 0; e < 2; ++e) {
                const float p = seen ? metal::fast::exp2(scores[j].thread_elements()[e] - next) : 0.0f;
                probabilities[j].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, ushort(1));
        tile_sum += simd_shuffle_xor(tile_sum, ushort(8));
        denominator = metal::fma(denominator, carry, tile_sum);
        maximum = next;
        for (uint d = 0; d < DB; ++d) {
            output[d].thread_elements()[0] *= carry;
            output[d].thread_elements()[1] *= carry;
        }
        for (uint d = 0; d < DB; ++d) {
            for (uint j = 0; j < KB; ++j) {
                simdgroup_matrix<S, 8, 8> v;
                simdgroup_load(v, values + j * 8 * PITCH + d * 8, PITCH);
                simdgroup_multiply_accumulate(output[d], probabilities[j], v, output[d]);
            }
        }
    }

    const uint row = first_row + fm;
    if (row >= rows)
        return;
    const float inverse = 1.0f / denominator;
    for (uint d = 0; d < DB; ++d)
        for (uint e = 0; e < 2; ++e)
            if (d * 8 + fn + e < W)
                out[ulong(row) * heads * W + ulong(head) * W + d * 8 + fn + e] =
                    S(output[d].thread_elements()[e] * inverse);
}

#if SEISMIC_HAS_TENSOR_OPS
// The attend body on Metal 4 tensor operations: simdgroups 0..3 own 16 query
// rows each and multiply (S = Q K^T and the output's P V are `matmul2d`
// products in execution_simdgroup scope), while all eight stage. S goes to
// the simdgroup's `slot` (16 x ATTEND_KEYS floats), where lanes own rows (2
// lanes per row, 16 columns each) for the online softmax as `attend`'s lanes
// do; the half probabilities follow it in the slot as the P V left operand,
// and per-row carries and inverses reach the cooperative output's rows
// through the slot's last 16 floats. `slots` is ATTEND_TENSOR_SLOTS memory.
constant constexpr uint ATTEND_TENSOR_ROWS = 16;
constant constexpr uint ATTEND_TENSOR_SLOT = ATTEND_TENSOR_ROWS * ATTEND_KEYS * 3 / 2 + ATTEND_TENSOR_ROWS;
#define ATTEND_TENSOR_SLOTS(name) \
    threadgroup float name[vision::ATTEND_ROWS / vision::ATTEND_TENSOR_ROWS * vision::ATTEND_TENSOR_SLOT]

// Each lane's row value (published by the row's first of LANES lanes) into
// the output's row tensor `rows`.
template <uint LANES, typename Rows>
inline void attend_load_rows(threadgroup float *published, uint lane, float value, thread Rows &rows) {
    if (lane % LANES == 0)
        published[lane / LANES] = value;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < rows.get_capacity(); ++i)
        if (rows.is_valid_element(i))
            rows[i] = published[rows.get_multidimensional_index(i)[0]];
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename S, uint W, uint WP>
inline void attend_tensor(device const S *qkv, device S *out, uint rows, uint heads, float scale,
    device const int *spans, threadgroup S *keys, threadgroup S *values, threadgroup uint *bounds,
    threadgroup float *slots, uint tile, uint head, uint thread_index, uint simd, uint lane) {
    constexpr uint THREADS = ATTEND_ROWS * 4;
    constexpr uint ROWS = ATTEND_TENSOR_ROWS;
    constexpr uint KEYS = ATTEND_KEYS;
    constexpr int32_t PITCH = attend_pitch<WP>();
    constexpr uint LANES = 32 / ROWS;
    constexpr uint COLUMNS = KEYS / LANES;
    typedef metal::extents<int32_t, WP, ROWS> q_extents;
    typedef metal::extents<int32_t, WP, KEYS> kv_extents;
    typedef metal::extents<int32_t, KEYS, ROWS> s_extents;
    typedef metal::tensor<device S, q_extents, metal::tensor_inline> q_tensor;
    typedef metal::tensor<threadgroup S, kv_extents, metal::tensor_inline> kv_tensor;
    typedef metal::tensor<threadgroup float, s_extents, metal::tensor_inline> s_tensor;
    typedef metal::tensor<threadgroup half, s_extents, metal::tensor_inline> p_tensor;
    typedef mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(ROWS, KEYS, WP, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        metal::execution_simdgroup> score_op;
    typedef mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(ROWS, WP, KEYS, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        metal::execution_simdgroup> output_op;
    const ulong width = ulong(heads) * WP;
    const ulong row_stride = 3 * width;
    const float scale2 = scale * 1.4426950408889634f;
    const bool computes = simd < ATTEND_ROWS / ROWS;
    const uint owner = computes ? simd : 0;
    const uint first_row = tile * ATTEND_ROWS + owner * ROWS;
    threadgroup float *slot = slots + owner * ATTEND_TENSOR_SLOT;
    threadgroup half *probabilities = reinterpret_cast<threadgroup half *>(slot + ROWS * KEYS);
    threadgroup float *published = slot + ROWS * KEYS * 3 / 2;

    // This lane's row's keys, and the keys the tile walks.
    const uint row = lane / LANES;
    const uint column0 = (lane % LANES) * COLUMNS;
    uint row_first = 0, row_end = rows, walk_first = 0, walk_end = rows;
    if (spans) {
        const uint span_row = metal::min(first_row + row, rows - 1);
        row_first = uint(spans[span_row * 2]);
        row_end = uint(spans[span_row * 2 + 1]);
        if (thread_index == 0) {
            uint low = rows, high = 0;
            for (uint r = tile * ATTEND_ROWS; r < metal::min(tile * ATTEND_ROWS + ATTEND_ROWS, rows); ++r) {
                low = metal::min(low, uint(spans[r * 2]));
                high = metal::max(high, uint(spans[r * 2 + 1]));
            }
            bounds[0] = low;
            bounds[1] = high;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        walk_first = bounds[0];
        walk_end = bounds[1];
    }

    q_tensor q(const_cast<device S *>(qkv) + ulong(first_row) * row_stride + ulong(head) * WP, q_extents(),
        metal::array<int32_t, 2>{1, int32_t(row_stride)});
    kv_tensor k(keys, kv_extents(), metal::array<int32_t, 2>{1, PITCH});
    kv_tensor v(values, kv_extents(), metal::array<int32_t, 2>{1, PITCH});
    s_tensor st(slot, s_extents(), metal::array<int32_t, 2>{1, int32_t(KEYS)});
    p_tensor pt(probabilities, s_extents(), metal::array<int32_t, 2>{1, int32_t(KEYS)});
    score_op score;
    output_op product;
    auto scores = score.template get_destination_cooperative_tensor<q_tensor, kv_tensor, float>();
    auto output = product.template get_destination_cooperative_tensor<p_tensor, kv_tensor, float>();
    auto output_rows = product.template get_row_reduction_destination_cooperative_tensor<p_tensor, kv_tensor, float>();
    for (uint16_t i = 0; i < output.get_capacity(); ++i)
        if (output.is_valid_element(i))
            output[i] = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;

    for (uint first = walk_first; first < walk_end; first += KEYS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        attend_stage<S, WP, THREADS>(keys, qkv, row_stride, width + ulong(head) * WP, first, rows, thread_index);
        attend_stage<S, WP, THREADS>(values, qkv, row_stride, 2 * width + ulong(head) * WP, first, rows,
            thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!computes)
            continue;
        score.run(q, k, scores);
        scores.store(st);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const bool whole = first + KEYS <= rows;
        float x[COLUMNS];
        float tile_maximum = -INFINITY;
        for (uint j = 0; j < COLUMNS; ++j) {
            const uint key = first + column0 + j;
            float value = slot[row * KEYS + column0 + j] * scale2;
            if ((!whole && key >= rows) || key < row_first || key >= row_end)
                value = -INFINITY;
            x[j] = value;
            tile_maximum = metal::max(tile_maximum, value);
        }
        for (ushort offset = 1; offset < LANES; offset <<= 1)
            tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, offset));
        const float next = metal::max(maximum, tile_maximum);
        // A row that has seen no key of its span yet keeps its (empty) state.
        const bool seen = next > -INFINITY;
        const float carry = seen ? metal::fast::exp2(maximum - next) : 1.0f;
        float tile_sum = 0.0f;
        for (uint j = 0; j < COLUMNS; ++j) {
            const float p = seen ? metal::fast::exp2(x[j] - next) : 0.0f;
            probabilities[row * KEYS + column0 + j] = half(p);
            tile_sum += p;
        }
        for (ushort offset = 1; offset < LANES; offset <<= 1)
            tile_sum += simd_shuffle_xor(tile_sum, offset);
        denominator = metal::fma(denominator, carry, tile_sum);
        maximum = next;
        if (!simd_all(carry == 1.0f)) {
            attend_load_rows<LANES>(published, lane, carry, output_rows);
            for (uint16_t i = 0; i < output.get_capacity(); ++i)
                if (output.is_valid_element(i))
                    output[i] *= *output_rows.map_iterator(output.get_iterator(i));
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        product.run(pt, v, output);
    }

    if (!computes)
        return;
    attend_load_rows<LANES>(published, lane, 1.0f / denominator, output_rows);
    for (uint16_t i = 0; i < output.get_capacity(); ++i) {
        if (!output.is_valid_element(i))
            continue;
        const auto index = output.get_multidimensional_index(i);
        const uint out_row = first_row + index[1];
        const uint column = index[0];
        if (out_row < rows && column < W)
            out[ulong(out_row) * heads * W + ulong(head) * W + column] =
                S(output[i] * *output_rows.map_iterator(output.get_iterator(i)));
    }
}
#endif

} // namespace vision
