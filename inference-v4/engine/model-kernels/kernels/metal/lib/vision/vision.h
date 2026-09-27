// Shared pieces of the Qwen3-VL vision entries (`qwen_vision_stem`,
// `qwen_vision_block`, `qwen_vision_merger`). The projections run on the
// projection library's GEMM (`projection::gemm` over dense weight rows) with
// the vision epilogues below; the layer norm, the 2D rotary embedding and the
// full (non-causal) attention are the vision-specific launches.
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

// ---------------------------------------------------------------------------
// GEMM epilogues. `store(m, n, acc)` receives the F32 product of output
// (m, n); the projection is acc + bias[n].

// round_A(projection).
template <typename A, typename B>
struct output_bias {
    device uchar *y;
    ulong stride;
    device const uchar *bias;
    void store(uint m, uint n, float value) const {
        element::put<A>(y, ulong(m) * stride + n, value + element::at<B>(bias, n));
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// round_A(gelu(round_A(projection))), tanh or erf form.
template <typename A, typename B, bool ERF>
struct output_bias_gelu {
    device uchar *y;
    ulong stride;
    device const uchar *bias;
    void store(uint m, uint n, float value) const {
        const float projected = A::round(value + element::at<B>(bias, n));
        element::put<A>(y, ulong(m) * stride + n, ERF ? gelu_erf(projected) : gelu_tanh(projected));
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// The F32 projection, plus an F32 residual row when `residual` is bound.
template <typename B>
struct output_bias_f32 {
    device float *y;
    ulong stride;
    device const uchar *bias;
    device const float *residual;
    ulong residual_stride;
    void store(uint m, uint n, float value) const {
        const float projected = value + element::at<B>(bias, n);
        y[ulong(m) * stride + n] = residual ? residual[ulong(m) * residual_stride + n] + projected : projected;
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// ---------------------------------------------------------------------------
// Layer norm of one F32 row of `width` values (THREADS threads): two-pass
// centered F32 statistics, out = round_A(centered * inverse * weight + bias).
// `partials` holds THREADS / 32 floats.
template <uint THREADS, typename A, typename WN, typename BN>
inline void layer_norm(device const float *row, device uchar *out, device const uchar *weight,
    device const uchar *bias, uint width, float epsilon, threadgroup float *partials, uint thread_index,
    uint sg, uint lane) {
    float sum = 0.0f;
    for (uint i = thread_index; i < width; i += THREADS)
        sum += row[i];
    const float mean = reduce::group_sum<THREADS / 32>(sum, partials, sg, lane) / float(width);
    float squares = 0.0f;
    for (uint i = thread_index; i < width; i += THREADS) {
        const float centered = row[i] - mean;
        squares = metal::fma(centered, centered, squares);
    }
    const float inverse =
        metal::rsqrt(reduce::group_sum<THREADS / 32>(squares, partials, sg, lane) / float(width) + epsilon);
    for (uint i = thread_index; i < width; i += THREADS)
        element::put<A>(out, i, (row[i] - mean) * inverse * element::at<WN>(weight, i) + element::at<BN>(bias, i));
}

// ---------------------------------------------------------------------------
// 2D rotary embedding, in place, of one head row of width W = 4P held by one
// simdgroup (lane l owns columns [l * E, l * E + E), E = W / 32): column
// i < 2P pairs with i + 2P (16 lanes away); pair p = i % 2P turns by
// coordinates[p / P] * 10000^(-(p % P) / P).
template <typename A, uint W>
inline void rotate(device typename A::storage *head_row, device const int *coordinates, uint lane) {
    static_assert(W % 64 == 0, "a rotated head row is two 16-lane halves");
    constexpr uint E = W / 32;
    constexpr uint P = W / 4;
    float x[E];
    for (uint i = 0; i < E; ++i)
        x[i] = A::load(head_row[lane * E + i]);
    float partner[E];
    for (uint i = 0; i < E; ++i)
        partner[i] = simd_shuffle_xor(x[i], ushort(16));
    for (uint i = 0; i < E; ++i) {
        const uint column = lane * E + i;
        const uint pair = column % (2 * P);
        const float frequency = metal::precise::exp(-9.210340371976184f * float(pair % P) / float(P));
        const float angle = float(coordinates[pair / P]) * frequency;
        const float c = metal::precise::cos(angle), s = metal::precise::sin(angle);
        const float rotated = column < 2 * P ? x[i] * c - partner[i] * s : x[i] * c + partner[i] * s;
        head_row[lane * E + i] = A::store(rotated);
    }
}

// ---------------------------------------------------------------------------
// Full attention. A threadgroup owns ATTEND_ROWS query rows of one head
// (simdgroup s: rows 8s .. 8s + 7 of the tile) and walks every key in tiles
// of ATTEND_KEYS: K and V staged in threadgroup memory, scores = Q K^T on the
// matrix units (Q's fragments held in registers), scaled into the exp2
// domain in F32, the online softmax, and the F32 output accumulated from
// probabilities rounded to half. `qkv` is the [rows, 3, H, W] projection with
// its queries and keys rotated; rows past `rows` of the last query tile are
// read (the buffer holds whole tiles) but never stored.
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

template <typename S, uint W>
inline void attend(device const S *qkv, device S *out, uint rows, uint heads, float scale,
    threadgroup S *keys, threadgroup S *values, uint tile, uint head, uint thread_index, uint simd,
    uint lane) {
    constexpr uint THREADS = ATTEND_ROWS * 4;
    constexpr uint PITCH = attend_pitch<W>();
    constexpr uint DB = W / 8;
    constexpr uint KB = ATTEND_KEYS / 8;
    const ulong width = ulong(heads) * W;
    const ulong row_stride = 3 * width;
    const float scale2 = scale * 1.4426950408889634f;

    // This lane's fragment coordinates: row fm, columns fn and fn + 1.
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + ((lane / 2) % 4);
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const uint first_row = tile * ATTEND_ROWS + simd * 8;

    simdgroup_matrix<S, 8, 8> q[DB];
    for (uint d = 0; d < DB; ++d)
        simdgroup_load(q[d], qkv + ulong(first_row) * row_stride + ulong(head) * W + d * 8, row_stride);
    simdgroup_matrix<float, 8, 8> output[DB];
    for (uint d = 0; d < DB; ++d)
        output[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float maximum = -INFINITY;
    float denominator = 0.0f;

    for (uint first = 0; first < rows; first += ATTEND_KEYS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        attend_stage<S, W, THREADS>(keys, qkv, row_stride, width + ulong(head) * W, first, rows, thread_index);
        attend_stage<S, W, THREADS>(values, qkv, row_stride, 2 * width + ulong(head) * W, first, rows,
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
                float s = scores[j].thread_elements()[e] * scale2;
                if (!whole && first + j * 8 + fn + e >= rows)
                    s = -INFINITY;
                scores[j].thread_elements()[e] = s;
                tile_maximum = metal::max(tile_maximum, s);
            }
        }
        tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(1)));
        tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(8)));
        const float next = metal::max(maximum, tile_maximum);
        const float carry = metal::fast::exp2(maximum - next);
        simdgroup_matrix<half, 8, 8> probabilities[KB];
        float tile_sum = 0.0f;
        for (uint j = 0; j < KB; ++j) {
            for (uint e = 0; e < 2; ++e) {
                const float p = metal::fast::exp2(scores[j].thread_elements()[e] - next);
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
            out[ulong(row) * width + ulong(head) * W + d * 8 + fn + e] =
                S(output[d].thread_elements()[e] * inverse);
}

} // namespace vision
