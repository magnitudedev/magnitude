// Shared pieces of the Qwen3-VL vision entries on CUDA (`qwen_vision_stem`,
// `qwen_vision_block`, `qwen_vision_merger`): a dense-weight GEMM on the
// tensor cores with the vision epilogues, the layer norm, the 2D rotary
// embedding and the full (non-causal) attention.
//
// The projection family (`projection.cuh`) runs on packed mma16 weights; the
// projector's matrices are dense f16 or bf16 rows, so the vision GEMM stages
// them as stored. Its MMA element is the activation's; a weight of the other
// 16-bit type is converted per fragment.
//
// Numerics (`vision.seismic`): the residual stream is F32; the published
// intermediates are rounded to the activation element A. Every dense operand
// is bound canonically (row-major, unit innermost stride).

#include "../core/activation.cuh"
#include "../core/reduce.cuh"

namespace vision {

using element::u16;
using element::u32;
using element::u64;
using element::u8;

template <class A, class B> struct Same {
    static constexpr bool value = false;
};
template <class A> struct Same<A, A> {
    static constexpr bool value = true;
};

// ---------------------------------------------------------------------------
// Scalar functions.

__device__ __forceinline__ float gelu_tanh(float value) {
    const float argument = 0.7978845608028654f * (value + 0.044715f * value * value * value);
    const float hyperbolic = 2.0f / (1.0f + expf(-2.0f * argument)) - 1.0f;
    return 0.5f * value * (1.0f + hyperbolic);
}

__device__ __forceinline__ float gelu_erf(float value) {
    return 0.5f * value * (1.0f + erff(value * 0.7071067811865476f));
}

// ---------------------------------------------------------------------------
// MMA element of a 16-bit dense kind: the tensor-core operation and the
// conversion of a packed pair of the other 16-bit kind.

template <class W> struct Mma;
template <> struct Mma<element::F16> {
    __device__ static __forceinline__ void run(float (&acc)[4], const u32 (&a)[4], const u32 (&b)[2]) {
        seismic_mma_m16n8k16_f16(acc, a, b);
    }
    template <class From> __device__ static __forceinline__ u32 convert(u32 pair) {
        if constexpr (Same<From, element::F16>::value) {
            return pair;
        } else {
            const float2 value = From::unpack2(pair);
            return seismic_pack_f16x2(value.x, value.y);
        }
    }
};
template <> struct Mma<element::Bf16> {
    __device__ static __forceinline__ void run(float (&acc)[4], const u32 (&a)[4], const u32 (&b)[2]) {
        seismic_mma_m16n8k16_bf16(acc, a, b);
    }
    template <class From> __device__ static __forceinline__ u32 convert(u32 pair) {
        if constexpr (Same<From, element::Bf16>::value) {
            return pair;
        } else {
            const float2 value = From::unpack2(pair);
            return seismic_pack_bf16x2(value.x, value.y);
        }
    }
};

// ---------------------------------------------------------------------------
// GEMM: y[m, n] = epilogue(sum_k x[m, k] * w[n, k]) over x [M, K] (element X,
// row stride `x_stride`) and w [N, K] (element W, rows of K). K is a multiple
// of GEMM_K. Block tile GEMM_M x GEMM_N x GEMM_K in two cp.async stages; eight
// warps as 2 x 4, each owning 64 x 32 outputs (4 x 4 m16n8 tiles).

constexpr u32 GEMM_M = 128;
constexpr u32 GEMM_N = 128;
constexpr u32 GEMM_K = 32;
constexpr u32 GEMM_THREADS = 256;
constexpr u32 GEMM_PITCH = GEMM_K + 8; // elements per staged row

struct GemmShared {
    u16 x[2][GEMM_M * GEMM_PITCH];
    u16 w[2][GEMM_N * GEMM_PITCH];
};

// Stages k-step `step` of `rows` rows from `first` (rows at or past `limit`
// are zero) into `tile`.
__device__ __forceinline__ void gemm_stage(u16 *tile, const u8 *base, u64 stride, u32 first, u32 limit,
                                           u32 step) {
    constexpr u32 CHUNKS = GEMM_K * 2 / 16; // 16-byte chunks per row
    for (u32 item = threadIdx.x; item < GEMM_M * CHUNKS; item += GEMM_THREADS) {
        const u32 row = item / CHUNKS, chunk = item % CHUNKS;
        const bool inside = first + row < limit;
        const u8 *source = base + ((u64)(inside ? first + row : first) * stride + (u64)step * GEMM_K + chunk * 8) * 2;
        seismic_cp_async_16_zfill(tile + row * GEMM_PITCH + chunk * 8, source, inside ? 16u : 0u);
    }
}

// Epilogues receive (m, n, value(n), value(n + 1)) for n even, n + 1 < N.
template <class X, class W, class Epilogue>
__device__ __forceinline__ void gemm(GemmShared &shared, const u8 *x, u64 x_stride, const u8 *w, u32 m_rows,
                                     u32 n_rows, u32 k, const Epilogue &epilogue) {
    static_assert(W::bytes == 2 && X::bytes == 2, "the vision GEMM stages 16-bit operands");
    const u32 m0 = blockIdx.y * GEMM_M, n0 = blockIdx.x * GEMM_N;
    const u32 lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const u32 wm = (warp / 4) * 64, wn = (warp % 4) * 32;
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[i][j][e] = 0.0f;
    const u32 steps = k / GEMM_K;
    gemm_stage(shared.x[0], x, x_stride, m0, m_rows, 0);
    gemm_stage(shared.w[0], w, k, n0, n_rows, 0);
    seismic_cp_async_commit();
    for (u32 step = 0; step < steps; ++step) {
        const u32 stage = step % 2;
        if (step + 1 < steps) {
            gemm_stage(shared.x[1 - stage], x, x_stride, m0, m_rows, step + 1);
            gemm_stage(shared.w[1 - stage], w, k, n0, n_rows, step + 1);
        }
        seismic_cp_async_commit();
        seismic_cp_async_wait<1>();
        __syncthreads();
        const u16 *xs = shared.x[stage];
        const u16 *ws = shared.w[stage];
#pragma unroll
        for (u32 kk = 0; kk < GEMM_K; kk += 16) {
            u32 a[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                seismic_ldmatrix_x4(a[i], xs + (wm + i * 16 + lane % 16) * GEMM_PITCH + kk + (lane / 16) * 8);
#pragma unroll
            for (int jj = 0; jj < 4; jj += 2) {
                u32 b[4];
                seismic_ldmatrix_x4(b, ws + (wn + jj * 8 + lane % 8 + (lane / 16) * 8) * GEMM_PITCH + kk
                                           + ((lane / 8) % 2) * 8);
                const u32 b0[2] = {Mma<X>::template convert<W>(b[0]), Mma<X>::template convert<W>(b[1])};
                const u32 b1[2] = {Mma<X>::template convert<W>(b[2]), Mma<X>::template convert<W>(b[3])};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    Mma<X>::run(acc[i][jj], a[i], b0);
                    Mma<X>::run(acc[i][jj + 1], a[i], b1);
                }
            }
        }
        __syncthreads();
    }
    const u32 g = lane / 4, t = lane % 4;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const u32 n = n0 + wn + j * 8 + 2 * t;
            if (n >= n_rows)
                continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const u32 m = m0 + wm + i * 16 + g + 8 * h;
                if (m < m_rows)
                    epilogue(m, n, acc[i][j][2 * h], acc[i][j][2 * h + 1]);
            }
        }
}

// ---------------------------------------------------------------------------
// GEMM epilogues: the projection is acc + bias[n].

// round_A(projection).
template <class A, class B> struct Bias {
    u8 *y;
    u64 stride;
    const u8 *bias;
    __device__ __forceinline__ void operator()(u32 m, u32 n, float first, float second) const {
        element::put2<A>(y, (u64)m * stride + n, first + element::at<B>(bias, n),
                         second + element::at<B>(bias, n + 1));
    }
};

// round_A(gelu(round_A(projection))), tanh or erf form.
template <class A, class B, bool ERF> struct BiasGelu {
    u8 *y;
    u64 stride;
    const u8 *bias;
    __device__ __forceinline__ float value(u32 n, float acc) const {
        const float projected = A::round(acc + element::at<B>(bias, n));
        return ERF ? gelu_erf(projected) : gelu_tanh(projected);
    }
    __device__ __forceinline__ void operator()(u32 m, u32 n, float first, float second) const {
        element::put2<A>(y, (u64)m * stride + n, value(n, first), value(n + 1, second));
    }
};

// The F32 projection, plus an F32 residual row when `residual` is bound.
template <class B> struct BiasF32 {
    float *y;
    u64 stride;
    const u8 *bias;
    const float *residual;
    u64 residual_stride;
    __device__ __forceinline__ float value(u32 m, u32 n, float acc) const {
        const float projected = acc + element::at<B>(bias, n);
        return residual ? residual[(u64)m * residual_stride + n] + projected : projected;
    }
    __device__ __forceinline__ void operator()(u32 m, u32 n, float first, float second) const {
        *reinterpret_cast<float2 *>(y + (u64)m * stride + n) = make_float2(value(m, n, first), value(m, n + 1, second));
    }
};

// ---------------------------------------------------------------------------
// Layer norm of one F32 row of `width` values (the whole block): two-pass
// centered F32 statistics, out = round_A(centered * inverse * weight + bias).
// `partials` holds one float per warp.

template <class A, class WN, class BN>
__device__ __forceinline__ void layer_norm(const float *row, u8 *out, const u8 *weight, const u8 *bias, u32 width,
                                           float epsilon, float *partials) {
    float sum = 0.0f;
    for (u32 i = threadIdx.x; i < width; i += blockDim.x)
        sum += row[i];
    const float mean = reduce::group_sum(sum, partials) / float(width);
    float squares = 0.0f;
    for (u32 i = threadIdx.x; i < width; i += blockDim.x) {
        const float centered = row[i] - mean;
        squares = fmaf(centered, centered, squares);
    }
    const float inverse = rsqrtf(reduce::group_sum(squares, partials) / float(width) + epsilon);
    for (u32 i = threadIdx.x; i < width; i += blockDim.x)
        element::put<A>(out, i, (row[i] - mean) * inverse * element::at<WN>(weight, i) + element::at<BN>(bias, i));
}

// ---------------------------------------------------------------------------
// 2D rotary embedding, in place, of one head row of width W = 4P held by one
// warp (lane l owns columns [l * E, l * E + E), E = W / 32): column i < 2P
// pairs with i + 2P (16 lanes away); pair p = i % 2P turns by
// coordinates[p / P] * 10000^(-(p % P) / P).
template <class A, u32 W> __device__ __forceinline__ void rotate(u8 *head_row, const int *coordinates, u32 lane) {
    static_assert(W % 64 == 0, "a rotated head row is two 16-lane halves");
    constexpr u32 E = W / 32;
    constexpr u32 P = W / 4;
    float x[E], partner[E];
#pragma unroll
    for (u32 i = 0; i < E; ++i)
        x[i] = element::at<A>(head_row, lane * E + i);
#pragma unroll
    for (u32 i = 0; i < E; ++i)
        partner[i] = seismic_shfl_xor_f32(x[i], 16);
#pragma unroll
    for (u32 i = 0; i < E; ++i) {
        const u32 column = lane * E + i;
        const u32 pair = column % (2 * P);
        const float frequency = expf(-9.210340371976184f * float(pair % P) / float(P));
        const float angle = float(coordinates[pair / P]) * frequency;
        float s, c;
        sincosf(angle, &s, &c);
        element::put<A>(head_row, lane * E + i, column < 2 * P ? x[i] * c - partner[i] * s : x[i] * c + partner[i] * s);
    }
}

// ---------------------------------------------------------------------------
// Full attention over the [rows, 3, H, W] projection (queries and keys
// rotated). A block owns ATTEND_ROWS query rows of one head, 16 per warp,
// and walks every key in tiles of ATTEND_KEYS staged by cp.async in two
// stages: scores = Q K^T on the tensor cores (Q's fragments in registers),
// scaled into the exp2 domain in F32, the online softmax, and the F32 output
// accumulated from probabilities rounded to A.

constexpr u32 ATTEND_ROWS = 64;
constexpr u32 ATTEND_KEYS = 64;
constexpr u32 ATTEND_THREADS = ATTEND_ROWS * 2; // four warps

template <u32 W> struct AttendShared {
    static constexpr u32 PITCH = W + 8;
    u16 q[ATTEND_ROWS * PITCH];
    u16 k[2][ATTEND_KEYS * PITCH];
    u16 v[2][ATTEND_KEYS * PITCH];
};

// Rows [first, first + count) of the projection part at element column
// `column` into `tile`; rows at or past `rows` are zero.
template <u32 W>
__device__ __forceinline__ void attend_stage(u16 *tile, const u8 *qkv, u64 row_stride, u64 column, u32 first,
                                             u32 count, u32 rows) {
    constexpr u32 CHUNKS = W * 2 / 16;
    for (u32 item = threadIdx.x; item < count * CHUNKS; item += ATTEND_THREADS) {
        const u32 row = item / CHUNKS, chunk = item % CHUNKS;
        const bool inside = first + row < rows;
        const u8 *source = qkv + ((u64)(inside ? first + row : 0) * row_stride + column + chunk * 8) * 2;
        seismic_cp_async_16_zfill(tile + row * AttendShared<W>::PITCH + chunk * 8, source, inside ? 16u : 0u);
    }
}

template <class A, u32 W>
__device__ __forceinline__ void attend(AttendShared<W> &shared, const u8 *qkv, u8 *out, u32 rows, u32 heads,
                                       float scale) {
    constexpr u32 PITCH = AttendShared<W>::PITCH;
    constexpr u32 DK = W / 16;           // k16 steps over the head width
    constexpr u32 KN = ATTEND_KEYS / 8;  // n8 score tiles per key tile
    constexpr u32 DN = W / 8;            // n8 output tiles
    const u64 width = (u64)heads * W;
    const u64 row_stride = 3 * width;
    const u32 head = blockIdx.y;
    const u32 first_row = blockIdx.x * ATTEND_ROWS;
    const u32 lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const u32 g = lane / 4, t = lane % 4;
    const float scale2 = scale * 1.4426950408889634f;

    attend_stage<W>(shared.q, qkv, row_stride, (u64)head * W, first_row, ATTEND_ROWS, rows);
    attend_stage<W>(shared.k[0], qkv, row_stride, width + (u64)head * W, 0, ATTEND_KEYS, rows);
    attend_stage<W>(shared.v[0], qkv, row_stride, 2 * width + (u64)head * W, 0, ATTEND_KEYS, rows);
    seismic_cp_async_commit();

    u32 q[DK][4];
    float output[DN][4];
#pragma unroll
    for (u32 d = 0; d < DN; ++d)
#pragma unroll
        for (u32 e = 0; e < 4; ++e)
            output[d][e] = 0.0f;
    const float negative_infinity = -__int_as_float(0x7f800000);
    float maximum[2] = {negative_infinity, negative_infinity};
    float denominator[2] = {0.0f, 0.0f};

    const u32 tiles = (rows + ATTEND_KEYS - 1) / ATTEND_KEYS;
    for (u32 tile = 0; tile < tiles; ++tile) {
        const u32 stage = tile % 2;
        if (tile + 1 < tiles) {
            const u32 next = (tile + 1) * ATTEND_KEYS;
            attend_stage<W>(shared.k[1 - stage], qkv, row_stride, width + (u64)head * W, next, ATTEND_KEYS, rows);
            attend_stage<W>(shared.v[1 - stage], qkv, row_stride, 2 * width + (u64)head * W, next, ATTEND_KEYS,
                            rows);
        }
        seismic_cp_async_commit();
        seismic_cp_async_wait<1>();
        __syncthreads();
        if (tile == 0) {
#pragma unroll
            for (u32 d = 0; d < DK; ++d)
                seismic_ldmatrix_x4(q[d], shared.q + (warp * 16 + lane % 16) * PITCH + d * 16 + (lane / 16) * 8);
        }
        const u16 *ks = shared.k[stage];
        const u16 *vs = shared.v[stage];
        float scores[KN][4];
#pragma unroll
        for (u32 j = 0; j < KN; ++j)
#pragma unroll
            for (u32 e = 0; e < 4; ++e)
                scores[j][e] = 0.0f;
#pragma unroll
        for (u32 d = 0; d < DK; ++d) {
#pragma unroll
            for (u32 j = 0; j < KN; j += 2) {
                u32 b[4];
                seismic_ldmatrix_x4(b, ks + (j * 8 + lane % 8 + (lane / 16) * 8) * PITCH + d * 16 + ((lane / 8) % 2) * 8);
                const u32 b0[2] = {b[0], b[1]};
                const u32 b1[2] = {b[2], b[3]};
                Mma<A>::run(scores[j], q[d], b0);
                Mma<A>::run(scores[j + 1], q[d], b1);
            }
        }
        const u32 key0 = tile * ATTEND_KEYS;
        float tile_maximum[2] = {negative_infinity, negative_infinity};
#pragma unroll
        for (u32 j = 0; j < KN; ++j)
#pragma unroll
            for (u32 e = 0; e < 4; ++e) {
                float s = scores[j][e] * scale2;
                if (key0 + j * 8 + 2 * t + (e % 2) >= rows)
                    s = negative_infinity;
                scores[j][e] = s;
                tile_maximum[e / 2] = fmaxf(tile_maximum[e / 2], s);
            }
        float carry[2];
#pragma unroll
        for (u32 h = 0; h < 2; ++h) {
            tile_maximum[h] = fmaxf(tile_maximum[h], seismic_shfl_xor_f32(tile_maximum[h], 1));
            tile_maximum[h] = fmaxf(tile_maximum[h], seismic_shfl_xor_f32(tile_maximum[h], 2));
            const float next = fmaxf(maximum[h], tile_maximum[h]);
            carry[h] = exp2f(maximum[h] - next);
            maximum[h] = next;
        }
        // Probabilities as A fragments: score tiles 2c and 2c + 1 are the
        // k16 step c of the P V product.
        u32 p[KN / 2][4];
        float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
        for (u32 j = 0; j < KN; ++j) {
            float v[4];
#pragma unroll
            for (u32 e = 0; e < 4; ++e) {
                v[e] = exp2f(scores[j][e] - maximum[e / 2]);
                tile_sum[e / 2] += v[e];
            }
            p[j / 2][(j % 2) * 2 + 0] = A::pack2(v[0], v[1]);
            p[j / 2][(j % 2) * 2 + 1] = A::pack2(v[2], v[3]);
        }
#pragma unroll
        for (u32 h = 0; h < 2; ++h) {
            tile_sum[h] += seismic_shfl_xor_f32(tile_sum[h], 1);
            tile_sum[h] += seismic_shfl_xor_f32(tile_sum[h], 2);
            denominator[h] = fmaf(denominator[h], carry[h], tile_sum[h]);
        }
#pragma unroll
        for (u32 d = 0; d < DN; ++d)
#pragma unroll
            for (u32 e = 0; e < 4; ++e)
                output[d][e] *= carry[e / 2];
#pragma unroll
        for (u32 c = 0; c < KN / 2; ++c) {
            // p[c] = (row g, keys 16c + 2t..), (row g + 8, ..), (row g, keys 16c + 8 + 2t..), (row g + 8, ..)
            const u32 a[4] = {p[c][0], p[c][1], p[c][2], p[c][3]};
#pragma unroll
            for (u32 d = 0; d < DN; d += 2) {
                u32 b[4];
                seismic_ldmatrix_x4_trans(b, vs + (c * 16 + lane % 8 + ((lane / 8) % 2) * 8) * PITCH + d * 8
                                                 + (lane / 16) * 8);
                const u32 b0[2] = {b[0], b[1]};
                const u32 b1[2] = {b[2], b[3]};
                Mma<A>::run(output[d], a, b0);
                Mma<A>::run(output[d + 1], a, b1);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (u32 h = 0; h < 2; ++h) {
        const u32 row = first_row + warp * 16 + g + 8 * h;
        if (row >= rows)
            continue;
        const float inverse = 1.0f / denominator[h];
#pragma unroll
        for (u32 d = 0; d < DN; ++d)
            element::put2<A>(out, (u64)row * width + (u64)head * W + d * 8 + 2 * t, output[d][2 * h] * inverse,
                             output[d][2 * h + 1] * inverse);
    }
}

} // namespace vision
