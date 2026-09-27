// Shared pieces of the vision tower entries on CUDA (`vision_*` in
// vision.seismic): a dense-weight GEMM on the tensor cores with the vision
// epilogues, the row norm, the 2D rotary embedding and the full
// (non-causal) attention.
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

__device__ __forceinline__ float gelu_quick(float value) {
    return value / (1.0f + expf(-1.702f * value));
}

// The activation of `vision_linear`: 1 tanh GELU, 2 erf GELU, 3 quick GELU.
__device__ __forceinline__ float activate(int code, float value) {
    return code == 1 ? gelu_tanh(value) : code == 2 ? gelu_erf(value) : gelu_quick(value);
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
// of 8 (whole 16-byte chunks). Block tile GEMM_M x GEMM_N x GEMM_K in two cp.async stages; eight
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
                                           u32 step, u32 k) {
    constexpr u32 CHUNKS = GEMM_K * 2 / 16; // 16-byte chunks per row
    for (u32 item = threadIdx.x; item < GEMM_M * CHUNKS; item += GEMM_THREADS) {
        const u32 row = item / CHUNKS, chunk = item % CHUNKS;
        const u32 column = step * GEMM_K + chunk * 8;
        const bool inside = first + row < limit && column < k;
        const u8 *source = base + ((u64)(first + row < limit ? first + row : first) * stride + column) * 2;
        seismic_cp_async_16_zfill(tile + row * GEMM_PITCH + chunk * 8, source, inside ? 16u : 0u);
    }
}

// Stages k-step `step` of F32 weight rows as X (rounded to the MMA element,
// a relative 2^-9 for bf16, 2^-12 for f16): plain loads, converted, stored.
template <class X>
__device__ __forceinline__ void gemm_stage_f32(u16 *tile, const u8 *base, u64 stride, u32 first, u32 limit,
                                               u32 step, u32 k) {
    constexpr u32 CHUNKS = GEMM_K / 8; // eight weights per chunk
    for (u32 item = threadIdx.x; item < GEMM_M * CHUNKS; item += GEMM_THREADS) {
        const u32 row = item / CHUNKS, chunk = item % CHUNKS;
        const u32 column = step * GEMM_K + chunk * 8;
        float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (first + row < limit && column < k) {
            const float *source = reinterpret_cast<const float *>(base) + (u64)(first + row) * stride + column;
            if (column + 8 <= k && k % 4 == 0) {
                const float4 a = reinterpret_cast<const float4 *>(source)[0];
                const float4 b = reinterpret_cast<const float4 *>(source)[1];
                v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
            } else {
                for (u32 i = 0; i < 8 && column + i < k; ++i)
                    v[i] = source[i];
            }
        }
        *reinterpret_cast<uint4 *>(tile + row * GEMM_PITCH + chunk * 8) =
            make_uint4(X::pack2(v[0], v[1]), X::pack2(v[2], v[3]), X::pack2(v[4], v[5]), X::pack2(v[6], v[7]));
    }
}

// The element a weight tile is staged in: 16-bit weights as stored, F32 ones
// as the MMA element X.
template <class X, class W> struct Staged {
    typedef W type;
};
template <class X> struct Staged<X, element::F32> {
    typedef X type;
};

template <class X, class W>
__device__ __forceinline__ void gemm_stage_weights(u16 *tile, const u8 *w, u32 first, u32 limit, u32 step, u32 k) {
    if constexpr (W::bytes == 4)
        gemm_stage_f32<X>(tile, w, k, first, limit, step, k);
    else
        gemm_stage(tile, w, k, first, limit, step, k);
}

// Epilogues receive (m, n, value(n), value(n + 1)) for n even, n + 1 < N.
// Weights are f16, bf16 (staged as stored) or F32 (staged as X).
template <class X, class W, class Epilogue>
__device__ __forceinline__ void gemm(GemmShared &shared, const u8 *x, u64 x_stride, const u8 *w, u32 m_rows,
                                     u32 n_rows, u32 k, const Epilogue &epilogue) {
    static_assert(X::bytes == 2, "the vision GEMM stages 16-bit activations");
    typedef typename Staged<X, W>::type WS;
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
    // A partial last step stages zeros past K.
    const u32 steps = (k + GEMM_K - 1) / GEMM_K;
    gemm_stage(shared.x[0], x, x_stride, m0, m_rows, 0, k);
    gemm_stage_weights<X, W>(shared.w[0], w, n0, n_rows, 0, k);
    seismic_cp_async_commit();
    for (u32 step = 0; step < steps; ++step) {
        const u32 stage = step % 2;
        if (step + 1 < steps) {
            gemm_stage(shared.x[1 - stage], x, x_stride, m0, m_rows, step + 1, k);
            gemm_stage_weights<X, W>(shared.w[1 - stage], w, n0, n_rows, step + 1, k);
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
                const u32 b0[2] = {Mma<X>::template convert<WS>(b[0]), Mma<X>::template convert<WS>(b[1])};
                const u32 b1[2] = {Mma<X>::template convert<WS>(b[2]), Mma<X>::template convert<WS>(b[3])};
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
// The `vision_linear` epilogue: the projection is acc, plus bias[n] when
// BIAS, clamped to [minimum, maximum] when CLAMP. With `activation` nonzero
// y = Y(act(round_A(projection))); with GATE y = Y(round_A(projection) ·
// gate[m, n]); otherwise y = Y(projection), plus the F32 residual row when
// RESIDUAL.
template <class A, class B, class Y, bool BIAS, bool RESIDUAL, bool GATE, bool CLAMP> struct Linear {
    u8 *y;
    u64 stride;
    const u8 *bias;
    const float *residual;
    u64 residual_stride;
    const u8 *gate;
    u64 gate_stride;
    float minimum;
    float maximum;
    int activation;
    __device__ __forceinline__ float value(u32 m, u32 n, float acc) const {
        float projected = acc;
        if constexpr (BIAS)
            projected = acc + element::at<B>(bias, n);
        if constexpr (CLAMP)
            projected = fminf(fmaxf(projected, minimum), maximum);
        if (activation != 0)
            return activate(activation, A::round(projected));
        if constexpr (GATE)
            return A::round(projected) * element::at<A>(gate, (u64)m * gate_stride + n);
        if constexpr (RESIDUAL)
            return residual[(u64)m * residual_stride + n] + projected;
        return projected;
    }
    __device__ __forceinline__ void operator()(u32 m, u32 n, float first, float second) const {
        element::put2<Y>(y, (u64)m * stride + n, value(m, n, first), value(m, n + 1, second));
    }
};

// ---------------------------------------------------------------------------
// Norm of one F32 row of `width` values (the whole block): two-pass F32
// statistics, centered (a layer norm) or not (the mean taken as 0, a
// root-mean-square norm), out[i * stride] = Y(centered * inverse [* weight]
// [+ bias]). `partials` holds one float per warp.

template <class Y, class WN, class BN, bool WEIGHT, bool BIAS>
__device__ __forceinline__ void norm(const float *row, u8 *out, u64 stride, const u8 *weight, const u8 *bias,
                                     u32 width, bool centered, float epsilon, float *partials) {
    float mean = 0.0f;
    if (centered) {
        float sum = 0.0f;
        for (u32 i = threadIdx.x; i < width; i += blockDim.x)
            sum += row[i];
        mean = reduce::group_sum(sum, partials) / float(width);
    }
    float squares = 0.0f;
    for (u32 i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = row[i] - mean;
        squares = fmaf(value, value, squares);
    }
    const float inverse = rsqrtf(reduce::group_sum(squares, partials) / float(width) + epsilon);
    // One expression per form, as the layer norm has always been written.
    for (u32 i = threadIdx.x; i < width; i += blockDim.x) {
        float value;
        if constexpr (WEIGHT && BIAS)
            value = (row[i] - mean) * inverse * element::at<WN>(weight, i) + element::at<BN>(bias, i);
        else if constexpr (WEIGHT)
            value = (row[i] - mean) * inverse * element::at<WN>(weight, i);
        else if constexpr (BIAS)
            value = (row[i] - mean) * inverse + element::at<BN>(bias, i);
        else
            value = (row[i] - mean) * inverse;
        element::put<Y>(out, (u64)i * stride, value);
    }
}

// ---------------------------------------------------------------------------
// One attention operand head row of width W = 4P, prepared by one warp (lane
// l owns columns l, l + 32, ...): RMS-normalized with `norm` (`head_norm`:
// x · rsqrt(Σx² / W + epsilon) · norm[i]) when `normed` (`norm` null: unit
// weights), then, when `rotated`, the 2D rotary embedding: column i < 2P
// pairs with i + 2P; pair p = i % 2P turns by coordinates[p / P] ·
// base^(-(p % P) / P), `log_base` = ln(base). Written rounded to A at
// `target`, zero from column W to WP.
template <class A, u32 W, u32 WP>
__device__ __forceinline__ void prepare_head(const u8 *source, u8 *target, bool normed, const float *norm,
                                             float epsilon, bool rotated, const int *coordinates, float log_base,
                                             u32 lane) {
    constexpr u32 P = W / 4;
    float inverse = 1.0f;
    if (normed) {
        float squares = 0.0f;
        for (u32 i = lane; i < W; i += 32) {
            const float x = element::at<A>(source, i);
            squares += x * x;
        }
#pragma unroll
        for (u32 offset = 16; offset > 0; offset /= 2)
            squares += seismic_shfl_xor_f32(squares, offset);
        inverse = rsqrtf(squares / float(W) + epsilon);
    }
    const auto value = [&](u32 i) {
        const float x = element::at<A>(source, i);
        return normed ? (norm ? x * inverse * norm[i] : x * inverse) : x;
    };
    for (u32 i = lane; i < W; i += 32) {
        const float x = value(i);
        float published = x;
        if (rotated) {
            const u32 pair = i % (2 * P);
            const float frequency = expf(-log_base * float(pair % P) / float(P));
            const float angle = float(coordinates[pair / P]) * frequency;
            float s, c;
            sincosf(angle, &s, &c);
            const float partner = value(i < 2 * P ? i + 2 * P : i - 2 * P);
            published = i < 2 * P ? x * c - partner * s : x * c + partner * s;
        }
        element::put<A>(target, i, published);
    }
    for (u32 i = W + lane; i < WP; i += 32)
        element::put<A>(target, i, 0.0f);
}

// ---------------------------------------------------------------------------
// Non-causal attention over the [rows, 3, H, WP] operand rows (queries and
// keys prepared, zero past the head width W). A block owns ATTEND_ROWS query
// rows of one head, 16 per warp, and walks the keys of its rows' spans in
// tiles of KEYS staged by cp.async in two stages: scores = Q K^T on the
// tensor cores (Q's fragments in registers), scaled into the exp2 domain in
// F32, the online softmax, and the F32 output accumulated from probabilities
// rounded to A. Without `spans` every row attends to every row; with them row
// r attends to rows [spans[2r], spans[2r + 1]), the block walking the union
// of its rows' spans. Wide heads stage 32-key tiles to fit 48 KiB.

constexpr u32 ATTEND_ROWS = 64;
constexpr u32 ATTEND_THREADS = ATTEND_ROWS * 2; // four warps

template <u32 WP> struct AttendShared {
    static constexpr u32 PITCH = WP + 8;
    static constexpr u32 KEYS = WP <= 64 ? 64 : 32;
    u16 q[ATTEND_ROWS * PITCH];
    u16 k[2][KEYS * PITCH];
    u16 v[2][KEYS * PITCH];
    u32 bounds[2];
};

// Rows [first, first + count) of the operand part at element column
// `column` into `tile`; rows at or past `rows` are zero.
template <u32 WP>
__device__ __forceinline__ void attend_stage(u16 *tile, const u8 *qkv, u64 row_stride, u64 column, u32 first,
                                             u32 count, u32 rows) {
    constexpr u32 CHUNKS = WP * 2 / 16;
    for (u32 item = threadIdx.x; item < count * CHUNKS; item += ATTEND_THREADS) {
        const u32 row = item / CHUNKS, chunk = item % CHUNKS;
        const bool inside = first + row < rows;
        const u8 *source = qkv + ((u64)(inside ? first + row : 0) * row_stride + column + chunk * 8) * 2;
        seismic_cp_async_16_zfill(tile + row * AttendShared<WP>::PITCH + chunk * 8, source, inside ? 16u : 0u);
    }
}

template <class A, u32 W, u32 WP>
__device__ __forceinline__ void attend(AttendShared<WP> &shared, const u8 *qkv, u8 *out, u32 rows, u32 heads,
                                       float scale, const int *spans) {
    constexpr u32 PITCH = AttendShared<WP>::PITCH;
    constexpr u32 ATTEND_KEYS = AttendShared<WP>::KEYS;
    constexpr u32 DK = WP / 16;          // k16 steps over the head width
    constexpr u32 KN = ATTEND_KEYS / 8;  // n8 score tiles per key tile
    constexpr u32 DN = WP / 8;           // n8 output tiles
    const u64 width = (u64)heads * WP;
    const u64 row_stride = 3 * width;
    const u32 head = blockIdx.y;
    const u32 first_row = blockIdx.x * ATTEND_ROWS;
    const u32 lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const u32 g = lane / 4, t = lane % 4;
    const float scale2 = scale * 1.4426950408889634f;

    // The keys of this lane's two rows (g and g + 8 of its warp's 16), and
    // the keys the block walks.
    u32 row_first[2] = {0, 0}, row_end[2] = {rows, rows};
    u32 walk_first = 0, walk_end = rows;
    if (spans) {
#pragma unroll
        for (u32 h = 0; h < 2; ++h) {
            const u32 row = min(first_row + warp * 16 + g + 8 * h, rows - 1);
            row_first[h] = (u32)spans[row * 2];
            row_end[h] = (u32)spans[row * 2 + 1];
        }
        if (threadIdx.x == 0) {
            u32 low = rows, high = 0;
            for (u32 r = first_row; r < min(first_row + ATTEND_ROWS, rows); ++r) {
                low = min(low, (u32)spans[r * 2]);
                high = max(high, (u32)spans[r * 2 + 1]);
            }
            shared.bounds[0] = low;
            shared.bounds[1] = high;
        }
        __syncthreads();
        walk_first = shared.bounds[0];
        walk_end = shared.bounds[1];
    }

    attend_stage<WP>(shared.q, qkv, row_stride, (u64)head * WP, first_row, ATTEND_ROWS, rows);
    attend_stage<WP>(shared.k[0], qkv, row_stride, width + (u64)head * WP, walk_first, ATTEND_KEYS, rows);
    attend_stage<WP>(shared.v[0], qkv, row_stride, 2 * width + (u64)head * WP, walk_first, ATTEND_KEYS, rows);
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

    const u32 tiles = (walk_end - walk_first + ATTEND_KEYS - 1) / ATTEND_KEYS;
    for (u32 tile = 0; tile < tiles; ++tile) {
        const u32 stage = tile % 2;
        if (tile + 1 < tiles) {
            const u32 next = walk_first + (tile + 1) * ATTEND_KEYS;
            attend_stage<WP>(shared.k[1 - stage], qkv, row_stride, width + (u64)head * WP, next, ATTEND_KEYS, rows);
            attend_stage<WP>(shared.v[1 - stage], qkv, row_stride, 2 * width + (u64)head * WP, next, ATTEND_KEYS,
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
        const u32 key0 = walk_first + tile * ATTEND_KEYS;
        float tile_maximum[2] = {negative_infinity, negative_infinity};
#pragma unroll
        for (u32 j = 0; j < KN; ++j)
#pragma unroll
            for (u32 e = 0; e < 4; ++e) {
                const u32 key = key0 + j * 8 + 2 * t + (e % 2);
                float s = scores[j][e] * scale2;
                if (key >= rows || key < row_first[e / 2] || key >= row_end[e / 2])
                    s = negative_infinity;
                scores[j][e] = s;
                tile_maximum[e / 2] = fmaxf(tile_maximum[e / 2], s);
            }
        // A row that has seen no key of its span yet keeps its (empty) state.
        float carry[2];
        bool seen[2];
#pragma unroll
        for (u32 h = 0; h < 2; ++h) {
            tile_maximum[h] = fmaxf(tile_maximum[h], seismic_shfl_xor_f32(tile_maximum[h], 1));
            tile_maximum[h] = fmaxf(tile_maximum[h], seismic_shfl_xor_f32(tile_maximum[h], 2));
            const float next = fmaxf(maximum[h], tile_maximum[h]);
            seen[h] = next > negative_infinity;
            carry[h] = seen[h] ? exp2f(maximum[h] - next) : 1.0f;
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
                v[e] = seen[e / 2] ? exp2f(scores[j][e] - maximum[e / 2]) : 0.0f;
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
            if (d * 8 + 2 * t < W)
                element::put2<A>(out, ((u64)row * heads + head) * W + d * 8 + 2 * t, output[d][2 * h] * inverse,
                                 output[d][2 * h + 1] * inverse);
    }
}

} // namespace vision
