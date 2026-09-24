// K1 packed projection family for CUDA: y[m, n] = sum_k x[m, k] * W[n, k]
// over mma16 weights, with a prologue that forms x and an epilogue that
// publishes y. Entry sources are thin instantiations.
//
// Numerics. Activations are rounded to the activation element A (bf16 or
// f16). GEMV: codes enter `mma.sync.m16n8k16` as exact 16-bit integers; each
// representation group (32 codes for q4k/q5k/q8, 16 for q6k) accumulates in
// F32 from zero and is folded as acc += scale * P - bias * sum(x), the
// group-factored form. GEMM: weights are dequantized to A per value,
// round_A(code * round_A(scale) - round_A(bias)) (a rounding of 2^-9 relative
// in bf16, far below the formats' own quantization step), and accumulate in
// F32 without a per-group fold. The GEMM's INT8 candidate: see
// `QuantizedRows`. Reductions across K run in a fixed order. A row's GEMV
// result does not depend on M (every activation row is its own MMA column),
// so rows 1..GEMV_ROWS of one request match M = 1 bit for bit.
//
// Rows. A prologue that is not a plain read of A forms its rows with
// `form_row` (the staging launch into global scratch, or at M = 1 the GEMV
// block itself into its shared memory), so every path reads the same A rows.
//
// GEMV (M <= GEMV_ROWS = 16): swap-AB. The 16 weight rows of a tile are the A
// operand, the M activation rows the N = 8 columns of NB column blocks (NB 1
// for M <= 8, 2 up to 16). A warp owns TPW consecutive tiles of one weight
// segment over a 1/KSPLIT share of K; KSPLIT warps reduce through shared
// memory in part order. Codes arrive in fragment order, one 16 B
// non-coherent, L1-bypassing load per lane per 64-code k-block (high-bit
// planes as whole-superblock loads redistributed by shuffles); the codes and
// packed coefficients of the next superblock are in flight in registers while
// the current one decodes (an explicit L2 prefetch, and pipelining across
// consecutive tiles, measured no gain on top of that). Group sums of x come
// from the same B operands through an all-ones A fragment.
//
// GEMM (M > GEMV_ROWS): a staging launch writes the prologue's rows when the
// prologue is not a plain read of A; the main launch runs BM x 128 x 64 block
// tiles over a STAGES-deep cp.async pipeline (activation rows, code chunks and
// coefficient words), ldmatrix activation fragments, and weight B fragments
// dequantized from the staged mma16 chunks (an A-fragment register pair of a
// 16-row weight tile is the B fragment of one of its 8-row halves).
#include "common/packets.cuh"
#include "common/reduce.cuh"

namespace projection {

using element::u8;
using element::u16;
using element::u32;
using element::u64;
using packets::Coefficients;
using packets::S8Pair;
using packets::Same;
typedef element::Act Act;

static_assert(Act::bytes == 2, "the CUDA projection family requires a bf16 or f16 activation element");
using Op = typename packets::OperandOf<Act>::type;

// The largest row count the GEMV serves; larger counts run the GEMM.
constexpr u32 GEMV_ROWS = 16;

__device__ __forceinline__ float silu(float value) { return value / (1.0f + expf(-value)); }

__device__ __forceinline__ void named_barrier(u32 id, u32 threads) {
    asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// The segment of a segmented projection holding work item `index`, given the
// items (GEMV tile groups or GEMM block columns) of each segment in order;
// `index` becomes segment-local. Returns N past the last segment.
template <int N> __device__ __forceinline__ int locate_segment(u64 &index, const u64 (&items)[N]) {
#pragma unroll
    for (int segment = 0; segment < N; ++segment) {
        if (index < items[segment])
            return segment;
        index -= items[segment];
    }
    return N;
}

// ---------------------------------------------------------------------------
// Row maps: which input row feeds projected row m.

struct AllRows {
    __device__ __forceinline__ u64 operator()(u32 m) const { return m; }
};
struct SelectedRows {
    const int *rows;
    __device__ __forceinline__ u64 operator()(u32 m) const { return (u64)rows[m]; }
};

// ---------------------------------------------------------------------------
// Prologues. Each forms x[m, k] rounded to A from FACTORS per-row factors:
//   prepare_row(shared, m, tmp) whole block: factors of row m at shared[0..]
//   value(factors, m, k)        x[m, k]
// A GEMV reads its operands through `pair(factors, m, k)` (the operand bits
// of (x[m, k], x[m, k + 1])), which only the prologues read in place provide.
// `STAGED` is false when the input can be read in place.

template <class E, class Rows> struct Plain {
    const u8 *x;
    u64 stride;
    Rows rows;
    static constexpr bool STAGED = !Same<E, Act>::value || !Same<Rows, AllRows>::value;
    static constexpr int FACTORS = 0;
    __device__ __forceinline__ void prepare_row(float *, u32, float *) const {}
    __device__ __forceinline__ float value(const float *, u32 m, u64 k) const {
        return Act::round(element::at<E>(x, rows(m) * stride + k));
    }
    __device__ __forceinline__ u32 pair(const float *, u32 m, u64 k) const {
        static_assert(!STAGED, "a GEMV reads staged rows");
        return *reinterpret_cast<const u32 *>(x + (rows(m) * stride + k) * 2);
    }
};

// RMS normalization of an F32 residual: x = round_A(t * rsqrt(sum t^2 / K + eps) * w).
template <class NORM, class Rows> struct Rms {
    const float *x;
    u64 stride;
    const u8 *norm;
    float eps;
    u64 width;
    Rows rows;
    static constexpr bool STAGED = true;
    static constexpr int FACTORS = 1;
    __device__ __forceinline__ float squares(u32 m, u32 first, u32 step) const {
        const float *row = x + rows(m) * stride;
        float total = 0.0f;
        for (u64 k = first; k < width; k += step)
            total = seismic_fma_rn(row[k], row[k], total);
        return total;
    }
    __device__ __forceinline__ float inverse(float total) const { return rsqrtf(total / (float)width + eps); }
    __device__ __forceinline__ void prepare_row(float *shared, u32 m, float *scratch) const {
        const float total = reduce::group_sum(squares(m, threadIdx.x, blockDim.x), scratch);
        if (threadIdx.x == 0)
            shared[0] = inverse(total);
    }
    __device__ __forceinline__ float formed(float inv, u32 m, u64 k) const {
        return x[rows(m) * stride + k] * inv * element::at<NORM>(norm, k);
    }
    __device__ __forceinline__ float value(const float *factors, u32 m, u64 k) const {
        return Act::round(formed(factors[0], m, k));
    }
};

// Per-head RMS of `mixed` gated by SiLU(z), heads of HEAD columns:
// x = round_A(round_A(v * rsqrt(sum_head v^2 / HEAD + eps) * w[k % HEAD]) * round_A(silu(z))).
template <class MIXED, class Z, class NORM, u32 HEAD, u32 HEADS, class Rows> struct GatedRms {
    const u8 *mixed;
    u64 mixed_stride;
    const u8 *z;
    u64 z_stride;
    const u8 *norm;
    float eps;
    Rows rows;
    static constexpr bool STAGED = true;
    static constexpr int FACTORS = HEADS;
    __device__ __forceinline__ float inverse(u32 m, u32 head) const {
        const u64 base = rows(m) * mixed_stride + (u64)head * HEAD;
        float total = 0.0f;
        for (u32 k = threadIdx.x % 32; k < HEAD; k += 32) {
            const float v = element::at<MIXED>(mixed, base + k);
            total = seismic_fma_rn(v, v, total);
        }
        total = seismic_warp_sum_f32(total);
        return rsqrtf(total / (float)HEAD + eps);
    }
    __device__ __forceinline__ void prepare_row(float *shared, u32 m, float *) const {
        const u32 warps = blockDim.x / 32;
        for (u32 head = threadIdx.x / 32; head < HEADS; head += warps) {
            const float value = inverse(m, head);
            if (threadIdx.x % 32 == 0)
                shared[head] = value;
        }
    }
    // The gated value with the head's inverse RMS `inv`.
    __device__ __forceinline__ float gated(float inv, u32 m, u64 k) const {
        const float v = element::at<MIXED>(mixed, rows(m) * mixed_stride + k);
        const float normalized = Act::round(v * inv * element::at<NORM>(norm, k % HEAD));
        const float activated = Act::round(silu(element::at<Z>(z, rows(m) * z_stride + k)));
        return normalized * activated;
    }
    __device__ __forceinline__ float value(const float *factors, u32 m, u64 k) const {
        return Act::round(gated(factors[k / HEAD], m, k));
    }
};

// The prologue's row m as A into out[m * K .. m * K + K), formed by the whole
// block: the staging launch (one block per row, global scratch) and a GEMV
// block at M = 1 (its shared memory) both form rows with it.
template <class Pro> __device__ __forceinline__ void form_row(const Pro &pro, u32 m, u64 K, u8 *out) {
    static_assert(Pro::STAGED, "an input read in place is not formed");
    __shared__ float factors[Pro::FACTORS > 0 ? Pro::FACTORS : 1];
    __shared__ float scratch[32];
    pro.prepare_row(factors, m, scratch);
    __syncthreads();
    for (u64 k = threadIdx.x; k < K; k += blockDim.x)
        element::put<Act>(out, m * K + k, pro.value(factors, m, k));
}

// int8 activations (the GEMM's INT8 candidate): rows quantized per 32-code
// group as q8_1 does, x ~= d * q with d = max|x| / 127 and q = rint(x / d),
// staged by `stage_row_s8` as `q` ([M, width] bytes in the s8 fragments'
// virtual k order, packets.cuh) and `groups` (each group's (d, d * sum q)).
struct QuantizedRows {
    const u8 *q;
    u64 width;
    const float2 *groups;
};

// Real offset within a 32-code group of virtual s8 position v.
__device__ __forceinline__ u32 s8_real_offset(u32 v) {
    const u32 half = v / 16, t = (v % 16) / 4, j = v % 4;
    return half * 16 + 2 * t + (j % 2) + 8 * (j / 2);
}

// Staging launch of the INT8 candidate, one block per row m: the prologue's
// row x[m, :] (rounded to A) quantized per 32 codes into `q` ([M, K] bytes,
// virtual order) and `groups` ([M, K / 32] (d, d * sum q)).
template <class Pro>
__device__ __forceinline__ void stage_row_s8(const Pro &pro, u32 m, u64 K, u8 *q, float2 *groups) {
    __shared__ float factors[Pro::FACTORS > 0 ? Pro::FACTORS : 1];
    __shared__ float scratch[32];
    pro.prepare_row(factors, m, scratch);
    __syncthreads();
    for (u64 group = threadIdx.x; group < K / 32; group += blockDim.x) {
        float values[32];
        float maximum = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            values[j] = pro.value(factors, m, group * 32 + j);
            maximum = fmaxf(maximum, fabsf(values[j]));
        }
        const float d = maximum / 127.0f;
        int total = 0;
        u32 words[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            u32 word = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int code = d == 0.0f ? 0 : __float2int_rn(values[s8_real_offset(4 * w + b)] / d);
                total += code;
                word |= ((u32)code & 0xFFu) << (8 * b);
            }
            words[w] = word;
        }
        uint4 *out = reinterpret_cast<uint4 *>(q + m * K + group * 32);
        out[0] = make_uint4(words[0], words[1], words[2], words[3]);
        out[1] = make_uint4(words[4], words[5], words[6], words[7]);
        groups[m * (K / 32) + group] = make_float2(d, d * (float)total);
    }
}

// ---------------------------------------------------------------------------
// Epilogues: called once per (m, n) with the F32 projection (and, paired, the
// second stream's projection of the same n).

// y[m, offset + n] rounded to the stored element E.
template <class E> struct Store {
    u8 *out;
    u64 stride;
    u64 offset;
    __device__ __forceinline__ void operator()(u32 m, u64 n, float value, float) const {
        element::put<E>(out, m * stride + offset + n, value);
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float first, float second, float, float) const {
        element::put2<E>(out, m * stride + offset + n, first, second);
    }
};

// out[m, n] = residual[rows(m), n] + round_A(projection).
template <class Rows> struct Residual {
    const float *residual;
    u64 residual_stride;
    Rows rows;
    float *out;
    u64 stride;
    __device__ __forceinline__ void operator()(u32 m, u64 n, float value, float) const {
        out[m * stride + n] = residual[rows(m) * residual_stride + n] + Act::round(value);
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float first, float second, float, float) const {
        const u64 source = rows(m) * residual_stride + n;
        const u64 target = m * stride + n;
        if (source % 2 != 0 || target % 2 != 0) {
            (*this)(m, n, first, 0.0f);
            (*this)(m, n + 1, second, 0.0f);
            return;
        }
        const float2 base = reinterpret_cast<const float2 *>(residual)[source / 2];
        reinterpret_cast<float2 *>(out)[target / 2] = make_float2(base.x + Act::round(first), base.y + Act::round(second));
    }
};

// out[m, n] = round(round_A(silu(round_A(gate))) * round_A(up)).
template <class E> struct SiluMul {
    u8 *out;
    u64 stride;
    __device__ static __forceinline__ float value(float gate, float up) {
        return Act::round(silu(Act::round(gate))) * Act::round(up);
    }
    __device__ __forceinline__ void operator()(u32 m, u64 n, float gate, float up) const {
        element::put<E>(out, m * stride + n, value(gate, up));
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float gate0, float gate1, float up0, float up1) const {
        element::put2<E>(out, m * stride + n, value(gate0, up0), value(gate1, up1));
    }
};

// The absent second stream of an unpaired projection.
struct NoWeight {
    static constexpr int GROUP = 16;
    static constexpr int GROUPS = 4;
    static constexpr bool BIAS = false;
    static constexpr bool DENSE = false;
    static constexpr int CHUNKS = 0;
    struct Raw {};
    struct Block {};
    struct Super {};
    __device__ __forceinline__ Super fetch_superblock(u64, u64, u64, u32) const { return Super{}; }
    __device__ __forceinline__ Raw raw(const Super &, int, u32) const { return Raw{}; }
    __device__ __forceinline__ Block block(u64, u64, u32) const { return Block{}; }
};

// ---------------------------------------------------------------------------
// GEMV (M <= GEMV_ROWS).

// Operand sums of x over each 32-code half of a k-block, for the lane's
// C-fragment columns 2t and 2t+1 of each 8-row column block.
template <int NB> struct GroupSums {
    float half[NB][2][2];
};

template <int NB, class W>
__device__ __forceinline__ void gemv_accumulate(float (&acc)[NB][4], const typename W::Raw &raw, const W &w,
                                                const typename W::Block &block, int q, const u32 (&b)[NB][4][2],
                                                const GroupSums<NB> &sums) {
    Coefficients<W::GROUPS> c;
    w.coefficients(block, q, c);
    constexpr int STEPS = W::GROUP / 16;
#pragma unroll
    for (int group = 0; group < W::GROUPS; ++group) {
        float p[NB][4];
#pragma unroll
        for (int nb = 0; nb < NB; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                p[nb][e] = 0.0f;
#pragma unroll
        for (int j = 0; j < STEPS; ++j) {
            u32 a[4];
            w.template decode<Op>(raw, group * STEPS + j, a);
#pragma unroll
            for (int nb = 0; nb < NB; ++nb)
                Op::mma(p[nb], a, b[nb][group * STEPS + j]);
        }
        // p[nb]: (row g, col 2t), (row g, col 2t+1), (row g+8, col 2t), (row g+8, col 2t+1).
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[nb][e] = seismic_fma_rn(c.scale[e / 2][group], p[nb][e], acc[nb][e]);
            if constexpr (W::BIAS) {
                static_assert(W::GROUP == 32, "biased groups span half a k-block");
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[nb][e] = seismic_fma_rn(-c.bias[e / 2][group], sums.half[nb][group][e % 2], acc[nb][e]);
            }
        }
    }
}

// One warp of a tile group: tiles [tile0, tile0 + TPW) of a segment with
// `tiles` tiles and `rows` valid rows, superblock share `part` of KSPLIT, for
// M <= 8 * NB activation rows (column block nb holds rows 8 nb .. 8 nb + 7).
// Each iteration decodes one superblock (4 k-blocks) while the codes and
// packed coefficients of the next are in flight in registers. `pro` reads
// its A operands in place (`pair`); `reduce` is the group's reduction area,
// `barrier` its named barrier.
template <int TPW, int KSPLIT, int NB, class Pro, class WA, class WB, class Epi>
__device__ __forceinline__ void gemv_group(const Pro &pro, u32 M, u64 kblocks, u64 tile0, u64 tiles, u64 rows,
                                           const WA &wa, const WB &wb, const Epi &epi, float *reduce, u32 barrier,
                                           u32 part) {
    static_assert(Pro::FACTORS == 0, "a GEMV reads its operands in place");
    constexpr bool PAIR = !Same<WB, NoWeight>::value;
    constexpr bool BIAS = WA::BIAS || WB::BIAS;
    const u32 lane = threadIdx.x % 32;
    const u32 g = lane / 4;
    const u32 t = lane % 4;
    const float *factors = nullptr;
    float acc[2][TPW][NB][4];
#pragma unroll
    for (int stream = 0; stream < 2; ++stream)
#pragma unroll
        for (int i = 0; i < TPW; ++i)
#pragma unroll
            for (int nb = 0; nb < NB; ++nb)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[stream][i][nb][e] = 0.0f;
    const u64 superblocks = (kblocks + 3) / 4;
    const u64 begin = superblocks * part / KSPLIT;
    const u64 end = superblocks * (part + 1) / KSPLIT;
    // Codes and packed coefficients of one superblock of the group's tiles.
    struct Stage {
        typename WA::Super ra[TPW];
        typename WB::Super rb[TPW];
        typename WA::Block ba[TPW];
        typename WB::Block bb[TPW];
    };
    auto load = [&](Stage &stage, u64 sb) {
#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            if (tile0 + i >= tiles)
                continue;
            stage.ra[i] = wa.fetch_superblock(tile0 + i, sb, kblocks, lane);
            if constexpr (PAIR)
                stage.rb[i] = wb.fetch_superblock(tile0 + i, sb, kblocks, lane);
            stage.ba[i] = wa.block(tile0 + i, sb, lane);
            if constexpr (PAIR)
                stage.bb[i] = wb.block(tile0 + i, sb, lane);
        }
    };
    auto compute = [&](const Stage &stage, u64 sb) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (4 * sb + q >= kblocks)
                break;
            const u64 k0 = (4 * sb + q) * 64;
            u32 b[NB][4][2];
#pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                const u32 m = 8 * nb + g;
                const bool feeds = m < M;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    b[nb][s][0] = feeds ? pro.pair(factors, m, k0 + 16 * s + 2 * t) : 0u;
                    b[nb][s][1] = feeds ? pro.pair(factors, m, k0 + 16 * s + 2 * t + 8) : 0u;
                }
            }
            GroupSums<NB> sums;
            if constexpr (BIAS) {
                const u32 ones[4] = {Op::ONES, Op::ONES, Op::ONES, Op::ONES};
#pragma unroll
                for (int nb = 0; nb < NB; ++nb)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        Op::mma(x, ones, b[nb][2 * h]);
                        Op::mma(x, ones, b[nb][2 * h + 1]);
                        sums.half[nb][h][0] = x[0];
                        sums.half[nb][h][1] = x[1];
                    }
            }
#pragma unroll
            for (int i = 0; i < TPW; ++i)
                if (tile0 + i < tiles) {
                    gemv_accumulate<NB>(acc[0][i], wa.raw(stage.ra[i], q, lane), wa, stage.ba[i], q, b, sums);
                    if constexpr (PAIR)
                        gemv_accumulate<NB>(acc[1][i], wb.raw(stage.rb[i], q, lane), wb, stage.bb[i], q, b, sums);
                }
        }
    };
    // The next superblock's loads are in flight while the current one decodes.
    Stage current;
    if (begin < end)
        load(current, begin);
    for (u64 sb = begin; sb < end; ++sb) {
        Stage next;
        if (sb + 1 < end)
            load(next, sb + 1);
        compute(current, sb);
        current = next;
    }
    if constexpr (KSPLIT > 1) {
        constexpr int VALUES = 2 * TPW * NB * 4;
#pragma unroll
        for (int v = 0; v < VALUES; ++v)
            reduce[(part * VALUES + v) * 32 + lane] = acc[v / (TPW * NB * 4)][(v / (NB * 4)) % TPW][(v / 4) % NB][v % 4];
        named_barrier(barrier, KSPLIT * 32);
        if (part != 0)
            return;
#pragma unroll
        for (int v = 0; v < VALUES; ++v) {
            float total = reduce[v * 32 + lane];
            for (int other = 1; other < KSPLIT; ++other)
                total += reduce[(other * VALUES + v) * 32 + lane];
            acc[v / (TPW * NB * 4)][(v / (NB * 4)) % TPW][(v / 4) % NB][v % 4] = total;
        }
    }
#pragma unroll
    for (int i = 0; i < TPW; ++i) {
        const u64 tile = tile0 + i;
        if (tile >= tiles)
            continue;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const u64 n = tile * 16 + g + 8 * r;
            if (n >= rows)
                continue;
#pragma unroll
            for (int nb = 0; nb < NB; ++nb)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const u32 m = 8 * nb + 2 * t + c;
                    if (m < M)
                        epi(m, n, acc[0][i][nb][2 * r + c], acc[1][i][nb][2 * r + c]);
                }
        }
    }
}

// Launch shape of a GEMV: WARPS warps per block in groups of KSPLIT; a group
// owns TPW tiles for up to 8 * NB activation rows. Threads per block = 32 *
// WARPS; tile groups per block = WARPS / KSPLIT.
template <int WARPS_, int TPW_, int KSPLIT_, int NB_> struct GemvShape {
    static constexpr int WARPS = WARPS_;
    static constexpr int TPW = TPW_;
    static constexpr int KSPLIT = KSPLIT_;
    static constexpr int NB = NB_;
    static constexpr int GROUPS = WARPS / KSPLIT;
    static_assert(WARPS % KSPLIT == 0, "KSPLIT divides WARPS");
    static_assert(GROUPS <= 15, "one named barrier per group");
    static_assert(NB >= 1 && 8 * NB <= (int)GEMV_ROWS, "column blocks of 8 rows up to GEMV_ROWS");
    static constexpr int REDUCE = KSPLIT > 1 ? KSPLIT * 2 * TPW * NB * 4 * 32 : 1;
    __device__ static __forceinline__ u32 group() { return threadIdx.x / 32 / KSPLIT; }
    __device__ static __forceinline__ u32 part() { return threadIdx.x / 32 % KSPLIT; }
    // Global tile-group index of this warp.
    __device__ static __forceinline__ u64 tile_group() { return (u64)blockIdx.x * GROUPS + group(); }
};

// Shared memory of a GEMV block: the KSPLIT reduction of each tile group.
template <class Shape, class Pro> struct GemvShared {
    float reduce[Shape::GROUPS][Shape::REDUCE];
};

// Tile groups of a segment of `rows` rows.
template <class Shape> __device__ __forceinline__ u64 gemv_groups(u64 rows) {
    return ((rows + 15) / 16 + Shape::TPW - 1) / Shape::TPW;
}

// Run a GEMV warp over one segment: `group` is the segment-local tile group.
template <class Shape, class Pro, class WA, class WB, class Epi, class Shared>
__device__ __forceinline__ void gemv_segment(Shared &shared, const Pro &pro, u32 M, u64 kblocks, u64 group,
                                             u64 rows, const WA &wa, const WB &wb, const Epi &epi) {
    gemv_group<Shape::TPW, Shape::KSPLIT, Shape::NB>(pro, M, kblocks, group * Shape::TPW, (rows + 15) / 16, rows,
                                                     wa, wb, epi, shared.reduce[Shape::group()], 1 + Shape::group(),
                                                     Shape::part());
}

// ---------------------------------------------------------------------------
// GEMM (M > GEMV_ROWS).

// Block tile BM activation rows x 128 weight rows (8 tiles) x 64 codes, warps
// WM x WN; each warp owns (BM / WM) x (128 / WN). A pipeline stage holds one
// k-block: the activation rows (A at pitch 144, or s8 at pitch 80 followed by
// each row's two (d, d * sum q) groups, 16 B), and per weight stream the code
// chunks of the 8 tiles (1024 B per tile, the largest representation)
// followed by the 128 rows' coefficient words (16 B per row). Each warp
// decodes its own rows' coefficients into its area (16 B per row per stream).
template <int BM_, int WM_, int WN_, int STAGES_> struct GemmShape {
    static constexpr int BM = BM_;
    static constexpr int WM = WM_;
    static constexpr int WN = WN_;
    static constexpr int STAGES = STAGES_;
    static constexpr int WARPS = WM * WN;
    static constexpr int THREADS = 32 * WARPS;
    static constexpr int MI = BM / WM / 16; // m16 tiles per warp
    static constexpr int NJ = 8 / WN;       // weight tiles per warp
    static_assert(MI * 16 * WM == BM && NJ * WN == 8, "warp tiling covers the block tile");
    template <bool S8> static constexpr int A_PITCH = S8 ? 80 : 144;
    // Activation bytes per row per stage (s8 rows carry their groups).
    template <bool S8> static constexpr int A_ROW = S8 ? 96 : 144;
    static constexpr int W_TILE = 1024;
    static constexpr int W_CODES = 8 * W_TILE;
    static constexpr int W_STREAM = W_CODES + 128 * 16;
    static constexpr int COEF_WARP = NJ * 16 * 16;
    // Shared bytes (the declared `shared_bytes` of a GEMM launch):
    //   STAGES * (BM * 144 + STREAMS * 10240) + STREAMS * WARPS * NJ * 256
    // (the 16-bit layout, which also covers S8's smaller rows: INT8 has no
    // effect with dense weights, so a configuration's GEMM may run either).
    template <int STREAMS, bool S8> static constexpr int STAGE_BYTES = BM * A_ROW<S8> + STREAMS * W_STREAM;
    template <int STREAMS, bool S8>
    static constexpr int SHARED_BYTES = STAGES * STAGE_BYTES<STREAMS, S8> + STREAMS * WARPS * COEF_WARP;
};

// The raw fragments of weight tile `tile`, k-block `kblock` for the lane: from
// the stage's staged chunks, or for dense weights (nothing staged) from
// global memory.
template <class W>
__device__ __forceinline__ typename W::Raw gemm_raw(const W &w, const u8 *staged_tile, u64 tile, u64 kblock, u32 lane) {
    if constexpr (W::DENSE)
        return w.fetch(tile, kblock, lane);
    else
        return w.from_shared(staged_tile, lane);
}

__device__ __forceinline__ void cp_async_4_zfill(void *shared, const void *global, u32 source_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(seismic_shared_address(shared)), "l"(global),
                 "r"(source_bytes)
                 : "memory");
}

// One stream's part of a stage: the code chunks of tiles [tile0, tile0 + 8)
// and the coefficient words of their rows (zero past `tiles`).
template <class W, int THREADS>
__device__ __forceinline__ void gemm_load_weight(const W &w, u64 tile0, u64 tiles, u64 kblock, u8 *stage) {
    for (int c = threadIdx.x; c < 8 * W::CHUNKS; c += THREADS) {
        const u64 j = c / W::CHUNKS;
        const u32 q = c % W::CHUNKS;
        const bool valid = tile0 + j < tiles;
        seismic_cp_async_16_zfill(stage + j * 1024 + q * 16, w.chunk_source(valid ? tile0 + j : tile0, kblock, q),
                                  valid ? 16u : 0u);
    }
    u8 *words = stage + 8 * 1024;
    for (int c = threadIdx.x; c < 128 * W::COEF_WORDS; c += THREADS) {
        const u32 row = c / W::COEF_WORDS;
        const int word = c % W::COEF_WORDS;
        const bool valid = tile0 + row / 16 < tiles;
        cp_async_4_zfill(words + row * 16 + word * 4, w.coef_word(tile0 * 16 + (valid ? row : 0), kblock, word),
                         valid ? 4u : 0u);
    }
}

// The warp's decoded coefficients of k-block `kblock`: its NJ tiles' rows
// (block-tile rows first..first + NJ * 16), area[row * GROUPS + group] as
// F32 (scale, -bias), the bias absent when the representation has none.
template <class W, int NJ>
__device__ __forceinline__ void gemm_decode_coefficients(const W &w, const u8 *words, u32 first, u64 kblock,
                                                         float *area) {
    if constexpr (W::DENSE)
        return;
    constexpr int ITEMS = NJ * 16 * W::GROUPS;
    constexpr int WORDS = W::BIAS ? 2 : 1;
    for (int item = threadIdx.x % 32; item < ITEMS; item += 32) {
        const u32 row = item / W::GROUPS;
        const uint4 staged = *reinterpret_cast<const uint4 *>(words + (first + row) * 16);
        const u32 row_words[4] = {staged.x, staged.y, staged.z, staged.w};
        float scale, bias;
        w.staged_coefficient(row_words, kblock, item % W::GROUPS, scale, bias);
        area[item * WORDS] = scale;
        if constexpr (W::BIAS)
            area[item * WORDS + 1] = -bias;
    }
    __syncwarp();
}

// (scale, -bias) of (row, group) in a warp's decoded area.
template <class W> __device__ __forceinline__ float2 gemm_coefficient(const float *area, u32 row, int group) {
    if constexpr (W::BIAS)
        return reinterpret_cast<const float2 *>(area)[row * W::GROUPS + group];
    else
        return make_float2(area[row * W::GROUPS + group], 0.0f);
}

// The A pair of two codes (an exact A pair, as `decode` forms it) of one row:
// round_A(code * scale - bias) per half, from F32 arithmetic.
__device__ __forceinline__ u32 dequantize_pair(u32 codes, float2 coefficient) {
    const float2 pair = Op::unpack(codes);
    return Op::pack(seismic_fma_rn(pair.x, coefficient.x, coefficient.y),
                    seismic_fma_rn(pair.y, coefficient.x, coefficient.y));
}

// Epilogue of columns n and n + 1 (n even, n + 1 < rows): the epilogue's
// `pair` when it has one, else two element calls.
template <class Epi>
__device__ __forceinline__ auto epilogue_pair(const Epi &epi, u32 m, u64 n, float a0, float a1, float b0, float b1,
                                              int) -> decltype(epi.pair(m, n, a0, a1, b0, b1), void()) {
    epi.pair(m, n, a0, a1, b0, b1);
}
template <class Epi>
__device__ __forceinline__ void epilogue_pair(const Epi &epi, u32 m, u64 n, float a0, float a1, float b0, float b1,
                                              long) {
    epi(m, n, a0, b0);
    epi(m, n + 1, a1, b1);
}

// The C fragment of one warp tile: acc[j][mi][h][e] is row m0 + mi * 16 + g +
// 8 * (e / 2), column (tile0 + j) * 16 + h * 8 + 2 * t + e % 2.
template <int NJ, int MI, class Epi>
__device__ __forceinline__ void gemm_publish(const float (&acc)[2][NJ][MI][2][4], u64 m0, u64 tile0, u32 M, u64 rows,
                                             const Epi &epi) {
    const u32 lane = threadIdx.x % 32;
    const u32 g = lane / 4;
    const u32 t = lane % 4;
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const u64 m = m0 + mi * 16 + g + 8 * r;
                    const u64 n = (tile0 + j) * 16 + h * 8 + 2 * t;
                    if (m >= M || n >= rows)
                        continue;
                    if (n + 1 < rows)
                        epilogue_pair(epi, (u32)m, n, acc[0][j][mi][h][2 * r], acc[0][j][mi][h][2 * r + 1],
                                      acc[1][j][mi][h][2 * r], acc[1][j][mi][h][2 * r + 1], 0);
                    else
                        epi((u32)m, n, acc[0][j][mi][h][2 * r], acc[1][j][mi][h][2 * r]);
                }
}

// acc[mi][h] += x . w over the 32-code half `half` (k16 steps 2 * half,
// 2 * half + 1) of the warp's weight tile j, its codes dequantized to A with
// the warp's decoded coefficients (a holds the half's two steps).
template <class Shape, class W>
__device__ __forceinline__ void gemm_tile_half(float acc[Shape::MI][2][4], const W &w, const typename W::Raw &raw,
                                               const float *coefficients, int j_tile, int half,
                                               const u32 a[2][Shape::MI][4]) {
    const u32 g = (threadIdx.x % 32) / 4;
    constexpr int STEPS = W::GROUP / 16;
#pragma unroll
    for (int local = 0; local < 2; ++local) {
        const int step = 2 * half + local;
        // Registers 0 and 2 hold row g of the tile, 1 and 3 row g + 8.
        u32 r[4];
        w.template decode<Op>(raw, step, r);
        // Dense weights decode to their operand values directly.
        if constexpr (!W::DENSE) {
            const float2 low = gemm_coefficient<W>(coefficients, j_tile * 16 + g, step / STEPS);
            const float2 high = gemm_coefficient<W>(coefficients, j_tile * 16 + g + 8, step / STEPS);
            r[0] = dequantize_pair(r[0], low);
            r[2] = dequantize_pair(r[2], low);
            r[1] = dequantize_pair(r[1], high);
            r[3] = dequantize_pair(r[3], high);
        }
        const u32 b0[2] = {r[0], r[2]};
        const u32 b1[2] = {r[1], r[3]};
#pragma unroll
        for (int mi = 0; mi < Shape::MI; ++mi) {
            Op::mma(acc[mi][0], a[local][mi], b0);
            Op::mma(acc[mi][1], a[local][mi], b1);
        }
    }
}

// The A-typed activation rows of a GEMM at a fixed row stride (elements):
// row m starts at `act + m * stride`. Any row source gives `row(m)` for
// m < M, the rows the GEMM reads.
struct ActivationRows {
    const u8 *act;
    u64 stride;
    __device__ __forceinline__ const u8 *row(u64 m) const { return act + m * stride * 2; }
};

// One GEMM block over segment-local block column `nblock` (tiles
// [8 * nblock, 8 * nblock + 8)), activation rows [BM * blockIdx.y, ...) of
// `source` (the A-typed input [M, K], staged or in place) and K share `part`
// of `parts` (split-K: the epilogue is then a `PartialStore` and
// `split_finalize` applies the real one). Rows at or past M read zeros; a warp
// whose rows all lie there decodes and multiplies nothing.
template <class Shape, class Rows, class WA, class WB, class Epi>
__device__ __forceinline__ void gemm_segment(u8 *shared, const Rows &source, u32 M, u64 K, u64 nblock, u64 rows,
                                             const WA &wa, const WB &wb, const Epi &epi, u64 part = 0,
                                             u64 parts = 1) {
    constexpr bool PAIR = !Same<WB, NoWeight>::value;
    constexpr int STREAMS = PAIR ? 2 : 1;
    constexpr int THREADS = Shape::THREADS;
    constexpr int STAGE = Shape::template STAGE_BYTES<STREAMS, false>;
    constexpr int PITCH = Shape::template A_PITCH<false>;
    constexpr int WEIGHTS = Shape::BM * Shape::template A_ROW<false>;
    constexpr int MI = Shape::MI;
    constexpr int NJ = Shape::NJ;
    const u32 lane = threadIdx.x % 32;
    const u32 warp = threadIdx.x / 32;
    const u32 wm = warp % Shape::WM;
    const u32 wn = warp / Shape::WM;
    const u64 m0 = (u64)blockIdx.y * Shape::BM;
    const u64 tile0 = nblock * 8;
    const u64 tiles = (rows + 15) / 16;
    const u64 kbegin = K / 64 * part / parts;
    const u64 kblocks = K / 64 * (part + 1) / parts - kbegin;
    float *coefficients =
        reinterpret_cast<float *>(shared + Shape::STAGES * STAGE + warp * STREAMS * Shape::COEF_WARP);
    const bool active = m0 + wm * (MI * 16) < M;

    // A thread's activation chunks (16 bytes of one tile row per k-block)
    // keep their rows across k-blocks, so their addresses are formed once;
    // a chunk past M reads zeros from a valid address.
    constexpr int CHUNKS = (Shape::BM * 8 + THREADS - 1) / THREADS;
    const u8 *chunk[CHUNKS];
    bool valid[CHUNKS];
#pragma unroll
    for (int i = 0; i < CHUNKS; ++i) {
        const int c = threadIdx.x + i * THREADS;
        const u64 row = m0 + c / 8;
        valid[i] = c < Shape::BM * 8 && row < M;
        chunk[i] = source.row(valid[i] ? row : 0) + (c % 8) * 16;
    }

    // `index` counts k-blocks of this share.
    auto load = [&](int stage, u64 index) {
        const u64 kb = kbegin + index;
        u8 *base = shared + stage * STAGE;
#pragma unroll
        for (int i = 0; i < CHUNKS; ++i) {
            const int c = threadIdx.x + i * THREADS;
            if (c < Shape::BM * 8)
                seismic_cp_async_16_zfill(base + (c / 8) * PITCH + (c % 8) * 16, chunk[i] + kb * 128,
                                          valid[i] ? 16u : 0u);
        }
        gemm_load_weight<WA, THREADS>(wa, tile0, tiles, kb, base + WEIGHTS);
        if constexpr (PAIR)
            gemm_load_weight<WB, THREADS>(wb, tile0, tiles, kb, base + WEIGHTS + Shape::W_STREAM);
    };

    float acc[2][NJ][MI][2][4];
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        acc[s][j][mi][h][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < Shape::STAGES - 1; ++s) {
        if ((u64)s < kblocks)
            load(s, s);
        seismic_cp_async_commit();
    }
    for (u64 kb = 0; kb < kblocks; ++kb) {
        seismic_cp_async_wait<Shape::STAGES - 2>();
        __syncthreads();
        const u64 next = kb + Shape::STAGES - 1;
        if (next < kblocks)
            load((int)(next % Shape::STAGES), next);
        seismic_cp_async_commit();
        if (!active)
            continue;
        const u8 *base = shared + (kb % Shape::STAGES) * STAGE;
        gemm_decode_coefficients<WA, NJ>(wa, base + WEIGHTS + Shape::W_CODES, wn * NJ * 16, kbegin + kb,
                                         coefficients);
        if constexpr (PAIR)
            gemm_decode_coefficients<WB, NJ>(wb, base + WEIGHTS + Shape::W_STREAM + Shape::W_CODES, wn * NJ * 16,
                                             kbegin + kb, coefficients + Shape::COEF_WARP / 4);
        typename WA::Raw raw_a[NJ];
        typename WB::Raw raw_b[NJ];
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            raw_a[j] = gemm_raw(wa, base + WEIGHTS + (wn * NJ + j) * 1024, tile0 + wn * NJ + j, kbegin + kb, lane);
            if constexpr (PAIR)
                raw_b[j] = gemm_raw(wb, base + WEIGHTS + Shape::W_STREAM + (wn * NJ + j) * 1024, tile0 + wn * NJ + j,
                                    kbegin + kb, lane);
        }
        // Per 32-code half: the activation fragments of its two k16 steps.
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            u32 a[2][MI][4];
#pragma unroll
            for (int local = 0; local < 2; ++local)
#pragma unroll
                for (int mi = 0; mi < MI; ++mi) {
                    const int s = 2 * half + local;
                    const u32 row = wm * (MI * 16) + mi * 16;
                    seismic_ldmatrix_x4(a[local][mi],
                                        base + (row + lane % 16) * PITCH + (s * 16 + (lane / 16) * 8) * 2);
                }
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                gemm_tile_half<Shape>(acc[0][j], wa, raw_a[j], coefficients, j, half, a);
                if constexpr (PAIR)
                    gemm_tile_half<Shape>(acc[1][j], wb, raw_b[j], coefficients + Shape::COEF_WARP / 4, j, half, a);
            }
        }
    }
    seismic_cp_async_wait<0>();
    gemm_publish(acc, m0 + wm * (MI * 16), tile0 + wn * NJ, M, rows, epi);
}

// The INT8 candidate's GEMM: the same block tiling and pipeline over s8
// activations staged by `stage_row_s8` (BM rows x 64 bytes per k-block, pitch
// 80, plus each row's two (d, d * sum q) groups) and s8 weight fragments
// (packets.cuh virtual order). Per 32-code group one m16n8k32 s8 MMA into an
// exact int32 product (two, half-masked, for 16-code groups), folded as
// acc += scale * (d_x * P) - bias * (d_x * sum q_x).
template <class Shape, class W>
__device__ __forceinline__ void gemm_tile_s8(float acc[Shape::MI][2][4], const W &w, const u8 *staged_tile,
                                             const float *coefficients, int j_tile, const u32 a[2][Shape::MI][4],
                                             const float2 x[2][Shape::MI][2]) {
    static_assert(!W::DENSE, "the INT8 GEMM needs packed weights");
    const u32 lane = threadIdx.x % 32;
    const u32 t = lane % 4;
    const typename W::Raw raw = w.from_shared(staged_tile, lane);
    auto fold = [&](const int (&p)[Shape::MI][2][4], int group, int h) {
#pragma unroll
        for (int half = 0; half < 2; ++half)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const float2 k = gemm_coefficient<W>(coefficients, j_tile * 16 + half * 8 + 2 * t + c, group);
#pragma unroll
                for (int mi = 0; mi < Shape::MI; ++mi)
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        float &value = acc[mi][half][2 * r + c];
                        value = seismic_fma_rn(k.x, x[h][mi][r].x * (float)p[mi][half][2 * r + c], value);
                        if constexpr (W::BIAS)
                            value = seismic_fma_rn(k.y, x[h][mi][r].y, value);
                    }
            }
    };
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const S8Pair first = w.s8(raw, 2 * h);
        const S8Pair second = w.s8(raw, 2 * h + 1);
        if constexpr (W::GROUP == 32) {
            const u32 b0[2] = {first.row_g, second.row_g};
            const u32 b1[2] = {first.row_g8, second.row_g8};
            int p[Shape::MI][2][4];
#pragma unroll
            for (int mi = 0; mi < Shape::MI; ++mi) {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    p[mi][0][e] = p[mi][1][e] = 0;
                seismic_mma_m16n8k32_s8(p[mi][0], a[h][mi], b0);
                seismic_mma_m16n8k32_s8(p[mi][1], a[h][mi], b1);
            }
            fold(p, h, h);
        } else {
            static_assert(W::GROUP == 16, "groups of 16 or 32 codes");
#pragma unroll
            for (int part = 0; part < 2; ++part) {
                const S8Pair &step = part == 0 ? first : second;
                const u32 b0[2] = {part == 0 ? step.row_g : 0u, part == 1 ? step.row_g : 0u};
                const u32 b1[2] = {part == 0 ? step.row_g8 : 0u, part == 1 ? step.row_g8 : 0u};
                int p[Shape::MI][2][4];
#pragma unroll
                for (int mi = 0; mi < Shape::MI; ++mi) {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        p[mi][0][e] = p[mi][1][e] = 0;
                    seismic_mma_m16n8k32_s8(p[mi][0], a[h][mi], b0);
                    seismic_mma_m16n8k32_s8(p[mi][1], a[h][mi], b1);
                }
                fold(p, 2 * h + part, h);
            }
        }
    }
}

template <class Shape, class WA, class WB, class Epi>
__device__ __forceinline__ void gemm_segment_s8(u8 *shared, const QuantizedRows &x, u32 M, u64 K, u64 nblock,
                                                u64 rows, const WA &wa, const WB &wb, const Epi &epi, u64 part = 0,
                                                u64 parts = 1) {
    constexpr bool PAIR = !Same<WB, NoWeight>::value;
    constexpr int STREAMS = PAIR ? 2 : 1;
    constexpr int THREADS = Shape::THREADS;
    constexpr int STAGE = Shape::template STAGE_BYTES<STREAMS, true>;
    constexpr int PITCH = Shape::template A_PITCH<true>;
    constexpr int WEIGHTS = Shape::BM * Shape::template A_ROW<true>;
    constexpr int MI = Shape::MI;
    constexpr int NJ = Shape::NJ;
    const u32 lane = threadIdx.x % 32;
    const u32 warp = threadIdx.x / 32;
    const u32 wm = warp % Shape::WM;
    const u32 wn = warp / Shape::WM;
    const u32 g = lane / 4;
    const u64 m0 = (u64)blockIdx.y * Shape::BM;
    const u64 tile0 = nblock * 8;
    const u64 tiles = (rows + 15) / 16;
    const u64 kbegin = K / 64 * part / parts;
    const u64 kblocks = K / 64 * (part + 1) / parts - kbegin;
    float *coefficients =
        reinterpret_cast<float *>(shared + Shape::STAGES * STAGE + warp * STREAMS * Shape::COEF_WARP);
    // As in `gemm_segment`: a warp whose rows all lie at or past M only loads.
    const bool active = m0 + wm * (MI * 16) < M;

    auto load = [&](int stage, u64 index) {
        const u64 kb = kbegin + index;
        u8 *base = shared + stage * STAGE;
        for (int c = threadIdx.x; c < Shape::BM * 4; c += THREADS) {
            const u64 row = m0 + c / 4;
            const bool valid = row < M;
            seismic_cp_async_16_zfill(base + (c / 4) * PITCH + (c % 4) * 16,
                                      x.q + (valid ? row : 0) * K + kb * 64 + (c % 4) * 16, valid ? 16u : 0u);
        }
        u8 *staged_groups = base + Shape::BM * PITCH;
        for (int r = threadIdx.x; r < Shape::BM; r += THREADS) {
            const u64 row = m0 + r;
            const bool valid = row < M;
            seismic_cp_async_16_zfill(staged_groups + r * 16, x.groups + (valid ? row : 0) * (K / 32) + kb * 2,
                                      valid ? 16u : 0u);
        }
        gemm_load_weight<WA, THREADS>(wa, tile0, tiles, kb, base + WEIGHTS);
        if constexpr (PAIR)
            gemm_load_weight<WB, THREADS>(wb, tile0, tiles, kb, base + WEIGHTS + Shape::W_STREAM);
    };

    float acc[2][NJ][MI][2][4];
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        acc[s][j][mi][h][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < Shape::STAGES - 1; ++s) {
        if ((u64)s < kblocks)
            load(s, s);
        seismic_cp_async_commit();
    }
    for (u64 kb = 0; kb < kblocks; ++kb) {
        seismic_cp_async_wait<Shape::STAGES - 2>();
        __syncthreads();
        const u64 next = kb + Shape::STAGES - 1;
        if (next < kblocks)
            load((int)(next % Shape::STAGES), next);
        seismic_cp_async_commit();
        if (!active)
            continue;
        const u8 *base = shared + (kb % Shape::STAGES) * STAGE;
        gemm_decode_coefficients<WA, NJ>(wa, base + WEIGHTS + Shape::W_CODES, wn * NJ * 16, kbegin + kb,
                                         coefficients);
        if constexpr (PAIR)
            gemm_decode_coefficients<WB, NJ>(wb, base + WEIGHTS + Shape::W_STREAM + Shape::W_CODES, wn * NJ * 16,
                                             kbegin + kb, coefficients + Shape::COEF_WARP / 4);
        // Activation fragments and (d, d * sum q) of the two 32-code groups.
        u32 a[2][MI][4];
        float2 groups[2][MI][2];
        const float2 *staged_groups = reinterpret_cast<const float2 *>(base + Shape::BM * PITCH);
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                const u32 row = wm * (MI * 16) + mi * 16;
                seismic_ldmatrix_x4(a[h][mi], base + (row + lane % 16) * PITCH + h * 32 + (lane / 16) * 16);
                groups[h][mi][0] = staged_groups[(row + g) * 2 + h];
                groups[h][mi][1] = staged_groups[(row + g + 8) * 2 + h];
            }
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int tile_local = wn * NJ + j;
            gemm_tile_s8<Shape>(acc[0][j], wa, base + WEIGHTS + tile_local * 1024, coefficients, j, a, groups);
            if constexpr (PAIR)
                gemm_tile_s8<Shape>(acc[1][j], wb, base + WEIGHTS + Shape::W_STREAM + tile_local * 1024,
                                    coefficients + Shape::COEF_WARP / 4, j, a, groups);
        }
    }
    seismic_cp_async_wait<0>();
    gemm_publish(acc, m0 + wm * (MI * 16), tile0 + wn * NJ, M, rows, epi);
}

// ---------------------------------------------------------------------------
// Operand paths of an entry. The GEMV always runs the 16-bit path; the GEMM
// runs it or, with S8, the INT8 candidate (q8_1 activations and s8 MMAs). The
// staging launch (one block per row) writes the rows a launch does not read in
// place: the q8_1 rows (S8 GEMM: K bytes per row in `staged`, the (d, d * sum
// q) groups in `groups`), else the prologue's A rows (a prologue that is not
// read in place, M > 1: K A elements per row in `staged`).
// Whether the INT8 candidate applies to an entry's weights: it multiplies
// q8_1 activations with packed codes, so with any dense weight INT8 has no
// effect and the GEMM runs the 16-bit path.
template <class... W> constexpr bool quantizable = (!W::DENSE && ...);

template <bool S8, class Pro>
__device__ __forceinline__ void stage_row(const Pro &pro, u32 m, u32 M, u64 K, u8 *staged, void *groups) {
    if (S8 && M > GEMV_ROWS)
        stage_row_s8(pro, m, K, staged, reinterpret_cast<float2 *>(groups));
    else if constexpr (Pro::STAGED)
        form_row(pro, m, K, staged);
}

// The GEMV activation source. A prologue read in place is its own source.
// Any other prologue's rows are the staging launch's A rows, except at M = 1,
// where the whole GEMV block forms its row into `row` (its dynamic shared
// memory, K A elements) instead of waiting for a staging launch. `make` is
// block-collective.
template <bool STAGED, class Pro> struct GemvSourceOf {
    using type = Pro;
    __device__ static __forceinline__ type make(const Pro &pro, u8 *, u32, u64, const u8 *) { return pro; }
};
template <class Pro> struct GemvSourceOf<true, Pro> {
    using type = Plain<Act, AllRows>;
    __device__ static __forceinline__ type make(const Pro &pro, u8 *row, u32 M, u64 K, const u8 *staged) {
        if (M > 1)
            return type{staged, K, AllRows{}};
        form_row(pro, 0, K, row);
        __syncthreads();
        return type{row, K, AllRows{}};
    }
};
template <class Pro> using GemvSource = GemvSourceOf<Pro::STAGED, Pro>;

// One GEMM block on either path: `act` is the A rows (row stride
// `act_stride`) for the 16-bit path or the staged s8 rows (with their
// `groups`) for S8.
template <class Shape, bool S8, class WA, class WB, class Epi>
__device__ __forceinline__ void gemm_run(u8 *shared, const u8 *act, u64 act_stride, const void *groups, u32 M,
                                         u64 K, u64 nblock, u64 rows, const WA &wa, const WB &wb, const Epi &epi,
                                         u64 part = 0, u64 parts = 1) {
    if constexpr (S8)
        gemm_segment_s8<Shape>(shared, QuantizedRows{act, K, reinterpret_cast<const float2 *>(groups)}, M, K, nblock,
                               rows, wa, wb, epi, part, parts);
    else
        gemm_segment<Shape>(shared, ActivationRows{act, act_stride}, M, K, nblock, rows, wa, wb, epi, part, parts);
}

// Block columns of a segment of `rows` rows.
__device__ __forceinline__ u64 gemm_columns(u64 rows) { return ((rows + 15) / 16 + 7) / 8; }

// Split-K: each K share stores its F32 projection into
// partials[part][stream][m][column] ([parts, STREAMS, M, columns]);
// `offset` places the segment's rows among the entry's columns.
template <int STREAMS> struct PartialStore {
    float *partials;
    u64 rows_m;
    u64 columns;
    u64 offset;
    u64 part;
    __device__ __forceinline__ void operator()(u32 m, u64 n, float first, float second) const {
        float *slice = partials + (part * STREAMS * rows_m + m) * columns + offset + n;
        slice[0] = first;
        if constexpr (STREAMS == 2)
            slice[rows_m * columns] = second;
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float first0, float first1, float second0,
                                         float second1) const {
        u8 *slice = reinterpret_cast<u8 *>(partials);
        const u64 index = (part * STREAMS * rows_m + m) * columns + offset + n;
        element::put2<element::F32>(slice, index, first0, first1);
        if constexpr (STREAMS == 2)
            element::put2<element::F32>(slice, index + rows_m * columns, second0, second1);
    }
};

// The finalize launch of a split-K GEMM: sums each (m, column) over the K
// shares in part order and hands the totals to `apply(m, column, first,
// second)`. One thread per element.
template <int STREAMS, class Apply>
__device__ __forceinline__ void split_finalize(const float *partials, u64 parts, u64 rows_m, u64 columns,
                                               const Apply &apply) {
    const u64 element = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= rows_m * columns)
        return;
    const u32 m = (u32)(element / columns);
    const u64 column = element % columns;
    float first = 0.0f, second = 0.0f;
    for (u64 part = 0; part < parts; ++part) {
        const float *slice = partials + (part * STREAMS * rows_m + m) * columns + column;
        first += slice[0];
        if constexpr (STREAMS == 2)
            second += slice[rows_m * columns];
    }
    apply(m, column, first, second);
}

} // namespace projection
