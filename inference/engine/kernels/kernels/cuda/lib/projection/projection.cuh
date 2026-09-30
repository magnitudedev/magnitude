// K1 packed projection family for CUDA: y[m, n] = sum_k x[m, k] * W[n, k]
// over mma16 weights, with a prologue that forms x and an epilogue that
// publishes y. Entry sources are thin instantiations.
//
// Numerics. Activations are rounded to the activation element A (bf16 or
// f16). GEMV: codes enter `mma.sync.m16n8k16` as exact 16-bit integers; each
// representation group (32 codes for q4k/q5k/q8/iq4, 16 for q6k; iq4 codes
// enter as their exact table values) accumulates in
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
// tiles over a STAGES-deep cp.async pipeline (activation rows and code chunks
// per k-block, the weight rows' coefficient records once per superblock),
// ldmatrix activation fragments, and weight B fragments dequantized from the
// staged mma16 chunks (an A-fragment register pair of a 16-row weight tile is
// the B fragment of one of its 8-row halves). Blocks walk the grid column-major
// in groups of row bands, so co-resident blocks share a block column's
// weights. Entries run it in two row bands with their own launches:
// `SmallGemm` up to SMALL_GEMM_ROWS (64-row tiles, the INT8 candidate) and
// `LargeGemm` beyond (128-row tiles, the 16-bit path, its copies issued by a
// rotating set of warps: the launches' tuning parameter ROTATE). Entries with
// few output columns also split it over K up to SPLIT_ROWS rows. Measured on
// GB10 for every K1 entry (17 to 512 rows, both operand paths, split 1/2/4):
// up to 64 rows want the small tiles, beyond them the large ones, and the
// few-column entries the split up to 128 rows.
#include "../core/activation.cuh"
#include "../core/functions.cuh"
#include <seismic/packets.cuh>
#include "../core/reduce.cuh"

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

#include "stage.cuh"

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

// Activation-generic GLU (paired), `function` a `functions::` code (SiLU
// gives SiluMul's bits):
// out[m, n] = round(round_A(act(round_A(gate))) * round_A(up)).
template <class E> struct Glu {
    u8 *out;
    u64 stride;
    int function;
    __device__ __forceinline__ float value(float gate, float up) const {
        return Act::round(functions::activate(function, Act::round(gate))) * Act::round(up);
    }
    __device__ __forceinline__ void operator()(u32 m, u64 n, float gate, float up) const {
        element::put<E>(out, m * stride + n, value(gate, up));
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float gate0, float gate1, float up0, float up1) const {
        element::put2<E>(out, m * stride + n, value(gate0, up0), value(gate1, up1));
    }
};

// A plain activated projection (up-only feed-forward, ReLU²):
// out[m, n] = round(act(round_A(projection))).
template <class E> struct Activated {
    u8 *out;
    u64 stride;
    int function;
    __device__ __forceinline__ float value(float projected) const {
        return functions::activate(function, Act::round(projected));
    }
    __device__ __forceinline__ void operator()(u32 m, u64 n, float projected, float) const {
        element::put<E>(out, m * stride + n, value(projected));
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float first, float second, float, float) const {
        element::put2<E>(out, m * stride + n, value(first), value(second));
    }
};

// The F32 product of two projections of one input (paired), unrounded:
// out[m, n] = gate * up.
struct Mul {
    float *out;
    u64 stride;
    __device__ __forceinline__ void operator()(u32 m, u64 n, float gate, float up) const {
        out[m * stride + n] = gate * up;
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float gate0, float gate1, float up0, float up1) const {
        out[m * stride + n] = gate0 * up0;
        out[m * stride + n + 1] = gate1 * up1;
    }
};

// An activated projection times an external F32 multiplier:
// out[m, n] = round(round_A(act(round_A(projection))) * external[m, n]).
template <class E> struct ActivatedMul {
    u8 *out;
    u64 stride;
    const float *external;
    u64 external_stride;
    int function;
    __device__ __forceinline__ float value(u32 m, u64 n, float projected) const {
        return Act::round(functions::activate(function, Act::round(projected))) * external[m * external_stride + n];
    }
    __device__ __forceinline__ void operator()(u32 m, u64 n, float projected, float) const {
        element::put<E>(out, m * stride + n, value(m, n, projected));
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float first, float second, float, float) const {
        element::put2<E>(out, m * stride + n, value(m, n, first), value(m, n + 1, second));
    }
};

// F32 logits, softcapped in F32 from the accumulator when `cap` > 0:
// out[m, n] = cap > 0 ? cap * tanh(projection / cap) : projection.
struct Logits {
    u8 *out;
    u64 stride;
    float cap;
    __device__ __forceinline__ float value(float projected) const {
        return cap > 0.0f ? functions::softcap(cap, projected) : projected;
    }
    __device__ __forceinline__ void operator()(u32 m, u64 n, float projected, float) const {
        element::put<element::F32>(out, m * stride + n, value(projected));
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float first, float second, float, float) const {
        element::put2<element::F32>(out, m * stride + n, value(first), value(second));
    }
};

// The absent second stream of an unpaired projection.
struct NoWeight {
    static constexpr int GROUP = 16;
    static constexpr int GROUPS = 4;
    static constexpr bool BIAS = false;
    static constexpr bool DENSE = false;
    static constexpr int CHUNKS = 0;
    static constexpr int SUPER_WORDS = 0;
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
// chunks of the 8 tiles (`TILE`: the representation's CHUNKS 16 B chunks per
// tile). The 128 rows' coefficient records (`RECORD` bytes per row: the
// representation's SUPER_WORDS words, 16 B aligned) are staged once per
// superblock into one of two record buffers, with the stage of the
// superblock's first k-block (or of the share's first). Each warp decodes its
// own rows' coefficients of every k-block into its area (16 B per row per
// stream).
template <int BM_, int WM_, int WN_, int STAGES_, bool ORDERED_ = false, int ROTATE_ = 1> struct GemmShape {
    static constexpr int BM = BM_;
    static constexpr int WM = WM_;
    static constexpr int WN = WN_;
    static constexpr int STAGES = STAGES_;
    static constexpr bool ORDERED = ORDERED_;
    static constexpr int WARPS = WM * WN;
    static constexpr int THREADS = 32 * WARPS;
    // A stage's copies are issued by LOADERS threads, a set of WARPS / ROTATE
    // warps that rotates with the k-block (`gemm_segment`).
    static constexpr int ROTATE = ROTATE_;
    static constexpr int LOADERS = THREADS / ROTATE;
    static_assert(WARPS % ROTATE == 0 && BM * 8 % LOADERS == 0, "loader sets of whole warps and chunks");
    static constexpr int MI = BM / WM / 16; // m16 tiles per warp
    static constexpr int NJ = 8 / WN;       // weight tiles per warp
    static_assert(MI * 16 * WM == BM && NJ * WN == 8, "warp tiling covers the block tile");
    static_assert(STAGES >= 2 && STAGES <= 5, "a record buffer is reloaded only after its superblock's k-blocks");
    template <bool S8> static constexpr int A_PITCH = S8 ? 80 : 144;
    // Activation bytes per row per stage (s8 rows carry their groups).
    template <bool S8> static constexpr int A_ROW = S8 ? 96 : 144;
    template <class W> static constexpr int TILE = W::CHUNKS * 16;
    template <class W> static constexpr int RECORD = (W::SUPER_WORDS * 4 + 15) / 16 * 16;
    static constexpr int COEF_WARP = NJ * 16 * 16;
    template <class WA, class WB, bool S8>
    static constexpr int STAGE_BYTES = BM * A_ROW<S8> + 8 * (TILE<WA> + TILE<WB>);
    // One record buffer: both streams' 128 rows.
    template <class WA, class WB> static constexpr int RECORDS = 128 * (RECORD<WA> + RECORD<WB>);
    template <class WA, class WB, bool S8>
    static constexpr int LAYOUT_BYTES = STAGES * STAGE_BYTES<WA, WB, S8> + 2 * RECORDS<WA, WB> +
                                        (Same<WB, NoWeight>::value ? 1 : 2) * WARPS * COEF_WARP;
    // Shared bytes (the declared `shared_bytes` of a GEMM launch), a bound on
    // every representation's layout:
    //   STAGES * (BM * 144 + STREAMS * 10240) + STREAMS * WARPS * NJ * 256
    // (the 16-bit layout, which also covers S8's smaller rows: INT8 has no
    // effect with dense weights, so a configuration's GEMM may run either).
    // Per stream the largest codes (q8: 1024 B per tile, 16 B records) and
    // records (q6k, 32 B per row, 768 B per tile) take STAGES * 8192 + 4096
    // and STAGES * 6144 + 8192 of its STAGES * 10240 bytes.
    template <int STREAMS>
    static constexpr int SHARED_BYTES = STAGES * (BM * 144 + STREAMS * 10240) + STREAMS * WARPS * COEF_WARP;
};

// The block column and row band of a GEMM block over a grid of block columns
// (x) by row bands (y). An ordered shape walks the blocks in groups of BANDS
// row bands: within a group, consecutive blocks take the group's bands of one
// block column in turn, so the co-resident blocks of a column share its weight
// tiles (fetched once from memory per group) while the group's activation rows
// stay in L2. Unordered shapes take (x, y) as they are. Measured on GB10 (4B
// expand and output at 512 and 2048 rows): the column-major walk cuts the q8
// expand's time by 40% at 512 rows (its weights exceed L2), and 4 bands per
// group keep the output's activation rows (K = 9216) in L2 at 2048 rows,
// where 16 bands lose a quarter of its rate.
constexpr u64 BANDS = 4;
template <class Shape> __device__ __forceinline__ u64 gemm_column() {
    if constexpr (!Shape::ORDERED)
        return blockIdx.x;
    const u64 linear = blockIdx.x + (u64)gridDim.x * blockIdx.y;
    const u64 group = linear / ((u64)gridDim.x * BANDS);
    const u64 bands = min((u64)gridDim.y - group * BANDS, BANDS);
    return (linear - group * gridDim.x * BANDS) / bands;
}
template <class Shape> __device__ __forceinline__ u64 gemm_band() {
    if constexpr (!Shape::ORDERED)
        return blockIdx.y;
    const u64 linear = blockIdx.x + (u64)gridDim.x * blockIdx.y;
    const u64 group = linear / ((u64)gridDim.x * BANDS);
    const u64 bands = min((u64)gridDim.y - group * BANDS, BANDS);
    return group * BANDS + (linear - group * gridDim.x * BANDS) % bands;
}

// The GEMM row bands of the K1 entries (declared as their `_gemm_small` and
// `_gemm` launches). Each warp owns one weight tile of the block tile and all
// its activation rows (one warp row: measured faster than two at every row
// count, most at 512 rows; sixteen warps in two warp rows measured slower
// too). Shared bytes: small 3 * (64 * 144 + STREAMS * 10240) + STREAMS *
// 2048, large 2 * (128 * 144 + STREAMS * 10240) + STREAMS * 2048.
//
// The large band's loader rotation is its launches' tuning parameter ROTATE
// (1, 2 or 4 loader sets). Measured on GB10 (4B expand and output, 512 and
// 2048 rows): the q5k, q6k and q8 expands and every output want 2 sets (up
// to +10%), the q4k expand 4 or 1, the mxfp4 expand 1 (2 lose a fifth). A
// third stage, a producer warp (nine warps leave 168 registers: the
// multiplying warps spill) and bulk (TMA) copies of the 128-byte activation
// rows all measured slower.
constexpr u32 SMALL_GEMM_ROWS = 64;
using SmallGemm = GemmShape<64, 1, 8, 3, true>;
template <unsigned ROTATE> using LargeGemm = GemmShape<128, 1, 8, 2, true, ROTATE>;
// Entries with few output columns split their GEMM over K into SPLIT shares
// up to SPLIT_ROWS rows (the declarations' grid z `1 + min(1, 128 / rows)`,
// partials scratch and finalize launch): their block columns alone leave most
// of the device idle there.
constexpr u32 SPLIT_ROWS = 128;
constexpr u32 SPLIT = 2;

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

// One stream's part of a stage: the code chunks (TILE bytes per tile) of tiles
// [tile0, tile0 + 8) (zero past `tiles`), by threads `first`, `first + STRIDE`, ...
template <class W, int STRIDE, int TILE>
__device__ __forceinline__ void gemm_load_codes(const W &w, u64 tile0, u64 tiles, u64 kblock, u8 *codes, u32 first) {
    for (int c = first; c < 8 * W::CHUNKS; c += STRIDE) {
        const u64 j = c / W::CHUNKS;
        const u32 q = c % W::CHUNKS;
        const bool valid = tile0 + j < tiles;
        seismic_cp_async_16_zfill(codes + j * TILE + q * 16, w.chunk_source(valid ? tile0 + j : tile0, kblock, q),
                                  valid ? 16u : 0u);
    }
}

// One stream's part of a record buffer: the coefficient records (RECORD bytes
// per row) of superblock `superblock` of the rows of tiles [tile0, tile0 + 8),
// zero past `tiles` and for words of k-blocks at or past `kblocks` (a partial
// last superblock), by threads `first`, `first + STRIDE`, ...
template <class W, int STRIDE, int RECORD>
__device__ __forceinline__ void gemm_load_records(const W &w, u64 tile0, u64 tiles, u64 superblock, u64 kblocks,
                                                  u8 *records, u32 first) {
    for (int c = first; c < 128 * W::SUPER_WORDS; c += STRIDE) {
        const u32 row = c / W::SUPER_WORDS;
        const int word = c % W::SUPER_WORDS;
        const bool valid = tile0 + row / 16 < tiles && 4 * superblock + W::word_kblock(word) < kblocks;
        cp_async_4_zfill(records + row * RECORD + word * 4,
                         w.super_word(tile0 * 16 + (valid ? row : 0), valid ? superblock : 0, valid ? word : 0),
                         valid ? 4u : 0u);
    }
}

// The warp's decoded coefficients of k-block `kblock`: its NJ tiles' rows
// (block-tile rows first..first + NJ * 16) from their records (the record
// buffer of the k-block's superblock), area[row * GROUPS + group] as F32
// (scale, -bias), the bias absent when the representation has none.
template <class W, int NJ, int RECORD>
__device__ __forceinline__ void gemm_decode_coefficients(const W &w, const u8 *records, u32 first, u64 kblock,
                                                         float *area) {
    if constexpr (W::DENSE)
        return;
    constexpr int ITEMS = NJ * 16 * W::GROUPS;
    constexpr int WORDS = W::BIAS ? 2 : 1;
    for (int item = threadIdx.x % 32; item < ITEMS; item += 32) {
        const u32 row = item / W::GROUPS;
        float scale, bias;
        w.staged_coefficient(reinterpret_cast<const u32 *>(records + (first + row) * RECORD), kblock,
                             item % W::GROUPS, scale, bias);
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

// A weight's second-level scale (NVFP4 `.scale`, per tensor or per expert)
// on the F32 accumulator, before the wrapped epilogue: `first` scales the
// projection (a paired epilogue's first stream), `second` the second
// stream. Entries wrap their epilogue only when a scale port is present
// (static extent 1; `scaling`), so an unscaled entry compiles as before.
template <class Epi> struct Scaled {
    Epi epi;
    float first, second;
    __device__ __forceinline__ void operator()(u32 m, u64 n, float a, float b) const {
        epi(m, n, a * first, b * second);
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float a0, float a1, float b0, float b1) const {
        epilogue_pair(epi, m, n, a0 * first, a1 * first, b0 * second, b1 * second, 0);
    }
};

template <bool SCALED> struct scaling;
template <> struct scaling<false> {
    template <class Epi> using type = Epi;
    template <class Epi> __device__ __forceinline__ static Epi wrap(const Epi &epi, float, float) { return epi; }
};
template <> struct scaling<true> {
    template <class Epi> using type = Scaled<Epi>;
    template <class Epi> __device__ __forceinline__ static Scaled<Epi> wrap(const Epi &epi, float first, float second) {
        return Scaled<Epi>{epi, first, second};
    }
};

// A scale port's value at `index` along its expert axis (0 for a
// per-tensor port; `stride` that axis' stride), or 1 for an absent port
// (static `extent` 0: the port is never read).
__device__ __forceinline__ float scale_factor(const u8 *scale, u64 extent, u64 stride, u64 index) {
    return extent == 0 ? 1.0f : reinterpret_cast<const float *>(scale)[index * stride];
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
//
// A representation with `SCALED_GEMM` keeps its codes exact in the operand
// (they are small integers) and scales each group's F32 partial product, as
// the GEMV does: acc += scale * (x . code) - bias * sum(x), the group's
// activation sum formed by the same MMA against an all-ones operand. Its
// weights are then never rounded to A, whose 8-bit mantissa (bf16) would
// otherwise round every dequantized weight.
template <class Shape, class W>
__device__ __forceinline__ void gemm_tile_half(float acc[Shape::MI][2][4], const W &w, const typename W::Raw &raw,
                                               const float *coefficients, int j_tile, int half,
                                               const u32 a[2][Shape::MI][4]) {
    const u32 g = (threadIdx.x % 32) / 4;
    constexpr int STEPS = W::GROUP / 16;
    if constexpr (W::SCALED_GEMM) {
        const u32 t = threadIdx.x % 4;
        const u32 ones[2] = {Op::ONES, Op::ONES};
        u32 r[2][4];
#pragma unroll
        for (int local = 0; local < 2; ++local)
            w.template decode<Op>(raw, 2 * half + local, r[local]);
        // One partial product per coefficient group: both steps of the half
        // for 32-code groups, each step for 16-code groups.
        constexpr int FOLDS = 2 / STEPS;
#pragma unroll
        for (int fold = 0; fold < FOLDS; ++fold) {
            const int group = (2 * half) / STEPS + fold;
            // The C columns of this lane: weight rows h * 8 + 2t + c, with
            // their (scale, -bias).
            float2 k[2][2];
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 2; ++c)
                    k[h][c] = gemm_coefficient<W>(coefficients, j_tile * 16 + h * 8 + 2 * t + c, group);
#pragma unroll
            for (int mi = 0; mi < Shape::MI; ++mi) {
                float p[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
                // sum(x) of the group per C row (every column equal).
                float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int local = fold * STEPS; local < (fold + 1) * STEPS; ++local) {
                    const u32 b0[2] = {r[local][0], r[local][2]};
                    const u32 b1[2] = {r[local][1], r[local][3]};
                    Op::mma(p[0], a[local][mi], b0);
                    Op::mma(p[1], a[local][mi], b1);
                    if constexpr (W::BIAS)
                        Op::mma(s, a[local][mi], ones);
                }
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        acc[mi][h][e] = seismic_fma_rn(k[h][e % 2].x, p[h][e], acc[mi][h][e]);
                        if constexpr (W::BIAS)
                            acc[mi][h][e] = seismic_fma_rn(k[h][e % 2].y, s[e], acc[mi][h][e]);
                    }
            }
        }
        return;
    }
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

// The weight side of a GEMM block's shared memory: STAGES stages (BM
// activation rows of A_ROW<S8> bytes, then each stream's codes), the two
// record buffers (each stream's records) and the warps' coefficient areas
// (each stream's). `load` stages k-block `index` of the share (with its
// superblock's records when it starts one or the share) by threads `first`,
// `first + STRIDE`, ...; `decode` forms the warp's coefficients of the share's
// k-block `index` from its record buffer.
template <class Shape, class WA, class WB, bool S8> struct GemmWeights {
    static constexpr bool PAIR = !Same<WB, NoWeight>::value;
    static constexpr int STREAMS = PAIR ? 2 : 1;
    static constexpr int ROWS = Shape::BM * Shape::template A_ROW<S8>;
    static constexpr int TILE_A = Shape::template TILE<WA>;
    static constexpr int TILE_B = Shape::template TILE<WB>;
    static constexpr int RECORD_A = Shape::template RECORD<WA>;
    static constexpr int RECORD_B = Shape::template RECORD<WB>;
    static constexpr int STAGE = Shape::template STAGE_BYTES<WA, WB, S8>;
    static constexpr int BUFFER = Shape::template RECORDS<WA, WB>;
    static constexpr int RECORDS = Shape::STAGES * STAGE;
    static constexpr int AREAS = RECORDS + 2 * BUFFER;
    static_assert(Shape::template LAYOUT_BYTES<WA, WB, S8> == AREAS + STREAMS * Shape::WARPS * Shape::COEF_WARP,
                  "the layout of GemmShape");
    static_assert(Shape::template LAYOUT_BYTES<WA, WB, S8> <= Shape::template SHARED_BYTES<STREAMS>,
                  "the declared shared bytes bound every representation's layout");

    u8 *shared;
    WA wa;
    WB wb;
    u64 tile0, tiles, kbegin, ktotal;

    __device__ __forceinline__ u8 *stage(int s) const { return shared + s * STAGE; }
    __device__ __forceinline__ const u8 *codes(int s, int stream, int tile) const {
        return shared + s * STAGE + ROWS + (stream == 0 ? tile * TILE_A : 8 * TILE_A + tile * TILE_B);
    }
    __device__ __forceinline__ const u8 *records(u64 kb, int stream) const {
        return shared + RECORDS + (kb / 4) % 2 * BUFFER + (stream == 0 ? 0 : 128 * RECORD_A);
    }
    __device__ __forceinline__ float *area(u32 warp, int stream) const {
        return reinterpret_cast<float *>(shared + AREAS + (warp * STREAMS + stream) * Shape::COEF_WARP);
    }
    template <int STRIDE> __device__ __forceinline__ void load(int s, u64 index, u32 first) const {
        const u64 kb = kbegin + index;
        gemm_load_codes<WA, STRIDE, TILE_A>(wa, tile0, tiles, kb, stage(s) + ROWS, first);
        if constexpr (PAIR)
            gemm_load_codes<WB, STRIDE, TILE_B>(wb, tile0, tiles, kb, stage(s) + ROWS + 8 * TILE_A, first);
        if (index == 0 || kb % 4 == 0) {
            u8 *buffer = shared + RECORDS + (kb / 4) % 2 * BUFFER;
            gemm_load_records<WA, STRIDE, RECORD_A>(wa, tile0, tiles, kb / 4, ktotal, buffer, first);
            if constexpr (PAIR)
                gemm_load_records<WB, STRIDE, RECORD_B>(wb, tile0, tiles, kb / 4, ktotal, buffer + 128 * RECORD_A,
                                                        first);
        }
    }
    __device__ __forceinline__ void decode(u64 index, u32 warp, u32 first) const {
        const u64 kb = kbegin + index;
        gemm_decode_coefficients<WA, Shape::NJ, RECORD_A>(wa, records(kb, 0), first, kb, area(warp, 0));
        if constexpr (PAIR)
            gemm_decode_coefficients<WB, Shape::NJ, RECORD_B>(wb, records(kb, 1), first, kb, area(warp, 1));
    }
};

// One warp's multiply of the share's staged k-block `index` (in stage
// `stage`) into its accumulators: the warp decodes its rows' coefficients,
// then per 32-code half multiplies the half's two k16 steps of activation
// fragments by its tiles' dequantized weight fragments.
template <class Shape, class Weights>
__device__ __forceinline__ void gemm_multiply(float (&acc)[2][Shape::NJ][Shape::MI][2][4], const Weights &weights,
                                              int stage, u64 index, u32 warp, u32 lane) {
    constexpr int PITCH = Shape::template A_PITCH<false>;
    constexpr int MI = Shape::MI;
    constexpr int NJ = Shape::NJ;
    const u32 wm = warp % Shape::WM;
    const u32 wn = warp / Shape::WM;
    const u8 *base = weights.stage(stage);
    const u64 kb = weights.kbegin + index;
    weights.decode(index, warp, wn * NJ * 16);
    const float *coefficients_a = weights.area(warp, 0);
    const float *coefficients_b = weights.area(warp, 1);
    typename decltype(weights.wa)::Raw raw_a[NJ];
    typename decltype(weights.wb)::Raw raw_b[NJ];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const u64 tile = weights.tile0 + wn * NJ + j;
        raw_a[j] = gemm_raw(weights.wa, weights.codes(stage, 0, wn * NJ + j), tile, kb, lane);
        if constexpr (Weights::PAIR)
            raw_b[j] = gemm_raw(weights.wb, weights.codes(stage, 1, wn * NJ + j), tile, kb, lane);
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        u32 a[2][MI][4];
#pragma unroll
        for (int local = 0; local < 2; ++local)
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                const int s = 2 * half + local;
                const u32 row = wm * (MI * 16) + mi * 16;
                seismic_ldmatrix_x4(a[local][mi], base + (row + lane % 16) * PITCH + (s * 16 + (lane / 16) * 8) * 2);
            }
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            gemm_tile_half<Shape>(acc[0][j], weights.wa, raw_a[j], coefficients_a, j, half, a);
            if constexpr (Weights::PAIR)
                gemm_tile_half<Shape>(acc[1][j], weights.wb, raw_b[j], coefficients_b, j, half, a);
        }
    }
}

// One GEMM block over segment-local block column `nblock` (tiles
// [8 * nblock, 8 * nblock + 8)), activation rows [BM * gemm_band(), ...) of
// `source` (the A-typed input [M, K], staged or in place) and K share `part`
// of `parts` (split-K: the epilogue is then a `PartialStore` and
// `split_finalize` applies the real one). Rows at or past M read zeros; a warp
// whose rows all lie there decodes and multiplies nothing.
template <class Shape, class Rows, class WA, class WB, class Epi>
__device__ __forceinline__ void gemm_segment(u8 *shared, const Rows &source, u32 M, u64 K, u64 nblock, u64 rows,
                                             const WA &wa, const WB &wb, const Epi &epi, u64 part = 0,
                                             u64 parts = 1) {
    using Weights = GemmWeights<Shape, WA, WB, false>;
    constexpr int LOADERS = Shape::LOADERS;
    constexpr int PITCH = Shape::template A_PITCH<false>;
    constexpr int MI = Shape::MI;
    constexpr int NJ = Shape::NJ;
    const u32 lane = threadIdx.x % 32;
    const u32 warp = threadIdx.x / 32;
    const u32 wm = warp % Shape::WM;
    const u32 wn = warp / Shape::WM;
    const u64 m0 = gemm_band<Shape>() * Shape::BM;
    const u64 tile0 = nblock * 8;
    const u64 kbegin = K / 64 * part / parts;
    const u64 kblocks = K / 64 * (part + 1) / parts - kbegin;
    const Weights weights{shared, wa, wb, tile0, (rows + 15) / 16, kbegin, K / 64};
    const bool active = m0 + wm * (MI * 16) < M;

    // The loader set of the share's k-block `index` issues its stage: the
    // loaders' other warps multiply meanwhile, their tensor pipes busy while
    // the loaders stall on the load-store unit (ROTATE is tuned: the best set
    // size depends on the representation's copy and decode work).
    const u32 loader = threadIdx.x % LOADERS;
    // A loader's activation chunks (16 bytes of one tile row per k-block)
    // keep their rows across k-blocks, so their addresses are formed once;
    // a chunk past M reads zeros from a valid address.
    constexpr int CHUNKS = Shape::BM * 8 / LOADERS;
    const u8 *chunk[CHUNKS];
    bool valid[CHUNKS];
#pragma unroll
    for (int i = 0; i < CHUNKS; ++i) {
        const int c = (int)loader + i * LOADERS;
        const u64 row = m0 + c / 8;
        valid[i] = row < M;
        chunk[i] = source.row(valid[i] ? row : 0) + (c % 8) * 16;
    }
    // `index` counts k-blocks of this share.
    auto load = [&](int stage, u64 index) {
        if (threadIdx.x / LOADERS != index % Shape::ROTATE)
            return;
        const u64 kb = kbegin + index;
        u8 *base = weights.stage(stage);
#pragma unroll
        for (int i = 0; i < CHUNKS; ++i) {
            const int c = (int)loader + i * LOADERS;
            seismic_cp_async_16_zfill(base + (c / 8) * PITCH + (c % 8) * 16, chunk[i] + kb * 128,
                                      valid[i] ? 16u : 0u);
        }
        weights.template load<LOADERS>(stage, index, loader);
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
        if (active)
            gemm_multiply<Shape>(acc, weights, (int)(kb % Shape::STAGES), kb, warp, lane);
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
// An s8 MMA product as F32, exactly: |p| <= 32 * 127 * 128 < 2^22, so
// p + 1.5 * 2^23 is an integer-valued float whose low mantissa bits are p. One
// integer add and one F32 add replace the quarter-rate integer-to-float
// conversion, which bounded the fold at about half the s8 MMA rate.
__device__ __forceinline__ float gemm_s8_product(int p) {
    return __int_as_float(p + 0x4B400000) - 12582912.0f;
}

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
                        value = seismic_fma_rn(k.x, x[h][mi][r].x * gemm_s8_product(p[mi][half][2 * r + c]), value);
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
    using Weights = GemmWeights<Shape, WA, WB, true>;
    constexpr bool PAIR = Weights::PAIR;
    constexpr int THREADS = Shape::THREADS;
    constexpr int PITCH = Shape::template A_PITCH<true>;
    constexpr int MI = Shape::MI;
    constexpr int NJ = Shape::NJ;
    const u32 lane = threadIdx.x % 32;
    const u32 warp = threadIdx.x / 32;
    const u32 wm = warp % Shape::WM;
    const u32 wn = warp / Shape::WM;
    const u32 g = lane / 4;
    const u64 m0 = gemm_band<Shape>() * Shape::BM;
    const u64 tile0 = nblock * 8;
    const u64 kbegin = K / 64 * part / parts;
    const u64 kblocks = K / 64 * (part + 1) / parts - kbegin;
    const Weights weights{shared, wa, wb, tile0, (rows + 15) / 16, kbegin, K / 64};
    const float *coefficients_a = weights.area(warp, 0);
    const float *coefficients_b = weights.area(warp, 1);
    // As in `gemm_segment`: a warp whose rows all lie at or past M only loads.
    const bool active = m0 + wm * (MI * 16) < M;

    auto load = [&](int stage, u64 index) {
        const u64 kb = kbegin + index;
        u8 *base = weights.stage(stage);
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
        weights.template load<THREADS>(stage, index, threadIdx.x);
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
        const int stage = (int)(kb % Shape::STAGES);
        const u8 *base = weights.stage(stage);
        weights.decode(kb, warp, wn * NJ * 16);
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
            gemm_tile_s8<Shape>(acc[0][j], wa, weights.codes(stage, 0, tile_local), coefficients_a, j, a, groups);
            if constexpr (PAIR)
                gemm_tile_s8<Shape>(acc[1][j], wb, weights.codes(stage, 1, tile_local), coefficients_b, j, a, groups);
        }
    }
    seismic_cp_async_wait<0>();
    gemm_publish(acc, m0 + wm * (MI * 16), tile0 + wn * NJ, M, rows, epi);
}

// ---------------------------------------------------------------------------
// Operand paths of an entry. The GEMV and the large-band GEMM run the 16-bit
// path; the small-band GEMM runs it or, with S8, the INT8 candidate (q8_1
// activations and s8 MMAs). The staging launches (one block per row) write
// the rows a launch does not read in place: `_stage` the prologue's A rows (a
// prologue that is not read in place, M > 1: K A elements per row in
// `staged`), `_stage_s8` the INT8 candidate's q8_1 rows (K bytes per row in
// `staged`, the (d, d * sum q) groups in `groups`).
// Whether the INT8 candidate applies to an entry's weights: it multiplies
// q8_1 activations with packed codes, so with any dense weight INT8 has no
// effect: the GEMM runs the 16-bit path, and `_stage_s8` stages what that
// path reads.
template <class... W> constexpr bool quantizable = (!W::DENSE && ...);

template <bool S8, class Pro>
__device__ __forceinline__ void stage_row(const Pro &pro, u32 m, u64 K, u8 *staged, void *groups) {
    if constexpr (S8)
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

// One GEMM block of a single-stream entry with few output columns: its K
// share's partials when the launch is split (grid z = SPLIT shares up to
// SPLIT_ROWS rows, summed by the finalize launch), else published through
// `epi`.
template <class Shape, bool S8, class W, class Epi>
__device__ __forceinline__ void gemm_run_split(u8 *shared, const u8 *act, u64 act_stride, const void *groups, u32 M,
                                               u64 K, u64 nblock, u64 rows, const W &w, const Epi &epi,
                                               float *partials) {
    if (gridDim.z > 1)
        gemm_run<Shape, S8>(shared, act, act_stride, groups, M, K, nblock, rows, w, NoWeight{},
                            PartialStore<1>{partials, M, rows, 0, blockIdx.z}, blockIdx.z, gridDim.z);
    else
        gemm_run<Shape, S8>(shared, act, act_stride, groups, M, K, nblock, rows, w, NoWeight{}, epi);
}

} // namespace projection
