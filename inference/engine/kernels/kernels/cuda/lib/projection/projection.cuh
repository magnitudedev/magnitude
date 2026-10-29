// K1 packed projection family for CUDA: y[m, n] = sum_k x[m, k] * W[n, k]
// over mma16 weights, with a prologue that forms x and an epilogue that
// publishes y. Entry sources are thin instantiations.
//
// Numerics. Activations are rounded to the activation element A (bf16 or
// f16). GEMV: codes enter `mma.sync.m16n8k16` as exact 16-bit integers; each
// representation group (32 codes for q4k/q5k/q8/iq4, 16 for q6k; iq4 codes
// enter as their exact table values) accumulates in
// F32 from zero and is folded as acc += scale * P - bias * sum(x), the
// group-factored form. GEMM: SCALED_GEMM representations also retain exact
// codes and fold each group's F32 partial output with its decoded coefficient.
// Other packed representations form round_A(code * scale - bias) from F32
// coefficient arithmetic and accumulate those operands in F32. Dense weights
// enter as their operand values. The GEMM's INT8 candidate quantizes activation
// rows (`int8_activations`): see `QuantizedRows`. Reductions across K run in a
// fixed order. A row's GEMV
// result does not depend on M (every activation row is its own MMA column),
// so rows 1..GEMV_ROWS of one request match M = 1 bit for bit under one
// launch configuration (TPW, KSPLIT).
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
// consecutive tiles, measured no gain on top of that), except that a staged
// weight's group copies its tiles' coefficient fields into shared memory as
// whole-row runs once (`GemvStaged`). Group sums of x come
// from the same B operands through an all-ones A fragment. Vector-coded
// weights run in entry order (`gemv_entries_group`): k permuted within each
// scale group so a lane's A values are half an entry, looked up once per row.
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
// few output columns also split it over K up to SPLIT_ROWS rows. The current
// row schedule uses small tiles through 64 rows, large tiles beyond, and splits
// few-column entries through 128 rows. Numerical-class admission is separate
// from this schedule; a row count does not grant activation quantization.
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

// Semantic Act owns prologue/epilogue rounding. A physical operand policy owns
// reusable row storage and the tensor operands; choosing it is a separate
// numerical transformation, never permission inferred from the row schedule.
struct SemanticGemmOperands {
    using Storage = Act;
    using Operand = Op;
    static constexpr bool EXACT_CODES = true;
    static constexpr bool MATERIALIZE = false;
    static constexpr bool FP4 = false;
};
struct F16GemmOperands {
    using Storage = element::F16;
    using Operand = packets::OpF16;
    static constexpr bool EXACT_CODES = false;
    static constexpr bool MATERIALIZE = true;
    static constexpr bool FP4 = false;
};
// NVFP4 operand rows staged packed (`PackedRows`) and multiplied with NVFP4
// weights on FP4 tensor operations (`FP4_MMA`, `gemm_multiply_fp4`). Every
// product is exact, as the 16-bit operands' are; the row's global scale
// multiplies its sums at publication.
struct Fp4GemmOperands {
    using Storage = Act;
    static constexpr bool MATERIALIZE = true;
    static constexpr bool FP4 = true;
};
template <unsigned Representation> struct GemmOperands;
template <> struct GemmOperands<0> { using type = SemanticGemmOperands; };
template <> struct GemmOperands<1> { using type = F16GemmOperands; };

// A when C, else B.
template <bool C, class A, class B> struct Pick {
    using Type = A;
};
template <class A, class B> struct Pick<false, A, B> {
    using Type = B;
};


// The largest row count the GEMV serves; larger counts run the GEMM.
constexpr u32 GEMV_ROWS = 16;

__device__ __forceinline__ float silu(float value) { return value / (1.0f + expf(-value)); }

// The named barrier of a GEMV block's tile group `group`, for the group's
// KSPLIT warps: barrier FIRST + group of GROUPS. A block reserves the hardware
// barriers 0 .. FIRST + GROUPS - 1 (ptxas counts up to the largest id) of an
// SM's 16, so the ids bound how many synchronizing blocks, of its own launch
// or another, share its SM (`GemvResidency`): with RESIDENCY R > 0 the block
// takes the top ids 16 / R - GROUPS .. 16 / R - 1 and at most R blocks fit
// (R = 1 holds the SM alone); with 0 it takes ids 1 .. GROUPS and registers
// and shared memory alone bound the blocks per SM. The id is selected among
// compile-time constants (a run-time id would reserve all 16 whatever the
// block's residency).
template <int GROUPS_, int RESIDENCY> struct GroupBarrier {
    static constexpr int GROUPS = GROUPS_;
    static constexpr int FIRST = RESIDENCY == 0 ? 1 : 16 / RESIDENCY - GROUPS;
    static_assert(FIRST >= 1, "barrier 0 is the block's own");
    u32 group;
    template <int ID = FIRST> __device__ __forceinline__ void sync(u32 threads) const {
        if constexpr (ID < FIRST + GROUPS - 1)
            if (group + FIRST != ID) {
                sync<ID + 1>(threads);
                return;
            }
        asm volatile("barrier.sync %0, %1;" ::"n"(ID), "r"(threads) : "memory");
    }
};

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

// The operands of a GEMM over weights W: FP4 tensor operations where the
// weights take NVFP4 operand rows and the device has them, else
// `OPERAND`'s.
template <unsigned OPERAND, class... W>
using GemmOperandsOf = typename Pick<FP4_MMA && (packets::nvfp4_operands<W>::value && ...), Fp4GemmOperands,
                                     typename GemmOperands<OPERAND>::type>::Type;

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

// `scatter`: each of the STREAMS hyper-connection streams of the residual
// plus the projection times the stream's inject value,
//   out[m, s * width + n] = residual[rows(m), s * width + n] + round_A(projection) * inject[m, s],
// or with INJECTED false (one stream) `Residual`'s sum.
template <class Rows, u32 STREAMS, bool INJECTED> struct Scatter {
    const float *residual;
    u64 residual_stride;
    Rows rows;
    float *out;
    u64 stride;
    const float *inject;
    u64 inject_stride;
    u64 width;
    __device__ __forceinline__ void operator()(u32 m, u64 n, float value, float) const {
        const float y = Act::round(value);
#pragma unroll
        for (u32 s = 0; s < STREAMS; ++s) {
            const float base = residual[rows(m) * residual_stride + s * width + n];
            out[m * stride + s * width + n] = INJECTED ? base + y * inject[m * inject_stride + s] : base + y;
        }
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float first, float second, float, float) const {
        (*this)(m, n, first, 0.0f);
        (*this)(m, n + 1, second, 0.0f);
    }
};

// A routed sublayer's column n joining its row's residual (`routed_output`,
// `routed_combine`): residual + selected + shared, in that order, or (INJECTED)
// for each of the STREAMS streams residual[s] + (selected + shared) * inject[s].
// `out` and `residual` point at the row, `inject` at its inject values.
template <u32 STREAMS, bool INJECTED>
__device__ __forceinline__ void routed_join(float *out, const float *residual, const float *inject, u64 width, u64 n,
                                            float selected, float shared) {
    if constexpr (!INJECTED) {
        out[n] = residual[n] + selected + shared;
    } else {
#pragma unroll
        for (u32 s = 0; s < STREAMS; ++s)
            out[s * width + n] = residual[s * width + n] + (selected + shared) * inject[s];
    }
}

// An output entry's epilogue: `Residual`, or `Scatter` over STREAMS streams
// when the entry takes inject values. `make` forms either from every field
// (`Residual` reads the leading ones).
template <class Rows, u32 STREAMS, bool INJECTED> struct OutputOf {
    using type = Scatter<Rows, STREAMS, INJECTED>;
    __device__ static __forceinline__ type make(const float *residual, u64 residual_stride, Rows rows, float *out,
                                                u64 stride, const float *inject, u64 inject_stride, u64 width) {
        return type{residual, residual_stride, rows, out, stride, inject, inject_stride, width};
    }
};
template <class Rows> struct OutputOf<Rows, 1, false> {
    using type = Residual<Rows>;
    __device__ static __forceinline__ type make(const float *residual, u64 residual_stride, Rows rows, float *out,
                                                u64 stride, const float *, u64, u64) {
        return type{residual, residual_stride, rows, out, stride};
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
template <class E, class Publication = E> struct Glu {
    u8 *out;
    u64 stride;
    int function;
    __device__ __forceinline__ float value(float gate, float up) const {
        const float product = Act::round(functions::activate(function, Act::round(gate))) * Act::round(up);
        if constexpr (Same<E, Publication>::value)
            return product;
        else
            return Publication::round(product);
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
template <class E, class Publication = E> struct Activated {
    u8 *out;
    u64 stride;
    int function;
    __device__ __forceinline__ float value(float projected) const {
        const float activated = functions::activate(function, Act::round(projected));
        if constexpr (Same<E, Publication>::value)
            return activated;
        else
            return Publication::round(activated);
    }
    __device__ __forceinline__ void operator()(u32 m, u64 n, float projected, float) const {
        element::put<E>(out, m * stride + n, value(projected));
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float first, float second, float, float) const {
        element::put2<E>(out, m * stride + n, value(first), value(second));
    }
};

// A mix site's low-rank rows (`stream_mix_down`) over S streams:
// out[m, n] = round_E(silu(projection / S)).
template <class E> struct MixLow {
    u8 *out;
    u64 stride;
    float streams;
    __device__ __forceinline__ float value(float projected) const {
        const float v = projected / streams;
        return v / (1.0f + expf(-v));
    }
    __device__ __forceinline__ void operator()(u32 m, u64 n, float projected, float) const {
        element::put<E>(out, m * stride + n, value(projected));
    }
};

// A mix site's inject values (`stream_mix_down`) over S streams:
// out[m, n] = 2 sigmoid(projection / S) (F32).
struct MixInject {
    float *out;
    u64 stride;
    float streams;
    __device__ __forceinline__ void operator()(u32 m, u64 n, float projected, float) const {
        out[m * stride + n] = 2.0f / (1.0f + expf(-projected / streams));
    }
};

// A mix site's per-stream partials (`stream_mix_up`): column n of stream
// n / width gated by its projection,
// out[m, n] = sigmoid(projection) * x[m, n] * inverse[m, n / width] * w[n] / S (F32).
template <class NORM> struct Mix {
    float *out;
    u64 stride;
    const float *x;
    u64 x_stride;
    const float *inverse;
    u64 inverse_stride;
    const u8 *norm;
    u64 width;
    float streams;
    __device__ __forceinline__ void operator()(u32 m, u64 n, float projected, float) const {
        const float normalized = x[m * x_stride + n] * inverse[m * inverse_stride + n / width]
            * element::at<NORM>(norm, n);
        const float gate = 1.0f / (1.0f + expf(-projected));
        out[m * stride + n] = gate * normalized / streams;
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
    __device__ __forceinline__ Raw fetch(u64, u64, u32) const { return Raw{}; }
    __device__ __forceinline__ Raw raw(const Super &, int, u32) const { return Raw{}; }
    __device__ __forceinline__ Block block(u64, u64, u32) const { return Block{}; }
};

// ---------------------------------------------------------------------------
// GEMV (M <= GEMV_ROWS).

// Operand sums of x over each GROUP-code group of a k-block (a biased
// weight's coefficient group), for the lane's C-fragment columns 2t and 2t+1
// of each 8-row column block.
template <int NB, int GROUP> struct GroupSums {
    static_assert(GROUP == 16 || GROUP == 32, "groups of 16 or 32 codes");
    float group[NB][64 / GROUP][2];
};

// The coefficient group a weight's GEMV takes operand sums over: its own
// where it has a bias, else none (0).
template <class W> __host__ __device__ constexpr int bias_group() {
    if constexpr (W::BIAS)
        return W::GROUP;
    else
        return 0;
}

// The operand sums of each GROUP-code group (16 or 32) of k-block operands b:
// one MMA against an all-ones operand per step, accumulated over the group's
// steps.
template <int NB, int GROUP>
__device__ __forceinline__ void group_sums(const u32 (&b)[NB][4][2], GroupSums<NB, GROUP> &sums) {
    const u32 ones[4] = {Op::ONES, Op::ONES, Op::ONES, Op::ONES};
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int h = 0; h < 64 / GROUP; ++h) {
            float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            // The group's steps as explicit MMAs: a step loop, even unrolled,
            // compiles 32-code groups to other code than two calls.
            Op::mma(x, ones, b[nb][GROUP / 16 * h]);
            if constexpr (GROUP == 32)
                Op::mma(x, ones, b[nb][GROUP / 16 * h + 1]);
            sums.group[nb][h][0] = x[0];
            sums.group[nb][h][1] = x[1];
        }
}

template <int NB, class W, int SUM_GROUP>
__device__ __forceinline__ void gemv_accumulate(float (&acc)[NB][4], const typename W::Raw &raw, const W &w,
                                                const typename W::Block &block, int q, const u32 (&b)[NB][4][2],
                                                const GroupSums<NB, SUM_GROUP> &sums) {
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
                static_assert(W::GROUP == SUM_GROUP, "a biased weight takes the sums of its own groups");
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[nb][e] = seismic_fma_rn(-c.bias[e / 2][group], sums.group[nb][group][e % 2], acc[nb][e]);
            }
        }
    }
}

// Whether a weight's GEMV reads its k-blocks in the permuted column order
// (`fetch_superblock_permuted`): dense weights, whose rows load as whole
// vectors that way.
template <class W, class = void> struct GemvPermuted {
    static constexpr bool value = false;
};
template <class W> struct GemvPermuted<W, decltype((void)W::GEMV_PERMUTED)> {
    static constexpr bool value = W::GEMV_PERMUTED;
};

// Whether a weight's GEMV stages its tiles' coefficient fields in shared
// memory (`W::GEMV_STAGED`, see `packets::Direct::stage`).
template <class W, class = void> struct GemvStaged {
    static constexpr bool value = false;
};
template <class W> struct GemvStaged<W, decltype((void)W::GEMV_STAGED)> {
    static constexpr bool value = W::GEMV_STAGED;
};
// The staged bytes of one tile of W (0 when W is not staged).
template <class W> __host__ __device__ constexpr u64 gemv_staged_tile(u64 kblocks) {
    if constexpr (GemvStaged<W>::value)
        return W::staged_bytes(kblocks);
    else
        return 0;
}

// Whether a weight's GEMV runs in entry order (`W::GEMV_ENTRIES`): vector
// codes, whose eight values an entry names together, run
// `gemv_entries_group`, which looks each entry up once per row instead of
// once per two values.
template <class W, class = void> struct GemvEntries {
    static constexpr bool value = false;
};
template <class W> struct GemvEntries<W, decltype((void)W::GEMV_ENTRIES)> {
    static constexpr bool value = W::GEMV_ENTRIES;
};

// Whether a weight's GEMV reads its superblocks as records
// (`W::GEMV_RECORDS`, see `gemv_records_group`).
template <class W, class = void> struct GemvRecords {
    static constexpr bool value = false;
};
template <class W> struct GemvRecords<W, decltype((void)W::GEMV_RECORDS)> {
    static constexpr bool value = W::GEMV_RECORDS;
};

// The K-split reduction and epilogue of a GEMV group's accumulators (the
// MMA C-fragment places: (row g, col 2t), (row g, col 2t+1), (row g+8, col
// 2t), (row g+8, col 2t+1) of each column block).
template <int TPW, int KSPLIT, int NB, class Epi, class Barrier>
__device__ __forceinline__ void gemv_finish(float (&acc)[2][TPW][NB][4], u32 M, u64 tile0, u64 tiles, u64 rows,
                                            const Epi &epi, float *reduce, const Barrier &barrier, u32 part) {
    const u32 lane = threadIdx.x % 32;
    const u32 g = lane / 4;
    const u32 t = lane % 4;
    if constexpr (KSPLIT > 1) {
        constexpr int VALUES = 2 * TPW * NB * 4;
#pragma unroll
        for (int v = 0; v < VALUES; ++v)
            reduce[(part * VALUES + v) * 32 + lane] = acc[v / (TPW * NB * 4)][(v / (NB * 4)) % TPW][(v / 4) % NB][v % 4];
        barrier.sync(KSPLIT * 32);
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

// One warp of a tile group: tiles [tile0, tile0 + TPW) of a segment with
// `tiles` tiles and `rows` valid rows, superblock share `part` of KSPLIT, for
// M <= 8 * NB activation rows (column block nb holds rows 8 nb .. 8 nb + 7).
// Each iteration decodes one superblock (4 k-blocks) while the codes and
// packed coefficients of the next are in flight in registers. `ready()`
// returns the activation source, which reads its A operands in place
// (`pair`); it runs once the first superblock's weight loads are issued, so
// it may wait for the previous launch (`seismic_dependency_wait`) while they
// are in flight. A group past the segment (tile0 >= tiles) loads nothing but
// still runs `ready()`, which may be block-collective. `reduce` is the
// group's reduction area, `barrier` its named barrier. With STAGING (the
// launch's shared `staging` area holds the group's tiles: `gemv_staging`)
// an unpaired staged weight (`GemvStaged`) reads its coefficients from the
// group's copy of its tiles' fields, issued with the first superblock's
// loads and complete before the first decode.
template <int TPW, int KSPLIT, int NB, bool STAGING, class Ready, class WA, class WB, class Epi, class Barrier>
__device__ __forceinline__ void gemv_mma_group(const Ready &ready, u32 M, u64 kblocks, u64 tile0, u64 tiles,
                                               u64 rows, const WA &wa, const WB &wb, const Epi &epi, float *reduce,
                                               const Barrier &barrier, u32 part, u8 *staging) {
    constexpr bool PAIR = !Same<WB, NoWeight>::value;
    constexpr bool STAGED = STAGING && !PAIR && GemvStaged<WA>::value;
    // Both weights read permuted, or neither: the activations follow one order.
    constexpr bool PERMUTED = GemvPermuted<WA>::value && (!PAIR || GemvPermuted<WB>::value);
    // The groups the streams take operand sums over: A's (or B's, where A has
    // none), and B's apart only where both are biased in groups that differ.
    constexpr int SUM_A = bias_group<WA>(), SUM_B = bias_group<WB>();
    constexpr int SUM_GROUP = SUM_A != 0 ? SUM_A : SUM_B != 0 ? SUM_B : 32;
    constexpr int OTHER_GROUP = SUM_B != 0 && SUM_B != SUM_GROUP ? SUM_B : SUM_GROUP;
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
            if constexpr (PERMUTED) {
                stage.ra[i] = wa.fetch_superblock_permuted(tile0 + i, sb, kblocks, lane);
                if constexpr (PAIR)
                    stage.rb[i] = wb.fetch_superblock_permuted(tile0 + i, sb, kblocks, lane);
            } else {
                stage.ra[i] = wa.fetch_superblock(tile0 + i, sb, kblocks, lane);
                if constexpr (PAIR)
                    stage.rb[i] = wb.fetch_superblock(tile0 + i, sb, kblocks, lane);
            }
            if constexpr (!STAGED)
                stage.ba[i] = wa.block(tile0 + i, sb, lane);
            if constexpr (PAIR)
                stage.bb[i] = wb.block(tile0 + i, sb, lane);
        }
    };
    // The staged coefficient fields of tile tile0 + i.
    const u64 staged_tile = gemv_staged_tile<WA>(kblocks);
    // The next superblock's loads are in flight while the current one
    // decodes; the first one's are issued before the source is ready.
    Stage current;
    if constexpr (STAGED) {
#pragma unroll
        for (int i = 0; i < TPW; ++i)
            if (tile0 + i < tiles)
                wa.stage(tile0 + i, kblocks, staging + i * staged_tile, part * 32 + lane, KSPLIT * 32);
    }
    if (tile0 < tiles && begin < end)
        load(current, begin);
    const auto pro = ready();
    static_assert(decltype(ready())::FACTORS == 0, "a GEMV reads its operands in place");
    if (tile0 >= tiles)
        return;
    if constexpr (STAGED) {
        seismic_cp_async_wait_all();
        barrier.sync(KSPLIT * 32);
    }
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
                    // The B fragment's columns 2t, 2t + 1 and 2t + 8, 2t + 9 of
                    // step s, or their permuted places (`GemvPermuted`).
                    const u64 low = PERMUTED ? 16 * t + 4 * s : 16 * s + 2 * t;
                    const u64 high = PERMUTED ? low + 2 : low + 8;
                    b[nb][s][0] = feeds ? pro.pair(factors, m, k0 + low) : 0u;
                    b[nb][s][1] = feeds ? pro.pair(factors, m, k0 + high) : 0u;
                }
            }
            GroupSums<NB, SUM_GROUP> sums;
            if constexpr (SUM_A != 0 || SUM_B != 0)
                group_sums(b, sums);
            GroupSums<NB, OTHER_GROUP> other;
            if constexpr (OTHER_GROUP != SUM_GROUP)
                group_sums(b, other);
#pragma unroll
            for (int i = 0; i < TPW; ++i)
                if (tile0 + i < tiles) {
                    if constexpr (STAGED)
                        gemv_accumulate<NB>(acc[0][i], wa.raw(stage.ra[i], q, lane), wa,
                                            wa.staged_block(staging + i * staged_tile, kblocks, sb, lane), q, b,
                                            sums);
                    else
                        gemv_accumulate<NB>(acc[0][i], wa.raw(stage.ra[i], q, lane), wa, stage.ba[i], q, b, sums);
                    if constexpr (PAIR) {
                        if constexpr (OTHER_GROUP != SUM_GROUP)
                            gemv_accumulate<NB>(acc[1][i], wb.raw(stage.rb[i], q, lane), wb, stage.bb[i], q, b, other);
                        else
                            gemv_accumulate<NB>(acc[1][i], wb.raw(stage.rb[i], q, lane), wb, stage.bb[i], q, b, sums);
                    }
                }
        }
    };
    for (u64 sb = begin; sb < end; ++sb) {
        Stage next;
        if (sb + 1 < end)
            load(next, sb + 1);
        compute(current, sb);
        current = next;
    }
    gemv_finish<TPW, KSPLIT, NB>(acc, M, tile0, tiles, rows, epi, reduce, barrier, part);
}

// One k-block of an entry-order GEMV (`W::GEMV_ENTRIES`) for one tile:
// rows g and g + 8 of the lane (`raw`, the k-block's entries of both), the
// k-block's scales (`c`), the M <= 8 NB activation rows as the N columns of
// NB column blocks. Each scale group's steps accumulate into a fresh MMA
// sum, which then takes the group's scale once per row (and, biased, the
// group's operand sum, one MMA against an all-ones operand per step, its
// bias); every column is its own activation row, so a row's bits do not
// depend on M.
template <int NB, class Pro, class W>
__device__ __forceinline__ void gemv_entries_kblock(float (&acc)[NB][4], const W &, const typename W::Raw &raw,
                                                    const Coefficients<W::GROUPS> &c, const Pro &pro, u32 M,
                                                    u64 kblock) {
    const u32 lane = threadIdx.x % 32;
    const u32 g = lane / 4, t = lane % 4;
    const float *factors = nullptr;
#pragma unroll
    for (int group = 0; group < W::GROUPS; ++group) {
        const int entry = W::gemv_entry(group, t);
        float d[NB][4], sums[NB][4];
#pragma unroll
        for (int nb = 0; nb < NB; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                d[nb][e] = sums[nb][e] = 0.0f;
#pragma unroll
        for (int step = 0; step < W::GROUP_STEPS; ++step) {
            const int h = W::gemv_half(step, t);
            u32 a[4];
            W::template entry_pairs<Op>(raw.g, entry, h, a[0], a[2]);
            W::template entry_pairs<Op>(raw.g8, entry, h, a[1], a[3]);
            const u64 k = kblock * 64 + 8 * entry + 4 * h;
#pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                const u32 m = 8 * nb + g;
                const u32 b[2] = {m < M ? pro.pair(factors, m, k) : 0u, m < M ? pro.pair(factors, m, k + 2) : 0u};
                Op::mma(d[nb], a, b);
                if constexpr (W::BIAS) {
                    const u32 ones[4] = {Op::ONES, Op::ONES, Op::ONES, Op::ONES};
                    Op::mma(sums[nb], ones, b);
                }
            }
        }
#pragma unroll
        for (int nb = 0; nb < NB; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[nb][e] = seismic_fma_rn(c.scale[e / 2][group], d[nb][e], acc[nb][e]);
                if constexpr (W::BIAS)
                    acc[nb][e] = seismic_fma_rn(-c.bias[e / 2][group], sums[nb][e], acc[nb][e]);
            }
    }
}

// One warp of a tile group of an entry-order weight (and its pair, when WB is
// one): tiles [tile0, tile0 + TPW) over superblock share `part` of KSPLIT.
// Each pass loads one superblock per tile and stream whole (rows g and g + 8
// of its k-blocks, and their coefficient words, every load in flight at
// once) and then decodes it; the next pass's loads follow the decode (on GB10
// a lane with more superblocks in flight, prefetched or batched, measured
// slower). The first superblock's loads are issued before `ready()`.
template <int TPW, int KSPLIT, int NB, class Ready, class WA, class WB, class Epi, class Barrier>
__device__ __forceinline__ void gemv_entries_group(const Ready &ready, u32 M, u64 kblocks, u64 tile0, u64 tiles,
                                                   u64 rows, const WA &wa, const WB &wb, const Epi &epi,
                                                   float *reduce, const Barrier &barrier, u32 part) {
    constexpr bool PAIR = !Same<WB, NoWeight>::value;
    const u32 lane = threadIdx.x % 32;
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
    struct Stage {
        typename WA::Raw ra[TPW][4];
        typename WB::Raw rb[TPW][4];
        typename WA::Block ba[TPW];
        typename WB::Block bb[TPW];
    };
    auto load = [&](Stage &stage, u64 sb) {
#pragma unroll
        for (int i = 0; i < TPW; ++i)
            if (tile0 + i < tiles) {
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    if (4 * sb + q < kblocks) {
                        stage.ra[i][q] = wa.fetch(tile0 + i, 4 * sb + q, lane);
                        if constexpr (PAIR)
                            stage.rb[i][q] = wb.fetch(tile0 + i, 4 * sb + q, lane);
                    }
                stage.ba[i] = wa.block(tile0 + i, sb, lane);
                if constexpr (PAIR)
                    stage.bb[i] = wb.block(tile0 + i, sb, lane);
            }
    };
    Stage stage;
    if (tile0 < tiles && begin < end)
        load(stage, begin);
    const auto pro = ready();
    static_assert(decltype(ready())::FACTORS == 0, "a GEMV reads its operands in place");
    if (tile0 >= tiles)
        return;
    for (u64 sb = begin; sb < end; ++sb) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const u64 kblock = 4 * sb + q;
#pragma unroll
            for (int i = 0; i < TPW; ++i) {
                if (kblock >= kblocks || tile0 + i >= tiles)
                    continue;
                Coefficients<WA::GROUPS> ca;
                wa.coefficients(stage.ba[i], q, ca);
                gemv_entries_kblock<NB>(acc[0][i], wa, stage.ra[i][q], ca, pro, M, kblock);
                if constexpr (PAIR) {
                    Coefficients<WB::GROUPS> cb;
                    wb.coefficients(stage.bb[i], q, cb);
                    gemv_entries_kblock<NB>(acc[1][i], wb, stage.rb[i][q], cb, pro, M, kblock);
                }
            }
        }
        if (sb + 1 < end)
            load(stage, sb + 1);
    }
    gemv_finish<TPW, KSPLIT, NB>(acc, M, tile0, tiles, rows, epi, reduce, barrier, part);
}

// One warp of a tile group of a records weight (`W::GEMV_RECORDS`; WB its
// pair of the same representation, or none): tiles [tile0, tile0 + TPW)
// over superblock share `part` of KSPLIT. A superblock of a tile is one
// record whose bytes the lanes load 16 at a time, contiguous over the warp;
// the next superblock's records are in flight while the current one decodes.
// As in `gemv_entries_kblock`, the k of a step is permuted within its scale
// group so that lane t's four A values of a row are half of its entry
// 4 group + t, and its B values are x at the same places: one 16-byte read
// of x covers a group's two steps. Each group's MMA sum takes its scale once
// per row, so a row's bits do not depend on M. The group's warps form the
// representation's decode table (`W::Table`) in shared memory while the first
// records load, and synchronize on the group's barrier.
template <int TPW, int KSPLIT, int NB, class Ready, class WA, class WB, class Epi, class Barrier>
__device__ __forceinline__ void gemv_records_group(const Ready &ready, u32 M, u64 kblocks, u64 tile0, u64 tiles,
                                                   u64 rows, const WA &wa, const WB &wb, const Epi &epi,
                                                   float *reduce, const Barrier &barrier, u32 part) {
    constexpr bool PAIR = !Same<WB, NoWeight>::value;
    static_assert(!PAIR || Same<WA, WB>::value, "paired records weights share one representation");
    __shared__ typename WA::Table tables[Barrier::GROUPS];
    typename WA::Table &table = tables[barrier.group];
    const u32 lane = threadIdx.x % 32;
    const u32 g = lane / 4, t = lane % 4;
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
    struct Stage {
        typename WA::Record ra[TPW];
        typename WA::Record rb[TPW];
    };
    auto load = [&](Stage &stage, u64 sb) {
#pragma unroll
        for (int i = 0; i < TPW; ++i)
            if (tile0 + i < tiles) {
                stage.ra[i] = wa.record(tile0 + i, sb, lane);
                if constexpr (PAIR)
                    stage.rb[i] = wb.record(tile0 + i, sb, lane);
            }
    };
    Stage stage;
    if (tile0 < tiles) {
        if (begin < end)
            load(stage, begin);
        WA::form_table(table, part * 32 + lane, KSPLIT * 32);
    }
    const auto pro = ready();
    static_assert(decltype(ready())::FACTORS == 0, "a GEMV reads its operands in place");
    if (tile0 >= tiles)
        return;
    barrier.sync(KSPLIT * 32);
    // One record of one tile and stream into its accumulators.
    auto accumulate = [&](float (&sum)[NB][4], const typename WA::Record &record, u64 sb) {
        const typename WA::Scales scale = WA::scales(record, t);
        // Records hold whole superblocks: all four k-blocks exist.
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const u64 kblock = 4 * sb + q;
#pragma unroll
            for (int group = 0; group < WA::GROUPS; ++group) {
                // Entry 4 group + t of rows g and g + 8: values 0..3 (step
                // 0) and 4..7 (step 1) of the lane's A pairs.
                u32 rg[4], rg8[4];
                WA::template operands<Op>(table, record.w[2 * q] >> (16 * group), rg);
                WA::template operands<Op>(table, record.w[2 * q + 1] >> (16 * group), rg8);
                const u64 k = kblock * 64 + 8 * (4 * group + t);
                float d[NB][4];
#pragma unroll
                for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        d[nb][e] = 0.0f;
                    const u32 m = 8 * nb + g;
                    const uint4 x = m < M ? pro.octet(factors, m, k) : make_uint4(0u, 0u, 0u, 0u);
                    const u32 a0[4] = {rg[0], rg8[0], rg[1], rg8[1]}, b0[2] = {x.x, x.y};
                    const u32 a1[4] = {rg[2], rg8[2], rg[3], rg8[3]}, b1[2] = {x.z, x.w};
                    Op::mma(d[nb], a0, b0);
                    Op::mma(d[nb], a1, b1);
                }
#pragma unroll
                for (int nb = 0; nb < NB; ++nb)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        sum[nb][e] = seismic_fma_rn(scale(e / 2, 2 * q + group), d[nb][e], sum[nb][e]);
            }
        }
    };
    for (u64 sb = begin; sb < end; ++sb) {
        const Stage current = stage;
        if (sb + 1 < end)
            load(stage, sb + 1);
#pragma unroll
        for (int i = 0; i < TPW; ++i)
            if (tile0 + i < tiles) {
                accumulate(acc[0][i], current.ra[i], sb);
                if constexpr (PAIR)
                    accumulate(acc[1][i], current.rb[i], sb);
            }
    }
    gemv_finish<TPW, KSPLIT, NB>(acc, M, tile0, tiles, rows, epi, reduce, barrier, part);
}

// One warp of a tile group: `gemv_records_group` for records weights,
// `gemv_entries_group` for entry-order weights, otherwise `gemv_mma_group`
// (`staging`: see there).
template <int TPW, int KSPLIT, int NB, bool STAGING = false, class Ready, class WA, class WB, class Epi,
          class Barrier>
__device__ __forceinline__ void gemv_group(const Ready &ready, u32 M, u64 kblocks, u64 tile0, u64 tiles, u64 rows,
                                           const WA &wa, const WB &wb, const Epi &epi, float *reduce,
                                           const Barrier &barrier, u32 part, u8 *staging = nullptr) {
    if constexpr (GemvRecords<WA>::value && (Same<WB, NoWeight>::value || Same<WA, WB>::value))
        gemv_records_group<TPW, KSPLIT, NB>(ready, M, kblocks, tile0, tiles, rows, wa, wb, epi, reduce, barrier,
                                            part);
    else if constexpr (GemvEntries<WA>::value && (Same<WB, NoWeight>::value || GemvEntries<WB>::value))
        gemv_entries_group<TPW, KSPLIT, NB>(ready, M, kblocks, tile0, tiles, rows, wa, wb, epi, reduce, barrier,
                                            part);
    else
        gemv_mma_group<TPW, KSPLIT, NB, STAGING>(ready, M, kblocks, tile0, tiles, rows, wa, wb, epi, reduce, barrier,
                                                 part, staging);
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

// Coefficient staging bytes of a GEMV tile group (`gemv_mma_group`) over
// weights WS... of `kblocks` k-blocks: the most any staged weight takes for
// the group's TPW tiles, or 0 when none is staged or the block's groups
// would take more than STAGING_BYTES with their reduction areas.
constexpr u64 STAGING_BYTES = 40960;
template <class Shape, class... WS> __host__ __device__ constexpr u64 gemv_staging(u64 kblocks) {
    u64 bytes = 0;
    ((bytes = gemv_staged_tile<WS>(kblocks) > bytes ? gemv_staged_tile<WS>(kblocks) : bytes), ...);
    bytes *= Shape::TPW;
    return Shape::GROUPS * (bytes + 4 * Shape::REDUCE) <= STAGING_BYTES ? bytes : 0;
}

// Shared memory of a GEMV block: the KSPLIT reduction of each tile group,
// and its coefficient staging (STAGING bytes, `gemv_staging`).
template <class Shape, class Pro, u64 STAGING = 0> struct GemvShared {
    float reduce[Shape::GROUPS][Shape::REDUCE];
    uint4 staging[Shape::GROUPS][STAGING > 0 ? STAGING / 16 : 1];
    static constexpr bool STAGED = STAGING > 0;
};

// Tile groups of a segment of `rows` rows.
template <class Shape> __device__ __forceinline__ u64 gemv_groups(u64 rows) {
    return ((rows + 15) / 16 + Shape::TPW - 1) / Shape::TPW;
}

// Blocks of a launch of `rows` rows (one segment, `gemv_groups` blocks of
// Shape::GROUPS tile groups).
template <class Shape> __host__ __device__ constexpr u64 gemv_grid(u64 rows) {
    return (((rows + 15) / 16 + Shape::TPW - 1) / Shape::TPW + Shape::GROUPS - 1) / Shape::GROUPS;
}

// How many blocks of a GEMV launch of weights WA (and WB) share an SM
// (`GroupBarrier`): a launch of GRID blocks (0 when the kernel does not state
// its grid) on a device of SEISMIC_MULTIPROCESSORS SMs.
//
// A streaming weight's warp keeps whole superblocks of 2 to 4 KiB per tile in
// flight: one block per SM already draws the device's memory rate, and the
// block holds its SM alone. Measured on GB10 (Nemotron 3.5 Bolt decode, 4096
// context, against these weights sharing their SMs): the q8 output head
// 2400 -> 2308 us, a q8 attention input projection 107 -> 92 us, an NVFP4
// state-space input projection 196 -> 193 us, a step 25.0 -> 24.6 ms.
//
// A records weight's warp (`GemvRecords`) keeps one 1 KiB record per tile in
// flight, so its launch needs its blocks resident together, up to four per
// SM. A launch that states its grid takes exactly the blocks per SM that
// hold it in one wave, ceil(GRID / SMs): fewer leave a second wave of a few
// blocks, each re-forming its source row, behind the first; more let the SMs
// that free first take more than their share, since a programmatic launch's
// blocks are placed as its predecessor's retire, each freed SM filling to
// its residency. The kernel declares the same residency as its launch bounds
// (`gemv_min_blocks`), so its registers admit it. Measured on GB10 (Bolt, 48
// SMs, the IQ2_XXS shared expert decode chain): the up (168 blocks) at 4
// rather than 3 per SM 33.3 -> 28.5 us; the down that follows (128 blocks)
// at 3 rather than 4 per SM 28.6 -> 25.4 us. A records launch that does not
// state its grid shares its SM as registers allow.
template <class Shape, class WA, class WB, u64 GRID> struct GemvResidency {
    static constexpr bool RECORDS =
        GemvRecords<WA>::value && (Same<WB, NoWeight>::value || GemvRecords<WB>::value);
    // The most blocks whose barrier ids fit an SM's 16 (barrier 0 is each
    // block's own), and the records design's four.
    static constexpr u64 MOST = 16 / (Shape::GROUPS + 1) < 4 ? 16 / (Shape::GROUPS + 1) : 4;
    __host__ __device__ static constexpr u64 wave() {
        const u64 per = (GRID + SEISMIC_MULTIPROCESSORS - 1) / SEISMIC_MULTIPROCESSORS;
        return per < 1 ? 1 : per > MOST ? MOST : per;
    }
    static constexpr int value = !RECORDS ? 1 : GRID == 0 ? 0 : (int)wave();
};

// The launch bounds' minimum blocks per SM of a GEMV kernel of that launch.
template <class Shape, class WA, class WB, u64 GRID> __host__ __device__ constexpr int gemv_min_blocks() {
    return GemvResidency<Shape, WA, WB, GRID>::value > 1 ? GemvResidency<Shape, WA, WB, GRID>::value : 1;
}

// Run a GEMV warp over one segment of a launch of GRID blocks (0: not
// stated, see `GemvResidency`): `group` is the segment-local tile group.
// `ready()` returns the activation source once the warp's first weight loads
// are issued (see `gemv_group`).
template <class Shape, u64 GRID = 0, class Ready, class WA, class WB, class Epi, class Shared>
__device__ __forceinline__ void gemv_segment_ready(Shared &shared, const Ready &ready, u32 M, u64 kblocks, u64 group,
                                                   u64 rows, const WA &wa, const WB &wb, const Epi &epi) {
    gemv_group<Shape::TPW, Shape::KSPLIT, Shape::NB, Shared::STAGED>(
        ready, M, kblocks, group * Shape::TPW, (rows + 15) / 16, rows, wa, wb, epi, shared.reduce[Shape::group()],
        GroupBarrier<Shape::GROUPS, GemvResidency<Shape, WA, WB, GRID>::value>{Shape::group()}, Shape::part(),
        reinterpret_cast<u8 *>(shared.staging[Shape::group()]));
}

template <class Shape, u64 GRID = 0, class Pro, class WA, class WB, class Epi, class Shared>
__device__ __forceinline__ void gemv_segment(Shared &shared, const Pro &pro, u32 M, u64 kblocks, u64 group,
                                             u64 rows, const WA &wa, const WB &wb, const Epi &epi) {
    gemv_segment_ready<Shape, GRID>(shared, [&] { return pro; }, M, kblocks, group, rows, wa, wb, epi);
}

// Ready callbacks of a programmatic dependent (SEISMIC_PROGRAMMATIC_DEPENDENCY):
// wait for the previous launch, let the next one start, and return the
// source: `after_dependency` for a source read in place,
// `source_after_dependency` for a `GemvSource` (block-collective at M = 1).
template <class Pro> __device__ __forceinline__ auto after_dependency(const Pro &pro) {
    return [pro] {
        seismic_dependency_start();
        return pro;
    };
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
// own rows' coefficients of every k-block into its area (per row per stream,
// a k-block's (scale, -bias) of each group: `AreaRow`).
// Bytes of one staged code chunk of W: 16, or `W::CHUNK_BYTES` (a
// representation whose k-block's bytes are 8-byte pieces of its storage).
template <class W, class = void> struct ChunkBytes {
    static constexpr int value = 16;
};
template <class W> struct ChunkBytes<W, decltype((void)W::CHUNK_BYTES)> {
    static constexpr int value = W::CHUNK_BYTES;
};

// Bytes of a row of a warp's coefficient area for W: the F32 scale (and
// bias) of each of a k-block's groups (`gemm_decode_coefficients`), at least
// 16 (the declared shared bytes' term).
template <class W> __host__ __device__ constexpr int area_row() {
    return W::GROUPS * (W::BIAS ? 2 : 1) * 4 > 16 ? W::GROUPS * (W::BIAS ? 2 : 1) * 4 : 16;
}

template <int BM_, int WM_, int WN_, int STAGES_, bool ORDERED_ = false, int ROTATE_ = 1, int SPAN_ = 1>
struct GemmShape {
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
    // K-blocks staged together (`gemm_segment`): a span's consecutive stages
    // are issued at once, its weight copies interleaved so that one warp copy
    // covers a row's codes of all SPAN k-blocks.
    static constexpr int SPAN = SPAN_;
    static_assert(SPAN == 1 || (ROTATE == 1 && STAGES % SPAN == 0 && STAGES / SPAN >= 2),
                  "a spanned pipeline rotates whole spans through at least two span slots");
    static constexpr int MI = BM / WM / 16; // m16 tiles per warp
    static constexpr int NJ = 8 / WN;       // weight tiles per warp
    static_assert(MI * 16 * WM == BM && NJ * WN == 8, "warp tiling covers the block tile");
    static_assert(STAGES >= 2 && STAGES <= 5, "a record buffer is reloaded only after its superblock's k-blocks");
    template <bool S8> static constexpr int A_PITCH = S8 ? 80 : 144;
    // Activation bytes per row per stage (s8 rows carry their groups).
    template <bool S8> static constexpr int A_ROW = S8 ? 96 : 144;
    template <class W> static constexpr int TILE = W::CHUNKS * ChunkBytes<W>::value;
    template <class W> static constexpr int RECORD = (W::SUPER_WORDS * 4 + 15) / 16 * 16;
    // The 16 B a row the declared shared bytes take for a warp's
    // coefficient area of one stream, and the area of each stream of WA and
    // WB: the larger of their rows' (`area_row`).
    static constexpr int COEF_WARP = NJ * 16 * 16;
    template <class WA, class WB> __host__ __device__ static constexpr int coef_area() {
        return NJ * 16 * (area_row<WA>() > area_row<WB>() ? area_row<WA>() : area_row<WB>());
    }
    template <class WA, class WB, bool S8>
    static constexpr int STAGE_BYTES = BM * A_ROW<S8> + 8 * (TILE<WA> + TILE<WB>);
    // One record buffer: both streams' 128 rows.
    template <class WA, class WB> static constexpr int RECORDS = 128 * (RECORD<WA> + RECORD<WB>);
    template <class WA, class WB, bool S8>
    static constexpr int LAYOUT_BYTES = STAGES * STAGE_BYTES<WA, WB, S8> + 2 * RECORDS<WA, WB> +
                                        (Same<WB, NoWeight>::value ? 1 : 2) * WARPS * coef_area<WA, WB>();
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
__device__ __forceinline__ void cp_async_8_zfill(void *shared, const void *global, u32 source_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(seismic_shared_address(shared)), "l"(global),
                 "r"(source_bytes)
                 : "memory");
}
// One staged chunk of CHUNK bytes (`ChunkBytes`), zero when not `valid`.
template <int CHUNK> __device__ __forceinline__ void cp_async_chunk(u8 *shared, const u8 *global, bool valid) {
    if constexpr (CHUNK == 16)
        seismic_cp_async_16_zfill(shared, global, valid ? 16u : 0u);
    else
        cp_async_8_zfill(shared, global, valid ? 8u : 0u);
}

// One stream's part of a stage: the code chunks (TILE bytes per tile) of tiles
// [tile0, tile0 + 8) (zero past `tiles`), by threads `first`, `first + STRIDE`, ...
template <class W, int STRIDE, int TILE>
__device__ __forceinline__ void gemm_load_codes(const W &w, u64 tile0, u64 tiles, u64 kblock, u8 *codes, u32 first) {
    constexpr int CHUNK = ChunkBytes<W>::value;
    for (int c = first; c < 8 * W::CHUNKS; c += STRIDE) {
        const u64 j = c / W::CHUNKS;
        const u32 q = c % W::CHUNKS;
        const bool valid = tile0 + j < tiles;
        cp_async_chunk<CHUNK>(codes + j * TILE + q * CHUNK, w.chunk_source(valid ? tile0 + j : tile0, kblock, q),
                              valid);
    }
}

// One stream's codes of k-blocks [kblock, kblock + count) (count <= SPAN)
// into the stages at codes, codes + STAGE, ...: chunk c of the span copies
// k-block c % SPAN, so the SPAN copies of one chunk position (a row's
// consecutive codes in the row-major packed layouts) are issued by adjacent
// lanes of one warp copy. Measured on GB10: one warp copy's 32 contiguous
// bytes per row (a q4 k-block) stream from memory at 108 GB/s, 64 bytes at
// 205 GB/s, 128 bytes at 231 GB/s, whatever the number of copies in flight.
template <class W, int STRIDE, int TILE, int SPAN, int STAGE>
__device__ __forceinline__ void gemm_load_codes_span(const W &w, u64 tile0, u64 tiles, u64 kblock, u64 count,
                                                     u8 *codes, u32 first) {
    for (int c = first; c < SPAN * 8 * W::CHUNKS; c += STRIDE) {
        const u64 d = c % SPAN;
        if (d >= count)
            continue;
        const int position = c / SPAN;
        const u64 j = position / W::CHUNKS;
        const u32 q = position % W::CHUNKS;
        const bool valid = tile0 + j < tiles;
        cp_async_chunk<ChunkBytes<W>::value>(codes + d * STAGE + j * TILE + q * ChunkBytes<W>::value,
                                             w.chunk_source(valid ? tile0 + j : tile0, kblock + d, q), valid);
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
template <class W, int NJ, int RECORD, int TILE>
__device__ __forceinline__ void gemm_decode_coefficients(const W &w, const u8 *records, const u8 *codes, u32 first,
                                                         u64 kblock, float *area) {
    if constexpr (W::DENSE)
        return;
    constexpr int ITEMS = NJ * 16 * W::GROUPS;
    constexpr int WORDS = W::BIAS ? 2 : 1;
    for (int item = threadIdx.x % 32; item < ITEMS; item += 32) {
        const u32 row = item / W::GROUPS;
        float scale, bias;
        w.staged_coefficient(reinterpret_cast<const u32 *>(records + (first + row) * RECORD),
                             codes + (first + row) / 16 * TILE, (first + row) % 16, kblock, item % W::GROUPS, scale,
                             bias);
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
template <class Operand = Op>
__device__ __forceinline__ u32 dequantize_pair(u32 codes, float2 coefficient) {
    const float2 pair = Operand::unpack(codes);
    return Operand::pack(seismic_fma_rn(pair.x, coefficient.x, coefficient.y),
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
template <class Shape, class Operands = SemanticGemmOperands, class W>
__device__ __forceinline__ void gemm_tile_half(float acc[Shape::MI][2][4], const W &w, const typename W::Raw &raw,
                                               const float *coefficients, int j_tile, int half,
                                               const u32 a[2][Shape::MI][4]) {
    using Operand = typename Operands::Operand;
    const u32 g = (threadIdx.x % 32) / 4;
    constexpr int STEPS = W::GROUP / 16;
    if constexpr (W::SCALED_GEMM && Operands::EXACT_CODES) {
        const u32 t = threadIdx.x % 4;
        const u32 ones[2] = {Operand::ONES, Operand::ONES};
        u32 r[2][4];
#pragma unroll
        for (int local = 0; local < 2; ++local)
            w.template decode<Operand>(raw, 2 * half + local, r[local]);
        // One partial product per coefficient group the half meets: both
        // steps of the half for groups of 32 codes or more, each step for
        // 16-code groups.
        constexpr int FOLD_STEPS = STEPS < 2 ? STEPS : 2;
        constexpr int FOLDS = 2 / FOLD_STEPS;
#pragma unroll
        for (int fold = 0; fold < FOLDS; ++fold) {
            const int group = (2 * half + fold * FOLD_STEPS) / STEPS;
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
                for (int local = fold * FOLD_STEPS; local < (fold + 1) * FOLD_STEPS; ++local) {
                    const u32 b0[2] = {r[local][0], r[local][2]};
                    const u32 b1[2] = {r[local][1], r[local][3]};
                    Operand::mma(p[0], a[local][mi], b0);
                    Operand::mma(p[1], a[local][mi], b1);
                    if constexpr (W::BIAS)
                        Operand::mma(s, a[local][mi], ones);
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
        w.template decode<Operand>(raw, step, r);
        // Dense weights decode to their operand values directly.
        if constexpr (!W::DENSE) {
            const float2 low = gemm_coefficient<W>(coefficients, j_tile * 16 + g, step / STEPS);
            const float2 high = gemm_coefficient<W>(coefficients, j_tile * 16 + g + 8, step / STEPS);
            r[0] = dequantize_pair<Operand>(r[0], low);
            r[2] = dequantize_pair<Operand>(r[2], low);
            r[1] = dequantize_pair<Operand>(r[1], high);
            r[3] = dequantize_pair<Operand>(r[3], high);
        }
        const u32 b0[2] = {r[0], r[2]};
        const u32 b1[2] = {r[1], r[3]};
#pragma unroll
        for (int mi = 0; mi < Shape::MI; ++mi) {
            Operand::mma(acc[mi][0], a[local][mi], b0);
            Operand::mma(acc[mi][1], a[local][mi], b1);
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
    static_assert(Shape::template LAYOUT_BYTES<WA, WB, S8> == AREAS + STREAMS * Shape::WARPS * Shape::template coef_area<WA, WB>(),
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
        return reinterpret_cast<float *>(shared + AREAS + (warp * STREAMS + stream) * Shape::template coef_area<WA, WB>());
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
    // The weights of the share's k-blocks [index, index + count) into stages
    // s, s + 1, ... (`gemm_load_codes_span`), and the records of each
    // superblock that starts among them (or of the share's first).
    template <int STRIDE, int SPAN> __device__ __forceinline__ void load_span(int s, u64 index, u64 count,
                                                                             u32 first) const {
        const u64 kb = kbegin + index;
        gemm_load_codes_span<WA, STRIDE, TILE_A, SPAN, STAGE>(wa, tile0, tiles, kb, count, stage(s) + ROWS, first);
        if constexpr (PAIR)
            gemm_load_codes_span<WB, STRIDE, TILE_B, SPAN, STAGE>(wb, tile0, tiles, kb, count,
                                                                  stage(s) + ROWS + 8 * TILE_A, first);
        for (u64 d = 0; d < count; ++d) {
            if (index + d != 0 && (kb + d) % 4 != 0)
                continue;
            u8 *buffer = shared + RECORDS + (kb + d) / 4 % 2 * BUFFER;
            gemm_load_records<WA, STRIDE, RECORD_A>(wa, tile0, tiles, (kb + d) / 4, ktotal, buffer, first);
            if constexpr (PAIR)
                gemm_load_records<WB, STRIDE, RECORD_B>(wb, tile0, tiles, (kb + d) / 4, ktotal,
                                                        buffer + 128 * RECORD_A, first);
        }
    }
    __device__ __forceinline__ void decode(int s, u64 index, u32 warp, u32 first) const {
        const u64 kb = kbegin + index;
        gemm_decode_coefficients<WA, Shape::NJ, RECORD_A, TILE_A>(wa, records(kb, 0), codes(s, 0, 0), first, kb,
                                                                  area(warp, 0));
        if constexpr (PAIR)
            gemm_decode_coefficients<WB, Shape::NJ, RECORD_B, TILE_B>(wb, records(kb, 1), codes(s, 1, 0), first, kb,
                                                                      area(warp, 1));
    }
};

// One warp's multiply of the share's staged k-block `index` (in stage
// `stage`) into its accumulators: the warp decodes its rows' coefficients,
// then per 32-code half multiplies the half's two k16 steps of activation
// fragments by its tiles' dequantized weight fragments.
template <class Shape, class Operands = SemanticGemmOperands, class Weights>
__device__ __forceinline__ void gemm_multiply(float (&acc)[2][Shape::NJ][Shape::MI][2][4], const Weights &weights,
                                              int stage, u64 index, u32 warp, u32 lane) {
    constexpr int PITCH = Shape::template A_PITCH<false>;
    constexpr int MI = Shape::MI;
    constexpr int NJ = Shape::NJ;
    const u32 wm = warp % Shape::WM;
    const u32 wn = warp / Shape::WM;
    const u8 *base = weights.stage(stage);
    const u64 kb = weights.kbegin + index;
    weights.decode(stage, index, warp, wn * NJ * 16);
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
            gemm_tile_half<Shape, Operands>(acc[0][j], weights.wa, raw_a[j], coefficients_a, j, half, a);
            if constexpr (Weights::PAIR)
                gemm_tile_half<Shape, Operands>(acc[1][j], weights.wb, raw_b[j], coefficients_b, j, half, a);
        }
    }
}

// D += A B over one 64-value k-block on FP4 tensor operations (`FP4_MMA`):
// `a` the A fragment of E2M1 codes (rows g, g + 8 of registers 0, 1 and 2,
// 3; values 8t .. 8t + 7 in registers 0 and 1, 32 + 8t .. in 2 and 3), `b`
// the B fragment (column g, values 8t .. 8t + 7, then 32 + 8t ..), `sa` the
// four UE4M3 scales of row g (lane t = 0) or g + 8 (t = 1), `sb` column g's
// (t = 0).
__device__ __forceinline__ void mma_fp4(float (&d)[4], const u32 (&a)[4], const u32 (&b)[2], u32 sa, u32 sb) {
#if PROJECTION_FP4_MMA
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32."
                 "ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
#endif
}

// acc[mi][h] += x . w over a staged k-block of NVFP4 weight tile `chunk`
// (rows row0 .. row0 + 15 of the block tile, their scale records at
// `records`) on FP4 tensor operations. The B fragment of 8-row half h is
// gathered from the tile's mma16 chunks: its register r holds values
// 8t .. 8t + 7 of 32r .. 32r + 31, half t % 2 of k16 step s = 2r + t / 2,
// whose byte tt is lane (g, tt)'s step-s word's nibbles 2 (t % 2) + h and
// 2 (t % 2) + h + 4 (registry `PackedRowLayout::code_bit`).
template <class Shape, int RECORD>
__device__ __forceinline__ void gemm_tile_fp4(float acc[Shape::MI][2][4], const u8 *chunk, const u8 *records, u32 row0,
                                              u64 kblock, const u32 (&a)[Shape::MI][4], const u32 (&sa)[Shape::MI]) {
    const u32 lane = threadIdx.x % 32;
    const u32 g = lane / 4, t = lane % 4;
    u32 words[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int tt = 0; tt < 4; ++tt)
            words[r][tt] = *reinterpret_cast<const u32 *>(chunk + (g * 4 + tt) * 16 + 4 * (2 * r + t / 2));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const u32 p = 4 * (2 * (t % 2) + h);
        u32 b[2] = {0u, 0u};
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int tt = 0; tt < 4; ++tt)
                b[r] |= (((words[r][tt] >> p) & 0xFu) | ((words[r][tt] >> (p + 12)) & 0xF0u)) << (8 * tt);
        const u32 sb = *reinterpret_cast<const u32 *>(records + (row0 + h * 8 + g) * RECORD + (kblock % 4) * 4);
#pragma unroll
        for (int mi = 0; mi < Shape::MI; ++mi)
            mma_fp4(acc[mi][h], a[mi], b, sa[mi], sb);
    }
}

// One warp's multiply of the share's staged k-block `index` (in stage
// `stage`) on FP4 tensor operations: the A fragments and row scales of its
// packed activation rows (`gemm_segment` stages a row's 32 code bytes and 4
// scale fields of a k-block at its 16-bit row's place), then each weight
// tile's halves (`gemm_tile_fp4`).
template <class Shape, class Weights>
__device__ __forceinline__ void gemm_multiply_fp4(float (&acc)[2][Shape::NJ][Shape::MI][2][4],
                                                  const Weights &weights, int stage, u64 index, u32 warp, u32 lane) {
    static_assert(packets::nvfp4_operands<decltype(weights.wa)>::value &&
                      (!Weights::PAIR || packets::nvfp4_operands<decltype(weights.wb)>::value),
                  "FP4 tensor operations multiply W4A4 NVFP4 weights");
    constexpr int PITCH = Shape::template A_PITCH<false>;
    constexpr int MI = Shape::MI;
    constexpr int NJ = Shape::NJ;
    const u32 wm = warp % Shape::WM;
    const u32 wn = warp / Shape::WM;
    const u32 g = lane / 4, t = lane % 4;
    const u8 *base = weights.stage(stage);
    const u64 kb = weights.kbegin + index;
    u32 a[MI][4], sa[MI];
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
        const u8 *row = base + (wm * (MI * 16) + mi * 16 + g) * PITCH;
        const u8 *row8 = row + 8 * PITCH;
        a[mi][0] = *reinterpret_cast<const u32 *>(row + 4 * t);
        a[mi][1] = *reinterpret_cast<const u32 *>(row8 + 4 * t);
        a[mi][2] = *reinterpret_cast<const u32 *>(row + 16 + 4 * t);
        a[mi][3] = *reinterpret_cast<const u32 *>(row8 + 16 + 4 * t);
        sa[mi] = *reinterpret_cast<const u32 *>((t == 1 ? row8 : row) + 32);
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const u32 row0 = (wn * NJ + j) * 16;
        gemm_tile_fp4<Shape, Weights::RECORD_A>(acc[0][j], weights.codes(stage, 0, wn * NJ + j), weights.records(kb, 0),
                                                row0, kb, a, sa);
        if constexpr (Weights::PAIR)
            gemm_tile_fp4<Shape, Weights::RECORD_B>(acc[1][j], weights.codes(stage, 1, wn * NJ + j),
                                                    weights.records(kb, 1), row0, kb, a, sa);
    }
}

// An epilogue over the sums of packed NVFP4 operand rows: each row's sums
// times its global scale (exact, a power of two), then `epi`.
template <class Rows, class Epi> struct RowGlobal {
    Rows rows;
    u64 K;
    Epi epi;
    __device__ __forceinline__ float global(u32 m) const { return PackedRows::global(rows.row(m), K); }
    __device__ __forceinline__ void operator()(u32 m, u64 n, float a, float b) const {
        const float g = global(m);
        epi(m, n, __fmul_rn(a, g), __fmul_rn(b, g));
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float a0, float a1, float b0, float b1) const {
        const float g = global(m);
        epilogue_pair(epi, m, n, __fmul_rn(a0, g), __fmul_rn(a1, g), __fmul_rn(b0, g), __fmul_rn(b1, g), 0);
    }
};

// One GEMM block over segment-local block column `nblock` (tiles
// [8 * nblock, 8 * nblock + 8)), activation rows [BM * gemm_band(), ...) of
// `source` (the A-typed input [M, K], staged or in place) and K share `part`
// of `parts` (split-K: the epilogue is then a `PartialStore` and
// `split_finalize` applies the real one). Rows at or past M read zeros; a warp
// whose rows all lie there decodes and multiplies nothing.
template <class Shape, class Operands = SemanticGemmOperands, class Rows, class WA, class WB, class Epi>
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
    // Packed NVFP4 operand rows (`Fp4GemmOperands`) stage a row's 32 code
    // bytes of a k-block (two pieces) and its 4 scale fields (a third) at the
    // 16-bit row's place; a loader's pieces too keep their rows.
    constexpr int PIECES = Operands::FP4 ? (Shape::BM * 3 + LOADERS - 1) / LOADERS : 1;
    const u8 *piece[PIECES];
    int place[PIECES];
    bool present[PIECES];
#pragma unroll
    for (int i = 0; i < PIECES; ++i) {
        const int c = (int)loader + i * LOADERS;
        const u64 row = m0 + c / 3;
        present[i] = Operands::FP4 && c < Shape::BM * 3 && row < M;
        piece[i] = source.row(present[i] ? row : 0) + (c % 3 < 2 ? 16 * (c % 3) : PackedRows::scales(K));
        place[i] = (c / 3) * PITCH + 16 * (c % 3);
    }
    // Stage k-block `kb`'s activation rows into stage `base`.
    auto load_rows = [&](u8 *base, u64 kb) {
        if constexpr (Operands::FP4) {
#pragma unroll
            for (int i = 0; i < PIECES; ++i) {
                const int c = (int)loader + i * LOADERS;
                if (c >= Shape::BM * 3)
                    continue;
                if (c % 3 < 2)
                    seismic_cp_async_16_zfill(base + place[i], piece[i] + kb * 32, present[i] ? 16u : 0u);
                else
                    cp_async_4_zfill(base + place[i], piece[i] + kb * 4, present[i] ? 4u : 0u);
            }
        } else {
#pragma unroll
            for (int i = 0; i < CHUNKS; ++i) {
                const int c = (int)loader + i * LOADERS;
                seismic_cp_async_16_zfill(base + (c / 8) * PITCH + (c % 8) * 16, chunk[i] + kb * 128,
                                          valid[i] ? 16u : 0u);
            }
        }
    };
    auto multiply = [&](float (&acc)[2][NJ][MI][2][4], int stage, u64 index) {
        if constexpr (Operands::FP4)
            gemm_multiply_fp4<Shape>(acc, weights, stage, index, warp, lane);
        else
            gemm_multiply<Shape, Operands>(acc, weights, stage, index, warp, lane);
    };
    // `index` counts k-blocks of this share.
    auto load = [&](int stage, u64 index) {
        if (threadIdx.x / LOADERS != index % Shape::ROTATE)
            return;
        load_rows(weights.stage(stage), kbegin + index);
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

    if constexpr (Shape::SPAN == 1) {
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
                multiply(acc, (int)(kb % Shape::STAGES), kb);
        }
    } else {
        // Spans of SPAN k-blocks rotate through STAGES / SPAN span slots: the
        // span `slots - 1` ahead is issued while one span is multiplied.
        constexpr int SPAN = Shape::SPAN;
        constexpr int SLOTS = Shape::STAGES / SPAN;
        const u64 spans = (kblocks + SPAN - 1) / SPAN;
        auto load_span = [&](int slot, u64 span) {
            const u64 index = span * SPAN;
            const u64 count = min((u64)SPAN, kblocks - index);
            for (u64 d = 0; d < count; ++d)
                load_rows(weights.stage(slot * SPAN + (int)d), kbegin + index + d);
            weights.template load_span<LOADERS, SPAN>(slot * SPAN, index, count, loader);
        };
#pragma unroll
        for (int s = 0; s < SLOTS - 1; ++s) {
            if ((u64)s < spans)
                load_span(s, s);
            seismic_cp_async_commit();
        }
        for (u64 span = 0; span < spans; ++span) {
            seismic_cp_async_wait<SLOTS - 2>();
            __syncthreads();
            const u64 next = span + SLOTS - 1;
            if (next < spans)
                load_span((int)(next % SLOTS), next);
            seismic_cp_async_commit();
            if (active)
#pragma unroll
                for (int d = 0; d < SPAN; ++d) {
                    const u64 kb = span * SPAN + d;
                    if (kb < kblocks)
                        multiply(acc, (int)(span % SLOTS) * SPAN + d, kb);
                }
        }
    }
    seismic_cp_async_wait<0>();
    if constexpr (Operands::FP4)
        gemm_publish(acc, m0 + wm * (MI * 16), tile0 + wn * NJ, M, rows, RowGlobal<Rows, Epi>{source, K, epi});
    else
        gemm_publish(acc, m0 + wm * (MI * 16), tile0 + wn * NJ, M, rows, epi);
}

// The INT8 candidate's GEMM: the same block tiling and pipeline over s8
// activations staged by `stage_row_s8` (BM rows x 64 bytes per k-block, pitch
// 80, plus each row's two (d, d * sum q) groups) and s8 weight fragments
// (packets.cuh virtual order). Per 32-code group one m16n8k32 s8 MMA into an
// exact int32 product (two, half-masked, for 16-code groups), folded as
// acc += scale * (d_x * P) - bias * (d_x * sum q_x); a biased 16-code group's
// sum q_x is its own, the exact int32 product of the same half-masked MMA
// against all-ones weights.
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
    // A biased 16-code group's bias term: d_x times the exact sum of its 16
    // activation codes (every column of the half-masked s8 MMA against
    // all-ones weights), where a group of 32 codes or more takes the staged
    // d_x sum q_x.
    auto fold_bias16 = [&](int group, int h, int part) {
        const u32 ones[2] = {part == 0 ? 0x01010101u : 0u, part == 1 ? 0x01010101u : 0u};
#pragma unroll
        for (int mi = 0; mi < Shape::MI; ++mi) {
            int q[4] = {0, 0, 0, 0};
            seismic_mma_m16n8k32_s8(q, a[h][mi], ones);
#pragma unroll
            for (int half = 0; half < 2; ++half)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const float2 k = gemm_coefficient<W>(coefficients, j_tile * 16 + half * 8 + 2 * t + c, group);
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        float &value = acc[mi][half][2 * r + c];
                        value = seismic_fma_rn(k.y, x[h][mi][r].x * gemm_s8_product(q[2 * r]), value);
                    }
                }
        }
    };
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
                        if constexpr (W::BIAS && W::GROUP >= 32)
                            value = seismic_fma_rn(k.y, x[h][mi][r].y, value);
                    }
            }
    };
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const S8Pair first = w.s8(raw, 2 * h);
        const S8Pair second = w.s8(raw, 2 * h + 1);
        if constexpr (W::GROUP >= 32) {
            // Half h's 32 codes, of group 32 h / GROUP.
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
            fold(p, 32 * h / W::GROUP, h);
        } else {
            static_assert(W::GROUP == 16, "groups of 16 codes, or of 32 or more");
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
                if constexpr (W::BIAS)
                    fold_bias16(2 * h + part, h, part);
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
        weights.decode(stage, kb, warp, wn * NJ * 16);
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

// Operand eligibility is independent of row scheduling. GEMV reads semantic
// Act; either matrix tile band can consume exact semantic operands, explicitly
// admitted S8 activations, or a separately admitted materialized F16 operand
// view. The common producer owns the selected representation and its group
// metadata. S8 multiplies packed codes, so dense weights retain the semantic
// path rather than silently quantizing their activations, and weights that
// take NVFP4 operand rows keep the rows their recipe defines.
template <class... W>
constexpr bool quantizable = ((!W::DENSE && !packets::nvfp4_operands<W>::value) && ...);

// A selected materialized operand view owns its storage even when its type
// equals semantic Act. Its producer must write every row; in-place eligibility
// of the semantic source cannot turn that write into a no-op.
// FP4 tensor operations' views are packed rows at the 16-bit rows' pitch.
template <class Operands, class Pro>
__device__ __forceinline__ void materialize_operand_row(const Pro &pro, u32 m, u64 K, u8 *staged) {
    if constexpr (Operands::FP4)
        form_packed_row(pro, m, K, staged + m * K * 2);
    else
        form_row<typename Operands::Storage>(pro, m, K, staged);
}

// Row m of the K-wide rows of element E at `rows` as NVFP4 operands in place
// (a product the next GEMM takes as `Operands`), by THREADS threads: packed
// for FP4 tensor operations, else as `Operands::Storage` values, each
// 16-value block read by the thread that rewrites it.
template <class Operands, class E, u64 K, int THREADS>
__device__ __forceinline__ void form_operands_in_place(u8 *rows, u32 m) {
    using Raw = Plain<E, AllRows>;
    const Nvfp4Rows<Raw> pro{Raw{rows, K, AllRows{}}};
    if constexpr (Operands::FP4)
        form_packed_row<(K / 16 + THREADS - 1) / THREADS>(pro, m, K, rows + m * K * 2);
    else
        form_row<typename Operands::Storage>(pro, m, K, rows);
}

// One producer contract for every row schedule. GEMV retains semantic Act;
// either matrix tile band reads the selected representation. Plain semantic
// sources remain in place, while an owned physical view is always produced.
template <bool S8, class Operands, class Pro>
__device__ __forceinline__ void stage_projection_row(const Pro &pro, u32 m, u32 rows, u64 K,
                                                     u8 *staged, void *groups) {
    static_assert(!S8 || Same<Operands, SemanticGemmOperands>::value,
                  "activation quantization and coefficient conversion are distinct policies");
    if (rows <= GEMV_ROWS) {
        if constexpr (Pro::STAGED)
            form_row(pro, m, K, staged);
    } else if constexpr (S8) {
        stage_row_s8(pro, m, K, staged, reinterpret_cast<float2 *>(groups));
    } else if constexpr (Pro::STAGED || Operands::MATERIALIZE) {
        materialize_operand_row<Operands>(pro, m, K, staged);
    }
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
template <class Pro>
__device__ __forceinline__ auto source_after_dependency(const Pro &pro, u8 *row, u32 M, u64 K, const u8 *staged) {
    return [=] {
        seismic_dependency_start();
        return GemvSource<Pro>::make(pro, row, M, K, staged);
    };
}

// One GEMM block on either path: `act` is the A rows (row stride
// `act_stride`) for the 16-bit path or the staged s8 rows (with their
// `groups`) for S8.
template <class Shape, bool S8, class Operands = SemanticGemmOperands, class WA, class WB, class Epi>
__device__ __forceinline__ void gemm_run(u8 *shared, const u8 *act, u64 act_stride, const void *groups, u32 M,
                                         u64 K, u64 nblock, u64 rows, const WA &wa, const WB &wb, const Epi &epi,
                                         u64 part = 0, u64 parts = 1) {
    static_assert(!S8 || Same<Operands, SemanticGemmOperands>::value,
                  "quantized activation GEMM requires the semantic operand policy");
    if constexpr (S8)
        gemm_segment_s8<Shape>(shared, QuantizedRows{act, K, reinterpret_cast<const float2 *>(groups)}, M, K, nblock,
                               rows, wa, wb, epi, part, parts);
    else
        gemm_segment<Shape, Operands>(shared, ActivationRows{act, act_stride}, M, K, nblock, rows, wa, wb, epi, part, parts);
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
template <class Shape, bool S8, class Operands = SemanticGemmOperands, class W, class Epi>
__device__ __forceinline__ void gemm_run_split(u8 *shared, const u8 *act, u64 act_stride, const void *groups, u32 M,
                                               u64 K, u64 nblock, u64 rows, const W &w, const Epi &epi,
                                               float *partials) {
    if (gridDim.z > 1)
        gemm_run<Shape, S8, Operands>(shared, act, act_stride, groups, M, K, nblock, rows, w, NoWeight{},
                            PartialStore<1>{partials, M, rows, 0, blockIdx.z}, blockIdx.z, gridDim.z);
    else
        gemm_run<Shape, S8, Operands>(shared, act, act_stride, groups, M, K, nblock, rows, w, NoWeight{}, epi);
}

} // namespace projection
