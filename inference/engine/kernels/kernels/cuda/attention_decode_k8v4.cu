#define KERNEL_A SEISMIC_ELEMENT_A
#define ATTENTION_AFFINE_HISTORY 1
// Sink logits (SK): the merge adds each head's sink to its merged denominator
// (`attention::normalizer`).
#define ATTENTION_SINKS SEISMIC_DIM_SK
#define ATTENTION_SINKS_BUFFER reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SINKS))
#define ATTENTION_SINK_AT(query_head) (static_cast<attention::u64>(query_head) * SEISMIC_SINKS_STRIDE_1)
#include "lib/attention/inputs.cuh"
#define ATTENTION_QUERY_GROUP SEISMIC_DIM_G
#define ATTENTION_INTERLEAVED SEISMIC_DIM_I
#define ATTENTION_SEPARATE SEISMIC_DIM_U
// attention_decode_k8v4 (M <= 8): `attention_decode`'s split-KV decode over
// affine K8/V4 history.
//
// L1 `attention_decode_k8v4_partial`, one block per (kv head, partition, row),
// partitioned and sliced over the query group exactly as the dense entry. Keys
// score as the sum over groups of scale * (q . code) + zero * sum(q) and
// values accumulate (p * scale) * code plus the bias sum(p * zero) of each
// lane's group, carried through the online softmax by the same factor as the
// accumulator (`absorb`), so no element is decoded. History rows are read as
// stored; a fresh row is prepared and quantized in registers exactly as the
// append encodes it (`AffineHistory::quantize`), so it is absorbed as history
// holds it. Its batches of TOKENS keys sit at fixed positions of the row's
// own key sequence (`attention::KeyCursor`), so the order of a row's sums does
// not depend on which of its keys are still fresh. Partition 0 of a
// layer with fresh rows appends the row's key and value encoded. L2
// `attention_decode_k8v4_merge`: the dense entry's merge and gate.
//
// Grouped-query matrix form (MATRIX = 1): `attention_decode`'s, over affine
// history: producer warps decode code tiles into F16 operands beside the MMA
// warps (`attention::prefill::attend`).

// Both forms' shared code (an entry's includes expand once, whatever the
// branch).
#include "lib/attention/attention.cuh"
#include "lib/attention/decode_merge.cuh"

#if SEISMIC_TUNE_MATRIX

#define ATTENTION_STAGES SEISMIC_TUNE_STAGES
#define ATTENTION_COLUMNS SEISMIC_TUNE_COLUMNS
#define ATTENTION_Q_REGISTERS SEISMIC_TUNE_QREG
#define ATTENTION_PRODUCER_WARPS (SEISMIC_TUNE_WARPS * SEISMIC_TUNE_COLUMNS)
#include "lib/attention/prefill.cuh"

extern "C" __global__ void __launch_bounds__(attention::prefill::MMA_THREADS + attention::prefill::PRODUCERS)
    attention_decode_k8v4_partial(SEISMIC_KERNEL_PARAMS) {
    // Let a programmatic dependent start (its first weight loads overlap this
    // launch; it still waits for this launch's completion).
    seismic_dependents_launch();
    attention::prefill::attend(
        ATTENTION_INPUTS(),
        attention::AffineHistory{
            reinterpret_cast<const attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY_CODES)),
            SEISMIC_OUT_HISTORY_KEY_COEFFICIENTS,
            reinterpret_cast<const attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE_CODES)),
            SEISMIC_OUT_HISTORY_VALUE_COEFFICIENTS,
            SEISMIC_HISTORY_KEY_CODES_STRIDE_0, SEISMIC_HISTORY_KEY_CODES_STRIDE_1,
            SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_0, SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_1,
            SEISMIC_HISTORY_VALUE_CODES_STRIDE_0, SEISMIC_HISTORY_VALUE_CODES_STRIDE_1,
            SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_0, SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_1,
            SEISMIC_PARAM_SLAB_ROWS},
        attention::prefill::DecodeRows<SEISMIC_TUNE_PARTS>{
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS))},
        attention::prefill::Block{static_cast<int>(blockIdx.z), static_cast<int>(blockIdx.x),
                                static_cast<int>(blockIdx.y), SEISMIC_TUNE_PARTS});
}

#else

namespace {

using attention::Act;
using attention::DPL;
using attention::G;
using attention::KV;
using attention::W;
using attention::u32;
using attention::u64;

typedef attention::LaneCodes<attention::KEY_BITS> KeyCodes;
typedef attention::LaneCodes<attention::VALUE_BITS> ValueCodes;

constexpr int WARPS = SEISMIC_TUNE_WARPS;
constexpr int PARTS = SEISMIC_TUNE_PARTS;
// The G query heads split into SLICES slices of H heads, as in
// `attention_decode`: warp w holds slice w % SLICES over key group w / SLICES.
constexpr int SLICES = SEISMIC_TUNE_SLICES;
constexpr int H = G / SLICES;
constexpr int GROUPS = WARPS / SLICES;
// History keys a warp keeps in flight: affine rows are 2.6x smaller than
// dense ones, so twice the dense batch keeps as many bytes in flight (fewer
// for heads wider than 256, whose lanes hold 16 dimensions per query head).
constexpr int TOKENS = DPL >= 16 ? 2 : (H >= 8 ? 4 : 8);
typedef attention::Heads<H> State;

// Absorb `count` (1..N) affine-coded keys and values, with this lane's group
// pairs `kc` and `vc`, into `state` and this lane's per-head `bias` (exp2
// domain; `q` scaled into it, `qsum` the sums of this lane's dimensions).
template <int N>
__device__ __forceinline__ void absorb(State &state, float (&bias)[H],
                                       const float (&q)[H][DPL], const float (&qsum)[H],
                                       const u32 (&k)[N][KeyCodes::words],
                                       const float2 (&kc)[N],
                                       const u32 (&v)[N][ValueCodes::words],
                                       const float2 (&vc)[N], int count, int lane) {
    float score[N][H];
#pragma unroll
    for (int t = 0; t < N; ++t) {
        float key[DPL];
#pragma unroll
        for (int d = 0; d < DPL; ++d) key[d] = KeyCodes::code(k[t], d);
#pragma unroll
        for (int h = 0; h < H; ++h) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < DPL; ++d) dot = __fmaf_rn(q[h][d], key[d], dot);
            score[t][h] = __fmaf_rn(kc[t].x, dot, kc[t].y * qsum[h]);
        }
    }
    attention::score_sums(score, lane);
    float value[N][DPL];
#pragma unroll
    for (int t = 0; t < N; ++t)
#pragma unroll
        for (int d = 0; d < DPL; ++d) value[t][d] = ValueCodes::code(v[t], d);
#pragma unroll
    for (int h = 0; h < H; ++h) {
        float maximum = state.maximum[h];
#pragma unroll
        for (int t = 0; t < N; ++t)
            if (t < count) maximum = fmaxf(maximum, score[t][h]);
        // exp2(-inf) = 0 for the first keys of an empty state.
        const float carry = seismic_ex2_approx(state.maximum[h] - maximum);
        float denominator = state.denominator[h] * carry;
        float offset = bias[h] * carry;
#pragma unroll
        for (int d = 0; d < DPL; ++d) state.accumulator[h][d] *= carry;
#pragma unroll
        for (int t = 0; t < N; ++t) {
            if (t < count) {
                const float p = seismic_ex2_approx(score[t][h] - maximum);
                denominator += p;
                offset = __fmaf_rn(p, vc[t].y, offset);
                const float weight = p * vc[t].x;
#pragma unroll
                for (int d = 0; d < DPL; ++d)
                    state.accumulator[h][d] =
                        __fmaf_rn(weight, value[t][d], state.accumulator[h][d]);
            }
        }
        state.denominator[h] = denominator;
        state.maximum[h] = maximum;
        bias[h] = offset;
    }
}

// Fold the per-head bias into the accumulator (it then carries the whole
// attended sum), for the publication.
__device__ __forceinline__ void fold(State &state, float (&bias)[H]) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
#pragma unroll
        for (int d = 0; d < DPL; ++d) state.accumulator[h][d] += bias[h];
        bias[h] = 0.0f;
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(WARPS * 32)
    attention_decode_k8v4_partial(SEISMIC_KERNEL_PARAMS) {
    // Let a programmatic dependent start (its first weight loads overlap this
    // launch; it still waits for this launch's completion).
    seismic_dependents_launch();
    const attention::Inputs in = ATTENTION_INPUTS();
    const attention::AffineHistory history{
        reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY_CODES)),
        SEISMIC_OUT_HISTORY_KEY_COEFFICIENTS,
        reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE_CODES)),
        SEISMIC_OUT_HISTORY_VALUE_COEFFICIENTS,
        SEISMIC_HISTORY_KEY_CODES_STRIDE_0, SEISMIC_HISTORY_KEY_CODES_STRIDE_1,
        SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_0, SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_1,
        SEISMIC_HISTORY_VALUE_CODES_STRIDE_0, SEISMIC_HISTORY_VALUE_CODES_STRIDE_1,
        SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_0, SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_1,
        SEISMIC_PARAM_SLAB_ROWS};
    float *partials = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    float *statistics = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS));
    const int kv = blockIdx.x;
    const int part = blockIdx.y;
    const u64 row = blockIdx.z;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    extern __shared__ float shared[];
    float *queries = shared;                  // [G][W]
    float *exchange = queries + G * W;        // [WARPS][W]
    float *warp_stats = exchange + WARPS * W; // [WARPS][H][2]
    float *own = exchange + warp * W;

    const float query_scale = in.scale * attention::LOG2E;
    for (int h = warp; h < G; h += WARPS) {
        float x[DPL];
        attention::prepared_query(in, row, kv * G + h, x, own, lane);
#pragma unroll
        for (int d = 0; d < DPL; ++d) queries[h * W + lane * DPL + d] = x[d] * query_scale;
    }
    if (attention::FRESH && part == 0 && warp == WARPS - 1) {
        float k[DPL];
        attention::prepared_key(in, row, kv, k, own, lane);
        attention::append(in, history, row, kv, k, lane);
    }
    __syncthreads();

    const int head0 = (warp % SLICES) * H;
    float q[H][DPL];
    float qsum[H];
#pragma unroll
    for (int h = 0; h < H; ++h) {
        element::f32_span(queries + (head0 + h) * W + lane * DPL, q[h]);
        float sum = 0.0f;
#pragma unroll
        for (int d = 0; d < DPL; ++d) sum += q[h][d];
        qsum[h] = sum;
    }
    // This lane's (scale, zero) pair within a vector's pairs.
    const int pair = lane / attention::PAIR_LANES;

    // Equal partitions of the row's keys, then equal key-group ranges of this
    // one.
    const u64 spans = SEISMIC_DIM_R;
    const long long total = attention::visible_total(in, row, spans, lane);
    const attention::Range partition =
        attention::partition(attention::Range{0, total}, attention::partition_run(total, 1, PARTS), part);
    const attention::Range slice = attention::partition(
        partition, (partition.hi - partition.lo + GROUPS - 1) / GROUPS, warp / SLICES);

    State state;
    attention::clear(state);
    float bias[H];
#pragma unroll
    for (int h = 0; h < H; ++h) bias[h] = 0.0f;

    // The slice's keys absorb in batches of TOKENS at fixed positions
    // slice.lo + i * TOKENS of the row's own key sequence, read through its
    // cursor (`attention::KeyCursor`), as the dense entry's: a key's batch,
    // and so the row's reduction order, depends on its position alone.
    const attention::AffineReads reads = history.reads();
    attention::KeyCursor cursor = attention::KeyCursor::at(in, row, spans, reads.slab_rows, slice.lo, lane);
    // The cursor's slab's first vectors.
    attention::AffineReads::Vectors slab{};
    // A history row's lane share of codes and pairs, read as stored.
    auto read_history = [&](const attention::AffineReads::Vectors &at, u32 (&k)[KeyCodes::words],
                            u32 (&v)[ValueCodes::words], float2 &kc, float2 &vc) {
        KeyCodes::load(at.key_codes, lane, k);
        ValueCodes::load(at.value_codes, lane, v);
        kc = attention::coefficients(at.key_pairs[pair]);
        vc = attention::coefficients(at.value_pairs[pair]);
    };
    for (long long position = slice.lo; position < slice.hi; position += TOKENS) {
        const int count = static_cast<int>(min(static_cast<long long>(TOKENS), slice.hi - position));
        u32 k[TOKENS][KeyCodes::words];
        u32 v[TOKENS][ValueCodes::words];
        float2 kc[TOKENS];
        float2 vc[TOKENS];
        // The batch's first key; a batch of TOKENS keys inside one history run
        // (a span's part of one slab) reads straight from the run's rows,
        // each key a row stride past the first.
        bool moved;
        const int first = cursor.seek(position, moved);
        if (cursor.historical() && moved) slab = reads.slab_vectors(cursor.slab, kv);
        if (cursor.historical() && count == TOKENS && position + TOKENS <= cursor.run_end) {
            const attention::AffineReads::Vectors head = slab.at(reads, static_cast<u32>(first) - cursor.slab_lo);
#pragma unroll
            for (int t = 0; t < TOKENS; ++t) read_history(head.at(reads, t), k[t], v[t], kc[t], vc[t]);
        } else {
#pragma unroll
            for (int t = 0; t < TOKENS; ++t) {
                // Past the slice, zero codes and pairs: `absorb` skips them.
                if (t >= count) {
#pragma unroll
                    for (int j = 0; j < KeyCodes::words; ++j) k[t][j] = 0u;
#pragma unroll
                    for (int j = 0; j < ValueCodes::words; ++j) v[t][j] = 0u;
                    kc[t] = vc[t] = make_float2(0.0f, 0.0f);
                    continue;
                }
                const int token = cursor.seek(position + t, moved);
                if (cursor.historical()) {
                    // A history row, read as stored.
                    if (moved) slab = reads.slab_vectors(cursor.slab, kv);
                    read_history(slab.at(reads, static_cast<u32>(token) - cursor.slab_lo), k[t], v[t], kc[t],
                                 vc[t]);
                } else {
                    // A fresh row, encoded in registers as the append encodes it.
                    float key[DPL];
                    float value[DPL];
                    attention::prepared_key(in, token, kv, key, own, lane);
                    attention::fresh_value(in, token, kv, value, lane);
                    const attention::AffineHistory::Encoding e = attention::AffineHistory::quantize(key, value);
#pragma unroll
                    for (int j = 0; j < KeyCodes::words; ++j) k[t][j] = e.key.codes[j];
#pragma unroll
                    for (int j = 0; j < ValueCodes::words; ++j) v[t][j] = e.value.codes[j];
                    kc[t] = make_float2(e.key.scale, e.key.zero);
                    vc[t] = make_float2(e.value.scale, e.value.zero);
                }
            }
        }
        absorb(state, bias, q, qsum, k, kc, v, vc, count, lane);
    }
    fold(state, bias);
    attention::publish<WARPS, PARTS, SLICES>(state, exchange, warp_stats, partials, statistics, row,
                                             kv, part, warp, lane);
}

#endif

extern "C" __global__ void attention_decode_k8v4_merge(SEISMIC_KERNEL_PARAMS) {
    // Let a programmatic dependent start (its first weight loads overlap this
    // launch; it still waits for this launch's completion).
    seismic_dependents_launch();
    attention::decode_gate<SEISMIC_TUNE_PARTS>(
        ATTENTION_INPUTS(),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS)),
        SEISMIC_RESULT_0_OUT, blockIdx.x, blockIdx.y, threadIdx.x);
}
