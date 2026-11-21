// Sink logits (SK): the merge adds each head's sink to its merged denominator
// (`attention::normalizer`).
#define KERNEL_A SEISMIC_ELEMENT_A
#define ATTENTION_SINKS SEISMIC_DIM_SK
#define ATTENTION_SINKS_BUFFER reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SINKS))
#define ATTENTION_SINK_AT(query_head) (static_cast<attention::u64>(query_head) * SEISMIC_SINKS_STRIDE_1)
#include "lib/attention/inputs.cuh"
#define ATTENTION_QUERY_GROUP SEISMIC_DIM_G
#define ATTENTION_INTERLEAVED SEISMIC_DIM_I
#define ATTENTION_SEPARATE SEISMIC_DIM_U
// attention_decode (M <= 8): split-KV decode attention.
//
// L1 `attention_decode_partial`, one block per (kv head, partition, row
// tile). A row's tokens (its visible spans in order, then its fresh span)
// split into PARTS partitions (the vector form's contiguous, the matrix
// form's contiguous runs of its key tiles); partition 0 of a layer with fresh
// rows appends the rows' K/V. Each partition publishes one partial (maximum,
// denominator, accumulator) per (row, query head). L2
// `attention_decode_merge`, one block per (query head, row): the partitions
// merge in partition order, then the output gate.
//
// Vector form (MATRIX = 0), row tiles of one row: the block prepares the
// row's G queries (rounded to the activation element as the contract
// publishes them, then scaled into the exp2 domain); within a partition each
// warp scans a contiguous range of its key group for the query heads of its
// slice, lanes owning W / 32 dimensions, so every K/V row is read once per
// slice of the G query heads of its kv head, in batches of TOKENS keys at
// fixed positions of the row's own key sequence (`attention::KeyCursor`), so
// a row's reduction order does not depend on which of its keys are still
// fresh. Warp states merge in key-group order.
//
// Grouped-query matrix form (MATRIX = 1): the tensor-core flash body
// (`attention::prefill::attend` with decode rows) over row tiles of WARPS * 16
// / G rows, their query heads the matrix rows; K/V tiles are read once per
// kv head, at fixed positions of each row's key sequence, each partition
// taking a contiguous run of a row's tiles sized by that row's own count.

// Both forms' shared code (an entry's includes expand once, whatever the
// branch).
#include "lib/attention/attention.cuh"
#include "lib/attention/decode_merge.cuh"

#if SEISMIC_TUNE_MATRIX

#define ATTENTION_STAGES SEISMIC_TUNE_STAGES
#define ATTENTION_COLUMNS SEISMIC_TUNE_COLUMNS
#define ATTENTION_Q_REGISTERS SEISMIC_TUNE_QREG
#define ATTENTION_PRODUCER_WARPS 0
// Two or more column warps over a head wider than 256 split the scores by key piece.
#define ATTENTION_SPLIT_SCORES 1
#include "lib/attention/prefill.cuh"

extern "C" __global__ void __launch_bounds__(attention::prefill::MMA_THREADS)
    attention_decode_partial(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::attend(
        ATTENTION_INPUTS(),
        attention::DenseHistory{
            SEISMIC_OUT_HISTORY_KEY,
            SEISMIC_OUT_HISTORY_VALUE,
            SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,
            SEISMIC_HISTORY_VALUE_STRIDE_0, SEISMIC_HISTORY_VALUE_STRIDE_1,
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
using attention::u64;

constexpr int WARPS = SEISMIC_TUNE_WARPS;
constexpr int PARTS = SEISMIC_TUNE_PARTS;
// The G query heads split into SLICES slices of H heads: warp w holds slice
// w % SLICES over key group w / SLICES (so wide heads keep their queries and
// outputs in registers), and each key/value row is loaded once per slice.
constexpr int SLICES = SEISMIC_TUNE_SLICES;
constexpr int H = G / SLICES;
constexpr int GROUPS = WARPS / SLICES;
// History tokens a warp keeps in flight (fewer when H query heads of state
// or heads wider than 256 already fill the registers).
constexpr int TOKENS = (DPL >= 16 || H >= 8) ? 2 : 4;

}  // namespace

extern "C" __global__ void __launch_bounds__(WARPS * 32)
    attention_decode_partial(SEISMIC_KERNEL_PARAMS) {
    const attention::Inputs in = ATTENTION_INPUTS();
    const attention::DenseHistory history{
        SEISMIC_OUT_HISTORY_KEY,
        SEISMIC_OUT_HISTORY_VALUE,
        SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,
        SEISMIC_HISTORY_VALUE_STRIDE_0, SEISMIC_HISTORY_VALUE_STRIDE_1, SEISMIC_PARAM_SLAB_ROWS};
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
#pragma unroll
    for (int h = 0; h < H; ++h) element::f32_span(queries + (head0 + h) * W + lane * DPL, q[h]);

    // Equal partitions of the row's keys, then equal key-group ranges of this
    // one.
    const u64 spans = SEISMIC_DIM_R;
    const long long total = attention::visible_total(in, row, spans, lane);
    const attention::Range partition =
        attention::partition(attention::Range{0, total}, attention::partition_run(total, 1, PARTS), part);
    const attention::Range slice = attention::partition(
        partition, (partition.hi - partition.lo + GROUPS - 1) / GROUPS, warp / SLICES);

    attention::Heads<H> state;
    attention::clear(state);

    // The slice's keys absorb in batches of TOKENS at fixed positions
    // slice.lo + i * TOKENS of the row's own key sequence, read through its
    // cursor (`attention::KeyCursor`): a key's batch, and so the row's
    // reduction order, depends on its position alone, whichever of its keys
    // are still fresh.
    attention::KeyCursor cursor = attention::KeyCursor::at(in, row, spans, history.slab_rows, slice.lo, lane);
    // The cursor's slab's first vectors.
    const unsigned char *key_slab = nullptr;
    const unsigned char *value_slab = nullptr;
    const u64 key_step = history.key_row * Act::bytes;
    const u64 value_step = history.value_row * Act::bytes;
    // Key `position` (at or past the cursor's) into `k` and `v`: a history
    // row from its slab, a fresh row prepared as history holds it.
    // Row `within` of the cursor's slab, as stored.
    auto read_history = [&](u64 within, float (&k)[DPL], float (&v)[DPL]) {
        element::span<Act, DPL, true>(key_slab + within * key_step, lane * DPL, k);
        element::span<Act, DPL, true>(value_slab + within * value_step, lane * DPL, v);
    };
    auto read = [&](long long position, float (&k)[DPL], float (&v)[DPL]) {
        bool moved;
        const int token = cursor.seek(position, moved);
        if (cursor.historical()) {
            if (moved) {
                key_slab = history.key_slab(cursor.slab, kv);
                value_slab = history.value_slab(cursor.slab, kv);
            }
            read_history(static_cast<unsigned>(token) - cursor.slab_lo, k, v);
        } else {
            attention::prepared_key(in, token, kv, k, own, lane);
            attention::fresh_value(in, token, kv, v, lane);
        }
    };
    for (long long position = slice.lo; position < slice.hi; position += TOKENS) {
        const int count = static_cast<int>(min(static_cast<long long>(TOKENS), slice.hi - position));
        float k[TOKENS][DPL];
        float v[TOKENS][DPL];
        // The batch's first key; a batch of TOKENS keys inside one history run
        // (a span's part of one slab) reads straight from the run's rows, each
        // key a row stride past the first.
        read(position, k[0], v[0]);
        if (cursor.historical() && count == TOKENS && position + TOKENS <= cursor.run_end) {
            const u64 first = static_cast<unsigned>(cursor.lo + static_cast<int>(position - cursor.offset)) - cursor.slab_lo;
#pragma unroll
            for (int t = 1; t < TOKENS; ++t) {
                read_history(first + t, k[t], v[t]);
            }
        } else {
#pragma unroll
            for (int t = 1; t < TOKENS; ++t) {
                if (t < count) {
                    read(position + t, k[t], v[t]);
                } else {
#pragma unroll
                    for (int d = 0; d < DPL; ++d) k[t][d] = v[t][d] = 0.0f;
                }
            }
        }
        float score[TOKENS][H];
        attention::scores(q, k, score, lane);
        attention::absorb(state, score, v, count);
    }

    attention::publish<WARPS, PARTS, SLICES>(state, exchange, warp_stats, partials, statistics, row, kv,
                                     part, warp, lane);
}

#endif


extern "C" __global__ void attention_decode_merge(SEISMIC_KERNEL_PARAMS) {
    attention::decode_gate<SEISMIC_TUNE_PARTS>(
        ATTENTION_INPUTS(),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS)),
        SEISMIC_RESULT_0_OUT, blockIdx.x, blockIdx.y, threadIdx.x);
}
