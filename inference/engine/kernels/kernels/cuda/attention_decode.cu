// attention_decode (M <= 8): split-KV decode attention.
//
// L1 `attention_decode_partial`, one block per (kv head, partition, row): the
// block prepares the row's G queries (rounded to the activation element as
// the contract publishes them, then scaled into the exp2 domain) and, in
// partition 0 of a layer with fresh rows, appends the row's K/V. A row's
// tokens (its visible spans in order, then its fresh span) split into PARTS
// contiguous partitions; within a partition each warp scans a contiguous
// range of its key group for the query heads of its slice, lanes owning W /
// 32 dimensions, so every K/V row is read once per slice of the G query heads
// of its kv head. Warp states merge in key-group order into one partial
// (maximum, denominator, accumulator) per (row, query head, partition). L2
// `attention_decode_merge`, one block per (query head, row): the partitions
// merge in partition order, then the output gate.

#include "lib/attention/attention.cuh"

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
// already fill the registers).
constexpr int TOKENS = H >= 8 ? 2 : 4;

}  // namespace

extern "C" __global__ void __launch_bounds__(WARPS * 32)
    attention_decode_partial(SEISMIC_KERNEL_PARAMS) {
    const attention::Inputs in = ATTENTION_INPUTS();
    const attention::DenseHistory history{
        reinterpret_cast<const attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY)),
        reinterpret_cast<const attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE)),
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
    const long long total = attention::visible_total(in, row, spans);
    const attention::Range partition =
        attention::partition(attention::Range{0, total}, (total + PARTS - 1) / PARTS, part);
    const attention::Range slice = attention::partition(
        partition, (partition.hi - partition.lo + GROUPS - 1) / GROUPS, warp / SLICES);

    attention::Heads<H> state;
    attention::clear(state);

    long long offset = 0;
    for (u64 span = 0; span <= spans && offset < slice.hi; ++span) {
        const attention::Span s = attention::span(in, row, span, spans);
        const long long length = s.hi > s.lo ? s.hi - s.lo : 0;
        const long long first = max(slice.lo, offset);
        const long long last = min(slice.hi, offset + length);
        if (first < last) {
            const int lo = s.lo + static_cast<int>(first - offset);
            const int hi = s.lo + static_cast<int>(last - offset);
            if (span < spans) {
                // Each slab's part of the span resolves its first rows once;
                // later rows are row strides past them (a per-token slab
                // lookup divides 64-bit rows).
                for (int part_lo = lo; part_lo < hi;) {
                    const int part_hi = history.slab_end(part_lo, hi);
                    const auto *keys = history.key_vector(part_lo, kv);
                    const auto *values = history.value_vector(part_lo, kv);
                    const u64 key_step = history.key_row * Act::bytes;
                    const u64 value_step = history.value_row * Act::bytes;
                    for (int token = part_lo; token < part_hi; token += TOKENS) {
                        const int count = min(TOKENS, part_hi - token);
                        float k[TOKENS][DPL];
                        float v[TOKENS][DPL];
#pragma unroll
                        for (int t = 0; t < TOKENS; ++t) {
                            if (t < count) {
                                const u64 at = static_cast<u64>(token + t - part_lo);
                                element::span<Act, DPL, true>(keys + at * key_step, lane * DPL, k[t]);
                                element::span<Act, DPL, true>(values + at * value_step, lane * DPL,
                                                              v[t]);
                            } else {
#pragma unroll
                                for (int d = 0; d < DPL; ++d) k[t][d] = v[t][d] = 0.0f;
                            }
                        }
                        float score[TOKENS][H];
                        attention::scores(q, k, score);
                        attention::absorb(state, score, v, count);
                    }
                    part_lo = part_hi;
                }
            } else {
                for (int token = lo; token < hi; ++token) {
                    float k[1][DPL];
                    float v[1][DPL];
                    attention::prepared_key(in, token, kv, k[0], own, lane);
                    attention::fresh_value(in, token, kv, v[0], lane);
                    float score[1][H];
                    attention::scores(q, k, score);
                    attention::absorb(state, score, v, 1);
                }
            }
        }
        offset += length;
    }

    attention::publish<WARPS, PARTS, SLICES>(state, exchange, warp_stats, partials, statistics, row, kv,
                                     part, warp, lane);
}

extern "C" __global__ void attention_decode_merge(SEISMIC_KERNEL_PARAMS) {
    attention::decode_gate<PARTS>(
        ATTENTION_INPUTS(),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS)),
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), blockIdx.x, blockIdx.y, threadIdx.x);
}
