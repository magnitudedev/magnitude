// gated_delta_project_convolved: `gated_delta_project`'s GEMV and batched
// GEMV launches (RMS prologue over the F32 residual, one segmented projection
// qkv | z | alpha | beta), whose qkv segment also convolves. Its column
// epilogue (`ConvolvedChannel`) receives every row's sum of a channel and,
// for each row it publishes, forms `gated_delta_step`'s convolved channel
// (`recurrent::convolve`'s F32 tap chain over the rows rounded to A, then
// SiLU) and the row's share of the successor window. Channels of the window
// rows must be contiguous (unit stride), as the step requires.
#define KERNEL_W0 SEISMIC_QKV_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_ALPHA_WEIGHT
#define KERNEL_W3 SEISMIC_BETA_WEIGHT
#include "lib/projection/projection.h"
#include "lib/recurrent/versions.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_INPUT_NORM) norm_element;

// The slot of each row, resolved once per threadgroup before the projection
// so the epilogue's window reads do not wait on the slot tables: `lo` < 0 on
// a row outside every slot; the source and successor window banks as
// addresses.
struct ConvolvedRow {
    long lo;
    long stop;
    long kept;
    long taped;
    ulong source;
    ulong target;
};

// Threads 0..M-1 resolve row `thread_index`'s slot into `rows`; the caller
// barriers before the epilogue reads them.
inline void convolved_rows(versions::Slots slots, device const ulong *window, threadgroup ConvolvedRow *rows,
    uint thread_index, constant ulong *seismic_words) {
    if (thread_index >= SEISMIC_DIM_M)
        return;
    ConvolvedRow row;
    row.lo = -1;
    const ulong index = versions::slot_index_of_row(slots, long(thread_index), seismic_words);
    if (index < SEISMIC_DIM_B) {
        const versions::Slot slot = versions::slot_of(slots, index, seismic_words);
        row.lo = slot.lo;
        row.stop = slot.stop;
        row.kept = long(SEISMIC_DIM_C) - 1 + versions::tape_rows(slot, seismic_words);
        row.taped = slot.taped;
        row.source = reinterpret_cast<ulong>(
            versions::bank<activation::storage>(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words));
        row.target = reinterpret_cast<ulong>(
            versions::bank<activation::storage>(window, slot.target, SEISMIC_WINDOW_STRIDE_0, seismic_words));
    }
    rows[thread_index] = row;
}

// The qkv segment's epilogue, for (row m, channel n):
// - the projection value round_A(sum) (`projection::Store`);
// - the convolved channel, zero on a row outside every slot: over taps
//   0..C-1 of slot-local row `local`, fma(w[n, tap], x, sum) from 0 with x the
//   source version's window row `local + tap - (C - 1)` before the slot, else
//   the slot's row, rounded to A; then sum / (1 + exp(-sum));
// - the successor window (the C - 1 raw rows before the slot's publication
//   row, then the raw rows of its tape): the row's raw value when the window
//   keeps it, and, from the slot's first row, the source window rows it keeps.
// sums[index] by a select over the whole array, so the array stays in
// registers (a dynamic index would place it in stack memory).
template <uint N>
inline float row_sum(thread const float (&sums)[N], ulong index) {
    float value = sums[0];
    PROJECTION_UNROLL for (uint j = 1; j < N; ++j)
        value = index == j ? sums[j] : value;
    return value;
}

struct ConvolvedChannel {
    device uchar *projection;
    ulong projection0, projection1;
    device float *convolved;
    ulong convolved0, convolved1;
    device const float *convolution;
    threadgroup const ConvolvedRow *rows;
    constant ulong *seismic_words;

    template <uint N>
    void store_column(uint m, uint n, thread const float (&sums)[N]) const {
        typedef activation::storage Storage;
        const float raw = row_sum(sums, m);
        reinterpret_cast<device Storage *>(projection)[ulong(m) * projection0 + ulong(n) * projection1] =
            activation::store(raw);
        device float *value = convolved + ulong(m) * convolved0 + ulong(n) * convolved1;
        const ConvolvedRow row = rows[m];
        if (row.lo < 0) {
            *value = 0.0f;
            return;
        }
        const long taps = long(SEISMIC_DIM_C) - 1;
        const long local = long(m) - row.lo;
        const ulong channel = ulong(n);
        device const Storage *source = reinterpret_cast<device const Storage *>(row.source);
        float sum = 0.0f;
        PROJECTION_UNROLL for (uint tap = 0; tap < SEISMIC_DIM_C; ++tap) {
            const long position = local + long(tap) - taps;
            const float input = position < 0
                ? activation::load(source[ulong(row.taped + taps + position) * SEISMIC_WINDOW_STRIDE_1 + channel])
                : activation::round(row_sum(sums, ulong(row.lo + position)));
            sum = metal::fma(convolution[channel * SEISMIC_CONVOLUTION_STRIDE_0 + tap * SEISMIC_CONVOLUTION_STRIDE_1],
                input, sum);
        }
        *value = sum / (1.0f + metal::exp(-sum));
        device Storage *target = reinterpret_cast<device Storage *>(row.target);
        const long kept = local - row.stop + taps;
        if (kept >= 0 && kept < row.kept)
            target[ulong(kept) * SEISMIC_WINDOW_STRIDE_1 + channel] = activation::store(raw);
        if (local == 0) {
            for (long tap = 0; row.stop + tap < taps; ++tap)
                target[ulong(tap) * SEISMIC_WINDOW_STRIDE_1 + channel] =
                    source[ulong(row.taped + row.stop + tap) * SEISMIC_WINDOW_STRIDE_1 + channel];
        }
    }
};

#define RECURRENT_PROJECT_ARGUMENTS                                                     \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *input_norm [[buffer(SEISMIC_BUFFER_INPUT_NORM)]],               \
    device const uchar *qkv_weight [[buffer(SEISMIC_BUFFER_QKV_WEIGHT)]],               \
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],             \
    device const uchar *alpha_weight [[buffer(SEISMIC_BUFFER_ALPHA_WEIGHT)]],           \
    device const uchar *beta_weight [[buffer(SEISMIC_BUFFER_BETA_WEIGHT)]],             \
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],             \
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],                     \
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],                             \
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],           \
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],           \
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],         \
    device const ulong *window [[buffer(SEISMIC_BUFFER_WINDOW)]],                       \
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device float *convolved [[buffer(SEISMIC_RESULT_1_BUFFER)]],                        \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define RECURRENT_PROJECT_OPERANDS                                                      \
    const uint k = uint(SEISMIC_DIM_H);                                                 \
    const uint qkv_rows = uint((2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W);  \
    const uint gate_rows = uint(SEISMIC_DIM_NV * SEISMIC_DIM_W);                        \
    const uint head_rows = uint(SEISMIC_DIM_NV);                                        \
    projection::Rms<activation, norm_element, projection::AllRows> in{hidden, SEISMIC_HIDDEN_STRIDE_0, \
        SEISMIC_HIDDEN_STRIDE_1, input_norm, SEISMIC_INPUT_NORM_STRIDE_0,               \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {}};                            \
    projection::Weights<packets::W0> qkv{qkv_weight, KERNEL_W0_LAYOUT(k), k};           \
    projection::Weights<packets::W1> gate{gate_weight, KERNEL_W1_LAYOUT(k), k};         \
    projection::Weights<packets::W2> alpha{alpha_weight, KERNEL_W2_LAYOUT(k), k};       \
    projection::Weights<packets::W3> beta{beta_weight, KERNEL_W3_LAYOUT(k), k};         \
    ConvolvedChannel qkv_out{result, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, \
        convolved, SEISMIC_RESULT_1_STRIDE_0, SEISMIC_RESULT_1_STRIDE_1, convolution,   \
        slot_rows, seismic_words};                                                      \
    projection::Store<activation> gate_out{result, SEISMIC_RESULT_0_STRIDE_0,           \
        SEISMIC_RESULT_0_STRIDE_1, qkv_rows};                                           \
    projection::Store<activation> alpha_out{result, SEISMIC_RESULT_0_STRIDE_0,          \
        SEISMIC_RESULT_0_STRIDE_1, qkv_rows + gate_rows};                               \
    projection::Store<activation> beta_out{result, SEISMIC_RESULT_0_STRIDE_0,           \
        SEISMIC_RESULT_0_STRIDE_1, qkv_rows + gate_rows + head_rows}

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_CONVOLVED_GEMV
template <uint ROWS, uint LANES>
kernel void gated_delta_project_convolved_gemv(RECURRENT_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup ConvolvedRow slot_rows[16];
    RECURRENT_PROJECT_OPERANDS;
    uint per = simdgroups * ROWS * (32u / LANES);
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    uint t0 = (qkv_rows + per - 1) / per, t1 = (gate_rows + per - 1) / per;
    uint t2 = (head_rows + per - 1) / per;
    // The last simdgroup resolves the rows' slots after its share of the norm;
    // the body's opening barrier publishes them to the epilogue.
    if (tile < t0 && sg + 1 == simdgroups)
        convolved_rows({segments, stop, previous_bank, previous_tape, following_bank}, window, slot_rows, lane,
            seismic_words);
    if (tile < t0) {
        PROJECTION_FOR_ROWS(rows, projection::gemv_columns_runtime<packets::W0, ROWS, MAXM, LANES>(
            x, qkv_out, qkv, rows, qkv_rows, k, tile, shared, simdgroups, sg, lane));
    } else if (tile < t0 + t1) {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W1, ROWS, MAXM, LANES>(
            x, gate_out, gate, rows, gate_rows, k, tile - t0, shared, simdgroups, sg, lane));
    } else if (tile < t0 + t1 + t2) {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W2, ROWS, MAXM, LANES>(
            x, alpha_out, alpha, rows, head_rows, k, tile - t0 - t1, shared, simdgroups, sg, lane));
    } else {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W3, ROWS, MAXM, LANES>(
            x, beta_out, beta, rows, head_rows, k, tile - t0 - t1 - t2, shared, simdgroups, sg, lane));
    }
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_CONVOLVED_BATCH
template <uint BATCH_ROWS>
kernel void gated_delta_project_convolved_batch(RECURRENT_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup ConvolvedRow slot_rows[16];
    RECURRENT_PROJECT_OPERANDS;
    uint per = simdgroups * BATCH_ROWS * 8u;
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    uint t0 = (qkv_rows + per - 1) / per, t1 = (gate_rows + per - 1) / per;
    uint t2 = (head_rows + per - 1) / per;
    // The last simdgroup resolves the rows' slots after its share of the norm;
    // the body's opening barrier publishes them to the epilogue.
    if (tile < t0 && sg + 1 == simdgroups)
        convolved_rows({segments, stop, previous_bank, previous_tape, following_bank}, window, slot_rows, lane,
            seismic_words);
    if (tile < t0)
        projection::gemv_batch_columns_runtime<packets::W0, BATCH_ROWS>(x, qkv_out, qkv, rows, qkv_rows, k,
            tile, shared, simdgroups, sg, lane);
    else if (tile < t0 + t1)
        projection::gemv_batch_runtime<packets::W1, BATCH_ROWS>(x, gate_out, gate, rows, gate_rows, k,
            tile - t0, shared, simdgroups, sg, lane);
    else if (tile < t0 + t1 + t2)
        projection::gemv_batch_runtime<packets::W2, BATCH_ROWS>(x, alpha_out, alpha, rows, head_rows, k,
            tile - t0 - t1, shared, simdgroups, sg, lane);
    else
        projection::gemv_batch_runtime<packets::W3, BATCH_ROWS>(x, beta_out, beta, rows, head_rows, k,
            tile - t0 - t1 - t2, shared, simdgroups, sg, lane);
}
#endif
