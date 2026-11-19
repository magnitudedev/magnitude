// gated_delta_project_convolved: `gated_delta_project`'s GEMV and batched
// launches (RMS prologue over the F32 residual, the `dense` projection,
// segmented, qkv | z | alpha | beta), whose qkv segment also convolves. For each row
// of a channel, `ConvolvedChannel` forms `gated_delta_step`'s convolved
// channel (`recurrent::convolve`'s F32 tap chain over the rows rounded to A,
// then SiLU) and the row's share of the successor window: the batched class
// as its column epilogue, from every row's sum of the channel; the GEMV
// classes in a pass after their qkv projection, from the values the
// threadgroup published. Channels of the window rows must be contiguous
// (unit stride), as the step requires.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_QKV_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_ALPHA_WEIGHT
#define KERNEL_W3 SEISMIC_BETA_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/operator.h"
#include "lib/recurrent/versions.h"
#if defined(SEISMIC_FORMING_GATED_DELTA_PROJECT_CONVOLVED_GEMV) || defined(SEISMIC_FORMING_GATED_DELTA_PROJECT_CONVOLVED_GEMV_ROWS) || defined(SEISMIC_FORMING_GATED_DELTA_PROJECT_CONVOLVED_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

typedef activation::Act activation_element;
typedef ELEMENT_OF(SEISMIC_INPUT_NORM) norm_element;
// The rows every weight takes (the normalized rows, or NVFP4 operands).
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4,
    packets::W2_OPERANDS_NVFP4, packets::W3_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

// The slot of each row, resolved once per threadgroup before the projection
// so the epilogue's window reads do not wait on the slot tables: `lo` < 0 on
// a row outside every slot; the source and successor window banks as
// resolved entries.
struct ConvolvedRow {
    long lo;
    long stop;
    long kept;
    long taped;
    slab::Table<activation_element> source;
    slab::Table<activation_element> target;
};

// Threads 0..M-1 resolve row `thread_index`'s slot into `rows`; the caller
// barriers before the epilogue reads them.
inline void convolved_rows(versions::Slots slots, device const slab::Table<activation_element> *window,
    threadgroup ConvolvedRow *rows, uint thread_index, constant ulong *seismic_words) {
    if (thread_index >= SEISMIC_DIM_M)
        return;
    ConvolvedRow row;
    row.lo = -1;
    const ulong index = versions::slot_index_of_row(slots, long(thread_index), seismic_words);
    if (index < SEISMIC_DIM_B) {
        const versions::Slot slot = versions::slot_of(slots, index, seismic_words);
        row.lo = slot.lo;
        row.stop = slot.stop;
        row.kept = long(SEISMIC_DIM_C) - 1 + versions::tape_rows(slot, long(SEISMIC_DIM_T));
        row.taped = slot.taped;
        row.source = versions::bank_entry(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
        row.target = versions::bank_entry(window, slot.target, SEISMIC_WINDOW_STRIDE_0, seismic_words);
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
    element::Out<activation_element> projection;
    ulong projection0, projection1;
    element::Out<element::F32> convolved;
    ulong convolved0, convolved1;
    device const float *convolution;
    threadgroup const ConvolvedRow *rows;
    constant ulong *seismic_words;

    // Channel n of row m, given `value(j)`, row j's projection value of
    // channel n rounded to A.
    template <typename Value>
    void convolve(uint m, uint n, Value value) const {
        const element::Out<element::F32> out = convolved.offset(ulong(m) * convolved0 + ulong(n) * convolved1);
        const ConvolvedRow row = rows[m];
        if (row.lo < 0) {
            out.put(0, 0.0f);
            return;
        }
        const long taps = long(SEISMIC_DIM_C) - 1;
        const long local = long(m) - row.lo;
        const ulong channel = ulong(n);
        device const activation_element::storage *source = slab::view(row.source);
        float sum = 0.0f;
        PROJECTION_UNROLL for (uint tap = 0; tap < SEISMIC_DIM_C; ++tap) {
            const long position = local + long(tap) - taps;
            const float input = position < 0
                ? activation_element::load(source[ulong(row.taped + taps + position) * SEISMIC_WINDOW_STRIDE_1 + channel])
                : value(ulong(row.lo + position));
            sum = metal::fma(convolution[channel * SEISMIC_CONVOLUTION_STRIDE_0 + tap * SEISMIC_CONVOLUTION_STRIDE_1],
                input, sum);
        }
        out.put(0, sum / (1.0f + metal::exp(-sum)));
        const element::Out<activation_element> target = slab::out(row.target);
        const long kept = local - row.stop + taps;
        if (kept >= 0 && kept < row.kept)
            target.put(ulong(kept) * SEISMIC_WINDOW_STRIDE_1 + channel, value(ulong(m)));
        if (local == 0) {
            for (long tap = 0; row.stop + tap < taps; ++tap)
                target.move_stored(ulong(tap) * SEISMIC_WINDOW_STRIDE_1 + channel, slab::out(row.source),
                    ulong(row.taped + row.stop + tap) * SEISMIC_WINDOW_STRIDE_1 + channel, 1);
        }
    }

    // The GEMV classes' store: the projection value alone (the pass after
    // the projection convolves).
    void store(uint m, uint n, float value) const {
        projection.put(ulong(m) * projection0 + ulong(n) * projection1, value);
    }

    // The batched class's column epilogue: the projection value, then the
    // channel from every row's sum of it.
    template <uint N>
    void store_column(uint m, uint n, thread const float (&sums)[N]) const {
        projection.put(ulong(m) * projection0 + ulong(n) * projection1, row_sum(sums, m));
        convolve(m, n, [&](ulong j) { return activation_element::round(row_sum(sums, j)); });
    }

    // The GEMV classes' pass after their projection of channels first ..
    // first + count, which this threadgroup published: every thread takes
    // (row, channel) items in turn, reading the published values, so the
    // items run at once. A lane group's column epilogue would run its rows'
    // items one after another.
    void convolve_channels(uint first, uint count, uint m_rows, uint thread_index, uint threads) const {
        for (uint item = thread_index; item < count * m_rows; item += threads) {
            const uint n = first + item / m_rows, m = item % m_rows;
            convolve(m, n, [&](ulong j) { return projection.at(j * projection0 + ulong(n) * projection1); });
        }
    }
};

#define PROJECTION_ARGUMENTS                                                            \
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
    SEISMIC_OUT_WINDOW_PARAM(window),                                                   \
    SEISMIC_RESULT_0_PARAM(result_cells),                                               \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    SEISMIC_RESULT_1_PARAM(convolved_cells),                                            \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define PROJECTION_N0 ((2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W)
#define PROJECTION_N1 (SEISMIC_DIM_NV * SEISMIC_DIM_W)
#define PROJECTION_N2 SEISMIC_DIM_NV
#define PROJECTION_N3 SEISMIC_DIM_NV

// `raw` is the residual rows under the norm prologue; `in` the normalized
// rows of the matrix GEMV; the qkv segment's epilogue convolves, from the
// rows' slots (`slot_rows`, resolved by its threadgroups first).
#define RECURRENT_PROJECT_RAW                                                           \
    projection::InputOf<activation_element, norm_element, projection::AllRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>::type \
        raw{hidden, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1, input_norm,        \
        SEISMIC_INPUT_NORM_STRIDE_0, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {}}
#define PROJECTION_OPERANDS                                                             \
    const uint m = uint(SEISMIC_DIM_M), k = uint(SEISMIC_DIM_H);                        \
    RECURRENT_PROJECT_RAW;                                                              \
    const projection::Plain<activation_element, projection::AllRows> in{normalized, k, 1, k, {}}; \
    threadgroup ConvolvedRow slot_rows[16];                                             \
    projection::Weights<packets::W0> w0{{qkv_weight, KERNEL_W0_LAYOUT(k), k}};          \
    projection::Weights<packets::W1> w1{{gate_weight, KERNEL_W1_LAYOUT(k), k}};         \
    projection::Weights<packets::W2> w2{{alpha_weight, KERNEL_W2_LAYOUT(k), k}};        \
    projection::Weights<packets::W3> w3{{beta_weight, KERNEL_W3_LAYOUT(k), k}};         \
    ConvolvedChannel out0{element::Out<activation_element>(result_cells), SEISMIC_RESULT_0_STRIDE_0, \
        SEISMIC_RESULT_0_STRIDE_1, element::Out<element::F32>(convolved_cells),         \
        SEISMIC_RESULT_1_STRIDE_0, SEISMIC_RESULT_1_STRIDE_1, convolution, slot_rows,   \
        seismic_words};                                                                 \
    projection::Store<activation_element> out1{element::Out<activation_element>(result_cells), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, \
        PROJECTION_FIRST_1};                                                            \
    projection::Store<activation_element> out2{element::Out<activation_element>(result_cells), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, \
        PROJECTION_FIRST_2};                                                            \
    projection::Store<activation_element> out3{element::Out<activation_element>(result_cells), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, \
        PROJECTION_FIRST_3}

// The GEMV's rows: the norm applied as they stage, from their sums of
// squares over the threadgroup.
#define PROJECTION_GEMV_INPUT(x)                                                        \
    PROJECTION_SQUARES_SHARED(squares, decltype(raw)::parts);                           \
    projection::threadgroup_squares_runtime(raw, m, squares, simdgroups, sg, lane);     \
    const auto x = projection::shared_norm(raw, squares)
#define PROJECTION_FRAGMENTS normalized
#define PROJECTION_ZEROED(s) false
#define PROJECTION_BOUNDED 0u
#define PROJECTION_BOUND_TABLES {}
// The qkv segment: the batched launch's column epilogue receives every row's
// sum of a channel; a GEMV threadgroup convolves the channels it published
// after storing them.
#define PROJECTION_COLUMNS(s) (s == 0)
// The slots' rows, resolved before the convolution reads them: a GEMV
// threadgroup's last simdgroup resolves them beside the others' GEMV, and the
// body's own barrier publishes them to the convolution pass; the batched
// launch resolves them first and waits.
#define PROJECTION_GEMV_PROLOGUE(s)                                                     \
    if (s == 0 && sg + 1 == simdgroups)                                                 \
        convolved_rows({segments, stop, previous_bank, previous_tape, following_bank}, window, slot_rows, lane, \
            seismic_words);
#define PROJECTION_BATCH_PROLOGUE(s)                                                    \
    if (s == 0) {                                                                       \
        if (sg == 0)                                                                    \
            convolved_rows({segments, stop, previous_bank, previous_tape, following_bank}, window, slot_rows, lane, \
                seismic_words);                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    }
#define PROJECTION_GEMV_EPILOGUE(s, first, count)                                       \
    if (s == 0) {                                                                       \
        threadgroup_barrier(mem_flags::mem_device);                                     \
        out0.convolve_channels(first, count, m, sg * 32u + lane, simdgroups * 32u);     \
    }

// The normalized rows into `normalized` for the matrix GEMV, one row a
// threadgroup.
#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_CONVOLVED_NORMALIZE
template <uint MATRIX>
kernel void gated_delta_project_convolved_normalize(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    const uint k = uint(SEISMIC_DIM_H);
    RECURRENT_PROJECT_RAW;
    Operands::normalize<256>(raw, item, normalized, k, norms, thread_index);
}
#endif

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_CONVOLVED_GEMV
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void gated_delta_project_convolved_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_CONVOLVED_BATCH
template <uint MATRIX, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void gated_delta_project_convolved_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_CONVOLVED_GEMV_ROWS
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void gated_delta_project_convolved_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_CONVOLVED_GEMV_PAIR
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void gated_delta_project_convolved_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMV(PAIR, TILED);
}
#endif
