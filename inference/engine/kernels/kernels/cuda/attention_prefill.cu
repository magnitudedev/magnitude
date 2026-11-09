// Sink logits (SK): an unsplit tile's direct store and the split merge add
// each head's sink to its denominator (`attention::normalizer`).
#define ATTENTION_SINKS SEISMIC_DIM_SK
#define ATTENTION_SINKS_BUFFER reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SINKS))
#define ATTENTION_SINK_AT(query_head) (static_cast<attention::u64>(query_head) * SEISMIC_SINKS_STRIDE_1)
#include "lib/attention/inputs.cuh"
#define ATTENTION_QUERY_GROUP SEISMIC_DIM_G
#define ATTENTION_INTERLEAVED SEISMIC_DIM_I
#define ATTENTION_SEPARATE SEISMIC_DIM_U
// attention_prefill (M >= 16): flash attention on tensor cores over dense
// history (bodies in lib/attention/prefill.cuh).

#define ATTENTION_STAGES SEISMIC_TUNE_STAGES
#define ATTENTION_COLUMNS SEISMIC_TUNE_COLUMNS
#define ATTENTION_Q_REGISTERS SEISMIC_TUNE_QREG
#define ATTENTION_PRODUCER_WARPS 0
#include "lib/attention/prefill.cuh"

// Dense history planes [T, KV, W] (activation element).
#define HISTORY()                                                                            \
    attention::DenseHistory {                                                                \
        reinterpret_cast<const attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY)),   \
        reinterpret_cast<const attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE)), \
            SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,                      \
            SEISMIC_HISTORY_VALUE_STRIDE_0, SEISMIC_HISTORY_VALUE_STRIDE_1,                  \
            SEISMIC_PARAM_SLAB_ROWS                                                          \
    }

extern "C" __global__ void __launch_bounds__(256) attention_prefill_prepare(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::prepare(
        ATTENTION_INPUTS(), HISTORY(),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES)),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS)),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES)),
        attention::key_rounding(SEISMIC_DIM_KR != 0, SEISMIC_PARAM_KEY_BLOCK, SEISMIC_PARAM_KEY_ELEMENT,
                                SEISMIC_PARAM_KEY_SCALE, SEISMIC_PARAM_KEY_AMAX_FLOOR));
}

// A launch's WorkList selection (none when B is 0) and its list scratch:
// each query tile holds the key tiles its rows read, each row at most the
// bitmap's 32 * B * BR keys in at most R + 1 runs.
#define PREFILL_LIST_CAPACITY                                                                               \
    static_cast<attention::u32>(attention::prefill::QT *                                                    \
        ((32 * SEISMIC_DIM_B * SEISMIC_DIM_BR + attention::prefill::KEYS - 1) / attention::prefill::KEYS + SEISMIC_DIM_R + 1))
// The worklist scratch: the query tiles' (span, first) lists, their counts,
// then the rows' span ordinals.
#define PREFILL_LIST_TILES \
    static_cast<attention::u64>((SEISMIC_DIM_M + attention::prefill::QT - 1) / attention::prefill::QT)
#define PREFILL_WORKLIST() reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_WORKLIST))
#define SELECTION()                                                                                         \
    attention::prefill::Selection {                                                                         \
        reinterpret_cast<const attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTION)),                    \
            PREFILL_WORKLIST() + PREFILL_LIST_TILES * PREFILL_LIST_CAPACITY * 2 + PREFILL_LIST_TILES,       \
            PREFILL_WORKLIST(),                                                                             \
            reinterpret_cast<const attention::u32 *>(PREFILL_WORKLIST() + PREFILL_LIST_TILES * PREFILL_LIST_CAPACITY * 2), \
            static_cast<attention::u32>(SEISMIC_DIM_B), static_cast<int>(SEISMIC_DIM_BR),                   \
            PREFILL_LIST_CAPACITY                                                                           \
    }

// The attend grid's key partitions (its z extent, attention.seismic). An
// index-list source's state (KS > 0) is the partition after them.
#define PREFILL_PARTS                                                                     \
    max(1ull, (static_cast<unsigned long long>(SEISMIC_TUNE_SPLIT_GROUPS) + PREFILL_LIST_TILES * SEISMIC_DIM_KV - 1) / \
                  (PREFILL_LIST_TILES * SEISMIC_DIM_KV))
#define PREFILL_PICKED(records, width)                                                    \
    (SEISMIC_DIM_KS > 0 ? (records) + PREFILL_PARTS * SEISMIC_DIM_M * SEISMIC_DIM_KV * attention::G * (width) \
                        : nullptr)
#define SPLIT()                                                                           \
    attention::prefill::Split {                                                           \
        reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),          \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS)),    \
            reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_COUNTS)), \
            PREFILL_PICKED(reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)), attention::W), \
            PREFILL_PICKED(reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS)), 2) \
    }

extern "C" __global__ void __launch_bounds__(attention::prefill::MMA_THREADS, 1)
    attention_prefill_attend(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::attend(
        ATTENTION_INPUTS(), HISTORY(),
        attention::prefill::PrefillRows{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES),
                                      SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS),
                                      SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES),
                                      SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SPLIT()},
        attention::prefill::Block{static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y),
                                static_cast<int>(blockIdx.z), static_cast<int>(gridDim.z)},
        SELECTION());
}

// The index-list source (KS > 0, `attention::prefill::picked`): block (row,
// query head), one warp, after L1 put the queries; its state is the
// partition after the key partitions' (`PREFILL_PICKED`).
extern "C" __global__ void attention_prefill_picks(SEISMIC_KERNEL_PARAMS) {
    const attention::Inputs in = ATTENTION_INPUTS();
    const auto *rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTED_ROWS));
    const auto *selected = reinterpret_cast<const attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTED));
    const attention::u64 row = blockIdx.x;
    attention::prefill::picked<attention::prefill::Operands<attention::DenseHistory>>(
        reinterpret_cast<const attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES)), SPLIT(),
        [&](int pick) { return rows[row * SEISMIC_SELECTED_ROWS_STRIDE_0 + pick * SEISMIC_SELECTED_ROWS_STRIDE_1]; },
        static_cast<int>(SEISMIC_DIM_KS),
        [&](int picked, int kv) -> const attention::u8 * {
            return slab::row(selected, picked, SEISMIC_PARAM_SELECTED_SLAB_ROWS,
                             SEISMIC_SELECTED_STRIDE_0 * attention::Act::bytes)
                + static_cast<attention::u64>(kv) * SEISMIC_SELECTED_STRIDE_1 * attention::Act::bytes;
        },
        in.scale * attention::LOG2E, row, static_cast<int>(blockIdx.y), static_cast<int>(threadIdx.x));
}

extern "C" __global__ void attention_prefill_merge(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::merge_partitions(ATTENTION_INPUTS(), SPLIT(), SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
}

// The output rotary (RO = 1): block (row, query head), thread per pair.
extern "C" __global__ void attention_prefill_rotate(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::output_rotary(ATTENTION_INPUTS(), SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), blockIdx.x,
                                      static_cast<int>(blockIdx.y), static_cast<int>(threadIdx.x));
}

// The WorkList's list launch (B > 0): each query tile's read key tiles
// (`attention::prefill::list`).
extern "C" __global__ void __launch_bounds__(256) attention_prefill_list(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::list(ATTENTION_INPUTS(),
                             PREFILL_WORKLIST() + PREFILL_LIST_TILES * PREFILL_LIST_CAPACITY * 2 + PREFILL_LIST_TILES,
                             PREFILL_WORKLIST(),
                             reinterpret_cast<attention::u32 *>(PREFILL_WORKLIST() + PREFILL_LIST_TILES * PREFILL_LIST_CAPACITY * 2),
                             reinterpret_cast<const attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTION)),
                             static_cast<attention::u32>(SEISMIC_DIM_B), static_cast<int>(SEISMIC_DIM_BR),
                             PREFILL_LIST_CAPACITY, static_cast<int>(blockIdx.x));
}
