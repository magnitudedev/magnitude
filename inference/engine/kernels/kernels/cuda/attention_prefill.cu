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
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES)));
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

#define SPLIT()                                                                           \
    attention::prefill::Split {                                                           \
        reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),          \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS)),    \
            reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_COUNTS)) \
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

extern "C" __global__ void attention_prefill_merge(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::merge_partitions(ATTENTION_INPUTS(), SPLIT(), SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
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
