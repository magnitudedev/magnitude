#include "lib/attention/inputs.cuh"
#define ATTENTION_QUERY_GROUP SEISMIC_DIM_G
#define ATTENTION_INTERLEAVED SEISMIC_DIM_I
#define ATTENTION_SEPARATE SEISMIC_DIM_U
// attention_prefill_k8v4 (M >= 16): `attention_prefill` over affine K8/V4
// history (bodies in lib/attention/prefill.cuh): the prepare launch appends
// encoded rows; the attend launch's producer warps decode code tiles into F16
// operands beside its MMA warps (every product is F16).

#define ATTENTION_STAGES SEISMIC_TUNE_STAGES
#define ATTENTION_COLUMNS SEISMIC_TUNE_COLUMNS
#define ATTENTION_Q_REGISTERS SEISMIC_TUNE_QREG
#define ATTENTION_PRODUCER_WARPS SEISMIC_TUNE_PRODUCERS
#include "lib/attention/prefill.cuh"

// Affine history planes: codes [T, KV, W * B / 32] u32 and group (scale,
// zero) pairs [T, KV, W / 16] f16, one aligned u32 per pair.
#define HISTORY()                                                                              \
    attention::AffineHistory {                                                                 \
        reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY_CODES)),      \
            reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY_COEFFICIENTS)), \
            reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE_CODES)),    \
            reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE_COEFFICIENTS)), \
            SEISMIC_HISTORY_KEY_CODES_STRIDE_0, SEISMIC_HISTORY_KEY_CODES_STRIDE_1,             \
            SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_0, SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_1, \
            SEISMIC_HISTORY_VALUE_CODES_STRIDE_0, SEISMIC_HISTORY_VALUE_CODES_STRIDE_1,         \
            SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_0,                                        \
            SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_1, SEISMIC_PARAM_SLAB_ROWS                \
    }

extern "C" __global__ void __launch_bounds__(256)
    attention_prefill_k8v4_prepare(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::prepare(
        ATTENTION_INPUTS(), HISTORY(),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES)),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS)),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES)));
}

#define SPLIT()                                                                           \
    attention::prefill::Split {                                                           \
        reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),          \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS)),    \
            reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_COUNTS)) \
    }

extern "C" __global__ void __launch_bounds__(attention::prefill::MMA_THREADS + attention::prefill::PRODUCERS, 1)
    attention_prefill_k8v4_attend(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::attend(
        ATTENTION_INPUTS(), HISTORY(),
        attention::prefill::PrefillRows{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES),
                                      SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS),
                                      SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES),
                                      SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SPLIT()},
        attention::prefill::Block{static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y),
                                static_cast<int>(blockIdx.z), static_cast<int>(gridDim.z)});
}

extern "C" __global__ void attention_prefill_k8v4_merge(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::merge_partitions(ATTENTION_INPUTS(), SPLIT(), SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
}
