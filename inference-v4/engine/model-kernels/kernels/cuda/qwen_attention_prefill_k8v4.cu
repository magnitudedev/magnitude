// qwen_attention_prefill_k8v4 (M >= 16): `qwen_attention_prefill` over
// affine K8/V4 history (bodies in common/attention_prefill.cuh): the prepare
// launch appends encoded rows; the attend launch's producer warps decode code
// tiles into F16 operands beside its MMA warps (every product is F16).

#include "common/attention_prefill.cuh"

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
            SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_1                                         \
    }

extern "C" __global__ void __launch_bounds__(256)
    qwen_attention_prefill_k8v4_prepare(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::prepare(
        ATTENTION_INPUTS(), HISTORY(),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES)),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS)),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES)));
}

extern "C" __global__ void __launch_bounds__(attention::prefill::WARPS * 64, 1)
    qwen_attention_prefill_k8v4_attend(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::attend(ATTENTION_INPUTS(), HISTORY(),
                               SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES),
                               SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS),
                               SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES),
                               SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
}
