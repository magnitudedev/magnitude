// The CPU's answer to a streamed decode combine (`routed_down_sited`,
// `routed_output_sited` in routed.seismic): the counterpart of
// `metal/lib/routed/sited.h`. The includer includes the protocol's
// `streaming/site.hpp` first, and binds the entry's step, sources, site
// and site_output ports.
#define ROUTED_SITED

// Whether the CPU answered the step's request (its slot `slot`, `q` the
// step): a choice of the step is the CPU's, and invocation 0 reads the
// slot of the layer's site table (`streaming_await_slot`; the park before
// the call opened once the slot was answered); the outcome reaches the
// workgroup through its group memory. A step with no CPU choice reads
// neither table. Every invocation calls it.
bool routed_site_answered(const uint64_t q, const uint64_t slot) {
    if (!streaming_site_any(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), SEISMIC_SOURCES_STRIDE_0,
            SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, uint(SEISMIC_DIM_M) * uint(SEISMIC_DIM_K)))
        return false;
    if (gl_LocalInvocationIndex == 0u)
        seismic_shared_i32[0] = streaming_await_slot(
            SEISMIC_PTR(SEISMIC_BUFFER_SITE) + slot * SEISMIC_SITE_STRIDE_0 * 4ul, SEISMIC_SITE_STRIDE_1, q) ? 1 : 0;
    barrier();
    return seismic_shared_i32[0] != 0;
}

// The CPU's weighted sum of row m at channel n (`site_output[slot, m, n]`)
// added to `selected` where the CPU answered the step (`decode_selected`),
// else `selected` as it is.
float routed_sited(const float selected, const bool answered, const uint64_t slot, const uint64_t m,
    const uint64_t n) {
    if (!answered)
        return selected;
    return selected + element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_SITE_OUTPUT) + (slot * SEISMIC_SITE_OUTPUT_STRIDE_0
        + m * SEISMIC_SITE_OUTPUT_STRIDE_1 + n * SEISMIC_SITE_OUTPUT_STRIDE_2) * 4ul);
}
