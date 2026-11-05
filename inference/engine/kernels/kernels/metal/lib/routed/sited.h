// The CPU's answer of a streamed decode combine (`routed_down_sited`,
// `routed_output_sited`): thread 0 waits on the step's slot of the layer's
// site table when a choice of the step is the CPU's (`streaming_site_any`,
// `streaming_await_slot`); then every publishing thread adds row m's answer
// (`site_output[slot, m, n]`), read through system-coherent loads. The
// includer includes `streaming/site.hpp` and `routed.h` first.

namespace routed {

// Channel n of row m's published projections, in A (a fused output's slot K
// the shared expert's).
struct PublishedProjections {
    device const uchar *published;
    constant ulong *seismic_words;
    ulong m;
    uint n;
    float at(uint k) const thread {
        return Act::load(reinterpret_cast<device const typename Act::storage *>(published)[
            m * SEISMIC_PUBLISHED_STRIDE_0 + ulong(k) * SEISMIC_PUBLISHED_STRIDE_1 + ulong(n) * SEISMIC_PUBLISHED_STRIDE_2]);
    }
};

// Whether the CPU answered step `q`'s request at `slot`, the step's slot of
// the site table, `ready` the group's word. Every thread calls it.
inline bool site_answered(device const int *sources, SYSTEM const int *slot, ulong q, threadgroup int *ready, uint tid,
    constant ulong *seismic_words) {
    if (tid == 0) {
        const uint choices = uint(SEISMIC_DIM_M) * uint(SEISMIC_DIM_K);
        *ready = streaming_site_any(sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1,
                     SEISMIC_SOURCES_STRIDE_2, choices)
            && streaming_await_slot(slot, SEISMIC_SITE_STRIDE_1, q);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *ready != 0;
}

// Channel n of row m of the slot's answer `output` (`site_output`) where the
// CPU answered the step, else none. By every publishing thread: where
// answered, the fence is the acquire of the output the CPU wrote before its
// answer.
inline Answer answer_at(SYSTEM const float *output, bool answered, ulong m, uint n, constant ulong *seismic_words) {
    if (!answered)
        return Answer{false, 0.0f};
    streaming_system_fence();
    return Answer{true, output[m * SEISMIC_SITE_OUTPUT_STRIDE_1 + ulong(n) * SEISMIC_SITE_OUTPUT_STRIDE_2]};
}

} // namespace routed
