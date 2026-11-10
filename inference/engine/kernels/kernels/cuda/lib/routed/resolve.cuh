// A streamed decode route's resolve and assignment (`streaming/resolve.hpp`,
// which the includer includes first) over the ports `routed_route_streamed`
// and `routed_select_streamed` share: the group's step-stamped map
// ([GM, MAP_WORDS], the layer's expert e at row `home + e`), the layer's
// tier map (TM words, none without a host tier), the step pair, the step's
// sources, routing record and file table. Every thread of the
// selecting block calls it (`routed_resolve`), `words` its group words
// (`STREAMING_RESOLVE_WORDS`).

// The bank's words a resolve reads and writes, as its kernel binds them
// (`ROUTED_RESOLVE_PORTS`, in the kernel's own scope).
struct ResolvePorts {
    const int *routes, *step;
    int *sources, *routing, *file;
    const int *map, *tier, *policy;
};

#define ROUTED_RESOLVE_PORTS                                                                                \
    ResolvePorts {                                                                                          \
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)),                                  \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)),                                \
            reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES)),                                   \
            reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTING)),                                   \
            reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_FILE)),                                      \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_MAP)),                                 \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TIER)),                                \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_POLICY))                               \
    }

// The resolve over `bank` (`whole`: a whole group's).
__device__ __forceinline__ void routed_resolve(const ResolvePorts &bank, bool whole, int *words,
                                               const seismic_words_t &seismic_words_value) {
    streaming_resolve(bank.routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
        static_cast<unsigned>(SEISMIC_DIM_M) * static_cast<unsigned>(SEISMIC_DIM_K), bank.step,
        SEISMIC_STEP_STRIDE_0, whole, bank.map, SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1, bank.tier,
        SEISMIC_TIER_STRIDE_0, static_cast<unsigned>(SEISMIC_DIM_TM), static_cast<unsigned>(SEISMIC_PARAM_SITED),
        bank.sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1,
        SEISMIC_SOURCES_STRIDE_2, bank.routing, SEISMIC_ROUTING_STRIDE_0, SEISMIC_ROUTING_STRIDE_1,
        static_cast<unsigned>(SEISMIC_DIM_FLIGHT), static_cast<unsigned>(SEISMIC_DIM_P), bank.file,
        SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1,
        bank.policy, words,
        threadIdx.x, blockDim.x);
}
