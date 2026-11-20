// A streamed decode route's resolve and assignment (`streaming/resolve.hpp`,
// which the includer includes first) over the ports `routed_route_streamed`
// and `routed_select_streamed` share: the group's step-stamped map
// ([GM, MAP_WORDS], the layer's expert e at row `home + e`), the layer's
// tier map (TM words, none without a host tier), the step pair, the step's
// sources, routing record, file table, the layer's front ring, its landing
// launch's grid, and the
// routes' weights (`ROUTED_RESOLVE_PORTS`' buffer
// and its strides: the route's scores, the general form's weights). Every thread of the
// selecting block calls it (`routed_resolve`), `words` its group words
// (`STREAMING_RESOLVE_WORDS`).

// The bank's words a resolve reads and writes, as its kernel binds them
// (`ROUTED_RESOLVE_PORTS`, in the kernel's own scope).
struct ResolvePorts {
    const int *routes, *step;
    int *sources, *routing, *file, *front, *land_grid;
    const int *map, *tier, *policy;
    // The routes' weights, read as their f32 bits, and their strides.
    const int *weights;
    unsigned long long weights0, weights1;
};

#define ROUTED_RESOLVE_PORTS(WEIGHTS, W0, W1)                                                                \
    ResolvePorts {                                                                                          \
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)),                                  \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)),                                \
            reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES)),                                   \
            reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTING)),                                   \
            reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_FILE)),                                      \
            reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_FRONT)),                                     \
            reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_LAND_GRID)),                                 \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_MAP)),                                 \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TIER)),                                \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_POLICY)),                              \
            reinterpret_cast<const int *>(WEIGHTS), W0, W1                                                  \
    }

// The resolve over `bank` (`whole`: a whole group's).
__device__ __forceinline__ void routed_resolve(const ResolvePorts &bank, bool whole, int *words,
                                               const seismic_words_t &seismic_words_value) {
    streaming_resolve(bank.routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
        static_cast<unsigned>(SEISMIC_DIM_M) * static_cast<unsigned>(SEISMIC_DIM_K), bank.weights, bank.weights0,
        bank.weights1, bank.step,
        SEISMIC_STEP_STRIDE_0, whole, bank.map, SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1, bank.tier,
        SEISMIC_TIER_STRIDE_0, static_cast<unsigned>(SEISMIC_DIM_TM), static_cast<unsigned>(SEISMIC_PARAM_SITED),
        bank.sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1,
        SEISMIC_SOURCES_STRIDE_2, bank.routing, SEISMIC_ROUTING_STRIDE_0, SEISMIC_ROUTING_STRIDE_1,
        static_cast<unsigned>(SEISMIC_DIM_FLIGHT), static_cast<unsigned>(SEISMIC_DIM_P), bank.file,
        SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1,
        bank.policy, bank.front, SEISMIC_FRONT_STRIDE_0, bank.land_grid, SEISMIC_LAND_GRID_STRIDE_0,
        static_cast<int>(SEISMIC_PARAM_LAND_GROUPS), words,
        threadIdx.x, blockDim.x);
}

// A streamed route's selecting warps write each row's key of every expert
// into the step's slot of its key records (`keys`, `streaming_record_key`),
// which placement ranks every expert by. `ROUTED_KEYS_RECORD` opens the slot
// once, before the rows' selections; `ROUTED_ROW_KEYS(r)` is row r's sink.
#define ROUTED_KEYS_RECORD                                                                                  \
    int *const keys_record = streaming_slot_of(reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_KEYS)),   \
        SEISMIC_KEYS_STRIDE_0,                                                                              \
        streaming_step(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0), \
        static_cast<unsigned>(SEISMIC_DIM_FLIGHT))
#define ROUTED_ROW_KEYS(r)                                                                                  \
    [&](unsigned e, float key) {                                                                            \
        streaming_record_key(keys_record, SEISMIC_KEYS_STRIDE_1, SEISMIC_KEYS_STRIDE_2,                     \
            static_cast<unsigned>(r), e, __float_as_int(key));                                              \
    }
