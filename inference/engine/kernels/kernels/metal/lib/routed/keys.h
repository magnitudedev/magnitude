// A streamed route's selection keys (`routed_route_streamed`,
// `routed_select_streamed`), in its kernel's scope after
// `streaming/resolve.hpp`: its selecting lanes write each row's key of
// every expert into the step's slot of its key records (`keys`,
// `streaming_record_key`), which placement ranks every expert by.
// `ROUTED_KEYS_RECORD` opens the slot once, before the rows' selections;
// `ROUTED_ROW_KEYS(r)` is row r's sink.
#define ROUTED_KEYS_RECORD                                                                  \
    SYSTEM int *const keys_record = streaming_slot_of(keys, SEISMIC_KEYS_STRIDE_0,          \
        streaming_step(step, SEISMIC_STEP_STRIDE_0), uint(SEISMIC_DIM_FLIGHT))
#define ROUTED_ROW_KEYS(r)                                                                  \
    [&](uint e, float key) {                                                                \
        streaming_record_key(keys_record, SEISMIC_KEYS_STRIDE_1, SEISMIC_KEYS_STRIDE_2, uint(r), e, \
            as_type<int>(key));                                                             \
    }
