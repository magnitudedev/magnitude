// Included before the selection (`selection.glsl`, `route.glsl`), which then
// stores each winner at its group word as well (`routed_group_route`): a
// plain device store is not visible to another invocation even across the
// group's barrier, so the resolve reads the routes from group memory.
#if defined(SELECT_K) || defined(ROUTE_K)
#error "include lib/routed/streamed.glsl before the selection"
#endif
#include "../streaming/control.glsl"
#define STREAMING_ROUTES_GROUPED
#include "../../../streaming/resolve.hpp"

// The resolve's group words begin here, for the selection's stores and the
// resolve's reads alike.
const uint ROUTED_RESOLVE_GROUP = 0u;

// Choice (row, slot)'s expert at its group word, which the resolve reads.
void routed_group_route(uint64_t row, uint slot, int expert) {
    const uint streaming_group = ROUTED_RESOLVE_GROUP;
    STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(uint(row) * uint(SEISMIC_DIM_K) + slot)) = expert;
}

// Selection and its resolve share one workgroup; a whole group's
// layer (`whole`) resolves its hits and routing record alone. Its threads are
// numbered by SEISMIC_THREAD, as the selection's: a group word stored through
// an address or under control derived from the local index would make every
// group word the selection reads local-derived, which no subgroup operation
// may take.
void routed_resolve(const bool whole) {
    streaming_resolve(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES), SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
        uint(SEISMIC_DIM_M * SEISMIC_DIM_K), SEISMIC_PTR(SEISMIC_BUFFER_STEP), SEISMIC_STEP_STRIDE_0, whole,
        SEISMIC_PTR(SEISMIC_BUFFER_MAP), SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1,
        SEISMIC_PTR(SEISMIC_BUFFER_TIER), SEISMIC_TIER_STRIDE_0, uint(SEISMIC_DIM_TM), uint(SEISMIC_PARAM_SITED),
        SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1,
        SEISMIC_SOURCES_STRIDE_2, SEISMIC_PTR(SEISMIC_BUFFER_ROUTING), SEISMIC_ROUTING_STRIDE_0,
        SEISMIC_ROUTING_STRIDE_1, uint(SEISMIC_DIM_FLIGHT), uint(SEISMIC_DIM_P),
        SEISMIC_PTR(SEISMIC_BUFFER_FILE), SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1,
        SEISMIC_PTR(SEISMIC_BUFFER_POLICY), ROUTED_RESOLVE_GROUP,
        SEISMIC_THREAD, gl_WorkGroupSize.x);
}
