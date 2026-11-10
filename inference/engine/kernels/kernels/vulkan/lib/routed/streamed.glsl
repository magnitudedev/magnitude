#include "../streaming/control.glsl"
#include "../../../streaming/resolve.hpp"
#include "../../../streaming/hint.hpp"

// Selection and its resolve share one workgroup, so the protocol's
// publication barrier covers the routes it just wrote; a whole group's
// layer (`whole`) resolves its hits and routing record alone.
void routed_resolve(const bool whole) {
    streaming_resolve(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES), SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
        uint(SEISMIC_DIM_M * SEISMIC_DIM_K), SEISMIC_PTR(SEISMIC_BUFFER_STEP), SEISMIC_STEP_STRIDE_0, whole,
        SEISMIC_PTR(SEISMIC_BUFFER_MAP), SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1,
        SEISMIC_PTR(SEISMIC_BUFFER_TIER), SEISMIC_TIER_STRIDE_0, uint(SEISMIC_DIM_TM), uint(SEISMIC_PARAM_SITED),
        SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1,
        SEISMIC_SOURCES_STRIDE_2, SEISMIC_PTR(SEISMIC_BUFFER_ROUTING), SEISMIC_ROUTING_STRIDE_0,
        SEISMIC_ROUTING_STRIDE_1, uint(SEISMIC_DIM_FLIGHT), uint(SEISMIC_DIM_P),
        SEISMIC_PTR(SEISMIC_BUFFER_FILE), SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1,
        SEISMIC_PTR(SEISMIC_BUFFER_POLICY), 0u,
        gl_LocalInvocationIndex, gl_WorkGroupSize.x);
}

// Lead `lead`'s hints into its ring row, under the step's low word.
void routed_hint_publish(const uint lead) {
    streaming_hint(uint(SEISMIC_DIM_M), SEISMIC_PTR(SEISMIC_BUFFER_RINGS) + uint64_t(lead) * SEISMIC_RINGS_STRIDE_0 * 4ul,
        SEISMIC_RINGS_STRIDE_1,
        streaming_device_load(SEISMIC_PTR(SEISMIC_BUFFER_STEP) + SEISMIC_CONST_STEP_LOW * SEISMIC_STEP_STRIDE_0 * 4ul),
        0u, gl_LocalInvocationIndex, gl_WorkGroupSize.x);
}
