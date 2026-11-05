// How every expert selection holds and orders its experts (`rank_experts`,
// functions.seismic): `routed_route`'s `routed_select_row` and
// `routed_select` (select.cuh). The counterpart of
// `metal/lib/routed/ranked.h`, `vulkan/lib/routed/ranked.glsl` and
// `cpu/lib/routed/ranked.rs`.
//   candidates: lane `lane` of a warp holds experts lane + 32 i,
//     ceil(E / 32) registers; when E is not a multiple of 32, lanes at or
//     past E % 32 hold one fewer, and their last register is an excluded
//     entry, held as a taken one (`routed_select`: ranked -inf, expert -1;
//     the route: `out`).
//   ranked: a NaN as -inf. A bit test, which no math mode folds.
//   precedes: descending ranked value, ties to the higher expert index.
//     Over ranked values it is a total order, so every rank names an
//     expert whatever a row holds, and an excluded entry precedes nothing.
//   out: an entry out of the ranking, for a selection that keeps no expert
//     array (`routed_select_row`): a NaN, which no ranked value is, so
//     `precedes` neither chooses it nor lets it hold a best. NVRTC compiles
//     without fast math, and PTX comparisons of a NaN are false.

namespace selection {

__device__ __forceinline__ float out() {
    return __int_as_float(0x7fc00000);
}

// The experts lane `lane` holds of E: ceil(E / 32) on the first E % 32
// lanes, E / 32 on the others (a constant, E / 32, when 32 divides E).
template <unsigned E> __device__ __forceinline__ unsigned candidates(unsigned lane) {
    return E / 32 + (E % 32 != 0 && lane < E % 32 ? 1u : 0u);
}

__device__ __forceinline__ float ranked(float value) {
    return (__float_as_uint(value) & 0x7fffffffu) > 0x7f800000u ? -__int_as_float(0x7f800000) : value;
}

__device__ __forceinline__ bool precedes(float value, int expert, float best, int best_expert) {
    return value > best || (value == best && expert > best_expert);
}

} // namespace selection
