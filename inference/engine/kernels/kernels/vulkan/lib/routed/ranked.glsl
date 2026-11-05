// How every expert selection holds and orders its experts (`rank_experts`,
// functions.seismic): `routed_route_select` and `routed_select`
// (select.glsl). The counterpart of `metal/lib/routed/ranked.h`,
// `cuda/lib/routed/ranked.cuh` and `cpu/lib/routed/ranked.rs`.
//   selection_candidates: lane `lane` of a subgroup holds experts
//     lane + 32 i, ceil(E / 32) registers; when E is not a multiple of 32,
//     lanes at or past E % 32 hold one fewer, and their last register is an
//     excluded entry (ranked -inf, expert -1, as a taken one).
//   selection_ranked: a NaN as -inf. A bit test, which no math mode folds.
//   selection_precedes: descending ranked value, ties to the higher expert
//     index. Over ranked values it is a total order, so every rank names an
//     expert whatever a row holds, and an excluded entry precedes nothing.

// The experts lane `lane` holds of `experts`: ceil(E / 32) on the first
// E % 32 lanes, E / 32 on the others (a constant, E / 32, when 32 divides a
// constant E).
uint selection_candidates(const uint experts, const uint lane) {
    return experts / 32u + ((experts % 32u) != 0u && lane < experts % 32u ? 1u : 0u);
}

float selection_ranked(float value) {
    return (floatBitsToUint(value) & 0x7fffffffu) > 0x7f800000u ? -uintBitsToFloat(0x7f800000u) : value;
}

bool selection_precedes(float value, int expert, float best, int best_expert) {
    return value > best || (value == best && expert > best_expert);
}
