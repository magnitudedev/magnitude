// Rotary-embedding angle evaluation shared by the decoder attention
// (`lib/attention/attention.glsl`) and the vision block (`lib/vision/vision.glsl`).
//
// This file is independent of any entry ABI.

// sin and cos of an F32 rotary angle (|angle| up to the context length). The
// angle is reduced to [-pi, pi] by a three-part 2*pi (Cody-Waite, fused
// products) and evaluated with the hardware functions, which are accurate on
// that interval.
float rotary_sincos(float angle, out float cosine) {
    const float turns = roundEven(angle * 0.15915494309189535);
    float reduced = seismic_fma_rn(-turns, 6.28125, angle);
    reduced = seismic_fma_rn(-turns, 0.0019354820251464844, reduced);
    reduced = seismic_fma_rn(-turns, -1.7484555314695172e-07, reduced);
    cosine = cos(reduced);
    return sin(reduced);
}

// The rotated pairs of a head row, the counterpart of `rotary::layout`
// (`metal/lib/core/rotary.h`, `cuda/lib/core/rotary.cuh`): the row's P pairs
// lie over the 2P columns from `first`: pair p (first + p, first + p + P), or
// (first + 2p, first + 2p + 1) when `adjacent`. The including entry builds
// its layout from its constants (P, and `first` = S when RT, `adjacent` when
// RA).
struct rotary_layout {
    uint p;
    uint first;
    bool adjacent;
};

bool rotary_rotated(uint column, uint p, uint first) {
    return column - first < 2u * p;
}
uint rotary_pair(uint column, uint p, uint first, bool adjacent) {
    const uint j = column - first;
    return adjacent ? j / 2u : j % max(p, 1u);
}
bool rotary_leads(uint column, uint p, uint first, bool adjacent) {
    const uint j = column - first;
    return adjacent ? j % 2u == 0u : j < p;
}
uint rotary_partner(uint column, uint p, uint first, bool adjacent) {
    if (adjacent)
        return column ^ 1u;
    return rotary_leads(column, p, first, adjacent) ? column + p : column - p;
}
// The leading column of pair `q`.
uint rotary_leading(uint q, rotary_layout pairs) {
    return pairs.first + (pairs.adjacent ? 2u * q : q);
}

// Pair `q`'s turn at `coordinates` (an address stepping by
// `coordinate_stride` elements): coordinate axis components[q] at
// frequencies[q], its cosine and sine scaled by amplitudes[q]. Table
// addresses step by their element strides.
void rotary_turn(uint64_t coordinates, uint64_t coordinate_stride, uint64_t components, uint64_t component_stride,
    uint64_t frequencies, uint64_t frequency_stride, uint64_t amplitudes, uint64_t amplitude_stride, uint q,
    out float cosine, out float sine) {
    const int axis = element_i32_at(components + uint64_t(q) * component_stride * 4ul);
    const float angle = float(element_i32_at(coordinates + uint64_t(axis) * coordinate_stride * 4ul))
        * element_f32_at(frequencies + uint64_t(q) * frequency_stride * 4ul);
    float c;
    const float s = rotary_sincos(angle, c);
    const float amplitude = element_f32_at(amplitudes + uint64_t(q) * amplitude_stride * 4ul);
    cosine = c * amplitude;
    sine = s * amplitude;
}

// The lane rotation of one head row held by a logical subgroup: lane l holds
// columns [l E, (l + 1) E), E = `LANE_E` (which the including entry defines).
// The rotated columns are whole lanes (P and `first` multiples of E), so a
// half-split pair's partner lives P / E lanes away and an adjacent pair's in
// the same lane (or, at one column a lane, the neighbouring one).
#ifdef LANE_E

// One row's turns for a lane's rotated columns (`rotary_turn`).
struct rotary_turns {
    float cosine[LANE_E];
    float sine[LANE_E];
};

// The turns at `coordinates` (an address stepping by `coordinate_stride`
// elements).
rotary_turns rotary_lane_turns(uint64_t coordinates, uint64_t coordinate_stride, uint64_t components,
    uint64_t component_stride, uint64_t frequencies, uint64_t frequency_stride, uint64_t amplitudes,
    uint64_t amplitude_stride, rotary_layout pairs, uint lane) {
    const uint p = pairs.p, first = pairs.first;
    const bool adjacent = pairs.adjacent;
    rotary_turns t;
    [[unroll]] for (uint i = 0u; i < LANE_E; ++i) {
        t.cosine[i] = 1.0;
        t.sine[i] = 0.0;
    }
    if (p == 0u || !rotary_rotated(lane * LANE_E, p, first))
        return t;
    [[unroll]] for (uint i = 0u; i < LANE_E; ++i)
        rotary_turn(coordinates, coordinate_stride, components, component_stride, frequencies, frequency_stride,
            amplitudes, amplitude_stride, rotary_pair(lane * LANE_E + i, p, first, adjacent), t.cosine[i], t.sine[i]);
    return t;
}

// Lane `lane`'s columns `x` of the row, its pairs rotated by `t`. The whole
// logical subgroup calls it.
void rotary_lane_rotate(inout float x[LANE_E], rotary_turns t, rotary_layout pairs, uint lane) {
    const uint p = pairs.p, first = pairs.first;
    const bool adjacent = pairs.adjacent;
    if (p == 0u)
        return;
    const bool turned = rotary_rotated(lane * LANE_E, p, first);
    float other[LANE_E];
    if (adjacent && LANE_E > 1u) {
        [[unroll]] for (uint i = 0u; i < LANE_E; ++i)
            other[i] = x[i ^ 1u];
    } else {
        const uint partner_lane = turned ? rotary_partner(lane * LANE_E, p, first, adjacent) / LANE_E : lane;
        [[unroll]] for (uint i = 0u; i < LANE_E; ++i)
            other[i] = seismic_shuffle(x[i], partner_lane);
    }
    if (!turned)
        return;
    [[unroll]] for (uint i = 0u; i < LANE_E; ++i)
        x[i] = rotary_leads(lane * LANE_E + i, p, first, adjacent) ? x[i] * t.cosine[i] - other[i] * t.sine[i]
                                                                   : x[i] * t.cosine[i] + other[i] * t.sine[i];
}

#endif
