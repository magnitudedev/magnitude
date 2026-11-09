// Rotary position embeddings of one head row held in a simdgroup's registers
// (`rotate_pairs`, attention.seismic): the reduced-angle sine and cosine, the
// layout of a row's rotated pairs over its 32 lanes, a row's amplitude-scaled
// table and the rotation itself. Shared by the attention library, the pooled
// rows and the indexers.
//
// This file is independent of any entry ABI: a caller names its layout with
// `rotary::layout<W, P, FIRST, ADJACENT>`.

namespace rotary {

// sin and cos of an F32 rotary angle (|angle| up to the context length). The
// angle is reduced exactly enough to [-pi, pi] by a three-part 2*pi
// (Cody-Waite, fused products) and evaluated with the fast functions, which
// are accurate on that interval; the precise library functions cost tens of
// microseconds per head row here.
inline float sincos(float angle, thread float &cosine) {
    const float turns = metal::rint(angle * 0.15915494309189535f);
    float reduced = metal::fma(-turns, 6.28125f, angle);
    reduced = metal::fma(-turns, 0.0019354820251464844f, reduced);
    reduced = metal::fma(-turns, -1.7484555314695172e-07f, reduced);
    cosine = metal::fast::cos(reduced);
    return metal::fast::sin(reduced);
}

// A head row of W columns over a simdgroup, lane l holding columns
// [l E, (l + 1) E), E = W / 32: its P rotary pairs over the 2P columns from
// FIRST, pair p (FIRST + p, FIRST + p + P), or (FIRST + 2p, FIRST + 2p + 1)
// when ADJACENT.
template <uint W, uint P, uint FIRST, bool ADJACENT>
struct layout {
    // An enumerator: Metal admits no static data member at program scope.
    enum : uint { E = W / 32 };
    static_assert(W % 32 == 0, "a head row must cover whole lanes");
    static_assert(P % E == 0, "each rotary half must cover whole lanes");
    static_assert(!ADJACENT || E % 2 == 0 || E == 1, "an adjacent pair must lie within one lane or span two");

    // The place of `column` among the rotated columns: whether it is
    // rotated, its pair, whether it leads the pair, its pair's other column.
    static bool rotated(uint column) {
        return column - FIRST < 2 * P;
    }
    static uint pair(uint column) {
        const uint j = column - FIRST;
        return ADJACENT ? j / 2 : j % P;
    }
    static bool leads(uint column) {
        const uint j = column - FIRST;
        return ADJACENT ? j % 2 == 0 : j < P;
    }
    static uint partner(uint column) {
        if (ADJACENT)
            return column ^ 1u;
        return leads(column) ? column + P : column - P;
    }

    // One row's table for lane `lane`'s rotated columns: pair p turns by
    // coordinate axis components[p] at frequencies[p], its cosine and sine
    // scaled by amplitudes[p]. The same table rotates every head of the row.
    struct table {
        float cosine[E];
        float sine[E];
        table(device const int *coordinates, device const int *components, device const float *frequencies,
            device const float *amplitudes, uint lane) {
            if (P == 0 || !rotated(lane * E))
                return;
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < E; ++i) {
                const uint p = pair(lane * E + i);
                float c;
                const float s = sincos(float(coordinates[components[p]]) * frequencies[p], c);
                cosine[i] = c * amplitudes[p];
                sine[i] = s * amplitudes[p];
            }
        }
    };

    // Lane `lane`'s columns `x` of the row, its pairs rotated by `t`. The
    // rotated columns are whole lanes; a half-split pair's partner lives
    // P / E lanes away, an adjacent pair's in the same lane (or, at one
    // column a lane, the neighbouring one). The whole simdgroup calls it.
    static void rotate(thread const table &t, uint lane, thread float (&x)[E]) {
        if (P == 0)
            return;
        const bool turned = rotated(lane * E);
        float other[E];
        if (ADJACENT && E > 1) {
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < E; ++i)
                other[i] = x[i ^ 1u];
        } else {
            const uint partner_lane = turned ? partner(lane * E) / E : lane;
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < E; ++i)
                other[i] = simd_shuffle(x[i], ushort(partner_lane));
        }
        if (!turned)
            return;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < E; ++i) {
            x[i] = leads(lane * E + i) ? x[i] * t.cosine[i] - other[i] * t.sine[i]
                                       : x[i] * t.cosine[i] + other[i] * t.sine[i];
        }
    }
};

} // namespace rotary
