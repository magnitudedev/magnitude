// The general indexer's Metal helpers (`indexer_keys`, `indexer_score`,
// `rows_topk`; contracts in indexer.seismic): a head row prepared over a
// simdgroup (its RMS norm and rotary, as the portable `head_rotary`), and
// the threadgroup scan the selection places its picks with. The helpers
// that read a layout take it as template arguments, so this file names no
// entry ABI symbol.
//
// `sincos`, the lane rotary and `group_exclusive` are the counterparts of
// `indexer_select.metal`'s, which leave with that entry (the Qwen4
// migration onto these entries).

namespace indexer {

// The rotary angle's sine and cosine, reduced as the attention entries reduce
// it (`attention::sincos`): to [-pi, pi] by a three-part 2*pi, then the fast
// functions, accurate on that interval.
inline float sincos(float angle, thread float &cosine) {
    const float turns = metal::rint(angle * 0.15915494309189535f);
    float reduced = metal::fma(-turns, 6.28125f, angle);
    reduced = metal::fma(-turns, 0.0019354820251464844f, reduced);
    reduced = metal::fma(-turns, -1.7484555314695172e-07f, reduced);
    cosine = metal::fast::cos(reduced);
    return metal::fast::sin(reduced);
}

// A head row of W = 32 E columns over a simdgroup, lane l holding columns
// [l E, (l + 1) E): its pairs' layout (`rotate_pairs`) is P pairs over the
// 2P columns from FIRST (S with trailing pairs, else 0), pair p turning
// (2p, 2p + 1) when ADJACENT, else (p, p + P). The rotated columns are whole
// lanes (P is a multiple of E); an adjacent pair lies within a lane (E even)
// or spans two (E = 1).
template <uint E, uint P, uint FIRST, bool ADJACENT>
struct Lanes {
    static constant constexpr uint W = 32 * E;

    // RMS-normalized with `norm` (the portable `head_norm`): x * inv * norm,
    // inv = rsqrt(sum(x^2) / W + epsilon). The whole simdgroup calls it.
    static void normalize(thread float (&x)[E], device const float *norm, float epsilon, uint lane) {
        float squares = 0.0f;
        for (uint i = 0; i < E; ++i)
            squares = metal::fma(x[i], x[i], squares);
        squares = simd_sum(squares);
        const float inverse = metal::rsqrt(squares / float(W) + epsilon);
        for (uint i = 0; i < E; ++i)
            x[i] = x[i] * inverse * norm[lane * E + i];
    }

    // Its pairs turned by the row's rotary at `coordinates`: lo' = lo c - hi s,
    // hi' = hi c + lo s, c and s the cosine and sine of the pair's angle
    // times its amplitude. The whole simdgroup calls it.
    static void rotate(thread float (&x)[E], device const int *coordinates, device const int *components,
        device const float *frequencies, device const float *amplitudes, uint lane) {
        if (P == 0)
            return;
        const uint first = lane * E;
        const bool rotated = first - FIRST < 2 * P;
        float partner[E];
        if (ADJACENT && E > 1) {
            for (uint i = 0; i < E; ++i)
                partner[i] = x[i ^ 1u];
        } else {
            const uint offset = first - FIRST;
            const uint other = ADJACENT ? (first ^ 1u) : (offset < P ? first + P : first - P);
            const uint partner_lane = rotated ? other / E : lane;
            for (uint i = 0; i < E; ++i)
                partner[i] = simd_shuffle(x[i], ushort(partner_lane));
        }
        if (!rotated)
            return;
        for (uint i = 0; i < E; ++i) {
            const uint j = first + i - FIRST;
            const uint pair = ADJACENT ? j / 2 : j % P;
            const bool leads = ADJACENT ? j % 2 == 0 : j < P;
            float c;
            float s = sincos(float(coordinates[components[pair]]) * frequencies[pair], c);
            c *= amplitudes[pair];
            s *= amplitudes[pair];
            x[i] = leads ? x[i] * c - partner[i] * s : x[i] * c + partner[i] * s;
        }
    }
};

// The exclusive prefix of `value` over the 256 threads of a threadgroup in
// thread order, and in `total` their sum. Every thread calls it; `totals`
// holds the 8 simdgroups' sums.
inline uint group_exclusive(uint value, threadgroup uint *totals, uint simd, uint lane, thread uint &total) {
    const uint inclusive = simd_prefix_inclusive_sum(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 31)
        totals[simd] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0;
    total = 0;
    for (uint other = 0; other < 8; ++other) {
        const uint sum = totals[other];
        before += other < simd ? sum : 0u;
        total += sum;
    }
    return before + inclusive - value;
}

// The length of span [lo, hi): its rows, or none when it is empty.
inline uint span_length(int lo, int hi) {
    return hi > lo ? uint(hi - lo) : 0u;
}

} // namespace indexer
