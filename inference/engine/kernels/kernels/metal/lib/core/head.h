// A head row held by a simdgroup: its RMS norm (the portable `head_norm`),
// shared by the attention library, the pooled rows and the indexers. Its
// rotation is `rotary::layout`'s, its rounding `rounding::round_lanes`. The
// counterpart of `cuda/lib/core/head.cuh`.
//
// This file is independent of any entry ABI.

namespace head {

// A head row of 32 E columns over a simdgroup, lane l holding columns
// [l E, (l + 1) E), RMS-normalized with `norm`: x * inv * norm,
// inv = rsqrt(sum(x^2) / W + epsilon). The whole simdgroup calls it.
template <uint E>
inline void rms_norm(thread float (&x)[E], device const float *norm, float epsilon, uint lane) {
    float squares = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < E; ++i)
        squares = metal::fma(x[i], x[i], squares);
    squares = simd_sum(squares);
    const float inverse = metal::rsqrt(squares / float(32 * E) + epsilon);
    if constexpr (E % 4 == 0) {
        device const float *lane_norm = norm + lane * E;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < E; i += 4) {
            const float4 w = float4(*reinterpret_cast<device const packed_float4 *>(lane_norm + i));
            x[i + 0] = x[i + 0] * inverse * w.x;
            x[i + 1] = x[i + 1] * inverse * w.y;
            x[i + 2] = x[i + 2] * inverse * w.z;
            x[i + 3] = x[i + 3] * inverse * w.w;
        }
    } else {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < E; ++i)
            x[i] = x[i] * inverse * norm[lane * E + i];
    }
}

} // namespace head
