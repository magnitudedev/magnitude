// The order-preserving key of an F32 (`ordered_key`, sampling.seismic): a
// larger non-NaN float has a larger key, -0.0 and 0.0 share one key, every
// key of a non-NaN float is nonzero, and every NaN has the one key
// 0xFFC00000 (above +inf), as on every backend. Shared by the sampling
// shape and the indexer's top-K; `value` also by the progressive readout.
//
// This file is independent of any entry ABI.

namespace order {

// The key of a float its caller holds to be no NaN (the sampling shape's,
// which refuses a row holding one before it keys any value).
inline uint finite_key(float value) {
    const uint bits = as_type<uint>(value + 0.0f);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

inline uint key(float value) {
    return metal::isnan(value) ? 0xffc00000u : finite_key(value);
}

// The float of the key `k` of a non-NaN float (0.0 for the shared zero key).
inline float value(uint k) {
    return as_type<float>((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

} // namespace order
