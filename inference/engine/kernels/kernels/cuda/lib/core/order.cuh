// The order-preserving key of an F32 (`ordered_key`, sampling.seismic): a
// larger float has a larger key, -0.0 and 0.0 share one key, every NaN has
// the one key 0xFFC00000 above +inf's, and every key of a float is nonzero.
// Shared by the sampling shape and the indexer's top-K. The counterpart of
// `metal/lib/core/order.h`.
//
// This file is independent of any entry ABI.

#pragma once

namespace order {

// The key of a float its caller holds to be no NaN (the sampling shape's,
// which refuses a row holding one before it keys any value).
__device__ __forceinline__ unsigned finite_key(float value) {
    const unsigned bits = __float_as_uint(value + 0.0f);
    return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
}

__device__ __forceinline__ unsigned key(float value) {
    return isnan(value) ? 0xFFC00000u : finite_key(value);
}

// The float of non-NaN key `k` (0.0 for the shared zero key).
__device__ __forceinline__ float value(unsigned k) {
    return __uint_as_float((k & 0x80000000u) != 0u ? k & 0x7fffffffu : ~k);
}

}  // namespace order
