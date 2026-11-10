// The order-preserving key of an F32 (`ordered_key`, sampling.seismic): a
// larger float has a larger key, -0.0 and 0.0 share one key, every NaN has the
// one key 0xFFC00000 above +inf's, and every key of a float is nonzero. Used by
// the indexer's top-K. The counterpart of `metal/lib/core/order.h` and
// `cuda/lib/core/order.cuh`.
//
// This file is independent of any entry ABI.

// The key of a float its caller holds to be no NaN.
uint order_finite_key(float value) {
    const uint bits = floatBitsToUint(value + 0.0);
    return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
}

uint order_key(float value) {
    return isnan(value) ? 0xFFC00000u : order_finite_key(value);
}
