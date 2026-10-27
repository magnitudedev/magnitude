// Seismic native library (Metal): the dialect prelude of the shared
// `repack.h` (`<seismic/repack.h>`), whose docs this follows.
#pragma once

#define SEISMIC_REPACK_FUNCTION constexpr
#define SEISMIC_REPACK_LOAD(address) (*(address))
#define SEISMIC_REPACK_TABLE constant const uint *
#define SEISMIC_REPACK_INLINE inline
#define SEISMIC_REPACK_PRIVATE thread
#define SEISMIC_REPACK_STATIC static constant constexpr
#define SEISMIC_REPACK_LOOP _Pragma("clang loop unroll(full)")

namespace repack {

// The value ops' float arithmetic, on float bits.
inline uint f32_multiply(uint a, uint b) { return as_type<uint>(as_type<float>(a) * as_type<float>(b)); }
inline uint f16_to_f32(uint bits) { return as_type<uint>(float(as_type<half>(ushort(bits)))); }
inline uint i32_to_f32(uint bits) { return as_type<uint>(float(as_type<int>(bits))); }

}  // namespace repack
