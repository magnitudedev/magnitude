// Seismic native library (CUDA): the dialect prelude of the shared
// `repack.h` (`<seismic/repack.cuh>`), whose docs this follows.
#pragma once

#define SEISMIC_REPACK_FUNCTION __host__ __device__ constexpr
// A table word: at run time through the read-only cache, whose load
// (`__ldg`) is no constant expression, and in constant evaluation a plain
// read. NVRTC compiles C++17 (`nvrtc::COMPILE_OPTIONS`), where its front end
// provides `__builtin_is_constant_evaluated` (libcu++ gates its own use of it
// on C++14 or later for NVRTC).
#define SEISMIC_REPACK_LOAD(address) (__builtin_is_constant_evaluated() ? *(address) : __ldg(address))
#define SEISMIC_REPACK_TABLE const unsigned int *
#define SEISMIC_REPACK_INLINE __device__ __forceinline__
#define SEISMIC_REPACK_PRIVATE
#define SEISMIC_REPACK_STATIC static constexpr
#define SEISMIC_REPACK_LOOP _Pragma("unroll")

namespace repack {

typedef unsigned int uint;

// The value ops' float arithmetic, on float bits, rounded to nearest (no
// contraction into a fused multiply-add).
__device__ __forceinline__ uint f32_multiply(uint a, uint b) {
    return __float_as_uint(__fmul_rn(__uint_as_float(a), __uint_as_float(b)));
}
__device__ __forceinline__ uint f16_to_f32(uint bits) { return __float_as_uint(seismic_f16_to_f32((unsigned short)bits)); }
__device__ __forceinline__ uint i32_to_f32(uint bits) { return __float_as_uint(__int2float_rn(int(bits))); }

}  // namespace repack
