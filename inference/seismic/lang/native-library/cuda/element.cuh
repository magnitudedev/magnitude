// Seismic native library (CUDA): dense element types, over the CUDA prelude's
// conversions. `#include <seismic/element.cuh>`. Every type converts one
// stored element to F32 (`load`) and rounds F32 to the element (`store`,
// `round`, round-to-nearest-even). Tensors are addressed in elements of their
// own dtype from a byte base: `at`, `put`, `put2`, and `span` (N contiguous
// elements into registers).
//
// The element of a tensor or element parameter is selected from its ABI
// prefix: `ELEMENT_OF(SEISMIC_NORM)`.

// 1 when the macro `<prefix><suffix>` is defined as 1, else 0; usable in both
// `#if` and constant expressions.
#define ELEMENT_CAT_(a, b) a##b
#define ELEMENT_CAT(a, b) ELEMENT_CAT_(a, b)
#define ELEMENT_SECOND_(a, b, ...) b
#define ELEMENT_SECOND(...) ELEMENT_SECOND_(__VA_ARGS__, 0, 0)
#define ELEMENT_PROBE_1 ~, 1
#define ELEMENT_IS_ONE(value) ELEMENT_SECOND(ELEMENT_CAT(ELEMENT_PROBE_, value))
#define ELEMENT_HAS(prefix, suffix) ELEMENT_IS_ONE(ELEMENT_CAT(prefix, suffix))

namespace element {

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

struct Bf16 {
    typedef u16 storage;
    static constexpr u32 bytes = 2;
    __device__ static __forceinline__ float load(storage value) { return seismic_bf16_to_f32(value); }
    __device__ static __forceinline__ storage store(float value) { return seismic_f32_to_bf16(value); }
    __device__ static __forceinline__ float round(float value) { return load(store(value)); }
    // Two elements in 32 bits, the first in the low half.
    __device__ static __forceinline__ u32 pack2(float first, float second) {
        return seismic_pack_bf16x2(first, second);
    }
    __device__ static __forceinline__ float2 unpack2(u32 bits) { return seismic_unpack_bf16x2(bits); }
};

struct F16 {
    typedef u16 storage;
    static constexpr u32 bytes = 2;
    __device__ static __forceinline__ float load(storage value) { return seismic_f16_to_f32(value); }
    __device__ static __forceinline__ storage store(float value) { return seismic_f32_to_f16(value); }
    __device__ static __forceinline__ float round(float value) { return load(store(value)); }
    __device__ static __forceinline__ u32 pack2(float first, float second) {
        return seismic_pack_f16x2(first, second);
    }
    __device__ static __forceinline__ float2 unpack2(u32 bits) { return seismic_unpack_f16x2(bits); }
};

struct F32 {
    typedef float storage;
    static constexpr u32 bytes = 4;
    __device__ static __forceinline__ float load(storage value) { return value; }
    __device__ static __forceinline__ storage store(float value) { return value; }
    __device__ static __forceinline__ float round(float value) { return value; }
};

// Element `index` of a dense tensor of element type E.
template <class E> __device__ __forceinline__ float at(const u8 *base, u64 index) {
    return E::load(reinterpret_cast<const typename E::storage *>(base)[index]);
}

template <class E> __device__ __forceinline__ void put(u8 *base, u64 index, float value) {
    reinterpret_cast<typename E::storage *>(base)[index] = E::store(value);
}

// Elements index and index + 1: one vector store when index is even.
template <class E> __device__ __forceinline__ void put2(u8 *base, u64 index, float first, float second) {
    if (index % 2 != 0) {
        put<E>(base, index, first);
        put<E>(base, index + 1, second);
    } else if constexpr (E::bytes == 4) {
        reinterpret_cast<float2 *>(base)[index / 2] = make_float2(first, second);
    } else {
        reinterpret_cast<u32 *>(base)[index / 2] = E::pack2(first, second);
    }
}

// The elements of E in 32 bits (one f32 or two 16-bit elements) into `out`.
template <class E> __device__ __forceinline__ void unpack(u32 bits, float *out) {
    if constexpr (E::bytes == 4) {
        out[0] = __uint_as_float(bits);
    } else {
        const float2 pair = E::unpack2(bits);
        out[0] = pair.x;
        out[1] = pair.y;
    }
}

// `N` contiguous elements starting at `index` into F32 registers, with the
// widest aligned vector loads the span size allows. The span start is aligned
// to the span's byte size (callers index whole spans of a canonical row).
// `NC` selects the read-only non-coherent path, admitted only for memory no
// thread writes during the launch.
template <class E, int N, bool NC>
__device__ __forceinline__ void span(const u8 *base, u64 index, float (&out)[N]) {
    constexpr int bytes = N * E::bytes;
    constexpr int per_word = 4 / E::bytes;
    const u8 *address = base + index * E::bytes;
    if constexpr (bytes % 16 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < bytes / 16; ++chunk) {
            uint4 word;
            if constexpr (NC) {
                word = seismic_ld_nc_v4(address + chunk * 16);
            } else {
                word = *reinterpret_cast<const uint4 *>(address + chunk * 16);
            }
            unpack<E>(word.x, out + chunk * 4 * per_word + 0 * per_word);
            unpack<E>(word.y, out + chunk * 4 * per_word + 1 * per_word);
            unpack<E>(word.z, out + chunk * 4 * per_word + 2 * per_word);
            unpack<E>(word.w, out + chunk * 4 * per_word + 3 * per_word);
        }
    } else if constexpr (bytes % 8 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < bytes / 8; ++chunk) {
            uint2 word;
            if constexpr (NC) {
                word = seismic_ld_nc_v2(address + chunk * 8);
            } else {
                word = *reinterpret_cast<const uint2 *>(address + chunk * 8);
            }
            unpack<E>(word.x, out + chunk * 2 * per_word + 0 * per_word);
            unpack<E>(word.y, out + chunk * 2 * per_word + 1 * per_word);
        }
    } else if constexpr (bytes % 4 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < bytes / 4; ++chunk) {
            const u32 word = NC ? seismic_ld_nc_u32(address + chunk * 4)
                                : *reinterpret_cast<const u32 *>(address + chunk * 4);
            unpack<E>(word, out + chunk * per_word);
        }
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = at<E>(base, index + i);
        }
    }
}

// `N` contiguous F32 values (global or shared) into registers and back, in
// 16-byte vectors when N % 4 == 0.
template <int N> __device__ __forceinline__ void f32_span(const float *base, float (&out)[N]) {
    if constexpr (N % 4 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < N / 4; ++chunk) {
            const float4 word = reinterpret_cast<const float4 *>(base)[chunk];
            out[chunk * 4 + 0] = word.x;
            out[chunk * 4 + 1] = word.y;
            out[chunk * 4 + 2] = word.z;
            out[chunk * 4 + 3] = word.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = base[i];
        }
    }
}

template <int N> __device__ __forceinline__ void f32_span_store(float *base, const float (&values)[N]) {
    if constexpr (N % 4 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < N / 4; ++chunk) {
            reinterpret_cast<float4 *>(base)[chunk] =
                make_float4(values[chunk * 4], values[chunk * 4 + 1], values[chunk * 4 + 2], values[chunk * 4 + 3]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            base[i] = values[i];
        }
    }
}

// An F32 scalar argument from its argument word.
__device__ __forceinline__ float word_f32(u64 word) { return __uint_as_float(static_cast<u32>(word)); }

// The element type of a dense representation kind: 0 = f32, 1 = bf16,
// 2 = f16. Other kinds have none.
template <int KIND> struct Of {
    static_assert(KIND >= 0, "a dense f32, bf16 or f16 representation is required");
};
template <> struct Of<0> { typedef F32 type; };
template <> struct Of<1> { typedef Bf16 type; };
template <> struct Of<2> { typedef F16 type; };

} // namespace element

#define ELEMENT_KIND(prefix)                                                                       \
    (ELEMENT_HAS(prefix, _REPRESENTATION_F32)    ? 0                                               \
     : ELEMENT_HAS(prefix, _REPRESENTATION_BF16) ? 1                                               \
     : ELEMENT_HAS(prefix, _REPRESENTATION_F16)  ? 2                                               \
                                                 : -1)
// The element type of the dense tensor or element parameter `prefix`.
#define ELEMENT_OF(prefix) typename element::Of<ELEMENT_KIND(prefix)>::type
