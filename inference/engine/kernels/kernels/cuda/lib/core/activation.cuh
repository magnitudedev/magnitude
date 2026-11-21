// `activation::Act`: the activation element `A`, for entries that bind one —
// the engine's element-parameter convention over Seismic's dense element
// types (<seismic/element.cuh>). The counterpart of
// `metal/lib/core/activation.h` and `vulkan/lib/core/activation.glsl`.
//
// An entry with an A element binds it before including this file, as it
// binds a weight slot:
//     #define KERNEL_A SEISMIC_ELEMENT_A
// so a header names only symbols every including entry has.

#include <seismic/element.cuh>

namespace activation {
// The activation element A, when the entry binds one.
#if defined(KERNEL_A)
typedef ELEMENT_OF(KERNEL_A) Act;
#endif

// Four contiguous F32 values into registers: one 16-byte float4 load when
// aligned, four scalar loads otherwise.
__device__ __forceinline__ void load4_f32_aligned(const float *src, float (&values)[4]) {
    if ((reinterpret_cast<element::u64>(src) & 15u) == 0) {
        const float4 word = *reinterpret_cast<const float4 *>(src);
        values[0] = word.x;
        values[1] = word.y;
        values[2] = word.z;
        values[3] = word.w;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            values[i] = src[i];
    }
}

// Four contiguous elements into `out`: vector store via `put_span<4>` when
// aligned (for F32, 16-byte aligned; 16-bit elements check 8-byte alignment
// inside `put_span<4>`), four scalar stores otherwise.
template <class E>
__device__ __forceinline__ void put4_aligned(element::Out<E> out, element::u64 index, const float (&values)[4]) {
    if constexpr (E::bytes == 4) {
        if ((reinterpret_cast<element::u64>(out.stored(index)) & 15u) == 0) {
            out.template put_span<4>(index, values);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                out.put(index + i, values[i]);
        }
    } else {
        out.template put_span<4>(index, values);
    }
}
} // namespace activation
