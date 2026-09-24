// Shared device code of the CUDA token mixers (attention and gated delta
// recurrence): activation-element access, span loads into registers and
// small numeric helpers. The activation element is `SEISMIC_ELEMENT_A`
// (f32, f16 or bf16). Every buffer is addressed in elements of its own dtype.

namespace mx {

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#define MX_ACT_BYTES 4
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16) || defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
#define MX_ACT_BYTES 2
#else
#error "token mixers need a dense f32, f16 or bf16 activation element"
#endif

// One activation element to F32.
__device__ __forceinline__ float act_load(const u8 *base, u64 element) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return reinterpret_cast<const float *>(base)[element];
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return seismic_f16_to_f32(reinterpret_cast<const u16 *>(base)[element]);
#else
    return seismic_bf16_to_f32(reinterpret_cast<const u16 *>(base)[element]);
#endif
}

// F32 rounded to the activation element (round to nearest even), as F32.
__device__ __forceinline__ float act_round(float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return seismic_f16_to_f32(seismic_f32_to_f16(value));
#else
    return seismic_bf16_to_f32(seismic_f32_to_bf16(value));
#endif
}

__device__ __forceinline__ void act_store(u8 *base, u64 element, float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    reinterpret_cast<float *>(base)[element] = value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    reinterpret_cast<u16 *>(base)[element] = seismic_f32_to_f16(value);
#else
    reinterpret_cast<u16 *>(base)[element] = seismic_f32_to_bf16(value);
#endif
}

// Unpack 32 bits of activation elements (two 16-bit or one f32) into `out`.
__device__ __forceinline__ void act_unpack(u32 bits, float *out) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    out[0] = __uint_as_float(bits);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    const float2 pair = seismic_unpack_f16x2(bits);
    out[0] = pair.x;
    out[1] = pair.y;
#else
    const float2 pair = seismic_unpack_bf16x2(bits);
    out[0] = pair.x;
    out[1] = pair.y;
#endif
}

#define MX_PER_WORD (4 / MX_ACT_BYTES)

// `N` contiguous activation elements starting at `element` into F32 registers,
// with the widest aligned vector loads the span size allows. The span start
// is aligned to the span's byte size (callers index whole spans of a
// canonical row). `NC` selects the read-only non-coherent path, admitted only
// for memory no thread writes during the launch.
template <int N, bool NC>
__device__ __forceinline__ void act_span(const u8 *base, u64 element, float (&out)[N]) {
    constexpr int bytes = N * MX_ACT_BYTES;
    const u8 *address = base + element * MX_ACT_BYTES;
    if constexpr (bytes % 16 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < bytes / 16; ++chunk) {
            uint4 word;
            if constexpr (NC) {
                word = seismic_ld_nc_v4(address + chunk * 16);
            } else {
                word = *reinterpret_cast<const uint4 *>(address + chunk * 16);
            }
            act_unpack(word.x, out + chunk * 4 * MX_PER_WORD + 0 * MX_PER_WORD);
            act_unpack(word.y, out + chunk * 4 * MX_PER_WORD + 1 * MX_PER_WORD);
            act_unpack(word.z, out + chunk * 4 * MX_PER_WORD + 2 * MX_PER_WORD);
            act_unpack(word.w, out + chunk * 4 * MX_PER_WORD + 3 * MX_PER_WORD);
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
            act_unpack(word.x, out + chunk * 2 * MX_PER_WORD + 0 * MX_PER_WORD);
            act_unpack(word.y, out + chunk * 2 * MX_PER_WORD + 1 * MX_PER_WORD);
        }
    } else if constexpr (bytes % 4 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < bytes / 4; ++chunk) {
            const u32 word = NC ? seismic_ld_nc_u32(address + chunk * 4)
                                : *reinterpret_cast<const u32 *>(address + chunk * 4);
            act_unpack(word, out + chunk * MX_PER_WORD);
        }
    } else {
#pragma unroll
        for (int index = 0; index < N; ++index) {
            out[index] = act_load(base, element + index);
        }
    }
}

// `N` contiguous F32 values into registers (16-byte vectors when N % 4 == 0).
template <int N>
__device__ __forceinline__ void f32_span(const float *base, float (&out)[N]) {
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
        for (int index = 0; index < N; ++index) {
            out[index] = base[index];
        }
    }
}

template <int N>
__device__ __forceinline__ void f32_span_store(float *base, const float (&values)[N]) {
    if constexpr (N % 4 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < N / 4; ++chunk) {
            reinterpret_cast<float4 *>(base)[chunk] =
                make_float4(values[chunk * 4], values[chunk * 4 + 1], values[chunk * 4 + 2],
                            values[chunk * 4 + 3]);
        }
    } else {
#pragma unroll
        for (int index = 0; index < N; ++index) {
            base[index] = values[index];
        }
    }
}

__device__ __forceinline__ float word_f32(u64 word) {
    return __uint_as_float(static_cast<u32>(word));
}

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

}  // namespace mx
