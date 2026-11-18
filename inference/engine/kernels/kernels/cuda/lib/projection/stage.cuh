// stage shared projection mechanisms; included at the original declaration point.
// ---------------------------------------------------------------------------
// Prologues. Each forms x[m, k] rounded to A from FACTORS per-row factors:
//   prepare_row(shared, m, tmp) whole block: factors of row m at shared[0..]
//   value(factors, m, k)        x[m, k]
// A GEMV reads its operands through `pair(factors, m, k)` (the operand bits
// of (x[m, k], x[m, k + 1])), which only the prologues read in place provide.
// `STAGED` is false when the input can be read in place.

template <class E, class Rows> struct Plain {
    const u8 *x;
    u64 stride;
    Rows rows;
    static constexpr bool STAGED = !Same<E, Act>::value || !Same<Rows, AllRows>::value;
    static constexpr int FACTORS = 0;
    __device__ __forceinline__ void prepare_row(float *, u32, float *) const {}
    __device__ __forceinline__ float value(const float *, u32 m, u64 k) const {
        return Act::round(element::at<E>(x, rows(m) * stride + k));
    }
    // A view's base and stride are only element-aligned: the wide loads are
    // taken where they hold, single 16-bit loads elsewhere.
    __device__ __forceinline__ u32 pair(const float *, u32 m, u64 k) const {
        static_assert(!STAGED, "a GEMV reads staged rows");
        const u8 *at = x + (rows(m) * stride + k) * 2;
        if (stride % 2 == 0 && ((u64)x & 3) == 0)
            return *reinterpret_cast<const u32 *>(at);
        const u16 *values = reinterpret_cast<const u16 *>(at);
        return (u32)values[0] | (u32)values[1] << 16;
    }
    // The operand bits of x[m, k .. k + 7], k a multiple of 8 (`gemv_rows`).
    __device__ __forceinline__ uint4 octet(const float *, u32 m, u64 k) const {
        static_assert(!STAGED, "a GEMV reads staged rows");
        const u8 *at = x + (rows(m) * stride + k) * 2;
        if (stride % 8 == 0 && ((u64)x & 15) == 0)
            return *reinterpret_cast<const uint4 *>(at);
        const u16 *values = reinterpret_cast<const u16 *>(at);
        u32 w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            w[j] = (u32)values[2 * j] | (u32)values[2 * j + 1] << 16;
        return make_uint4(w[0], w[1], w[2], w[3]);
    }
};

// A scalar source with independent row and column strides. It publishes
// semantic Act into owned staging; it is never a direct tensor operand view.
template <class E, class Rows> struct Strided {
    const u8 *x;
    u64 row_stride;
    u64 column_stride;
    Rows rows;
    static constexpr bool STAGED = true;
    static constexpr int FACTORS = 0;
    __device__ __forceinline__ void prepare_row(float *, u32, float *) const {}
    __device__ __forceinline__ float value(const float *, u32 m, u64 k) const {
        return Act::round(element::at<E>(x, rows(m) * row_stride + k * column_stride));
    }
};

// RMS normalization of an F32 residual: x = round_A(t * rsqrt(sum t^2 / K + eps) * w).
template <class NORM, class Rows> struct Rms {
    const float *x;
    u64 stride;
    const u8 *norm;
    float eps;
    u64 width;
    Rows rows;
    static constexpr bool STAGED = true;
    static constexpr int FACTORS = 1;
    __device__ __forceinline__ float squares(u32 m, u32 first, u32 step) const {
        const float *row = x + rows(m) * stride;
        float total = 0.0f;
        for (u64 k = first; k < width; k += step)
            total = seismic_fma_rn(row[k], row[k], total);
        return total;
    }
    __device__ __forceinline__ float inverse(float total) const { return rsqrtf(total / (float)width + eps); }
    __device__ __forceinline__ void prepare_row(float *shared, u32 m, float *scratch) const {
        const float total = reduce::group_sum(squares(m, threadIdx.x, blockDim.x), scratch);
        if (threadIdx.x == 0)
            shared[0] = inverse(total);
    }
    __device__ __forceinline__ float formed(float inv, u32 m, u64 k) const {
        return x[rows(m) * stride + k] * inv * element::at<NORM>(norm, k);
    }
    __device__ __forceinline__ float value(const float *factors, u32 m, u64 k) const {
        return Act::round(formed(factors[0], m, k));
    }
};

// The sum in part order of the PARTS width-wide parts of an F32 row, a
// hyper-connection mix site's per-stream partials (`form_input` without a
// norm): x = round_A(sum_p t[p * width + k]). Its fields are `Rms`'s, so an
// entry selects either by its norm's extent (`Input`); `norm` and `eps` are
// not read.
template <class NORM, class Rows, u64 PARTS> struct Summed {
    const float *x;
    u64 stride;
    const u8 *norm;
    float eps;
    u64 width;
    Rows rows;
    static constexpr bool STAGED = true;
    static constexpr int FACTORS = 0;
    __device__ __forceinline__ void prepare_row(float *, u32, float *) const {}
    __device__ __forceinline__ float value(const float *, u32 m, u64 k) const {
        const float *row = x + rows(m) * stride;
        float sum = 0.0f;
#pragma unroll
        for (u64 part = 0; part < PARTS; ++part)
            sum = seismic_add_rn(sum, row[part * width + k]);
        return Act::round(sum);
    }
};

// The per-stream RMS normalization of STREAMS hyper-connection streams of
// `width` columns each (`group_rms`, `stream_mix_down`), one factor (the
// stream's inverse) per stream:
// x = round_A(t[k] * inverse[k / width] * w[k]).
template <class NORM, class Rows, u32 STREAMS> struct GroupRms {
    const float *x;
    u64 stride;
    const u8 *norm;
    float eps;
    u64 width;
    Rows rows;
    static constexpr bool STAGED = true;
    static constexpr int FACTORS = STREAMS;
    __device__ __forceinline__ float squares(u32 m, u32 stream, u32 first, u32 step) const {
        const float *row = x + rows(m) * stride + stream * width;
        float total = 0.0f;
        for (u64 k = first; k < width; k += step)
            total = seismic_fma_rn(row[k], row[k], total);
        return total;
    }
    __device__ __forceinline__ void prepare_row(float *shared, u32 m, float *scratch) const {
#pragma unroll 1
        for (u32 stream = 0; stream < STREAMS; ++stream) {
            const float total = reduce::group_sum(squares(m, stream, threadIdx.x, blockDim.x), scratch);
            if (threadIdx.x == 0)
                shared[stream] = rsqrtf(total / (float)width + eps);
        }
    }
    __device__ __forceinline__ float formed(const float *factors, u32 m, u64 k) const {
        return x[rows(m) * stride + k] * factors[k / width] * element::at<NORM>(norm, k);
    }
    __device__ __forceinline__ float value(const float *factors, u32 m, u64 k) const {
        return Act::round(formed(factors, m, k));
    }
};

// An entry's input prologue (`form_input`): RMS normalization with a norm
// of NORM_EXTENT values, or the sum of PARTS partials without one.
template <class NORM, class Rows, u64 NORM_EXTENT, u64 PARTS> struct InputOf {
    using type = Rms<NORM, Rows>;
};
template <class NORM, class Rows, u64 PARTS> struct InputOf<NORM, Rows, 0, PARTS> {
    using type = Summed<NORM, Rows, PARTS>;
};
template <class NORM, class Rows, u64 NORM_EXTENT, u64 PARTS>
using Input = typename InputOf<NORM, Rows, NORM_EXTENT, PARTS>::type;

// ---------------------------------------------------------------------------
// NVFP4 operands (docs/precision.md, "NVFP4 operands"; the portable
// `nvfp4_row`): a product over NVFP4 weights takes each row x as
// q * g * s, which A holds exactly. g, the row's global scale, is the least
// power of two at or above amax(|x|) / (6 * 448); s, a 16-value block's
// scale, is amax(|block|) / (6 * g) rounded to nearest even onto UE4M3; q
// is x / (g * s) rounded to nearest even onto E2M1, saturating at 6 (zero
// where s is). Every operation rounds once, to nearest even.

namespace nvfp4 {

__device__ __forceinline__ float global_scale(float amax) {
    const u32 bits = __float_as_uint(__fdiv_rn(amax, 2688.0f));
    const u32 exponent = (bits >> 23) + ((bits & 0x7fffffu) != 0u ? 1u : 0u);
    return __uint_as_float(min(max(exponent, 1u), 254u) << 23);
}

// v >= 0 onto UE4M3 (exponent bias 7, three mantissa bits, subnormal
// quantum 2^-9), saturating at 448.
__device__ __forceinline__ float ue4m3(float v) {
    const u32 exponent = max((__float_as_uint(v) >> 23) & 255u, 121u);
    const float quantum = __uint_as_float((exponent - 3u) << 23);
    return fminf(__fmul_rn(rintf(__fdiv_rn(v, quantum)), quantum), 448.0f);
}

// v onto E2M1's magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6 with v's sign,
// saturating at 6; NaN stays NaN.
__device__ __forceinline__ float e2m1(float v) {
    const float a = fabsf(v);
    const float m = a <= 0.25f   ? 0.0f
                    : a < 0.75f  ? 0.5f
                    : a <= 1.25f ? 1.0f
                    : a < 1.75f  ? 1.5f
                    : a <= 2.5f  ? 2.0f
                    : a < 3.5f   ? 3.0f
                    : a <= 5.0f  ? 4.0f
                    : a > 5.0f   ? 6.0f
                                 : a;
    return v < 0.0f ? -m : m;
}

// The field of s, a UE4M3 value (`ue4m3`'s): exponent bias 7 above the
// subnormal quantum 2^-9.
__device__ __forceinline__ u32 ue4m3_field(float s) {
    const u32 bits = __float_as_uint(s);
    return s < 0.015625f ? (u32)__fmul_rn(s, 512.0f) : (((bits >> 23) - 120u) << 3) | ((bits >> 20) & 7u);
}

// The code of q, an E2M1 value (`e2m1`'s): the magnitude's index among 0,
// 0.5, 1, 1.5, 2, 3, 4, 6 with q's sign at bit 3 (zero positive).
__device__ __forceinline__ u32 e2m1_code(float q) {
    const float a = fabsf(q);
    const u32 index = a < 2.0f ? (u32)__fmul_rn(a, 2.0f) : a < 4.0f ? 2u + (u32)a : a < 6.0f ? 6u : 7u;
    return index | (q < 0.0f ? 8u : 0u);
}

// Row m's global scale g, by the whole block.
template <class Pro>
__device__ __forceinline__ float row_global(const Pro &pro, const float *factors, u32 m, u64 K, float *scratch) {
    float amax = 0.0f;
    for (u64 k = threadIdx.x; k < K; k += blockDim.x)
        amax = fmaxf(amax, fabsf(pro.value(factors, m, k)));
    return global_scale(reduce::group_max(amax, scratch));
}

// The 16-value block `block` of row m under global scale g: its UE4M3
// scale s and each value's E2M1 code value q, the operand q * g * s.
template <class Pro>
__device__ __forceinline__ void block_operands(const Pro &pro, const float *factors, u32 m, u64 block, float g,
                                               float &s, float (&q)[16]) {
    float maximum = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        q[j] = pro.value(factors, m, block * 16 + j);
        maximum = fmaxf(maximum, fabsf(q[j]));
    }
    s = ue4m3(__fdiv_rn(maximum, __fmul_rn(6.0f, g)));
    const float d = __fmul_rn(g, s);
#pragma unroll
    for (int j = 0; j < 16; ++j)
        q[j] = d == 0.0f ? 0.0f : e2m1(__fdiv_rn(q[j], d));
}

// The packed form of a block's codes (value j in nibble j % 8 of word
// j / 8).
__device__ __forceinline__ uint2 packed_codes(const float (&q)[16]) {
    uint2 codes = make_uint2(0u, 0u);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        codes.x |= e2m1_code(q[j]) << (4 * j);
        codes.y |= e2m1_code(q[8 + j]) << (4 * j);
    }
    return codes;
}

} // namespace nvfp4

// FP4 tensor operations: block-scaled `mma.sync` with E2M1 operands and
// UE4M3 scales per 16 values, in the sm_12x family's architecture-specific
// feature set (formations target it from compute capability 9.0). Where a
// device has them, the rows a GEMM takes as NVFP4 operands are staged packed
// (`PackedRows`) and multiplied by them (`Fp4GemmOperands`).
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ / 100 == 12
#define PROJECTION_FP4_MMA 1
#else
#define PROJECTION_FP4_MMA 0
#endif
constexpr bool FP4_MMA = PROJECTION_FP4_MMA;

// NVFP4 operand rows packed for FP4 tensor operations: a row of K values
// holds K / 2 bytes of E2M1 codes (value k in nibble k % 2 of byte k / 2,
// `nvfp4::e2m1_code`), then its K / 16 UE4M3 block scale fields, then its
// global scale g (F32): value k is code * scale * g, `form_row`'s value.
// K / 2 + K / 16 + 4 bytes fit the 2 K of a staged 16-bit row, at whose
// pitch packed rows are staged.
struct PackedRows {
    __device__ static __forceinline__ u64 scales(u64 K) { return K / 2; }
    __device__ static __forceinline__ u64 global(u64 K) { return K / 2 + K / 16; }
    __device__ static __forceinline__ float global(const u8 *row, u64 K) {
        return *reinterpret_cast<const float *>(row + global(K));
    }
};

// A prologue whose rows a product over NVFP4 weights takes: always staged,
// as NVFP4 operands (`form_row`).
template <class Pro> struct Nvfp4Rows : Pro {
    static constexpr bool STAGED = true;
    static constexpr bool NVFP4 = true;
};

template <class Pro, class = void> struct nvfp4_rows {
    static constexpr bool value = false;
};
template <class Pro> struct nvfp4_rows<Pro, decltype(void(Pro::NVFP4))> {
    static constexpr bool value = Pro::NVFP4;
};

// The operand recipe of a product over the weights whose `operands(W)` are
// `RECIPES`, which the rows they share all take: NVFP4 (1) or rounded A (0).
template <bool... RECIPES> struct OperandRecipe {
    static constexpr bool NVFP4 = (RECIPES || ...);
    static_assert(NVFP4 == (RECIPES && ...), "weights that share rows share their operand recipe");
};

// The prologue of rows a product of operand recipe NVFP4 takes.
template <bool NVFP4, class Pro> struct OperandsOf {
    using type = Pro;
};
template <class Pro> struct OperandsOf<true, Pro> {
    using type = Nvfp4Rows<Pro>;
};
template <bool NVFP4, class Pro> using Operands = typename OperandsOf<NVFP4, Pro>::type;

// The prologue's row m as A into out[m * K .. m * K + K), formed by the whole
// block: the staging launch (one block per row, global scratch) and a GEMV
// block at M = 1 (its shared memory, `shared_row`) both form rows with it. A
// row of NVFP4 operands is formed per 16-value block (K a multiple of 16).
// With `published` given, the row's FACTORS factors (a group prologue's
// per-stream inverses) are also published to published[f * step].
template <class Storage = Act, class Pro>
__device__ __forceinline__ void form_row(const Pro &pro, u32 m, u64 K, u8 *out,
                                         const element::Out<element::F32> *published = nullptr, u64 step = 1) {
    static_assert(Storage::bytes == 2, "projection physical operands are 16-bit");
    __shared__ float factors[Pro::FACTORS > 0 ? Pro::FACTORS : 1];
    __shared__ float scratch[32];
    pro.prepare_row(factors, m, scratch);
    __syncthreads();
    if (published != nullptr && threadIdx.x < (unsigned)Pro::FACTORS)
        published->put(threadIdx.x * step, factors[threadIdx.x]);
    if constexpr (nvfp4_rows<Pro>::value) {
        const float g = nvfp4::row_global(pro, factors, m, K, scratch);
        for (u64 block = threadIdx.x; block < K / 16; block += blockDim.x) {
            float s, q[16];
            nvfp4::block_operands(pro, factors, m, block, g, s, q);
            const float d = __fmul_rn(g, s);
#pragma unroll
            for (int j = 0; j < 16; ++j)
                element::put<Storage>(out, m * K + block * 16 + j, __fmul_rn(q[j], d));
        }
    } else {
        for (u64 k = threadIdx.x; k < K; k += blockDim.x)
            element::put<Storage>(out, m * K + k, pro.value(factors, m, k));
    }
}

// The prologue's row m as packed NVFP4 operands (`PackedRows`) at `row`,
// formed by the whole block as `form_row` forms its values. With BLOCKS
// > 0 the row may be the prologue's own input (a product formed in place):
// each thread first forms its at most BLOCKS 16-value blocks, and the block
// writes once every thread has read.
template <int BLOCKS = 0, class Pro>
__device__ __forceinline__ void form_packed_row(const Pro &pro, u32 m, u64 K, u8 *row) {
    static_assert(nvfp4_rows<Pro>::value, "packed rows are NVFP4 operands");
    __shared__ float factors[Pro::FACTORS > 0 ? Pro::FACTORS : 1];
    __shared__ float scratch[32];
    pro.prepare_row(factors, m, scratch);
    __syncthreads();
    const float g = nvfp4::row_global(pro, factors, m, K, scratch);
    auto put = [&](u64 block, uint2 codes, float s) {
        *reinterpret_cast<uint2 *>(row + block * 8) = codes;
        row[PackedRows::scales(K) + block] = (u8)nvfp4::ue4m3_field(s);
    };
    if constexpr (BLOCKS == 0) {
        for (u64 block = threadIdx.x; block < K / 16; block += blockDim.x) {
            float s, q[16];
            nvfp4::block_operands(pro, factors, m, block, g, s, q);
            put(block, nvfp4::packed_codes(q), s);
        }
    } else {
        uint2 codes[BLOCKS];
        float scales[BLOCKS];
#pragma unroll
        for (int i = 0; i < BLOCKS; ++i) {
            const u64 block = threadIdx.x + (u64)i * blockDim.x;
            if (block < K / 16) {
                float q[16];
                nvfp4::block_operands(pro, factors, m, block, g, scales[i], q);
                codes[i] = nvfp4::packed_codes(q);
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < BLOCKS; ++i) {
            const u64 block = threadIdx.x + (u64)i * blockDim.x;
            if (block < K / 16)
                put(block, codes[i], scales[i]);
        }
    }
    if (threadIdx.x == 0)
        *reinterpret_cast<float *>(row + PackedRows::global(K)) = g;
}

// int8 activations (the GEMM's INT8 candidate): rows quantized per 32-code
// group, x ~= d * q by the rule every backend's INT8 activations follow
// (`s8_scale`, `s8_code`: the CPU's, `seismic-native-cpu` `quant.rs`), staged
// by `stage_row_s8` as `q` ([M, width] bytes in the s8 fragments' virtual k
// order, packets.cuh, in entry order where ENTRIES: `s8_real_offset`) and
// `groups` (each group's (d, d * sum q)). Any s8 row source gives `row(m)`
// (the row's codes) and `groups_of(m)` (its groups) for m < M, and
// `entries()`, their order: the rows the int8 GEMM reads.
template <bool ENTRIES> struct QuantizedRows {
    __host__ __device__ static constexpr bool entries() { return ENTRIES; }
    const u8 *q;
    u64 width;
    const float2 *groups;
    __device__ __forceinline__ const u8 *row(u64 m) const { return q + m * width; }
    __device__ __forceinline__ const float2 *groups_of(u64 m) const { return groups + m * (width / 32); }
};

// Real offset within a 32-code group of virtual s8 position v (lane t of
// the m16n8k32 fragment holds virtual 4t .. 4t + 3 and 16 + 4t ..), in the
// weights' order (`packets::Packed::s8_half`). Slot order: the k16 steps'
// slots, real {2t, 2t + 1, 2t + 8, 2t + 9} of each step. Entry order
// (ENTRIES, grid codes): lane t's eight are real 8t .. 8t + 7, one entry.
template <bool ENTRIES> __device__ __forceinline__ u32 s8_real_offset(u32 v) {
    const u32 half = v / 16, t = (v % 16) / 4, j = v % 4;
    if constexpr (ENTRIES)
        return 8 * t + 4 * half + j;
    else
        return half * 16 + 2 * t + (j % 2) + 8 * (j / 2);
}

// The scale of a group whose largest magnitude is `maximum` (NaN ignored):
// `maximum / 127` rounded to nearest, or where that is subnormal, rounded up
// onto the subnormal grid, so `maximum / d <= 127` and `d` is zero only for
// an all-zero group.
__device__ __forceinline__ float s8_scale(float maximum) {
    const float least_normal = __uint_as_float(0x00800000u);
    float d = __fdiv_rn(maximum, 127.0f);
    // F64 holds `d * 127` exactly.
    if (d < least_normal && (double)d * 127.0 < (double)maximum)
        d = __uint_as_float(__float_as_uint(d) + 1u);
    return d;
}

// The code of `x` under the scale `d` (`s8_scale`): the exact quotient
// rounded once, half away from zero; NaN, and a zero scale, code 0. Where
// `d` is normal the F32 quotient is within half an ulp of the exact one, so
// only a quotient that lands on a half decides by the sign of its remainder;
// where `d` is subnormal the F64 quotient of two F32 values rounds like the
// exact one.
__device__ __forceinline__ int s8_code(float x, float d) {
    if (d == 0.0f)
        return 0;
    const float magnitude = fabsf(x);
    float rounded;
    if (d < __uint_as_float(0x00800000u)) {
        const double quotient = (double)magnitude / (double)d;
        if (!(quotient == quotient))
            return 0;
        rounded = (float)round(quotient);
    } else {
        const float quotient = __fdiv_rn(magnitude, d);
        if (!(quotient == quotient))
            return 0;
        rounded = roundf(quotient);
        // A tie in F32 is below the exact quotient where the remainder is negative.
        if (quotient - truncf(quotient) == 0.5f && fmaf(-quotient, d, magnitude) < 0.0f)
            rounded = truncf(quotient);
    }
    return x < 0.0f ? -(int)rounded : (int)rounded;
}

// Staging launch of the INT8 candidate, one block per row m: the prologue's
// row x[m, :] (rounded to A) quantized per 32 codes into `q` ([M, K] bytes,
// virtual order, ENTRIES: `s8_real_offset`) and `groups` ([M, K / 32]
// (d, d * sum q)).
template <bool ENTRIES, class Pro>
__device__ __forceinline__ void stage_row_s8(const Pro &pro, u32 m, u64 K, u8 *q, float2 *groups) {
    __shared__ float factors[Pro::FACTORS > 0 ? Pro::FACTORS : 1];
    __shared__ float scratch[32];
    pro.prepare_row(factors, m, scratch);
    __syncthreads();
    for (u64 group = threadIdx.x; group < K / 32; group += blockDim.x) {
        float values[32];
        float maximum = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            values[j] = pro.value(factors, m, group * 32 + j);
            maximum = fmaxf(maximum, fabsf(values[j]));
        }
        const float d = s8_scale(maximum);
        int total = 0;
        u32 words[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            u32 word = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int code = s8_code(values[s8_real_offset<ENTRIES>(4 * w + b)], d);
                total += code;
                word |= ((u32)code & 0xFFu) << (8 * b);
            }
            words[w] = word;
        }
        uint4 *out = reinterpret_cast<uint4 *>(q + m * K + group * 32);
        out[0] = make_uint4(words[0], words[1], words[2], words[3]);
        out[1] = make_uint4(words[4], words[5], words[6], words[7]);
        groups[m * (K / 32) + group] = make_float2(d, d * (float)total);
    }
}

// ---------------------------------------------------------------------------
// Reference-order prologues (`OrderedRms`, `OrderedSummed`, `OrderedInput`):
// 8-lane reduction (`((l0+l4)+(l2+l6))+((l1+l5)+(l3+l7))`) and `1 / sqrtf(...)`.

__device__ __forceinline__ float ordered_combine8(float v) {
    const unsigned mask = 0xffu << (threadIdx.x & 24u);
    v = seismic_add_rn(v, __shfl_xor_sync(mask, v, 4));
    v = seismic_add_rn(v, __shfl_xor_sync(mask, v, 2));
    v = seismic_add_rn(v, __shfl_xor_sync(mask, v, 1));
    return v;
}

template <class NORM, class Rows> struct OrderedRms {
    const float *x;
    u64 stride;
    const u8 *norm;
    float eps;
    u64 width;
    Rows rows;
    u64 col_stride = 1;
    u64 norm_stride = 1;
    static constexpr bool STAGED = true;
    static constexpr int FACTORS = 1;
    __device__ __forceinline__ void ordered_inverses(u32 m_rows, float *inverses, u32 first_row = 0u,
                                                     float *scratch = nullptr) const {
        const u32 tid = threadIdx.x, threads = blockDim.x;
        const u32 group_id = tid / 8u, u = tid & 7u, groups = threads / 8u;
        if (scratch != nullptr && col_stride == 1ull && m_rows == 1u) {
            const float *row_ptr = x + rows(first_row) * stride;
            float sum = 0.0f;
            for (u64 k0 = 0; k0 < width; k0 += 512ull) {
                const u32 chunk = (u32)min(512ull, width - k0);
                for (u32 c = tid; c < chunk; c += threads)
                    scratch[c] = row_ptr[k0 + c];
                __syncthreads();
                if (group_id == 0u) {
                    u32 c = u;
                    for (; c + 56u < chunk; c += 64u) {
                        const float v0 = scratch[c];
                        const float v1 = scratch[c + 8u];
                        const float v2 = scratch[c + 16u];
                        const float v3 = scratch[c + 24u];
                        const float v4 = scratch[c + 32u];
                        const float v5 = scratch[c + 40u];
                        const float v6 = scratch[c + 48u];
                        const float v7 = scratch[c + 56u];
                        sum = seismic_fma_rn(v0, v0, sum);
                        sum = seismic_fma_rn(v1, v1, sum);
                        sum = seismic_fma_rn(v2, v2, sum);
                        sum = seismic_fma_rn(v3, v3, sum);
                        sum = seismic_fma_rn(v4, v4, sum);
                        sum = seismic_fma_rn(v5, v5, sum);
                        sum = seismic_fma_rn(v6, v6, sum);
                        sum = seismic_fma_rn(v7, v7, sum);
                    }
                    for (; c < chunk; c += 8u) {
                        const float v = scratch[c];
                        sum = seismic_fma_rn(v, v, sum);
                    }
                }
                __syncthreads();
            }
            const float total_sq = ordered_combine8(sum);
            if (tid == 0u)
                inverses[0] = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(total_sq, (float)width) + eps));
            __syncthreads();
            return;
        }
        for (u32 base = 0u; base < m_rows; base += groups) {
            const u32 r = base + group_id;
            float sum = 0.0f;
            if (r < m_rows) {
                const float *row_ptr = x + rows(first_row + r) * stride;
                if (col_stride == 1ull) {
                    u64 k0 = 0;
                    for (; k0 + 64ull <= width; k0 += 64ull) {
                        const u64 c = k0 + u;
                        const float v0 = row_ptr[c];
                        const float v1 = row_ptr[c + 8ull];
                        const float v2 = row_ptr[c + 16ull];
                        const float v3 = row_ptr[c + 24ull];
                        const float v4 = row_ptr[c + 32ull];
                        const float v5 = row_ptr[c + 40ull];
                        const float v6 = row_ptr[c + 48ull];
                        const float v7 = row_ptr[c + 56ull];
                        sum = seismic_fma_rn(v0, v0, sum);
                        sum = seismic_fma_rn(v1, v1, sum);
                        sum = seismic_fma_rn(v2, v2, sum);
                        sum = seismic_fma_rn(v3, v3, sum);
                        sum = seismic_fma_rn(v4, v4, sum);
                        sum = seismic_fma_rn(v5, v5, sum);
                        sum = seismic_fma_rn(v6, v6, sum);
                        sum = seismic_fma_rn(v7, v7, sum);
                    }
                    for (u64 c = k0 + u; c < width; c += 8ull) {
                        const float v = row_ptr[c];
                        sum = seismic_fma_rn(v, v, sum);
                    }
                } else {
                    for (u64 c = u; c < width; c += 8ull) {
                        const float v = row_ptr[c * col_stride];
                        sum = seismic_fma_rn(v, v, sum);
                    }
                }
            }
            const float total_sq = ordered_combine8(sum);
            if (r < m_rows && u == 0u)
                inverses[r] = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(total_sq, (float)width) + eps));
        }
        __syncthreads();
    }
    __device__ __forceinline__ float ordered_at(u32 m, u64 c, float inv) const {
        return Act::round(x[rows(m) * stride + c * col_stride] * inv * element::at<NORM>(norm, c * norm_stride));
    }
    __device__ __forceinline__ void prepare_row(float *shared, u32 m, float *) const {
        __shared__ float ordered_sram[512];
        ordered_inverses(1u, shared, m, ordered_sram);
    }
    __device__ __forceinline__ float value(const float *factors, u32 m, u64 k) const {
        return ordered_at(m, k, factors[0]);
    }
};

template <class NORM, class Rows, u64 PARTS> struct OrderedSummed {
    const float *x;
    u64 stride;
    const u8 *norm;
    float eps;
    u64 width;
    Rows rows;
    u64 col_stride = 1;
    u64 norm_stride = 1;
    static constexpr bool STAGED = true;
    static constexpr int FACTORS = 0;
    __device__ __forceinline__ void ordered_inverses(u32 m_rows, float *inverses, u32 = 0u, float * = nullptr) const {
        for (u32 r = threadIdx.x; r < m_rows; r += blockDim.x)
            inverses[r] = 1.0f;
        __syncthreads();
    }
    __device__ __forceinline__ float ordered_at(u32 m, u64 c, float) const {
        const float *row = x + rows(m) * stride;
        float sum = 0.0f;
#pragma unroll
        for (u64 part = 0; part < PARTS; ++part)
            sum = seismic_add_rn(sum, row[(part * width + c) * col_stride]);
        return Act::round(sum);
    }
    __device__ __forceinline__ void prepare_row(float *, u32, float *) const {}
    __device__ __forceinline__ float value(const float *, u32 m, u64 k) const {
        return ordered_at(m, k, 1.0f);
    }
};

template <class NORM, class Rows, u64 NORM_EXTENT, u64 PARTS> struct OrderedInputOf {
    using type = OrderedRms<NORM, Rows>;
};
template <class NORM, class Rows, u64 PARTS> struct OrderedInputOf<NORM, Rows, 0, PARTS> {
    using type = OrderedSummed<NORM, Rows, PARTS>;
};
template <class NORM, class Rows, u64 NORM_EXTENT, u64 PARTS>
using OrderedInput = typename OrderedInputOf<NORM, Rows, NORM_EXTENT, PARTS>::type;

template <class Pro, class Out>
__device__ __forceinline__ float ordered_stage_row(const Pro &pro, u32 m, u64 K, Out out, u64 stride0, u64 stride1) {
    __shared__ float inv_shared[1];
    __shared__ float stage_scratch[512];
    pro.ordered_inverses(1u, inv_shared, m, stage_scratch);
    const float inv = inv_shared[0];
    for (u64 c = threadIdx.x; c < K; c += blockDim.x)
        out.put(m * stride0 + c * stride1, pro.ordered_at(m, c, inv));
    return inv;
}
