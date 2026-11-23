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
    static constexpr bool VEC4 = true;
    __device__ __forceinline__ void prepare_row(float *, u32, float *) const {}
    __device__ __forceinline__ float value(const float *, u32 m, u64 k) const {
        return Act::round(element::at<E>(x, rows(m) * stride + k));
    }
    __device__ __forceinline__ bool can_vec4(u32 m) const {
        const u8 *row = x + rows(m) * stride * E::bytes;
        return (reinterpret_cast<u64>(row) & (4 * E::bytes - 1)) == 0ull;
    }
    __device__ __forceinline__ void value4(const float *, u32 m, u64 k, float (&out)[4]) const {
        element::span<E, 4, false>(x, rows(m) * stride + k, out);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            out[j] = Act::round(out[j]);
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
    static constexpr bool VEC4 = true;
    __device__ __forceinline__ void prepare_row(float *, u32, float *) const {}
    __device__ __forceinline__ float value(const float *, u32 m, u64 k) const {
        return Act::round(element::at<E>(x, rows(m) * row_stride + k * column_stride));
    }
    __device__ __forceinline__ bool can_vec4(u32 m) const {
        if (column_stride != 1ull)
            return false;
        const u8 *row = x + rows(m) * row_stride * E::bytes;
        return (reinterpret_cast<u64>(row) & (4 * E::bytes - 1)) == 0ull;
    }
    __device__ __forceinline__ void value4(const float *, u32 m, u64 k, float (&out)[4]) const {
        element::span<E, 4, false>(x, rows(m) * row_stride + k, out);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            out[j] = Act::round(out[j]);
    }
};

// Coalesced float4 square sum over `row[0 .. width)` with exact FMA
// accumulation, falling back to scalar when `row` is not 16-byte aligned.
__device__ __forceinline__ float row_squares(const float *row, u64 width, u32 first, u32 step) {
    float total = 0.0f;
    if ((reinterpret_cast<u64>(row) & 15ull) == 0ull) {
        const u64 vec_width = width & ~3ull;
        const u64 step4 = (u64)step * 4ull;
        for (u64 k = (u64)first * 4ull; k < vec_width; k += step4) {
            const float4 v = *reinterpret_cast<const float4 *>(row + k);
            total = seismic_fma_rn(v.x, v.x, total);
            total = seismic_fma_rn(v.y, v.y, total);
            total = seismic_fma_rn(v.z, v.z, total);
            total = seismic_fma_rn(v.w, v.w, total);
        }
        for (u64 k = vec_width + first; k < width; k += step)
            total = seismic_fma_rn(row[k], row[k], total);
    } else {
        for (u64 k = first; k < width; k += step)
            total = seismic_fma_rn(row[k], row[k], total);
    }
    return total;
}

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
    static constexpr bool VEC4 = true;
    __device__ __forceinline__ float squares(u32 m, u32 first, u32 step) const {
        return row_squares(x + rows(m) * stride, width, first, step);
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
    __device__ __forceinline__ bool can_vec4(u32 m) const {
        const float *row = x + rows(m) * stride;
        return (reinterpret_cast<u64>(row) & 15ull) == 0ull &&
               (reinterpret_cast<u64>(norm) & (4 * NORM::bytes - 1)) == 0ull;
    }
    __device__ __forceinline__ void value4(const float *factors, u32 m, u64 k, float (&out)[4]) const {
        const float4 v = *reinterpret_cast<const float4 *>(x + rows(m) * stride + k);
        float w[4];
        element::span<NORM, 4, true>(norm, k, w);
        const float inv = factors[0];
        out[0] = Act::round(v.x * inv * w[0]);
        out[1] = Act::round(v.y * inv * w[1]);
        out[2] = Act::round(v.z * inv * w[2]);
        out[3] = Act::round(v.w * inv * w[3]);
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
    static constexpr bool VEC4 = true;
    __device__ __forceinline__ void prepare_row(float *, u32, float *) const {}
    __device__ __forceinline__ float value(const float *, u32 m, u64 k) const {
        const float *row = x + rows(m) * stride;
        float sum = 0.0f;
#pragma unroll
        for (u64 part = 0; part < PARTS; ++part)
            sum = seismic_add_rn(sum, row[part * width + k]);
        return Act::round(sum);
    }
    __device__ __forceinline__ bool can_vec4(u32 m) const {
        const float *row = x + rows(m) * stride;
        return (reinterpret_cast<u64>(row) & 15ull) == 0ull && (PARTS <= 1 || (width & 3ull) == 0ull);
    }
    __device__ __forceinline__ void value4(const float *, u32 m, u64 k, float (&out)[4]) const {
        const float *row = x + rows(m) * stride;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma unroll
        for (u64 part = 0; part < PARTS; ++part) {
            const float4 v = *reinterpret_cast<const float4 *>(row + part * width + k);
            s0 = seismic_add_rn(s0, v.x);
            s1 = seismic_add_rn(s1, v.y);
            s2 = seismic_add_rn(s2, v.z);
            s3 = seismic_add_rn(s3, v.w);
        }
        out[0] = Act::round(s0);
        out[1] = Act::round(s1);
        out[2] = Act::round(s2);
        out[3] = Act::round(s3);
    }
};

// The per-stream RMS normalization of STREAMS hyper-connection streams of
// `width` columns each (`group_rms`, `stream_mix_down`), one factor (the
// stream's inverse) per stream:
// x = round_A(t[k] * inverse[k / width] * w[k]).
template <class NORM, class Rows, u32 STREAMS, u64 WIDTH = 0> struct GroupRms {
    const float *x;
    u64 stride;
    const u8 *norm;
    float eps;
    u64 width;
    Rows rows;
    static constexpr bool STAGED = true;
    static constexpr int FACTORS = STREAMS;
    static constexpr bool VEC4 = true;
    __device__ __forceinline__ u64 stream_of(u64 k) const {
        if constexpr (STREAMS == 1)
            return 0ull;
        else if constexpr (WIDTH != 0)
            return k / WIDTH;
        else
            return k / width;
    }
    __device__ __forceinline__ float squares(u32 m, u32 stream, u32 first, u32 step) const {
        return row_squares(x + rows(m) * stride + stream * width, width, first, step);
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
        return x[rows(m) * stride + k] * factors[stream_of(k)] * element::at<NORM>(norm, k);
    }
    __device__ __forceinline__ float value(const float *factors, u32 m, u64 k) const {
        return Act::round(formed(factors, m, k));
    }
    __device__ __forceinline__ bool can_vec4(u32 m) const {
        const float *row = x + rows(m) * stride;
        return (width & 3ull) == 0ull && (reinterpret_cast<u64>(row) & 15ull) == 0ull &&
               (reinterpret_cast<u64>(norm) & (4 * NORM::bytes - 1)) == 0ull;
    }
    __device__ __forceinline__ void value4(const float *factors, u32 m, u64 k, float (&out)[4]) const {
        const float4 v = *reinterpret_cast<const float4 *>(x + rows(m) * stride + k);
        float w[4];
        element::span<NORM, 4, true>(norm, k, w);
        const float inv = factors[stream_of(k)];
        out[0] = Act::round(v.x * inv * w[0]);
        out[1] = Act::round(v.y * inv * w[1]);
        out[2] = Act::round(v.z * inv * w[2]);
        out[3] = Act::round(v.w * inv * w[3]);
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

template <class Pro, class = void> struct has_vec4 {
    static constexpr bool value = false;
};
template <class Pro> struct has_vec4<Pro, decltype(void(Pro::VEC4))> {
    static constexpr bool value = Pro::VEC4;
};

template <class Pro> __device__ __forceinline__ bool pro_can_vec4(const Pro &pro, u32 m) {
    if constexpr (has_vec4<Pro>::value)
        return pro.can_vec4(m);
    else
        return false;
}

template <class Pro>
__device__ __forceinline__ void pro_value4(const Pro &pro, const float *factors, u32 m, u64 k, float (&out)[4]) {
    if constexpr (has_vec4<Pro>::value) {
        pro.value4(factors, m, k, out);
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            out[j] = pro.value(factors, m, k + j);
    }
}

// ---------------------------------------------------------------------------
// NVFP4 operand rows (<seismic/activation.cuh>, `packets::nvfp4`): the row's
// global scale over its peak, and each 16-value block's scale and codes.

namespace nvfp4_stage {

// Row m's global scale g, by the whole block.
template <class Pro>
__device__ __forceinline__ float row_global(const Pro &pro, const float *factors, u32 m, u64 K, float *scratch) {
    float amax = 0.0f;
    if (pro_can_vec4(pro, m)) {
        const u64 k4 = K & ~3ull;
        const u64 step4 = (u64)blockDim.x * 4ull;
        for (u64 k = (u64)threadIdx.x * 4ull; k < k4; k += step4) {
            float v[4];
            pro_value4(pro, factors, m, k, v);
            amax = packets::peak_with(amax, v[0]);
            amax = packets::peak_with(amax, v[1]);
            amax = packets::peak_with(amax, v[2]);
            amax = packets::peak_with(amax, v[3]);
        }
        for (u64 k = k4 + threadIdx.x; k < K; k += blockDim.x)
            amax = packets::peak_with(amax, pro.value(factors, m, k));
    } else {
        for (u64 k = threadIdx.x; k < K; k += blockDim.x)
            amax = packets::peak_with(amax, pro.value(factors, m, k));
    }
    return packets::nvfp4::global_scale(reduce::group_peak(amax, scratch));
}

// The 16-value block `block` of row m under global scale g: its UE4M3
// scale s and each value's E2M1 code value q, the operand q * g * s; s NaN
// and every q 0 under a NaN g (the block's own peak need not carry NaN).
template <class Pro>
__device__ __forceinline__ void block_operands(const Pro &pro, const float *factors, u32 m, u64 block, float g,
                                               float &s, float (&q)[16]) {
    float maximum = 0.0f;
    const u64 base = block * 16;
    if (pro_can_vec4(pro, m)) {
#pragma unroll
        for (int j = 0; j < 16; j += 4) {
            float v[4];
            pro_value4(pro, factors, m, base + j, v);
            q[j + 0] = v[0];
            q[j + 1] = v[1];
            q[j + 2] = v[2];
            q[j + 3] = v[3];
            maximum = fmaxf(maximum, fabsf(v[0]));
            maximum = fmaxf(maximum, fabsf(v[1]));
            maximum = fmaxf(maximum, fabsf(v[2]));
            maximum = fmaxf(maximum, fabsf(v[3]));
        }
    } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            q[j] = pro.value(factors, m, base + j);
            maximum = fmaxf(maximum, fabsf(q[j]));
        }
    }
    s = packets::nvfp4::block_scale(maximum, g);
    const float d = __fmul_rn(g, s);
#pragma unroll
    for (int j = 0; j < 16; ++j)
        q[j] = packets::nvfp4::code_value(q[j], d);
}

// The packed form of a block's codes (value j in nibble j % 8 of word
// j / 8).
__device__ __forceinline__ uint2 packed_codes(const float (&q)[16]) {
    uint2 codes = make_uint2(0u, 0u);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        codes.x |= packets::nvfp4::e2m1_code(q[j]) << (4 * j);
        codes.y |= packets::nvfp4::e2m1_code(q[8 + j]) << (4 * j);
    }
    return codes;
}

} // namespace nvfp4_stage

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
// `packets::nvfp4::e2m1_code`), then its K / 16 UE4M3 block scale fields, then its
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
    const element::Out<Storage> dst = element::out_over_scratch<Storage>(out);
    const u64 row_base = (u64)m * K;
    if constexpr (nvfp4_rows<Pro>::value) {
        const float g = nvfp4_stage::row_global(pro, factors, m, K, scratch);
        for (u64 block = threadIdx.x; block < K / 16; block += blockDim.x) {
            float s, q[16];
            nvfp4_stage::block_operands(pro, factors, m, block, g, s, q);
            const float d = __fmul_rn(g, s);
#pragma unroll
            for (int j = 0; j < 16; ++j)
                q[j] = __fmul_rn(q[j], d);
            dst.template put_span<16>(row_base + block * 16, q);
        }
    } else if (pro_can_vec4(pro, m)) {
        const u64 k4 = K & ~3ull;
        const u64 step4 = (u64)blockDim.x * 4ull;
        for (u64 k = (u64)threadIdx.x * 4ull; k < k4; k += step4) {
            float vals[4];
            pro_value4(pro, factors, m, k, vals);
            dst.template put_span<4>(row_base + k, vals);
        }
        for (u64 k = k4 + threadIdx.x; k < K; k += blockDim.x)
            dst.put(row_base + k, pro.value(factors, m, k));
    } else {
        for (u64 k = threadIdx.x; k < K; k += blockDim.x)
            dst.put(row_base + k, pro.value(factors, m, k));
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
    const float g = nvfp4_stage::row_global(pro, factors, m, K, scratch);
    auto put = [&](u64 block, uint2 codes, float s) {
        *reinterpret_cast<uint2 *>(row + block * 8) = codes;
        row[PackedRows::scales(K) + block] = (u8)packets::nvfp4::ue4m3_field(s);
    };
    if constexpr (BLOCKS == 0) {
        for (u64 block = threadIdx.x; block < K / 16; block += blockDim.x) {
            float s, q[16];
            nvfp4_stage::block_operands(pro, factors, m, block, g, s, q);
            put(block, nvfp4_stage::packed_codes(q), s);
        }
    } else {
        uint2 codes[BLOCKS];
        float scales[BLOCKS];
#pragma unroll
        for (int i = 0; i < BLOCKS; ++i) {
            const u64 block = threadIdx.x + (u64)i * blockDim.x;
            if (block < K / 16) {
                float q[16];
                nvfp4_stage::block_operands(pro, factors, m, block, g, scales[i], q);
                codes[i] = nvfp4_stage::packed_codes(q);
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
// (`packets::int8_scale_of`, `packets::int8_code_of`, <seismic/activation.cuh>), staged
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
// weights' order (`W::s8_half`). Slot order: the k16 steps'
// slots, real {2t, 2t + 1, 2t + 8, 2t + 9} of each step. Entry order
// (ENTRIES, grid codes): lane t's eight are real 8t .. 8t + 7, one entry.
template <bool ENTRIES> __device__ __forceinline__ u32 s8_real_offset(u32 v) {
    const u32 half = v / 16, t = (v % 16) / 4, j = v % 4;
    if constexpr (ENTRIES)
        return 8 * t + 4 * half + j;
    else
        return half * 16 + 2 * t + (j % 2) + 8 * (j / 2);
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
    const bool vec4 = pro_can_vec4(pro, m);
    for (u64 group = threadIdx.x; group < K / 32; group += blockDim.x) {
        float values[32];
        float maximum = 0.0f;
        const u64 base = group * 32;
        if (vec4) {
#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                float v[4];
                pro_value4(pro, factors, m, base + j, v);
                values[j + 0] = v[0];
                values[j + 1] = v[1];
                values[j + 2] = v[2];
                values[j + 3] = v[3];
                maximum = packets::peak_with(maximum, v[0]);
                maximum = packets::peak_with(maximum, v[1]);
                maximum = packets::peak_with(maximum, v[2]);
                maximum = packets::peak_with(maximum, v[3]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                values[j] = pro.value(factors, m, base + j);
                maximum = packets::peak_with(maximum, values[j]);
            }
        }
        const float d = packets::int8_scale_of(maximum);
        int total = 0;
        u32 words[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            u32 word = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int code = packets::int8_code_of(values[s8_real_offset<ENTRIES>(4 * w + b)], d);
                total += code;
                word |= ((u32)code & 0xFFu) << (8 * b);
            }
            words[w] = word;
        }
        uint4 *out = reinterpret_cast<uint4 *>(q + (u64)m * K + group * 32);
        out[0] = make_uint4(words[0], words[1], words[2], words[3]);
        out[1] = make_uint4(words[4], words[5], words[6], words[7]);
        groups[(u64)m * (K / 32) + group] = make_float2(d, d * (float)total);
    }
}

// Single-warp quantization of one plain activation row (`Pro::FACTORS == 0`,
// no block-wide barrier) into its s8 codes and (d, d * sum q) groups, bit-for-bit
// identical to `stage_row_s8`. Lets a 256-thread CTA process 8 rows in parallel.
template <bool ENTRIES, class Pro>
__device__ __forceinline__ void stage_row_s8_warp(const Pro &pro, u32 m, u64 K, u8 *q, float2 *groups) {
    static_assert(Pro::FACTORS == 0, "stage_row_s8_warp requires a factor-free prologue");
    const float factors[1] = {0.0f};
    const bool vec4 = pro_can_vec4(pro, m);
    for (u64 group = threadIdx.x & 31u; group < K / 32; group += 32) {
        float values[32];
        float maximum = 0.0f;
        const u64 base = group * 32;
        if (vec4) {
#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                float v[4];
                pro_value4(pro, factors, m, base + j, v);
                values[j + 0] = v[0];
                values[j + 1] = v[1];
                values[j + 2] = v[2];
                values[j + 3] = v[3];
                maximum = packets::peak_with(maximum, v[0]);
                maximum = packets::peak_with(maximum, v[1]);
                maximum = packets::peak_with(maximum, v[2]);
                maximum = packets::peak_with(maximum, v[3]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                values[j] = pro.value(factors, m, base + j);
                maximum = packets::peak_with(maximum, values[j]);
            }
        }
        const float d = packets::int8_scale_of(maximum);
        int total = 0;
        u32 words[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            u32 word = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int code = packets::int8_code_of(values[s8_real_offset<ENTRIES>(4 * w + b)], d);
                total += code;
                word |= ((u32)code & 0xFFu) << (8 * b);
            }
            words[w] = word;
        }
        uint4 *out = reinterpret_cast<uint4 *>(q + (u64)m * K + group * 32);
        out[0] = make_uint4(words[0], words[1], words[2], words[3]);
        out[1] = make_uint4(words[4], words[5], words[6], words[7]);
        groups[(u64)m * (K / 32) + group] = make_float2(d, d * (float)total);
    }
}

// Single-pass staging of row `m` for mixed segmented projections (some segments
// INT8-admitted, others 16-bit): writes the 16-bit `Storage` row into `act_out`
// and the s8 codes + groups into `q` / `groups` from the same normalized values.
template <bool ENTRIES, class Storage = Act, class Pro>
__device__ __forceinline__ void stage_row_s8_and_act(const Pro &pro, u32 m, u64 K, u8 *act_out, u8 *q,
                                                     float2 *groups) {
    static_assert(Storage::bytes == 2, "projection physical operands are 16-bit");
    __shared__ float factors[Pro::FACTORS > 0 ? Pro::FACTORS : 1];
    __shared__ float scratch[32];
    pro.prepare_row(factors, m, scratch);
    __syncthreads();
    const element::Out<Storage> act_dst = element::out_over_scratch<Storage>(act_out);
    const u64 row_base = (u64)m * K;
    const bool vec4 = pro_can_vec4(pro, m);
    for (u64 group = threadIdx.x; group < K / 32; group += blockDim.x) {
        float values[32];
        float maximum = 0.0f;
        const u64 base = group * 32;
        if (vec4) {
#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                float v[4];
                pro_value4(pro, factors, m, base + j, v);
                values[j + 0] = v[0];
                values[j + 1] = v[1];
                values[j + 2] = v[2];
                values[j + 3] = v[3];
                maximum = packets::peak_with(maximum, v[0]);
                maximum = packets::peak_with(maximum, v[1]);
                maximum = packets::peak_with(maximum, v[2]);
                maximum = packets::peak_with(maximum, v[3]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                values[j] = pro.value(factors, m, base + j);
                maximum = packets::peak_with(maximum, values[j]);
            }
        }
        act_dst.template put_span<32>(row_base + base, values);
        const float d = packets::int8_scale_of(maximum);
        int total = 0;
        u32 words[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            u32 word = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int code = packets::int8_code_of(values[s8_real_offset<ENTRIES>(4 * w + b)], d);
                total += code;
                word |= ((u32)code & 0xFFu) << (8 * b);
            }
            words[w] = word;
        }
        uint4 *out = reinterpret_cast<uint4 *>(q + row_base + base);
        out[0] = make_uint4(words[0], words[1], words[2], words[3]);
        out[1] = make_uint4(words[4], words[5], words[6], words[7]);
        groups[(u64)m * (K / 32) + group] = make_float2(d, d * (float)total);
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

// Accumulate `sp[u], sp[u + 8], ... < count` in exact 8-lane reference order.
__device__ __forceinline__ float ordered_accumulate_sram8(const float *sp, u32 count, u32 u, float sum) {
    u32 c = u;
    for (; c + 56u < count; c += 64u) {
        const float v0 = sp[c];
        const float v1 = sp[c + 8u];
        const float v2 = sp[c + 16u];
        const float v3 = sp[c + 24u];
        const float v4 = sp[c + 32u];
        const float v5 = sp[c + 40u];
        const float v6 = sp[c + 48u];
        const float v7 = sp[c + 56u];
        sum = seismic_fma_rn(v0, v0, sum);
        sum = seismic_fma_rn(v1, v1, sum);
        sum = seismic_fma_rn(v2, v2, sum);
        sum = seismic_fma_rn(v3, v3, sum);
        sum = seismic_fma_rn(v4, v4, sum);
        sum = seismic_fma_rn(v5, v5, sum);
        sum = seismic_fma_rn(v6, v6, sum);
        sum = seismic_fma_rn(v7, v7, sum);
    }
    for (; c < count; c += 8u) {
        const float val = sp[c];
        sum = seismic_fma_rn(val, val, sum);
    }
    return sum;
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
    static constexpr bool VEC4 = true;
    __device__ __forceinline__ void preload_row_regs(u32 m, u64 c, float4 &r0, float4 &r1) const {
        const float *row_ptr = x + rows(m) * stride;
        r0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        r1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (c + 8ull <= width) {
            if ((reinterpret_cast<u64>(row_ptr) & 15ull) == 0ull) {
                const float4 *src4 = reinterpret_cast<const float4 *>(row_ptr + c);
                r0 = src4[0];
                r1 = src4[1];
            } else {
                r0 = make_float4(row_ptr[c], row_ptr[c + 1ull], row_ptr[c + 2ull], row_ptr[c + 3ull]);
                r1 = make_float4(row_ptr[c + 4ull], row_ptr[c + 5ull], row_ptr[c + 6ull], row_ptr[c + 7ull]);
            }
        } else if (c < width) {
            r0 = make_float4((c + 0ull < width) ? row_ptr[c + 0ull] : 0.0f,
                             (c + 1ull < width) ? row_ptr[c + 1ull] : 0.0f,
                             (c + 2ull < width) ? row_ptr[c + 2ull] : 0.0f,
                             (c + 3ull < width) ? row_ptr[c + 3ull] : 0.0f);
            r1 = make_float4((c + 4ull < width) ? row_ptr[c + 4ull] : 0.0f,
                             (c + 5ull < width) ? row_ptr[c + 5ull] : 0.0f,
                             (c + 6ull < width) ? row_ptr[c + 6ull] : 0.0f,
                             (c + 7ull < width) ? row_ptr[c + 7ull] : 0.0f);
        }
    }
    __device__ __forceinline__ void ordered_inverses(u32 m_rows, float *inverses, u32 first_row = 0u,
                                                     float *scratch = nullptr) const {
        const u32 tid = threadIdx.x, threads = blockDim.x;
        const u32 group_id = tid / 8u, u = tid & 7u, groups = threads / 8u;
        if (scratch != nullptr && col_stride == 1ull && m_rows == 1u) {
            const float *row_ptr = x + rows(first_row) * stride;
            const bool aligned4 =
                (reinterpret_cast<u64>(row_ptr) & 15ull) == 0ull && (reinterpret_cast<u64>(scratch) & 15ull) == 0ull;
            float sum = 0.0f;
            for (u64 k0 = 0; k0 < width; k0 += 512ull) {
                const u32 chunk = (u32)min(512ull, width - k0);
                if (aligned4) {
                    const u32 chunk4 = chunk >> 2u;
                    const float4 *src4 = reinterpret_cast<const float4 *>(row_ptr + k0);
                    float4 *dst4 = reinterpret_cast<float4 *>(scratch);
                    for (u32 c4 = tid; c4 < chunk4; c4 += threads)
                        dst4[c4] = src4[c4];
                    const u32 tail = chunk4 << 2u;
                    for (u32 c = tail + tid; c < chunk; c += threads)
                        scratch[c] = row_ptr[k0 + c];
                } else {
                    for (u32 c = tid; c < chunk; c += threads)
                        scratch[c] = row_ptr[k0 + c];
                }
                __syncthreads();
                if (group_id == 0u)
                    sum = ordered_accumulate_sram8(scratch, chunk, u, sum);
                __syncthreads();
            }
            if (group_id == 0u) {
                const float total_sq = ordered_combine8(sum);
                if (tid == 0u)
                    inverses[0] = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(total_sq, (float)width) + eps));
            }
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
    __device__ __forceinline__ bool can_vec4(u32 m) const {
        const float *row = x + rows(m) * stride;
        return col_stride == 1ull && norm_stride == 1ull &&
               (reinterpret_cast<u64>(row) & 15ull) == 0ull &&
               (reinterpret_cast<u64>(norm) & (4 * NORM::bytes - 1)) == 0ull;
    }
    __device__ __forceinline__ void ordered_at4(u32 m, u64 c, float inv, float (&out)[4]) const {
        const float4 v = *reinterpret_cast<const float4 *>(x + rows(m) * stride + c);
        float w[4];
        element::span<NORM, 4, true>(norm, c, w);
        out[0] = Act::round(v.x * inv * w[0]);
        out[1] = Act::round(v.y * inv * w[1]);
        out[2] = Act::round(v.z * inv * w[2]);
        out[3] = Act::round(v.w * inv * w[3]);
    }
    __device__ __forceinline__ bool can_vec8(u32 m) const {
        const float *row = x + rows(m) * stride;
        return col_stride == 1ull && norm_stride == 1ull &&
               (reinterpret_cast<u64>(row) & 15ull) == 0ull &&
               (reinterpret_cast<u64>(norm) & 15ull) == 0ull;
    }
    __device__ __forceinline__ void ordered_at8(u32 m, u64 c, float inv, float (&out)[8]) const {
        const float *row = x + rows(m) * stride + c;
        const float4 x0 = *reinterpret_cast<const float4 *>(row);
        const float4 x1 = *reinterpret_cast<const float4 *>(row + 4ull);
        float w[8];
        element::span<NORM, 8, true>(norm, c, w);
        out[0] = Act::round(x0.x * inv * w[0]);
        out[1] = Act::round(x0.y * inv * w[1]);
        out[2] = Act::round(x0.z * inv * w[2]);
        out[3] = Act::round(x0.w * inv * w[3]);
        out[4] = Act::round(x1.x * inv * w[4]);
        out[5] = Act::round(x1.y * inv * w[5]);
        out[6] = Act::round(x1.z * inv * w[6]);
        out[7] = Act::round(x1.w * inv * w[7]);
    }
    __device__ __forceinline__ void prepare_row(float *shared, u32 m, float *) const {
        __shared__ float4 ordered_sram4[128];
        ordered_inverses(1u, shared, m, reinterpret_cast<float *>(ordered_sram4));
    }
    __device__ __forceinline__ float value(const float *factors, u32 m, u64 k) const {
        return ordered_at(m, k, factors[0]);
    }
    __device__ __forceinline__ void value4(const float *factors, u32 m, u64 k, float (&out)[4]) const {
        ordered_at4(m, k, factors[0], out);
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
    static constexpr bool VEC4 = true;
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
    __device__ __forceinline__ bool can_vec4(u32 m) const {
        const float *row = x + rows(m) * stride;
        return col_stride == 1ull && (reinterpret_cast<u64>(row) & 15ull) == 0ull &&
               (PARTS <= 1 || (width & 3ull) == 0ull);
    }
    __device__ __forceinline__ void ordered_at4(u32 m, u64 c, float, float (&out)[4]) const {
        const float *row = x + rows(m) * stride;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma unroll
        for (u64 part = 0; part < PARTS; ++part) {
            const float4 v = *reinterpret_cast<const float4 *>(row + part * width + c);
            s0 = seismic_add_rn(s0, v.x);
            s1 = seismic_add_rn(s1, v.y);
            s2 = seismic_add_rn(s2, v.z);
            s3 = seismic_add_rn(s3, v.w);
        }
        out[0] = Act::round(s0);
        out[1] = Act::round(s1);
        out[2] = Act::round(s2);
        out[3] = Act::round(s3);
    }
    __device__ __forceinline__ bool can_vec8(u32 m) const { return can_vec4(m); }
    __device__ __forceinline__ void ordered_at8(u32 m, u64 c, float inv, float (&out)[8]) const {
        float lo[4], hi[4];
        ordered_at4(m, c, inv, lo);
        ordered_at4(m, c + 4ull, inv, hi);
        out[0] = lo[0];
        out[1] = lo[1];
        out[2] = lo[2];
        out[3] = lo[3];
        out[4] = hi[0];
        out[5] = hi[1];
        out[6] = hi[2];
        out[7] = hi[3];
    }
    __device__ __forceinline__ void prepare_row(float *, u32, float *) const {}
    __device__ __forceinline__ float value(const float *, u32 m, u64 k) const {
        return ordered_at(m, k, 1.0f);
    }
    __device__ __forceinline__ void value4(const float *, u32 m, u64 k, float (&out)[4]) const {
        ordered_at4(m, k, 1.0f, out);
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

// Reference-order RMS inverse over preloaded register chunks (`CHUNKS * 2048` max width):
// streams `r0[ch], r1[ch]` through `sram` (`sram_floats` floats, >= 256, 16-byte aligned)
// so warp 0 lanes 0..7 accumulate `x[u], x[u + 8], ...` in bit-exact reference order.
template <u32 CHUNKS, class NORM, class Rows>
__device__ __forceinline__ float ordered_preload_inverse(const OrderedRms<NORM, Rows> &pro, u64 K,
                                                         const float4 (&r0)[CHUNKS], const float4 (&r1)[CHUNKS],
                                                         float *sram, u32 sram_floats, float *inv_shared) {
    const u32 tid = threadIdx.x;
    const u32 cap_threads = sram_floats >> 3u;
    const u32 cap_floats = cap_threads << 3u;
    float4 *sram4 = reinterpret_cast<float4 *>(sram);
    float sum = 0.0f;
    for (u32 base_slot = 0u; (u64)base_slot * 8ull < K; base_slot += cap_threads) {
        if (base_slot > 0u)
            __syncthreads();
        if (tid < 256u) {
#pragma unroll
            for (u32 ch = 0u; ch < CHUNKS; ++ch) {
                const u32 slot = ch * 256u + tid;
                if (slot >= base_slot && slot < base_slot + cap_threads) {
                    const u32 local_slot = slot - base_slot;
                    sram4[2u * local_slot] = r0[ch];
                    sram4[2u * local_slot + 1u] = r1[ch];
                }
            }
        }
        __syncthreads();
        if (tid < 8u) {
            const u32 lim = (u32)min(K - (u64)base_slot * 8ull, (u64)cap_floats);
            sum = ordered_accumulate_sram8(sram, lim, tid, sum);
        }
    }
    if (tid < 8u) {
        const float total_sq = ordered_combine8(sum);
        if (tid == 0u)
            inv_shared[0] = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(total_sq, (float)K) + pro.eps));
    }
    __syncthreads();
    return inv_shared[0];
}

template <class Pro, class Out>
__device__ __forceinline__ void ordered_stage_store_row(const Pro &pro, u32 m, u64 K, float inv, Out out, u64 stride0,
                                                        u64 stride1) {
    const u64 row_base = (u64)m * stride0;
    if (stride1 == 1ull && pro.can_vec8(m) && (reinterpret_cast<u64>(out.stored(row_base)) & 15ull) == 0ull) {
        const u64 k8 = K & ~7ull;
        const u64 step8 = (u64)blockDim.x * 8ull;
        for (u64 c = (u64)threadIdx.x * 8ull; c < k8; c += step8) {
            float vals[8];
            pro.ordered_at8(m, c, inv, vals);
            out.template put_span<8>(row_base + c, vals);
        }
        for (u64 c = k8 + threadIdx.x; c < K; c += blockDim.x)
            out.put(row_base + c, pro.ordered_at(m, c, inv));
    } else {
        for (u64 c = threadIdx.x; c < K; c += blockDim.x)
            out.put(row_base + c * stride1, pro.ordered_at(m, c, inv));
    }
}

template <class NORM, class Rows, class Out>
__device__ __forceinline__ float ordered_stage_row(const OrderedRms<NORM, Rows> &pro, u32 m, u64 K, Out out,
                                                   u64 stride0, u64 stride1) {
    __shared__ float inv_shared[1];
    __shared__ float4 stage_scratch4[512];
    float *stage_scratch = reinterpret_cast<float *>(stage_scratch4);
    const u64 row_base = (u64)m * stride0;
    if (blockDim.x == 256u && K == pro.width && K > 0ull && K <= 8192ull && pro.col_stride == 1ull) {
        const u32 tid = threadIdx.x;
        float4 r0[4], r1[4];
#pragma unroll
        for (u32 ch = 0u; ch < 4u; ++ch) {
            r0[ch] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            r1[ch] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const u64 c = (u64)(ch * 256u + tid) * 8ull;
            if (c < K)
                pro.preload_row_regs(m, c, r0[ch], r1[ch]);
        }
        const float inv = ordered_preload_inverse<4u>(pro, K, r0, r1, stage_scratch, 2048u, inv_shared);
        const bool fast_put = stride1 == 1ull && pro.norm_stride == 1ull &&
                              (reinterpret_cast<u64>(pro.norm) & 15ull) == 0ull &&
                              (reinterpret_cast<u64>(out.stored(row_base)) & 15ull) == 0ull;
#pragma unroll
        for (u32 ch = 0u; ch < 4u; ++ch) {
            const u64 c = (u64)(ch * 256u + tid) * 8ull;
            if (c < K) {
                if (fast_put && c + 8ull <= K) {
                    float w[8];
                    element::span<NORM, 8, true>(pro.norm, c, w);
                    float vals[8];
                    vals[0] = Act::round(r0[ch].x * inv * w[0]);
                    vals[1] = Act::round(r0[ch].y * inv * w[1]);
                    vals[2] = Act::round(r0[ch].z * inv * w[2]);
                    vals[3] = Act::round(r0[ch].w * inv * w[3]);
                    vals[4] = Act::round(r1[ch].x * inv * w[4]);
                    vals[5] = Act::round(r1[ch].y * inv * w[5]);
                    vals[6] = Act::round(r1[ch].z * inv * w[6]);
                    vals[7] = Act::round(r1[ch].w * inv * w[7]);
                    out.template put_span<8>(row_base + c, vals);
                } else {
                    const float raw_v[8] = {
                        r0[ch].x, r0[ch].y, r0[ch].z, r0[ch].w, r1[ch].x, r1[ch].y, r1[ch].z, r1[ch].w};
                    for (u32 j = 0u; j < 8u && c + j < K; ++j) {
                        const float val = Act::round(
                            raw_v[j] * inv * element::at<NORM>(pro.norm, (c + j) * pro.norm_stride));
                        out.put(row_base + (c + j) * stride1, val);
                    }
                }
            }
        }
        return inv;
    }
    pro.ordered_inverses(1u, inv_shared, m, stage_scratch);
    const float inv = inv_shared[0];
    ordered_stage_store_row(pro, m, K, inv, out, stride0, stride1);
    return inv;
}

template <class Pro, class Out>
__device__ __forceinline__ float ordered_stage_row(const Pro &pro, u32 m, u64 K, Out out, u64 stride0, u64 stride1) {
    __shared__ float inv_shared[1];
    __shared__ float4 stage_scratch4[128];
    pro.ordered_inverses(1u, inv_shared, m, reinterpret_cast<float *>(stage_scratch4));
    const float inv = inv_shared[0];
    ordered_stage_store_row(pro, m, K, inv, out, stride0, stride1);
    return inv;
}
