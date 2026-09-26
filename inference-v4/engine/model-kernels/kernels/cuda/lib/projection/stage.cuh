// stage shared projection mechanisms; included at the original declaration point.
// ---------------------------------------------------------------------------
// Row maps: which input row feeds projected row m.

struct AllRows {
    __device__ __forceinline__ u64 operator()(u32 m) const { return m; }
};
struct SelectedRows {
    const int *rows;
    __device__ __forceinline__ u64 operator()(u32 m) const { return (u64)rows[m]; }
};

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
    __device__ __forceinline__ u32 pair(const float *, u32 m, u64 k) const {
        static_assert(!STAGED, "a GEMV reads staged rows");
        return *reinterpret_cast<const u32 *>(x + (rows(m) * stride + k) * 2);
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

// Per-head RMS of `mixed` gated by SiLU(z), heads of HEAD columns:
// x = round_A(round_A(v * rsqrt(sum_head v^2 / HEAD + eps) * w[k % HEAD]) * round_A(silu(z))).
template <class MIXED, class Z, class NORM, u32 HEAD, u32 HEADS, class Rows> struct GatedRms {
    const u8 *mixed;
    u64 mixed_stride;
    const u8 *z;
    u64 z_stride;
    const u8 *norm;
    float eps;
    Rows rows;
    static constexpr bool STAGED = true;
    static constexpr int FACTORS = HEADS;
    __device__ __forceinline__ float inverse(u32 m, u32 head) const {
        const u64 base = rows(m) * mixed_stride + (u64)head * HEAD;
        float total = 0.0f;
        for (u32 k = threadIdx.x % 32; k < HEAD; k += 32) {
            const float v = element::at<MIXED>(mixed, base + k);
            total = seismic_fma_rn(v, v, total);
        }
        total = seismic_warp_sum_f32(total);
        return rsqrtf(total / (float)HEAD + eps);
    }
    __device__ __forceinline__ void prepare_row(float *shared, u32 m, float *) const {
        const u32 warps = blockDim.x / 32;
        for (u32 head = threadIdx.x / 32; head < HEADS; head += warps) {
            const float value = inverse(m, head);
            if (threadIdx.x % 32 == 0)
                shared[head] = value;
        }
    }
    // The gated value with the head's inverse RMS `inv`.
    __device__ __forceinline__ float gated(float inv, u32 m, u64 k) const {
        const float v = element::at<MIXED>(mixed, rows(m) * mixed_stride + k);
        const float normalized = Act::round(v * inv * element::at<NORM>(norm, k % HEAD));
        const float activated = Act::round(silu(element::at<Z>(z, rows(m) * z_stride + k)));
        return normalized * activated;
    }
    __device__ __forceinline__ float value(const float *factors, u32 m, u64 k) const {
        return Act::round(gated(factors[k / HEAD], m, k));
    }
};

// The prologue's row m as A into out[m * K .. m * K + K), formed by the whole
// block: the staging launch (one block per row, global scratch) and a GEMV
// block at M = 1 (its shared memory) both form rows with it.
template <class Pro> __device__ __forceinline__ void form_row(const Pro &pro, u32 m, u64 K, u8 *out) {
    static_assert(Pro::STAGED, "an input read in place is not formed");
    __shared__ float factors[Pro::FACTORS > 0 ? Pro::FACTORS : 1];
    __shared__ float scratch[32];
    pro.prepare_row(factors, m, scratch);
    __syncthreads();
    for (u64 k = threadIdx.x; k < K; k += blockDim.x)
        element::put<Act>(out, m * K + k, pro.value(factors, m, k));
}

// int8 activations (the GEMM's INT8 candidate): rows quantized per 32-code
// group as q8_1 does, x ~= d * q with d = max|x| / 127 and q = rint(x / d),
// staged by `stage_row_s8` as `q` ([M, width] bytes in the s8 fragments'
// virtual k order, packets.cuh) and `groups` (each group's (d, d * sum q)).
struct QuantizedRows {
    const u8 *q;
    u64 width;
    const float2 *groups;
};

// Real offset within a 32-code group of virtual s8 position v.
__device__ __forceinline__ u32 s8_real_offset(u32 v) {
    const u32 half = v / 16, t = (v % 16) / 4, j = v % 4;
    return half * 16 + 2 * t + (j % 2) + 8 * (j / 2);
}

// Staging launch of the INT8 candidate, one block per row m: the prologue's
// row x[m, :] (rounded to A) quantized per 32 codes into `q` ([M, K] bytes,
// virtual order) and `groups` ([M, K / 32] (d, d * sum q)).
template <class Pro>
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
        const float d = maximum / 127.0f;
        int total = 0;
        u32 words[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            u32 word = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int code = d == 0.0f ? 0 : __float2int_rn(values[s8_real_offset(4 * w + b)] / d);
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

