// readout_features_rows: the final RMS prologue over the demanded rows
// (`out_rows` gather), published as A. One block per output row.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using element::u64;
using element::u8;

extern "C" __global__ void readout_features_rows(SEISMIC_KERNEL_PARAMS) {
    using Act = activation::Act;
    using Norm = ELEMENT_OF(SEISMIC_NORM);
    __shared__ float partials[32];
    const u64 row = blockIdx.x;
    const u64 width = SEISMIC_DIM_D;
    const u64 width4 = width & ~3ull;
    const u64 step4 = static_cast<u64>(blockDim.x) * 4ull;
    const int *out_rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS));
    const u64 src_row = static_cast<u64>(out_rows[row * SEISMIC_OUT_ROWS_STRIDE_0]);
    const float *src = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN))
        + src_row * SEISMIC_HIDDEN_STRIDE_0;
    const auto features = SEISMIC_RESULT_0_OUT.offset(row * SEISMIC_RESULT_0_STRIDE_0);
    const bool hidden_vec = SEISMIC_HIDDEN_STRIDE_2 == 1ull && (reinterpret_cast<u64>(src) & 15ull) == 0ull;
    const bool features_vec = SEISMIC_RESULT_0_STRIDE_1 == 1ull &&
                              (reinterpret_cast<u64>(features.stored(0)) & (4ull * Act::bytes - 1ull)) == 0ull;
    if constexpr (SEISMIC_DIM_NM > 0) {
        const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
        const u8 *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
        const bool store_vec = hidden_vec &&
                               features_vec &&
                               SEISMIC_NORM_STRIDE_0 == 1ull &&
                               (reinterpret_cast<u64>(norm) & (4ull * Norm::bytes - 1ull)) == 0ull;
        float squares = 0.0f;
        if constexpr (SEISMIC_DIM_D <= 8192ull) {
            constexpr u64 CACHE = 8ull;
            float4 cached[CACHE];
            float tail_cached = 0.0f;
            const bool can_cache = hidden_vec && width <= CACHE * step4;
            const u64 base = static_cast<u64>(threadIdx.x) * 4ull;
            const u64 tail = width4 + threadIdx.x;
            if (can_cache) {
#pragma unroll
                for (u64 slot = 0; slot < CACHE; ++slot) {
                    const u64 i = base + slot * step4;
                    if (i < width4) {
                        const float4 v = *reinterpret_cast<const float4 *>(src + i);
                        cached[slot] = v;
                        squares = seismic_fma_rn(v.x, v.x, squares);
                        squares = seismic_fma_rn(v.y, v.y, squares);
                        squares = seismic_fma_rn(v.z, v.z, squares);
                        squares = seismic_fma_rn(v.w, v.w, squares);
                    }
                }
                if (tail < width) {
                    tail_cached = src[tail];
                    squares = seismic_fma_rn(tail_cached, tail_cached, squares);
                }
            } else if (hidden_vec) {
                for (u64 i = base; i < width4; i += step4) {
                    const float4 v = *reinterpret_cast<const float4 *>(src + i);
                    squares = seismic_fma_rn(v.x, v.x, squares);
                    squares = seismic_fma_rn(v.y, v.y, squares);
                    squares = seismic_fma_rn(v.z, v.z, squares);
                    squares = seismic_fma_rn(v.w, v.w, squares);
                }
                for (u64 i = tail; i < width; i += blockDim.x) {
                    const float v = src[i];
                    squares = seismic_fma_rn(v, v, squares);
                }
            } else {
                for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
                    const float v = src[i * SEISMIC_HIDDEN_STRIDE_2];
                    squares = seismic_fma_rn(v, v, squares);
                }
            }
            const float total = reduce::group_sum(squares, partials);
            const float inverse = rsqrtf(total / (float)width + epsilon);
            if (can_cache && store_vec) {
#pragma unroll
                for (u64 slot = 0; slot < CACHE; ++slot) {
                    const u64 i = base + slot * step4;
                    if (i < width4) {
                        const float4 v = cached[slot];
                        float nv[4];
                        element::span<Norm, 4, true>(norm, i, nv);
                        float out[4];
                        out[0] = (v.x * inverse) * nv[0];
                        out[1] = (v.y * inverse) * nv[1];
                        out[2] = (v.z * inverse) * nv[2];
                        out[3] = (v.w * inverse) * nv[3];
                        features.put_span<4>(i, out);
                    }
                }
                if (tail < width) {
                    features.put(tail, (tail_cached * inverse) * element::at<Norm>(norm, tail));
                }
            } else if (can_cache) {
#pragma unroll
                for (u64 slot = 0; slot < CACHE; ++slot) {
                    const u64 i = base + slot * step4;
                    if (i < width4) {
                        const float4 v = cached[slot];
                        features.put((i + 0ull) * SEISMIC_RESULT_0_STRIDE_1,
                                     (v.x * inverse) * element::at<Norm>(norm, (i + 0ull) * SEISMIC_NORM_STRIDE_0));
                        features.put((i + 1ull) * SEISMIC_RESULT_0_STRIDE_1,
                                     (v.y * inverse) * element::at<Norm>(norm, (i + 1ull) * SEISMIC_NORM_STRIDE_0));
                        features.put((i + 2ull) * SEISMIC_RESULT_0_STRIDE_1,
                                     (v.z * inverse) * element::at<Norm>(norm, (i + 2ull) * SEISMIC_NORM_STRIDE_0));
                        features.put((i + 3ull) * SEISMIC_RESULT_0_STRIDE_1,
                                     (v.w * inverse) * element::at<Norm>(norm, (i + 3ull) * SEISMIC_NORM_STRIDE_0));
                    }
                }
                if (tail < width) {
                    features.put(tail * SEISMIC_RESULT_0_STRIDE_1,
                                 (tail_cached * inverse) * element::at<Norm>(norm, tail * SEISMIC_NORM_STRIDE_0));
                }
            } else if (store_vec) {
                for (u64 i = base; i < width4; i += step4) {
                    const float4 v = *reinterpret_cast<const float4 *>(src + i);
                    float nv[4];
                    element::span<Norm, 4, true>(norm, i, nv);
                    float out[4];
                    out[0] = (v.x * inverse) * nv[0];
                    out[1] = (v.y * inverse) * nv[1];
                    out[2] = (v.z * inverse) * nv[2];
                    out[3] = (v.w * inverse) * nv[3];
                    features.put_span<4>(i, out);
                }
                for (u64 i = tail; i < width; i += blockDim.x) {
                    features.put(i, (src[i] * inverse) * element::at<Norm>(norm, i));
                }
            } else {
                for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
                    features.put(i * SEISMIC_RESULT_0_STRIDE_1,
                                 (src[i * SEISMIC_HIDDEN_STRIDE_2] * inverse) *
                                     element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0));
                }
            }
        } else {
            if (hidden_vec) {
                for (u64 i = static_cast<u64>(threadIdx.x) * 4ull; i < width4; i += step4) {
                    const float4 v = *reinterpret_cast<const float4 *>(src + i);
                    squares = seismic_fma_rn(v.x, v.x, squares);
                    squares = seismic_fma_rn(v.y, v.y, squares);
                    squares = seismic_fma_rn(v.z, v.z, squares);
                    squares = seismic_fma_rn(v.w, v.w, squares);
                }
                for (u64 i = width4 + threadIdx.x; i < width; i += blockDim.x) {
                    const float v = src[i];
                    squares = seismic_fma_rn(v, v, squares);
                }
            } else {
                for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
                    const float v = src[i * SEISMIC_HIDDEN_STRIDE_2];
                    squares = seismic_fma_rn(v, v, squares);
                }
            }
            const float total = reduce::group_sum(squares, partials);
            const float inverse = rsqrtf(total / (float)width + epsilon);
            if (store_vec) {
                for (u64 i = static_cast<u64>(threadIdx.x) * 4ull; i < width4; i += step4) {
                    const float4 v = *reinterpret_cast<const float4 *>(src + i);
                    float nv[4];
                    element::span<Norm, 4, true>(norm, i, nv);
                    float out[4];
                    out[0] = (v.x * inverse) * nv[0];
                    out[1] = (v.y * inverse) * nv[1];
                    out[2] = (v.z * inverse) * nv[2];
                    out[3] = (v.w * inverse) * nv[3];
                    features.put_span<4>(i, out);
                }
                for (u64 i = width4 + threadIdx.x; i < width; i += blockDim.x) {
                    features.put(i, (src[i] * inverse) * element::at<Norm>(norm, i));
                }
            } else {
                for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
                    features.put(i * SEISMIC_RESULT_0_STRIDE_1,
                                 (src[i * SEISMIC_HIDDEN_STRIDE_2] * inverse) *
                                     element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0));
                }
            }
        }
    } else {
        (void)partials;
        const bool sum_vec = hidden_vec &&
                             features_vec &&
                             (SEISMIC_DIM_PT == 1 || (SEISMIC_HIDDEN_STRIDE_1 & 3ull) == 0ull);
        if (sum_vec) {
            for (u64 i = static_cast<u64>(threadIdx.x) * 4ull; i < width4; i += step4) {
                float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (u64 part = 0; part < SEISMIC_DIM_PT; ++part) {
                    const float4 v = *reinterpret_cast<const float4 *>(src + part * SEISMIC_HIDDEN_STRIDE_1 + i);
                    out[0] = seismic_add_rn(out[0], v.x);
                    out[1] = seismic_add_rn(out[1], v.y);
                    out[2] = seismic_add_rn(out[2], v.z);
                    out[3] = seismic_add_rn(out[3], v.w);
                }
                features.put_span<4>(i, out);
            }
            for (u64 i = width4 + threadIdx.x; i < width; i += blockDim.x) {
                float sum = 0.0f;
#pragma unroll
                for (u64 part = 0; part < SEISMIC_DIM_PT; ++part)
                    sum = seismic_add_rn(sum, src[part * SEISMIC_HIDDEN_STRIDE_1 + i]);
                features.put(i, sum);
            }
        } else {
            for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
                float sum = 0.0f;
#pragma unroll
                for (u64 part = 0; part < SEISMIC_DIM_PT; ++part)
                    sum = seismic_add_rn(sum, src[part * SEISMIC_HIDDEN_STRIDE_1 + i * SEISMIC_HIDDEN_STRIDE_2]);
                features.put(i * SEISMIC_RESULT_0_STRIDE_1, sum);
            }
        }
    }
}
