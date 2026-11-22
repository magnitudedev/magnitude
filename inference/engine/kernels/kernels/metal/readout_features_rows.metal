// readout_features_rows: the final RMS normalization of the `out_rows` hidden
// rows, published in the activation type. One threadgroup per output row.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef activation::Act Act;
typedef ELEMENT_OF(SEISMIC_NORM) Norm;

kernel void readout_features_rows(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],
    SEISMIC_RESULT_0_PARAM(features),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    const element::Out<Act> out(features);
    const ulong width = SEISMIC_DIM_D;
    const ulong width4 = width & ~3ul;
    const ulong src_row = ulong(out_rows[ulong(row) * SEISMIC_OUT_ROWS_STRIDE_0]);
    device const float *src = hidden + src_row * SEISMIC_HIDDEN_STRIDE_0;
    const ulong out_row = ulong(row) * SEISMIC_RESULT_0_STRIDE_0;
    if constexpr (SEISMIC_DIM_NM > 0) {
        const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
        const bool hidden_vec = SEISMIC_HIDDEN_STRIDE_2 == 1ul;
        const bool store_vec = hidden_vec && SEISMIC_NORM_STRIDE_0 == 1ul && SEISMIC_RESULT_0_STRIDE_1 == 1ul;
        float squares = 0.0f;
        if constexpr (SEISMIC_DIM_D <= 8192ul) {
            constexpr ulong CACHE = 8ul;
            float4 cached[CACHE];
            float tail_cached = 0.0f;
            const ulong base = ulong(thread_index) * 4ul;
            const ulong tail = width4 + thread_index;
            if (hidden_vec) {
#pragma unroll
                for (ulong slot = 0ul; slot < CACHE; ++slot) {
                    const ulong i = base + slot * 1024ul;
                    if (i < width4) {
                        const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(src + i));
                        cached[slot] = v;
                        squares = metal::fma(v.x, v.x, squares);
                        squares = metal::fma(v.y, v.y, squares);
                        squares = metal::fma(v.z, v.z, squares);
                        squares = metal::fma(v.w, v.w, squares);
                    }
                }
                if (tail < width) {
                    tail_cached = src[tail];
                    squares = metal::fma(tail_cached, tail_cached, squares);
                }
            } else {
                for (ulong i = thread_index; i < width; i += 256u) {
                    const float v = src[i * SEISMIC_HIDDEN_STRIDE_2];
                    squares = metal::fma(v, v, squares);
                }
            }
            const float total = reduce::group_sum<8>(squares, partials, sg, lane);
            const float inverse = metal::rsqrt(total / float(width) + epsilon);
            if (store_vec) {
                device const Norm::storage *norm_base = reinterpret_cast<device const Norm::storage *>(norm);
#pragma unroll
                for (ulong slot = 0ul; slot < CACHE; ++slot) {
                    const ulong i = base + slot * 1024ul;
                    if (i < width4) {
                        const float4 v = cached[slot];
                        const float4 nv = Norm::load4(norm_base + i);
                        out.put4(out_row + i, (v * inverse) * nv);
                    }
                }
                if (tail < width) {
                    out.put(out_row + tail, (tail_cached * inverse) * element::at<Norm>(norm, tail));
                }
            } else if (hidden_vec) {
#pragma unroll
                for (ulong slot = 0ul; slot < CACHE; ++slot) {
                    const ulong i = base + slot * 1024ul;
                    if (i < width4) {
                        const float4 v = cached[slot];
                        out.put(out_row + (i + 0ul) * SEISMIC_RESULT_0_STRIDE_1,
                            (v.x * inverse) * element::at<Norm>(norm, (i + 0ul) * SEISMIC_NORM_STRIDE_0));
                        out.put(out_row + (i + 1ul) * SEISMIC_RESULT_0_STRIDE_1,
                            (v.y * inverse) * element::at<Norm>(norm, (i + 1ul) * SEISMIC_NORM_STRIDE_0));
                        out.put(out_row + (i + 2ul) * SEISMIC_RESULT_0_STRIDE_1,
                            (v.z * inverse) * element::at<Norm>(norm, (i + 2ul) * SEISMIC_NORM_STRIDE_0));
                        out.put(out_row + (i + 3ul) * SEISMIC_RESULT_0_STRIDE_1,
                            (v.w * inverse) * element::at<Norm>(norm, (i + 3ul) * SEISMIC_NORM_STRIDE_0));
                    }
                }
                if (tail < width) {
                    out.put(out_row + tail * SEISMIC_RESULT_0_STRIDE_1,
                        (tail_cached * inverse) * element::at<Norm>(norm, tail * SEISMIC_NORM_STRIDE_0));
                }
            } else {
                for (ulong i = thread_index; i < width; i += 256u) {
                    out.put(out_row + i * SEISMIC_RESULT_0_STRIDE_1,
                        (src[i * SEISMIC_HIDDEN_STRIDE_2] * inverse)
                            * element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0));
                }
            }
        } else {
            if (hidden_vec) {
                for (ulong i = ulong(thread_index) * 4ul; i < width4; i += 1024ul) {
                    const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(src + i));
                    squares = metal::fma(v.x, v.x, squares);
                    squares = metal::fma(v.y, v.y, squares);
                    squares = metal::fma(v.z, v.z, squares);
                    squares = metal::fma(v.w, v.w, squares);
                }
                for (ulong i = width4 + thread_index; i < width; i += 256u) {
                    const float v = src[i];
                    squares = metal::fma(v, v, squares);
                }
            } else {
                for (ulong i = thread_index; i < width; i += 256u) {
                    const float v = src[i * SEISMIC_HIDDEN_STRIDE_2];
                    squares = metal::fma(v, v, squares);
                }
            }
            const float total = reduce::group_sum<8>(squares, partials, sg, lane);
            const float inverse = metal::rsqrt(total / float(width) + epsilon);
            if (store_vec) {
                device const Norm::storage *norm_base = reinterpret_cast<device const Norm::storage *>(norm);
                for (ulong i = ulong(thread_index) * 4ul; i < width4; i += 1024ul) {
                    const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(src + i));
                    const float4 nv = Norm::load4(norm_base + i);
                    out.put4(out_row + i, (v * inverse) * nv);
                }
                for (ulong i = width4 + thread_index; i < width; i += 256u) {
                    out.put(out_row + i, (src[i] * inverse) * element::at<Norm>(norm, i));
                }
            } else {
                for (ulong i = thread_index; i < width; i += 256u) {
                    out.put(out_row + i * SEISMIC_RESULT_0_STRIDE_1,
                        (src[i * SEISMIC_HIDDEN_STRIDE_2] * inverse)
                            * element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0));
                }
            }
        }
    } else {
        (void)partials;
        if (SEISMIC_HIDDEN_STRIDE_2 == 1ul && SEISMIC_RESULT_0_STRIDE_1 == 1ul) {
            for (ulong i = ulong(thread_index) * 4ul; i < width4; i += 1024ul) {
                float4 sum = float4(0.0f);
                for (ulong part = 0; part < SEISMIC_DIM_PT; ++part) {
                    const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(
                        src + part * SEISMIC_HIDDEN_STRIDE_1 + i));
                    sum += v;
                }
                out.put4(out_row + i, sum);
            }
            for (ulong i = width4 + thread_index; i < width; i += 256u) {
                float sum = 0.0f;
                for (ulong part = 0; part < SEISMIC_DIM_PT; ++part)
                    sum += src[part * SEISMIC_HIDDEN_STRIDE_1 + i];
                out.put(out_row + i, sum);
            }
        } else {
            for (ulong i = thread_index; i < width; i += 256u) {
                float sum = 0.0f;
                for (ulong part = 0; part < SEISMIC_DIM_PT; ++part)
                    sum += src[part * SEISMIC_HIDDEN_STRIDE_1 + i * SEISMIC_HIDDEN_STRIDE_2];
                out.put(out_row + i * SEISMIC_RESULT_0_STRIDE_1, sum);
            }
        }
    }
}
