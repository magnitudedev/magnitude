// corrector_rows: one block of 256 threads per row (4 elements per thread per
// iteration): values + bias, then by form the identity, silu or unit length
// (the row's sum of squares reduced over the block), plus the residual row
// when R = 1, published to Y.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

extern "C" __global__ void corrector_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    using ResultElem = ELEMENT_OF(SEISMIC_RESULT_0);
    __shared__ float partials[32];
    const float *values = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_VALUES));
    const float *bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS));
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const u64 row = blockIdx.x;
    const u64 width = SEISMIC_DIM_N;
    const u64 width4 = width & ~3ull;
    const int form = (int)SEISMIC_PARAM_FORM;
    const float *v_row = values + row * SEISMIC_VALUES_STRIDE_0;
    const float *r_row = residual + row * SEISMIC_RESIDUAL_STRIDE_1;
    const auto result_row = SEISMIC_RESULT_0_OUT.offset(row * SEISMIC_RESULT_0_STRIDE_0);

    const bool bias_vec = SEISMIC_VALUES_STRIDE_1 == 1ull && SEISMIC_BIAS_STRIDE_0 == 1ull &&
                          ((reinterpret_cast<u64>(v_row) | reinterpret_cast<u64>(bias)) & 15ull) == 0ull;

    // Form 2's length, max(sqrt(sum of squares), epsilon).
    float length = 1.0f;
    if (form == 2) {
        float squares = 0.0f;
        if (bias_vec) {
            const u64 step4 = static_cast<u64>(blockDim.x) * 4ull;
            for (u64 column = static_cast<u64>(threadIdx.x) * 4ull; column < width4; column += step4) {
                const float4 vv = *reinterpret_cast<const float4 *>(v_row + column);
                const float4 bv = *reinterpret_cast<const float4 *>(bias + column);
                const float b0 = vv.x + bv.x;
                const float b1 = vv.y + bv.y;
                const float b2 = vv.z + bv.z;
                const float b3 = vv.w + bv.w;
                squares = seismic_fma_rn(b0, b0, squares);
                squares = seismic_fma_rn(b1, b1, squares);
                squares = seismic_fma_rn(b2, b2, squares);
                squares = seismic_fma_rn(b3, b3, squares);
            }
            for (u64 column = width4 + threadIdx.x; column < width; column += blockDim.x) {
                const float value = v_row[column] + bias[column];
                squares = seismic_fma_rn(value, value, squares);
            }
        } else {
            for (u64 column = threadIdx.x; column < width; column += blockDim.x) {
                const float value =
                    v_row[column * SEISMIC_VALUES_STRIDE_1] + bias[column * SEISMIC_BIAS_STRIDE_0];
                squares = seismic_fma_rn(value, value, squares);
            }
        }
        length = fmaxf(sqrtf(reduce::group_sum(squares, partials)), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON));
    }

    const bool out_vec =
        bias_vec && SEISMIC_RESULT_0_STRIDE_1 == 1ull &&
        (reinterpret_cast<u64>(result_row.stored(0)) & (4ull * ResultElem::bytes - 1ull)) == 0ull &&
        (SEISMIC_DIM_R == 0 || (SEISMIC_RESIDUAL_STRIDE_2 == 1ull && (reinterpret_cast<u64>(r_row) & 15ull) == 0ull));

    if (out_vec) {
        const u64 step4 = static_cast<u64>(blockDim.x) * 4ull;
        for (u64 column = static_cast<u64>(threadIdx.x) * 4ull; column < width4; column += step4) {
            const float4 vv = *reinterpret_cast<const float4 *>(v_row + column);
            const float4 bv = *reinterpret_cast<const float4 *>(bias + column);
            float out[4];
            out[0] = vv.x + bv.x;
            out[1] = vv.y + bv.y;
            out[2] = vv.z + bv.z;
            out[3] = vv.w + bv.w;
            if (form == 1) {
                out[0] = out[0] / (1.0f + expf(-out[0]));
                out[1] = out[1] / (1.0f + expf(-out[1]));
                out[2] = out[2] / (1.0f + expf(-out[2]));
                out[3] = out[3] / (1.0f + expf(-out[3]));
            } else if (form == 2) {
                out[0] = out[0] / length;
                out[1] = out[1] / length;
                out[2] = out[2] / length;
                out[3] = out[3] / length;
            }
            if (SEISMIC_DIM_R == 1) {
                const float4 rv = *reinterpret_cast<const float4 *>(r_row + column);
                out[0] += rv.x;
                out[1] += rv.y;
                out[2] += rv.z;
                out[3] += rv.w;
            }
            result_row.put_span<4>(column, out);
        }
        for (u64 column = width4 + threadIdx.x; column < width; column += blockDim.x) {
            float value = v_row[column] + bias[column];
            if (form == 1)
                value = value / (1.0f + expf(-value));
            else if (form == 2)
                value = value / length;
            if (SEISMIC_DIM_R == 1)
                value += r_row[column];
            result_row.put(column, value);
        }
    } else {
        for (u64 column = threadIdx.x; column < width; column += blockDim.x) {
            float value = v_row[column * SEISMIC_VALUES_STRIDE_1] + bias[column * SEISMIC_BIAS_STRIDE_0];
            if (form == 1)
                value = value / (1.0f + expf(-value));
            else if (form == 2)
                value = value / length;
            if (SEISMIC_DIM_R == 1)
                value += r_row[column * SEISMIC_RESIDUAL_STRIDE_2];
            result_row.put(column * SEISMIC_RESULT_0_STRIDE_1, value);
        }
    }
}
