// corrector_nodes: 4 elements per thread of node (slot s, mask row j,
// candidate k) on a (column, node, slot) grid: its token, pass, pass bias,
// feature, feature bias, slot and rank values, then the token bias when
// TB = 1, summed in the portable body's order.
#include <seismic/element.cuh>

extern "C" __global__ void corrector_nodes(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *token = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN));
    const float *token_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN_BIAS));
    const float *pass = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PASS));
    const float *pass_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PASS_BIAS));
    const float *feature = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_FEATURE));
    const float *feature_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_FEATURE_BIAS));
    const float *slot_rows = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SLOT));
    const float *rank_rows = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RANK));
    const auto result = SEISMIC_RESULT_0_OUT;
    const u64 slots = SEISMIC_DIM_S, rows = SEISMIC_DIM_J, count = SEISMIC_DIM_K, width = SEISMIC_DIM_H;
    const u64 col = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4ull;
    if (col >= width)
        return;
    const u64 nodes_per_slot = rows * count;
    const u64 end = (col + 4ull < width) ? (col + 4ull) : width;
    const bool unit_strides = SEISMIC_TOKEN_STRIDE_3 == 1ull && SEISMIC_PASS_STRIDE_2 == 1ull &&
                              SEISMIC_PASS_BIAS_STRIDE_0 == 1ull && SEISMIC_FEATURE_STRIDE_3 == 1ull &&
                              SEISMIC_FEATURE_BIAS_STRIDE_0 == 1ull && SEISMIC_SLOT_STRIDE_1 == 1ull &&
                              SEISMIC_RANK_STRIDE_1 == 1ull && SEISMIC_RESULT_0_STRIDE_3 == 1ull &&
                              (SEISMIC_DIM_TB == 0 || SEISMIC_TOKEN_BIAS_STRIDE_1 == 1ull);

    for (u64 slot = blockIdx.z; slot < slots; slot += gridDim.z) {
        for (u64 node = blockIdx.y; node < nodes_per_slot; node += gridDim.y) {
            const u64 mask = node / count;
            const u64 candidate = node % count;
            const float *t_row = token + slot * SEISMIC_TOKEN_STRIDE_0 + mask * SEISMIC_TOKEN_STRIDE_1 +
                                 candidate * SEISMIC_TOKEN_STRIDE_2;
            const float *p_row = pass + slot * SEISMIC_PASS_STRIDE_0 + mask * SEISMIC_PASS_STRIDE_1;
            const float *f_row = feature + slot * SEISMIC_FEATURE_STRIDE_0 + mask * SEISMIC_FEATURE_STRIDE_1 +
                                 candidate * SEISMIC_FEATURE_STRIDE_2;
            const float *s_row = slot_rows + mask * SEISMIC_SLOT_STRIDE_0;
            const float *r_row = rank_rows + candidate * SEISMIC_RANK_STRIDE_0;
            const u64 out_base = slot * SEISMIC_RESULT_0_STRIDE_0 + mask * SEISMIC_RESULT_0_STRIDE_1 +
                                 candidate * SEISMIC_RESULT_0_STRIDE_2;

            u64 ptr_bits = reinterpret_cast<u64>(t_row + col) | reinterpret_cast<u64>(p_row + col) |
                           reinterpret_cast<u64>(pass_bias + col) | reinterpret_cast<u64>(f_row + col) |
                           reinterpret_cast<u64>(feature_bias + col) | reinterpret_cast<u64>(s_row + col) |
                           reinterpret_cast<u64>(r_row + col) |
                           reinterpret_cast<u64>(result.stored(out_base + col));
            if (SEISMIC_DIM_TB == 1)
                ptr_bits |= reinterpret_cast<u64>(token_bias + col);

            if (col + 4ull <= width && unit_strides && (ptr_bits & 15ull) == 0ull) {
                const float4 tv = *reinterpret_cast<const float4 *>(t_row + col);
                const float4 pv = *reinterpret_cast<const float4 *>(p_row + col);
                const float4 pbv = *reinterpret_cast<const float4 *>(pass_bias + col);
                const float4 fv = *reinterpret_cast<const float4 *>(f_row + col);
                const float4 fbv = *reinterpret_cast<const float4 *>(feature_bias + col);
                const float4 sv = *reinterpret_cast<const float4 *>(s_row + col);
                const float4 rv = *reinterpret_cast<const float4 *>(r_row + col);
                float out[4];
                out[0] = tv.x + pv.x + pbv.x + fv.x + fbv.x + sv.x + rv.x;
                out[1] = tv.y + pv.y + pbv.y + fv.y + fbv.y + sv.y + rv.y;
                out[2] = tv.z + pv.z + pbv.z + fv.z + fbv.z + sv.z + rv.z;
                out[3] = tv.w + pv.w + pbv.w + fv.w + fbv.w + sv.w + rv.w;
                if (SEISMIC_DIM_TB == 1) {
                    const float4 tbv = *reinterpret_cast<const float4 *>(token_bias + col);
                    out[0] += tbv.x;
                    out[1] += tbv.y;
                    out[2] += tbv.z;
                    out[3] += tbv.w;
                }
                result.put_span<4>(out_base + col, out);
            } else {
                for (u64 c = col; c < end; ++c) {
                    float value = t_row[c * SEISMIC_TOKEN_STRIDE_3];
                    value += p_row[c * SEISMIC_PASS_STRIDE_2];
                    value += pass_bias[c * SEISMIC_PASS_BIAS_STRIDE_0];
                    value += f_row[c * SEISMIC_FEATURE_STRIDE_3];
                    value += feature_bias[c * SEISMIC_FEATURE_BIAS_STRIDE_0];
                    value += s_row[c * SEISMIC_SLOT_STRIDE_1];
                    value += r_row[c * SEISMIC_RANK_STRIDE_1];
                    if (SEISMIC_DIM_TB == 1)
                        value += token_bias[c * SEISMIC_TOKEN_BIAS_STRIDE_1];
                    result.put(out_base + c * SEISMIC_RESULT_0_STRIDE_3, value);
                }
            }
        }
    }
}
