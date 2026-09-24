// qwen_embedding_rows: one block per token row gathers and decodes its table
// row through the K1 decoders (mma16 lane chunks) or reads a dense table.
// Results: the row rounded to A, and the same values as F32.
#if defined(SEISMIC_TABLE_KIND_PACKED)
#define SJ_W0 SEISMIC_TABLE
#endif
#include "common/packets.cuh"

extern "C" __global__ void qwen_embedding_rows(SEISMIC_KERNEL_PARAMS) {
    const sj::u64 row = blockIdx.x;
    if (row >= SEISMIC_DIM_M)
        return;
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    sj::u8 *embedded = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    float *wide = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER));
    const sj::u64 token = (sj::u64)tokens[row * SEISMIC_TOKENS_STRIDE_0];
    const sj::u64 out0 = row * SEISMIC_RESULT_0_STRIDE_0;
    const sj::u64 out1 = row * SEISMIC_RESULT_1_STRIDE_0;
#if defined(SEISMIC_TABLE_KIND_PACKED)
    const auto table = SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_TABLE));
    // Work item (k-block, column pair t) decodes 16 values of the row.
    for (sj::u64 item = threadIdx.x; item < SEISMIC_DIM_D / 64 * 4; item += blockDim.x) {
        const sj::u64 kb = item / 4;
        const sj::u32 t = (sj::u32)(item % 4);
        float values[16];
        sj::row_values16(table, token, kb, t, values);
        const sj::u32 offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
        for (int s = 0; s < 4; ++s)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const sj::u64 column = kb * 64 + 16 * s + offsets[j];
                const float value = sj::Act::round(values[4 * s + j]);
                sj::Act::store(embedded, out0 + column * SEISMIC_RESULT_0_STRIDE_1, value);
                wide[out1 + column * SEISMIC_RESULT_1_STRIDE_1] = value;
            }
    }
#else
    using Table = sj::Dense<SJ_DENSE_KIND(SEISMIC_TABLE)>;
    const sj::u8 *table = SEISMIC_PTR(SEISMIC_BUFFER_TABLE);
    for (sj::u64 column = threadIdx.x; column < SEISMIC_DIM_D; column += blockDim.x) {
        const float value =
            sj::Act::round(Table::load(table, token * SEISMIC_TABLE_STRIDE_0 + column * SEISMIC_TABLE_STRIDE_1));
        sj::Act::store(embedded, out0 + column * SEISMIC_RESULT_0_STRIDE_1, value);
        wide[out1 + column * SEISMIC_RESULT_1_STRIDE_1] = value;
    }
#endif
}
