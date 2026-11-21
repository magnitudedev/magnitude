// repeat_rows: S scaled copies of each row, four columns across all S streams
// per thread (loads each source quad once and scales into every stream).
#include <seismic/element.cuh>

extern "C" __global__ void repeat_rows(SEISMIC_KERNEL_PARAMS) {
    using element::u32;
    using element::u64;
    const u32 h = static_cast<u32>(SEISMIC_DIM_H);
    const u32 streams = static_cast<u32>(SEISMIC_DIM_S);
    const u32 vecs_per_row = (h + 3u) / 4u;
    const u64 index = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<u64>(SEISMIC_DIM_M) * vecs_per_row)
        return;
    const u64 row = index / vecs_per_row;
    const u32 column = static_cast<u32>(index - row * vecs_per_row) * 4u;
    const float *src_row =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROWS)) + row * SEISMIC_ROWS_STRIDE_0;
    const float *scales = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES));
    const auto dst_row = SEISMIC_RESULT_0_OUT.offset(row * SEISMIC_RESULT_0_STRIDE_0);
    if (column + 4u <= h) {
        float4 values;
        if (SEISMIC_ROWS_STRIDE_1 == 1 && (reinterpret_cast<u64>(src_row + column) & 15u) == 0u) {
            values = *reinterpret_cast<const float4 *>(src_row + column);
        } else {
            values = make_float4(src_row[static_cast<u64>(column) * SEISMIC_ROWS_STRIDE_1],
                                 src_row[static_cast<u64>(column + 1u) * SEISMIC_ROWS_STRIDE_1],
                                 src_row[static_cast<u64>(column + 2u) * SEISMIC_ROWS_STRIDE_1],
                                 src_row[static_cast<u64>(column + 3u) * SEISMIC_ROWS_STRIDE_1]);
        }
        const bool unit_dst = SEISMIC_RESULT_0_STRIDE_1 == 1;
        for (u32 s = 0; s < streams; ++s) {
            const float scale = scales[static_cast<u64>(s) * SEISMIC_SCALES_STRIDE_0];
            const u64 offset = static_cast<u64>(s) * h + column;
            const float scaled[4] = {values.x * scale, values.y * scale, values.z * scale, values.w * scale};
            if (unit_dst && (reinterpret_cast<u64>(dst_row.stored(offset)) & 15u) == 0u) {
                dst_row.put_span<4>(offset, scaled);
            } else {
                dst_row.put(offset * SEISMIC_RESULT_0_STRIDE_1, scaled[0]);
                dst_row.put((offset + 1u) * SEISMIC_RESULT_0_STRIDE_1, scaled[1]);
                dst_row.put((offset + 2u) * SEISMIC_RESULT_0_STRIDE_1, scaled[2]);
                dst_row.put((offset + 3u) * SEISMIC_RESULT_0_STRIDE_1, scaled[3]);
            }
        }
    } else {
        for (u32 c = column; c < h; ++c) {
            const float value = src_row[static_cast<u64>(c) * SEISMIC_ROWS_STRIDE_1];
            for (u32 s = 0; s < streams; ++s) {
                const float scale = scales[static_cast<u64>(s) * SEISMIC_SCALES_STRIDE_0];
                dst_row.put((static_cast<u64>(s) * h + c) * SEISMIC_RESULT_0_STRIDE_1, value * scale);
            }
        }
    }
}
