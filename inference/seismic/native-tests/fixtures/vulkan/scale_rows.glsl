// ROWS rows per group; the group's lanes stride over the columns.
void scale_rows_body() {
    const Out result = SEISMIC_RESULT_0_OUT;
    const float factor = uintBitsToFloat(uint(SEISMIC_PARAM_FACTOR));
    for (uint local = 0u; local < SEISMIC_TUNE_ROWS; ++local) {
        const uint64_t row = uint64_t(gl_WorkGroupID.x) * SEISMIC_TUNE_ROWS + local;
        if (row >= SEISMIC_DIM_M) {
            return;
        }
        seismic_f32 x = seismic_f32(SCALE_ROW(row));
        for (uint64_t column = gl_LocalInvocationID.x; column < SEISMIC_DIM_N; column += gl_WorkGroupSize.x) {
            out_put(result, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
                x[uint(column * SEISMIC_X_STRIDE_1)].v * factor);
        }
    }
}
