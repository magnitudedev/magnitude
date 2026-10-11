// readout_exact_rows: level three of a certified selection (`readout.seismic`).
// One warp per 32 vocabulary rows keeps the rows level two leaves (`fine`,
// `floor`) and projects them exactly over every plane, with the full pass's
// own routine (`lib/readout/progressive.cuh`); −inf elsewhere.
#include "lib/readout/progressive.cuh"

#define F32_AT(buffer) reinterpret_cast<float *>(SEISMIC_PTR(buffer))
#define F32_IN(buffer) reinterpret_cast<const float *>(SEISMIC_PTR(buffer))

#ifdef SEISMIC_FORMING_READOUT_EXACT_ROWS_GATHER
extern "C" __global__ void readout_exact_rows_gather(SEISMIC_KERNEL_PARAMS) {
    const progressive::Planes planes{reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_TOP)),
                                     SEISMIC_TOP_STRIDE_0,
                                     reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_BIT3)),
                                     SEISMIC_BIT3_STRIDE_0,
                                     reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_REST)),
                                     SEISMIC_REST_STRIDE_0, SEISMIC_REST_STRIDE_1, SEISMIC_REST_STRIDE_2,
                                     reinterpret_cast<const unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES)),
                                     SEISMIC_SCALES_STRIDE_0};
    const progressive::Selection selection{
        reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_DRAWS)), SEISMIC_DRAWS_STRIDE_0,
        SEISMIC_DRAWS_STRIDE_1, F32_IN(SEISMIC_BUFFER_TEMPERATURE), SEISMIC_TEMPERATURE_STRIDE_0,
        reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_MASK)), SEISMIC_MASK_STRIDE_0,
        SEISMIC_MASK_STRIDE_1, reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONSTRAINED)),
        SEISMIC_CONSTRAINED_STRIDE_0};
    // Radius column 1: the 5-bit view's.
    progressive::gather<false>(SEISMIC_PTR(SEISMIC_BUFFER_FEATURES), SEISMIC_FEATURES_STRIDE_0, planes,
                               F32_IN(SEISMIC_BUFFER_RADIUS) + SEISMIC_RADIUS_STRIDE_1, nullptr,
                               SEISMIC_RADIUS_STRIDE_0, F32_IN(SEISMIC_BUFFER_FINE), SEISMIC_FINE_STRIDE_0,
                               SEISMIC_FINE_STRIDE_1, F32_IN(SEISMIC_BUFFER_FLOOR), SEISMIC_FLOOR_STRIDE_0,
                               F32_IN(SEISMIC_BUFFER_LENGTH), SEISMIC_LENGTH_STRIDE_0, selection,
                               F32_AT(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
                               nullptr, 0, nullptr, SEISMIC_DIM_O, SEISMIC_DIM_V, SEISMIC_DIM_D);
}
#endif
