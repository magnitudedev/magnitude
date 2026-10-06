// readout_refine_rows: level two of a certified selection (`readout.seismic`).
// One warp per 32 vocabulary rows keeps the rows level one leaves (`coarse`,
// `floor`), adds bit 3 to their logits (the 5-bit view) and publishes the
// raised threshold (`lib/readout/progressive.cuh`).
#include "lib/readout/progressive.cuh"

#define F32_AT(buffer) reinterpret_cast<float *>(SEISMIC_PTR(buffer))
#define F32_IN(buffer) reinterpret_cast<const float *>(SEISMIC_PTR(buffer))

#ifdef SEISMIC_FORMING_READOUT_REFINE_ROWS_GATHER
extern "C" __global__ void readout_refine_rows_gather(SEISMIC_KERNEL_PARAMS) {
    const progressive::Planes planes{nullptr, 0, reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_BIT3)),
                                     SEISMIC_BIT3_STRIDE_0, nullptr, 0, 0, 0,
                                     reinterpret_cast<const unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES)),
                                     SEISMIC_SCALES_STRIDE_0};
    const progressive::Selection selection{
        reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_DRAWS)), SEISMIC_DRAWS_STRIDE_0,
        SEISMIC_DRAWS_STRIDE_1, F32_IN(SEISMIC_BUFFER_TEMPERATURE), SEISMIC_TEMPERATURE_STRIDE_0,
        reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_MASK)), SEISMIC_MASK_STRIDE_0,
        SEISMIC_MASK_STRIDE_1, reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONSTRAINED)),
        SEISMIC_CONSTRAINED_STRIDE_0};
    const float *radius = F32_IN(SEISMIC_BUFFER_RADIUS);
    // Radius columns 0 and 1: the 4- and 5-bit views'.
    progressive::gather<true>(SEISMIC_PTR(SEISMIC_BUFFER_FEATURES), SEISMIC_FEATURES_STRIDE_0, planes, radius,
                              radius + SEISMIC_RADIUS_STRIDE_1, SEISMIC_RADIUS_STRIDE_0, F32_IN(SEISMIC_BUFFER_COARSE),
                              SEISMIC_COARSE_STRIDE_0, SEISMIC_COARSE_STRIDE_1, F32_IN(SEISMIC_BUFFER_FLOOR),
                              SEISMIC_FLOOR_STRIDE_0, F32_IN(SEISMIC_BUFFER_LENGTH), SEISMIC_LENGTH_STRIDE_0,
                              selection, F32_AT(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0,
                              SEISMIC_RESULT_0_STRIDE_1, F32_AT(SEISMIC_RESULT_1_BUFFER), SEISMIC_RESULT_1_STRIDE_0,
                              reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_BOUNDS)),
                              SEISMIC_DIM_O, SEISMIC_DIM_V, SEISMIC_DIM_D);
}
#endif
