// readout_top_rows: level one of a certified selection (`readout.seismic`).
// `form` (one block per row) writes the rows' final norm (`features`), their
// lengths and per-group sums; `scan` projects every vocabulary row onto the
// 4-bit view and publishes each row's threshold (`lib/readout/progressive.cuh`).
#include "lib/readout/progressive.cuh"

using Pro = projection::Input<ELEMENT_OF(SEISMIC_NORM), projection::SelectedRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;

#define F32_AT(buffer) reinterpret_cast<float *>(SEISMIC_PTR(buffer))
#define F32_IN(buffer) reinterpret_cast<const float *>(SEISMIC_PTR(buffer))

#ifdef SEISMIC_FORMING_READOUT_TOP_ROWS_FORM
extern "C" __global__ void readout_top_rows_form(SEISMIC_KERNEL_PARAMS) {
    const Pro pro{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,
                  SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), SEISMIC_DIM_D,
                  projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))}};
    const progressive::F32Out lengths = SEISMIC_RESULT_3_OUT;
    progressive::form<true>(pro, SEISMIC_DIM_D, SEISMIC_RESULT_2_OUT, SEISMIC_RESULT_2_STRIDE_0, &lengths,
                            SEISMIC_RESULT_3_STRIDE_0,
                            F32_AT(SEISMIC_BUFFER_SCRATCH_GROUPS));
}
#endif

#ifdef SEISMIC_FORMING_READOUT_TOP_ROWS_SCAN
template <unsigned ROWS>
__global__ void readout_top_rows_scan(SEISMIC_KERNEL_PARAMS) {
    const progressive::Planes planes{reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_TOP)),
                                     SEISMIC_TOP_STRIDE_0, nullptr, 0, nullptr, 0, 0, 0,
                                     reinterpret_cast<const unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES)),
                                     SEISMIC_SCALES_STRIDE_0};
    const progressive::Selection selection{
        reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_DRAWS)), SEISMIC_DRAWS_STRIDE_0,
        SEISMIC_DRAWS_STRIDE_1, F32_IN(SEISMIC_BUFFER_TEMPERATURE), SEISMIC_TEMPERATURE_STRIDE_0,
        reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_MASK)), SEISMIC_MASK_STRIDE_0,
        SEISMIC_MASK_STRIDE_1, reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONSTRAINED)),
        SEISMIC_CONSTRAINED_STRIDE_0, __uint_as_float((unsigned)SEISMIC_PARAM_SOFTCAP)};
    __shared__ uint4 staged[SEISMIC_CONST_VERIFY_ROWS * PROGRESSIVE_CHUNK / 8];
    __shared__ float sums[SEISMIC_CONST_VERIFY_ROWS * 32];
    __shared__ unsigned lowest[SEISMIC_CONST_VERIFY_ROWS];
    const unsigned rows = SEISMIC_DIM_O;
    const progressive::F32Out threshold = SEISMIC_RESULT_1_OUT;
    // Radius column 0: the 4-bit view's.
#define TOP_SCAN(MAXM)                                                                                               \
    progressive::scan<MAXM, ROWS, false>(                                                                            \
        SEISMIC_RESULT_2_OUT.stored(), SEISMIC_RESULT_2_STRIDE_0, F32_AT(SEISMIC_BUFFER_SCRATCH_GROUPS),             \
        planes, F32_IN(SEISMIC_BUFFER_RADIUS), SEISMIC_RADIUS_STRIDE_0,                                              \
        reinterpret_cast<const float *>(SEISMIC_RESULT_3_OUT.stored()), SEISMIC_RESULT_3_STRIDE_0, selection,        \
        SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, &threshold,                      \
        SEISMIC_RESULT_1_STRIDE_0,                                                                                   \
        reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_BOUNDS)), rows, SEISMIC_DIM_V, SEISMIC_DIM_D, staged, sums, lowest)
    if (rows <= 1)
        TOP_SCAN(1);
    else if (rows <= 2)
        TOP_SCAN(2);
    else if (rows <= 4)
        TOP_SCAN(4);
    else
        TOP_SCAN(8);
}
#endif
