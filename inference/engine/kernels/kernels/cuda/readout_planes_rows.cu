// readout_planes_rows: the exact head logits of the `out_rows` rows from the
// progressive planes (`readout.seismic`). `form` (one block per row) writes the
// final norm to scratch; `scan` projects every vocabulary row exactly, with
// the routine the certified exact level shares (`lib/readout/progressive.cuh`).
#include "lib/readout/progressive.cuh"

using Pro = projection::Rms<ELEMENT_OF(SEISMIC_NORM), projection::SelectedRows>;

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_FORM
extern "C" __global__ void readout_planes_rows_form(SEISMIC_KERNEL_PARAMS) {
    const Pro pro{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,
                  SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), SEISMIC_DIM_D,
                  projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))}};
    progressive::form<false>(pro, SEISMIC_DIM_D, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_FEATURES), SEISMIC_DIM_D, nullptr,
                             0, nullptr);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_SCAN
template <unsigned ROWS>
__global__ void readout_planes_rows_scan(SEISMIC_KERNEL_PARAMS) {
    const progressive::Planes planes{reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_TOP)),
                                     SEISMIC_TOP_STRIDE_0,
                                     reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_BIT3)),
                                     SEISMIC_BIT3_STRIDE_0,
                                     reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_REST)),
                                     SEISMIC_REST_STRIDE_0, SEISMIC_REST_STRIDE_1, SEISMIC_REST_STRIDE_2,
                                     reinterpret_cast<const unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES)),
                                     SEISMIC_SCALES_STRIDE_0};
    const progressive::Selection selection{};
    __shared__ uint4 staged[PROGRESSIVE_ROWS * PROGRESSIVE_CHUNK / 8];
    // Rows 8b..8b + 7 of row block b (the grid's second axis).
    const unsigned first = blockIdx.y * PROGRESSIVE_ROWS;
    const unsigned rows = min(PROGRESSIVE_ROWS, (unsigned)SEISMIC_DIM_O - first);
    const projection::u8 *block = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_FEATURES) + (unsigned long long)first *
                                  SEISMIC_DIM_D * projection::Act::bytes;
    float *out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + first * SEISMIC_RESULT_0_STRIDE_0;
#define PLANES_SCAN(MAXM)                                                                                            \
    progressive::scan<MAXM, ROWS, true>(block, SEISMIC_DIM_D, nullptr, planes, nullptr, 0, nullptr, 0, selection,    \
                                        out, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, nullptr, 0,       \
                                        nullptr, rows, SEISMIC_DIM_V, SEISMIC_DIM_D, staged, nullptr, nullptr)
    if (rows <= 1)
        PLANES_SCAN(1);
    else if (rows <= 2)
        PLANES_SCAN(2);
    else if (rows <= 4)
        PLANES_SCAN(4);
    else
        PLANES_SCAN(8);
}
#endif
