// readout_top_rows: level one of a certified selection (`readout.seismic`):
// the `dense` projection's `planes` form (`lib/projection/operator.cuh`) onto
// the 4-bit view, with the bound epilogue (`lib/readout/progressive.cuh`).
// `stage` forms the rows' final norm (`features`) with their lengths and
// per-group sums; `planes` projects every vocabulary row onto the view and
// publishes each row's threshold.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/readout/progressive.cuh"
#include "lib/projection/operator.cuh"

using Pro = projection::Input<ELEMENT_OF(SEISMIC_NORM), projection::SelectedRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;

#define F32_AT(buffer) reinterpret_cast<float *>(SEISMIC_PTR(buffer))
#define F32_IN(buffer) reinterpret_cast<const float *>(SEISMIC_PTR(buffer))

#define PROJECTION_M SEISMIC_DIM_O
#define PROJECTION_K SEISMIC_DIM_D
#define PROJECTION_N SEISMIC_DIM_V
#define PROJECTION_ROWS(g)                                                                                        \
    Pro {                                                                                                         \
        F32_IN(SEISMIC_BUFFER_HIDDEN), SEISMIC_HIDDEN_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_NORM),                  \
            __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), SEISMIC_DIM_D,                                      \
            projection::SelectedRows {                                                                            \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))                                   \
        }                                                                                                         \
    }
// The rows are the level's `features`, which the later levels read.
#define PROJECTION_PLANES_STAGED SEISMIC_RESULT_2_OUT
#define PROJECTION_PLANES_FORMATION                                                                               \
    progressive::Measured {                                                                                       \
        SEISMIC_RESULT_3_OUT, SEISMIC_RESULT_3_STRIDE_0, F32_AT(SEISMIC_BUFFER_SCRATCH_GROUPS),                   \
            SEISMIC_DIM_D                                                                                         \
    }
#define PROJECTION_PLANES_VIEW                                                                                    \
    progressive::Top {                                                                                            \
        progressive::Planes{reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_TOP)),                   \
                            SEISMIC_TOP_STRIDE_0,                                                                 \
                            nullptr,                                                                              \
                            0,                                                                                    \
                            nullptr,                                                                              \
                            0,                                                                                    \
                            0,                                                                                    \
                            0,                                                                                    \
                            reinterpret_cast<const unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES)),         \
                            SEISMIC_SCALES_STRIDE_0},                                                             \
            F32_IN(SEISMIC_BUFFER_SCRATCH_GROUPS), sums                                                           \
    }
// Radius column 0: the 4-bit view's.
#define PROJECTION_EPILOGUE(g)                                                                                    \
    progressive::Bounds {                                                                                         \
        SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, F32_IN(SEISMIC_BUFFER_RADIUS), \
            SEISMIC_RADIUS_STRIDE_0, reinterpret_cast<const float *>(SEISMIC_RESULT_3_OUT.stored()),              \
            SEISMIC_RESULT_3_STRIDE_0,                                                                            \
            progressive::Selection{reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_DRAWS)),          \
                                   SEISMIC_DRAWS_STRIDE_0,                                                        \
                                   SEISMIC_DRAWS_STRIDE_1,                                                        \
                                   F32_IN(SEISMIC_BUFFER_TEMPERATURE),                                            \
                                   SEISMIC_TEMPERATURE_STRIDE_0,                                                  \
                                   reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_MASK)),          \
                                   SEISMIC_MASK_STRIDE_0,                                                         \
                                   SEISMIC_MASK_STRIDE_1,                                                         \
                                   reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONSTRAINED)),        \
                                   SEISMIC_CONSTRAINED_STRIDE_0,                                                  \
                                   __uint_as_float((unsigned)SEISMIC_PARAM_SOFTCAP)},                             \
            SEISMIC_RESULT_1_OUT, SEISMIC_RESULT_1_STRIDE_0,                                                      \
            reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_BOUNDS)), lowest                      \
    }

#ifdef SEISMIC_FORMING_READOUT_TOP_ROWS_STAGE
extern "C" __global__ void readout_top_rows_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_PLANES_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_READOUT_TOP_ROWS_PLANES
template <unsigned ROWS>
__global__ void readout_top_rows_planes(SEISMIC_KERNEL_PARAMS) {
    // The view's staged group sums and the epilogue's keys.
    __shared__ float sums[PROGRESSIVE_ROWS * 32];
    __shared__ unsigned lowest[PROGRESSIVE_ROWS];
    PROJECTION_PLANES(ROWS);
}
#endif
