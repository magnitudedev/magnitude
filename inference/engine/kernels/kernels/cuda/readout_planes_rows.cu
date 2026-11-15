// readout_planes_rows: the exact head logits of the `out_rows` rows from the
// progressive planes (`readout.seismic`): the `dense` projection's `planes`
// form (`lib/projection/operator.cuh`) onto the exact codes, by the routine
// the certified exact level shares (`lib/readout/progressive.cuh`), over the
// rows' final norm, which `stage` forms.
#include "lib/readout/progressive.cuh"
#include "lib/projection/operator.cuh"

using Pro = projection::Input<ELEMENT_OF(SEISMIC_NORM), projection::SelectedRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;

#define PROJECTION_M SEISMIC_DIM_O
#define PROJECTION_K SEISMIC_DIM_D
#define PROJECTION_N SEISMIC_DIM_V
#define PROJECTION_ROWS(g)                                                                                        \
    Pro {                                                                                                         \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,              \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), SEISMIC_DIM_D,     \
            projection::SelectedRows {                                                                            \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))                                   \
        }                                                                                                         \
    }
#define PROJECTION_PLANES_STAGED element::out_over_scratch<projection::Act>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED))
#define PROJECTION_PLANES_FORMATION progressive::Rows{}
#define PROJECTION_PLANES_VIEW                                                                                    \
    progressive::Exact {                                                                                          \
        progressive::Planes {                                                                                     \
            reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_TOP)), SEISMIC_TOP_STRIDE_0,             \
                reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_BIT3)), SEISMIC_BIT3_STRIDE_0,       \
                reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_REST)), SEISMIC_REST_STRIDE_0,       \
                SEISMIC_REST_STRIDE_1, SEISMIC_REST_STRIDE_2,                                                     \
                reinterpret_cast<const unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES)),                     \
                SEISMIC_SCALES_STRIDE_0                                                                           \
        }                                                                                                         \
    }
#define PROJECTION_EPILOGUE(g)                                                                                    \
    progressive::Logits {                                                                                         \
        SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0,                                                          \
            SEISMIC_RESULT_0_STRIDE_1, __uint_as_float((unsigned)SEISMIC_PARAM_SOFTCAP)                           \
    }

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_STAGE
extern "C" __global__ void readout_planes_rows_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_PLANES_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_PLANES
template <unsigned ROWS>
__global__ void readout_planes_rows_planes(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_PLANES(ROWS);
}
#endif
