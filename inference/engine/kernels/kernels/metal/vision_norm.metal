// vision_norm: one threadgroup per result row r = cell * G + member, which
// normalizes source row r (or order[r] when NO = 1) and publishes it to Y at
// member * H (side by side) or, with `interleave`, channel by channel at
// stride G. Body in lib/vision/vision.h.
#include "lib/vision/vision.h"

typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) output_element;
typedef ELEMENT_OF(SEISMIC_WEIGHT) weight_element;
typedef ELEMENT_OF(SEISMIC_BIAS) bias_element;

#define NORM_THREADS 256

kernel void vision_norm(device const float *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],
    device const uchar *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    device const int *order [[buffer(SEISMIC_BUFFER_ORDER)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[NORM_THREADS / 32];
    const uint g = uint(SEISMIC_DIM_G), h = uint(SEISMIC_DIM_H);
    const uint from = SEISMIC_DIM_NO == 1 ? uint(order[row]) : row;
    const uint cell = row / g, member = row % g;
    const bool interleave = int(SEISMIC_PARAM_INTERLEAVE) != 0;
    const ulong first = ulong(cell) * SEISMIC_RESULT_0_STRIDE_0 + (interleave ? member : ulong(member) * h);
    vision::norm<NORM_THREADS, output_element, weight_element, bias_element, SEISMIC_DIM_NW == 1,
        SEISMIC_DIM_NB == 1>(source + ulong(from) * h, result + first * output_element::bytes, interleave ? g : 1,
        weight, bias, h, int(SEISMIC_PARAM_CENTERED) != 0, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), partials,
        thread_index, sg, lane);
}
