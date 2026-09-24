// qwen_vision_stem: the patch projection as two GEMMs over the projection
// library, one per temporal frame (K = C * P * P each): the first stores its
// F32 product, the second adds it, the bias and the bilinear position-table
// blend, publishing the F32 residual stream. Pixels enter the matrix units as
// f16 (a relative rounding of 2^-11).
#include "common/vision.h"

typedef vision::dense_packet<ELEMENT_KIND(SEISMIC_TEMPORAL_WEIGHT_0)>::type weight_0_packet;
typedef vision::dense_packet<ELEMENT_KIND(SEISMIC_TEMPORAL_WEIGHT_1)>::type weight_1_packet;
typedef ELEMENT_OF(SEISMIC_TEMPORAL_WEIGHT_0) weight_0_element;
typedef ELEMENT_OF(SEISMIC_TEMPORAL_WEIGHT_1) weight_1_element;
typedef ELEMENT_OF(SEISMIC_BIAS) bias_element;
typedef ELEMENT_OF(SEISMIC_TABLE) table_element;

// Frame `frame` of the pixel rows [M, C, 2, P, P] as the GEMM's A operand:
// x[m, k] with k = channel * P * P + (y * P + x), rounded to f16.
struct input_pixels {
    typedef element::F16 activation;
    device const float *pixels;
    ulong row_stride;
    uint frame;
    uint area;
    uint4 words8(uint m, uint k) const {
        const uint channel = k / area, within = k % area;
        device const float *at = pixels + ulong(m) * row_stride + (ulong(channel) * 2 + frame) * area + within;
        const float4 a = *reinterpret_cast<device const float4 *>(at);
        const float4 b = *reinterpret_cast<device const float4 *>(at + 4);
        return uint4(as_type<uint>(half2(a.x, a.y)), as_type<uint>(half2(a.z, a.w)),
            as_type<uint>(half2(b.x, b.y)), as_type<uint>(half2(b.z, b.w)));
    }
};

// The first frame's F32 product.
struct output_partial {
    device float *partial;
    uint columns;
    void store(uint m, uint n, float value) const { partial[ulong(m) * columns + n] = value; }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// (partial + product + bias) + the four table corners scaled by their
// coefficients, summed in order.
struct output_stem {
    device float *y;
    ulong stride;
    device const float *partial;
    device const uchar *bias;
    device const uchar *table;
    device const int *indices;
    device const float *coefficients;
    uint columns;
    void store(uint m, uint n, float value) const {
        const float projected = partial[ulong(m) * columns + n] + value + element::at<bias_element>(bias, n);
        float blended = 0.0f;
        for (uint i = 0; i < 4; ++i)
            blended += element::at<table_element>(table, ulong(indices[m * 4 + i]) * columns + n)
                * coefficients[m * 4 + i];
        y[ulong(m) * stride + n] = projected + blended;
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

#define STEM_ARGUMENTS                                                                           \
    device const float *pixels [[buffer(SEISMIC_BUFFER_PIXELS)]],                                \
    device const uchar *temporal_weight_0 [[buffer(SEISMIC_BUFFER_TEMPORAL_WEIGHT_0)]],          \
    device const uchar *temporal_weight_1 [[buffer(SEISMIC_BUFFER_TEMPORAL_WEIGHT_1)]],          \
    device const uchar *bias [[buffer(SEISMIC_BUFFER_BIAS)]],                                    \
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],                                  \
    device const int *indices [[buffer(SEISMIC_BUFFER_INDICES)]],                                \
    device const float *coefficients [[buffer(SEISMIC_BUFFER_COEFFICIENTS)]],                    \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                                    \
    device float *partial [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIAL)]],                            \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define STEM_PIXELS(frame)                                                                       \
    const uint area = uint(SEISMIC_DIM_P * SEISMIC_DIM_P);                                       \
    const uint k = uint(SEISMIC_DIM_C) * area;                                                   \
    input_pixels in{pixels, SEISMIC_PIXELS_STRIDE_0, frame, area}

kernel void qwen_vision_stem_first(STEM_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    STEM_PIXELS(0);
    const uint h = uint(SEISMIC_DIM_H);
    output_partial out{partial, h};
    auto w = vision::weight<weight_0_packet>(temporal_weight_0, SEISMIC_TEMPORAL_WEIGHT_0_STRIDE_0,
        weight_0_element::bytes, k);
    projection::gemm<weight_0_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), h, k,
        tile.y, tile.x, shared, sg, lane);
}

kernel void qwen_vision_stem_second(STEM_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    STEM_PIXELS(1);
    const uint h = uint(SEISMIC_DIM_H);
    output_stem out{result, SEISMIC_RESULT_0_STRIDE_0, partial, bias, table, indices, coefficients, h};
    auto w = vision::weight<weight_1_packet>(temporal_weight_1, SEISMIC_TEMPORAL_WEIGHT_1_STRIDE_0,
        weight_1_element::bytes, k);
    projection::gemm<weight_1_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), h, k,
        tile.y, tile.x, shared, sg, lane);
}
