// vision_patch_stem: the patch projection as one GEMM per frame over the
// projection library (K = C * P * P each): with two frames the first stores its
// F32 product; the last adds it, the bias and the bilinear position-table
// blend, publishing the F32 residual stream. Pixels enter the matrix units as
// f16 (a relative rounding of 2^-11).
#include "lib/vision/vision.h"

typedef vision::dense_packet<ELEMENT_KIND(SEISMIC_FRAME_WEIGHT)>::type weight_0_packet;
typedef vision::dense_packet<ELEMENT_KIND(SEISMIC_NEXT_FRAME_WEIGHT)>::type weight_1_packet;
typedef ELEMENT_OF(SEISMIC_FRAME_WEIGHT) weight_0_element;
typedef ELEMENT_OF(SEISMIC_NEXT_FRAME_WEIGHT) weight_1_element;
typedef ELEMENT_OF(SEISMIC_BIAS) bias_element;
typedef ELEMENT_OF(SEISMIC_TABLE) table_element;

constant constexpr uint FRAMES = 1 + SEISMIC_DIM_S;

// Frame `frame` of the pixel rows [M, C, FRAMES, P, P] as the GEMM's A
// operand: x[m, k] with k = channel * P * P + (y * P + x), rounded to f16.
struct input_pixels {
    typedef element::F16 activation;
    device const float *pixels;
    ulong row_stride;
    uint frame;
    uint area;
    uint k;
    uint4 words8(uint m, uint k0) const {
        float v[8];
        for (uint i = 0; i < 8; ++i) {
            const uint column = k0 + i;
            const uint channel = column / area, within = column % area;
            v[i] = column < k ? pixels[ulong(m) * row_stride + (ulong(channel) * FRAMES + frame) * area + within] : 0.0f;
        }
        return uint4(as_type<uint>(half2(v[0], v[1])), as_type<uint>(half2(v[2], v[3])),
            as_type<uint>(half2(v[4], v[5])), as_type<uint>(half2(v[6], v[7])));
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

// ((partial) + product (+ bias)) + the four table corners scaled by their
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
        float projected = value;
        if (SEISMIC_DIM_S == 1)
            projected = partial[ulong(m) * columns + n] + value;
        if (SEISMIC_DIM_NB == 1)
            projected = projected + element::at<bias_element>(bias, n);
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
    device const uchar *frame_weight [[buffer(SEISMIC_BUFFER_FRAME_WEIGHT)]],                    \
    device const uchar *next_frame_weight [[buffer(SEISMIC_BUFFER_NEXT_FRAME_WEIGHT)]],          \
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
    input_pixels in{pixels, SEISMIC_PIXELS_STRIDE_0, frame, area, k}

kernel void vision_patch_stem_first(STEM_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    STEM_PIXELS(0);
    const uint h = uint(SEISMIC_DIM_H);
    output_partial out{partial, h};
    auto w = vision::weight<weight_0_packet>(frame_weight, SEISMIC_FRAME_WEIGHT_STRIDE_0, weight_0_element::bytes, k);
    projection::gemm<weight_0_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), h, k,
        tile.y, tile.x, shared, sg, lane);
}

// The last frame (the only one without a next frame).
kernel void vision_patch_stem_last(STEM_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    STEM_PIXELS(SEISMIC_DIM_S);
    const uint h = uint(SEISMIC_DIM_H);
    output_stem out{result, SEISMIC_RESULT_0_STRIDE_0, partial, bias, table, indices, coefficients, h};
    if (SEISMIC_DIM_S == 1) {
        auto w = vision::weight<weight_1_packet>(next_frame_weight, SEISMIC_NEXT_FRAME_WEIGHT_STRIDE_1,
            weight_1_element::bytes, k);
        projection::gemm<weight_1_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), h, k,
            tile.y, tile.x, shared, sg, lane);
    } else {
        auto w = vision::weight<weight_0_packet>(frame_weight, SEISMIC_FRAME_WEIGHT_STRIDE_0,
            weight_0_element::bytes, k);
        projection::gemm<weight_0_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), h, k,
            tile.y, tile.x, shared, sg, lane);
    }
}
