// vision_patch_stem: the patch projection as one GEMM per frame over the
// projection library (K = C * P * P each): with two frames the first stores its
// F32 product; the last adds it, the bias and the bilinear position-table
// blend, publishing the F32 residual stream. Pixels enter the matrix units as
// f16 (a relative rounding of 2^-11).
#include "lib/vision/vision.h"

typedef ELEMENT_OF(SEISMIC_FRAME_WEIGHT) weight_0_element;
typedef ELEMENT_OF(SEISMIC_NEXT_FRAME_WEIGHT) weight_1_element;
typedef packets::Dense<weight_0_element> weight_0_packet;
typedef packets::Dense<weight_1_element> weight_1_packet;
typedef ELEMENT_OF(SEISMIC_BIAS) bias_element;
typedef ELEMENT_OF(SEISMIC_TABLE) table_element;

constant constexpr uint FRAMES = 1 + SEISMIC_DIM_S;
constant constexpr uint AREA = uint(SEISMIC_DIM_P * SEISMIC_DIM_P);
constant constexpr uint K = uint(SEISMIC_DIM_C) * AREA;

template <typename E>
inline float2 load2(device const uchar *base, ulong index) {
    device const typename E::storage *src = reinterpret_cast<device const typename E::storage *>(base) + index;
    const vec<typename E::storage, 2> raw =
        vec<typename E::storage, 2>(*reinterpret_cast<device const typename E::packed2 *>(src));
    return float2(E::load(raw.x), E::load(raw.y));
}

// Frame `frame` of the pixel rows [M, C, FRAMES, P, P] as the GEMM's A
// operand: x[m, k] with k = channel * P * P + (y * P + x), rounded to f16.
struct input_pixels {
    typedef element::F16 activation;
    device const float *pixels;
    ulong row_stride;
    uint frame;
    float4 load4(device const float *row_pixels, uint col, thread uint &channel, thread uint &within) const {
        if (col >= K)
            return float4(0.0f);
        if (within + 4u <= AREA) {
            const float4 vals = float4(*reinterpret_cast<device const packed_float4 *>(
                row_pixels + (ulong(channel) * FRAMES + frame) * AREA + within));
            within += 4u;
            if (within == AREA) {
                within = 0u;
                ++channel;
            }
            return vals;
        }
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 4u; ++i) {
            if (col + i < K) {
                v[i] = row_pixels[(ulong(channel) * FRAMES + frame) * AREA + within];
                if (++within == AREA) {
                    within = 0u;
                    ++channel;
                }
            }
        }
        return float4(v[0], v[1], v[2], v[3]);
    }
    uint4 words8(uint m, uint k0) const {
        if (k0 >= K)
            return uint4(0);
        uint channel = k0 / AREA;
        uint within = k0 % AREA;
        device const float *row_pixels = pixels + ulong(m) * row_stride;
        const float4 lo = load4(row_pixels, k0, channel, within);
        const float4 hi = load4(row_pixels, k0 + 4u, channel, within);
        return uint4(as_type<uint>(half2(lo.x, lo.y)), as_type<uint>(half2(lo.z, lo.w)),
            as_type<uint>(half2(hi.x, hi.y)), as_type<uint>(half2(hi.z, hi.w)));
    }
};

// The first frame's F32 product.
struct output_partial {
    device float *partial;
    uint columns;
    void store(uint m, uint n, float value) const { partial[ulong(m) * columns + n] = value; }
    void store2(uint m, uint n, float first, float second) const {
        *reinterpret_cast<device packed_float2 *>(partial + ulong(m) * columns + n) = packed_float2(first, second);
    }
};

// ((partial) + product (+ bias)) + (with NT = 1) the four table corners
// scaled by their coefficients, summed in order.
struct output_stem {
    element::Out<element::F32> y;
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
        if (SEISMIC_DIM_NT == 1) {
            const int4 idx = int4(*reinterpret_cast<device const packed_int4 *>(indices + ulong(m) * 4));
            const float4 coef = float4(*reinterpret_cast<device const packed_float4 *>(coefficients + ulong(m) * 4));
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < 4; ++i)
                blended += element::at<table_element>(table, ulong(idx[i]) * columns + n) * coef[i];
        }
        y.put(ulong(m) * stride + n, projected + blended);
    }
    void store2(uint m, uint n, float first, float second) const {
        float p0 = first, p1 = second;
        if (SEISMIC_DIM_S == 1) {
            const float2 p = float2(*reinterpret_cast<device const packed_float2 *>(partial + ulong(m) * columns + n));
            p0 = p.x + first;
            p1 = p.y + second;
        }
        if (SEISMIC_DIM_NB == 1) {
            const float2 b = load2<bias_element>(bias, n);
            p0 = p0 + b.x;
            p1 = p1 + b.y;
        }
        float b0 = 0.0f, b1 = 0.0f;
        if (SEISMIC_DIM_NT == 1) {
            const int4 idx = int4(*reinterpret_cast<device const packed_int4 *>(indices + ulong(m) * 4));
            const float4 coef = float4(*reinterpret_cast<device const packed_float4 *>(coefficients + ulong(m) * 4));
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < 4; ++i) {
                const float2 t = load2<table_element>(table, ulong(idx[i]) * columns + n);
                b0 += t.x * coef[i];
                b1 += t.y * coef[i];
            }
        }
        y.put2(ulong(m) * stride + n, p0 + b0, p1 + b1);
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
    SEISMIC_RESULT_0_PARAM(result),                                                              \
    device float *partial [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIAL)]],                            \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define STEM_PIXELS(frame)                                                                       \
    input_pixels in{pixels, SEISMIC_PIXELS_STRIDE_0, frame}

kernel void vision_patch_stem_first(STEM_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    STEM_PIXELS(0);
    const uint h = uint(SEISMIC_DIM_H);
    output_partial out{partial, h};
    auto w = vision::weight<weight_0_element>(frame_weight, SEISMIC_FRAME_WEIGHT_STRIDE_0, K);
    projection::gemm<weight_0_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), h, K,
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
    output_stem out{element::Out<element::F32>(result), SEISMIC_RESULT_0_STRIDE_0, partial, bias, table, indices, coefficients, h};
    if (SEISMIC_DIM_S == 1) {
        auto w = vision::weight<weight_1_element>(next_frame_weight, SEISMIC_NEXT_FRAME_WEIGHT_STRIDE_1, K);
        projection::gemm<weight_1_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), h, K,
            tile.y, tile.x, shared, sg, lane);
    } else {
        auto w = vision::weight<weight_0_element>(frame_weight, SEISMIC_FRAME_WEIGHT_STRIDE_0, K);
        projection::gemm<weight_0_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), h, K,
            tile.y, tile.x, shared, sg, lane);
    }
}
