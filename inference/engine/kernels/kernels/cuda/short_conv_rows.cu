// short_conv_rows (contract and portable body in short_conv.seismic): the
// gated causal depthwise convolution of one short-convolution layer. A thread
// owns 4 contiguous channels across a 16-row tile (item y < ceil_div(M, 16))
// or publishes 4 contiguous channels of slot b's successor window (item y =
// ceil_div(M, 16) + b); the y grid visits items with a stride (capped at
// 65535). Marching across contiguous rows of a slot keeps the filter weights
// and rolling C - 1 tap history in registers for C <= 4, while preserving the
// exact tap-ascending F32 FMA chain times the gate. The successor window is no
// slot's source, so the publishing threads run beside the row threads.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"
#include "lib/recurrent/versions.cuh"

namespace short_conv {

typedef versions::u64 u64;

constexpr u64 ROW_TILE = 16;

struct Operands {
    const float *projection;
    const float *convolution;
    element::SlabOut<element::F32> window;
    element::Out<activation::Act> gated;
    versions::Slots slots;
};

__device__ __forceinline__ float4 load_projection4(const float *projection, u64 row, u64 col, u64 lanes,
                                                   const seismic_words_t &seismic_words_value) {
    const u64 base = row * SEISMIC_PROJECTION_STRIDE_0 + col * SEISMIC_PROJECTION_STRIDE_1;
    if (lanes == 4 && SEISMIC_PROJECTION_STRIDE_1 == 1 &&
        ((reinterpret_cast<u64>(projection + base) & 15u) == 0)) {
        return *reinterpret_cast<const float4 *>(projection + base);
    }
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (u64 k = 0; k < 4; ++k) {
        if (k < lanes)
            v[k] = projection[base + k * SEISMIC_PROJECTION_STRIDE_1];
    }
    return make_float4(v[0], v[1], v[2], v[3]);
}

__device__ __forceinline__ float4 load_window4(const element::Stored<element::F32> &source, u64 tap_row,
                                               u64 channel, u64 lanes,
                                               const seismic_words_t &seismic_words_value) {
    const u64 base = tap_row * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2;
    if (lanes == 4 && SEISMIC_WINDOW_STRIDE_2 == 1 &&
        ((reinterpret_cast<u64>(source.bytes(base)) & 15u) == 0)) {
        return *reinterpret_cast<const float4 *>(source.bytes(base));
    }
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (u64 k = 0; k < 4; ++k) {
        if (k < lanes)
            v[k] = source.at(base + k * SEISMIC_WINDOW_STRIDE_2);
    }
    return make_float4(v[0], v[1], v[2], v[3]);
}

__device__ __forceinline__ void store_window4(const element::Out<element::F32> &target, u64 tap_row, u64 channel,
                                              u64 lanes, float4 val,
                                              const seismic_words_t &seismic_words_value) {
    const u64 base = tap_row * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2;
    const float vals[4] = {val.x, val.y, val.z, val.w};
    if (lanes == 4 && SEISMIC_WINDOW_STRIDE_2 == 1 &&
        ((reinterpret_cast<u64>(target.stored(base)) & 15u) == 0)) {
        target.put_span<4>(base, vals);
        return;
    }
#pragma unroll
    for (u64 k = 0; k < 4; ++k) {
        if (k < lanes)
            target.put(base + k * SEISMIC_WINDOW_STRIDE_2, vals[k]);
    }
}

__device__ __forceinline__ void store_gated4(const element::Out<activation::Act> &gated, u64 row, u64 channel,
                                             u64 lanes, float4 val,
                                             const seismic_words_t &seismic_words_value) {
    const u64 base = row * SEISMIC_RESULT_0_STRIDE_0 + channel * SEISMIC_RESULT_0_STRIDE_1;
    const float vals[4] = {val.x, val.y, val.z, val.w};
    if (lanes == 4 && SEISMIC_RESULT_0_STRIDE_1 == 1) {
        constexpr u64 align_mask = static_cast<u64>(activation::Act::bytes) * 4u - 1u;
        if ((reinterpret_cast<u64>(gated.stored(base)) & align_mask) == 0) {
            gated.put_span<4>(base, vals);
            return;
        }
    }
#pragma unroll
    for (u64 k = 0; k < 4; ++k) {
        if (k < lanes)
            gated.put(base + k * SEISMIC_RESULT_0_STRIDE_1, vals[k]);
    }
}

// Publishes channels `channel..channel+lanes` of slot `index`'s successor
// window: the source version's rows move as stored, the projection's rows are
// stored.
__device__ __forceinline__ void publish(const Operands &in, u64 index, u64 channel, u64 lanes,
                                        const seismic_words_t &seismic_words_value) {
    const int taps = static_cast<int>(SEISMIC_DIM_C) - 1;
    const u64 bank_elements = SEISMIC_WINDOW_STRIDE_0;
    const versions::Slot slot = versions::slot_of(in.slots, index, seismic_words_value);
    if (slot.lo == slot.hi) return;
    const element::Stored<element::F32> source =
        versions::bank_view(in.window, slot.source, bank_elements, seismic_words_value);
    const element::Out<element::F32> target =
        versions::bank(in.window, slot.target, bank_elements, seismic_words_value);
    const int published = taps + versions::tape_rows(slot, static_cast<int>(SEISMIC_DIM_T));
    for (int tap = 0; tap < published; ++tap) {
        const int position = slot.stop + tap - taps;
        if (position < 0) {
            const u64 to = static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2;
            const u64 from = static_cast<u64>(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1 +
                             channel * SEISMIC_WINDOW_STRIDE_2;
            if (lanes == 4 && SEISMIC_WINDOW_STRIDE_2 == 1) {
                target.move_stored(to, source.offset(from), 4);
            } else {
#pragma unroll
                for (u64 k = 0; k < 4; ++k) {
                    if (k < lanes)
                        target.move_stored(to + k * SEISMIC_WINDOW_STRIDE_2,
                                           source.offset(from + k * SEISMIC_WINDOW_STRIDE_2), 1);
                }
            }
        } else {
            const float4 val = load_projection4(in.projection, static_cast<u64>(slot.lo + position), channel,
                                                lanes, seismic_words_value);
            store_window4(target, static_cast<u64>(tap), channel, lanes, val, seismic_words_value);
        }
    }
}

// Register-tiled gated convolution for small compile-time tap counts C_STATIC
// (1..4): loads convolution weights once per tile and maintains the rolling
// C_STATIC - 1 tap history in registers across each contiguous slot run.
template <int C_STATIC>
__device__ __forceinline__ void convolve_tile_static(const Operands &in, int tile_lo, int tile_hi,
                                                     u64 slot_index, u64 channel, u64 lanes,
                                                     const seismic_words_t &seismic_words_value) {
    constexpr int TAPS = C_STATIC - 1;
    const u64 bank_elements = SEISMIC_WINDOW_STRIDE_0;

    float w[4][C_STATIC];
#pragma unroll
    for (u64 k = 0; k < 4; ++k) {
        if (k < lanes) {
            const u64 ch_off = (channel + k) * SEISMIC_CONVOLUTION_STRIDE_0;
#pragma unroll
            for (int tap = 0; tap < C_STATIC; ++tap)
                w[k][tap] = in.convolution[ch_off + static_cast<u64>(tap) * SEISMIC_CONVOLUTION_STRIDE_1];
        } else {
#pragma unroll
            for (int tap = 0; tap < C_STATIC; ++tap)
                w[k][tap] = 0.0f;
        }
    }

    int row = tile_lo;
    while (row < tile_hi) {
        if (slot_index == SEISMIC_DIM_B) {
            const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (; row < tile_hi; ++row)
                store_gated4(in.gated, static_cast<u64>(row), channel, lanes, zero, seismic_words_value);
            return;
        }
        const versions::Slot slot = versions::slot_of(in.slots, slot_index, seismic_words_value);
        const element::Stored<element::F32> source =
            versions::bank_view(in.window, slot.source, bank_elements, seismic_words_value);
        const int run_end = tile_hi < slot.hi ? tile_hi : slot.hi;

        float4 ring[TAPS > 0 ? TAPS : 1];
        if constexpr (TAPS > 0) {
            const int local0 = row - slot.lo;
#pragma unroll
            for (int t = 0; t < TAPS; ++t) {
                const int pos = local0 + t - TAPS;
                ring[t] = pos < 0
                              ? load_window4(source, static_cast<u64>(slot.taped + TAPS + pos), channel, lanes,
                                             seismic_words_value)
                              : load_projection4(in.projection, static_cast<u64>(slot.lo + pos), channel, lanes,
                                                 seismic_words_value);
            }
        }

        for (int r = row; r < run_end; ++r) {
            const float4 cur_u =
                load_projection4(in.projection, static_cast<u64>(r), channel, lanes, seismic_words_value);
            const float4 gate = load_projection4(in.projection, static_cast<u64>(r), SEISMIC_DIM_CH + channel,
                                                 lanes, seismic_words_value);

            float sx = 0.0f, sy = 0.0f, sz = 0.0f, sw = 0.0f;
            if constexpr (TAPS > 0) {
#pragma unroll
                for (int t = 0; t < TAPS; ++t) {
                    sx = seismic_fma_rn(w[0][t], ring[t].x, sx);
                    sy = seismic_fma_rn(w[1][t], ring[t].y, sy);
                    sz = seismic_fma_rn(w[2][t], ring[t].z, sz);
                    sw = seismic_fma_rn(w[3][t], ring[t].w, sw);
                }
            }
            sx = seismic_fma_rn(w[0][TAPS], cur_u.x, sx);
            sy = seismic_fma_rn(w[1][TAPS], cur_u.y, sy);
            sz = seismic_fma_rn(w[2][TAPS], cur_u.z, sz);
            sw = seismic_fma_rn(w[3][TAPS], cur_u.w, sw);

            const float4 out_val =
                make_float4(__fmul_rn(gate.x, sx), __fmul_rn(gate.y, sy), __fmul_rn(gate.z, sz),
                            __fmul_rn(gate.w, sw));
            store_gated4(in.gated, static_cast<u64>(r), channel, lanes, out_val, seismic_words_value);

            if constexpr (TAPS > 0) {
                if constexpr (TAPS > 1) {
#pragma unroll
                    for (int t = 0; t + 1 < TAPS; ++t)
                        ring[t] = ring[t + 1];
                }
                ring[TAPS > 0 ? TAPS - 1 : 0] = cur_u;
            }
        }

        row = run_end;
        if (row < tile_hi)
            slot_index = versions::slot_index_of_row(in.slots, row, seismic_words_value);
    }
}

// General 4-channel fallback for arbitrary C > 4.
__device__ __forceinline__ void convolve_tile_general(const Operands &in, int tile_lo, int tile_hi,
                                                      u64 slot_index, u64 channel, u64 lanes,
                                                      const seismic_words_t &seismic_words_value) {
    const int taps = static_cast<int>(SEISMIC_DIM_C) - 1;
    const u64 bank_elements = SEISMIC_WINDOW_STRIDE_0;
    int row = tile_lo;
    while (row < tile_hi) {
        if (slot_index == SEISMIC_DIM_B) {
            const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (; row < tile_hi; ++row)
                store_gated4(in.gated, static_cast<u64>(row), channel, lanes, zero, seismic_words_value);
            return;
        }
        const versions::Slot slot = versions::slot_of(in.slots, slot_index, seismic_words_value);
        const element::Stored<element::F32> source =
            versions::bank_view(in.window, slot.source, bank_elements, seismic_words_value);
        const int run_end = tile_hi < slot.hi ? tile_hi : slot.hi;
        for (int r = row; r < run_end; ++r) {
            const int local = r - slot.lo;
            float sx = 0.0f, sy = 0.0f, sz = 0.0f, sw = 0.0f;
            for (int tap = 0; tap <= taps; ++tap) {
                const int position = local + tap - taps;
                const float4 value =
                    position < 0
                        ? load_window4(source, static_cast<u64>(slot.taped + taps + position), channel, lanes,
                                       seismic_words_value)
                        : load_projection4(in.projection, static_cast<u64>(slot.lo + position), channel, lanes,
                                           seismic_words_value);
                const u64 tap_off = static_cast<u64>(tap) * SEISMIC_CONVOLUTION_STRIDE_1;
                sx = seismic_fma_rn(in.convolution[channel * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off], value.x, sx);
                if (lanes > 1)
                    sy = seismic_fma_rn(in.convolution[(channel + 1) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off],
                                        value.y, sy);
                if (lanes > 2)
                    sz = seismic_fma_rn(in.convolution[(channel + 2) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off],
                                        value.z, sz);
                if (lanes > 3)
                    sw = seismic_fma_rn(in.convolution[(channel + 3) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off],
                                        value.w, sw);
            }
            const float4 gate = load_projection4(in.projection, static_cast<u64>(r), SEISMIC_DIM_CH + channel,
                                                 lanes, seismic_words_value);
            const float4 out_val =
                make_float4(__fmul_rn(gate.x, sx), __fmul_rn(gate.y, sy), __fmul_rn(gate.z, sz),
                            __fmul_rn(gate.w, sw));
            store_gated4(in.gated, static_cast<u64>(r), channel, lanes, out_val, seismic_words_value);
        }
        row = run_end;
        if (row < tile_hi)
            slot_index = versions::slot_index_of_row(in.slots, row, seismic_words_value);
    }
}

__device__ __forceinline__ void convolve_tile(const Operands &in, u64 tile_index, u64 channel, u64 lanes,
                                              const seismic_words_t &seismic_words_value) {
    const int tile_lo = static_cast<int>(tile_index * ROW_TILE);
    const u64 end = tile_index * ROW_TILE + ROW_TILE;
    const int tile_hi = static_cast<int>(end < SEISMIC_DIM_M ? end : SEISMIC_DIM_M);
    const u64 slot_index = versions::slot_index_of_row(in.slots, tile_lo, seismic_words_value);
    if (slot_index == SEISMIC_DIM_B) {
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int r = tile_lo; r < tile_hi; ++r)
            store_gated4(in.gated, static_cast<u64>(r), channel, lanes, zero, seismic_words_value);
        return;
    }
    switch (SEISMIC_DIM_C) {
    case 3:
        convolve_tile_static<3>(in, tile_lo, tile_hi, slot_index, channel, lanes, seismic_words_value);
        break;
    case 4:
        convolve_tile_static<4>(in, tile_lo, tile_hi, slot_index, channel, lanes, seismic_words_value);
        break;
    case 2:
        convolve_tile_static<2>(in, tile_lo, tile_hi, slot_index, channel, lanes, seismic_words_value);
        break;
    case 1:
        convolve_tile_static<1>(in, tile_lo, tile_hi, slot_index, channel, lanes, seismic_words_value);
        break;
    default:
        convolve_tile_general(in, tile_lo, tile_hi, slot_index, channel, lanes, seismic_words_value);
        break;
    }
}

} // namespace short_conv

extern "C" __global__ void short_conv_rows(SEISMIC_KERNEL_PARAMS) {
    typedef versions::u64 u64;
    const short_conv::Operands in{
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION)),
        SEISMIC_OUT_WINDOW, SEISMIC_RESULT_0_OUT, VERSIONS_SLOTS()};
    const u64 channel = (static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (channel >= SEISMIC_DIM_CH)
        return;
    const u64 remaining = SEISMIC_DIM_CH - channel;
    const u64 lanes = remaining < 4 ? remaining : 4;
    const u64 rows = SEISMIC_DIM_M;
    const u64 row_tiles = (rows + short_conv::ROW_TILE - 1) / short_conv::ROW_TILE;
    const u64 total_items = row_tiles + SEISMIC_DIM_B;
    for (u64 item = blockIdx.y; item < total_items; item += gridDim.y) {
        if (item >= row_tiles)
            short_conv::publish(in, item - row_tiles, channel, lanes, seismic_words_value);
        else
            short_conv::convolve_tile(in, item, channel, lanes, seismic_words_value);
    }
}
