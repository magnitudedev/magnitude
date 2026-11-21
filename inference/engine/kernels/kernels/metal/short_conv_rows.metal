// short_conv_rows (contract and portable body in short_conv.seismic): the
// gated causal depthwise convolution of one short-convolution layer. A thread
// owns 4 contiguous channels across a 16-row tile (grid y < ceil_div(M, 16))
// or publishes 4 contiguous channels of slot b's successor window (grid y =
// ceil_div(M, 16) + b). Marching across contiguous rows of a slot keeps the
// filter weights and rolling C - 1 tap history in registers for C <= 4, while
// preserving the exact tap-ascending F32 FMA chain times the gate. The
// successor window is no slot's source, so the publishing threads run beside
// the row threads.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/recurrent/versions.h"

namespace short_conv {

constant constexpr ulong ROW_TILE = 16;

inline float4 load_projection4(device const float *projection, ulong row, ulong col, uint lanes,
                               constant ulong *seismic_words) {
    const ulong base = row * SEISMIC_PROJECTION_STRIDE_0 + col * SEISMIC_PROJECTION_STRIDE_1;
    if (lanes == 4 && SEISMIC_PROJECTION_STRIDE_1 == 1)
        return element::F32::load4(projection + base);
    float4 v = float4(0.0f);
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 4; ++k) {
        if (k < lanes)
            v[k] = projection[base + ulong(k) * SEISMIC_PROJECTION_STRIDE_1];
    }
    return v;
}

inline float4 load_window4(device const float *source, ulong tap_row, ulong channel, uint lanes,
                           constant ulong *seismic_words) {
    const ulong base = tap_row * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2;
    if (lanes == 4 && SEISMIC_WINDOW_STRIDE_2 == 1)
        return element::F32::load4(source + base);
    float4 v = float4(0.0f);
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 4; ++k) {
        if (k < lanes)
            v[k] = source[base + ulong(k) * SEISMIC_WINDOW_STRIDE_2];
    }
    return v;
}

inline void store_window4(element::Out<element::F32> target, ulong tap_row, ulong channel, uint lanes,
                          float4 val, constant ulong *seismic_words) {
    const ulong base = tap_row * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2;
    if (lanes == 4 && SEISMIC_WINDOW_STRIDE_2 == 1) {
        target.put4(base, val);
        return;
    }
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 4; ++k) {
        if (k < lanes)
            target.put(base + ulong(k) * SEISMIC_WINDOW_STRIDE_2, val[k]);
    }
}

inline void store_gated4(element::Out<activation::Act> result, ulong row, ulong channel, uint lanes,
                         float4 val, constant ulong *seismic_words) {
    const ulong base = row * SEISMIC_RESULT_0_STRIDE_0 + channel * SEISMIC_RESULT_0_STRIDE_1;
    if (lanes == 4 && SEISMIC_RESULT_0_STRIDE_1 == 1) {
        result.put4(base, val);
        return;
    }
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 4; ++k) {
        if (k < lanes)
            result.put(base + ulong(k) * SEISMIC_RESULT_0_STRIDE_1, val[k]);
    }
}

inline void publish(device const float *projection,
                    device const slab::Table<element::F32> *window,
                    versions::Slots slots,
                    ulong index, ulong channel, uint lanes,
                    constant ulong *seismic_words) {
    const long taps = long(SEISMIC_DIM_C) - 1;
    const versions::Slot slot = versions::slot_of(slots, index, seismic_words);
    if (slot.lo == slot.hi) return;
    device const float *source = versions::bank_view(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    const element::Out<element::F32> target =
        versions::bank(window, slot.target, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    const long published = taps + versions::tape_rows(slot, long(SEISMIC_DIM_T));
    for (long tap = 0; tap < published; ++tap) {
        const long position = slot.stop + tap - taps;
        if (position < 0) {
            const ulong to = ulong(tap) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2;
            const ulong from = ulong(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1
                + channel * SEISMIC_WINDOW_STRIDE_2;
            if (lanes == 4 && SEISMIC_WINDOW_STRIDE_2 == 1) {
                target.put4(to, element::F32::load4(source + from));
            } else {
                for (uint k = 0; k < lanes; ++k)
                    target.copy_stored(to + ulong(k) * SEISMIC_WINDOW_STRIDE_2, source,
                                       from + ulong(k) * SEISMIC_WINDOW_STRIDE_2, 1);
            }
        } else {
            const float4 val =
                load_projection4(projection, ulong(slot.lo + position), channel, lanes, seismic_words);
            store_window4(target, ulong(tap), channel, lanes, val, seismic_words);
        }
    }
}

template <int C_STATIC>
inline void convolve_tile_static(device const float *projection,
                                 device const float *convolution,
                                 device const slab::Table<element::F32> *window,
                                 element::Out<activation::Act> result,
                                 versions::Slots slots,
                                 long tile_lo, long tile_hi,
                                 ulong slot_index, ulong channel, uint lanes,
                                 constant ulong *seismic_words) {
    constexpr int TAPS = C_STATIC - 1;
    const ulong channels = SEISMIC_DIM_CH;

    float4 w[C_STATIC];
    _Pragma("clang loop unroll(full)")
    for (int tap = 0; tap < C_STATIC; ++tap)
        w[tap] = float4(0.0f);
    if (lanes == 4) {
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < 4; ++k) {
            const ulong ch_off = (channel + ulong(k)) * SEISMIC_CONVOLUTION_STRIDE_0;
            _Pragma("clang loop unroll(full)")
            for (int tap = 0; tap < C_STATIC; ++tap)
                w[tap][k] = convolution[ch_off + ulong(tap) * SEISMIC_CONVOLUTION_STRIDE_1];
        }
    } else {
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < 4; ++k) {
            if (k < lanes) {
                const ulong ch_off = (channel + ulong(k)) * SEISMIC_CONVOLUTION_STRIDE_0;
                _Pragma("clang loop unroll(full)")
                for (int tap = 0; tap < C_STATIC; ++tap)
                    w[tap][k] = convolution[ch_off + ulong(tap) * SEISMIC_CONVOLUTION_STRIDE_1];
            }
        }
    }

    long row = tile_lo;
    while (row < tile_hi) {
        if (slot_index == SEISMIC_DIM_B) {
            const float4 zero = float4(0.0f);
            for (; row < tile_hi; ++row)
                store_gated4(result, ulong(row), channel, lanes, zero, seismic_words);
            return;
        }
        const versions::Slot slot = versions::slot_of(slots, slot_index, seismic_words);
        device const float *source = versions::bank_view(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
        const long run_end = tile_hi < slot.hi ? tile_hi : slot.hi;

        float4 ring[TAPS > 0 ? TAPS : 1];
        if constexpr (TAPS > 0) {
            const long local0 = row - slot.lo;
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < TAPS; ++t) {
                const long pos = local0 + long(t) - long(TAPS);
                ring[t] = pos < 0
                    ? load_window4(source, ulong(slot.taped + long(TAPS) + pos), channel, lanes, seismic_words)
                    : load_projection4(projection, ulong(slot.lo + pos), channel, lanes, seismic_words);
            }
        }

        for (long r = row; r < run_end; ++r) {
            const float4 cur_u = load_projection4(projection, ulong(r), channel, lanes, seismic_words);
            const float4 gate = load_projection4(projection, ulong(r), channels + channel, lanes, seismic_words);

            float4 sum = float4(0.0f);
            if constexpr (TAPS > 0) {
                _Pragma("clang loop unroll(full)")
                for (int t = 0; t < TAPS; ++t)
                    sum = metal::fma(w[t], ring[t], sum);
            }
            sum = metal::fma(w[TAPS], cur_u, sum);
            store_gated4(result, ulong(r), channel, lanes, gate * sum, seismic_words);

            if constexpr (TAPS > 0) {
                if constexpr (TAPS > 1) {
                    _Pragma("clang loop unroll(full)")
                    for (int t = 0; t + 1 < TAPS; ++t)
                        ring[t] = ring[t + 1];
                }
                ring[TAPS > 0 ? TAPS - 1 : 0] = cur_u;
            }
        }

        row = run_end;
        if (row < tile_hi)
            slot_index = versions::slot_index_of_row(slots, row, seismic_words);
    }
}

inline void convolve_tile_general(device const float *projection,
                                  device const float *convolution,
                                  device const slab::Table<element::F32> *window,
                                  element::Out<activation::Act> result,
                                  versions::Slots slots,
                                  long tile_lo, long tile_hi,
                                  ulong slot_index, ulong channel, uint lanes,
                                  constant ulong *seismic_words) {
    const long taps = long(SEISMIC_DIM_C) - 1;
    const ulong channels = SEISMIC_DIM_CH;
    long row = tile_lo;
    while (row < tile_hi) {
        if (slot_index == SEISMIC_DIM_B) {
            const float4 zero = float4(0.0f);
            for (; row < tile_hi; ++row)
                store_gated4(result, ulong(row), channel, lanes, zero, seismic_words);
            return;
        }
        const versions::Slot slot = versions::slot_of(slots, slot_index, seismic_words);
        device const float *source = versions::bank_view(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
        const long run_end = tile_hi < slot.hi ? tile_hi : slot.hi;
        for (long r = row; r < run_end; ++r) {
            const long local = r - slot.lo;
            float4 sum = float4(0.0f);
            for (long tap = 0; tap <= taps; ++tap) {
                const long position = local + tap - taps;
                const float4 value = position < 0
                    ? load_window4(source, ulong(slot.taped + taps + position), channel, lanes, seismic_words)
                    : load_projection4(projection, ulong(slot.lo + position), channel, lanes, seismic_words);
                float4 wt = float4(0.0f);
                const ulong tap_off = ulong(tap) * SEISMIC_CONVOLUTION_STRIDE_1;
                _Pragma("clang loop unroll(full)")
                for (uint k = 0; k < 4; ++k) {
                    if (k < lanes)
                        wt[k] = convolution[(channel + ulong(k)) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off];
                }
                sum = metal::fma(wt, value, sum);
            }
            const float4 gate = load_projection4(projection, ulong(r), channels + channel, lanes, seismic_words);
            store_gated4(result, ulong(r), channel, lanes, gate * sum, seismic_words);
        }
        row = run_end;
        if (row < tile_hi)
            slot_index = versions::slot_index_of_row(slots, row, seismic_words);
    }
}

} // namespace short_conv

kernel void short_conv_rows(
    device const float *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    SEISMIC_OUT_WINDOW_PARAM(window),
    SEISMIC_RESULT_0_PARAM(gated),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    const ulong channel = (ulong(group.x) * threadgroup_shape.x + thread_index) * 4;
    if (channel >= SEISMIC_DIM_CH)
        return;
    const ulong remaining = SEISMIC_DIM_CH - channel;
    const uint lanes = remaining < 4 ? uint(remaining) : 4u;
    const versions::Slots slots{segments, stop, previous_bank, previous_tape, following_bank};
    const ulong rows = SEISMIC_DIM_M;
    const ulong row_tiles = (rows + short_conv::ROW_TILE - 1) / short_conv::ROW_TILE;
    if (group.y >= row_tiles) {
        short_conv::publish(projection, window, slots, ulong(group.y) - row_tiles, channel, lanes, seismic_words);
        return;
    }
    const element::Out<activation::Act> result(gated);
    const long tile_lo = long(ulong(group.y) * short_conv::ROW_TILE);
    const ulong end = ulong(group.y) * short_conv::ROW_TILE + short_conv::ROW_TILE;
    const long tile_hi = long(end < rows ? end : rows);
    const ulong slot_index = versions::slot_index_of_row(slots, tile_lo, seismic_words);
    if (slot_index == SEISMIC_DIM_B) {
        const float4 zero = float4(0.0f);
        for (long r = tile_lo; r < tile_hi; ++r)
            short_conv::store_gated4(result, ulong(r), channel, lanes, zero, seismic_words);
        return;
    }
    switch (SEISMIC_DIM_C) {
    case 3:
        short_conv::convolve_tile_static<3>(projection, convolution, window, result, slots,
                                            tile_lo, tile_hi, slot_index, channel, lanes, seismic_words);
        break;
    case 4:
        short_conv::convolve_tile_static<4>(projection, convolution, window, result, slots,
                                            tile_lo, tile_hi, slot_index, channel, lanes, seismic_words);
        break;
    case 2:
        short_conv::convolve_tile_static<2>(projection, convolution, window, result, slots,
                                            tile_lo, tile_hi, slot_index, channel, lanes, seismic_words);
        break;
    case 1:
        short_conv::convolve_tile_static<1>(projection, convolution, window, result, slots,
                                            tile_lo, tile_hi, slot_index, channel, lanes, seismic_words);
        break;
    default:
        short_conv::convolve_tile_general(projection, convolution, window, result, slots,
                                          tile_lo, tile_hi, slot_index, channel, lanes, seismic_words);
        break;
    }
}
