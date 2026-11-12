// The decode down projections of the routed entries (`down.h`, `output.h`):
// a simdgroup projects a few channels of one row against its product row
// with the GEMV's step and group sum. Its own header so that the routed
// GEMMs' libraries (`routed.h` alone) compile no GEMV.

#include "routed.h"
#include "../projection/gemv.h"

namespace routed {

// The product packets a simdgroup of the decode down projections stages at
// once (`project_channels`), of SIMDGROUPS side by side, at LANES lanes over
// a row of COLUMN_PACKETS: its share of 16 KB of threadgroup memory, at most
// the row, or none. Below 16 lanes always: several lane groups share each
// packet and a lane would hold several. At 32 lanes never: one lane group
// reads each packet once, straight into registers. At 16 lanes, where both
// lane groups read the whole row, only where AT_16 (`routed_down`, the one
// projection where it is measured) and a lane would hold more than 2
// packets of the row: past that the staged form is faster at every timed
// shape, at or below it slower; why is open, as it reads neither ROWS nor
// the format, which the gain also depends on. On M4 Pro, routed_down at
// ROWS 1 LANES 16, MXFP4, timed, µs a launch unstaged / staged: 3.5 packets
// a lane (LFM2) 75.7 / 63.2, 3.6 (Nemotron Lightning) 121.7 / 90.8, 4.5
// (DeepSeek V4.1) 270.2 / 202.9, 5.25 (Nemotron Super) 381.5 / 250.3; 1.4
// (Gemma 4 26B) 57.9 / 68.1 and 2 (Laguna) 116.7 / 152.0. Each entry's
// declaration bounds the weight packets a lane then holds (routed.seismic).
template <uint SIMDGROUPS, uint LANES, uint COLUMN_PACKETS, bool AT_16>
constexpr uint stage_packets() {
    constexpr uint SHARE = 16384u / (SIMDGROUPS * 64u);
    constexpr bool STAGED = LANES < 16u || (AT_16 && LANES == 16u && (COLUMN_PACKETS + 15u) / 16u > 2u);
    return !STAGED ? 0u : COLUMN_PACKETS < SHARE ? COLUMN_PACKETS : SHARE;
}

// A simdgroup's region of threadgroup memory, in floats: its stage of STAGE
// packets (16 floats each) while it projects, then its TILE projections.
template <uint TILE, uint STAGE>
constexpr uint projection_region() {
    return 16u * STAGE > TILE ? 16u * STAGE : TILE;
}

// The decode down projections of one simdgroup (the choices of a row, and
// the shared expert, run side by side, one simdgroup each): channels
// first + g + G * r (lane group g of G = 32 / LANES, r < ROWS) of weights `w`
// against the one-row product `in` of COLUMNS values, over its first
// `extent` values alone (a multiple of the packet's 32 columns; COLUMNS for
// every value), stored unrounded at region[g + G * r] (`region` its
// `projection_region<TILE, STAGE>()` floats).
// A down row is short (20 packets at Flash Next's 640 columns), so its
// lanes are few (LANES down to 4) and a lane group owns several channels:
// that keeps ROWS * ceil(packets / LANES) weight loads in flight a lane,
// each lane issuing the weight loads of all its channels before it
// accumulates any. With no stage (STAGE 0, `stage_packets`) a lane reads its
// packets of the product straight from device memory, without a barrier.
// Otherwise the row's packets run in chunks of STAGE: the simdgroup stages a
// chunk of the product at the start of `region` (no threadgroup barrier),
// shared by every lane group. In a chunk (the whole row without a stage) the
// lanes of a group own its packets sub, sub + LANES, ..., summed in that
// order, then across the group.
template <typename W, uint ROWS, uint LANES, uint COLUMNS, uint STAGE, typename In>
inline void project_channels(thread const In &in, thread const projection::Weights<W> &w, uint first,
    uint rows, uint extent, threadgroup float *region, uint lane) {
    constexpr uint G = 32u / LANES;
    constexpr uint COLUMN_PACKETS = (COLUMNS + 31u) / 32u;
    // The packets read: those of the extent.
    const uint P = min(COLUMN_PACKETS, (extent + 31u) / 32u);
    const uint g = lane / LANES, sub = lane % LANES;
    uint n[ROWS];
    PROJECTION_UNROLL
    for (uint r = 0; r < ROWS; ++r)
        n[r] = min(first + g + G * r, rows - 1u);
    float acc[ROWS];
    PROJECTION_UNROLL
    for (uint r = 0; r < ROWS; ++r)
        acc[r] = 0.0f;
    if constexpr (STAGE == 0) {
        // Packets per lane.
        constexpr uint PL = (COLUMN_PACKETS + LANES - 1u) / LANES;
        typename W::packet packets[ROWS][PL];
        PROJECTION_UNROLL
        for (uint r = 0; r < ROWS; ++r) {
            PROJECTION_UNROLL
            for (uint i = 0; i < PL; ++i) {
                const uint p = sub + LANES * i;
                if (p < P)
                    packets[r][i] = w.packet(n[r], p);
            }
        }
        uint4 x[PL][4];
        PROJECTION_UNROLL
        for (uint i = 0; i < PL; ++i) {
            const uint p = sub + LANES * i;
            PROJECTION_UNROLL
            for (uint step = 0; step < 4; ++step)
                x[i][step] = p < P ? in.words8(0, 32u * p + 8u * step) : uint4(0);
        }
        PROJECTION_UNROLL
        for (uint i = 0; i < PL; ++i) {
            if (sub + LANES * i >= P)
                continue;
            PROJECTION_UNROLL
            for (uint step = 0; step < 4; ++step) {
                float4 xe, xo;
                Act::split8(x[i][step], xe, xo);
                PROJECTION_UNROLL
                for (uint r = 0; r < ROWS; ++r) {
                    float4 we, wo;
                    W::values(packets[r][i], step, we, wo);
                    acc[r] = projection::gemv_step_dot(acc[r], we, wo, xe, xo);
                }
            }
        }
    } else {
        constexpr uint CHUNK = COLUMN_PACKETS < STAGE ? COLUMN_PACKETS : STAGE;
        // Packets per lane of a chunk.
        constexpr uint PL = (CHUNK + LANES - 1u) / LANES;
        threadgroup uint4 *staged = reinterpret_cast<threadgroup uint4 *>(region);
        for (uint base = 0; base < P; base += CHUNK) {
            const uint count = min(CHUNK, P - base);
            typename W::packet packets[ROWS][PL];
            PROJECTION_UNROLL
            for (uint r = 0; r < ROWS; ++r) {
                PROJECTION_UNROLL
                for (uint i = 0; i < PL; ++i) {
                    const uint p = sub + LANES * i;
                    if (p < count)
                        packets[r][i] = w.packet(n[r], base + p);
                }
            }
            // The previous chunk's reads of the stage end before this one's
            // writes; its writes before any lane reads them.
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint word = lane; word < 4u * count; word += 32u)
                staged[word] = in.words8(0, 32u * base + 8u * word);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            PROJECTION_UNROLL
            for (uint i = 0; i < PL; ++i) {
                const uint p = sub + LANES * i;
                if (p >= count)
                    continue;
                PROJECTION_UNROLL
                for (uint step = 0; step < 4; ++step) {
                    float4 xe, xo;
                    Act::split8(staged[4u * p + step], xe, xo);
                    PROJECTION_UNROLL
                    for (uint r = 0; r < ROWS; ++r) {
                        float4 we, wo;
                        W::values(packets[r][i], step, we, wo);
                        acc[r] = projection::gemv_step_dot(acc[r], we, wo, xe, xo);
                    }
                }
            }
        }
        // The last chunk's reads of the stage end before the projections
        // take its place.
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    PROJECTION_UNROLL
    for (uint r = 0; r < ROWS; ++r) {
        const float total = projection::gemv_group_sum<LANES>(acc[r]);
        if (sub == 0)
            region[g + G * r] = total;
    }
}

} // namespace routed
