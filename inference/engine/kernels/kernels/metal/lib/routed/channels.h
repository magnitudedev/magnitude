// The decode down projections of the routed entries (`down.h`, `output.h`):
// a simdgroup projects a few channels of one row against its product row
// with the GEMV's step and group sum. Its own header so that the routed
// GEMMs' libraries (`routed.h` alone) compile no GEMV.

#include "routed.h"
#include "../projection/gemv.h"

namespace routed {

// The product packets a simdgroup of the decode down projections stages at
// once (`project_channels`), of SIMDGROUPS side by side, at LANES lanes a
// row. At 16 or 32 lanes none: one or two lane groups read each packet of
// the product, so a lane loads its own packets straight into registers, and
// a stage would buy no load while its round trip through threadgroup memory
// lengthens every launch. Below 16 lanes, several lane groups share each
// packet and a lane would hold several: their share of 16 KB of threadgroup
// memory, at most COLUMN_PACKETS.
template <uint SIMDGROUPS, uint LANES, uint COLUMN_PACKETS>
constexpr uint stage_packets() {
    constexpr uint SHARE = 16384u / (SIMDGROUPS * 64u);
    return LANES >= 16u ? 0u : COLUMN_PACKETS < SHARE ? COLUMN_PACKETS : SHARE;
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
