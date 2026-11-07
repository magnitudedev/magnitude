// The decode down projections of the routed entries (`down.h`, `output.h`):
// a simdgroup projects a few channels of one row against its staged product
// row with the GEMV's step and group sum. Its own header so that the routed
// GEMMs' libraries (`routed.h` alone) compile no GEMV.

#include "routed.h"
#include "../projection/gemv.h"

namespace routed {

// The product packets a simdgroup of the decode down projections stages at
// once (`project_channels`), of SIMDGROUPS side by side: their share of
// 16 KB of threadgroup memory, at most COLUMN_PACKETS.
template <uint SIMDGROUPS, uint COLUMN_PACKETS>
constexpr uint stage_packets() {
    constexpr uint SHARE = 16384u / (SIMDGROUPS * 64u);
    return COLUMN_PACKETS < SHARE ? COLUMN_PACKETS : SHARE;
}

// The decode down projections of one simdgroup (the choices of a row, and
// the shared expert, run side by side, one simdgroup each): channels
// first + g + G * r (lane group g of G = 32 / LANES, r < ROWS) of weights `w`
// against the one-row product `in` of COLUMNS values, over its first
// `extent` values alone (a multiple of the packet's 32 columns; COLUMNS for
// every value), stored unrounded at projected[g + G * r].
// A down row is short (20 packets at Flash Next's 640 columns), so its
// lanes are few (LANES down to 4) and a lane group owns several channels:
// that keeps ROWS * ceil(packets / LANES) weight loads in flight a lane.
// The row's packets run in chunks of STAGE: the simdgroup stages a chunk of
// the product in `staged` (STAGE * 4 words, its own; no threadgroup
// barrier), shared by every lane group, and its lanes issue the weight loads
// of all their channels for the chunk before they accumulate any. In a
// chunk the lanes of a group own its packets sub, sub + LANES, ..., summed
// in that order, then across the group.
template <typename W, uint ROWS, uint LANES, uint COLUMNS, uint STAGE, typename In>
inline void project_channels(thread const In &in, thread const projection::Weights<W> &w, uint first,
    uint rows, uint extent, threadgroup uint4 *staged, threadgroup float *projected, uint lane) {
    constexpr uint G = 32u / LANES;
    constexpr uint COLUMN_PACKETS = (COLUMNS + 31u) / 32u;
    constexpr uint CHUNK = COLUMN_PACKETS < STAGE ? COLUMN_PACKETS : STAGE;
    // Packets per lane of a chunk.
    constexpr uint PL = (CHUNK + LANES - 1u) / LANES;
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
    PROJECTION_UNROLL
    for (uint r = 0; r < ROWS; ++r) {
        const float total = projection::gemv_group_sum<LANES>(acc[r]);
        if (sub == 0)
            projected[g + G * r] = total;
    }
}

} // namespace routed
