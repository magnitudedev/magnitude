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
    constexpr uint SHARE = (uint(SEISMIC_DEVICE_SHARED_BYTES_PER_GROUP) / 2u) / (SIMDGROUPS * 64u);
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
// One-row accumulation of `project_channels` into `acc[ROWS]`, using `region`
// as staging when `STAGE > 0`.
template <typename W, uint ROWS, uint LANES, uint COLUMNS, uint STAGE, typename In>
inline void project_channels_one(thread const In &in, thread const projection::Weights<W> &w,
    thread const uint (&n)[ROWS], uint P, threadgroup float *region, uint lane, thread float (&acc)[ROWS]) {
    constexpr uint COLUMN_PACKETS = (COLUMNS + 31u) / 32u;
    const uint sub = lane % LANES;
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
                    projection::gemv_step_weights<W>(packets[r][i], step, we, wo);
                    acc[r] = projection::gemv_step_add<W>(acc[r], packets[r][i], step, we, wo, xe, xo);
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
            if (base > 0u)
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
                        projection::gemv_step_weights<W>(packets[r][i], step, we, wo);
                        acc[r] = projection::gemv_step_add<W>(acc[r], packets[r][i], step, we, wo, xe, xo);
                    }
                }
            }
        }
    }
}

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
    project_channels_one<W, ROWS, LANES, COLUMNS, STAGE>(in, w, n, P, region, lane, acc);
    if constexpr (STAGE > 0) {
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

// Operands of up to MAXM rows formed by one simdgroup alone (no threadgroup
// barrier): plain rows unchanged, or NVFP4 rows with per-row global scales in
// registers (`projection::RegGlobals<MAXM>`).
template <bool NVFP4, uint MAXM>
struct lone_operands;

template <uint MAXM>
struct lone_operands<false, MAXM> {
    template <typename In>
    static In form(In in, uint, uint, uint) {
        return in;
    }
};

template <uint MAXM>
struct lone_operands<true, MAXM> {
    template <typename In>
    static projection::Nvfp4Rows<In, projection::RegGlobals<MAXM>> form(In in, uint m_rows, uint columns, uint lane) {
        projection::RegGlobals<MAXM> globals = {};
        for (uint m = 0; m < m_rows; ++m) {
            uint amax = 0u;
            for (uint k = 8u * lane; k < columns; k += 256u) {
                float4 even, odd;
                in.load8(m, k, 0.0f, even, odd);
                amax = packets::magnitude_peak(amax, even, odd);
            }
            globals.g[m] = packets::nvfp4::global_scale(as_type<float>(simd_max(amax)));
        }
        return projection::Nvfp4Rows<In, projection::RegGlobals<MAXM>>{in, globals};
    }
};

// Accumulate one loaded weight packet of ROWS channels into MAXM rows'
// accumulators, decoding each weight once across all rows.
template <typename W, uint ROWS, uint MAXM, typename In>
inline void project_channels_step(thread const In &in, thread const typename W::packet (&pkt)[ROWS], uint col_base,
    uint m_rows, thread float (&acc)[ROWS][MAXM]) {
    PROJECTION_UNROLL
    for (uint step = 0; step < 4; ++step) {
        const uint col = col_base + 8u * step;
        if constexpr (ROWS <= 4u) {
            float4 we[ROWS], wo[ROWS];
            PROJECTION_UNROLL
            for (uint r = 0; r < ROWS; ++r)
                projection::gemv_step_weights<W>(pkt[r], step, we[r], wo[r]);
            PROJECTION_UNROLL
            for (uint m = 0; m < MAXM; ++m) {
                if (m < m_rows) {
                    float4 xe, xo;
                    Act::split8(in.words8(m, col), xe, xo);
                    PROJECTION_UNROLL
                    for (uint r = 0; r < ROWS; ++r)
                        acc[r][m] = projection::gemv_step_add<W>(acc[r][m], pkt[r], step, we[r], wo[r], xe, xo);
                }
            }
        } else {
            uint4 xw[MAXM];
            PROJECTION_UNROLL
            for (uint m = 0; m < MAXM; ++m)
                xw[m] = m < m_rows ? in.words8(m, col) : uint4(0);
            PROJECTION_UNROLL
            for (uint r = 0; r < ROWS; ++r) {
                float4 we, wo;
                projection::gemv_step_weights<W>(pkt[r], step, we, wo);
                PROJECTION_UNROLL
                for (uint m = 0; m < MAXM; ++m) {
                    if (m < m_rows) {
                        float4 xe, xo;
                        Act::split8(xw[m], xe, xo);
                        acc[r][m] = projection::gemv_step_add<W>(acc[r][m], pkt[r], step, we, wo, xe, xo);
                    }
                }
            }
        }
    }
}

// Threadgroup-staged activation rows for `project_channels_step`: row `m`
// begins at `m * STRIDE_WORDS` `uint4` words (8 activation values per word).
template <uint STRIDE_WORDS>
struct StagedRows {
    threadgroup const uint4 *staged;
    inline uint4 words8(uint m, uint col) const {
        return staged[m * STRIDE_WORDS + (col >> 3)];
    }
};

// Project one contiguous packet slice `[pkt_base, pkt_base + count)` into
// `acc`, reading activations from `in` at column `32u * (in_pkt_base + p)`.
// Software-pipelines `current`/`next` weight packets for `ROWS <= 2 && MAXM < 8`
// (wider rows or 8-row batches hold too many live registers for Apple GPU
// occupancy).
template <typename W, uint ROWS, uint LANES, uint MAXM, typename In>
inline void project_channels_slice(thread const In &in, thread const projection::Weights<W> &w,
    thread const uint (&n)[ROWS], uint pkt_base, uint in_pkt_base, uint count, uint m_rows, uint sub,
    thread float (&acc)[ROWS][MAXM]) {
    if constexpr (ROWS <= 2u && MAXM < 8u) {
        typename W::packet current[ROWS], next[ROWS];
        if (sub < count) {
            PROJECTION_UNROLL
            for (uint r = 0; r < ROWS; ++r)
                current[r] = w.packet(n[r], pkt_base + sub);
        }
        for (uint p = sub; p < count; p += LANES) {
            if (p + LANES < count) {
                PROJECTION_UNROLL
                for (uint r = 0; r < ROWS; ++r)
                    next[r] = w.packet(n[r], pkt_base + p + LANES);
            }
            project_channels_step<W, ROWS, MAXM>(in, current, 32u * (in_pkt_base + p), m_rows, acc);
            if (p + LANES < count) {
                PROJECTION_UNROLL
                for (uint r = 0; r < ROWS; ++r)
                    current[r] = next[r];
            }
        }
    } else {
        for (uint p = sub; p < count; p += LANES) {
            typename W::packet pkt[ROWS];
            PROJECTION_UNROLL
            for (uint r = 0; r < ROWS; ++r)
                pkt[r] = w.packet(n[r], pkt_base + p);
            project_channels_step<W, ROWS, MAXM>(in, pkt, 32u * (in_pkt_base + p), m_rows, acc);
        }
    }
}

// Fixed threadgroup width of the gathered summed projections (`routed_output`,
// `routed_down`, and their `*_project` forms): 4 simdgroups (128 threads)
// cooperatively staging one shared activation chunk for the threadgroup's
// unique expert so occupancy stays high (4 KB at LANES <= 8) and every MAXM
// stays staged.
constexpr constant uint GATHERED_SUMMED_SIMDGROUPS = 4u;

template <uint LANES, bool ONE = false>
struct GatheredSummedShared {
    static constexpr uint stage_pkts() { return LANES >= 8u ? LANES : 8u; }
    static constexpr uint stage_rows() { return ONE ? 1u : 4u; }
    uint4 staged[stage_rows() * 4u * stage_pkts()];
    uint products[ONE ? 1u : SEISMIC_CONST_VERIFY_ROWS];
    ulong offsets[ONE ? 1u : SEISMIC_CONST_VERIFY_ROWS];
};

// Batched decode down projection of `m_rows` (<= MAXM) rows against one weight
// matrix `w` in a single weight pass, with all `GATHERED_SUMMED_SIMDGROUPS`
// simdgroups cooperatively staging each `STAGE_PKTS` chunk (a multiple of
// `LANES`, preserving `project_channels`' exact per-lane packet order so every
// row keeps its one-row bits) into `staged`, and each active simdgroup
// (`first < H`) publishing its channels directly to `published_rows[m]`.
template <typename W, uint ROWS, uint LANES, uint MAXM, uint COLUMNS, uint H, bool NVFP4, typename In,
    typename Published>
inline void project_channels_published(thread const In &raw, thread const projection::Weights<W> &w, uint m_rows,
    threadgroup const uint *published_rows, uint first, uint rows, uint extent, threadgroup uint4 *staged,
    Published published, uint tid, uint lane) {
    constexpr uint G = 32u / LANES;
    constexpr uint COLUMN_PACKETS = (COLUMNS + 31u) / 32u;
    constexpr uint STAGE_PKTS = GatheredSummedShared<LANES>::stage_pkts();
    constexpr uint ROW_WORDS = 4u * STAGE_PKTS;
    constexpr uint THREADS = 32u * GATHERED_SUMMED_SIMDGROUPS;
    const uint P = min(COLUMN_PACKETS, (extent + 31u) / 32u);
    const uint g = lane / LANES, sub = lane % LANES;
    const bool active_sg = first < H;
    const auto in = lone_operands<NVFP4, MAXM>::form(raw, m_rows, COLUMNS, lane);
    uint n[ROWS];
    PROJECTION_UNROLL
    for (uint r = 0; r < ROWS; ++r)
        n[r] = min(first + g + G * r, rows - 1u);
    float acc[ROWS][MAXM];
    PROJECTION_UNROLL
    for (uint r = 0; r < ROWS; ++r) {
        PROJECTION_UNROLL
        for (uint m = 0; m < MAXM; ++m)
            acc[r][m] = 0.0f;
    }
    const StagedRows<ROW_WORDS> staged_in{staged};
    for (uint base = 0; base < P; base += STAGE_PKTS) {
        const uint count = min(STAGE_PKTS, P - base);
        if (base > 0u)
            threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint words_per_row = 4u * count;
        const uint total_words = m_rows * words_per_row;
        for (uint idx = tid; idx < total_words; idx += THREADS) {
            const uint m = idx / words_per_row;
            const uint word = idx - m * words_per_row;
            staged[m * ROW_WORDS + word] = in.words8(m, 32u * base + 8u * word);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active_sg)
            project_channels_slice<W, ROWS, LANES, MAXM>(staged_in, w, n, base, 0u, count, m_rows, sub, acc);
    }
    if (!active_sg)
        return;
    PROJECTION_UNROLL
    for (uint r = 0; r < ROWS; ++r) {
        const uint channel = first + g + G * r;
        PROJECTION_UNROLL
        for (uint m = 0; m < MAXM; ++m) {
            if (m < m_rows) {
                const float total = projection::gemv_group_sum<LANES>(acc[r][m]);
                if (sub == (r * MAXM + m) % LANES && channel < H)
                    publish_channel<H>(published, published_rows[m], channel, total);
            }
        }
    }
}

// Gathered routed expert down projection: all `GATHERED_SUMMED_SIMDGROUPS`
// simdgroups cooperatively unpack compact worklist entry `entry` into `tg` and
// project every choice of it in one weight pass over this simdgroup's channel
// slice `[first, first + TILE)`, publishing each choice's slot directly into
// `published`.
template <typename W, uint ROWS, uint LANES, uint SLOTS, uint COLUMNS, uint H, bool NVFP4, bool ONE = false,
    typename Published>
inline void project_gathered_expert(device const uchar *product, ulong p0, ulong p1, ulong p2,
    device const WorkEntry &entry, device const ulong *expert_down, device const ulong *tier_down,
    packets::Rows16 layout, uint first, uint tid, uint lane, threadgroup GatheredSummedShared<LANES, ONE> &tg,
    Published published) {
    constexpr uint TILE = (32u / LANES) * ROWS;
    const uint m_rows = ONE ? 1u : entry.count;
    const slab::Source source{entry.source};
    if (tid < m_rows) {
        const uint packed = entry.packed[tid];
        const uint m = packed >> 16u;
        const uint k = packed & 0xffffu;
        tg.products[tid] = m * SLOTS + k;
        tg.offsets[tid] = ulong(m) * p0 + ulong(k) * p1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (source.none()) {
        if (first < H) {
            for (uint r = 0; r < m_rows; ++r)
                publish_zero<TILE, H>(published, tg.products[r], first, lane);
        }
        return;
    }
    const bool aligned = p2 == 1ul && (p0 & 7ul) == 0ul && (p1 & 7ul) == 0ul;
    const auto down = expert_matrix<W>(Experts{expert_down, tier_down}, source, layout, COLUMNS);
    if constexpr (ONE) {
        const GatheredProduct in{product, tg.offsets, p2, COLUMNS, aligned};
        project_channels_published<W, ROWS, LANES, 1u, COLUMNS, H, NVFP4>(
            in, down, 1u, tg.products, first, H, COLUMNS, tg.staged, published, tid, lane);
    } else {
        for (uint base = 0u; base < m_rows; base += 4u) {
            const uint chunk = min(4u, m_rows - base);
            if (base > 0u)
                threadgroup_barrier(mem_flags::mem_threadgroup);
            const GatheredProduct in{product, tg.offsets + base, p2, COLUMNS, aligned};
            PROJECTION_FOR_ROWS_24(chunk,
                (project_channels_published<W, ROWS, LANES, MAXM, COLUMNS, H, NVFP4>(
                    in, down, chunk, tg.products + base, first, H, COLUMNS, tg.staged, published, tid, lane)));
        }
    }
}

} // namespace routed
