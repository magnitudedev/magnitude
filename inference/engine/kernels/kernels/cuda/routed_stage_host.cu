// Unified device-classified 3-tier decode staging (`routed_stage_host` in
// routed.seismic) for CUDA:
//
// 1. Classifies every choice in `front_sources` into:
//    - `VRAM_HIT` (`source >= 0`): already served by `DecodeHits` (masked to -1
//      in `back_sources` when `READY_RING_MASK_HITS != 0`, else preserved).
//    - `HOST_STAGE` (`source <= -2`, unique host-RAM count `<= DECODE_STAGE_SLOTS`
//      and `< policy[POLICY_CE_MIN_EXPERTS]`): deduplicated by physical
//      `tier_record`, assigned VRAM scratch slot `stage_base + s`, copied by
//      all 4 CTAs x 256 threads via coalesced `uint4` `__ldcs` (`UNROLL=4`).
//    - `HOST_WORK` (NVMe miss `land_grid[0] != 0`, CPU-sited `site >= 0`, or
//      host-RAM burst `>= ce_min` / `> DECODE_STAGE_SLOTS`): wakes `DecodeReactor`
//      via `ready_slot[READY_RING_READY_GEN]`.
// 2. When `host_work == 0`, CTA 0 thread 0 executes `__threadfence_system()`
//    and `st.release.sys.global.u32 [land_gate_addr], (u32)gen`, satisfying
//    the top-level `BatchMemOp(WAIT_VALUE_32: land_gate >= g_L)` purely on
//    device with zero CPU wakeup.
// 3. When `host_work > 0`, CTA 0 thread 0 publishes `ready_slot` with
//    `st.release.sys.global.u64 [&ready_slot[READY_RING_READY_GEN]], gen`,
//    after `back_sources` is written, so `DecodeReactor` services only residual
//    `HOST_WORK` and preserves any SM-staged `back_sources` entries.

#include <seismic/slab.cuh>
#include "lib/streaming/control.cuh"
#include <seismic/element.cuh>

extern "C" __global__ void routed_stage_host(SEISMIC_KERNEL_PARAMS) {
    __shared__ int s_tiers[SEISMIC_CONST_DECODE_STAGE_SLOTS];
    __shared__ unsigned s_stage_count;
    __shared__ int s_flags;
    __shared__ int s_stage_base;
    __shared__ int s_mask_hits;

    const unsigned K = static_cast<unsigned>(SEISMIC_DIM_K);
    const unsigned choices = static_cast<unsigned>(SEISMIC_DIM_M) * K;
    const int *front = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FRONT_SOURCES));
    int *back = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_BACK_SOURCES));
    const int *policy = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_POLICY));
    const int *step = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    const int *land_grid = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LAND_GRID));
    int *ready = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_READY_SLOT));

    if (threadIdx.x == 0) {
        const int ce_min = streaming_load(policy + SEISMIC_CONST_POLICY_CE_MIN_EXPERTS * SEISMIC_POLICY_STRIDE_0);
        const int stage_base = streaming_load(ready + SEISMIC_CONST_READY_RING_STAGE_BASE * SEISMIC_READY_SLOT_STRIDE_0);
        const int mask_hits = streaming_load(ready + SEISMIC_CONST_READY_RING_MASK_HITS * SEISMIC_READY_SLOT_STRIDE_0);
        const bool file_pending = (land_grid[0] != 0);
        bool cpu_sited = false;
        unsigned unique_host_ram = 0u;
        for (unsigned c = 0; c < choices; ++c) {
            const u64 at = u64(c / K) * SEISMIC_FRONT_SOURCES_STRIDE_1 + u64(c % K) * SEISMIC_FRONT_SOURCES_STRIDE_2;
            const int src = front[SEISMIC_CONST_SOURCE_RECORD * SEISMIC_FRONT_SOURCES_STRIDE_0 + at];
            const int site = front[SEISMIC_CONST_SOURCE_SITE * SEISMIC_FRONT_SOURCES_STRIDE_0 + at];
            if (site >= 0) {
                cpu_sited = true;
            }
            if (src <= -2) {
                const int tr = -2 - src;
                bool seen = false;
                const unsigned cap = unique_host_ram < static_cast<unsigned>(SEISMIC_CONST_DECODE_STAGE_SLOTS)
                                         ? unique_host_ram
                                         : static_cast<unsigned>(SEISMIC_CONST_DECODE_STAGE_SLOTS);
                for (unsigned i = 0; i < cap; ++i) {
                    if (s_tiers[i] == tr) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) {
                    if (unique_host_ram < static_cast<unsigned>(SEISMIC_CONST_DECODE_STAGE_SLOTS)) {
                        s_tiers[unique_host_ram] = tr;
                    }
                    ++unique_host_ram;
                }
            }
        }
        const unsigned max_stage = static_cast<unsigned>(
            (SEISMIC_DIM_E < SEISMIC_CONST_DECODE_STAGE_SLOTS) ? SEISMIC_DIM_E : SEISMIC_CONST_DECODE_STAGE_SLOTS);
        const bool use_sm_stage = (unique_host_ram > 0u) && (stage_base >= 0) &&
                                  (unique_host_ram <= max_stage) &&
                                  (ce_min > 0) && (unique_host_ram < static_cast<unsigned>(ce_min));
        const bool ce_dma = (unique_host_ram > 0u) && !use_sm_stage;
        int flags = 0;
        if (file_pending)
            flags |= SEISMIC_CONST_READY_FLAG_FILE_PENDING;
        if (cpu_sited)
            flags |= SEISMIC_CONST_READY_FLAG_CPU_SITED;
        if (ce_dma)
            flags |= SEISMIC_CONST_READY_FLAG_CE_DMA;
        s_stage_count = use_sm_stage ? unique_host_ram : 0u;
        s_flags = flags;
        s_stage_base = stage_base;
        s_mask_hits = mask_hits;
    }
    __syncthreads();

    const unsigned stage_count = s_stage_count;
    const int stage_base = s_stage_base;

    // CTA 0 writes `back_sources` and either self-gates `land_gate` (host_work == 0)
    // or publishes `ready_slot` (host_work > 0).
    if (blockIdx.x == 0) {
        const int mask_hits = s_mask_hits;
        for (unsigned c = threadIdx.x; c < choices; c += blockDim.x) {
            const u64 f_at = u64(c / K) * SEISMIC_FRONT_SOURCES_STRIDE_1 + u64(c % K) * SEISMIC_FRONT_SOURCES_STRIDE_2;
            const u64 b_at = u64(c / K) * SEISMIC_BACK_SOURCES_STRIDE_1 + u64(c % K) * SEISMIC_BACK_SOURCES_STRIDE_2;
            const int src = front[SEISMIC_CONST_SOURCE_RECORD * SEISMIC_FRONT_SOURCES_STRIDE_0 + f_at];
            const int site = front[SEISMIC_CONST_SOURCE_SITE * SEISMIC_FRONT_SOURCES_STRIDE_0 + f_at];
            int back_rec = -1;
            if (src >= 0) {
                back_rec = (mask_hits != 0) ? -1 : src;
            } else if (src <= -2) {
                if (stage_count > 0u) {
                    const int tr = -2 - src;
                    for (unsigned i = 0; i < stage_count; ++i) {
                        if (s_tiers[i] == tr) {
                            back_rec = stage_base + static_cast<int>(i);
                            break;
                        }
                    }
                } else {
                    // Residual CE_DMA miss: preserve `-2 - tier_record` for DecodeReactor.
                    back_rec = src;
                }
            }
            streaming_store(back + SEISMIC_CONST_SOURCE_RECORD * SEISMIC_BACK_SOURCES_STRIDE_0 + b_at, back_rec);
            streaming_store(back + SEISMIC_CONST_SOURCE_SITE * SEISMIC_BACK_SOURCES_STRIDE_0 + b_at, site);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            const u64 gen = streaming_load_u64(ready + SEISMIC_CONST_READY_RING_GEN * SEISMIC_READY_SLOT_STRIDE_0);
            if (s_flags == 0) {
                const u64 gate_addr =
                    streaming_load_u64(ready + SEISMIC_CONST_READY_RING_GATE_ADDR * SEISMIC_READY_SLOT_STRIDE_0);
                if (gate_addr != 0ull) {
                    const unsigned int g32 = static_cast<unsigned int>(gen);
                    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(gate_addr), "r"(g32) : "memory");
                }
            } else {
                const u64 q = streaming_step(step, SEISMIC_STEP_STRIDE_0);
                const u64 seq_idx = SEISMIC_CONST_READY_RING_SEQUENCE * SEISMIC_READY_SLOT_STRIDE_0;
                streaming_store(ready + seq_idx, static_cast<int>(q & 0xFFFFFFFFull));
                streaming_store(ready + seq_idx + 1, static_cast<int>(q >> 32));
                streaming_store(ready + SEISMIC_CONST_READY_RING_FLAGS * SEISMIC_READY_SLOT_STRIDE_0, s_flags);
                streaming_store_u64_release(
                    ready + SEISMIC_CONST_READY_RING_READY_GEN * SEISMIC_READY_SLOT_STRIDE_0, gen);
            }
        }
    }

    if (stage_count == 0u)
        return;

    // All 4 CTAs x 256 threads = 1024 threads execute coalesced uint4 __ldcs
    // UNROLL=4 copies from host-tier slabs into VRAM scratch `stage_base + s`.
    const u64 tid = static_cast<u64>(blockIdx.x) * static_cast<u64>(blockDim.x) + static_cast<u64>(threadIdx.x);
    const u64 total_threads = static_cast<u64>(gridDim.x) * static_cast<u64>(blockDim.x);
    const unsigned char *t_gate = SEISMIC_PTR(SEISMIC_BUFFER_TIER_GATE);
    const unsigned char *t_up = SEISMIC_PTR(SEISMIC_BUFFER_TIER_UP);
    const unsigned char *t_down = SEISMIC_PTR(SEISMIC_BUFFER_TIER_DOWN);
    unsigned char *e_gate = SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE);
    unsigned char *e_up = SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP);
    unsigned char *e_down = SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN);
    const u64 gate_bytes = (SEISMIC_DIM_GV > 0) ? ELEMENT_MATRIX_BYTES(SEISMIC_TIER_GATE) : 0ull;
    const u64 up_bytes = ELEMENT_MATRIX_BYTES(SEISMIC_TIER_UP);
    const u64 down_bytes = ELEMENT_MATRIX_BYTES(SEISMIC_TIER_DOWN);

    // `expert_gate`/`expert_up`/`expert_down` are bound to the disjoint tail
    // scratch slice `[stage_base .. stage_base + DECODE_STAGE_SLOTS)` so
    // `DagOrder` forks `routed_stage_host` in parallel with `DecodeHits`
    // (which reads `[0 .. stage_base)`). Record index `i` in this slice writes
    // physical slot `stage_base + i`, matching `back_sources` for `DecodeBack`.
    for (unsigned i = 0; i < stage_count; ++i) {
        const u64 src_rec = static_cast<u64>(s_tiers[i]);
        const u64 dst_rec = static_cast<u64>(i);
        if (gate_bytes > 0ull) {
            streaming::copy_rows(slab::region(t_gate, src_rec), 0, slab::region(e_gate, dst_rec), 0, 1, gate_bytes, tid,
                                 total_threads);
        }
        streaming::copy_rows(slab::region(t_up, src_rec), 0, slab::region(e_up, dst_rec), 0, 1, up_bytes, tid,
                             total_threads);
        streaming::copy_rows(slab::region(t_down, src_rec), 0, slab::region(e_down, dst_rec), 0, 1, down_bytes, tid,
                             total_threads);
    }
}
