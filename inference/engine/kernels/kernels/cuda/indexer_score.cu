// indexer_score on CUDA (contract and portable body in indexer.seismic), as
// Metal's (`metal/indexer_score.metal`):
//
// A block of 256 threads per (256 scores, row). Each warp first scans the
// row's spans 32 at a time (`indexer::warp_span_row`) to get the listed row
// count and each lane's key-plane row for ordinal j = 256 x + t. A tile
// holding some of the row's listed rows then prepares the row's H query heads
// into shared memory (padded by 4 floats per head so cross-head lane reads are
// bank-conflict-free), a warp per head (8 heads at a time): widened to F32,
// RMS-normalized with the query norm when NQ = 1, rotated at the row's
// coordinates (`rotary::layout`, exchanging through the head's own shared
// row) and rounded with the indexer's trained rounding, as `indexer_keys`
// prepares a key. Head weights are loaded directly into per-lane registers.
// When all H prepared heads and the 8 warp key buffers fit in 48 KiB of shared
// memory, every head is prepared once and each warp scores its 32 ordinals one
// key at a time (each lane owning ceil(H / 32) heads). Wider shapes tile H in
// power-of-two chunks HEAD_TILE fitting 48 KiB, splitting each head across
// 32 / HEAD_TILE lanes (100% warp utilization) and accumulating across tiles
// before scaling. Every score past the list is -inf.
//
// Dynamic extents, strides and parameters read `seismic_words`.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"
#include "lib/core/head.cuh"
#include "lib/core/rotary.cuh"
#include "lib/core/rounding.cuh"
#include "lib/indexer/indexer.cuh"
#include <seismic/slab.cuh>

namespace score {

typedef unsigned int u32;
typedef unsigned long long u64;
typedef activation::Act Act;

constexpr int W = 2 * SEISMIC_DIM_P + SEISMIC_DIM_S;
constexpr int W4 = W / 4;
constexpr int E = W / 32;
constexpr int H = SEISMIC_DIM_H;
constexpr int Q_STRIDE = W + 4;
constexpr int SMEM_BUDGET_FLOATS = SEISMIC_DEVICE_SHARED_BYTES_PER_GROUP / 4;
constexpr bool SINGLE_TILE = (H * Q_STRIDE + 8 * W) <= SMEM_BUDGET_FLOATS;

__host__ __device__ constexpr int select_head_tile() {
    if (SINGLE_TILE) return H;
    const int max_heads = (SMEM_BUDGET_FLOATS - 8 * W) / Q_STRIDE;
    if (max_heads >= 32) return 32;
    if (max_heads >= 16) return 16;
    if (max_heads >= 8) return 8;
    if (max_heads >= 4) return 4;
    if (max_heads >= 2) return 2;
    return 1;
}

constexpr int HEAD_TILE = select_head_tile();
constexpr int HEADS_PER_LANE = (H + 31) / 32;
constexpr int LANES_PER_HEAD = SINGLE_TILE ? 1 : 32 / HEAD_TILE;

static_assert(W % 32 == 0, "indexer head width must be a multiple of 32");
static_assert(H >= 1 && H <= 64, "indexer head count must be in 1..=64");
static_assert(HEAD_TILE >= 1 && (HEAD_TILE * Q_STRIDE + 8 * W) <= SMEM_BUDGET_FLOATS,
              "indexer_score tile must fit in device shared memory");
static_assert(SINGLE_TILE || (32 % HEAD_TILE == 0), "HEAD_TILE must divide warp size");

typedef ROTARY_ENTRY_LAYOUT(W) query_layout;

} // namespace score

extern "C" __global__ void indexer_score(SEISMIC_KERNEL_PARAMS) {
    using namespace score;
    // Shared memory layout:
    //   prepared[HEAD_TILE][Q_STRIDE]  (Q_STRIDE = W + 4, 16-byte aligned, conflict-free across heads)
    //   k_buf[8][W]                    (per-warp coalesced key staging buffer, 16-byte aligned)
    extern __shared__ float smem[];
    float *prepared = smem;
    float *k_buf = prepared + HEAD_TILE * Q_STRIDE;

    const u64 row = blockIdx.y;
    const u64 n = SEISMIC_DIM_N;
    const u64 r = SEISMIC_DIM_R;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int *row_spans = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SPANS)) + row * SEISMIC_SPANS_STRIDE_0;
    const u64 span_stride = SEISMIC_SPANS_STRIDE_1;
    const u64 bound_stride = SEISMIC_SPANS_STRIDE_2;
    const u64 ordinal = static_cast<u64>(blockIdx.x) * 256 + threadIdx.x;
    // The row's listed rows (those of its spans, at most N scored) and each
    // lane's key-plane row.
    int at = -1;
    u64 listed = indexer::warp_span_row(row_spans, r, span_stride, bound_stride, ordinal, lane, at);
    listed = listed < n ? listed : n;
    const element::Out<element::F32> out = SEISMIC_RESULT_0_OUT.offset(row * SEISMIC_RESULT_0_STRIDE_0);
    const u64 out_stride = SEISMIC_RESULT_0_STRIDE_1;
    const float neg_inf = -__int_as_float(0x7f800000);
    if (static_cast<u64>(blockIdx.x) * 256 >= listed) {
        if (ordinal < n) out.put(ordinal * out_stride, neg_inf);
        return;
    }

    const float epsilon = element::word_f32(SEISMIC_PARAM_EPSILON);
    const rounding::Rounding trained =
        rounding::of(SEISMIC_PARAM_BLOCK, SEISMIC_PARAM_ELEMENT, SEISMIC_PARAM_ROUNDING_SCALE, SEISMIC_PARAM_AMAX_FLOOR);
    const rotary::table table = ROTARY_ENTRY_TABLE(
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES)) + row * SEISMIC_COORDINATES_STRIDE_0,
        SEISMIC_COORDINATES_STRIDE_1);
    const element::u8 *query = SEISMIC_PTR(SEISMIC_BUFFER_QUERY);

    auto prepare_head = [&](int head, float *prepared_head) {
        float x[E];
#pragma unroll
        for (int i = 0; i < E; ++i)
            x[i] = element::at<Act>(query, row * SEISMIC_QUERY_STRIDE_0 + static_cast<u64>(head) * SEISMIC_QUERY_STRIDE_1 +
                                               static_cast<u64>(lane * E + i) * SEISMIC_QUERY_STRIDE_2);
        // A rounded query is rounded from its norm and rotation held in A.
        if constexpr (SEISMIC_DIM_NQ != 0) {
            head::rms_norm(x, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM)),
                           SEISMIC_QUERY_NORM_STRIDE_1, epsilon, lane);
            rounding::hold_lanes<Act>(x, trained);
        }
        // The head's own shared row is its rotation's exchange until it holds
        // the prepared head.
        query_layout::rotate(x, table, prepared_head, lane);
        rounding::hold_lanes<Act>(x, trained);
        rounding::round_lanes<E>(x, trained);
#pragma unroll
        for (int i = 0; i < E; ++i) prepared_head[lane * E + i] = x[i];
    };

    const u64 warp_base = static_cast<u64>(blockIdx.x) * 256 + static_cast<u64>(warp) * 32;
    const int warp_keys = warp_base < listed ? static_cast<int>(listed - warp_base < 32 ? listed - warp_base : 32) : 0;
    const u32 *keys = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_KEYS));
    const float scale = element::word_f32(SEISMIC_PARAM_SCALE);
    float *k_warp = k_buf + warp * W;

    if constexpr (SINGLE_TILE) {
        for (int head = warp; head < H; head += 8) prepare_head(head, prepared + head * Q_STRIDE);
        __syncthreads();

        // Load each lane's head weight(s) directly into registers.
        float w[HEADS_PER_LANE];
#pragma unroll
        for (int p = 0; p < HEADS_PER_LANE; ++p) {
            const int head = p * 32 + lane;
            if constexpr (SEISMIC_DIM_HW != 0) {
                const element::u8 *head_weights = SEISMIC_PTR(SEISMIC_BUFFER_HEAD_WEIGHTS);
                w[p] = head < H
                           ? element::at<Act>(head_weights,
                                              row * SEISMIC_HEAD_WEIGHTS_STRIDE_0 +
                                                  static_cast<u64>(head) * SEISMIC_HEAD_WEIGHTS_STRIDE_1)
                           : 0.0f;
            } else {
                w[p] = 1.0f;
            }
        }

        float my_score = neg_inf;
        for (int j = 0; j < warp_keys; ++j) {
            const int key_row = __shfl_sync(0xffffffffu, at, j);
            const float *k = reinterpret_cast<const float *>(
                slab::row(keys, static_cast<u64>(key_row), SEISMIC_PARAM_SLAB_ROWS, W * sizeof(float)));
#pragma unroll
            for (int c = lane; c < W4; c += 32)
                reinterpret_cast<float4 *>(k_warp)[c] = reinterpret_cast<const float4 *>(k)[c];
            __syncwarp();

            float dot[HEADS_PER_LANE];
#pragma unroll
            for (int p = 0; p < HEADS_PER_LANE; ++p) dot[p] = 0.0f;
            for (int column = 0; column < W; column += 4) {
                const float4 kx = *reinterpret_cast<const float4 *>(k_warp + column);
#pragma unroll
                for (int p = 0; p < HEADS_PER_LANE; ++p) {
                    const int head = p * 32 + lane;
                    if (head < H) {
                        const float4 q = *reinterpret_cast<const float4 *>(prepared + head * Q_STRIDE + column);
                        float d = dot[p];
                        d = __fmaf_rn(q.x, kx.x, d);
                        d = __fmaf_rn(q.y, kx.y, d);
                        d = __fmaf_rn(q.z, kx.z, d);
                        d = __fmaf_rn(q.w, kx.w, d);
                        dot[p] = d;
                    }
                }
            }
            float val = 0.0f;
#pragma unroll
            for (int p = 0; p < HEADS_PER_LANE; ++p) {
                const int head = p * 32 + lane;
                if (head < H) val += fmaxf(dot[p], 0.0f) * w[p];
            }
            const float total = seismic_warp_sum_f32(val);
            if (lane == j) my_score = scale * total;
            __syncwarp();
        }
        if (ordinal < n) out.put(ordinal * out_stride, my_score);
    } else {
        // Head-tiled path: tile H query heads in chunks of HEAD_TILE so shared
        // memory stays <= 48 KiB. Within each tile, each warp splits every head
        // across LANES_PER_HEAD = 32 / HEAD_TILE lanes (100% warp utilization;
        // local_head = lane % HEAD_TILE, sub_lane = lane / HEAD_TILE gives
        // conflict-free prepared reads and broadcast k_warp reads).
        const int local_head = lane & (HEAD_TILE - 1);
        const int sub_lane = lane / HEAD_TILE;
        const float *prepared_local = prepared + local_head * Q_STRIDE;
        float my_score = lane < warp_keys ? 0.0f : neg_inf;

        for (int h0 = 0; h0 < H; h0 += HEAD_TILE) {
            const int tile_heads = (H - h0) < HEAD_TILE ? (H - h0) : HEAD_TILE;
            for (int lh = warp; lh < tile_heads; lh += 8) prepare_head(h0 + lh, prepared + lh * Q_STRIDE);
            __syncthreads();

            const int head = h0 + local_head;
            float w_local = 1.0f;
            if constexpr (SEISMIC_DIM_HW != 0) {
                if (sub_lane == 0 && head < H) {
                    const element::u8 *head_weights = SEISMIC_PTR(SEISMIC_BUFFER_HEAD_WEIGHTS);
                    w_local = element::at<Act>(
                        head_weights,
                        row * SEISMIC_HEAD_WEIGHTS_STRIDE_0 + static_cast<u64>(head) * SEISMIC_HEAD_WEIGHTS_STRIDE_1);
                }
            }

            for (int j = 0; j < warp_keys; ++j) {
                const int key_row = __shfl_sync(0xffffffffu, at, j);
                const float *k = reinterpret_cast<const float *>(
                    slab::row(keys, static_cast<u64>(key_row), SEISMIC_PARAM_SLAB_ROWS, W * sizeof(float)));
#pragma unroll
                for (int c = lane; c < W4; c += 32)
                    reinterpret_cast<float4 *>(k_warp)[c] = reinterpret_cast<const float4 *>(k)[c];
                __syncwarp();

                float dot = 0.0f;
                if (head < H) {
                    for (int c4 = sub_lane; c4 < W4; c4 += LANES_PER_HEAD) {
                        const int column = c4 * 4;
                        const float4 kx = *reinterpret_cast<const float4 *>(k_warp + column);
                        const float4 q = *reinterpret_cast<const float4 *>(prepared_local + column);
                        dot = __fmaf_rn(q.x, kx.x, dot);
                        dot = __fmaf_rn(q.y, kx.y, dot);
                        dot = __fmaf_rn(q.z, kx.z, dot);
                        dot = __fmaf_rn(q.w, kx.w, dot);
                    }
                }
#pragma unroll
                for (int offset = 16; offset >= HEAD_TILE; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffffu, dot, offset);

                const float val = (sub_lane == 0 && head < H) ? fmaxf(dot, 0.0f) * w_local : 0.0f;
                const float tile_sum = seismic_warp_sum_f32(val);
                if (lane == j) my_score += tile_sum;
                __syncwarp();
            }

            if (h0 + HEAD_TILE < H) __syncthreads();
        }

        if (lane < warp_keys) my_score *= scale;
        if (ordinal < n) out.put(ordinal * out_stride, my_score);
    }
}
