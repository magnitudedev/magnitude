// Decode output (M <= 8); the CUDA form of `metal/routed_output.metal`.
// Block (x, y) works on the output channels of tile group x for y =
// m * (K + 1) + slot (slot < K is choice (m, slot), slot K the shared
// expert) and publishes each projection A-rounded into scratch. The last
// block of a channel tile to arrive combines its channels: each published
// choice projection weighted by its score in slot order, then residual +
// selected + shared * c, the arithmetic of one block carrying every slot in
// turn. Running the projections side by side keeps enough weight streams in
// flight to reach bandwidth.
// - `routed_output` (one row): block (x, slot) projects slot's down
//   projection; the last of the K + 1 blocks of tile x combines.
// - `routed_output_gathered` (several rows): choice (m, slot) projects only
//   when it is its expert's first choice, the down projection of every
//   choice of that expert (`routed::expert_choices`), so each chosen expert
//   streams once; slot K of row 0 projects the shared expert for every row.
//   The last of tile x's M * (K + 1) blocks combines every row.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#define KERNEL_W1 SEISMIC_SHARED_DOWN
#include "lib/routed/routed.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;

struct PublishEpi {
    float *published;
    __device__ __forceinline__ void operator()(unsigned, projection::u64 n, float value, float) const {
        published[n] = element::Act::round(value);
    }
};

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT
template <int TPW, int KSPLIT>
__global__ void routed_output(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    constexpr projection::u64 CHANNELS = Shape::GROUPS * TPW * 16;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    __shared__ unsigned last;
    const projection::u64 H = SEISMIC_DIM_H, K = SEISMIC_DIM_K;
    const projection::u64 m = blockIdx.y / (K + 1), slot = blockIdx.y % (K + 1);
    const projection::u64 group = Shape::tile_group();
    float *published = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED)) + m * (K + 1) * H;
    unsigned *arrivals = reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS));

    if (group < projection::gemv_groups<Shape>(H)) {
        const PublishEpi epi{published + slot * H};
        if (slot < K) {
            const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
            const projection::u64 expert =
                (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1];
            const Pro pro{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT)
                              + (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + slot * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * 2,
                          0, projection::AllRows{}};
            projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(pro), 1u, SEISMIC_DIM_F / 64, group, H,
                                            KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), expert),
                                            projection::NoWeight{}, epi);
        } else {
            const Pro pro{SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT) + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * 2, 0,
                          projection::AllRows{}};
            projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(pro), 1u, SEISMIC_DIM_S / 64, group, H,
                                            KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)),
                                            projection::NoWeight{}, epi);
        }
    } else {
        // Wait before arriving: the arrival and the combine follow the
        // launch before.
        seismic_dependency_start();
    }

    // Arrive (sync scratch: zero when the launch starts, restored by the last).
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned *counter = arrivals + m * gridDim.x + blockIdx.x;
        const bool is_last = atomicAdd(counter, 1u) == K;
        if (is_last) {
            __threadfence();
            atomicExch(counter, 0u);
        }
        last = is_last;
    }
    __syncthreads();
    if (!last)
        return;

    const float *scores = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));
    const float *residual =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)) + m * SEISMIC_RESIDUAL_STRIDE_0;
    float *out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + m * SEISMIC_RESULT_0_STRIDE_0;
    const float coefficient =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT))[m * SEISMIC_COEFFICIENT_STRIDE_0];
    const projection::u64 first = blockIdx.x * CHANNELS;
    for (projection::u64 n = first + threadIdx.x; n < first + CHANNELS && n < H; n += blockDim.x) {
        float selected = 0.0f;
        for (projection::u64 k = 0; k < K; ++k)
            selected = __fmaf_rn(scores[m * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1],
                                 __ldcg(published + k * H + n), selected);
        out[n] = residual[n] + selected + __ldcg(published + K * H + n) * coefficient;
    }
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_GATHERED
// Each GEMV row's projection A-rounded at its published offset.
struct GatheredPublishEpi {
    float *published;
    const projection::u64 *offsets;
    __device__ __forceinline__ void operator()(unsigned m, projection::u64 n, float value, float) const {
        published[offsets[m] + n] = element::Act::round(value);
    }
};

template <int TPW, int KSPLIT>
__global__ void routed_output_gathered(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    constexpr projection::u64 CHANNELS = Shape::GROUPS * TPW * 16;
    __shared__ projection::GemvShared<Shape, routed::GatheredRows> shared;
    __shared__ projection::u64 rows_at[8], published_at[8];
    __shared__ unsigned count, last;
    const projection::u64 H = SEISMIC_DIM_H, K = SEISMIC_DIM_K, M = SEISMIC_DIM_M;
    const projection::u64 m = blockIdx.y / (K + 1), slot = blockIdx.y % (K + 1);
    const projection::u64 group = Shape::tile_group();
    float *published = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED));
    unsigned *arrivals = reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS));
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));

    // The routes come from the launch before the expansion, which has
    // completed (the expansion lets this launch start only after its own
    // wait), so every block reads them and issues its first weight loads
    // before waiting.
    const unsigned choice = (unsigned)(m * K + slot);
    const bool projects = slot < K
        ? routed::first_choice(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, (unsigned)K, choice)
        : m == 0;
    if (projects && threadIdx.x == 0) {
        if (slot < K) {
            projection::u32 chosen[8];
            count = routed::expert_choices(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, (unsigned)M,
                                           (unsigned)K, choice, chosen);
            for (unsigned r = 0; r < count; ++r) {
                const projection::u64 row = chosen[r] / K, choice_slot = chosen[r] % K;
                rows_at[r] = row * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + choice_slot * SEISMIC_EXPERT_PRODUCT_STRIDE_1;
                published_at[r] = (row * (K + 1) + choice_slot) * H;
            }
        } else {
            count = (unsigned)M;
            for (unsigned r = 0; r < count; ++r) {
                rows_at[r] = r * SEISMIC_SHARED_PRODUCT_STRIDE_0;
                published_at[r] = (r * (K + 1) + K) * H;
            }
        }
    }
    __syncthreads();
    if (projects) {
        const GatheredPublishEpi epi{published, published_at};
        if (slot < K) {
            const projection::u64 expert = (projection::u64)routed::route(
                routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, (unsigned)K, choice);
            const routed::GatheredRows rows{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT), rows_at};
            projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(rows), count,
                                                  SEISMIC_DIM_F / 64, group, H,
                                                  KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), expert),
                                                  projection::NoWeight{}, epi);
        } else {
            const routed::GatheredRows rows{SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT), rows_at};
            projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(rows), count,
                                                  SEISMIC_DIM_S / 64, group, H,
                                                  KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)),
                                                  projection::NoWeight{}, epi);
        }
    } else {
        seismic_dependency_start();
    }

    // Arrive (sync scratch: zero when the launch starts, restored by the last).
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned *counter = arrivals + blockIdx.x;
        const bool is_last = atomicAdd(counter, 1u) == gridDim.y - 1;
        if (is_last) {
            __threadfence();
            atomicExch(counter, 0u);
        }
        last = is_last;
    }
    __syncthreads();
    if (!last)
        return;

    const float *scores = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));
    const float *coefficients = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT));
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    float *out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const projection::u64 first = blockIdx.x * CHANNELS;
    for (projection::u64 item = threadIdx.x; item < M * CHANNELS; item += blockDim.x) {
        const projection::u64 row = item / CHANNELS, n = first + item % CHANNELS;
        if (n >= H)
            continue;
        const float *row_published = published + row * (K + 1) * H;
        float selected = 0.0f;
        for (projection::u64 k = 0; k < K; ++k)
            selected = __fmaf_rn(scores[row * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1],
                                 __ldcg(row_published + k * H + n), selected);
        out[row * SEISMIC_RESULT_0_STRIDE_0 + n] =
            residual[row * SEISMIC_RESIDUAL_STRIDE_0 + n] + selected
            + __ldcg(row_published + K * H + n) * coefficients[row * SEISMIC_COEFFICIENT_STRIDE_0];
    }
}
#endif
