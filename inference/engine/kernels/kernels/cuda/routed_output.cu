// Decode output (M <= 8); the CUDA form of `metal/routed_output.metal`.
// Block (x, y) projects one of row m's K + 1 down projections (y = m * (K + 1)
// + slot: slot < K is choice slot, slot K the shared expert) over the output
// channels of tile group x, and publishes each channel A-rounded into
// scratch. The last of a (x, m)'s K + 1 blocks to arrive combines its
// channels: each published choice projection weighted by its score in slot
// order, then residual + selected + shared * c, the arithmetic of one block
// carrying every slot in turn. Running the slots side by side keeps enough
// weight streams in flight to reach bandwidth.
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
