// Routing for M rows; the CUDA form of `metal/qwen_routed_route.metal`, in
// four launches: `qwen_routed_normalize` (M > 8, one block per row) publishes
// the normalized rows once; `qwen_routed_logits_gemv` (M <= 8, one warp per
// logit column, normalizing on the fly and publishing the normalized rows) or
// `qwen_routed_logits_gemm` (32 rows x 32 columns per block over shared
// tiles) forms the [M, E + 1] logits, column E being the shared-expert gate;
// `qwen_routed_select` (one warp per row) forms the softmax, selects the K
// winners by K rounds of warp argmax (ties to the higher expert index) and
// computes the shared-expert coefficient from column E. Every logit accumulates over the hidden axis in the same order
// for every expert, so experts with equal router rows tie exactly.

__device__ __forceinline__ float routed_load_norm(const unsigned char *base, unsigned long long logical) {
#if defined(SEISMIC_NORM_REPRESENTATION_F32)
    return reinterpret_cast<const float *>(base)[logical];
#elif defined(SEISMIC_NORM_REPRESENTATION_F16)
    return seismic_f16_to_f32(reinterpret_cast<const unsigned short *>(base)[logical]);
#elif defined(SEISMIC_NORM_REPRESENTATION_BF16)
    return seismic_bf16_to_f32(reinterpret_cast<const unsigned short *>(base)[logical]);
#else
#error "routed norm must be dense"
#endif
}

__device__ __forceinline__ float routed_load_router(const unsigned char *base, unsigned long long logical) {
#if defined(SEISMIC_ROUTER_REPRESENTATION_F32)
    return reinterpret_cast<const float *>(base)[logical];
#elif defined(SEISMIC_ROUTER_REPRESENTATION_F16)
    return seismic_f16_to_f32(reinterpret_cast<const unsigned short *>(base)[logical]);
#elif defined(SEISMIC_ROUTER_REPRESENTATION_BF16)
    return seismic_bf16_to_f32(reinterpret_cast<const unsigned short *>(base)[logical]);
#else
#error "routed router must be dense"
#endif
}

__device__ __forceinline__ float routed_round_activation(float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return seismic_f16_to_f32(seismic_f32_to_f16(value));
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    return seismic_bf16_to_f32(seismic_f32_to_bf16(value));
#else
#error "routed activation must be dense"
#endif
}

__device__ __forceinline__ float routed_load_activation(const unsigned char *base, unsigned long long logical) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return reinterpret_cast<const float *>(base)[logical];
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return seismic_f16_to_f32(reinterpret_cast<const unsigned short *>(base)[logical]);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    return seismic_bf16_to_f32(reinterpret_cast<const unsigned short *>(base)[logical]);
#endif
}

// Stores a value already rounded to the activation dtype.
__device__ __forceinline__ void routed_store_activation(unsigned char *base, unsigned long long logical, float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    reinterpret_cast<float *>(base)[logical] = value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    reinterpret_cast<unsigned short *>(base)[logical] = seismic_f32_to_f16(value);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    reinterpret_cast<unsigned short *>(base)[logical] = (unsigned short)(__float_as_uint(value) >> 16);
#endif
}

constexpr unsigned ROUTE_THREADS = 256;
constexpr unsigned ROUTE_WARPS = ROUTE_THREADS / 32;

extern "C" __global__ void qwen_routed_normalize(SEISMIC_KERNEL_PARAMS) {
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const unsigned char *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    __shared__ float partials[ROUTE_WARPS];
    const unsigned thread = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const float eps = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPS));
    float squares = 0.0f;
    for (unsigned source = thread; source < SEISMIC_DIM_H; source += ROUTE_THREADS) {
        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        squares = seismic_fma_rn(value, value, squares);
    }
    squares = seismic_warp_sum_f32(squares);
    if (thread % 32 == 0)
        partials[thread / 32] = squares;
    __syncthreads();
    float total = 0.0f;
    for (unsigned index = 0; index < ROUTE_WARPS; ++index)
        total = seismic_add_rn(total, partials[index]);
    const float inverse = rsqrtf(seismic_add_rn(total / static_cast<float>(SEISMIC_DIM_H), eps));
    for (unsigned source = thread; source < SEISMIC_DIM_H; source += ROUTE_THREADS) {
        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        routed_store_activation(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
            routed_round_activation(seismic_mul_rn(seismic_mul_rn(value, inverse),
                routed_load_norm(norm, source * SEISMIC_NORM_STRIDE_0))));
    }
}

// The router logits scratch is [M, E + 1]: column E holds the shared-expert
// gate logit (normalized . shared_router), formed like an expert's logit.
__device__ __forceinline__ unsigned long long routed_logit_index(unsigned long long row, unsigned long long expert) {
    return row * (SEISMIC_DIM_E + 1) + expert;
}

// The weight of logit column `expert` at hidden coordinate `source`.
__device__ __forceinline__ float routed_column_weight(const unsigned char *router, const float *shared_router,
    unsigned long long expert, unsigned long long source) {
    return expert < SEISMIC_DIM_E
        ? routed_load_router(router, expert * SEISMIC_ROUTER_STRIDE_0 + source * SEISMIC_ROUTER_STRIDE_1)
        : shared_router[source * SEISMIC_SHARED_ROUTER_STRIDE_0];
}

// M <= 8, one launch before the selection: the block computes its rows' RMS
// inverses (in the normalize launch's order), then each warp forms one logit
// column, normalizing each element as it reads it; block 0 publishes the
// normalized rows.
extern "C" __global__ void qwen_routed_logits_gemv(SEISMIC_KERNEL_PARAMS) {
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const unsigned char *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const unsigned char *router = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER);
    const float *shared_router = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER));
    unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    __shared__ float parts[8][ROUTE_WARPS];
    __shared__ float inverses[8];
    const unsigned thread = threadIdx.x, lane = thread % 32, warp = thread / 32;
    const unsigned rows = static_cast<unsigned>(SEISMIC_DIM_M);
    const float eps = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPS));
    // The normalize launch's sum order (256 strided partials, warp sums, then
    // the eight warp partials in order): item (row, part) is part `part`'s
    // warp sum, and the items spread over every warp.
    for (unsigned item = warp; item < rows * ROUTE_WARPS; item += SEISMIC_TUNE_SIMDGROUPS) {
        const unsigned row = item / ROUTE_WARPS, part = item % ROUTE_WARPS;
        float squares = 0.0f;
        for (unsigned source = part * 32 + lane; source < SEISMIC_DIM_H; source += ROUTE_THREADS) {
            const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
            squares = seismic_fma_rn(value, value, squares);
        }
        squares = seismic_warp_sum_f32(squares);
        if (lane == 0)
            parts[row][part] = squares;
    }
    __syncthreads();
    if (thread < rows) {
        float total = 0.0f;
        for (unsigned part = 0; part < ROUTE_WARPS; ++part)
            total = seismic_add_rn(total, parts[thread][part]);
        inverses[thread] = rsqrtf(seismic_add_rn(total / static_cast<float>(SEISMIC_DIM_H), eps));
    }
    __syncthreads();
    if (blockIdx.x == 0) {
        for (unsigned item = thread; item < rows * static_cast<unsigned>(SEISMIC_DIM_H); item += 32 * SEISMIC_TUNE_SIMDGROUPS) {
            const unsigned long long row = item / SEISMIC_DIM_H, source = item % SEISMIC_DIM_H;
            const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
            routed_store_activation(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
                routed_round_activation(seismic_mul_rn(seismic_mul_rn(value, inverses[row]),
                    routed_load_norm(norm, source * SEISMIC_NORM_STRIDE_0))));
        }
    }
    const unsigned long long expert = static_cast<unsigned long long>(blockIdx.x) * SEISMIC_TUNE_SIMDGROUPS + warp;
    if (expert > SEISMIC_DIM_E)
        return;
    // Lane l owns sources 4l + 128 i .. + 3 (four independent loads per
    // step), accumulated in source order within the lane.
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned base = 4 * lane; base < SEISMIC_DIM_H; base += 128) {
        float weight[4], scale[4];
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            weight[j] = routed_column_weight(router, shared_router, expert, base + j);
            scale[j] = routed_load_norm(norm, (base + j) * SEISMIC_NORM_STRIDE_0);
        }
#pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            if (row < rows) {
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + (base + j) * SEISMIC_RESIDUAL_STRIDE_1];
                    sums[row] = seismic_fma_rn(routed_round_activation(seismic_mul_rn(seismic_mul_rn(value, inverses[row]),
                        scale[j])), weight[j], sums[row]);
                }
            }
        }
    }
#pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        const float sum = seismic_warp_sum_f32(sums[row]);
        if (row < rows && lane == 0)
            logits[routed_logit_index(row, expert)] = sum;
    }
}

constexpr unsigned ROUTE_TILE = 32;

extern "C" __global__ void qwen_routed_logits_gemm(SEISMIC_KERNEL_PARAMS) {
    const unsigned char *router = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER);
    const float *shared_router = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER));
    const unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    __shared__ float rows_tile[ROUTE_TILE][ROUTE_TILE + 1];
    __shared__ float experts_tile[ROUTE_TILE][ROUTE_TILE + 1];
    const unsigned thread = threadIdx.x;
    const unsigned long long expert0 = static_cast<unsigned long long>(blockIdx.x) * ROUTE_TILE;
    const unsigned long long row0 = static_cast<unsigned long long>(blockIdx.y) * ROUTE_TILE;
    // Thread t owns row t / 8 and experts 4 (t % 8) .. 4 (t % 8) + 3.
    const unsigned r = thread / 8, e0 = 4 * (thread % 8);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned long long k0 = 0; k0 < SEISMIC_DIM_H; k0 += ROUTE_TILE) {
        __syncthreads();
        for (unsigned item = thread; item < ROUTE_TILE * ROUTE_TILE; item += ROUTE_THREADS) {
            const unsigned i = item / ROUTE_TILE, k = item % ROUTE_TILE;
            const unsigned long long row = row0 + i, expert = expert0 + i;
            rows_tile[i][k] = row < SEISMIC_DIM_M
                ? routed_load_activation(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + (k0 + k) * SEISMIC_RESULT_0_STRIDE_1)
                : 0.0f;
            experts_tile[i][k] = expert <= SEISMIC_DIM_E
                ? routed_column_weight(router, shared_router, expert, k0 + k)
                : 0.0f;
        }
        __syncthreads();
#pragma unroll 8
        for (unsigned k = 0; k < ROUTE_TILE; ++k) {
            const float x = rows_tile[r][k];
#pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                acc[j] = seismic_fma_rn(x, experts_tile[e0 + j][k], acc[j]);
        }
    }
    const unsigned long long row = row0 + r;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const unsigned long long expert = expert0 + e0 + j;
        if (row < SEISMIC_DIM_M && expert <= SEISMIC_DIM_E)
            logits[routed_logit_index(row, expert)] = acc[j];
    }
}

// Descending probability, ties to the higher expert index.
__device__ __forceinline__ bool routed_precedes(float probability, int expert, float best, int best_expert) {
    return probability > best || (probability == best && expert > best_expert);
}

extern "C" __global__ void qwen_routed_select(SEISMIC_KERNEL_PARAMS) {
    int *routes = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    float *scores = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));
    float *coefficient = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER));
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    constexpr unsigned PER_LANE = SEISMIC_DIM_E / 32;
    const unsigned lane = threadIdx.x % 32;
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * ROUTE_WARPS + threadIdx.x / 32;
    if (row >= SEISMIC_DIM_M)
        return;
    const float negative_infinity = -__int_as_float(0x7f800000);
    const float *values = logits + routed_logit_index(row, 0);
    // Lane `lane` holds experts lane + 32 i.
    float probability[PER_LANE];
    float maximum = negative_infinity;
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i) {
        probability[i] = values[lane + 32 * i];
        maximum = fmaxf(maximum, probability[i]);
    }
    maximum = seismic_warp_max_f32(maximum);
    float total = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i) {
        probability[i] = expf(probability[i] - maximum);
        total = seismic_add_rn(total, probability[i]);
    }
    total = seismic_warp_sum_f32(total);
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i)
        probability[i] = probability[i] / total;

    float chosen[SEISMIC_DIM_K];
    for (unsigned rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = negative_infinity;
        int best_expert = -1;
#pragma unroll
        for (unsigned i = 0; i < PER_LANE; ++i) {
            const int expert = static_cast<int>(lane + 32 * i);
            if (routed_precedes(probability[i], expert, best, best_expert)) {
                best = probability[i];
                best_expert = expert;
            }
        }
        const float winner_probability = seismic_warp_max_f32(best);
        const int winner = seismic_redux_max_s32(best == winner_probability ? best_expert : -1);
#pragma unroll
        for (unsigned i = 0; i < PER_LANE; ++i)
            if (winner >= 0 && static_cast<unsigned>(winner) == lane + 32 * i)
                probability[i] = negative_infinity;
        chosen[rank] = winner_probability;
        if (lane == 0) {
            const unsigned long long slot = SEISMIC_DIM_K - 1 - rank;
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = winner;
            scores[row * SEISMIC_SCORES_STRIDE_0 + slot * SEISMIC_SCORES_STRIDE_1] = winner_probability;
        }
    }
    if (SEISMIC_PARAM_NORMALIZE != 0) {
        // The slot-order sum: slot s holds rank K - 1 - s.
        float denominator = 0.0f;
        for (unsigned slot = 0; slot < SEISMIC_DIM_K; ++slot)
            denominator = seismic_add_rn(denominator, chosen[SEISMIC_DIM_K - 1 - slot]);
        if (lane == 0)
            for (unsigned slot = 0; slot < SEISMIC_DIM_K; ++slot)
                scores[row * SEISMIC_SCORES_STRIDE_0 + slot * SEISMIC_SCORES_STRIDE_1] =
                    chosen[SEISMIC_DIM_K - 1 - slot] / denominator;
    }

    if (lane == 0)
        coefficient[row * SEISMIC_RESULT_1_STRIDE_0] = 1.0f / (1.0f + expf(-values[SEISMIC_DIM_E]));
}
