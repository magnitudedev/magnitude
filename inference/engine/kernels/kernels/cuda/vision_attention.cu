// vision_attention: the prepare launch writes each row's queries and keys
// (head-normalized with NQ, rotated) and its values (normalized with NV) into
// the [M, 3, H, WP] operand rows (W = 4P, WP = W rounded up to 16, the pad
// zero); the attend launch runs 64 query rows of one head per block over the
// keys of their spans. Bodies in lib/vision/vision.cuh.
#include "lib/vision/vision.cuh"

using vision::u32;
using vision::u64;
using vision::u8;
typedef element::Act A;

constexpr u32 W = 4 * SEISMIC_DIM_P;
constexpr u32 WP = (W + 15) / 16 * 16;
#define OPERANDS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS)

// One warp per (row, part, head): queries and keys (normalized with NQ)
// rotated, values normalized with NV (weightless) or copied.
extern "C" __global__ void vision_attention_prepare(SEISMIC_KERNEL_PARAMS) {
    const u64 item = (u64)blockIdx.x * 8 + threadIdx.x / 32;
    const u64 row = item / (3 * SEISMIC_DIM_H);
    if (row >= SEISMIC_DIM_M)
        return;
    const u64 part = item % (3 * SEISMIC_DIM_H) / SEISMIC_DIM_H;
    const u64 head = item % SEISMIC_DIM_H;
    const u8 *source = part == 0   ? SEISMIC_PTR(SEISMIC_BUFFER_QUERY)
                       : part == 1 ? SEISMIC_PTR(SEISMIC_BUFFER_KEY)
                                   : SEISMIC_PTR(SEISMIC_BUFFER_VALUE);
    const float *norm = part == 0   ? reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM))
                        : part == 1 ? reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM))
                                    : nullptr;
    const bool normed = part < 2 ? SEISMIC_DIM_NQ == 1 : SEISMIC_DIM_NV == 1;
    const int *coordinates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES));
    vision::prepare_head<A, W, WP>(source + (row * SEISMIC_DIM_H + head) * W * A::bytes,
                                   OPERANDS + ((row * 3 + part) * SEISMIC_DIM_H + head) * WP * A::bytes, normed, norm,
                                   __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), part < 2, coordinates + row * 2,
                                   __uint_as_float((unsigned)SEISMIC_PARAM_LOG_BASE), threadIdx.x % 32);
}

extern "C" __global__ void vision_attention_attend(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::AttendShared<WP> shared;
    const float scale = (int)SEISMIC_PARAM_UNIT_SCALE != 0 ? 1.0f : rsqrtf(float(W));
    const int *spans =
        SEISMIC_DIM_WS == 1 ? reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SPANS)) : nullptr;
    vision::attend<A, W, WP>(shared, OPERANDS, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), (u32)SEISMIC_DIM_M, SEISMIC_DIM_H,
                             scale, spans);
}
