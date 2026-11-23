// Grouped combine; the CUDA form of `metal/routed_combine.metal`:
//   stage    the normalized rows' operands: their s8 rows and groups (INT8),
//            or their F16 operand rows (OPERAND);
//   expand   paired GEMM of the shared expert's gate/up over the normalized
//            rows (read in place unless staged) with SiLU . mul into
//            `product` [M, S];
//   quantize each product row's s8 row and groups (INT8);
//   down     shared down GEMM whose epilogue unpermutes the grouped expert
//            outputs in slot order: residual + selected + round_A(shared) * c.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_SHARED_DOWN
#define KERNEL_W1 SEISMIC_SHARED_GATE
#define KERNEL_W2 SEISMIC_SHARED_UP
#include "lib/projection/projection.cuh"

template <unsigned BM> struct CombineShape;
template <> struct CombineShape<32> { using type = projection::GemmShape<32, 2, 4, 3>; };
template <> struct CombineShape<64> { using type = projection::GemmShape<64, 2, 4, 3>; };
template <> struct CombineShape<128> { using type = projection::GemmShape<128, 2, 4, 2>; };
using activation::Act;
// Whether both GEMMs run on the int8 path (INT8 = 1, which the declaration
// admits where every weight is admitted to it).
#define GROUPED_S8                                                                                         \
    (projection::int8_path<INT8, PACKETS_FORM_INT8(KERNEL_W0), PACKETS_FORM_INT8(KERNEL_W1),               \
                           PACKETS_FORM_INT8(KERNEL_W2)>())
// The order of the int8 rows each GEMM reads (`projection::gemm_s8_entries`),
// as the stage and quantize launches form them: the expansion's of the gate
// and up weights, the down GEMM's of the down weight.
#define EXPAND_ENTRIES (projection::gemm_s8_entries<GROUPED_S8, packets::W1, packets::W2>())
#define DOWN_ENTRIES (projection::gemm_s8_entries<GROUPED_S8, packets::W0>())

// Exact unsigned 32-bit divmod by a runtime-invariant divisor >= 1:
// shift/mask for powers of two (`magic == 0`), Granlund-Montgomery `__umulhi`
// reciprocal otherwise.
struct FastDiv32 {
    projection::u32 d;
    projection::u32 magic;
    projection::u32 shift;

    __device__ __forceinline__ static FastDiv32 make(projection::u32 divisor) {
        const projection::u32 s = 31u - (projection::u32)__clz((int)divisor);
        if ((divisor & (divisor - 1u)) == 0u)
            return FastDiv32{divisor, 0u, s};
        const projection::u64 rem = ((projection::u64)1 << (s + 1u)) - (projection::u64)divisor;
        const projection::u32 m = (projection::u32)((rem << 32) / (projection::u64)divisor) + 1u;
        return FastDiv32{divisor, m, s};
    }

    __device__ __forceinline__ void divmod(projection::u32 n, projection::u32 &q, projection::u32 &r) const {
        if (magic == 0u) {
            q = n >> shift;
            r = n & (d - 1u);
        } else {
            const projection::u32 t = __umulhi(n, magic);
            q = (t + ((n - t) >> 1u)) >> shift;
            r = n - q * d;
        }
    }
};

struct CombineEpi {
    element::Out<element::F32> out;
    projection::u64 out_stride;
    const float *residual;
    projection::u64 residual_stride;
    const projection::u8 *__restrict__ expert_output;
    FastDiv32 tile_div;
    projection::u64 block_stride;
    projection::u64 lane_stride;
    const int *__restrict__ inverse;
    projection::u64 inverse_stride;
    const float *__restrict__ scores;
    projection::u64 scores_stride;
    const float *__restrict__ coefficient;
    const float *__restrict__ inject;
    projection::u64 inject_stride;
    __device__ __forceinline__ void operator()(unsigned m, projection::u64 n, float value, float) const {
        const int *__restrict__ inv_row = inverse + m * inverse_stride;
        const float *__restrict__ score_row = scores + m * scores_stride;
        const float coeff = __ldg(&coefficient[m]);
        float selected = 0.0f;
        for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k) {
            const projection::u32 position = (projection::u32)__ldg(&inv_row[k]);
            projection::u32 bucket, lane;
            tile_div.divmod(position, bucket, lane);
            const float projected = element::at<Act>(
                expert_output, (projection::u64)bucket * block_stride + (projection::u64)lane * lane_stride + n);
            selected = __fmaf_rn(__ldg(&score_row[k]), projected, selected);
        }
        projection::routed_join<SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>(
            out.offset(m * out_stride), residual + m * residual_stride, inject + m * inject_stride, SEISMIC_DIM_H, n,
            selected, Act::round(value) * coeff);
    }
    __device__ __forceinline__ void pair(projection::u32 m, projection::u64 n, float a0, float a1, float, float) const {
        const int *__restrict__ inv_row = inverse + m * inverse_stride;
        const float *__restrict__ score_row = scores + m * scores_stride;
        const float coeff = __ldg(&coefficient[m]);
        float selected0 = 0.0f;
        float selected1 = 0.0f;
        for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k) {
            const projection::u32 position = (projection::u32)__ldg(&inv_row[k]);
            projection::u32 bucket, lane;
            tile_div.divmod(position, bucket, lane);
            const projection::u64 base = (projection::u64)bucket * block_stride + (projection::u64)lane * lane_stride + n;
            const projection::u8 *address = expert_output + base * Act::bytes;
            const float2 projected = ((reinterpret_cast<projection::u64>(address) & 3u) == 0u)
                                         ? Act::unpack2(*reinterpret_cast<const projection::u32 *>(address))
                                         : make_float2(element::at<Act>(expert_output, base),
                                                       element::at<Act>(expert_output, base + 1));
            const float score = __ldg(&score_row[k]);
            selected0 = __fmaf_rn(score, projected.x, selected0);
            selected1 = __fmaf_rn(score, projected.y, selected1);
        }
        projection::routed_join_pair<SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>(
            out.offset(m * out_stride), residual + m * residual_stride, inject + m * inject_stride, SEISMIC_DIM_H, n,
            selected0, selected1, Act::round(a0) * coeff, Act::round(a1) * coeff);
    }
};

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void routed_combine_stage(SEISMIC_KERNEL_PARAMS) {
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    const projection::Plain<Act, projection::AllRows> source{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED), SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}};
    if constexpr (GROUPED_S8)
        projection::stage_row_s8<EXPAND_ENTRIES>(
            source, blockIdx.x, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED),
                                 reinterpret_cast<float2 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)));
    else
        projection::materialize_operand_row<Operands>(source, blockIdx.x, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED));
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_EXPAND
template <unsigned INT8, unsigned OPERAND, unsigned BM, unsigned SPLIT>
__global__ void routed_combine_expand(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = GROUPED_S8;
    constexpr bool STAGED = S8 || OPERAND == 1;
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    using Shape = typename CombineShape<BM>::type;
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_S))
        return;
    if constexpr (SPLIT > 1) {
        const projection::PartialStore<2> epi{
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_EXPAND_PARTIALS)), SEISMIC_DIM_M,
            SEISMIC_DIM_S, 0, blockIdx.z};
        projection::gemm_run<Shape, S8, EXPAND_ENTRIES, Operands>(
            reinterpret_cast<projection::u8 *>(dynamic_shared),
            STAGED ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED) : SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED),
            STAGED ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS),
            (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H, blockIdx.x, SEISMIC_DIM_S,
            KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE)),
            KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP)), epi, blockIdx.z, SPLIT);
    } else {
        const projection::Glu<typename Operands::Storage, Act> epi{
            element::out_over_scratch<typename Operands::Storage>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT)),
            SEISMIC_DIM_S, 0};
        projection::gemm_run<Shape, S8, EXPAND_ENTRIES, Operands>(
            reinterpret_cast<projection::u8 *>(dynamic_shared),
            STAGED ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED) : SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED),
            STAGED ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS),
            (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H, blockIdx.x, SEISMIC_DIM_S,
            KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE)),
            KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP)), epi);
    }
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_EXPAND_FINALIZE
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT>
__global__ void routed_combine_expand_finalize(SEISMIC_KERNEL_PARAMS) {
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    const projection::Glu<typename Operands::Storage, Act> epi{
        element::out_over_scratch<typename Operands::Storage>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT)),
        SEISMIC_DIM_S, 0};
    projection::split_finalize<2>(reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_EXPAND_PARTIALS)),
                                  SPLIT, SEISMIC_DIM_M, SEISMIC_DIM_S, epi);
}
#endif

// Each product row's s8 row and groups, the int8 down GEMM's operands (a
// launch of INT8 = 1 alone, 8 rows per 256-thread block, 1 warp/row).
#ifdef SEISMIC_FORMING_ROUTED_COMBINE_QUANTIZE
template <unsigned INT8, unsigned OPERAND>
__global__ void routed_combine_quantize(SEISMIC_KERNEL_PARAMS) {
    const projection::u64 row = (projection::u64)blockIdx.x * 8u + (threadIdx.x >> 5);
    if (row >= SEISMIC_DIM_M)
        return;
    projection::stage_row_s8_warp<DOWN_ENTRIES>(
        projection::Plain<Act, projection::AllRows>{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT), SEISMIC_DIM_S,
                                                    projection::AllRows{}},
        (projection::u32)row, SEISMIC_DIM_S, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_CODES),
        reinterpret_cast<float2 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_GROUPS)));
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN
template <unsigned INT8, unsigned OPERAND, unsigned BM>
__global__ void routed_combine_down(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = GROUPED_S8;
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    using Shape = typename CombineShape<BM>::type;
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_H))
        return;
    const CombineEpi epi{SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,
                         SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_OUTPUT), FastDiv32::make((projection::u32)SEISMIC_DIM_T),
                         SEISMIC_EXPERT_OUTPUT_STRIDE_0, SEISMIC_EXPERT_OUTPUT_STRIDE_1,
                         reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE)), SEISMIC_INVERSE_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)), SEISMIC_SCORES_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT)),
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INJECT)), SEISMIC_INJECT_STRIDE_0};
    projection::gemm_run<Shape, S8, DOWN_ENTRIES, Operands>(reinterpret_cast<projection::u8 *>(dynamic_shared),
                             S8 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_CODES) : SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT),
                             SEISMIC_DIM_S, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_GROUPS),
                             (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_S, blockIdx.x, SEISMIC_DIM_H,
                             KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)), projection::NoWeight{}, epi);
}
#endif
