// The shared expert's expansion in a decode route's launch, shared by the
// routes that also expand it (`routed_route_shared`, `routed_route_streamed`);
// the CUDA form of `metal/lib/routed/shared.h`. Both bind
// `routed_route_shared`'s shared-expert ports (`shared_gate`, `shared_up`,
// the product `shared_product` as result 2, of static extent S) and its
// SIMDGROUPS, TPW and KSPLIT. The includer defines KERNEL_W1 and KERNEL_W2
// as the shared gate and up weights and includes `route.cuh` first. Apart
// from `route.cuh`, which `routed_route` also includes and which so names no
// symbol of these ports.
//
// A shared block forms the decode rows in its shared memory as a router
// block forms them (`route::decode_inverses`, `route::Input`: the published
// `normalized` bits; where the shared weights take NVFP4 operands, those
// rows' operands as `routed_expand`'s stage launch forms them), then runs
// `routed_expand`'s shared row: the swap-AB paired GEMV over the shared
// expert's gate and up rows with the SiLU . mul epilogue, SIMDGROUPS /
// KSPLIT tile groups of TPW tiles a block, KSPLIT splitting K across a
// group's warps. A row's result depends only on TPW and KSPLIT, never on the
// warps a block has, so for equal KSPLIT the shared product carries
// `routed_expand`'s bits.

#include "../projection/projection.cuh"

// The shared rows run `routed_expand`'s paired GEMV, which reads 16-bit
// activations in place.
static_assert(element::Act::bytes == 2, "a route that expands the shared expert requires a bf16 or f16 activation");

namespace shared_expert {

using Shape = projection::GemvShape<SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_TPW, SEISMIC_TUNE_KSPLIT, 1>;
// The source rows of the GEMV: A values (or NVFP4 operands), read in place.
using Rows = projection::Plain<projection::Act, projection::AllRows>;

// The rows the shared gate/up weights take: the normalized rows, or NVFP4
// operands.
constexpr bool NVFP4 = projection::OperandRecipe<packets::W1_OPERANDS_NVFP4, packets::W2_OPERANDS_NVFP4>::NVFP4;

// The decode rows `in` (their RMS `inverses`) into `out` (M x H A
// elements), each with NVFP4 formed in place as `form_row` forms a staged
// row. Every thread of the block calls it; the rows are the block's when it
// returns.
__device__ __forceinline__ void rows(const route::Input &in, const float *inverses, projection::u8 *out,
                                     const seismic_words_t &seismic_words_value) {
    const unsigned count = static_cast<unsigned>(SEISMIC_DIM_M);
    const projection::u64 hidden = SEISMIC_DIM_H;
    for (projection::u64 item = threadIdx.x; item < count * hidden; item += blockDim.x)
        element::put<projection::Act>(out, item,
            in.value(item / hidden, item % hidden, inverses[item / hidden], seismic_words_value));
    __syncthreads();
    if constexpr (NVFP4) {
        const projection::Nvfp4Rows<Rows> operands{Rows{out, hidden, projection::AllRows{}}};
        for (unsigned row = 0; row < count; ++row) {
            projection::form_row(operands, row, hidden, out);
            __syncthreads();
        }
    }
}

// Shared-expert block `block` (counted from the first after the blocks
// before the shared expert's) over the `gate` and `up` weights into
// `product`: its rows formed in `memory` (the launch's dynamic shared
// memory, M x H A elements) from the decode rows `in` once the launch's
// predecessor completed, through the block's `parts` and `inverses`; the
// GEMV's first weight loads are issued before. Every thread calls it.
__device__ __forceinline__ void block(const route::Input &in, const projection::u8 *gate, const projection::u8 *up,
                                      projection::u8 *product, float (*parts)[route::WARPS], float *inverses,
                                      projection::u8 *memory, unsigned block,
                                      const seismic_words_t &seismic_words_value) {
    __shared__ projection::GemvShared<Shape, Rows> gemv;
    const auto ready = [&] {
        seismic_dependency_start();
        route::decode_inverses(in.residual, parts, inverses, seismic_words_value);
        rows(in, inverses, memory, seismic_words_value);
        return Rows{memory, SEISMIC_DIM_H, projection::AllRows{}};
    };
    projection::gemv_segment_ready<Shape>(gemv, ready, static_cast<unsigned>(SEISMIC_DIM_M), SEISMIC_DIM_H / 64,
        static_cast<projection::u64>(block) * Shape::GROUPS + Shape::group(), SEISMIC_DIM_S, KERNEL_W1_AT(gate),
        KERNEL_W2_AT(up), projection::SiluMul<projection::Act>{product, SEISMIC_RESULT_2_STRIDE_0});
}

} // namespace shared_expert
