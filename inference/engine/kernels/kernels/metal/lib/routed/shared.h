// The shared expert's expansion in a decode route's launch, shared by the
// routes that also expand it (`routed_route_shared`, `routed_route_streamed`):
// both bind `routed_route_shared`'s shared-expert ports (`shared_gate`,
// `shared_up`, the product `shared_product` as result 2, of static extent
// S) and its SIMDGROUPS, ROWS and LANES. The includer defines KERNEL_W1 and
// KERNEL_W2 as the shared gate and up weights and includes `route.h` first.
// Apart from `route.h`, which `routed_route` also includes and which so
// names no symbol of these ports.

// The shared rows run `routed_expand`'s paired GEMV, which packs 16-bit
// activations (`lib/routed/routed.h`); F32 activations take `routed_route`.
static_assert(element::Act::bytes == 2, "a route that expands the shared expert requires a bf16 or f16 activation");
// The rows the shared gate/up weights take (the normalized rows, or NVFP4
// operands), formed while the shared GEMV stages them.
constant constexpr bool shared_nvfp4 =
    projection::OperandRecipe<packets::W1_OPERANDS_NVFP4, packets::W2_OPERANDS_NVFP4>::NVFP4;

// Shared-expert threadgroup `tile` of the decode rows `in`: `routed_expand`'s
// shared row, a K1 paired GEMV of ROWS weight rows per lane group of LANES
// lanes over the shared expert's gate and up rows, with the SiLU . mul
// epilogue. `globals` is the entry's `PROJECTION_OPERANDS_SHARED` of
// `shared_nvfp4`. Every thread calls it.
inline void shared_group(thread const RouteRows &in, device const uchar *shared_gate, device const uchar *shared_up,
    device uchar *shared_product, constant ulong *seismic_words, uint tile, threadgroup float *globals,
    threadgroup uchar *shared, uint simd, uint lane) {
    const uint rows = uint(SEISMIC_DIM_M);
    const uint k = uint(SEISMIC_DIM_H);
    const projection::Weights<packets::W1> gate{shared_gate, KERNEL_W1_LAYOUT(k), k, nullptr};
    const projection::Weights<packets::W2> up{shared_up, KERNEL_W2_LAYOUT(k), k, nullptr};
    const projection::SiluMul<Act> out{shared_product, SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1};
    const auto x = projection::operands<shared_nvfp4>::rows(in, rows, k, globals, SEISMIC_TUNE_SIMDGROUPS, simd, lane);
    PROJECTION_FOR_ROWS(rows,
        (projection::gemv_paired_runtime<packets::W1, packets::W2, SEISMIC_TUNE_ROWS, MAXM, SEISMIC_TUNE_LANES>(x,
            out, gate, up, rows, SEISMIC_DIM_S, k, tile, shared, SEISMIC_TUNE_SIMDGROUPS, simd, lane)));
}
