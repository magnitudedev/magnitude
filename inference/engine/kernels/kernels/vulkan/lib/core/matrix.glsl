// The cooperative-matrix fragments of the subgroup 16 x 16 x 16 products
// (`SEISMIC_HAS_MATRIX`) shared by the projection and attention bodies.
//
// This file is independent of any entry ABI.

#if SEISMIC_HAS_MATRIX
#define MATRIX_FRAGMENT_A coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MATRIX_FRAGMENT_B coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define MATRIX_FRAGMENT_C coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>

// The zero accumulator. Every accumulator clear takes it from this one
// function: the AMD Windows driver (32.0.21017, LLPC) faults in
// vkCreateComputePipelines when one cooperative-matrix constant is used by
// two functions of a module.
MATRIX_FRAGMENT_C matrix_zero() { return MATRIX_FRAGMENT_C(0.0); }
#endif
