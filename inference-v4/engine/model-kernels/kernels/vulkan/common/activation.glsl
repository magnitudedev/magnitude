// `ELEMENT_ACT`: the element kind of the activation element A, for entries
// that bind A (the counterpart of `element::Act` in `metal/common/element.h`
// and `cuda/common/element.cuh`). Its ladder names A's ABI symbols, which the
// build admits only in entries with an element parameter A, so it is a
// header of its own rather than part of `common/element.glsl`.
#include "common/element.glsl"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
#define ELEMENT_ACT ELEMENT_BF16
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
#define ELEMENT_ACT ELEMENT_F16
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#define ELEMENT_ACT ELEMENT_F32
#else
#error "the activation element A must be dense f32, bf16 or f16"
#endif
