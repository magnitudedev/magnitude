// `element::Act`: the activation element `A`, for entries that bind one — the
// engine's element-parameter convention over Seismic's dense element types
// (<seismic/element.cuh>). The counterpart of `metal/lib/core/activation.h` and
// `vulkan/lib/core/activation.glsl`.

#include <seismic/element.cuh>

// The ABI prefix of the activation element A. It is pasted, not written out:
// entries without an A element include this file too, and a header may name
// only ABI symbols every including entry has.
#define ELEMENT_A ELEMENT_CAT(SEISMIC, _ELEMENT_A)

namespace element {
// The activation element A, when the entry binds one.
#if ELEMENT_HAS(ELEMENT_A, _REPRESENTATION_BF16)
typedef Bf16 Act;
#elif ELEMENT_HAS(ELEMENT_A, _REPRESENTATION_F16)
typedef F16 Act;
#elif ELEMENT_HAS(ELEMENT_A, _REPRESENTATION_F32)
typedef F32 Act;
#endif
} // namespace element
