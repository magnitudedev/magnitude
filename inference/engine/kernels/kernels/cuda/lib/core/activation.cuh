// `activation::Act`: the activation element `A`, for entries that bind one —
// the engine's element-parameter convention over Seismic's dense element
// types (<seismic/element.cuh>). The counterpart of
// `metal/lib/core/activation.h` and `vulkan/lib/core/activation.glsl`.
//
// An entry with an A element binds it before including this file, as it
// binds a weight slot:
//     #define KERNEL_A SEISMIC_ELEMENT_A
// so a header names only symbols every including entry has.

#include <seismic/element.cuh>

namespace activation {
// The activation element A, when the entry binds one.
#if defined(KERNEL_A)
typedef ELEMENT_OF(KERNEL_A) Act;
#endif
} // namespace activation
