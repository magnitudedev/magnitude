// `activation::Act`: the activation element `A`, for entries that bind one
// with `#define KERNEL_A SEISMIC_ELEMENT_A` before their first include — the
// engine's element-parameter convention over Seismic's dense element types
// (<seismic/element.h>). The counterpart of `cuda/lib/core/activation.cuh` and
// `vulkan/lib/core/activation.glsl`.

#include <seismic/element.h>

#ifdef KERNEL_A
namespace activation {
// The activation element A.
typedef ELEMENT_OF(KERNEL_A) Act;
} // namespace activation
#endif
