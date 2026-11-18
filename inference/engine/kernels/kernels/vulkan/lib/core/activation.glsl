// `ACTIVATION_A`: the element kind of the activation element `A`, for
// entries that bind one with `#define KERNEL_A SEISMIC_ELEMENT_A` before their
// first include, so a header names only symbols every including entry has
// (the counterpart of `activation::Act` in `metal/lib/core/activation.h` and
// `cuda/lib/core/activation.cuh`).
#include <seismic/element.glsl>

#if defined(KERNEL_A)
#define ACTIVATION_A ELEMENT_OF(KERNEL_A)
#endif
