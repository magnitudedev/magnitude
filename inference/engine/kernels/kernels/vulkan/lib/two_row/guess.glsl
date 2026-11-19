// Vulkan's shim under the chain `guess_chain` runs
// (`two_row/guess.hpp`), which it includes: a handle is a word's address.
#include <seismic/element.glsl>
#define GUESS_FN
#define GUESS_READ uint64_t
#define GUESS_WORDS uint64_t
#define GUESS_AT(handle, index) ((handle) + (index) * 4ul)
#define guess_load(handle) element_i32_at(handle)
#define guess_store(handle, value) element_i32_put(handle, value)
#define u32 uint
#define u64 uint64_t
#include "../../../two_row/guess.hpp"
