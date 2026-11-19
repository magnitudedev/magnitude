// CUDA's shim under the chain `guess_chain` runs
// (`two_row/guess.hpp`), which it includes.
#define GUESS_FN __device__ inline
#define GUESS_READ const int *
#define GUESS_WORDS int *
#define GUESS_AT(handle, index) ((handle) + (index))
#define guess_load(handle) (*(handle))
#define guess_store(handle, value) (*(handle) = (value))
typedef unsigned u32;
typedef unsigned long long u64;
#include "../../../two_row/guess.hpp"
