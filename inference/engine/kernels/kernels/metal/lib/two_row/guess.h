// Metal's shim under the chain `guess_chain` runs
// (`two_row/guess.hpp`), which it includes.
#define GUESS_FN inline
#define GUESS_READ device const int *
#define GUESS_WORDS device int *
#define GUESS_AT(handle, index) ((handle) + (index))
#define guess_load(handle) (*(handle))
#define guess_store(handle, value) (*(handle) = (value))
typedef uint u32;
typedef ulong u64;
#include "../../../two_row/guess.hpp"
