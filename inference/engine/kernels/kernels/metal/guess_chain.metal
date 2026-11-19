// guess_chain (guess.seismic): each chaining row's next guesses and each
// continuing row's token row and kind, by one thread (two_row/guess.hpp).
#include "lib/two_row/guess.h"

kernel void guess_chain(
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device const int *chaining [[buffer(SEISMIC_BUFFER_CHAINING)]],
    device const int *selected [[buffer(SEISMIC_BUFFER_SELECTED)]],
    device const int *drafted [[buffer(SEISMIC_BUFFER_DRAFTED)]],
    device const int *continuing [[buffer(SEISMIC_BUFFER_CONTINUING)]],
    device int *guesses [[buffer(SEISMIC_BUFFER_GUESSES)]],
    device int *next_selected [[buffer(SEISMIC_BUFFER_NEXT_SELECTED)]],
    device int *next_guesses [[buffer(SEISMIC_BUFFER_NEXT_GUESSES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0)
        guess_chain_rows(tokens, SEISMIC_TOKENS_STRIDE_0, SEISMIC_TOKENS_STRIDE_1, chaining, SEISMIC_CHAINING_STRIDE_0,
            SEISMIC_CHAINING_STRIDE_1, selected, SEISMIC_SELECTED_STRIDE_0, SEISMIC_SELECTED_STRIDE_1, drafted,
            SEISMIC_DRAFTED_STRIDE_0, SEISMIC_DRAFTED_STRIDE_1, continuing, SEISMIC_CONTINUING_STRIDE_0,
            SEISMIC_CONTINUING_STRIDE_1, guesses, SEISMIC_GUESSES_STRIDE_0, SEISMIC_GUESSES_STRIDE_1, next_selected,
            SEISMIC_NEXT_SELECTED_STRIDE_0, SEISMIC_NEXT_SELECTED_STRIDE_1, next_guesses, SEISMIC_NEXT_GUESSES_STRIDE_0,
            SEISMIC_NEXT_GUESSES_STRIDE_1, uint(SEISMIC_DIM_S), uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_N));
}
