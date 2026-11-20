// Two-row decode's chain (`guess_chain` in guess.seismic), written once, in
// the language MSL, CUDA C++ and GLSL have in common, by one thread. Its
// includer defines first:
// - `GUESS_FN`, a function's qualifier; `u32` and `u64`;
// - `GUESS_READ` and `GUESS_WORDS`, the handles of read-only and read-write
//   i32 words; `GUESS_AT(handle, index)`, the handle of word `index` past
//   `handle`; `guess_load(handle)` and `guess_store(handle, value)`.
// Every tensor is passed as its handle and its strides in words.

// `guess_chain`: each chaining row's chain of next guesses and their kinds
// (its consistent front rows' selections, then the head's selection on the
// deepest of them, guess.seismic's rule), then each continuing row's token
// row and kind. Every output row is cleared first, since a continuing row's
// kind lands on the row before it.
GUESS_FN void guess_chain_rows(GUESS_READ tokens, u64 tokens0, u64 tokens1, GUESS_READ chaining, u64 chaining0,
    u64 chaining1, GUESS_READ selected, u64 selected0, u64 selected1, GUESS_READ drafted, u64 drafted0, u64 drafted1,
    GUESS_READ continuing, u64 continuing0, u64 continuing1, GUESS_WORDS guesses, u64 guesses0, u64 guesses1,
    GUESS_WORDS next_selected, u64 next_selected0, u64 next_selected1, GUESS_WORDS next_guesses, u64 next_guesses0,
    u64 next_guesses1, u32 rows, u32 token_rows, u32 continued) {
    for (u32 s = 0u; s < rows; ++s) {
        guess_store(GUESS_AT(guesses, u64(s) * guesses0), 0);
        guess_store(GUESS_AT(guesses, u64(s) * guesses0 + guesses1), 0);
    }
    for (u32 r = 0u; r < continued; ++r) {
        guess_store(GUESS_AT(next_selected, u64(r) * next_selected0), 0);
        guess_store(GUESS_AT(next_selected, u64(r) * next_selected0 + next_selected1), 0);
        guess_store(GUESS_AT(next_guesses, u64(r) * next_guesses0), 0);
        guess_store(GUESS_AT(next_guesses, u64(r) * next_guesses0 + next_guesses1), 0);
    }
    for (u32 s = 0u; s < rows; ++s) {
        const u64 at = u64(s) * chaining0;
        const int t = guess_load(GUESS_AT(chaining, at + u64(SEISMIC_CONST_CHAINING_ROW) * chaining1));
        const int front = guess_load(GUESS_AT(chaining, at + u64(SEISMIC_CONST_CHAINING_FRONT) * chaining1));
        if (t < 0 || u32(t) >= token_rows || guess_load(GUESS_AT(selected, u64(s) * selected0 + selected1)) != 0
            || guess_load(GUESS_AT(tokens, u64(t) * tokens0 + tokens1)) != 0)
            continue;
        int last = guess_load(GUESS_AT(selected, u64(s) * selected0));
        u32 deepest = 0u;
        for (int depth = 1; depth <= front && s + u32(depth) < rows && u32(t + depth) < token_rows; ++depth) {
            const u64 row = u64(s) + u64(depth);
            const u64 token = u64(t + depth);
            if (guess_load(GUESS_AT(selected, row * selected0 + selected1)) != 0
                || guess_load(GUESS_AT(tokens, token * tokens0 + tokens1)) != 0
                || guess_load(GUESS_AT(tokens, token * tokens0)) != last)
                break;
            last = guess_load(GUESS_AT(selected, row * selected0));
            guess_store(GUESS_AT(guesses, (row - u64(1)) * guesses0), last);
            guess_store(GUESS_AT(guesses, (row - u64(1)) * guesses0 + guesses1), SEISMIC_CONST_GUESS_CHAINED);
            deepest = u32(depth);
        }
        const u64 entry = u64(s + deepest);
        if (guess_load(GUESS_AT(drafted, entry * drafted0 + drafted1)) == 0) {
            guess_store(GUESS_AT(guesses, entry * guesses0), guess_load(GUESS_AT(drafted, entry * drafted0)));
            guess_store(GUESS_AT(guesses, entry * guesses0 + guesses1), SEISMIC_CONST_GUESS_HEAD);
        }
    }
    for (u32 r = 0u; r < continued; ++r) {
        const u64 at = u64(r) * continuing0;
        const int s = guess_load(GUESS_AT(continuing, at + u64(SEISMIC_CONST_CONTINUING_SLOT) * continuing1));
        const int depth = guess_load(GUESS_AT(continuing, at + u64(SEISMIC_CONST_CONTINUING_DEPTH) * continuing1));
        if (s < 0 || u32(s) >= rows)
            continue;
        const u64 slot = u64(s);
        if (depth == 0) {
            guess_store(GUESS_AT(next_selected, u64(r) * next_selected0),
                guess_load(GUESS_AT(selected, slot * selected0)));
            guess_store(GUESS_AT(next_selected, u64(r) * next_selected0 + next_selected1),
                guess_load(GUESS_AT(selected, slot * selected0 + selected1)));
            continue;
        }
        const bool held = guess_load(GUESS_AT(chaining, slot * chaining0 + u64(SEISMIC_CONST_CHAINING_ROW) * chaining1)) >= 0
            && depth >= 1
            && depth <= guess_load(GUESS_AT(chaining, slot * chaining0 + u64(SEISMIC_CONST_CHAINING_FRONT) * chaining1)) + 1
            && u32(s + depth - 1) < rows;
        const u64 row = u64(s + depth - 1);
        int token = guess_load(GUESS_AT(selected, slot * selected0));
        int guess = 0;
        int kind = 0;
        if (held && guess_load(GUESS_AT(guesses, row * guesses0 + guesses1)) != 0) {
            token = guess_load(GUESS_AT(guesses, row * guesses0));
            guess = token;
            kind = guess_load(GUESS_AT(guesses, row * guesses0 + guesses1));
        }
        guess_store(GUESS_AT(next_selected, u64(r) * next_selected0), token);
        guess_store(GUESS_AT(next_selected, u64(r) * next_selected0 + next_selected1), 0);
        if (r >= 1u) {
            guess_store(GUESS_AT(next_guesses, u64(r - 1u) * next_guesses0), guess);
            guess_store(GUESS_AT(next_guesses, u64(r - 1u) * next_guesses0 + next_guesses1), kind);
        }
    }
}
