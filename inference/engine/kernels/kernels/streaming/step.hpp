// Expert streaming, a step's words (`routed_step` in routed.seismic), over
// the backend's shim (streaming/contract.hpp), by one thread: the words the
// host wrote before submitting the step (`sequence`: its pair and front
// rows), then its guesses' kinds (STEP_GUESSES), each front row r's (paired
// with exact row r - 1) in its field from bit STEP_GUESS_SHIFT(r), the kind
// word of `guesses[r - 1]`; the host sets a front row's bit only past row 0
// and below VERIFY_ROWS and the graph's rows. `guesses` is (guess, kind) rows of `rows` rows (0 for a
// graph whose steps run no front row), strides `guesses0` and `guesses1`.
// Every later entry of the step reads the words in order on the same queue.

STREAMING_FN void streaming_step_words(STREAMING_DEVICE_WORDS step, u64 step0, STREAMING_DEVICE_READ sequence,
    u64 sequence0, STREAMING_DEVICE_READ guesses, u64 guesses0, u64 guesses1, u32 rows) {
    const int front = streaming_device_load(STREAMING_AT(sequence, SEISMIC_CONST_STEP_FRONT * sequence0));
    u32 kinds = 0u;
    for (u32 row = 1; row < rows; ++row) {
        if (((u32(front) >> row) & 1u) == 0u)
            continue;
        const int kind = streaming_device_load(STREAMING_AT(guesses, u64(row - 1) * guesses0 + guesses1));
        kinds |= u32(kind) << SEISMIC_CONST_STEP_GUESS_SHIFT(row);
    }
    streaming_device_store(STREAMING_AT(step, SEISMIC_CONST_STEP_LOW * step0),
        streaming_device_load(STREAMING_AT(sequence, SEISMIC_CONST_STEP_LOW * sequence0)));
    streaming_device_store(STREAMING_AT(step, SEISMIC_CONST_STEP_HIGH * step0),
        streaming_device_load(STREAMING_AT(sequence, SEISMIC_CONST_STEP_HIGH * sequence0)));
    streaming_device_store(STREAMING_AT(step, SEISMIC_CONST_STEP_FRONT * step0), front);
    streaming_device_store(STREAMING_AT(step, SEISMIC_CONST_STEP_GUESSES * step0), int(kinds));
}
