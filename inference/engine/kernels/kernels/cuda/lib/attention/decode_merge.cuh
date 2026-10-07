// Output-only helpers kept out of the append phase's ABI.
namespace attention {
// The decode merge of one (query head, row) column: the row's PARTS
// partitions in partition order, then the output gate.
template <int PARTS>
__device__ __forceinline__ void decode_gate(const Inputs &in, const float *partials,
                                            const float *statistics, u8 *result, int query_head,
                                            u64 row, int column) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const u64 first = (row * KV * G + query_head) * PARTS;
    float denominator, accumulator;
    merge(statistics + first * 2, 2, partials + first * W + column, W, PARTS, denominator,
          accumulator);
    const float attended = gated(in, row, query_head, column, accumulator / fmaxf(denominator, 1e-30f));
    element::put<Act>(result,
                      row * SEISMIC_RESULT_0_STRIDE_0 + query_head * SEISMIC_RESULT_0_STRIDE_1 +
                          column * SEISMIC_RESULT_0_STRIDE_2,
                      attended);
}

} // namespace attention
