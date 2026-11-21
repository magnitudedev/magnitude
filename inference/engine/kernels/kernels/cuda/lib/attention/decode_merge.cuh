// Output-only helpers kept out of the append phase's ABI.
namespace attention {
// The decode merge of one (query head, row) column: the row's PARTS
// partitions in partition order, then the output gate.
template <int PARTS>
__device__ __forceinline__ void decode_gate(const Inputs &in, const float *partials,
                                            const float *statistics, element::Out<Act> result, int query_head,
                                            u64 row, int column) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const u64 first = (row * KV * G + query_head) * PARTS;
    constexpr int MAX_SHARED_PARTS = 256;
    float normalized;
    if constexpr (PARTS <= MAX_SHARED_PARTS) {
        __shared__ float shared_weights[PARTS];
        __shared__ float shared_inv_norm;
        if (threadIdx.x < 32) {
            float denominator;
            const float maximum = merge_weights<PARTS>(
                statistics + first * 2, shared_weights, denominator, static_cast<int>(threadIdx.x));
            if (threadIdx.x == 0) {
                shared_inv_norm = 1.0f / normalizer(in, query_head, maximum, denominator);
            }
        }
        __syncthreads();
        float accumulator = 0.0f;
        const float *values = partials + first * W + column;
        for (int p = 0; p < PARTS; ++p) {
            const float weight = shared_weights[p];
            if (weight > 0.0f) {
                accumulator = __fmaf_rn(weight, values[static_cast<u64>(p) * W], accumulator);
            }
        }
        normalized = accumulator * shared_inv_norm;
    } else {
        float denominator, accumulator;
        const float maximum = merge(statistics + first * 2, 2, partials + first * W + column, W, PARTS,
                                    denominator, accumulator);
        normalized = accumulator / normalizer(in, query_head, maximum, denominator);
    }
    const float attended = gated(in, row, query_head, column, normalized);
    result.put(row * SEISMIC_RESULT_0_STRIDE_0 + query_head * SEISMIC_RESULT_0_STRIDE_1 +
                   column * SEISMIC_RESULT_0_STRIDE_2,
               attended);
}

} // namespace attention
