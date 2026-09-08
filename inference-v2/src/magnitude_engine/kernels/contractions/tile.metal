// Each lane visits the same K positions for every row-tile size. Only encoded
// loads are shared: rows never share accumulators or change reduction domains.
template <typename T, int BITS, int PACK, int K, int N, int GROUP, int R>
void magnitude_contract(
    const device T* x, const device uint* w, const device T* scales,
    const device T* biases, const thread int* rows, uint first, uint lane,
    thread float (&acc)[R][4]) {
    constexpr uint KW = K * BITS / 32, KG = K / GROUP;
    for (uint k = lane * PACK; k < K; k += 32 * PACK) {
        AffinePack<BITS, PACK> weights[4];
        for (uint c = 0; c < 4; ++c) {
            uint channel = min(first + c, uint(N - 1));
            size_t g = size_t(channel) * KG + k / GROUP;
            weights[c].load(w + size_t(channel) * KW + k * BITS / 32,
                            float(scales[g]), float(biases[g]));
        }
        for (uint r = 0; r < R; ++r) {
            if (rows[r] < 0) continue;
            float values[PACK];
            float sum = magnitude_load<T, BITS, PACK>(x + size_t(rows[r]) * K + k, values);
            for (uint c = 0; c < 4; ++c) acc[r][c] += weights[c].dot(values, sum);
        }
    }
}
