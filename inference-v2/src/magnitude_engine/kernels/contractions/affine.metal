// A step owns encoded arithmetic; the driver owns ordered traversal and shared input loads.
template<typename T, int BITS, int PACK, int K, int N, int GROUP, int R>
struct AffineStep {
    const device uint* weight;
    const device T* scales;
    const device T* biases;
    AffinePack<BITS, PACK> prepared[4];
    struct State { float acc[R][4]; };
    using Result = MagnitudeFragment<T, R * 4>;

    void prepare(uint k, uint first) {
        for (uint c = 0; c < 4; ++c) {
            uint column = min(first + c, uint(N - 1));
            size_t g = size_t(column) * (K / GROUP) + k / GROUP;
            prepared[c].load(weight + size_t(column) * (K * BITS / 32) + k * BITS / 32,
                             float(scales[g]), float(biases[g]));
        }
    }
    void step(thread State& state, uint row, const thread float* values, float sum) const {
        for (uint c = 0; c < 4; ++c) state.acc[row][c] += prepared[c].dot(values, sum);
    }
    Result finish(thread State& state, uint lane) const {
        Result result;
        for (uint r = 0; r < R; ++r)
            for (uint c = 0; c < 4; ++c) result.values[r * 4 + c] = T(simd_sum(state.acc[r][c]));
        return result;
    }
};

template<typename T, int BITS, int PACK, int K, int R, bool PREPARED, typename Input, typename Sums, typename Body>
inline typename Body::Result magnitude_affine_fold(
    Input x, Sums sums, const thread int* rows, uint first, uint lane, thread Body& body) {
    typename Body::State state = {};
    for (uint k = lane * PACK; k < K; k += 32 * PACK) {
        body.prepare(k, first);
        for (uint r = 0; r < R; ++r) {
            if (rows[r] < 0) continue;
            float values[PACK];
            float sum;
            if constexpr (PREPARED) {
                for (uint i = 0; i < PACK; ++i) values[i] = x[size_t(rows[r]) * K + k + i];
                sum = sums[size_t(rows[r]) * (K / PACK) + k / PACK];
            } else {
                sum = magnitude_load<T, BITS, PACK>(x + size_t(rows[r]) * K + k, values);
            }
            body.step(state, r, values, sum);
        }
    }
    return body.finish(state, lane);
}
