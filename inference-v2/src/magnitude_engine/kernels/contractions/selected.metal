// Each body owns its coefficients; the driver shares route grouping and input packs.
template<typename T, int BITS, int PACK, int K, int N, int GROUP, int R>
struct SelectedStep {
    const device uint* weight;
    const device T* scales;
    const device T* biases;
    const device uint* shared_weight;
    const device T* shared_scales;
    const device T* shared_biases;
    AffineStep<T, BITS, PACK, K, N, GROUP, R> active;
    using State = typename AffineStep<T, BITS, PACK, K, N, GROUP, R>::State;
    using Result = typename AffineStep<T, BITS, PACK, K, N, GROUP, R>::Result;
    void select(uint bank, bool shared) {
        size_t offset = size_t(shared ? 0 : bank) * N;
        active.weight = (shared ? shared_weight : weight) + offset * (K * BITS / 32);
        active.scales = (shared ? shared_scales : scales) + offset * (K / GROUP);
        active.biases = (shared ? shared_biases : biases) + offset * (K / GROUP);
    }
    void prepare(uint k, uint first) { active.prepare(k, first); }
    void step(thread State& state, uint row, const thread float* values, float sum) const {
        active.step(state, row, values, sum);
    }
    Result finish(thread State& state, uint lane) const { return active.finish(state, lane); }
};

template<typename T, int BITS, int PACK, int K, int R,
         int SLOTS, int BANKS, bool SHARED, bool PER_SLOT, typename Indices, typename Body>
inline typename Body::Result magnitude_selected(
    const device T* x, Indices ids, const thread int* slots,
    uint first, uint lane, thread Body& body) {
    uint experts[R];
    for (uint r = 0; r < R; ++r) experts[r] = slots[r] >= 0 ? ids[slots[r]] : uint(-1);
    typename Body::State state = {};
    for (uint leader = 0; leader < R; ++leader) {
        if (slots[leader] < 0) continue;
        bool seen = false;
        for (uint p = 0; p < leader; ++p) seen |= experts[p] == experts[leader];
        if (seen) continue;
        int rows[R];
        for (uint r = 0; r < R; ++r)
            rows[r] = experts[r] == experts[leader] ? (PER_SLOT ? slots[r] : slots[r] / SLOTS) : -1;
        body.select(experts[leader], SHARED && experts[leader] == BANKS);
        for (uint k = lane * PACK; k < K; k += 32 * PACK) {
            body.prepare(k, first);
            for (uint r = 0; r < R; ++r) {
                if (rows[r] < 0) continue;
                float values[PACK];
                float sum = magnitude_load<T, BITS, PACK>(x + size_t(rows[r]) * K + k, values);
                body.step(state, r, values, sum);
            }
        }
    }
    return body.finish(state, lane);
}
