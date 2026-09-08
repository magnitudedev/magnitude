// Numerical bank selection is independent of the physical row permutation.
template<typename T, int BITS, int PACK, int K, int N, int GROUP, int R,
         int SLOTS, int BANKS, bool SHARED, bool PER_SLOT, typename Indices>
inline MagnitudeFragment<T, R * 4> magnitude_selected(
    const device T* x, Indices ids,
    const device uint* w, const device T* s, const device T* bias,
    const device uint* sw, const device T* ss, const device T* sb,
    const thread int* slots, uint first, uint lane) {
    uint experts[R];
    for (uint r = 0; r < R; ++r) experts[r] = slots[r] >= 0 ? ids[slots[r]] : uint(-1);
    float acc[R][4] = {0};
    for (uint leader = 0; leader < R; ++leader) {
        if (slots[leader] < 0) continue;
        bool seen = false;
        for (uint p = 0; p < leader; ++p) seen |= experts[p] == experts[leader];
        if (seen) continue;
        int rows[R];
        for (uint r = 0; r < R; ++r)
            rows[r] = experts[r] == experts[leader] ? (PER_SLOT ? slots[r] : slots[r] / SLOTS) : -1;
        bool shared = SHARED && experts[leader] == BANKS;
        size_t offset = size_t(shared ? 0 : experts[leader]) * N;
        magnitude_contract<T, BITS, PACK, K, N, GROUP, R>(x,
            (shared ? sw : w) + offset * (K * BITS / 32),
            (shared ? ss : s) + offset * (K / GROUP),
            (shared ? sb : bias) + offset * (K / GROUP), rows, first, lane, acc);
    }
    MagnitudeFragment<T, R * 4> result;
    for (uint r = 0; r < R; ++r)
        for (uint c = 0; c < 4; ++c) result.values[r * 4 + c] = T(simd_sum(acc[r][c]));
    return result;
}
