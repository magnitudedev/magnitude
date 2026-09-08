uint lane = thread_index_in_simdgroup;
uint first = threadgroup_position_in_grid.y * 8 + simdgroup_index_in_threadgroup * 4;
uint base = threadgroup_position_in_grid.z * R;
constexpr uint SLOTS = TOPK + SHARED;
uint slots[R], experts[R];
for (uint r = 0; r < R; ++r) {
    slots[r] = base + r < M ? order[base + r] : M;
    experts[r] = slots[r] < M ? ids[slots[r]] : uint(-1);
}
float gate[R][4] = {0}, up[R][4] = {0};
for (uint leader = 0; leader < R; ++leader) {
    if (slots[leader] == M) continue;
    bool seen = false;
    for (uint p = 0; p < leader; ++p) seen |= experts[p] == experts[leader];
    if (seen) continue;
    bool shared = SHARED && experts[leader] == E;
    uint expert = shared ? 0 : experts[leader];
    int rows[R];
    for (uint r = 0; r < R; ++r)
        rows[r] = experts[r] == experts[leader] ? int(slots[r] / SLOTS) : -1;
    constexpr uint KW = K * BITS / 32, KG = K / GROUP;
    size_t offset = size_t(expert) * N;
    magnitude_contract<T, BITS, PACK, K, N, GROUP, R>(x,
        (shared ? wsg : wg) + offset * KW,
        (shared ? ssg : sg_) + offset * KG, (shared ? bsg : bg_) + offset * KG,
        rows, first, lane, gate);
    magnitude_contract<T, BITS, PACK, K, N, GROUP, R>(x,
        (shared ? wsu : wu) + offset * KW,
        (shared ? ssu : su_) + offset * KG, (shared ? bsu : bu_) + offset * KG,
        rows, first, lane, up);
}
for (uint r = 0; r < R; ++r) for (uint c = 0; c < 4; ++c) {
    float g = simd_sum(gate[r][c]), u = simd_sum(up[r][c]);
    if (lane == 0 && slots[r] < M && first + c < N) {
        T gr = T(g), ur = T(u);
        T e = T(metal::exp(metal::abs(float(gr))));
        auto y = 1 / (1 + e);
        T sigmoid = T(gr < 0 ? y : 1 - y);
        out[size_t(slots[r]) * N + first + c] = T(T(gr * sigmoid) * ur);
    }
}
