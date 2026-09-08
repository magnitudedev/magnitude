uint lane = thread_index_in_simdgroup;
uint first = threadgroup_position_in_grid.y * 8 + simdgroup_index_in_threadgroup * 4;
uint base = threadgroup_position_in_grid.z * R;
uint slots[R], experts[R];
for (uint r = 0; r < R; ++r) {
    slots[r] = base + r < M ? order[base + r] : M;
    experts[r] = slots[r] < M ? ids[slots[r]] : uint(-1);
}
float acc[R][4] = {0};
for (uint leader = 0; leader < R; ++leader) {
    if (slots[leader] == M) continue;
    bool seen = false;
    for (uint p = 0; p < leader; ++p) seen |= experts[p] == experts[leader];
    if (seen) continue;
    bool shared = SHARED && experts[leader] == E;
    uint expert = shared ? 0 : experts[leader];
    int rows[R];
    for (uint r = 0; r < R; ++r)
        rows[r] = experts[r] == experts[leader] ? int(slots[r]) : -1;
    constexpr uint KW = K * BITS / 32, KG = K / GROUP;
    size_t offset = size_t(expert) * N;
    magnitude_contract<T, BITS, PACK, K, N, GROUP, R>(x,
        (shared ? wsh : w) + offset * KW,
        (shared ? ssh : scales) + offset * KG, (shared ? bsh : biases) + offset * KG,
        rows, first, lane, acc);
}
for (uint r = 0; r < R; ++r) for (uint c = 0; c < 4; ++c) {
    float value = simd_sum(acc[r][c]);
    if (lane == 0 && slots[r] < M && first + c < N)
        out[size_t(slots[r]) * N + first + c] = T(value);
}
