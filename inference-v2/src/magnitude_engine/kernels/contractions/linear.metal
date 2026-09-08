uint lane = thread_index_in_simdgroup;
uint first = threadgroup_position_in_grid.y * 8 + simdgroup_index_in_threadgroup * 4;
uint base = threadgroup_position_in_grid.z * R;
int rows[R];
for (uint r = 0; r < R; ++r) rows[r] = base + r < M ? int(base + r) : -1;
float acc[R][4] = {0};
if constexpr (PREPARED) {
    magnitude_contract<T, BITS, PACK, K, N, GROUP, R, true, X>(
        x, w, scales, biases, rows, first, lane, acc, sums);
} else {
    magnitude_contract<T, BITS, PACK, K, N, GROUP, R>(
        x, w, scales, biases, rows, first, lane, acc);
}
for (uint r = 0; r < R; ++r) for (uint c = 0; c < 4; ++c) {
    float value = simd_sum(acc[r][c]);
    if (lane == 0 && rows[r] >= 0 && first + c < N)
        out[size_t(rows[r]) * N + first + c] = T(value);
}
