uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint first = threadgroup_position_in_grid.y * 8 + sg * 4;
uint row = threadgroup_position_in_grid.z;
constexpr uint KW = K * BITS / 32, KG = K / GROUP;
float combined[4] = {0};
constexpr uint SLOTS = TOPK + SHARED;
for (uint slot = 0; slot < SLOTS; ++slot) {
    bool shared = SHARED && slot == TOPK;
    uint expert = shared ? 0 : inds[row * TOPK + slot];
    const device uint* selected_w = shared ? wsh : w;
    const device T* selected_s = shared ? ssh : scales;
    const device T* selected_b = shared ? bsh : biases;
    T score = shared ? shared_score[row] : scores[row * TOPK + slot];
    float acc[1][4] = {0};
    int rows[1] = {int(row * SLOTS + slot)};
    size_t base = size_t(expert) * N;
    magnitude_contract<T, BITS, PACK, K, N, GROUP, 1>(
        x, selected_w + base * KW, selected_s + base * KG, selected_b + base * KG,
        rows, first, lane, acc);
    for (uint r = 0; r < 4; ++r) {
        float value = simd_sum(acc[0][r]);
        // MLX's short column reduction adds in the output dtype.
        T contribution = T(T(value) * score);
        combined[r] = float(T(combined[r] + float(contribution)));
    }
}
if (lane == 0) for (uint r = 0; r < 4; ++r)
    out[row * N + first + r] = T(combined[r]);
