// A lane retains fixed consecutive coordinates; token order is never reassociated.
uint lane = thread_position_in_grid.x;
uint channel = thread_position_in_grid.y;
uint owner = thread_position_in_grid.z;
if (channel >= DV) return;
uint batch = owner / HV, head = owner % HV, key_head = head / (HV / HK);
uint steps = FIXED_TOKENS ? TOKENS : uint(length[0]);
constexpr uint W = DK / 32;
float memory[W];
size_t base = (size_t(owner) * DV + channel) * DK;
for (uint i = 0; i < W; ++i) memory[i] = initial[base + lane * W + i];
auto next_key = k + (size_t(batch) * steps * HK + key_head) * DK;
auto next_value = v + (size_t(batch) * steps * HV + head) * DV + channel;
auto next_decay = decay + size_t(batch) * steps * HV + head;
auto next_beta = beta + size_t(batch) * steps * HV + head;
#if !STATE_ONLY
auto next_query = q + (size_t(batch) * steps * HK + key_head) * DK;
auto next_output = output + (size_t(batch) * steps * HV + head) * DV + channel;
#endif
for (uint t = 0; t < steps; ++t) {
    float remembered = 0.0f;
    for (uint i = 0; i < W; ++i) {
        memory[i] *= float(next_decay[0]);
        remembered += memory[i] * float(next_key[lane * W + i]);
    }
    remembered = simd_sum(remembered);
    float residual = (float(next_value[0]) - remembered) * float(next_beta[0]);
#if !STATE_ONLY
    float answer = 0.0f;
#endif
    for (uint i = 0; i < W; ++i) {
        uint coordinate = lane * W + i;
        memory[i] += residual * float(next_key[coordinate]);
#if !STATE_ONLY
        answer += memory[i] * float(next_query[coordinate]);
#endif
    }
#if !STATE_ONLY
    answer = simd_sum(answer);
    if (lane == 0) next_output[0] = In(answer);
    next_query += HK * DK;
    next_output += HV * DV;
#endif
    next_key += HK * DK;
    next_value += HV * DV;
    next_decay += HV;
    next_beta += HV;
}
for (uint i = 0; i < W; ++i) final[base + lane * W + i] = memory[i];
