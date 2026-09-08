uint lane = thread_position_in_grid.x;
uint channel = thread_position_in_grid.y;
uint owner = thread_position_in_grid.z;
if (channel >= DV) return;
uint batch = owner / HV;
uint head = owner % HV;
uint key_head = head / (HV / HK);
constexpr uint WIDTH = DK / 32;
const uint tokens = SHORT_T > 0 ? SHORT_T : length[0];
float memory[WIDTH];
size_t base = (size_t(owner) * DV + channel) * DK;
for (uint i = 0; i < WIDTH; ++i)
    memory[i] = initial[base + lane * WIDTH + i];
// Resolve each row/head once. Advancing constant strides avoids wide
// per-token address products in the register-resident recurrence loop.
auto next_key = k + (size_t(batch) * tokens * HK + key_head) * DK;
auto next_value = v + (size_t(batch) * tokens * HV + head) * DV + channel;
auto next_decay = decay + size_t(batch) * tokens * HV + head;
auto next_beta = beta + size_t(batch) * tokens * HV + head;
#if !STATE_ONLY
auto next_query = q + (size_t(batch) * tokens * HK + key_head) * DK;
#endif
#if !STATE_ONLY
auto next_output = output + (size_t(batch) * tokens * HV + head) * DV + channel;
#endif
for (uint t = 0; t < tokens; ++t) {
    float remembered = 0.0f;
    for (uint i = 0; i < WIDTH; ++i) {
        memory[i] *= float(*next_decay);
        remembered += memory[i] * float(next_key[lane * WIDTH + i]);
    }
    remembered = simd_sum(remembered);
    float residual = (float(*next_value) - remembered) * float(*next_beta);
    float answer = 0.0f;
    for (uint i = 0; i < WIDTH; ++i) {
        uint coordinate = lane * WIDTH + i;
        memory[i] += residual * float(next_key[coordinate]);
        #if !STATE_ONLY
answer += memory[i] * float(next_query[coordinate]);
#endif
    }
    #if !STATE_ONLY
answer = simd_sum(answer); if (lane == 0) *next_output = In(answer);
#endif
    next_key += HK * DK;
    next_value += HV * DV;
    next_decay += HV;
    next_beta += HV;
    #if !STATE_ONLY
next_query += HK * DK;
#endif
    #if !STATE_ONLY
next_output += HV * DV;
#endif
}
for (uint i = 0; i < WIDTH; ++i)
    final[base + lane * WIDTH + i] = memory[i];
