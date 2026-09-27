uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint head = threadgroup_position_in_grid.y;
uint row = threadgroup_position_in_grid.z;
uint batch = row / COUNT, token = row % COUNT;
bool query = head < HQ;
uint h = query ? head : head - HQ;
size_t base = size_t(row) * (2 * HQ + 2 * HK) * WIDTH;
size_t src = base + (query ? h * 2 * WIDTH : (2 * HQ + h) * WIDTH);
float x[4], sum = 0.0f;
for (uint i = 0; i < 4; ++i) {
    uint d = 4 * tid + i;
    x[i] = d < WIDTH ? float(projected[src + d]) : 0.0f;
    sum += x[i] * x[i];
}
threadgroup float partial[32];
threadgroup T normalized[WIDTH];
if (sg == 0) partial[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
sum = simd_sum(sum);
if (lane == 0) partial[sg] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    sum = simd_sum(partial[lane]);
    if (lane == 0) partial[0] = metal::precise::rsqrt(
        sum / float(WIDTH) + (query ? QEPS : KEPS));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = 0; i < 4; ++i) {
    uint d = 4 * tid + i;
    if (d >= WIDTH) break;
    normalized[d] = T(T(x[i] * partial[0]) * (query ? query_weight[d] : key_weight[d]));
    if (query) gates[(size_t(row) * HQ + h) * WIDTH + d] = projected[src + WIDTH + d];
    else values[((size_t(batch) * HK + h) * COUNT + token) * WIDTH + d]
        = projected[base + (2 * HQ + HK + h) * WIDTH + d];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
auto output = query ? queries : keys;
size_t dst = ((size_t(batch) * (query ? HQ : HK) + h) * COUNT + token) * WIDTH;
for (uint d = tid; d < WIDTH; d += THREADS) {
    float result = float(normalized[d]);
    if (d < ROTARY) {
        bool first = d < ROTARY / 2;
        uint frequency = first ? d : d - ROTARY / 2;
        uint pair = first ? d + ROTARY / 2 : frequency;
        float angle = float(positions[batch] + token) * frequencies[frequency];
        float c = metal::cos(angle), s = metal::sin(angle);
        float other = float(normalized[pair]);
        result = first ? result * c - other * s : result * c + other * s;
    }
    output[dst + d] = T(result);
}
