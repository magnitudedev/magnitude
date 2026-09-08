uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
uint row = threadgroup_position_in_grid.y;
size_t base = size_t(row) * BLOCKS;
float peak = -INFINITY;
for (uint b = lane; b < BLOCKS; b += 32) peak = max(peak, maximum[base + b]);
peak = simd_max(peak);
float norm = 0.0f;
for (uint b = lane; b < BLOCKS; b += 32)
    if (denominator[base + b] > 0)
        norm += exp(maximum[base + b] - peak) * denominator[base + b];
norm = simd_sum(norm);
float values[DV / 32];
for (uint i = 0; i < DV / 32; ++i) values[i] = 0.0f;
for (uint b = sg; b < BLOCKS; b += GROUPS) {
    float factor = denominator[base + b] > 0 ? exp(maximum[base + b] - peak) : 0.0f;
    for (uint i = 0; i < DV / 32; ++i)
        values[i] += factor * partial[(base + b) * DV + lane + 32 * i];
}
threadgroup float scratch[GROUPS * DV];
for (uint i = 0; i < DV / 32; ++i) scratch[sg * DV + lane + 32 * i] = values[i];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) for (uint i = 0; i < DV / 32; ++i) {
    float value = 0.0f;
    for (uint g = 0; g < GROUPS; ++g) value += scratch[g * DV + lane + 32 * i];
    output[row * DV + lane + 32 * i] = In(value / norm);
}
