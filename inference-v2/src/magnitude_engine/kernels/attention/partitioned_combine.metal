uint lane = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;
const uint splits = layout[2];
size_t base = size_t(row) * splits * (DV + 2);
float maximum = -INFINITY;
for (uint split = lane; split < splits; split += 32)
    maximum = max(maximum, partial[base + split * (DV + 2) + DV]);
maximum = simd_max(maximum);
float denominator = 0.0f;
float result[DV / 32];
for (uint i = 0; i < DV / 32; ++i) result[i] = 0.0f;
for (uint split = 0; split < splits; ++split) {
    size_t address = base + split * (DV + 2);
    float weight = exp(partial[address + DV] - maximum);
    denominator += weight * partial[address + DV + 1];
    for (uint i = 0; i < DV / 32; ++i)
        result[i] += weight * partial[address + lane + 32 * i];
}
for (uint i = 0; i < DV / 32; ++i)
    output[row * DV + lane + 32 * i] = Out(result[i] / denominator);
