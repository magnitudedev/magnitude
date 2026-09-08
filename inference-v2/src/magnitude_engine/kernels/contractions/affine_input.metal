uint pack = thread_position_in_grid.x;
if (pack >= M * K / PACK) return;
float values[PACK];
float sum = magnitude_load<T, BITS, PACK>(x + size_t(pack) * PACK, values);
for (uint i = 0; i < PACK; ++i) prepared[size_t(pack) * PACK + i] = values[i];
sums[pack] = sum;
