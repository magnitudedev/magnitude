uint lane = thread_position_in_grid.x;
uint split = thread_position_in_grid.y;
uint row = thread_position_in_grid.z;
const uint splits = layout[2];
if (split >= splits) return;
uint sequence = row / ((HQ / HEADS) * TQ);
uint head = ((row / TQ) % (HQ / HEADS)) * HEADS;
uint token = row % TQ;
uint kh = head / (HQ / HK);
uint visible = positions[sequence] + token + 1;
uint partition_base = 0;
if (layout[4] > 0 && positions[sequence] + 1 > uint(layout[4]))
    partition_base = positions[sequence] + 1 - uint(layout[4]);
uint first = partition_base + split * SPAN;
if (layout[4] > 0 && visible > uint(layout[4]))
    first = max(first, visible - uint(layout[4]));
uint last = min(partition_base + (split + 1) * SPAN, visible);
float query[HEADS][DK / 32];
float accumulator[HEADS][DV / 32];
float maximum[HEADS];
float denominator[HEADS];
for (uint h = 0; h < HEADS; ++h) {
    for (uint i = 0; i < DK / 32; ++i)
        query[h][i] = float(queries[
            ((sequence * HQ + head + h) * TQ + token) * DK + lane + 32 * i]);
    for (uint i = 0; i < DV / 32; ++i) accumulator[h][i] = 0.0f;
    maximum[h] = -INFINITY;
    denominator[h] = 0.0f;
}
uint position = first;
while (position < last) {
    // Resolve a physical page once, then consume its contiguous keys.
    uint page_index = position / layout[1];
    uint page_end = min(last, (page_index + 1) * layout[1]);
    bool in_tail = TAIL > 0 && position >= uint(tail_starts[sequence]);
    if (TAIL > 0 && !in_tail) page_end = min(page_end, uint(tail_starts[sequence]));
    uint physical = pages[sequence * layout[3] + page_index] * layout[1]
        + position % layout[1];
    size_t address = in_tail
        ? (size_t(sequence) * HK + kh) * TAIL + position - uint(tail_starts[sequence])
        : size_t(kh) * layout[0] + physical;
    auto source_keys = in_tail ? tail_keys : keys;
    auto source_values = in_tail ? tail_values : values;
    for (; position < page_end; ++position, ++address) {
        float key[DK / 32];
        float value[DV / 32];
        for (uint i = 0; i < DK / 32; ++i)
            key[i] = float(source_keys[address * DK + lane + 32 * i]);
        for (uint i = 0; i < DV / 32; ++i)
            value[i] = float(source_values[address * DV + lane + 32 * i]);
        for (uint h = 0; h < HEADS; ++h) {
            float score = 0.0f;
            for (uint i = 0; i < DK / 32; ++i) score += query[h][i] * key[i];
            score = simd_sum(score) * scale[0];
            float next_maximum = max(maximum[h], score);
            float previous_weight = exp(maximum[h] - next_maximum);
            float weight = exp(score - next_maximum);
            denominator[h] = denominator[h] * previous_weight + weight;
            for (uint i = 0; i < DV / 32; ++i)
                accumulator[h][i] = accumulator[h][i] * previous_weight + weight * value[i];
            maximum[h] = next_maximum;
        }
    }
}
for (uint h = 0; h < HEADS; ++h) {
    size_t output_row = (sequence * HQ + head + h) * TQ + token;
    size_t destination = (output_row * splits + split) * (DV + 2);
    for (uint i = 0; i < DV / 32; ++i)
        partial[destination + lane + 32 * i] = accumulator[h][i];
    if (lane == 0) {
        partial[destination + DV] = maximum[h];
        partial[destination + DV + 1] = denominator[h];
    }
}
