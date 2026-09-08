uint index = thread_position_in_grid.x;
if (index >= ROWS * WIDTH) return;
uint row = index / WIDTH, channel = index % WIDTH;
RouteStep<T, SLOTS, SHARED, decltype(scores), decltype(shared_score)> step{scores, shared_score};
float state = step.initial();
for (uint slot = 0; slot < SLOTS; ++slot)
    state = step.step(state, x[(size_t(row) * SLOTS + slot) * WIDTH + channel], row, slot);
out[index] = step.finish(state);
