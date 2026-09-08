uint index = thread_position_in_grid.x;
if (index >= M * N) return;
uint row = index / N, channel = index % N;
constexpr uint SLOTS = TOPK + SHARED;
float combined = 0;
// Scheduling may sort assignments; numerical combination never does.
for (uint slot = 0; slot < SLOTS; ++slot) {
    T score = SHARED && slot == TOPK ? shared_score[row] : scores[row * TOPK + slot];
    T contribution = T(x[(size_t(row) * SLOTS + slot) * N + channel] * score);
    combined = float(T(combined + float(contribution)));
}
out[index] = T(combined);
