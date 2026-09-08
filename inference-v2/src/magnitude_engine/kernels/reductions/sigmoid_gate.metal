uint i = thread_position_in_grid.x;
if (i >= N) return;
T x = gates[SHARED ? i / WIDTH : i];
// Preserve MLX's dtype arithmetic and its precise unary exponential.
// Default exp can round differently after fusion, including for BF16.
T exponential = T(metal::precise::exp(metal::abs(x)));
auto y = 1 / (1 + exponential);
T gate = T((x < 0) ? y : 1 - y);
output[i] = values[i] * gate;
