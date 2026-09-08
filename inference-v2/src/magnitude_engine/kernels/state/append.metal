uint i = thread_position_in_grid.x;
constexpr uint NK = B * H * C * K;
if (i >= B * H * C * (K + V)) return;
bool key = i < NK;
uint j = key ? i : i - NK;
uint width = key ? K : V;
uint channel = j % width;
uint position = (j / width) % C;
uint head = (j / width / C) % H;
uint row = j / width / C / H;
int token = int(position) - offsets[row];
if (token >= 0 && token < N) {
    uint source = ((row * H + head) * N + uint(token)) * width + channel;
    output[i] = key ? keys[source] : values[source];
} else {
    output[i] = previous[i];
}
