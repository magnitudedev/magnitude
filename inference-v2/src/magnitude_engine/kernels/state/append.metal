// Functional storage update: every output has exactly one writer, inputs are read-only.
uint i = thread_position_in_grid.x;
if (i >= SIZE) return;
bool key = i < BATCH * HEADS * CAPACITY * DK;
uint j = key ? i : i - BATCH * HEADS * CAPACITY * DK;
uint width = key ? DK : DV;
uint channel = j % width;
uint position = (j / width) % CAPACITY;
uint head = (j / width / CAPACITY) % HEADS;
uint row = j / width / CAPACITY / HEADS;
int token = int(position) - offsets[row];
if (token >= 0 && token < COUNT) {
    uint source = ((row * HEADS + head) * COUNT + uint(token)) * width + channel;
    output[i] = key ? keys[source] : values[source];
} else {
    output[i] = previous[i];
}
