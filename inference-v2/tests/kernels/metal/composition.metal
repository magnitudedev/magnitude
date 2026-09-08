inline float bump(float x) { return x + 2.0f; }
inline float pair_subtract(float current, float partner) { return current - partner; }

// A second row algorithm: no RMS-specific core interface or planner rule.
template<uint WIDTH, uint THREADS, typename Body>
inline void row_transform(const thread Body& body, uint row, uint tid) {
    for (uint column = tid; column < WIDTH; column += THREADS) {
        size_t index = size_t(row) * WIDTH + column;
        float original = body.input(index);
        body.output(index, original * 2.0f + 1.0f, original);
    }
}

template<typename T> inline T identity(T x) { return x; }
