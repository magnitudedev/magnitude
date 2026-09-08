template <typename T, int BITS, int PACK>
float magnitude_load(const device T* x, thread float* values) {
    float sum = 0.0f;
    if (BITS == 4) {
        for (int i = 0; i < PACK; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            values[i] = float(x[i]);
            values[i + 1] = float(x[i + 1]) / 16.0f;
            values[i + 2] = float(x[i + 2]) / 256.0f;
            values[i + 3] = float(x[i + 3]) / 4096.0f;
        }
    } else {
        for (int i = 0; i < PACK; ++i) { sum += x[i]; values[i] = float(x[i]); }
    }
    return sum;
}

// Retain the encoded pack across input rows. Dequantizing first would change
// the affine bias and multiply/add rounding of the independent-row operation.
template <int BITS, int PACK>
struct AffinePack {
    uint words[PACK * BITS / 32];
    float scale, bias;

    void load(const device uint* weights, float s, float b) {
        scale = s; bias = b;
        for (int i = 0; i < PACK * BITS / 32; ++i) words[i] = weights[i];
    }

    float dot(const thread float* x, float sum) const {
        float value = 0.0f;
        if (BITS == 4) {
            for (int i = 0; i < PACK / 4; ++i) {
                uint w = (words[i / 2] >> ((i % 2) * 16)) & 0xffff;
                value += (x[4 * i] * (w & 0x000f)
                    + x[4 * i + 1] * (w & 0x00f0)
                    + x[4 * i + 2] * (w & 0x0f00)
                    + x[4 * i + 3] * (w & 0xf000));
            }
        } else {
            for (int i = 0; i < PACK; ++i)
                value += x[i] * ((words[i / 4] >> ((i % 4) * 8)) & 0xff);
        }
        return scale * value + sum * bias;
    }
};
