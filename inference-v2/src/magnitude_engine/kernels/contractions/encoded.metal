template <typename T, int BITS, int PACK>
float magnitude_load(const device T* x, thread float* values) {
    float sum = 0.0f;
    if (BITS == 4) {
        #pragma clang loop unroll(full)
        for (int i = 0; i < PACK; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            values[i] = float(x[i]);
            values[i + 1] = float(x[i + 1]) / 16.0f;
            values[i + 2] = float(x[i + 2]) / 256.0f;
            values[i + 3] = float(x[i + 3]) / 4096.0f;
        }
    } else {
        #pragma clang loop unroll(full)
        for (int i = 0; i < PACK; ++i) { sum += x[i]; values[i] = float(x[i]); }
    }
    return sum;
}

// Reuse the unscaled integer coefficients, not dequantized weights. Four-bit
// masks are q * 2^(4j), q <= 15, j <= 3; eight-bit coefficients are q <= 255.
// Both are exactly representable in half. Their products, affine correction and
// accumulation remain float, in the independent-row operation's original order.
template <int BITS, int PACK, bool REUSE = false>
struct AffinePack {
    uint words[PACK * BITS / 32];
    half coefficients[PACK];
    float scale, bias;

    void load(const device uint* weights, float s, float b) {
        scale = s; bias = b;
        #pragma clang loop unroll(full)
        for (int i = 0; i < PACK * BITS / 32; ++i) words[i] = weights[i];
        if constexpr (REUSE) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < PACK; ++i) {
                if constexpr (BITS == 4) {
                    uint halfword = (words[i / 8] >> (((i / 4) % 2) * 16)) & 0xffff;
                    coefficients[i] = half(halfword & (0xf << ((i % 4) * 4)));
                } else {
                    coefficients[i] = half((words[i / 4] >> ((i % 4) * 8)) & 0xff);
                }
            }
        }
    }

    float dot(const thread float* x, float sum) const {
        float value = 0.0f;
        if (BITS == 4) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < PACK / 4; ++i) {
                if constexpr (REUSE) {
                    value += (x[4 * i] * coefficients[4 * i]
                        + x[4 * i + 1] * coefficients[4 * i + 1]
                        + x[4 * i + 2] * coefficients[4 * i + 2]
                        + x[4 * i + 3] * coefficients[4 * i + 3]);
                } else {
                    uint w = (words[i / 2] >> ((i % 2) * 16)) & 0xffff;
                    value += (x[4 * i] * (w & 0x000f)
                        + x[4 * i + 1] * (w & 0x00f0)
                        + x[4 * i + 2] * (w & 0x0f00)
                        + x[4 * i + 3] * (w & 0xf000));
                }
            }
        } else {
            #pragma clang loop unroll(full)
            for (int i = 0; i < PACK; ++i) {
                float coefficient = REUSE ? float(coefficients[i])
                    : float((words[i / 4] >> ((i % 4) * 8)) & 0xff);
                value += x[i] * coefficient;
            }
        }
        return scale * value + sum * bias;
    }
};
