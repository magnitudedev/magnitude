// F16 staging of F32 or BF16 values on Vulkan. The counterpart of the scaled
// staging in `projection.glsl` and `attention/prefill.glsl`, which both take
// their exponent here.
//
// Values whose largest magnitude is `largest` are staged scaled by 2^-e, the
// least exponent e >= 0 that holds the largest below 2^15, so F16 holds every
// one: e = 0 unless `largest` reaches 2^15 (an infinity takes none, its
// F16 is infinite either way). Scaling by a power of two keeps every value
// within F16's range exact, and its consumer takes 2^e back exactly.
int staging_f16_exponent(float largest) {
    if (largest < 32768.0 || isinf(largest))
        return 0;
    int e;
    frexp(largest, e);
    return e - 15;
}
