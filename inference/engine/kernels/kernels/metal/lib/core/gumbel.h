// The sampler's Gumbel noise (`sample_rows`, `gumbel_rows`): Philox4x32-10 on
// counter (token, draw[3], draw[4], draw[5]) and key (draw[1], draw[2]), then
// −log(−log u) with u = ((x >> 9) + 0.5) · 2⁻²³; zero unless draw[0] == 1.
// A draw is the row's six words, `stride` words apart. Every kernel that
// scores tokens for the sampler adds this noise, so their scores agree bit for
// bit.

namespace gumbel {

inline uint2 multiply(uint left, uint right) {
    ulong product = ulong(left) * ulong(right);
    return uint2(uint(product >> 32), uint(product));
}

inline float noise(uint token, device const uint *draw, ulong stride) {
    if (draw[0] != 1u)
        return 0.0f;
    uint4 counter(token, draw[3 * stride], draw[4 * stride], draw[5 * stride]);
    uint2 key(draw[stride], draw[2 * stride]);
    for (uint round = 0; round < 10; ++round) {
        uint2 p0 = multiply(3528531795u, counter.x);
        uint2 p1 = multiply(3449720151u, counter.z);
        counter = uint4(p1.x ^ counter.y ^ key.x, p1.y, p0.x ^ counter.w ^ key.y, p0.y);
        key += uint2(2654435769u, 3144134277u);
    }
    float uniform = (float(counter.x >> 9) + 0.5f) * 0.00000011920928955078125f;
    return -metal::log(-metal::log(uniform));
}

} // namespace gumbel
