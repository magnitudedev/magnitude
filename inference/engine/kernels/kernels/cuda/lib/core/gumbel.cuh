// The sampler's Gumbel noise (`sample_rows`, `gumbel_rows`): Philox4x32-10 on
// counter (token, draw[3], draw[4], draw[5]) and key (draw[1], draw[2]), then
// −log(−log u) with u = ((x >> 9) + 0.5) · 2⁻²³; zero unless draw[0] == 1.
// A draw is the row's six words, `stride` words apart. Every kernel that
// scores tokens for the sampler adds this noise, so their scores agree bit for
// bit.

namespace gumbel {

__device__ __forceinline__ float noise(unsigned token, const unsigned *draw, unsigned long long stride) {
    if (draw[0] != 1u)
        return 0.0f;
    unsigned c0 = token, c1 = draw[3 * stride], c2 = draw[4 * stride], c3 = draw[5 * stride];
    unsigned k0 = draw[stride], k1 = draw[2 * stride];
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        const unsigned hi0 = __umulhi(3528531795u, c0), lo0 = 3528531795u * c0;
        const unsigned hi1 = __umulhi(3449720151u, c2), lo1 = 3449720151u * c2;
        const unsigned next0 = hi1 ^ c1 ^ k0;
        const unsigned next2 = hi0 ^ c3 ^ k1;
        c0 = next0;
        c1 = lo1;
        c2 = next2;
        c3 = lo0;
        k0 += 2654435769u;
        k1 += 3144134277u;
    }
    const float uniform = ((float)(c0 >> 9) + 0.5f) * 0.00000011920928955078125f;
    return -logf(-logf(uniform));
}

} // namespace gumbel
