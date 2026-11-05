// `wide_accumulators`: the flash-attention body (the engine's
// `vulkan/lib/attention/flash.glsl` and its prefill loop) on its
// cooperative-matrix path with the whole head as one output window: W / 16
// output fragments of a 16-fragment accumulator array, 8 at W = 128 and 16 at
// W = 256.
// One subgroup scores its 16 query rows against N keys streamed through
// shared memory in tiles of 32, runs the online softmax, and accumulates
// P V. Buffers: q (16 x W f16), k and v (N x W f16), out (16 x W f32); the
// word: N. Its extensions are `WIDE_EXTENSIONS` (`probe.rs`), declared
// before the head.

#define FLASH_KEYS 32u
#define FLASH_MAX_W 256u
#define FLASH_SCRATCH_FLOATS 512u
#define FLASH_FACTORS 256u
#define FLASH_INF uintBitsToFloat(0x7f800000u)
#define LANE gl_SubgroupInvocationID
#define FRAGMENT_A coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define FRAGMENT_B coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define FRAGMENT_C coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define PITCH (W + 8u)
#define SHARED_BYTES (FLASH_KEYS * PITCH * 2u + FLASH_SCRATCH_FLOATS * 4u)

shared probe_shared_f32_t { float shared_f32[SHARED_BYTES / 4u]; };
shared probe_shared_f16_t { float16_t shared_f16[SHARED_BYTES / 2u]; };
shared probe_shared_u16_t { uint16_t shared_u16[SHARED_BYTES / 2u]; };
shared probe_shared_uvec4_t { uvec4 shared_uvec4[SHARED_BYTES / 16u]; };

layout(buffer_reference, scalar, buffer_reference_align = 2) readonly buffer flash_f16_rows { float16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer flash_uvec4_rows { uvec4 v[]; };

FRAGMENT_C matrix_zero() { return FRAGMENT_C(0.0); }

void flash_stage(uint64_t plane, int first, int end) {
    const uint pieces = W / 8u;
    for (uint item = gl_LocalInvocationIndex; item < FLASH_KEYS * pieces; item += gl_WorkGroupSize.x) {
        const uint k = item / pieces;
        const uint c = (item % pieces) * 8u;
        const int t = first + int(k);
        uvec4 bits = uvec4(0u);
        if (t < end)
            bits = flash_uvec4_rows(plane + (uint64_t(t) * W + c) * 2ul).v[0];
        shared_uvec4[(k * PITCH + c) / 8u] = bits;
    }
}

struct flash_scores_state {
    FRAGMENT_C s[2];
};

void flash_scores_add(uint64_t q, inout flash_scores_state state) {
    [[unroll]] for (uint d = 0u; d < FLASH_MAX_W; d += 16u) {
        if (d < W) {
            FRAGMENT_A a;
            coopMatLoad(a, flash_f16_rows(q).v, d, W, gl_CooperativeMatrixLayoutRowMajor);
            [[unroll]] for (uint j = 0u; j < 2u; ++j) {
                FRAGMENT_B b;
                coopMatLoad(b, shared_f16, j * 16u * PITCH + d, PITCH, gl_CooperativeMatrixLayoutColumnMajor);
                state.s[j] = coopMatMulAdd(a, b, state.s[j]);
            }
        }
    }
}

void flash_scores_publish(flash_scores_state state, uint scratch) {
    subgroupBarrier();
    coopMatStore(state.s[0], shared_f32, scratch, 32u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatStore(state.s[1], shared_f32, scratch + 16u, 32u, gl_CooperativeMatrixLayoutRowMajor);
    subgroupMemoryBarrierShared();
    subgroupBarrier();
}

struct flash_softmax {
    float maximum;
    float denominator;
};

float flash_online(inout flash_softmax state, inout float s[16]) {
    float tile = s[0];
    [[unroll]] for (uint j = 1u; j < 16u; ++j)
        tile = max(tile, s[j]);
    tile = max(tile, subgroupShuffleXor(tile, 16u));
    const float next = max(state.maximum, tile);
    const bool seen = next > -FLASH_INF;
    const float alpha = seen ? exp2(state.maximum - next) : 1.0;
    float sum = 0.0;
    [[unroll]] for (uint j = 0u; j < 16u; ++j) {
        s[j] = seen ? exp2(s[j] - next) : 0.0;
        sum += s[j];
    }
    state.denominator = state.denominator * alpha + sum;
    state.maximum = next;
    return alpha;
}

uint flash_publish_probabilities(uint scratch, float p[16]) {
    const uint p_half = 2u * scratch;
    subgroupBarrier();
    [[unroll]] for (uint j = 0u; j < 16u; ++j)
        shared_u16[p_half + (LANE % 16u) * 32u + 16u * (LANE / 16u) + j] = uint16_t(packHalf2x16(vec2(p[j], 0.0)) & 0xffffu);
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    return p_half;
}

struct flash_output {
    FRAGMENT_C c[FLASH_MAX_W / 16u];
};

void flash_rescale(uint scratch, float alpha, inout flash_output o) {
    if (subgroupAll(alpha == 1.0))
        return;
    [[unroll]] for (uint i = 0u; i < 8u; ++i)
        shared_f32[scratch + FLASH_FACTORS + (LANE % 16u) * 16u + 8u * (LANE / 16u) + i] = alpha;
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    FRAGMENT_C factors;
    coopMatLoad(factors, shared_f32, scratch + FLASH_FACTORS, 16u, gl_CooperativeMatrixLayoutRowMajor);
    [[unroll]] for (uint d = 0u; d < FLASH_MAX_W / 16u; ++d)
        if (d < W / 16u)
            o.c[d] = o.c[d] * factors;
    subgroupBarrier();
}

void flash_accumulate(uint p_half, inout flash_output o) {
    FRAGMENT_A p[2];
    coopMatLoad(p[0], shared_f16, p_half, 32u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(p[1], shared_f16, p_half + 16u, 32u, gl_CooperativeMatrixLayoutRowMajor);
    [[unroll]] for (uint d = 0u; d < FLASH_MAX_W / 16u; ++d) {
        if (d < W / 16u) {
            [[unroll]] for (uint j = 0u; j < 2u; ++j) {
                FRAGMENT_B v;
                coopMatLoad(v, shared_f16, j * 16u * PITCH + 16u * d, PITCH, gl_CooperativeMatrixLayoutRowMajor);
                o.c[d] = coopMatMulAdd(p[j], v, o.c[d]);
            }
        }
    }
}

float flash_output_value(inout flash_output o, uint scratch, uint q) {
    if (q % 8u == 0u) {
        subgroupBarrier();
        coopMatStore(o.c[q / 8u], shared_f32, scratch, 16u, gl_CooperativeMatrixLayoutRowMajor);
        subgroupMemoryBarrierShared();
        subgroupBarrier();
    }
    return shared_f32[scratch + 8u * LANE + q % 8u];
}

void main() {
    const uint64_t q = PROBE_ARGUMENT(0);
    const uint64_t k = PROBE_ARGUMENT(1);
    const uint64_t v = PROBE_ARGUMENT(2);
    probe_f32 out_rows = probe_f32(PROBE_ARGUMENT(3));
    const int n = int(PROBE_ARGUMENT(5));
    const uint scratch = (FLASH_KEYS * PITCH) / 2u;
    // log2(e) / sqrt(W): the scores' scale in the base-2 softmax.
    const float scale = 1.4426950408889634 * inversesqrt(float(W));
    flash_output o;
    [[unroll]] for (uint d = 0u; d < FLASH_MAX_W / 16u; ++d)
        if (d < W / 16u)
            o.c[d] = matrix_zero();
    flash_softmax softmax = flash_softmax(-FLASH_INF, 0.0);
    for (int first = 0; first < n; first += int(FLASH_KEYS)) {
        flash_scores_state scores;
        scores.s[0] = matrix_zero();
        scores.s[1] = matrix_zero();
        barrier();
        flash_stage(k, first, n);
        barrier();
        flash_scores_add(q, scores);
        flash_scores_publish(scores, scratch);
        float s[16];
        [[unroll]] for (uint j = 0u; j < 16u; ++j) {
            const int t = first + int(16u * (LANE / 16u) + j);
            s[j] = t < n ? shared_f32[scratch + (LANE % 16u) * 32u + 16u * (LANE / 16u) + j] * scale : -FLASH_INF;
        }
        const float alpha = flash_online(softmax, s);
        const uint p_half = flash_publish_probabilities(scratch, s);
        flash_rescale(scratch, alpha, o);
        barrier();
        flash_stage(v, first, n);
        barrier();
        flash_accumulate(p_half, o);
    }
    const float denominator = softmax.denominator + subgroupShuffleXor(softmax.denominator, 16u);
    barrier();
    [[unroll]] for (uint index = 0u; index < W / 2u; ++index) {
        const float value = flash_output_value(o, scratch, index);
        const uint r = (8u * LANE + index % 8u) / 16u;
        const uint column = 16u * (index / 8u) + (8u * LANE + index % 8u) % 16u;
        out_rows.v[r * W + column] = value / subgroupShuffle(denominator, r % 16u);
    }
}
