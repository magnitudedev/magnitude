// Tile pieces of the streaming ("flash") attention bodies: the gated
// attention prefill (`qwen_attention_prefill`) and the vision full attention
// (`qwen_vision_block`). Each subgroup owns a block of 16 query rows of head
// width `w` (a multiple of 16, at most FLASH_MAX_W); keys stream through
// the shared region in tiles of FLASH_KEYS rows as f16 with rows padded to
// w + 8. Per tile the block's scores S = Q K^T are formed in F32 (16x16x16
// cooperative matrices under SEISMIC_HAS_MATRIX, FMAs otherwise) and passed
// through the subgroup's shared scratch to the caller's scalar softmax,
// whose f16 probabilities multiply the staged V tile into F32 output
// accumulators. Callers run a row-maxima pass, then an accumulation pass per
// output window (FLASH_OUT_W columns), so the accumulators are never
// rescaled: the cooperative-matrix fragment layout is opaque.
//
// Lane layout of the scalar softmax: lane l owns row l % 16 and keys
// 16 (l / 16) .. + 15 of the tile.
//
// This file is independent of any entry ABI.
#include "common/element.glsl"

#define FLASH_KEYS 32u
#define FLASH_MAX_W 256u
// Floats of shared scratch per subgroup (scores 16 x 32; the output
// fragments reuse it).
#define FLASH_SCRATCH_FLOATS 512u

#if SEISMIC_HAS_MATRIX
#define FLASH_FRAGMENT_A coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define FLASH_FRAGMENT_B coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define FLASH_FRAGMENT_C coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#endif

layout(buffer_reference, scalar, buffer_reference_align = 2) readonly buffer flash_f16_rows { float16_t v[]; };

// Row pitch of a staged tile, in f16 elements.
uint flash_pitch(const uint w) { return w + 8u; }

// Copies rows [first, first + FLASH_KEYS) of a plane of `kind` (bf16 or f16
// elements; row t at element t * row_stride + column, w contiguous) into the
// f16 tile at shared half `base`; rows at or past `end` are zero. Every
// invocation of the workgroup takes part.
void flash_stage(const int kind, uint64_t plane, uint64_t row_stride, uint64_t column, int first, int end, const uint w,
    uint base) {
    const uint pieces = w / 8u;
    for (uint item = gl_LocalInvocationIndex; item < FLASH_KEYS * pieces; item += gl_WorkGroupSize.x) {
        const uint k = item / pieces;
        const uint c = (item % pieces) * 8u;
        const int t = first + int(k);
        uvec4 bits = uvec4(0u);
        if (t < end) {
            bits = element_uvec4_at(plane + (uint64_t(t) * row_stride + column + c) * 2ul);
            if (kind == ELEMENT_BF16) {
                vec4 even, odd;
                element_split8(ELEMENT_BF16, bits, even, odd);
                bits = element_pack8(ELEMENT_F16, even, odd);
            }
        }
        seismic_shared_uvec4[(base + k * flash_pitch(w) + c) / 8u] = bits;
    }
}

// The block's scores S = Q K^T (16 rows x FLASH_KEYS keys, F32) into its
// shared scratch (row-major, 32 floats per row). `q` is the block's first
// query row (f16 elements, rows `q_stride` apart); the K tile is at shared
// half `k_base`.
void flash_scores(uint64_t q, uint64_t q_stride, const uint w, uint k_base, uint scratch) {
    const uint pitch = flash_pitch(w);
#if SEISMIC_HAS_MATRIX
    FLASH_FRAGMENT_C s[2];
    s[0] = FLASH_FRAGMENT_C(0.0);
    s[1] = FLASH_FRAGMENT_C(0.0);
    [[unroll]] for (uint d = 0u; d < FLASH_MAX_W; d += 16u) {
        if (d < w) {
            FLASH_FRAGMENT_A a;
            coopMatLoad(a, flash_f16_rows(q).v, d, uint(q_stride), gl_CooperativeMatrixLayoutRowMajor);
            [[unroll]] for (uint j = 0u; j < 2u; ++j) {
                FLASH_FRAGMENT_B b;
                coopMatLoad(b, seismic_shared_f16, k_base + j * 16u * pitch + d, pitch, gl_CooperativeMatrixLayoutColumnMajor);
                s[j] = coopMatMulAdd(a, b, s[j]);
            }
        }
    }
    subgroupBarrier();
    coopMatStore(s[0], seismic_shared_f32, scratch, 32u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatStore(s[1], seismic_shared_f32, scratch + 16u, 32u, gl_CooperativeMatrixLayoutRowMajor);
    subgroupMemoryBarrierShared();
    subgroupBarrier();
#else
    const uint lane = gl_SubgroupInvocationID;
    const uint r = lane % 16u, h = lane / 16u;
    float s[16];
    [[unroll]] for (uint j = 0u; j < 16u; ++j)
        s[j] = 0.0;
    for (uint d = 0u; d < w; d += 2u) {
        const vec2 x = vec2(element_at(ELEMENT_F16, q, uint64_t(r) * q_stride + d),
            element_at(ELEMENT_F16, q, uint64_t(r) * q_stride + d + 1u));
        [[unroll]] for (uint j = 0u; j < 16u; ++j) {
            const vec2 k = unpackHalf2x16(seismic_shared_u32[(k_base + (16u * h + j) * pitch + d) / 2u]);
            s[j] = seismic_fma_rn(x.x, k.x, s[j]);
            s[j] = seismic_fma_rn(x.y, k.y, s[j]);
        }
    }
    subgroupBarrier();
    [[unroll]] for (uint j = 0u; j < 16u; ++j)
        seismic_shared_f32[scratch + r * 32u + 16u * h + j] = s[j];
    subgroupMemoryBarrierShared();
    subgroupBarrier();
#endif
}

// The lane's 16 raw scores of the last `flash_scores`.
void flash_lane_scores(uint scratch, out float s[16]) {
    const uint lane = gl_SubgroupInvocationID;
    [[unroll]] for (uint j = 0u; j < 16u; ++j)
        s[j] = seismic_shared_f32[scratch + (lane % 16u) * 32u + 16u * (lane / 16u) + j];
}

// Publishes the lane's 16 probabilities (f16) as the block's 16 x 32 P
// matrix over its scratch; returns P's shared half. Every lane must hold its
// scores already (`flash_lane_scores`).
uint flash_publish_probabilities(uint scratch, float p[16]) {
    const uint lane = gl_SubgroupInvocationID;
    const uint p_half = 2u * scratch;
    subgroupBarrier();
    [[unroll]] for (uint j = 0u; j < 16u; ++j)
        seismic_shared_f16[p_half + (lane % 16u) * 32u + 16u * (lane / 16u) + j] = float16_t(p[j]);
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    return p_half;
}

// The output accumulators of a block over one window of at most
// FLASH_OUT_W columns (columns [column0, column0 + FLASH_OUT_W) of w): wide
// heads accumulate their columns window by window, each over the whole key
// walk, so a subgroup holds at most 4 accumulator fragments: the NVIDIA 580
// compiler returns wrong outputs for some larger accumulator arrays (16
// fragments; 8 declared with 2 used), while 4 are reliable at every tested
// width. Every window is `flash_window(w)` columns, a compile-time constant,
// as window / 16 fragments, or window / 2 columns per lane (row lane % 16,
// columns (lane / 16) window / 2 ..). So w is at most FLASH_OUT_W or a
// multiple of it. (Wider windows cost fewer score passes: revisit per device.)
#define FLASH_OUT_W 64u

uint flash_window(const uint w) { return min(FLASH_OUT_W, w); }

struct flash_output {
#if SEISMIC_HAS_MATRIX
    FLASH_FRAGMENT_C c[FLASH_OUT_W / 16u];
#else
    float c[FLASH_OUT_W / 2u];
#endif
};

// Clears the fragments (columns) a window of w uses; the others stay unused.
void flash_output_clear(const uint w, inout flash_output o) {
#if SEISMIC_HAS_MATRIX
    [[unroll]] for (uint d = 0u; d < FLASH_OUT_W / 16u; ++d)
        if (d < flash_window(w) / 16u)
            o.c[d] = FLASH_FRAGMENT_C(0.0);
#else
    [[unroll]] for (uint d = 0u; d < FLASH_OUT_W / 2u; ++d)
        if (d < flash_window(w) / 2u)
            o.c[d] = 0.0;
#endif
}

// O += P V over the window's columns of the V tile at shared half `v_base`;
// P at shared half `p_half` (row pitch 32).
void flash_accumulate(uint p_half, uint v_base, const uint w, uint column0, inout flash_output o) {
    const uint pitch = flash_pitch(w);
    const uint window = flash_window(w);
#if SEISMIC_HAS_MATRIX
    FLASH_FRAGMENT_A p[2];
    coopMatLoad(p[0], seismic_shared_f16, p_half, 32u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(p[1], seismic_shared_f16, p_half + 16u, 32u, gl_CooperativeMatrixLayoutRowMajor);
    [[unroll]] for (uint d = 0u; d < FLASH_OUT_W / 16u; ++d) {
        if (d < window / 16u) {
            [[unroll]] for (uint j = 0u; j < 2u; ++j) {
                FLASH_FRAGMENT_B v;
                coopMatLoad(v, seismic_shared_f16, v_base + j * 16u * pitch + column0 + 16u * d, pitch,
                    gl_CooperativeMatrixLayoutRowMajor);
                o.c[d] = coopMatMulAdd(p[j], v, o.c[d]);
            }
        }
    }
#else
    const uint lane = gl_SubgroupInvocationID;
    const uint r = lane % 16u, h = lane / 16u;
    const uint columns = window / 2u;
    for (uint j = 0u; j < FLASH_KEYS; ++j) {
        const float p = float(seismic_shared_f16[p_half + r * 32u + j]);
        [[unroll]] for (uint c = 0u; c < FLASH_OUT_W / 2u; c += 2u) {
            if (c < columns) {
                const vec2 v = unpackHalf2x16(seismic_shared_u32[(v_base + j * pitch + column0 + h * columns + c) / 2u]);
                o.c[c] = seismic_fma_rn(p, v.x, o.c[c]);
                o.c[c + 1u] = seismic_fma_rn(p, v.y, o.c[c + 1u]);
            }
        }
    }
#endif
}

// The lane's outputs of the window: value q (q < window / 2) sits at block
// row `flash_output_row(q)`, column `flash_output_column(w, column0, q)` (of
// w). The matrix path publishes each fragment through the subgroup's
// scratch: call `flash_output_value` for q = 0 .. window / 2 - 1 in order
// from every lane, after a barrier that frees the scratch.
uint flash_output_row(uint q) {
#if SEISMIC_HAS_MATRIX
    return (8u * gl_SubgroupInvocationID + q % 8u) / 16u;
#else
    return gl_SubgroupInvocationID % 16u;
#endif
}

uint flash_output_column(const uint w, uint column0, uint q) {
#if SEISMIC_HAS_MATRIX
    return column0 + 16u * (q / 8u) + (8u * gl_SubgroupInvocationID + q % 8u) % 16u;
#else
    return column0 + (gl_SubgroupInvocationID / 16u) * (flash_window(w) / 2u) + q;
#endif
}

float flash_output_value(inout flash_output o, uint scratch, uint q) {
#if SEISMIC_HAS_MATRIX
    if (q % 8u == 0u) {
        subgroupBarrier();
        coopMatStore(o.c[q / 8u], seismic_shared_f32, scratch, 16u, gl_CooperativeMatrixLayoutRowMajor);
        subgroupMemoryBarrierShared();
        subgroupBarrier();
    }
    return seismic_shared_f32[scratch + 8u * gl_SubgroupInvocationID + q % 8u];
#else
    return o.c[q];
#endif
}
