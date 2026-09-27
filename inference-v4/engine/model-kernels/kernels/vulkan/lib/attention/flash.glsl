// Tile pieces of the streaming ("flash") attention bodies: the gated
// attention prefill (`gated_attention_prefill`) and the vision full attention
// (`qwen_vision_block`). Each subgroup owns a block of 16 query rows of head
// width `w` (a multiple of 16, at most FLASH_MAX_W); keys stream through
// the shared region in tiles of FLASH_KEYS rows as f16 with rows padded to
// w + 8. Per tile the block's scores S = Q K^T are formed in F32 (16x16x16
// cooperative matrices under SEISMIC_HAS_MATRIX, FMAs otherwise) and passed
// through the subgroup's shared scratch to the scalar online softmax
// (`flash_online`), whose f16 probabilities multiply the staged V tile into
// F32 output accumulators. Callers make one pass over the keys per output
// window (a caller constant of at most FLASH_MAX_W columns); when a row's
// running maximum grows, its accumulated outputs are rescaled by a
// component-wise product with a fragment of per-row factors staged in the
// scratch (`flash_rescale`), since the cooperative-matrix fragment layout is
// opaque.
//
// Lane layout of the scalar softmax: lane l owns row l % 16 and keys
// 16 (l / 16) .. + 15 of the tile.
//
// This file is independent of any entry ABI.
#include <seismic/element.glsl>

#define FLASH_KEYS 32u
#define FLASH_MAX_W 256u
// Floats of shared scratch per subgroup: the scores 16 x 32; then the f16 P
// (16 x 32) in its first half, the rescale factors (16 x 16) in its second
// half; the output fragments reuse it.
#define FLASH_SCRATCH_FLOATS 512u
#define FLASH_FACTORS 256u
#define FLASH_INF uintBitsToFloat(0x7f800000u)
// The widest output window a device takes in one pass: the whole head, or 4
// accumulator fragments where the compiler mishandles wider accumulator
// arrays (NVIDIA 580, `SEISMIC_HAS_WIDE_ACCUMULATORS` 0).
#if SEISMIC_HAS_WIDE_ACCUMULATORS
#define FLASH_WINDOW FLASH_MAX_W
#else
#define FLASH_WINDOW 64u
#endif

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
    const uint lane = SEISMIC_LANE;
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
    const uint lane = SEISMIC_LANE;
    [[unroll]] for (uint j = 0u; j < 16u; ++j)
        s[j] = seismic_shared_f32[scratch + (lane % 16u) * 32u + 16u * (lane / 16u) + j];
}

// Publishes the lane's 16 probabilities (f16) as the block's 16 x 32 P
// matrix over its scratch; returns P's shared half. Every lane must hold its
// scores already (`flash_lane_scores`).
uint flash_publish_probabilities(uint scratch, float p[16]) {
    const uint lane = SEISMIC_LANE;
    const uint p_half = 2u * scratch;
    subgroupBarrier();
    [[unroll]] for (uint j = 0u; j < 16u; ++j)
        seismic_shared_u16[p_half + (lane % 16u) * 32u + 16u * (lane / 16u) + j] = seismic_f32_to_f16(p[j]);
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    return p_half;
}

// The online softmax state of the lane's row: its running maximum (shared by
// both lanes of the row) and the lane's half of the running denominator.
struct flash_softmax {
    float maximum;
    float denominator;
};

flash_softmax flash_softmax_start() { return flash_softmax(-FLASH_INF, 0.0); }

// Turns the lane's 16 masked, scaled (log2) scores into probabilities
// exp2(s - m) against the row's new running maximum m, folds them into the
// denominator, and returns the factor exp2(m_old - m) that rescales the row's
// earlier accumulations (1 while nothing is seen).
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

// The row's whole denominator (both lanes' halves), on both lanes.
float flash_denominator(flash_softmax state) {
    return state.denominator + subgroupShuffleXor(state.denominator, 16u);
}

// The output accumulators of a block over one window of `window` columns
// (columns [column0, column0 + window) of w; window a compile-time constant
// dividing w, at most FLASH_MAX_W): window / 16 fragments, or window / 2
// columns per lane (row lane % 16, columns (lane / 16) window / 2 ..). A
// window narrower than w costs one more score pass per extra window.
uint flash_window(const uint w, const uint window) { return min(window, w); }

struct flash_output {
#if SEISMIC_HAS_MATRIX
    FLASH_FRAGMENT_C c[FLASH_MAX_W / 16u];
#else
    float c[FLASH_MAX_W / 2u];
#endif
};

// Clears the fragments (columns) a window uses; the others stay unused.
void flash_output_clear(const uint w, const uint window, inout flash_output o) {
#if SEISMIC_HAS_MATRIX
    [[unroll]] for (uint d = 0u; d < FLASH_MAX_W / 16u; ++d)
        if (d < flash_window(w, window) / 16u)
            o.c[d] = FLASH_FRAGMENT_C(0.0);
#else
    [[unroll]] for (uint d = 0u; d < FLASH_MAX_W / 2u; ++d)
        if (d < flash_window(w, window) / 2u)
            o.c[d] = 0.0;
#endif
}

// O <- diag(alpha) O for the lane rows' factors `alpha` (from
// `flash_online`). Skipped when no row's maximum grew. Every lane of the
// subgroup calls it, after `flash_publish_probabilities`.
void flash_rescale(uint scratch, float alpha, const uint w, const uint window, inout flash_output o) {
    if (seismic_subgroup_all(alpha == 1.0))
        return;
#if SEISMIC_HAS_MATRIX
    // The factors as a 16 x 16 accumulator fragment, row r all alpha_r: each
    // lane writes half of its row.
    const uint lane = SEISMIC_LANE;
    [[unroll]] for (uint i = 0u; i < 8u; ++i)
        seismic_shared_f32[scratch + FLASH_FACTORS + (lane % 16u) * 16u + 8u * (lane / 16u) + i] = alpha;
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    FLASH_FRAGMENT_C factors;
    coopMatLoad(factors, seismic_shared_f32, scratch + FLASH_FACTORS, 16u, gl_CooperativeMatrixLayoutRowMajor);
    [[unroll]] for (uint d = 0u; d < FLASH_MAX_W / 16u; ++d)
        if (d < flash_window(w, window) / 16u)
            o.c[d] = o.c[d] * factors;
    subgroupBarrier();
#else
    [[unroll]] for (uint d = 0u; d < FLASH_MAX_W / 2u; ++d)
        if (d < flash_window(w, window) / 2u)
            o.c[d] *= alpha;
#endif
}

// O += P V over the window's columns of the V tile at shared half `v_base`;
// P at shared half `p_half` (row pitch 32).
void flash_accumulate(uint p_half, uint v_base, const uint w, const uint window, uint column0, inout flash_output o) {
    const uint pitch = flash_pitch(w);
    const uint width = flash_window(w, window);
#if SEISMIC_HAS_MATRIX
    FLASH_FRAGMENT_A p[2];
    coopMatLoad(p[0], seismic_shared_f16, p_half, 32u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(p[1], seismic_shared_f16, p_half + 16u, 32u, gl_CooperativeMatrixLayoutRowMajor);
    [[unroll]] for (uint d = 0u; d < FLASH_MAX_W / 16u; ++d) {
        if (d < width / 16u) {
            [[unroll]] for (uint j = 0u; j < 2u; ++j) {
                FLASH_FRAGMENT_B v;
                coopMatLoad(v, seismic_shared_f16, v_base + j * 16u * pitch + column0 + 16u * d, pitch,
                    gl_CooperativeMatrixLayoutRowMajor);
                o.c[d] = coopMatMulAdd(p[j], v, o.c[d]);
            }
        }
    }
#else
    const uint lane = SEISMIC_LANE;
    const uint r = lane % 16u, h = lane / 16u;
    const uint columns = width / 2u;
    for (uint j = 0u; j < FLASH_KEYS; ++j) {
        const float p = seismic_f16_to_f32(seismic_shared_u16[p_half + r * 32u + j]);
        [[unroll]] for (uint c = 0u; c < FLASH_MAX_W / 2u; c += 2u) {
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
// row `flash_output_row(q)`, column `flash_output_column(w, window, column0,
// q)` (of w). The matrix path publishes each fragment through the subgroup's
// scratch: call `flash_output_value` for q = 0 .. window / 2 - 1 in order
// from every lane, after a barrier that frees the scratch.
uint flash_output_row(uint q) {
#if SEISMIC_HAS_MATRIX
    return (8u * SEISMIC_LANE + q % 8u) / 16u;
#else
    return SEISMIC_LANE % 16u;
#endif
}

uint flash_output_column(const uint w, const uint window, uint column0, uint q) {
#if SEISMIC_HAS_MATRIX
    return column0 + 16u * (q / 8u) + (8u * SEISMIC_LANE + q % 8u) % 16u;
#else
    return column0 + (SEISMIC_LANE / 16u) * (flash_window(w, window) / 2u) + q;
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
    return seismic_shared_f32[scratch + 8u * SEISMIC_LANE + q % 8u];
#else
    return o.c[q];
#endif
}
