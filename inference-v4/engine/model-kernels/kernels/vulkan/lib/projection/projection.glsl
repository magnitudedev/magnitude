// The packed projection family (K1): y[m, n] = epilogue(sum_k x[m, k] * W[n, k])
// where x is produced by a prologue from the entry's inputs. The counterpart
// of `metal/lib/projection/projection.h` and `cuda/lib/projection/projection.cuh`, with the
// same vocabulary (organization spec §B3), prefixed `projection_` in GLSL.
//
// Row maps: AllRows (table 0) or SelectedRows (an `out_rows` i32 table).
//
// Prologues (`projection_prologue`, kind PROJECTION_{PLAIN,RMS,GATED_RMS,
// GROUPED}) produce one activation value x[m, k], already rounded to the
// activation type A exactly as the entry's portable body publishes it:
//   Plain     x = A[row(m), k]
//   Rms       x = round_A(r[row(m), k] * inverse(m) * norm[k])
//   GatedRms  per value head h = k / W:
//             x = round_A(round_A(mixed * inverse(m, h) * norm[k % W]) * round_A(silu(z[row(m), k])))
//   Grouped   x = A[order[m], k], zero for a padding row (order -1)
// In the GEMV row class (M <= 8) every workgroup computes its rows' norm
// inverses and applies the prologue while staging, so a decode projection is
// one launch. The GEMM class (M > 8) runs a pre-pass launch
// (`projection_stage`) that writes the whole prologue output in A to scratch,
// read by the GEMM as a Plain operand: no GEMM tile repeats the prologue.
//
// Epilogues (`projection_epilogue`, kind PROJECTION_{STORE,RESIDUAL,SILU_MUL})
// receive the F32 dot product:
//   Store<E>  y = round_E(acc) (logits are Store<f32>)
//   Residual  y = residual[row(m), n] + round_A(acc)                    (F32)
//   SiluMul   y = round_A(round_A(silu(round_A(gate))) * round_A(up))
//
// A segmented projection is one launch over several weight tensors (each
// with its own packet kind and destination). Every workgroup belongs to one
// segment; entries map their workgroup index to a segment and call the GEMV
// or GEMM body with that segment's weights and epilogue.
//
// Row classes pick the launch:
// - GEMV (M <= 8): lane groups own weight rows, lanes own packets, and each
//   weight packet is decoded once and applied to every activation row with F32
//   FMAs. The activations are staged once per workgroup in the shared region
//   (A storage) with per-16-element sums for the factored biases.
// - GEMM (M > 8): TM x TN output tiles over subgroups of 32 x 32, stepping K
//   by 32. Activations are staged as f16 and weights decoded to f16 once per
//   tile. Under SEISMIC_HAS_MATRIX (RDNA3/4) cooperative-matrix 16x16x16
//   multiplies accumulate in F32; otherwise every lane runs a 4 x 8 register
//   tile of F32 FMAs over the same staged operands. Small-N outputs may split
//   K.
//
// GLSL has no templates. Operands are structs whose kind fields are
// compile-time constants at every construction, and the bodies take their
// shape (rows per lane group, lanes, activation-row bound, tile) as constant
// arguments: the driver inlines each call and folds every kind switch and
// constant loop, which instantiates the body exactly as a template would.
//
// Shared region use (bytes; entries size `shared_bytes` from these):
//   GEMV   PROJECTION_GEMV_SQUARES_BYTES (row-norm partial squares), then the
//          staging: min(8 * ceil_div(K, 32), 288) * 72
//   GEMM   (TM + TN) * 80 (the f16 A and B tiles, rows padded to 40)
//   stage pre-pass: 128 (one partial per subgroup)
//
// This file is independent of any entry ABI.
#include <seismic/packets.glsl>
#include "../core/reduce.glsl"

#include "stage.glsl"

#define PROJECTION_STORE 0
#define PROJECTION_RESIDUAL 1
#define PROJECTION_SILU_MUL 2

struct projection_epilogue {
    int kind;
    int act;            // Store: the stored element E; Residual/SiluMul: A
    uint64_t y;
    uint64_t y0;
    uint64_t y1;
    uint64_t column;    // Store: first destination column
    uint64_t residual;  // Residual: F32 residual rows
    uint64_t r0;
    uint64_t r1;
    uint64_t rows;      // Residual: row map of the residual (0: AllRows)
};

projection_epilogue projection_store(const int element, uint64_t y, uint64_t stride0, uint64_t stride1, uint64_t column) {
    return projection_epilogue(PROJECTION_STORE, element, y, stride0, stride1, column, 0ul, 0ul, 0ul, 0ul);
}

projection_epilogue projection_residual(const int act, uint64_t y, uint64_t stride0, uint64_t stride1,
    uint64_t residual, uint64_t residual0, uint64_t residual1, uint64_t rows) {
    return projection_epilogue(PROJECTION_RESIDUAL, act, y, stride0, stride1, 0ul, residual, residual0, residual1, rows);
}

projection_epilogue projection_silu_mul(const int act, uint64_t y, uint64_t stride0, uint64_t stride1) {
    return projection_epilogue(PROJECTION_SILU_MUL, act, y, stride0, stride1, 0ul, 0ul, 0ul, 0ul, 0ul);
}

// Publishes one output of a Store or Residual epilogue.
void projection_put(projection_epilogue out_, uint m, uint n, float value) {
    if (out_.kind == PROJECTION_STORE) {
        element_put(out_.act, out_.y, uint64_t(m) * out_.y0 + (out_.column + uint64_t(n)) * out_.y1, value);
    } else {
        const uint64_t source = uint64_t(projection_row(out_.rows, m)) * out_.r0 + uint64_t(n) * out_.r1;
        element_f32_put(out_.y + (uint64_t(m) * out_.y0 + uint64_t(n) * out_.y1) * 4ul,
            element_f32_at(out_.residual + source * 4ul) + element_round(out_.act, value));
    }
}

// Publishes one SiluMul feature from its gate and up sums.
void projection_put_pair(projection_epilogue out_, uint m, uint n, float gate_sum, float up_sum) {
    const float gate = element_round(out_.act, gate_sum);
    const float up = element_round(out_.act, up_sum);
    element_put(out_.act, out_.y, uint64_t(m) * out_.y0 + uint64_t(n) * out_.y1,
        projection_silu_rounded(out_.act, gate) * up);
}

void projection_emit(projection_epilogue out_, const bool paired, uint m, uint n, float first, float second) {
    if (paired)
        projection_put_pair(out_, m, n, first, second);
    else
        projection_put(out_, m, n, first);
}

// Rows of one weight tensor from row `first` (expert-stacked weights start
// at their expert's first row), optionally gathered through a row table.
struct projection_weights {
    uint64_t base;
    packets_rows16 geometry;
    uint k;
    uint64_t table;     // 0: rows in order
};

uint64_t projection_weights_row(projection_weights w, uint n) {
    const uint64_t r = w.table == 0ul ? uint64_t(n) : uint64_t(element_i32_at(w.table + uint64_t(n) * 4ul));
    return w.base + r * w.geometry.stride;
}

packets_packet projection_packet(const int kind, projection_weights w, uint n, uint p) {
    return packets_load(kind, projection_weights_row(w, n), w.geometry, p, w.k);
}

// ---------------------------------------------------------------------------
// The stage pre-pass of the GEMM class.

// Workgroup `item` normalizes one (row, group) of a normed prologue and
// stores that group's prologue output, in A, at x[m * columns + group * width
// ..]. The GEMM launches then read `x` as a Plain operand. Shared: one float
// per subgroup at float 0.
void projection_stage(projection_prologue in_, uint item, uint64_t x, uint columns) {
    const uint thread = gl_LocalInvocationIndex, threads = gl_WorkGroupSize.x;
    const uint groups = projection_groups(in_), width = projection_width(in_);
    const uint m = item / groups, group = item % groups;
    float squares = 0.0;
    for (uint i = thread; i < width; i += threads) {
        const float v = projection_norm_input(in_, m, group, i);
        squares = seismic_fma_rn(v, v, squares);
    }
    squares = reduce_group_sum(squares, 0u);
    const float inverse = inversesqrt(seismic_div_rn(squares, float(width)) + in_.eps);
    const uint first = group * width;
    const uint64_t row = uint64_t(m) * columns;
    const bool vector = (columns & 7u) == 0u && (first & 7u) == 0u;
    for (uint i = 8u * thread; i < width; i += 8u * threads) {
        vec4 even, odd;
        projection_load8(in_, m, first + i, inverse, even, odd);
        if (vector && i + 8u <= width) {
            element_uvec4_put(x + (row + first + i) * 2ul, element_pack8(in_.act, even, odd));
        } else {
            for (uint j = 0u; j < 8u && i + j < width; ++j)
                element_put(in_.act, x, row + first + i + j, (j & 1u) != 0u ? odd[j >> 1] : even[j >> 1]);
        }
    }
}

// ---------------------------------------------------------------------------
// GEMV (M <= 8).
//
// A workgroup of SG subgroups; a subgroup is 32 / LANES lane groups; lane
// group g of subgroup sg owns the R weight rows from
// ((tile * SG + sg) * (32 / LANES) + g) * R, and its lanes own the packets
// sub, sub + LANES, ... of those rows. The prologue output is staged per
// workgroup in chunks of at most 288 / m_rows packets (a multiple of LANES):
// the eight values of chunk-local packet p, step s and row m as one uvec4 in
// A storage, and the packet's two 16-column sums. Weight packets are
// double-buffered in registers: a lane loads its next packet before
// accumulating the current one, and its first packet before the staging.
//
// Norms: PROJECTION_NORM_NONE (Plain), PROJECTION_NORM_SHARED (Rms: the rows'
// square sums are reduced first into PROJECTION_RMS_PARTS parts in the shared
// region, each by one subgroup, and summed in a fixed tree, so the inverses do
// not depend on the workgroup shape), PROJECTION_NORM_LANES (GatedRms: each
// head's square sum is reduced across the head_width / 8 lanes that stage it).

#define PROJECTION_NORM_NONE 0
#define PROJECTION_NORM_SHARED 1
#define PROJECTION_NORM_LANES 2

#define PROJECTION_GEMV_STAGE_PACKETS 288u
// Row-norm partial squares at float 0: 8 rows x PROJECTION_RMS_PARTS.
#define PROJECTION_GEMV_SQUARES_BYTES 256
#define PROJECTION_GEMV_STAGE_WORD 16u      // first uvec4 of the staging
#define PROJECTION_GEMV_MAX_R 4
#define PROJECTION_GEMV_MAX_M 8
#define PROJECTION_GEMV_SUMS (PROJECTION_GEMV_MAX_R * PROJECTION_GEMV_MAX_M)

// Weight rows of one GEMV workgroup (segmented entries map workgroups to
// segments by it).
uint projection_gemv_rows(const uint sg_count, const uint r_rows, const uint lanes) {
    return sg_count * r_rows * (32u / lanes);
}

// The first weight row of this invocation's lane group.
uint projection_gemv_first_row(const uint r_rows, const uint lanes, uint tile) {
    return ((tile * gl_NumSubgroups + gl_SubgroupID) * (32u / lanes) + gl_SubgroupInvocationID / lanes) * r_rows;
}

// Row-norm square sums of the workgroup's rows into the shared region;
// subgroup sg reduces (row, part) items sg, sg + SG, ... The GEMV body's
// opening barrier publishes them.
void projection_gemv_squares(projection_prologue in_, uint m_rows) {
    const uint slots = PROJECTION_RMS_PARTS;
    for (uint item = gl_SubgroupID; item < m_rows * slots; item += gl_NumSubgroups) {
        const float sum = seismic_subgroup_sum_f32(
            projection_rms_squares(in_, item / slots, item % slots, gl_SubgroupInvocationID));
        if (gl_SubgroupInvocationID == 0u)
            seismic_shared_f32[item] = sum;
    }
}

float projection_gemv_inverse(projection_prologue in_, uint m) {
    const uint at = m * PROJECTION_RMS_PARTS;
    const float sum = ((seismic_shared_f32[at] + seismic_shared_f32[at + 1u])
                          + (seismic_shared_f32[at + 2u] + seismic_shared_f32[at + 3u]))
        + ((seismic_shared_f32[at + 4u] + seismic_shared_f32[at + 5u])
            + (seismic_shared_f32[at + 6u] + seismic_shared_f32[at + 7u]));
    return inversesqrt(seismic_div_rn(sum, float(in_.columns)) + in_.eps);
}

// The staged prologue output of x[m, k..k+8).
void projection_gemv_prologue8(projection_prologue in_, const int norm, uint m, uint k, out vec4 even, out vec4 odd) {
    if (norm == PROJECTION_NORM_SHARED) {
        projection_load8(in_, m, k, projection_gemv_inverse(in_, m), even, odd);
    } else if (norm == PROJECTION_NORM_LANES) {
        // head_width / 8 adjacent lanes load one head's columns in order: each
        // lane sums its eight values' squares, a butterfly over the group
        // gives every lane the head's square sum.
        const projection_gated8 v = projection_gated_inputs8(in_, m, k);
        float squares = 0.0;
        [[unroll]] for (uint j = 0u; j < 4u; ++j) {
            squares = seismic_fma_rn(v.me[j], v.me[j], squares);
            squares = seismic_fma_rn(v.mo[j], v.mo[j], squares);
        }
        for (uint offset = 1u; offset < in_.head_width / 8u; offset <<= 1)
            squares += subgroupShuffleXor(squares, offset);
        projection_gated_finish8(in_, v, inversesqrt(seismic_div_rn(squares, float(in_.head_width)) + in_.eps), even, odd);
    } else {
        projection_load8(in_, m, k, 0.0, even, odd);
    }
}

// Stage packets first .. first + count of every activation row: one
// invocation per eight columns (row m, packet, step); the four steps of a
// packet sit on adjacent lanes, which add their sums pairwise into the
// packet's two 16-column sums.
void projection_gemv_stage(projection_prologue in_, const int norm, uint m_rows, uint first, uint count, uint chunk) {
    const uint sums = 4u * (PROJECTION_GEMV_STAGE_WORD + 4u * m_rows * chunk);
    for (uint item = gl_LocalInvocationIndex; item < 4u * m_rows * count; item += gl_WorkGroupSize.x) {
        const uint step = item & 3u, packet = item >> 2;
        const uint m = packet / count, local = packet - m * count;
        vec4 even, odd;
        projection_gemv_prologue8(in_, norm, m, 32u * (first + local) + 8u * step, even, odd);
        seismic_shared_uvec4[PROJECTION_GEMV_STAGE_WORD + (m * 4u + step) * chunk + local] = element_pack8(in_.act, even, odd);
        const vec4 pair = even + odd;
        float sum = (pair.x + pair.y) + (pair.z + pair.w);
        sum += subgroupShuffleXor(sum, 1u);
        if ((step & 1u) == 0u)
            seismic_shared_f32[sums + 2u * (m * chunk + local) + (step >> 1)] = sum;
    }
}

// One packet of the R weight rows (and, paired, of the second tensor).
void projection_gemv_load(const int wk, const int uk, const bool paired, const uint r_rows, projection_weights w,
    projection_weights u, uint first_row, uint rows, uint p, out packets_packet a[PROJECTION_GEMV_MAX_R],
    out packets_packet b[PROJECTION_GEMV_MAX_R]) {
    [[unroll]] for (uint r = 0u; r < PROJECTION_GEMV_MAX_R; ++r) {
        if (r < r_rows) {
            const uint n = min(first_row + r, rows - 1u);
            a[r] = projection_packet(wk, w, n, p);
            if (paired)
                b[r] = projection_packet(uk, u, n, p);
        }
    }
}

// The dot product of eight codes (even, odd) with eight activations, in
// column order: one multiply, then fused multiply-adds (`dot()` under
// NoContraction would lower to separate multiplies and adds).
float projection_dot8(vec4 ce, vec4 co, vec4 xe, vec4 xo) {
    float sum = ce.x * xe.x;
    sum = seismic_fma_rn(co.x, xo.x, sum);
    sum = seismic_fma_rn(ce.y, xe.y, sum);
    sum = seismic_fma_rn(co.y, xo.y, sum);
    sum = seismic_fma_rn(ce.z, xe.z, sum);
    sum = seismic_fma_rn(co.z, xo.z, sum);
    sum = seismic_fma_rn(ce.w, xe.w, sum);
    return seismic_fma_rn(co.w, xo.w, sum);
}

// Accumulate one loaded packet (chunk-local `local`) for every activation row.
void projection_gemv_accumulate(const int wk, const int uk, const bool paired, const uint r_rows, const uint maxm,
    const int act, inout float acc[PROJECTION_GEMV_SUMS], inout float acc2[PROJECTION_GEMV_SUMS],
    packets_packet a[PROJECTION_GEMV_MAX_R], packets_packet b[PROJECTION_GEMV_MAX_R], uint local, uint m_rows,
    uint chunk) {
    const uint sums = 4u * (PROJECTION_GEMV_STAGE_WORD + 4u * m_rows * chunk);
    [[unroll]] for (uint step = 0u; step < 4u; ++step) {
        vec4 ae[PROJECTION_GEMV_MAX_R], ao[PROJECTION_GEMV_MAX_R], be[PROJECTION_GEMV_MAX_R], bo[PROJECTION_GEMV_MAX_R];
        [[unroll]] for (uint r = 0u; r < PROJECTION_GEMV_MAX_R; ++r) {
            if (r < r_rows) {
                packets_codes(wk, a[r], step, ae[r], ao[r]);
                if (paired)
                    packets_codes(uk, b[r], step, be[r], bo[r]);
            }
        }
        [[unroll]] for (uint m = 0u; m < PROJECTION_GEMV_MAX_M; ++m) {
            if (m < maxm && m < m_rows) {
                vec4 xe, xo;
                element_split8(act, seismic_shared_uvec4[PROJECTION_GEMV_STAGE_WORD + (m * 4u + step) * chunk + local], xe, xo);
                [[unroll]] for (uint r = 0u; r < PROJECTION_GEMV_MAX_R; ++r) {
                    if (r < r_rows) {
                        const uint at = r * PROJECTION_GEMV_MAX_M + m;
                        acc[at] = seismic_fma_rn(packets_scale(wk, a[r], step), projection_dot8(ae[r], ao[r], xe, xo), acc[at]);
                        if (paired)
                            acc2[at] = seismic_fma_rn(packets_scale(uk, b[r], step), projection_dot8(be[r], bo[r], xe, xo), acc2[at]);
                    }
                }
            }
        }
    }
    if (packets_biased(wk) || (paired && packets_biased(uk))) {
        [[unroll]] for (uint m = 0u; m < PROJECTION_GEMV_MAX_M; ++m) {
            if (m < maxm && m < m_rows) {
                const vec2 s = vec2(seismic_shared_f32[sums + 2u * (m * chunk + local)],
                    seismic_shared_f32[sums + 2u * (m * chunk + local) + 1u]);
                [[unroll]] for (uint r = 0u; r < PROJECTION_GEMV_MAX_R; ++r) {
                    if (r < r_rows) {
                        const uint at = r * PROJECTION_GEMV_MAX_M + m;
                        if (packets_biased(wk)) {
                            if (packets_groups(wk) == 1u) {
                                acc[at] = seismic_fma_rn(packets_bias(wk, a[r], 0u), s.x + s.y, acc[at]);
                            } else {
                                acc[at] = seismic_fma_rn(packets_bias(wk, a[r], 0u), s.x, acc[at]);
                                acc[at] = seismic_fma_rn(packets_bias(wk, a[r], 1u), s.y, acc[at]);
                            }
                        }
                        if (paired && packets_biased(uk)) {
                            if (packets_groups(uk) == 1u) {
                                acc2[at] = seismic_fma_rn(packets_bias(uk, b[r], 0u), s.x + s.y, acc2[at]);
                            } else {
                                acc2[at] = seismic_fma_rn(packets_bias(uk, b[r], 0u), s.x, acc2[at]);
                                acc2[at] = seismic_fma_rn(packets_bias(uk, b[r], 1u), s.y, acc2[at]);
                            }
                        }
                    }
                }
            }
        }
    }
}

// The sums of one GEMV workgroup with the activation-row bound `maxm` (1, 2, 4
// or 8): every lane of a lane group receives the group's totals of its R
// weight rows (from projection_gemv_first_row) and every activation row, at
// totals[r * PROJECTION_GEMV_MAX_M + m] (totals2: the paired weight).
void projection_gemv_sums(const int wk, const int uk, const bool paired, const uint r_rows, const uint lanes,
    const uint maxm, const int norm, projection_prologue in_, projection_weights w, projection_weights u, uint m_rows,
    uint rows, uint k, uint tile, out float totals[PROJECTION_GEMV_SUMS], out float totals2[PROJECTION_GEMV_SUMS]) {
    const uint sub = gl_SubgroupInvocationID % lanes;
    const uint first_row = projection_gemv_first_row(r_rows, lanes, tile);
    const bool owns_rows = first_row < rows;
    const uint packets = (k + 31u) / 32u;
    packets_packet current_a[PROJECTION_GEMV_MAX_R], current_b[PROJECTION_GEMV_MAX_R];
    packets_packet next_a[PROJECTION_GEMV_MAX_R], next_b[PROJECTION_GEMV_MAX_R];
    if (owns_rows && sub < packets)
        projection_gemv_load(wk, uk, paired, r_rows, w, u, first_row, rows, sub, current_a, current_b);
    const uint chunk = min(packets, (PROJECTION_GEMV_STAGE_PACKETS / m_rows) / lanes * lanes);
    float acc[PROJECTION_GEMV_SUMS], acc2[PROJECTION_GEMV_SUMS];
    [[unroll]] for (uint i = 0u; i < PROJECTION_GEMV_SUMS; ++i) {
        acc[i] = 0.0;
        acc2[i] = 0.0;
    }
    for (uint first = 0u; first < packets; first += chunk) {
        const uint count = min(chunk, packets - first);
        // The previous chunk's or call's reads of the staging finish before
        // this one writes it (and the row-norm squares become visible).
        barrier();
        projection_gemv_stage(in_, norm, m_rows, first, count, chunk);
        barrier();
        if (owns_rows) {
            for (uint local = sub; local < count; local += lanes) {
                if (first + local + lanes < packets)
                    projection_gemv_load(wk, uk, paired, r_rows, w, u, first_row, rows, first + local + lanes, next_a, next_b);
                projection_gemv_accumulate(wk, uk, paired, r_rows, maxm, in_.act, acc, acc2, current_a, current_b, local,
                    m_rows, chunk);
                current_a = next_a;
                current_b = next_b;
            }
        }
    }
    [[unroll]] for (uint i = 0u; i < PROJECTION_GEMV_SUMS; ++i) {
        const uint r = i / PROJECTION_GEMV_MAX_M, m = i % PROJECTION_GEMV_MAX_M;
        totals[i] = 0.0;
        totals2[i] = 0.0;
        if (r < r_rows && m < maxm && m < m_rows) {
            totals[i] = reduce_lanes_sum(acc[i], lanes);
            if (paired)
                totals2[i] = reduce_lanes_sum(acc2[i], lanes);
        }
    }
}

// One GEMV workgroup with the activation-row bound `maxm`: the sums, then the
// epilogue of each (weight row, activation row) from one lane of its group.
void projection_gemv_bounded(const int wk, const int uk, const bool paired, const uint r_rows, const uint lanes,
    const uint maxm, const int norm, projection_prologue in_, projection_epilogue out_, projection_weights w,
    projection_weights u, uint m_rows, uint rows, uint k, uint tile) {
    float totals[PROJECTION_GEMV_SUMS], totals2[PROJECTION_GEMV_SUMS];
    projection_gemv_sums(wk, uk, paired, r_rows, lanes, maxm, norm, in_, w, u, m_rows, rows, k, tile, totals, totals2);
    const uint sub = gl_SubgroupInvocationID % lanes;
    const uint first_row = projection_gemv_first_row(r_rows, lanes, tile);
    [[unroll]] for (uint r = 0u; r < PROJECTION_GEMV_MAX_R; ++r) {
        [[unroll]] for (uint m = 0u; m < PROJECTION_GEMV_MAX_M; ++m) {
            const uint at = r * PROJECTION_GEMV_MAX_M + m;
            if (r < r_rows && m < maxm && m < m_rows && sub == (r * maxm + m) % lanes && first_row + r < rows)
                projection_emit(out_, paired, m, first_row + r, totals[at], totals2[at]);
        }
    }
}

// The GEMV of one segment: weight kinds `wk` (and `uk` of the up weights when
// paired), R rows per lane group, LANES lanes per group. Rms prologues first
// reduce their rows' squares (norm SHARED). Every invocation of the workgroup
// must call it (it contains barriers).
void projection_gemv(const int wk, const int uk, const bool paired, const uint r_rows, const uint lanes, const int norm,
    projection_prologue in_, projection_epilogue out_, projection_weights w, projection_weights u, uint m_rows,
    uint rows, uint k, uint tile) {
    if (norm == PROJECTION_NORM_SHARED)
        projection_gemv_squares(in_, m_rows);
    if (m_rows <= 1u)
        projection_gemv_bounded(wk, uk, paired, r_rows, lanes, 1u, norm, in_, out_, w, u, m_rows, rows, k, tile);
    else if (m_rows <= 2u)
        projection_gemv_bounded(wk, uk, paired, r_rows, lanes, 2u, norm, in_, out_, w, u, m_rows, rows, k, tile);
    else if (m_rows <= 4u)
        projection_gemv_bounded(wk, uk, paired, r_rows, lanes, 4u, norm, in_, out_, w, u, m_rows, rows, k, tile);
    else
        projection_gemv_bounded(wk, uk, paired, r_rows, lanes, 8u, norm, in_, out_, w, u, m_rows, rows, k, tile);
}

// ---------------------------------------------------------------------------
// GEMM (M > 8).
//
// A TM x TN output tile per workgroup of (TM / WM) x (TN / WN) subgroups, each
// owning a WM x WN sub-tile (WM, WN in {32, 64}), stepping K by 32. The shared
// region holds the A tile (TM rows) and the B tile (TN weight rows) as f16
// with rows padded to 40 elements. While a step multiplies, every invocation
// already holds the next step's activation values and raw weight packets in
// registers, loaded before the multiply; it stores them after the step's
// barrier. Neither the tile nor the sub-tile shape changes a result: every
// output is the same ascending-K chain (16-deep matrix steps, or single FMAs on
// the plain path).
//
// Tiles past the last row (`m_rows`) are cheap: their activation rows are not
// read, and each subgroup multiplies only its 16-row fragment rows that hold a
// row before `live_rows`.
//
// Range: bf16 activations can exceed f16's range (a SiLU * up product reaches
// 1e5). With a bf16 activation every tile first scans its rows for their
// largest magnitude and stages each row scaled by an exact power of two
// 2^-e (e >= 0, only when the row needs it) so it stays below 2^15; the
// outputs are scaled back by 2^e. Scaling by powers of two is exact for every
// value within f16's normal range. The per-row maximum bits and exponent sit
// in the pad words of the row's A tile slot (halves 32..39), which the K loop
// never writes.
//
// A GEMM launch declares TM * TN * 32 / (WM * WN) invocations and (TM + TN) *
// 80 shared bytes; TM in {32, 64, 128}, TN in {64, 128}, WM <= TM, WN <= TN.

#define PROJECTION_GEMM_K 32u
#define PROJECTION_GEMM_LDA 40u
// Staged items per invocation and step: at most WM * WN / (8 TN) activation
// items and WM * WN / (16 TM) weight items.
#define PROJECTION_GEMM_MAX_ITEMS 8
// Matrix fragments (16 x 16) of a 64 x 64 sub-tile; plain 32 x 32 blocks.
#define PROJECTION_GEMM_MAX_FRAGMENTS 16
#define PROJECTION_GEMM_MAX_BLOCKS 4
// u32 offsets, within a row's A slot, of its maximum bits and scale exponent.
#define PROJECTION_GEMM_ROW_MAXIMUM 16u
#define PROJECTION_GEMM_ROW_EXPONENT 17u

#if SEISMIC_HAS_MATRIX
#define PROJECTION_FRAGMENT_A coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define PROJECTION_FRAGMENT_B coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define PROJECTION_FRAGMENT_C coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#endif

// The accumulators of one subgroup's WM x WN sub-tile: (WM / 16) x (WN / 16)
// fragments (fragment (i, j) at i * (WN / 16) + j), or per 32 x 32 block
// (bi, bj) (block bi * (WN / 32) + bj) a 4 x 8 register tile per lane (rows 4
// (lane / 4) .., columns 8 (lane % 4) .. of the block), and the scale 2^e of
// each of the lane's output rows (see `projection_gemm_pair`).
struct projection_gemm_acc {
#if SEISMIC_HAS_MATRIX
    PROJECTION_FRAGMENT_C c[PROJECTION_GEMM_MAX_FRAGMENTS];
#else
    float c[32 * PROJECTION_GEMM_MAX_BLOCKS];
#endif
    float scale[8];
};

// Invocations of a GEMM workgroup.
uint projection_gemm_threads(const uint tm, const uint tn, const uint wm, const uint wn) {
    return tm * tn * 32u / (wm * wn);
}

// The u32 index of word `word` of tile row `row`'s A slot.
uint projection_gemm_row_word(uint row, uint word) { return row * (PROJECTION_GEMM_LDA / 2u) + word; }

// The row scan of a bf16 activation: every tile row's exponent e = max(0,
// exponent(max |x|) - 14), stored in its A slot. Every invocation takes part;
// the caller has freed the region.
void projection_gemm_scan_rows(projection_prologue in_, const uint tm, uint m0, uint m_rows) {
    for (uint row = gl_LocalInvocationIndex; row < tm; row += gl_WorkGroupSize.x)
        seismic_shared_u32[projection_gemm_row_word(row, PROJECTION_GEMM_ROW_MAXIMUM)] = 0u;
    barrier();
    const uint pieces = (in_.columns + 7u) / 8u;
    for (uint item = gl_LocalInvocationIndex; item < tm * pieces; item += gl_WorkGroupSize.x) {
        const uint row = item / pieces;
        if (m0 + row >= m_rows)
            continue;
        vec4 even, odd;
        projection_load8(in_, m0 + row, 8u * (item % pieces), 0.0, even, odd);
        const vec4 largest = max(abs(even), abs(odd));
        const float value = max(max(largest.x, largest.y), max(largest.z, largest.w));
        atomicMax(seismic_shared_u32[projection_gemm_row_word(row, PROJECTION_GEMM_ROW_MAXIMUM)], floatBitsToUint(value));
    }
    barrier();
    for (uint row = gl_LocalInvocationIndex; row < tm; row += gl_WorkGroupSize.x) {
        const float largest = uintBitsToFloat(seismic_shared_u32[projection_gemm_row_word(row, PROJECTION_GEMM_ROW_MAXIMUM)]);
        int exponent = 0;
        if (largest >= 32768.0 && !isinf(largest)) {
            int e;
            frexp(largest, e);
            exponent = e - 15;
        }
        seismic_shared_u32[projection_gemm_row_word(row, PROJECTION_GEMM_ROW_EXPONENT)] = uint(exponent);
    }
    barrier();
}

// The staging exponent of tile row `row` (0 without a scan).
int projection_gemm_row_exponent(projection_prologue in_, uint row) {
    return in_.act == ELEMENT_BF16 ? int(seismic_shared_u32[projection_gemm_row_word(row, PROJECTION_GEMM_ROW_EXPONENT)]) : 0;
}

// The activation items (8 columns of one tile row) an invocation stages per
// step, and its weight items (half-packets of one tensor's tile rows).
uint projection_gemm_a_items(const uint tm, const uint threads) { return (4u * tm + threads - 1u) / threads; }
uint projection_gemm_b_items(const uint count, const uint threads) { return (2u * count + threads - 1u) / threads; }

// Stage one step's A registers: item j is columns 8 (item % 4) .. of tile row
// item / 4, as f16 words (scaled by the row's 2^-e).
void projection_gemm_load_a(projection_prologue in_, const uint tm, const uint threads, uint m0, uint m_rows, uint k0,
    out uvec4 regs[PROJECTION_GEMM_MAX_ITEMS]) {
    [[unroll]] for (uint j = 0u; j < projection_gemm_a_items(tm, threads); ++j) {
        const uint item = gl_LocalInvocationIndex + j * threads;
        const uint row = item / 4u, column = k0 + 8u * (item % 4u);
        regs[j] = uvec4(0u);
        if (row < tm && m0 + row < m_rows && column < in_.columns) {
            vec4 even, odd;
            projection_load8(in_, m0 + row, column, 0.0, even, odd);
            if (in_.act == ELEMENT_BF16) {
                const int exponent = projection_gemm_row_exponent(in_, row);
                even = ldexp(even, ivec4(-exponent));
                odd = ldexp(odd, ivec4(-exponent));
            }
            regs[j] = element_pack8(ELEMENT_F16, even, odd);
        }
    }
}

void projection_gemm_store_a(const uint tm, const uint threads, uvec4 regs[PROJECTION_GEMM_MAX_ITEMS]) {
    [[unroll]] for (uint j = 0u; j < projection_gemm_a_items(tm, threads); ++j) {
        const uint item = gl_LocalInvocationIndex + j * threads;
        const uint row = item / 4u;
        if (row < tm)
            seismic_shared_uvec4[row * (PROJECTION_GEMM_LDA / 8u) + item % 4u] = regs[j];
    }
}

// One step's weight registers: item j is the half-packet (16 codes)
// item % 2 of the tensor's tile-local row item / 2.
struct projection_gemm_b {
    packets_packet packet[PROJECTION_GEMM_MAX_ITEMS];
    bool valid[PROJECTION_GEMM_MAX_ITEMS];
};

void projection_gemm_load_b(const int kind, const uint threads, projection_weights w, uint first, const uint count,
    uint rows, uint k0, uint k, out projection_gemm_b regs) {
    [[unroll]] for (uint j = 0u; j < projection_gemm_b_items(count, threads); ++j) {
        const uint item = gl_LocalInvocationIndex + j * threads;
        const uint local = item / 2u;
        const uint p = k0 / 32u;
        regs.valid[j] = local < count && 32u * p < k;
        if (regs.valid[j])
            regs.packet[j] = projection_packet(kind, w, min(first + local, rows - 1u), p);
    }
}

// Decode the staged packets (scale * code + bias, rounded to f16). Local row
// r of this tensor becomes B row tile_row0 + r * spacing (a paired tile
// interleaves gate and up rows); B starts at uvec4 `b_word`.
void projection_gemm_store_b(const int kind, const uint threads, projection_gemm_b regs, const uint count,
    uint tile_row0, uint spacing, uint b_word) {
    [[unroll]] for (uint j = 0u; j < projection_gemm_b_items(count, threads); ++j) {
        const uint item = gl_LocalInvocationIndex + j * threads;
        const uint local = item / 2u, within = item % 2u;
        if (local >= count)
            continue;
        const uint at = b_word + (tile_row0 + local * spacing) * (PROJECTION_GEMM_LDA / 8u) + 2u * within;
        if (regs.valid[j]) {
            const float scale = packets_scale(kind, regs.packet[j], 2u * within);
            const float bias = packets_bias(kind, regs.packet[j], packets_groups(kind) == 1u ? 0u : within);
            [[unroll]] for (uint s = 0u; s < 2u; ++s) {
                vec4 even, odd;
                packets_codes(kind, regs.packet[j], 2u * within + s, even, odd);
                [[unroll]] for (uint i = 0u; i < 4u; ++i) {
                    even[i] = seismic_fma_rn(scale, even[i], bias);
                    odd[i] = seismic_fma_rn(scale, odd[i], bias);
                }
                seismic_shared_uvec4[at + s] = element_pack8(ELEMENT_F16, even, odd);
            }
        } else {
            seismic_shared_uvec4[at] = uvec4(0u);
            seismic_shared_uvec4[at + 1u] = uvec4(0u);
        }
    }
}

// The subgroup's sub-tile origin in the tile.
uint projection_gemm_row0(const uint tn, const uint wm, const uint wn) { return (gl_SubgroupID / (tn / wn)) * wm; }
uint projection_gemm_column0(const uint tn, const uint wn) { return (gl_SubgroupID % (tn / wn)) * wn; }

// The multiply of one staged step: `live` fragment rows (0 .. WM / 16) of 16.
void projection_gemm_multiply(const uint tm, const uint tn, const uint wm, const uint wn, uint live,
    inout projection_gemm_acc acc) {
    const uint row0 = projection_gemm_row0(tn, wm, wn), column0 = projection_gemm_column0(tn, wn);
    const uint b0 = tm * PROJECTION_GEMM_LDA;   // B tile, in f16 elements
#if SEISMIC_HAS_MATRIX
    const uint fm = wm / 16u, fn = wn / 16u;
    [[unroll]] for (uint kk = 0u; kk < PROJECTION_GEMM_K; kk += 16u) {
        PROJECTION_FRAGMENT_B b[4];
        [[unroll]] for (uint j = 0u; j < fn; ++j)
            coopMatLoad(b[j], seismic_shared_f16, b0 + (column0 + 16u * j) * PROJECTION_GEMM_LDA + kk,
                PROJECTION_GEMM_LDA, gl_CooperativeMatrixLayoutColumnMajor);
        [[unroll]] for (uint i = 0u; i < fm; ++i) {
            if (i < live) {
                PROJECTION_FRAGMENT_A a;
                coopMatLoad(a, seismic_shared_f16, (row0 + 16u * i) * PROJECTION_GEMM_LDA + kk, PROJECTION_GEMM_LDA,
                    gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint j = 0u; j < fn; ++j)
                    acc.c[fn * i + j] = coopMatMulAdd(a, b[j], acc.c[fn * i + j]);
            }
        }
    }
#else
    const uint lane = gl_SubgroupInvocationID;
    const uint bn = wn / 32u;
    [[unroll]] for (uint block = 0u; block < (wm / 32u) * bn; ++block) {
        const uint bi = block / bn, bj = block % bn;
        const uint r0 = row0 + 32u * bi + 4u * (lane / 4u), c0 = column0 + 32u * bj + 8u * (lane % 4u);
        // Block row bi holds fragment rows 2 bi and 2 bi + 1.
        if (live <= 2u * bi || (live == 2u * bi + 1u && r0 >= row0 + 32u * bi + 16u))
            continue;
        // Pairs of K (one f16x2 word per row) keep the reads vectorized.
        [[unroll]] for (uint kk = 0u; kk < PROJECTION_GEMM_K; kk += 2u) {
            vec2 a[4], b[8];
            [[unroll]] for (uint i = 0u; i < 4u; ++i)
                a[i] = unpackHalf2x16(seismic_shared_u32[((r0 + i) * PROJECTION_GEMM_LDA + kk) / 2u]);
            [[unroll]] for (uint j = 0u; j < 8u; ++j)
                b[j] = unpackHalf2x16(seismic_shared_u32[(b0 + (c0 + j) * PROJECTION_GEMM_LDA + kk) / 2u]);
            [[unroll]] for (uint i = 0u; i < 4u; ++i) {
                [[unroll]] for (uint j = 0u; j < 8u; ++j) {
                    const uint at = 32u * block + 8u * i + j;
                    acc.c[at] = seismic_fma_rn(a[i].x, b[j].x, acc.c[at]);
                    acc.c[at] = seismic_fma_rn(a[i].y, b[j].y, acc.c[at]);
                }
            }
        }
    }
#endif
}

// The K loop over one TM x TN tile, steps [step_begin, step_end): `uk` and
// `u` are the second weights of a paired tile (rows interleaved with w's).
void projection_gemm_accumulate(const int wk, const int uk, const bool paired, const uint tm, const uint tn,
    const uint wm, const uint wn, projection_prologue in_, projection_weights w, projection_weights u, uint first,
    uint rows, uint m0, uint m_rows, uint k, uint step_begin, uint step_end, uint live_rows,
    out projection_gemm_acc acc) {
    const uint threads = projection_gemm_threads(tm, tn, wm, wn);
    const uint count = paired ? tn / 2u : tn;
    const uint spacing = paired ? 2u : 1u;
    const uint b_word = tm * (PROJECTION_GEMM_LDA / 8u);
#if SEISMIC_HAS_MATRIX
    [[unroll]] for (uint i = 0u; i < (wm / 16u) * (wn / 16u); ++i)
        acc.c[i] = PROJECTION_FRAGMENT_C(0.0);
    const uint scales = wm / 16u;
#else
    [[unroll]] for (uint i = 0u; i < wm * wn / 32u; ++i)
        acc.c[i] = 0.0;
    const uint scales = wm / 8u;
#endif
    [[unroll]] for (uint i = 0u; i < scales; ++i)
        acc.scale[i] = 1.0;
    const uint row_first = m0 + projection_gemm_row0(tn, wm, wn);
    const uint live_limit = min(m_rows, live_rows);
    const uint live = row_first >= live_limit ? 0u : min(wm / 16u, (live_limit - row_first + 15u) / 16u);
    if (step_begin >= step_end)
        return;
    if (in_.act == ELEMENT_BF16) {
        // A previous tile's epilogue may still read the region.
        barrier();
        projection_gemm_scan_rows(in_, tm, m0, m_rows);
    }
    uvec4 a_regs[PROJECTION_GEMM_MAX_ITEMS];
    projection_gemm_b w_regs, u_regs;
    uint k0 = step_begin * PROJECTION_GEMM_K;
    projection_gemm_load_a(in_, tm, threads, m0, m_rows, k0, a_regs);
    projection_gemm_load_b(wk, threads, w, first, count, rows, k0, k, w_regs);
    if (paired)
        projection_gemm_load_b(uk, threads, u, first, count, rows, k0, k, u_regs);
    // A previous tile's epilogue may still read the region.
    barrier();
    projection_gemm_store_a(tm, threads, a_regs);
    projection_gemm_store_b(wk, threads, w_regs, count, 0u, spacing, b_word);
    if (paired)
        projection_gemm_store_b(uk, threads, u_regs, count, 1u, spacing, b_word);
    barrier();
    for (uint t = step_begin; t < step_end; ++t) {
        const bool more = t + 1u < step_end;
        const uint k1 = (t + 1u) * PROJECTION_GEMM_K;
        if (more) {
            projection_gemm_load_a(in_, tm, threads, m0, m_rows, k1, a_regs);
            projection_gemm_load_b(wk, threads, w, first, count, rows, k1, k, w_regs);
            if (paired)
                projection_gemm_load_b(uk, threads, u, first, count, rows, k1, k, u_regs);
        }
        projection_gemm_multiply(tm, tn, wm, wn, live, acc);
        barrier();
        if (more) {
            projection_gemm_store_a(tm, threads, a_regs);
            projection_gemm_store_b(wk, threads, w_regs, count, 0u, spacing, b_word);
            if (paired)
                projection_gemm_store_b(uk, threads, u_regs, count, 1u, spacing, b_word);
        }
        barrier();
    }
    if (in_.act == ELEMENT_BF16) {
        // The scales of the lane's output rows, read before the pairs reuse
        // the region.
        const uint row0 = projection_gemm_row0(tn, wm, wn), lane = gl_SubgroupInvocationID;
        [[unroll]] for (uint i = 0u; i < scales; ++i) {
#if SEISMIC_HAS_MATRIX
            const uint row = row0 + 16u * i + lane / 2u;
#else
            const uint row = row0 + 32u * (i / 4u) + 4u * (lane / 4u) + i % 4u;
#endif
            acc.scale[i] = ldexp(1.0, projection_gemm_row_exponent(in_, row));
        }
        barrier();
    }
}

// Hand every lane its sub-tile outputs in pairs of adjacent columns (c,
// c + 1), c even, so paired tiles see (gate, up): per lane
// `projection_gemm_pairs` pairs; pair q sits at tile row
// `projection_gemm_pair_row` and column `projection_gemm_pair_column`. The
// matrix path publishes each 16 x 16 fragment through the subgroup's 1 KiB
// of the (now free) tile region (pairs 4 f .. 4 f + 3 are fragment f); the
// plain path already holds a 4 x 8 register tile per block (pairs 16 b ..
// 16 b + 15 are block b).
uint projection_gemm_pairs(const uint wm, const uint wn) { return wm * wn / 64u; }

uint projection_gemm_pair_row(const uint tn, const uint wm, const uint wn, uint q) {
    const uint lane = gl_SubgroupInvocationID;
#if SEISMIC_HAS_MATRIX
    // Fragment q / 4 = (i, j); within it, pair 4 lane + q % 4 of 128 (8
    // pairs per row).
    return projection_gemm_row0(tn, wm, wn) + 16u * ((q / 4u) / (wn / 16u)) + (4u * lane + q % 4u) / 8u;
#else
    return projection_gemm_row0(tn, wm, wn) + 32u * ((q / 16u) / (wn / 32u)) + 4u * (lane / 4u) + (q % 16u) / 4u;
#endif
}

uint projection_gemm_pair_column(const uint tn, const uint wn, uint q) {
    const uint lane = gl_SubgroupInvocationID;
#if SEISMIC_HAS_MATRIX
    return projection_gemm_column0(tn, wn) + 16u * ((q / 4u) % (wn / 16u)) + 2u * ((4u * lane + q % 4u) % 8u);
#else
    return projection_gemm_column0(tn, wn) + 32u * ((q / 16u) % (wn / 32u)) + 8u * (lane % 4u) + 2u * (q % 4u);
#endif
}

// The lane's pair q of the accumulators, scaled back by its row's 2^e. The
// matrix path must call it for q = 0 .. projection_gemm_pairs - 1 in order
// from every lane of the subgroup, after a barrier that frees the tile region.
vec2 projection_gemm_pair(inout projection_gemm_acc acc, const uint wn, uint q) {
#if SEISMIC_HAS_MATRIX
    const uint scratch = 256u * gl_SubgroupID;   // floats
    const uint pair = 4u * gl_SubgroupInvocationID + q % 4u;
    if (q % 4u == 0u) {
        subgroupBarrier();
        coopMatStore(acc.c[q / 4u], seismic_shared_f32, scratch, 16u, gl_CooperativeMatrixLayoutRowMajor);
        subgroupMemoryBarrierShared();
        subgroupBarrier();
    }
    return acc.scale[(q / 4u) / (wn / 16u)]
        * vec2(seismic_shared_f32[scratch + 2u * pair], seismic_shared_f32[scratch + 2u * pair + 1u]);
#else
    const uint block = q / 16u, within = q % 16u;
    const uint at = 32u * block + 8u * (within / 4u) + 2u * (within % 4u);
    return acc.scale[4u * (block / (wn / 32u)) + within / 4u] * vec2(acc.c[at], acc.c[at + 1u]);
#endif
}

// One GEMM tile of a plain (unpaired) projection: output rows tm_index * TM
// .., weight rows tn_index * TN .. of `w`. Rows at or past `live_rows` are
// known zero (padding): their products are skipped, and they store the
// epilogue of 0.
void projection_gemm(const int wk, const uint tm, const uint tn, const uint wm, const uint wn,
    projection_prologue in_, projection_epilogue out_, projection_weights w, uint m_rows, uint rows, uint k,
    uint tm_index, uint tn_index, uint live_rows) {
    const uint first = tn_index * tn;
    projection_gemm_acc acc;
    projection_gemm_accumulate(wk, wk, false, tm, tn, wm, wn, in_, w, w, first, rows, tm_index * tm, m_rows, k, 0u,
        (k + PROJECTION_GEMM_K - 1u) / PROJECTION_GEMM_K, live_rows, acc);
    [[unroll]] for (uint q = 0u; q < projection_gemm_pairs(wm, wn); ++q) {
        const vec2 c = projection_gemm_pair(acc, wn, q);
        const uint m = tm_index * tm + projection_gemm_pair_row(tn, wm, wn, q);
        const uint n = first + projection_gemm_pair_column(tn, wn, q);
        if (m < m_rows && n < rows)
            projection_put(out_, m, n, c.x);
        if (m < m_rows && n + 1u < rows)
            projection_put(out_, m, n + 1u, c.y);
    }
}

// Split-K. Part `part` of `split` runs the K steps [part * steps / split,
// (part + 1) * steps / split) of one plain GEMM tile and stores its raw F32
// sums to partials[(part * m_rows + m) * rows + n].
void projection_gemm_part(const int wk, const uint tm, const uint tn, const uint wm, const uint wn,
    projection_prologue in_, uint64_t partials, projection_weights w, uint m_rows, uint rows, uint k, uint split,
    uint part, uint tm_index, uint tn_index) {
    const uint first = tn_index * tn;
    const uint steps = (k + PROJECTION_GEMM_K - 1u) / PROJECTION_GEMM_K;
    projection_gemm_acc acc;
    projection_gemm_accumulate(wk, wk, false, tm, tn, wm, wn, in_, w, w, first, rows, tm_index * tm, m_rows, k,
        part * steps / split, (part + 1u) * steps / split, m_rows, acc);
    const uint64_t own = partials + uint64_t(part) * m_rows * rows * 4ul;
    [[unroll]] for (uint q = 0u; q < projection_gemm_pairs(wm, wn); ++q) {
        const vec2 c = projection_gemm_pair(acc, wn, q);
        const uint m = tm_index * tm + projection_gemm_pair_row(tn, wm, wn, q);
        const uint n = first + projection_gemm_pair_column(tn, wn, q);
        if (m < m_rows && n < rows)
            element_f32_put(own + (uint64_t(m) * rows + n) * 4ul, c.x);
        if (m < m_rows && n + 1u < rows)
            element_f32_put(own + (uint64_t(m) * rows + n + 1u) * 4ul, c.y);
    }
}

// The split-K finalize: output `index` = m * rows + n sums its parts in order
// and applies the epilogue.
void projection_gemm_finalize(projection_epilogue out_, uint64_t partials, uint m_rows, uint rows, uint split, uint index) {
    if (index >= m_rows * rows)
        return;
    float total = 0.0;
    for (uint part = 0u; part < split; ++part)
        total += element_f32_at(partials + (uint64_t(part) * m_rows * rows + index) * 4ul);
    projection_put(out_, index / rows, index % rows, total);
}

// One GEMM tile of a paired projection: TN / 2 features of gate and up, the
// tile's weight rows interleaved (gate, up), so a lane's column pair is one
// feature. `live_rows` as for `projection_gemm`.
void projection_gemm_paired(const int gk, const int uk, const uint tm, const uint tn, const uint wm, const uint wn,
    projection_prologue in_, projection_epilogue out_, projection_weights gate, projection_weights up, uint m_rows,
    uint rows, uint k, uint tm_index, uint tn_index, uint live_rows) {
    const uint first = tn_index * (tn / 2u);
    projection_gemm_acc acc;
    projection_gemm_accumulate(gk, uk, true, tm, tn, wm, wn, in_, gate, up, first, rows, tm_index * tm, m_rows, k, 0u,
        (k + PROJECTION_GEMM_K - 1u) / PROJECTION_GEMM_K, live_rows, acc);
    [[unroll]] for (uint q = 0u; q < projection_gemm_pairs(wm, wn); ++q) {
        const vec2 c = projection_gemm_pair(acc, wn, q);
        const uint m = tm_index * tm + projection_gemm_pair_row(tn, wm, wn, q);
        const uint n = first + projection_gemm_pair_column(tn, wn, q) / 2u;
        if (m < m_rows && n < rows)
            projection_put_pair(out_, m, n, c.x, c.y);
    }
}
