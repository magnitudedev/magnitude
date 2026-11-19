// The packed projection family (K1): y[m, n] = epilogue(sum_k x[m, k] * W[n, k])
// where x is produced by a prologue from the entry's inputs. The counterpart
// of `metal/lib/projection/projection.h` and `cuda/lib/projection/projection.cuh`, with the
// same vocabulary (organization spec §B3), prefixed `projection_` in GLSL.
//
// Row maps: AllRows (table 0) or SelectedRows (an `out_rows` i32 table).
//
// Prologues (`projection_prologue`, kind PROJECTION_{PLAIN,RMS,GROUPED})
// produce one activation value x[m, k], already rounded to the activation
// type A exactly as the entry's portable body publishes it:
//   Plain     x = A[row(m), k]
//   Rms       x = round_A(r[row(m), k] * inverse(m) * norm[k])
//   Grouped   x = A[order[m], k], zero for a padding row (order -1)
// In the GEMV row class (M <= 8) every workgroup computes its rows' norm
// inverses and applies the prologue while staging, so a decode projection is
// one launch. The GEMM class (M > 8) runs a pre-pass launch
// (`projection_stage`) that writes the whole prologue output in A to scratch,
// read by the GEMM as a Plain operand: no GEMM tile repeats the prologue.
// Over NVFP4 weights the rows enter as NVFP4 operands (`projection_operands`,
// the portable `operand_rows`): the GEMV forms them while staging, the stage
// launches form them into the scratch the GEMM reads.
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
// or GEMM body with that segment's weights and epilogue. A GEMM launch calls
// the GEMM body once, with the segment's kind as a workgroup-uniform value:
// only the weight staging branches on it, over the kinds the entry's slots
// bind, each a preprocessor group (<seismic/packets.glsl>). Instantiating the
// body once per segment instead defeats the NVIDIA proprietary compiler
// (580.159.03): a launch holding two or more GEMM bodies ran about 100x slower
// than the same tiles through one body.
//
// Row classes pick the launch:
// - GEMV (M <= 8): lane groups own weight rows, lanes own packets, and each
//   weight packet is decoded once and applied to every activation row with F32
//   FMAs. The activations are staged once per workgroup in the shared region
//   (A storage) with per-16-element sums for the factored biases.
// - GEMM (M > 8): TM x TN output tiles over subgroups of 32 x 32, stepping K
//   by 32. Activations are staged as f16 and weights decoded to f16 once per
//   tile. Under SEISMIC_HAS_MATRIX (cooperative-matrix devices) 16x16x16
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
//   GEMV   PROJECTION_GEMV_SQUARES_BYTES (row-norm partial squares), the
//          NVFP4 operand rows' global scales (PROJECTION_GEMV_GLOBALS_BYTES,
//          only when the weights take NVFP4 operands), then the staging:
//          min(8 * ceil_div(K, 32), 288) * 72, or, staged as F32,
//          min(8 * ceil_div(K, 32), 144) * 136
//   GEMM   (TM + TN) * 80 (the f16 A and B tiles, rows padded to 40)
//   stage pre-pass: 128 (one partial per subgroup)
//
// This file is independent of any entry ABI.
#include <seismic/packets.glsl>
#include <seismic/activation.glsl>
#include "../core/reduce.glsl"
#include "../core/functions.glsl"
#include "../core/matrix.glsl"
#include "../core/staging.glsl"

// The workgroup's invocations, which `SEISMIC_THREAD` (the prelude's) numbers
// 0 .. PROJECTION_THREADS - 1 for work that reaches a subgroup operation or
// memory read in lane order.
#define PROJECTION_THREADS (gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z)

#include "stage.glsl"
#include "join.glsl"


#define PROJECTION_STORE 0
#define PROJECTION_RESIDUAL 1
#define PROJECTION_SILU_MUL 2
#define PROJECTION_GLU 3
#define PROJECTION_ACTIVATED 4
#define PROJECTION_MUL 5
#define PROJECTION_ACTIVATED_MUL 6
#define PROJECTION_LOGITS 7
#define PROJECTION_SCATTER 8
#define PROJECTION_MIX_LOW 9
#define PROJECTION_MIX_INJECT 10
#define PROJECTION_MIX 11
#define PROJECTION_SCATTER_COMBED 12

// Further epilogues (the Metal and CUDA vocabulary):
//   Glu           y = round_A(round_A(act(round_A(gate))) * round_A(up))  (paired;
//                 `glu`, functions.seismic: clamped where the entry's LB
//                 states a limit, as `projection_put_pair` says)
//   Activated     y = round_A(act(round_A(acc)))                  (up-only, ReLU²)
//   Mul           y = gate * up                           (paired; F32, unrounded)
//   ActivatedMul  y = round_A(round_A(act(round_A(acc))) * external[m, n])
//   Logits        y = acc, or cap * tanh(acc / cap) when cap > 0              (F32)
// with `act` a FUNCTIONS_* code (`../core/functions.glsl`).
struct projection_epilogue {
    int kind;
    int act;            // Residual, Scatter*, SiluMul, Glu, Activated*: ELEMENT_* of A
    Out y;              // the result's owner handle
    Out y_f32;          // F32-result epilogues' owner handle (`y` when unsegmented)
    uint64_t y0;
    uint64_t y1;
    uint64_t column;    // Store: first destination column; Mix: the norm weights
    uint64_t residual;  // Residual: F32 residual rows; ActivatedMul: F32 multipliers; Mix: F32 streams
    uint64_t r0;
    uint64_t r1;
    uint64_t rows;      // Residual: row map of the residual (0: AllRows)
    int function;       // Glu, Activated, ActivatedMul: FUNCTIONS_* code; Mix: the norm's ELEMENT_*
    float cap;          // Logits: the softcap (0: none)
    float scale;        // a weight's second-level scale on the accumulator (paired: the gate stream's)
    float scale2;       // paired: the up stream's second-level scale
    uint64_t inject;    // Scatter: F32 inject rows [M, streams]; Mix: F32 inverses [M, streams]
    uint64_t inject0;
    uint streams;       // Scatter, Mix*: hyper-connection streams of `width` columns
    uint width;
    uint64_t outputs;   // the rows' outputs: 0, row m's at m y0; else row m's at the element offset
                        // outputs[m] (I32, `projection_gathered`)
};

// A weight's second-level scale (NVFP4 `.scale`, per tensor or per expert)
// on the F32 accumulator, before the epilogue. The library applies it only
// in entries that define PROJECTION_SCALED before including this file (an
// entry with a scale port of static extent 1), so an unscaled entry
// compiles as before.
projection_epilogue projection_scaled(projection_epilogue out_, float first, float second) {
    out_.scale = first;
    out_.scale2 = second;
    return out_;
}

// A scale port's value at `index` along its expert axis (0 for a
// per-tensor port; `stride` that axis' stride), or 1 for an absent port
// (static `extent` 0: the port is never read).
float projection_scale_factor(uint64_t scale, uint64_t extent, uint64_t stride, uint64_t index) {
    return extent == 0ul ? 1.0 : element_f32_at(scale + index * stride * 4ul);
}

projection_epilogue projection_store(Out y, uint64_t stride0, uint64_t stride1, uint64_t column) {
    return projection_epilogue(PROJECTION_STORE, 0, y, y, stride0, stride1, column, 0ul, 0ul, 0ul, 0ul, 0, 0.0, 1.0,
        1.0, 0ul, 0ul, 1u, 0u, 0ul);
}

projection_epilogue projection_residual(const int act, Out y, uint64_t stride0, uint64_t stride1,
    uint64_t residual, uint64_t residual0, uint64_t residual1, uint64_t rows) {
    return projection_epilogue(PROJECTION_RESIDUAL, act, y, y, stride0, stride1, 0ul, residual, residual0, residual1,
        rows, 0, 0.0, 1.0, 1.0, 0ul, 0ul, 1u, 0u, 0ul);
}

// `scatter`: `streams` hyper-connection streams of `width` columns, each the
// residual plus the output times the stream's inject value (`inject` rows of
// stride `inject0`); `combed`, each target stream d the residual streams
// mixed by the comb after the inject values (sum_s comb[s, d] residual[s],
// sources ascending) plus the output times inject[d].
projection_epilogue projection_scatter(const int act, Out y, uint64_t stride0, uint64_t stride1,
    uint64_t residual, uint64_t residual0, uint64_t residual1, uint64_t rows, uint64_t inject, uint64_t inject0,
    uint streams, uint width, const bool combed) {
    return projection_epilogue(combed ? PROJECTION_SCATTER_COMBED : PROJECTION_SCATTER, act, y, y, stride0, stride1,
        0ul, residual, residual0, residual1, rows, 0, 0.0, 1.0, 1.0, inject, inject0, streams, width, 0ul);
}

// An output entry's epilogue by its inject extent `inject_extent`: the
// residual, or the scatter into `streams` streams of `width` columns (combed
// when the inject row carries a comb after its `streams` values).
projection_epilogue projection_output(const int act, Out y, uint64_t stride0, uint64_t stride1,
    uint64_t residual, uint64_t residual0, uint64_t residual1, uint64_t rows, uint64_t inject, uint64_t inject0,
    const uint inject_extent, const uint streams, uint width) {
    return inject_extent != 0u
        ? projection_scatter(act, y, stride0, stride1, residual, residual0, residual1, rows, inject, inject0,
            streams, width, inject_extent > streams)
        : projection_residual(act, y, stride0, stride1, residual, residual0, residual1, rows);
}

// A mix site's low-rank rows (`stream_mix_down`): round_A(silu(v / streams)).
// Carries both result handles so the segmented GEMM keeps both `Out` kinds
// constant across its branches.
projection_epilogue projection_mix_low(Out low_y, Out inject_y, uint64_t stride0, uint64_t stride1, uint streams) {
    return projection_epilogue(PROJECTION_MIX_LOW, 0, low_y, inject_y, stride0, stride1, 0ul, 0ul, 0ul, 0ul, 0ul, 0,
        0.0, 1.0, 1.0, 0ul, 0ul, streams, 0u, 0ul);
}

// A mix site's inject values (`stream_mix_down`): 2 sigmoid(v / streams), F32.
projection_epilogue projection_mix_inject(Out low_y, Out inject_y, uint64_t stride0, uint64_t stride1, uint streams) {
    return projection_epilogue(PROJECTION_MIX_INJECT, 0, low_y, inject_y, stride0, stride1, 0ul, 0ul, 0ul, 0ul, 0ul, 0,
        0.0, 1.0, 1.0, 0ul, 0ul, streams, 0u, 0ul);
}

// A mix site's per-stream partials (`stream_mix_up`): column n of stream
// n / width, sigmoid(v) * x[m, n] * inverse[m, n / width] * norm[n] / streams,
// F32 (`x` the streams' rows, `inverse` their [M, streams] inverses).
projection_epilogue projection_mix(Out y, uint64_t stride0, uint64_t stride1, uint64_t x, uint64_t x0,
    uint64_t x1, uint64_t inverse, uint64_t inverse0, uint64_t norm, const int norm_kind, uint streams, uint width) {
    return projection_epilogue(PROJECTION_MIX, 0, y, y, stride0, stride1, norm, x, x0, x1, 0ul, norm_kind,
        0.0, 1.0, 1.0, inverse, inverse0, streams, width, 0ul);
}

projection_epilogue projection_silu_mul(const int act, Out y, uint64_t stride0, uint64_t stride1) {
    return projection_epilogue(PROJECTION_SILU_MUL, act, y, y, stride0, stride1, 0ul, 0ul, 0ul, 0ul, 0ul, 0, 0.0, 1.0,
        1.0, 0ul, 0ul, 1u, 0u, 0ul);
}

projection_epilogue projection_glu(const int act, Out y, uint64_t stride0, uint64_t stride1, int function) {
    return projection_epilogue(PROJECTION_GLU, act, y, y, stride0, stride1, 0ul, 0ul, 0ul, 0ul, 0ul, function, 0.0, 1.0,
        1.0, 0ul, 0ul, 1u, 0u, 0ul);
}

projection_epilogue projection_activated(const int act, Out y, uint64_t stride0, uint64_t stride1, int function) {
    return projection_epilogue(PROJECTION_ACTIVATED, act, y, y, stride0, stride1, 0ul, 0ul, 0ul, 0ul, 0ul, function,
        0.0, 1.0, 1.0, 0ul, 0ul, 1u, 0u, 0ul);
}

projection_epilogue projection_mul(Out y, uint64_t stride0, uint64_t stride1) {
    return projection_epilogue(PROJECTION_MUL, 0, y, y, stride0, stride1, 0ul, 0ul, 0ul, 0ul, 0ul, 0, 0.0, 1.0,
        1.0, 0ul, 0ul, 1u, 0u, 0ul);
}

projection_epilogue projection_activated_mul(const int act, Out y, uint64_t stride0, uint64_t stride1,
    uint64_t multiplier, uint64_t multiplier0, uint64_t multiplier1, int function) {
    return projection_epilogue(PROJECTION_ACTIVATED_MUL, act, y, y, stride0, stride1, 0ul, multiplier, multiplier0,
        multiplier1, 0ul, function, 0.0, 1.0, 1.0, 0ul, 0ul, 1u, 0u, 0ul);
}

projection_epilogue projection_logits(Out y, uint64_t stride0, uint64_t stride1, float cap) {
    return projection_epilogue(PROJECTION_LOGITS, 0, y, y, stride0, stride1, 0ul, 0ul, 0ul, 0ul, 0ul, 0, cap, 1.0,
        1.0, 0ul, 0ul, 1u, 0u, 0ul);
}

// A routed sublayer's column n of row m joining the residual (`routed_output`,
// `routed_combine`): residual + selected + expert, in that order (`expert` the
// shared expert's weighted output), or with an inject (`inject_extent` != 0)
// for each of the `streams` streams of `width` columns
// residual[s] + (selected + expert) * inject[s].
void projection_routed_join(Out y, uint64_t y0, uint64_t y1, uint64_t residual, uint64_t residual0,
    uint64_t residual1, uint64_t inject, uint64_t inject0, const uint inject_extent, const uint streams, uint width,
    uint64_t m, uint64_t n, float selected, float expert) {
    if (inject_extent == 0u) {
        out_put(y, m * y0 + n * y1,
            element_f32_at(residual + (m * residual0 + n * residual1) * 4ul) + selected + expert);
        return;
    }
    for (uint s = 0u; s < streams; ++s) {
        const uint64_t column = uint64_t(s * width) + n;
        out_put(y, m * y0 + column * y1,
            element_f32_at(residual + (m * residual0 + column * residual1) * 4ul)
                + (selected + expert) * element_f32_at(inject + (m * inject0 + s) * 4ul));
    }
}

// Where row m's outputs start (element offset): m y0, or where the
// epilogue's outputs are gathered (`projection_gathered`), outputs[m].
uint64_t projection_out_row(projection_epilogue out_, uint m) {
    return out_.outputs == 0ul ? uint64_t(m) * out_.y0 : uint64_t(element_i32_at(out_.outputs + uint64_t(m) * 4ul));
}

// `out_` with GEMV row m's outputs at element offset `outputs[m]` (I32 a
// row) instead of m y0: a gathered GEMV's rows publish into the choices they
// were gathered for.
projection_epilogue projection_gathered(projection_epilogue out_, uint64_t outputs) {
    out_.outputs = outputs;
    return out_;
}

// Publishes one output of an unpaired epilogue.
void projection_put(projection_epilogue out_, uint m, uint n, float value) {
#if defined(PROJECTION_SCALED)
    value *= out_.scale;
#endif
    const uint64_t at = projection_out_row(out_, m) + uint64_t(n) * out_.y1;
    if (out_.kind == PROJECTION_STORE) {
        out_put(out_.y, projection_out_row(out_, m) + (out_.column + uint64_t(n)) * out_.y1, value);
    } else if (out_.kind == PROJECTION_ACTIVATED) {
        out_put(out_.y, at, functions_activate(out_.function, element_round(out_.act, value)));
    } else if (out_.kind == PROJECTION_ACTIVATED_MUL) {
        const float activated = element_round(out_.act,
            functions_activate(out_.function, element_round(out_.act, value)));
        const uint64_t source = uint64_t(m) * out_.r0 + uint64_t(n) * out_.r1;
        out_put(out_.y, at, activated * element_f32_at(out_.residual + source * 4ul));
    } else if (out_.kind == PROJECTION_LOGITS) {
        out_put(out_.y_f32, at, out_.cap > 0.0 ? functions_softcap(out_.cap, value) : value);
    } else if (out_.kind == PROJECTION_SCATTER || out_.kind == PROJECTION_SCATTER_COMBED) {
        join_scatter(out_offset(out_.y, projection_out_row(out_, m)), out_.y1,
            out_.residual + uint64_t(projection_row(out_.rows, m)) * out_.r0 * 4ul, out_.r1,
            out_.inject + uint64_t(m) * out_.inject0 * 4ul, out_.streams, out_.width,
            out_.kind == PROJECTION_SCATTER_COMBED, uint64_t(n), element_round(out_.act, value));
    } else if (out_.kind == PROJECTION_MIX_LOW) {
        const float v = seismic_div_rn(value, float(out_.streams));
        out_put(out_.y, at, seismic_div_rn(v, 1.0 + exp(-v)));
    } else if (out_.kind == PROJECTION_MIX_INJECT) {
        out_put(out_.y_f32, at,
            seismic_div_rn(2.0, 1.0 + exp(-seismic_div_rn(value, float(out_.streams)))));
    } else if (out_.kind == PROJECTION_MIX) {
        const float normalized = element_f32_at(out_.residual + (uint64_t(m) * out_.r0 + uint64_t(n) * out_.r1) * 4ul)
            * element_f32_at(out_.inject + (uint64_t(m) * out_.inject0 + uint64_t(n / out_.width)) * 4ul)
            * element_at(out_.function, out_.column, uint64_t(n));
        const float gate = seismic_div_rn(1.0, 1.0 + exp(-value));
        out_put(out_.y_f32, at, seismic_div_rn(gate * normalized, float(out_.streams)));
    } else {
        const uint64_t source = uint64_t(projection_row(out_.rows, m)) * out_.r0 + uint64_t(n) * out_.r1;
        out_put(out_.y, at, element_f32_at(out_.residual + source * 4ul) + element_round(out_.act, value));
    }
}

// Publishes one paired feature from its gate and up sums.
void projection_put_pair(projection_epilogue out_, uint m, uint n, float gate_sum, float up_sum) {
#if defined(PROJECTION_SCALED)
    gate_sum *= out_.scale;
    up_sum *= out_.scale2;
#endif
    const uint64_t at = projection_out_row(out_, m) + uint64_t(n) * out_.y1;
    if (out_.kind == PROJECTION_MUL) {
        out_put(out_.y_f32, at, gate_sum * up_sum);
        return;
    }
#if defined(PROJECTION_GLU_LIMIT)
    // PROJECTION_GLU_LIMIT, the entry's nonzero LB: the F32 bits of its
    // expansion's limit L. A Glu clamps the rounded gate to (-inf, L] and up
    // to [-L, L], each written `x > L ? L : x` so a NaN stays NaN. An entry
    // without one compiles none.
    const float limit = uintBitsToFloat(uint(PROJECTION_GLU_LIMIT));
    float gate = element_round(out_.act, gate_sum);
    float up = element_round(out_.act, up_sum);
    if (out_.kind == PROJECTION_GLU) {
        gate = gate > limit ? limit : gate;
        up = up > limit ? limit : up;
        up = up < -limit ? -limit : up;
    }
#else
    const float gate = element_round(out_.act, gate_sum);
    const float up = element_round(out_.act, up_sum);
#endif
    const float activated = out_.kind == PROJECTION_GLU
        ? element_round(out_.act, functions_activate(out_.function, gate))
        : projection_silu_rounded(out_.act, gate);
    out_put(out_.y, at, activated * up);
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
    uint64_t table_stride;
    uint packet_first;  // the stored packet of packet 0 (a window of whole packets; `k` stays the row's)
};

// An entry whose table may select past the tensor's rows states them
// (`PROJECTION_TABLE_ROWS`): such a row reads row 0, whose results the entry
// masks.
uint64_t projection_weights_row(projection_weights w, uint n) {
    uint64_t r = w.table == 0ul ? uint64_t(n) : uint64_t(element_i32_at(w.table + uint64_t(n) * w.table_stride * 4ul));
#ifdef PROJECTION_TABLE_ROWS
    r = r < uint64_t(PROJECTION_TABLE_ROWS) ? r : 0ul;
#endif
    return packets_row(w.base, w.geometry, r);
}

packets_packet projection_packet(const int kind, projection_weights w, uint n, uint p) {
    return packets_load(kind, projection_weights_row(w, n), w.geometry, w.packet_first + p, w.k);
}

// ---------------------------------------------------------------------------
// The stage pre-pass of the GEMM class.

// The Rms norm inverse of columns first .. first + width of row m, reduced
// across the workgroup (shared: one float per subgroup at float 0).
float projection_stage_inverse(projection_prologue in_, uint m, uint first, uint width) {
    const uint thread = SEISMIC_THREAD, threads = PROJECTION_THREADS;
    float squares = 0.0;
    seismic_strided(uint, i, thread, width, threads) {
        const float v = projection_norm_input(in_, m, first + i);
        squares = seismic_fma_rn(v, v, squares);
    }
    squares = reduce_group_sum(squares, 0u);
    return inversesqrt(seismic_div_rn(squares, float(width)) + in_.eps);
}

// The largest magnitude of row m's prologue output under its norm inverse
// (NaN wins: `nvfp4_peak`), reduced across the workgroup (shared: one
// float per subgroup at float 0, which every invocation may still be
// reading: a normalization's squares).
float projection_row_maximum(projection_prologue in_, uint m, float inverse) {
    const uint thread = SEISMIC_THREAD, threads = PROJECTION_THREADS;
    uint amax = 0u;
    seismic_strided(uint, i, 8u * thread, in_.columns, 8u * threads) {
        vec4 even, odd;
        projection_load8(in_, m, i, inverse, even, odd);
        amax = nvfp4_peak(amax, even, odd);
    }
    return uintBitsToFloat(reduce_group_max_u32(amax, 0u));
}

// Workgroup `m` stores row m's prologue output (an Rms prologue's normalized
// by the row's inverse) as its weights' operands (`projection_operands`), in
// A, at x[m * columns ..]. The GEMM launches then read `x` as a Plain
// operand. Shared: one float per subgroup at float 0.
void projection_stage(projection_prologue in_, uint m, uint64_t x, uint columns) {
    const uint thread = gl_LocalInvocationIndex, threads = gl_WorkGroupSize.x;
    const uint width = in_.columns;
    float inverse = 0.0;
    if (in_.kind == PROJECTION_RMS)
        inverse = projection_stage_inverse(in_, m, 0u, width);
    const float g = in_.operands == PROJECTION_OPERANDS_NVFP4
        ? nvfp4_global_scale(projection_row_maximum(in_, m, inverse))
        : 0.0;
    const uint64_t row = uint64_t(m) * columns;
    const bool vector = (columns & 7u) == 0u;
    for (uint i = 8u * thread; i < width; i += 8u * threads) {
        vec4 even, odd;
        projection_operand_load8(in_, m, i, inverse, g, even, odd);
        if (vector && i + 8u <= width) {
            element_uvec4_put(x + (row + i) * 2ul, element_pack8(in_.act, even, odd));
        } else {
            for (uint j = 0u; j < 8u && i + j < width; ++j)
                element_put(in_.act, x, row + i + j, (j & 1u) != 0u ? odd[j >> 1] : even[j >> 1]);
        }
    }
}

// `projection_stage` over one hyper-connection stream (`group_rms`): the
// workgroup normalizes columns stream * width .. of row m of an Rms
// prologue over whole rows of streams by the stream's own inverse, stores
// them in A at x[m * columns + stream * width ..] and returns the inverse.
float projection_stage_stream(projection_prologue in_, uint m, uint stream, uint width, uint64_t x, uint columns) {
    const uint thread = gl_LocalInvocationIndex, threads = gl_WorkGroupSize.x;
    const uint first = stream * width;
    const float inverse = projection_stage_inverse(in_, m, first, width);
    const uint64_t row = uint64_t(m) * columns + first;
    const bool vector = (columns & 7u) == 0u && (first & 7u) == 0u;
    for (uint i = 8u * thread; i < width; i += 8u * threads) {
        vec4 even, odd;
        projection_load8(in_, m, first + i, inverse, even, odd);
        if (vector && i + 8u <= width) {
            element_uvec4_put(x + (row + i) * 2ul, element_pack8(in_.act, even, odd));
        } else {
            for (uint j = 0u; j < 8u && i + j < width; ++j)
                element_put(in_.act, x, row + i + j, (j & 1u) != 0u ? odd[j >> 1] : even[j >> 1]);
        }
    }
    return inverse;
}

// The A row (of element kind `act`) at `row`, `columns` values, formed in
// place as NVFP4 operands by the workgroup: each invocation reads and
// rewrites whole 16-value blocks, so none reads a value another has
// rewritten (the grouped expert products a down projection takes). Shared:
// one float per subgroup at float 0. The blocks are numbered by
// `SEISMIC_THREAD`: `projection_row_maximum` reads the row in lane order.
void projection_form_in_place(const int act, uint64_t row, uint columns) {
    const uint thread = SEISMIC_THREAD, threads = PROJECTION_THREADS;
    const projection_prologue in_ = projection_operands(projection_plain(act, row, 0ul, 1ul, columns, 0ul),
        PROJECTION_OPERANDS_NVFP4);
    const float g = nvfp4_global_scale(projection_row_maximum(in_, 0u, 0.0));
    for (uint block = 16u * thread; block < columns; block += 16u * threads) {
        vec4 even[2], odd[2];
        projection_operand_load8(in_, 0u, block, 0.0, g, even[0], odd[0]);
        projection_operand_load8(in_, 0u, block + 8u, 0.0, g, even[1], odd[1]);
        for (uint j = 0u; j < 16u && block + j < columns; ++j)
            element_put(act, row, uint64_t(block + j),
                (j & 1u) != 0u ? odd[j >> 3][(j & 7u) >> 1] : even[j >> 3][(j & 7u) >> 1]);
    }
}

// ---------------------------------------------------------------------------
// GEMV (M <= 8).
//
// A workgroup of SG subgroups; a subgroup is 32 / LANES lane groups; lane
// group g of subgroup sg owns the R weight rows from
// ((tile * SG + sg) * (32 / LANES) + g) * R, and its lanes own the packets
// sub, sub + LANES, ... of those rows. The prologue output is staged per
// workgroup in chunks of at most 288 / maxm packets (a multiple of LANES):
// the eight values of chunk-local packet p, step s and row m as one uvec4 in
// A storage, and the packet's two 16-column sums. Unpaired projections and
// the bound 1 double-buffer weight packets in registers: a lane loads its next
// packet before accumulating the current one, and its first packet before the
// staging. A paired projection under a larger bound gives those registers to
// its 2 * R * maxm accumulators and loads each packet before accumulating it:
// on an RTX 3090 (NVIDIA 595), double-buffering paired weights under bounds
// 2 to 8 spilled accumulators or doubled the registers per invocation, costing
// dense_expand up to 26%, while single-buffering cost up to 5% on unpaired
// projections and 4% under the bound 1.
//
// Norms: PROJECTION_NORM_NONE (Plain), PROJECTION_NORM_SHARED (Rms: the rows'
// square sums are reduced first into PROJECTION_RMS_PARTS parts in the shared
// region, each by one subgroup, and summed in a fixed tree, so the inverses do
// not depend on the workgroup shape).

#define PROJECTION_NORM_NONE 0
#define PROJECTION_NORM_SHARED 1

#define PROJECTION_GEMV_STAGE_PACKETS 288u
// Row-norm partial squares at float 0: 8 rows x PROJECTION_RMS_PARTS.
#define PROJECTION_GEMV_SQUARES_BYTES 256
// NVFP4 operand rows' global scales after the squares, one float a row
// (`projection_gemv_globals`), in PROJECTION_GEMV_GLOBALS_BYTES.
#define PROJECTION_GEMV_GLOBALS 64u
#define PROJECTION_GEMV_GLOBALS_BYTES 32

// The first uvec4 of the staging: after the squares, and the global scales
// of NVFP4 operand rows.
uint projection_gemv_stage_word(projection_prologue in_) {
    return (PROJECTION_GEMV_SQUARES_BYTES
               + (in_.operands == PROJECTION_OPERANDS_NVFP4 ? PROJECTION_GEMV_GLOBALS_BYTES : 0)) / 16u;
}

// A staged packet's bytes a row: its values in A storage, or as F32 where
// `f32`, and its two 16-column sums.
uint projection_gemv_packet_bytes(const bool f32) {
    return f32 ? 136u : 72u;
}

// The packets of every activation row the GEMV stages at once:
// PROJECTION_GEMV_STAGE_PACKETS (half of them where `f32`), or, where the
// launch holds its weights' tables at its region's end
// (`packets_reserved_words`), what the region holds before them, which its
// declaration sizes to at least 8 * LANES (`dense.seismic`).
uint projection_gemv_stage_packets(projection_prologue in_, const bool f32) {
    const uint most = f32 ? PROJECTION_GEMV_STAGE_PACKETS / 2u : PROJECTION_GEMV_STAGE_PACKETS;
#if defined(PACKETS_STAGING)
    if (packets_reserved_words() != 0u)
        return min(most, (4u * (SEISMIC_SHARED_U32 - packets_reserved_words()) - 16u * projection_gemv_stage_word(in_))
            / projection_gemv_packet_bytes(f32));
#endif
    return most;
}

// The staging's uvec4 of activation row m's eight values at step `step` of
// chunk-local packet `local`: in A storage, or, where `f32`, the even values
// as F32 there and the odd ones `chunk` on.
uint projection_gemv_x_word(const bool f32, uint word, uint m, uint step, uint chunk, uint local) {
    return word + (f32 ? 2u : 1u) * (m * 4u + step) * chunk + local;
}

// The staging's float of the first of the packets' 16-column sums.
uint projection_gemv_sums_float(const bool f32, uint word, const uint maxm, uint chunk) {
    return 4u * (word + (f32 ? 8u : 4u) * maxm * chunk);
}

// The eight staged values at `at` (`projection_gemv_x_word`).
void projection_gemv_x8(const int act, const bool f32, uint at, uint chunk, out vec4 xe, out vec4 xo) {
    if (f32) {
        xe = uintBitsToFloat(seismic_shared_uvec4[at]);
        xo = uintBitsToFloat(seismic_shared_uvec4[at + chunk]);
    } else {
        element_split8(act, seismic_shared_uvec4[at], xe, xo);
    }
}
// The register arrays are sized for the largest R and bound; every loop over
// weight rows runs to `r_rows` and every loop over activation rows to `maxm`,
// both literals in each pipeline, so the driver unrolls only the iterations
// that run (its compile time grows with the unrolled body).
#define PROJECTION_GEMV_MAX_R 4
#define PROJECTION_GEMV_MAX_M 8
#ifndef SEISMIC_CONST_VERIFY_ROWS
#error "projection.glsl needs SEISMIC_CONST_VERIFY_ROWS"
#endif
#if SEISMIC_CONST_VERIFY_ROWS > PROJECTION_GEMV_MAX_M
#error "the GEMV classes serve at most PROJECTION_GEMV_MAX_M rows"
#endif
#define PROJECTION_GEMV_SUMS (PROJECTION_GEMV_MAX_R * PROJECTION_GEMV_MAX_M)

// Weight rows of one GEMV workgroup (segmented entries map workgroups to
// segments by it).
uint projection_gemv_rows(const uint sg_count, const uint r_rows, const uint lanes) {
    return sg_count * r_rows * (32u / lanes);
}

// The first weight row of this invocation's lane group.
uint projection_gemv_first_row(const uint r_rows, const uint lanes, uint tile) {
    return ((tile * SEISMIC_SUBGROUPS + SEISMIC_SUBGROUP) * (32u / lanes) + SEISMIC_LANE / lanes) * r_rows;
}

// Row-norm square sums of the workgroup's rows into the shared region;
// subgroup sg reduces (row, part) items sg, sg + SG, ... The GEMV body's
// opening barrier publishes them.
void projection_gemv_squares(projection_prologue in_, uint m_rows) {
    const uint slots = PROJECTION_RMS_PARTS;
    if (in_.kind == PROJECTION_ORDERED_RMS || in_.kind == PROJECTION_ORDERED_SUMMED) {
        projection_ordered_inverses(in_, m_rows, 0u, slots, 0u, m_rows * slots <= 64u, 64u);
        return;
    }
    // A logical subgroup past the items (on 64 lanes, beside one with an
    // item) reduces the last item again and stores nothing.
    bool held;
    for (uint item = SEISMIC_SUBGROUP; seismic_steer(held, item < m_rows * slots); item += SEISMIC_SUBGROUPS) {
        const uint at = held ? item : m_rows * slots - 1u;
        const float sum = seismic_subgroup_sum_f32(projection_rms_squares(in_, at / slots, at % slots, SEISMIC_LANE));
        if (held && SEISMIC_LANE == 0u)
            seismic_shared_f32[item] = sum;
    }
}

float projection_gemv_inverse(projection_prologue in_, uint m) {
    const uint at = m * PROJECTION_RMS_PARTS;
    if (in_.kind == PROJECTION_ORDERED_RMS || in_.kind == PROJECTION_ORDERED_SUMMED)
        return seismic_shared_f32[at];
    const float sum = ((seismic_shared_f32[at] + seismic_shared_f32[at + 1u])
                          + (seismic_shared_f32[at + 2u] + seismic_shared_f32[at + 3u]))
        + ((seismic_shared_f32[at + 4u] + seismic_shared_f32[at + 5u])
            + (seismic_shared_f32[at + 6u] + seismic_shared_f32[at + 7u]));
    return inversesqrt(seismic_div_rn(sum, float(in_.columns)) + in_.eps);
}

// The norm inverse of row m (0 without a norm).
float projection_gemv_norm_inverse(projection_prologue in_, const int norm, uint m) {
    return norm == PROJECTION_NORM_SHARED ? projection_gemv_inverse(in_, m) : 0.0;
}

// The global scales of the workgroup's NVFP4 operand rows into the shared
// region: after a barrier that publishes the row-norm squares (and orders
// the previous call's staging reads), subgroup sg reduces rows sg, sg + SG,
// ... The staging's opening barrier publishes them.
void projection_gemv_globals(projection_prologue in_, const int norm, uint m_rows) {
    barrier();
    // As `projection_gemv_squares` takes its items.
    bool held;
    for (uint m = SEISMIC_SUBGROUP; seismic_steer(held, m < m_rows); m += SEISMIC_SUBGROUPS) {
        const uint row = held ? m : m_rows - 1u;
        const float inverse = projection_gemv_norm_inverse(in_, norm, row);
        uint amax = 0u;
        seismic_strided(uint, k, 8u * SEISMIC_LANE, in_.columns, 8u * 32u) {
            vec4 even, odd;
            projection_load8(in_, row, k, inverse, even, odd);
            amax = nvfp4_peak(amax, even, odd);
        }
        amax = seismic_redux_max_u32(amax);
        if (held && SEISMIC_LANE == 0u)
            seismic_shared_f32[PROJECTION_GEMV_GLOBALS + m] = nvfp4_global_scale(uintBitsToFloat(amax));
    }
}

// The staged prologue output of x[m, k..k+8), as the prologue's operands.
void projection_gemv_prologue8(projection_prologue in_, const int norm, uint m, uint k, out vec4 even, out vec4 odd) {
    const float g = in_.operands == PROJECTION_OPERANDS_NVFP4 ? seismic_shared_f32[PROJECTION_GEMV_GLOBALS + m] : 0.0;
    projection_operand_load8(in_, m, k, projection_gemv_norm_inverse(in_, norm, m), g, even, odd);
}

// Stage packets first .. first + count of every activation row: one
// invocation per eight columns (row m, packet, step); the four steps of a
// packet sit on adjacent lanes, which add their sums pairwise into the
// packet's two 16-column sums. Where `f32`, the values are rounded to A and
// stored as F32, the same values a reader takes without splitting A.
void projection_gemv_stage(projection_prologue in_, const int norm, const uint maxm, const bool f32, uint m_rows,
    uint first, uint count, uint chunk) {
    const uint word = projection_gemv_stage_word(in_);
    const uint sums = projection_gemv_sums_float(f32, word, maxm, chunk);
    // Each logical subgroup takes 32 items a round, and all its lanes take
    // part in the pairing shuffle: it runs inside the loop, so the loop
    // steers whole subgroups and a lane past the items stages nothing (items
    // come in whole packets, so a pair is staged or not together).
    const uint items = 4u * m_rows * count;
    bool held;
    for (uint base = 32u * SEISMIC_SUBGROUP; seismic_steer(held, base < items); base += 32u * SEISMIC_SUBGROUPS) {
        const uint item = base + SEISMIC_LANE;
        const uint step = item & 3u, packet = item >> 2;
        // Under the bound 1 every packet is of row 0, without a division.
        const uint m = maxm == 1u ? 0u : packet / count;
        const uint local = maxm == 1u ? packet : packet - m * count;
        float sum = 0.0;
        if (item < items) {
            vec4 even, odd;
            projection_gemv_prologue8(in_, norm, m, 32u * (first + local) + 8u * step, even, odd);
            const uint at = projection_gemv_x_word(f32, word, m, step, chunk, local);
            if (f32) {
                vec4 e, o;
                element_split8(in_.act, element_pack8(in_.act, even, odd), e, o);
                seismic_shared_uvec4[at] = floatBitsToUint(e);
                seismic_shared_uvec4[at + chunk] = floatBitsToUint(o);
            } else {
                seismic_shared_uvec4[at] = element_pack8(in_.act, even, odd);
            }
            const vec4 pair = even + odd;
            sum = (pair.x + pair.y) + (pair.z + pair.w);
        }
        sum += subgroupShuffleXor(sum, 1u);
        if (item < items && (step & 1u) == 0u)
            seismic_shared_f32[sums + 2u * (m * chunk + local) + (step >> 1)] = sum;
    }
}

// One packet of the R weight rows (and, paired, of the second tensor).
void projection_gemv_load(const int wk, const int uk, const bool paired, const uint r_rows, projection_weights w,
    projection_weights u, uint first_row, uint rows, uint p, out packets_packet a[PROJECTION_GEMV_MAX_R],
    out packets_packet b[PROJECTION_GEMV_MAX_R]) {
    [[unroll]] for (uint r = 0u; r < r_rows; ++r) {
        const uint n = min(first_row + r, rows - 1u);
        a[r] = projection_packet(wk, w, n, p);
        if (paired)
            b[r] = projection_packet(uk, u, n, p);
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

// The dot product of sub-step `step` of `packet`'s codes with eight
// activations: the codes decoded (`packets_codes`, `arithmetic` where asked)
// into projection_dot8, or, for codes that are signs decoded by
// `arithmetic`, the activations with their signs flipped where a code is -1
// and added in the same order: a sign's product is exact, so the roundings
// are projection_dot8's.
float projection_code_dot8(const int kind, packets_packet packet, uint step, const bool arithmetic, vec4 xe, vec4 xo) {
    if (arithmetic && packets_signs(kind)) {
        uvec4 me, mo;
        packets_sign_masks(packet, step, me, mo);
        const vec4 e = uintBitsToFloat(floatBitsToUint(xe) ^ me), o = uintBitsToFloat(floatBitsToUint(xo) ^ mo);
        return ((((((e.x + o.x) + e.y) + o.y) + e.z) + o.z) + e.w) + o.w;
    }
    vec4 ce, co;
    packets_codes(kind, packet, step, arithmetic, ce, co);
    return projection_dot8(ce, co, xe, xo);
}

// Accumulate one loaded packet (chunk-local `local`) for every activation row
// of the staging from uvec4 `word` (as F32 where `f32`), its codes decoded by
// `arithmetic` where asked (`packets_codes`).
void projection_gemv_accumulate(const int wk, const int uk, const bool paired, const uint r_rows, const uint maxm,
    const bool f32, const bool arithmetic, const int act, inout float acc[PROJECTION_GEMV_SUMS],
    inout float acc2[PROJECTION_GEMV_SUMS], packets_packet a[PROJECTION_GEMV_MAX_R],
    packets_packet b[PROJECTION_GEMV_MAX_R], uint local, uint m_rows, uint chunk, uint word) {
    const uint sums = projection_gemv_sums_float(f32, word, maxm, chunk);
    [[unroll]] for (uint step = 0u; step < 4u; ++step) {
        if (maxm == 1u) {
            // One activation row: each weight row's codes are decoded where
            // they are used, never held for all R rows at once.
            if (m_rows > 0u) {
                vec4 xe, xo;
                projection_gemv_x8(act, f32, projection_gemv_x_word(f32, word, 0u, step, chunk, local), chunk, xe, xo);
                [[unroll]] for (uint r = 0u; r < r_rows; ++r) {
                    const uint at = r * PROJECTION_GEMV_MAX_M;
                    acc[at] = seismic_fma_rn(packets_scale(wk, a[r], step),
                        projection_code_dot8(wk, a[r], step, arithmetic, xe, xo), acc[at]);
                    if (paired)
                        acc2[at] = seismic_fma_rn(packets_scale(uk, b[r], step),
                            projection_code_dot8(uk, b[r], step, arithmetic, xe, xo), acc2[at]);
                }
            }
        } else {
            vec4 ae[PROJECTION_GEMV_MAX_R], ao[PROJECTION_GEMV_MAX_R], be[PROJECTION_GEMV_MAX_R], bo[PROJECTION_GEMV_MAX_R];
            [[unroll]] for (uint r = 0u; r < r_rows; ++r) {
                packets_codes(wk, a[r], step, arithmetic, ae[r], ao[r]);
                if (paired)
                    packets_codes(uk, b[r], step, arithmetic, be[r], bo[r]);
            }
            [[unroll]] for (uint m = 0u; m < maxm; ++m) {
                if (m < m_rows) {
                    vec4 xe, xo;
                    projection_gemv_x8(act, f32, projection_gemv_x_word(f32, word, m, step, chunk, local), chunk, xe,
                        xo);
                    [[unroll]] for (uint r = 0u; r < r_rows; ++r) {
                        const uint at = r * PROJECTION_GEMV_MAX_M + m;
                        acc[at] = seismic_fma_rn(packets_scale(wk, a[r], step), projection_dot8(ae[r], ao[r], xe, xo), acc[at]);
                        if (paired)
                            acc2[at] = seismic_fma_rn(packets_scale(uk, b[r], step), projection_dot8(be[r], bo[r], xe, xo), acc2[at]);
                    }
                }
            }
        }
    }
#if PACKETS_ANY_BIASED
    if (packets_biased(wk) || (paired && packets_biased(uk))) {
        [[unroll]] for (uint m = 0u; m < maxm; ++m) {
            if (m < m_rows) {
                const vec2 s = vec2(seismic_shared_f32[sums + 2u * (m * chunk + local)],
                    seismic_shared_f32[sums + 2u * (m * chunk + local) + 1u]);
                [[unroll]] for (uint r = 0u; r < r_rows; ++r) {
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
#endif
}

// The sums of one GEMV workgroup with the activation-row bound `maxm` (1, 2, 4
// or 8): every lane of a lane group receives the group's totals of its R
// weight rows (from projection_gemv_first_row) and every activation row, at
// totals[r * PROJECTION_GEMV_MAX_M + m] (totals2: the paired weight).
void projection_gemv_sums(const int wk, const int uk, const bool paired, const uint r_rows, const uint lanes,
    const uint maxm, const bool f32, const bool arithmetic, const int norm, projection_prologue in_,
    projection_weights w, projection_weights u, uint m_rows, uint rows, uint k, uint tile,
    out float totals[PROJECTION_GEMV_SUMS], out float totals2[PROJECTION_GEMV_SUMS]) {
    const uint sub = SEISMIC_LANE % lanes;
    const uint packets = (k + 31u) / 32u;
    // Every packet at once where the staging holds them, else a multiple of
    // LANES of them.
    const uint staged = projection_gemv_stage_packets(in_, f32) / maxm;
    const uint chunk = staged >= packets ? packets : staged / lanes * lanes;
    const bool double_buffer = !paired || maxm == 1u;
    packets_packet current_a[PROJECTION_GEMV_MAX_R], current_b[PROJECTION_GEMV_MAX_R];
    if (double_buffer) {
        const uint first_row = projection_gemv_first_row(r_rows, lanes, tile);
        if (first_row < rows && sub < packets)
            projection_gemv_load(wk, uk, paired, r_rows, w, u, first_row, rows, sub, current_a, current_b);
    }
    if (in_.operands == PROJECTION_OPERANDS_NVFP4)
        projection_gemv_globals(in_, norm, m_rows);
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
        projection_gemv_stage(in_, norm, maxm, f32, m_rows, first, count, chunk);
        barrier();
        // Recomputed from builtins rather than held across the chunk loop.
        const uint first_row = projection_gemv_first_row(r_rows, lanes, tile);
        if (first_row < rows) {
            // Every lane of a group runs the same iterations (chunk is a
            // multiple of LANES, so aligned packets keep count aligned too);
            // only the last of an unaligned count skips its lanes past it.
            for (uint base = 0u; base < count; base += lanes) {
                const uint local = base + sub;
                if (packets % lanes == 0u || local < count) {
                    if (double_buffer) {
                        // The next packet is copied only when it was loaded:
                        // copying the unwritten registers past the last
                        // packet is undefined (the NVIDIA driver then zeroed
                        // paired dense sums). With aligned packets the whole
                        // group has a next packet or none.
                        packets_packet next_a[PROJECTION_GEMV_MAX_R], next_b[PROJECTION_GEMV_MAX_R];
                        const bool more = packets % lanes == 0u ? first + base + lanes < packets
                                                                : first + local + lanes < packets;
                        if (more)
                            projection_gemv_load(wk, uk, paired, r_rows, w, u, first_row, rows, first + local + lanes,
                                next_a, next_b);
                        projection_gemv_accumulate(wk, uk, paired, r_rows, maxm, f32, arithmetic, in_.act, acc, acc2,
                            current_a, current_b, local, m_rows, chunk, projection_gemv_stage_word(in_));
                        if (more) {
                            current_a = next_a;
                            current_b = next_b;
                        }
                    } else {
                        packets_packet a[PROJECTION_GEMV_MAX_R], b[PROJECTION_GEMV_MAX_R];
                        projection_gemv_load(wk, uk, paired, r_rows, w, u, first_row, rows, first + local, a, b);
                        projection_gemv_accumulate(wk, uk, paired, r_rows, maxm, f32, arithmetic, in_.act, acc, acc2,
                            a, b, local, m_rows, chunk, projection_gemv_stage_word(in_));
                    }
                }
            }
        }
    }
    [[unroll]] for (uint i = 0u; i < PROJECTION_GEMV_SUMS; ++i) {
        totals[i] = 0.0;
        totals2[i] = 0.0;
    }
    [[unroll]] for (uint r = 0u; r < r_rows; ++r) {
        [[unroll]] for (uint m = 0u; m < maxm; ++m) {
            const uint at = r * PROJECTION_GEMV_MAX_M + m;
            if (m < m_rows) {
                totals[at] = reduce_lanes_sum(acc[at], lanes);
                if (paired)
                    totals2[at] = reduce_lanes_sum(acc2[at], lanes);
            }
        }
    }
}

// The GEMV of one segment: weight kinds `wk` (and `uk` of the up weights when
// paired), R rows per lane group, LANES lanes per group, and the
// activation-row bound `maxm` (1, 2, 4 or 8; m_rows <= maxm), activations
// staged as F32 where `f32` (`projection_gemv_stage`) and codes decoded by
// `arithmetic` where asked (`packets_codes`, and signs by the
// activations' sign bits, `projection_code_dot8`): the sums, then the
// epilogue of each (weight row, activation row) from one lane of its group.
// Rms prologues first reduce their rows' squares (norm SHARED). Every
// invocation of the workgroup must call it (it contains barriers).
//
// One bound per pipeline: every call passes `maxm` as a literal, and an entry
// serving several bounds has one entry function (and one launch) per bound.
// The body unrolls over MAX_R x MAX_M x 4 steps, and the NVIDIA proprietary
// driver's pipeline creation is superlinear in function size: one pipeline
// holding the four bounds behind a runtime row-count switch took 10-25 s to
// create (dense_expand, rtx3090), against about 1 s for one bound, while a
// single bound of 8 for every row count lost 16% of one-row decode bandwidth.
void projection_gemv(const int wk, const int uk, const bool paired, const uint r_rows, const uint lanes,
    const uint maxm, const bool f32, const bool arithmetic, const int norm, projection_prologue in_,
    projection_epilogue out_, projection_weights w, projection_weights u, uint m_rows, uint rows, uint k, uint tile) {
    if (norm == PROJECTION_NORM_SHARED)
        projection_gemv_squares(in_, m_rows);
    float totals[PROJECTION_GEMV_SUMS], totals2[PROJECTION_GEMV_SUMS];
    projection_gemv_sums(wk, uk, paired, r_rows, lanes, maxm, f32, arithmetic, norm, in_, w, u, m_rows, rows, k, tile,
        totals, totals2);
    const uint sub = SEISMIC_LANE % lanes;
    const uint first_row = projection_gemv_first_row(r_rows, lanes, tile);
    [[unroll]] for (uint r = 0u; r < r_rows; ++r) {
        [[unroll]] for (uint m = 0u; m < maxm; ++m) {
            const uint at = r * PROJECTION_GEMV_MAX_M + m;
            if (m < m_rows && sub == (r * maxm + m) % lanes && first_row + r < rows)
                projection_emit(out_, paired, m, first_row + r, totals[at], totals2[at]);
        }
    }
}



// ---------------------------------------------------------------------------
// Reference-order (`ordered`) forms (`projection_ordered_stage_row`,
// `projection_ordered_gemv`, `projection_ordered_gemm`): 8-lane dot products
// matching the CPU reference order bit for bit.

float projection_ordered_stage_row(projection_prologue in_, uint m, Out out_, uint64_t stride0, uint64_t stride1) {
    const uint tid = SEISMIC_THREAD, threads = SEISMIC_SUBGROUPS * 32u;
    projection_ordered_inverses(in_, 1u, 0u, 1u, m, true, 4u);
    const float inv = seismic_shared_f32[0];
    for (uint c = tid; c < in_.columns; c += threads)
        out_put(out_, uint64_t(m) * stride0 + uint64_t(c) * stride1, projection_ordered_at(in_, m, c, inv));
    return inv;
}

void projection_ordered_transpose8(inout float v[8], uint u) {
    [[unroll]] for (uint i = 0u; i < 8u; i += 2u) {
        const float send = (u & 1u) != 0u ? v[i] : v[i + 1u];
        const float recv = subgroupShuffleXor(send, 1u);
        if ((u & 1u) != 0u)
            v[i] = recv;
        else
            v[i + 1u] = recv;
    }
    [[unroll]] for (uint g = 0u; g < 8u; g += 4u) {
        [[unroll]] for (uint j = 0u; j < 2u; ++j) {
            const uint i = g + j;
            const float send = (u & 2u) != 0u ? v[i] : v[i + 2u];
            const float recv = subgroupShuffleXor(send, 2u);
            if ((u & 2u) != 0u)
                v[i] = recv;
            else
                v[i + 2u] = recv;
        }
    }
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        const float send = (u & 4u) != 0u ? v[i] : v[i + 4u];
        const float recv = subgroupShuffleXor(send, 4u);
        if ((u & 4u) != 0u)
            v[i] = recv;
        else
            v[i + 4u] = recv;
    }
}

void projection_ordered_gemv(const int kind, const bool publish, const uint maxm, projection_prologue in_,
    projection_epilogue out_, projection_weights w, Out pub_out, uint64_t pub_s0, uint64_t pub_s1, bool publish_rows,
    uint m, uint n, uint k, uint local_tile) {
    const uint sg = SEISMIC_SUBGROUP, lane = SEISMIC_LANE;
    const uint tid = sg * 32u + lane, threads = SEISMIC_SUBGROUPS * 32u;
    const uint q = lane >> 3u, u = lane & 7u;
    const uint groups = SEISMIC_SUBGROUPS * 4u;
    projection_ordered_inverses(in_, m, 0u, 1u, 0u, true, 64u);
    const uint w_row = local_tile * groups + sg * 4u + q;
    const uint hw_first_row = local_tile * groups + 4u * SEISMIC_HARDWARE_SUBGROUP;
    float acc[8];
    [[unroll]] for (uint r = 0u; r < 8u; ++r)
        acc[r] = 0.0;
    for (uint k0 = 0u; k0 < k; k0 += 256u) {
        const uint chunk = min(256u, k - k0);
        for (uint idx = tid; idx < m * chunk; idx += threads) {
            const uint r = chunk == 256u ? (idx >> 8u) : (idx / chunk);
            const uint c = chunk == 256u ? (idx & 255u) : (idx % chunk);
            const float val = projection_ordered_at(in_, r, k0 + c, seismic_shared_f32[r]);
            seismic_shared_f32[64u + r * 256u + c] = val;
            if (publish && publish_rows)
                out_put(pub_out, uint64_t(r) * pub_s0 + uint64_t(k0 + c) * pub_s1, val);
        }
        barrier();
        if (hw_first_row < n) {
            for (uint sub = 0u; sub < chunk; sub += 64u) {
                const uint p_own = (k0 + sub) / 32u + (u >> 2u);
                const uint s_own = u & 3u;
                const uint base_col = k0 + sub + 8u * u;
                float wt[8];
                if (w_row < n && base_col < k) {
                    const packets_packet pkt = projection_packet(kind, w, w_row, p_own);
                    vec4 even, odd;
                    packets_exact_values(kind, pkt, s_own, even, odd);
                    [[unroll]] for (uint j = 0u; j < 4u; ++j) {
                        wt[2u * j] = (base_col + 2u * j < k) ? even[j] : 0.0;
                        wt[2u * j + 1u] = (base_col + 2u * j + 1u < k) ? odd[j] : 0.0;
                    }
                } else {
                    [[unroll]] for (uint j = 0u; j < 8u; ++j)
                        wt[j] = 0.0;
                }
                projection_ordered_transpose8(wt, u);
                if (w_row < n) {
                    if (sub + 64u <= chunk) {
                        [[unroll]] for (uint s = 0u; s < 8u; ++s) {
                            const uint col = sub + 8u * s + u;
                            const float w_val = wt[s];
                            [[unroll]] for (uint r = 0u; r < maxm; ++r) {
                                if (r < m)
                                    acc[r] = seismic_fma_rn(seismic_shared_f32[64u + r * 256u + col], w_val, acc[r]);
                            }
                        }
                    } else {
                        [[unroll]] for (uint s = 0u; s < 8u; ++s) {
                            const uint col = sub + 8u * s + u;
                            if (col < chunk) {
                                const float w_val = wt[s];
                                [[unroll]] for (uint r = 0u; r < maxm; ++r) {
                                    if (r < m)
                                        acc[r] = seismic_fma_rn(seismic_shared_f32[64u + r * 256u + col], w_val, acc[r]);
                                }
                            }
                        }
                    }
                }
            }
        }
        barrier();
    }
    [[unroll]] for (uint r = 0u; r < maxm; ++r) {
        if (r < m) {
            const float sum = projection_ordered_combine8(acc[r]);
            if (w_row < n && u == 0u)
                projection_put(out_, r, w_row, sum);
        }
    }
}

void projection_ordered_gemm(const int kind, projection_prologue in_, projection_epilogue out_, projection_weights w,
    uint m, uint n, uint k, uint tile_m, uint tile_n) {
    const uint sg = SEISMIC_SUBGROUP, lane = SEISMIC_LANE;
    const uint gn = lane >> 3u, u = lane & 7u;
    const uint act_lo = tile_m * 64u + sg * 4u;
    const uint act_hi = act_lo + 32u;
    const bool has_lo = act_lo < m;
    const bool has_hi = act_hi < m;
    const bool has_cols = (tile_n * 32u + 8u * gn) < n;
    const uint my_row0 = act_lo + gn;
    const uint my_row1 = act_hi + gn;
    const uint c_dec = lane, s_dec = sg;
    const uint dec_row = tile_n * 32u + c_dec;
    float acc0[4][8];
    float acc1[4][8];
    [[unroll]] for (uint i = 0u; i < 4u; ++i)
        [[unroll]] for (uint j = 0u; j < 8u; ++j) {
            acc0[i][j] = 0.0;
            acc1[i][j] = 0.0;
        }
    const uint packets = (k + 31u) / 32u;
    if (packets == 0u)
        return;
    const uint64_t my_src_row0 = my_row0 < m ? uint64_t(projection_row(in_.rows, my_row0)) * in_.x0 : 0ul;
    const uint64_t my_src_row1 = my_row1 < m ? uint64_t(projection_row(in_.rows, my_row1)) * in_.x0 : 0ul;
    if (sg < 4u) {
        const uint base_col = 8u * s_dec;
        if (dec_row < n && base_col < k) {
            const packets_packet pkt = projection_packet(kind, w, dec_row, 0u);
            vec4 even, odd;
            packets_exact_values(kind, pkt, s_dec, even, odd);
            [[unroll]] for (uint j = 0u; j < 4u; ++j) {
                seismic_shared_f32[(8u * s_dec + 2u * j) * 36u + c_dec] = (base_col + 2u * j < k) ? even[j] : 0.0;
                seismic_shared_f32[(8u * s_dec + 2u * j + 1u) * 36u + c_dec] =
                    (base_col + 2u * j + 1u < k) ? odd[j] : 0.0;
            }
        } else {
            [[unroll]] for (uint j = 0u; j < 8u; ++j)
                seismic_shared_f32[(8u * s_dec + j) * 36u + c_dec] = 0.0;
        }
    }
    for (uint p = 0u; p < packets; ++p) {
        const uint k0 = 32u * p;
        float cur_x0[4];
        float cur_x1[4];
        [[unroll]] for (uint s = 0u; s < 4u; ++s) {
            cur_x0[s] = 0.0;
            cur_x1[s] = 0.0;
        }
        if (has_lo) {
            [[unroll]] for (uint s = 0u; s < 4u; ++s) {
                const uint col = k0 + 8u * s + u;
                cur_x0[s] = (my_row0 < m && col < k) ? stored_at(in_.x, my_src_row0 + uint64_t(col) * in_.x1) : 0.0;
            }
        }
        if (has_hi) {
            [[unroll]] for (uint s = 0u; s < 4u; ++s) {
                const uint col = k0 + 8u * s + u;
                cur_x1[s] = (my_row1 < m && col < k) ? stored_at(in_.x, my_src_row1 + uint64_t(col) * in_.x1) : 0.0;
            }
        }
        barrier();
        const uint cur_buf = (p & 1u) * 1152u;
        float next_w[8];
        [[unroll]] for (uint j = 0u; j < 8u; ++j)
            next_w[j] = 0.0;
        if (p + 1u < packets && sg < 4u) {
            const uint base_col = k0 + 32u + 8u * s_dec;
            if (dec_row < n && base_col < k) {
                const packets_packet pkt = projection_packet(kind, w, dec_row, p + 1u);
                vec4 even, odd;
                packets_exact_values(kind, pkt, s_dec, even, odd);
                [[unroll]] for (uint j = 0u; j < 4u; ++j) {
                    next_w[2u * j] = (base_col + 2u * j < k) ? even[j] : 0.0;
                    next_w[2u * j + 1u] = (base_col + 2u * j + 1u < k) ? odd[j] : 0.0;
                }
            }
        }
        // A logical subgroup without rows (on 64 lanes, beside one with
        // them) shuffles its zeros and accumulates nothing.
        bool lo;
        if (seismic_steer(lo, has_lo)) {
            [[unroll]] for (uint s = 0u; s < 4u; ++s) {
                const float x0 = seismic_shuffle(cur_x0[s], u);
                const float x1 = seismic_shuffle(cur_x0[s], 8u + u);
                const float x2 = seismic_shuffle(cur_x0[s], 16u + u);
                const float x3 = seismic_shuffle(cur_x0[s], 24u + u);
                float y0 = 0.0, y1 = 0.0, y2 = 0.0, y3 = 0.0;
                bool hi;
                if (seismic_steer(hi, has_hi)) {
                    y0 = seismic_shuffle(cur_x1[s], u);
                    y1 = seismic_shuffle(cur_x1[s], 8u + u);
                    y2 = seismic_shuffle(cur_x1[s], 16u + u);
                    y3 = seismic_shuffle(cur_x1[s], 24u + u);
                }
                if (lo && has_cols && (k0 + 32u <= k || k0 + 8u * s + u < k)) {
                    const uint row_off = cur_buf + (8u * s + u) * 36u + 8u * gn;
                    float wc[8];
                    [[unroll]] for (uint j = 0u; j < 8u; ++j)
                        wc[j] = seismic_shared_f32[row_off + j];
                    [[unroll]] for (uint j = 0u; j < 8u; ++j) {
                        acc0[0][j] = seismic_fma_rn(x0, wc[j], acc0[0][j]);
                        acc0[1][j] = seismic_fma_rn(x1, wc[j], acc0[1][j]);
                        acc0[2][j] = seismic_fma_rn(x2, wc[j], acc0[2][j]);
                        acc0[3][j] = seismic_fma_rn(x3, wc[j], acc0[3][j]);
                    }
                    if (has_hi) {
                        [[unroll]] for (uint j = 0u; j < 8u; ++j) {
                            acc1[0][j] = seismic_fma_rn(y0, wc[j], acc1[0][j]);
                            acc1[1][j] = seismic_fma_rn(y1, wc[j], acc1[1][j]);
                            acc1[2][j] = seismic_fma_rn(y2, wc[j], acc1[2][j]);
                            acc1[3][j] = seismic_fma_rn(y3, wc[j], acc1[3][j]);
                        }
                    }
                }
            }
        }
        if (p + 1u < packets && sg < 4u) {
            const uint dst_buf = ((p + 1u) & 1u) * 1152u;
            [[unroll]] for (uint j = 0u; j < 8u; ++j)
                seismic_shared_f32[dst_buf + (8u * s_dec + j) * 36u + c_dec] = next_w[j];
        }
    }
    // Every lane of a subgroup with rows takes part in the combine; a lane
    // without columns, or (on 64 lanes) a logical subgroup without rows,
    // combines zeros and publishes nothing.
    bool lo;
    if (seismic_steer(lo, has_lo)) {
        [[unroll]] for (uint i = 0u; i < 4u; ++i) {
            const uint out_r = act_lo + i;
            [[unroll]] for (uint j = 0u; j < 8u; ++j) {
                const float sum = projection_ordered_combine8(acc0[i][j]);
                const uint out_c = tile_n * 32u + 8u * gn + j;
                if (u == 0u && out_r < m && out_c < n)
                    projection_put(out_, out_r, out_c, sum);
            }
        }
    }
    bool hi;
    if (seismic_steer(hi, has_hi)) {
        [[unroll]] for (uint i = 0u; i < 4u; ++i) {
            const uint out_r = act_hi + i;
            [[unroll]] for (uint j = 0u; j < 8u; ++j) {
                const float sum = projection_ordered_combine8(acc1[i][j]);
                const uint out_c = tile_n * 32u + 8u * gn + j;
                if (u == 0u && out_r < m && out_c < n)
                    projection_put(out_, out_r, out_c, sum);
            }
        }
    }
}
