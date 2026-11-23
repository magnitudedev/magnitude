// The packed projection family's GEMM (M > 8), over `projection.glsl`'s
// prologues, weights and epilogues; its includer includes that file first.
// This file is independent of any entry ABI.
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
// The declarations default to 64 x 128 tiles of 32 x 32 sub-tiles (256
// invocations). On an RTX 3090 (NVIDIA 595.91.07, attention_project at
// Qwen3.6 35B prefill shapes) its pipelines' statistics separate the slow
// shapes: every 64 x 64 sub-tile reaches the 255-register ceiling and spills
// (6.2-7.1x the best), and 128 x 128 tiles of 32 x 32 sub-tiles take 128
// registers in each of 512 invocations, the whole register file (6.6x).
// 64 x 128 / 32 x 32 (128 registers, 256 invocations) is the best there, and
// on GB10 its recorded time is 1.12x the best of each GEMM entry (geometric
// mean), where 128 x 128 / 32 x 32 was 1.15x.

// The sub-tile (WM x WN) the template cannot run: the checker excludes it
// from every implementation that includes this template and tunes its
// sub-tile (SEISMIC_TUNE_SUB_M, SEISMIC_TUNE_SUB_N), whatever it declares.
#define PROJECTION_GEMM_SPILLING_SUB_M 64
#define PROJECTION_GEMM_SPILLING_SUB_N 64

// The tile shape is the pipeline's constant, which the includer defines
// before including this file: TM x TN tiles of WM x WN sub-tiles,
// PROJECTION_GEMM_TM, _TN, _WM and _WN (`operator.glsl`: the declaration's
// tuned TILE_M x TILE_N and SUB_M x SUB_N). Every register array of the GEMM
// is sized by it exactly and every loop over one runs to a literal bound, and
// the arrays are the invocation's private variables, read and written in
// place by every GEMM function, never an aggregate passed between functions.
// On the NVIDIA proprietary compiler (GB10) arrays sized for the largest shape
// and passed so (the staged weight packets, the accumulators) sat in 2 KiB of
// local memory an invocation.
#if !defined(PROJECTION_GEMM_TM) || !defined(PROJECTION_GEMM_TN) || !defined(PROJECTION_GEMM_WM) \
    || !defined(PROJECTION_GEMM_WN)
#error "gemm.glsl needs the GEMM shape, PROJECTION_GEMM_TM, _TN, _WM and _WN"
#endif

#define PROJECTION_GEMM_K 32u
#define PROJECTION_GEMM_LDA 40u
// The shape as unsigned constants.
#define PROJECTION_TM uint(PROJECTION_GEMM_TM)
#define PROJECTION_TN uint(PROJECTION_GEMM_TN)
#define PROJECTION_WM uint(PROJECTION_GEMM_WM)
#define PROJECTION_WN uint(PROJECTION_GEMM_WN)
// Invocations of a GEMM workgroup. The staging spreads its items over them by
// `SEISMIC_THREAD`, never by the local invocation index: the matrix path's
// cooperative-matrix operations take what it stages (the prelude's rule).
#define PROJECTION_GEMM_THREADS (PROJECTION_TM * PROJECTION_TN * 32u / (PROJECTION_WM * PROJECTION_WN))
// Staged items per invocation and step: activation items (8 columns of one
// tile row) and weight items (half-packets of one tensor's tile rows, TN of
// them unpaired, TN / 2 of each tensor paired).
#define PROJECTION_GEMM_A_ITEMS ((4u * PROJECTION_TM + PROJECTION_GEMM_THREADS - 1u) / PROJECTION_GEMM_THREADS)
#define PROJECTION_GEMM_B_ITEMS ((2u * PROJECTION_TN + PROJECTION_GEMM_THREADS - 1u) / PROJECTION_GEMM_THREADS)
// Matrix fragments (16 x 16) of the sub-tile, or plain 32 x 32 blocks, and the
// lane's output rows that carry a scale.
#if SEISMIC_HAS_MATRIX
#define PROJECTION_GEMM_ACCUMULATORS ((PROJECTION_WM / 16u) * (PROJECTION_WN / 16u))
#define PROJECTION_GEMM_SCALES (PROJECTION_WM / 16u)
#else
#define PROJECTION_GEMM_ACCUMULATORS (32u * (PROJECTION_WM / 32u) * (PROJECTION_WN / 32u))
#define PROJECTION_GEMM_SCALES (PROJECTION_WM / 8u)
#endif
// u32 offsets, within a row's A slot, of its maximum bits and scale exponent.
#define PROJECTION_GEMM_ROW_MAXIMUM 16u
#define PROJECTION_GEMM_ROW_EXPONENT 17u

// The accumulators of the subgroup's WM x WN sub-tile: (WM / 16) x (WN / 16)
// fragments (fragment (i, j) at i * (WN / 16) + j), or per 32 x 32 block
// (bi, bj) (block bi * (WN / 32) + bj) a 4 x 8 register tile per lane (rows 4
// (lane / 4) .., columns 8 (lane % 4) .. of the block), and the scale 2^e of
// each of the lane's output rows (see `projection_gemm_pair`).
#if SEISMIC_HAS_MATRIX
MATRIX_FRAGMENT_C projection_gemm_c[PROJECTION_GEMM_ACCUMULATORS];
#else
float projection_gemm_c[PROJECTION_GEMM_ACCUMULATORS];
#endif
float projection_gemm_scale[PROJECTION_GEMM_SCALES];
// One step's staged registers: activation item j as f16 words, and weight
// item j, the half-packet (16 codes) item % 2 of the tensor's tile-local row
// item / 2, of the weight and of a paired tile's second weight.
uvec4 projection_gemm_a[PROJECTION_GEMM_A_ITEMS];
packets_packet projection_gemm_w[PROJECTION_GEMM_B_ITEMS];
bool projection_gemm_w_valid[PROJECTION_GEMM_B_ITEMS];
packets_packet projection_gemm_u[PROJECTION_GEMM_B_ITEMS];
bool projection_gemm_u_valid[PROJECTION_GEMM_B_ITEMS];

// The u32 index of word `word` of tile row `row`'s A slot.
uint projection_gemm_row_word(uint row, uint word) { return row * (PROJECTION_GEMM_LDA / 2u) + word; }

// The row scan of a bf16 activation: every tile row's exponent e = max(0,
// exponent(max |x|) - 14), stored in its A slot. Every invocation takes part;
// the caller has freed the region.
void projection_gemm_scan_rows(projection_prologue in_, uint m0, uint m_rows) {
    for (uint row = SEISMIC_THREAD; row < PROJECTION_TM; row += PROJECTION_GEMM_THREADS)
        seismic_shared_u32[projection_gemm_row_word(row, PROJECTION_GEMM_ROW_MAXIMUM)] = 0u;
    barrier();
    const uint pieces = (in_.columns + 7u) / 8u;
    for (uint item = SEISMIC_THREAD; item < PROJECTION_TM * pieces; item += PROJECTION_GEMM_THREADS) {
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
    for (uint row = SEISMIC_THREAD; row < PROJECTION_TM; row += PROJECTION_GEMM_THREADS) {
        const float largest = uintBitsToFloat(seismic_shared_u32[projection_gemm_row_word(row, PROJECTION_GEMM_ROW_MAXIMUM)]);
        seismic_shared_u32[projection_gemm_row_word(row, PROJECTION_GEMM_ROW_EXPONENT)] = uint(staging_f16_exponent(largest));
    }
    barrier();
}

// The staging exponent of tile row `row` (0 without a scan).
int projection_gemm_row_exponent(projection_prologue in_, uint row) {
    return in_.act == ELEMENT_BF16 ? int(seismic_shared_u32[projection_gemm_row_word(row, PROJECTION_GEMM_ROW_EXPONENT)]) : 0;
}

// The weight items of a tensor of `count` tile rows.
uint projection_gemm_b_items(const uint count) {
    return (2u * count + PROJECTION_GEMM_THREADS - 1u) / PROJECTION_GEMM_THREADS;
}

// Stage one step's activation registers: item j is columns 8 (item % 4) .. of
// tile row item / 4, as f16 words (scaled by the row's 2^-e).
void projection_gemm_load_a(projection_prologue in_, uint m0, uint m_rows, uint k0) {
    [[unroll]] for (uint j = 0u; j < PROJECTION_GEMM_A_ITEMS; ++j) {
        const uint item = SEISMIC_THREAD + j * PROJECTION_GEMM_THREADS;
        const uint row = item / 4u, column = k0 + 8u * (item % 4u);
        projection_gemm_a[j] = uvec4(0u);
        if (row < PROJECTION_TM && m0 + row < m_rows && column < in_.columns) {
            vec4 even, odd;
            projection_load8(in_, m0 + row, column, 0.0, even, odd);
            if (in_.act == ELEMENT_BF16) {
                const int exponent = projection_gemm_row_exponent(in_, row);
                even = ldexp(even, ivec4(-exponent));
                odd = ldexp(odd, ivec4(-exponent));
            }
            projection_gemm_a[j] = element_pack8(ELEMENT_F16, even, odd);
        }
    }
}

void projection_gemm_store_a() {
    [[unroll]] for (uint j = 0u; j < PROJECTION_GEMM_A_ITEMS; ++j) {
        const uint item = SEISMIC_THREAD + j * PROJECTION_GEMM_THREADS;
        const uint row = item / 4u;
        if (row < PROJECTION_TM)
            seismic_shared_uvec4[row * (PROJECTION_GEMM_LDA / 8u) + item % 4u] = projection_gemm_a[j];
    }
}

// Load one step's packets of a tensor of `count` tile rows from `first`, into
// the weight's registers or (`partner`) the second weight's.
void projection_gemm_load_b_of(const int kind, const bool partner, projection_weights w, uint first,
    const uint count, uint rows, uint k0, uint k) {
    [[unroll]] for (uint j = 0u; j < projection_gemm_b_items(count); ++j) {
        const uint item = SEISMIC_THREAD + j * PROJECTION_GEMM_THREADS;
        const uint local = item / 2u;
        const uint p = k0 / 32u;
        const bool valid = local < count && 32u * p < k;
        if (partner)
            projection_gemm_u_valid[j] = valid;
        else
            projection_gemm_w_valid[j] = valid;
        if (valid) {
            const packets_packet packet = projection_packet(kind, w, min(first + local, rows - 1u), p);
            if (partner)
                projection_gemm_u[j] = packet;
            else
                projection_gemm_w[j] = packet;
        }
    }
}

// Decode the staged packets (their values, `packets_values`, rounded to
// f16). Local row r of this tensor becomes B row tile_row0 + r * spacing (a
// paired tile interleaves gate and up rows); B starts at uvec4 `b_word`.
void projection_gemm_store_b_of(const int kind, const bool partner, const uint count, uint tile_row0,
    uint spacing, uint b_word) {
    [[unroll]] for (uint j = 0u; j < projection_gemm_b_items(count); ++j) {
        const uint item = SEISMIC_THREAD + j * PROJECTION_GEMM_THREADS;
        const uint local = item / 2u, within = item % 2u;
        if (local >= count)
            continue;
        const uint at = b_word + (tile_row0 + local * spacing) * (PROJECTION_GEMM_LDA / 8u) + 2u * within;
        if (partner ? projection_gemm_u_valid[j] : projection_gemm_w_valid[j]) {
            [[unroll]] for (uint s = 0u; s < 2u; ++s) {
                vec4 even, odd;
                packets_values(kind, partner ? projection_gemm_u[j] : projection_gemm_w[j], 2u * within + s, even,
                    odd);
                seismic_shared_uvec4[at + s] = element_pack8(ELEMENT_F16, even, odd);
            }
        } else {
            seismic_shared_uvec4[at] = uvec4(0u);
            seismic_shared_uvec4[at + 1u] = uvec4(0u);
        }
    }
}

// The weight staging of a kind that may be chosen per workgroup (a segmented
// launch, see the file header): one preprocessor group per distinct kind the
// slots bind, each a constant-kind staging.
#define PROJECTION_GEMM_LOAD_B(KIND) \
    if (kind == KIND) { \
        projection_gemm_load_b_of(KIND, partner, w, first, count, rows, k0, k); \
        return; \
    }
#define PROJECTION_GEMM_STORE_B(KIND) \
    if (kind == KIND) { \
        projection_gemm_store_b_of(KIND, partner, count, tile_row0, spacing, b_word); \
        return; \
    }

void projection_gemm_load_b(int kind, const bool partner, projection_weights w, uint first, const uint count,
    uint rows, uint k0, uint k) {
#if defined(KERNEL_W0)
    PROJECTION_GEMM_LOAD_B(PACKETS_W0)
#endif
#if defined(KERNEL_W1) && PACKETS_W1 != PACKETS_W0
    PROJECTION_GEMM_LOAD_B(PACKETS_W1)
#endif
#if defined(KERNEL_W2) && PACKETS_W2 != PACKETS_W0 && PACKETS_W2 != PACKETS_W1
    PROJECTION_GEMM_LOAD_B(PACKETS_W2)
#endif
#if defined(KERNEL_W3) && PACKETS_W3 != PACKETS_W0 && PACKETS_W3 != PACKETS_W1 && PACKETS_W3 != PACKETS_W2
    PROJECTION_GEMM_LOAD_B(PACKETS_W3)
#endif
#if defined(KERNEL_W4) && PACKETS_W4 != PACKETS_W0 && PACKETS_W4 != PACKETS_W1 && PACKETS_W4 != PACKETS_W2 && PACKETS_W4 != PACKETS_W3
    PROJECTION_GEMM_LOAD_B(PACKETS_W4)
#endif
#if defined(KERNEL_W5) && PACKETS_W5 != PACKETS_W0 && PACKETS_W5 != PACKETS_W1 && PACKETS_W5 != PACKETS_W2 && PACKETS_W5 != PACKETS_W3 && PACKETS_W5 != PACKETS_W4
    PROJECTION_GEMM_LOAD_B(PACKETS_W5)
#endif
#if defined(KERNEL_W6) && PACKETS_W6 != PACKETS_W0 && PACKETS_W6 != PACKETS_W1 && PACKETS_W6 != PACKETS_W2 && PACKETS_W6 != PACKETS_W3 && PACKETS_W6 != PACKETS_W4 && PACKETS_W6 != PACKETS_W5
    PROJECTION_GEMM_LOAD_B(PACKETS_W6)
#endif
#if defined(KERNEL_W7) && PACKETS_W7 != PACKETS_W0 && PACKETS_W7 != PACKETS_W1 && PACKETS_W7 != PACKETS_W2 && PACKETS_W7 != PACKETS_W3 && PACKETS_W7 != PACKETS_W4 && PACKETS_W7 != PACKETS_W5 && PACKETS_W7 != PACKETS_W6
    PROJECTION_GEMM_LOAD_B(PACKETS_W7)
#endif
}

void projection_gemm_store_b(int kind, const bool partner, const uint count, uint tile_row0, uint spacing,
    uint b_word) {
#if defined(KERNEL_W0)
    PROJECTION_GEMM_STORE_B(PACKETS_W0)
#endif
#if defined(KERNEL_W1) && PACKETS_W1 != PACKETS_W0
    PROJECTION_GEMM_STORE_B(PACKETS_W1)
#endif
#if defined(KERNEL_W2) && PACKETS_W2 != PACKETS_W0 && PACKETS_W2 != PACKETS_W1
    PROJECTION_GEMM_STORE_B(PACKETS_W2)
#endif
#if defined(KERNEL_W3) && PACKETS_W3 != PACKETS_W0 && PACKETS_W3 != PACKETS_W1 && PACKETS_W3 != PACKETS_W2
    PROJECTION_GEMM_STORE_B(PACKETS_W3)
#endif
#if defined(KERNEL_W4) && PACKETS_W4 != PACKETS_W0 && PACKETS_W4 != PACKETS_W1 && PACKETS_W4 != PACKETS_W2 && PACKETS_W4 != PACKETS_W3
    PROJECTION_GEMM_STORE_B(PACKETS_W4)
#endif
#if defined(KERNEL_W5) && PACKETS_W5 != PACKETS_W0 && PACKETS_W5 != PACKETS_W1 && PACKETS_W5 != PACKETS_W2 && PACKETS_W5 != PACKETS_W3 && PACKETS_W5 != PACKETS_W4
    PROJECTION_GEMM_STORE_B(PACKETS_W5)
#endif
#if defined(KERNEL_W6) && PACKETS_W6 != PACKETS_W0 && PACKETS_W6 != PACKETS_W1 && PACKETS_W6 != PACKETS_W2 && PACKETS_W6 != PACKETS_W3 && PACKETS_W6 != PACKETS_W4 && PACKETS_W6 != PACKETS_W5
    PROJECTION_GEMM_STORE_B(PACKETS_W6)
#endif
#if defined(KERNEL_W7) && PACKETS_W7 != PACKETS_W0 && PACKETS_W7 != PACKETS_W1 && PACKETS_W7 != PACKETS_W2 && PACKETS_W7 != PACKETS_W3 && PACKETS_W7 != PACKETS_W4 && PACKETS_W7 != PACKETS_W5 && PACKETS_W7 != PACKETS_W6
    PROJECTION_GEMM_STORE_B(PACKETS_W7)
#endif
}

// The subgroup's sub-tile origin in the tile.
uint projection_gemm_row0() { return (SEISMIC_SUBGROUP / (PROJECTION_TN / PROJECTION_WN)) * PROJECTION_WM; }
uint projection_gemm_column0() { return (SEISMIC_SUBGROUP % (PROJECTION_TN / PROJECTION_WN)) * PROJECTION_WN; }

// The multiply of one staged step: `live` fragment rows (0 .. WM / 16) of 16.
void projection_gemm_multiply(uint live) {
    const uint row0 = projection_gemm_row0(), column0 = projection_gemm_column0();
    const uint b0 = PROJECTION_TM * PROJECTION_GEMM_LDA;   // B tile, in f16 elements
#if SEISMIC_HAS_MATRIX
    const uint fm = PROJECTION_WM / 16u, fn = PROJECTION_WN / 16u;
    [[unroll]] for (uint kk = 0u; kk < PROJECTION_GEMM_K; kk += 16u) {
        MATRIX_FRAGMENT_B b[PROJECTION_WN / 16u];
        [[unroll]] for (uint j = 0u; j < fn; ++j)
            coopMatLoad(b[j], seismic_shared_f16, b0 + (column0 + 16u * j) * PROJECTION_GEMM_LDA + kk,
                PROJECTION_GEMM_LDA, gl_CooperativeMatrixLayoutColumnMajor);
        [[unroll]] for (uint i = 0u; i < fm; ++i) {
            if (i < live) {
                MATRIX_FRAGMENT_A a;
                coopMatLoad(a, seismic_shared_f16, (row0 + 16u * i) * PROJECTION_GEMM_LDA + kk, PROJECTION_GEMM_LDA,
                    gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint j = 0u; j < fn; ++j)
                    projection_gemm_c[fn * i + j] = coopMatMulAdd(a, b[j], projection_gemm_c[fn * i + j]);
            }
        }
    }
#else
    const uint lane = SEISMIC_LANE;
    const uint bn = PROJECTION_WN / 32u;
    [[unroll]] for (uint block = 0u; block < (PROJECTION_WM / 32u) * bn; ++block) {
        const uint bi = block / bn, bj = block % bn;
        const uint r0 = row0 + 32u * bi + 4u * (lane / 4u), c0 = column0 + 32u * bj + 8u * (lane % 4u);
        // Block row bi holds fragment rows 2 bi and 2 bi + 1.
        if (live <= 2u * bi || (live == 2u * bi + 1u && r0 >= row0 + 32u * bi + 16u))
            continue;
        // Pairs of K (one f16x2 word per row) keep the reads vectorized.
        [[unroll]] for (uint kk = 0u; kk < PROJECTION_GEMM_K; kk += 2u) {
            vec2 a[4], b[8];
            [[unroll]] for (uint i = 0u; i < 4u; ++i)
                a[i] = seismic_unpack_f16x2(seismic_shared_u32[((r0 + i) * PROJECTION_GEMM_LDA + kk) / 2u]);
            [[unroll]] for (uint j = 0u; j < 8u; ++j)
                b[j] = seismic_unpack_f16x2(seismic_shared_u32[(b0 + (c0 + j) * PROJECTION_GEMM_LDA + kk) / 2u]);
            [[unroll]] for (uint i = 0u; i < 4u; ++i) {
                [[unroll]] for (uint j = 0u; j < 8u; ++j) {
                    const uint at = 32u * block + 8u * i + j;
                    projection_gemm_c[at] = seismic_fma_rn(a[i].x, b[j].x, projection_gemm_c[at]);
                    projection_gemm_c[at] = seismic_fma_rn(a[i].y, b[j].y, projection_gemm_c[at]);
                }
            }
        }
    }
#endif
}

// The K loop over one TM x TN tile, steps [step_begin, step_end): `uk` and
// `u` are the second weights of a paired tile (rows interleaved with w's).
void projection_gemm_accumulate(const int wk, const int uk, const bool paired, projection_prologue in_,
    projection_weights w, projection_weights u, uint first, uint rows, uint m0, uint m_rows, uint k, uint step_begin,
    uint step_end, uint live_rows) {
    const uint count = paired ? PROJECTION_TN / 2u : PROJECTION_TN;
    const uint spacing = paired ? 2u : 1u;
    const uint b_word = PROJECTION_TM * (PROJECTION_GEMM_LDA / 8u);
    [[unroll]] for (uint i = 0u; i < PROJECTION_GEMM_ACCUMULATORS; ++i) {
#if SEISMIC_HAS_MATRIX
        projection_gemm_c[i] = matrix_zero();
#else
        projection_gemm_c[i] = 0.0;
#endif
    }
    [[unroll]] for (uint i = 0u; i < PROJECTION_GEMM_SCALES; ++i)
        projection_gemm_scale[i] = 1.0;
    const uint row_first = m0 + projection_gemm_row0();
    const uint live_limit = min(m_rows, live_rows);
    const uint live = row_first >= live_limit ? 0u : min(PROJECTION_WM / 16u, (live_limit - row_first + 15u) / 16u);
    if (step_begin >= step_end)
        return;
    if (in_.act == ELEMENT_BF16) {
        // A previous tile's epilogue may still read the region.
        barrier();
        projection_gemm_scan_rows(in_, m0, m_rows);
    }
    uint k0 = step_begin * PROJECTION_GEMM_K;
    projection_gemm_load_a(in_, m0, m_rows, k0);
    projection_gemm_load_b(wk, false, w, first, count, rows, k0, k);
    if (paired)
        projection_gemm_load_b(uk, true, u, first, count, rows, k0, k);
    // A previous tile's epilogue may still read the region.
    barrier();
    projection_gemm_store_a();
    projection_gemm_store_b(wk, false, count, 0u, spacing, b_word);
    if (paired)
        projection_gemm_store_b(uk, true, count, 1u, spacing, b_word);
    barrier();
    for (uint t = step_begin; t < step_end; ++t) {
        const bool more = t + 1u < step_end;
        const uint k1 = (t + 1u) * PROJECTION_GEMM_K;
        if (more) {
            projection_gemm_load_a(in_, m0, m_rows, k1);
            projection_gemm_load_b(wk, false, w, first, count, rows, k1, k);
            if (paired)
                projection_gemm_load_b(uk, true, u, first, count, rows, k1, k);
        }
        projection_gemm_multiply(live);
        barrier();
        if (more) {
            projection_gemm_store_a();
            projection_gemm_store_b(wk, false, count, 0u, spacing, b_word);
            if (paired)
                projection_gemm_store_b(uk, true, count, 1u, spacing, b_word);
        }
        barrier();
    }
    if (in_.act == ELEMENT_BF16) {
        // The scales of the lane's output rows, read before the pairs reuse
        // the region.
        const uint row0 = projection_gemm_row0(), lane = SEISMIC_LANE;
        [[unroll]] for (uint i = 0u; i < PROJECTION_GEMM_SCALES; ++i) {
#if SEISMIC_HAS_MATRIX
            const uint row = row0 + 16u * i + lane / 2u;
#else
            const uint row = row0 + 32u * (i / 4u) + 4u * (lane / 4u) + i % 4u;
#endif
            projection_gemm_scale[i] = ldexp(1.0, projection_gemm_row_exponent(in_, row));
        }
        barrier();
    }
}

// Hand every lane its sub-tile outputs in pairs of adjacent columns (c,
// c + 1), c even, so paired tiles see (gate, up): per lane
// PROJECTION_GEMM_PAIRS pairs; pair q sits at tile row
// `projection_gemm_pair_row` and column `projection_gemm_pair_column`. The
// matrix path publishes each 16 x 16 fragment through the subgroup's 1 KiB
// of the (now free) tile region (pairs 4 f .. 4 f + 3 are fragment f); the
// plain path already holds a 4 x 8 register tile per block (pairs 16 b ..
// 16 b + 15 are block b).
#define PROJECTION_GEMM_PAIRS (PROJECTION_WM * PROJECTION_WN / 64u)

uint projection_gemm_pair_row(uint q) {
    const uint lane = SEISMIC_LANE;
#if SEISMIC_HAS_MATRIX
    // Fragment q / 4 = (i, j); within it, pair 4 lane + q % 4 of 128 (8
    // pairs per row).
    return projection_gemm_row0() + 16u * ((q / 4u) / (PROJECTION_WN / 16u)) + (4u * lane + q % 4u) / 8u;
#else
    return projection_gemm_row0() + 32u * ((q / 16u) / (PROJECTION_WN / 32u)) + 4u * (lane / 4u) + (q % 16u) / 4u;
#endif
}

uint projection_gemm_pair_column(uint q) {
    const uint lane = SEISMIC_LANE;
#if SEISMIC_HAS_MATRIX
    return projection_gemm_column0() + 16u * ((q / 4u) % (PROJECTION_WN / 16u)) + 2u * ((4u * lane + q % 4u) % 8u);
#else
    return projection_gemm_column0() + 32u * ((q / 16u) % (PROJECTION_WN / 32u)) + 8u * (lane % 4u) + 2u * (q % 4u);
#endif
}

// The lane's pair q of the accumulators, scaled back by its row's 2^e. The
// matrix path must call it for q = 0 .. PROJECTION_GEMM_PAIRS - 1 in order
// from every lane of the subgroup, after a barrier that frees the tile region.
vec2 projection_gemm_pair(uint q) {
#if SEISMIC_HAS_MATRIX
    const uint scratch = 256u * SEISMIC_SUBGROUP;   // floats
    const uint pair = 4u * SEISMIC_LANE + q % 4u;
    if (q % 4u == 0u) {
        subgroupBarrier();
        // Each fragment by a constant index: indexed by q / 4 at run time
        // (the callers' pair loops hold barriers, which the NVIDIA
        // proprietary compiler leaves rolled), the accumulators, and every
        // multiply-add on them, sat in local memory (GB10: 224-240 B, 1.7x the
        // time). A case past the last fragment is never taken.
        switch (q / 4u) {
#define PROJECTION_GEMM_STORE_FRAGMENT(f) \
        case f: \
            coopMatStore(projection_gemm_c[min(f, PROJECTION_GEMM_ACCUMULATORS - 1u)], seismic_shared_f32, scratch, 16u, \
                gl_CooperativeMatrixLayoutRowMajor); \
            break;
        PROJECTION_GEMM_STORE_FRAGMENT(0u) PROJECTION_GEMM_STORE_FRAGMENT(1u) PROJECTION_GEMM_STORE_FRAGMENT(2u)
        PROJECTION_GEMM_STORE_FRAGMENT(3u) PROJECTION_GEMM_STORE_FRAGMENT(4u) PROJECTION_GEMM_STORE_FRAGMENT(5u)
        PROJECTION_GEMM_STORE_FRAGMENT(6u) PROJECTION_GEMM_STORE_FRAGMENT(7u) PROJECTION_GEMM_STORE_FRAGMENT(8u)
        PROJECTION_GEMM_STORE_FRAGMENT(9u) PROJECTION_GEMM_STORE_FRAGMENT(10u) PROJECTION_GEMM_STORE_FRAGMENT(11u)
        PROJECTION_GEMM_STORE_FRAGMENT(12u) PROJECTION_GEMM_STORE_FRAGMENT(13u) PROJECTION_GEMM_STORE_FRAGMENT(14u)
        PROJECTION_GEMM_STORE_FRAGMENT(15u)
#undef PROJECTION_GEMM_STORE_FRAGMENT
        }
        subgroupMemoryBarrierShared();
        subgroupBarrier();
    }
    return projection_gemm_scale[(q / 4u) / (PROJECTION_WN / 16u)]
        * vec2(seismic_shared_f32[scratch + 2u * pair], seismic_shared_f32[scratch + 2u * pair + 1u]);
#else
    // Pair q's two accumulators are 2 q and 2 q + 1, written over `q` itself
    // (the packet decoder's coefficient order).
    const uint block = q / 16u, within = q % 16u;
    const uint at = 2u * q;
    return projection_gemm_scale[4u * (block / (PROJECTION_WN / 32u)) + within / 4u]
        * vec2(projection_gemm_c[at], projection_gemm_c[at + 1u]);
#endif
}

// One GEMM tile of a plain (unpaired) projection: output rows tm_index * TM
// .., weight rows tn_index * TN .. of `w`. Rows at or past `live_rows` are
// known zero (padding): their products are skipped, and they store the
// epilogue of 0.
void projection_gemm(const int wk, projection_prologue in_, projection_epilogue out_, projection_weights w,
    uint m_rows, uint rows, uint k, uint tm_index, uint tn_index, uint live_rows) {
    const uint first = tn_index * PROJECTION_TN;
    projection_gemm_accumulate(wk, wk, false, in_, w, w, first, rows, tm_index * PROJECTION_TM, m_rows, k, 0u,
        (k + PROJECTION_GEMM_K - 1u) / PROJECTION_GEMM_K, live_rows);
    [[unroll]] for (uint q = 0u; q < PROJECTION_GEMM_PAIRS; ++q) {
        const vec2 c = projection_gemm_pair(q);
        const uint m = tm_index * PROJECTION_TM + projection_gemm_pair_row(q);
        const uint n = first + projection_gemm_pair_column(q);
        if (m < m_rows && n < rows)
            projection_put(out_, m, n, c.x);
        if (m < m_rows && n + 1u < rows)
            projection_put(out_, m, n + 1u, c.y);
    }
}

// Split-K. Part `part` of `split` runs the K steps [part * steps / split,
// (part + 1) * steps / split) of one plain GEMM tile and stores its raw F32
// sums to partials[(part * m_rows + m) * rows + n].
void projection_gemm_part(const int wk, projection_prologue in_, uint64_t partials, projection_weights w,
    uint m_rows, uint rows, uint k, uint split, uint part, uint tm_index, uint tn_index) {
    const uint first = tn_index * PROJECTION_TN;
    const uint steps = (k + PROJECTION_GEMM_K - 1u) / PROJECTION_GEMM_K;
    projection_gemm_accumulate(wk, wk, false, in_, w, w, first, rows, tm_index * PROJECTION_TM, m_rows, k,
        part * steps / split, (part + 1u) * steps / split, m_rows);
    const uint64_t own = partials + uint64_t(part) * m_rows * rows * 4ul;
    [[unroll]] for (uint q = 0u; q < PROJECTION_GEMM_PAIRS; ++q) {
        const vec2 c = projection_gemm_pair(q);
        const uint m = tm_index * PROJECTION_TM + projection_gemm_pair_row(q);
        const uint n = first + projection_gemm_pair_column(q);
        if (m < m_rows && n < rows)
            element_f32_put(own + (uint64_t(m) * rows + n) * 4ul, c.x);
        if (m < m_rows && n + 1u < rows)
            element_f32_put(own + (uint64_t(m) * rows + n + 1u) * 4ul, c.y);
    }
}

// The split-K finalize: output (m, n) sums its parts in order and applies the
// epilogue.
void projection_gemm_finalize(projection_epilogue out_, uint64_t partials, uint m_rows, uint rows, uint split, uint m, uint n) {
    if (m >= m_rows || n >= rows)
        return;
    const uint64_t index = uint64_t(m) * rows + n;
    float total = 0.0;
    for (uint part = 0u; part < split; ++part)
        total += element_f32_at(partials + (uint64_t(part) * m_rows * rows + index) * 4ul);
    projection_put(out_, m, n, total);
}

// One GEMM tile of a paired projection: TN / 2 features of gate and up, the
// tile's weight rows interleaved (gate, up), so a lane's column pair is one
// feature. `live_rows` as for `projection_gemm`.
void projection_gemm_paired(const int gk, const int uk, projection_prologue in_, projection_epilogue out_,
    projection_weights gate, projection_weights up, uint m_rows, uint rows, uint k, uint tm_index, uint tn_index,
    uint live_rows) {
    const uint first = tn_index * (PROJECTION_TN / 2u);
    projection_gemm_accumulate(gk, uk, true, in_, gate, up, first, rows, tm_index * PROJECTION_TM, m_rows, k, 0u,
        (k + PROJECTION_GEMM_K - 1u) / PROJECTION_GEMM_K, live_rows);
    [[unroll]] for (uint q = 0u; q < PROJECTION_GEMM_PAIRS; ++q) {
        const vec2 c = projection_gemm_pair(q);
        const uint m = tm_index * PROJECTION_TM + projection_gemm_pair_row(q);
        const uint n = first + projection_gemm_pair_column(q) / 2u;
        if (m < m_rows && n < rows)
            projection_put_pair(out_, m, n, c.x, c.y);
    }
}
