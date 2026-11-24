// The Metal chunked delta-rule entries (`gated_delta_chunk`,
// `kimi_delta_chunk`; contracts in recurrent.seismic) over the rule's gates
// (`recurrent::HeadRule`, `kimi::ChannelRule`): the chunked WY form over
// pieces of 16 rows (`METAL_CHUNK_PIECE`), in three launches:
//
// `rch_prepare`, one threadgroup (256 threads) per (piece, key head):
// convolves the piece's q, k and member value heads' v rows once (sliding
// window per warp), L2-normalizes q (scaled by W^-1/2) and k, forms the
// decayed K K^T and Q K^T products in F32 (using `simdgroup_float8x8` tiles
// for DECAYS == 1), solves Tb = (I + A)^-1 diag(beta) by parallel forward
// substitution across all member heads, and writes split-F16 operands (Q,
// K_hi, K_lo; or per-head decayed Q, K_hi, K_lo, U_hi, U_lo for Kimi), F16 V,
// Tb, D, and F32 decay gates into the `pieces` scratch.
//
// `rch_scan`, one threadgroup per (ROWS state rows, value head, slot): each
// simdgroup keeps 16 state rows in registers:
// - With `SEISMIC_HAS_TENSOR_OPS` (M5+): a single full-width cooperative
//   tensor via `mpp::tensor_ops::matmul2d` over split-F16 operands with F32
//   accumulation.
// - Without `SEISMIC_HAS_TENSOR_OPS` (M1-M4): native F32 `simdgroup_float8x8`
//   (widening `Q`, `K_hi + K_lo`, `V`, `Tb`, `D` during threadgroup staging
//   for `W <= 128`, avoiding 3x split-F16 MMA overhead since M1-M4 runs F32
//   simdgroup matrix multiply at full rate).
//
// Slots of at most RECURRENT_SEQUENTIAL_ROWS rows and rows after `stop`
// advance row-sequentially with the step's bits (`recurrent::advance_rows`).
//
// The gate launch: one simdgroup per (row, value head) gates the row's raw
// outputs into the result (`recurrent::gate_row`).

#define RCH_PIECE SEISMIC_CONST_METAL_CHUNK_PIECE
#define RCH_ROWS SEISMIC_TUNE_ROWS
#define RCH_THREADS (2 * RCH_ROWS)
#define RCH_SIMDGROUPS (RCH_ROWS / 16)
#define RCH_WIDTH SEISMIC_DIM_W
#define RCH_TILES (RCH_WIDTH / 8)
#define RCH_W_BLOCKS (RCH_WIDTH / 32)
#define RCH_CPL (RCH_WIDTH / 32)
#define RCH_KEY_HEADS SEISMIC_DIM_NK
#define RCH_VALUE_HEADS SEISMIC_DIM_NV
#define RCH_GROUP (RCH_VALUE_HEADS / RCH_KEY_HEADS)
#define RCH_PREPARE_THREADS 256u
#define RCH_PREPARE_WARPS 8u

#define RCH_PADDED(n) ((n) % 32 == 8 || (n) % 32 == 24 ? (n) : (n) + 8)
#define RCH_PREP_STRIDE RCH_PADDED(RCH_WIDTH)
#define RCH_SEQ_SPAN (RCH_WIDTH <= 128 ? 8u : 4u)
#define RCH_SEQ_INPUT_STRIDE RCH_PADDED(RCH_WIDTH)
#define RCH_SEQ_VALUE_STRIDE RCH_PADDED(RCH_ROWS)

#define RCH_Q_STRIDE (RCH_WIDTH + 8)
#define RCH_V_STRIDE (RCH_ROWS + 8)

#include "convolution.h"

namespace rch {

static_assert(RCH_PIECE == 16, "Metal chunked delta rule uses 16-row pieces");
static_assert(RCH_VALUE_HEADS % RCH_KEY_HEADS == 0 && RCH_GROUP <= 8, "chunk head grouping");
static_assert(RCH_WIDTH % 32 == 0 && RCH_ROWS % 16 == 0 && RCH_WIDTH % RCH_ROWS == 0, "chunk geometry");

constant constexpr uint RECORD_V = 0u;
constant constexpr uint RECORD_TB = uint(RCH_PIECE) * uint(RCH_WIDTH) * 2u;
constant constexpr uint RECORD_D = RECORD_TB + uint(RCH_PIECE) * uint(RCH_PIECE) * 2u;

inline uint scratch_pieces(recurrent::Slot slot) {
    return slot.hi - slot.lo <= RECURRENT_SEQUENTIAL_ROWS ? 0u : uint(recurrent::pieces_before<RCH_PIECE>(slot));
}

inline uint first_piece(device const int *segments, device const int *stop, device const int *previous_bank,
    device const int *previous_tape, device const int *following_bank, ulong slot_index,
    constant ulong *seismic_words) {
    uint total = 0u;
    for (ulong slot = 0; slot < slot_index; ++slot) {
        total += scratch_pieces(versions::slot_of({segments, stop, previous_bank, previous_tape, following_bank},
            slot, seismic_words));
    }
    return total;
}

inline uint member_head(uint key_head, uint member, constant ulong *seismic_words) {
    return SEISMIC_PARAM_GROUPED != 0 ? key_head * RCH_GROUP + member : key_head + member * RCH_KEY_HEADS;
}

constexpr uint prepare_row_run(uint parts) {
    uint best = RCH_PIECE;
    for (uint run = RCH_PIECE; run >= 1u; run >>= 1) {
        const uint runs = parts * (RCH_PIECE / run);
        const uint rows = (runs + RCH_PREPARE_WARPS - 1u) / RCH_PREPARE_WARPS * run;
        const uint best_runs = parts * (RCH_PIECE / best);
        if (rows < (best_runs + RCH_PREPARE_WARPS - 1u) / RCH_PREPARE_WARPS * best) {
            best = run;
        }
    }
    return best;
}

inline void load_conv_input(device const recurrent::Storage *projection,
    device const slab::Table<activation::Act> *window, recurrent::Slot slot, long position, ulong channel,
    thread float (&out)[RCH_CPL], constant ulong *seismic_words) {
    device const recurrent::Storage *ptr = position < 0
        ? versions::bank_view(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words)
            + ulong(slot.taped + long(RECURRENT_TAPS) - 1 + position) * SEISMIC_WINDOW_STRIDE_1 + channel
        : projection + ulong(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0 + channel;
    if constexpr (RCH_CPL % 4 == 0) {
        RECURRENT_UNROLL for (uint k = 0; k < RCH_CPL; k += 4) {
            const float4 v = activation::Act::load4(ptr + k);
            out[k + 0] = v.x;
            out[k + 1] = v.y;
            out[k + 2] = v.z;
            out[k + 3] = v.w;
        }
    } else {
        RECURRENT_UNROLL for (uint k = 0; k < RCH_CPL; ++k) {
            out[k] = activation::Act::load(ptr[k]);
        }
    }
}

inline void store_half_cpl(device half *dst, thread const float (&values)[RCH_CPL]) {
    if constexpr (RCH_CPL % 4 == 0) {
        RECURRENT_UNROLL for (uint k = 0; k < RCH_CPL; k += 4) {
            *reinterpret_cast<device half4 *>(dst + k) =
                half4(half(values[k]), half(values[k + 1]), half(values[k + 2]), half(values[k + 3]));
        }
    } else if constexpr (RCH_CPL == 2) {
        *reinterpret_cast<device half2 *>(dst) = half2(half(values[0]), half(values[1]));
    } else {
        dst[0] = half(values[0]);
    }
}

inline void store_half_split_cpl(device half *dst_hi, device half *dst_lo, thread const float (&values)[RCH_CPL]) {
    if constexpr (RCH_CPL % 4 == 0) {
        RECURRENT_UNROLL for (uint k = 0; k < RCH_CPL; k += 4) {
            const half4 h = half4(half(values[k]), half(values[k + 1]), half(values[k + 2]), half(values[k + 3]));
            const half4 l = half4(half(values[k] - float(h.x)), half(values[k + 1] - float(h.y)),
                half(values[k + 2] - float(h.z)), half(values[k + 3] - float(h.w)));
            *reinterpret_cast<device half4 *>(dst_hi + k) = h;
            *reinterpret_cast<device half4 *>(dst_lo + k) = l;
        }
    } else if constexpr (RCH_CPL == 2) {
        const half2 h = half2(half(values[0]), half(values[1]));
        const half2 l = half2(half(values[0] - float(h.x)), half(values[1] - float(h.y)));
        *reinterpret_cast<device half2 *>(dst_hi) = h;
        *reinterpret_cast<device half2 *>(dst_lo) = l;
    } else {
        const half h = half(values[0]);
        dst_hi[0] = h;
        dst_lo[0] = half(values[0] - float(h));
    }
}

inline void store_float_cpl(device float *dst, thread const float (&values)[RCH_CPL]) {
    if constexpr (RCH_CPL % 4 == 0) {
        RECURRENT_UNROLL for (uint k = 0; k < RCH_CPL; k += 4) {
            *reinterpret_cast<device float4 *>(dst + k) =
                float4(values[k], values[k + 1], values[k + 2], values[k + 3]);
        }
    } else if constexpr (RCH_CPL == 2) {
        *reinterpret_cast<device float2 *>(dst) = float2(values[0], values[1]);
    } else {
        dst[0] = values[0];
    }
}

// Piece scratch layout, templated by DECAYS (1 = Gated DeltaNet, W = Kimi).
template <uint DECAYS>
struct Layout;

template <>
struct Layout<1> {
    static constant constexpr ulong MATRIX_HALFS = ulong(RCH_PIECE) * ulong(RCH_WIDTH);
    static constant constexpr ulong OPERAND_BYTES = ulong(RCH_KEY_HEADS) * 3ul * MATRIX_HALFS * 2ul;
    static constant constexpr ulong RECORD_GATES = RECORD_D + ulong(RCH_PIECE) * ulong(RCH_PIECE) * 2ul;
    static constant constexpr uint GATES_FLOATS = 2u * RCH_PIECE + 4u;
    static constant constexpr ulong RECORD_BYTES = RECORD_GATES + ulong(GATES_FLOATS) * 4ul;
    static constant constexpr ulong PIECE_BYTES = OPERAND_BYTES + ulong(RCH_VALUE_HEADS) * RECORD_BYTES;

    static inline device uchar *record(device uchar *pieces, uint piece, uint head) {
        return pieces + ulong(piece) * PIECE_BYTES + OPERAND_BYTES + ulong(head) * RECORD_BYTES;
    }
    static inline device half *operands(device uchar *pieces, uint piece, uint key_head, uint) {
        return reinterpret_cast<device half *>(pieces + ulong(piece) * PIECE_BYTES
            + ulong(key_head) * 3ul * MATRIX_HALFS * 2ul);
    }
    static inline device half *q_ptr(device half *ops) { return ops; }
    static inline device half *k_hi_ptr(device half *ops) { return ops + MATRIX_HALFS; }
    static inline device half *k_lo_ptr(device half *ops) { return ops + 2ul * MATRIX_HALFS; }
    static inline device half *u_hi_ptr(device half *ops) { return ops + MATRIX_HALFS; }
    static inline device half *u_lo_ptr(device half *ops) { return ops + 2ul * MATRIX_HALFS; }

    static inline float gamma(threadgroup const float *gates, uint t) { return gates[t]; }
    static inline float tail(threadgroup const float *gates, uint t) { return gates[RCH_PIECE + t]; }
    static inline float factor(threadgroup const float *gates, uint) { return gates[2u * RCH_PIECE]; }
};

template <uint DECAYS>
struct Layout {
    static constant constexpr ulong MATRIX_HALFS = ulong(RCH_PIECE) * ulong(RCH_WIDTH);
    static constant constexpr ulong RECORD_ROWS = RECORD_D + ulong(RCH_PIECE) * ulong(RCH_PIECE) * 2ul;
    static constant constexpr ulong RECORD_GATES = RECORD_ROWS + 5ul * MATRIX_HALFS * 2ul;
    static constant constexpr uint GATES_FLOATS = RCH_WIDTH;
    static constant constexpr ulong RECORD_BYTES = RECORD_GATES + ulong(GATES_FLOATS) * 4ul;
    static constant constexpr ulong PIECE_BYTES = ulong(RCH_VALUE_HEADS) * RECORD_BYTES;
    static_assert(RECORD_BYTES == 196ul * ulong(RCH_WIDTH) + 1024ul, "Kimi piece record size");

    static inline device uchar *record(device uchar *pieces, uint piece, uint head) {
        return pieces + ulong(piece) * PIECE_BYTES + ulong(head) * RECORD_BYTES;
    }
    static inline device half *operands(device uchar *pieces, uint piece, uint, uint head) {
        return reinterpret_cast<device half *>(record(pieces, piece, head) + RECORD_ROWS);
    }
    static inline device half *q_ptr(device half *ops) { return ops; }
    static inline device half *k_hi_ptr(device half *ops) { return ops + MATRIX_HALFS; }
    static inline device half *k_lo_ptr(device half *ops) { return ops + 2ul * MATRIX_HALFS; }
    static inline device half *u_hi_ptr(device half *ops) { return ops + 3ul * MATRIX_HALFS; }
    static inline device half *u_lo_ptr(device half *ops) { return ops + 4ul * MATRIX_HALFS; }

    static inline float gamma(threadgroup const float *, uint) { return 1.0f; }
    static inline float tail(threadgroup const float *, uint) { return 1.0f; }
    static inline float factor(threadgroup const float *gates, uint col) { return gates[col]; }
};

inline void solve_tb(threadgroup const float (&system)[RCH_PIECE][RCH_PIECE + 1], float b,
    device half *tb, uint lane16) {
    float x[RCH_PIECE];
    RECURRENT_UNROLL for (uint i = 0; i < RCH_PIECE; ++i) {
        float val = (i == lane16) ? 1.0f : 0.0f;
        RECURRENT_UNROLL for (uint k = 0; k < i; ++k) {
            val = metal::fma(-system[i][k], x[k], val);
        }
        x[i] = val;
    }
    RECURRENT_UNROLL for (uint i = 0; i < RCH_PIECE; ++i) {
        tb[i * RCH_PIECE + lane16] = half(x[i] * b);
    }
}

inline void convolve_run(device const recurrent::Storage *projection, device const float *convolution,
    device const slab::Table<activation::Act> *window, recurrent::Slot slot, long piece_first, long piece_len,
    uint first_row, uint row_run, uint channel_base, uint column, bool normalize, bool is_query,
    threadgroup float *tg_dst, uint tg_stride, uint tg_row_offset, device half *dev_hi, device half *dev_lo,
    constant ulong *seismic_words) {
    const ulong channel = ulong(channel_base) + ulong(column);
    float weights[RECURRENT_TAPS][RCH_CPL];
    RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; ++c) {
        RECURRENT_UNROLL for (uint tap = 0; tap < RECURRENT_TAPS; ++tap) {
            weights[tap][c] = convolution[(channel + c) * SEISMIC_CONVOLUTION_STRIDE_0
                + tap * SEISMIC_CONVOLUTION_STRIDE_1];
        }
    }
    const long local = piece_first - slot.lo + long(first_row);
    float inputs[RECURRENT_TAPS][RCH_CPL];
    RECURRENT_UNROLL for (uint tap = 0; tap < RECURRENT_TAPS - 1; ++tap) {
        if (long(first_row) < piece_len) {
            load_conv_input(projection, window, slot, local + long(tap) - long(RECURRENT_TAPS - 1), channel,
                inputs[tap], seismic_words);
        } else {
            RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; ++c) inputs[tap][c] = 0.0f;
        }
    }
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_NORM_EPSILON));
    const float scale = is_query ? metal::rsqrt(float(RCH_WIDTH)) : 1.0f;
    for (uint i = 0; i < row_run; ++i) {
        const uint t = first_row + i;
        if (long(t) < piece_len) {
            load_conv_input(projection, window, slot, local + long(i), channel, inputs[RECURRENT_TAPS - 1],
                seismic_words);
        } else {
            RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; ++c) inputs[RECURRENT_TAPS - 1][c] = 0.0f;
        }
        float values[RCH_CPL];
        float squares = 0.0f;
        RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; ++c) {
            float sum = weights[RECURRENT_TAPS - 1][c] * inputs[RECURRENT_TAPS - 1][c];
            RECURRENT_UNROLL for (uint tap = 0; tap < RECURRENT_TAPS - 1; ++tap) {
                sum = metal::fma(weights[tap][c], inputs[tap][c], sum);
            }
            values[c] = long(t) < piece_len ? sum / (1.0f + metal::exp(-sum)) : 0.0f;
            squares = metal::fma(values[c], values[c], squares);
        }
        RECURRENT_UNROLL for (uint tap = 0; tap < RECURRENT_TAPS - 1; ++tap) {
            RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; ++c) {
                inputs[tap][c] = inputs[tap + 1][c];
            }
        }
        if (normalize) {
            const float inverse = long(t) < piece_len ? metal::rsqrt(simd_sum(squares) + epsilon) * scale : 0.0f;
            RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; ++c) {
                values[c] *= inverse;
                if (tg_dst != nullptr) {
                    tg_dst[(t - tg_row_offset) * tg_stride + column + c] = values[c];
                }
            }
            if (dev_hi != nullptr) {
                if (dev_lo != nullptr) {
                    store_half_split_cpl(dev_hi + ulong(t) * RCH_WIDTH + column,
                        dev_lo + ulong(t) * RCH_WIDTH + column, values);
                } else {
                    store_half_cpl(dev_hi + ulong(t) * RCH_WIDTH + column, values);
                }
            }
        } else {
            store_half_cpl(dev_hi + ulong(t) * RCH_WIDTH + column, values);
        }
    }
}

// Threadgroup memory for `rch_prepare`, declared in the kernel entry point.
template <uint DECAYS>
struct PrepareShared;

template <>
struct PrepareShared<1> {
    static constant constexpr bool SPLIT_Q = (RCH_WIDTH > 128);
    static constant constexpr uint Q_ROWS = SPLIT_Q ? 8u : RCH_PIECE;
    static constant constexpr uint STRIDE = SPLIT_Q ? (RCH_WIDTH + 1u) : RCH_PREP_STRIDE;
    static constant constexpr uint SYS_COUNT = SPLIT_Q ? 1u : RCH_GROUP;
    float keys[RCH_PIECE][STRIDE];
    float queries[Q_ROWS][STRIDE];
    float kk[RCH_PIECE][RCH_PIECE];
    float qk[RCH_PIECE][RCH_PIECE];
    float cumulative[RCH_GROUP][RCH_PIECE];
    float beta[RCH_GROUP][RCH_PIECE];
    float system[SYS_COUNT][RCH_PIECE][RCH_PIECE + 1];
};

template <uint DECAYS>
struct PrepareShared {
    static_assert(RCH_WIDTH <= 128, "Kimi prepare threadgroup memory fits to W 128");
    float queries[RCH_PIECE][RCH_WIDTH + 1];
    float keys[RCH_PIECE][RCH_WIDTH + 1];
    float decays[RCH_PIECE][RCH_WIDTH + 1];
    float beta[RCH_GROUP][RCH_PIECE];
    float system[RCH_PIECE][RCH_PIECE + 1];
};

// Per-rule prepare body: DECAYS == 1 (Gated DeltaNet) vs DECAYS == W (Kimi).
template <typename Rule, uint DECAYS = Rule::DECAYS>
struct PrepareBody;

template <typename Rule>
struct PrepareBody<Rule, 1> {
    using Form = Layout<1>;
    static constant constexpr bool SPLIT_Q = PrepareShared<1>::SPLIT_Q;
    static constant constexpr uint STRIDE = PrepareShared<1>::STRIDE;

    static inline void run(Rule rule, device const recurrent::Storage *projection, device const float *convolution,
        device const slab::Table<activation::Act> *window, device uchar *pieces,
        threadgroup PrepareShared<1> &sh, recurrent::Slot slot, uint global_piece, long piece_first, long piece_len,
        uint key_head, uint thread_index, uint simdgroup, uint lane, constant ulong *seismic_words) {
        device half *ops = Form::operands(pieces, global_piece, key_head, 0u);
        const uint column = lane * RCH_CPL;

        // Main convolution pass: q (when !SPLIT_Q), k, and member value heads' v.
        constexpr uint MAIN_PARTS = (SPLIT_Q ? 1u : 2u) + RCH_GROUP;
        constexpr uint ROW_RUN = prepare_row_run(MAIN_PARTS);
        constexpr uint RUNS = RCH_PIECE / ROW_RUN;
        for (uint item = simdgroup; item < MAIN_PARTS * RUNS; item += RCH_PREPARE_WARPS) {
            const uint part = item % MAIN_PARTS;
            const uint first_row = ROW_RUN * (item / MAIN_PARTS);
            if (!SPLIT_Q && part == 0u) {
                convolve_run(projection, convolution, window, slot, piece_first, piece_len, first_row, ROW_RUN,
                    key_head * RCH_WIDTH, column, true, true, &sh.queries[0][0], STRIDE, 0u,
                    Form::q_ptr(ops), nullptr, seismic_words);
            } else if ((!SPLIT_Q && part == 1u) || (SPLIT_Q && part == 0u)) {
                convolve_run(projection, convolution, window, slot, piece_first, piece_len, first_row, ROW_RUN,
                    (RCH_KEY_HEADS + key_head) * RCH_WIDTH, column, true, false, &sh.keys[0][0], STRIDE, 0u,
                    Form::k_hi_ptr(ops), Form::k_lo_ptr(ops), seismic_words);
            } else {
                const uint member = part - (SPLIT_Q ? 1u : 2u);
                const uint vh = member_head(key_head, member, seismic_words);
                device half *v_dst = reinterpret_cast<device half *>(Form::record(pieces, global_piece, vh) + RECORD_V);
                convolve_run(projection, convolution, window, slot, piece_first, piece_len, first_row, ROW_RUN,
                    (2u * RCH_KEY_HEADS + vh) * RCH_WIDTH, column, false, false, nullptr, 0u, 0u,
                    v_dst, nullptr, seismic_words);
            }
        }

        // Gates and inclusive prefix sum of log decays (16-lane prefix sum per member).
        if (simdgroup < (RCH_GROUP + 1u) / 2u) {
            const uint member = thread_index / RCH_PIECE;
            const uint t = thread_index % RCH_PIECE;
            float log_decay = 0.0f;
            float b = 0.0f;
            if (member < RCH_GROUP && long(t) < piece_len) {
                const uint vh = member_head(key_head, member, seismic_words);
                const auto hg = rule.gates(vh, seismic_words);
                log_decay = hg.at(ulong(piece_first) + t, 0ul, b, seismic_words);
            }
            RECURRENT_UNROLL for (ushort offset = 1; offset < RCH_PIECE; offset <<= 1) {
                const float up = simd_shuffle_up(log_decay, offset);
                if (t >= offset) log_decay += up;
            }
            if (member < RCH_GROUP) {
                sh.cumulative[member][t] = log_decay;
                sh.beta[member][t] = b;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint t = thread_index / RCH_PIECE;
        const uint s = thread_index % RCH_PIECE;
        if constexpr (!SPLIT_Q) {
            // Fast path (W <= 128): compute the 6 lower-triangle 8x8 tiles of K K^T and Q K^T
            // in parallel using simdgroup_float8x8 across simdgroups 0..5.
            if (simdgroup < 6u) {
                const bool is_q = simdgroup >= 3u;
                const uint tile_id = is_q ? (simdgroup - 3u) : simdgroup;
                const uint row_tile = tile_id == 0u ? 0u : 1u;
                const uint col_tile = tile_id == 2u ? 1u : 0u;
                threadgroup const float *left_base = (is_q ? &sh.queries[0][0] : &sh.keys[0][0])
                    + (8u * row_tile) * STRIDE;
                threadgroup const float *right_base = &sh.keys[0][0] + (8u * col_tile) * STRIDE;
                simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
                    simdgroup_float8x8 a_tile, b_tile;
                    simdgroup_load(a_tile, left_base + 8u * c, STRIDE);
                    simdgroup_load(b_tile, right_base + 8u * c, STRIDE, ulong2(0), true);
                    simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
                }
                threadgroup float *dst = (is_q ? &sh.qk[0][0] : &sh.kk[0][0])
                    + (8u * row_tile) * RCH_PIECE + 8u * col_tile;
                simdgroup_store(acc, dst, RCH_PIECE);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float key_dot = (s < t && long(t) < piece_len) ? sh.kk[t][s] : 0.0f;
            const float query_dot = (s <= t && long(t) < piece_len) ? sh.qk[t][s] : 0.0f;
            RECURRENT_UNROLL for (uint member = 0; member < RCH_GROUP; ++member) {
                const uint vh = member_head(key_head, member, seismic_words);
                device uchar *rec = Form::record(pieces, global_piece, vh);
                const float gt = sh.cumulative[member][t];
                const float gs = sh.cumulative[member][s];
                const float decay = (s <= t && long(t) < piece_len) ? (s < t ? metal::exp(gt - gs) : 1.0f) : 0.0f;
                sh.system[member][t][s] = (s < t) ? sh.beta[member][t] * decay * key_dot : 0.0f;
                reinterpret_cast<device half *>(rec + RECORD_D)[t * RCH_PIECE + s] =
                    half((s <= t && long(t) < piece_len) ? decay * query_dot : 0.0f);
                if (thread_index < RCH_PIECE) {
                    const float last = sh.cumulative[member][RCH_PIECE - 1];
                    device float *gates = reinterpret_cast<device float *>(rec + Form::RECORD_GATES);
                    gates[s] = metal::exp(gs);
                    gates[RCH_PIECE + s] = metal::exp(last - gs);
                    if (s == 0u) gates[2u * RCH_PIECE] = metal::exp(last);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Solve Tb = (I + A)^-1 diag(beta) for ALL member value heads in parallel.
            const uint member = thread_index / RCH_PIECE;
            const uint lane16 = thread_index % RCH_PIECE;
            if (member < RCH_GROUP) {
                const uint vh = member_head(key_head, member, seismic_words);
                device half *tb = reinterpret_cast<device half *>(Form::record(pieces, global_piece, vh) + RECORD_TB);
                solve_tb(sh.system[member], sh.beta[member][lane16], tb, lane16);
            }
        } else {
            // W > 128 fallback: K K^T from keys, then Q in two 8-row halves.
            float kk0 = 0.0f, kk1 = 0.0f;
            if (s <= t && long(t) < piece_len) {
                for (uint c = 0; c < RCH_WIDTH; c += 2) {
                    kk0 = metal::fma(sh.keys[t][c], sh.keys[s][c], kk0);
                    kk1 = metal::fma(sh.keys[t][c + 1], sh.keys[s][c + 1], kk1);
                }
            }
            const float key_dot = kk0 + kk1;
            float query_dot = 0.0f;
            for (uint half_idx = 0; half_idx < 2u; ++half_idx) {
                if (simdgroup < 8u) {
                    convolve_run(projection, convolution, window, slot, piece_first, piece_len,
                        half_idx * 8u + simdgroup, 1u, key_head * RCH_WIDTH, column, true, true,
                        &sh.queries[0][0], STRIDE, half_idx * 8u, Form::q_ptr(ops), nullptr, seismic_words);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (t >= half_idx * 8u && t < (half_idx + 1u) * 8u && s <= t && long(t) < piece_len) {
                    float qk0 = 0.0f, qk1 = 0.0f;
                    for (uint c = 0; c < RCH_WIDTH; c += 2) {
                        qk0 = metal::fma(sh.queries[t - half_idx * 8u][c], sh.keys[s][c], qk0);
                        qk1 = metal::fma(sh.queries[t - half_idx * 8u][c + 1], sh.keys[s][c + 1], qk1);
                    }
                    query_dot = qk0 + qk1;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint member = 0; member < RCH_GROUP; ++member) {
                const uint vh = member_head(key_head, member, seismic_words);
                device uchar *rec = Form::record(pieces, global_piece, vh);
                const float gt = sh.cumulative[member][t];
                const float gs = sh.cumulative[member][s];
                const float decay = (s <= t && long(t) < piece_len) ? (s < t ? metal::exp(gt - gs) : 1.0f) : 0.0f;
                sh.system[0][t][s] = (s < t) ? sh.beta[member][t] * decay * key_dot : 0.0f;
                reinterpret_cast<device half *>(rec + RECORD_D)[t * RCH_PIECE + s] =
                    half((s <= t && long(t) < piece_len) ? decay * query_dot : 0.0f);
                if (thread_index < RCH_PIECE) {
                    const float last = sh.cumulative[member][RCH_PIECE - 1];
                    device float *gates = reinterpret_cast<device float *>(rec + Form::RECORD_GATES);
                    gates[s] = metal::exp(gs);
                    gates[RCH_PIECE + s] = metal::exp(last - gs);
                    if (s == 0u) gates[2u * RCH_PIECE] = metal::exp(last);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (thread_index < RCH_PIECE) {
                    solve_tb(sh.system[0], sh.beta[member][thread_index],
                        reinterpret_cast<device half *>(rec + RECORD_TB), thread_index);
                }
                if (member + 1u < RCH_GROUP) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }
};

template <typename Rule, uint DECAYS>
struct PrepareBody {
    using Form = Layout<DECAYS>;

    static inline void run(Rule rule, device const recurrent::Storage *projection, device const float *convolution,
        device const slab::Table<activation::Act> *window, device uchar *pieces,
        threadgroup PrepareShared<DECAYS> &sh, recurrent::Slot slot, uint global_piece, long piece_first,
        long piece_len, uint key_head, uint thread_index, uint simdgroup, uint lane, constant ulong *seismic_words) {
        const uint column = lane * RCH_CPL;
        constexpr uint PARTS = 2u + RCH_GROUP;
        constexpr uint ROW_RUN = prepare_row_run(PARTS);
        constexpr uint RUNS = RCH_PIECE / ROW_RUN;
        for (uint item = simdgroup; item < PARTS * RUNS; item += RCH_PREPARE_WARPS) {
            const uint part = item % PARTS;
            const uint first_row = ROW_RUN * (item / PARTS);
            if (part == 0u) {
                convolve_run(projection, convolution, window, slot, piece_first, piece_len, first_row, ROW_RUN,
                    key_head * RCH_WIDTH, column, true, true, &sh.queries[0][0], RCH_WIDTH + 1u, 0u,
                    nullptr, nullptr, seismic_words);
            } else if (part == 1u) {
                convolve_run(projection, convolution, window, slot, piece_first, piece_len, first_row, ROW_RUN,
                    (RCH_KEY_HEADS + key_head) * RCH_WIDTH, column, true, false, &sh.keys[0][0], RCH_WIDTH + 1u, 0u,
                    nullptr, nullptr, seismic_words);
            } else {
                const uint member = part - 2u;
                const uint vh = member_head(key_head, member, seismic_words);
                device half *v_dst = reinterpret_cast<device half *>(Form::record(pieces, global_piece, vh) + RECORD_V);
                convolve_run(projection, convolution, window, slot, piece_first, piece_len, first_row, ROW_RUN,
                    (2u * RCH_KEY_HEADS + vh) * RCH_WIDTH, column, false, false, nullptr, 0u, 0u,
                    v_dst, nullptr, seismic_words);
            }
        }
        if (thread_index < RCH_GROUP * RCH_PIECE) {
            const uint member = thread_index / RCH_PIECE;
            const uint t = thread_index % RCH_PIECE;
            float b = 0.0f;
            if (long(t) < piece_len) {
                const uint vh = member_head(key_head, member, seismic_words);
                const auto hg = rule.gates(vh, seismic_words);
                (void)hg.at(ulong(piece_first) + t, 0ul, b, seismic_words);
            }
            sh.beta[member][t] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint t = thread_index / RCH_PIECE;
        const uint s = thread_index % RCH_PIECE;
        for (uint member = 0; member < RCH_GROUP; ++member) {
            const uint vh = member_head(key_head, member, seismic_words);
            const auto hg = rule.gates(vh, seismic_words);
            for (uint row = simdgroup; row < RCH_PIECE; row += RCH_PREPARE_WARPS) {
                if (long(row) < piece_len) {
                    if constexpr (RCH_CPL % 4 == 0) {
                        RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; c += 4) {
                            float unused_b;
                            const float4 ld = hg.at4(ulong(piece_first) + row, column + c, unused_b, seismic_words);
                            sh.decays[row][column + c + 0] = ld.x;
                            sh.decays[row][column + c + 1] = ld.y;
                            sh.decays[row][column + c + 2] = ld.z;
                            sh.decays[row][column + c + 3] = ld.w;
                        }
                    } else {
                        RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; ++c) {
                            float unused_b;
                            sh.decays[row][column + c] = hg.at(ulong(piece_first) + row, column + c, unused_b,
                                seismic_words);
                        }
                    }
                } else {
                    RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; ++c) sh.decays[row][column + c] = 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = thread_index; c < RCH_WIDTH; c += RCH_PREPARE_THREADS) {
                float sum = 0.0f;
                RECURRENT_UNROLL for (uint row = 0; row < RCH_PIECE; ++row) {
                    sum += sh.decays[row][c];
                    sh.decays[row][c] = sum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float kk0 = 0.0f, kk1 = 0.0f, qk0 = 0.0f, qk1 = 0.0f;
            if (s <= t) {
                for (uint c = 0; c < RCH_WIDTH; c += 2) {
                    const float d0 = metal::exp(sh.decays[t][c] - sh.decays[s][c]);
                    const float d1 = metal::exp(sh.decays[t][c + 1] - sh.decays[s][c + 1]);
                    kk0 = metal::fma(sh.keys[t][c] * sh.keys[s][c], d0, kk0);
                    kk1 = metal::fma(sh.keys[t][c + 1] * sh.keys[s][c + 1], d1, kk1);
                    qk0 = metal::fma(sh.queries[t][c] * sh.keys[s][c], d0, qk0);
                    qk1 = metal::fma(sh.queries[t][c + 1] * sh.keys[s][c + 1], d1, qk1);
                }
            }
            sh.system[t][s] = (s < t) ? sh.beta[member][t] * (kk0 + kk1) : 0.0f;
            device uchar *rec = Form::record(pieces, global_piece, vh);
            reinterpret_cast<device half *>(rec + RECORD_D)[t * RCH_PIECE + s] =
                half(long(t) < piece_len ? (qk0 + qk1) : 0.0f);
            device half *ops = Form::operands(pieces, global_piece, key_head, vh);
            float last[RCH_CPL];
            RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; ++c) last[c] = sh.decays[RCH_PIECE - 1][column + c];
            if (simdgroup == 0u) {
                float gate_vals[RCH_CPL];
                RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; ++c) gate_vals[c] = metal::exp(last[c]);
                store_float_cpl(reinterpret_cast<device float *>(rec + Form::RECORD_GATES) + column, gate_vals);
            }
            for (uint row = simdgroup; row < RCH_PIECE; row += RCH_PREPARE_WARPS) {
                float q_dec[RCH_CPL], k_dec[RCH_CPL], u_dec[RCH_CPL];
                RECURRENT_UNROLL for (uint c = 0; c < RCH_CPL; ++c) {
                    const float g = sh.decays[row][column + c];
                    const float gamma = metal::exp(g);
                    const float key = sh.keys[row][column + c];
                    q_dec[c] = sh.queries[row][column + c] * gamma;
                    k_dec[c] = key * gamma;
                    u_dec[c] = key * metal::exp(last[c] - g);
                }
                const ulong off = ulong(row) * RCH_WIDTH + column;
                store_half_cpl(Form::q_ptr(ops) + off, q_dec);
                store_half_split_cpl(Form::k_hi_ptr(ops) + off, Form::k_lo_ptr(ops) + off, k_dec);
                store_half_split_cpl(Form::u_hi_ptr(ops) + off, Form::u_lo_ptr(ops) + off, u_dec);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (thread_index < RCH_PIECE) {
                solve_tb(sh.system, sh.beta[member][thread_index], reinterpret_cast<device half *>(rec + RECORD_TB),
                    thread_index);
            }
            if (member + 1u < RCH_GROUP) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
};

constant constexpr bool F32_SCAN = (RCH_WIDTH <= 128);

// Threadgroup memory for `rch_scan`: staged piece operands aliased with the
// sequential fallback buffers (`advance_rows`).
template <uint DECAYS, bool F32 = F32_SCAN>
struct StageBuf;

template <uint DECAYS>
struct StageBuf<DECAYS, true> {
    float q[RCH_PIECE * RCH_Q_STRIDE];
    float k[RCH_PIECE * RCH_Q_STRIDE];
    float v[RCH_PIECE * RCH_V_STRIDE];
    float tb[RCH_PIECE * RCH_PIECE];
    float d[RCH_PIECE * RCH_PIECE];
    float gates[Layout<DECAYS>::GATES_FLOATS];
};

template <uint DECAYS>
struct StageBuf<DECAYS, false> {
    half q[RCH_PIECE * RCH_Q_STRIDE];
    half k_hi[RCH_PIECE * RCH_Q_STRIDE];
    half k_lo[RCH_PIECE * RCH_Q_STRIDE];
    half v[RCH_PIECE * RCH_V_STRIDE];
    half tb[RCH_PIECE * RCH_PIECE];
    half d[RCH_PIECE * RCH_PIECE];
    float gates[Layout<DECAYS>::GATES_FLOATS];
};

template <uint DECAYS>
struct SeqBuf {
    float queries[RCH_SEQ_SPAN * RCH_SEQ_INPUT_STRIDE];
    float keys[RCH_SEQ_SPAN * RCH_SEQ_INPUT_STRIDE];
    float values[RCH_SEQ_SPAN * RCH_SEQ_VALUE_STRIDE];
    float gates[RCH_SEQ_SPAN * (DECAYS + 1u)];
};

template <uint DECAYS>
union ScanShared {
    StageBuf<DECAYS, F32_SCAN> stage;
    SeqBuf<DECAYS> seq;
};

// Widen split-F16 (hi + lo) into a [16, RCH_Q_STRIDE] threadgroup float buffer.
inline void stage_matrix_w_split_f32(threadgroup float *dst, device const half *src_hi,
    device const half *src_lo, uint thread_index) {
    constexpr uint QUADS = (RCH_PIECE * RCH_WIDTH) / 4u;
    constexpr uint ROW_QUADS = RCH_WIDTH / 4u;
    for (uint item = thread_index; item < QUADS; item += RCH_THREADS) {
        const uint r = item / ROW_QUADS;
        const uint c = 4u * (item % ROW_QUADS);
        const ulong off = ulong(r) * RCH_WIDTH + c;
        const float4 hi = float4(*reinterpret_cast<device const half4 *>(src_hi + off));
        const float4 lo = float4(*reinterpret_cast<device const half4 *>(src_lo + off));
        *reinterpret_cast<threadgroup float4 *>(dst + r * RCH_Q_STRIDE + c) = hi + lo;
    }
}

template <uint DECAYS>
inline void stage_piece_f32(threadgroup StageBuf<DECAYS, true> &st, device uchar *pieces, uint global_piece,
    uint key_head, uint head, uint block_row0, uint thread_index) {
    using Form = Layout<DECAYS>;
    device half *ops = Form::operands(pieces, global_piece, key_head, head);
    device const uchar *rec = Form::record(pieces, global_piece, head);
    {
        device const half *q_src = Form::q_ptr(ops);
        constexpr uint QUADS = (RCH_PIECE * RCH_WIDTH) / 4u;
        constexpr uint ROW_QUADS = RCH_WIDTH / 4u;
        for (uint item = thread_index; item < QUADS; item += RCH_THREADS) {
            const uint r = item / ROW_QUADS;
            const uint c = 4u * (item % ROW_QUADS);
            *reinterpret_cast<threadgroup float4 *>(st.q + r * RCH_Q_STRIDE + c) =
                float4(*reinterpret_cast<device const half4 *>(q_src + ulong(r) * RCH_WIDTH + c));
        }
    }
    stage_matrix_w_split_f32(st.k, Form::k_hi_ptr(ops), Form::k_lo_ptr(ops), thread_index);
    {
        device const half *v_src = reinterpret_cast<device const half *>(rec + RECORD_V) + block_row0;
        constexpr uint V_QUADS = (RCH_PIECE * RCH_ROWS) / 4u;
        constexpr uint ROW_QUADS = RCH_ROWS / 4u;
        for (uint item = thread_index; item < V_QUADS; item += RCH_THREADS) {
            const uint r = item / ROW_QUADS;
            const uint c = 4u * (item % ROW_QUADS);
            *reinterpret_cast<threadgroup float4 *>(st.v + r * RCH_V_STRIDE + c) =
                float4(*reinterpret_cast<device const half4 *>(v_src + ulong(r) * RCH_WIDTH + c));
        }
    }
    {
        device const half4 *tb_src = reinterpret_cast<device const half4 *>(rec + RECORD_TB);
        device const half4 *d_src = reinterpret_cast<device const half4 *>(rec + RECORD_D);
        constexpr uint SQ_QUADS = (RCH_PIECE * RCH_PIECE) / 4u;
        for (uint item = thread_index; item < SQ_QUADS; item += RCH_THREADS) {
            reinterpret_cast<threadgroup float4 *>(st.tb)[item] = float4(tb_src[item]);
            reinterpret_cast<threadgroup float4 *>(st.d)[item] = float4(d_src[item]);
        }
    }
    {
        device const float4 *g_src = reinterpret_cast<device const float4 *>(rec + Form::RECORD_GATES);
        constexpr uint G_QUADS = Form::GATES_FLOATS / 4u;
        for (uint item = thread_index; item < G_QUADS; item += RCH_THREADS) {
            reinterpret_cast<threadgroup float4 *>(st.gates)[item] = g_src[item];
        }
    }
}

// Copy a [16, W] device half matrix into a [16, RCH_Q_STRIDE] threadgroup buffer.
inline void stage_matrix_w(threadgroup half *dst, device const half *src, uint thread_index) {
    constexpr uint QUADS = (RCH_PIECE * RCH_WIDTH) / 4u;
    constexpr uint ROW_QUADS = RCH_WIDTH / 4u;
    for (uint item = thread_index; item < QUADS; item += RCH_THREADS) {
        const uint r = item / ROW_QUADS;
        const uint c = 4u * (item % ROW_QUADS);
        *reinterpret_cast<threadgroup half4 *>(dst + r * RCH_Q_STRIDE + c) =
            *reinterpret_cast<device const half4 *>(src + ulong(r) * RCH_WIDTH + c);
    }
}

template <uint DECAYS>
inline void stage_piece_half(threadgroup StageBuf<DECAYS, false> &st, device uchar *pieces, uint global_piece,
    uint key_head, uint head, uint block_row0, uint thread_index) {
    using Form = Layout<DECAYS>;
    device half *ops = Form::operands(pieces, global_piece, key_head, head);
    device const uchar *rec = Form::record(pieces, global_piece, head);
    stage_matrix_w(st.q, Form::q_ptr(ops), thread_index);
    stage_matrix_w(st.k_hi, Form::k_hi_ptr(ops), thread_index);
    stage_matrix_w(st.k_lo, Form::k_lo_ptr(ops), thread_index);
    {
        device const half *v_src = reinterpret_cast<device const half *>(rec + RECORD_V) + block_row0;
        constexpr uint V_QUADS = (RCH_PIECE * RCH_ROWS) / 4u;
        constexpr uint ROW_QUADS = RCH_ROWS / 4u;
        for (uint item = thread_index; item < V_QUADS; item += RCH_THREADS) {
            const uint r = item / ROW_QUADS;
            const uint c = 4u * (item % ROW_QUADS);
            *reinterpret_cast<threadgroup half4 *>(st.v + r * RCH_V_STRIDE + c) =
                *reinterpret_cast<device const half4 *>(v_src + ulong(r) * RCH_WIDTH + c);
        }
    }
    {
        device const half4 *tb_src = reinterpret_cast<device const half4 *>(rec + RECORD_TB);
        device const half4 *d_src = reinterpret_cast<device const half4 *>(rec + RECORD_D);
        constexpr uint SQ_QUADS = (RCH_PIECE * RCH_PIECE) / 4u;
        for (uint item = thread_index; item < SQ_QUADS; item += RCH_THREADS) {
            reinterpret_cast<threadgroup half4 *>(st.tb)[item] = tb_src[item];
            reinterpret_cast<threadgroup half4 *>(st.d)[item] = d_src[item];
        }
    }
    {
        device const float4 *g_src = reinterpret_cast<device const float4 *>(rec + Form::RECORD_GATES);
        constexpr uint G_QUADS = Form::GATES_FLOATS / 4u;
        for (uint item = thread_index; item < G_QUADS; item += RCH_THREADS) {
            reinterpret_cast<threadgroup float4 *>(st.gates)[item] = g_src[item];
        }
    }
}

inline void split_sg_tile(simdgroup_float8x8 src, thread simdgroup_half8x8 &hi, thread simdgroup_half8x8 &lo) {
    const float f0 = src.thread_elements()[0];
    const float f1 = src.thread_elements()[1];
    const half h0 = half(f0);
    const half h1 = half(f1);
    hi.thread_elements()[0] = h0;
    hi.thread_elements()[1] = h1;
    lo.thread_elements()[0] = half(f0 - float(h0));
    lo.thread_elements()[1] = half(f1 - float(h1));
}

} // namespace rch

// L1: one threadgroup (256 threads) per (piece, key head).
template <typename Rule>
inline void rch_prepare(Rule rule, device const recurrent::Storage *projection, device const float *convolution,
    device const int *segments, device const int *stop, device const int *previous_bank,
    device const int *previous_tape, device const int *following_bank,
    device const slab::Table<activation::Act> *window, device uchar *pieces,
    threadgroup rch::PrepareShared<Rule::DECAYS> &sh, uint3 group, uint thread_index,
    uint simdgroup, uint lane, constant ulong *seismic_words) {
    uint remaining = group.x;
    recurrent::Slot slot;
    long piece_first = 0;
    long piece_len = 0;
    bool owned = false;
    for (ulong index = 0; index < SEISMIC_DIM_B && !owned; ++index) {
        const recurrent::Slot candidate = versions::slot_of({segments, stop, previous_bank, previous_tape,
            following_bank}, index, seismic_words);
        const uint count = rch::scratch_pieces(candidate);
        if (remaining < count) {
            slot = candidate;
            recurrent::piece_of<RCH_PIECE>(candidate, remaining, piece_first, piece_len);
            owned = true;
        } else {
            remaining -= count;
        }
    }
    if (!owned) return;
    rch::PrepareBody<Rule>::run(rule, projection, convolution, window, pieces, sh, slot, group.x, piece_first,
        piece_len, group.y, thread_index, simdgroup, lane, seismic_words);
}

// L2: one threadgroup per (ROWS state rows, value head, slot).
template <typename Rule>
inline void rch_scan(Rule rule, device const recurrent::Storage *projection, device const float *convolution,
    device const int *segments, device const int *stop, device const int *previous_bank,
    device const int *previous_tape, device const int *following_bank,
    device const slab::Table<activation::Act> *window, device const slab::Table<element::F32> *delta,
    device const slab::Table<element::F32> *tape, device uchar *pieces, device recurrent::Storage *mixed,
    threadgroup rch::ScanShared<Rule::DECAYS> &shared, uint3 group, uint thread_index, uint simdgroup,
    uint lane_index, constant ulong *seismic_words) {
    constexpr uint DECAYS = Rule::DECAYS;
    using Form = rch::Layout<DECAYS>;
    const ulong width = RCH_WIDTH;
    const ulong head = group.y;
    const ulong slot = group.z;
    const ulong block_row0 = ulong(group.x) * RCH_ROWS;
    const ulong row0 = block_row0 + ulong(simdgroup) * 16;
    const ulong key_head = recurrent::key_head(head, seismic_words);
    const recurrent::Slot geometry = versions::slot_of({segments, stop, previous_bank, previous_tape,
        following_bank}, slot, seismic_words);
    if (geometry.lo == geometry.hi) return;
    const ulong source = geometry.source;
    const ulong target = geometry.target;
    if (group.x == 0) {
        recurrent::publish_window(projection, window, geometry, head, thread_index, RCH_THREADS, seismic_words);
    }

#define RCH_SEQUENTIAL(begin)                                                                               \
    recurrent::advance_rows<16, RCH_ROWS, RCH_SEQ_SPAN, RCH_SEQ_INPUT_STRIDE, RCH_SEQ_VALUE_STRIDE>(        \
        recurrent::Convolving{projection, window, convolution, geometry}, rule.gates(head, seismic_words),   \
        delta, tape, mixed, geometry, begin, head, block_row0, row0, shared.seq.queries, shared.seq.keys,   \
        shared.seq.values, shared.seq.gates, shared.seq.gates + RCH_SEQ_SPAN, thread_index, RCH_THREADS,    \
        simdgroup, lane_index, seismic_words)

    if (geometry.hi - geometry.lo <= RECURRENT_SEQUENTIAL_ROWS) {
        RCH_SEQUENTIAL(geometry.lo);
        return;
    }

    const ulong state_stride = SEISMIC_DELTA_STRIDE_2;
    const ulong lane_state = head * SEISMIC_DELTA_STRIDE_1 + row0 * state_stride;
    const element::Out<element::F32> published = versions::bank(delta, target, SEISMIC_DELTA_STRIDE_0,
        seismic_words).offset(lane_state);
    device const float *initial = versions::bank_view(delta, source, SEISMIC_DELTA_STRIDE_0, seismic_words)
        + lane_state;
    const uint first_global = rch::first_piece(segments, stop, previous_bank, previous_tape, following_bank,
        slot, seismic_words);
    const ulong pieces_before = recurrent::pieces_before<RCH_PIECE>(geometry);

    // Non-tensor-ops path (M1-M4): native F32 simdgroup_float8x8 for W <= 128
    // (1x MMA count instead of 3x), or half staging fallback for W > 128.
    const ushort lane = ushort(lane_index);
    const ushort quad = lane / 4;
    const ushort row = (quad & 4) + ((lane / 2) % 4);
    const ushort column = (quad & 2) * 2 + (lane % 2) * 2;

    simdgroup_float8x8 state[2][RCH_TILES];
    RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
        RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
            simdgroup_load(state[h][c], initial + 8 * h * state_stride + 8 * c, state_stride, ulong2(0), true);
        }
    }
    for (long entry = 0; entry < geometry.taped; ++entry) {
        device const float *tape_row = recurrent::tape_view(tape, source, entry, seismic_words);
        const float head_factor = DECAYS == 1
            ? tape_row[recurrent::tape_decay<DECAYS>(head, 0, seismic_words)] : 0.0f;
        float2 innovation[2];
        RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
            innovation[h] = float2(*reinterpret_cast<device const packed_float2 *>(
                tape_row + RECURRENT_TAPE_U + head * width + row0 + 8 * h + column));
        }
        RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
            const float factor = DECAYS == 1 ? head_factor
                : tape_row[recurrent::tape_decay<DECAYS>(head, 8 * c + row, seismic_words)];
            const float key_value = tape_row[RECURRENT_TAPE_K + key_head * width + 8 * c + row];
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                state[h][c].thread_elements()[0] *= factor;
                state[h][c].thread_elements()[0] = metal::fma(innovation[h].x, key_value,
                    state[h][c].thread_elements()[0]);
                state[h][c].thread_elements()[1] *= factor;
                state[h][c].thread_elements()[1] = metal::fma(innovation[h].y, key_value,
                    state[h][c].thread_elements()[1]);
            }
        }
    }
    if (geometry.stop == 0) {
        RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
            RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
                published.put_tile(8 * h * state_stride + 8 * c, state[h][c], state_stride, true);
            }
        }
    }

    for (ulong piece = 0; piece < pieces_before; ++piece) {
        long first;
        long length;
        recurrent::piece_of<RCH_PIECE>(geometry, piece, first, length);
        const uint global_piece = first_global + uint(piece);

        if constexpr (rch::F32_SCAN) {
        rch::stage_piece_f32<DECAYS>(shared.stage, pieces, global_piece, uint(key_head), uint(head),
            uint(block_row0), thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // X = K S^T and Y = Q S^T: [2][2] of 8x8 tiles in native F32.
        simdgroup_float8x8 removed[2][2];
        simdgroup_float8x8 output[2][2];
        RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                removed[p][h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                output[p][h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
            RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
                simdgroup_float8x8 k_tile, q_tile;
                const uint off = (8u * p) * RCH_Q_STRIDE + 8u * c;
                simdgroup_load(k_tile, shared.stage.k + off, RCH_Q_STRIDE);
                simdgroup_load(q_tile, shared.stage.q + off, RCH_Q_STRIDE);
                RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                    simdgroup_multiply_accumulate(removed[p][h], k_tile, state[h][c], removed[p][h]);
                    simdgroup_multiply_accumulate(output[p][h], q_tile, state[h][c], output[p][h]);
                }
            }
        }

        if constexpr (DECAYS > 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            device half *ops = Form::operands(pieces, global_piece, uint(key_head), uint(head));
            rch::stage_matrix_w_split_f32(shared.stage.k, Form::u_hi_ptr(ops), Form::u_lo_ptr(ops), thread_index);
        }

        // Z = V - diag(gamma) X, U = Tb Z, O = diag(gamma) Y + D U.
        RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
            const float g = Form::gamma(shared.stage.gates, 8u * p + row);
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                simdgroup_float8x8 val;
                simdgroup_load(val, shared.stage.v + (8u * p) * RCH_V_STRIDE + simdgroup * 16u + 8u * h,
                    RCH_V_STRIDE);
                RECURRENT_UNROLL for (ushort e = 0; e < 2; ++e) {
                    removed[p][h].thread_elements()[e] = val.thread_elements()[e]
                        - removed[p][h].thread_elements()[e] * g;
                    output[p][h].thread_elements()[e] *= g;
                }
            }
        }
        simdgroup_float8x8 update[2][2];
        RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                update[p][h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        RECURRENT_UNROLL for (uint q = 0; q < 2; ++q) {
            RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
                simdgroup_float8x8 tb_tile;
                simdgroup_load(tb_tile, shared.stage.tb + (8u * p) * RCH_PIECE + 8u * q, RCH_PIECE);
                RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                    simdgroup_multiply_accumulate(update[p][h], tb_tile, removed[q][h], update[p][h]);
                }
            }
        }
        RECURRENT_UNROLL for (uint q = 0; q < 2; ++q) {
            RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
                simdgroup_float8x8 d_tile;
                simdgroup_load(d_tile, shared.stage.d + (8u * p) * RCH_PIECE + 8u * q, RCH_PIECE);
                RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                    simdgroup_multiply_accumulate(output[p][h], d_tile, update[q][h], output[p][h]);
                }
            }
        }
        RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
            const uint piece_row = 8u * p + row;
            const float tl = Form::tail(shared.stage.gates, piece_row);
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                if (long(piece_row) < length) {
                    device recurrent::Storage *destination = mixed
                        + recurrent::raw_index(ulong(first + long(piece_row)), head, row0 + 8 * h + column,
                            seismic_words);
                    *reinterpret_cast<device activation::Act::packed2 *>(destination) =
                        activation::Act::packed2(activation::Act::store2(
                            float2(output[p][h].thread_elements()[0], output[p][h].thread_elements()[1])));
                }
                update[p][h].thread_elements()[0] *= tl;
                update[p][h].thread_elements()[1] *= tl;
            }
        }

        if constexpr (DECAYS > 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // S^T <- diag(f) S^T + K'^T U.
        RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
            const float factor = Form::factor(shared.stage.gates, 8u * c + row);
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                state[h][c].thread_elements()[0] *= factor;
                state[h][c].thread_elements()[1] *= factor;
            }
            RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
                simdgroup_float8x8 k_t;
                simdgroup_load(k_t, shared.stage.k + (8u * p) * RCH_Q_STRIDE + 8u * c, RCH_Q_STRIDE, ulong2(0), true);
                RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                    simdgroup_multiply_accumulate(state[h][c], k_t, update[p][h], state[h][c]);
                }
            }
        }

        } else {
        rch::stage_piece_half<DECAYS>(shared.stage, pieces, global_piece, uint(key_head), uint(head),
            uint(block_row0), thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 removed[2][2];
        simdgroup_float8x8 removed_lo[2][2];
        simdgroup_float8x8 removed_klo[2][2];
        simdgroup_float8x8 output[2][2];
        RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                removed[p][h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                removed_lo[p][h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                removed_klo[p][h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                output[p][h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
            simdgroup_half8x8 s_hi[2], s_lo[2];
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                rch::split_sg_tile(state[h][c], s_hi[h], s_lo[h]);
            }
            RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
                simdgroup_half8x8 k_hi, k_lo, q_hi;
                const uint off = (8u * p) * RCH_Q_STRIDE + 8u * c;
                simdgroup_load(k_hi, shared.stage.k_hi + off, RCH_Q_STRIDE);
                simdgroup_load(k_lo, shared.stage.k_lo + off, RCH_Q_STRIDE);
                simdgroup_load(q_hi, shared.stage.q + off, RCH_Q_STRIDE);
                RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                    simdgroup_multiply_accumulate(removed[p][h], k_hi, s_hi[h], removed[p][h]);
                    simdgroup_multiply_accumulate(output[p][h], q_hi, s_hi[h], output[p][h]);
                    simdgroup_multiply_accumulate(removed_lo[p][h], k_hi, s_lo[h], removed_lo[p][h]);
                    simdgroup_multiply_accumulate(removed_klo[p][h], k_lo, s_hi[h], removed_klo[p][h]);
                }
            }
        }

        simdgroup_half8x8 z_hi[2][2], z_lo[2][2];
        RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
            const float g = Form::gamma(shared.stage.gates, 8u * p + row);
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                simdgroup_half8x8 val_h;
                simdgroup_load(val_h, shared.stage.v + (8u * p) * RCH_V_STRIDE + simdgroup * 16u + 8u * h,
                    RCH_V_STRIDE);
                simdgroup_float8x8 z;
                RECURRENT_UNROLL for (ushort e = 0; e < 2; ++e) {
                    const float x = removed[p][h].thread_elements()[e]
                        + removed_lo[p][h].thread_elements()[e] + removed_klo[p][h].thread_elements()[e];
                    z.thread_elements()[e] = float(val_h.thread_elements()[e]) - x * g;
                    output[p][h].thread_elements()[e] *= g;
                }
                rch::split_sg_tile(z, z_hi[p][h], z_lo[p][h]);
            }
        }

        simdgroup_float8x8 update[2][2];
        RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                update[p][h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        RECURRENT_UNROLL for (uint q = 0; q < 2; ++q) {
            RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
                simdgroup_half8x8 tb_tile;
                simdgroup_load(tb_tile, shared.stage.tb + (8u * p) * RCH_PIECE + 8u * q, RCH_PIECE);
                RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                    simdgroup_multiply_accumulate(update[p][h], tb_tile, z_hi[q][h], update[p][h]);
                    simdgroup_multiply_accumulate(update[p][h], tb_tile, z_lo[q][h], update[p][h]);
                }
            }
        }

        simdgroup_half8x8 u_hi[2][2];
        RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                u_hi[p][h].thread_elements()[0] = half(update[p][h].thread_elements()[0]);
                u_hi[p][h].thread_elements()[1] = half(update[p][h].thread_elements()[1]);
            }
        }
        RECURRENT_UNROLL for (uint q = 0; q < 2; ++q) {
            RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
                simdgroup_half8x8 d_tile;
                simdgroup_load(d_tile, shared.stage.d + (8u * p) * RCH_PIECE + 8u * q, RCH_PIECE);
                RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                    simdgroup_multiply_accumulate(output[p][h], d_tile, u_hi[q][h], output[p][h]);
                }
            }
        }
        simdgroup_half8x8 w_hi[2][2], w_lo[2][2];
        RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
            const uint piece_row = 8u * p + row;
            const float tl = Form::tail(shared.stage.gates, piece_row);
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                if (long(piece_row) < length) {
                    device recurrent::Storage *destination = mixed
                        + recurrent::raw_index(ulong(first + long(piece_row)), head, row0 + 8 * h + column,
                            seismic_words);
                    *reinterpret_cast<device activation::Act::packed2 *>(destination) =
                        activation::Act::packed2(activation::Act::store2(
                            float2(output[p][h].thread_elements()[0], output[p][h].thread_elements()[1])));
                }
                update[p][h].thread_elements()[0] *= tl;
                update[p][h].thread_elements()[1] *= tl;
                rch::split_sg_tile(update[p][h], w_hi[p][h], w_lo[p][h]);
            }
        }

        RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
            const float factor = Form::factor(shared.stage.gates, 8u * c + row);
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                state[h][c].thread_elements()[0] *= factor;
                state[h][c].thread_elements()[1] *= factor;
            }
            RECURRENT_UNROLL for (uint p = 0; p < 2; ++p) {
                simdgroup_half8x8 k_hi_t, k_lo_t;
                const uint off = (8u * p) * RCH_Q_STRIDE + 8u * c;
                simdgroup_load(k_hi_t, shared.stage.k_hi + off, RCH_Q_STRIDE, ulong2(0), true);
                simdgroup_load(k_lo_t, shared.stage.k_lo + off, RCH_Q_STRIDE, ulong2(0), true);
                RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                    simdgroup_multiply_accumulate(state[h][c], k_hi_t, w_hi[p][h], state[h][c]);
                    simdgroup_multiply_accumulate(state[h][c], k_hi_t, w_lo[p][h], state[h][c]);
                    simdgroup_multiply_accumulate(state[h][c], k_lo_t, w_hi[p][h], state[h][c]);
                }
            }
        }
        }

        if (piece + 1 == pieces_before) {
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
                    published.put_tile(8 * h * state_stride + 8 * c, state[h][c], state_stride, true);
                }
            }
        } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // The rows after the stop row advance row-sequentially from the published
    // state (their innovations feed the tape).
    if (geometry.stop < geometry.hi - geometry.lo) {
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        RCH_SEQUENTIAL(geometry.lo + geometry.stop);
    }
#undef RCH_SEQUENTIAL
}
