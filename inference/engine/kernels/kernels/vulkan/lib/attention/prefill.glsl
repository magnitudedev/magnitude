// The body of the prefill attention entries (`attention_prefill` over dense
// history, `attention_prefill_k8v4` over affine K8/V4 history). The Vulkan
// form of `metal/attention_prefill.metal`: the same three
// launches and partition contract; an entry passes its history buffers as an
// `attention_history` (lib/attention/history.glsl), which is all that differs.
//
// L1 (prepare): one subgroup per (row, query or kv head), rows padded to whole
// QT tiles, prepares the query or the key (rounded to A) and stores it to
// scratch as f16, the attention operand type; the value is copied beside the
// key, and K/V are appended to the history at the row's destination (dense in
// A, or encoded). Queries sit at [KV][rows][G][W], so the QT x G matrix rows
// of one (tile, kv head) are consecutive rows of stride W.
//
// L2 (attend): workgroup (QT-row tile, kv head, key partition); its QT * G
// matrix rows (token-major, head-minor) split into 16-row blocks, one per
// subgroup. The tile's key tiles (each span's union interval in 32-key steps,
// spans then fresh) split into consecutive runs of at least 16 tiles over
// the partitions. Each partition makes one online-softmax pass of
// `lib/attention/flash.glsl` (scores, exp2(s - running max), rescaled P V) over its key tiles, K and V staged
// as f16 (history converted from A or decoded straight to f16, fresh rows
// from scratch). A tile served by one partition stores its gated output
// directly; otherwise each partition stores (partial output, maximum,
// denominator) and L3 merges them.
//
// L3 (merge): workgroup (QT-row tile, query head), one invocation per column.
//
// A head wider than FLASH_MAX_W stages its K tile in column chunks of
// FLASH_MAX_W, scored in turn, and its V tile per output window.
//
// Shared bytes: the K/V tile (32 x (min(W, FLASH_MAX_W) + 8) f16), then 2 KiB
// per subgroup.
//
// Decode rows (PREFILL_DECODE, the decode entries' grouped-query matrix form,
// PREFILL_ROWS = 16 * SIMDS): L2 runs as workgroup (kv head, partition,
// QT-row tile) of the decode's partial launch. Unless L1 ran
// (PREFILL_PREPARED, the entry's choice for a query group whose prepared
// queries exceed the shared bytes), it prepares its own query tile into
// shared memory after the subgroup scratch ([ROWS][W + 8] f16) and its fresh
// K/V tiles (prepared and rounded as L1 would), partition 0 appending the
// tile's rows. The key tiles, then with an index-list source (PREFILL_PICKS)
// each row's pick tiles in row order, split into equal runs over
// PREFILL_PARTS partitions, and every partition, an empty one too
// (denominator 0), stores its partial in the decode layout
// ((row * KV * G + head) * PARTS + part) for the decode merge.
#include "history.glsl"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#error "prefill attention stages 2-byte activations"
#endif

#define PREFILL_KEYS FLASH_KEYS
// A WorkList prefill (the entry's PREFILL_SELECTED: `indexer_select`'s
// per-row block bitmap, PREFILL_SELECTION_WORDS words a row, the
// PREFILL_BLOCK_ROWS keys of a block) walks the key tiles its list launch
// (`prefill_list`) listed for its query tile and masks every key per row.
// The entry names the bitmap and the list launch's scratch: each row's key
// ordinal at the start of each of its spans ([M][R + 1] i32), each query
// tile's listed (span, first) pairs ([tiles][PREFILL_LIST_CAPACITY]) and
// their counts.
#ifndef PREFILL_SELECTED
#define PREFILL_SELECTED false
#endif
#ifdef PREFILL_DECODE
#define PREFILL_SELECTED_PASS false
#else
#define PREFILL_SELECTED_PASS PREFILL_SELECTED
#endif
// An index-list source (the entry's PREFILL_PICKS: its KS > 0). For prefill
// rows it is one more softmax state per (row, query head) after the key
// partitions' (`prefill_picks`, a subgroup over the row's picks), which the
// merge folds in, so every tile publishes partials then. For decode rows its
// picks are key tiles after the row's visible ones: the entry names the
// row's PREFILL_PICK_COUNT picks (PREFILL_PICK_ROWS, [M][picks] i32, -1 for
// none) into the PREFILL_PICK_PLANE slab table ([rows][KV][W] A,
// PREFILL_PICK_SLAB_ROWS rows a slab), each read as key and value.
#ifndef PREFILL_PICKS
#define PREFILL_PICKS false
#endif
#ifndef PREFILL_PICK_COUNT
#define PREFILL_PICK_COUNT 0u
#define PREFILL_PICK_ROWS 0ul
#define PREFILL_PICK_PLANE 0ul
#define PREFILL_PICK_SLAB_ROWS 1ul
#endif
// Whether L1 (`prefill_prepare`) put the queries and fresh rows in scratch:
// always for prefill rows; for decode rows as the entry defines it.
#ifndef PREFILL_DECODE
#define PREFILL_PREPARED 1
#elif !defined(PREFILL_PREPARED)
#define PREFILL_PREPARED 0
#endif
// PREFILL_ROWS, the matrix rows of a workgroup, is the entry's (and with
// PREFILL_DECODE, PREFILL_PARTS its partitions).

// The prefill entries' scratch planes ([KV][rows][G][W], [M][KV][W],
// [M][KV][W] f16 and the tile partition counts), named by the entry: the
// entry ABI is not this file's. Decode rows use the first three where L1
// runs (PREFILL_PREPARED), and no counts.
struct prefill_scratch {
    uint64_t queries;
    uint64_t keys;
    uint64_t values;
    uint64_t counts;
};

// The index-list source's state (PREFILL_PICKS), named by the entry:
// [M][KV * G][W] and [M][KV * G][2] F32 after the key partitions' partials
// and statistics; zero without picks.
struct prefill_picked {
    uint64_t partials;
    uint64_t statistics;
};
#define PREFILL_QT (PREFILL_ROWS / ATTENTION_G)
#ifdef PREFILL_DECODE
#define PREFILL_MIN_TILES 1u
#else
#define PREFILL_MIN_TILES 16u
#endif
// Output columns per pass over the keys: the whole head up to FLASH_MAX_W, so
// every key tile's scores and history decode happen once, where the device
// compiles wide accumulator arrays (`flash.glsl`).
#define PREFILL_WINDOW FLASH_WINDOW

// ---------------------------------------------------------------------------
// L1.

// The lane's E columns of batch row `row`'s prepared key (rounded by the
// entry's key rounding and to A, as history stores it) or value for
// `kv_head`, one subgroup. A stored-key value is the prepared key.
void prefill_fresh_row(const bool is_key, uint64_t row, uint kv_head, uint lane, out float x[ATTENTION_E]) {
    const uint64_t source = (row * ATTENTION_KV + kv_head) * ATTENTION_W * uint64_t(ELEMENT_BYTES(ELEMENT_ACT));
    const float epsilon = element_word_f32(SEISMIC_PARAM_EPSILON);
    if (is_key || ATTENTION_STORED_KEY) {
        attention_prepare(SEISMIC_PTR(SEISMIC_BUFFER_KEY) + source, ATTENTION_NORM,
            SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM), SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES) + row * 16ul,
            SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS), SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES),
            ATTENTION_AMPLITUDE_BUFFER, epsilon, lane, ATTENTION_KEY_ROUNDING, x);
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            x[i] = element_round(ELEMENT_ACT, x[i]);
    } else {
        attention_value(SEISMIC_PTR(SEISMIC_BUFFER_VALUE) + source, ATTENTION_VALUE_NORM_BUFFER, epsilon, lane, x);
    }
}

void prefill_prepare(attention_history h, prefill_scratch scratch_planes) {
    const uint W = ATTENTION_W, E = ATTENTION_E, KV = ATTENTION_KV, G = ATTENTION_G;
    const uint lane = SEISMIC_LANE;
    const uint64_t rows = SEISMIC_DIM_M;
    const uint64_t padded = (rows + PREFILL_QT - 1ul) / PREFILL_QT * PREFILL_QT;
    const uint64_t item = uint64_t(gl_WorkGroupID.x) * SEISMIC_SUBGROUPS + SEISMIC_SUBGROUP;
    const uint64_t row = item / (KV * (G + 1u));
    const uint head = uint(item % (KV * (G + 1u)));
    if (row >= padded)
        return;
    const uint64_t queries = scratch_planes.queries;
    if (head < KV * G) {
        const uint kv_head = head / G, g = head % G;
        const uint64_t at = ((uint64_t(kv_head) * padded + row) * G + g) * W + lane * E;
        float x[ATTENTION_E];
        if (row < rows) {
            attention_prepare(ATTENTION_QUERY_BUFFER + ((row * KV * G + head) * ATTENTION_QUERY_STRIDE) * 2ul,
                ATTENTION_QUERY_NORM, SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM),
                SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES) + row * 16ul,
                SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS), SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES),
                ATTENTION_AMPLITUDE_BUFFER, element_word_f32(SEISMIC_PARAM_EPSILON), lane, rounding_none(), x);
        } else {
            [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                x[i] = 0.0;
        }
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            element_put(ELEMENT_F16, queries, at + i, element_round(ELEMENT_ACT, x[i]));
        return;
    }
    if (row >= rows || !ATTENTION_FRESH)
        return;
    const uint kv_head = head - KV * G;
    const uint64_t source = (row * KV + kv_head) * W;
    float k[ATTENTION_E], v[ATTENTION_E];
    prefill_fresh_row(true, row, kv_head, lane, k);
    // A stored-key value is the prepared key.
    if (ATTENTION_STORED_KEY) {
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            v[i] = k[i];
    } else {
        prefill_fresh_row(false, row, kv_head, lane, v);
    }
    const uint64_t keys = scratch_planes.keys;
    const uint64_t values = scratch_planes.values;
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
        element_put(ELEMENT_F16, keys, source + lane * E + i, k[i]);
        element_put(ELEMENT_F16, values, source + lane * E + i, v[i]);
    }
    const int destination = element_i32_at(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATIONS) + row * 4ul);
    if (destination < 0)
        return;
    history_append(h, true, destination, kv_head, lane, k);
    history_append(h, false, destination, kv_head, lane, v);
}

// ---------------------------------------------------------------------------
// L2.

// Columns [column, column + w) of the lane's E columns of a row, as f16, into
// row `row` of the tile at shared half `base` (row pitch `pitch`).
void prefill_put_row(uint base, uint pitch, uint row, uint column, const uint w, uint lane, float x[ATTENTION_E]) {
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
        const uint c = lane * ATTENTION_E + i;
        if (c >= column && c < column + w)
            seismic_shared_u16[base + row * pitch + c - column] = seismic_f32_to_f16(x[i]);
    }
}

// Stages columns [column, column + w) of rows [first, first + 32) of one kv
// head as the f16 tile at shared half 0 (row pitch w + 8): history (dense or
// decoded) or the batch's fresh rows (from L1's f16 scratch, or for decode
// rows prepared here, a subgroup per row).
void prefill_stage(attention_history h, prefill_scratch scratch_planes, const bool is_key, bool historical, int first,
    int end, uint kv_head, uint column, const uint w) {
    if (historical) {
        history_stage(h, is_key, first, end, kv_head, column, w, 0u);
    } else {
#if !PREFILL_PREPARED
        for (uint r = SEISMIC_SUBGROUP; r < FLASH_KEYS; r += SEISMIC_SUBGROUPS) {
            const int t = first + int(r);
            float x[ATTENTION_E];
            if (t < end) {
                prefill_fresh_row(is_key, uint64_t(t), kv_head, SEISMIC_LANE, x);
            } else {
                [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                    x[i] = 0.0;
            }
            prefill_put_row(0u, flash_pitch(w), r, column, w, SEISMIC_LANE, x);
        }
#else
        const uint64_t plane =
            is_key ? scratch_planes.keys : scratch_planes.values;
        flash_stage(ELEMENT_F16, plane, uint64_t(ATTENTION_KV * ATTENTION_W), uint64_t(kv_head * ATTENTION_W + column),
            first, end, w, 0u);
#endif
    }
}

#ifdef PREFILL_DECODE
// Row `row`'s pick `pick`: its row of the pick plane, or -1 for none (also
// past the row's picks).
int prefill_pick(uint64_t row, uint pick) {
    return pick < PREFILL_PICK_COUNT
        ? element_i32_at(PREFILL_PICK_ROWS + (row * uint64_t(PREFILL_PICK_COUNT) + uint64_t(pick)) * 4ul)
        : -1;
}

// Stages columns [column, column + w) of row `row`'s picks [pick0, pick0 +
// 32) of one kv head as the f16 tile at shared half 0 (row pitch w + 8),
// gathered from the pick plane; a pick of -1 stages zeros.
void prefill_stage_picks(uint64_t row, uint pick0, uint kv_head, uint column, const uint w) {
    const uint pieces = w / 8u;
    const uint64_t head_bytes = uint64_t(ATTENTION_W) * ELEMENT_BYTES(ELEMENT_ACT);
    for (uint item = gl_LocalInvocationIndex; item < FLASH_KEYS * pieces; item += gl_WorkGroupSize.x) {
        const uint k = item / pieces;
        const uint c = (item % pieces) * 8u;
        const int index = prefill_pick(row, pick0 + k);
        uvec4 bits = uvec4(0u);
        if (index >= 0) {
            bits = element_uvec4_at(history_address(PREFILL_PICK_PLANE, uint64_t(index), PREFILL_PICK_SLAB_ROWS,
                uint64_t(ATTENTION_KV) * head_bytes, head_bytes, kv_head) + uint64_t(column + c) * ELEMENT_BYTES(ELEMENT_ACT));
            if (ELEMENT_ACT == ELEMENT_BF16) {
                vec4 even, odd;
                element_split8(ELEMENT_BF16, bits, even, odd);
                bits = element_pack8(ELEMENT_F16, even, odd);
            }
        }
        seismic_shared_uvec4[(k * flash_pitch(w) + c) / 8u] = bits;
    }
}
#endif

// The union [lo, hi) of the tile's non-empty row intervals and the
// intersection [common_lo, common_hi) of all its row intervals, for span
// `index`. Every subgroup computes the same four bounds from the tile rows.
ivec4 prefill_interval(uint64_t visible, uint64_t fresh, uint64_t tile_first, uint64_t rows, uint64_t spans, uint64_t index) {
    int union_lo = 0x7fffffff, union_hi = int(0x80000000);
    int common_lo = int(0x80000000), common_hi = 0x7fffffff;
    for (uint i = 0u; i < PREFILL_QT; ++i) {
        const uint64_t row = tile_first + i;
        if (row >= rows)
            break;
        int lo, hi;
        attention_span(visible, fresh, row, spans, index, lo, hi);
        if (hi > lo) {
            union_lo = min(union_lo, lo);
            union_hi = max(union_hi, hi);
        }
        common_lo = max(common_lo, lo);
        common_hi = min(common_hi, hi);
    }
    return ivec4(union_lo, union_hi, common_lo, common_hi);
}

uint prefill_span_tiles(ivec4 interval) {
    return interval.y > interval.x ? uint(interval.y - interval.x + int(PREFILL_KEYS) - 1) / PREFILL_KEYS : 0u;
}

// The key tiles of span `index` (none past the fresh span).
uint prefill_index_tiles(uint64_t visible, uint64_t fresh, uint64_t tile_first, uint64_t rows, uint64_t spans,
    uint64_t index) {
    return index > spans ? 0u : prefill_span_tiles(prefill_interval(visible, fresh, tile_first, rows, spans, index));
}

// The query tile's key tiles over all its spans. Lane l counts spans l, l +
// 32, ...; the whole subgroup calls it.
uint prefill_total_tiles(uint64_t visible, uint64_t fresh, uint64_t tile_first, uint64_t rows, uint64_t spans) {
    uint total = 0u;
    for (uint64_t index = SEISMIC_LANE; index <= spans; index += 32ul)
        total += prefill_index_tiles(visible, fresh, tile_first, rows, spans, index);
    return seismic_subgroup_add(total);
}

// The span holding the query tile's key tile `tile` (in span order) and the
// tiles before it: the first span in 0..=spans whose tiles end past `tile`,
// or (spans + 1, total) when there is none. The whole subgroup calls it, 32
// spans at a time.
uvec2 prefill_locate_tile(uint64_t visible, uint64_t fresh, uint64_t tile_first, uint64_t rows, uint64_t spans,
    uint tile) {
    uint carried = 0u;
    for (uint64_t base = 0ul; base <= spans; base += 32ul) {
        uint found;
        if (reduce_lanes_locate(prefill_index_tiles(visible, fresh, tile_first, rows, spans, base + SEISMIC_LANE),
                tile, carried, found))
            return uvec2(uint(base) + found, carried);
    }
    return uvec2(uint(spans) + 1u, carried);
}

// The lane's 16 scaled, masked scores of the tile starting at key `first`;
// a key outside the row's span bounds scores -inf (bounds are read only
// outside the common interval).
void prefill_lane_scores(uint scratch, float scale, int first, bool inside, int row_lo, int row_hi, out float s[16]) {
    flash_lane_scores(scratch, s);
    const uint h = SEISMIC_LANE / 16u;
    [[unroll]] for (uint j = 0u; j < 16u; ++j) {
        s[j] *= scale;
        const int t = first + int(16u * h + j);
        if (!inside && !(t >= row_lo && t < row_hi))
            s[j] = -ATTENTION_INF;
    }
}

#if !defined(PREFILL_DECODE)
// Whether row `row` reads key `key` of span `index` (bounds from `lo`) under
// the WorkList: the bit of the key's block; a block past the bitmap is not
// read.
// `base` is the row's key ordinal at the span's start.
bool prefill_reads(uint64_t row, int base, int key, int lo) {
    if (!PREFILL_SELECTED)
        return true;
    const int ordinal = base + key - lo;
    const uint block = uint(ordinal) / uint(PREFILL_BLOCK_ROWS);
    if (block / 32u >= uint(PREFILL_SELECTION_WORDS))
        return false;
    const uint word = element_u32_at(PREFILL_SELECTION_BUFFER
        + (row * uint64_t(PREFILL_SELECTION_WORDS) + uint64_t(block / 32u)) * 4ul);
    return ((word >> (block % 32u)) & 1u) != 0u;
}

// The list launch (PREFILL_SELECTED): workgroup per QT-row query tile, 256
// invocations. It writes each tile row's span ordinals, then lists, in L2's
// tile order (each span's union interval in PREFILL_KEYS-key tiles, spans
// then fresh), the key tiles some row of the tile reads. Shared u32: a flag
// per invocation, then the running count.
void prefill_list() {
    const uint64_t visible = SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE);
    const uint64_t fresh = SEISMIC_PTR(SEISMIC_BUFFER_FRESH);
    const uint64_t rows = SEISMIC_DIM_M;
    const uint64_t spans = SEISMIC_DIM_R;
    const uint tile = gl_WorkGroupID.x;
    const uint thread = gl_LocalInvocationIndex;
    const uint threads = gl_WorkGroupSize.x;
    const uint64_t tile_first = uint64_t(tile) * PREFILL_QT;
    if (thread < PREFILL_QT && tile_first + thread < rows) {
        const uint64_t row = tile_first + thread;
        int base = 0;
        for (uint64_t index = 0ul; index <= spans; ++index) {
            int lo, hi;
            attention_span(visible, fresh, row, spans, index, lo, hi);
            element_i32_put(PREFILL_BASES_BUFFER + (row * (spans + 1ul) + index) * 4ul, base);
            base += max(hi - lo, 0);
        }
    }
    if (thread == 0u)
        seismic_shared_u32[threads] = 0u;
    barrier();
    const uint total = prefill_total_tiles(visible, fresh, tile_first, rows, spans);
    for (uint first_candidate = 0u; first_candidate < total; first_candidate += threads) {
        const uint candidate = first_candidate + thread;
        bool read = false;
        int entry_span = 0, entry_first = 0;
        if (candidate < total) {
            uint before = 0u;
            ivec4 interval = ivec4(0);
            for (uint64_t index = 0ul; index <= spans; ++index) {
                interval = prefill_interval(visible, fresh, tile_first, rows, spans, index);
                const uint n = prefill_span_tiles(interval);
                if (candidate < before + n) {
                    entry_span = int(index);
                    entry_first = interval.x + int((candidate - before) * PREFILL_KEYS);
                    break;
                }
                before += n;
            }
            const int end = min(entry_first + int(PREFILL_KEYS), interval.y);
            for (uint i = 0u; i < PREFILL_QT && !read; ++i) {
                const uint64_t row = tile_first + i;
                if (row >= rows)
                    break;
                // The row's ordinal at the span's start (the bases this
                // launch writes are read by the attend launch only).
                int base = 0;
                int lo = 0, hi = 0;
                for (uint64_t index = 0ul; index < uint64_t(entry_span); ++index) {
                    attention_span(visible, fresh, row, spans, index, lo, hi);
                    base += max(hi - lo, 0);
                }
                attention_span(visible, fresh, row, spans, uint64_t(entry_span), lo, hi);
                const int first_key = max(entry_first, lo);
                const int last_key = min(end, hi);
                if (first_key >= last_key)
                    continue;
                for (int key = first_key; key < last_key && !read; ++key)
                    read = prefill_reads(row, base, key, lo);
            }
        }
        seismic_shared_u32[thread] = read ? 1u : 0u;
        barrier();
        uint offset = seismic_shared_u32[threads];
        for (uint other = 0u; other < thread; ++other)
            offset += seismic_shared_u32[other];
        if (read && offset < PREFILL_LIST_CAPACITY) {
            const uint64_t slot = PREFILL_LISTS_BUFFER + ((uint64_t(tile) * PREFILL_LIST_CAPACITY + offset) * 2ul) * 4ul;
            element_i32_put(slot, entry_span);
            element_i32_put(slot + 4ul, entry_first);
        }
        barrier();
        if (thread == 0u) {
            uint added = 0u;
            for (uint other = 0u; other < threads; ++other)
                added += seismic_shared_u32[other];
            seismic_shared_u32[threads] += added;
        }
        barrier();
    }
    if (thread == 0u)
        element_u32_put(PREFILL_LISTED_BUFFER + uint64_t(tile) * 4ul, min(seismic_shared_u32[threads], PREFILL_LIST_CAPACITY));
}
#endif

// Adds the block's scores of columns [column, column + w) of the staged K
// tile: from L1's scratch rows (`scratch_rows`) or the workgroup's query tile
// (`tile_half`).
void prefill_scores_add(uint64_t scratch_rows, uint tile_half, uint column, const uint w, inout flash_scores_state scores) {
#if PREFILL_PREPARED
    flash_scores_add(scratch_rows + uint64_t(column) * 2ul, uint64_t(ATTENTION_W), w, 0u, scores);
#else
    flash_scores_add_shared(tile_half + column, ATTENTION_W + 8u, w, 0u, scores);
#endif
}

// Where a key tile's rows come from: history, the batch's fresh rows, or (a
// decode row's) picks.
#define PREFILL_FROM_HISTORY 0u
#define PREFILL_FROM_FRESH 1u
#define PREFILL_FROM_PICKS 2u

// Stages columns [column, column + w) of the key or value tile of `source`
// whose first key is `first` (a pick tile: its first pick of row `row`).
void prefill_stage_from(attention_history h, prefill_scratch scratch_planes, const bool is_key, uint source, int first,
    int end, uint64_t row, uint kv_head, uint column, const uint w) {
#ifdef PREFILL_DECODE
    if (source == PREFILL_FROM_PICKS) {
        prefill_stage_picks(row, uint(first), kv_head, column, w);
        return;
    }
#endif
    prefill_stage(h, scratch_planes, is_key, source == PREFILL_FROM_HISTORY, first, end, kv_head, column, w);
}

// One key tile of `source` from key `first` (rows before `end`; a pick
// tile: from pick `first` of row `pick_row`): the block's scores, each row's
// keys outside [row_lo, row_hi) masked unless the tile lies in every row's
// interval (`inside`), and a pick tile's keys masked for every row but its
// own (the lane's row is `token`) and where a pick is -1; the online
// softmax; O += P V over the pass's output window. The whole workgroup calls
// it.
void prefill_absorb(attention_history h, prefill_scratch scratch_planes, uint source, int first, int end, bool inside,
    int row_lo, int row_hi, uint64_t pick_row, uint64_t token, uint kv_head, const uint chunk, uint column0,
    const uint width,
    bool computes, uint64_t block_queries, uint shared_queries, uint scratch, float scale,
    inout flash_softmax softmax, inout flash_output o) {
    flash_scores_state scores;
    flash_scores_clear(scores);
    for (uint column = 0u; column < ATTENTION_W; column += chunk) {
        barrier();
        prefill_stage_from(h, scratch_planes, true, source, first, end, pick_row, kv_head, column, chunk);
        barrier();
        if (computes)
            prefill_scores_add(block_queries, shared_queries, column, chunk, scores);
    }
    flash_scores_publish(scores, scratch);
    float s[16];
    prefill_lane_scores(scratch, scale, first, inside, row_lo, row_hi, s);
#ifdef PREFILL_DECODE
    if (source == PREFILL_FROM_PICKS) {
        const uint lane_half = SEISMIC_LANE / 16u;
        [[unroll]] for (uint j = 0u; j < 16u; ++j)
            if (token != pick_row || prefill_pick(pick_row, uint(first) + 16u * lane_half + j) < 0)
                s[j] = -ATTENTION_INF;
    }
#endif
    const float alpha = flash_online(softmax, s);
    const uint p_half = flash_publish_probabilities(scratch, s);
    flash_rescale(scratch, alpha, ATTENTION_W, PREFILL_WINDOW, o);
    barrier();
    prefill_stage_from(h, scratch_planes, false, source, first, end, pick_row, kv_head, column0, width);
    barrier();
    if (computes)
        flash_accumulate(p_half, 0u, width, width, 0u, o);
}

void prefill_attend(attention_history h, prefill_scratch scratch_planes) {
    const uint W = ATTENTION_W, KV = ATTENTION_KV, G = ATTENTION_G;
    const uint64_t visible = SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE);
    const uint64_t fresh = SEISMIC_PTR(SEISMIC_BUFFER_FRESH);
    const uint64_t rows = SEISMIC_DIM_M;
    const uint64_t spans = SEISMIC_DIM_R;
    const uint64_t padded = (rows + PREFILL_QT - 1ul) / PREFILL_QT * PREFILL_QT;
    const float scale = element_word_f32(SEISMIC_PARAM_SCALE) * ATTENTION_LOG2E;
    const uint lane = SEISMIC_LANE;
#ifdef PREFILL_DECODE
    const uint kv_head = gl_WorkGroupID.x;
    const uint part = gl_WorkGroupID.y;
    const uint tile = gl_WorkGroupID.z;
    const uint parts = PREFILL_PARTS;
#else
    // Query tiles dispatch last-first: in a causal chunk the last tiles see
    // the most keys, and starting them first shortens the grid's tail.
    const uint tile = gl_NumWorkGroups.x - 1u - gl_WorkGroupID.x;
    const uint kv_head = gl_WorkGroupID.y;
    const uint part = gl_WorkGroupID.z;
    // The grid's key partitions: max(1, ceil(SPLIT_GROUPS / (tiles * KV))).
    const uint parts = gl_NumWorkGroups.z;
#endif
    const uint64_t tile_first = uint64_t(tile) * PREFILL_QT;
    // K stages in chunks of at most FLASH_MAX_W columns, scored in turn; V
    // stages the pass's output window.
    const uint chunk = min(W, FLASH_MAX_W);
    const uint width = flash_window(W, PREFILL_WINDOW);
    const uint scratch = (FLASH_KEYS * flash_pitch(chunk)) / 2u + SEISMIC_SUBGROUP * FLASH_SCRATCH_FLOATS;
#if !PREFILL_PREPARED
    // The query tile [ROWS][W + 8] f16, after every subgroup's scratch.
    const uint queries = FLASH_KEYS * flash_pitch(chunk) + 2u * SEISMIC_SUBGROUPS * FLASH_SCRATCH_FLOATS;
    for (uint i = SEISMIC_SUBGROUP; i < PREFILL_ROWS; i += SEISMIC_SUBGROUPS) {
        const uint64_t token = tile_first + i / G;
        float x[ATTENTION_E];
        if (i < PREFILL_QT * G && token < rows) {
            attention_prepare(ATTENTION_QUERY_BUFFER + ((token * KV * G + kv_head * G + i % G) * ATTENTION_QUERY_STRIDE)
                    * uint64_t(ELEMENT_BYTES(ELEMENT_ACT)), ATTENTION_QUERY_NORM,
                SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM), SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES) + token * 16ul,
                SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS), SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES),
                ATTENTION_AMPLITUDE_BUFFER, element_word_f32(SEISMIC_PARAM_EPSILON), lane, rounding_none(), x);
            [[unroll]] for (uint j = 0u; j < ATTENTION_E; ++j)
                x[j] = element_round(ELEMENT_ACT, x[j]);
        } else {
            [[unroll]] for (uint j = 0u; j < ATTENTION_E; ++j)
                x[j] = 0.0;
        }
        prefill_put_row(queries, W + 8u, i, 0u, W, lane, x);
    }
    if (ATTENTION_FRESH && part == 0u) {
        for (uint64_t token = tile_first + SEISMIC_SUBGROUP; token < min(tile_first + PREFILL_QT, rows);
             token += SEISMIC_SUBGROUPS) {
            const int destination = element_i32_at(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATIONS) + token * 4ul);
            if (destination < 0)
                continue;
            float x[ATTENTION_E];
            prefill_fresh_row(true, token, kv_head, lane, x);
            history_append(h, true, destination, kv_head, lane, x);
            prefill_fresh_row(false, token, kv_head, lane, x);
            history_append(h, false, destination, kv_head, lane, x);
        }
    }
    barrier();
#endif

    uint total_tiles = prefill_total_tiles(visible, fresh, tile_first, rows, spans);
    // The tiles of the spans; a decode row's pick tiles follow them.
    const uint visible_tiles = total_tiles;
#ifdef PREFILL_DECODE
    // Each row of the tile has its own picks, in tiles of PREFILL_KEYS.
    const uint pick_tiles = (PREFILL_PICK_COUNT + PREFILL_KEYS - 1u) / PREFILL_KEYS;
    if (PREFILL_PICKS)
        total_tiles += PREFILL_QT * pick_tiles;
#else
    // A WorkList's key tiles are its query tile's listed ones.
    if (PREFILL_SELECTED)
        total_tiles = element_u32_at(PREFILL_LISTED_BUFFER + uint64_t(tile) * 4ul);
#endif
    const uint per = max(PREFILL_MIN_TILES, (total_tiles + parts - 1u) / parts);
    const uint used = max(1u, (total_tiles + per - 1u) / per);
    if (part >= used) {
#ifdef PREFILL_DECODE
        // An empty partition's (maximum, denominator 0) per row.
        for (uint i = gl_LocalInvocationIndex; i < PREFILL_QT * G; i += gl_WorkGroupSize.x) {
            const uint64_t token = tile_first + i / G;
            if (token >= rows)
                continue;
            const uint64_t slot = (token * KV * G + kv_head * G + i % G) * PREFILL_PARTS + part;
            element_f32_put(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS) + slot * 8ul, -ATTENTION_INF);
            element_f32_put(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS) + slot * 8ul + 4ul, 0.0);
        }
#endif
        return;
    }
#ifndef PREFILL_DECODE
    if (part == 0u && kv_head == 0u && gl_LocalInvocationIndex == 0u)
        element_u32_put(scratch_planes.counts + uint64_t(tile) * 4ul, used);
#endif
    const uint tiles_lo = part * per;
    const uint tiles_hi = min(tiles_lo + per, total_tiles);
    // The partition's walk over the spans starts on the span holding its
    // first tile and ends past its last.
    const uvec2 located = PREFILL_SELECTED_PASS
        ? uvec2(0u)
        : prefill_locate_tile(visible, fresh, tile_first, rows, spans, tiles_lo);

    // The subgroup's block: matrix rows 16 sg .. of the tile's QT x G rows;
    // the lane's softmax row is lane % 16. When G does not divide ROWS, the
    // rows past QT x G are padding: they score whatever query scratch follows
    // (the declaration adds ROWS rows of slack) and store nothing.
    const uint block_row = 16u * SEISMIC_SUBGROUP;
    const uint lane_row = block_row + lane % 16u;
    const uint64_t token = tile_first + lane_row / G;
    const bool valid = lane_row < PREFILL_QT * G && token < rows;
    // A block of padding rows only (the batch's last, partial tile, or with
    // decode rows most blocks) skips the matrix products and stores nothing;
    // it runs every other step.
    const bool computes = block_row < PREFILL_QT * G && tile_first + block_row / G < rows;
    // The block's queries: L1's scratch rows, or the workgroup's query tile.
#if PREFILL_PREPARED
    const uint64_t block_queries = scratch_planes.queries
        + ((uint64_t(kv_head) * padded + tile_first) * G + block_row) * W * 2ul;
    const uint shared_queries = 0u;
#else
    const uint64_t block_queries = 0ul;
    const uint shared_queries = queries + block_row * (W + 8u);
#endif

    const uint64_t heads = uint64_t(KV * G);
    const uint64_t query = ATTENTION_QUERY_BUFFER;
    const uint64_t gated = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const uint64_t partials = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS);
    const uint64_t statistics = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS);
    flash_output o;

    // One online-softmax pass per output window of PREFILL_WINDOW columns.
    const uint windows = (W + PREFILL_WINDOW - 1u) / PREFILL_WINDOW;
    for (uint pass = 0u; pass < windows; ++pass) {
        const uint column0 = pass * PREFILL_WINDOW;
        flash_output_clear(W, PREFILL_WINDOW, o);
        flash_softmax softmax = flash_softmax_start();
#ifndef PREFILL_DECODE
        // A WorkList walks its listed key tiles, every key masked per row.
        for (uint listed = tiles_lo; PREFILL_SELECTED && listed < tiles_hi; ++listed) {
            const uint64_t slot = PREFILL_LISTS_BUFFER + ((uint64_t(tile) * PREFILL_LIST_CAPACITY + listed) * 2ul) * 4ul;
            const uint64_t index = uint64_t(element_i32_at(slot));
            const int first = element_i32_at(slot + 4ul);
            const ivec4 interval = prefill_interval(visible, fresh, tile_first, rows, spans, index);
            const bool historical = index < spans;
            int row_lo = 0, row_hi = 0, row_base = 0;
            if (valid) {
                attention_span(visible, fresh, token, spans, index, row_lo, row_hi);
                row_base = element_i32_at(PREFILL_BASES_BUFFER + (token * (spans + 1ul) + index) * 4ul);
            }
            flash_scores_state scores;
            flash_scores_clear(scores);
            for (uint column = 0u; column < W; column += chunk) {
                barrier();
                prefill_stage(h, scratch_planes, true, historical, first, interval.y, kv_head, column, chunk);
                barrier();
                if (computes)
                    prefill_scores_add(block_queries, shared_queries, column, chunk, scores);
            }
            flash_scores_publish(scores, scratch);
            float s[16];
            prefill_lane_scores(scratch, scale, first, false, row_lo, row_hi, s);
            const uint lane_half = SEISMIC_LANE / 16u;
            [[unroll]] for (uint j = 0u; j < 16u; ++j) {
                const int t = first + int(16u * lane_half + j);
                if (valid && t >= row_lo && t < row_hi && !prefill_reads(token, row_base, t, row_lo))
                    s[j] = -ATTENTION_INF;
            }
            const float alpha = flash_online(softmax, s);
            const uint p_half = flash_publish_probabilities(scratch, s);
            flash_rescale(scratch, alpha, W, PREFILL_WINDOW, o);
            barrier();
            prefill_stage(h, scratch_planes, false, historical, first, interval.y, kv_head, column0, width);
            barrier();
            if (computes)
                flash_accumulate(p_half, 0u, width, width, 0u, o);
        }
#endif
        uint tiles_before = located.y;
        for (uint64_t index = located.x; index <= spans && tiles_before < tiles_hi && !PREFILL_SELECTED_PASS;
             ++index) {
            const ivec4 interval = prefill_interval(visible, fresh, tile_first, rows, spans, index);
            const uint span_tiles = prefill_span_tiles(interval);
            const uint span_first = tiles_before;
            tiles_before += span_tiles;
            if (span_tiles == 0u || span_first + span_tiles <= tiles_lo || span_first >= tiles_hi)
                continue;
            const uint own_lo = max(tiles_lo, span_first) - span_first;
            const uint own_hi = min(tiles_hi, span_first + span_tiles) - span_first;
            const bool historical = index < spans;
            int row_lo = 0, row_hi = 0;
            if (valid)
                attention_span(visible, fresh, token, spans, index, row_lo, row_hi);
            for (uint own = own_lo; own < own_hi; ++own) {
                const int first = interval.x + int(own * PREFILL_KEYS);
                const bool inside = first >= interval.z && first + int(PREFILL_KEYS) <= interval.w;
                prefill_absorb(h, scratch_planes, historical ? PREFILL_FROM_HISTORY : PREFILL_FROM_FRESH, first,
                    interval.y, inside, row_lo, row_hi, 0ul, token, kv_head, chunk, column0, width, computes,
                    block_queries, shared_queries, scratch, scale, softmax, o);
            }
        }
#ifdef PREFILL_DECODE
        // The tile rows' pick tiles, after the visible tiles: each attended
        // by its own row alone.
        for (uint picked = max(tiles_lo, visible_tiles); PREFILL_PICKS && picked < tiles_hi; ++picked) {
            const uint ordinal = picked - visible_tiles;
            const uint64_t pick_row = tile_first + ordinal / pick_tiles;
            if (pick_row >= rows)
                break;
            prefill_absorb(h, scratch_planes, PREFILL_FROM_PICKS, int((ordinal % pick_tiles) * PREFILL_KEYS), 0, true,
                0, 0, pick_row, token, kv_head, chunk, column0, width, computes, block_queries, shared_queries,
                scratch, scale, softmax, o);
        }
#endif
        const float maximum = softmax.maximum;
        const float denominator = flash_denominator(softmax);
        // Every subgroup of a used partition reaches here; the scratch is
        // free.
        barrier();
        // Unrolled, so every fragment index is a constant.
        [[unroll]] for (uint q = 0u; q < PREFILL_WINDOW / 2u; ++q) {
            if (q >= flash_window(W, PREFILL_WINDOW) / 2u)
                break;
            const float value = flash_output_value(o, scratch, q);
            const uint r = block_row + flash_output_row(q);
            const uint column = flash_output_column(W, PREFILL_WINDOW, column0, q);
            const uint64_t out_token = tile_first + r / G;
            const uint64_t out_head = kv_head * G + r % G;
            // The row's softmax state lives on lanes r % 16 and r % 16 + 16.
            const float row_maximum = seismic_shuffle(maximum, r % 16u);
            const float row_denominator = seismic_shuffle(denominator, r % 16u);
            if (r < PREFILL_QT * G && out_token < rows) {
#ifdef PREFILL_DECODE
                const uint64_t slot = (out_token * heads + out_head) * PREFILL_PARTS + part;
                element_f32_put(partials + (slot * W + column) * 4ul, value);
                if (column == 0u) {
                    element_f32_put(statistics + slot * 8ul, row_maximum);
                    element_f32_put(statistics + slot * 8ul + 4ul, row_denominator);
                }
#else
                if (used > 1u || PREFILL_PICKS) {
                    const uint64_t slot = (uint64_t(part) * rows + out_token) * heads + out_head;
                    element_f32_put(partials + (slot * W + column) * 4ul, value);
                    if (column == 0u) {
                        element_f32_put(statistics + slot * 8ul, row_maximum);
                        element_f32_put(statistics + slot * 8ul + 4ul, row_denominator);
                    }
                } else {
                    attention_store_gated(query, ATTENTION_GATE_BUFFER, gated, out_token, out_head, column,
                        attention_attended(value, row_maximum, row_denominator, out_head));
                }
#endif
            }
        }
    }
}

#ifndef PREFILL_DECODE
// ---------------------------------------------------------------------------
// The index-list source (PREFILL_PICKS), after L1 put the queries:
// workgroup (row, query head), one subgroup. Its softmax state over the row's
// `picks` picks (`selected_rows`, [M][picks] i32, -1 for none; each the kv
// head's vector of the `selected` plane, [rows][KV][W] A in slabs of
// `selected_slab_rows` rows, read as key and value) is `picked`'s.
void prefill_picks(prefill_scratch scratch_planes, prefill_picked picked, uint64_t selected_rows, uint picks,
    uint64_t selected, uint64_t selected_slab_rows) {
    const uint W = ATTENTION_W, E = ATTENTION_E, KV = ATTENTION_KV, G = ATTENTION_G;
    const uint64_t rows = SEISMIC_DIM_M;
    const uint64_t padded = (rows + PREFILL_QT - 1ul) / PREFILL_QT * PREFILL_QT;
    const uint64_t row = gl_WorkGroupID.x;
    const uint head = gl_WorkGroupID.y;
    const uint kv_head = head / G;
    const uint lane = SEISMIC_LANE;
    const float scale = element_word_f32(SEISMIC_PARAM_SCALE) * ATTENTION_LOG2E;
    const uint64_t query = ((uint64_t(kv_head) * padded + row) * G + head % G) * W + lane * E;
    float q[ATTENTION_E], accumulated[ATTENTION_E];
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
        q[i] = element_at(ELEMENT_F16, scratch_planes.queries, query + i);
        accumulated[i] = 0.0;
    }
    const uint64_t head_bytes = uint64_t(W) * ELEMENT_BYTES(ELEMENT_ACT);
    float maximum = -ATTENTION_INF;
    float denominator = 0.0;
    for (uint pick = 0u; pick < picks; ++pick) {
        const int index = element_i32_at(selected_rows + (row * picks + pick) * 4ul);
        if (index < 0)
            continue;
        float x[ATTENTION_E];
        const uint64_t vector = history_address(selected, uint64_t(index), selected_slab_rows,
            uint64_t(KV) * head_bytes, head_bytes, kv_head);
        attention_load_row(vector, uint64_t(lane) * E, x);
        float dot = 0.0;
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            dot = seismic_fma_rn(q[i], x[i], dot);
        const float score = seismic_subgroup_sum_f32(dot) * scale;
        const float next = max(maximum, score);
        const float carry = exp2(maximum - next);
        const float weight = exp2(score - next);
        denominator = seismic_fma_rn(denominator, carry, weight);
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            accumulated[i] = seismic_fma_rn(accumulated[i], carry, weight * x[i]);
        maximum = next;
    }
    const uint64_t state = row * KV * G + head;
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
        element_f32_put(picked.partials + (state * W + lane * E + i) * 4ul, accumulated[i]);
    if (lane == 0u) {
        element_f32_put(picked.statistics + state * 8ul, maximum);
        element_f32_put(picked.statistics + state * 8ul + 4ul, denominator);
    }
}

// ---------------------------------------------------------------------------
// L3: workgroup (QT-row tile, query head), one invocation per column. A tile
// that took several key partitions, or any tile with an index-list source,
// merges each of its rows' partitions in partition order, then the picked
// state, and applies the gate; other tiles were stored by L2.
void prefill_merge(prefill_scratch scratch_planes, prefill_picked picked) {
    const uint64_t heads = uint64_t(ATTENTION_KV * ATTENTION_G);
    const uint tile = gl_WorkGroupID.x;
    const uint64_t head = gl_WorkGroupID.y;
    const uint column = gl_LocalInvocationIndex;
    const uint count = element_u32_at(scratch_planes.counts + uint64_t(tile) * 4ul);
    if (count <= 1u && !PREFILL_PICKS)
        return;
    const uint64_t rows = SEISMIC_DIM_M;
    for (uint64_t row = uint64_t(tile) * PREFILL_QT; row < min(uint64_t(tile + 1u) * PREFILL_QT, rows); ++row) {
        const uint64_t slot = row * heads + head;
        vec3 merged = attention_merge_state(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS),
            SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS), slot, rows * heads, count, column);
        if (PREFILL_PICKS) {
            const float denominator = element_f32_at(picked.statistics + slot * 8ul + 4ul);
            if (denominator > 0.0) {
                const float maximum = element_f32_at(picked.statistics + slot * 8ul);
                const float next = max(merged.x, maximum);
                const float carry = exp2(merged.x - next);
                const float weight = exp2(maximum - next);
                merged = vec3(next, seismic_fma_rn(denominator, weight, merged.y * carry),
                    seismic_fma_rn(element_f32_at(picked.partials + (slot * ATTENTION_W + column) * 4ul), weight,
                        merged.z * carry));
            }
        }
        attention_store_gated(ATTENTION_QUERY_BUFFER, ATTENTION_GATE_BUFFER, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), row,
            head, column, attention_attended(merged.z, merged.x, merged.y, head));
    }
}

#endif

#ifdef PREFILL_DECODE
// The decode merge of the matrix form: workgroup (query head, row), one
// invocation per column; every one of the row's PREFILL_PARTS partitions
// stored a partial (an empty one with denominator 0), merged in partition
// order, then the output gate.
void prefill_decode_merge() {
    const uint64_t head = gl_WorkGroupID.x, row = gl_WorkGroupID.y;
    const uint column = gl_LocalInvocationIndex;
    const float attended = attention_merge(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS),
        SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS),
        (row * ATTENTION_KV * ATTENTION_G + head) * PREFILL_PARTS, 1ul, PREFILL_PARTS, column, head);
    attention_store_gated(ATTENTION_QUERY_BUFFER, ATTENTION_GATE_BUFFER, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), row,
        head, column, attended);
}
#endif
