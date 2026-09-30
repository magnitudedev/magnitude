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
// PREFILL_ROWS = 16 * SIMDS): only L2 runs, as workgroup (kv head, partition,
// QT-row tile) of the decode's partial launch. It prepares its own query tile
// into shared memory after the subgroup scratch ([ROWS][W + 8] f16) and its
// fresh K/V tiles (prepared and rounded as L1 would), partition 0 appending
// the tile's rows; the key tiles split into equal runs over PREFILL_PARTS
// partitions, and every partition, an empty one too (denominator 0), stores
// its partial in the decode layout ((row * KV * G + head) * PARTS + part) for
// the decode merge.
#include "history.glsl"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#error "prefill attention stages 2-byte activations"
#endif

#define PREFILL_KEYS FLASH_KEYS
// PREFILL_ROWS, the matrix rows of a workgroup, is the entry's (and with
// PREFILL_DECODE, PREFILL_PARTS its partitions).

// The prefill entries' scratch planes ([KV][rows][G][W], [M][KV][W],
// [M][KV][W] f16 and the tile partition counts), named by the entry: the
// entry ABI is not this file's. Decode rows use none.
struct prefill_scratch {
    uint64_t queries;
    uint64_t keys;
    uint64_t values;
    uint64_t counts;
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

#ifndef PREFILL_DECODE

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
                SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM), SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES) + row * 16ul,
                SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS), SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES),
                ATTENTION_AMPLITUDE_BUFFER, element_word_f32(SEISMIC_PARAM_EPSILON), lane, x);
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
    attention_prepare(SEISMIC_PTR(SEISMIC_BUFFER_KEY) + source * 2ul, SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM),
        SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES) + row * 16ul, SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS),
        SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES), ATTENTION_AMPLITUDE_BUFFER,
        element_word_f32(SEISMIC_PARAM_EPSILON), lane, k);
    attention_value(SEISMIC_PTR(SEISMIC_BUFFER_VALUE) + source * 2ul, ATTENTION_VALUE_NORM_BUFFER,
        element_word_f32(SEISMIC_PARAM_EPSILON), lane, v);
    const uint64_t keys = scratch_planes.keys;
    const uint64_t values = scratch_planes.values;
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
        k[i] = element_round(ELEMENT_ACT, k[i]);
        element_put(ELEMENT_F16, keys, source + lane * E + i, k[i]);
        element_put(ELEMENT_F16, values, source + lane * E + i, v[i]);
    }
    const int destination = element_i32_at(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATIONS) + row * 4ul);
    if (destination < 0)
        return;
    history_append(h, true, destination, kv_head, lane, k);
    history_append(h, false, destination, kv_head, lane, v);
}
#endif

// ---------------------------------------------------------------------------
// L2.

// The lane's E columns of batch row `row`'s prepared key (rounded to A, as
// history stores it) or value for `kv_head`, one subgroup.
void prefill_fresh_row(const bool is_key, uint64_t row, uint kv_head, uint lane, out float x[ATTENTION_E]) {
    const uint64_t source = (row * ATTENTION_KV + kv_head) * ATTENTION_W * uint64_t(ELEMENT_BYTES(ELEMENT_ACT));
    const float epsilon = element_word_f32(SEISMIC_PARAM_EPSILON);
    if (is_key) {
        attention_prepare(SEISMIC_PTR(SEISMIC_BUFFER_KEY) + source, SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM),
            SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES) + row * 16ul, SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS),
            SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES), ATTENTION_AMPLITUDE_BUFFER, epsilon, lane, x);
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            x[i] = element_round(ELEMENT_ACT, x[i]);
    } else {
        attention_value(SEISMIC_PTR(SEISMIC_BUFFER_VALUE) + source, ATTENTION_VALUE_NORM_BUFFER, epsilon, lane, x);
    }
}

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
#ifdef PREFILL_DECODE
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
#ifdef PREFILL_DECODE
    // The query tile [ROWS][W + 8] f16, after every subgroup's scratch.
    const uint queries = FLASH_KEYS * flash_pitch(chunk) + 2u * SEISMIC_SUBGROUPS * FLASH_SCRATCH_FLOATS;
    for (uint i = SEISMIC_SUBGROUP; i < PREFILL_ROWS; i += SEISMIC_SUBGROUPS) {
        const uint64_t token = tile_first + i / G;
        float x[ATTENTION_E];
        if (i < PREFILL_QT * G && token < rows) {
            attention_prepare(ATTENTION_QUERY_BUFFER + ((token * KV * G + kv_head * G + i % G) * ATTENTION_QUERY_STRIDE)
                    * uint64_t(ELEMENT_BYTES(ELEMENT_ACT)),
                SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM), SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES) + token * 16ul,
                SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS), SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES),
                ATTENTION_AMPLITUDE_BUFFER, element_word_f32(SEISMIC_PARAM_EPSILON), lane, x);
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

    uint total_tiles = 0u;
    for (uint64_t index = 0ul; index <= spans; ++index)
        total_tiles += prefill_span_tiles(prefill_interval(visible, fresh, tile_first, rows, spans, index));
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

    // The subgroup's block: matrix rows 16 sg .. of the tile's QT x G rows;
    // the lane's softmax row is lane % 16. When G does not divide ROWS, the
    // rows past QT x G are padding: they score whatever query scratch follows
    // (the declaration adds ROWS rows of slack) and store nothing.
    const uint block_row = 16u * SEISMIC_SUBGROUP;
    const uint lane_row = block_row + lane % 16u;
    const uint64_t token = tile_first + lane_row / G;
    const bool valid = lane_row < PREFILL_QT * G && token < rows;
    // A block of padding rows only (the tile's last rows, or with decode rows
    // most blocks) takes part in the staging and the barriers, not the
    // products: its subgroup is uniform in this.
    const bool computes = block_row < PREFILL_QT * G && tile_first + block_row / G < rows;
#ifndef PREFILL_DECODE
    const uint64_t block_queries = scratch_planes.queries
        + ((uint64_t(kv_head) * padded + tile_first) * G + block_row) * W * 2ul;
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
        uint tiles_before = 0u;
        for (uint64_t index = 0ul; index <= spans; ++index) {
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
                flash_scores_state scores;
                flash_scores_clear(scores);
                for (uint column = 0u; column < W; column += chunk) {
                    barrier();
                    prefill_stage(h, scratch_planes, true, historical, first, interval.y, kv_head, column, chunk);
                    barrier();
                    if (computes) {
#ifdef PREFILL_DECODE
                        flash_scores_add_shared(queries + block_row * (W + 8u) + column, W + 8u, chunk, 0u, scores);
#else
                        flash_scores_add(block_queries + uint64_t(column) * 2ul, uint64_t(W), chunk, 0u, scores);
#endif
                    }
                }
                uint p_half = 0u;
                if (computes) {
                    flash_scores_publish(scores, scratch);
                    const bool inside = first >= interval.z && first + int(PREFILL_KEYS) <= interval.w;
                    float s[16];
                    prefill_lane_scores(scratch, scale, first, inside, row_lo, row_hi, s);
                    const float alpha = flash_online(softmax, s);
                    p_half = flash_publish_probabilities(scratch, s);
                    flash_rescale(scratch, alpha, W, PREFILL_WINDOW, o);
                }
                barrier();
                prefill_stage(h, scratch_planes, false, historical, first, interval.y, kv_head, column0, width);
                barrier();
                if (computes)
                    flash_accumulate(p_half, 0u, width, width, 0u, o);
            }
        }
        const float maximum = softmax.maximum;
        const float denominator = flash_denominator(softmax);
        // Every subgroup of a used partition reaches here; the scratch is
        // free.
        barrier();
        // Unrolled, so every fragment index is a constant.
        [[unroll]] for (uint q = 0u; q < PREFILL_WINDOW / 2u; ++q) {
            if (!computes || q >= flash_window(W, PREFILL_WINDOW) / 2u)
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
                if (used > 1u) {
                    const uint64_t slot = (uint64_t(part) * rows + out_token) * heads + out_head;
                    element_f32_put(partials + (slot * W + column) * 4ul, value);
                    if (column == 0u) {
                        element_f32_put(statistics + slot * 8ul, row_maximum);
                        element_f32_put(statistics + slot * 8ul + 4ul, row_denominator);
                    }
                } else {
                    attention_store_gated(query, ATTENTION_GATE_BUFFER, gated, out_token, out_head, column,
                        seismic_div_rn(value, max(row_denominator, 1e-30)));
                }
#endif
            }
        }
    }
}

#ifndef PREFILL_DECODE
// ---------------------------------------------------------------------------
// L3: workgroup (QT-row tile, query head), one invocation per column. A tile
// that took several key partitions merges each of its rows' partitions in
// partition order and applies the gate; other tiles were stored by L2.
void prefill_merge(prefill_scratch scratch_planes) {
    const uint64_t heads = uint64_t(ATTENTION_KV * ATTENTION_G);
    const uint tile = gl_WorkGroupID.x;
    const uint64_t head = gl_WorkGroupID.y;
    const uint column = gl_LocalInvocationIndex;
    const uint count = element_u32_at(scratch_planes.counts + uint64_t(tile) * 4ul);
    if (count <= 1u)
        return;
    const uint64_t rows = SEISMIC_DIM_M;
    for (uint64_t row = uint64_t(tile) * PREFILL_QT; row < min(uint64_t(tile + 1u) * PREFILL_QT, rows); ++row) {
        const float attended = attention_merge(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS),
            SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS), row * heads + head, rows * heads, count, column);
        attention_store_gated(ATTENTION_QUERY_BUFFER, ATTENTION_GATE_BUFFER, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), row,
            head, column, attended);
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
        (row * ATTENTION_KV * ATTENTION_G + head) * PREFILL_PARTS, 1ul, PREFILL_PARTS, column);
    attention_store_gated(ATTENTION_QUERY_BUFFER, ATTENTION_GATE_BUFFER, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), row,
        head, column, attended);
}
#endif
