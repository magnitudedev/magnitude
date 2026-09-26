// The body of the prefill attention entries (`gated_attention_prefill` over
// dense history, `gated_attention_prefill_k8v4` over affine K8/V4 history).
// The Vulkan form of `metal/gated_attention_prefill.metal`: the same three
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
// Shared bytes: the K/V tile (32 x (W + 8) f16), then 2 KiB per subgroup.
#include "history.glsl"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#error "prefill attention stages 2-byte activations"
#endif

#define PREFILL_KEYS FLASH_KEYS
#define PREFILL_MIN_TILES 16u
#define PREFILL_QT (uint(SEISMIC_TUNE_ROWS) / ATTENTION_G)
// Output columns per pass over the keys: the whole head, so every key tile's
// scores and history decode happen once, where the device compiles wide
// accumulator arrays (`flash.glsl`).
#define PREFILL_WINDOW FLASH_WINDOW

// ---------------------------------------------------------------------------
// L1.

void prefill_prepare(attention_history h) {
    const uint W = ATTENTION_W, E = ATTENTION_E, KV = ATTENTION_KV, G = ATTENTION_G;
    const uint lane = SEISMIC_LANE;
    const uint64_t rows = SEISMIC_DIM_M;
    const uint64_t padded = (rows + PREFILL_QT - 1ul) / PREFILL_QT * PREFILL_QT;
    const uint64_t item = uint64_t(gl_WorkGroupID.x) * SEISMIC_SUBGROUPS + SEISMIC_SUBGROUP;
    const uint64_t row = item / (KV * (G + 1u));
    const uint head = uint(item % (KV * (G + 1u)));
    if (row >= padded)
        return;
    const uint64_t queries = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES);
    if (head < KV * G) {
        const uint kv_head = head / G, g = head % G;
        const uint64_t at = ((uint64_t(kv_head) * padded + row) * G + g) * W + lane * E;
        float x[ATTENTION_E];
        if (row < rows) {
            attention_prepare(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_GATE) + ((row * KV * G + head) * 2ul * W) * 2ul,
                SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM), SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES) + row * 16ul,
                SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS), SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES),
                element_word_f32(SEISMIC_PARAM_EPSILON), lane, x);
        } else {
            [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                x[i] = 0.0;
        }
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            element_put(ELEMENT_F16, queries, at + i, element_round(ELEMENT_ACT, x[i]));
        return;
    }
    if (row >= rows)
        return;
    const uint kv_head = head - KV * G;
    const uint64_t source = (row * KV + kv_head) * W;
    float k[ATTENTION_E], v[ATTENTION_E];
    attention_prepare(SEISMIC_PTR(SEISMIC_BUFFER_KEY) + source * 2ul, SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM),
        SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES) + row * 16ul, SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS),
        SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES), element_word_f32(SEISMIC_PARAM_EPSILON), lane, k);
    attention_load_row(SEISMIC_PTR(SEISMIC_BUFFER_VALUE), source + lane * E, v);
    const uint64_t keys = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS);
    const uint64_t values = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES);
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

// ---------------------------------------------------------------------------
// L2.

// Stages rows [first, first + 32) of one kv head as the f16 tile at shared
// half 0: history (dense or decoded) or the batch's fresh f16 scratch rows.
void prefill_stage(attention_history h, const bool is_key, bool historical, int first, int end, uint kv_head) {
    if (historical) {
        history_stage(h, is_key, first, end, kv_head, 0u);
    } else {
        const uint64_t plane =
            is_key ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS) : SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES);
        flash_stage(ELEMENT_F16, plane, uint64_t(ATTENTION_KV * ATTENTION_W), uint64_t(kv_head * ATTENTION_W), first, end, ATTENTION_W, 0u);
    }
}

// The union [lo, hi) of the tile's non-empty row intervals and the
// intersection [common_lo, common_hi) of all its row intervals, for span
// `index` (every subgroup computes it; lanes < QT hold the tile's rows).
ivec4 prefill_interval(uint64_t visible, uint64_t fresh, uint64_t tile_first, uint64_t rows, uint64_t spans, uint64_t index) {
    const uint lane = SEISMIC_LANE;
    const uint64_t tile_row = tile_first + lane;
    const bool row_valid = lane < PREFILL_QT && tile_row < rows;
    int lo = 0, hi = 0;
    if (row_valid)
        attention_span(visible, fresh, tile_row, spans, index, lo, hi);
    const bool nonempty = row_valid && hi > lo;
    return ivec4(seismic_subgroup_min(nonempty ? lo : 0x7fffffff), seismic_subgroup_max(nonempty ? hi : int(0x80000000)),
        seismic_subgroup_max(row_valid ? lo : int(0x80000000)), seismic_subgroup_min(row_valid ? hi : 0x7fffffff));
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

void prefill_attend(attention_history h) {
    const uint W = ATTENTION_W, KV = ATTENTION_KV, G = ATTENTION_G;
    const uint64_t visible = SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE);
    const uint64_t fresh = SEISMIC_PTR(SEISMIC_BUFFER_FRESH);
    const uint64_t rows = SEISMIC_DIM_M;
    const uint64_t spans = SEISMIC_DIM_R;
    const uint64_t padded = (rows + PREFILL_QT - 1ul) / PREFILL_QT * PREFILL_QT;
    const float scale = element_word_f32(SEISMIC_PARAM_SCALE) * ATTENTION_LOG2E;
    const uint lane = SEISMIC_LANE;
    // Query tiles dispatch last-first: in a causal chunk the last tiles see
    // the most keys, and starting them first shortens the grid's tail.
    const uint tile = gl_NumWorkGroups.x - 1u - gl_WorkGroupID.x;
    const uint kv_head = gl_WorkGroupID.y;
    const uint part = gl_WorkGroupID.z;
    const uint64_t tile_first = uint64_t(tile) * PREFILL_QT;
    const uint scratch = (FLASH_KEYS * flash_pitch(W)) / 2u + SEISMIC_SUBGROUP * FLASH_SCRATCH_FLOATS;

    uint total_tiles = 0u;
    for (uint64_t index = 0ul; index <= spans; ++index)
        total_tiles += prefill_span_tiles(prefill_interval(visible, fresh, tile_first, rows, spans, index));
    // The grid's key partitions: max(1, ceil(SPLIT_GROUPS / (tiles * KV))).
    const uint parts = gl_NumWorkGroups.z;
    const uint per = max(PREFILL_MIN_TILES, (total_tiles + parts - 1u) / parts);
    const uint used = max(1u, (total_tiles + per - 1u) / per);
    if (part >= used)
        return;
    if (part == 0u && kv_head == 0u && gl_LocalInvocationIndex == 0u)
        element_u32_put(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_COUNTS) + uint64_t(tile) * 4ul, used);
    const uint tiles_lo = part * per;
    const uint tiles_hi = min(tiles_lo + per, total_tiles);

    // The subgroup's block: matrix rows 16 sg .. of the tile's QT x G rows;
    // the lane's softmax row is lane % 16.
    const uint block_row = 16u * SEISMIC_SUBGROUP;
    const uint lane_row = block_row + lane % 16u;
    const uint64_t token = tile_first + lane_row / G;
    const bool valid = token < rows;
    const uint64_t block_queries = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES)
        + ((uint64_t(kv_head) * padded + tile_first) * G + block_row) * W * 2ul;

    const uint64_t heads = uint64_t(KV * G);
    const uint64_t query_gate = SEISMIC_PTR(SEISMIC_BUFFER_QUERY_GATE);
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
                barrier();
                prefill_stage(h, true, historical, first, interval.y, kv_head);
                barrier();
                flash_scores(block_queries, uint64_t(W), W, 0u, scratch);
                const bool inside = first >= interval.z && first + int(PREFILL_KEYS) <= interval.w;
                float s[16];
                prefill_lane_scores(scratch, scale, first, inside, row_lo, row_hi, s);
                const float alpha = flash_online(softmax, s);
                const uint p_half = flash_publish_probabilities(scratch, s);
                flash_rescale(scratch, alpha, W, PREFILL_WINDOW, o);
                barrier();
                prefill_stage(h, false, historical, first, interval.y, kv_head);
                barrier();
                flash_accumulate(p_half, 0u, W, PREFILL_WINDOW, column0, o);
            }
        }
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
            if (out_token < rows) {
                if (used > 1u) {
                    const uint64_t slot = (uint64_t(part) * rows + out_token) * heads + out_head;
                    element_f32_put(partials + (slot * W + column) * 4ul, value);
                    if (column == 0u) {
                        element_f32_put(statistics + slot * 8ul, row_maximum);
                        element_f32_put(statistics + slot * 8ul + 4ul, row_denominator);
                    }
                } else {
                    attention_store_gated(query_gate, gated, out_token, out_head, column,
                        seismic_div_rn(value, max(row_denominator, 1e-30)));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// L3: workgroup (QT-row tile, query head), one invocation per column. A tile
// that took several key partitions merges each of its rows' partitions in
// partition order and applies the gate; other tiles were stored by L2.
void prefill_merge() {
    const uint64_t heads = uint64_t(ATTENTION_KV * ATTENTION_G);
    const uint tile = gl_WorkGroupID.x;
    const uint64_t head = gl_WorkGroupID.y;
    const uint column = gl_LocalInvocationIndex;
    const uint count = element_u32_at(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_COUNTS) + uint64_t(tile) * 4ul);
    if (count <= 1u)
        return;
    const uint64_t rows = SEISMIC_DIM_M;
    for (uint64_t row = uint64_t(tile) * PREFILL_QT; row < min(uint64_t(tile + 1u) * PREFILL_QT, rows); ++row) {
        const float attended = attention_merge(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS),
            SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS), row * heads + head, rows * heads, count, column);
        attention_store_gated(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_GATE), SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), row, head,
            column, attended);
    }
}
