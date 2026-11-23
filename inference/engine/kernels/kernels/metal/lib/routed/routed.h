// Shared pieces of the routed (mixture-of-experts) Metal entries built on the
// projection family: the record an expert's weights are read from, the plain
// activation prologue they share, the grouped-row prologue of the expert
// GEMMs, the live-row count of a grouped block and the choice of a block's
// GEMM form. Every form gives the GEMM's bits, so a row's output depends on
// its own row alone, never on how many rows share its block. The grouped
// projection's other forms (TALL, INT8, PACK) take whole blocks, each row's
// result its own but under PACK, which packs a block's adjacent rows in
// pairs (as dense's, `int8_token_packing`): their grids, the grouped rows'
// positions and the forms' scratch are here. Weights are
// bound through the `packets.h` slots (`KERNEL_W0..3`); an expert-stacked
// weight a launch reads per step's source is a record port, bound as its
// records' address table, and an expert's weights are rows 0.. of its
// record's base (`slab::Experts::matrix`, in the bound weights' space), the
// record a `slab::Source` names; a resident layer's expert is its own
// record (`decode_source`).

#include <seismic/slab.h>
#include "../projection/projection.h"

namespace routed {

typedef activation::Act Act;
static_assert(Act::bytes == 2, "the Metal routed family requires a bf16 or f16 activation element");

// An expert-stacked weight's record tables, in the bound weights' space.
typedef slab::Experts<packets::space> Experts;

// The source a decode expert entry reads choice (m, k)'s weights from, of
// expert `expert` (`decode_source`, routed.seismic): the expert itself
// without sources (`rows` 0, a resident layer: record `expert` of its
// experts' table), else the step's source (`sources`' SOURCE_RECORD row at
// strides `row_stride`, `choice_stride`): a bank record, a host tier record
// (the secondary table, which on unified memory the entries read in place as
// they read the bank's), or none (a choice that reads no record and
// contributes nothing; when is stated there).
inline slab::Source decode_source(device const int *sources, ulong rows, ulong row_stride, ulong choice_stride,
                                  ulong m, ulong k, int expert) {
    return slab::Source{rows == 0 ? expert : sources[m * row_stride + k * choice_stride]};
}

// A choice with no record's product row (`decode_source` none): `count` zero A
// elements at `out`, element stride `stride`, by `threads` threads from
// `first`.
inline void zero_row(element::Out<Act> out, ulong stride, uint count, uint first, uint threads) {
    for (uint i = first; i < count; i += threads)
        out.put(ulong(i) * stride, 0.0f);
}

inline void write_grid(device int *grid, uint x, uint y) {
    grid[0] = int(x);
    grid[1] = int(y);
    grid[2] = 1;
}

// The grids of an expert pass over `rows` grouped rows and `used` blocks of
// `tile_rows` rows (`routed_experts`): a stage threadgroup per row, a
// quantize threadgroup per block row, and `subtiles` row tiles of each used
// block by the expand's and down's column tiles.
inline void expert_grids(uint rows, uint used, uint tile_rows, uint subtiles, uint expand_columns,
    uint down_columns, device int *stage, device int *expand, device int *quantize, device int *down) {
    write_grid(stage, rows, 1);
    write_grid(expand, expand_columns, used * subtiles);
    write_grid(quantize, used * tile_rows, 1);
    write_grid(down, down_columns, used * subtiles);
}

// Rows of weight tensor `base` starting at row `first`.
template <typename W>
inline projection::Weights<W> weights(packets::weight_bytes base, packets::Rows16 layout, ulong first, uint k) {
    return projection::Weights<W>{{base, layout, k}, nullptr, first};
}

// The weights of record `record` of an expert-stacked weight's record table
// `records`: rows 0.. of the record's base.
template <typename W>
inline projection::Weights<W> record_weights(device const ulong *records, int record, packets::Rows16 layout,
    uint k) {
    return weights<W>(slab::region<const uchar, packets::space>(records, ulong(record)), layout, 0, k);
}

// The weights of an expert's matrix where `source` (not none) names it.
template <typename W>
inline projection::Weights<W> expert_matrix(Experts experts, slab::Source source, packets::Rows16 layout, uint k) {
    return weights<W>(experts.matrix(source), layout, 0, k);
}

// The gathered decode form (`choice`, GATHERED): the threadgroups of choice
// `at` (flat m * K + k) run only where it is its expert's first choice, and
// then project that expert's weights once over the activation row of every
// choice of it, so a step reads each chosen expert's records once. A row
// chooses an expert at most once, so an expert has at most M (at most
// VERIFY_ROWS) choices. A GEMV row's sums never depend on the other rows
// (`gemv.h`), so every choice keeps its one-row bits.
struct Gathered {
    // The expert, and its choices' count: 0 where `at` is not its first.
    int expert;
    uint count;
    // Choice r's activation row, and its product row's element offset
    // (m * stride0 + k * stride1).
    uint rows[SEISMIC_CONST_VERIFY_ROWS];
    uint products[SEISMIC_CONST_VERIFY_ROWS];
};

// Compact unique-expert worklist emitted once by a 1-simdgroup `*_gather`
// launch for the gathered decode forms (`choice` and `summed`), replacing
// per-threadgroup `M * K` global route scans and enabling hardware indirect
// dispatch over only the active unique `(expert, source)` groups.
struct WorkEntry {
    int expert;
    int source;
    uint count;
    uint pad;
    // Flat choice encoded as `(m << 16) | k` (M <= VERIFY_ROWS, K < 65536).
    uint packed[SEISMIC_CONST_VERIFY_ROWS];
};

struct Worklist {
    uint count;
    uint one_count;
    uint multi_count;
    uint capacity;
    WorkEntry entries[1];
};

// Unpack a compact worklist entry into `Gathered` with product offsets at
// strides `p0`, `p1`.
inline Gathered unpack_work_entry(device const WorkEntry &entry, ulong p0, ulong p1) {
    Gathered chosen;
    chosen.expert = entry.expert;
    chosen.count = entry.count;
    for (uint r = 0u; r < entry.count; ++r) {
        const uint packed = entry.packed[r];
        const uint m = packed >> 16u;
        const uint k = packed & 0xffffu;
        chosen.rows[r] = m;
        chosen.products[r] = uint(ulong(m) * p0 + ulong(k) * p1);
    }
    return chosen;
}

constexpr constant uint WORKLIST_CACHE = 256u;
constexpr constant uint WORKLIST_CACHE_WORDS = 2u * WORKLIST_CACHE + SEISMIC_CONST_VERIFY_ROWS;

// Build the compact unique-expert worklist in `worklist` using one simdgroup
// (32 lanes), partitioning singletons (`r == 1`) from the front of
// `worklist->entries` (`gather_one_grid`: `(x_groups, one_count, 1)`) and
// multi-row experts (`r >= 2`) from the back (`gather_grid`:
// `(x_groups, multi_count + extra_y, 1)`). `cache` holds `WORKLIST_CACHE_WORDS`
// ints in threadgroup memory (declared in the `*_gather` kernel).
template <typename Active>
inline void build_worklist(device const int *routes, ulong r0, ulong r1, device const int *sources, ulong source_rows,
    ulong src1, ulong src2, uint m_rows, uint slots, uint x_groups, uint extra_y, device Worklist *worklist,
    device int *gather_grid, device int *gather_one_grid, threadgroup int *cache, uint lane, Active active) {
    constexpr uint CACHE = WORKLIST_CACHE;
    threadgroup int *cached_expert = cache;
    threadgroup int *cached_source = cache + CACHE;
    threadgroup uint *tmp_packed = reinterpret_cast<threadgroup uint *>(cache + 2u * CACHE);

    const uint total = m_rows * slots;
    uint one_count = 0u;
    uint multi_count = 0u;
    auto emit_entry = [&](int ex, int src, uint count) {
        const uint slot = count == 1u ? one_count++ : (total - 1u - multi_count++);
        if (lane < count)
            worklist->entries[slot].packed[lane] = tmp_packed[lane];
        if (lane == 0u) {
            worklist->entries[slot].expert = ex;
            worklist->entries[slot].source = src;
            worklist->entries[slot].count = count;
            worklist->entries[slot].pad = 0u;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    };
    if (total <= CACHE) {
        uint active_mask[CACHE / 32u];
        uint claimed[CACHE / 32u];
        for (uint w = 0u; w < CACHE / 32u; ++w) {
            active_mask[w] = 0u;
            claimed[w] = 0u;
        }
        for (uint base = 0u; base < total; base += 32u) {
            const uint c = base + lane;
            bool is_active = false;
            if (c < total) {
                const uint m = c / slots, k = c % slots;
                const int ex = routes[ulong(m) * r0 + ulong(k) * r1];
                const slab::Source src = decode_source(sources, source_rows, src1, src2, m, k, ex);
                cached_expert[c] = ex;
                cached_source[c] = src.code;
                is_active = active(c, src);
            }
            active_mask[base >> 5u] = uint(uint64_t(metal::simd_ballot(is_active)));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint at = 0u; at < total; ++at) {
            const uint word = at >> 5u, bit = 1u << (at & 31u);
            if (((active_mask[word] & ~claimed[word]) & bit) == 0u)
                continue;
            const int ex = cached_expert[at];
            const int src = cached_source[at];
            if (src == -1) {
                claimed[word] |= bit;
                if (lane == 0u)
                    tmp_packed[0] = ((at / slots) << 16u) | (at % slots);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                emit_entry(ex, -1, 1u);
                continue;
            }
            uint count = 0u;
            const uint start_base = at & ~31u;
            for (uint base = start_base; base < total; base += 32u) {
                const uint c = base + lane;
                const uint w = base >> 5u;
                const bool live = c < total && c >= at && ((active_mask[w] & ~claimed[w]) & (1u << lane)) != 0u;
                const bool match = live && (c == at || (cached_expert[c] == ex && cached_source[c] == src));
                const uint mask = uint(uint64_t(metal::simd_ballot(match)));
                if (mask != 0u) {
                    claimed[w] |= mask;
                    const uint lt_mask = (1u << lane) - 1u;
                    const uint pos = count + metal::popcount(mask & lt_mask);
                    if (match && pos < uint(SEISMIC_CONST_VERIFY_ROWS)) {
                        const uint m = c / slots, k = c % slots;
                        tmp_packed[pos] = (m << 16u) | k;
                    }
                    count = metal::min(count + metal::popcount(mask), uint(SEISMIC_CONST_VERIFY_ROWS));
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            emit_entry(ex, src, count);
        }
    } else {
        for (uint at = 0u; at < total; ++at) {
            const uint at_m = at / slots, at_k = at % slots;
            const int ex = routes[ulong(at_m) * r0 + ulong(at_k) * r1];
            const slab::Source at_src = decode_source(sources, source_rows, src1, src2, at_m, at_k, ex);
            if (!active(at, at_src))
                continue;
            if (at_src.none()) {
                if (lane == 0u)
                    tmp_packed[0] = (at_m << 16u) | at_k;
                simdgroup_barrier(mem_flags::mem_threadgroup);
                emit_entry(ex, -1, 1u);
                continue;
            }
            bool earlier_hit = false;
            uint count = 0u;
            for (uint base = 0u; base < total; base += 32u) {
                const uint c = base + lane;
                bool match = false;
                if (c < total) {
                    if (c == at) {
                        match = true;
                    } else {
                        const uint m = c / slots, k = c % slots;
                        const int oex = routes[ulong(m) * r0 + ulong(k) * r1];
                        if (oex == ex) {
                            const slab::Source osrc = decode_source(sources, source_rows, src1, src2, m, k, oex);
                            match = osrc.code == at_src.code && active(c, osrc);
                        }
                    }
                }
                const uint mask = uint(uint64_t(metal::simd_ballot(match)));
                if (base < at) {
                    const uint earlier = at - base >= 32u ? 0xffffffffu : ((1u << (at - base)) - 1u);
                    if ((mask & earlier) != 0u) {
                        earlier_hit = true;
                        break;
                    }
                }
                if (mask != 0u) {
                    const uint lt_mask = (1u << lane) - 1u;
                    const uint pos = count + metal::popcount(mask & lt_mask);
                    if (match && pos < uint(SEISMIC_CONST_VERIFY_ROWS)) {
                        const uint m = c / slots, k = c % slots;
                        tmp_packed[pos] = (m << 16u) | k;
                    }
                    count = metal::min(count + metal::popcount(mask), uint(SEISMIC_CONST_VERIFY_ROWS));
                }
            }
            if (!earlier_hit) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
                emit_entry(ex, at_src.code, count);
            }
        }
    }
    if (lane == 0u) {
        worklist->count = one_count + multi_count;
        worklist->one_count = one_count;
        worklist->multi_count = multi_count;
        worklist->capacity = total;
        write_grid(gather_one_grid, x_groups, one_count);
        write_grid(gather_grid, x_groups, multi_count + extra_y);
    }
}

inline void build_worklist(device const int *routes, ulong r0, ulong r1, device const int *sources, ulong source_rows,
    ulong src1, ulong src2, uint m_rows, uint slots, uint x_groups, uint extra_y, device Worklist *worklist,
    device int *gather_grid, device int *gather_one_grid, threadgroup int *cache, uint lane) {
    build_worklist(routes, r0, r1, sources, source_rows, src1, src2, m_rows, slots, x_groups, extra_y, worklist,
        gather_grid, gather_one_grid, cache, lane, [](uint, slab::Source) { return true; });
}

inline void build_worklist(device const int *routes, ulong r0, ulong r1, uint m_rows, uint slots, uint x_groups,
    uint extra_y, device Worklist *worklist, device int *gather_grid, device int *gather_one_grid,
    threadgroup int *cache, uint lane) {
    build_worklist(routes, r0, r1, nullptr, 0, 0, 0, m_rows, slots, x_groups, extra_y, worklist, gather_grid,
        gather_one_grid, cache, lane);
}

// The activation rows a gathered GEMV reads: row r is `rows[r]`.
struct Listed {
    thread const uint *rows;
    uint at(uint m) const { return rows[m]; }
};

// A gathered GEMV's epilogue: its row r is stored where product row
// `products[r]` (an element offset) lies, by an epilogue whose row stride is
// one element.
template <typename Out>
struct ChoiceRows {
    Out out;
    thread const uint *products;
    void store(uint m, uint n, float value) const { out.store(products[m], n, value); }
    void store2(uint m, uint n, float first, float second) const { out.store2(products[m], n, first, second); }
    void store_pair(uint m, uint n, float gate, float up) const { out.store_pair(products[m], n, gate, up); }
};

// The gathered decode down projections (`summed`, GATHERED) publish each
// slot's projection in A ([M, SLOTS, H], SLOTS the K choices then the shared
// expert's where it has one): stored by the launch's threadgroups, loaded by
// the last to arrive, so through memory declared coherent
// (`arrive::coherent_device`).
typedef coherent(device) device Act::storage *PublishedSlots;

// Channel `channel` of a projection published in A at row `row` of `published`.
template <uint SLOTS>
struct OutPublishedSlots {
    element::Out<Act> out;
    ulong s0, s1, s2;
};

template <uint H>
inline void publish_channel(PublishedSlots published, uint row, uint channel, float value) {
    published[ulong(row) * H + channel] = Act::store(value);
}

template <uint H, uint SLOTS>
inline void publish_channel(OutPublishedSlots<SLOTS> published, uint row, uint channel, float value) {
    const ulong base = ulong(row / SLOTS) * published.s0 + ulong(row % SLOTS) * published.s1;
    published.out.put(base + ulong(channel) * published.s2, value);
}

template <uint TILE, uint H, typename Published>
inline void publish_zero(Published published, uint row, uint first, uint lane) {
    for (uint channel = lane; channel < TILE; channel += 32)
        if (first + channel < H)
            publish_channel<H>(published, row, first + channel, 0.0f);
}

// Gathered rows of a [M, K, columns] A product tensor: row r lies at element
// offset `offsets[r]`, elements `stride` apart. Vectorized when `stride == 1`
// and every row offset is 8-element aligned (`aligned`).
struct GatheredProduct {
    typedef Act activation;
    device const uchar *x;
    threadgroup const ulong *offsets;
    ulong stride;
    uint columns;
    bool aligned;
    uint width() const { return columns; }
    void load8(uint m, uint k, float, thread float4 &even, thread float4 &odd) const {
        device const typename Act::storage *row =
            reinterpret_cast<device const typename Act::storage *>(x) + offsets[m];
        projection::load8_storage<Act>(row, stride, k, columns, aligned, even, odd);
    }
    uint4 words8(uint m, uint k) const {
        device const typename Act::storage *row =
            reinterpret_cast<device const typename Act::storage *>(x) + offsets[m];
        return projection::words8_storage<Act>(row, stride, k, columns, aligned);
    }
};

// Channel n of row m's projections where a gathered decode down launch
// published them (`project_channels_published`), A-rounded, read by its last
// threadgroup to arrive.
template <uint SLOTS, uint H>
struct ArrivedProjections {
    PublishedSlots published;
    ulong m;
    uint n;
    float at(uint k) const thread {
        return Act::load(published[(m * SLOTS + k) * H + n]);
    }
    float4 at4(uint k) const thread {
        const vec<typename Act::storage, 4> raw = vec<typename Act::storage, 4>(
            *reinterpret_cast<coherent(device) device const typename Act::packed4 *>(
                published + (m * SLOTS + k) * H + n));
        return float4(Act::load(raw.x), Act::load(raw.y), Act::load(raw.z), Act::load(raw.w));
    }
};

// A plain activation prologue over rows of a [rows, columns] A tensor.
inline projection::Plain<Act, projection::AllRows> activation(device const uchar *x, ulong stride0, ulong stride1,
    uint columns) {
    return projection::Plain<Act, projection::AllRows>{x, stride0, stride1, columns, {}};
}

// The live rows of a grouped block: its `order` entries are one expert's rows
// followed by -1 padding, so the count is the index of the first -1. The
// expert GEMMs skip the MMAs of padding rows (`live_rows`).
inline uint block_rows(device const int *order, ulong stride, uint rows) {
    uint low = 0, high = rows;
    while (low < high) {
        uint middle = (low + high) / 2u;
        if (order[ulong(middle) * stride] >= 0)
            low = middle + 1u;
        else
            high = middle;
    }
    return low;
}

// The grouped prologue of the expert GEMMs: row m is tile row m % T of block
// m / T (T `tile_rows`, blocks `block_stride` entries of `order` apart), and
// reads activation row `order[m / T, m % T]`; padding rows (-1) read zeros.
// A block's GEMM takes its own rows (`order` at the block, m below T); the
// forms whose operand covers every block take the grouped rows' positions,
// block b's from b * T.
struct Grouped {
    typedef Act activation;
    device const uchar *x;
    ulong stride0, stride1;
    uint columns;
    device const int *order;
    ulong order_stride;
    ulong block_stride;
    uint tile_rows;
    uint groups() const { return 0; }
    uint width() const { return 1; }
    float norm_input(uint, uint, uint) const { return 0.0f; }
    float epsilon() const { return 0.0f; }
    uint inverse_index(uint, uint) const { return 0; }
    int source(uint m) const {
        return order[ulong(m / tile_rows) * block_stride + ulong(m % tile_rows) * order_stride];
    }
    float value(uint m, uint k, float) const {
        int row = source(m);
        return row < 0 ? 0.0f
            : Act::load(reinterpret_cast<device const typename Act::storage *>(x)
                [ulong(row) * stride0 + ulong(k) * stride1]);
    }
    uint4 words8(uint m, uint k) const {
        int row = source(m);
        if (row < 0)
            return uint4(0);
        device const typename Act::storage *base =
            reinterpret_cast<device const typename Act::storage *>(x) + ulong(row) * stride0;
        return projection::words8_storage<Act>(base, stride1, k, columns, stride1 == 1 && (stride0 & 7u) == 0);
    }
    void load8(uint m, uint k, float, thread float4 &even, thread float4 &odd) const {
        int row = source(m);
        if (row < 0) {
            even = odd = float4(0.0f);
            return;
        }
        device const typename Act::storage *base =
            reinterpret_cast<device const typename Act::storage *>(x) + ulong(row) * stride0;
        projection::load8_storage<Act>(base, stride1, k, columns, stride1 == 1 && (stride0 & 7u) == 0, even, odd);
    }
    // x[m, k] and x[m, k + 1] as one storage word, zero for a padding row or
    // a row at or past `m_rows` (`projection::tall_relayout`).
    uint words2(uint m, uint k, uint m_rows) const {
        int row = m < m_rows ? source(m) : -1;
        if (row < 0)
            return 0u;
        device const typename Act::storage *base =
            reinterpret_cast<device const typename Act::storage *>(x) + ulong(row) * stride0;
        if (stride1 == 1)
            return as_type<uint>(ushort2(*reinterpret_cast<device const packed_ushort2 *>(base + k)));
        return uint(as_type<ushort>(base[ulong(k) * stride1])) | uint(as_type<ushort>(base[ulong(k + 1u) * stride1])) << 16u;
    }
};

// The products of the grouped rows (`product`, a row a position) as the down
// GEMM's forms over every block read them: padding positions (`order` -1)
// read zeros, as dense pads its rows, whatever the expansion left there.
struct GroupedProducts {
    typedef Act activation;
    projection::Plain<Act, projection::AllRows> rows;
    Grouped grouped;
    void load8(uint m, uint k, float inverse, thread float4 &even, thread float4 &odd) const {
        if (grouped.source(m) < 0) {
            even = odd = float4(0.0f);
            return;
        }
        rows.load8(m, k, inverse, even, odd);
    }
};

// Block `block`'s rows in the tall GEMM's order (`projection::tall_relayout`):
// whole 64-row tiles of `columns` storage elements, ceil(tile_rows / 64) a
// block.
inline device uchar *block_fragments(device uchar *fragments, uint block, uint tile_rows, uint columns) {
    return fragments + ulong(block) * ((tile_rows + 63u) / 64u) * 64u * columns * Act::bytes;
}

// The forms' scratch (`projection routed for metal`), each GEMM's in turn.
// `operand`: the GEMM's rows in its form, TALL's in the tall GEMM's order
// (`block_fragments`), INT8's quantized and PACK's coded over `rows` rows of
// `columns` columns (the capacity of every launch of the form). `tables`:
// over stacked experts, each used block's expert's ordinal among the used
// experts first (`ordinals`, `blocks` words in 16-byte units), then each used
// expert's tables of the GEMM's `weights` weights, ordinal o's weight j at
// o * weights + j (`weight_table`); one set over dense weights (blocks 0,
// ordinal 0). A weight of `rows` rows by `columns` columns takes the INT8
// form's block coefficients and biases, or the PACK form's block factors
// (twice them where each run has its code sums, `packing_codes::sums`) and
// weight scales (`*_table_bytes`).
// Top two bits of each ordinal word encode the block's pairing role when an
// even block `2k` and `2k + 1` belong to the same expert so they coalesce into
// a 64-row `PACK_TOKENS = 4` tile without extra scratch or grid changes.
constexpr constant uint ORDINAL_MASK = 0x3fffffffu;
constexpr constant uint ORDINAL_PAIR_EVEN = 0x40000000u;
constexpr constant uint ORDINAL_PAIR_ODD = 0x80000000u;
constexpr constant uint ORDINAL_PAIR_MASK = ORDINAL_PAIR_EVEN | ORDINAL_PAIR_ODD;

inline device int *ordinals(device uchar *tables) {
    return reinterpret_cast<device int *>(tables);
}
// Coalesced 64-row tall-tile descriptors for T == 32: each entry encodes
// `(first_block << 1) | (paired ? 1 : 0)`, stored in the second 16-byte-aligned
// region of `tables` so it never collides with `ordinals` or `weight_table`.
inline device uint *tall_tiles(device uchar *tables, uint blocks) {
    return reinterpret_cast<device uint *>(tables + ulong((blocks + 3u) / 4u) * 16u);
}
inline uint write_tall_tiles(device const int *blocks, ulong stride, uint used, device uint *tiles) {
    uint count = 0u;
    for (uint b = 0u; b < used;) {
        const bool paired = (b + 1u < used) && (blocks[ulong(b + 1u) * stride] == blocks[ulong(b) * stride]);
        tiles[count++] = (b << 1u) | (paired ? 1u : 0u);
        b += paired ? 2u : 1u;
    }
    return count;
}
inline int block_ordinal(device uchar *tables, uint block) {
    return int(uint(ordinals(tables)[block]) & ORDINAL_MASK);
}
inline uint block_pair_role(device uchar *tables, uint block) {
    return uint(ordinals(tables)[block]) & ORDINAL_PAIR_MASK;
}
inline ulong int8_coef_bytes(uint rows, uint columns) {
    return ulong(rows) * metal::max(columns / 32u, 3u) * 4u;
}
inline ulong int8_table_bytes(uint rows, uint columns) {
    return int8_coef_bytes(rows, columns) + ulong((rows + 31u) / 32u * 32u) * ((columns + 1023u) / 1024u) * 64u;
}
inline ulong packing_table_bytes(uint rows, uint columns, bool sums) {
    return ulong(rows) * (columns / 32u) * (sums ? 8u : 4u) + ulong(rows) * 4u;
}
inline device uchar *weight_table(device uchar *tables, uint blocks, int ordinal, uint weights, uint weight,
    ulong bytes) {
    return tables + ulong((blocks + 3u) / 4u) * 32u + (ulong(ordinal) * weights + weight) * bytes;
}
// The INT8 and PACK forms' scratch of a weight whose table is `table` (the
// operand's own passes take no table: `rows` 0); PACK's a
// `projection::packing_scratch`, of `packing.h`, which its entries include,
// whose tokens' entries and weights' factors are twice as wide where each run
// has its code sums (`sums`). `packing_table` is the weight's table alone,
// which the coefficient pass reads (its operand pointers none).
inline projection::int8_scratch int8_forms(device uchar *operand, ulong operand_rows, device uchar *table,
    uint rows, uint columns) {
    device uchar *row_scales = operand + operand_rows * columns;
    device uchar *block_sums = row_scales + operand_rows * (columns / 32u) * 2u;
    return projection::int8_scratch{operand, reinterpret_cast<device half *>(row_scales),
        reinterpret_cast<device half *>(block_sums), reinterpret_cast<device float *>(table),
        reinterpret_cast<device half *>(table + int8_coef_bytes(rows, columns))};
}
template <typename Packing>
inline Packing packing_table(device uchar *table, uint rows, uint columns, bool sums) {
    return Packing{nullptr, nullptr, nullptr, reinterpret_cast<device half *>(table),
        reinterpret_cast<device float *>(table + ulong(rows) * (columns / 32u) * (sums ? 8u : 4u))};
}
template <typename Packing>
inline Packing packing_forms(device uchar *operand, ulong operand_rows, device uchar *table, uint rows,
    uint columns, bool sums) {
    Packing forms = packing_table<Packing>(table, rows, columns, sums);
    device uchar *token_factors = operand + operand_rows * columns * 2u + 1024u;
    forms.operand = reinterpret_cast<device float *>(operand);
    forms.tokens = reinterpret_cast<device half *>(token_factors);
    forms.token_scales =
        reinterpret_cast<device float *>(token_factors + operand_rows * (columns / 32u) * (sums ? 8u : 4u));
    return forms;
}

// An epilogue of one block's rows reached at the grouped rows' positions:
// position m is the block's row m - first.
template <typename Out>
struct Positioned {
    Out out;
    uint first;
    void store(uint m, uint n, float value) const { out.store(m - first, n, value); }
    void store2(uint m, uint n, float a, float b) const { out.store2(m - first, n, a, b); }
    void store_pair(uint m, uint n, float gate, float up) const { out.store_pair(m - first, n, gate, up); }
    auto row(uint m) const { return projection::row_of(out, m - first); }
};
template <typename Out>
inline Positioned<Out> positioned(thread const Out &out, uint first) {
    return Positioned<Out>{out, first};
}

// An epilogue of grouped blocks' output rows (`[B, T, H]`): grouped row `m`
// maps to block `m / tile_rows`, row `m % tile_rows`, so a 64-row paired tile
// spanning two consecutive blocks stays exact even for strided outputs.
template <typename OutElem>
struct BlockOutput {
    element::Out<OutElem> y;
    ulong stride0;
    ulong stride1;
    ulong stride2;
    uint tile_rows;
    struct Row {
        element::Out<OutElem> row;
        ulong stride2;
        void store(uint n, float value) const { row.put(ulong(n) * stride2, value); }
    };
    Row row(uint m) const {
        return Row{y.offset(ulong(m / tile_rows) * stride0 + ulong(m % tile_rows) * stride1), stride2};
    }
    void store(uint m, uint n, float value) const { row(m).store(n, value); }
};

// Whether used block `b` is its expert's first: an expert's blocks are
// consecutive (`routed_group`), and its first forms what they share (the
// weights' tables of the INT8 and PACK forms, one an expert).
inline bool first_block(device const int *blocks, ulong stride, uint b) {
    return b == 0u || blocks[ulong(b - 1u) * stride] != blocks[ulong(b) * stride];
}

// Each of the `used` blocks' expert ordinals among the used experts, tagged
// with its pairing role when an even block `2k` and `2k + 1` belong to the
// same expert (sharing one 64-token packed operand tile `tile_m = k`).
inline void write_ordinals(device const int *blocks, ulong stride, uint used, device int *ordinals) {
    int ordinal = -1;
    for (uint b = 0; b < used; ++b) {
        if (first_block(blocks, stride, b))
            ++ordinal;
        uint flags = 0u;
        if ((b & 1u) == 0u) {
            if (b + 1u < used && blocks[ulong(b + 1u) * stride] == blocks[ulong(b) * stride])
                flags = ORDINAL_PAIR_EVEN;
        } else if (blocks[ulong(b - 1u) * stride] == blocks[ulong(b) * stride]) {
            flags = ORDINAL_PAIR_ODD;
        }
        ordinals[b] = int(uint(ordinal) | flags);
    }
}

// A grouped GEMM's form past its default tiles: TALL, INT8 or PACK (the
// forms' `INT8` and `PACK` parameters, each 0 or 1).
enum class gemm_form : uint { tall, int8, pack };
inline gemm_form form_of(int int8, int pack) {
    return int8 != 0 ? gemm_form::int8 : pack != 0 ? gemm_form::pack : gemm_form::tall;
}

// The grids of a grouped GEMM's form over `used` blocks of `tile_rows` rows
// (`routed_experts`): the GEMM's `n` weight rows of `width` columns, which
// `served` weights expand (its features `n` each). The form's row pass into
// `own`, the GEMM's own grid, whose default tiles the form replaces (TALL's
// relayout, a block a row; INT8's quantization and PACK's codes, over the
// blocks' positions); its tables pass into `tables` (a used block a row);
// its tiles into `tiles`: TALL's 64-row tiles, INT8's and PACK's 32-row tiles
// of 128 weight rows (64 features where paired; on a device with tensor
// operations PACK's the default form's two 32 x 64 there).
inline void form_grids(gemm_form form, uint used, uint tile_rows, uint n, uint width, uint served, device int *own,
    device int *tables, device int *tiles) {
    switch (form) {
    case gemm_form::tall:
        write_grid(own, ((tile_rows + 63u) / 64u * width + 255u) / 256u, used);
        write_grid(tiles, (n * served + 31u) / 32u, used * ((tile_rows + 63u) / 64u));
        return;
    case gemm_form::int8:
        write_grid(own, used * tile_rows, 1);
        write_grid(tables, (n * ((width + 255u) / 256u) + 1023u) / 1024u, used);
        break;
    case gemm_form::pack:
        write_grid(own, used * tile_rows / 4u, 1);
        write_grid(tables, n / 32u, used);
        break;
    }
    write_grid(tiles, (n * served + 127u) / 128u, used * (tile_rows / 32u));
}

// The expert projections of one grouped block tile: the TM x TN GEMM over
// the block's `live` rows, or, on simdgroup matrices, its one-row form
// (`projection::gemm_row`) when they fit one fragment row, with the same
// bits and without the GEMM's threadgroup-wide weight staging, which few
// rows cannot amortize.
// A tensor-operation device runs the GEMM alone: its products are one
// `matmul2d` over the staged tile, whose summation order no other form can
// be shown to reproduce.
template <typename G, typename U, uint TM, uint TN, typename In, typename Out>
inline void expert_paired(thread const In &in, thread const Out &out, thread const projection::Weights<G> &gate,
    thread const projection::Weights<U> &up, uint live, uint m_rows, uint rows, uint k, uint tm, uint tn,
    threadgroup uchar *shared, uint sg, uint lane) {
#if !SEISMIC_HAS_TENSOR_OPS
    if (live <= projection::gemm_row_rows) {
        if (tm == 0)
            projection::gemm_row_paired<G, U, TM, TN>(in, out, gate, up, live, rows, k, tn, shared, sg, lane);
        return;
    }
#endif
    projection::gemm_paired<G, U, TM, TN>(in, out, gate, up, m_rows, rows, k, tm, tn, shared, sg, lane, live);
}

template <typename W, uint TM, uint TN, typename In, typename Out>
inline void expert_plain(thread const In &in, thread const Out &out, thread const projection::Weights<W> &w,
    uint live, uint m_rows, uint rows, uint k, uint tm, uint tn, threadgroup uchar *shared, uint sg, uint lane) {
#if !SEISMIC_HAS_TENSOR_OPS
    if (live <= projection::gemm_row_rows) {
        if (tm == 0)
            projection::gemm_row<W, TM, TN>(in, out, w, live, rows, k, tn, shared, sg, lane);
        return;
    }
#endif
    projection::gemm<W, TM, TN>(in, out, w, m_rows, rows, k, tm, tn, shared, sg, lane, live);
}

// A resident combine's projections of one channel in threadgroup memory:
// slot k's at `projected[k * stride + tid]`, A-rounded.
struct TileProjections {
    threadgroup const float *projected;
    uint stride;
    uint tid;
    float at(uint k) const thread {
        return Act::round(projected[k * stride + tid]);
    }
    float4 at4(uint k) const thread {
        const float4 raw = float4(*reinterpret_cast<threadgroup const packed_float4 *>(projected + k * stride + tid));
        return float4(Act::round(raw.x), Act::round(raw.y), Act::round(raw.z), Act::round(raw.w));
    }
};

} // namespace routed
