// The packed-rows GEMV (`PACKED_ROWS`), the GEMV row classes' form over the
// token-packing GEMM's operand: its own header, since only a single
// projection's `gemv_packed` launch runs it. The includer includes
// `packing.h` and `gemv.h` first.

namespace projection {
// Packed rows (`PACKED_ROWS` = N, 2 to VERIFY_ROWS rows; packing.h's lanes):
// N activation rows share one F32 operand, so one FMA of the chain carries N
// products. Per (row, run of 32 / folds columns) a row's values become integer
// codes under the held scale `packing_lane_scales` certifies for its lane
// (its sum limit against the weights' packed centre, with the run's packing
// bound where the weight has one, its codes against |x|_inf), and an element
// is the N rows' codes, `packing_element`. A run of a weight
// row accumulates A = sum_l 2^shift(l) S_l of its stored codes minus their
// centre (the PACK form's operand, from the decoder's codes) against the
// elements; `packing_split` gives each lane's sum, which folds into its row's
// result with the row's held scale and the packet's block factors
// (`packing_block_factors`: scale 2^15 / 64 and min term). A row's result
// depends on the rows it is packed with: the precision system admits the
// form per unit, at any row count.
//
// The staging pass is the GEMV's (`gemv_body`): once per threadgroup and
// chunk of packets, over every weight row of the threadgroup, the mapping
// space reaching 512 rows a threadgroup. A chunk holds a packet's elements
// (two float4 a step and group of N rows) and each row's held scale and code
// sum / 64 a run: `gemv_packed_packet_bytes` a packet.

// The packed operand of weights W: `available` for a decoder admitted to the
// packed form (`packing_codes`) whose codes enter without residue bits. Every
// admitted decoder's lane codes enter as its values + PACK_OFFSET
// (`lane_codes`), its values its codes less their zero code (`C::zero`,
// `packets::centered_codes`), so the operand is the decoder's codes moved by
// PACK_OFFSET - PACK_CENTRE - C::zero, one add.
template <typename W>
struct gemv_packed_codes {
    static constant constexpr bool available = false;
};
template <typename P, typename C, typename K, typename F>
struct gemv_packed_codes<packets::Packet<P, C, K, F>> {
    typedef packets::Packet<P, C, K, F> W;
    static constant constexpr bool available = packing_codes<W>::available && F::pack_residue == 0;
    static constant constexpr uint centre = F::pack_centre, folds = K::groups;
    typedef packing_terms<K, F::pack_centre, F::pack_offset> terms;
    // Step `step`'s eight codes less the centre, exact small integers.
    static void codes(thread const typename W::packet &k, uint step, thread float4 &even, thread float4 &odd) {
        C::template codes<P>(k.codes, step, even, odd);
        constexpr float shift = float(int(F::pack_offset) - int(F::pack_centre) - int(C::zero));
        even += shift;
        odd += shift;
    }
    // Run `run`'s factors: its scale 2^15 / 64 and its min term.
    static float2 factors(thread const typename W::packet &k, uint run) {
        const typename terms::entry entry = packing_block_factors<K, F::pack_centre, F::pack_offset>(k.coefficients);
        if constexpr (terms::sums)
            return run == 0u ? entry.xy : entry.zw;
        else if constexpr (folds == 2)
            return float2(entry[run], 0.0f);
        else
            return entry;
    }
};

// A staged packet's bytes: per step and group of N rows two float4 of
// elements, per row and run a float2 (held scale, code sum / 64).
template <uint N, uint MAXM, uint FOLDS>
constexpr uint gemv_packed_packet_bytes() {
    return 4u * 2u * ((MAXM + N - 1u) / N) * 16u + MAXM * FOLDS * 8u;
}

// Stage packets first .. first + count of MAXM rows as elements of N rows
// (rows at or past m_rows code zero), each element's held scales and its
// bottom lane's code step `packing_lane_scales`' over its rows' norms and,
// with BOUNDS, the run's packing bound. One item a (packet, step, group);
// every thread takes every round, since a run's norms cross its items' lanes
// (four consecutive items a packet, two a run of 16).
template <uint N, uint MAXM, uint FOLDS, uint CENTRE, uint BOUNDS, typename In>
inline void gemv_packed_stage(thread const In &in, thread const packing_bounds<BOUNDS> &bounds, uint m_rows,
    uint first, uint count, threadgroup float4 *elements, threadgroup float2 *tokens, uint thread_index,
    uint threads) {
    typedef packing_lanes<N> lanes;
    constexpr uint G = (MAXM + N - 1u) / N, RUN = 4u / FOLDS;
    const uint total = 4u * G * count;
    for (uint base = 0; base < total; base += threads) {
        const uint item = base + thread_index;
        const uint step = item & 3u, packet = item >> 2;
        const uint group = packet / count, local = packet - group * count;
        const float squares = BOUNDS != 0 && item < total ? bounds.at((first + local) * FOLDS + step / RUN) : 0.0f;
        float4 values_even[N], values_odd[N], even[N], odd[N];
        float3 norms[N];
        PROJECTION_UNROLL
        for (uint l = 0; l < N; ++l) {
            const uint m = group * N + l;
            values_even[l] = values_odd[l] = float4(0.0f);
            if (item < total && m < m_rows)
                in.load8(m, 32u * (first + local) + 8u * step, 0.0f, values_even[l], values_odd[l]);
            norms[l] = packing_norms(values_even[l], values_odd[l], FOLDS);
        }
        float held[N];
        const float bottom = packing_lane_scales<lanes, N, FOLDS, (BOUNDS != 0)>(norms, squares, CENTRE, held);
        PROJECTION_UNROLL
        for (uint l = 0; l < N; ++l) {
            const uint m = group * N + l;
            const float sum =
                packing_code(values_even[l], values_odd[l], held[l], l == 0 ? bottom : 1.0f, FOLDS, even[l], odd[l]);
            if (item < total && m < MAXM && step % RUN == 0u)
                tokens[(m * FOLDS + step / RUN) * count + local] = float2(held[l], sum * (1.0f / 64.0f));
        }
        if (item < total) {
            elements[gemv_word(local, step, 2u * group, count)] = packing_element<N>(even) * (1.0f / lanes::element);
            elements[gemv_word(local, step, 2u * group + 1u, count)] = packing_element<N>(odd) * (1.0f / lanes::element);
        }
    }
}

// Accumulate one packet of R weight rows into the MAXM rows' results: a run's
// chain per (weight row, group), then its split and fold.
template <typename W, uint N, uint R, uint MAXM>
inline void gemv_packed_accumulate(thread float (&acc)[R][MAXM], thread const typename W::packet (&packets)[R],
    uint local, uint count, threadgroup const float4 *elements, threadgroup const float2 *tokens) {
    typedef gemv_packed_codes<W> coded;
    constexpr uint G = (MAXM + N - 1u) / N, FOLDS = coded::folds, RUN = 4u / FOLDS;
    float sums[R][G];
    PROJECTION_UNROLL
    for (uint step = 0; step < 4; ++step) {
        float4 we[R], wo[R];
        PROJECTION_UNROLL
        for (uint r = 0; r < R; ++r)
            coded::codes(packets[r], step, we[r], wo[r]);
        PROJECTION_UNROLL
        for (uint g = 0; g < G; ++g) {
            const float4 xe = elements[gemv_word(local, step, 2u * g, count)];
            const float4 xo = elements[gemv_word(local, step, 2u * g + 1u, count)];
            PROJECTION_UNROLL
            for (uint r = 0; r < R; ++r)
                sums[r][g] = gemv_step_dot(step % RUN == 0u ? 0.0f : sums[r][g], we[r], wo[r], xe, xo);
        }
        if ((step + 1u) % RUN != 0u)
            continue;
        const uint run = step / RUN;
        PROJECTION_UNROLL
        for (uint r = 0; r < R; ++r) {
            const float2 factors = coded::factors(packets[r], run);
            PROJECTION_UNROLL
            for (uint g = 0; g < G; ++g) {
                float lanes[N];
                packing_split<N>(sums[r][g], lanes);
                PROJECTION_UNROLL
                for (uint l = 0; l < N; ++l) {
                    const uint m = g * N + l;
                    if (m < MAXM) {
                        const float2 token = tokens[(m * FOLDS + run) * count + local];
                        acc[r][m] = metal::fma(token.x, metal::fma(lanes[l], factors.x, token.y * factors.y), acc[r][m]);
                    }
                }
            }
        }
    }
}

// The packed-rows GEMV of one threadgroup: `gemv_body`'s mapping, staging
// chunks and register double-buffering, with the packed staging and chain.
template <typename W, uint N, uint R, uint MAXM, uint LANES, uint BOUNDS, typename In, typename Out>
inline void gemv_packed(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const packing_bounds<BOUNDS> &bounds, uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    typedef gemv_packed_codes<W> coded;
    static_assert(coded::available, "PACKED_ROWS names weights whose packed codes enter without residue bits");
    static_assert(N == 2 || N == 3, "packing_lanes holds two or three lanes");
    static_assert(LANES == 32 || LANES == 16 || LANES == 8, "a GEMV lane group is 8, 16 or 32 lanes");
    constexpr uint G = (MAXM + N - 1u) / N, FOLDS = coded::folds;
    constexpr uint bytes = gemv_packed_packet_bytes<N, MAXM, FOLDS>();
    uint group = lane / LANES, sub = lane % LANES;
    uint first_row = ((tile * simdgroups + sg) * (32u / LANES) + group) * R;
    bool active = first_row < rows;
    uint packets = (k + 31u) / 32u;
    gemv_rows<W, W, false, R, false> at;
    gemv_packets<W, W, false, R> current, next;
    if (active) {
        at.locate(w, w, first_row, rows);
        if (sub < packets)
            current.load(w, w, at, sub);
    }
    typename early<Out>::values loaded = {};
    if (gemv_emit<false>::template ahead<false, R, MAXM, LANES>() && active && sub < R * MAXM
        && sub % MAXM < m_rows && first_row + sub / MAXM < rows)
        loaded = early<Out>::load(out, sub % MAXM, first_row + sub / MAXM);
    const uint chunk = min(packets, (gemv_stage_packet_rows * gemv_packet_row_bytes / bytes) / LANES * LANES);
    threadgroup float4 *elements = reinterpret_cast<threadgroup float4 *>(shared);
    threadgroup float2 *tokens = reinterpret_cast<threadgroup float2 *>(shared + chunk * 4u * 2u * G * 16u);
    float acc[R][MAXM];
    PROJECTION_UNROLL
    for (uint r = 0; r < R; ++r)
        PROJECTION_UNROLL
        for (uint m = 0; m < MAXM; ++m)
            acc[r][m] = 0.0f;
    for (uint first = 0; first < packets; first += chunk) {
        uint count = min(chunk, packets - first);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemv_packed_stage<N, MAXM, FOLDS, coded::centre, BOUNDS>(in, bounds, m_rows, first, count, elements, tokens,
            sg * 32u + lane, simdgroups * 32u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint local = sub; local < count; local += LANES) {
                if (first + local + LANES < packets)
                    next.load(w, w, at, first + local + LANES);
                gemv_packed_accumulate<W, N, R, MAXM>(acc, current.a, local, count, elements, tokens);
                current = next;
            }
        }
    }
    gemv_emit<false>::run<false, R, MAXM, LANES>(out, acc, acc, m_rows, rows, first_row, sub, loaded);
}

// The packed rows' bound in {2, 4, 8}: their staging holds MAXM rows.
#define PROJECTION_FOR_PACKED_ROWS(rows, ...)                                   \
    do {                                                                        \
        if ((rows) <= 2) { constexpr uint MAXM = 2; __VA_ARGS__; }              \
        else if ((rows) <= 4) { constexpr uint MAXM = 4; __VA_ARGS__; }         \
        else { constexpr uint MAXM = 8; __VA_ARGS__; }                          \
    } while (0)

} // namespace projection
