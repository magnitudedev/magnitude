// The GEMV row classes of the packed projection family (M <= 16 activation
// rows, below an entry's BATCH_FROM): the GEMV over lane groups, its walk by
// packet class and its form over row tiles (`projection.h` describes the
// family, its row classes and the in-threadgroup prologue they share with the
// batched GEMV).
//
// Its own header so that a library that runs no GEMV (a GEMM's, a batched
// GEMV's) compiles none of this text: a library is compiled whole, and what
// it does not run should not be part of it. The includer includes
// `projection.h` first.

namespace projection {
// ---------------------------------------------------------------------------
// GEMV.
//
// Each weight is decoded once to F32 (`fma(scale, code, bias)`, one rounding,
// the dequantized value of the portable body) and multiplies every activation
// row with one FMA, in column order. A lane's sum therefore depends only on
// the packets it owns (LANES), never on M, R or SG: every row of an M-row
// GEMV is bit-identical to the same row computed alone.

// Threadgroup memory of one GEMV, a `threadgroup uchar *` bound at
// [[threadgroup(0)]]: the staged activations. A GEMV launch over K columns
// and M rows declares
//   shared_bytes (min(ceil_div(K, 32) * B, 448) * 64)
// with B = min(M, 2) + 2 * min(M / 3, 1) + 4 * min(M / 5, 1), the row count
// rounded up to its instantiated bound MAXM (`PROJECTION_FOR_ROWS`): at most
// `gemv_stage_packet_rows` staged packet rows of `gemv_packet_row_bytes`.
constant constexpr uint gemv_stage_packet_rows = 448;
constant constexpr uint gemv_packet_row_bytes = 4 * 16;

// Stage packets first .. first + count of MAXM activation rows (rows at or
// past m_rows are zero); all threads of the threadgroup (whole simdgroups)
// take part, one thread per eight columns. The eight activation values of
// chunk-local packet p, step s (columns 32p + 8s ..) and row m are one uint4
// in A storage at words[(p * 4 + s) * MAXM + m], so a lane reads the rows of
// its step consecutively.
template <uint MAXM, typename In>
inline void gemv_stage(thread const In &in, uint m_rows, uint first, uint count, threadgroup uint4 *words,
    uint thread_index, uint threads) {
    typedef typename In::activation A;
    for (uint item = thread_index; item < 4u * MAXM * count; item += threads) {
        uint step = item & 3u, packet = item >> 2;
        uint m = packet / count, local = packet - m * count;
        float4 even = float4(0.0f), odd = float4(0.0f);
        if (m < m_rows)
            in.load8(m, 32u * (first + local) + 8u * step, 0.0f, even, odd);
        words[(local * 4u + step) * MAXM + m] = A::pack8(even, odd);
    }
}

// The R weight rows of a lane group (and, paired, of the second tensor)
// from `first_row` (a multiple of R below `rows`), located once
// (`Weights::locate`), rows of one tile from the first
// (`Weights::together`). DERIVED: the group's one row is named instead, and
// its lanes derive it at each load.
template <typename W, typename U, bool PAIRED, uint R, bool DERIVED = false>
struct gemv_rows {
    typename Weights<W>::located a[R];
    typename Weights<U>::located b[R];
    uint only;
    void locate(thread const Weights<W> &w, thread const Weights<U> &u, uint first_row, uint rows) {
        static_assert(!DERIVED || R == 1u, "a lane derives one row");
        if (DERIVED) {
            only = min(first_row, rows - 1);
            return;
        }
        const bool a_together = w.template together<R>(rows);
        const bool b_together = PAIRED && u.template together<R>(rows);
        PROJECTION_UNROLL
        for (uint r = 0; r < R; ++r) {
            uint n = min(first_row + r, rows - 1);
            if (r > 0 && a_together)
                a[r] = Weights<W>::after(a[0], r);
            else
                a[r] = w.locate(n);
            if (PAIRED) {
                if (r > 0 && b_together)
                    b[r] = Weights<U>::after(b[0], r);
                else
                    b[r] = u.locate(n);
            }
        }
    }
};

// One packet of R weight rows (and, paired, of R rows of the second tensor).
template <typename W, typename U, bool PAIRED, uint R>
struct gemv_packets {
    typename W::packet a[R];
    typename U::packet b[R];
    template <bool DERIVED>
    void load(thread const Weights<W> &w, thread const Weights<U> &u,
        thread const gemv_rows<W, U, PAIRED, R, DERIVED> &at, uint p) {
        if (DERIVED) {
            a[0] = w.packet(at.only, p);
            if (PAIRED)
                b[0] = u.packet(at.only, p);
            return;
        }
        PROJECTION_UNROLL
        for (uint r = 0; r < R; ++r) {
            a[r] = w.packet(at.a[r], p);
            if (PAIRED)
                b[r] = u.packet(at.b[r], p);
        }
    }
};

// acc += the eight products of one step, in column order.
inline float gemv_step_dot(float acc, float4 we, float4 wo, float4 xe, float4 xo) {
    PROJECTION_UNROLL
    for (uint j = 0; j < 4; ++j) {
        acc = metal::fma(we[j], xe[j], acc);
        acc = metal::fma(wo[j], xo[j], acc);
    }
    return acc;
}

// Accumulate one loaded packet (chunk-local `local`) into the accumulators of
// all MAXM staged rows.
template <typename W, typename U, bool PAIRED, uint R, uint MAXM, typename A>
inline void gemv_accumulate(thread float (&acc)[R][MAXM], thread float (&acc2)[R][MAXM],
    thread const gemv_packets<W, U, PAIRED, R> &packets, uint local, threadgroup const uint4 *words) {
    PROJECTION_UNROLL
    for (uint step = 0; step < 4; ++step) {
        float4 ae[R], ao[R], be[R], bo[R];
        PROJECTION_UNROLL
        for (uint r = 0; r < R; ++r) {
            W::values(packets.a[r], step, ae[r], ao[r]);
            if (PAIRED)
                U::values(packets.b[r], step, be[r], bo[r]);
        }
        threadgroup const uint4 *x = words + (local * 4u + step) * MAXM;
        PROJECTION_UNROLL
        for (uint m = 0; m < MAXM; ++m) {
            float4 xe, xo;
            A::split8(x[m], xe, xo);
            PROJECTION_UNROLL
            for (uint r = 0; r < R; ++r) {
                acc[r][m] = gemv_step_dot(acc[r][m], ae[r], ao[r], xe, xo);
                if (PAIRED)
                    acc2[r][m] = gemv_step_dot(acc2[r][m], be[r], bo[r], xe, xo);
            }
        }
    }
}

// Weight rows of one GEMV threadgroup (segmented entries map threadgroups to
// segments by it).
template <uint SG, uint R, uint LANES>
constexpr uint gemv_threadgroup_rows() {
    return SG * R * (32u / LANES);
}

// Sum of `value` over the LANES lanes of this lane's group.
template <uint LANES>
inline float gemv_group_sum(float value) {
    if (LANES == 32)
        return simd_sum(value);
    for (ushort offset = LANES / 2; offset > 0; offset >>= 1)
        value += simd_shuffle_xor(value, offset);
    return value;
}

// Publishes a GEMV lane group's sums. Plain and paired epilogues get (m, n)
// from the lane that owns it. A column epilogue (`store_column(m, n, sums)`,
// plain only) gets from that lane every activation row's sum of output n,
// which each lane of the group holds after the group sums: the convolution
// along the rows of `gated_delta_project_convolved`.
template <bool COLUMNS>
struct gemv_emit;
template <>
struct gemv_emit<false> {
    // Whether each lane emits at most one output, so the lane loads that
    // output's epilogue values ahead (`early`).
    template <bool PAIRED, uint R, uint MAXM, uint LANES>
    static constexpr bool ahead() {
        return !PAIRED && R * MAXM <= LANES;
    }
    template <bool PAIRED, uint R, uint MAXM, uint LANES, typename Out>
    static void run(thread const Out &out, thread float (&acc)[R][MAXM], thread float (&acc2)[R][MAXM],
        uint m_rows, uint rows, uint first_row, uint sub, thread const typename early<Out>::values &loaded) {
        for (uint r = 0; r < R; ++r) {
            for (uint m = 0; m < MAXM; ++m) {
                if (m < m_rows) {
                    float total = gemv_group_sum<LANES>(acc[r][m]);
                    float total2 = PAIRED ? gemv_group_sum<LANES>(acc2[r][m]) : 0.0f;
                    if (sub == (r * MAXM + m) % LANES && first_row + r < rows) {
                        if constexpr (ahead<PAIRED, R, MAXM, LANES>())
                            early<Out>::store(out, m, first_row + r, total, loaded);
                        else
                            emit<PAIRED>::run(out, m, first_row + r, total, total2);
                    }
                }
            }
        }
    }
};
template <>
struct gemv_emit<true> {
    template <bool PAIRED, uint R, uint MAXM, uint LANES>
    static constexpr bool ahead() {
        return false;
    }
    template <bool PAIRED, uint R, uint MAXM, uint LANES, typename Out>
    static void run(thread const Out &out, thread float (&acc)[R][MAXM], thread float (&)[R][MAXM],
        uint m_rows, uint rows, uint first_row, uint sub, thread const typename early<Out>::values &) {
        static_assert(!PAIRED, "a column epilogue is plain");
        for (uint r = 0; r < R; ++r) {
            float column[MAXM];
            for (uint m = 0; m < MAXM; ++m) {
                column[m] = 0.0f;
                if (m < m_rows)
                    column[m] = gemv_group_sum<LANES>(acc[r][m]);
            }
            for (uint m = 0; m < MAXM; ++m) {
                if (m < m_rows && sub == (r * MAXM + m) % LANES && first_row + r < rows)
                    out.store_column(m, first_row + r, column);
            }
        }
    }
};

// One GEMV threadgroup of SG simdgroups. A simdgroup is 32 / LANES lane
// groups; lane group g of simdgroup sg owns the R weight rows from
// ((tile * SG + sg) * (32 / LANES) + g) * R, and its lanes own the packets
// sub, sub + LANES, ... of those rows. The prologue output is staged per
// threadgroup in chunks of at most `gemv_stage_packet_rows / MAXM` packets
// (a multiple of LANES). Weight packets are double-buffered in registers: a
// lane loads its next packet before accumulating the current one, and its
// first packet before the staging, so the weight stream never waits on it.
// `first_packet` is the stored packet of the GEMV's packet 0: a GEMV over a
// range of whole packets of the weight rows (K split by stream in
// `stream_mix_down`) reads packets first_packet .. first_packet +
// ceil(k / 32) - 1, the rows' stored length unchanged. It is the call's
// argument, not a field of the weight view, so no other reader of a view
// (the GEMMs among them) carries it.
template <typename W, typename U, bool PAIRED, uint R, uint MAXM, uint LANES, bool COLUMNS = false, typename In,
    typename Out>
inline void gemv_body(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint simdgroups, uint sg, uint lane, uint first_packet = 0) {
    static_assert(LANES == 32 || LANES == 16 || LANES == 8, "a GEMV lane group is 8, 16 or 32 lanes");
    typedef typename In::activation A;
    uint group = lane / LANES, sub = lane % LANES;
    uint first_row = ((tile * simdgroups + sg) * (32u / LANES) + group) * R;
    bool active = first_row < rows;
    uint packets = (k + 31u) / 32u;
    // A lane group of one row derives it at each load: across four activation
    // rows' accumulators the located row costs more than the arithmetic
    // (`Weights::locate`).
    gemv_rows<W, U, PAIRED, R, (R == 1u)> at;
    gemv_packets<W, U, PAIRED, R> current, next;
    if (active) {
        at.locate(w, u, first_row, rows);
        if (sub < packets)
            current.load(w, u, at, first_packet + sub);
    }
    // The epilogue values of the output this lane emits, with its first
    // packet (`early`): lane sub emits output r * MAXM + m == sub.
    typename early<Out>::values loaded = {};
    if (gemv_emit<COLUMNS>::template ahead<PAIRED, R, MAXM, LANES>() && active && sub < R * MAXM
        && sub % MAXM < m_rows && first_row + sub / MAXM < rows)
        loaded = early<Out>::load(out, sub % MAXM, first_row + sub / MAXM);
    uint chunk = min(packets, (gemv_stage_packet_rows / MAXM) / LANES * LANES);
    threadgroup uint4 *words = reinterpret_cast<threadgroup uint4 *>(shared);
    float acc[R][MAXM], acc2[R][MAXM];
    for (uint r = 0; r < R; ++r)
        for (uint m = 0; m < MAXM; ++m)
            acc[r][m] = acc2[r][m] = 0.0f;
    for (uint first = 0; first < packets; first += chunk) {
        uint count = min(chunk, packets - first);
        // A threadgroup may run several GEMVs in turn: the previous chunk's or
        // call's reads of the threadgroup memory finish before this one
        // writes it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemv_stage<MAXM>(in, m_rows, first, count, words, sg * 32u + lane, simdgroups * 32u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint local = sub; local < count; local += LANES) {
                if (first + local + LANES < packets)
                    next.load(w, u, at, first_packet + first + local + LANES);
                gemv_accumulate<W, U, PAIRED, R, MAXM, A>(acc, acc2, current, local, words);
                current = next;
            }
        }
    }
    gemv_emit<COLUMNS>::template run<PAIRED, R, MAXM, LANES>(out, acc, acc2, m_rows, rows, first_row, sub, loaded);
}

template <typename W, uint SG, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared, uint sg, uint lane) {
    gemv_body<W, W, false, R, MAXM, LANES>(in, out, w, w, m_rows, rows, k, tile, shared, SG, sg, lane);
}

template <typename W, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv_runtime(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane, uint first_packet = 0) {
    gemv_body<W, W, false, R, MAXM, LANES>(in, out, w, w, m_rows, rows, k, tile, shared,
        simdgroups, sg, lane, first_packet);
}

// `gemv_runtime` with a column epilogue (`gemv_emit<true>`).
template <typename W, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv_columns_runtime(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    gemv_body<W, W, false, R, MAXM, LANES, true>(in, out, w, w, m_rows, rows, k, tile, shared,
        simdgroups, sg, lane);
}

template <typename G, typename U, uint SG, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv_paired(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint sg, uint lane) {
    gemv_body<G, U, true, R, MAXM, LANES>(in, out, gate, up, m_rows, rows, k, tile, shared, SG, sg, lane);
}

template <typename G, typename U, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv_paired_runtime(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    gemv_body<G, U, true, R, MAXM, LANES>(in, out, gate, up, m_rows, rows, k, tile, shared,
        simdgroups, sg, lane);
}

// ---------------------------------------------------------------------------
// The row mapping's GEMV walked by packet class (`rows32` weights): a form
// selected by the launch, with the row mapping's results.
//
// The threadgroup keeps the row mapping's SG * R * 32 / LANES weight rows
// and its LANES partial sums per row (class c: packets c, c + LANES, ... in
// column order), but its lanes are regrouped: with `span` = SG * 32 / LANES
// row groups in the threadgroup, lanes b * span .. b * span + span - 1 of
// simdgroup sg are those row groups at class sg + b * SG. A simdgroup's
// lanes are then consecutive rows of a tile at the same packet, so its code
// loads are runs of adjacent bytes in lane order, where the row mapping asks
// for LANES separate cache lines a kilobyte apart at every step (on the M1,
// `dense_expand` over Q8_0 at one row: 438 against 420 µs).
//
// A row's class sums then sit in different simdgroups. `gemv_group_sum`
// pairs classes c and c + LANES / 2 first, then by halving offsets: the
// levels down to offset SG pair lanes of one simdgroup, and the levels below
// pair simdgroups, each a round of the threadgroup memory. Every result is
// therefore bit-identical to the row mapping's with the same LANES. The
// rounds cost a barrier each, which the walk repays for one or two
// activation rows on the M1 and not on the M4 Pro.

// Whether the class walk serves: R consecutive rows of a tile per lane, a
// class per simdgroup and lane block, lane groups `gemv_group_sum` adds by
// its own tree (32 lanes are `simd_sum`'s order, which the walk cannot
// reproduce), and the staged activations' threadgroup memory holding a float
// per row, activation row and tensor of every simdgroup.
template <typename W, typename U, bool PAIRED, uint R, uint MAXM, uint LANES>
inline bool gemv_classes_serve(thread const Weights<W> &w, thread const Weights<U> &u, uint rows, uint packets,
    uint simdgroups) {
    return w.template together<R>(rows) && (!PAIRED || u.template together<R>(rows)) && LANES < 32u &&
        simdgroups <= LANES &&
        8u * MAXM * simdgroups * (simdgroups * 32u / LANES) * R <=
            min(packets * MAXM, gemv_stage_packet_rows) * gemv_packet_row_bytes;
}

template <typename W, typename U, bool PAIRED, uint R, uint MAXM, uint LANES, typename In, typename Out>
inline void gemv_classes(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    typedef typename In::activation A;
    uint threads = simdgroups * 32u, thread_index = sg * 32u + lane;
    uint span = threads / LANES;
    uint sub = sg + (lane / span) * simdgroups, within = lane % span;
    uint first_row = (tile * span + within) * R;
    bool active = first_row < rows;
    uint packets = (k + 31u) / 32u;
    gemv_rows<W, U, PAIRED, R> at;
    gemv_packets<W, U, PAIRED, R> current, next;
    if (active) {
        at.locate(w, u, first_row, rows);
        if (sub < packets)
            current.load(w, u, at, sub);
    }
    uint chunk = min(packets, (gemv_stage_packet_rows / MAXM) / LANES * LANES);
    threadgroup uint4 *words = reinterpret_cast<threadgroup uint4 *>(shared);
    float acc[R][MAXM], acc2[R][MAXM];
    for (uint r = 0; r < R; ++r)
        for (uint m = 0; m < MAXM; ++m)
            acc[r][m] = acc2[r][m] = 0.0f;
    for (uint first = 0; first < packets; first += chunk) {
        uint count = min(chunk, packets - first);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemv_stage<MAXM>(in, m_rows, first, count, words, thread_index, threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint local = sub; local < count; local += LANES) {
                if (first + local + LANES < packets)
                    next.load(w, u, at, first + local + LANES);
                gemv_accumulate<W, U, PAIRED, R, MAXM, A>(acc, acc2, current, local, words);
                current = next;
            }
        }
    }
    // The levels that pair classes of one simdgroup.
    PROJECTION_UNROLL
    for (uint r = 0; r < R; ++r) {
        PROJECTION_UNROLL
        for (uint m = 0; m < MAXM; ++m) {
            for (ushort offset = 16; offset >= span; offset >>= 1) {
                acc[r][m] += simd_shuffle_xor(acc[r][m], offset);
                if (PAIRED)
                    acc2[r][m] += simd_shuffle_xor(acc2[r][m], offset);
            }
        }
    }
    // The levels that pair simdgroups: every simdgroup publishes its rows'
    // sums, and at the level of offset `o` simdgroup c < o adds simdgroup
    // c + o's and publishes the result for the next level. The sums stay in
    // the walk's registers. The staged activations are read before the sums
    // overwrite them.
    threadgroup float *sums = reinterpret_cast<threadgroup float *>(shared);
    const uint plane = simdgroups * span * R, mine = (sg * span + within) * R;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < span) {
        PROJECTION_UNROLL
        for (uint r = 0; r < R; ++r) {
            PROJECTION_UNROLL
            for (uint m = 0; m < MAXM; ++m) {
                if (m < m_rows) {
                    sums[(2u * m) * plane + mine + r] = acc[r][m];
                    if (PAIRED)
                        sums[(2u * m + 1u) * plane + mine + r] = acc2[r][m];
                }
            }
        }
    }
    for (uint offset = simdgroups / 2u; offset > 0u; offset >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < offset && lane < span) {
            const uint other = mine + offset * span * R;
            PROJECTION_UNROLL
            for (uint r = 0; r < R; ++r) {
                PROJECTION_UNROLL
                for (uint m = 0; m < MAXM; ++m) {
                    if (m < m_rows) {
                        acc[r][m] += sums[(2u * m) * plane + other + r];
                        sums[(2u * m) * plane + mine + r] = acc[r][m];
                        if (PAIRED) {
                            acc2[r][m] += sums[(2u * m + 1u) * plane + other + r];
                            sums[(2u * m + 1u) * plane + mine + r] = acc2[r][m];
                        }
                    }
                }
            }
        }
    }
    PROJECTION_UNROLL
    for (uint r = 0; r < R; ++r) {
        PROJECTION_UNROLL
        for (uint m = 0; m < MAXM; ++m) {
            if (m < m_rows && first_row + r < rows && sg == 0u && lane < span)
                emit<PAIRED>::run(out, m, first_row + r, acc[r][m], acc2[r][m]);
        }
    }
}

// The row GEMV of a launch whose form parameter may select the class walk
// (CLASSES): the walk where it serves, the row mapping otherwise.
template <typename W, typename U, bool PAIRED, uint R, uint MAXM, uint LANES, bool CLASSES, typename In,
    typename Out>
inline void gemv_form(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    if (CLASSES && gemv_classes_serve<W, U, PAIRED, R, MAXM, LANES>(w, u, rows, (k + 31u) / 32u, simdgroups)) {
        gemv_classes<W, U, PAIRED, R, MAXM, LANES>(in, out, w, u, m_rows, rows, k, tile, shared, simdgroups, sg,
            lane);
        return;
    }
    gemv_body<W, U, PAIRED, R, MAXM, LANES>(in, out, w, u, m_rows, rows, k, tile, shared, simdgroups, sg, lane);
}

// ---------------------------------------------------------------------------
// The GEMV over row tiles (`rows32` weights, M up to `gemv_tile_rows`): a
// form of its own, selected by the launch.
//
// A thread owns one weight row. The threadgroup keeps the row mapping's
// SG * R * 32 / LANES weight rows, so a row has LANES / R threads, thread
// t = sg * 32 + lane owning row t % rows-of-the-threadgroup: the lanes of a
// simdgroup are consecutive rows of one tile at the same packets, their code
// loads are adjacent, and they walk the packets in step, so every lane reads
// its row's coefficient planes once per run of packets (`packets::Block`).
//
// Summation order. A row's packets are split into LANES contiguous ranges,
// in column order and as evenly as they divide (the first packets % LANES
// ranges hold one packet more). A range is summed in column order from zero,
// and the LANES range sums are added pairwise (`gemv_pair_sum`: neighbours
// first). A thread owns R neighbouring ranges and adds them itself; the rest
// crosses the threadgroup, since a row's threads are in different
// simdgroups. The order depends on the row's packet count and LANES only,
// never on M or on how the ranges are staged: every row of an M-row call in
// this form is bit-identical to the same row computed alone in this form. It
// is not the row mapping's order.
//
// The activations are staged in chunks of the same window of every range
// (`gemv_chunk`), so all threads work in every chunk. The launch's
// threadgroup memory holds, besides the staged activations, one float per
// thread and tensor: a launch of SG simdgroups declares
//   shared_bytes (max(<the GEMV staging above>, SG * 128))
// (SG * 256 for a paired GEMV).

// Activation rows the form serves. From 3 rows up it loses to the row
// mapping on q4k and q8 (M6: 136 against 119 µs at 3 rows, both tuned).
constant constexpr uint gemv_tile_rows = 2;

// A row's packets split into `ranges` contiguous ranges: range s holds the
// `length(s)` packets from `start(s)`.
struct gemv_split {
    uint ranges;
    uint base;    // packets of every range
    uint extra;   // ranges holding one more
    uint start(uint range) const { return range * base + min(range, extra); }
    uint length(uint range) const { return base + (range < extra ? 1u : 0u); }
    uint longest() const { return base + min(extra, 1u); }
};

// One staged chunk: the at most `window` packets from `offset` of every
// range, a range's packets in consecutive slots and the ranges in order.
struct gemv_chunk {
    gemv_split split;
    uint offset;
    uint longer;    // staged packets of a range holding one more
    uint shorter;   // staged packets of the other ranges
    uint count(uint range) const { return range < split.extra ? longer : shorter; }
    uint slot(uint range) const {
        return range < split.extra ? range * longer : split.extra * longer + (range - split.extra) * shorter;
    }
    uint slots() const { return split.extra * longer + (split.ranges - split.extra) * shorter; }
    // The row packet staged in `slot`.
    uint packet(uint slot) const {
        uint head = split.extra * longer;
        uint range, at;
        if (slot < head) {
            range = slot / longer;
            at = slot - range * longer;
        } else {
            range = (slot - head) / shorter;
            at = slot - head - range * shorter;
            range += split.extra;
        }
        return split.start(range) + offset + at;
    }
};
inline gemv_chunk gemv_chunk_at(gemv_split split, uint offset, uint window) {
    uint longer = split.base + 1u > offset ? min(split.base + 1u - offset, window) : 0u;
    uint shorter = split.base > offset ? min(split.base - offset, window) : 0u;
    return gemv_chunk{split, offset, longer, shorter};
}

// Stage a chunk of MAXM activation rows as `gemv_stage` does, slot q of the
// chunk at words[(q * 4 + s) * MAXM + m].
template <uint MAXM, typename In>
inline void gemv_stage_chunk(thread const In &in, uint m_rows, gemv_chunk chunk, threadgroup uint4 *words,
    uint thread_index, uint threads) {
    typedef typename In::activation A;
    uint count = chunk.slots();
    for (uint item = thread_index; item < 4u * MAXM * count; item += threads) {
        uint step = item & 3u, packet = item >> 2;
        uint m = packet / count, slot = packet - m * count;
        float4 even = float4(0.0f), odd = float4(0.0f);
        if (m < m_rows)
            in.load8(m, 32u * chunk.packet(slot) + 8u * step, 0.0f, even, odd);
        words[(slot * 4u + step) * MAXM + m] = A::pack8(even, odd);
    }
}

// The sum of COUNT values in a pairwise tree, neighbours first.
template <uint COUNT>
inline float gemv_pair_sum(thread float (&v)[COUNT]) {
    PROJECTION_UNROLL
    for (uint width = 1; width < COUNT; width *= 2) {
        PROJECTION_UNROLL
        for (uint i = 0; i < COUNT; i += 2 * width)
            v[i] += v[i + width];
    }
    return v[0];
}

// One contiguous range of a row's packets, walked in order: the packet being
// accumulated, the one loaded ahead of it, and the row's coefficient run.
template <typename W>
struct gemv_walk {
    uint next;   // the packet loaded ahead
    uint end;
    typename W::packet current, ahead;
    typename packets::Block<W>::state run;
};

// Load the walk's next packet ahead (none past the range's end) and step.
template <typename W>
inline void gemv_walk_load(thread gemv_walk<W> &walk, thread const Weights<W> &w,
    thread const typename Weights<W>::located &at, bool first) {
    if (walk.next < walk.end) {
        if (first || walk.next % packets::Block<W>::packets == 0u)
            walk.run = w.run(at, walk.next);
        walk.ahead = w.packet(at, walk.next, walk.run);
    }
    walk.next += 1u;
}

template <typename W, typename U, bool PAIRED, uint R, uint MAXM, uint LANES, typename In, typename Out>
inline void gemv_tile_row(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    static_assert(LANES == 32 || LANES == 16 || LANES == 8, "a GEMV row has 8, 16 or 32 packet ranges");
    static_assert(LANES % R == 0, "a thread owns a whole share of its row's ranges");
    static_assert(gemv_stage_packet_rows / MAXM >= LANES, "a chunk stages a packet of every range");
    typedef typename In::activation A;
    constexpr uint sharers = LANES / R;
    uint threads = simdgroups * 32u, thread_index = sg * 32u + lane;
    uint span = threads / sharers;
    uint within = thread_index % span, share = thread_index / span;
    // A thread past the last weight row computes that row and emits nothing.
    bool active = tile * span + within < rows;
    uint n = min(tile * span + within, rows - 1u);
    uint packets = (k + 31u) / 32u;
    gemv_split split{LANES, packets / LANES, packets % LANES};
    typename Weights<W>::located a_at = w.locate(n);
    typename Weights<U>::located b_at = u.locate(n);
    gemv_walk<W> a[R];
    gemv_walk<U> b[R];
    PROJECTION_UNROLL
    for (uint j = 0; j < R; ++j) {
        uint range = share * R + j;
        a[j].next = split.start(range);
        a[j].end = a[j].next + split.length(range);
        gemv_walk_load(a[j], w, a_at, true);
        if (PAIRED) {
            b[j].next = a[j].next - 1u;
            b[j].end = a[j].end;
            gemv_walk_load(b[j], u, b_at, true);
        }
    }
    constexpr uint window = (gemv_stage_packet_rows / MAXM) / LANES;
    threadgroup uint4 *words = reinterpret_cast<threadgroup uint4 *>(shared);
    float acc[R][MAXM], acc2[R][MAXM];
    PROJECTION_UNROLL
    for (uint j = 0; j < R; ++j) {
        PROJECTION_UNROLL
        for (uint m = 0; m < MAXM; ++m)
            acc[j][m] = acc2[j][m] = 0.0f;
    }
    for (uint offset = 0; offset < split.longest(); offset += window) {
        gemv_chunk chunk = gemv_chunk_at(split, offset, window);
        // A threadgroup may run several GEMVs in turn: the previous chunk's or
        // call's reads of the threadgroup memory finish before this one
        // writes it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemv_stage_chunk<MAXM>(in, m_rows, chunk, words, thread_index, threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint at = 0; at < (split.extra == 0u ? chunk.shorter : chunk.longer); ++at) {
            PROJECTION_UNROLL
            for (uint j = 0; j < R; ++j) {
                uint range = share * R + j;
                if (at < chunk.count(range)) {
                    a[j].current = a[j].ahead;
                    gemv_walk_load(a[j], w, a_at, false);
                    if (PAIRED) {
                        b[j].current = b[j].ahead;
                        gemv_walk_load(b[j], u, b_at, false);
                    }
                    threadgroup const uint4 *x = words + (chunk.slot(range) + at) * 4u * MAXM;
                    PROJECTION_UNROLL
                    for (uint step = 0; step < 4; ++step) {
                        float4 ae, ao, be, bo;
                        W::values(a[j].current, step, ae, ao);
                        if (PAIRED)
                            U::values(b[j].current, step, be, bo);
                        PROJECTION_UNROLL
                        for (uint m = 0; m < MAXM; ++m) {
                            float4 xe, xo;
                            A::split8(x[step * MAXM + m], xe, xo);
                            acc[j][m] = gemv_step_dot(acc[j][m], ae, ao, xe, xo);
                            if (PAIRED)
                                acc2[j][m] = gemv_step_dot(acc2[j][m], be, bo, xe, xo);
                        }
                    }
                }
            }
        }
    }
    // Each activation row's sums cross the threadgroup in a round of their
    // own: every thread publishes its ranges' sum and the row's first thread
    // adds them.
    threadgroup float *sums = reinterpret_cast<threadgroup float *>(shared);
    PROJECTION_UNROLL
    for (uint m = 0; m < MAXM; ++m) {
        if (m < m_rows) {
            float own[R], own2[R];
            PROJECTION_UNROLL
            for (uint j = 0; j < R; ++j) {
                own[j] = acc[j][m];
                own2[j] = acc2[j][m];
            }
            // The staged activations (and the previous round) are read
            // before the sums overwrite them.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sums[thread_index] = gemv_pair_sum<R>(own);
            if (PAIRED)
                sums[threads + thread_index] = gemv_pair_sum<R>(own2);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active && share == 0u) {
                float part[sharers], part2[sharers];
                PROJECTION_UNROLL
                for (uint s = 0; s < sharers; ++s) {
                    part[s] = sums[within + s * span];
                    part2[s] = PAIRED ? sums[threads + within + s * span] : 0.0f;
                }
                emit<PAIRED>::run(out, m, n, gemv_pair_sum<sharers>(part), gemv_pair_sum<sharers>(part2));
            }
        }
    }
}

// Whether a GEMV of `m_rows` activation rows over `w` takes the tile form.
template <typename W>
inline bool gemv_tile_row_serves(thread const Weights<W> &w, uint m_rows) {
    return w.tile() > 1u && m_rows <= gemv_tile_rows;
}

// Instantiate the tile form for the activation-row bound MAXM in {1, 2}.
#define PROJECTION_FOR_TILE_ROWS(rows, ...)                                     \
    do {                                                                        \
        if ((rows) <= 1) { constexpr uint MAXM = 1; __VA_ARGS__; }              \
        else { constexpr uint MAXM = 2; __VA_ARGS__; }                          \
    } while (0)

// Instantiate a GEMV body for the activation-row bound MAXM in {1, 2, 4, 8}.
#define PROJECTION_FOR_ROWS(rows, ...)                                          \
    do {                                                                        \
        if ((rows) <= 1) { constexpr uint MAXM = 1; __VA_ARGS__; }              \
        else if ((rows) <= 2) { constexpr uint MAXM = 2; __VA_ARGS__; }         \
        else if ((rows) <= 4) { constexpr uint MAXM = 4; __VA_ARGS__; }         \
        else { constexpr uint MAXM = 8; __VA_ARGS__; }                          \
    } while (0)

// The same in the launches of an entry that launches its GEMV per row count,
// one for one activation row, one for two and one for three up to BATCH_FROM
// (at most 8): each instantiates only the bounds it runs. A kernel holds the
// registers of every bound it instantiates, and a kernel that also holds a
// wider body's runs slower for bounds it never takes (on the M1 10% at one
// row beside the 8-row body, and 10% at two rows beside the 4- and 8-row
// bodies: `dense_output` over Q4_K 254 against 281 µs). A wrapper body names
// its launch's count, PROJECTION_FOR_<count>_ROWS.
#define PROJECTION_FOR_ONE_ROWS(rows, ...)                                      \
    do { constexpr uint MAXM = 1; __VA_ARGS__; } while (0)
#define PROJECTION_FOR_ONE_TILE_ROWS(rows, ...) PROJECTION_FOR_ONE_ROWS(rows, __VA_ARGS__)
#define PROJECTION_FOR_PAIR_ROWS(rows, ...)                                     \
    do { constexpr uint MAXM = 2; __VA_ARGS__; } while (0)
#define PROJECTION_FOR_PAIR_TILE_ROWS(rows, ...) PROJECTION_FOR_PAIR_ROWS(rows, __VA_ARGS__)
#define PROJECTION_FOR_SEVERAL_ROWS(rows, ...)                                  \
    do {                                                                        \
        if ((rows) <= 4) { constexpr uint MAXM = 4; __VA_ARGS__; }              \
        else { constexpr uint MAXM = 8; __VA_ARGS__; }                          \
    } while (0)
#define PROJECTION_FOR_SEVERAL_TILE_ROWS(rows, ...)                             \
    do { constexpr uint MAXM = 2; __VA_ARGS__; } while (0)

} // namespace projection
