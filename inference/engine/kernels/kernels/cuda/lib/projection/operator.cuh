// The bodies of the `dense` projection's forms for CUDA (`dense.seismic`),
// once for every entry that projects: an entry's kernel states its prologue
// (the rows it projects), its epilogue and its weights, and runs the form its
// launch names here. A projection is `paired` (two weights, `W0` and `W1`,
// one activation row serves, combined by the epilogue), `single` (one
// weight, `W0`) or `segmented` (below).
//
// The includer defines, before including this file:
//   KERNEL_W0 (and KERNEL_W1)   its weight slots, as `projection.cuh` takes them;
//   Pro                         its prologue type (`projection::Rms`,
//                               `Plain`, under `Operands` for NVFP4 weights);
//   PROJECTION_M, _K, _N, _G    the rows, the columns, each weight's rows (K a
//                               constant) and the groups of its weight (1
//                               where it has none): group g projects its own
//                               columns of the rows by its own N weight rows
//                               into its own N columns, at grid z, the forms
//                               that stage whole rows running only G = 1;
//   PROJECTION_ROWS             the prologue value (the rows as given);
//   PROJECTION_EPILOGUE         the epilogue value;
//   PROJECTION_WEIGHTS          `wa, wb` (`wb` is `projection::NoWeight{}`
//                               where single), the weights at their rows;
//   PROJECTION_WA, _WB          their types; PROJECTION_WEIGHT_TYPES (a
//                               `Types<..>`, a segment set's `Weights`) the
//                               weights whose operands the GEMMs take;
//   PROJECTION_INT8             `projection::int8_path<INT8, ...>()`'s
//                               admitted list: each weight's `form(W, INT8)`;
//   PROJECTION_PLACE, _PLACE_STRIDE
//                               the rows a GEMM reads in place when nothing
//                               stages them (the staging itself, K apart,
//                               where the prologue always does);
//   PROJECTION_STAGING, _GROUPS, _PARTIALS
//                               the scratch the projection declares.
// An entry whose weights take NVFP4 operands or whose prologue forms its rows
// declares the stage launch's output in `staged`.
//
// A plane set (`planes.cuh`) takes the stage launch at every row count and
// the `planes` form; its includer defines PROJECTION_M, _K, _N, _ROWS and
// _EPILOGUE as above, and
//   PROJECTION_PLANES_STAGED    the rows the stage launch forms and the
//                               `planes` launch reads (K apart), as an
//                               `element::Out<Act>` (scratch, or a result);
//   PROJECTION_PLANES_VIEW      the view its rows are projected onto;
//   PROJECTION_PLANES_FORMATION what its stage launch forms beside the rows.

#pragma once
#include "projection.cuh"
#include "planes.cuh"

namespace projection {
namespace op {

// Whether a GEMM reads its rows from the staging launch: the int8 and
// materialized operand paths always do, and so does a prologue that forms
// its rows. A function, not a class's constant (see `int8_path`).
template <class Pro, bool S8, class Operands> __host__ __device__ constexpr bool staged_rows() {
    return S8 || Operands::MATERIALIZE || Pro::STAGED;
}

// The weights' coefficient staging of a GEMV block over K columns, where
// STAGING is 1: only an unpaired staged weight stages (`gemv_mma_group`).
template <int STAGING, class Shape, u64 K, class WA, class WB> __host__ __device__ constexpr u64 gemv_staged_bytes() {
    return STAGING == 1 && Same<WB, NoWeight>::value ? gemv_staging<Shape, WA>(K / 64) : 0;
}

// The launch's grid where it states it (RESIDENCY 1 or 2), else 0.
template <int KSPLIT, int RESIDENCY, u64 N> __host__ __device__ constexpr u64 gemv_stated_grid() {
    return RESIDENCY == 0 ? 0 : gemv_grid<GemvShape<8, 1, KSPLIT, 1>>(N);
}

// The launch's blocks per SM, which its launch bounds admit.
template <int KSPLIT, int RESIDENCY, u64 N, class WA, class WB> __host__ __device__ constexpr int gemv_blocks() {
    using Shape = GemvShape<8, 1, KSPLIT, 1>;
    return gemv_min_blocks<Shape, WA, WB, gemv_stated_grid<KSPLIT, RESIDENCY, N>(), RESIDENCY>();
}

// The GEMV's rows once the previous launch is done: at M = 1 the block's
// (block-collective, `shared_row`, or with S8 (INT8 = 1) `shared_s8_row`),
// else the prologue's or stage launch's (with S8 `staged` codes and `groups`).
template <bool S8, bool ENTRIES, u64 K, class Pro>
__device__ __forceinline__ auto gemv_one_row_after_dependency(const Pro &pro) {
    if constexpr (S8)
        return shared_s8_row_after_dependency<ENTRIES>(pro, K);
    else
        return shared_row_after_dependency(pro, K);
}
template <bool S8, u64 K, class Pro>
__device__ __forceinline__ auto gemv_rows_after_dependency(const Pro &pro, const u8 *staged, const void *groups) {
    if constexpr (S8)
        return s8_rows_after_dependency(staged, groups, K);
    else
        return source_after_dependency(pro, K, staged);
}

// The GEMV over NB column blocks of 8 rows, a launch of GRID blocks (0: not
// stated, `GemvResidency`). Every warp runs it, a group past the rows
// included: the block first stages its weights' grid points (`stage_points`),
// then once the warps' first weight loads are in flight it waits for the
// previous launch, and reads its rows (`gemv_rows_after_dependency`).
template <int NB, int KSPLIT, int STAGING, u64 GRID, int RESIDENCY, u64 K, bool S8, class Pro, class WA, class WB,
          class Epi>
__device__ __forceinline__ void gemv(const Pro &pro, const u8 *staged, const void *groups, u32 M, u64 N,
                                     const WA &wa, const WB &wb, const Epi &epi) {
    using Shape = GemvShape<8, 1, KSPLIT, NB>;
    using Readers = typename GemvPointReaders<S8, WA, WB>::type;
    using Points = typename PointsOf<Readers>::type;
    __shared__ GemvShared<Shape, typename GemvSource<Pro>::type, gemv_staged_bytes<STAGING, Shape, K, WA, WB>(),
                          Points>
        shared;
    stage_points(Readers{});
    if constexpr (NB == 1)
        if (M == 1) {
            constexpr bool ENTRIES = s8_entries<S8, WA, WB>();
            gemv_segment_ready<Shape, GRID, RESIDENCY>(shared, gemv_one_row_after_dependency<S8, ENTRIES, K>(pro), 1u,
                                                       K / 64, Shape::tile_group(), N, wa, wb, epi);
            return;
        }
    gemv_segment_ready<Shape, GRID, RESIDENCY>(shared, gemv_rows_after_dependency<S8, K>(pro, staged, groups), M,
                                               K / 64, Shape::tile_group(), N, wa, wb, epi);
}

// The stage launch's row: the prologue's rows for the GEMV above (M <= 16),
// or the operands the matrix bands read (int8 rows in the GEMV's order,
// GEMV_ENTRIES, or the GEMM's, GEMM_ENTRIES).
template <bool S8, bool GEMV_ENTRIES, bool GEMM_ENTRIES, class Operands, class Pro>
__device__ __forceinline__ void stage(const Pro &pro, u32 M, u64 K, u8 *staged, void *groups) {
    seismic_dependency_start();
    stage_projection_row<S8, GEMV_ENTRIES, GEMM_ENTRIES, Operands>(pro, blockIdx.x, M, K, staged, groups);
}

// One GEMM block: K share `part` of `parts` of a single weight (published as
// partials when split, which `finalize` sums), or both weights of a pair
// whole.
template <class Shape, bool S8, bool ENTRIES, class Operands, class WA, class WB, class Epi>
__device__ __forceinline__ void gemm(u8 *shared, const u8 *rows, u64 stride, const void *groups, u32 M, u64 K, u64 N,
                                     const WA &wa, const WB &wb, const Epi &epi, float *partials, u64 part,
                                     u64 parts) {
    seismic_dependency_start();
    const u64 column = gemm_column<Shape>();
    if (column < gemm_columns(N)) {
        if constexpr (Same<WB, NoWeight>::value)
            gemm_run_split<Shape, S8, ENTRIES, Operands>(shared, rows, stride, groups, M, K, column, N, wa, epi,
                                                         partials, part, parts);
        else
            gemm_run<Shape, S8, ENTRIES, Operands>(shared, rows, stride, groups, M, K, column, N, wa, wb, epi);
    }
}

// The finalize launch of a split GEMM (SPLIT shares): each element summed
// over the K shares in part order, into the epilogue.
template <class Epi>
__device__ __forceinline__ void finalize(const float *partials, u64 parts, u32 M, u64 N, const Epi &epi) {
    seismic_dependency_start();
    split_finalize<1>(partials, parts, M, N, [&](unsigned m, unsigned long long n, float value, float) {
        epi(m, n, value, 0.0f);
    });
}


// ---------------------------------------------------------------------------
// The `segmented` variant: one to four segments, each of one weight or of a
// served pair (two weights of the same rows one activation row serves,
// combined by the segment's epilogue), projecting the rows into its own
// columns of the result, in order, grouped as the declaration groups them
// (`declared_served`). The includer names its `Segments` as a `SegmentSet`
// and builds it (`PROJECTION_SEGMENTS`) as `{rows, {weights},
// {epilogues}}`: `rows[COUNT]`, the weights in segment order
// (`packets::W0..`) and one epilogue a segment (their types may differ).

// A segment set is built of exactly what its segments have: `Members` take
// exactly one value for each of their types and `Rows` exactly COUNT, so an
// initializer that omits or adds a weight, an epilogue or a segment's rows does
// not compile.
template <class... Ts> struct Members {
    __host__ __device__ constexpr Members() {}
};
template <class T, class... Ts> struct Members<T, Ts...> {
    T head;
    Members<Ts...> tail;
    __host__ __device__ constexpr Members(T first, Ts... rest) : head(first), tail(rest...) {}
};
template <int I, class T, class... Ts> __host__ __device__ constexpr const auto &member(const Members<T, Ts...> &m) {
    if constexpr (I == 0)
        return m.head;
    else
        return member<I - 1>(m.tail);
}

template <int COUNT> struct Rows {
    unsigned long long value[COUNT];
    template <class... Is>
    __host__ __device__ constexpr Rows(Is... rows) : value{static_cast<unsigned long long>(rows)...} {
        static_assert(sizeof...(Is) == COUNT, "a segment set takes the rows of each of its segments");
    }
    __host__ __device__ constexpr unsigned long long operator[](int segment) const { return value[segment]; }
};

using projection::Types;

// The mask of the segments with no rows, from the row counts an entry gives its
// segment set (static dimensions: a count that is not known when the kernel is
// formed does not compile). The entry builds `EMPTY_` from the very counts it
// builds its set with, so the two cannot disagree.
template <class... Is> __host__ __device__ constexpr int empty_mask(Is... rows) {
    const u64 counts[] = {static_cast<u64>(rows)...};
    int mask = 0;
    for (int i = 0; i < static_cast<int>(sizeof...(Is)); ++i)
        mask |= counts[i] == 0 ? 1 << i : 0;
    return mask;
}

#if SEISMIC_PROJECTION_SEGMENTS
// The weights segment `segment` serves, as the declaration groups them
// (`project .. segmented`): the ABI prefix's, one or two, none past the last.
// No kernel states its grouping.
__host__ __device__ constexpr int declared_served(int segment) {
    return segment == 0   ? SEISMIC_SEGMENT_WEIGHTS_0
           : segment == 1 ? SEISMIC_SEGMENT_WEIGHTS_1
           : segment == 2 ? SEISMIC_SEGMENT_WEIGHTS_2
                          : SEISMIC_SEGMENT_WEIGHTS_3;
}

// The segments of a projection: one epilogue each (`Os`), serving the weights
// of `Ws` in segment order as the declaration groups them (`declared_served`),
// and ZEROES: 1 where a call may zero segments 0 and 1, which share an epilogue
// type, and EMPTY_: the segments with no rows (`empty_mask`; none by default).
template <int ZEROES_, class Ws, class Os, int EMPTY_ = 0> struct SegmentSet;
template <int ZEROES_, class... Ws, class... Os, int EMPTY_>
struct SegmentSet<ZEROES_, Types<Ws...>, Types<Os...>, EMPTY_> {
    enum { COUNT = sizeof...(Os), ZEROES = ZEROES_ };
    static_assert(COUNT == SEISMIC_PROJECTION_SEGMENTS, "a segment set has an epilogue for each declared segment");
    static_assert(sizeof...(Ws) == declared_served(0) + declared_served(1) + declared_served(2) + declared_served(3),
                  "a segment set holds the weights its declared segments serve");
    using Weights = Types<Ws...>;
    __host__ __device__ static constexpr int served(int segment) { return declared_served(segment); }
    // Whether segment `segment` has no rows, known when the kernel is formed:
    // its body is not compiled, as nothing can reach it.
    __host__ __device__ static constexpr bool empty(int segment) { return (EMPTY_ >> segment) & 1; }
    Rows<COUNT> rows;
    Members<Ws...> w;
    Members<Os...> o;
    __host__ __device__ constexpr SegmentSet(Rows<COUNT> r, Members<Ws...> weights, Members<Os...> epilogues)
        : rows(r), w(weights), o(epilogues) {}
};
#endif

template <unsigned OPERAND, class List> struct OperandsOf;
template <unsigned OPERAND, class... Ws> struct OperandsOf<OPERAND, Types<Ws...>> {
    using type = projection::GemmOperandsOf<OPERAND, Ws...>;
};

template <int STAGING, class Shape, u64 K, class List> struct StagingOf;
template <int STAGING, class Shape, u64 K, class... Ws> struct StagingOf<STAGING, Shape, K, Types<Ws...>> {
    __host__ __device__ static constexpr u64 bytes() { return STAGING == 1 ? gemv_staging<Shape, Ws...>(K / 64) : 0; }
};

template <int I, class Segments> __device__ __forceinline__ const auto &weight_of(const Segments &s) {
    return member<I>(s.w);
}

// The first weight (in segment order) of segment I.
template <class Segments, int I> __host__ __device__ constexpr int first_weight() {
    int at = 0;
    for (int segment = 0; segment < I; ++segment)
        at += Segments::served(segment);
    return at;
}

template <int I, class Segments> __device__ __forceinline__ const auto &out_of(const Segments &s) {
    return member<I>(s.o);
}

// Zeroes the results of segment `segment` (0 or 1) in a block of rows and
// columns. Its rows and result are selected by value: indexing `Segments` at
// a runtime segment would place all of it, weights included, in local
// memory, written by every thread of every launch.
template <class Segments>
__device__ __forceinline__ void zero_segment(const Segments &s, int segment, unsigned M, unsigned row_start,
                                             unsigned row_count, u64 column_start, unsigned column_count,
                                             unsigned tid, unsigned threads) {
    const auto out = segment == 0 ? member<0>(s.o) : member<1>(s.o);
    const u64 columns = segment == 0 ? s.rows[0] : s.rows[1];
    for (unsigned item = tid; item < row_count * column_count; item += threads) {
        const unsigned row = row_start + item / column_count;
        const u64 column = column_start + item % column_count;
        if (row < M && column < columns)
            out(row, column, 0.0f, 0.0f);
    }
}

// One segment's GEMV warp group, over its weight or served pair.
template <int I, class Shape, class Ready, class Segments, class Shared>
__device__ __forceinline__ void gemv_segment_of(Shared &shared, const Ready &x, u32 M, u64 kblocks, u64 group,
                                                const Segments &s) {
    constexpr int A = first_weight<Segments, I>();
    if constexpr (Segments::served(I) == 2)
        gemv_segment_ready<Shape>(shared, x, M, kblocks, group, s.rows[I], weight_of<A>(s), weight_of<A + 1>(s),
                                  out_of<I>(s));
    else
        gemv_segment_ready<Shape>(shared, x, M, kblocks, group, s.rows[I], weight_of<A>(s), NoWeight{}, out_of<I>(s));
}

// The segmented GEMV's warp over its source `x` (block-collective at M = 1),
// a group past the last segment included, which has no rows.
template <class Shape, class Ready, class Segments, class Shared>
__device__ __forceinline__ void gemv_segmented_from(Shared &shared, const Ready &x, u32 M, u64 kblocks, u64 group,
                                                    int segment, const Segments &s, unsigned zero) {
    constexpr int LAST = Segments::COUNT - 1;
    constexpr int LAST_WEIGHT = first_weight<Segments, LAST>() + Segments::served(LAST) - 1;
    if constexpr (Segments::ZEROES) if (zero != 0 && segment < 2) {
        const auto zeroed = [&] {
            const auto source = x();
            zero_segment(s, segment, M, 0, M, group * Shape::TPW * 16, Shape::TPW * 16,
                         threadIdx.x % (32 * Shape::KSPLIT), 32 * Shape::KSPLIT);
            return source;
        };
        gemv_segment_ready<Shape>(shared, zeroed, M, kblocks, 0, 0, weight_of<LAST_WEIGHT>(s), NoWeight{},
                                  out_of<LAST>(s));
        return;
    }
    switch (segment) {
    case 0:
        if constexpr (!Segments::empty(0))
            gemv_segment_of<0, Shape>(shared, x, M, kblocks, group, s);
        break;
    case 1:
        if constexpr (Segments::COUNT > 1 && !Segments::empty(1))
            gemv_segment_of<1, Shape>(shared, x, M, kblocks, group, s);
        break;
    case 2:
        if constexpr (Segments::COUNT > 2 && !Segments::empty(2))
            gemv_segment_of<2, Shape>(shared, x, M, kblocks, group, s);
        break;
    case 3:
        if constexpr (Segments::COUNT > 3 && !Segments::empty(3))
            gemv_segment_of<3, Shape>(shared, x, M, kblocks, group, s);
        break;
    default:
        gemv_segment_ready<Shape>(shared, x, M, kblocks, 0, 0, weight_of<LAST_WEIGHT>(s), NoWeight{}, out_of<LAST>(s));
        break;
    }
}

// The segmented GEMV over NB column blocks of 8 rows. With `zero` != 0 the
// results of segments 0 and 1 are zeros: their groups store them and are
// otherwise past the last segment. It reads its rows as `gemv` does.
template <int NB, int KSPLIT, int STAGING, u64 K, bool S8, class Weights, class Pro, class Segments>
__device__ __forceinline__ void gemv_segmented(const Pro &pro, const u8 *staged, const void *rows, u32 M,
                                               const Segments &s, unsigned zero) {
    using Shape = GemvShape<8, 1, KSPLIT, NB>;
    using Points = typename PointsOf<Weights>::type;
    __shared__ GemvShared<Shape, typename GemvSource<Pro>::type, StagingOf<STAGING, Shape, K, Weights>::bytes(), Points>
        shared;
    stage_points(Weights{});
    const u64 kblocks = K / 64;
    u64 group = Shape::tile_group();
    u64 groups[Segments::COUNT];
#pragma unroll
    for (int i = 0; i < Segments::COUNT; ++i)
        groups[i] = gemv_segment_groups<Shape>(s.rows[i]);
    const int segment = locate_segment(group, groups);
    if constexpr (NB == 1)
        if (M == 1) {
            constexpr bool ENTRIES = s8_entries_of<S8>(Weights{});
            gemv_segmented_from<Shape>(shared, gemv_one_row_after_dependency<S8, ENTRIES, K>(pro), 1u, kblocks, group,
                                       segment, s, zero);
            return;
        }
    gemv_segmented_from<Shape>(shared, gemv_rows_after_dependency<S8, K>(pro, staged, rows), M, kblocks, group,
                               segment, s, zero);
}

// One segment's GEMM block, over its weight or served pair.
template <int I, class Shape, bool S8, bool ENTRIES, class Operands, class Segments>
__device__ __forceinline__ void gemm_segment_of(u8 *shared, const u8 *rows, const void *groups, u32 M, u64 K,
                                                u64 column, const Segments &s) {
    constexpr int A = first_weight<Segments, I>();
    if constexpr (Segments::served(I) == 2)
        gemm_run<Shape, S8, ENTRIES, Operands>(shared, rows, K, groups, M, K, column, s.rows[I], weight_of<A>(s),
                                               weight_of<A + 1>(s), out_of<I>(s));
    else
        gemm_run<Shape, S8, ENTRIES, Operands>(shared, rows, K, groups, M, K, column, s.rows[I], weight_of<A>(s),
                                               NoWeight{}, out_of<I>(s));
}

// The segmented GEMM of one row band over the staged rows: block column
// `gemm_column` of the segments in order.
template <class Shape, bool S8, bool ENTRIES, class Operands, class Segments>
__device__ __forceinline__ void gemm_segmented(u8 *shared, const u8 *rows, const void *groups, u32 M, u64 K,
                                               const Segments &s, unsigned zero) {
    seismic_dependency_start();
    u64 column = gemm_column<Shape>();
    u64 columns[Segments::COUNT];
#pragma unroll
    for (int i = 0; i < Segments::COUNT; ++i)
        columns[i] = gemm_columns(s.rows[i]);
    const int segment = locate_segment(column, columns);
    if constexpr (Segments::ZEROES) if (zero != 0 && segment < 2) {
        zero_segment(s, segment, M, gemm_band<Shape>() * Shape::BM, Shape::BM, column * 128, 128, threadIdx.x,
                     Shape::THREADS);
        return;
    }
    switch (segment) {
    case 0:
        if constexpr (!Segments::empty(0))
            gemm_segment_of<0, Shape, S8, ENTRIES, Operands>(shared, rows, groups, M, K, column, s);
        break;
    case 1:
        if constexpr (Segments::COUNT > 1 && !Segments::empty(1))
            gemm_segment_of<1, Shape, S8, ENTRIES, Operands>(shared, rows, groups, M, K, column, s);
        break;
    case 2:
        if constexpr (Segments::COUNT > 2 && !Segments::empty(2))
            gemm_segment_of<2, Shape, S8, ENTRIES, Operands>(shared, rows, groups, M, K, column, s);
        break;
    case 3:
        if constexpr (Segments::COUNT > 3 && !Segments::empty(3))
            gemm_segment_of<3, Shape, S8, ENTRIES, Operands>(shared, rows, groups, M, K, column, s);
        break;
    }
}

// One unpaired segment's split-K GEMM block into `partials` at `offset` among
// the segments' `total_cols` columns.
template <int I, class Shape, bool S8, bool ENTRIES, class Operands, class Segments>
__device__ __forceinline__ void gemm_segment_split_of(u8 *shared, const u8 *rows, const void *groups, u32 M, u64 K,
                                                      u64 column, const Segments &s, float *partials, u64 total_cols,
                                                      u64 offset, u64 part, u64 parts) {
    static_assert(Segments::served(I) == 1, "split segmented GEMM segments are unpaired");
    constexpr int A = first_weight<Segments, I>();
    gemm_run<Shape, S8, ENTRIES, Operands>(shared, rows, K, groups, M, K, column, s.rows[I], weight_of<A>(s),
                                           NoWeight{}, PartialStore<1>{partials, M, total_cols, offset, part}, part,
                                           parts);
}

// The segmented GEMM with optional split-K (`parts > 1`): when `parts == 1`,
// publishes directly through the segments' epilogues (`gemm_segmented`); when
// `parts > 1`, each K share writes its F32 partials for `finalize_segmented`.
template <class Shape, bool S8, bool ENTRIES, class Operands, class Segments>
__device__ __forceinline__ void gemm_segmented_split(u8 *shared, const u8 *rows, const void *groups, u32 M, u64 K,
                                                     const Segments &s, float *partials, u64 part, u64 parts,
                                                     unsigned zero) {
    if (parts <= 1) {
        gemm_segmented<Shape, S8, ENTRIES, Operands>(shared, rows, groups, M, K, s, zero);
        return;
    }
    seismic_dependency_start();
    u64 column = gemm_column<Shape>();
    u64 columns[Segments::COUNT];
    u64 offsets[Segments::COUNT];
    u64 total_cols = 0;
#pragma unroll
    for (int i = 0; i < Segments::COUNT; ++i) {
        columns[i] = gemm_columns(s.rows[i]);
        offsets[i] = total_cols;
        total_cols += s.rows[i];
    }
    const int segment = locate_segment(column, columns);
    switch (segment) {
    case 0:
        if constexpr (!Segments::empty(0))
            gemm_segment_split_of<0, Shape, S8, ENTRIES, Operands>(shared, rows, groups, M, K, column, s, partials,
                                                                   total_cols, offsets[0], part, parts);
        break;
    case 1:
        if constexpr (Segments::COUNT > 1 && !Segments::empty(1))
            gemm_segment_split_of<1, Shape, S8, ENTRIES, Operands>(shared, rows, groups, M, K, column, s, partials,
                                                                   total_cols, offsets[1], part, parts);
        break;
    case 2:
        if constexpr (Segments::COUNT > 2 && !Segments::empty(2))
            gemm_segment_split_of<2, Shape, S8, ENTRIES, Operands>(shared, rows, groups, M, K, column, s, partials,
                                                                   total_cols, offsets[2], part, parts);
        break;
    case 3:
        if constexpr (Segments::COUNT > 3 && !Segments::empty(3))
            gemm_segment_split_of<3, Shape, S8, ENTRIES, Operands>(shared, rows, groups, M, K, column, s, partials,
                                                                   total_cols, offsets[3], part, parts);
        break;
    }
}

// The finalize launch of a split-K segmented GEMM: sums each element over the
// `parts` K shares in exact part order and hands it to its segment's epilogue.
template <class Segments>
__device__ __forceinline__ void finalize_segmented(const float *partials, u64 parts, u32 M, const Segments &s) {
    seismic_dependency_start();
    u64 total_cols = 0;
#pragma unroll
    for (int i = 0; i < Segments::COUNT; ++i)
        total_cols += s.rows[i];
    split_finalize<1>(partials, parts, M, total_cols, [&](unsigned m, unsigned long long col, float value, float) {
        u64 offset = 0;
        if constexpr (!Segments::empty(0)) {
            if (col < s.rows[0]) {
                out_of<0>(s)(m, col, value, 0.0f);
                return;
            }
            offset += s.rows[0];
        }
        if constexpr (Segments::COUNT > 1 && !Segments::empty(1)) {
            if (col < offset + s.rows[1]) {
                out_of<1>(s)(m, col - offset, value, 0.0f);
                return;
            }
            offset += s.rows[1];
        }
        if constexpr (Segments::COUNT > 2 && !Segments::empty(2)) {
            if (col < offset + s.rows[2]) {
                out_of<2>(s)(m, col - offset, value, 0.0f);
                return;
            }
            offset += s.rows[2];
        }
        if constexpr (Segments::COUNT > 3 && !Segments::empty(3)) {
            if (col < offset + s.rows[3]) {
                out_of<3>(s)(m, col - offset, value, 0.0f);
                return;
            }
        }
    });
}

#ifdef PROJECTION_PARTS
// Parts: a segmented projection whose K is PARTS parts of PART_K columns, each
// part's rows of their own (its own norm group), splits its GEMV along K by
// part. Block tile group g serves part g / per (`per` a part's tile groups
// over the segments) over its window of PART_K columns, into F32 partial
// sums: output n of row m from part p at partials[(m * PARTS + p) * outputs
// + n], `outputs` the segments' rows in order. The last block to arrive adds
// every output's parts in part order through the segments' epilogues. The
// segments are unpaired. The includer includes `lib/core/arrive.cuh` and
// defines PROJECTION_PARTS before including this file.

template <class W> __device__ __forceinline__ Window<W> window_of(const W &w, u64 superblock) {
    return Window<W>(w, superblock);
}

// One part's partial sums of the segments' outputs, from column `column`.
struct PartPartial {
    float *out;
    u64 row;
    u64 column;
    __device__ __forceinline__ void operator()(unsigned m, u64 n, float projected, float) const {
        out[m * row + column + n] = projected;
    }
};

// One segment's GEMV warp group over its weight's window of the part, from
// the part's `shared_row` at one row, else from its rows `rows`.
template <int I, class Shape, class Row, class Rows, class Segments, class Shared>
__device__ __forceinline__ void gemv_part_of(Shared &shared, const Row &row, const Rows &rows, u32 M, u64 kblocks,
                                             u64 group, u64 window, const PartPartial &partial, const Segments &s) {
    static_assert(Segments::served(I) == 1, "a parts projection's segments are unpaired");
    const auto weight = window_of(weight_of<first_weight<Segments, I>()>(s), window);
    if constexpr (Shape::NB == 1)
        if (M == 1) {
            gemv_segment_ready<Shape>(shared, row, 1u, kblocks, group, s.rows[I], weight, NoWeight{}, partial);
            return;
        }
    gemv_segment_ready<Shape>(shared, rows, M, kblocks, group, s.rows[I], weight, NoWeight{}, partial);
}

// Every output's parts, added in part order, through its segment's epilogue.
template <int I, class Segments>
__device__ __forceinline__ void store_part_sum(const Segments &s, unsigned m, u64 n, float sum) {
    out_of<I>(s)(m, n, sum, 0.0f);
}

// The parts GEMV over NB column blocks of 8 rows. `row(part, first)` is part
// `part`'s one row as a GEMV ready callback giving its `shared_row` (`first`:
// the part's first block, which publishes what the part's prologue
// publishes), `rows(part)` its rows past one row as one giving the rows of
// prologue type Part.
template <int NB, int KSPLIT, int PARTS, u64 PART_K, class Part, class Row, class Rows, class Segments>
__device__ __forceinline__ void gemv_parts(const Row &row, const Rows &rows, float *partials, unsigned *arrivals,
                                           u32 M, const Segments &s) {
    using Shape = GemvShape<8, 1, KSPLIT, NB>;
    __shared__ GemvShared<Shape, typename GemvSource<Part>::type> shared;
    constexpr u64 kblocks = PART_K / 64;
    u64 extents[Segments::COUNT], groups[Segments::COUNT], firsts[Segments::COUNT];
    u64 outputs = 0, per = 0;
#pragma unroll
    for (int i = 0; i < Segments::COUNT; ++i) {
        extents[i] = s.rows[i];
        groups[i] = gemv_segment_groups<Shape>(s.rows[i]);
        firsts[i] = outputs;
        outputs += s.rows[i];
        per += groups[i];
    }
    const u64 tile_group = Shape::tile_group();
    const unsigned part = (unsigned)(tile_group / per);
    u64 group = tile_group % per;
    const bool first = group < Shape::GROUPS;
    const int segment = locate_segment(group, groups);
    const auto x = row(part, first);
    const auto xs = rows(part);
    const u64 window = part * (kblocks / 4);
    const auto partial = [&](int i) { return PartPartial{partials, PARTS * outputs, part * outputs + firsts[i]}; };
    switch (segment) {
    case 0:
        if constexpr (!Segments::empty(0))
            gemv_part_of<0, Shape>(shared, x, xs, M, kblocks, group, window, partial(0), s);
        break;
    case 1:
        if constexpr (Segments::COUNT > 1 && !Segments::empty(1))
            gemv_part_of<1, Shape>(shared, x, xs, M, kblocks, group, window, partial(1), s);
        break;
    case 2:
        if constexpr (Segments::COUNT > 2 && !Segments::empty(2))
            gemv_part_of<2, Shape>(shared, x, xs, M, kblocks, group, window, partial(2), s);
        break;
    case 3:
        if constexpr (Segments::COUNT > 3 && !Segments::empty(3))
            gemv_part_of<3, Shape>(shared, x, xs, M, kblocks, group, window, partial(3), s);
        break;
    }
    __shared__ unsigned last;
    if (!arrive::last(arrivals, gridDim.x, &last))
        return;
    for (u64 item = threadIdx.x; item < M * outputs; item += blockDim.x) {
        const unsigned m = (unsigned)(item / outputs);
        u64 n = item % outputs;
        float sum = 0.0f;
        for (unsigned p = 0; p < PARTS; ++p)
            sum += __ldcg(partials + (m * PARTS + p) * outputs + n);
        switch (locate_segment(n, extents)) {
        case 0:
            store_part_sum<0>(s, m, n, sum);
            break;
        case 1:
            if constexpr (Segments::COUNT > 1)
                store_part_sum<1>(s, m, n, sum);
            break;
        case 2:
            if constexpr (Segments::COUNT > 2)
                store_part_sum<2>(s, m, n, sum);
            break;
        case 3:
            if constexpr (Segments::COUNT > 3)
                store_part_sum<3>(s, m, n, sum);
            break;
        }
    }
}
#endif

} // namespace op
} // namespace projection

// ---------------------------------------------------------------------------
// The launches' bodies. An entry writes each launch's kernel
// `entry_<launch>` under its forming guard, with the launch's code parameters
// (`dense.seismic`) as its template and `SEISMIC_KERNEL_PARAMS` as its
// arguments.

#define PROJECTION_S8 (projection::int8_path<INT8, PROJECTION_INT8>())
// The order of the int8 rows the GEMV reads, its weights' own
// (`projection::s8_entries`), and the GEMM's (`projection::gemm_s8_entries`),
// in which the stage launch forms the GEMM's rows.
#define PROJECTION_S8_ENTRIES (projection::s8_entries_of<PROJECTION_S8>(PROJECTION_WEIGHT_TYPES{}))
#define PROJECTION_GEMM_ENTRIES (projection::gemm_s8_entries_of<PROJECTION_S8>(PROJECTION_WEIGHT_TYPES{}))
#define PROJECTION_OPERANDS typename projection::op::OperandsOf<OPERAND, PROJECTION_WEIGHT_TYPES>::type

// The group grid z names: literally 0 at G = 1, so every ungrouped address
// and split part compiles to its ungrouped form.
#define PROJECTION_GROUP(z) (PROJECTION_G == 1 ? 0u : (unsigned)(z))

// `stage`: the rows each GEMM reads (group 0's, the forms that stage whole
// rows run only G = 1), from the entry's prologue, or from the rows `ROWS`
// where the rows are `staged` (formed from a source of their own).
#define PROJECTION_STAGE_OF(ROWS)                                                                                  \
    projection::op::stage<PROJECTION_S8, PROJECTION_S8_ENTRIES, PROJECTION_GEMM_ENTRIES, PROJECTION_OPERANDS>(     \
        ROWS, PROJECTION_M, PROJECTION_K, PROJECTION_STAGING, PROJECTION_GROUPS)
#define PROJECTION_STAGE PROJECTION_STAGE_OF(PROJECTION_ROWS(0u))

// A GEMV launch over NB column blocks, group g at grid z (its kernel states
// `__launch_bounds__(256, PROJECTION_GEMV_BLOCKS(KSPLIT, RESIDENCY))` itself,
// which the checker reads).
#define PROJECTION_GEMV_BLOCKS(KSPLIT, RESIDENCY)                                                                  \
    (projection::op::gemv_blocks<KSPLIT, RESIDENCY, PROJECTION_N, PROJECTION_WA, PROJECTION_WB>())
#define PROJECTION_GEMV(NB, KSPLIT, STAGING, GRID, RESIDENCY)                                                      \
    do {                                                                                                           \
        const unsigned group_ = PROJECTION_GROUP(blockIdx.z);                                                      \
        projection::op::gemv<NB, KSPLIT, STAGING, GRID, RESIDENCY, PROJECTION_K, false>(                           \
            PROJECTION_ROWS(group_), PROJECTION_STAGING, PROJECTION_GROUPS, (unsigned)PROJECTION_M, PROJECTION_N,  \
            PROJECTION_WEIGHTS(group_), PROJECTION_EPILOGUE(group_));                                              \
    } while (0)
#define PROJECTION_GEMV_GRID(KSPLIT, RESIDENCY)                                                                    \
    (projection::op::gemv_stated_grid<KSPLIT, RESIDENCY, PROJECTION_N>())

// A GEMM launch at grid z = group * splits + part: rows from the stage
// launch, or in place.
#define PROJECTION_GEMM(SHAPE, OPERANDS_)                                                                          \
    do {                                                                                                           \
        constexpr bool S8 = PROJECTION_S8;                                                                         \
        constexpr bool STAGED = projection::op::staged_rows<Pro, S8, OPERANDS_>();                                 \
        extern __shared__ uint4 dynamic_shared[];                                                                  \
        const unsigned splits_ = gridDim.z / PROJECTION_G;                                                         \
        const unsigned group_ = PROJECTION_GROUP(blockIdx.z / splits_);                                            \
        projection::op::gemm<SHAPE, S8, PROJECTION_GEMM_ENTRIES, OPERANDS_>(                                       \
            reinterpret_cast<projection::u8 *>(dynamic_shared),                                                    \
            STAGED ? PROJECTION_STAGING : PROJECTION_PLACE(group_),                                                \
            STAGED ? PROJECTION_K : PROJECTION_PLACE_STRIDE, PROJECTION_GROUPS, (unsigned)PROJECTION_M,            \
            PROJECTION_K, PROJECTION_N, PROJECTION_WEIGHTS(group_), PROJECTION_EPILOGUE(group_),                   \
            PROJECTION_PARTIALS(group_, splits_), blockIdx.z % splits_, splits_);                                  \
    } while (0)

#define PROJECTION_FINALIZE                                                                                        \
    do {                                                                                                           \
        const unsigned group_ = PROJECTION_GROUP(blockIdx.z);                                                      \
        projection::op::finalize(PROJECTION_PARTIALS(group_, SPLIT), SPLIT, PROJECTION_M, PROJECTION_N,            \
                                 PROJECTION_EPILOGUE(group_));                                                     \
    } while (0)

// A plane set's launches: `stage`, its rows (every row count), and `planes`,
// ROWS vocabulary rows a warp over the block's rows (grid y).
#define PROJECTION_PLANES_STAGE                                                                                    \
    projection::planes::stage(PROJECTION_ROWS(0u), PROJECTION_K, PROJECTION_PLANES_STAGED, PROJECTION_PLANES_FORMATION)
#define PROJECTION_PLANES(ROWS)                                                                                    \
    projection::planes::run<ROWS>(PROJECTION_PLANES_STAGED.stored(), (unsigned)PROJECTION_M, PROJECTION_N,        \
                                  PROJECTION_K,                                                                  \
                                  PROJECTION_PLANES_VIEW, PROJECTION_EPILOGUE(0u))

// A segmented projection's launches (`PROJECTION_SEGMENTS`: its `Segments`;
// `PROJECTION_ZERO`: nonzero where segments 0 and 1 are zeros for the call).
#define PROJECTION_SEGMENTED_GEMV(NB, KSPLIT, STAGING)                                                             \
    projection::op::gemv_segmented<NB, KSPLIT, STAGING, PROJECTION_K, false, PROJECTION_WEIGHT_TYPES>(             \
        PROJECTION_ROWS(0u), PROJECTION_STAGING, PROJECTION_GROUPS, (unsigned)PROJECTION_M, PROJECTION_SEGMENTS,   \
        (unsigned)PROJECTION_ZERO)
#define PROJECTION_SEGMENTED_GEMM(SHAPE, OPERANDS_)                                                                \
    do {                                                                                                           \
        constexpr bool S8 = PROJECTION_S8;                                                                         \
        extern __shared__ uint4 dynamic_shared[];                                                                  \
        projection::op::gemm_segmented<SHAPE, S8, PROJECTION_GEMM_ENTRIES, OPERANDS_>(                             \
            reinterpret_cast<projection::u8 *>(dynamic_shared), PROJECTION_STAGING, PROJECTION_GROUPS,             \
            (unsigned)PROJECTION_M, PROJECTION_K, PROJECTION_SEGMENTS, (unsigned)PROJECTION_ZERO);                 \
    } while (0)

// ---------------------------------------------------------------------------
// Reference-order (`ordered`) forms (`ordered_stage`, `ordered_gemv`,
// `ordered_gemm`): 8-lane dot products (`((l0+l4)+(l2+l6))+((l1+l5)+(l3+l7))`)
// matching the CPU reference order bit for bit.

namespace projection {
namespace op {

template <bool PUBLISH, u32 MAXM, class Pro, class WA, class Epi, class PubOut>
__device__ __forceinline__ void ordered_gemv_block_rows(u8 *shared_raw, const Pro &pro, PubOut pub_out, u64 pub_s0,
                                                        u64 pub_s1, bool publish_rows, u32 M, u64 K, u64 N,
                                                        u64 local_tile, const WA &wa, const Epi &epi) {
    float *smem_inv = reinterpret_cast<float *>(shared_raw);
    float *smem_x = reinterpret_cast<float *>(shared_raw + 256);
    const u32 tid = threadIdx.x;
    const u32 group_id = tid / 8u;
    const u32 u = tid & 7u;
    constexpr u32 GROUPS = 32u;
    pro.ordered_inverses(M, smem_inv, 0u, smem_x);
    const u64 w_row = local_tile * GROUPS + group_id;
    float acc[MAXM];
#pragma unroll
    for (u32 r = 0; r < MAXM; ++r)
        acc[r] = 0.0f;
    for (u64 k0 = 0; k0 < K; k0 += 256) {
        const u32 chunk = (u32)min(256ull, K - k0);
        const u32 chunk4 = chunk >> 2u;
        const u32 tail0 = chunk4 << 2u;
        const u32 row_step = blockDim.x >> 6u;
        const u32 r0 = tid >> 6u;
        const u32 c4 = tid & 63u;
        for (u32 r = r0; r < M; r += row_step) {
            const float inv = smem_inv[r];
            float *dst_row = smem_x + r * 256u;
            if (pro.can_vec4(r)) {
                if (c4 < chunk4) {
                    const u32 c = c4 << 2u;
                    float vals[4];
                    pro.ordered_at4(r, k0 + c, inv, vals);
                    reinterpret_cast<float4 *>(dst_row)[c4] = make_float4(vals[0], vals[1], vals[2], vals[3]);
                    if constexpr (PUBLISH) {
                        if (publish_rows) {
                            if (pub_s1 == 1ull) {
                                pub_out.template put_span<4>((u64)r * pub_s0 + k0 + c, vals);
                            } else {
#pragma unroll
                                for (u32 j = 0u; j < 4u; ++j)
                                    pub_out.put((u64)r * pub_s0 + (k0 + c + j) * pub_s1, vals[j]);
                            }
                        }
                    }
                }
                if (c4 < chunk - tail0) {
                    const u32 c = tail0 + c4;
                    const float val = pro.ordered_at(r, k0 + c, inv);
                    dst_row[c] = val;
                    if constexpr (PUBLISH) {
                        if (publish_rows)
                            pub_out.put((u64)r * pub_s0 + (k0 + c) * pub_s1, val);
                    }
                }
            } else {
                for (u32 c = c4; c < chunk; c += 64u) {
                    const float val = pro.ordered_at(r, k0 + c, inv);
                    dst_row[c] = val;
                    if constexpr (PUBLISH) {
                        if (publish_rows)
                            pub_out.put((u64)r * pub_s0 + (k0 + c) * pub_s1, val);
                    }
                }
            }
        }
        __syncthreads();
        if (w_row < N) {
            const u32 packets = (chunk + 31u) / 32u;
            for (u32 p = 0; p < packets; ++p) {
                float wv[4];
                packets::row_packet8(wa, w_row, N, k0 / 32 + p, K, u, wv);
                if (chunk == 256u) {
#pragma unroll
                    for (u32 s = 0; s < 4u; ++s) {
                        const u32 col = 32u * p + 8u * s + u;
#pragma unroll
                        for (u32 r = 0; r < MAXM; ++r) {
                            if (r < M)
                                acc[r] = seismic_fma_rn(smem_x[r * 256u + col], wv[s], acc[r]);
                        }
                    }
                } else {
#pragma unroll
                    for (u32 s = 0; s < 4u; ++s) {
                        const u32 col = 32u * p + 8u * s + u;
                        if (col < chunk) {
#pragma unroll
                            for (u32 r = 0; r < MAXM; ++r) {
                                if (r < M)
                                    acc[r] = seismic_fma_rn(smem_x[r * 256u + col], wv[s], acc[r]);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    if (w_row < N) {
#pragma unroll
        for (u32 r = 0; r < MAXM; ++r) {
            if (r < M) {
                const float sum = ordered_combine8(acc[r]);
                if (u == 0u)
                    epi(r, w_row, sum, 0.0f);
            }
        }
    }
}

template <bool PUBLISH, class Pro, class WA, class Epi, class PubOut>
__device__ __forceinline__ void ordered_gemv_block(u8 *shared_raw, const Pro &pro, PubOut pub_out, u64 pub_s0,
                                                   u64 pub_s1, bool publish_rows, u32 M, u64 K, u64 N, u64 local_tile,
                                                   const WA &wa, const Epi &epi) {
    if (M <= 1u)
        ordered_gemv_block_rows<PUBLISH, 1u>(shared_raw, pro, pub_out, pub_s0, pub_s1, publish_rows, M, K, N,
                                             local_tile, wa, epi);
    else if (M <= 2u)
        ordered_gemv_block_rows<PUBLISH, 2u>(shared_raw, pro, pub_out, pub_s0, pub_s1, publish_rows, M, K, N,
                                             local_tile, wa, epi);
    else if (M <= 4u)
        ordered_gemv_block_rows<PUBLISH, 4u>(shared_raw, pro, pub_out, pub_s0, pub_s1, publish_rows, M, K, N,
                                             local_tile, wa, epi);
    else
        ordered_gemv_block_rows<PUBLISH, 8u>(shared_raw, pro, pub_out, pub_s0, pub_s1, publish_rows, M, K, N,
                                             local_tile, wa, epi);
}

template <class WA, class Epi>
__device__ __forceinline__ void ordered_gemm_block(u8 *shared_raw, const u8 *rows, u64 row_stride, u64 col_stride,
                                                   u32 M, u64 K, u64 N, u64 tile_m, u64 tile_n, const WA &wa,
                                                   const Epi &epi) {
    float *cols_tile = reinterpret_cast<float *>(shared_raw);
    const u32 tid = threadIdx.x;
    const u32 sg = tid / 32u;
    const u32 lane = tid & 31u;
    const u32 gn = lane >> 3u;
    const u32 u = lane & 7u;
    const u32 r_dec = tid / 8u;
    const u32 act_lo = (u32)(tile_m * 64ull + sg * 4u);
    const u32 act_hi = act_lo + 32u;
    const bool has_lo = act_lo < M;
    const bool has_hi = act_hi < M;
    const bool has_cols = (tile_n * 32ull + 8ull * gn) < N;
    const u32 my_row0 = act_lo + gn;
    const u32 my_row1 = act_hi + gn;
    const u64 w_row = tile_n * 32ull + r_dec;
    float acc0[4][8], acc1[4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc0[i][j] = 0.0f;
            acc1[i][j] = 0.0f;
        }
    const u64 packets = (K + 31ull) / 32ull;
    if (packets == 0ull)
        return;
    {
        float wv[4];
        packets::row_packet8(wa, w_row, N, 0ull, K, u, wv);
#pragma unroll
        for (int s = 0; s < 4; ++s)
            cols_tile[(8u * (u32)s + u) * 36u + r_dec] = wv[s];
    }
    for (u64 p = 0; p < packets; ++p) {
        const u64 k0 = 32ull * p;
        float cur_x0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float cur_x1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (has_lo) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const u64 col = k0 + 8ull * (u64)s + u;
                cur_x0[s] =
                    (my_row0 < M && col < K) ? element::at<Act>(rows, (u64)my_row0 * row_stride + col * col_stride) : 0.0f;
            }
        }
        if (has_hi) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const u64 col = k0 + 8ull * (u64)s + u;
                cur_x1[s] =
                    (my_row1 < M && col < K) ? element::at<Act>(rows, (u64)my_row1 * row_stride + col * col_stride) : 0.0f;
            }
        }
        __syncthreads();
        const float *cur_cols = cols_tile + (p & 1ull) * 1152ull;
        float next_w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (p + 1ull < packets)
            packets::row_packet8(wa, w_row, N, p + 1ull, K, u, next_w);
        if (has_lo) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const float x0 = __shfl_sync(0xffffffffu, cur_x0[s], u);
                const float x1 = __shfl_sync(0xffffffffu, cur_x0[s], 8u + u);
                const float x2 = __shfl_sync(0xffffffffu, cur_x0[s], 16u + u);
                const float x3 = __shfl_sync(0xffffffffu, cur_x0[s], 24u + u);
                float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
                if (has_hi) {
                    y0 = __shfl_sync(0xffffffffu, cur_x1[s], u);
                    y1 = __shfl_sync(0xffffffffu, cur_x1[s], 8u + u);
                    y2 = __shfl_sync(0xffffffffu, cur_x1[s], 16u + u);
                    y3 = __shfl_sync(0xffffffffu, cur_x1[s], 24u + u);
                }
                if (has_cols && (k0 + 32ull <= K || k0 + 8ull * (u64)s + u < K)) {
                    const float4 *row_f4 = reinterpret_cast<const float4 *>(cur_cols + (8u * (u32)s + u) * 36u);
                    const float4 wc0 = row_f4[2u * gn];
                    const float4 wc1 = row_f4[2u * gn + 1u];
                    const float wc[8] = {wc0.x, wc0.y, wc0.z, wc0.w, wc1.x, wc1.y, wc1.z, wc1.w};
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc0[0][j] = seismic_fma_rn(x0, wc[j], acc0[0][j]);
                        acc0[1][j] = seismic_fma_rn(x1, wc[j], acc0[1][j]);
                        acc0[2][j] = seismic_fma_rn(x2, wc[j], acc0[2][j]);
                        acc0[3][j] = seismic_fma_rn(x3, wc[j], acc0[3][j]);
                    }
                    if (has_hi) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            acc1[0][j] = seismic_fma_rn(y0, wc[j], acc1[0][j]);
                            acc1[1][j] = seismic_fma_rn(y1, wc[j], acc1[1][j]);
                            acc1[2][j] = seismic_fma_rn(y2, wc[j], acc1[2][j]);
                            acc1[3][j] = seismic_fma_rn(y3, wc[j], acc1[3][j]);
                        }
                    }
                }
            }
        }
        if (p + 1ull < packets) {
            float *dst = cols_tile + ((p + 1ull) & 1ull) * 1152ull;
#pragma unroll
            for (int s = 0; s < 4; ++s)
                dst[(8u * (u32)s + u) * 36u + r_dec] = next_w[s];
        }
    }
    if (has_lo && has_cols) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const u32 out_r = act_lo + (u32)i;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float sum = ordered_combine8(acc0[i][j]);
                const u64 out_c = tile_n * 32ull + 8ull * gn + (u64)j;
                if (u == 0u && out_r < M && out_c < N)
                    epi(out_r, out_c, sum, 0.0f);
            }
        }
    }
    if (has_hi && has_cols) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const u32 out_r = act_hi + (u32)i;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float sum = ordered_combine8(acc1[i][j]);
                const u64 out_c = tile_n * 32ull + 8ull * gn + (u64)j;
                if (u == 0u && out_r < M && out_c < N)
                    epi(out_r, out_c, sum, 0.0f);
            }
        }
    }
}

template <int I> struct SegmentIndex { static constexpr int value = I; };

template <int I, bool PUBLISH, unsigned TPW, unsigned KSPLIT, class ProFn, class UnordPro, class PubOut, class Segments>
__device__ __forceinline__ void ordered_gemv_segment_of(u8 *shared_raw, const ProFn &pro_fn, const UnordPro &unord_pro,
                                                        const u8 *staged, PubOut pub_out, u64 pub_s0, u64 pub_s1,
                                                        bool publish_rows, u32 M, u64 K, u64 local, const Segments &s) {
    constexpr int A = first_weight<Segments, I>();
    if constexpr (Segments::served(I) == 2) {
        using Shape = GemvShape<8, TPW, KSPLIT, 1>;
        __shared__ GemvShared<Shape, typename GemvSource<UnordPro>::type, 0> gemv_shared;
        const u64 group = local * Shape::GROUPS + Shape::group();
        if (M == 1)
            gemv_segment_ready<Shape>(gemv_shared, [&] { return shared_row(unord_pro, K); }, 1u, K / 64, group,
                                      s.rows[I], weight_of<A>(s), weight_of<A + 1>(s), out_of<I>(s));
        else
            gemv_segment_ready<Shape>(gemv_shared, [&] { return GemvSource<UnordPro>::make(unord_pro, K, staged); },
                                      M, K / 64, group, s.rows[I], weight_of<A>(s), weight_of<A + 1>(s),
                                      out_of<I>(s));
    } else {
        ordered_gemv_block<PUBLISH>(shared_raw, pro_fn(SegmentIndex<I>{}), pub_out, pub_s0, pub_s1, publish_rows, M, K,
                                    s.rows[I], local, weight_of<A>(s), out_of<I>(s));
    }
}

template <bool PUBLISH, u64 K, unsigned TPW = 1, unsigned KSPLIT = 1, class ProFn, class UnordPro, class PubOut,
          class Segments>
__device__ __forceinline__ u64 ordered_gemv_segmented_each(u8 *shared_raw, const ProFn &pro_fn,
                                                           const UnordPro &unord_pro, const u8 *staged, PubOut pub_out,
                                                           u64 pub_s0, u64 pub_s1, u32 M, const Segments &s,
                                                           unsigned zero_mask) {
    seismic_dependency_start();
    constexpr u64 UNORD_PER = (8ull / KSPLIT) * (16ull * TPW);
    const bool seg_ord[4] = {SEISMIC_SEGMENT_ORDERED_0 != 0, SEISMIC_SEGMENT_ORDERED_1 != 0,
                             SEISMIC_SEGMENT_ORDERED_2 != 0, SEISMIC_SEGMENT_ORDERED_3 != 0};
    u64 tiles[Segments::COUNT];
    u64 ordered_tiles = 0;
#pragma unroll
    for (int i = 0; i < Segments::COUNT; ++i) {
        const u64 per = seg_ord[i] ? 32ull : UNORD_PER;
        tiles[i] = (s.rows[i] + per - 1ull) / per;
        if (seg_ord[i])
            ordered_tiles += tiles[i];
    }
    u64 local = blockIdx.x;
    const int segment = locate_segment(local, tiles);
    if constexpr (Segments::ZEROES) if ((zero_mask & (1u << segment)) != 0u) {
        const u32 per = seg_ord[segment] ? 32u : static_cast<u32>(UNORD_PER);
        zero_segment(s, segment, M, 0, M, local * per, per, threadIdx.x, blockDim.x);
        return ordered_tiles;
    }
    const bool pub = blockIdx.x == 0;
    switch (segment) {
    case 0:
        if constexpr (!Segments::empty(0))
            ordered_gemv_segment_of<0, PUBLISH, TPW, KSPLIT>(shared_raw, pro_fn, unord_pro, staged, pub_out, pub_s0,
                                                             pub_s1, pub, M, K, local, s);
        break;
    case 1:
        if constexpr (Segments::COUNT > 1 && !Segments::empty(1))
            ordered_gemv_segment_of<1, PUBLISH, TPW, KSPLIT>(shared_raw, pro_fn, unord_pro, staged, pub_out, pub_s0,
                                                             pub_s1, pub, M, K, local, s);
        break;
    case 2:
        if constexpr (Segments::COUNT > 2 && !Segments::empty(2))
            ordered_gemv_segment_of<2, PUBLISH, TPW, KSPLIT>(shared_raw, pro_fn, unord_pro, staged, pub_out, pub_s0,
                                                             pub_s1, pub, M, K, local, s);
        break;
    case 3:
        if constexpr (Segments::COUNT > 3 && !Segments::empty(3))
            ordered_gemv_segment_of<3, PUBLISH, TPW, KSPLIT>(shared_raw, pro_fn, unord_pro, staged, pub_out, pub_s0,
                                                             pub_s1, pub, M, K, local, s);
        break;
    }
    return ordered_tiles;
}

template <class Pro> struct SamePro {
    const Pro &pro;
    template <class Tag> __device__ __forceinline__ const Pro &operator()(Tag) const { return pro; }
};

template <bool PUBLISH, u64 K, unsigned TPW = 1, unsigned KSPLIT = 1, class Pro, class UnordPro, class PubOut,
          class Segments>
__device__ __forceinline__ u64 ordered_gemv_segmented(u8 *shared_raw, const Pro &pro, const UnordPro &unord_pro,
                                                      const u8 *staged, PubOut pub_out, u64 pub_s0, u64 pub_s1, u32 M,
                                                      const Segments &s, unsigned zero_mask) {
    return ordered_gemv_segmented_each<PUBLISH, K, TPW, KSPLIT>(shared_raw, SamePro<Pro>{pro}, unord_pro,
                                                                staged, pub_out, pub_s0, pub_s1, M, s, zero_mask);
}

template <int I, class Segments>
__device__ __forceinline__ void ordered_gemm_segment_of(u8 *shared_raw, const u8 *rows, u64 row_stride, u64 col_stride,
                                                        u32 M, u64 K, u64 local, const Segments &s) {
    constexpr int A = first_weight<Segments, I>();
    ordered_gemm_block(shared_raw, rows, row_stride, col_stride, M, K, s.rows[I], blockIdx.y, local, weight_of<A>(s),
                       out_of<I>(s));
}

template <class Segments>
__device__ __forceinline__ void ordered_gemm_segmented(u8 *shared_raw, const u8 *rows, u64 row_stride, u64 col_stride,
                                                       u32 M, u64 K, const Segments &s, unsigned zero) {
    seismic_dependency_start();
    u64 tiles[Segments::COUNT];
#pragma unroll
    for (int i = 0; i < Segments::COUNT; ++i)
        tiles[i] = (s.rows[i] + 31ull) / 32ull;
    u64 local = blockIdx.x;
    const int segment = locate_segment(local, tiles);
    if constexpr (Segments::ZEROES) if ((zero & (1u << segment)) != 0u) {
        zero_segment(s, segment, M, blockIdx.y * 64u, 64u, local * 32ull, 32u, threadIdx.x, blockDim.x);
        return;
    }
    switch (segment) {
    case 0:
        if constexpr (!Segments::empty(0))
            ordered_gemm_segment_of<0>(shared_raw, rows, row_stride, col_stride, M, K, local, s);
        break;
    case 1:
        if constexpr (Segments::COUNT > 1 && !Segments::empty(1))
            ordered_gemm_segment_of<1>(shared_raw, rows, row_stride, col_stride, M, K, local, s);
        break;
    case 2:
        if constexpr (Segments::COUNT > 2 && !Segments::empty(2))
            ordered_gemm_segment_of<2>(shared_raw, rows, row_stride, col_stride, M, K, local, s);
        break;
    case 3:
        if constexpr (Segments::COUNT > 3 && !Segments::empty(3))
            ordered_gemm_segment_of<3>(shared_raw, rows, row_stride, col_stride, M, K, local, s);
        break;
    }
}

} // namespace op
} // namespace projection
