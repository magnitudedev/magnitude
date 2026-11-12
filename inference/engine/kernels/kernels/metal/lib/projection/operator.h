// The bodies of the `dense` projection's forms (`dense.seismic`), once
// for every entry that projects: an entry's kernel states its prologue (the
// rows `x` it projects), its epilogue (`out`) and its weights, and runs the
// form its launch names here. A projection is `paired` (two weights, `w0`
// and `w1`, one activation row serves, combined by `out.store_pair`) or
// `single` (one weight, `w0`, which the bodies also take as `w1`).
//
// The includer includes `projection.h` first, and `gemv.h` where it forms a
// GEMV launch.

// The GEMV of a launch that serves COUNT (ONE, PAIR, SEVERAL) rows of `x`:
// with TILED = 1 the row tiles where every weight's rows are tiled, else the
// row mapping (walked by packet class with TILED = 2). `n` is the rows of
// each weight, `k` their columns; the kernel provides `tile`, `shared`,
// `simdgroups`, `sg` and `lane`.
#define PROJECTION_GEMV(W0, W1, PAIRED, ROWS, LANES, TILED, COUNT, x, out, w0, w1, rows, n, k)                  \
    if (TILED == 1 && projection::gemv_tile_row_serves(w0, rows) &&                                             \
        (!(PAIRED) || projection::gemv_tile_row_serves(w1, rows))) {                                            \
        PROJECTION_FOR_##COUNT##_TILE_ROWS(rows,                                                                \
            projection::gemv_tile_row<W0, W1, PAIRED, ROWS, MAXM, LANES>(                                       \
                x, out, w0, w1, rows, n, k, tile, shared, simdgroups, sg, lane));                               \
        return;                                                                                                 \
    }                                                                                                           \
    PROJECTION_FOR_##COUNT##_ROWS(rows,                                                                         \
        projection::gemv_form<W0, W1, PAIRED, ROWS, MAXM, LANES, (TILED == 2)>(                                 \
            x, out, w0, w1, rows, n, k, tile, shared, simdgroups, sg, lane))

// The batched launch (BATCH_FROM..16 rows of `x`, `m` of them): with
// MATRIX = 1 the matrix GEMV, which the projection admits only where it
// serves every weight, else the batched rows in the first threadgroups of
// the launch.
#define PROJECTION_BATCH(W0, W1, PAIRED, BATCH_ROWS, BATCH_PARTS, MATRIX, x, out, w0, w1, m, n, k)               \
    if constexpr (MATRIX == 1) {                                                                                \
        if constexpr (PAIRED)                                                                                   \
            projection::gemv_matrix_paired<W0, W1, BATCH_PARTS>(x, out, w0, w1, m, n, k, tile, shared,          \
                simdgroups, sg, lane);                                                                          \
        else                                                                                                    \
            projection::gemv_matrix<W0, BATCH_PARTS>(x, out, w0, m, n, k, tile, shared, simdgroups, sg, lane);  \
        return;                                                                                                 \
    }                                                                                                           \
    if (tile * simdgroups * BATCH_ROWS * 8u >= n)                                                               \
        return;                                                                                                 \
    if constexpr (PAIRED)                                                                                       \
        projection::gemv_batch_paired_runtime<W0, W1, BATCH_ROWS>(x, out, w0, w1, m, n, k, tile, shared,        \
            simdgroups, sg, lane);                                                                              \
    else                                                                                                        \
        projection::gemv_batch_runtime<W0, BATCH_ROWS>(x, out, w0, m, n, k, tile, shared, simdgroups, sg, lane)
