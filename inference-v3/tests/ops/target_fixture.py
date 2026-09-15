"""Deterministic matrix-analysis doubles for host-only operation tests."""

def matrix_query(tiles):
    def query(dtype, m, n, k, threads, a_scope, b_scope, c_scope):
        return next((tile for tile in tiles if tile.input_dtype == dtype), None)
    return query
