// A resident launch's three phases between grid barriers (route.seismic
// `resident_reverse`): threadgroup g owns elements g, g + G, ... of a grid
// of G (`threadgroups_per_grid`, the runtime's RESIDENT at most N), its
// threads one element each in turn. Its phases read other threadgroups'
// writes, so `x` and `y` are `coherent(device)`.

kernel void resident_reverse(
    device const int *skip [[buffer(SEISMIC_BUFFER_SKIP)]],
    coherent(device) device float *x [[buffer(SEISMIC_BUFFER_X)]],
    coherent(device) device float *y [[buffer(SEISMIC_BUFFER_Y)]],
    device atomic_uint *words [[buffer(SEISMIC_BUFFER_SCRATCH_BARRIER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup uint flag;
    seismic_grid::Barrier barrier = SEISMIC_GRID_BARRIER(RESIDENT_REVERSE, words, groups);
    const ulong n = SEISMIC_DIM_N;
    const ulong xs = SEISMIC_X_STRIDE_0, ys = SEISMIC_Y_STRIDE_0;
    for (ulong i = ulong(group) * threads + tid; i < n; i += ulong(groups) * threads)
        x[i * xs] = float(i) + 1.0f;
    if (!(skip[0] != 0 && group == 0) && seismic_grid::sync(barrier, &flag, tid)) {
        for (ulong i = ulong(group) * threads + tid; i < n; i += ulong(groups) * threads)
            y[i * ys] = x[(n - 1 - i) * xs] * 2.0f;
        if (seismic_grid::sync(barrier, &flag, tid)) {
            for (ulong i = ulong(group) * threads + tid; i < n; i += ulong(groups) * threads)
                x[i * xs] = y[(n - 1 - i) * ys] + 1.0f;
        }
    }
    seismic_grid::depart(barrier, tid);
}
