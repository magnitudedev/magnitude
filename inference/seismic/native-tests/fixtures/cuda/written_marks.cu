// `written_marks` (conditional.seismic): the grid written, every mark
// cleared, then each block of the written grid marks its own (a
// written-grid body, run by the generated grid-stride kernel).

#ifdef SEISMIC_FORMING_WRITTEN_MARKS_GRID
__global__ void written_marks_grid(SEISMIC_KERNEL_PARAMS) {
    int *grid = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GRID));
    grid[0] = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COUNT))[0];
    grid[1] = 2;
    grid[2] = 1;
}
#endif

#ifdef SEISMIC_FORMING_WRITTEN_MARKS_CLEAR
__global__ void written_marks_clear(SEISMIC_KERNEL_PARAMS) {
    int *marks = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_MARKS));
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < SEISMIC_DIM_N * 2)
        marks[index / SEISMIC_DIM_N * SEISMIC_MARKS_STRIDE_0 + index % SEISMIC_DIM_N * SEISMIC_MARKS_STRIDE_1] = 0;
}
#endif

#ifdef SEISMIC_FORMING_WRITTEN_MARKS
__device__ void written_marks_block(SEISMIC_KERNEL_PARAMS) {
    int *marks = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_MARKS));
    if (threadIdx.x == 0 && gridDim.y == 2)
        marks[blockIdx.y * SEISMIC_MARKS_STRIDE_0 + blockIdx.x * SEISMIC_MARKS_STRIDE_1] = 1;
}
#endif
