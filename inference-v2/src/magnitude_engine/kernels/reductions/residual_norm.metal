// Match MLX RMS: four consecutive values per thread, then SIMD reductions.
// Wider rows use 1024 threads and retain the same ordered chunk traversal.
constexpr int ELEMS = 4;
constexpr int CHUNK = THREADS * ELEMS;
uint t   = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
threadgroup float part[32];
threadgroup float inv_sh[1];
const size_t rowbase = (size_t)row * D;
float xs[NCHUNK * ELEMS];
float ss2 = 0.0f;
_Pragma("clang loop unroll(full)") for (int c = 0; c < NCHUNK; ++c) {
    const uint idx = c * CHUNK + t * ELEMS;
    if (idx + ELEMS <= D) {
        _Pragma("clang loop unroll(full)") for (int i = 0; i < ELEMS; ++i) {
            float v = float(T(float(x[rowbase + idx + i]) + float(a[rowbase + idx + i])));
            xs[c * ELEMS + i] = v;
            xnew[rowbase + idx + i] = static_cast<T>(v);
            ss2 += v * v;
        }
    }
}
ss2 = simd_sum(ss2);
if (sg == 0) part[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) part[sg] = ss2;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float tot2 = simd_sum(part[lane]);
    if (lane == 0) inv_sh[0] = metal::precise::rsqrt(tot2 / float(D) + EPS);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv2 = inv_sh[0];
_Pragma("clang loop unroll(full)") for (int c = 0; c < NCHUNK; ++c) {
    const uint idx = c * CHUNK + t * ELEMS;
    if (idx + ELEMS <= D) {
        _Pragma("clang loop unroll(full)") for (int i = 0; i < ELEMS; ++i) {
            float n = static_cast<float>(static_cast<T>(xs[c * ELEMS + i] * inv2));
            normalized[rowbase + idx + i] = static_cast<T>(n * static_cast<float>(w1[idx + i]));
        }
    }
}
