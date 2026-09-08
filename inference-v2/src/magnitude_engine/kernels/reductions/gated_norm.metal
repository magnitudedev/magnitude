uint t = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
threadgroup float partial[32];
threadgroup float inverse;
float values[4];
float sum = 0.0f;
for (uint i = 0; i < 4; ++i) {
    values[i] = float(x[row * D + t * 4 + i]);
    sum += values[i] * values[i];
}
sum = simd_sum(sum);
if (sg == 0) partial[lane] = 0;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) partial[sg] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float total = simd_sum(partial[lane]);
    if (lane == 0) inverse = metal::precise::rsqrt(total / float(D) + EPS);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = 0; i < 4; ++i) {
    uint column = t * 4 + i;
    uint index = row * D + column;
    T normed = T(float(T(values[i] * inverse)) * float(w[column]));
    float z = float(gate[index]);
    float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
    float sigmoid = z < 0.0f ? e : 1.0f - e;
    out[index] = T((z * sigmoid) * float(normed));
}
