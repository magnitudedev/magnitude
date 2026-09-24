
// ---------------------------------------------------------------------------
// Seismic CUDA device library. Every native CUDA source receives this prelude
// ahead of its generated ABI macros; sources compile under NVRTC with no
// vendor headers. Warp-collective helpers require all 32 lanes of the warp to
// execute them convergently. Global pointers are generic pointers into global
// memory; shared pointers are generic pointers into shared memory.
// ---------------------------------------------------------------------------

// Conversions (same rounding as the compiler's PTX emitter).
__device__ __forceinline__ float seismic_bf16_to_f32(unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}
__device__ __forceinline__ unsigned short seismic_f32_to_bf16(float value) {
    unsigned short result;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}
__device__ __forceinline__ float seismic_f16_to_f32(unsigned short value) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
    return result;
}
__device__ __forceinline__ unsigned short seismic_f32_to_f16(float value) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}

// Packed pairs: `lo` occupies bits 0..15 (the lower address in memory), `hi`
// bits 16..31. Packing rounds to nearest even.
__device__ __forceinline__ unsigned seismic_pack_bf16x2(float lo, float hi) {
    unsigned result;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(hi), "f"(lo));
    return result;
}
__device__ __forceinline__ unsigned seismic_pack_f16x2(float lo, float hi) {
    unsigned result;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(hi), "f"(lo));
    return result;
}
__device__ __forceinline__ float2 seismic_unpack_bf16x2(unsigned pair) {
    return make_float2(__uint_as_float(pair << 16), __uint_as_float(pair & 0xffff0000u));
}
__device__ __forceinline__ float2 seismic_unpack_f16x2(unsigned pair) {
    float lo, hi;
    asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.f32.f16 %0, l;\n\tcvt.f32.f16 %1, h;\n\t}"
        : "=f"(lo), "=f"(hi)
        : "r"(pair));
    return make_float2(lo, hi);
}

// Explicitly rounded arithmetic. Formation runs with `--fmad=false`, so a
// fused multiply-add happens only where a kernel writes `seismic_fma_rn`.
__device__ __forceinline__ float seismic_fma_rn(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}
__device__ __forceinline__ float seismic_mul_rn(float a, float b) {
    float result;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}
__device__ __forceinline__ float seismic_add_rn(float a, float b) {
    float result;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

// Hardware approximations (MUFU). Not correctly rounded: admitted only where
// an entry's precision gate allows them. Subnormals are preserved.
__device__ __forceinline__ float seismic_ex2_approx(float x) {
    float result;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}
__device__ __forceinline__ float seismic_lg2_approx(float x) {
    float result;
    asm("lg2.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}
__device__ __forceinline__ float seismic_rcp_approx(float x) {
    float result;
    asm("rcp.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}
__device__ __forceinline__ float seismic_rsqrt_approx(float x) {
    float result;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}
__device__ __forceinline__ float seismic_tanh_approx(float x) {
    float result;
    asm("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

// Address spaces.
__device__ __forceinline__ unsigned seismic_shared_address(const void* shared) {
    unsigned long long address;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(address) : "l"(shared));
    return (unsigned)address;
}

// Non-coherent (read-only path) global loads. The memory must not be written
// while the kernel runs. `_na` does not allocate in L1 (streamed data).
__device__ __forceinline__ unsigned seismic_ld_nc_u32(const void* global) {
    unsigned value;
    asm("ld.global.nc.u32 %0, [%1];" : "=r"(value) : "l"(global));
    return value;
}
__device__ __forceinline__ uint2 seismic_ld_nc_v2(const void* global) {
    uint2 value;
    asm("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(value.x), "=r"(value.y) : "l"(global));
    return value;
}
__device__ __forceinline__ uint4 seismic_ld_nc_v4(const void* global) {
    uint4 value;
    asm("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(global));
    return value;
}
__device__ __forceinline__ uint4 seismic_ld_nc_na_v4(const void* global) {
    uint4 value;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(global));
    return value;
}

// Prefetch the line holding `global` into L2.
__device__ __forceinline__ void seismic_prefetch_l2(const void* global) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(global));
}

// Asynchronous 16-byte global-to-shared copies (L2 only, bypassing L1). Both
// addresses are 16-byte aligned. `_zfill` copies `source_bytes` (0..16) bytes
// and writes zeros for the rest; `global` must still be a valid address.
// A copy is complete for the issuing thread after the wait that retires its
// group; other threads observe it only after a following barrier.
__device__ __forceinline__ void seismic_cp_async_16(void* shared, const void* global) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 ::"r"(seismic_shared_address(shared)), "l"(global)
                 : "memory");
}
__device__ __forceinline__ void seismic_cp_async_16_zfill(void* shared, const void* global,
                                                          unsigned source_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 ::"r"(seismic_shared_address(shared)), "l"(global), "r"(source_bytes)
                 : "memory");
}
__device__ __forceinline__ void seismic_cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
// Wait until at most `PENDING` committed groups are still in flight.
template <int PENDING>
__device__ __forceinline__ void seismic_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(PENDING) : "memory");
}
__device__ __forceinline__ void seismic_cp_async_wait_all() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}

// Warp shuffles over the full warp. Out-of-range `up`/`down` sources return
// the caller's own value.
__device__ __forceinline__ unsigned seismic_shfl_xor_u32(unsigned value, unsigned lane_mask) {
    unsigned result;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=r"(result) : "r"(value), "r"(lane_mask));
    return result;
}
__device__ __forceinline__ unsigned seismic_shfl_idx_u32(unsigned value, unsigned source_lane) {
    unsigned result;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=r"(result) : "r"(value), "r"(source_lane));
    return result;
}
__device__ __forceinline__ unsigned seismic_shfl_down_u32(unsigned value, unsigned delta) {
    unsigned result;
    asm volatile("shfl.sync.down.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=r"(result) : "r"(value), "r"(delta));
    return result;
}
__device__ __forceinline__ unsigned seismic_shfl_up_u32(unsigned value, unsigned delta) {
    unsigned result;
    asm volatile("shfl.sync.up.b32 %0, %1, %2, 0x0, 0xffffffff;"
                 : "=r"(result) : "r"(value), "r"(delta));
    return result;
}
__device__ __forceinline__ float seismic_shfl_xor_f32(float value, unsigned lane_mask) {
    return __uint_as_float(seismic_shfl_xor_u32(__float_as_uint(value), lane_mask));
}
__device__ __forceinline__ float seismic_shfl_idx_f32(float value, unsigned source_lane) {
    return __uint_as_float(seismic_shfl_idx_u32(__float_as_uint(value), source_lane));
}
__device__ __forceinline__ float seismic_shfl_down_f32(float value, unsigned delta) {
    return __uint_as_float(seismic_shfl_down_u32(__float_as_uint(value), delta));
}
__device__ __forceinline__ float seismic_shfl_up_f32(float value, unsigned delta) {
    return __uint_as_float(seismic_shfl_up_u32(__float_as_uint(value), delta));
}

// Butterfly reductions: every lane receives the result, combined in the fixed
// order xor 16, 8, 4, 2, 1 (bitwise identical on every lane and every run).
__device__ __forceinline__ float seismic_warp_sum_f32(float value) {
    for (unsigned mask = 16; mask > 0; mask >>= 1) {
        value = seismic_add_rn(value, seismic_shfl_xor_f32(value, mask));
    }
    return value;
}
__device__ __forceinline__ float seismic_warp_max_f32(float value) {
    for (unsigned mask = 16; mask > 0; mask >>= 1) {
        value = fmaxf(value, seismic_shfl_xor_f32(value, mask));
    }
    return value;
}

// Integer warp reductions in one instruction; every lane receives the result.
__device__ __forceinline__ unsigned seismic_redux_add_u32(unsigned value) {
    unsigned result;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(result) : "r"(value));
    return result;
}
__device__ __forceinline__ unsigned seismic_redux_min_u32(unsigned value) {
    unsigned result;
    asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(result) : "r"(value));
    return result;
}
__device__ __forceinline__ unsigned seismic_redux_max_u32(unsigned value) {
    unsigned result;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(result) : "r"(value));
    return result;
}
__device__ __forceinline__ int seismic_redux_add_s32(int value) {
    int result;
    asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;" : "=r"(result) : "r"(value));
    return result;
}
__device__ __forceinline__ int seismic_redux_min_s32(int value) {
    int result;
    asm volatile("redux.sync.min.s32 %0, %1, 0xffffffff;" : "=r"(result) : "r"(value));
    return result;
}
__device__ __forceinline__ int seismic_redux_max_s32(int value) {
    int result;
    asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(result) : "r"(value));
    return result;
}

// Four-way byte dot products accumulated into `acc`: bytes of `a` and `b` in
// little-endian order, signed (s8) or unsigned (u8).
__device__ __forceinline__ int seismic_dp4a_s8(int a, int b, int acc) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(acc));
    return result;
}
__device__ __forceinline__ int seismic_dp4a_u8s8(unsigned a, int b, int acc) {
    int result;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(acc));
    return result;
}

// ldmatrix: 8x8 matrices of 16-bit elements from shared memory. Lane
// 8*i + r supplies the 16-byte-aligned address of row r of matrix i (lanes
// 0..7 for x1, 0..15 for x2, all for x4). Register i of lane l receives, from
// matrix i, row l/4, columns 2*(l%4) and 2*(l%4)+1 (low half first); with
// `_trans`, column l/4, rows 2*(l%4) and 2*(l%4)+1.
__device__ __forceinline__ void seismic_ldmatrix_x1(unsigned (&fragment)[1], const void* shared) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                 : "=r"(fragment[0])
                 : "r"(seismic_shared_address(shared)));
}
__device__ __forceinline__ void seismic_ldmatrix_x2(unsigned (&fragment)[2], const void* shared) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(fragment[0]), "=r"(fragment[1])
                 : "r"(seismic_shared_address(shared)));
}
__device__ __forceinline__ void seismic_ldmatrix_x4(unsigned (&fragment)[4], const void* shared) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
                 : "r"(seismic_shared_address(shared)));
}
__device__ __forceinline__ void seismic_ldmatrix_x1_trans(unsigned (&fragment)[1],
                                                          const void* shared) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                 : "=r"(fragment[0])
                 : "r"(seismic_shared_address(shared)));
}
__device__ __forceinline__ void seismic_ldmatrix_x2_trans(unsigned (&fragment)[2],
                                                          const void* shared) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                 : "=r"(fragment[0]), "=r"(fragment[1])
                 : "r"(seismic_shared_address(shared)));
}
__device__ __forceinline__ void seismic_ldmatrix_x4_trans(unsigned (&fragment)[4],
                                                          const void* shared) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
                 : "r"(seismic_shared_address(shared)));
}

// Warp-level tensor-core MMA, acc += A * B, with lane l = 4g + t (g = l/4,
// t = l%4). Accumulator (16x8 f32 or s32): acc[0], acc[1] = row g, columns
// 2t, 2t+1; acc[2], acc[3] = row g+8, same columns.
//
// m16n8k16, 16-bit elements (f16 or bf16), each register a packed pair with
// the lower k in the low half:
//   A (16x16, row-major): a[0] = row g, k 2t..2t+1; a[1] = row g+8, k 2t..2t+1;
//                         a[2] = row g, k 2t+8..2t+9; a[3] = row g+8, k 2t+8..2t+9.
//   B (16x8, k by n):     b[0] = column g, k 2t..2t+1; b[1] = column g, k 2t+8..2t+9.
//
// m16n8k32, signed 8-bit elements, four per register with the lowest k in the
// lowest byte:
//   A (16x32): a[0] = row g, k 4t..4t+3; a[1] = row g+8, k 4t..4t+3;
//              a[2] = row g, k 4t+16..4t+19; a[3] = row g+8, k 4t+16..4t+19.
//   B (32x8):  b[0] = column g, k 4t..4t+3; b[1] = column g, k 4t+16..4t+19.
__device__ __forceinline__ void seismic_mma_m16n8k16_f16(float (&acc)[4], const unsigned (&a)[4],
                                                         const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void seismic_mma_m16n8k16_bf16(float (&acc)[4], const unsigned (&a)[4],
                                                          const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void seismic_mma_m16n8k32_s8(int (&acc)[4], const unsigned (&a)[4],
                                                        const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};"
        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
