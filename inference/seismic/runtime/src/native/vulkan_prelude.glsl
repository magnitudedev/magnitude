// ---------------------------------------------------------------------------
// Seismic Vulkan device library, the counterpart of the CUDA prelude. Every
// native Vulkan source receives it after `#version`, the extensions and the
// device feature macros, ahead of its generated ABI macros.
//
// Every floating-point operation of the formed module is `NoContraction`, and
// the module runs with RTE rounding and signed-zero/Inf/NaN preservation for
// fp32, and for fp16 where the device has fp16 arithmetic
// (`SEISMIC_HAS_FLOAT16`), so `+` and `*` are single correctly rounded
// operations (on the NVIDIA proprietary driver fp32 rounds by the device's
// default rather than a declared mode). Without fp16 arithmetic, fp16 is a
// storage format only: kernels compute in fp32 and convert with the helpers
// below. Float division is always `seismic_div_rn`, never a bare `/`: GLSL
// `/` is only 2.5 ULP, and NVIDIA's compiler traps on it under
// `RoundingModeRTE 32`.
//
// Kernels are written for a 32-lane logical subgroup (§6.3), executed by all
// 32 lanes convergently where a helper says so. On 32-lane hardware it is the
// hardware subgroup; on 64-lane hardware (`SEISMIC_SUBGROUP_LANES` 64, AMD
// GCN) each hardware subgroup holds two, lanes 0-31 and 32-63. Kernels name
// lanes and subgroups only through `SEISMIC_LANE`, `SEISMIC_SUBGROUP` and
// `SEISMIC_SUBGROUPS`, and reduce, vote, ballot or shuffle by absolute lane
// only through the `seismic_subgroup_*`, `seismic_ballot` and
// `seismic_shuffle` helpers (seismic-build refuses the subgroup built-ins
// otherwise). Shuffles by an xor mask below 32 stay within a logical
// subgroup, and subgroup barriers cover it, on both; kernels use those
// directly. An up or down delta stays within it only where the source lane
// does (lane - delta >= 0, lane + delta < 32): past the edge it reads the
// other logical subgroup on 64-lane hardware and an undefined value on
// 32-lane hardware, so a caller discards what an edge lane receives.
//
// A kernel that reads its subgroups declares its workgroup X extent a
// multiple of 64, and formation requires its subgroups full, so every hardware
// subgroup is full on both widths, and `SEISMIC_THREAD` (logical subgroup, then lane)
// numbers the workgroup's invocations 0 .. size - 1. A kernel that spreads
// work over the invocations and then reduces, votes or shuffles indexes by
// `SEISMIC_THREAD`, never by the local invocation index: Vulkan does not say
// which local indices share a subgroup, nor in which lane order.
// ---------------------------------------------------------------------------

#if SEISMIC_SUBGROUP_LANES == 32
#define SEISMIC_LANE gl_SubgroupInvocationID
#define SEISMIC_SUBGROUP gl_SubgroupID
#define SEISMIC_SUBGROUPS gl_NumSubgroups
#define SEISMIC_LANE_BASE 0u
#define seismic_subgroup_add(value) subgroupAdd(value)
#define seismic_subgroup_min(value) subgroupMin(value)
#define seismic_subgroup_max(value) subgroupMax(value)
#define seismic_subgroup_all(value) subgroupAll(value)
// One logical subgroup is the hardware subgroup: a value it holds alike is
// already the hardware subgroup's, and its join is the value itself.
// Formation requires each argument uniform, then removes the calls before the
// module ships, so the joins cost nothing (`uniformity.rs`, `formation.rs`).
// The parameters are `const` so that glslang passes them by value, which
// formation's removal requires. `seismic_hardware_any` is `seismic_steer`'s
// own, never a kernel's (seismic-build refuses it in kernels).
bool seismic_hardware_any(const bool value) { return value; }
uint seismic_hardware_first(const uint value) { return value; }
uvec2 seismic_hardware_first(const uvec2 value) { return value; }
// The region it steers runs only where `predicate` holds: `live` is true.
#define seismic_steer(live, predicate) (live = true, seismic_hardware_any(predicate))
#elif SEISMIC_SUBGROUP_LANES == 64
#define SEISMIC_LANE (gl_SubgroupInvocationID & 31u)
#define SEISMIC_SUBGROUP (2u * gl_SubgroupID + (gl_SubgroupInvocationID >> 5))
#define SEISMIC_SUBGROUPS ((gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z + 31u) >> 5)
#define SEISMIC_LANE_BASE (gl_SubgroupInvocationID & 32u)
#define seismic_subgroup_add(value) subgroupClusteredAdd(value, 32u)
#define seismic_subgroup_min(value) subgroupClusteredMin(value, 32u)
#define seismic_subgroup_max(value) subgroupClusteredMax(value, 32u)
#define seismic_subgroup_all(value) (subgroupClusteredAnd(uint(value), 32u) != 0u)
#define seismic_hardware_first(value) subgroupBroadcastFirst(value)
// The region it steers runs where `predicate` holds in either logical
// subgroup: `live` is the caller's own.
#define seismic_steer(live, predicate) (live = (predicate), subgroupAny(live))
#else
#error "SEISMIC_SUBGROUP_LANES must be 32 or 64"
#endif
// The value of `value` on logical lane `lane` of the caller's logical
// subgroup.
#define seismic_shuffle(value, lane) subgroupShuffle(value, SEISMIC_LANE_BASE + (lane))
// The lanes of the caller's logical subgroup for which `predicate` holds: bit
// l for logical lane l.
uint seismic_ballot(bool predicate) {
    const uvec4 lanes = subgroupBallot(predicate);
#if SEISMIC_SUBGROUP_LANES == 32
    return lanes.x;
#else
    return SEISMIC_LANE_BASE == 0u ? lanes.x : lanes.y;
#endif
}
// The invocation's place in the workgroup: logical subgroup, then lane.
#define SEISMIC_THREAD (SEISMIC_SUBGROUP * 32u + SEISMIC_LANE)
// The first logical subgroup of the caller's hardware subgroup, uniform across
// it: for hardware-uniform rounds and trip counts.
#define SEISMIC_HARDWARE_SUBGROUP (gl_SubgroupID * (SEISMIC_SUBGROUP_LANES / 32u))
// Votes over the whole hardware subgroup of a value each invocation holds
// its own, uniform across it on both widths (unlike the logical
// `seismic_subgroup_all`): for steering control flow.
#define seismic_uniform_all(value) subgroupAll(value)
#define seismic_uniform_any(value) subgroupAny(value)
// `seismic_steer(live, predicate)` steers the hardware subgroup by a
// predicate each logical subgroup holds alike, and binds `live`, a `bool` the
// caller declares, to whether the caller's logical subgroup takes part. It is
// the whole condition of an `if`, `while` or `for` (seismic-build refuses it
// elsewhere, and as the condition of a `break` or `continue`, so a loop it
// steers exits at its header), and `live` means it only inside the region
// the condition guards: the body, or the rest of the function after
// `if (!seismic_steer(...)) return`. The region runs where the predicate
// holds in either logical subgroup of the hardware subgroup; `live` is the
// caller's own predicate on 64-lane hardware and the literal `true` on
// 32-lane hardware, where the region runs only where it holds, so a 32-lane
// module keeps nothing of either the vote or `live`.
// `seismic_hardware_first` broadcasts a value each logical subgroup holds
// alike from the first: the value itself on 32-lane hardware.
//
// `seismic_strided_from(T, i, origin, offset, end, step)` heads a loop over
// `i` = `origin` + `offset`, then up by `step`, below `end`: `origin`, `end`
// and `step` uniform, each invocation its own `offset` below `step` (a lane,
// or `SEISMIC_THREAD`); `seismic_strided(T, i, offset, end, step)` is the
// loop from `origin` 0. Where the device has maximal reconvergence
// (`SEISMIC_HAS_MAXIMAL_RECONVERGENCE` 1: the seal pass declares
// `MaximallyReconvergesKHR` on a module that relies on it) it is the plain
// loop, each invocation leaving at its own bound and every
// invocation rejoined after it, so a subgroup operation after the loop sees
// them all. Elsewhere every invocation runs the rounds of the first, `i`
// held by the guard `i < end` alone, so the trip count is uniform and a
// subgroup operation after the loop runs where Vulkan's core model has the
// invocations together. Its body is one statement or block, as a `for`'s; it
// holds no subgroup operation either way (the uniformity check refuses one in
// the plain loop, whose invocations leave apart), and no `continue` or
// `break` (in the rounds each is taken under the guard, which makes the loop
// lane-varying).
//
// `seismic_strided_bounded(T, i, origin, offset, end, bound, step)` is the
// loop whose `end` is each invocation's own, below a uniform `bound`: the
// plain loop where the device has maximal reconvergence, and elsewhere the
// rounds up to `bound`, guarded by `end`. `seismic_strided_from` is the loop
// whose `end` is its `bound`.
//
// `seismic_rejoin()` rejoins the invocations ahead of a subgroup operation
// that follows loops they left apart, at a point where a workgroup barrier
// is legal: nothing where the device has maximal reconvergence, elsewhere
// that barrier. Sources leave the mode to these: they do not read
// `SEISMIC_HAS_MAXIMAL_RECONVERGENCE` (`seismic-build` refuses it).
#if SEISMIC_HAS_MAXIMAL_RECONVERGENCE
#define seismic_strided_bounded(T, i, origin, offset, end, bound, step) \
    for (T i = (origin) + (offset); i < (end); i += (step))
#define seismic_rejoin()
#else
#define seismic_strided_bounded(T, i, origin, offset, end, bound, step) \
    for (T i##_round = (origin), i = (origin) + (offset); i##_round < (bound); i##_round += (step), i += (step)) \
        if (i < (end))
#define seismic_rejoin() barrier()
#endif
#define seismic_strided_from(T, i, origin, offset, end, step) \
    seismic_strided_bounded(T, i, origin, offset, end, end, step)
#define seismic_strided(T, i, offset, end, step) seismic_strided_from(T, i, T(0), offset, end, step)

// Element references: one element per block, so a reference indexes like a
// pointer (`seismic_f32(address)[i].v`) and advances by the element size.
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer seismic_f32 { float v; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer seismic_f16 { float16_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer seismic_bf16 { uint16_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 1) buffer seismic_u8 { uint8_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 1) buffer seismic_i8 { int8_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer seismic_u16 { uint16_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer seismic_u32 { uint v; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer seismic_i32 { int v; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer seismic_u64 { uint64_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer seismic_uvec2 { uvec2 v; };
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer seismic_uvec4 { uvec4 v; };

// Conversions. bf16 rounds to nearest even in integer code (NaN becomes the
// canonical 0x7fff, as CUDA's cvt.rn.bf16.f32). fp16 conversions round by the
// module's RTE mode where the device has fp16 arithmetic, and otherwise in
// integer code to nearest even with subnormals preserved (the CPU library's
// `f32_to_f16` and `f16_to_f32`). `seismic_f16_decode` is the exact integer
// decode on every device: an in-register round trip
// (`seismic_f16_decode(seismic_f32_to_f16(x))`) uses it, because a hardware
// encode-decode pair is a pair of conversions the driver may fold, leaving
// `x` unrounded (an f16 subnormal on the NVIDIA proprietary driver and
// llvmpipe).
float seismic_bf16_to_f32(uint16_t value) {
    return uintBitsToFloat(uint(value) << 16);
}
uint16_t seismic_f32_to_bf16(float value) {
    const uint bits = floatBitsToUint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return uint16_t(0x7fffu);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
float seismic_f16_decode(uint16_t value) {
    const uint bits = uint(value);
    const uint sign = (bits & 0x8000u) << 16;
    const uint exponent = (bits >> 10) & 0x1fu;
    const uint mantissa = bits & 0x3ffu;
    if (exponent == 0x1fu)
        return uintBitsToFloat(sign | 0x7f800000u | (mantissa << 13));
    if (exponent != 0u)
        return uintBitsToFloat(sign | ((exponent + 112u) << 23) | (mantissa << 13));
    if (mantissa == 0u)
        return uintBitsToFloat(sign);
    // A subnormal half, mantissa * 2^-24: its leading one becomes the
    // implicit bit.
    const uint shift = uint(10 - findMSB(mantissa));
    return uintBitsToFloat(sign | ((113u - shift) << 23) | (((mantissa << shift) & 0x3ffu) << 13));
}
#if SEISMIC_HAS_FLOAT16
float seismic_f16_to_f32(uint16_t value) {
    return float(uint16BitsToFloat16(value));
}
uint16_t seismic_f32_to_f16(float value) {
    return float16BitsToUint16(float16_t(value));
}
#else
float seismic_f16_to_f32(uint16_t value) {
    return seismic_f16_decode(value);
}
uint16_t seismic_f32_to_f16(float value) {
    const uint bits = floatBitsToUint(value);
    const uint sign = (bits >> 16) & 0x8000u;
    const int exponent = int((bits >> 23) & 0xffu);
    const uint mantissa = bits & 0x7fffffu;
    if (exponent == 0xff)
        return uint16_t(sign | 0x7c00u | (mantissa == 0u ? 0u : 0x200u | (mantissa >> 13)));
    const int unbiased = exponent - 127;
    if (unbiased > 15)
        return uint16_t(sign | 0x7c00u);
    if (unbiased >= -14) {
        // Normal half: round the 13 dropped mantissa bits to nearest even.
        const uint kept = (uint(unbiased + 15) << 10) | (mantissa >> 13);
        const uint dropped = mantissa & 0x1fffu;
        const bool up = dropped > 0x1000u || (dropped == 0x1000u && (kept & 1u) != 0u);
        return uint16_t(sign | (kept + uint(up)));
    }
    if (unbiased < -25)
        return uint16_t(sign);
    // Subnormal half: the implicit bit joins the mantissa.
    const uint full = mantissa | 0x800000u;
    const uint shift = uint(-unbiased - 1);
    const uint kept = full >> shift;
    const uint dropped = full & ((1u << shift) - 1u);
    const uint midpoint = 1u << (shift - 1u);
    const bool up = dropped > midpoint || (dropped == midpoint && (kept & 1u) != 0u);
    return uint16_t(sign | (kept + uint(up)));
}
#endif

// Packed pairs: `lo` occupies bits 0..15 (the lower address in memory), `hi`
// bits 16..31. Packing rounds to nearest even.
uint seismic_pack_bf16x2(float lo, float hi) {
    return uint(seismic_f32_to_bf16(lo)) | (uint(seismic_f32_to_bf16(hi)) << 16);
}
uint seismic_pack_f16x2(float lo, float hi) {
    return uint(seismic_f32_to_f16(lo)) | (uint(seismic_f32_to_f16(hi)) << 16);
}
vec2 seismic_unpack_bf16x2(uint pair) {
    return vec2(uintBitsToFloat(pair << 16), uintBitsToFloat(pair & 0xffff0000u));
}
vec2 seismic_unpack_f16x2(uint pair) {
    return vec2(seismic_f16_to_f32(uint16_t(pair & 0xffffu)), seismic_f16_to_f32(uint16_t(pair >> 16)));
}

// Exact integer helpers of the correctly rounded operations below.

// Round the 27-bit significand `q` (leading one at bit 26, three extra bits)
// with sticky bit `sticky` at biased exponent `exponent` to nearest even, and
// pack it (subnormal and overflowing results included).
float seismic_round_pack(uint sign, int exponent, uint q, bool sticky) {
    if (exponent >= 255)
        return uintBitsToFloat(sign | 0x7f800000u);
    if (exponent <= 0) {
        const uint shift = uint(min(1 - exponent, 31));
        sticky = sticky || (q & ((1u << shift) - 1u)) != 0u;
        q >>= shift;
        exponent = 0;
    }
    const uint rest = q & 7u;
    uint significand = q >> 3;
    if (rest > 4u || (rest == 4u && (sticky || (significand & 1u) != 0u)))
        significand += 1u;
    const uint base = exponent == 0 ? 0u : uint(exponent - 1) << 23;
    return uintBitsToFloat(sign | min(base + significand, 0x7f800000u));
}

// A finite nonzero operand as a 24-bit significand (leading one at bit 23)
// and an unbiased exponent.
uint seismic_normalize(uint bits, out int exponent) {
    const uint field = (bits >> 23) & 0xffu;
    const uint significand = bits & 0x7fffffu;
    if (field == 0u) {
        const int shift = 23 - findMSB(significand);
        exponent = -126 - shift;
        return significand << shift;
    }
    exponent = int(field) - 127;
    return significand | 0x800000u;
}

// Explicitly rounded arithmetic. `seismic_fma_rn` is one fused, correctly
// rounded multiply-add where the device has `VK_KHR_shader_fma` (formation
// binds `fma` to `OpFmaKHR`). Elsewhere it is GLSL `fma`, which the driver
// may fuse or round twice. `seismic_div_rn`'s fast path takes its residuals
// from `fma`, so there its quotient is correctly rounded only where `fma` is
// fused.
float seismic_add_rn(float a, float b) {
    return a + b;
}
float seismic_mul_rn(float a, float b) {
    return a * b;
}
float seismic_fma_rn(float a, float b, float c) {
    return fma(a, b, c);
}

// Coordinates without 64-bit division, which drivers expand into long
// instruction sequences (Mesa's Intel compiler about 2,300 instructions,
// copied by every unrolled loop around it). A 32-bit value split by any extent
// needs only a 32-bit division; a grid-stride walk splits its 32-bit start and
// stride once and then carries its coordinates. A 64-bit carry is a select,
// never a branch around the subtraction: NVIDIA 580's compiler crashes on the
// branch.

// `value` = `quotient` × `extent` + `remainder`, with 0 ≤ `remainder` <
// `extent`, for `extent` ≥ 1.
void seismic_split(uint value, uint64_t extent, out uint64_t quotient, out uint64_t remainder) {
    if (extent > uint64_t(0xffffffffu)) {
        quotient = 0ul;
        remainder = uint64_t(value);
    } else {
        quotient = uint64_t(value / uint(extent));
        remainder = uint64_t(value % uint(extent));
    }
}

// `value` = `quotient` × `divisor` + `remainder`, with 0 ≤ `remainder` <
// `divisor`, for 1 ≤ `divisor` < 2^16: long division of the high word and two
// 16-bit digits, each partial remainder shifted by 16 staying below 2^32.
void seismic_divide(uint64_t value, uint divisor, out uint64_t quotient, out uint remainder) {
    const uint high = uint(value >> 32), low = uint(value);
    const uint q2 = high / divisor;
    uint rest = high % divisor;
    uint digit = (rest << 16) | (low >> 16);
    const uint q1 = digit / divisor;
    rest = digit % divisor;
    digit = (rest << 16) | (low & 0xffffu);
    quotient = (uint64_t(q2) << 32) | (uint64_t(q1) << 16) | uint64_t(digit / divisor);
    remainder = digit % divisor;
}

// Refuses, at compile time, a constant divisor outside 1 ≤ `value` < 2^16 (an array of
// negative size): `seismic_divide` and `seismic_quotient` give a wrong quotient for one.
// `NAME` is a distinct global per check.
#define REQUIRE_DIVISOR_BELOW_2_16(NAME, value) \
    const int NAME[(uint64_t(value) >= 1ul && uint64_t(value) < 65536ul) ? 1 : -1] = int[1](0)

// `value` / `divisor` (1 ≤ `divisor` < 2^16), as `seismic_divide`.
uint64_t seismic_quotient(uint64_t value, uint divisor) {
    uint64_t quotient;
    uint remainder;
    seismic_divide(value, divisor, quotient, remainder);
    return quotient;
}

// `rest` / `divisor` for a quotient below 2^32 (`rest` < `divisor` × 2^32,
// `divisor` ≥ 1), leaving `rest` % `divisor` in `rest`: restoring division,
// before the step at `bit` the rest is below `divisor` × 2^(`bit` + 1).
uint seismic_narrow_divide(inout uint64_t rest, uint64_t divisor) {
    uint quotient = 0u;
    [[dont_unroll]] for (int bit = 31; bit >= 0; --bit) {
        const bool take = (rest >> bit) >= divisor;
        rest -= take ? divisor << bit : 0ul;
        quotient |= take ? 1u << bit : 0u;
    }
    return quotient;
}

// The quotient and remainder by `extent` of item `local` of group `group`
// of `per` items (`local` < `per`), index `group` × `per` + `local`, without
// forming that index: the group splits first, and the rest, below
// `extent` × `per`, holds fewer than `per` extents.
void seismic_split_group(uint group, uint per, uint local, uint64_t extent, out uint64_t quotient,
    out uint64_t remainder) {
    uint64_t groups;
    seismic_split(group, extent, groups, remainder);
    remainder = remainder * per + local;
    quotient = groups * per + seismic_narrow_divide(remainder, extent);
}

// A strided walk over the indices of a flattened [x, y, z, w] extent: it
// visits index = `start`, `start` + `stride`, ... below `count` (the x extent
// × `extents`, the y, z and w extents), at
// `axes` = (index / (y × z × w), index / (z × w) % y, index / w % z, index % w).
// A walk over fewer axes gives the outer of y, z, w extent 1.
//
//     for (seismic_walk at = seismic_walk_grid(rows * width, u64vec3(1ul, 1ul, width));
//             at.index < rows * width; seismic_walk_next(at)) {
//         const uint64_t row = at.axes.x, column = at.axes.w; ... }
//
// `start` and `stride` are 32-bit, so are their coordinates, and splitting
// them needs only 32-bit division; each step adds the stride's coordinates and
// carries, each coordinate staying below twice its extent before its carry.
struct seismic_walk {
    uint64_t index, stride;
    u64vec4 axes, steps;
    u64vec3 extents;
};

// The coordinates of 32-bit `value` over the y, z, w `extents`.
u64vec4 seismic_axes(uint value, u64vec3 extents) {
    u64vec4 axes;
    uint64_t rest;
    seismic_split(value, extents.z, rest, axes.w);
    seismic_split(uint(rest), extents.y, rest, axes.z);
    seismic_split(uint(rest), extents.x, axes.x, axes.y);
    return axes;
}

seismic_walk seismic_walk_begin(uint start, uint stride, uint64_t count, u64vec3 extents) {
    seismic_walk walk;
    walk.index = uint64_t(start);
    walk.stride = uint64_t(stride);
    walk.extents = extents;
    walk.axes = walk.steps = u64vec4(0ul);
    // An empty extent is never visited; a nonempty one has every extent ≥ 1.
    if (count != 0ul) {
        walk.axes = seismic_axes(start, extents);
        walk.steps = seismic_axes(stride, extents);
    }
    return walk;
}

// The x grid-stride walk: every invocation of the x grid from its global
// index. gl_GlobalInvocationID.x is 32-bit, so the grid's x invocations are.
// A macro: the kernel declares its workgroup size after this library.
#define seismic_walk_grid(count, extents) \
    seismic_walk_begin(gl_GlobalInvocationID.x, gl_NumWorkGroups.x * gl_WorkGroupSize.x, count, extents)

void seismic_walk_next(inout seismic_walk walk) {
    walk.index += walk.stride;
    walk.axes += walk.steps;
    bool carry = walk.axes.w >= walk.extents.z;
    walk.axes.w -= carry ? walk.extents.z : 0ul;
    walk.axes.z += carry ? 1ul : 0ul;
    carry = walk.axes.z >= walk.extents.y;
    walk.axes.z -= carry ? walk.extents.y : 0ul;
    walk.axes.y += carry ? 1ul : 0ul;
    carry = walk.axes.y >= walk.extents.x;
    walk.axes.y -= carry ? walk.extents.x : 0ul;
    walk.axes.x += carry ? 1ul : 0ul;
}

// Correctly rounded fp32 division and square root (CUDA's `/` and `sqrtf`
// under --prec-div/--prec-sqrt). Division of operands and quotient well
// inside the normal range takes a fused exact-residual path; any other
// division takes the same path on the significands. Integer square root
// handles every operand.

// The correctly rounded a / b for operands and quotient within 2^+-100, where
// every residual is exact: one correction from the exact residual, then the
// neighbor whose exact residual is smallest (division has no ties).
float seismic_div_core(float a, float b, float q0) {
    const float q1 = q0 + fma(-q0, b, a) / b;
    const uint bits = floatBitsToUint(q1);
    float best = q1;
    float residual = abs(fma(-q1, b, a));
    for (int step = -1; step <= 1; step += 2) {
        const float candidate = uintBitsToFloat(uint(int(bits) + step));
        const float candidate_residual = abs(fma(-candidate, b, a));
        if (candidate_residual < residual) {
            best = candidate;
            residual = candidate_residual;
        }
    }
    return best;
}

// a / b for any operands, on the core: the significands in [1, 2) divide in
// range, then the quotient takes back the exponents. Every division site
// inlines this path, often inside unrolled epilogues with live accumulators,
// so it holds no integer division: only the core again and bit arithmetic.
float seismic_div_exact(float a, float b) {
    const uint ua = floatBitsToUint(a), ub = floatBitsToUint(b);
    const uint sign = (ua ^ ub) & 0x80000000u;
    const float nan = uintBitsToFloat(0x7fffffffu);
    if (isnan(a) || isnan(b) || (isinf(a) && isinf(b)))
        return nan;
    const bool zero_a = (ua & 0x7fffffffu) == 0u, zero_b = (ub & 0x7fffffffu) == 0u;
    if (zero_a && zero_b)
        return nan;
    if (isinf(a) || zero_b)
        return uintBitsToFloat(sign | 0x7f800000u);
    if (isinf(b) || zero_a)
        return uintBitsToFloat(sign);
    int ea, eb;
    const uint ma = seismic_normalize(ua, ea), mb = seismic_normalize(ub, eb);
    const float x = uintBitsToFloat(0x3f800000u | (ma & 0x7fffffu));
    const float y = uintBitsToFloat(0x3f800000u | (mb & 0x7fffffu));
    // q = RN(x / y) in (1/2, 2), and the sign of the exact residual.
    const float q = seismic_div_core(x, y, x / y);
    const float residual = fma(-q, y, x);
    const uint bits = floatBitsToUint(q);
    const int exponent = int(bits >> 23) + ea - eb;
    // A normal result is q scaled exactly.
    if (exponent >= 255)
        return uintBitsToFloat(sign | 0x7f800000u);
    if (exponent >= 1)
        return uintBitsToFloat(sign | (uint(exponent) << 23) | (bits & 0x7fffffu));
    // A subnormal one rounds at bit 1 or above of the truncated quotient t,
    // the rest below t only sticky: t is q, or its predecessor when q is
    // above the quotient.
    const uint t = residual < 0.0 ? bits - 1u : bits;
    return seismic_round_pack(sign, int(t >> 23) + ea - eb, ((t & 0x7fffffu) | 0x800000u) << 3, residual != 0.0);
}

float seismic_div_rn(float a, float b) {
    // Operands and quotient within 2^+-100 keep every residual exact.
    const float q0 = a / b;
    const float low = 7.8886091e-31, high = 1.2676506e30;
    const bool fast = abs(a) >= low && abs(a) <= high && abs(b) >= low && abs(b) <= high
        && abs(q0) >= low && abs(q0) <= high;
    if (!fast)
        return seismic_div_exact(a, b);
    return seismic_div_core(a, b, q0);
}

// One digit step of `seismic_sqrt_rn`'s restoring square root, and four.
#define SEISMIC_SQRT_STEP \
    { \
        const uint64_t trial = root + one; \
        const bool take = rest >= trial; \
        rest -= take ? trial : 0ul; \
        root = (root >> 1) + (take ? one : 0ul); \
        one >>= 2; \
    }
#define SEISMIC_SQRT_STEPS4 SEISMIC_SQRT_STEP SEISMIC_SQRT_STEP SEISMIC_SQRT_STEP SEISMIC_SQRT_STEP
float seismic_sqrt_rn(float x) {
    const uint bits = floatBitsToUint(x);
    if (isnan(x) || (x < 0.0))
        return uintBitsToFloat(0x7fffffffu);
    if (isinf(x) || (bits & 0x7fffffffu) == 0u)
        return x;
    int exponent;
    const uint significand = seismic_normalize(bits, exponent);
    // x = significand * 2^(exponent - 23) = M * 2^t with t even.
    int t = exponent - 23 - 30;
    // An odd t moves one factor of two into m.
    const int odd = t & 1;
    uint64_t m = uint64_t(significand) << (30 + odd);
    t -= odd;
    // m lies in [2^53, 2^55), so its leading power of four is at most 2^54:
    // 28 digit steps from there (a leading step above the operand leaves the
    // root 0). Each step's conditional subtraction is selects, not a branch
    // (NVIDIA's 580 compiler crashes on a 64-bit subtraction under a 64-bit
    // compare). The steps are straight-line, not a loop: on the loop form
    // NVIDIA's 580 drivers computed odd exponents wrongly (GB10) or hung
    // (Ada) (docs/compatibility.md, "Driver defects").
    uint64_t rest = m, root = 0ul, one = 1ul << 54;
    SEISMIC_SQRT_STEPS4 SEISMIC_SQRT_STEPS4 SEISMIC_SQRT_STEPS4 SEISMIC_SQRT_STEPS4
    SEISMIC_SQRT_STEPS4 SEISMIC_SQRT_STEPS4 SEISMIC_SQRT_STEPS4
    bool sticky = rest != 0ul;
    int biased = 26 + t / 2 + 127;
    if (root >= (1ul << 27)) {
        sticky = sticky || (root & 1ul) != 0ul;
        root >>= 1;
        biased += 1;
    }
    return seismic_round_pack(0u, biased, uint(root), sticky);
}
#undef SEISMIC_SQRT_STEPS4
#undef SEISMIC_SQRT_STEP

// Hardware approximations. Not correctly rounded: admitted only where an
// entry's precision gate allows them. Their divisions are `seismic_div_rn`,
// as every float division is (a bare fp32 `/` does not form on the NVIDIA
// proprietary driver under `RoundingModeRTE 32`, §16.1).
float seismic_ex2_approx(float x) {
    return exp2(x);
}
float seismic_lg2_approx(float x) {
    return log2(x);
}
float seismic_rcp_approx(float x) {
    return seismic_div_rn(1.0, x);
}
float seismic_rsqrt_approx(float x) {
    return inversesqrt(x);
}
float seismic_tanh_approx(float x) {
    // tanh(x) = 1 - 2 / (2^(2x log2 e) + 1), saturated where the exponential
    // overflows.
    const float clamped = clamp(x, -15.0, 15.0);
    return 1.0 - seismic_div_rn(2.0, exp2(2.8853900817779268 * clamped) + 1.0);
}

// Subgroup shuffles within the logical subgroup (lane masks below 32; an
// up or down delta only where the source lane is within it, as above).
uint seismic_shfl_xor_u32(uint value, uint lane_mask) {
    return subgroupShuffleXor(value, lane_mask);
}
uint seismic_shfl_idx_u32(uint value, uint source_lane) {
    return seismic_shuffle(value, source_lane);
}
uint seismic_shfl_down_u32(uint value, uint delta) {
    return subgroupShuffleDown(value, delta);
}
uint seismic_shfl_up_u32(uint value, uint delta) {
    return subgroupShuffleUp(value, delta);
}
float seismic_shfl_xor_f32(float value, uint lane_mask) {
    return subgroupShuffleXor(value, lane_mask);
}
float seismic_shfl_idx_f32(float value, uint source_lane) {
    return seismic_shuffle(value, source_lane);
}
float seismic_shfl_down_f32(float value, uint delta) {
    return subgroupShuffleDown(value, delta);
}
float seismic_shfl_up_f32(float value, uint delta) {
    return subgroupShuffleUp(value, delta);
}

// Butterfly reductions over the logical subgroup: masks below 32 keep them
// inside it at either width, and every lane receives the same value, combined
// in the order xor 16, 8, 4, 2, 1.
float seismic_subgroup_sum_f32(float value) {
    [[unroll]] for (uint mask = 16u; mask > 0u; mask >>= 1)
        value = value + subgroupShuffleXor(value, mask);
    return value;
}
float seismic_subgroup_max_f32(float value) {
    [[unroll]] for (uint mask = 16u; mask > 0u; mask >>= 1)
        value = max(value, subgroupShuffleXor(value, mask));
    return value;
}

// Integer logical-subgroup reductions, where the order does not matter;
// every lane receives the result.
uint seismic_redux_add_u32(uint value) {
    return seismic_subgroup_add(value);
}
uint seismic_redux_min_u32(uint value) {
    return seismic_subgroup_min(value);
}
uint seismic_redux_max_u32(uint value) {
    return seismic_subgroup_max(value);
}
int seismic_redux_add_s32(int value) {
    return seismic_subgroup_add(value);
}
int seismic_redux_min_s32(int value) {
    return seismic_subgroup_min(value);
}
int seismic_redux_max_s32(int value) {
    return seismic_subgroup_max(value);
}

// Four-way byte dot products accumulated into `acc` without saturation (as
// CUDA's dp4a): bytes of `a` and `b` in little-endian order, signed (s8) or
// unsigned (u8).
int seismic_dp4a_s8(int a, int b, int acc) {
    return acc + dotPacked4x8EXT(a, b);
}
int seismic_dp4a_u8s8(uint a, int b, int acc) {
    return acc + dotPacked4x8EXT(a, b);
}
