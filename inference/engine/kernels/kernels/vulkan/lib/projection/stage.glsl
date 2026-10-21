// stage shared projection mechanisms; included at the original declaration point.
// ---------------------------------------------------------------------------
// Operands.

// Activation row m: AllRows (table 0) or SelectedRows.
uint projection_row(uint64_t table, uint m) {
    return table == 0ul ? m : uint(element_i32_at(table + uint64_t(m) * 4ul));
}

#define PROJECTION_PLAIN 0
#define PROJECTION_RMS 1
#define PROJECTION_GROUPED 2
#define PROJECTION_SUMMED 3

// The operand recipe of the weights a prologue's rows feed (the portable
// `operand_rows`): the rows as they are, or as NVFP4 operands. An entry reads
// its weights' recipe from the ABI by name (`defined(<W>_OPERANDS_NVFP4)`,
// see <seismic/element.glsl>).
#define PROJECTION_OPERANDS_ROUNDED 0
#define PROJECTION_OPERANDS_NVFP4 1

struct projection_prologue {
    int kind;
    int act;            // ELEMENT_* of the activation A (bf16 or f16)
    uint64_t x;         // Plain/Grouped: A rows; Rms/Summed: F32 rows
    uint64_t x0;        // element strides of x
    uint64_t x1;
    uint columns;       // K
    uint64_t rows;      // row map (0: AllRows); Grouped: the order row (i32)
    uint64_t rows_stride;
    uint64_t norm;      // Rms: norm weights
    uint64_t norm_stride;
    int norm_kind;      // ELEMENT_* of the norm
    float eps;
    uint parts;         // Summed: the parts of a row, `columns` wide each
    int operands;       // PROJECTION_OPERANDS_*: how the GEMV staging and the stage launches form the rows
};

projection_prologue projection_plain(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t rows) {
    return projection_prologue(PROJECTION_PLAIN, act, x, stride0, stride1, columns, rows, 1ul, 0ul, 0ul,
        ELEMENT_F32, 0.0, 1u, PROJECTION_OPERANDS_ROUNDED);
}

projection_prologue projection_rms(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint64_t norm,
    uint64_t norm_stride, const int norm_kind, float eps, uint columns, uint64_t rows) {
    return projection_prologue(PROJECTION_RMS, act, x, stride0, stride1, columns, rows, 1ul, norm, norm_stride,
        norm_kind, eps, 1u, PROJECTION_OPERANDS_ROUNDED);
}

// The sum in part order of the `parts` parts of an F32 row, a hyper-connection
// mix site's per-stream partials (`form_input` without a norm). Its square
// sums are zero and it ignores the inverse it is given.
projection_prologue projection_summed(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t rows, uint parts) {
    return projection_prologue(PROJECTION_SUMMED, act, x, stride0, stride1, columns, rows, 1ul, 0ul, 0ul,
        ELEMENT_F32, 0.0, parts, PROJECTION_OPERANDS_ROUNDED);
}

// An entry's input prologue (`form_input`) by its norm's extent `norm_extent`:
// `projection_rms`, or without a norm `projection_summed` over `parts` parts.
projection_prologue projection_input(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint64_t norm,
    uint64_t norm_stride, const int norm_kind, float eps, uint columns, uint64_t rows, const uint norm_extent,
    const uint parts) {
    return norm_extent == 0u
        ? projection_summed(act, x, stride0, stride1, columns, rows, parts)
        : projection_rms(act, x, stride0, stride1, norm, norm_stride, norm_kind, eps, columns, rows);
}

// Tile row m reads activation row order[m * order_stride]; -1 is padding.
projection_prologue projection_grouped(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t order, uint64_t order_stride) {
    return projection_prologue(PROJECTION_GROUPED, act, x, stride0, stride1, columns, order, order_stride, 0ul,
        0ul, ELEMENT_F32, 0.0, 1u, PROJECTION_OPERANDS_ROUNDED);
}

// The prologue's rows as the operands of weights of recipe `operands`
// (PROJECTION_OPERANDS_*), which the GEMV staging and the stage launches
// form (`projection_operand_load8`); the GEMM bodies read rows as they are,
// so a GEMM over NVFP4 weights reads the rows a stage launch formed.
projection_prologue projection_operands(projection_prologue in_, const int operands) {
    in_.operands = operands;
    return in_;
}

// Element k of a Summed row m: the sum in part order of its parts.
float projection_summed_at(projection_prologue in_, uint m, uint k) {
    const uint64_t row = uint64_t(projection_row(in_.rows, m)) * in_.x0;
    float sum = 0.0;
    for (uint part = 0u; part < in_.parts; ++part)
        sum = seismic_add_rn(sum,
            element_f32_at(in_.x + (row + uint64_t(part * in_.columns + k) * in_.x1) * 4ul));
    return sum;
}

// Eight consecutive elements of `kind` from element `index` of the tensor at
// `base` with element stride `stride`, as (0,2,4,6), (1,3,5,7); elements past
// the first `valid` are zero. `vector` allows one aligned load.
void projection_load8_strided(const int kind, uint64_t base, uint64_t index, uint64_t stride, uint valid,
    bool vector, out vec4 even, out vec4 odd) {
    if (vector && valid >= 8u) {
        element_load8(kind, base, index, even, odd);
        return;
    }
    float v[8];
    [[unroll]] for (uint i = 0u; i < 8u; ++i)
        v[i] = i < valid ? element_at(kind, base, index + uint64_t(i) * stride) : 0.0;
    even = vec4(v[0], v[2], v[4], v[6]);
    odd = vec4(v[1], v[3], v[5], v[7]);
}

// The input the Rms norm reduces: element i of row m.
float projection_norm_input(projection_prologue in_, uint m, uint i) {
    return element_f32_at(in_.x + (uint64_t(projection_row(in_.rows, m)) * in_.x0 + uint64_t(i) * in_.x1) * 4ul);
}

float projection_silu_rounded(const int act, float gate) {
    return element_round(act, seismic_div_rn(gate, 1.0 + exp(-gate)));
}

// x[m, k..k+8) (k a multiple of 8) as (even, odd) given the norm inverse of
// row m (Plain and Grouped ignore it); elements at or past `columns` are zero.
void projection_load8(projection_prologue in_, uint m, uint k, float inverse, out vec4 even, out vec4 odd) {
    const uint valid = k < in_.columns ? min(8u, in_.columns - k) : 0u;
    if (in_.kind == PROJECTION_PLAIN || in_.kind == PROJECTION_GROUPED) {
        int row = int(m);
        if (in_.kind == PROJECTION_GROUPED)
            row = element_i32_at(in_.rows + uint64_t(m) * in_.rows_stride * 4ul);
        else if (in_.rows != 0ul)
            row = int(projection_row(in_.rows, m));
        if (row < 0) {
            even = vec4(0.0);
            odd = vec4(0.0);
            return;
        }
        projection_load8_strided(in_.act, in_.x, uint64_t(row) * in_.x0 + uint64_t(k) * in_.x1, in_.x1, valid,
            in_.x1 == 1ul && (in_.x0 & 7ul) == 0ul, even, odd);
    } else if (in_.kind == PROJECTION_SUMMED) {
        float v[8];
        [[unroll]] for (uint i = 0u; i < 8u; ++i)
            v[i] = i < valid ? element_round(in_.act, projection_summed_at(in_, m, k + i)) : 0.0;
        even = vec4(v[0], v[2], v[4], v[6]);
        odd = vec4(v[1], v[3], v[5], v[7]);
    } else {
        const uint64_t row = uint64_t(projection_row(in_.rows, m)) * in_.x0;
        vec4 xe, xo;
        projection_load8_strided(ELEMENT_F32, in_.x, row + uint64_t(k) * in_.x1, in_.x1, valid,
            in_.x1 == 1ul && (in_.x0 & 3ul) == 0ul, xe, xo);
        if (valid == 8u) {
            vec4 ne, no;
            projection_load8_strided(in_.norm_kind, in_.norm, uint64_t(k) * in_.norm_stride, in_.norm_stride, 8u,
                in_.norm_stride == 1ul, ne, no);
            const vec4 e = xe * inverse * ne, o = xo * inverse * no;
            [[unroll]] for (uint i = 0u; i < 4u; ++i) {
                even[i] = element_round(in_.act, e[i]);
                odd[i] = element_round(in_.act, o[i]);
            }
        } else {
            float v[8];
            [[unroll]] for (uint i = 0u; i < 8u; ++i) {
                const float x = (i & 1u) != 0u ? xo[i >> 1] : xe[i >> 1];
                v[i] = i < valid
                    ? element_round(in_.act, x * inverse * element_at(in_.norm_kind, in_.norm, uint64_t(k + i) * in_.norm_stride))
                    : 0.0;
            }
            even = vec4(v[0], v[2], v[4], v[6]);
            odd = vec4(v[1], v[3], v[5], v[7]);
        }
    }
}

// The Rms row norm's square sum is split into PROJECTION_RMS_PARTS fixed
// parts; one lane's share of `part`: columns 4c .. 4c + 3 for
// c = 32 * part + lane + 32 * parts * j, in order.
#define PROJECTION_RMS_PARTS 8u

float projection_rms_squares(projection_prologue in_, uint m, uint part, uint lane) {
    if (in_.kind == PROJECTION_SUMMED)
        return 0.0;
    const uint64_t row = in_.x + uint64_t(projection_row(in_.rows, m)) * in_.x0 * 4ul;
    const bool vector = in_.x1 == 1ul && (in_.x0 & 3ul) == 0ul && (in_.columns & 3u) == 0u;
    float sum = 0.0;
    for (uint c = 4u * (32u * part + lane); c < in_.columns; c += 128u * PROJECTION_RMS_PARTS) {
        vec4 v;
        if (vector) {
            v = element_vec4_at(row + uint64_t(c) * 4ul);
        } else {
            [[unroll]] for (uint i = 0u; i < 4u; ++i)
                v[i] = c + i < in_.columns ? element_f32_at(row + uint64_t(c + i) * in_.x1 * 4ul) : 0.0;
        }
        sum = seismic_fma_rn(v.x, v.x, sum);
        sum = seismic_fma_rn(v.y, v.y, sum);
        sum = seismic_fma_rn(v.z, v.z, sum);
        sum = seismic_fma_rn(v.w, v.w, sum);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// NVFP4 operands (docs/precision.md, "NVFP4 operands"; the portable
// `nvfp4_row`): a product over NVFP4 weights takes each row x as q * g * s,
// g the row's global scale (the least power of two with 2688 g >= amax|x|),
// s a 16-value block's UE4M3 scale (nearest amax|block| / (6 g)) and q the
// E2M1 code nearest x / (g s), saturating at 6, every rounding to nearest
// even. The portable body divides with F32's correctly rounded quotient,
// which rounds onto each grid as the exact quotient does; GLSL's division is
// not correctly rounded, so every rounding here is decided without one:
// each grid point or midpoint times its divisor is exact in F32, and the
// dividend against it decides the exact quotient. Rows are finite.

// 2^(e - 127) for a biased exponent e in 1 .. 254.
float nvfp4_power(uint e) { return uintBitsToFloat(e << 23); }

// The global scale of a row whose largest magnitude is amax: the binade of
// an approximate amax / 2688, corrected by one exact comparison. A finite
// row's g is at most 2^117, so its inverse is normal.
float nvfp4_global_scale(float amax) {
    const uint bits = floatBitsToUint(amax * (1.0 / 2688.0));
    const uint near = clamp((bits >> 23) + ((bits & 0x7fffffu) != 0u ? 1u : 0u), 1u, 254u);
    if (near > 1u && 2688.0 * nvfp4_power(near - 1u) >= amax)
        return nvfp4_power(near - 1u);
    if (near < 254u && 2688.0 * nvfp4_power(near) < amax)
        return nvfp4_power(near + 1u);
    return nvfp4_power(near);
}

// g s for a block whose largest magnitude is block_max: s is the UE4M3
// value (exponent bias 7, three mantissa bits, subnormal quantum 2^-9)
// nearest w / 6 for w = block_max / g (exact: g is a power of two). The
// quantum is that of an approximate w / 6's binade (both binades' grids
// hold the boundary, the nearest value to any quotient that rounds across
// it); t = w / quantum is exact, and the approximate index t / 6 is
// corrected against its exact midpoints 6 index +- 3. A finite row's w is
// at most 2688, so s is at most 448.
float nvfp4_scale(float block_max, float g) {
    const float w = block_max * nvfp4_power(254u - (floatBitsToUint(g) >> 23));
    const uint exponent = max((floatBitsToUint(w * (1.0 / 6.0)) >> 23) & 255u, 121u);
    const float t = w * nvfp4_power(257u - exponent);
    const int near = int(roundEven(t * (1.0 / 6.0)));
    const bool odd = (near & 1) != 0;
    const float above = float(6 * near + 3), below = float(6 * near - 3);
    const int index = (t > above || (t == above && odd)) ? near + 1
        : (t < below || (t == below && odd))             ? near - 1
                                                         : near;
    return g * (float(index) * nvfp4_power(exponent - 3u));
}

// x's E2M1 code nearest x / d, saturating at 6, times d: each threshold
// times d is exact, so |x| against it decides the exact quotient (ties to
// the even code). A zero d gives (positive) zero.
float nvfp4_value(float x, float d) {
    if (d == 0.0)
        return 0.0;
    const float a = abs(x);
    const float m = a <= 0.25 * d ? 0.0
        : a < 0.75 * d            ? 0.5
        : a <= 1.25 * d           ? 1.0
        : a < 1.75 * d            ? 1.5
        : a <= 2.5 * d            ? 2.0
        : a < 3.5 * d             ? 3.0
        : a <= 5.0 * d            ? 4.0
                                  : 6.0;
    return (x < 0.0 ? -m : m) * d;
}

vec4 nvfp4_values(vec4 x, float d) {
    return vec4(nvfp4_value(x.x, d), nvfp4_value(x.y, d), nvfp4_value(x.z, d), nvfp4_value(x.w, d));
}

float nvfp4_magnitude(vec4 even, vec4 odd) {
    const vec4 m = max(abs(even), abs(odd));
    return max(max(m.x, m.y), max(m.z, m.w));
}

// x[m, k..k+8) as the operands the prologue's recipe forms, given the norm
// inverse of row m and, for NVFP4 operands, its global scale g: the 16-value
// block that holds k formed under g (K's tail past `columns` is zero, as
// `projection_load8` loads it).
void projection_operand_load8(projection_prologue in_, uint m, uint k, float inverse, float g, out vec4 even,
    out vec4 odd) {
    if (in_.operands == PROJECTION_OPERANDS_ROUNDED) {
        projection_load8(in_, m, k, inverse, even, odd);
        return;
    }
    const uint block = k & ~15u;
    vec4 e0, o0, e1, o1;
    projection_load8(in_, m, block, inverse, e0, o0);
    projection_load8(in_, m, block + 8u, inverse, e1, o1);
    const float d = nvfp4_scale(max(nvfp4_magnitude(e0, o0), nvfp4_magnitude(e1, o1)), g);
    const bool upper = (k & 8u) != 0u;
    even = nvfp4_values(upper ? e1 : e0, d);
    odd = nvfp4_values(upper ? o1 : o0, d);
}
