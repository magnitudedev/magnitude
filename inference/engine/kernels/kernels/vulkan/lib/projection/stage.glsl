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

struct projection_prologue {
    int kind;
    int act;            // ELEMENT_* of the activation A (bf16 or f16)
    uint64_t x;         // Plain/Grouped: A rows; Rms: F32 rows
    uint64_t x0;        // element strides of x
    uint64_t x1;
    uint columns;       // K
    uint64_t rows;      // row map (0: AllRows); Grouped: the order row (i32)
    uint64_t rows_stride;
    uint64_t norm;      // Rms: norm weights
    uint64_t norm_stride;
    int norm_kind;      // ELEMENT_* of the norm
    float eps;
};

projection_prologue projection_plain(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t rows) {
    return projection_prologue(PROJECTION_PLAIN, act, x, stride0, stride1, columns, rows, 1ul, 0ul, 0ul,
        ELEMENT_F32, 0.0);
}

projection_prologue projection_rms(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint64_t norm,
    uint64_t norm_stride, const int norm_kind, float eps, uint columns, uint64_t rows) {
    return projection_prologue(PROJECTION_RMS, act, x, stride0, stride1, columns, rows, 1ul, norm, norm_stride,
        norm_kind, eps);
}

// Tile row m reads activation row order[m * order_stride]; -1 is padding.
projection_prologue projection_grouped(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t order, uint64_t order_stride) {
    return projection_prologue(PROJECTION_GROUPED, act, x, stride0, stride1, columns, order, order_stride, 0ul,
        0ul, ELEMENT_F32, 0.0);
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

