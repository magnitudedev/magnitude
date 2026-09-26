// stage shared projection mechanisms; included at the original declaration point.
// ---------------------------------------------------------------------------
// Operands.

// Activation row m: AllRows (table 0) or SelectedRows.
uint projection_row(uint64_t table, uint m) {
    return table == 0ul ? m : uint(element_i32_at(table + uint64_t(m) * 4ul));
}

#define PROJECTION_PLAIN 0
#define PROJECTION_RMS 1
#define PROJECTION_GATED_RMS 2
#define PROJECTION_GROUPED 3

struct projection_prologue {
    int kind;
    int act;            // ELEMENT_* of the activation A (bf16 or f16)
    uint64_t x;         // Plain/Grouped: A rows; Rms: F32 rows; GatedRms: mixed [rows, heads, W] in A
    uint64_t x0;        // element strides of x (GatedRms: row, head, column)
    uint64_t x1;
    uint64_t x2;
    uint columns;       // K
    uint64_t rows;      // row map (0: AllRows); Grouped: the order row (i32)
    uint64_t rows_stride;
    uint64_t norm;      // Rms/GatedRms: norm weights
    uint64_t norm_stride;
    int norm_kind;      // ELEMENT_* of the norm
    float eps;
    uint64_t z;         // GatedRms: the projection row holding z (A)
    uint64_t z0;
    uint64_t z1;
    uint64_t z_column;
    uint heads;         // GatedRms: norm groups per row
    uint head_width;    // GatedRms: W (a multiple of 8, at most 256)
};

projection_prologue projection_plain(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t rows) {
    return projection_prologue(PROJECTION_PLAIN, act, x, stride0, stride1, 0ul, columns, rows, 1ul, 0ul, 0ul,
        ELEMENT_F32, 0.0, 0ul, 0ul, 0ul, 0ul, 0u, 0u);
}

projection_prologue projection_rms(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint64_t norm,
    uint64_t norm_stride, const int norm_kind, float eps, uint columns, uint64_t rows) {
    return projection_prologue(PROJECTION_RMS, act, x, stride0, stride1, 0ul, columns, rows, 1ul, norm, norm_stride,
        norm_kind, eps, 0ul, 0ul, 0ul, 0ul, 1u, columns);
}

projection_prologue projection_gated_rms(const int act, uint64_t mixed, uint64_t mixed0, uint64_t mixed1,
    uint64_t mixed2, uint64_t z, uint64_t z0, uint64_t z1, uint64_t z_column, uint64_t norm, uint64_t norm_stride,
    const int norm_kind, float eps, uint heads, uint head_width, uint64_t rows) {
    return projection_prologue(PROJECTION_GATED_RMS, act, mixed, mixed0, mixed1, mixed2, heads * head_width, rows, 1ul,
        norm, norm_stride, norm_kind, eps, z, z0, z1, z_column, heads, head_width);
}

// Tile row m reads activation row order[m * order_stride]; -1 is padding.
projection_prologue projection_grouped(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t order, uint64_t order_stride) {
    return projection_prologue(PROJECTION_GROUPED, act, x, stride0, stride1, 0ul, columns, order, order_stride, 0ul,
        0ul, ELEMENT_F32, 0.0, 0ul, 0ul, 0ul, 0ul, 0u, 0u);
}

// Norm groups per row and their width.
uint projection_groups(projection_prologue in_) { return in_.kind == PROJECTION_GATED_RMS ? in_.heads : 1u; }
uint projection_width(projection_prologue in_) {
    return in_.kind == PROJECTION_GATED_RMS ? in_.head_width : in_.columns;
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

float projection_mixed_at(projection_prologue in_, uint m, uint head, uint i) {
    return element_at(in_.act, in_.x,
        uint64_t(projection_row(in_.rows, m)) * in_.x0 + uint64_t(head) * in_.x1 + uint64_t(i) * in_.x2);
}

// The input a norm group reduces: element i of group `group` of row m.
float projection_norm_input(projection_prologue in_, uint m, uint group, uint i) {
    if (in_.kind == PROJECTION_GATED_RMS)
        return projection_mixed_at(in_, m, group, i);
    return element_f32_at(in_.x + (uint64_t(projection_row(in_.rows, m)) * in_.x0 + uint64_t(i) * in_.x1) * 4ul);
}

float projection_silu_rounded(const int act, float gate) {
    return element_round(act, seismic_div_rn(gate, 1.0 + exp(-gate)));
}

// The GatedRms inputs of columns k..k+7 (one head) and their prologue output.
struct projection_gated8 {
    vec4 me, mo, ze, zo, ne, no;
};

projection_gated8 projection_gated_inputs8(projection_prologue in_, uint m, uint k) {
    const uint head = k / in_.head_width, i = k % in_.head_width;
    const uint64_t row = uint64_t(projection_row(in_.rows, m));
    const uint64_t mixed_at = row * in_.x0 + uint64_t(head) * in_.x1 + uint64_t(i) * in_.x2;
    const uint64_t z_at = row * in_.z0 + (in_.z_column + uint64_t(k)) * in_.z1;
    projection_gated8 v;
    projection_load8_strided(in_.act, in_.x, mixed_at, in_.x2, 8u, in_.x2 == 1ul && (mixed_at & 7ul) == 0ul, v.me, v.mo);
    projection_load8_strided(in_.act, in_.z, z_at, in_.z1, 8u, in_.z1 == 1ul && (z_at & 7ul) == 0ul, v.ze, v.zo);
    projection_load8_strided(in_.norm_kind, in_.norm, uint64_t(i) * in_.norm_stride, in_.norm_stride, 8u,
        in_.norm_stride == 1ul, v.ne, v.no);
    return v;
}

void projection_gated_finish8(projection_prologue in_, projection_gated8 v, float inverse, out vec4 even, out vec4 odd) {
    const vec4 e = v.me * inverse * v.ne, o = v.mo * inverse * v.no;
    [[unroll]] for (uint j = 0u; j < 4u; ++j) {
        even[j] = element_round(in_.act, element_round(in_.act, e[j]) * projection_silu_rounded(in_.act, v.ze[j]));
        odd[j] = element_round(in_.act, element_round(in_.act, o[j]) * projection_silu_rounded(in_.act, v.zo[j]));
    }
}

// x[m, k..k+8) (k a multiple of 8) as (even, odd) given the norm inverse of
// row m's group of column k (Plain and Grouped ignore it); elements at or
// past `columns` are zero.
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
    } else if (in_.kind == PROJECTION_RMS) {
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
    } else {
        projection_gated_finish8(in_, projection_gated_inputs8(in_, m, k), inverse, even, odd);
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

