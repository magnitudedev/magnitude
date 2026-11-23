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
#define PROJECTION_ORDERED_RMS 4
#define PROJECTION_ORDERED_SUMMED 5

// The operand recipe of the weights a prologue's rows feed (the portable
// `operand_rows`): the rows as they are, or as NVFP4 operands. An entry reads
// its weights' recipe through their prefix (`PACKETS_OPERANDS_NVFP4(W)`,
// <seismic/producers.glsl>).
#define PROJECTION_OPERANDS_ROUNDED 0
#define PROJECTION_OPERANDS_NVFP4 1

struct projection_prologue {
    int kind;
    int act;            // ELEMENT_* of the activation A (bf16 or f16)
    Stored x;           // Plain/Grouped: stored rows; Rms/Summed: F32 rows
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

projection_prologue projection_plain(const int act, Stored x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t rows) {
    return projection_prologue(PROJECTION_PLAIN, act, x, stride0, stride1, columns, rows, 1ul, 0ul, 0ul,
        ELEMENT_F32, 0.0, 1u, PROJECTION_OPERANDS_ROUNDED);
}

projection_prologue projection_plain(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t rows) {
    return projection_plain(act, stored_at_buffer(act, x), stride0, stride1, columns, rows);
}

projection_prologue projection_rms(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint64_t norm,
    uint64_t norm_stride, const int norm_kind, float eps, uint columns, uint64_t rows) {
    return projection_prologue(PROJECTION_RMS, act, stored_at_buffer(ELEMENT_F32, x), stride0, stride1, columns, rows,
        1ul, norm, norm_stride, norm_kind, eps, 1u, PROJECTION_OPERANDS_ROUNDED);
}

// The sum in part order of the `parts` parts of an F32 row, a hyper-connection
// mix site's per-stream partials (`form_input` without a norm). Its square
// sums are zero and it ignores the inverse it is given.
projection_prologue projection_summed(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t rows, uint parts) {
    return projection_prologue(PROJECTION_SUMMED, act, stored_at_buffer(ELEMENT_F32, x), stride0, stride1, columns,
        rows, 1ul, 0ul, 0ul, ELEMENT_F32, 0.0, parts, PROJECTION_OPERANDS_ROUNDED);
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
projection_prologue projection_grouped(const int act, Stored x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t order, uint64_t order_stride) {
    return projection_prologue(PROJECTION_GROUPED, act, x, stride0, stride1, columns, order, order_stride, 0ul,
        0ul, ELEMENT_F32, 0.0, 1u, PROJECTION_OPERANDS_ROUNDED);
}

projection_prologue projection_grouped(const int act, uint64_t x, uint64_t stride0, uint64_t stride1, uint columns,
    uint64_t order, uint64_t order_stride) {
    return projection_grouped(act, stored_at_buffer(act, x), stride0, stride1, columns, order, order_stride);
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
        sum = seismic_add_rn(sum, stored_at(in_.x, row + uint64_t(part * in_.columns + k) * in_.x1));
    return sum;
}

// Eight consecutive elements from element `index` of `source` with element
// stride `stride`, as (0,2,4,6), (1,3,5,7); elements past the first `valid`
// are zero. `vector` allows one aligned load.
void projection_load8_strided(Stored source, uint64_t index, uint64_t stride, uint valid,
    bool vector, out vec4 even, out vec4 odd) {
    if (vector && valid >= 8u) {
        stored_load8(source, index, even, odd);
        return;
    }
    float v[8];
    [[unroll]] for (uint i = 0u; i < 8u; ++i)
        v[i] = i < valid ? stored_at(source, index + uint64_t(i) * stride) : 0.0;
    even = vec4(v[0], v[2], v[4], v[6]);
    odd = vec4(v[1], v[3], v[5], v[7]);
}

void projection_load8_strided(const int kind, uint64_t base, uint64_t index, uint64_t stride, uint valid,
    bool vector, out vec4 even, out vec4 odd) {
    projection_load8_strided(stored_at_buffer(kind, base), index, stride, valid, vector, even, odd);
}

// The input the Rms norm reduces: element i of row m.
float projection_norm_input(projection_prologue in_, uint m, uint i) {
    return stored_at(in_.x, uint64_t(projection_row(in_.rows, m)) * in_.x0 + uint64_t(i) * in_.x1);
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
        projection_load8_strided(in_.x, uint64_t(row) * in_.x0 + uint64_t(k) * in_.x1, in_.x1, valid,
            in_.x1 == 1ul && (in_.x0 & 7ul) == 0ul && stored_aligned16(in_.x), even, odd);
    } else if (in_.kind == PROJECTION_SUMMED || in_.kind == PROJECTION_ORDERED_SUMMED) {
        float v[8];
        [[unroll]] for (uint i = 0u; i < 8u; ++i)
            v[i] = i < valid ? element_round(in_.act, projection_summed_at(in_, m, k + i)) : 0.0;
        even = vec4(v[0], v[2], v[4], v[6]);
        odd = vec4(v[1], v[3], v[5], v[7]);
    } else {
        const uint64_t row = uint64_t(projection_row(in_.rows, m)) * in_.x0;
        vec4 xe, xo;
        projection_load8_strided(in_.x, row + uint64_t(k) * in_.x1, in_.x1, valid,
            in_.x1 == 1ul && stored_aligned16(stored_offset(in_.x, row)), xe, xo);
        if (valid == 8u) {
            vec4 ne, no;
            projection_load8_strided(in_.norm_kind, in_.norm, uint64_t(k) * in_.norm_stride, in_.norm_stride, 8u,
                in_.norm_stride == 1ul && (in_.norm & 15ul) == 0ul, ne, no);
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
// c = 32 * part + lane + 32 * parts * j, in order (`seismic_strided`: the
// caller reduces the shares over the subgroup).
#define PROJECTION_RMS_PARTS 8u

float projection_rms_squares(projection_prologue in_, uint m, uint part, uint lane) {
    if (in_.kind == PROJECTION_SUMMED)
        return 0.0;
    const uint64_t row = uint64_t(projection_row(in_.rows, m)) * in_.x0;
    const bool vector = in_.x1 == 1ul && (in_.x0 & 3ul) == 0ul && (in_.columns & 3u) == 0u;
    float sum = 0.0;
    seismic_strided(uint, c, 4u * (32u * part + lane), in_.columns, 128u * PROJECTION_RMS_PARTS) {
        vec4 v;
        if (vector) {
            v = stored_vec4_at(in_.x, row + uint64_t(c));
        } else {
            [[unroll]] for (uint i = 0u; i < 4u; ++i)
                v[i] = c + i < in_.columns ? stored_at(in_.x, row + uint64_t(c + i) * in_.x1) : 0.0;
        }
        sum = seismic_fma_rn(v.x, v.x, sum);
        sum = seismic_fma_rn(v.y, v.y, sum);
        sum = seismic_fma_rn(v.z, v.z, sum);
        sum = seismic_fma_rn(v.w, v.w, sum);
    }
    return sum;
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


// ---------------------------------------------------------------------------
// Reference-order prologues (`projection_ordered_rms`,
// `projection_ordered_summed`, `projection_ordered_input`): 8-lane reduction
// (`reduce_lanes_sum(v, 8u)` = xor 4, 2, 1) and `1 / sqrt(...)`.

projection_prologue projection_ordered_rms(const int act, uint64_t x, uint64_t stride0, uint64_t stride1,
    uint64_t norm, uint64_t norm_stride, const int norm_kind, float eps, uint columns, uint64_t rows) {
    return projection_prologue(PROJECTION_ORDERED_RMS, act, stored_at_buffer(ELEMENT_F32, x), stride0, stride1, columns,
        rows, 1ul, norm, norm_stride, norm_kind, eps, 1u, PROJECTION_OPERANDS_ROUNDED);
}

projection_prologue projection_ordered_summed(const int act, uint64_t x, uint64_t stride0, uint64_t stride1,
    uint columns, uint64_t rows, uint parts) {
    return projection_prologue(PROJECTION_ORDERED_SUMMED, act, stored_at_buffer(ELEMENT_F32, x), stride0, stride1,
        columns, rows, 1ul, 0ul, 0ul, ELEMENT_F32, 0.0, parts, PROJECTION_OPERANDS_ROUNDED);
}

projection_prologue projection_ordered_input(const int act, uint64_t x, uint64_t stride0, uint64_t stride1,
    uint64_t norm, uint64_t norm_stride, const int norm_kind, float eps, uint columns, uint64_t rows,
    const uint norm_extent, const uint parts) {
    return norm_extent == 0u
        ? projection_ordered_summed(act, x, stride0, stride1, columns, rows, parts)
        : projection_ordered_rms(act, x, stride0, stride1, norm, norm_stride, norm_kind, eps, columns, rows);
}

float projection_ordered_combine8(float value) {
    value = seismic_add_rn(value, subgroupShuffleXor(value, 4u));
    value = seismic_add_rn(value, subgroupShuffleXor(value, 2u));
    value = seismic_add_rn(value, subgroupShuffleXor(value, 1u));
    return value;
}

// Accumulate `seismic_shared_f32[scratch_off + u], +8, ... < count` in exact
// 8-lane reference order.
float projection_ordered_accumulate_sram8(uint scratch_off, uint count, uint u, float sum) {
    uint k0 = 0u;
    for (; k0 + 64u <= count; k0 += 64u) {
        const uint c = k0 + u;
        const float v0 = seismic_shared_f32[scratch_off + c];
        const float v1 = seismic_shared_f32[scratch_off + c + 8u];
        const float v2 = seismic_shared_f32[scratch_off + c + 16u];
        const float v3 = seismic_shared_f32[scratch_off + c + 24u];
        const float v4 = seismic_shared_f32[scratch_off + c + 32u];
        const float v5 = seismic_shared_f32[scratch_off + c + 40u];
        const float v6 = seismic_shared_f32[scratch_off + c + 48u];
        const float v7 = seismic_shared_f32[scratch_off + c + 56u];
        sum = seismic_fma_rn(v0, v0, sum);
        sum = seismic_fma_rn(v1, v1, sum);
        sum = seismic_fma_rn(v2, v2, sum);
        sum = seismic_fma_rn(v3, v3, sum);
        sum = seismic_fma_rn(v4, v4, sum);
        sum = seismic_fma_rn(v5, v5, sum);
        sum = seismic_fma_rn(v6, v6, sum);
        sum = seismic_fma_rn(v7, v7, sum);
    }
    seismic_strided_from(uint, c, k0, u, count, 8u) {
        const float v = seismic_shared_f32[scratch_off + c];
        sum = seismic_fma_rn(v, v, sum);
    }
    return sum;
}

void projection_ordered_inverses(projection_prologue in_, uint m_rows, uint inv_off, uint inv_stride,
    uint first_row, bool has_scratch, uint scratch_off) {
    const uint tid = SEISMIC_THREAD, threads = SEISMIC_SUBGROUPS * 32u;
    if (in_.kind == PROJECTION_ORDERED_SUMMED) {
        for (uint r = tid; r < m_rows; r += threads)
            seismic_shared_f32[inv_off + r * inv_stride] = 1.0;
        barrier();
        return;
    }
    const uint group_id = tid >> 3u, u = tid & 7u, groups = threads >> 3u;
    if (has_scratch && in_.x1 == 1ul && m_rows == 1u) {
        const uint64_t row = uint64_t(projection_row(in_.rows, first_row)) * in_.x0;
        const bool aligned = stored_aligned16(stored_offset(in_.x, row));
        float sum = 0.0;
        for (uint k0 = 0u; k0 < in_.columns; k0 += 512u) {
            if (k0 > 0u)
                barrier();
            const uint chunk = min(512u, in_.columns - k0);
            if (aligned) {
                const uint chunk4 = chunk & ~3u;
                for (uint c = tid * 4u; c < chunk4; c += threads * 4u) {
                    const vec4 v = stored_vec4_at(in_.x, row + uint64_t(k0 + c));
                    seismic_shared_f32[scratch_off + c] = v.x;
                    seismic_shared_f32[scratch_off + c + 1u] = v.y;
                    seismic_shared_f32[scratch_off + c + 2u] = v.z;
                    seismic_shared_f32[scratch_off + c + 3u] = v.w;
                }
                for (uint c = chunk4 + tid; c < chunk; c += threads)
                    seismic_shared_f32[scratch_off + c] = stored_at(in_.x, row + uint64_t(k0 + c));
            } else {
                for (uint c = tid; c < chunk; c += threads)
                    seismic_shared_f32[scratch_off + c] = stored_at(in_.x, row + uint64_t(k0 + c));
            }
            barrier();
            if (group_id == 0u)
                sum = projection_ordered_accumulate_sram8(scratch_off, chunk, u, sum);
        }
        const float total_sq = projection_ordered_combine8(sum);
        if (tid == 0u)
            seismic_shared_f32[inv_off] =
                seismic_div_rn(1.0, sqrt(seismic_div_rn(total_sq, float(in_.columns)) + in_.eps));
        barrier();
        return;
    }
    for (uint base = 0u; base < m_rows; base += groups) {
        const uint r = base + group_id;
        float sum = 0.0;
        if (r < m_rows) {
            const uint64_t row = uint64_t(projection_row(in_.rows, first_row + r)) * in_.x0;
            if (in_.x1 == 1ul) {
                uint k0 = 0u;
                for (; k0 + 64u <= in_.columns; k0 += 64u) {
                    const uint c = k0 + u;
                    const float v0 = stored_at(in_.x, row + uint64_t(c));
                    const float v1 = stored_at(in_.x, row + uint64_t(c + 8u));
                    const float v2 = stored_at(in_.x, row + uint64_t(c + 16u));
                    const float v3 = stored_at(in_.x, row + uint64_t(c + 24u));
                    const float v4 = stored_at(in_.x, row + uint64_t(c + 32u));
                    const float v5 = stored_at(in_.x, row + uint64_t(c + 40u));
                    const float v6 = stored_at(in_.x, row + uint64_t(c + 48u));
                    const float v7 = stored_at(in_.x, row + uint64_t(c + 56u));
                    sum = seismic_fma_rn(v0, v0, sum);
                    sum = seismic_fma_rn(v1, v1, sum);
                    sum = seismic_fma_rn(v2, v2, sum);
                    sum = seismic_fma_rn(v3, v3, sum);
                    sum = seismic_fma_rn(v4, v4, sum);
                    sum = seismic_fma_rn(v5, v5, sum);
                    sum = seismic_fma_rn(v6, v6, sum);
                    sum = seismic_fma_rn(v7, v7, sum);
                }
                seismic_strided_from(uint, c, k0, u, in_.columns, 8u) {
                    const float v = stored_at(in_.x, row + uint64_t(c));
                    sum = seismic_fma_rn(v, v, sum);
                }
            } else {
                seismic_strided(uint, c, u, in_.columns, 8u) {
                    const float v = stored_at(in_.x, row + uint64_t(c) * in_.x1);
                    sum = seismic_fma_rn(v, v, sum);
                }
            }
        }
        const float total_sq = projection_ordered_combine8(sum);
        if (r < m_rows && u == 0u)
            seismic_shared_f32[inv_off + r * inv_stride] =
                seismic_div_rn(1.0, sqrt(seismic_div_rn(total_sq, float(in_.columns)) + in_.eps));
    }
    barrier();
}

float projection_ordered_at(projection_prologue in_, uint m, uint c, float inv) {
    if (in_.kind == PROJECTION_ORDERED_SUMMED)
        return element_round(in_.act, projection_summed_at(in_, m, c));
    const uint64_t row = uint64_t(projection_row(in_.rows, m)) * in_.x0;
    const float v = stored_at(in_.x, row + uint64_t(c) * in_.x1);
    const float nw = element_at(in_.norm_kind, in_.norm, uint64_t(c) * in_.norm_stride);
    return element_round(in_.act, v * inv * nw);
}
