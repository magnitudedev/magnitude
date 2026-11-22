// The output entries' joins of an F32 output into the hyper-connection
// streams (`scatter`, functions.seismic): the inject scatter and, when the
// inject row carries one, its comb. Apart from the projection family so an
// entry with no activation element or weights (`stream_join`) joins through
// the same code. The counterpart of `cuda/lib/projection/join.cuh`.
//
// This file is independent of any entry ABI. Addresses are bytes; strides are
// in F32 elements.

// Column n of one row's `streams` streams of `width` columns: each target
// stream d of `result` (columns `result_stride` apart) the residual stream d of
// `residual` (columns `residual_stride` apart) plus y inject[d], or, `combed`,
// the residual streams mixed by the comb after the inject values
// (sum_s comb[s, d] residual[s], sources ascending, the comb source major)
// plus y inject[d]. `inject` is the row's inject values, contiguous.
void join_scatter(Out result, uint64_t result_stride, uint64_t residual, uint64_t residual_stride, uint64_t inject,
    uint streams, uint width, const bool combed, uint64_t n, float y) {
    if (!combed) {
        for (uint d = 0u; d < streams; ++d) {
            const uint64_t col = uint64_t(d) * uint64_t(width) + n;
            const float mixed = element_f32_at(residual + col * residual_stride * 4ul);
            out_put(result, col * result_stride, mixed + y * element_f32_at(inject + uint64_t(d) * 4ul));
        }
        return;
    }
    const uint64_t comb = inject + uint64_t(streams) * 4ul;
    if (streams <= 8u) {
        float src[8];
        for (uint s = 0u; s < streams; ++s)
            src[s] = element_f32_at(residual + (uint64_t(s) * uint64_t(width) + n) * residual_stride * 4ul);
        for (uint d = 0u; d < streams; ++d) {
            float mixed = 0.0;
            for (uint s = 0u; s < streams; ++s)
                mixed = mixed + element_f32_at(comb + uint64_t(s * streams + d) * 4ul) * src[s];
            out_put(result, (uint64_t(d) * uint64_t(width) + n) * result_stride,
                mixed + y * element_f32_at(inject + uint64_t(d) * 4ul));
        }
    } else {
        for (uint d = 0u; d < streams; ++d) {
            float mixed = 0.0;
            for (uint s = 0u; s < streams; ++s)
                mixed = mixed + element_f32_at(comb + uint64_t(s * streams + d) * 4ul)
                    * element_f32_at(residual + (uint64_t(s) * uint64_t(width) + n) * residual_stride * 4ul);
            out_put(result, (uint64_t(d) * uint64_t(width) + n) * result_stride,
                mixed + y * element_f32_at(inject + uint64_t(d) * 4ul));
        }
    }
}

vec4 join_detail_load4(uint64_t base, uint64_t stride, uint64_t column, bool aligned_vec4) {
    if (aligned_vec4)
        return element_vec4_at(base + column * 4ul);
    return vec4(
        element_f32_at(base + column * stride * 4ul),
        element_f32_at(base + (column + 1ul) * stride * 4ul),
        element_f32_at(base + (column + 2ul) * stride * 4ul),
        element_f32_at(base + (column + 3ul) * stride * 4ul));
}

void join_detail_put4(Out result, uint64_t stride, uint64_t column, vec4 value) {
    if (stride == 1ul) {
        out_put2(result, column, value.x, value.y);
        out_put2(result, column + 2ul, value.z, value.w);
    } else {
        out_put(result, column * stride, value.x);
        out_put(result, (column + 1ul) * stride, value.y);
        out_put(result, (column + 2ul) * stride, value.z);
        out_put(result, (column + 3ul) * stride, value.w);
    }
}

// Four contiguous columns n .. n + 3 of one row's `streams` streams of `width`
// columns: 128-bit `element_vec4_at` residual loads when unit-strided and
// 16-byte aligned, `out_put2` stores on unit result stride, hoisted `inject`
// and `comb` registers, and `vec4 src[8]` cache for combed streams.
void join_scatter4(Out result, uint64_t result_stride, uint64_t residual, uint64_t residual_stride, uint64_t inject,
    uint streams, uint width, const bool combed, uint64_t n, vec4 y) {
    const bool residual_vec4 = residual_stride == 1ul
        && ((residual + n * 4ul) & 15ul) == 0ul
        && (streams <= 1u || (width & 3u) == 0u);
    if (!combed) {
        for (uint d = 0u; d < streams; ++d) {
            const uint64_t col = uint64_t(d) * uint64_t(width) + n;
            const float inj = element_f32_at(inject + uint64_t(d) * 4ul);
            const vec4 base = join_detail_load4(residual, residual_stride, col, residual_vec4);
            join_detail_put4(result, result_stride, col, base + y * inj);
        }
        return;
    }
    const uint64_t comb = inject + uint64_t(streams) * 4ul;
    if (streams <= 8u) {
        vec4 src[8];
        for (uint s = 0u; s < streams; ++s)
            src[s] = join_detail_load4(residual, residual_stride, uint64_t(s) * uint64_t(width) + n, residual_vec4);
        for (uint d = 0u; d < streams; ++d) {
            const float inj = element_f32_at(inject + uint64_t(d) * 4ul);
            vec4 mixed = vec4(0.0);
            for (uint s = 0u; s < streams; ++s) {
                const float c = element_f32_at(comb + uint64_t(s * streams + d) * 4ul);
                mixed = mixed + c * src[s];
            }
            join_detail_put4(result, result_stride, uint64_t(d) * uint64_t(width) + n, mixed + y * inj);
        }
    } else {
        for (uint d = 0u; d < streams; ++d) {
            const float inj = element_f32_at(inject + uint64_t(d) * 4ul);
            vec4 mixed = vec4(0.0);
            for (uint s = 0u; s < streams; ++s) {
                const float c = element_f32_at(comb + uint64_t(s * streams + d) * 4ul);
                const vec4 sv = join_detail_load4(
                    residual, residual_stride, uint64_t(s) * uint64_t(width) + n, residual_vec4);
                mixed = mixed + c * sv;
            }
            join_detail_put4(result, result_stride, uint64_t(d) * uint64_t(width) + n, mixed + y * inj);
        }
    }
}
