// stage shared projection mechanisms; included at the original declaration point.
// ---------------------------------------------------------------------------
// Prologues. `load8(m, k, inverse, ...)` returns x[m, k..k+8) (k a multiple
// of 8) as (even, odd) float4s given the norm inverse of row m's group of
// column k (plain operands ignore it); elements at or past `columns` are
// zero. `value` is the scalar form. A normed prologue also exposes its norm
// groups (`groups`, `width`, `norm_input`, `epsilon`) to `device_normalize`.

template <typename A>
inline void load8_storage(device const typename A::storage *row, ulong stride, uint k, uint columns,
    bool vector, thread float4 &even, thread float4 &odd) {
    if (vector && k + 8u <= columns) {
        A::split8(*reinterpret_cast<device const uint4 *>(row + k), even, odd);
        return;
    }
    float v[8];
    for (uint i = 0; i < 8; ++i)
        v[i] = k + i < columns ? A::load(row[ulong(k + i) * stride]) : 0.0f;
    even = float4(v[0], v[2], v[4], v[6]);
    odd = float4(v[1], v[3], v[5], v[7]);
}

// The storage words of x[.., k..k+8) (k a multiple of 8) exactly as stored:
// element 2i in the low half of word i; elements at or past `columns` are
// zero. Plain operands expose them as `words8(m, k)`, which the GEMM stages
// unconverted.
template <typename A>
inline uint4 words8_full_storage(device const typename A::storage *row, ulong stride, uint k, bool vector) {
    if (vector)
        return *reinterpret_cast<device const uint4 *>(row + k);
    const uint w0 = uint(as_type<ushort>(row[ulong(k + 0u) * stride]))
        | (uint(as_type<ushort>(row[ulong(k + 1u) * stride])) << 16u);
    const uint w1 = uint(as_type<ushort>(row[ulong(k + 2u) * stride]))
        | (uint(as_type<ushort>(row[ulong(k + 3u) * stride])) << 16u);
    const uint w2 = uint(as_type<ushort>(row[ulong(k + 4u) * stride]))
        | (uint(as_type<ushort>(row[ulong(k + 5u) * stride])) << 16u);
    const uint w3 = uint(as_type<ushort>(row[ulong(k + 6u) * stride]))
        | (uint(as_type<ushort>(row[ulong(k + 7u) * stride])) << 16u);
    return uint4(w0, w1, w2, w3);
}

template <typename A>
inline uint4 words8_storage(device const typename A::storage *row, ulong stride, uint k, uint columns,
    bool vector) {
    if (k + 8u <= columns)
        return words8_full_storage<A>(row, stride, k, vector);
    uint w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u;
    if (k + 0u < columns) w0 |= uint(as_type<ushort>(row[ulong(k + 0u) * stride]));
    if (k + 1u < columns) w0 |= uint(as_type<ushort>(row[ulong(k + 1u) * stride])) << 16u;
    if (k + 2u < columns) w1 |= uint(as_type<ushort>(row[ulong(k + 2u) * stride]));
    if (k + 3u < columns) w1 |= uint(as_type<ushort>(row[ulong(k + 3u) * stride])) << 16u;
    if (k + 4u < columns) w2 |= uint(as_type<ushort>(row[ulong(k + 4u) * stride]));
    if (k + 5u < columns) w2 |= uint(as_type<ushort>(row[ulong(k + 5u) * stride])) << 16u;
    if (k + 6u < columns) w3 |= uint(as_type<ushort>(row[ulong(k + 6u) * stride]));
    if (k + 7u < columns) w3 |= uint(as_type<ushort>(row[ulong(k + 7u) * stride])) << 16u;
    return uint4(w0, w1, w2, w3);
}

// Eight norm weights of element type N from index k (a multiple of 8), as
// (even, odd); every index is in range.
template <typename N>
inline void norm8_scalar(device const uchar *norm, ulong stride, uint k, thread float4 &even,
    thread float4 &odd) {
    float v[8];
    for (uint i = 0; i < 8; ++i)
        v[i] = element::at<N>(norm, ulong(k + i) * stride);
    even = float4(v[0], v[2], v[4], v[6]);
    odd = float4(v[1], v[3], v[5], v[7]);
}

template <typename N>
struct norm8 {
    static void load(device const uchar *norm, ulong stride, uint k, thread float4 &even, thread float4 &odd) {
        norm8_scalar<N>(norm, stride, k, even, odd);
    }
};

// 16-bit norms with unit stride load as one uint4.
template <typename N>
inline void norm8_packed(device const uchar *norm, ulong stride, uint k, thread float4 &even,
    thread float4 &odd) {
    if (stride == 1)
        N::split8(*reinterpret_cast<device const uint4 *>(norm + ulong(k) * 2u), even, odd);
    else
        norm8_scalar<N>(norm, stride, k, even, odd);
}
template <>
struct norm8<element::Bf16> {
    static void load(device const uchar *norm, ulong stride, uint k, thread float4 &even, thread float4 &odd) {
        norm8_packed<element::Bf16>(norm, stride, k, even, odd);
    }
};
template <>
struct norm8<element::F16> {
    static void load(device const uchar *norm, ulong stride, uint k, thread float4 &even, thread float4 &odd) {
        norm8_packed<element::F16>(norm, stride, k, even, odd);
    }
};

template <typename A, typename Rows>
struct Plain {
    static_assert(A::bytes == 2, "the Metal projection family requires a bf16 or f16 activation element");
    typedef A activation;
    device const uchar *x;
    ulong stride0, stride1;
    uint columns;
    Rows rows;
    uint width() const { return columns; }
    float value(uint m, uint k, float) const {
        return A::load(reinterpret_cast<device const typename A::storage *>(x)
            [ulong(rows.at(m)) * stride0 + ulong(k) * stride1]);
    }
    void load8(uint m, uint k, float, thread float4 &even, thread float4 &odd) const {
        device const typename A::storage *row =
            reinterpret_cast<device const typename A::storage *>(x) + ulong(rows.at(m)) * stride0;
        load8_storage<A>(row, stride1, k, columns, stride1 == 1 && (stride0 & 7u) == 0, even, odd);
    }
    uint4 words8_full(uint m, uint k) const {
        device const typename A::storage *row =
            reinterpret_cast<device const typename A::storage *>(x) + ulong(rows.at(m)) * stride0;
        return words8_full_storage<A>(row, stride1, k, stride1 == 1 && (stride0 & 7u) == 0);
    }
    uint4 words8(uint m, uint k) const {
        device const typename A::storage *row =
            reinterpret_cast<device const typename A::storage *>(x) + ulong(rows.at(m)) * stride0;
        return words8_storage<A>(row, stride1, k, columns, stride1 == 1 && (stride0 & 7u) == 0);
    }
    // x[m, k] and x[m, k + 1] as one storage word (k + 1 < columns), zero
    // for a row at or past `m_rows`.
    uint words2(uint m, uint k, uint m_rows) const {
        if (m >= m_rows)
            return 0u;
        device const typename A::storage *row =
            reinterpret_cast<device const typename A::storage *>(x) + ulong(rows.at(m)) * stride0;
        if (stride1 == 1)
            return as_type<uint>(ushort2(*reinterpret_cast<device const packed_ushort2 *>(row + k)));
        return uint(as_type<ushort>(row[ulong(k) * stride1]))
            | uint(as_type<ushort>(row[ulong(k + 1u) * stride1])) << 16u;
    }
};

// The sum in part order of the PARTS `columns`-wide parts of an F32 row, a
// hyper-connection mix site's per-stream partials (`form_input` without a
// norm). It has `Rms`'s fields and interface, so an entry selects either by
// its norm's extent (`InputOf`): `norm`, `norm_stride` and `eps` are not
// read, and its square sums are zero (the inverse it is given is ignored).
template <typename A, typename N, typename Rows, uint PARTS>
struct Summed {
    static_assert(A::bytes == 2, "the Metal projection family requires a bf16 or f16 activation element");
    typedef A activation;
    device const float *x;
    ulong stride0, stride1;
    device const uchar *norm;
    ulong norm_stride;
    float eps;
    uint columns;
    Rows rows;
    uint groups() const { return 1; }
    uint width() const { return columns; }
    float norm_input(uint, uint, uint) const { return 0.0f; }
    float epsilon() const { return eps; }
    static constant constexpr uint parts = 8;
    float squares(uint, uint, uint, uint) const { return 0.0f; }
    float sum(uint m, uint column) const {
        device const float *row = x + ulong(rows.at(m)) * stride0;
        float total = 0.0f;
        for (uint part = 0; part < PARTS; ++part)
            total += row[ulong(part * columns + column) * stride1];
        return total;
    }
    float value(uint m, uint column, float) const { return A::round(sum(m, column)); }
    void load8(uint m, uint k, float, thread float4 &even, thread float4 &odd) const {
        float v[8];
        for (uint i = 0; i < 8; ++i)
            v[i] = k + i < columns ? A::round(sum(m, k + i)) : 0.0f;
        even = float4(v[0], v[2], v[4], v[6]);
        odd = float4(v[1], v[3], v[5], v[7]);
    }
};

template <typename A, typename N, typename Rows>
struct Rms;

// An entry's input prologue (`form_input`): RMS normalization with a norm
// of NORM_EXTENT values, or the sum of PARTS partials without one.
template <typename A, typename N, typename Rows, uint NORM_EXTENT, uint PARTS>
struct InputOf {
    typedef Rms<A, N, Rows> type;
};
template <typename A, typename N, typename Rows, uint PARTS>
struct InputOf<A, N, Rows, 0, PARTS> {
    typedef Summed<A, N, Rows, PARTS> type;
};

template <typename A, typename N, typename Rows>
struct Rms {
    static_assert(A::bytes == 2, "the Metal projection family requires a bf16 or f16 activation element");
    typedef A activation;
    device const float *x;
    ulong stride0, stride1;
    device const uchar *norm;
    ulong norm_stride;
    float eps;
    uint columns;
    Rows rows;
    uint groups() const { return 1; }
    uint width() const { return columns; }
    float norm_input(uint m, uint, uint i) const {
        return x[ulong(rows.at(m)) * stride0 + ulong(i) * stride1];
    }
    float epsilon() const { return eps; }
    // A row's square sum in `parts` parts. One lane's share of `part`: columns
    // 4c .. 4c + 3 for c = 32 * part + lane + 32 * parts * j, in order.
    static constant constexpr uint parts = 8;
    float squares(uint m, uint, uint part, uint lane) const {
        device const float *row = x + ulong(rows.at(m)) * stride0;
        bool vector = stride1 == 1 && (stride0 & 3u) == 0 && (columns & 3u) == 0;
        float sum = 0.0f;
        for (uint c = 4u * (32u * part + lane); c < columns; c += 128u * parts) {
            float4 v;
            if (vector) {
                v = *reinterpret_cast<device const float4 *>(row + c);
            } else {
                for (uint i = 0; i < 4; ++i)
                    v[i] = c + i < columns ? row[ulong(c + i) * stride1] : 0.0f;
            }
            sum = metal::fma(v.x, v.x, sum);
            sum = metal::fma(v.y, v.y, sum);
            sum = metal::fma(v.z, v.z, sum);
            sum = metal::fma(v.w, v.w, sum);
        }
        return sum;
    }
    float value(uint m, uint column, float inverse) const {
        float v = x[ulong(rows.at(m)) * stride0 + ulong(column) * stride1];
        return A::round(v * inverse * element::at<N>(norm, ulong(column) * norm_stride));
    }
    void load8(uint m, uint k, float inverse, thread float4 &even, thread float4 &odd) const {
        device const float *row = x + ulong(rows.at(m)) * stride0;
        if (stride1 == 1 && (stride0 & 3u) == 0 && k + 8u <= columns) {
            float4 a = *reinterpret_cast<device const float4 *>(row + k);
            float4 b = *reinterpret_cast<device const float4 *>(row + k + 4);
            float4 ne, no;
            norm8<N>::load(norm, norm_stride, k, ne, no);
            float4 e = float4(a.x, a.z, b.x, b.z) * inverse * ne;
            float4 o = float4(a.y, a.w, b.y, b.w) * inverse * no;
            for (uint i = 0; i < 4; ++i) {
                even[i] = A::round(e[i]);
                odd[i] = A::round(o[i]);
            }
            return;
        }
        float v[8];
        for (uint i = 0; i < 8; ++i)
            v[i] = k + i < columns
                ? A::round(row[ulong(k + i) * stride1] * inverse
                    * element::at<N>(norm, ulong(k + i) * norm_stride))
                : 0.0f;
        even = float4(v[0], v[2], v[4], v[6]);
        odd = float4(v[1], v[3], v[5], v[7]);
    }
};


// The per-stream RMS normalization of a row of hyper-connection streams
// (`group_rms`, `stream_mix_down`): `groups()` streams of `width()` columns,
// each normalized by its own inverse,
//   x = round_A(r[row(m), k] * inverse(m, k / width) * norm[k]),
// with `Rms`'s interface over the stream that holds column k (every stream's
// first column a multiple of 8, so no eight-column load straddles two).
template <typename A, typename N, typename Rows>
struct GroupRms {
    static_assert(A::bytes == 2, "the Metal projection family requires a bf16 or f16 activation element");
    typedef A activation;
    device const float *x;
    ulong stride0, stride1;
    device const uchar *norm;
    ulong norm_stride;
    float eps;
    uint columns;
    Rows rows;
    uint stream_width;
    uint groups() const { return columns / stream_width; }
    uint width() const { return stream_width; }
    float norm_input(uint m, uint group, uint i) const {
        return x[ulong(rows.at(m)) * stride0 + ulong(group * stream_width + i) * stride1];
    }
    float epsilon() const { return eps; }
    static constant constexpr uint parts = 8;
    // `Rms::squares` over stream `group`'s columns.
    float squares(uint m, uint group, uint part, uint lane) const {
        device const float *row = x + ulong(rows.at(m)) * stride0 + ulong(group * stream_width) * stride1;
        bool vector = stride1 == 1 && (stride0 & 3u) == 0 && (stream_width & 3u) == 0;
        float sum = 0.0f;
        for (uint c = 4u * (32u * part + lane); c < stream_width; c += 128u * parts) {
            float4 v;
            if (vector) {
                v = *reinterpret_cast<device const float4 *>(row + c);
            } else {
                for (uint i = 0; i < 4; ++i)
                    v[i] = c + i < stream_width ? row[ulong(c + i) * stride1] : 0.0f;
            }
            sum = metal::fma(v.x, v.x, sum);
            sum = metal::fma(v.y, v.y, sum);
            sum = metal::fma(v.z, v.z, sum);
            sum = metal::fma(v.w, v.w, sum);
        }
        return sum;
    }
    float value(uint m, uint column, float inverse) const {
        return Rms<A, N, Rows>{x, stride0, stride1, norm, norm_stride, eps, columns, rows}.value(m, column, inverse);
    }
    void load8(uint m, uint k, float inverse, thread float4 &even, thread float4 &odd) const {
        Rms<A, N, Rows>{x, stride0, stride1, norm, norm_stride, eps, columns, rows}.load8(m, k, inverse, even, odd);
    }
};

// The operand recipe of a product over the weights whose `operands(W)` are
// `RECIPES`, which the rows they share all take: NVFP4 (1) or rounded A (0).
template <bool... RECIPES>
struct OperandRecipe {
    static constant constexpr bool NVFP4 = (RECIPES || ...);
    static_assert(NVFP4 == (RECIPES && ...), "weights that share rows share their operand recipe");
};

// Row m's global scale g from threadgroup memory (`globals[m]`, the GEMV
// rows), or one row's held in a register (a normalizing threadgroup's).
struct RowGlobals {
    threadgroup const float *globals;
    float at(uint m) const { return globals[m]; }
};
struct OneGlobal {
    float g;
    float at(uint) const { return g; }
};

// The rows of prologue `In` as NVFP4 operands: `load8` forms the 16-value
// block that holds k (K's tail past `columns` is zero, as `In` loads it)
// under its row's global scale.
template <typename In, typename G>
struct Nvfp4Rows {
    typedef typename In::activation activation;
    In in;
    G globals;
    uint width() const { return in.width(); }
    void load8(uint m, uint k, float inverse, thread float4 &even, thread float4 &odd) const {
        const uint block = k & ~15u;
        float4 e0, o0, e1, o1;
        in.load8(m, block, inverse, e0, o0);
        in.load8(m, block + 8u, inverse, e1, o1);
        const float d = packets::nvfp4::scale(
            metal::max(packets::nvfp4::magnitude(e0, o0), packets::nvfp4::magnitude(e1, o1)), globals.at(m));
        const bool upper = (k & 8u) != 0u;
        even = packets::nvfp4::values(upper ? e1 : e0, d);
        odd = packets::nvfp4::values(upper ? o1 : o0, d);
    }
    // `load8`'s values as A storage words (exact: q g s fits A).
    uint4 words8_full(uint m, uint k) const { return words8(m, k); }
    uint4 words8(uint m, uint k) const {
        float4 even, odd;
        load8(m, k, 0.0f, even, odd);
        return activation::pack8(even, odd);
    }
};

// ---------------------------------------------------------------------------
// Reference-order prologues (`OrderedRms`, `OrderedSummed`, `OrderedInputOf`):
// 8-lane reduction (`((l0+l4)+(l2+l6))+((l1+l5)+(l3+l7))`) and `1 / sqrt(...)`.

inline float ordered_combine8(float v) {
    v = v + simd_shuffle_xor(v, ushort(4));
    v = v + simd_shuffle_xor(v, ushort(2));
    v = v + simd_shuffle_xor(v, ushort(1));
    return v;
}

// Accumulate `sp[u], sp[u + 8], ... < count` in exact 8-lane reference order.
inline float ordered_accumulate_sram8(threadgroup const float *sp, uint count, uint u, float sum) {
    uint c = u;
    for (; c + 56u < count; c += 64u) {
        const float v0 = sp[c], v1 = sp[c + 8u], v2 = sp[c + 16u], v3 = sp[c + 24u];
        const float v4 = sp[c + 32u], v5 = sp[c + 40u], v6 = sp[c + 48u], v7 = sp[c + 56u];
        sum = metal::fma(v0, v0, sum);
        sum = metal::fma(v1, v1, sum);
        sum = metal::fma(v2, v2, sum);
        sum = metal::fma(v3, v3, sum);
        sum = metal::fma(v4, v4, sum);
        sum = metal::fma(v5, v5, sum);
        sum = metal::fma(v6, v6, sum);
        sum = metal::fma(v7, v7, sum);
    }
    for (; c < count; c += 8u) {
        const float val = sp[c];
        sum = metal::fma(val, val, sum);
    }
    return sum;
}

// Power-of-two 2D thread partition across `(m_rows, width)` without runtime integer division.
inline void ordered_stage_partition(uint m_rows, uint width, uint tid, uint threads,
    thread uint &r0, thread uint &row_stride, thread uint &c0, thread uint &c_stride) {
    if (m_rows == 1u) {
        r0 = 0u;
        row_stride = 1u;
        c0 = tid;
        c_stride = threads;
        return;
    }
    const uint max_group = metal::min(threads, metal::max(16u, width));
    uint shift = 4u;
    if (max_group >= 32u) shift = 5u;
    if (max_group >= 64u) shift = 6u;
    if (max_group >= 128u) shift = 7u;
    if (max_group >= 256u) shift = 8u;
    if (max_group >= 512u) shift = 9u;
    c_stride = 1u << shift;
    row_stride = threads >> shift;
    r0 = tid >> shift;
    c0 = tid & (c_stride - 1u);
}

// 8x8 butterfly transpose within each 8-lane quarter (`q = lane >> 3`, `u = lane & 7`):
// lane `u` enters holding 8 contiguous values at `base + 8*u + 0..7` and exits
// holding `v[s] = val[base + 8*s + u]` (`s = 0..7`), matching the 8-lane reference order.
inline void ordered_transpose8(thread float (&v)[8], uint u) {
    PROJECTION_UNROLL
    for (uint i = 0u; i < 8u; i += 2u) {
        const float send = (u & 1u) != 0u ? v[i] : v[i + 1u];
        const float recv = simd_shuffle_xor(send, ushort(1u));
        if ((u & 1u) != 0u)
            v[i] = recv;
        else
            v[i + 1u] = recv;
    }
    PROJECTION_UNROLL
    for (uint g = 0u; g < 8u; g += 4u) {
        PROJECTION_UNROLL
        for (uint j = 0u; j < 2u; ++j) {
            const uint i = g + j;
            const float send = (u & 2u) != 0u ? v[i] : v[i + 2u];
            const float recv = simd_shuffle_xor(send, ushort(2u));
            if ((u & 2u) != 0u)
                v[i] = recv;
            else
                v[i + 2u] = recv;
        }
    }
    PROJECTION_UNROLL
    for (uint i = 0u; i < 4u; ++i) {
        const float send = (u & 4u) != 0u ? v[i] : v[i + 4u];
        const float recv = simd_shuffle_xor(send, ushort(4u));
        if ((u & 4u) != 0u)
            v[i] = recv;
        else
            v[i + 4u] = recv;
    }
}

template <typename A, typename N, typename Rows>
struct OrderedRms {
    static_assert(A::bytes == 2, "the Metal projection family requires a bf16 or f16 activation element");
    typedef A activation;
    device const float *x;
    ulong stride0, stride1;
    device const uchar *norm;
    ulong norm_stride;
    float eps;
    uint columns;
    Rows rows;
    uint width() const { return columns; }
    float row_inverse_lane(uint actual_row, bool active, uint lane) const {
        const uint u = lane & 7u;
        float sum = 0.0f;
        if (active) {
            device const float *row_ptr = x + ulong(rows.at(actual_row)) * stride0;
            if (stride1 == 1ul) {
                uint c = u;
                for (; c + 56u < columns; c += 64u) {
                    const float v0 = row_ptr[c];
                    const float v1 = row_ptr[c + 8u];
                    const float v2 = row_ptr[c + 16u];
                    const float v3 = row_ptr[c + 24u];
                    const float v4 = row_ptr[c + 32u];
                    const float v5 = row_ptr[c + 40u];
                    const float v6 = row_ptr[c + 48u];
                    const float v7 = row_ptr[c + 56u];
                    sum = metal::fma(v0, v0, sum);
                    sum = metal::fma(v1, v1, sum);
                    sum = metal::fma(v2, v2, sum);
                    sum = metal::fma(v3, v3, sum);
                    sum = metal::fma(v4, v4, sum);
                    sum = metal::fma(v5, v5, sum);
                    sum = metal::fma(v6, v6, sum);
                    sum = metal::fma(v7, v7, sum);
                }
                for (; c < columns; c += 8u) {
                    const float val = row_ptr[c];
                    sum = metal::fma(val, val, sum);
                }
            } else {
                for (uint c = u; c < columns; c += 8u) {
                    const float val = row_ptr[ulong(c) * stride1];
                    sum = metal::fma(val, val, sum);
                }
            }
        }
        const float total_sq = ordered_combine8(sum);
        return (active && u == 0u) ? (1.0f / metal::precise::sqrt(total_sq / float(columns) + eps)) : 0.0f;
    }
    void preload_row_regs(uint actual_row, uint c, thread float4 &r0, thread float4 &r1) const {
        device const float *row_ptr = x + ulong(rows.at(actual_row)) * stride0;
        r0 = float4(0.0f);
        r1 = float4(0.0f);
        if (c + 8u <= columns) {
            if ((ulong(row_ptr) & 15ul) == 0ul) {
                device const float4 *src4 = reinterpret_cast<device const float4 *>(row_ptr + c);
                r0 = src4[0];
                r1 = src4[1];
            } else {
                r0 = float4(row_ptr[c], row_ptr[c + 1u], row_ptr[c + 2u], row_ptr[c + 3u]);
                r1 = float4(row_ptr[c + 4u], row_ptr[c + 5u], row_ptr[c + 6u], row_ptr[c + 7u]);
            }
        } else if (c < columns) {
            r0 = float4(
                (c + 0u < columns) ? row_ptr[c + 0u] : 0.0f,
                (c + 1u < columns) ? row_ptr[c + 1u] : 0.0f,
                (c + 2u < columns) ? row_ptr[c + 2u] : 0.0f,
                (c + 3u < columns) ? row_ptr[c + 3u] : 0.0f);
            r1 = float4(
                (c + 4u < columns) ? row_ptr[c + 4u] : 0.0f,
                (c + 5u < columns) ? row_ptr[c + 5u] : 0.0f,
                (c + 6u < columns) ? row_ptr[c + 6u] : 0.0f,
                (c + 7u < columns) ? row_ptr[c + 7u] : 0.0f);
        }
    }
    void ordered_inverses(uint m_rows, threadgroup float *inverses, uint sg, uint lane, uint simdgroups,
        uint first_row = 0u) const {
        if (m_rows == 0u)
            return;
        const uint q_base = sg * 4u, quarters = simdgroups * 4u;
        for (uint base = q_base; base < m_rows; base += quarters) {
            const uint r = base + (lane >> 3u);
            const float inv = row_inverse_lane(first_row + r, r < m_rows, lane);
            if (r < m_rows && (lane & 7u) == 0u)
                inverses[r] = inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    void ordered_inverses_sram(uint m_rows, threadgroup float *inverses, threadgroup float *sram, uint sram_floats,
        uint sg, uint lane, uint simdgroups) const {
        if (m_rows == 0u)
            return;
        const uint quarters = simdgroups * 4u;
        if (stride1 != 1ul || columns == 0u || sram_floats < m_rows * 64u || m_rows > quarters) {
            ordered_inverses(m_rows, inverses, sg, lane, simdgroups);
            return;
        }
        const uint row_cap = metal::min((columns + 63u) & ~63u, (sram_floats / m_rows) & ~63u);
        const uint tid = sg * 32u + lane, threads = simdgroups * 32u;
        const uint q_idx = sg * 4u + (lane >> 3u), u = lane & 7u;
        float sum = 0.0f;
        for (uint k0 = 0u; k0 < columns; k0 += row_cap) {
            if (k0 > 0u)
                threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint chunk = metal::min(row_cap, columns - k0);
            if ((chunk & 3u) == 0u) {
                const uint chunk4 = chunk >> 2u;
                uint r0, row_stride, c0, c_stride;
                ordered_stage_partition(m_rows, chunk4, tid, threads, r0, row_stride, c0, c_stride);
                for (uint r = r0; r < m_rows; r += row_stride) {
                    device const float *row_base = x + ulong(rows.at(r)) * stride0 + k0;
                    threadgroup float4 *dst4 = reinterpret_cast<threadgroup float4 *>(sram + r * row_cap);
                    const bool aligned = (ulong(row_base) & 15ul) == 0ul;
                    for (uint c = c0; c < chunk4; c += c_stride) {
                        device const float *rp = row_base + c * 4u;
                        dst4[c] = aligned ? *reinterpret_cast<device const float4 *>(rp)
                                          : float4(rp[0], rp[1], rp[2], rp[3]);
                    }
                }
            } else {
                uint r0, row_stride, c0, c_stride;
                ordered_stage_partition(m_rows, chunk, tid, threads, r0, row_stride, c0, c_stride);
                for (uint r = r0; r < m_rows; r += row_stride) {
                    device const float *row_base = x + ulong(rows.at(r)) * stride0 + k0;
                    threadgroup float *dst = sram + r * row_cap;
                    for (uint c = c0; c < chunk; c += c_stride)
                        dst[c] = row_base[c];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (q_idx < m_rows)
                sum = ordered_accumulate_sram8(sram + q_idx * row_cap, chunk, u, sum);
        }
        float inv = 0.0f;
        if (q_idx < m_rows) {
            const float total_sq = ordered_combine8(sum);
            if (u == 0u)
                inv = 1.0f / metal::precise::sqrt(total_sq / float(columns) + eps);
        }
        if ((inverses < sram + m_rows * row_cap) && (sram < inverses + m_rows))
            threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q_idx < m_rows && u == 0u)
            inverses[q_idx] = inv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float ordered_at(uint m, uint c, float inv) const {
        const float v = x[ulong(rows.at(m)) * stride0 + ulong(c) * stride1];
        return A::round(v * inv * element::at<N>(norm, ulong(c) * norm_stride));
    }
    void load8(uint m, uint k, float inv, thread float4 &even, thread float4 &odd) const {
        Rms<A, N, Rows>{x, stride0, stride1, norm, norm_stride, eps, columns, rows}.load8(m, k, inv, even, odd);
    }
};

template <typename A, typename N, typename Rows, uint PARTS>
struct OrderedSummed {
    static_assert(A::bytes == 2, "the Metal projection family requires a bf16 or f16 activation element");
    typedef A activation;
    device const float *x;
    ulong stride0, stride1;
    device const uchar *norm;
    ulong norm_stride;
    float eps;
    uint columns;
    Rows rows;
    uint width() const { return columns; }
    float row_inverse_lane(uint, bool, uint) const { return 1.0f; }
    void ordered_inverses(uint m_rows, threadgroup float *inverses, uint sg, uint lane, uint simdgroups,
        uint = 0u) const {
        const uint tid = sg * 32u + lane, threads = simdgroups * 32u;
        for (uint r = tid; r < m_rows; r += threads)
            inverses[r] = 1.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    void ordered_inverses_sram(uint m_rows, threadgroup float *inverses, threadgroup float *, uint,
        uint sg, uint lane, uint simdgroups) const {
        ordered_inverses(m_rows, inverses, sg, lane, simdgroups);
    }
    float ordered_at(uint m, uint c, float) const {
        device const float *row = x + ulong(rows.at(m)) * stride0;
        float total = 0.0f;
        for (uint part = 0; part < PARTS; ++part)
            total = total + row[ulong(part * columns + c) * stride1];
        return A::round(total);
    }
    void load8(uint m, uint k, float inv, thread float4 &even, thread float4 &odd) const {
        Summed<A, N, Rows, PARTS>{x, stride0, stride1, norm, norm_stride, eps, columns, rows}.load8(m, k, inv, even, odd);
    }
};

template <typename A, typename N, typename Rows, uint NORM_EXTENT, uint PARTS>
struct OrderedInputOf {
    typedef OrderedRms<A, N, Rows> type;
};
template <typename A, typename N, typename Rows, uint PARTS>
struct OrderedInputOf<A, N, Rows, 0, PARTS> {
    typedef OrderedSummed<A, N, Rows, PARTS> type;
};

template <typename A, typename N, typename Rows>
inline void ordered_norm8_regs(thread const OrderedRms<A, N, Rows> &in, uint c, float inv,
    float4 r0, float4 r1, thread float4 &even, thread float4 &odd) {
    float4 ne, no;
    norm8<N>::load(in.norm, in.norm_stride, c, ne, no);
    const float4 e = float4(r0.x, r0.z, r1.x, r1.z) * inv * ne;
    const float4 o = float4(r0.y, r0.w, r1.y, r1.w) * inv * no;
    PROJECTION_UNROLL
    for (uint i = 0; i < 4; ++i) {
        even[i] = A::round(e[i]);
        odd[i] = A::round(o[i]);
    }
}

// Reference-order RMS inverse over preloaded register chunks (`CHUNKS * 2048` max width):
// streams `r0[ch], r1[ch]` through `shared` (`sram_floats` floats, >= 256) so simdgroup 0
// lanes 0..7 accumulate `x[u], x[u + 8], ...` in bit-exact reference order.
template <uint CHUNKS, typename A, typename N, typename Rows>
inline float ordered_preload_inverse(thread const OrderedRms<A, N, Rows> &in, uint k,
    thread const float4 (&r0)[CHUNKS], thread const float4 (&r1)[CHUNKS],
    threadgroup uchar *shared, uint sram_floats, uint sg, uint lane) {
    const uint tid = sg * 32u + lane;
    const uint cap_threads = sram_floats >> 3u;
    const uint cap_floats = cap_threads * 8u;
    threadgroup float4 *sram4 = reinterpret_cast<threadgroup float4 *>(shared);
    threadgroup const float *sp = reinterpret_cast<threadgroup const float *>(shared);
    float sum = 0.0f;
    for (uint base_slot = 0u; base_slot * 8u < k; base_slot += cap_threads) {
        if (base_slot > 0u)
            threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 256u) {
            PROJECTION_UNROLL
            for (uint ch = 0u; ch < CHUNKS; ++ch) {
                const uint slot = ch * 256u + tid;
                if (slot >= base_slot && slot < base_slot + cap_threads) {
                    const uint local_slot = slot - base_slot;
                    sram4[2u * local_slot] = r0[ch];
                    sram4[2u * local_slot + 1u] = r1[ch];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u && lane < 8u) {
            const uint lim = metal::min(k - base_slot * 8u, cap_floats);
            sum = ordered_accumulate_sram8(sp, lim, lane, sum);
        }
    }
    if (sg == 0u) {
        const float total_sq = ordered_combine8(sum);
        if (lane == 0u)
            reinterpret_cast<threadgroup float *>(shared)[0] =
                1.0f / metal::precise::sqrt(total_sq / float(k) + in.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = reinterpret_cast<threadgroup const float *>(shared)[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return inv;
}

// Fused single-read prologue for `gemv_prestaged_one` (`m == 1, k <= 8192`):
// preloads `x` into registers while simdgroup 0 reduces the reference-order RMS
// inverse, then writes normalized `uint4` words in `gemv_word` layout at `shared + 0..k*2`.
template <uint SIMDGROUPS, typename A, typename N, typename Rows>
inline bool ordered_stage_single_gemv_words(thread const OrderedRms<A, N, Rows> &in, uint k,
    threadgroup uchar *shared, uint sg, uint lane) {
    if constexpr (SIMDGROUPS < 8u)
        return false;
    if (k != in.columns || k == 0u || k > 8192u || (k & 31u) != 0u || in.stride1 != 1ul)
        return false;
    const uint tid = sg * 32u + lane;
    float4 r0[4], r1[4];
    PROJECTION_UNROLL
    for (uint ch = 0u; ch < 4u; ++ch) {
        r0[ch] = float4(0.0f);
        r1[ch] = float4(0.0f);
        const uint c = (ch * 256u + tid) * 8u;
        if (tid < 256u && c < k)
            in.preload_row_regs(0u, c, r0[ch], r1[ch]);
    }
    const uint sram_floats = metal::max(256u, ((k + 31u) >> 5u) * 16u);
    const float inv = ordered_preload_inverse<4u>(in, k, r0, r1, shared, sram_floats, sg, lane);
    if (tid < 256u) {
        const uint count = k >> 5u;
        threadgroup uint4 *words = reinterpret_cast<threadgroup uint4 *>(shared);
        PROJECTION_UNROLL
        for (uint ch = 0u; ch < 4u; ++ch) {
            const uint slot = ch * 256u + tid;
            const uint c = slot * 8u;
            if (c < k) {
                float4 even, odd;
                ordered_norm8_regs(in, c, inv, r0[ch], r1[ch], even, odd);
                words[(slot & 3u) * count + (slot >> 2u)] = A::pack8(even, odd);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return true;
}

template <uint SIMDGROUPS, typename In>
inline bool ordered_stage_single_gemv_words(thread const In &, uint, threadgroup uchar *, uint, uint) {
    return false;
}

// Fused single-read prologue for `ordered_gemv` (`m == 1, k <= 4096`, capped by
// `ORDERED_GEMV_SHARED = 8192` bytes = 4096 packed bf16/f16 elements):
// stages `x` into `shared` in lane-strided `uint4` packets (`[block * 8 + u]`).
template <bool PUBLISH, uint SIMDGROUPS, typename A, typename N, typename Rows, typename PubOut>
inline bool ordered_stage_single_gemv_raw(thread const OrderedRms<A, N, Rows> &in, uint k,
    threadgroup uchar *shared, PubOut pub_out, ulong pub_stride0, ulong pub_stride1, bool publish_rows,
    thread float &inv_out, uint sg, uint lane) {
    if constexpr (SIMDGROUPS < 8u)
        return false;
    if (k != in.columns || k == 0u || k > 4096u || (k > 512u && (k & 63u) != 0u) || in.stride1 != 1ul)
        return false;
    const uint tid = sg * 32u + lane;
    float4 r0[2], r1[2];
    PROJECTION_UNROLL
    for (uint ch = 0u; ch < 2u; ++ch) {
        r0[ch] = float4(0.0f);
        r1[ch] = float4(0.0f);
        const uint c = (ch * 256u + tid) * 8u;
        if (tid < 256u && c < k)
            in.preload_row_regs(0u, c, r0[ch], r1[ch]);
    }
    const uint sram_floats = metal::max(256u, ((k + 31u) >> 5u) * 16u);
    const float inv = ordered_preload_inverse<2u>(in, k, r0, r1, shared, sram_floats, sg, lane);
    inv_out = inv;
    const uint active_tids = ((k + 63u) & ~63u) >> 3u;
    if (tid < 256u) {
        PROJECTION_UNROLL
        for (uint ch = 0u; ch < 2u; ++ch) {
            const uint slot = ch * 256u + tid;
            if (slot < active_tids) {
                const uint c = slot * 8u;
                float4 even = float4(0.0f), odd = float4(0.0f);
                if (c + 8u <= k) {
                    ordered_norm8_regs(in, c, inv, r0[ch], r1[ch], even, odd);
                } else if (c < k) {
                    const float raw_v[8] = {
                        r0[ch].x, r0[ch].y, r0[ch].z, r0[ch].w, r1[ch].x, r1[ch].y, r1[ch].z, r1[ch].w};
                    float out_v[8];
                    PROJECTION_UNROLL
                    for (uint j = 0; j < 8; ++j) {
                        out_v[j] = (c + j < k)
                            ? A::round(raw_v[j] * inv * element::at<N>(in.norm, ulong(c + j) * in.norm_stride))
                            : 0.0f;
                    }
                    even = float4(out_v[0], out_v[2], out_v[4], out_v[6]);
                    odd = float4(out_v[1], out_v[3], out_v[5], out_v[7]);
                }
                if constexpr (PUBLISH) {
                    if (publish_rows && c < k) {
                        if (pub_stride1 == 1ul && (pub_stride0 & 7ul) == 0ul && c + 8u <= k) {
                            pub_out.put8(ulong(c), even, odd);
                        } else {
                            for (uint j = 0; j < 8; ++j) {
                                if (c + j < k) {
                                    const float val = (j & 1u) != 0u ? odd[j >> 1u] : even[j >> 1u];
                                    pub_out.put(ulong(c + j) * pub_stride1, val);
                                }
                            }
                        }
                    }
                }
                float xv[8] = {even.x, odd.x, even.y, odd.y, even.z, odd.z, even.w, odd.w};
                ordered_transpose8(xv, lane & 7u);
                reinterpret_cast<threadgroup uint4 *>(shared)[slot] = A::pack8(
                    float4(xv[0], xv[2], xv[4], xv[6]),
                    float4(xv[1], xv[3], xv[5], xv[7]));
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return true;
}

template <bool PUBLISH, uint SIMDGROUPS, typename In, typename PubOut>
inline bool ordered_stage_single_gemv_raw(thread const In &, uint, threadgroup uchar *, PubOut, ulong, ulong, bool,
    thread float &, uint, uint) {
    return false;
}

template <typename A, typename N, typename Rows, typename Out>
inline float ordered_stage_row(thread const OrderedRms<A, N, Rows> &in, Out out, ulong out_stride, ulong col_stride,
    uint row, uint m_rows, uint k, threadgroup float *scratch, uint sg, uint lane, uint scratch_floats = 8u) {
    if (row >= m_rows)
        return 0.0f;
    const uint tid = sg * 32u + lane;
    if (k == in.columns && k > 0u && k <= 8192u && in.stride1 == 1ul) {
        float4 r0[4], r1[4];
        PROJECTION_UNROLL
        for (uint ch = 0u; ch < 4u; ++ch) {
            r0[ch] = float4(0.0f);
            r1[ch] = float4(0.0f);
            const uint c = (ch * 256u + tid) * 8u;
            if (c < k)
                in.preload_row_regs(row, c, r0[ch], r1[ch]);
        }
        float inv;
        if (scratch_floats >= 256u) {
            inv = ordered_preload_inverse<4u>(in, k, r0, r1, reinterpret_cast<threadgroup uchar *>(scratch),
                scratch_floats, sg, lane);
        } else {
            if (sg == 0u) {
                const float inv0 = in.row_inverse_lane(row, lane < 8u, lane);
                if (lane == 0u)
                    scratch[0] = inv0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            inv = scratch[0];
        }
        const bool fast_put = (col_stride == 1ul && (out_stride & 7ul) == 0ul);
        PROJECTION_UNROLL
        for (uint ch = 0u; ch < 4u; ++ch) {
            const uint c = (ch * 256u + tid) * 8u;
            if (c < k) {
                if (fast_put && c + 8u <= k) {
                    float4 even, odd;
                    ordered_norm8_regs(in, c, inv, r0[ch], r1[ch], even, odd);
                    out.put8(ulong(row) * out_stride + ulong(c), even, odd);
                } else {
                    const float raw_v[8] = {
                        r0[ch].x, r0[ch].y, r0[ch].z, r0[ch].w, r1[ch].x, r1[ch].y, r1[ch].z, r1[ch].w};
                    for (uint j = 0; j < 8u && c + j < k; ++j) {
                        const float val = A::round(
                            raw_v[j] * inv * element::at<N>(in.norm, ulong(c + j) * in.norm_stride));
                        out.put(ulong(row) * out_stride + ulong(c + j) * col_stride, val);
                    }
                }
            }
        }
        return inv;
    }
    in.ordered_inverses(1u, scratch, sg, lane, 8u, row);
    const float inv = scratch[0];
    if (col_stride == 1ul && (out_stride & 7ul) == 0ul) {
        const uint k8 = k & ~7u;
        for (uint c = tid * 8u; c < k8; c += 2048u) {
            float4 even, odd;
            in.load8(row, c, inv, even, odd);
            out.put8(ulong(row) * out_stride + ulong(c), even, odd);
        }
        for (uint c = k8 + tid; c < k; c += 256u)
            out.put(ulong(row) * out_stride + ulong(c), in.ordered_at(row, c, inv));
    } else {
        for (uint c = tid; c < k; c += 256u)
            out.put(ulong(row) * out_stride + ulong(c) * col_stride, in.ordered_at(row, c, inv));
    }
    return inv;
}

template <typename A, typename N, typename Rows, uint PARTS, typename Out>
inline float ordered_stage_row(thread const OrderedSummed<A, N, Rows, PARTS> &in, Out out, ulong out_stride,
    ulong col_stride, uint row, uint m_rows, uint k, threadgroup float *, uint sg, uint lane, uint = 8u) {
    if (row >= m_rows)
        return 0.0f;
    const uint tid = sg * 32u + lane;
    if (col_stride == 1ul && (out_stride & 7ul) == 0ul) {
        const uint k8 = k & ~7u;
        for (uint c = tid * 8u; c < k8; c += 2048u) {
            float4 even, odd;
            in.load8(row, c, 1.0f, even, odd);
            out.put8(ulong(row) * out_stride + ulong(c), even, odd);
        }
        for (uint c = k8 + tid; c < k; c += 256u)
            out.put(ulong(row) * out_stride + ulong(c), in.ordered_at(row, c, 1.0f));
    } else {
        for (uint c = tid; c < k; c += 256u)
            out.put(ulong(row) * out_stride + ulong(c) * col_stride, in.ordered_at(row, c, 1.0f));
    }
    return 1.0f;
}
