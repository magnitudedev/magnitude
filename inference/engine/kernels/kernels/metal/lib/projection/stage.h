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
inline uint4 words8_storage(device const typename A::storage *row, ulong stride, uint k, uint columns,
    bool vector) {
    if (vector && k + 8u <= columns)
        return *reinterpret_cast<device const uint4 *>(row + k);
    uint4 words = uint4(0);
    for (uint i = 0; i < 8; ++i)
        if (k + i < columns)
            words[i / 2] |= uint(as_type<ushort>(row[ulong(k + i) * stride])) << (16u * (i & 1u));
    return words;
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
    typedef A activation;
    device const uchar *x;
    ulong stride0, stride1;
    uint columns;
    Rows rows;
    float value(uint m, uint k, float) const {
        return A::load(reinterpret_cast<device const typename A::storage *>(x)
            [ulong(rows.at(m)) * stride0 + ulong(k) * stride1]);
    }
    void load8(uint m, uint k, float, thread float4 &even, thread float4 &odd) const {
        device const typename A::storage *row =
            reinterpret_cast<device const typename A::storage *>(x) + ulong(rows.at(m)) * stride0;
        load8_storage<A>(row, stride1, k, columns, stride1 == 1 && (stride0 & 7u) == 0, even, odd);
    }
    uint4 words8(uint m, uint k) const {
        device const typename A::storage *row =
            reinterpret_cast<device const typename A::storage *>(x) + ulong(rows.at(m)) * stride0;
        return words8_storage<A>(row, stride1, k, columns, stride1 == 1 && (stride0 & 7u) == 0);
    }
};

template <typename A, typename N, typename Rows>
struct Rms {
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

// Gated per-head RMS·SiLU(z): `mixed` is [rows, heads, W] in A, `z` sits at
// column `z_column` of a row of `projection` (A).
template <typename A, typename N, typename Rows>
struct GatedRms {
    typedef A activation;
    device const uchar *mixed;
    ulong mixed0, mixed1, mixed2;
    device const uchar *projection;
    ulong projection0, projection1;
    ulong z_column;
    device const uchar *norm;
    ulong norm_stride;
    float eps;
    uint heads;
    uint head_width;
    Rows rows;
    uint groups() const { return heads; }
    uint width() const { return head_width; }
    float mixed_at(uint m, uint head, uint i) const {
        return A::load(reinterpret_cast<device const typename A::storage *>(mixed)
            [ulong(rows.at(m)) * mixed0 + ulong(head) * mixed1 + ulong(i) * mixed2]);
    }
    float norm_input(uint m, uint head, uint i) const { return mixed_at(m, head, i); }
    float epsilon() const { return eps; }
    float value(uint m, uint column, float inverse) const {
        uint head = column / head_width, i = column % head_width;
        float normalized = A::round(mixed_at(m, head, i) * inverse
            * element::at<N>(norm, ulong(i) * norm_stride));
        float gate = A::load(reinterpret_cast<device const typename A::storage *>(projection)
            [ulong(rows.at(m)) * projection0 + (z_column + column) * projection1]);
        float activated = A::round(gate / (1.0f + metal::exp(-gate)));
        return A::round(normalized * activated);
    }
    // The inputs of columns k..k+7 (one head; head_width is a multiple of 8)
    // as (even, odd): mixed, z and the norm weights.
    struct inputs8 {
        float4 me, mo, ze, zo, ne, no;
    };
    inputs8 load_inputs8(uint m, uint k) const {
        uint head = k / head_width, i = k % head_width;
        ulong mixed_at0 = ulong(rows.at(m)) * mixed0 + ulong(head) * mixed1 + ulong(i) * mixed2;
        ulong z_at = ulong(rows.at(m)) * projection0 + (z_column + k) * projection1;
        device const typename A::storage *mixed_row = reinterpret_cast<device const typename A::storage *>(mixed);
        device const typename A::storage *z_row = reinterpret_cast<device const typename A::storage *>(projection);
        inputs8 v;
        load8_storage<A>(mixed_row + mixed_at0, mixed2, 0, 8, mixed2 == 1 && (mixed_at0 & 7u) == 0, v.me, v.mo);
        load8_storage<A>(z_row + z_at, projection1, 0, 8, projection1 == 1 && (z_at & 7u) == 0, v.ze, v.zo);
        norm8<N>::load(norm, norm_stride, i, v.ne, v.no);
        return v;
    }
    void finish8(thread const inputs8 &v, float inverse, thread float4 &even, thread float4 &odd) const {
        float4 e = v.me * inverse * v.ne, o = v.mo * inverse * v.no;
        float4 ae = v.ze / (1.0f + metal::exp(-v.ze)), ao = v.zo / (1.0f + metal::exp(-v.zo));
        for (uint j = 0; j < 4; ++j) {
            even[j] = A::round(A::round(e[j]) * A::round(ae[j]));
            odd[j] = A::round(A::round(o[j]) * A::round(ao[j]));
        }
    }
    void load8(uint m, uint k, float inverse, thread float4 &even, thread float4 &odd) const {
        finish8(load_inputs8(m, k), inverse, even, odd);
    }
    // The prologue of columns k..k+7 when the head_width / 8 adjacent lanes of
    // this lane's aligned lane group load the head's columns in order (the
    // GEMV staging's layout): each lane sums the squares of its eight mixed
    // values in column order, and a butterfly over the group gives every lane
    // the head's square sum. No separate square-sum pass is needed, and the
    // sum does not depend on the threadgroup shape.
    void load8_across_lanes(uint m, uint k, thread float4 &even, thread float4 &odd) const {
        inputs8 v = load_inputs8(m, k);
        float squares = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            squares = metal::fma(v.me[j], v.me[j], squares);
            squares = metal::fma(v.mo[j], v.mo[j], squares);
        }
        for (ushort offset = 1; offset < head_width / 8u; offset <<= 1)
            squares += simd_shuffle_xor(squares, offset);
        finish8(v, metal::rsqrt(squares / float(head_width) + eps), even, odd);
    }
};

