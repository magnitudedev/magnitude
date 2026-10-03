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

