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

// ---------------------------------------------------------------------------
// NVFP4 operands (docs/precision.md, "NVFP4 operands"; the portable
// `nvfp4_row`): a product over NVFP4 weights takes each row x as q * g * s,
// g the row's global scale (the least power of two with 2688 g >= amax|x|),
// s a 16-value block's UE4M3 scale (nearest amax|block| / (6 g)) and q the
// E2M1 code nearest x / (g s), saturating at 6, every rounding to nearest
// even. The portable body divides with F32's correctly rounded quotient,
// which rounds onto each grid as the exact quotient does; Metal's division
// is not correctly rounded, so every rounding here is decided without one:
// each grid point or midpoint times its divisor is exact in F32, and the
// dividend against it decides the exact quotient. Rows are finite.

namespace nvfp4 {

// 2^(e - 127) for a biased exponent e in 1 .. 254.
inline float power(uint e) { return as_type<float>(e << 23); }

// The global scale of a row whose largest magnitude is amax: the binade of
// an approximate amax / 2688, corrected by one exact comparison. A finite
// row's g is at most 2^117, so its inverse is normal.
inline float global_scale(float amax) {
    const uint bits = as_type<uint>(amax * (1.0f / 2688.0f));
    const uint near = metal::clamp((bits >> 23) + ((bits & 0x7fffffu) != 0u ? 1u : 0u), 1u, 254u);
    if (near > 1u && 2688.0f * power(near - 1u) >= amax)
        return power(near - 1u);
    if (near < 254u && 2688.0f * power(near) < amax)
        return power(near + 1u);
    return power(near);
}

// g s for a block whose largest magnitude is block_max: s is the UE4M3
// value (exponent bias 7, three mantissa bits, subnormal quantum 2^-9)
// nearest w / 6 for w = block_max / g (exact: g is a power of two). The
// quantum is that of an approximate w / 6's binade (both binades' grids
// hold the boundary, the nearest value to any quotient that rounds across
// it); t = w / quantum is exact, and the approximate index t / 6 is
// corrected against its exact midpoints 6 index +- 3. A finite row's w is
// at most 2688, so s is at most 448.
inline float scale(float block_max, float g) {
    const float w = block_max * power(254u - (as_type<uint>(g) >> 23));
    const uint exponent = metal::max((as_type<uint>(w * (1.0f / 6.0f)) >> 23) & 255u, 121u);
    const float t = w * power(257u - exponent);
    const int near = int(metal::rint(t * (1.0f / 6.0f)));
    const bool odd = (near & 1) != 0;
    const float above = float(6 * near + 3), below = float(6 * near - 3);
    const int index = (t > above || (t == above && odd)) ? near + 1
        : (t < below || (t == below && odd))             ? near - 1
                                                         : near;
    return g * (float(index) * power(exponent - 3u));
}

// x's E2M1 code nearest x / d, saturating at 6, times d: each threshold
// times d is exact, so |x| against it decides the exact quotient (ties to
// the even code). A zero d gives (positive) zero.
inline float value(float x, float d) {
    if (d == 0.0f)
        return 0.0f;
    const float a = metal::abs(x);
    const float m = a <= 0.25f * d ? 0.0f
        : a < 0.75f * d            ? 0.5f
        : a <= 1.25f * d           ? 1.0f
        : a < 1.75f * d            ? 1.5f
        : a <= 2.5f * d            ? 2.0f
        : a < 3.5f * d             ? 3.0f
        : a <= 5.0f * d            ? 4.0f
                                   : 6.0f;
    return (x < 0.0f ? -m : m) * d;
}

inline float4 values(float4 x, float d) {
    return float4(value(x.x, d), value(x.y, d), value(x.z, d), value(x.w, d));
}

inline float magnitude(float4 even, float4 odd) {
    const float4 m = metal::max(metal::abs(even), metal::abs(odd));
    return metal::max(metal::max(m.x, m.y), metal::max(m.z, m.w));
}

} // namespace nvfp4

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
        const float d = nvfp4::scale(metal::max(nvfp4::magnitude(e0, o0), nvfp4::magnitude(e1, o1)), globals.at(m));
        const bool upper = (k & 8u) != 0u;
        even = nvfp4::values(upper ? e1 : e0, d);
        odd = nvfp4::values(upper ? o1 : o0, d);
    }
    // `load8`'s values as A storage words (exact: q g s fits A).
    uint4 words8(uint m, uint k) const {
        float4 even, odd;
        load8(m, k, 0.0f, even, odd);
        return activation::pack8(even, odd);
    }
};
