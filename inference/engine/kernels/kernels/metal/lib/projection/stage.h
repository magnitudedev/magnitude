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

// ---------------------------------------------------------------------------
// Non-finite activations (`seismic-native-cpu` `quant.rs`): a NaN or ±Inf in
// a group makes its scale NaN and its codes 0, so its products are NaN, as
// the dense product's would be. A group's peak is the maximum of its
// magnitudes' bits, whose integer order is the magnitudes' with every NaN
// above +Inf, so a NaN wins (`metal::max` of floats drops it); its scale
// maps a +Inf peak to NaN once per group.

// The peak of `held` (a peak's bits) and eight values' magnitudes.
inline uint magnitude_peak(uint held, float4 even, float4 odd) {
    const uint4 m = metal::max(as_type<uint4>(even) & 0x7fffffffu, as_type<uint4>(odd) & 0x7fffffffu);
    return metal::max(held, metal::max(metal::max(m.x, m.y), metal::max(m.z, m.w)));
}

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
// dividend against it decides the exact quotient. The row is g's group: a
// NaN or ±Inf in the row makes g NaN (its peak `magnitude_peak`'s), so
// every block's g s and every operand of the row is NaN.

namespace nvfp4 {

// 2^(e - 127) for a biased exponent e in 1 .. 254.
inline float power(uint e) { return as_type<float>(e << 23); }

// The global scale of a row whose largest magnitude is amax: the binade of
// an approximate amax / 2688, corrected by one exact comparison. A finite
// row's g is at most 2^117, so its inverse is normal; a NaN or +Inf amax
// gives NaN.
inline float global_scale(float amax) {
    if (!(amax < INFINITY))
        return NAN;
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
// at most 2688, so s is at most 448. A NaN g gives NaN.
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
// the even code). A zero d gives (positive) zero, a NaN d NaN.
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
                for (uint idx = tid; idx < m_rows * chunk4; idx += threads) {
                    const uint r = idx / chunk4, c4 = (idx - r * chunk4) * 4u;
                    device const float *rp = x + ulong(rows.at(r)) * stride0 + k0 + c4;
                    const float4 v = ((ulong(rp) & 15ul) == 0ul)
                        ? *reinterpret_cast<device const float4 *>(rp)
                        : float4(rp[0], rp[1], rp[2], rp[3]);
                    *reinterpret_cast<threadgroup float4 *>(sram + r * row_cap + c4) = v;
                }
            } else {
                for (uint idx = tid; idx < m_rows * chunk; idx += threads) {
                    const uint r = idx / chunk, c = idx - r * chunk;
                    sram[r * row_cap + c] = x[ulong(rows.at(r)) * stride0 + k0 + c];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (q_idx < m_rows) {
                threadgroup const float *sp = sram + q_idx * row_cap;
                uint c = u;
                for (; c + 56u < chunk; c += 64u) {
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
                for (; c < chunk; c += 8u) {
                    const float val = sp[c];
                    sum = metal::fma(val, val, sum);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q_idx < m_rows) {
            const float total_sq = ordered_combine8(sum);
            if (u == 0u)
                inverses[q_idx] = 1.0f / metal::precise::sqrt(total_sq / float(columns) + eps);
        }
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

// Fused single-read prologue for `gemv_prestaged_one`: preloads `x` into
// registers while simdgroup 0 reduces the reference-order RMS inverse, then
// writes normalized `uint4` words in `gemv_word` layout at `shared + 0..k*2`.
template <typename A, typename N, typename Rows>
inline float ordered_preload_inverse(thread const OrderedRms<A, N, Rows> &in, uint k,
    float4 r0, float4 r1, threadgroup uchar *shared, uint sg, uint lane) {
    const uint tid = sg * 32u + lane;
    const uint cap_threads = metal::max(32u, ((k + 31u) >> 5u) * 2u);
    const uint cap_floats = cap_threads * 8u;
    threadgroup float4 *sram4 = reinterpret_cast<threadgroup float4 *>(shared);
    threadgroup const float *sp = reinterpret_cast<threadgroup const float *>(shared);
    if (tid < cap_threads) {
        sram4[2u * tid] = r0;
        sram4[2u * tid + 1u] = r1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    if (sg == 0u && lane < 8u) {
        const uint lim0 = metal::min(k, cap_floats);
        uint c0 = lane;
        for (; c0 + 56u < lim0; c0 += 64u) {
            const float v0 = sp[c0], v1 = sp[c0 + 8u], v2 = sp[c0 + 16u], v3 = sp[c0 + 24u];
            const float v4 = sp[c0 + 32u], v5 = sp[c0 + 40u], v6 = sp[c0 + 48u], v7 = sp[c0 + 56u];
            sum = metal::fma(v0, v0, sum);
            sum = metal::fma(v1, v1, sum);
            sum = metal::fma(v2, v2, sum);
            sum = metal::fma(v3, v3, sum);
            sum = metal::fma(v4, v4, sum);
            sum = metal::fma(v5, v5, sum);
            sum = metal::fma(v6, v6, sum);
            sum = metal::fma(v7, v7, sum);
        }
        for (; c0 < lim0; c0 += 8u) {
            const float val = sp[c0];
            sum = metal::fma(val, val, sum);
        }
    }
    if (k > cap_floats) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= cap_threads && tid < 2u * cap_threads) {
            const uint slot = tid - cap_threads;
            sram4[2u * slot] = r0;
            sram4[2u * slot + 1u] = r1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u && lane < 8u) {
            const uint lim1 = k - cap_floats;
            uint c0 = lane;
            for (; c0 + 56u < lim1; c0 += 64u) {
                const float v0 = sp[c0], v1 = sp[c0 + 8u], v2 = sp[c0 + 16u], v3 = sp[c0 + 24u];
                const float v4 = sp[c0 + 32u], v5 = sp[c0 + 40u], v6 = sp[c0 + 48u], v7 = sp[c0 + 56u];
                sum = metal::fma(v0, v0, sum);
                sum = metal::fma(v1, v1, sum);
                sum = metal::fma(v2, v2, sum);
                sum = metal::fma(v3, v3, sum);
                sum = metal::fma(v4, v4, sum);
                sum = metal::fma(v5, v5, sum);
                sum = metal::fma(v6, v6, sum);
                sum = metal::fma(v7, v7, sum);
            }
            for (; c0 < lim1; c0 += 8u) {
                const float val = sp[c0];
                sum = metal::fma(val, val, sum);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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

template <uint SIMDGROUPS, typename A, typename N, typename Rows>
inline bool ordered_stage_single_gemv_words(thread const OrderedRms<A, N, Rows> &in, uint k,
    threadgroup uchar *shared, uint sg, uint lane) {
    if constexpr (SIMDGROUPS < 8u)
        return false;
    if (k != in.columns || k == 0u || k > 2048u || (k & 31u) != 0u || in.stride1 != 1ul)
        return false;
    const uint tid = sg * 32u + lane;
    const uint c = tid * 8u;
    float4 r0 = float4(0.0f), r1 = float4(0.0f);
    if (tid < 256u)
        in.preload_row_regs(0u, c, r0, r1);
    const float inv = ordered_preload_inverse(in, k, r0, r1, shared, sg, lane);
    if (tid < 256u && c < k) {
        float4 ne, no;
        norm8<N>::load(in.norm, in.norm_stride, c, ne, no);
        const float4 e = float4(r0.x, r0.z, r1.x, r1.z) * inv * ne;
        const float4 o = float4(r0.y, r0.w, r1.y, r1.w) * inv * no;
        float4 even, odd;
        PROJECTION_UNROLL
        for (uint i = 0; i < 4; ++i) {
            even[i] = A::round(e[i]);
            odd[i] = A::round(o[i]);
        }
        const uint count = k >> 5u;
        threadgroup uint4 *words = reinterpret_cast<threadgroup uint4 *>(shared);
        words[(tid & 3u) * count + (tid >> 2u)] = A::pack8(even, odd);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return true;
}

template <uint SIMDGROUPS, typename In>
inline bool ordered_stage_single_gemv_words(thread const In &, uint, threadgroup uchar *, uint, uint) {
    return false;
}

// Fused single-read prologue for `ordered_gemv` (`m == 1, k <= 2048`):
// stages `x` into `shared` in lane-strided `uint4` packets (`[block * 8 + u]`).
template <bool PUBLISH, uint SIMDGROUPS, typename A, typename N, typename Rows, typename PubOut>
inline bool ordered_stage_single_gemv_raw(thread const OrderedRms<A, N, Rows> &in, uint k,
    threadgroup uchar *shared, PubOut pub_out, ulong pub_stride0, ulong pub_stride1, bool publish_rows,
    thread float &inv_out, uint sg, uint lane) {
    if constexpr (SIMDGROUPS < 8u)
        return false;
    if (k != in.columns || k == 0u || k > 2048u || (k > 512u && (k & 63u) != 0u) || in.stride1 != 1ul)
        return false;
    const uint tid = sg * 32u + lane;
    const uint c = tid * 8u;
    float4 r0 = float4(0.0f), r1 = float4(0.0f);
    if (tid < 256u)
        in.preload_row_regs(0u, c, r0, r1);
    const float inv = ordered_preload_inverse(in, k, r0, r1, shared, sg, lane);
    inv_out = inv;
    const uint active_tids = ((k + 63u) & ~63u) >> 3u;
    if (tid < active_tids) {
        float4 even = float4(0.0f), odd = float4(0.0f);
        if (c + 8u <= k) {
            float4 ne, no;
            norm8<N>::load(in.norm, in.norm_stride, c, ne, no);
            const float4 e = float4(r0.x, r0.z, r1.x, r1.z) * inv * ne;
            const float4 o = float4(r0.y, r0.w, r1.y, r1.w) * inv * no;
            PROJECTION_UNROLL
            for (uint i = 0; i < 4; ++i) {
                even[i] = A::round(e[i]);
                odd[i] = A::round(o[i]);
            }
        } else if (c < k) {
            const float raw_v[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
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
        reinterpret_cast<threadgroup uint4 *>(shared)[tid] = A::pack8(
            float4(xv[0], xv[2], xv[4], xv[6]),
            float4(xv[1], xv[3], xv[5], xv[7]));
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
    uint row, uint m_rows, uint k, threadgroup float *scratch, uint sg, uint lane) {
    if (row >= m_rows)
        return 0.0f;
    const uint tid = sg * 32u + lane;
    if (k == in.columns && k > 0u && k <= 2048u && in.stride1 == 1ul) {
        const uint c = tid * 8u;
        float4 r0 = float4(0.0f), r1 = float4(0.0f);
        in.preload_row_regs(row, c, r0, r1);
        if (sg == 0u) {
            const float inv0 = in.row_inverse_lane(row, lane < 8u, lane);
            if (lane == 0u)
                scratch[0] = inv0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = scratch[0];
        if (c < k) {
            if (col_stride == 1ul && (out_stride & 7ul) == 0ul && c + 8u <= k) {
                float4 ne, no;
                norm8<N>::load(in.norm, in.norm_stride, c, ne, no);
                const float4 e = float4(r0.x, r0.z, r1.x, r1.z) * inv * ne;
                const float4 o = float4(r0.y, r0.w, r1.y, r1.w) * inv * no;
                float4 even, odd;
                PROJECTION_UNROLL
                for (uint i = 0; i < 4; ++i) {
                    even[i] = A::round(e[i]);
                    odd[i] = A::round(o[i]);
                }
                out.put8(ulong(row) * out_stride + ulong(c), even, odd);
            } else {
                const float raw_v[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
                for (uint j = 0; j < 8u && c + j < k; ++j) {
                    const float val = A::round(
                        raw_v[j] * inv * element::at<N>(in.norm, ulong(c + j) * in.norm_stride));
                    out.put(ulong(row) * out_stride + ulong(c + j) * col_stride, val);
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
    ulong col_stride, uint row, uint m_rows, uint k, threadgroup float *, uint sg, uint lane) {
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
