// The output joins of the projection family: an output entry's epilogue onto
// the F32 residual (`Residual`) or the hyper-connection streams (`Scatter`,
// with the comb), the routed sublayer's join, and the row maps they read the
// residual through. Apart from the projection family so an entry with no
// activation element (`stream_join`) joins through the same epilogues.
#pragma once

#include <seismic/element.cuh>

namespace projection {

using element::u32;
using element::u64;

// ---------------------------------------------------------------------------
// Row maps: which input row feeds projected row m.

struct AllRows {
    __device__ __forceinline__ u64 operator()(u32 m) const { return m; }
};
struct SelectedRows {
    const int *rows;
    __device__ __forceinline__ u64 operator()(u32 m) const { return (u64)rows[m]; }
};

// ---------------------------------------------------------------------------
// Joins: called once per (m, n) with the F32 projection, as epilogues.


// out[m, n] = residual[rows(m), n] + round_A(projection), A the element the
// projection is rounded to (F32: unrounded).
template <class A, class Rows> struct Residual {
    const float *residual;
    u64 residual_stride;
    Rows rows;
    element::Out<element::F32> out;
    u64 stride;
    __device__ __forceinline__ void operator()(u32 m, u64 n, float value, float) const {
        out.put(m * stride + n, residual[rows(m) * residual_stride + n] + A::round(value));
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float first, float second, float, float) const {
        const u64 source = rows(m) * residual_stride + n;
        const u64 target = m * stride + n;
        if (source % 2 != 0 || target % 2 != 0) {
            (*this)(m, n, first, 0.0f);
            (*this)(m, n + 1, second, 0.0f);
            return;
        }
        const float2 base = reinterpret_cast<const float2 *>(residual)[source / 2];
        out.put2(target, base.x + A::round(first), base.y + A::round(second));
    }
};

// `scatter`: each of the STREAMS hyper-connection streams of the residual
// plus the projection times the stream's inject value,
//   out[m, s * width + n] = residual[rows(m), s * width + n] + round_A(projection) * inject[m, s],
// or (COMBED) the streams mixed by the comb that follows the inject values,
//   out[m, d * width + n] = sum_s comb[m, s, d] residual[rows(m), s * width + n] + round_A(projection) * inject[m, d].
template <class A, class Rows, u32 STREAMS, bool COMBED> struct Scatter {
    const float *residual;
    u64 residual_stride;
    Rows rows;
    element::Out<element::F32> out;
    u64 stride;
    const float *inject;
    u64 inject_stride;
    u64 width;
    __device__ __forceinline__ void operator()(u32 m, u64 n, float value, float) const {
        const float y = A::round(value);
        const float *from = residual + rows(m) * residual_stride + n;
        const float *own = inject + m * inject_stride;
        const element::Out<element::F32> to = out.offset(m * stride + n);
        if constexpr (!COMBED) {
#pragma unroll
            for (u32 s = 0; s < STREAMS; ++s)
                to.put(s * width, from[s * width] + y * own[s]);
        } else {
            float streams[STREAMS];
#pragma unroll
            for (u32 s = 0; s < STREAMS; ++s)
                streams[s] = from[s * width];
            const float *comb = own + STREAMS;
#pragma unroll
            for (u32 d = 0; d < STREAMS; ++d) {
                float mixed = 0.0f;
#pragma unroll
                for (u32 s = 0; s < STREAMS; ++s)
                    mixed = mixed + comb[s * STREAMS + d] * streams[s];
                to.put(d * width, mixed + y * own[d]);
            }
        }
    }
    __device__ __forceinline__ void pair(u32 m, u64 n, float first, float second, float, float) const {
        (*this)(m, n, first, 0.0f);
        (*this)(m, n + 1, second, 0.0f);
    }
};

// A routed sublayer's column n joining its row's residual (`routed_output`,
// `routed_combine`): residual + selected + shared, in that order, or (INJECTED)
// for each of the STREAMS streams residual[s] + (selected + shared) * inject[s].
// `out` and `residual` point at the row, `inject` at its inject values.
template <u32 STREAMS, bool INJECTED>
__device__ __forceinline__ void routed_join(element::Out<element::F32> out, const float *residual, const float *inject, u64 width, u64 n,
                                            float selected, float shared) {
    if constexpr (!INJECTED) {
        out.put(n, residual[n] + selected + shared);
    } else {
#pragma unroll
        for (u32 s = 0; s < STREAMS; ++s)
            out.put(s * width + n, residual[s * width + n] + (selected + shared) * inject[s]);
    }
}

// Adjacent columns n and n + 1 joining their row's residual (`routed_combine`
// epilogue pair): `float2` residual loads and `out.put2` stores when 8-byte
// aligned, hoisting `inject[s]` once per pair.
template <u32 STREAMS, bool INJECTED>
__device__ __forceinline__ void routed_join_pair(element::Out<element::F32> out, const float *residual,
                                                 const float *inject, u64 width, u64 n, float selected0,
                                                 float selected1, float shared0, float shared1) {
    if constexpr (!INJECTED) {
        const float2 base = ((reinterpret_cast<u64>(residual + n) & 7u) == 0u)
                                ? *reinterpret_cast<const float2 *>(residual + n)
                                : make_float2(residual[n], residual[n + 1]);
        const float y0 = base.x + selected0 + shared0;
        const float y1 = base.y + selected1 + shared1;
        if ((reinterpret_cast<u64>(out.stored(n)) & 7u) == 0u) {
            out.put2(n, y0, y1);
        } else {
            out.put(n, y0);
            out.put(n + 1, y1);
        }
    } else {
        const float total0 = selected0 + shared0;
        const float total1 = selected1 + shared1;
#pragma unroll
        for (u32 s = 0; s < STREAMS; ++s) {
            const float gain = inject[s];
            const u64 at = s * width + n;
            const float2 base = ((reinterpret_cast<u64>(residual + at) & 7u) == 0u)
                                    ? *reinterpret_cast<const float2 *>(residual + at)
                                    : make_float2(residual[at], residual[at + 1]);
            const float y0 = base.x + total0 * gain;
            const float y1 = base.y + total1 * gain;
            if ((reinterpret_cast<u64>(out.stored(at)) & 7u) == 0u) {
                out.put2(at, y0, y1);
            } else {
                out.put(at, y0);
                out.put(at + 1, y1);
            }
        }
    }
}

// An output entry's epilogue: `Residual`, or `Scatter` over STREAMS streams
// when the entry takes inject values (COMBED: and a comb after them). `make`
// forms either from every field (`Residual` reads the leading ones).
template <class A, class Rows, u32 STREAMS, bool INJECTED, bool COMBED> struct OutputOf {
    using type = Scatter<A, Rows, STREAMS, COMBED>;
    __device__ static __forceinline__ type make(const float *residual, u64 residual_stride, Rows rows, element::Out<element::F32> out,
                                                u64 stride, const float *inject, u64 inject_stride, u64 width) {
        return type{residual, residual_stride, rows, out, stride, inject, inject_stride, width};
    }
};
template <class A, class Rows> struct OutputOf<A, Rows, 1, false, false> {
    using type = Residual<A, Rows>;
    __device__ static __forceinline__ type make(const float *residual, u64 residual_stride, Rows rows, element::Out<element::F32> out,
                                                u64 stride, const float *, u64, u64) {
        return type{residual, residual_stride, rows, out, stride};
    }
};


} // namespace projection
