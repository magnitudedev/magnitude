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
    __device__ __forceinline__ void quad(u32 m, u64 n, float4 values) const {
        const float y0 = A::round(values.x);
        const float y1 = A::round(values.y);
        const float y2 = A::round(values.z);
        const float y3 = A::round(values.w);
        const u64 source = rows(m) * residual_stride + n;
        const u64 target = m * stride + n;
        if (((reinterpret_cast<u64>(residual + source) | reinterpret_cast<u64>(out.stored(target))) & 15u) == 0) {
            const float4 base = *reinterpret_cast<const float4 *>(residual + source);
            const float result[4] = {base.x + y0, base.y + y1, base.z + y2, base.w + y3};
            out.put_span<4>(target, result);
            return;
        }
        const float *from = residual + source;
        out.put(target, from[0] + y0);
        out.put(target + 1, from[1] + y1);
        out.put(target + 2, from[2] + y2);
        out.put(target + 3, from[3] + y3);
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
        const float y0 = A::round(first);
        const float y1 = A::round(second);
        const u64 source = rows(m) * residual_stride + n;
        const u64 target = m * stride + n;
        const float *own = inject + m * inject_stride;
        float inj[STREAMS];
#pragma unroll
        for (u32 s = 0; s < STREAMS; ++s)
            inj[s] = own[s];
        const bool aligned =
            ((reinterpret_cast<u64>(residual + source) | reinterpret_cast<u64>(out.stored(target))) & 7u) == 0 &&
            ((source | target) & 1u) == 0 && (STREAMS == 1 || (width & 1u) == 0);
        if constexpr (!COMBED) {
            if (aligned) {
#pragma unroll
                for (u32 s = 0; s < STREAMS; ++s) {
                    const u64 offset = s * width;
                    const float2 base = *reinterpret_cast<const float2 *>(residual + source + offset);
                    out.put2(target + offset, base.x + y0 * inj[s], base.y + y1 * inj[s]);
                }
            } else {
                const float *from = residual + source;
                const element::Out<element::F32> to = out.offset(target);
#pragma unroll
                for (u32 s = 0; s < STREAMS; ++s) {
                    const u64 offset = s * width;
                    to.put(offset, from[offset] + y0 * inj[s]);
                    to.put(offset + 1, from[offset + 1] + y1 * inj[s]);
                }
            }
        } else {
            const float *comb_ptr = own + STREAMS;
            float comb[STREAMS * STREAMS];
#pragma unroll
            for (u32 i = 0; i < STREAMS * STREAMS; ++i)
                comb[i] = comb_ptr[i];
            float2 streams[STREAMS];
            if (aligned) {
#pragma unroll
                for (u32 s = 0; s < STREAMS; ++s)
                    streams[s] = *reinterpret_cast<const float2 *>(residual + source + s * width);
            } else {
                const float *from = residual + source;
#pragma unroll
                for (u32 s = 0; s < STREAMS; ++s) {
                    const u64 offset = s * width;
                    streams[s] = make_float2(from[offset], from[offset + 1]);
                }
            }
#pragma unroll
            for (u32 d = 0; d < STREAMS; ++d) {
                float mixed0 = 0.0f;
                float mixed1 = 0.0f;
#pragma unroll
                for (u32 s = 0; s < STREAMS; ++s) {
                    const float c = comb[s * STREAMS + d];
                    mixed0 = mixed0 + c * streams[s].x;
                    mixed1 = mixed1 + c * streams[s].y;
                }
                const float out0 = mixed0 + y0 * inj[d];
                const float out1 = mixed1 + y1 * inj[d];
                const u64 offset = d * width;
                if (aligned) {
                    out.put2(target + offset, out0, out1);
                } else {
                    out.put(target + offset, out0);
                    out.put(target + offset + 1, out1);
                }
            }
        }
    }
    __device__ __forceinline__ void quad(u32 m, u64 n, float4 values) const {
        const float y0 = A::round(values.x);
        const float y1 = A::round(values.y);
        const float y2 = A::round(values.z);
        const float y3 = A::round(values.w);
        const u64 source = rows(m) * residual_stride + n;
        const u64 target = m * stride + n;
        const float *own = inject + m * inject_stride;
        float inj[STREAMS];
#pragma unroll
        for (u32 s = 0; s < STREAMS; ++s)
            inj[s] = own[s];
        const bool aligned =
            ((reinterpret_cast<u64>(residual + source) | reinterpret_cast<u64>(out.stored(target))) & 15u) == 0 &&
            (STREAMS == 1 || (width & 3u) == 0);
        if constexpr (!COMBED) {
            if (aligned) {
#pragma unroll
                for (u32 s = 0; s < STREAMS; ++s) {
                    const u64 offset = s * width;
                    const float4 base = *reinterpret_cast<const float4 *>(residual + source + offset);
                    const float result[4] = {base.x + y0 * inj[s], base.y + y1 * inj[s], base.z + y2 * inj[s],
                                             base.w + y3 * inj[s]};
                    out.put_span<4>(target + offset, result);
                }
            } else {
                const float *from = residual + source;
                const element::Out<element::F32> to = out.offset(target);
#pragma unroll
                for (u32 s = 0; s < STREAMS; ++s) {
                    const u64 offset = s * width;
                    to.put(offset, from[offset] + y0 * inj[s]);
                    to.put(offset + 1, from[offset + 1] + y1 * inj[s]);
                    to.put(offset + 2, from[offset + 2] + y2 * inj[s]);
                    to.put(offset + 3, from[offset + 3] + y3 * inj[s]);
                }
            }
        } else {
            const float *comb_ptr = own + STREAMS;
            float comb[STREAMS * STREAMS];
#pragma unroll
            for (u32 i = 0; i < STREAMS * STREAMS; ++i)
                comb[i] = comb_ptr[i];
            float4 streams[STREAMS];
            if (aligned) {
#pragma unroll
                for (u32 s = 0; s < STREAMS; ++s)
                    streams[s] = *reinterpret_cast<const float4 *>(residual + source + s * width);
            } else {
                const float *from = residual + source;
#pragma unroll
                for (u32 s = 0; s < STREAMS; ++s) {
                    const u64 offset = s * width;
                    streams[s] = make_float4(from[offset], from[offset + 1], from[offset + 2], from[offset + 3]);
                }
            }
#pragma unroll
            for (u32 d = 0; d < STREAMS; ++d) {
                float mixed0 = 0.0f;
                float mixed1 = 0.0f;
                float mixed2 = 0.0f;
                float mixed3 = 0.0f;
#pragma unroll
                for (u32 s = 0; s < STREAMS; ++s) {
                    const float c = comb[s * STREAMS + d];
                    mixed0 = mixed0 + c * streams[s].x;
                    mixed1 = mixed1 + c * streams[s].y;
                    mixed2 = mixed2 + c * streams[s].z;
                    mixed3 = mixed3 + c * streams[s].w;
                }
                const float result[4] = {mixed0 + y0 * inj[d], mixed1 + y1 * inj[d], mixed2 + y2 * inj[d],
                                         mixed3 + y3 * inj[d]};
                const u64 offset = d * width;
                if (aligned) {
                    out.put_span<4>(target + offset, result);
                } else {
                    out.put(target + offset, result[0]);
                    out.put(target + offset + 1, result[1]);
                    out.put(target + offset + 2, result[2]);
                    out.put(target + offset + 3, result[3]);
                }
            }
        }
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
