// The output joins of the projection family: an output entry's epilogue onto
// the F32 residual (`Residual`) or the hyper-connection streams (`Scatter`,
// with the comb), the routed sublayer's join (`routed_join`,
// `routed_join_quad`), and the row maps they read the residual through. Apart
// from `projection.h` so an entry with no activation element (`stream_join`)
// joins through the same epilogues without compiling the projection bodies.
// The counterpart of `cuda/lib/projection/join.cuh`.
#pragma once

#include <seismic/element.h>

#ifndef PROJECTION_UNROLL
// Register-array loops are fully unrolled: an array indexed by a rolled loop
// lives in stack memory.
#define PROJECTION_UNROLL _Pragma("clang loop unroll(full)")
#endif

namespace projection {

// ---------------------------------------------------------------------------
// Row maps: which input row feeds projected row m.

struct AllRows {
    uint at(uint m) const { return m; }
};
struct SelectedRows {
    device const int *table;
    uint at(uint m) const { return uint(table[m]); }
};

// ---------------------------------------------------------------------------
// Output joins: F32 residual and hyper-connection scatter epilogues.

template <typename A, typename Rows>
struct Residual {
    element::Out<element::F32> y;
    ulong stride0, stride1;
    device const float *residual;
    ulong residual0, residual1;
    Rows rows;
    void store(uint m, uint n, float value) const {
        y.put(ulong(m) * stride0 + ulong(n) * stride1,
            residual[ulong(rows.at(m)) * residual0 + ulong(n) * residual1] + A::round(value));
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
    // Row m resolved once (its result row and its residual row), for a
    // caller that stores several columns of it.
    struct Row {
        element::Out<element::F32> y;
        device const float *residual;
        ulong stride1, residual1;
        void store(uint n, float value) const {
            y.put(ulong(n) * stride1, residual[ulong(n) * residual1] + A::round(value));
        }
        // Columns n .. n + 3: one vector load and store on unit strides.
        void store4(uint n, float4 values) const {
            float4 rounded;
            PROJECTION_UNROLL
            for (uint i = 0; i < 4; ++i)
                rounded[i] = A::round(values[i]);
            if (stride1 == 1 && residual1 == 1) {
                const float4 from = float4(*reinterpret_cast<device const packed_float4 *>(residual + n));
                y.put4(n, from + rounded);
                return;
            }
            PROJECTION_UNROLL
            for (uint i = 0; i < 4; ++i)
                y.put(ulong(n + i) * stride1, residual[ulong(n + i) * residual1] + rounded[i]);
        }
        // Columns n .. n + 7 from their sums as (even, odd): two vector
        // loads and stores on unit strides.
        void store8(uint n, float4 even, float4 odd) const {
            float4 low = float4(even.x, odd.x, even.y, odd.y), high = float4(even.z, odd.z, even.w, odd.w);
            PROJECTION_UNROLL
            for (uint i = 0; i < 4; ++i) {
                low[i] = A::round(low[i]);
                high[i] = A::round(high[i]);
            }
            if (stride1 == 1 && residual1 == 1) {
                device const packed_float4 *from = reinterpret_cast<device const packed_float4 *>(residual + n);
                y.put4(n, float4(from[0]) + low);
                y.put4(n + 4u, float4(from[1]) + high);
                return;
            }
            PROJECTION_UNROLL
            for (uint i = 0; i < 4; ++i) {
                y.put(ulong(n + i) * stride1, residual[ulong(n + i) * residual1] + low[i]);
                y.put(ulong(n + 4u + i) * stride1, residual[ulong(n + 4u + i) * residual1] + high[i]);
            }
        }
    };
    Row row(uint m) const {
        return Row{y.offset(ulong(m) * stride0), residual + ulong(rows.at(m)) * residual0, stride1, residual1};
    }
};

// `scatter`: each of the STREAMS hyper-connection streams of the residual
// plus the output times the stream's inject value,
//   y[m, s * width + n] = residual[row(m), s * width + n] + round_A(acc) * inject[m, s]   (F32),
// or (COMBED) the streams mixed by the comb that follows the inject values,
//   y[m, d * width + n] = sum_s comb[m, s, d] residual[row(m), s * width + n] + round_A(acc) * inject[m, d],
// with `Residual`'s interface (column strides are the streams' too).
template <typename A, typename Rows, uint STREAMS, bool COMBED>
struct Scatter {
    element::Out<element::F32> y;
    ulong stride0, stride1;
    device const float *residual;
    ulong residual0, residual1;
    Rows rows;
    device const float *inject;
    ulong inject0;
    uint width;
    struct Row {
        element::Out<element::F32> y;
        device const float *residual;
        ulong stride1, residual1;
        device const float *inject;
        uint width;
        void store(uint n, float value) const {
            const float rounded = A::round(value);
            if constexpr (!COMBED) {
                PROJECTION_UNROLL
                for (uint s = 0; s < STREAMS; ++s) {
                    const ulong column = ulong(s) * width + n;
                    y.put(column * stride1, residual[column * residual1] + rounded * inject[s]);
                }
                return;
            } else {
                float streams[STREAMS];
                PROJECTION_UNROLL
                for (uint s = 0; s < STREAMS; ++s) {
                    streams[s] = residual[(ulong(s) * width + n) * residual1];
                }
                device const float *comb = inject + STREAMS;
                PROJECTION_UNROLL
                for (uint d = 0; d < STREAMS; ++d) {
                    float mixed = 0.0f;
                    PROJECTION_UNROLL
                    for (uint s = 0; s < STREAMS; ++s) {
                        mixed = mixed + comb[s * STREAMS + d] * streams[s];
                    }
                    y.put((ulong(d) * width + n) * stride1, mixed + rounded * inject[d]);
                }
            }
        }
        void store4_with(uint n, float4 rounded, thread const float (&inj)[STREAMS],
            thread const float (&comb)[COMBED ? STREAMS * STREAMS : 1]) const {
            if constexpr (!COMBED) {
                if (stride1 == 1 && residual1 == 1) {
                    PROJECTION_UNROLL
                    for (uint s = 0; s < STREAMS; ++s) {
                        const ulong column = ulong(s) * width + n;
                        const float4 base = float4(*reinterpret_cast<device const packed_float4 *>(residual + column));
                        y.put4(column, base + rounded * inj[s]);
                    }
                    return;
                }
                PROJECTION_UNROLL
                for (uint s = 0; s < STREAMS; ++s) {
                    const ulong base_col = ulong(s) * width + n;
                    PROJECTION_UNROLL
                    for (uint i = 0; i < 4; ++i) {
                        const ulong column = base_col + i;
                        y.put(column * stride1, residual[column * residual1] + rounded[i] * inj[s]);
                    }
                }
            } else {
                if (stride1 == 1 && residual1 == 1) {
                    float4 streams[STREAMS];
                    PROJECTION_UNROLL
                    for (uint s = 0; s < STREAMS; ++s) {
                        streams[s] = float4(*reinterpret_cast<device const packed_float4 *>(
                            residual + ulong(s) * width + n));
                    }
                    PROJECTION_UNROLL
                    for (uint d = 0; d < STREAMS; ++d) {
                        float4 mixed = float4(0.0f);
                        PROJECTION_UNROLL
                        for (uint s = 0; s < STREAMS; ++s) {
                            mixed = mixed + comb[s * STREAMS + d] * streams[s];
                        }
                        y.put4(ulong(d) * width + n, mixed + rounded * inj[d]);
                    }
                    return;
                }
                PROJECTION_UNROLL
                for (uint i = 0; i < 4; ++i) {
                    float streams[STREAMS];
                    PROJECTION_UNROLL
                    for (uint s = 0; s < STREAMS; ++s) {
                        streams[s] = residual[(ulong(s) * width + n + i) * residual1];
                    }
                    PROJECTION_UNROLL
                    for (uint d = 0; d < STREAMS; ++d) {
                        float mixed = 0.0f;
                        PROJECTION_UNROLL
                        for (uint s = 0; s < STREAMS; ++s) {
                            mixed = mixed + comb[s * STREAMS + d] * streams[s];
                        }
                        y.put((ulong(d) * width + n + i) * stride1, mixed + rounded[i] * inj[d]);
                    }
                }
            }
        }
        void store4(uint n, float4 values) const {
            float4 rounded;
            PROJECTION_UNROLL
            for (uint i = 0; i < 4; ++i)
                rounded[i] = A::round(values[i]);
            float inj[STREAMS];
            PROJECTION_UNROLL
            for (uint s = 0; s < STREAMS; ++s)
                inj[s] = inject[s];
            float comb[COMBED ? STREAMS * STREAMS : 1];
            if constexpr (COMBED) {
                device const float *comb_ptr = inject + STREAMS;
                PROJECTION_UNROLL
                for (uint i = 0; i < STREAMS * STREAMS; ++i)
                    comb[i] = comb_ptr[i];
            }
            store4_with(n, rounded, inj, comb);
        }
        void store8(uint n, float4 even, float4 odd) const {
            float4 low = float4(even.x, odd.x, even.y, odd.y), high = float4(even.z, odd.z, even.w, odd.w);
            PROJECTION_UNROLL
            for (uint i = 0; i < 4; ++i) {
                low[i] = A::round(low[i]);
                high[i] = A::round(high[i]);
            }
            float inj[STREAMS];
            PROJECTION_UNROLL
            for (uint s = 0; s < STREAMS; ++s)
                inj[s] = inject[s];
            float comb[COMBED ? STREAMS * STREAMS : 1];
            if constexpr (COMBED) {
                device const float *comb_ptr = inject + STREAMS;
                PROJECTION_UNROLL
                for (uint i = 0; i < STREAMS * STREAMS; ++i)
                    comb[i] = comb_ptr[i];
            }
            store4_with(n, low, inj, comb);
            store4_with(n + 4u, high, inj, comb);
        }
    };
    Row row(uint m) const {
        return Row{y.offset(ulong(m) * stride0), residual + ulong(rows.at(m)) * residual0, stride1, residual1,
            inject + ulong(m) * inject0, width};
    }
    void store(uint m, uint n, float value) const { row(m).store(n, value); }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// An output entry's epilogue: `Residual`, or `Scatter` over STREAMS streams
// when the entry takes inject values (COMBED: and a comb after them). `make`
// forms either from every field (`Residual` reads the leading ones).
template <typename A, typename Rows, uint STREAMS, bool INJECTED, bool COMBED>
struct OutputOf {
    typedef Scatter<A, Rows, STREAMS, COMBED> type;
    static type make(element::Out<element::F32> y, ulong stride0, ulong stride1, device const float *residual, ulong residual0,
        ulong residual1, Rows rows, device const float *inject, ulong inject0, uint width) {
        return type{y, stride0, stride1, residual, residual0, residual1, rows, inject, inject0, width};
    }
};
template <typename A, typename Rows>
struct OutputOf<A, Rows, 1, false, false> {
    typedef Residual<A, Rows> type;
    static type make(element::Out<element::F32> y, ulong stride0, ulong stride1, device const float *residual, ulong residual0,
        ulong residual1, Rows rows, device const float *, ulong, uint) {
        return type{y, stride0, stride1, residual, residual0, residual1, rows};
    }
};

// A routed sublayer's column n joining its row's residual (`routed_output`,
// `routed_combine`): residual + selected + shared, in that order, or (INJECTED)
// for each of the STREAMS streams residual[s] + (selected + shared) * inject[s].
// `out`, `residual` and `inject` point at the row; strides are column strides.
template <uint STREAMS, bool INJECTED>
inline void routed_join(element::Out<element::F32> out, ulong out1, device const float *residual, ulong residual1,
    device const float *inject, ulong width, ulong n, float selected, float shared) {
    if constexpr (!INJECTED) {
        out.put(n * out1, residual[n * residual1] + selected + shared);
    } else {
        const float total = selected + shared;
        PROJECTION_UNROLL
        for (uint s = 0; s < STREAMS; ++s) {
            const ulong column = ulong(s) * width + n;
            out.put(column * out1, residual[column * residual1] + total * inject[s]);
        }
    }
}

// Four adjacent columns n .. n + 3 joining their row's residual: one
// `packed_float4` residual load and `out.put4` store per stream on unit
// column strides, hoisting `inject[s]` once per quad.
template <uint STREAMS, bool INJECTED>
inline void routed_join_quad(element::Out<element::F32> out, ulong out1, device const float *residual, ulong residual1,
    device const float *inject, ulong width, ulong n, float4 selected, float4 shared) {
    if (out1 == 1 && residual1 == 1) {
        if constexpr (!INJECTED) {
            const float4 base = float4(*reinterpret_cast<device const packed_float4 *>(residual + n));
            out.put4(n, (base + selected) + shared);
        } else {
            const float4 total = selected + shared;
            PROJECTION_UNROLL
            for (uint s = 0; s < STREAMS; ++s) {
                const float gain = inject[s];
                const ulong at = ulong(s) * width + n;
                const float4 base = float4(*reinterpret_cast<device const packed_float4 *>(residual + at));
                out.put4(at, base + total * gain);
            }
        }
        return;
    }
    PROJECTION_UNROLL
    for (uint i = 0; i < 4u; ++i)
        routed_join<STREAMS, INJECTED>(out, out1, residual, residual1, inject, width, n + i, selected[i], shared[i]);
}

} // namespace projection
