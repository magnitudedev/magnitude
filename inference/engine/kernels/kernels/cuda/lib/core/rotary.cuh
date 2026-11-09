// Rotary position embeddings of one head row held by a warp (`rotate_pairs`,
// attention.seismic): the layout of a row's rotated pairs over its 32 lanes
// and the rotation itself. Shared by the attention library, the pooled rows
// and the indexers. The counterpart of `metal/lib/core/rotary.h`.
//
// This file is independent of any entry ABI: a caller names its layout with
// `rotary::layout<W, P, FIRST, ADJACENT>` and its row's tables with
// `rotary::table`. An entry whose ABI declares P, S, RT and RA and the
// `rotary_components`, `rotary_frequencies` and `rotary_amplitudes` tensors
// names them with `ROTARY_ENTRY_LAYOUT` and `ROTARY_ENTRY_TABLE`, expanded
// where its words are in scope.

#pragma once

// The entry's layout of a W-column row: its P pairs over the leading columns,
// or the trailing ones (RT = 1, from S); half-split pairs, or adjacent ones
// (RA = 1).
#define ROTARY_ENTRY_LAYOUT(W)                                                                     \
    rotary::layout<(W), static_cast<int>(SEISMIC_DIM_P), SEISMIC_DIM_RT != 0 ? static_cast<int>(SEISMIC_DIM_S) : 0, \
                   SEISMIC_DIM_RA != 0>

// The entry's rotary tables at the row coordinates `coordinates`, stepping by
// `coordinate_stride`.
#define ROTARY_ENTRY_TABLE(coordinates, coordinate_stride)                                         \
    rotary::table {                                                                                \
        (coordinates), (coordinate_stride),                                                        \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS)),          \
            SEISMIC_ROTARY_COMPONENTS_STRIDE_0,                                                    \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES)),       \
            SEISMIC_ROTARY_FREQUENCIES_STRIDE_0,                                                   \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_AMPLITUDES)),        \
            SEISMIC_ROTARY_AMPLITUDES_STRIDE_0                                                     \
    }

namespace rotary {

// One row's rotary tables: pair p turns by coordinate axis components[p] at
// frequencies[p], its cosine and sine scaled by amplitudes[p]. Each pointer
// steps by its stride (the entry's ABI strides).
struct table {
    const int *coordinates;
    unsigned long long coordinate_stride;
    const int *components;
    unsigned long long component_stride;
    const float *frequencies;
    unsigned long long frequency_stride;
    const float *amplitudes;
    unsigned long long amplitude_stride;

    // Pair p's amplitude-scaled cosine and sine.
    __device__ __forceinline__ void turn(int p, float &cosine, float &sine) const {
        const int component = components[static_cast<unsigned long long>(p) * component_stride];
        const int coordinate = coordinates[static_cast<unsigned long long>(component) * coordinate_stride];
        const float angle = static_cast<float>(coordinate) * frequencies[static_cast<unsigned long long>(p) * frequency_stride];
        sincosf(angle, &sine, &cosine);
        const float amplitude = amplitudes[static_cast<unsigned long long>(p) * amplitude_stride];
        cosine *= amplitude;
        sine *= amplitude;
    }
};

// A head row of W columns over a warp, lane l holding columns
// [l E, (l + 1) E), E = W / 32: its P rotary pairs over the 2P columns from
// FIRST, pair p (FIRST + p, FIRST + p + P), or (FIRST + 2p, FIRST + 2p + 1)
// when ADJACENT.
template <int W, int P, int FIRST, bool ADJACENT>
struct layout {
    static constexpr int E = W / 32;
    static_assert(W % 32 == 0, "a head row must cover whole lanes");
    // An adjacent pair's columns differ in their lowest bit only (`partner`).
    static_assert(!ADJACENT || FIRST % 2 == 0, "adjacent pairs start at an even column");

    // The place of `column` among the rotated columns: whether it is
    // rotated, its pair, whether it leads the pair, its pair's other column.
    __device__ static __forceinline__ bool rotated(int column) {
        return static_cast<unsigned>(column - FIRST) < static_cast<unsigned>(2 * P);
    }
    __device__ static __forceinline__ int pair(int column) {
        const int j = column - FIRST;
        return ADJACENT ? j / 2 : j % (P > 0 ? P : 1);
    }
    __device__ static __forceinline__ bool leads(int column) {
        const int j = column - FIRST;
        return ADJACENT ? j % 2 == 0 : j < P;
    }
    __device__ static __forceinline__ int partner(int column) {
        if (ADJACENT) return column ^ 1;
        return leads(column) ? column + P : column - P;
    }
    // The leading column of pair `p`.
    __device__ static __forceinline__ int leading(int p) { return FIRST + (ADJACENT ? 2 * p : p); }

    // Lane `lane`'s columns `x` of the row, its pairs rotated by `t`.
    // `exchange` is W floats of shared memory private to the warp, through
    // which each column reads its partner. The whole warp calls it.
    __device__ static __forceinline__ void rotate(float (&x)[E], const table &t, float *exchange, int lane) {
        if constexpr (P == 0) return;
        const int first = lane * E;
#pragma unroll
        for (int i = 0; i < E; ++i) exchange[first + i] = x[i];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < E; ++i) {
            const int column = first + i;
            if (rotated(column)) {
                float cosine, sine;
                t.turn(pair(column), cosine, sine);
                const float other = exchange[partner(column)];
                x[i] = leads(column) ? x[i] * cosine - other * sine : x[i] * cosine + other * sine;
            }
        }
        __syncwarp();
    }
};

}  // namespace rotary
