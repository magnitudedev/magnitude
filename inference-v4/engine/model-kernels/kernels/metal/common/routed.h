// Shared pieces of the routed (mixture-of-experts) Metal entries: weight slot
// bindings onto the K1 packet decoders, the activation and MMA operand types,
// and the prologues/epilogues the routed projections add to the K1 family.
//
// Weight tensors are bound to slots before this file is included:
//     #define ROUTED_W0 SEISMIC_EXPERT_GATE
// which defines `routed::W0` (the `packets` decoder of the bound
// representation) and `ROUTED_W0_LAYOUT(k)` (its `rows16` geometry for rows
// of `k` logical values). Slots W0..W3 exist. Layout names are formed by
// token pasting from the bound prefix.

#include "common/projection.h"

#define ROUTED_CAT_(a, b) a##b
#define ROUTED_CAT(a, b) ROUTED_CAT_(a, b)
// 1 when the macro `<prefix><suffix>` is defined as 1, else 0.
#define ROUTED_SECOND_(a, b, ...) b
#define ROUTED_SECOND(...) ROUTED_SECOND_(__VA_ARGS__, 0, 0)
#define ROUTED_PROBE_1 ~, 1
#define ROUTED_IS_ONE(value) ROUTED_SECOND(ROUTED_CAT(ROUTED_PROBE_, value))
#define ROUTED_HAS(prefix, suffix) ROUTED_IS_ONE(ROUTED_CAT(prefix, suffix))

#define ROUTED_ROWS16_Q4K(P)                                                              \
    packets::rows16 { ROUTED_CAT(P, _ROW_STRIDE_BYTES), ROUTED_CAT(P, _PLANE_CODES_LO_ROW_OFFSET), 0, \
        ROUTED_CAT(P, _PLANE_SCALES_ROW_OFFSET), ROUTED_CAT(P, _PLANE_SUPERS_ROW_OFFSET) }
#define ROUTED_ROWS16_HIGH(P)                                                             \
    packets::rows16 { ROUTED_CAT(P, _ROW_STRIDE_BYTES), ROUTED_CAT(P, _PLANE_CODES_LO_ROW_OFFSET), \
        ROUTED_CAT(P, _PLANE_CODES_HI_ROW_OFFSET), ROUTED_CAT(P, _PLANE_SCALES_ROW_OFFSET),     \
        ROUTED_CAT(P, _PLANE_SUPERS_ROW_OFFSET) }
#define ROUTED_ROWS16_Q8(P)                                                               \
    packets::rows16 { ROUTED_CAT(P, _ROW_STRIDE_BYTES), ROUTED_CAT(P, _PLANE_CODES_ROW_OFFSET), 0, 0, \
        ROUTED_CAT(P, _PLANE_SUPERS_ROW_OFFSET) }
// A dense weight row is `k` contiguous elements.
#define ROUTED_ROWS16_DENSE(E, k) packets::rows16 { ulong(k) * E::bytes, 0, 0, 0, 0 }

// Binds slot `SLOT` to the decoder type `TYPE`.
#define ROUTED_BIND(SLOT, TYPE) namespace routed { typedef TYPE SLOT; }

#if defined(ROUTED_W0)
#if ROUTED_HAS(ROUTED_W0, _REPRESENTATION_F32)
ROUTED_BIND(W0, packets::dense<packets::f32>)
#define ROUTED_W0_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::f32, k)
#elif ROUTED_HAS(ROUTED_W0, _REPRESENTATION_BF16)
ROUTED_BIND(W0, packets::dense<packets::bf16>)
#define ROUTED_W0_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::bf16, k)
#elif ROUTED_HAS(ROUTED_W0, _REPRESENTATION_F16)
ROUTED_BIND(W0, packets::dense<packets::f16>)
#define ROUTED_W0_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::f16, k)
#elif !ROUTED_HAS(ROUTED_W0, _LAYOUT_ROWS16)
#error "ROUTED_W0 must be dense or in the rows16 layout"
#elif ROUTED_HAS(ROUTED_W0, _REPRESENTATION_Q4K)
ROUTED_BIND(W0, packets::q4k)
#define ROUTED_W0_LAYOUT(k) ROUTED_ROWS16_Q4K(ROUTED_W0)
#elif ROUTED_HAS(ROUTED_W0, _REPRESENTATION_Q5K)
ROUTED_BIND(W0, packets::q5k)
#define ROUTED_W0_LAYOUT(k) ROUTED_ROWS16_HIGH(ROUTED_W0)
#elif ROUTED_HAS(ROUTED_W0, _REPRESENTATION_Q6K)
ROUTED_BIND(W0, packets::q6k)
#define ROUTED_W0_LAYOUT(k) ROUTED_ROWS16_HIGH(ROUTED_W0)
#elif ROUTED_HAS(ROUTED_W0, _REPRESENTATION_Q8G32S)
ROUTED_BIND(W0, packets::q8)
#define ROUTED_W0_LAYOUT(k) ROUTED_ROWS16_Q8(ROUTED_W0)
#else
#error "ROUTED_W0: unsupported weight representation"
#endif
#endif

#if defined(ROUTED_W1)
#if ROUTED_HAS(ROUTED_W1, _REPRESENTATION_F32)
ROUTED_BIND(W1, packets::dense<packets::f32>)
#define ROUTED_W1_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::f32, k)
#elif ROUTED_HAS(ROUTED_W1, _REPRESENTATION_BF16)
ROUTED_BIND(W1, packets::dense<packets::bf16>)
#define ROUTED_W1_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::bf16, k)
#elif ROUTED_HAS(ROUTED_W1, _REPRESENTATION_F16)
ROUTED_BIND(W1, packets::dense<packets::f16>)
#define ROUTED_W1_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::f16, k)
#elif !ROUTED_HAS(ROUTED_W1, _LAYOUT_ROWS16)
#error "ROUTED_W1 must be dense or in the rows16 layout"
#elif ROUTED_HAS(ROUTED_W1, _REPRESENTATION_Q4K)
ROUTED_BIND(W1, packets::q4k)
#define ROUTED_W1_LAYOUT(k) ROUTED_ROWS16_Q4K(ROUTED_W1)
#elif ROUTED_HAS(ROUTED_W1, _REPRESENTATION_Q5K)
ROUTED_BIND(W1, packets::q5k)
#define ROUTED_W1_LAYOUT(k) ROUTED_ROWS16_HIGH(ROUTED_W1)
#elif ROUTED_HAS(ROUTED_W1, _REPRESENTATION_Q6K)
ROUTED_BIND(W1, packets::q6k)
#define ROUTED_W1_LAYOUT(k) ROUTED_ROWS16_HIGH(ROUTED_W1)
#elif ROUTED_HAS(ROUTED_W1, _REPRESENTATION_Q8G32S)
ROUTED_BIND(W1, packets::q8)
#define ROUTED_W1_LAYOUT(k) ROUTED_ROWS16_Q8(ROUTED_W1)
#else
#error "ROUTED_W1: unsupported weight representation"
#endif
#endif

#if defined(ROUTED_W2)
#if ROUTED_HAS(ROUTED_W2, _REPRESENTATION_F32)
ROUTED_BIND(W2, packets::dense<packets::f32>)
#define ROUTED_W2_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::f32, k)
#elif ROUTED_HAS(ROUTED_W2, _REPRESENTATION_BF16)
ROUTED_BIND(W2, packets::dense<packets::bf16>)
#define ROUTED_W2_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::bf16, k)
#elif ROUTED_HAS(ROUTED_W2, _REPRESENTATION_F16)
ROUTED_BIND(W2, packets::dense<packets::f16>)
#define ROUTED_W2_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::f16, k)
#elif !ROUTED_HAS(ROUTED_W2, _LAYOUT_ROWS16)
#error "ROUTED_W2 must be dense or in the rows16 layout"
#elif ROUTED_HAS(ROUTED_W2, _REPRESENTATION_Q4K)
ROUTED_BIND(W2, packets::q4k)
#define ROUTED_W2_LAYOUT(k) ROUTED_ROWS16_Q4K(ROUTED_W2)
#elif ROUTED_HAS(ROUTED_W2, _REPRESENTATION_Q5K)
ROUTED_BIND(W2, packets::q5k)
#define ROUTED_W2_LAYOUT(k) ROUTED_ROWS16_HIGH(ROUTED_W2)
#elif ROUTED_HAS(ROUTED_W2, _REPRESENTATION_Q6K)
ROUTED_BIND(W2, packets::q6k)
#define ROUTED_W2_LAYOUT(k) ROUTED_ROWS16_HIGH(ROUTED_W2)
#elif ROUTED_HAS(ROUTED_W2, _REPRESENTATION_Q8G32S)
ROUTED_BIND(W2, packets::q8)
#define ROUTED_W2_LAYOUT(k) ROUTED_ROWS16_Q8(ROUTED_W2)
#else
#error "ROUTED_W2: unsupported weight representation"
#endif
#endif

#if defined(ROUTED_W3)
#if ROUTED_HAS(ROUTED_W3, _REPRESENTATION_F32)
ROUTED_BIND(W3, packets::dense<packets::f32>)
#define ROUTED_W3_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::f32, k)
#elif ROUTED_HAS(ROUTED_W3, _REPRESENTATION_BF16)
ROUTED_BIND(W3, packets::dense<packets::bf16>)
#define ROUTED_W3_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::bf16, k)
#elif ROUTED_HAS(ROUTED_W3, _REPRESENTATION_F16)
ROUTED_BIND(W3, packets::dense<packets::f16>)
#define ROUTED_W3_LAYOUT(k) ROUTED_ROWS16_DENSE(packets::f16, k)
#elif !ROUTED_HAS(ROUTED_W3, _LAYOUT_ROWS16)
#error "ROUTED_W3 must be dense or in the rows16 layout"
#elif ROUTED_HAS(ROUTED_W3, _REPRESENTATION_Q4K)
ROUTED_BIND(W3, packets::q4k)
#define ROUTED_W3_LAYOUT(k) ROUTED_ROWS16_Q4K(ROUTED_W3)
#elif ROUTED_HAS(ROUTED_W3, _REPRESENTATION_Q5K)
ROUTED_BIND(W3, packets::q5k)
#define ROUTED_W3_LAYOUT(k) ROUTED_ROWS16_HIGH(ROUTED_W3)
#elif ROUTED_HAS(ROUTED_W3, _REPRESENTATION_Q6K)
ROUTED_BIND(W3, packets::q6k)
#define ROUTED_W3_LAYOUT(k) ROUTED_ROWS16_HIGH(ROUTED_W3)
#elif ROUTED_HAS(ROUTED_W3, _REPRESENTATION_Q8G32S)
ROUTED_BIND(W3, packets::q8)
#define ROUTED_W3_LAYOUT(k) ROUTED_ROWS16_Q8(ROUTED_W3)
#else
#error "ROUTED_W3: unsupported weight representation"
#endif
#endif

namespace routed {

// The activation element A.
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 Act;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 Act;
#else
#error "routed projections require a bf16 or f16 activation element"
#endif

// Rows of weight tensor `base` starting at row `first` (an expert's first
// row inside an [E, N, K] tensor).
template <typename W>
inline projection::weight_rows<W> rows_from(device const uchar *base, packets::rows16 layout, ulong first,
    uint k) {
    return projection::weight_rows<W>{base + first * layout.stride, layout, k, nullptr};
}

// A plain activation prologue over rows of a [rows, columns] A tensor.
inline projection::input_plain<Act> activation(device const uchar *x, ulong stride0, ulong stride1,
    uint columns) {
    return projection::input_plain<Act>{x, stride0, stride1, columns, projection::row_map{nullptr}};
}

// The live rows of a grouped block: its `order` entries are one expert's rows
// followed by -1 padding, so the count is the index of the first -1. The
// expert GEMMs skip the MMAs of padding rows (`live_rows`).
inline uint block_rows(device const int *order, ulong stride, uint rows) {
    uint low = 0, high = rows;
    while (low < high) {
        uint middle = (low + high) / 2u;
        if (order[ulong(middle) * stride] >= 0)
            low = middle + 1u;
        else
            high = middle;
    }
    return low;
}

// The grouped A-loader of the expert GEMMs: tile row m of a block reads
// activation row `order[m]`; padding rows (-1) read zeros.
struct input_grouped {
    typedef Act activation;
    device const uchar *x;
    ulong stride0, stride1;
    uint columns;
    device const int *order;
    ulong order_stride;
    uint groups() const { return 0; }
    uint width() const { return 1; }
    float norm_input(uint, uint, uint) const { return 0.0f; }
    float epsilon() const { return 0.0f; }
    uint inverse_index(uint, uint) const { return 0; }
    float value(uint m, uint k, float) const {
        int row = order[ulong(m) * order_stride];
        return row < 0 ? 0.0f
            : Act::load(reinterpret_cast<device const typename Act::storage *>(x)
                [ulong(row) * stride0 + ulong(k) * stride1]);
    }
    uint4 words8(uint m, uint k) const {
        int row = order[ulong(m) * order_stride];
        if (row < 0)
            return uint4(0);
        device const typename Act::storage *base =
            reinterpret_cast<device const typename Act::storage *>(x) + ulong(row) * stride0;
        return projection::words8_storage<Act>(base, stride1, k, columns, stride1 == 1 && (stride0 & 7u) == 0);
    }
};

} // namespace routed
