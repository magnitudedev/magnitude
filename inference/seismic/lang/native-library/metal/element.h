// Seismic native library (Metal): dense element types. `#include <seismic/element.h>`.
//
// Every type converts one stored element to F32 (`load`), rounds F32 to the
// element (`store`, `round`), and names its storage and MSL scalar (`native`,
// the matrix-operand and vector-load type).
//
// `store` is round-to-nearest-even. For bf16 it is written out on the bits; a
// conversion to `native` (`bfloat(x)`) is the compiler's rounding. They agree
// on every finite value.
//
// The element of a tensor or element parameter is selected from its ABI
// prefix: `ELEMENT_OF(SEISMIC_NORM)`.

// 1 when the macro `<prefix><suffix>` is defined as 1, else 0; usable in both
// `#if` and constant expressions.
#define ELEMENT_CAT_(a, b) a##b
#define ELEMENT_CAT(a, b) ELEMENT_CAT_(a, b)
#define ELEMENT_SECOND_(a, b, ...) b
#define ELEMENT_SECOND(...) ELEMENT_SECOND_(__VA_ARGS__, 0, 0)
#define ELEMENT_PROBE_1 ~, 1
#define ELEMENT_IS_ONE(value) ELEMENT_SECOND(ELEMENT_CAT(ELEMENT_PROBE_, value))
#define ELEMENT_HAS(prefix, suffix) ELEMENT_IS_ONE(ELEMENT_CAT(prefix, suffix))

namespace element {

struct Bf16 {
    typedef ushort storage;
    typedef bfloat native;
    static constant constexpr uint bytes = 2;
    static float load(storage value) { return as_type<float>(uint(value) << 16); }
    // A NaN stays a NaN, quieted with its sign (as the CPU's and softfloat's narrowing do): the
    // rounding increment alone carries a payload NaN into infinity, zero or the other sign.
    static storage store(float value) {
        const uint bits = as_type<uint>(value);
        const uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
        return ushort(select(rounded, bits | 0x00400000u, metal::isnan(value)) >> 16);
    }
    static float round(float value) { return load(store(value)); }
    // Four consecutive elements at any element alignment.
    static float4 load4(device const storage *from) {
        const ushort4 bits = ushort4(*reinterpret_cast<device const packed_ushort4 *>(from));
        return as_type<float4>(uint4(bits) << 16);
    }
    // Eight consecutive elements packed in a uint4: element 2i in the low
    // half of word i. Returns (0,2,4,6) and (1,3,5,7).
    static void split8(uint4 words, thread float4 &even, thread float4 &odd) {
        even = as_type<float4>(words << 16);
        odd = as_type<float4>(words & 0xffff0000u);
    }
    // The inverse of `split8` for values already exact in bf16.
    static uint4 pack8(float4 even, float4 odd) {
        return (as_type<uint4>(odd) & 0xffff0000u) | (as_type<uint4>(even) >> 16);
    }
    typedef packed_ushort2 packed2;
    typedef packed_ushort4 packed4;
    static ushort2 store2(float2 value) { return ushort2(store(value.x), store(value.y)); }
    static ushort4 store4(float4 value) { return ushort4(store(value.x), store(value.y), store(value.z), store(value.w)); }
};

struct F16 {
    typedef half storage;
    typedef half native;
    static constant constexpr uint bytes = 2;
    static float load(storage value) { return float(value); }
    static storage store(float value) { return half(value); }
    static float round(float value) { return float(half(value)); }
    static float4 load4(device const storage *from) {
        return float4(*reinterpret_cast<device const packed_half4 *>(from));
    }
    static void split8(uint4 words, thread float4 &even, thread float4 &odd) {
        half2 a = as_type<half2>(words.x), b = as_type<half2>(words.y);
        half2 c = as_type<half2>(words.z), d = as_type<half2>(words.w);
        even = float4(a.x, b.x, c.x, d.x);
        odd = float4(a.y, b.y, c.y, d.y);
    }
    static uint4 pack8(float4 even, float4 odd) {
        return uint4(as_type<uint>(half2(even.x, odd.x)), as_type<uint>(half2(even.y, odd.y)),
            as_type<uint>(half2(even.z, odd.z)), as_type<uint>(half2(even.w, odd.w)));
    }
    typedef packed_half2 packed2;
    typedef packed_half4 packed4;
    static half2 store2(float2 value) { return half2(value); }
    static half4 store4(float4 value) { return half4(value); }
};

struct F32 {
    typedef float storage;
    typedef float native;
    static constant constexpr uint bytes = 4;
    static float load(storage value) { return value; }
    static storage store(float value) { return value; }
    static float round(float value) { return value; }
    typedef packed_float2 packed2;
    typedef packed_float4 packed4;
    static float2 store2(float2 value) { return value; }
    static float4 store4(float4 value) { return value; }
    static float4 load4(device const storage *from) {
        return float4(*reinterpret_cast<device const packed_float4 *>(from));
    }
};

// Element `index` of a dense tensor of element type E.
template <typename E>
inline float at(device const uchar *base, ulong index) {
    return E::load(reinterpret_cast<device const typename E::storage *>(base)[index]);
}

template <typename E>
inline void put(device uchar *base, ulong index, float value) {
    reinterpret_cast<device typename E::storage *>(base)[index] = E::store(value);
}

// A floating result, or a floating tensor a kernel mutates, as the kernel writes it. The kernel gets
// `device Cell<E> *` from `SEISMIC_RESULT_<n>_PARAM(name)` / `SEISMIC_OUT_<NAME>_PARAM(name)`, or an
// `Out` over scratch from `out_over_scratch`: a cell has no raw accessor, and `Out` (with the
// library's `coherent.h` views of it) is its only reader and writer, so every element a kernel
// publishes is an element of E.
template <typename E> struct Out;
template <typename E> struct CoherentOut;
template <typename E> struct SystemOut;

template <typename E> class Cell {
    typename E::storage stored;
    friend struct Out<E>;
    friend struct CoherentOut<E>;
    friend struct SystemOut<E>;
};

template <typename E> struct Out {
    explicit Out(device Cell<E> *cells) : cells(cells) {}

    // Element `index`, as F32.
    float at(ulong index) const { return E::load(cells[index].stored); }
    // Four contiguous elements from `index` as F32 (any element alignment).
    float4 at4(ulong index) const { return E::load4(&cells[index].stored); }
    // Stores `value` rounded to E.
    void put(ulong index, float value) const { cells[index].stored = E::store(value); }
    // Stores elements index and index + 1.
    void put2(ulong index, float first, float second) const {
        *reinterpret_cast<device typename E::packed2 *>(&cells[index].stored) =
            typename E::packed2(E::store2(float2(first, second)));
    }
    // Stores four contiguous elements from `index` (any element alignment).
    void put4(ulong index, float4 values) const {
        *reinterpret_cast<device typename E::packed4 *>(&cells[index].stored) =
            typename E::packed4(E::store4(values));
    }
    // Stores the eight contiguous elements from `index` (16-byte aligned) that `split8` splits into
    // `even` (0,2,4,6) and `odd` (1,3,5,7), each rounded to E, in one 16-byte store.
    void put8(ulong index, float4 even, float4 odd) const {
        if constexpr (E::bytes == 4) {
            put4(index, float4(even.x, odd.x, even.y, odd.y));
            put4(index + 4, float4(even.z, odd.z, even.w, odd.w));
        } else {
            *reinterpret_cast<device uint4 *>(&cells[index].stored) =
                E::pack8(float4(E::round(even.x), E::round(even.y), E::round(even.z), E::round(even.w)),
                    float4(E::round(odd.x), E::round(odd.y), E::round(odd.z), E::round(odd.w)));
        }
    }
    // Whether this is a handle to memory, not the null handle (`Out(nullptr)`: no tensor).
    bool present() const { return cells != nullptr; }
    // Stores the 8x8 F32 tile at `index` of rows `stride` apart: only to an F32 tensor, since the
    // tile's storage is float.
    void put_tile(ulong index, simdgroup_float8x8 tile, ulong stride, bool transpose) const {
        static_assert(E::bytes == 4, "a float tile stores only to an F32 tensor");
        simdgroup_store(tile, reinterpret_cast<device float *>(&cells[index].stored), stride, ulong2(0, 0), transpose);
    }
    // The view from element `elements` on.
    Out offset(ulong elements) const { return Out(cells + elements); }
    // Moves `count` elements of `from`, which the owner stored, to `index` without decoding.
    void move_stored(ulong index, Out from, ulong from_index, ulong count) const {
        for (ulong i = 0; i < count; ++i)
            cells[index + i].stored = from.cells[from_index + i].stored;
    }
    // Copies `count` elements of `from`, already stored in E, to `index` without decoding.
    void copy_stored(ulong index, device const typename E::storage *from, ulong from_index, ulong count) const {
        for (ulong i = 0; i < count; ++i)
            cells[index + i].stored = from[from_index + i];
    }

private:
    device Cell<E> *cells;

    template <typename F> friend CoherentOut<F> coherent_device(Out<F> out);
    template <typename F> friend SystemOut<F> system(Out<F> out);
};

// Scratch at `scratch` (memory the kernel owns, not a tensor) as an `Out` of E: the one place scratch
// becomes an `Out`.
template <typename E> inline Out<E> out_over_scratch(device uchar *scratch) {
    return Out<E>(reinterpret_cast<device Cell<E> *>(scratch));
}

// N consecutive elements as F32; `from` is aligned to N elements. A multiple
// of four elements loads in vectors of four.
template <typename E, uint N>
inline void span(device const typename E::native *from, thread float (&x)[N]) {
    constexpr uint VECTORED = N % 4 == 0 ? N : 0;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < VECTORED; i += 4) {
        const float4 four = float4(*reinterpret_cast<device const vec<typename E::native, 4> *>(from + i));
        x[i] = four.x;
        x[i + 1] = four.y;
        x[i + 2] = four.z;
        x[i + 3] = four.w;
    }
    _Pragma("clang loop unroll(full)")
    for (uint i = VECTORED; i < N; ++i)
        x[i] = float(from[i]);
}

// The element type of a dense representation kind: 0 = f32, 1 = bf16,
// 2 = f16. Other kinds have none.
template <int KIND> struct Of;
template <> struct Of<0> { typedef F32 type; };
template <> struct Of<1> { typedef Bf16 type; };
template <> struct Of<2> { typedef F16 type; };

} // namespace element

#define ELEMENT_KIND(prefix)                                                                       \
    (ELEMENT_HAS(prefix, _REPRESENTATION_F32)    ? 0                                               \
     : ELEMENT_HAS(prefix, _REPRESENTATION_BF16) ? 1                                               \
     : ELEMENT_HAS(prefix, _REPRESENTATION_F16)  ? 2                                               \
                                                 : -1)
// The element type of the dense tensor or element parameter `prefix`.
#define ELEMENT_OF(prefix) typename element::Of<ELEMENT_KIND(prefix)>::type
