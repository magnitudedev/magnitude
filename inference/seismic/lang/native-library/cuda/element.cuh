// Seismic native library (CUDA): dense element types, over the CUDA prelude's
// conversions. `#include <seismic/element.cuh>`. Every type converts one
// stored element to F32 (`load`) and rounds F32 to the element (`store`,
// `round`, round-to-nearest-even). Tensors are addressed in elements of their
// own dtype from a byte base: `at`, `put`, `put2`, and `span` (N contiguous
// elements into registers).
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

#include <seismic/slab.cuh>

namespace element {

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

struct Bf16 {
    typedef u16 storage;
    static constexpr u32 bytes = 2;
    __device__ static __forceinline__ float load(storage value) { return seismic_bf16_to_f32(value); }
    __device__ static __forceinline__ storage store(float value) { return seismic_f32_to_bf16(value); }
    __device__ static __forceinline__ float round(float value) { return load(store(value)); }
    // Two elements in 32 bits, the first in the low half.
    __device__ static __forceinline__ u32 pack2(float first, float second) {
        return seismic_pack_bf16x2(first, second);
    }
    __device__ static __forceinline__ float2 unpack2(u32 bits) { return seismic_unpack_bf16x2(bits); }
};

struct F16 {
    typedef u16 storage;
    static constexpr u32 bytes = 2;
    __device__ static __forceinline__ float load(storage value) { return seismic_f16_to_f32(value); }
    __device__ static __forceinline__ storage store(float value) { return seismic_f32_to_f16(value); }
    __device__ static __forceinline__ float round(float value) { return load(store(value)); }
    __device__ static __forceinline__ u32 pack2(float first, float second) {
        return seismic_pack_f16x2(first, second);
    }
    __device__ static __forceinline__ float2 unpack2(u32 bits) { return seismic_unpack_f16x2(bits); }
};

struct F32 {
    typedef float storage;
    static constexpr u32 bytes = 4;
    __device__ static __forceinline__ float load(storage value) { return value; }
    __device__ static __forceinline__ storage store(float value) { return value; }
    __device__ static __forceinline__ float round(float value) { return value; }
};

// Element `index` of a dense tensor of element type E.
template <class E> __device__ __forceinline__ float at(const u8 *base, u64 index) {
    return E::load(reinterpret_cast<const typename E::storage *>(base)[index]);
}

template <class E> __device__ __forceinline__ void put(u8 *base, u64 index, float value) {
    reinterpret_cast<typename E::storage *>(base)[index] = E::store(value);
}

// Elements index and index + 1: one vector store when index is even.
template <class E> __device__ __forceinline__ void put2(u8 *base, u64 index, float first, float second) {
    if (index % 2 != 0) {
        put<E>(base, index, first);
        put<E>(base, index + 1, second);
    } else if constexpr (E::bytes == 4) {
        reinterpret_cast<float2 *>(base)[index / 2] = make_float2(first, second);
    } else {
        reinterpret_cast<u32 *>(base)[index / 2] = E::pack2(first, second);
    }
}

// The elements of E in 32 bits (one f32 or two 16-bit elements) into `out`.
template <class E> __device__ __forceinline__ void unpack(u32 bits, float *out) {
    if constexpr (E::bytes == 4) {
        out[0] = __uint_as_float(bits);
    } else {
        const float2 pair = E::unpack2(bits);
        out[0] = pair.x;
        out[1] = pair.y;
    }
}

// `N` contiguous elements starting at `index` into F32 registers, with the
// widest aligned vector loads the span size allows. The span start is aligned
// to the span's byte size (callers index whole spans of a canonical row).
// `NC` selects the read-only non-coherent path, admitted only for memory no
// thread writes during the launch.
template <class E, int N, bool NC>
__device__ __forceinline__ void span(const u8 *base, u64 index, float (&out)[N]) {
    constexpr int bytes = N * E::bytes;
    constexpr int per_word = 4 / E::bytes;
    const u8 *address = base + index * E::bytes;
    if constexpr (bytes % 16 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < bytes / 16; ++chunk) {
            uint4 word;
            if constexpr (NC) {
                word = seismic_ld_nc_v4(address + chunk * 16);
            } else {
                word = *reinterpret_cast<const uint4 *>(address + chunk * 16);
            }
            unpack<E>(word.x, out + chunk * 4 * per_word + 0 * per_word);
            unpack<E>(word.y, out + chunk * 4 * per_word + 1 * per_word);
            unpack<E>(word.z, out + chunk * 4 * per_word + 2 * per_word);
            unpack<E>(word.w, out + chunk * 4 * per_word + 3 * per_word);
        }
    } else if constexpr (bytes % 8 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < bytes / 8; ++chunk) {
            uint2 word;
            if constexpr (NC) {
                word = seismic_ld_nc_v2(address + chunk * 8);
            } else {
                word = *reinterpret_cast<const uint2 *>(address + chunk * 8);
            }
            unpack<E>(word.x, out + chunk * 2 * per_word + 0 * per_word);
            unpack<E>(word.y, out + chunk * 2 * per_word + 1 * per_word);
        }
    } else if constexpr (bytes % 4 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < bytes / 4; ++chunk) {
            const u32 word = NC ? seismic_ld_nc_u32(address + chunk * 4)
                                : *reinterpret_cast<const u32 *>(address + chunk * 4);
            unpack<E>(word, out + chunk * per_word);
        }
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = at<E>(base, index + i);
        }
    }
}

// `N` contiguous F32 values (global or shared) into registers and back, in
// 16-byte vectors when N % 4 == 0.
template <int N> __device__ __forceinline__ void f32_span(const float *base, float (&out)[N]) {
    if constexpr (N % 4 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < N / 4; ++chunk) {
            const float4 word = reinterpret_cast<const float4 *>(base)[chunk];
            out[chunk * 4 + 0] = word.x;
            out[chunk * 4 + 1] = word.y;
            out[chunk * 4 + 2] = word.z;
            out[chunk * 4 + 3] = word.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = base[i];
        }
    }
}

template <int N> __device__ __forceinline__ void f32_span_store(float *base, const float (&values)[N]) {
    if constexpr (N % 4 == 0) {
#pragma unroll
        for (int chunk = 0; chunk < N / 4; ++chunk) {
            reinterpret_cast<float4 *>(base)[chunk] =
                make_float4(values[chunk * 4], values[chunk * 4 + 1], values[chunk * 4 + 2], values[chunk * 4 + 3]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            base[i] = values[i];
        }
    }
}

template <class E> struct StoredValue;
template <class E> struct Stored;
template <class E> struct Out;
template <class E> struct SlabOut;
template <class E> __device__ __forceinline__ Stored<E> stored(const u8 *base, u64 index = 0);
template <class E> __device__ __forceinline__ Out<E> out_over_scratch(u8 *scratch);
template <class E> __device__ __forceinline__ Out<E> abi_out(u8 *address);
template <class E> __device__ __forceinline__ SlabOut<E> abi_slab_out(const u8 *address);

// One element of E already stored by an owner, held in a register for
// pipelined bit-exact publication into an `Out<E>` of the same element.
template <class E> struct StoredValue {
    __device__ __forceinline__ StoredValue() : bits{} {}

private:
    typename E::storage bits;
    __device__ __forceinline__ explicit StoredValue(typename E::storage raw) : bits(raw) {}
    friend struct Stored<E>;
    friend struct Out<E>;
};

// Read-only view of elements of E already stored by an owner (an input tensor
// of E, a slab view of E, or an `Out<E>`): readable as F32 (`at`, `span`) and
// movable bit-for-bit only into an `Out<E>` of the same element.
template <class E> struct Stored {
    __device__ __forceinline__ explicit Stored(decltype(nullptr)) : data(nullptr) {}
    __device__ __forceinline__ bool present() const { return data != nullptr; }
    __device__ __forceinline__ float at(u64 index = 0) const { return E::load(data[index]); }
    template <int N, bool NC = false>
    __device__ __forceinline__ void span(u64 index, float (&values)[N]) const {
        element::span<E, N, NC>(reinterpret_cast<const u8 *>(data), index, values);
    }
    __device__ __forceinline__ StoredValue<E> item(u64 index = 0) const { return StoredValue<E>(data[index]); }
    __device__ __forceinline__ const u8 *bytes(u64 index = 0) const {
        return reinterpret_cast<const u8 *>(data + index);
    }
    __device__ __forceinline__ Stored offset(u64 elements) const { return Stored(data + elements); }

private:
    const typename E::storage *data;
    __device__ __forceinline__ explicit Stored(const typename E::storage *elements) : data(elements) {}
    friend struct Out<E>;
    friend struct SlabOut<E>;
    template <class F> __device__ friend __forceinline__ Stored<F> stored(const u8 *base, u64 index);
};

template <class E> __device__ __forceinline__ Stored<E> stored(const u8 *base, u64 index) {
    return Stored<E>(reinterpret_cast<const typename E::storage *>(base) + index);
}

// A floating result, or a floating tensor a kernel mutates, as the kernel writes it: a handle whose
// only writes are owner stores, so every element it publishes is an element of E. The tensor's raw
// address never reaches the kernel (the ABI defines no buffer index for it); the kernel gets
// `SEISMIC_RESULT_<n>_OUT` / `SEISMIC_OUT_<name>`.
template <class E> struct Out {
    __device__ __forceinline__ explicit Out(decltype(nullptr)) : base(nullptr) {}
    __device__ __forceinline__ bool present() const { return base != nullptr; }

    // Element `index`, as F32.
    __device__ __forceinline__ float at(u64 index) const {
        return E::load(reinterpret_cast<const typename E::storage *>(base)[index]);
    }
    // Stores `value` rounded to E.
    __device__ __forceinline__ void put(u64 index, float value) const { element::put<E>(base, index, value); }
    // Stores elements index and index + 1 (one vector store when index is even).
    __device__ __forceinline__ void put2(u64 index, float first, float second) const {
        element::put2<E>(base, index, first, second);
    }
    // Stores `N` contiguous values from `index` (vector stores when the span allows).
    template <int N> __device__ __forceinline__ void put_span(u64 index, const float (&values)[N]) const {
        if constexpr (E::bytes == 4 && N % 4 == 0) {
            f32_span_store<N>(reinterpret_cast<float *>(base) + index, values);
            return;
        }
        if constexpr (E::bytes == 2 && N % 8 == 0) {
            u8 *address = base + index * E::bytes;
            if ((reinterpret_cast<u64>(address) & 15u) == 0) {
#pragma unroll
                for (int chunk = 0; chunk < N / 8; ++chunk) {
                    reinterpret_cast<uint4 *>(address)[chunk] =
                        make_uint4(E::pack2(values[chunk * 8], values[chunk * 8 + 1]),
                                   E::pack2(values[chunk * 8 + 2], values[chunk * 8 + 3]),
                                   E::pack2(values[chunk * 8 + 4], values[chunk * 8 + 5]),
                                   E::pack2(values[chunk * 8 + 6], values[chunk * 8 + 7]));
                }
                return;
            }
        }
        if constexpr (E::bytes == 2 && N % 4 == 0) {
            u8 *address = base + index * E::bytes;
            if ((reinterpret_cast<u64>(address) & 7u) == 0) {
#pragma unroll
                for (int chunk = 0; chunk < N / 4; ++chunk) {
                    reinterpret_cast<uint2 *>(address)[chunk] =
                        make_uint2(E::pack2(values[chunk * 4], values[chunk * 4 + 1]),
                                   E::pack2(values[chunk * 4 + 2], values[chunk * 4 + 3]));
                }
                return;
            }
        }
#pragma unroll
        for (int i = 0; i < N; i += 2) {
            if (i + 1 < N) {
                put2(index + i, values[i], values[i + 1]);
            } else {
                put(index + i, values[i]);
            }
        }
    }
    // Stores `value` rounded to E with a volatile store, for memory the host maps and reads while the kernel runs.
    __device__ __forceinline__ void put_system(u64 index, float value) const {
        *(reinterpret_cast<volatile typename E::storage *>(base) + index) = E::store(value);
    }
    // Stores one element of E already held in `value` without decoding.
    __device__ __forceinline__ void put_stored(u64 index, StoredValue<E> value) const {
        reinterpret_cast<typename E::storage *>(base)[index] = value.bits;
    }
    // Elements `index` on, as F32 registers (`span`'s contract: `N` contiguous, aligned to the span).
    template <int N, bool NC = false>
    __device__ __forceinline__ void span(u64 index, float (&values)[N]) const {
        element::span<E, N, NC>(base, index, values);
    }
    // The tensor from element `index` on, read-only, for vector, async-copy and staged loads of owner-written elements.
    __device__ __forceinline__ const u8 *stored(u64 index = 0) const { return base + index * E::bytes; }
    // Read-only typed view from element `index` on.
    __device__ __forceinline__ Stored<E> view(u64 index = 0) const {
        return Stored<E>(reinterpret_cast<const typename E::storage *>(base) + index);
    }
    // The view from element `elements` on.
    __device__ __forceinline__ Out offset(u64 elements) const { return Out(base + elements * E::bytes); }
    // Moves `count` elements of `from`, which an owner stored, to `index` without decoding: the only
    // span bit move into a result.
    __device__ __forceinline__ void move_stored(u64 index, Stored<E> from, u64 count = 1) const {
        typename E::storage *to = reinterpret_cast<typename E::storage *>(base) + index;
        const typename E::storage *source = from.data;
        const u64 bytes = count * E::bytes;
        if (((reinterpret_cast<u64>(to) | reinterpret_cast<u64>(source) | bytes) & 15u) == 0) {
            for (u64 i = 0; i < bytes / 16; ++i) {
                reinterpret_cast<uint4 *>(to)[i] = reinterpret_cast<const uint4 *>(source)[i];
            }
            return;
        }
        for (u64 i = 0; i < count; ++i) {
            to[i] = source[i];
        }
    }
    __device__ __forceinline__ void move_stored(u64 index, const Out &from, u64 from_index, u64 count) const {
        move_stored(index, from.view(from_index), count);
    }

private:
    u8 *base;
    __device__ __forceinline__ explicit Out(u8 *address) : base(address) {}
    friend struct SlabOut<E>;
    template <class F> __device__ friend __forceinline__ Out<F> out_over_scratch(u8 *scratch);
    template <class F> __device__ friend __forceinline__ Out<F> abi_out(u8 *address);
};

// A floating tensor stored in a slab (`&slab mut`) as the kernel writes it: the kernel gets the slab
// address table (`SEISMIC_OUT_<NAME>`), and reaches a writable region or row only as an `Out`.
template <class E> struct SlabOut {
    // The region of slab `slab_index`.
    __device__ __forceinline__ Out<E> region(u64 slab_index) const { return Out<E>(slab::region(table, slab_index)); }
    // Row `row_index` of rows of `row_elements` elements, `rows_per_slab` to a slab.
    __device__ __forceinline__ Out<E> row(u64 row_index, u64 rows_per_slab, u64 row_elements) const {
        return Out<E>(slab::row(table, row_index, rows_per_slab, row_elements * E::bytes));
    }
    // The slab address table for read-only addressing (`slab::region`, `slab::row`).
    __device__ __forceinline__ const u32 *reads() const { return table; }
    // Read-only view of slab `slab_index`.
    __device__ __forceinline__ Stored<E> view_region(u64 slab_index) const {
        return Stored<E>(reinterpret_cast<const typename E::storage *>(slab::region(table, slab_index)));
    }
    // Read-only view of row `row_index`.
    __device__ __forceinline__ Stored<E> view_row(u64 row_index, u64 rows_per_slab, u64 row_elements) const {
        return Stored<E>(
            reinterpret_cast<const typename E::storage *>(slab::row(table, row_index, rows_per_slab, row_elements * E::bytes)));
    }

private:
    const u32 *table;
    __device__ __forceinline__ explicit SlabOut(const u8 *address) : table(reinterpret_cast<const u32 *>(address)) {}
    template <class F> __device__ friend __forceinline__ SlabOut<F> abi_slab_out(const u8 *address);
};

// An `Out` over a launch's scratch buffer, for multi-launch staging between a producer and consumer.
template <class E> __device__ __forceinline__ Out<E> out_over_scratch(u8 *scratch) { return Out<E>(scratch); }

// Generated ABI entrypoints for `SEISMIC_RESULT_<n>_OUT` and `SEISMIC_OUT_<NAME>`; reserved outside ABI headers.
template <class E> __device__ __forceinline__ Out<E> abi_out(u8 *address) { return Out<E>(address); }
template <class E> __device__ __forceinline__ SlabOut<E> abi_slab_out(const u8 *address) { return SlabOut<E>(address); }

// An F32 scalar argument from its argument word.
__device__ __forceinline__ float word_f32(u64 word) { return __uint_as_float(static_cast<u32>(word)); }

// The element type of a dense representation kind: 0 = f32, 1 = bf16,
// 2 = f16. Other kinds have none.
template <int KIND> struct Of {
    static_assert(KIND >= 0, "a dense f32, bf16 or f16 representation is required");
};
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
