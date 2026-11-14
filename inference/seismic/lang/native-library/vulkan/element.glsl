// Seismic native library (Vulkan): dense element types, over the Vulkan
// prelude's conversions. `#include <seismic/element.glsl>`. The counterpart of
// <seismic/element.h> (Metal) and <seismic/element.cuh> (CUDA).
//
// GLSL has no templates, so an element type is a constant kind (`ELEMENT_F32`,
// `ELEMENT_BF16`, `ELEMENT_F16`) passed as the first argument of every
// operation. Kinds are compile-time constants, so the driver folds every kind
// switch away. Every kind converts one stored element to F32
// (`element_load`) and rounds F32 to the element (`element_store`,
// `element_round`, round to nearest even; bf16, and `element_round`'s f16
// decode, in integer code).
//
// glslang's preprocessor has no variadic macros and does not evaluate token
// pasting inside `#if`, so the Metal/CUDA `ELEMENT_HAS` probe cannot exist
// here: a kind is read from the ABI prefix only by a `defined(...)` ladder
// over the full prefix name, and the build admits only the including
// entry's ABI names, so a ladder lives where its element exists, in the
// kernel library that names the element; an entry reads the kind of any other
// dense element or tensor with its own ladder:
//     #if defined(SEISMIC_ELEMENT_E_REPRESENTATION_F32)
//     #define IMPORT_SOURCE_KIND ELEMENT_F32
//     #elif ...
//
// Memory is addressed by 64-bit byte addresses (`SEISMIC_PTR(...)` plus
// offsets). `element_at`/`element_put` index a tensor in elements of its own
// kind; the raw accessors (`element_u32_at`, `element_uvec4_at`, ...) load and
// store whole words at a byte address aligned to the word.

#define ELEMENT_F32 0
#define ELEMENT_BF16 1
#define ELEMENT_F16 2

// Bytes of one element of a kind.
#define ELEMENT_BYTES(kind) ((kind) == ELEMENT_F32 ? 4u : 2u)

// ---------------------------------------------------------------------------
// Raw words at byte addresses. Reads go through `readonly` references.
//
// Memory is exchanged between invocations of a dispatch only through group
// memory or subgroup operations, and between dispatches through the dispatch
// boundary: never by one invocation putting a word that another invocation
// of the same dispatch then reads here. The reads are `readonly` and the
// puts plain, and a Vulkan source enables no memory model, so a
// `memoryBarrierBuffer(); barrier();` pair orders only `coherent` accesses
// and the driver is free to serve a read from a path that never sees the
// put, or to move it across the barrier: an exchange that way is exact on
// one driver and loses words on another (GB10's `candidate_rows`).

layout(buffer_reference, scalar, buffer_reference_align = 1) readonly buffer element_ro_u8 { uint8_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 1) readonly buffer element_ro_i8 { int8_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 2) readonly buffer element_ro_u16 { uint16_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer element_ro_u32 { uint v; };
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer element_ro_u64 { uint64_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer element_ro_uvec2 { uvec2 v; };
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer element_ro_uvec4 { uvec4 v; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer element_ro_f32 { float v; };
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer element_ro_vec4 { vec4 v; };

uint element_u8_at(uint64_t address) { return uint(element_ro_u8(address).v); }
int element_i8_at(uint64_t address) { return int(element_ro_i8(address).v); }
uint element_u16_at(uint64_t address) { return uint(element_ro_u16(address).v); }
uint element_u32_at(uint64_t address) { return element_ro_u32(address).v; }
int element_i32_at(uint64_t address) { return int(element_ro_u32(address).v); }
uint64_t element_u64_at(uint64_t address) { return element_ro_u64(address).v; }
uvec2 element_uvec2_at(uint64_t address) { return element_ro_uvec2(address).v; }
uvec4 element_uvec4_at(uint64_t address) { return element_ro_uvec4(address).v; }
float element_f32_at(uint64_t address) { return element_ro_f32(address).v; }
vec4 element_vec4_at(uint64_t address) { return element_ro_vec4(address).v; }

void element_u8_put(uint64_t address, uint value) { seismic_u8(address).v = uint8_t(value); }
void element_u16_put(uint64_t address, uint value) { seismic_u16(address).v = uint16_t(value); }
void element_u32_put(uint64_t address, uint value) { seismic_u32(address).v = value; }
void element_i32_put(uint64_t address, int value) { seismic_i32(address).v = value; }
void element_f32_put(uint64_t address, float value) { seismic_f32(address).v = value; }
void element_uvec4_put(uint64_t address, uvec4 value) { seismic_uvec4(address).v = value; }
void element_uvec2_put(uint64_t address, uvec2 value) { seismic_uvec2(address).v = value; }

// Element `index` of an i32 tensor (index maps, token ids).
int element_i32_index(uint64_t base, uint64_t index) { return element_i32_at(base + index * 4ul); }

// An F32 scalar argument from its argument word.
float element_word_f32(uint64_t word) { return uintBitsToFloat(uint(word)); }

// ---------------------------------------------------------------------------
// Conversions of one kind.

float element_load(const int kind, uint bits) {
    if (kind == ELEMENT_F32)
        return uintBitsToFloat(bits);
    if (kind == ELEMENT_BF16)
        return seismic_bf16_to_f32(uint16_t(bits));
    return seismic_f16_to_f32(uint16_t(bits));
}

// The stored bits of `value` (16-bit kinds in the low half).
uint element_store(const int kind, float value) {
    if (kind == ELEMENT_F32)
        return floatBitsToUint(value);
    if (kind == ELEMENT_BF16)
        return uint(seismic_f32_to_bf16(value));
    return uint(seismic_f32_to_f16(value));
}

// `value` rounded to the element, as F32. The f16 rounding decodes its bits
// in exact integer code (`seismic_f16_decode`): a hardware decode of a
// hardware encode is a conversion pair the driver may fold away.
float element_round(const int kind, float value) {
    if (kind == ELEMENT_F32)
        return value;
    if (kind == ELEMENT_BF16)
        return seismic_bf16_to_f32(seismic_f32_to_bf16(value));
    return seismic_f16_decode(seismic_f32_to_f16(value));
}

// Two 16-bit elements in 32 bits, the first in the low half.
uint element_pack2(const int kind, float first, float second) {
    return kind == ELEMENT_BF16 ? seismic_pack_bf16x2(first, second) : seismic_pack_f16x2(first, second);
}

vec2 element_unpack2(const int kind, uint bits) {
    return kind == ELEMENT_BF16 ? seismic_unpack_bf16x2(bits) : seismic_unpack_f16x2(bits);
}

// Eight consecutive 16-bit elements packed in a uvec4 (element 2i in the low
// half of word i) as (0,2,4,6) and (1,3,5,7).
void element_split8(const int kind, uvec4 words, out vec4 even, out vec4 odd) {
    if (kind == ELEMENT_BF16) {
        even = uintBitsToFloat(words << 16);
        odd = uintBitsToFloat(words & 0xffff0000u);
    } else {
        const vec2 a = seismic_unpack_f16x2(words.x), b = seismic_unpack_f16x2(words.y);
        const vec2 c = seismic_unpack_f16x2(words.z), d = seismic_unpack_f16x2(words.w);
        even = vec4(a.x, b.x, c.x, d.x);
        odd = vec4(a.y, b.y, c.y, d.y);
    }
}

// The inverse of `element_split8`, rounding each value to the kind.
uvec4 element_pack8(const int kind, vec4 even, vec4 odd) {
    return uvec4(element_pack2(kind, even.x, odd.x), element_pack2(kind, even.y, odd.y),
        element_pack2(kind, even.z, odd.z), element_pack2(kind, even.w, odd.w));
}

// ---------------------------------------------------------------------------
// Tensors of one kind, indexed in elements from a byte base.

float element_at(const int kind, uint64_t base, uint64_t index) {
    if (kind == ELEMENT_F32)
        return element_f32_at(base + index * 4ul);
    return element_load(kind, element_u16_at(base + index * 2ul));
}

void element_put(const int kind, uint64_t base, uint64_t index, float value) {
    if (kind == ELEMENT_F32)
        element_f32_put(base + index * 4ul, value);
    else
        element_u16_put(base + index * 2ul, element_store(kind, value));
}

// Elements index and index + 1: one 32-bit store for an even index of a
// 16-bit kind.
void element_put2(const int kind, uint64_t base, uint64_t index, float first, float second) {
    if (kind != ELEMENT_F32 && index % 2ul == 0ul) {
        element_u32_put(base + index * 2ul, element_pack2(kind, first, second));
    } else {
        element_put(kind, base, index, first);
        element_put(kind, base, index + 1ul, second);
    }
}

// Eight contiguous elements from `index`, aligned to eight elements, as
// (even, odd).
void element_load8(const int kind, uint64_t base, uint64_t index, out vec4 even, out vec4 odd) {
    if (kind == ELEMENT_F32) {
        const vec4 a = element_vec4_at(base + index * 4ul);
        const vec4 b = element_vec4_at(base + index * 4ul + 16ul);
        even = vec4(a.x, a.z, b.x, b.z);
        odd = vec4(a.y, a.w, b.y, b.w);
    } else {
        element_split8(kind, element_uvec4_at(base + index * 2ul), even, odd);
    }
}

// ---------------------------------------------------------------------------
// A floating result, or a floating tensor a kernel mutates, as the kernel writes it: a handle whose
// only writes are the owner's, so every element it publishes is an element of its kind. The kernel
// gets `SEISMIC_RESULT_<n>_OUT` / `SEISMIC_OUT_<NAME>` from the ABI (the tensor's address and its
// own element kind; no buffer index is defined for it), or an `Out` over scratch from
// `out_over_scratch`. The fields are the owner's: no kernel reads them or
// builds an `Out` itself (the build rejects an authored source that does).
struct Out {
    uint64_t seismic_base;
    int seismic_kind;
};

// The one place a scratch buffer (a projection's partials, a routed product) becomes an `Out`.
Out out_over_scratch(const int kind, uint64_t scratch_address) { return Out(scratch_address, kind); }

float out_at(Out handle, uint64_t index) { return element_at(handle.seismic_kind, handle.seismic_base, index); }
void out_put(Out handle, uint64_t index, float value) {
    element_put(handle.seismic_kind, handle.seismic_base, index, value);
}
void out_put2(Out handle, uint64_t index, float first, float second) {
    element_put2(handle.seismic_kind, handle.seismic_base, index, first, second);
}
// The view from element `elements` on.
Out out_offset(Out handle, uint64_t elements) {
    return Out(handle.seismic_base + elements * uint64_t(ELEMENT_BYTES(handle.seismic_kind)), handle.seismic_kind);
}
// Eight consecutive elements from `index`, aligned to eight, as (0,2,4,6) and (1,3,5,7).
void out_put8(Out handle, uint64_t index, vec4 even, vec4 odd) {
    if (handle.seismic_kind == ELEMENT_F32) {
        const uint64_t at = handle.seismic_base + index * 4ul;
        element_uvec4_put(at, floatBitsToUint(vec4(even.x, odd.x, even.y, odd.y)));
        element_uvec4_put(at + 16ul, floatBitsToUint(vec4(even.z, odd.z, even.w, odd.w)));
    } else {
        element_uvec4_put(handle.seismic_base + index * 2ul, element_pack8(handle.seismic_kind, even, odd));
    }
}
// Moves `count` elements of `from`, which the owner stored, to `index` without decoding.
void out_move_stored(Out to, uint64_t index, Out from, uint64_t from_index, uint64_t count) {
    for (uint64_t i = 0ul; i < count; ++i) {
        if (to.seismic_kind == ELEMENT_F32)
            element_u32_put(to.seismic_base + (index + i) * 4ul,
                element_u32_at(from.seismic_base + (from_index + i) * 4ul));
        else
            element_u16_put(to.seismic_base + (index + i) * 2ul,
                element_u16_at(from.seismic_base + (from_index + i) * 2ul));
    }
}
// Copies `count` stored elements of `to`'s kind from `source_base` (from element `source_index`) to
// `index` without decoding, preserving subnormals and NaN payloads bit-for-bit.
void out_copy_stored(Out to, uint64_t index, uint64_t source_base, uint64_t source_index, uint64_t count) {
    for (uint64_t i = 0ul; i < count; ++i) {
        if (to.seismic_kind == ELEMENT_F32)
            element_u32_put(to.seismic_base + (index + i) * 4ul,
                element_u32_at(source_base + (source_index + i) * 4ul));
        else
            element_u16_put(to.seismic_base + (index + i) * 2ul,
                element_u16_at(source_base + (source_index + i) * 2ul));
    }
}

// ---------------------------------------------------------------------------
// A read-only view over stored elements of a known dense kind (`out_stored` of an `Out` or
// `SlabOut` region, or `stored_at_buffer` over an input buffer). The fields are the library's: no
// kernel reads them or builds a `Stored` itself (the build rejects an authored source that does).
struct Stored {
    uint64_t seismic_base;
    int seismic_kind;
};

// The read-only view from element `index` of `handle`, for loads of elements the owner stored.
Stored out_stored(Out handle, uint64_t index) {
    return Stored(handle.seismic_base + index * uint64_t(ELEMENT_BYTES(handle.seismic_kind)), handle.seismic_kind);
}
// The read-only view of an input buffer of `kind` at byte address `base`.
Stored stored_at_buffer(const int kind, uint64_t base) {
    return Stored(base, kind);
}
// The view from element `elements` on.
Stored stored_offset(Stored view, uint64_t elements) {
    return Stored(view.seismic_base + elements * uint64_t(ELEMENT_BYTES(view.seismic_kind)), view.seismic_kind);
}
// Whether `view` starts on a 16-byte boundary.
bool stored_aligned16(Stored view) { return (view.seismic_base & 15ul) == 0ul; }
float stored_at(Stored view, uint64_t index) {
    return element_at(view.seismic_kind, view.seismic_base, index);
}
// Elements `index` and `index + 1`: one 32-bit load for an even index of a 16-bit kind.
vec2 stored_at2(Stored view, uint64_t index) {
    if (view.seismic_kind != ELEMENT_F32 && index % 2ul == 0ul)
        return element_unpack2(view.seismic_kind, element_u32_at(view.seismic_base + index * 2ul));
    return vec2(stored_at(view, index), stored_at(view, index + 1ul));
}
// Eight contiguous elements from `index`, aligned to eight elements, as (even, odd).
void stored_load8(Stored view, uint64_t index, out vec4 even, out vec4 odd) {
    element_load8(view.seismic_kind, view.seismic_base, index, even, odd);
}
// Raw 32-bit word `index` (from byte offset `index * 4`) of `view`.
uint stored_u32_at(Stored view, uint64_t index) {
    return element_u32_at(view.seismic_base + index * 4ul);
}
// Raw 16-byte word `index` (from byte offset `index * 16`) of `view`.
uvec4 stored_uvec4_at(Stored view, uint64_t index) {
    return element_uvec4_at(view.seismic_base + index * 16ul);
}
// Four contiguous F32 elements from element `index` (aligned to 16 bytes) of an F32 `view`.
vec4 stored_vec4_at(Stored view, uint64_t index) {
    return element_vec4_at(view.seismic_base + index * 4ul);
}
