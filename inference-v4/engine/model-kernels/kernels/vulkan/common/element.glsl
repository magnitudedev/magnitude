// Dense element types of the Vulkan entries: activations, norm vectors, dense
// weights and state, over the Seismic Vulkan prelude's conversions. The
// counterpart of `metal/common/element.h` and `cuda/common/element.cuh`.
//
// GLSL has no templates, so an element type is a constant kind (`ELEMENT_F32`,
// `ELEMENT_BF16`, `ELEMENT_F16`) passed as the first argument of every
// operation. Kinds are compile-time constants, so the driver folds every kind
// switch away. Every kind converts one stored element to F32
// (`element_load`) and rounds F32 to the element (`element_store`,
// `element_round`, round to nearest even; bf16 in integer code).
//
// glslang's preprocessor has no variadic macros and does not evaluate token
// pasting inside `#if`, so the Metal/CUDA `ELEMENT_HAS` probe cannot exist
// here: a kind is read from the ABI prefix only by a `defined(...)` ladder
// over the full prefix name, and the build admits only the including
// entry's ABI names, so a ladder lives where its element exists:
// `common/activation.glsl` gives `ELEMENT_ACT` (the activation element A) to
// entries that bind A; an entry reads the kind of any other dense element or
// tensor with its own ladder:
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

layout(buffer_reference, scalar, buffer_reference_align = 1) readonly buffer element_ro_u8 { uint8_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 1) readonly buffer element_ro_i8 { int8_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 2) readonly buffer element_ro_u16 { uint16_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer element_ro_u32 { uint v; };
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer element_ro_uvec2 { uvec2 v; };
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer element_ro_uvec4 { uvec4 v; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer element_ro_f32 { float v; };
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer element_ro_vec4 { vec4 v; };

uint element_u8_at(uint64_t address) { return uint(element_ro_u8(address).v); }
int element_i8_at(uint64_t address) { return int(element_ro_i8(address).v); }
uint element_u16_at(uint64_t address) { return uint(element_ro_u16(address).v); }
uint element_u32_at(uint64_t address) { return element_ro_u32(address).v; }
int element_i32_at(uint64_t address) { return int(element_ro_u32(address).v); }
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

float element_round(const int kind, float value) {
    return kind == ELEMENT_F32 ? value : element_load(kind, element_store(kind, value));
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
        const vec2 a = unpackHalf2x16(words.x), b = unpackHalf2x16(words.y);
        const vec2 c = unpackHalf2x16(words.z), d = unpackHalf2x16(words.w);
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

// Four contiguous elements from `index`, aligned to four elements.
vec4 element_load4(const int kind, uint64_t base, uint64_t index) {
    if (kind == ELEMENT_F32)
        return element_vec4_at(base + index * 4ul);
    const uvec2 words = element_uvec2_at(base + index * 2ul);
    return vec4(element_unpack2(kind, words.x), element_unpack2(kind, words.y));
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
