// Seismic native library (Metal and CUDA): the word form of a registered
// conversion (`seismic_lang::repack::WordTable`, whose docs give the format),
// as a kernel reads it. `#include <seismic/repack.h>` (Metal) or
// `#include <seismic/repack.cuh>` (CUDA): the dialect's prelude
// (`metal/repack.h`, `cuda/repack.cuh`), then this text. A prelude defines
// SEISMIC_REPACK_FUNCTION, the qualifiers of a function of table words,
// which kernel code evaluates at compile time or at run time;
// SEISMIC_REPACK_LOAD(address), the read of a table word, the one every
// function here reads through (`table_word`), valid in either (but for a
// loop's displacements, `Looped`, read by plain loads);
// SEISMIC_REPACK_TABLE, a pointer to table words; SEISMIC_REPACK_INLINE,
// the qualifiers of a function only run; SEISMIC_REPACK_PRIVATE, the address
// space of a kernel's own objects; SEISMIC_REPACK_STATIC, those of a static
// constant member; SEISMIC_REPACK_LOOP, the attribute of a loop its compiler
// unrolls (a group's of more than one member, `Group`); and the value ops'
// float arithmetic
// (`f32_multiply`, `f16_to_f32`, `i32_to_f32`), the only arithmetic the
// dialects spell differently.
//
// A kernel names its conversion's table and span once
// (`SEISMIC_CONVERSION_<FROM>_<TO>_*`), on Metal:
//     constexpr constant uint conversion[] = {..._TABLE};
//     constexpr constant repack::WordForm form = {conversion, ..._STACK,
//         ..._DIRECTORY, ..._CLASSES};
// (on CUDA, `__device__ constexpr` for `constexpr constant`), and reads every
// word through these functions (`table_word` for a whole word), which are
// the only place the kernel decodes the table's bits. It executes a unit's
// record words through `unit_record` and a fixed block's through
// `fixed_record`, reading source packets through its own readers. Op codes
// are the ABI's `SEISMIC_REPACK_OP_<NAME>`.

namespace repack {

// A run of `length` (1..32) bits as a mask.
SEISMIC_REPACK_FUNCTION uint low_bits(uint length) { return 0xffffffffu >> (32u - length); }

// ---- pieces (`Piece`) ------------------------------------------------------

// The first source bit of a run, or a value's record.
SEISMIC_REPACK_FUNCTION uint piece_first(uint piece) { return piece & 0xffffu; }
SEISMIC_REPACK_FUNCTION uint piece_shift(uint piece) { return (piece >> 16) & 31u; }
SEISMIC_REPACK_FUNCTION uint piece_length(uint piece) { return ((piece >> 21) & 31u) + 1u; }
// Whether the piece places bits of a value (from `piece_value_bit`) rather
// than a run of source bits.
SEISMIC_REPACK_FUNCTION bool piece_is_value(uint piece) { return (piece >> 31) != 0; }
SEISMIC_REPACK_FUNCTION uint piece_value_bit(uint piece) { return (piece >> 26) & 31u; }

// ---- runs of pieces (a value's ops, a field's pieces) ----------------------

SEISMIC_REPACK_FUNCTION uint run_first(uint run) { return run & 0xffffu; }
SEISMIC_REPACK_FUNCTION uint run_count(uint run) { return run >> 16; }

// ---- ops (`WordStep`) ------------------------------------------------------

SEISMIC_REPACK_FUNCTION uint op_code(uint op) { return op & 15u; }
SEISMIC_REPACK_FUNCTION uint op_shift_bits(uint op) { return op >> 4; }
SEISMIC_REPACK_FUNCTION int op_exponent(uint op) { return int(op) >> 4; }
// A lookup's first table word, and log2 of its entry bytes.
SEISMIC_REPACK_FUNCTION uint op_lookup_first(uint op) { return (op >> 4) & 0xfffffu; }
SEISMIC_REPACK_FUNCTION uint op_lookup_shift(uint op) { return op >> 24; }

// A ScaleF32 op's factor 2^exponent, as the bits of a normal float.
SEISMIC_REPACK_FUNCTION uint op_scale_bits(uint op) { return uint(127 + op_exponent(op)) << 23; }

// ---- a span's slot map entries (`SlotEntry`) -------------------------------

SEISMIC_REPACK_FUNCTION uint slot_of(uint entry) { return entry & 0x3fffffffu; }
// The slot moves its runs by whole bytes in every unit of the class.
SEISMIC_REPACK_FUNCTION bool slot_whole(uint entry) { return (entry >> 31) != 0; }
// It moves them nowhere. (A class's units move every member of a run
// within the bytes it spans.)
SEISMIC_REPACK_FUNCTION bool slot_still(uint entry) { return ((entry >> 30) & 1u) != 0; }

// ---- a conversion's word form ----------------------------------------------

// The table, and its span's directory and class list (`WordSpan`).
struct WordForm {
    SEISMIC_REPACK_TABLE table;
    // The most words a value's ops hold at once (0 without values, when
    // no value is evaluated).
    uint stack;
    uint directory, classes;
};

// Table word `at`.
SEISMIC_REPACK_FUNCTION uint table_word(WordForm form, uint at) { return SEISMIC_REPACK_LOAD(&form.table[at]); }

// A unit's groups (their first table word, with their count `<< 16`; four
// table words each, `group_*`): unit `unit` is block `unit`, its header the
// table's word `unit`.
SEISMIC_REPACK_FUNCTION uint unit_groups(WordForm form, uint unit) { return table_word(form, unit); }

// Fixed block `f`'s groups and record words (the first, with their count
// `<< 16`), after a span of `units` units' headers.
SEISMIC_REPACK_FUNCTION uint fixed_groups(WordForm form, uint units, uint f) { return table_word(form, units + 2 * f); }
SEISMIC_REPACK_FUNCTION uint fixed_words(WordForm form, uint units, uint f) {
    return table_word(form, units + 2 * f + 1);
}

// The fixed block whose first record word is `word`.
SEISMIC_REPACK_FUNCTION uint fixed_block(WordForm form, uint units, uint word) {
    uint f = 0;
    while (run_first(fixed_words(form, units, f)) != word) ++f;
    return f;
}

// The group at table word `at`: its representative's word in the block, its
// members, and its members' displacements of the parameters that vary (bit p
// of `group_varying` for parameter p: 0 the word, 1 the shift, 2 + i the
// representative's i-th run or offset in op order): from table word
// `group_list`, each member's row, one word per varying parameter in order.
SEISMIC_REPACK_FUNCTION uint group_word(WordForm form, uint at) { return table_word(form, at + 1) & 0xffu; }
SEISMIC_REPACK_FUNCTION uint group_count(WordForm form, uint at) { return table_word(form, at + 1) >> 8; }
SEISMIC_REPACK_FUNCTION uint group_list(WordForm form, uint at) { return table_word(form, at + 2); }
SEISMIC_REPACK_FUNCTION uint group_varying(WordForm form, uint at) { return table_word(form, at + 3); }

SEISMIC_REPACK_FUNCTION uint bits_set(uint bits) {
    uint count = 0;
    for (; bits != 0; bits &= bits - 1) ++count;
    return count;
}

// Whether the members' displacements of parameter `p` vary; the words of a
// member's row; the table word of member 0's displacement of `p` (member
// `m`'s is `m` rows on).
SEISMIC_REPACK_FUNCTION bool varies(WordForm form, uint at, uint p) {
    return ((group_varying(form, at) >> p) & 1u) != 0;
}
SEISMIC_REPACK_FUNCTION uint row_words(WordForm form, uint at) { return bits_set(group_varying(form, at)); }
SEISMIC_REPACK_FUNCTION uint displacement_at(WordForm form, uint at, uint p) {
    return group_list(form, at) + bits_set(group_varying(form, at) & ((1u << p) - 1u));
}

// Entry `index` of the table of Lookup op `op`: signed, of 2^shift bytes,
// little-endian from the table's word `op_lookup_first(op)`.
SEISMIC_REPACK_FUNCTION uint lookup_entry(WordForm form, uint op, uint index) {
    const uint bits = 8u << op_lookup_shift(op), byte = index << op_lookup_shift(op);
    const uint entry = table_word(form, op_lookup_first(op) + byte / 4) >> (byte % 4 * 8);
    // Sign-extended from `bits` bits.
    const uint sign = 1u << (bits - 1u);
    return ((entry & low_bits(bits)) ^ sign) - sign;
}

// A value's ops (a run) and its first operand.
SEISMIC_REPACK_FUNCTION uint value_ops(WordForm form, uint record) { return table_word(form, record); }
SEISMIC_REPACK_FUNCTION uint value_operands(WordForm form, uint record) { return table_word(form, record + 1); }

// The runs a value's fields read.
SEISMIC_REPACK_FUNCTION uint value_runs(WordForm form, uint record) {
    const uint ops = value_ops(form, record);
    uint operand = value_operands(form, record), runs = 0;
    for (uint at = run_first(ops); at < run_first(ops) + run_count(ops); ++at) {
        const uint code = op_code(table_word(form, at));
        if (code == SEISMIC_REPACK_OP_FIELD) runs += run_count(table_word(form, operand++));
        else if (code == SEISMIC_REPACK_OP_OFFSET) ++operand;
    }
    return runs;
}

// The runs the representative piece `piece` of a group visits: its own,
// or its value's fields' (runs are numbered by representatives, each
// member visiting its representative's).
SEISMIC_REPACK_FUNCTION uint piece_runs(WordForm form, uint piece) {
    return piece_is_value(piece) ? value_runs(form, piece_first(piece)) : 1u;
}

// A unit's class and the table word of its displacements (two per slot,
// bytes then bits, two's complement); a class's representative unit and
// slot map.
SEISMIC_REPACK_FUNCTION uint unit_class(WordForm form, uint unit) {
    return table_word(form, form.directory + 2 * unit);
}
SEISMIC_REPACK_FUNCTION uint unit_displacements(WordForm form, uint unit) {
    return table_word(form, form.directory + 2 * unit + 1);
}
SEISMIC_REPACK_FUNCTION uint class_representative(WordForm form, uint c) {
    return table_word(form, form.classes + 2 * c);
}
SEISMIC_REPACK_FUNCTION uint class_slot_map(WordForm form, uint c) {
    return table_word(form, form.classes + 2 * c + 1);
}

// A unit's displacement of slot `slot` (`displacements` its table word
// `unit_displacements`): in whole bytes, and in bits within a byte (two's
// complement).
SEISMIC_REPACK_FUNCTION int displacement_bytes(WordForm form, uint displacements, uint slot) {
    return int(table_word(form, displacements + 2 * slot));
}
SEISMIC_REPACK_FUNCTION uint displacement_bits(WordForm form, uint displacements, uint slot) {
    return table_word(form, displacements + 2 * slot + 1);
}

// ---- compile-time iteration ------------------------------------------------

template <uint I> struct Index {
    SEISMIC_REPACK_STATIC uint value = I;
};

// `f(Index<I>{})` for every I in [FIRST, FIRST + COUNT).
template <uint FIRST, uint COUNT, typename F> SEISMIC_REPACK_INLINE void each(SEISMIC_REPACK_PRIVATE const F &f) {
    if constexpr (COUNT == 1) {
        f(Index<FIRST>{});
    } else if constexpr (COUNT > 1) {
        each<FIRST, COUNT / 2>(f);
        each<FIRST + COUNT / 2, COUNT - COUNT / 2>(f);
    }
}

// `f(Index<index>{})` for the runtime `index` in [FIRST, FIRST + COUNT).
template <uint FIRST, uint COUNT, typename F>
SEISMIC_REPACK_INLINE void dispatch(uint index, SEISMIC_REPACK_PRIVATE const F &f) {
    if constexpr (COUNT == 1) {
        f(Index<FIRST>{});
    } else {
        if (index < FIRST + COUNT / 2) dispatch<FIRST, COUNT / 2>(index, f);
        else dispatch<FIRST + COUNT / 2, COUNT - COUNT / 2>(index, f);
    }
}

// The largest power of two, to 16, dividing `bytes` (16 for 0): the owner's
// (`seismic_lang::repack::alignment`) rule, here because kernels apply it at
// compile time to their own constants (sizes, offsets), which they cannot
// receive.
SEISMIC_REPACK_FUNCTION uint alignment(uint bytes) {
    return bytes % 16 == 0 ? 16 : bytes % 8 == 0 ? 8 : bytes % 4 == 0 ? 4 : bytes % 2 == 0 ? 2 : 1;
}

// ---- executing record words ------------------------------------------------
//
// The table is a constant expression, and every record word a kernel computes
// is named at compile time: these run at compile time over the table, leaving
// each block's loads, shifts and value ops. A block's words are the OR of its
// groups' members: a group's of more than one member by one loop over them,
// which the compiler unrolls (SEISMIC_REPACK_LOOP), its displacements read
// from its list; a lone member's emitted, its displacements constants. They
// read one source packet through a reader `r` of the kernel's: `R::form()`, the conversion's
// word form, and `r.template run<AT, K>(delta)`, the bits of the run of
// source bits at piece AT moved by `delta` bits (a member's, which spans as
// many bytes), the K-th run its block's representatives visit (where the
// kernel loads them from). They iterate with function objects, whose calls
// are SEISMIC_REPACK_INLINE, as a kernel's large loop bodies do: a lambda's
// call cannot be, and the inliner leaves large ones out of line, their
// captures then read through generic pointers (CUDA, clang sm_80: loads lose
// `ld.global.nc` and `ld.shared`, and programs grow by up to 28% in PTX
// instructions).

// A group's lone member, its representative: no displacements.
struct Alone {
    template <uint P> SEISMIC_REPACK_INLINE int at() const { return 0; }
};

// Member `m` of the group at GROUP, `m` a loop's: its displacements read
// from its list by plain loads (not SEISMIC_REPACK_LOAD, whose CUDA load no
// compiler folds), which the loop's unrolling makes constants.
template <typename R, uint GROUP> struct Looped {
    uint m;
    template <uint P> SEISMIC_REPACK_INLINE int at() const {
        if constexpr (!varies(R::form(), GROUP, P)) {
            return 0;
        } else {
            constexpr uint first = displacement_at(R::form(), GROUP, P), row = row_words(R::form(), GROUP);
            return int(R::form().table[first + row * m]);
        }
    }
};

// ORs into `value` piece I of the run FIELD, a run of source bits, of the
// member `d` displaces: the first the K-th run its block visits and the P-th
// parameter.
template <typename R, uint FIELD, uint K, uint P, typename D> struct GatherPiece {
    SEISMIC_REPACK_PRIVATE const R &r;
    SEISMIC_REPACK_PRIVATE const D &d;
    SEISMIC_REPACK_PRIVATE uint &value;
    template <typename I> SEISMIC_REPACK_INLINE void operator()(I) const {
        constexpr uint at = run_first(FIELD) + I::value, shift = piece_shift(table_word(R::form(), at));
        value |= r.template run<at, K + I::value>(d.template at<P + I::value>()) << shift;
    }
};

// The OR of the pieces of the run FIELD of the member `d` displaces, the
// first the K-th run its block visits and the P-th parameter.
template <typename R, uint FIELD, uint K, uint P, typename D>
SEISMIC_REPACK_INLINE uint gather(SEISMIC_REPACK_PRIVATE const R &r, SEISMIC_REPACK_PRIVATE const D &d) {
    uint value = 0;
    each<0, run_count(FIELD)>(GatherPiece<R, FIELD, K, P, D>{r, d, value});
    return value;
}

// The ops [AT, AT + COUNT) of the value of the member `d` displaces over
// `stack` of HEIGHT words, its next operand at OPERAND, its next field's
// first run the K-th its block visits and its next parameter the P-th.
template <typename R, uint AT, uint COUNT, uint HEIGHT, uint OPERAND, uint K, uint P, typename D>
SEISMIC_REPACK_INLINE void execute(SEISMIC_REPACK_PRIVATE const R &r, SEISMIC_REPACK_PRIVATE const D &d,
                                   SEISMIC_REPACK_PRIVATE uint *stack) {
    if constexpr (COUNT > 0) {
        constexpr uint op = table_word(R::form(), AT), code = op_code(op);
        constexpr uint top = HEIGHT - 1;
        if constexpr (code == SEISMIC_REPACK_OP_FIELD) {
            constexpr uint field = table_word(R::form(), OPERAND);
            stack[HEIGHT] = gather<R, field, K, P>(r, d);
            execute<R, AT + 1, COUNT - 1, HEIGHT + 1, OPERAND + 1, K + run_count(field), P + run_count(field)>(
                r, d, stack);
        } else if constexpr (code == SEISMIC_REPACK_OP_OFFSET) {
            constexpr uint offset = table_word(R::form(), OPERAND);
            stack[top] += offset + uint(d.template at<P>());
            execute<R, AT + 1, COUNT - 1, HEIGHT, OPERAND + 1, K, P + 1>(r, d, stack);
        } else if constexpr (code == SEISMIC_REPACK_OP_OR || code == SEISMIC_REPACK_OP_MULTIPLY_F32 ||
                             code == SEISMIC_REPACK_OP_NEGATE_IF) {
            if constexpr (code == SEISMIC_REPACK_OP_OR) {
                stack[top - 1] |= stack[top];
            } else if constexpr (code == SEISMIC_REPACK_OP_MULTIPLY_F32) {
                stack[top - 1] = f32_multiply(stack[top - 1], stack[top]);
            } else {
                // `-value` where the sign is 1: `(value ^ -sign) + sign`
                // (`seismic_lang::repack::negate_if`).
                stack[top - 1] = (stack[top - 1] ^ (0u - stack[top])) + stack[top];
            }
            execute<R, AT + 1, COUNT - 1, HEIGHT - 1, OPERAND, K, P>(r, d, stack);
        } else {
            if constexpr (code == SEISMIC_REPACK_OP_SHIFT_LEFT) {
                stack[top] <<= op_shift_bits(op);
            } else if constexpr (code == SEISMIC_REPACK_OP_F16_TO_F32) {
                stack[top] = f16_to_f32(stack[top]);
            } else if constexpr (code == SEISMIC_REPACK_OP_I32_TO_F32) {
                stack[top] = i32_to_f32(stack[top]);
            } else if constexpr (code == SEISMIC_REPACK_OP_SCALE_F32) {
                // Times 2^exponent, a normal float built from its exponent bits.
                stack[top] = f32_multiply(stack[top], op_scale_bits(op));
            } else {
                static_assert(code == SEISMIC_REPACK_OP_LOOKUP, "a word table's op");
                stack[top] = lookup_entry(R::form(), op, stack[top]);
            }
            execute<R, AT + 1, COUNT - 1, HEIGHT, OPERAND, K, P>(r, d, stack);
        }
    }
}

// ORs into `out[w]` the member `d` displaces of the group at GROUP, w its
// word: its representative piece (a run of source bits, or a slice of a
// computed value) with each parameter displaced, its first run the K-th
// its block visits.
template <typename R, uint GROUP, uint K, typename D>
SEISMIC_REPACK_INLINE void member(SEISMIC_REPACK_PRIVATE const R &r, SEISMIC_REPACK_PRIVATE const D &d,
                                  SEISMIC_REPACK_PRIVATE uint *out) {
    constexpr uint piece = table_word(R::form(), GROUP), word0 = group_word(R::form(), GROUP);
    const uint word = word0 + uint(d.template at<0>());
    const uint shift = piece_shift(piece) + uint(d.template at<1>());
    if constexpr (piece_is_value(piece)) {
        constexpr uint record = piece_first(piece), ops = value_ops(R::form(), record);
        uint stack[R::form().stack];
        execute<R, run_first(ops), run_count(ops), 0, value_operands(R::form(), record), K, 2>(r, d, stack);
        out[word] |= ((stack[0] >> piece_value_bit(piece)) & low_bits(piece_length(piece))) << shift;
    } else {
        out[word] |= r.template run<GROUP, K>(d.template at<2>()) << shift;
    }
}

// The members of the groups at [AT, END) of a block into `out` (`member`),
// K the visit number of the first run of the group at AT: one pass over the
// block's groups, the next's K counted on.
template <typename R, uint AT, uint END, uint K>
SEISMIC_REPACK_INLINE void groups_from(SEISMIC_REPACK_PRIVATE const R &r, SEISMIC_REPACK_PRIVATE uint *out) {
    if constexpr (AT < END) {
        constexpr uint count = group_count(R::form(), AT);
        if constexpr (count > 1) {
            SEISMIC_REPACK_LOOP
            for (uint m = 0; m < count; ++m) member<R, AT, K>(r, Looped<R, AT>{m}, out);
        } else {
            member<R, AT, K>(r, Alone{}, out);
        }
        groups_from<R, AT + 4, END, K + piece_runs(R::form(), table_word(R::form(), AT))>(r, out);
    }
}

// The WORDS words of a block whose groups are GROUPS (`unit_groups`,
// `fixed_groups`) of the record of the packet `r` reads, into `out`.
template <typename R, uint GROUPS, uint WORDS>
SEISMIC_REPACK_INLINE void block_record(SEISMIC_REPACK_PRIVATE const R &r, SEISMIC_REPACK_PRIVATE uint *out) {
    for (uint i = 0; i < WORDS; ++i) out[i] = 0;
    groups_from<R, run_first(GROUPS), run_first(GROUPS) + 4 * run_count(GROUPS), 0>(r, out);
}

// The WORDS record words of unit UNIT (a class's representative) of the
// packet `r` reads, into `out`.
template <typename R, uint UNIT, uint WORDS>
SEISMIC_REPACK_INLINE void unit_record(SEISMIC_REPACK_PRIVATE const R &r, SEISMIC_REPACK_PRIVATE uint *out) {
    block_record<R, unit_groups(R::form(), UNIT), WORDS>(r, out);
}

// Record words [FIRST, FIRST + COUNT), a fixed block after a span of UNITS
// units (none for COUNT 0), of the packet `r` reads, into `out`.
template <typename R, uint UNITS, uint FIRST, uint COUNT>
SEISMIC_REPACK_INLINE void fixed_record(SEISMIC_REPACK_PRIVATE const R &r, SEISMIC_REPACK_PRIVATE uint *out) {
    if constexpr (COUNT > 0) {
        constexpr uint f = fixed_block(R::form(), UNITS, FIRST);
        static_assert(run_count(fixed_words(R::form(), UNITS, f)) == COUNT, "a fixed range is one fixed block");
        block_record<R, fixed_groups(R::form(), UNITS, f), COUNT>(r, out);
    }
}

}  // namespace repack
