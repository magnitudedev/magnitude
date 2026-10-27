//! The Vulkan dialect of the generated prefix (Vulkan backend spec §5.2):
//! `#version` and the floor's extensions, the device feature macros, the
//! device library, the argument block, the workgroup size and the typed
//! group-memory views, and the suffix that calls the launch's kernel.

use seismic_lang::registry::{self, RowPlaneRole};
use seismic_lang::repack::{
    Group, PieceBits, WordClass, WordLookup, WordStep, WordTable, WordUnit,
};

/// Device features a Vulkan prefix exposes as `SEISMIC_HAS_*` macros (and
/// the extensions they need). They are part of the rendered source, so of
/// the formation and tuning identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VulkanFeatures {
    /// Lanes of one hardware subgroup: 32, or 64 holding two logical
    /// subgroups (`SEISMIC_SUBGROUP_LANES`).
    pub(crate) subgroup_lanes: u32,
    /// fp16 arithmetic (`SEISMIC_HAS_FLOAT16`).
    pub(crate) float16: bool,
    /// Subgroup 16x16x16 cooperative matrix, f16->f32 and s8->s32.
    pub(crate) matrix: bool,
    /// Accumulator arrays of more than 4 matrix fragments compile correctly.
    pub(crate) wide_accumulators: bool,
    /// Mixed-signedness packed 4x8 dot product is accelerated.
    pub(crate) mixed_dot: bool,
    pub(crate) f32_atomic_add: bool,
    pub(crate) shared_int64_atomics: bool,
}

impl VulkanFeatures {
    /// The features a device with `facts` exposes.
    pub(crate) fn of(facts: &seismic_vulkan::FormationFacts) -> Self {
        Self {
            subgroup_lanes: facts.subgroup_lanes,
            float16: facts.float16,
            matrix: facts.matrix,
            wide_accumulators: facts.wide_accumulators,
            mixed_dot: facts.mixed_dot,
            f32_atomic_add: facts.f32_atomic_add,
            shared_int64_atomics: facts.shared_int64_atomics,
        }
    }
}

/// The typed views of a Vulkan launch's one group-memory region, in
/// specialization-constant order after the workgroup size (ids 3..): view
/// name suffix, GLSL element type and element bytes. Every view covers the
/// whole region, and all of them alias it.
const SHARED_VIEWS: [(&str, &str, u64); 8] = [
    ("F32", "float", 4),
    ("F16", "float16_t", 2),
    ("BF16", "uint16_t", 2),
    ("U8", "uint8_t", 1),
    ("U16", "uint16_t", 2),
    ("U32", "uint", 4),
    ("I32", "int", 4),
    ("UVEC4", "uvec4", 16),
];

/// First specialization-constant id of the shared views; ids 0..2 are the
/// workgroup size.
const SHARED_VIEW_CONSTANT: usize = 3;

/// Lengths of the shared views of a region of `shared_bytes` (at least one
/// element each), in specialization-constant order.
pub(crate) fn shared_view_lengths(shared_bytes: u64) -> Vec<u64> {
    SHARED_VIEWS
        .iter()
        .map(|(_, _, bytes)| shared_bytes.div_ceil(*bytes).max(1))
        .collect()
}

/// Group memory a pipeline reserves for `shared_bytes`: its largest view.
pub(crate) fn shared_footprint(shared_bytes: u64) -> u64 {
    SHARED_VIEWS
        .iter()
        .zip(shared_view_lengths(shared_bytes))
        .map(|((_, _, bytes), length)| bytes * length)
        .max()
        .expect("there are shared views")
}

/// The Vulkan device library (`vulkan_prelude.glsl`): element references,
/// conversions, rounded arithmetic, subgroup shuffles and fixed-order
/// reductions, packed dot products.
const PRELUDE: &str = include_str!("../vulkan_prelude.glsl");

/// Extensions every Vulkan source enables: the device floor's.
const EXTENSIONS: [&str; 21] = [
    "GL_GOOGLE_cpp_style_line_directive",
    "GL_EXT_buffer_reference",
    "GL_EXT_buffer_reference2",
    "GL_EXT_scalar_block_layout",
    "GL_EXT_shader_explicit_arithmetic_types",
    "GL_EXT_shader_16bit_storage",
    "GL_EXT_shader_8bit_storage",
    "GL_EXT_shared_memory_block",
    "GL_EXT_control_flow_attributes",
    "GL_EXT_integer_dot_product",
    "GL_EXT_spirv_intrinsics",
    "GL_KHR_memory_scope_semantics",
    "GL_KHR_shader_subgroup_basic",
    "GL_KHR_shader_subgroup_vote",
    "GL_KHR_shader_subgroup_arithmetic",
    "GL_KHR_shader_subgroup_ballot",
    "GL_KHR_shader_subgroup_shuffle",
    "GL_KHR_shader_subgroup_shuffle_relative",
    "GL_KHR_shader_subgroup_clustered",
    "GL_KHR_shader_subgroup_quad",
    "GL_EXT_shader_subgroup_extended_types_int8",
];

/// `#version`, the extensions, the device feature macros and the prelude.
pub(super) fn header(features: VulkanFeatures) -> String {
    let mut header = String::from("#version 460\n");
    let gated = [
        (
            features.float16,
            "GL_EXT_shader_subgroup_extended_types_float16",
        ),
        (features.matrix, "GL_KHR_cooperative_matrix"),
        (features.f32_atomic_add, "GL_EXT_shader_atomic_float"),
        (features.shared_int64_atomics, "GL_EXT_shader_atomic_int64"),
    ];
    for extension in EXTENSIONS
        .iter()
        .chain(gated.iter().filter(|(on, _)| *on).map(|(_, name)| name))
    {
        header.push_str(&format!("#extension {extension} : require\n"));
    }
    header.push_str(&format!(
        "#define SEISMIC_SUBGROUP_LANES {}\n",
        features.subgroup_lanes
    ));
    for (name, on) in [
        ("FLOAT16", features.float16),
        ("MATRIX", features.matrix),
        ("WIDE_ACCUMULATORS", features.wide_accumulators),
        ("MIXED_DOT", features.mixed_dot),
        ("F32_ATOMIC_ADD", features.f32_atomic_add),
        ("SHARED_INT64_ATOMICS", features.shared_int64_atomics),
    ] {
        header.push_str(&format!("#define SEISMIC_HAS_{name} {}\n", u8::from(on)));
    }
    header.push_str(PRELUDE);
    header
}

/// The argument block (buffer addresses, the scalar-result address, then
/// the words: there is no words buffer), read-only flags, workgroup size and
/// group-memory views of a source with `buffers` buffers.
pub(super) fn tail(buffers: usize, read_only: impl Iterator<Item = bool>) -> String {
    let mut tail = format!("#define SEISMIC_BUFFER_SCALAR_RESULTS {buffers}\n");
    tail.push_str(
        "layout(push_constant, scalar) uniform seismic_push_t { uint64_t seismic_arguments; };\n\
         layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer seismic_argument_block_t { uint64_t w[]; };\n\
         #define SEISMIC_PTR(index) (seismic_argument_block_t(seismic_arguments).w[index])\n",
    );
    tail.push_str(&format!(
        "#define SEISMIC_SCALAR_RESULTS (seismic_argument_block_t(seismic_arguments).w[{buffers}])\n\
         #define seismic_words (seismic_argument_block_t(seismic_arguments + {}ul).w)\n",
        (buffers + 1) * 8
    ));
    tail.push_str("#define SEISMIC_READONLY_(index) SEISMIC_READONLY_##index\n#define SEISMIC_READONLY(index) SEISMIC_READONLY_(index)\n");
    for (index, read_only) in read_only.enumerate() {
        tail.push_str(&format!("#define SEISMIC_READONLY_{index} {read_only}\n"));
    }
    tail.push_str("layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;\n");
    for (ordinal, (name, ty, bytes)) in SHARED_VIEWS.iter().enumerate() {
        let lower = name.to_ascii_lowercase();
        tail.push_str(&format!(
            "layout(constant_id = {}) const uint SEISMIC_SHARED_{name} = {}u;\nshared seismic_shared_{lower}_t {{ {ty} seismic_shared_{lower}[SEISMIC_SHARED_{name}]; }};\n",
            SHARED_VIEW_CONSTANT + ordinal,
            SHARED_DEFAULT_BYTES / bytes
        ));
    }
    tail
}

/// The group-memory region the compiler sees before specialization: the
/// largest any floor device has (64 KiB), so constant indices into the
/// views compile. Pipelines always specialize the lengths to the launch's
/// `shared_bytes`; drivers allocate only the specialized size.
const SHARED_DEFAULT_BYTES: u64 = 64 << 10;

/// Calls the launch's kernel, which formation names with `SEISMIC_KERNEL`.
pub(super) const SUFFIX: &str = "\nvoid main() {\n    SEISMIC_KERNEL();\n}\n";

/// A registered conversion's word form (`WordTable`, whose docs give the
/// table's format) as GLSL, whose compiler evaluates no table: the table is
/// evaluated here, each block left as straight-line code (its groups'
/// members' source loads, shifts and masks, and their values' ops, each
/// value once).
///
/// - `{prefix}_UNIT(packet, unit, words)`: the words of unit `unit` of the
///   code planes' span (`WordSpan`), the codes' then the high codes', of
///   the source packet at address `packet`, from `words[0]` (an array of a
///   record's words, as each share's). Each class's representative
///   unit is code once; a unit runs its class's code with its runs moved
///   by its displacements, selected by the unit.
/// - `{prefix}_<ROLE>_SHARE(packet, words)`, per group role of a row layout
///   (`RowPlaneRole::ALL` past the code roles; with no words without the
///   role): the words of the role's share, its runs the table's, from
///   `words[0]`. A row layout's record words are its code planes' span and
///   these shares.
///
/// A source packet is read in aligned words of up to its packet alignment
/// (`ExternalPacketLayout::packet_alignment`, at most 4), each within the
/// packet wherever a unit moves it.
pub(super) fn render_conversion_code(
    out: &mut String,
    prefix: &str,
    words: &WordTable,
    source: &registry::ExternalPacketLayout,
) {
    let span = words.span();
    let name = prefix.to_ascii_lowercase();
    let (alignment, packet_size) = (source.packet_alignment, source.packet_size);
    // A read's alignment is a size's the packet alignment allows.
    for read in [1u32, 2, 4].into_iter().filter(|size| *size <= alignment) {
        for size in [1, 2, 4].into_iter().filter(|size| *size <= read) {
            out.push_str(&format!(
                "layout(buffer_reference, scalar, buffer_reference_align = {read}) readonly buffer {name}_{size}_{read} {{ uint{bits}_t v[]; }};\n",
                bits = 8 * size
            ));
        }
    }
    render_lookups(out, &name, words);

    // The span's units, each one's class and displacements.
    out.push_str(&format!(
        "void {prefix}_UNIT(uint64_t packet, uint unit, out uint words[{}]) {{\n",
        words.words()
    ));
    out.push_str("    switch (unit) {\n");
    for index in 0..span.class_count {
        let members = words.class_units(index).collect::<Vec<_>>();
        for (unit, _) in &members {
            out.push_str(&format!("    case {unit}u:\n"));
        }
        let class = words.class(index);
        let mut code = RecordWord::new(words, &name, alignment, packet_size);
        code.moves = Some(Moves {
            index,
            class,
            members,
        });
        code.block(class.representative);
        out.push_str("    {\n");
        for line in &code.lines {
            out.push_str(&format!("        {line}\n"));
        }
        out.push_str("        return;\n    }\n");
    }
    out.push_str("    }\n}\n\n");

    // Each group role's share.
    for role in RowPlaneRole::ALL
        .into_iter()
        .filter(|role| !RowPlaneRole::CODES.contains(role))
    {
        out.push_str(&format!(
            "void {prefix}_{}_SHARE(uint64_t packet, out uint words[{}]) {{\n",
            super::native_macro(role.name()),
            words.words()
        ));
        if let Some(&(_, first, _)) = words.roles().iter().find(|(share, _, _)| *share == role) {
            let mut code = RecordWord::new(words, &name, alignment, packet_size);
            code.block(words.fixed_block(first));
            for line in &code.lines {
                out.push_str(&format!("    {line}\n"));
            }
        }
        out.push_str("}\n\n");
    }
}

/// Each lookup table the record words' values read: a function of an
/// entry's index giving the entry, sign-extended, named by the table's
/// first word.
fn render_lookups(out: &mut String, name: &str, words: &WordTable) {
    for (WordLookup { first, shift }, entries, table) in words.lookups() {
        let bytes = 1usize << shift;
        // The entries in the fewest bits each (unsigned when none is
        // negative): held in one 64-bit constant when they fit, read with a
        // shift; else in a constant array of the table's words.
        let signed = entries.iter().any(|entry| *entry < 0);
        let width = entries
            .iter()
            .map(|entry| {
                if signed {
                    33 - entry.max(&!entry).leading_zeros()
                } else {
                    32 - entry.leading_zeros()
                }
            })
            .max()
            .expect("a lookup table has entries")
            .max(1);
        let body = if entries.len() as u32 * width <= 64 {
            let packed = entries
                .iter()
                .enumerate()
                .fold(0u64, |packed, (index, entry)| {
                    packed
                        | (u64::from(*entry as u32) & (u64::MAX >> (64 - width)))
                            << (index as u32 * width)
                });
            let field = format!("uint({packed:#x}ul >> ({width}u * index))");
            if signed {
                format!(
                    "uint(int({field} << {high}u) >> {high}u)",
                    high = 32 - width
                )
            } else {
                format!("{field} & {:#x}u", u32::MAX >> (32 - width))
            }
        } else {
            out.push_str(&format!(
                "const uint {name}_entries_{first}[] = uint[]({});\n",
                table
                    .iter()
                    .map(|word| format!("{word:#x}u"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            let high = 32 - 8 * bytes;
            let word = format!("{name}_entries_{first}[index >> {}u]", 2 - shift);
            let entry = match shift {
                2 => word,
                _ => format!(
                    "({word} >> ((index & {}u) * {}u))",
                    (4 / bytes) - 1,
                    8 * bytes
                ),
            };
            match high {
                0 => entry,
                _ => format!("uint(int({entry} << {high}u) >> {high}u)"),
            }
        };
        out.push_str(&format!(
            "uint {name}_lookup_{first}(uint index) {{\n    return {body};\n}}\n"
        ));
    }
}

/// `length` bits of the word `bits` from bit `from` (all of them for a
/// length of 32), placed at bit `shift` of a word: masked unless the shifts
/// drop every other bit.
fn placed(bits: String, from: u32, length: u32, shift: u32) -> String {
    let bits = match from {
        0 => bits,
        from => format!("({bits} >> {from}u)"),
    };
    let bits = if from + length == 32 || shift + length == 32 {
        bits
    } else {
        format!("({bits} & {:#x}u)", u32::MAX >> (32 - length))
    };
    match shift {
        0 => bits,
        shift => format!("({bits} << {shift}u)"),
    }
}

/// A unit's movement of its class's representative runs (`WordSpan`): the
/// class (its index and slot map), and each member unit with its
/// displacements.
struct Moves<'a> {
    index: u32,
    class: WordClass<'a>,
    members: Vec<(u32, WordUnit<'a>)>,
}

/// Blocks' code, reading a source packet of `packet_size` bytes aligned to
/// `alignment`: its lines, each value once.
struct RecordWord<'a> {
    words: &'a WordTable,
    name: &'a str,
    alignment: u32,
    packet_size: u32,
    moves: Option<Moves<'a>>,
    /// The runs visited so far (`WordSpan`'s order).
    visited: usize,
    lines: Vec<String>,
    /// The values evaluated: each its record and its member's
    /// displacements of the value's runs and offsets.
    values: Vec<(u32, Vec<i32>)>,
    declared: Vec<String>,
}

impl<'a> RecordWord<'a> {
    fn new(words: &'a WordTable, name: &'a str, alignment: u32, packet_size: u32) -> Self {
        Self {
            words,
            name,
            alignment,
            packet_size,
            moves: None,
            visited: 0,
            lines: Vec::new(),
            values: Vec::new(),
            declared: Vec::new(),
        }
    }

    /// The words of block `block` into `words[0..]`: the OR of its groups'
    /// members, each member emitted (`Group`: GLSL has no front end that
    /// compiles a loop's members once, and NVIDIA's driver leaves
    /// `[[unroll]]` loops rolled).
    fn block(&mut self, block: u32) {
        let words = self.words;
        let mut terms = vec![Vec::new(); words.block_words(block).len()];
        for group in words.groups(block) {
            let first = self.visited;
            for m in 0..group.count {
                self.visited = first;
                let term = self.member(group, m);
                terms[group.word.wrapping_add(group.displacement(m, 0) as u32) as usize].push(term);
            }
            self.visited = first + group.runs().len();
        }
        for (word, terms) in terms.into_iter().enumerate() {
            let value = if terms.is_empty() {
                String::from("0u")
            } else {
                terms.join(" | ")
            };
            self.lines.push(format!("words[{word}] = {value};"));
        }
    }

    /// Member `m` of `group`, placed.
    fn member(&mut self, group: Group, m: u32) -> String {
        let piece = group.piece;
        let shift = piece.shift.wrapping_add(group.displacement(m, 1) as u32);
        match piece.bits {
            PieceBits::Source(bit) => {
                let bits = self.source_run(
                    bit.wrapping_add(group.displacement(m, 2) as u32),
                    piece.length,
                );
                placed(bits, 0, 32, shift)
            }
            PieceBits::Value { record, bit } => {
                let value = self.value(group, record, m);
                placed(value, bit, piece.length, shift)
            }
        }
    }

    /// Declares `line` (named `name`) once.
    fn declare(&mut self, name: String, line: String) {
        if !self.declared.contains(&name) {
            self.declared.push(name);
            self.lines.push(line);
        }
    }

    /// The `length` bits of the next run visited, from source bit `bit` in
    /// the table, in the low bits of a word: the aligned source words
    /// holding them (of one size, the widest within the packet wherever a
    /// unit moves them), little-endian, shifted down and masked unless the
    /// words end with the run. A moved run's words are read from the packet
    /// moved by its slot's byte displacement, and shifted by its bit
    /// displacement more.
    fn source_run(&mut self, bit: u32, length: u32) -> String {
        let run = self.visited;
        self.visited += 1;
        // A moved run's slot, whether it moves by whole bytes, its byte
        // displacements over the class's units, the alignment the moved
        // packets keep, and the line declaring the unit's displacements of
        // the slot, selected by the unit.
        let moved = self.moves.as_ref().and_then(|moves| {
            let entry = moves.class.entry(run);
            (!entry.still).then(|| {
                let slot = entry.slot;
                let bytes = moves
                    .members
                    .iter()
                    .map(|(_, unit)| unit.displacement(slot).0)
                    .collect::<Vec<_>>();
                let alignment = self.words.slot_alignment(moves.index, slot);
                let select = |part: fn((i32, i32)) -> i32| {
                    let ((_, last), rest) = moves.members.split_last().expect("a class has units");
                    rest.iter().rev().fold(
                        format!("{}", part(last.displacement(slot))),
                        |rest, (unit, moved)| {
                            format!(
                                "unit == {unit}u ? {} : {rest}",
                                part(moved.displacement(slot))
                            )
                        },
                    )
                };
                let line = format!(
                    "const int d{slot}b = {}, d{slot}s = {};",
                    select(|(bytes, _)| bytes),
                    select(|(_, bits)| bits)
                );
                (slot, entry.whole, bytes, alignment, line)
            })
        });
        let (low_byte, high_byte) = moved.as_ref().map_or((0, 0), |(_, _, bytes, _, _)| {
            (*bytes.iter().min().unwrap(), *bytes.iter().max().unwrap())
        });
        // Alignment the moved packets keep.
        let alignment = moved
            .as_ref()
            .map_or(self.alignment, |(_, _, _, alignment, _)| {
                self.alignment.min(*alignment)
            });
        let base = match &moved {
            None => String::from("packet"),
            Some((slot, _, _, _, line)) => {
                self.declare(format!("d{slot}"), line.clone());
                format!("packet + uint64_t(int64_t(d{slot}b))")
            }
        };
        let view = |this: &mut Self, size: u32, alignment: u32| {
            let view = match &moved {
                None => format!("source{size}_{alignment}"),
                Some((slot, _, _, _, _)) => format!("moved{slot}_{size}_{alignment}"),
            };
            this.declare(
                view.clone(),
                format!(
                    "{name}_{size}_{alignment} {view} = {name}_{size}_{alignment}({base});",
                    name = this.name
                ),
            );
            view
        };
        let (first, last) = (bit / 8, (bit + length - 1) / 8);
        let (start, size) = [4u32, 2, 1]
            .into_iter()
            .filter(|size| *size <= alignment)
            .map(|size| (first / size * size, size))
            .find(|(start, size)| {
                let end = (last / size + 1) * size;
                *start as i32 + low_byte >= 0 && end as i32 + high_byte <= self.packet_size as i32
            })
            .expect("a byte is a word");
        let end = (last / size + 1) * size;
        let span = 8 * (end - start);
        let wide = span > 32;
        let view = view(self, size, alignment);
        let words = (start / size..end / size)
            .map(|index| {
                let offset = 8 * (index * size - start);
                let load = format!("{view}.v[{index}]");
                let load = if wide {
                    format!("uint64_t({load})")
                } else {
                    format!("uint({load})")
                };
                match offset {
                    0 => load,
                    offset => format!("({load} << {offset}u)"),
                }
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let low = bit - 8 * start;
        let shift = match &moved {
            Some((slot, false, _, _, _)) => Some(format!("{low}u + uint(d{slot}s)")),
            _ => (low != 0).then(|| format!("{low}u")),
        };
        let shifted = match (shift, wide) {
            (None, false) => format!("({words})"),
            (None, true) => format!("uint({words})"),
            (Some(shift), false) => format!("(({words}) >> ({shift}))"),
            (Some(shift), true) => format!("uint(({words}) >> ({shift}))"),
        };
        let exact = matches!(moved, None | Some((_, true, _, _, _))) && low + length == span;
        if exact || length == 32 {
            shifted
        } else {
            format!("({shifted} & {:#x}u)", u32::MAX >> (32 - length))
        }
    }

    /// A local holding `group`'s member `m` of the value whose record is at
    /// `record` (the representative's), evaluating it on first use (its
    /// postfix ops over a stack of locals, visiting its runs), its runs and
    /// offsets displaced. The block advances past the group's runs
    /// (`block`).
    fn value(&mut self, group: Group, record: u32, m: u32) -> String {
        let words = self.words;
        let parameters = 2
            + group.runs().len() as u32
            + words
                .steps(record)
                .filter(|step| matches!(step, WordStep::Offset(_)))
                .count() as u32;
        let key = (
            record,
            (2..parameters)
                .map(|parameter| group.displacement(m, parameter))
                .collect::<Vec<_>>(),
        );
        if let Some(index) = self.values.iter().position(|known| *known == key) {
            return format!("v{index}");
        }
        let value = format!("v{}", self.values.len());
        let mut parameter = 2;
        let mut stack: Vec<String> = Vec::new();
        for (index, step) in words.steps(record).enumerate() {
            let expression = match step {
                WordStep::Field(pieces) => pieces
                    .into_iter()
                    .map(|piece| {
                        parameter += 1;
                        let bit = piece
                            .bit
                            .wrapping_add(group.displacement(m, parameter - 1) as u32);
                        placed(self.source_run(bit, piece.length), 0, 32, piece.shift)
                    })
                    .collect::<Vec<_>>()
                    .join(" | "),
                WordStep::Offset(offset) => {
                    parameter += 1;
                    let offset = offset.wrapping_add(group.displacement(m, parameter - 1) as u32);
                    format!("{} + {offset:#x}u", stack.pop().expect("an operand"))
                }
                WordStep::Or | WordStep::MultiplyF32 | WordStep::NegateIf => {
                    let right = stack.pop().expect("an operand");
                    let left = stack.pop().expect("an operand");
                    match step {
                        WordStep::Or => format!("{left} | {right}"),
                        WordStep::MultiplyF32 => format!(
                            "floatBitsToUint(seismic_mul_rn(uintBitsToFloat({left}), uintBitsToFloat({right})))"
                        ),
                        // `-value` where the sign is 1: `(value ^ -sign) + sign`
                        // (`seismic_lang::repack::negate_if`).
                        _ => format!("({left} ^ (0u - {right})) + {right}"),
                    }
                }
                WordStep::ShiftLeft(bits) => {
                    format!("{} << {bits}u", stack.pop().expect("an operand"))
                }
                WordStep::F16ToF32 => format!(
                    "floatBitsToUint(seismic_f16_to_f32(uint16_t({})))",
                    stack.pop().expect("an operand")
                ),
                WordStep::I32ToF32 => format!(
                    "floatBitsToUint(float(int({})))",
                    stack.pop().expect("an operand")
                ),
                // Times 2^exponent, a normal float built from its exponent
                // bits.
                WordStep::ScaleF32(exponent) => format!(
                    "floatBitsToUint(seismic_mul_rn(uintBitsToFloat({}), uintBitsToFloat({:#x}u)))",
                    stack.pop().expect("an operand"),
                    ((127 + exponent) as u32) << 23
                ),
                WordStep::Lookup(lookup) => format!(
                    "{}_lookup_{}({})",
                    self.name,
                    lookup.first,
                    stack.pop().expect("an operand")
                ),
            };
            let name = format!("{value}_{index}");
            self.lines
                .push(format!("const uint {name} = {expression};"));
            stack.push(name);
        }
        let result = stack.pop().expect("a value");
        assert!(stack.is_empty(), "a value's ops leave one word");
        self.lines.push(format!("const uint {value} = {result};"));
        self.values.push(key);
        value
    }
}

/// The prelude's correctly rounded helpers against the host oracle on the
/// first Vulkan GPU (§13.1 Numerics): `seismic_div_rn`, `seismic_sqrt_rn`,
/// `seismic_fma_rn` and the bf16 conversion over edge cases (zeros,
/// subnormals, extremes, infinities, NaN) and random operands.
#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::*;
    use seismic_native_target::{ProgramEntry, ProgramSource, Toolchain};
    use seismic_vulkan::direct::{DirectBatch, DirectLaunch};
    use seismic_vulkan::toolchain::VulkanToolchain;

    const KERNEL: &str = "
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(push_constant, scalar) uniform seismic_push_t { uint64_t seismic_arguments; };
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer words_t { uint64_t w[]; };
void numerics() {
    // The scalar-result slot carries the data buffer: [cases][count * 3 in][count * 4 out].
    const uint64_t base = words_t(seismic_arguments).w[0];
    const uint count = seismic_u32(base)[0].v;
    const uint i = gl_GlobalInvocationID.x;
    if (i >= count)
        return;
    seismic_u32 data = seismic_u32(base + 4ul);
    const float a = uintBitsToFloat(data[3u * i].v);
    const float b = uintBitsToFloat(data[3u * i + 1u].v);
    const float c = uintBitsToFloat(data[3u * i + 2u].v);
    const uint result = 3u * count + 4u * i;
    data[result].v = floatBitsToUint(seismic_div_rn(a, b));
    // |a| by its bits: a float `abs` would flush a subnormal where the
    // device flushes (NVIDIA).
    data[result + 1u].v = floatBitsToUint(seismic_sqrt_rn(uintBitsToFloat(floatBitsToUint(a) & 0x7fffffffu)));
    data[result + 2u].v = floatBitsToUint(seismic_fma_rn(a, b, c));
    data[result + 3u].v = uint(seismic_f32_to_bf16(a));
}
void main() {
    numerics();
}
";

    fn bf16(value: f32) -> u32 {
        let bits = value.to_bits();
        if value.is_nan() {
            return 0x7fff;
        }
        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16
    }

    fn same(expected: f32, actual: u32) -> bool {
        (expected.is_nan() && f32::from_bits(actual).is_nan()) || expected.to_bits() == actual
    }

    #[test]
    fn rounded_helpers_match_the_host_oracle() {
        let Some(description) = seismic_vulkan::discover().ok().and_then(|devices| {
            devices
                .into_iter()
                .find(|device| device.floor.is_ok() && device.facts.is_gpu())
        }) else {
            eprintln!("no Vulkan GPU meets the floor");
            return;
        };
        let device = seismic_vulkan::Device::open(description.facts.uuid).expect("device opens");
        let features = VulkanFeatures {
            subgroup_lanes: device.facts().subgroup_width().lanes(),
            float16: device.facts().float16,
            matrix: false,
            wide_accumulators: false,
            mixed_dot: false,
            f32_atomic_add: false,
            shared_int64_atomics: false,
        };
        let source = format!("{}{KERNEL}", header(features));
        let module = VulkanToolchain::new(device.clone())
            .compile(
                &ProgramSource {
                    text: source,
                    entries: vec![ProgramEntry {
                        symbol: "numerics".into(),
                        group_size: Some([64, 1, 1]),
                        constants: Vec::new(),
                    }],
                },
                None,
            )
            .expect("the numerics kernel forms");

        let specials = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            3.0,
            0.1,
            1e-45,
            -1e-45,
            1.1754942e-38,
            1.1754944e-38,
            2.3509887e-38,
            3.4028235e38,
            -3.4028235e38,
            1e30,
            1e-30,
            7.0e-39,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1.0000001,
            0.99999994,
            16777215.0,
            1.5e-44,
        ];
        let mut cases = Vec::new();
        for a in specials {
            for b in specials {
                cases.push((
                    a,
                    b,
                    specials[(a.to_bits() ^ b.to_bits()) as usize % specials.len()],
                ));
            }
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        while cases.len() < 1 << 16 {
            let word = next();
            // Uniform bits, and values of moderate exponent (the fast path).
            let a = f32::from_bits(word as u32);
            let b = f32::from_bits((word >> 32) as u32);
            let c = f32::from_bits(next() as u32);
            cases.push((a, b, c));
            let scale = |value: f32, word: u64| {
                value.abs().max(1e-30).min(1e30) * if word & 1 == 0 { 1.0 } else { -1.0 }
            };
            cases.push((scale(a, word), scale(b, word >> 1), scale(c, word >> 2)));
        }
        let count = cases.len() as u32;
        let mut bytes = count.to_le_bytes().to_vec();
        for (a, b, c) in &cases {
            for value in [a, b, c] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let total = bytes.len() + cases.len() * 16;
        let data = device.allocate(total as u64, 256, None).expect("allocation");
        device.write(&data, 0, &bytes).expect("upload");
        let mut batch = DirectBatch::new(&device).expect("batch");
        batch
            .launch(&DirectLaunch {
                module: &module,
                function: 0,
                buffers: &[],
                words: &[],
                scalar_results: (&data, 0),
                groups: [u64::from(count.div_ceil(64)), 1, 1],
            })
            .expect("launch");
        batch
            .commit()
            .and_then(|submission| submission.finish())
            .expect("run");
        let mut results = vec![0u8; cases.len() * 16];
        device
            .read(&data, bytes.len() as u64, &mut results)
            .expect("download");
        // Without `DenormPreserve 32` the driver's default applies (NVIDIA:
        // flush to zero, §7.3): `fma` with a subnormal operand or result is
        // then outside the oracle. The integer paths preserve subnormals.
        let flushes = !device.facts().denorm_preserve_32;
        let subnormal = |value: f32| value.is_subnormal();
        let mut failures = Vec::new();
        for (index, (a, b, c)) in cases.iter().enumerate() {
            let fma_defined = !flushes
                || ![*a, *b, *c, a.mul_add(*b, *c), a * b]
                    .into_iter()
                    .any(subnormal);
            let word = |slot: usize| {
                let at = index * 16 + slot * 4;
                u32::from_le_bytes(results[at..at + 4].try_into().expect("four bytes"))
            };
            let checks = [
                ("div", same(a / b, word(0)), (a / b).to_bits(), word(0)),
                (
                    "sqrt",
                    same(a.abs().sqrt(), word(1)),
                    a.abs().sqrt().to_bits(),
                    word(1),
                ),
                (
                    "fma",
                    !fma_defined || same(a.mul_add(*b, *c), word(2)),
                    a.mul_add(*b, *c).to_bits(),
                    word(2),
                ),
                ("bf16", bf16(*a) == word(3), bf16(*a), word(3)),
            ];
            for (name, ok, expected, actual) in checks {
                if !ok {
                    failures.push(format!(
                        "{name}({:#010x}, {:#010x}, {:#010x}): expected {expected:#010x}, got {actual:#010x}",
                        a.to_bits(),
                        b.to_bits(),
                        c.to_bits()
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} cases differ on {}:\n{}",
            failures.len(),
            cases.len(),
            device.facts().name,
            failures
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    const HALVES: &str = "
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(push_constant, scalar) uniform seismic_push_t { uint64_t seismic_arguments; };
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer words_t { uint64_t w[]; };
void halves() {
    // [count][count f32 in][count * 2 out]: the f16 of input i, and the f32
    // of half i & 0xffff.
    const uint64_t base = words_t(seismic_arguments).w[0];
    const uint count = seismic_u32(base)[0].v;
    const uint i = gl_GlobalInvocationID.x;
    if (i >= count)
        return;
    seismic_u32 data = seismic_u32(base + 4ul);
    data[count + 2u * i].v = uint(seismic_f32_to_f16(uintBitsToFloat(data[i].v)));
    data[count + 2u * i + 1u].v = floatBitsToUint(seismic_f16_to_f32(uint16_t(i & 0xffffu)));
}
void main() {
    halves();
}
";

    /// Without fp16 arithmetic, the prelude's integer fp16 conversions equal
    /// the CPU library's over every half and over f32 values at, between and
    /// around every half.
    #[test]
    fn integer_f16_conversions_match_the_cpu_library() {
        use seismic_native_cpu::element::{f16_to_f32, f32_to_f16};
        let Some(description) = seismic_vulkan::discover().ok().and_then(|devices| {
            devices
                .into_iter()
                .find(|device| device.floor.is_ok() && device.facts.is_gpu())
        }) else {
            eprintln!("no Vulkan GPU meets the floor");
            return;
        };
        let device = seismic_vulkan::Device::open(description.facts.uuid).expect("device opens");
        let features = VulkanFeatures {
            subgroup_lanes: device.facts().subgroup_width().lanes(),
            float16: false,
            matrix: false,
            wide_accumulators: false,
            mixed_dot: false,
            f32_atomic_add: false,
            shared_int64_atomics: false,
        };
        let source = format!("{}{HALVES}", header(features));
        let module = VulkanToolchain::new(device.clone())
            .compile(
                &ProgramSource {
                    text: source,
                    entries: vec![ProgramEntry {
                        symbol: "halves".into(),
                        group_size: Some([64, 1, 1]),
                        constants: Vec::new(),
                    }],
                },
                None,
            )
            .expect("the conversion kernel forms");

        let mut inputs = Vec::new();
        for half in 0..=u16::MAX {
            let value = f16_to_f32(half);
            let bits = value.to_bits();
            // The half, its f32 neighbors, and the point halfway to the next
            // half up in magnitude (a tie) with its neighbors.
            let next = f16_to_f32(half.wrapping_add(1) & 0x7fff | half & 0x8000).to_bits();
            let middle = bits / 2 + next / 2 + (bits & next & 1);
            for candidate in [
                bits,
                bits.wrapping_add(1),
                bits.wrapping_sub(1),
                middle,
                middle + 1,
                middle.wrapping_sub(1),
            ] {
                inputs.push(candidate);
            }
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        while inputs.len() % 65536 != 0 || inputs.len() < 1 << 19 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            inputs.push(state as u32);
        }
        let count = inputs.len() as u32;
        let mut bytes = count.to_le_bytes().to_vec();
        for input in &inputs {
            bytes.extend_from_slice(&input.to_le_bytes());
        }
        let total = bytes.len() + inputs.len() * 8;
        let data = device.allocate(total as u64, 256, None).expect("allocation");
        device.write(&data, 0, &bytes).expect("upload");
        let mut batch = DirectBatch::new(&device).expect("batch");
        batch
            .launch(&DirectLaunch {
                module: &module,
                function: 0,
                buffers: &[],
                words: &[],
                scalar_results: (&data, 0),
                groups: [u64::from(count.div_ceil(64)), 1, 1],
            })
            .expect("launch");
        batch
            .commit()
            .and_then(|submission| submission.finish())
            .expect("run");
        let mut results = vec![0u8; inputs.len() * 8];
        device
            .read(&data, bytes.len() as u64, &mut results)
            .expect("download");
        let word = |at: usize| {
            u32::from_le_bytes(results[4 * at..4 * at + 4].try_into().expect("four bytes"))
        };
        let mut failures = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            let expected = u32::from(f32_to_f16(f32::from_bits(*input)));
            if word(2 * index) != expected {
                failures.push(format!(
                    "f32_to_f16({input:#010x}): expected {expected:#06x}, got {:#06x}",
                    word(2 * index)
                ));
            }
            let half = (index & 0xffff) as u16;
            let expected = f16_to_f32(half).to_bits();
            if word(2 * index + 1) != expected {
                failures.push(format!(
                    "f16_to_f32({half:#06x}): expected {expected:#010x}, got {:#010x}",
                    word(2 * index + 1)
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} conversions differ on {}:\n{}",
            failures.len(),
            2 * inputs.len(),
            device.facts().name,
            failures
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
