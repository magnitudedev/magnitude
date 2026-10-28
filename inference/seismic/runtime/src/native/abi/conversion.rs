//! A registered conversion's word form (`WordTable`, whose docs give the
//! table's format) as each dialect's code. The table is evaluated here, once,
//! for every backend: a kernel's front end evaluates no table. Vulkan's code
//! is straight-line (each source word and value once); Metal's and CUDA's
//! gather the members' pieces placed alike into nested loops over them,
//! which their compilers unroll (`Code`).
//!
//! - `{prefix}_UNIT(source, unit, words)`: the words of unit `unit` of the
//!   code planes' span (`WordSpan`), the codes' then the high codes', from
//!   `words[0]`. Each class's representative unit is code once; a unit runs
//!   its class's code with its runs moved by its displacements, selected by
//!   the unit.
//! - `{prefix}_FIXED(source, words)`: every record word past the span (a
//!   fixed block's), at `words[word]` (an array of a record's words).
//!
//! `source` is a source packet in memory (Metal and Vulkan: read in aligned
//! words of up to its packet alignment, each within the packet wherever a
//! unit moves it), or (CUDA) the kernel's stage of a warp's rows: the code
//! stages each segment of its tile plan (`TilePlans`), unit by unit, by the
//! stage's `load(first, length, unit, moved, index)` and `store(value,
//! length, position, unit, index)`, a batch's loads before its stores
//! (`STAGE_BATCH`), then reads its rows' staged bytes at `stage.row()`.
//! CUDA's fixed words load apart, so a kernel can issue their source loads
//! with its unit's, before either is staged (their round trip to memory then
//! overlaps the unit's): `{prefix}_FIXED_LOAD(stage, loaded)` loads the
//! `{prefix}_FIXED_LOADS` units into `loaded`, and `{prefix}_FIXED(stage,
//! loaded, words)` stages them and places the words.

use super::Dialect;
use seismic_lang::registry;
use seismic_lang::repack::{
    alignment, Group, PieceBits, WordClass, WordLookup, WordStep, WordTable, WordUnit,
};

/// How a dialect spells the code's types and operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Spelling {
    /// Vulkan is not built on macOS; its code still renders in tests there.
    #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
    Glsl,
    Msl,
    Cuda,
}

impl Spelling {
    /// `expression` as a 32-bit unsigned integer.
    fn word(self, expression: &str) -> String {
        match self {
            Self::Cuda => format!("(unsigned)({expression})"),
            // GLSL and MSL alike.
            _ => format!("uint({expression})"),
        }
    }

    /// `expression` as a 64-bit unsigned integer.
    fn wide(self, expression: &str) -> String {
        match self {
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Self::Glsl => format!("uint64_t({expression})"),
            Self::Msl => format!("ulong({expression})"),
            Self::Cuda => format!("(unsigned long long)({expression})"),
        }
    }

    fn word_type(self) -> &'static str {
        match self {
            Self::Cuda => "unsigned",
            // GLSL and MSL alike.
            _ => "uint",
        }
    }

    /// A 64-bit unsigned literal's suffix.
    fn wide_suffix(self) -> &'static str {
        match self {
            Self::Cuda => "ull",
            // GLSL and MSL alike.
            _ => "ul",
        }
    }

    /// A function's qualifiers.
    fn function(self) -> &'static str {
        match self {
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Self::Glsl => "",
            // Inlined before it is optimized: optimized alone, a function
            // is general in the constants every call gives it.
            Self::Msl => "__attribute__((always_inline)) inline ",
            Self::Cuda => "__device__ __forceinline__ ",
        }
    }

    /// The parameter of a record's words (GLSL's an array of `words`).
    #[cfg_attr(all(target_os = "macos", not(test), not(feature = "coverage")), allow(unused_variables))]
    fn words_parameter(self, words: u32) -> String {
        match self {
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Self::Glsl => format!("out uint words[{words}]"),
            Self::Msl => String::from("thread uint *words"),
            Self::Cuda => String::from("unsigned *words"),
        }
    }

    /// The parameter of the source.
    fn source_parameter(self) -> &'static str {
        match self {
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Self::Glsl => "uint64_t packet",
            Self::Msl => "device const uchar *packet",
            Self::Cuda => "const S &stage",
        }
    }

    /// A constant array of `values`, named `name`.
    fn constant_array(self, name: &str, values: &[String]) -> String {
        let values = values.join(", ");
        match self {
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Self::Glsl => format!("const uint {name}[] = uint[]({values});\n"),
            Self::Msl => format!("constant uint {name}[] = {{{values}}};\n"),
            Self::Cuda => format!("static __device__ const unsigned {name}[] = {{{values}}};\n"),
        }
    }

    /// Entry `index` of the constant array `name`.
    fn array_entry(self, name: &str, index: &str) -> String {
        match self {
            Self::Cuda => format!("__ldg(&{name}[{index}])"),
            // GLSL and MSL alike.
            _ => format!("{name}[{index}]"),
        }
    }

    /// The float bits of the product of the float bits `a` and `b`,
    /// rounded to nearest (no contraction into a fused multiply-add).
    fn f32_multiply(self, a: &str, b: &str) -> String {
        match self {
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Self::Glsl => format!(
                "floatBitsToUint(seismic_mul_rn(uintBitsToFloat({a}), uintBitsToFloat({b})))"
            ),
            Self::Msl => format!("as_type<uint>(as_type<float>({a}) * as_type<float>({b}))"),
            Self::Cuda => format!(
                "__float_as_uint(__fmul_rn(__uint_as_float({a}), __uint_as_float({b})))"
            ),
        }
    }

    /// The float bits of the binary16 bits `bits`, widened.
    fn f16_to_f32(self, bits: &str) -> String {
        match self {
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Self::Glsl => format!("floatBitsToUint(seismic_f16_to_f32(uint16_t({bits})))"),
            Self::Msl => format!("as_type<uint>(float(as_type<half>(ushort({bits}))))"),
            Self::Cuda => {
                format!("__float_as_uint(seismic_f16_to_f32((unsigned short)({bits})))")
            }
        }
    }


    /// `value` (32 bits, sign-extended from bit `32 - high`).
    fn sign_extended(self, value: &str, high: u32) -> String {
        match self {
            Self::Cuda => format!("(unsigned)((int)({value} << {high}u) >> {high}u)"),
            // GLSL and MSL alike.
            _ => format!("uint(int({value} << {high}u) >> {high}u)"),
        }
    }
}

/// The dialects whose code reads a source packet in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PacketDialect {
    #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
    Glsl,
    Msl,
}

impl PacketDialect {
    fn spelling(self) -> Spelling {
        match self {
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Self::Glsl => Spelling::Glsl,
            Self::Msl => Spelling::Msl,
        }
    }
}

/// Where a conversion's code reads its source: a source packet in memory
/// (Metal, Vulkan), or a CUDA kernel's stage of a warp's rows, which the
/// code stages by the conversion's tile plans.
enum Source {
    Packet(PacketDialect),
    Stage(TilePlans),
}

/// The rows of a CUDA kernel's tile and the lanes of its warp, whose
/// staged units a stage deals to the lanes in turn (`{prefix}_TILE_ROWS`,
/// `{prefix}_LANES`, the kernel's).
const TILE_ROWS: u32 = 16;
const LANES: u32 = 32;

/// The CUDA kernel's tile plans of a conversion's word form: the source
/// bytes a warp stages for a tile and where. A tile's words, a unit's (plan
/// `c` for a unit of class `c`, its representative's block) or at a group's
/// first tile the fixed words (the last plan, the fixed blocks), read some of
/// its rows' source bytes: part 0, the bytes of runs that stay where the
/// table puts them, and part 1 + s, the representative's bytes of the runs
/// of slot s, which a unit moves by s's byte displacement. A run's bytes are
/// its group's members', from the lowest to the highest, so a member's
/// staged bits lie as far from its representative's as in the packet (a
/// unit moves each member within the bytes it spans). The warp stages
/// exactly those, coalesced, part by part, each maximal run of used bytes (a
/// segment) at a position aligned to its copy unit: the widest that the
/// source's packet alignment (at most 4, the widest unit a kernel reads a
/// source packet in), the segment's first byte and length and its part's
/// displacements allow.
struct TilePlans {
    /// Per plan its segments, each `[part, first byte, length, position,
    /// copy unit]`.
    segments: Vec<Vec<[u32; 5]>>,
    /// Per class the staged bit of each run its representative visits, in
    /// visiting order.
    positions: Vec<Vec<u32>>,
    /// The staged byte of each source byte the fixed words read (0 for any
    /// other).
    fixed: Vec<u32>,
    /// The words of a staged row: the most any plan stages.
    stage_words: u32,
}

impl TilePlans {
    fn of(words: &WordTable, source: &registry::ExternalPacketLayout) -> Self {
        let packet_size = source.packet_size;
        let span = words.span();
        let mut segments = Vec::new();
        let mut positions: Vec<Vec<u32>> = Vec::new();
        let mut fixed = Vec::new();
        let mut most = 0;
        for class in (0..span.class_count).map(Some).chain([None]) {
            let blocks = match class {
                Some(class) => vec![words.class(class).representative],
                None => (span.units..words.blocks()).collect(),
            };
            // Each run's part and first bit, and the bytes each part uses.
            let mut used: Vec<Vec<bool>> = Vec::new();
            let mut visits = Vec::new();
            for block in blocks {
                let mut run = 0;
                for group in words.groups(block) {
                    for (source, parameter) in group.runs() {
                        let part = class.map_or(0, |class| {
                            let entry = words.class(class).entry(run);
                            if entry.still {
                                0
                            } else {
                                1 + entry.slot as usize
                            }
                        });
                        // Its members' first bits (a unit moves each within
                        // the bytes it spans).
                        let firsts = (0..group.count).map(|member| {
                            source
                                .bit
                                .wrapping_add(group.displacement(member, parameter) as u32)
                        });
                        let low = firsts.clone().min().unwrap() / 8;
                        let high = (firsts.max().unwrap() + source.length - 1) / 8;
                        if used.len() <= part {
                            used.resize(part + 1, vec![false; packet_size as usize]);
                        }
                        used[part][low as usize..=high as usize].fill(true);
                        visits.push((part, source.bit));
                        run += 1;
                    }
                }
            }
            let mut plan = Vec::new();
            let mut position = vec![vec![0u32; packet_size as usize]; used.len()];
            let mut cursor = 0u32;
            for (part, used) in used.iter().enumerate() {
                // The alignment of every byte displacement the class's units
                // give the part's slot.
                let moved = class
                    .filter(|_| part > 0)
                    .map_or(16, |class| words.slot_alignment(class, part as u32 - 1));
                for byte in 0..packet_size {
                    if !used[byte as usize] {
                        continue;
                    }
                    if byte == 0 || !used[byte as usize - 1] {
                        let length = (byte..packet_size)
                            .take_while(|at| used[*at as usize])
                            .count() as u32;
                        let unit = [byte, length, moved]
                            .into_iter()
                            .map(alignment)
                            .fold(source.packet_alignment, u32::min);
                        cursor = cursor.div_ceil(unit) * unit;
                        plan.push([part as u32, byte, length, cursor, unit]);
                    }
                    position[part][byte as usize] = cursor;
                    cursor += 1;
                }
            }
            segments.push(plan);
            most = most.max(cursor);
            match class {
                Some(_) => positions.push(
                    visits
                        .into_iter()
                        .map(|(part, bit)| 8 * position[part][bit as usize / 8] + bit % 8)
                        .collect(),
                ),
                // (Every byte 0 where the fixed words read none.)
                None => {
                    fixed = position
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| vec![0; packet_size as usize])
                }
            }
        }
        Self {
            segments,
            positions,
            fixed,
            stage_words: most.div_ceil(4),
        }
    }
}

/// `prefix`'s conversion (`words`, from a source of `layout`'s packets) as
/// `dialect`'s code (module docs). CUDA's code reads a stage, by the
/// conversion's tile plans: `{prefix}_STAGE_WORDS` the words of a staged
/// row, `{prefix}_TILE_ROWS` and `{prefix}_LANES` the tile's rows and the
/// warp's lanes the plans deal staged units to.
pub(super) fn render_conversion_code(
    out: &mut String,
    prefix: &str,
    words: &WordTable,
    layout: &registry::ExternalPacketLayout,
    dialect: Dialect,
) {
    let source = match dialect {
        Dialect::Cuda(_) => Source::Stage(TilePlans::of(words, layout)),
        Dialect::Metal(_) => Source::Packet(PacketDialect::Msl),
        #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
        Dialect::Vulkan(_) => Source::Packet(PacketDialect::Glsl),
    };
    let spelling = match &source {
        Source::Packet(dialect) => dialect.spelling(),
        Source::Stage(_) => Spelling::Cuda,
    };
    let span = words.span();
    let name = prefix.to_ascii_lowercase();
    let (alignment, packet_size) = (layout.packet_alignment, layout.packet_size);
    match &source {
        #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
        Source::Packet(PacketDialect::Glsl) => {
            // A read's alignment is a size's the packet alignment allows.
            for read in [1u32, 2, 4].into_iter().filter(|size| *size <= alignment) {
                for size in [1, 2, 4].into_iter().filter(|size| *size <= read) {
                    out.push_str(&format!(
                        "layout(buffer_reference, scalar, buffer_reference_align = {read}) readonly buffer {name}_{size}_{read} {{ uint{bits}_t v[]; }};\n",
                        bits = 8 * size
                    ));
                }
            }
        }
        Source::Packet(PacketDialect::Msl) => {}
        Source::Stage(plans) => {
            out.push_str(&format!(
                "#define {prefix}_STAGE_WORDS {}\n#define {prefix}_TILE_ROWS {TILE_ROWS}\n#define {prefix}_LANES {LANES}\n",
                plans.stage_words
            ));
            render_readers(out, &name);
        }
    }
    let lookups = render_lookups(out, &name, words, spelling);
    let template = match spelling {
        Spelling::Cuda => "template <typename S> ",
        _ => "",
    };
    let function = spelling.function();
    let new = |reads| Code::new(words, &name, reads, &lookups, alignment, packet_size);

    // The span's units, each one's class and displacements.
    out.push_str(&format!(
        "{template}{function}void {prefix}_UNIT({}, {} unit, {}) {{\n    switch (unit) {{\n",
        spelling.source_parameter(),
        spelling.word_type(),
        spelling.words_parameter(words.words())
    ));
    for index in 0..span.class_count {
        let members = words.class_units(index).collect::<Vec<_>>();
        for (unit, _) in &members {
            out.push_str(&format!("    case {unit}u:\n"));
        }
        let class = words.class(index);
        let mut code = new(match &source {
            Source::Packet(dialect) => Reads::Packet(*dialect),
            Source::Stage(plans) => Reads::Stage(Plan::Unit(plans, index)),
        });
        code.moves = Some(Moves {
            index,
            class,
            members,
        });
        code.block(class.representative);
        out.push_str("    {\n");
        for line in code.finish().1 {
            out.push_str(&format!("        {line}\n"));
        }
        out.push_str("        return;\n    }\n");
    }
    out.push_str("    }\n}\n\n");

    // The fixed blocks' words; a stage's loads of them apart (CUDA), so a
    // kernel can issue them with its unit's.
    let mut code = new(match &source {
        Source::Packet(dialect) => Reads::Packet(*dialect),
        Source::Stage(plans) => Reads::Stage(Plan::Fixed(plans)),
    });
    for block in span.units..words.blocks() {
        code.block(block);
    }
    let (loads, lines) = code.finish();
    let loaded = match source {
        Source::Stage(_) => {
            out.push_str(&format!("#define {prefix}_FIXED_LOADS {}\n", loads.len()));
            out.push_str(&format!(
                "{template}{function}void {prefix}_FIXED_LOAD(const S &stage, unsigned *loaded) {{\n"
            ));
            for line in loads {
                out.push_str(&format!("    {line}\n"));
            }
            out.push_str("}\n\n");
            ", const unsigned *loaded"
        }
        Source::Packet(_) => "",
    };
    out.push_str(&format!(
        "{template}{function}void {prefix}_FIXED({}{loaded}, {}) {{\n",
        spelling.source_parameter(),
        spelling.words_parameter(words.words())
    ));
    for line in lines {
        out.push_str(&format!("    {line}\n"));
    }
    out.push_str("}\n\n");
}

/// The functions a stage's code reads its runs by (CUDA): `length` bits
/// (at most 32) in the low bits of a word, constant-folded where the code's
/// loops unroll. `{name}_staged(row, at, length)`: from staged bit `at` of a
/// stage's row; `{name}_staged_bytes(row, byte, shift, length, bytes)`: from
/// bit `shift` of the `bytes` staged bytes from `byte` (in one 32-bit word
/// where they lie in one).
fn render_readers(out: &mut String, name: &str) {
    let mask = "length == 32u ? 0xffffffffu : (1u << length) - 1u";
    out.push_str(&format!(
            "__device__ __forceinline__ unsigned {name}_staged(const unsigned *row, unsigned at, unsigned length) {{
    const unsigned word = at >> 5, shift = at & 31u;
    const unsigned bits = shift + length > 32u ? __funnelshift_r(row[word], row[word + 1u], shift) : row[word] >> shift;
    return bits & ({mask});
}}
__device__ __forceinline__ unsigned {name}_staged_bytes(const unsigned *row, unsigned byte, unsigned shift, unsigned length, unsigned bytes) {{
    const unsigned word = byte >> 2, at = 8u * (byte & 3u) + shift;
    if ((byte & 3u) + bytes <= 4u) return (row[word] >> at) & ({mask});
    return (unsigned)(((unsigned long long)row[word + 1u] << 32 | row[word]) >> at) & ({mask});
}}
"
    ));
}

/// Each lookup table the record words' values read: a function of an
/// entry's index giving the entry, sign-extended, named by the table's
/// first word. Returns each table's width: the bits its entries' values
/// occupy (32 where one is negative: sign-extended).
fn render_lookups(out: &mut String, name: &str, words: &WordTable, spelling: Spelling) -> Vec<(WordLookup, u32)> {
    let word = spelling.word_type();
    let mut widths = Vec::new();
    for (lookup, entries, table) in words.lookups() {
        let WordLookup { first, shift } = lookup;
        let bytes = 1usize << shift;
        // The index where at most two entries differ from theirs, those
        // selected; else the entries in the fewest bits each (unsigned when
        // none is negative): held in one 64-bit constant when they fit,
        // read with a shift; else in a constant array of the table's words.
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
        widths.push((lookup, if signed { 32 } else { width }));
        // The entries that differ from their index.
        let exceptions = entries
            .iter()
            .enumerate()
            .filter(|(index, entry)| **entry != *index as i32)
            .collect::<Vec<_>>();
        let body = if exceptions.len() <= 2 {
            // Two operations an exception, fewer than a packed table's
            // 64-bit shift, truncation and mask.
            exceptions.iter().rev().fold(String::from("index"), |rest, (index, entry)| {
                format!("index == {index}u ? {:#x}u : {rest}", **entry as u32)
            })
        } else if entries.len() as u32 * width <= 64 {
            let packed = entries
                .iter()
                .enumerate()
                .fold(0u64, |packed, (index, entry)| {
                    packed
                        | (u64::from(*entry as u32) & (u64::MAX >> (64 - width)))
                            << (index as u32 * width)
                });
            let field = spelling.word(&format!(
                "{packed:#x}{} >> ({width}u * index)",
                spelling.wide_suffix()
            ));
            if signed {
                spelling.sign_extended(&field, 32 - width)
            } else {
                format!("{field} & {:#x}u", u32::MAX >> (32 - width))
            }
        } else {
            let array = format!("{name}_entries_{first}");
            out.push_str(&spelling.constant_array(
                &array,
                &table
                    .iter()
                    .map(|word| format!("{word:#x}u"))
                    .collect::<Vec<_>>(),
            ));
            let high = 32 - 8 * bytes as u32;
            let word = spelling.array_entry(&array, &format!("index >> {}u", 2 - shift));
            let entry = match shift {
                2 => word,
                _ => format!(
                    "({word} >> ((index & {}u) * {}u))",
                    (4 / bytes) - 1,
                    8 * bytes
                ),
            };
            // An entry sign-extended where one is negative, else masked.
            match (high, signed) {
                (0, _) => entry,
                (_, true) => spelling.sign_extended(&entry, high),
                (_, false) => format!("{entry} & {:#x}u", u32::MAX >> high),
            }
        };
        out.push_str(&format!(
            "{}{word} {name}_lookup_{first}({word} index) {{\n    return {body};\n}}\n",
            spelling.function()
        ));
    }
    widths
}

/// `length` bits of the word `bits` (whose bits from `width` on are zero)
/// from bit `from`, placed at bit `shift` of a word: masked unless no other
/// bit is set or the shifts drop every other bit.
fn placed(bits: String, width: u32, from: u32, length: u32, shift: u32) -> String {
    let bits = match from {
        0 => bits,
        from => format!("({bits} >> {from}u)"),
    };
    let bits = if from + length >= width.min(32) || shift + length == 32 {
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

/// Where a block's code reads its runs: a source packet in memory, or a
/// stage by a plan (CUDA).
#[derive(Clone, Copy)]
enum Reads<'a> {
    Packet(PacketDialect),
    Stage(Plan<'a>),
}

/// Where a stage's code reads its runs (CUDA): a unit of class `c` by plan
/// `c`, or the fixed words by the last plan.
#[derive(Clone, Copy)]
enum Plan<'a> {
    Unit(&'a TilePlans, u32),
    Fixed(&'a TilePlans),
}

impl<'a> Plan<'a> {
    fn plans(self) -> &'a TilePlans {
        match self {
            Self::Unit(plans, _) | Self::Fixed(plans) => plans,
        }
    }

    fn segments(self) -> &'a [[u32; 5]] {
        let plans = self.plans();
        match self {
            Self::Unit(_, class) => &plans.segments[class as usize],
            Self::Fixed(_) => plans.segments.last().expect("the fixed words have a plan"),
        }
    }
}

/// A moved run: its slot, whether the slot moves its runs by whole bytes,
/// its byte displacements over the class's units, and the alignment the
/// moved runs keep.
#[derive(Clone)]
struct Moved {
    slot: u32,
    whole: bool,
    bytes: Vec<i32>,
    alignment: u32,
}

/// A piece of a block's member, placed (looped code, `Code::placed_block`):
/// its record bit and the bits of its value it places (a source run's own
/// bits: from 0) and the runs it reads.
#[derive(Clone)]
struct Placement {
    /// Its value's record (the first of the block's with its ops: a
    /// record's runs are a member's parameters); `None` for a source run's
    /// bits.
    record: Option<u32>,
    bit: i64,
    from: u32,
    length: u32,
    reads: Vec<Read>,
}

/// A run a placement reads: its slot where a unit moves it (and whether by
/// whole bytes), its address (its first bit in the source packet, or in a
/// stage's row) and length.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Read {
    slot: Option<(u32, bool)>,
    at: i64,
    length: u32,
}

impl Placement {
    /// Whether `next` continues this piece: the next bits of its value (of
    /// its source run, read from the next bits) at the next record bits,
    /// together at most a word.
    fn continued_by(&self, next: &Self) -> bool {
        let length = i64::from(self.length);
        next.bit == self.bit + length
            && self.length + next.length <= 32
            && next.record == self.record
            && match self.record {
                Some(_) => {
                    next.from == self.from + self.length && next.reads == self.reads
                }
                None => {
                    let (read, following) = (self.reads[0], next.reads[0]);
                    following.slot == read.slot && following.at == read.at + length
                }
            }
    }

    /// The pieces placed alike, up to where they read and place: the same
    /// value's bits, from runs of the same slots and lengths.
    fn shape(&self) -> (Option<u32>, u32, u32, Vec<(Option<(u32, bool)>, u32)>) {
        (
            self.record,
            self.from,
            self.length,
            self.reads.iter().map(|read| (read.slot, read.length)).collect(),
        )
    }

    /// The slots its reads move by (where a unit moves them).
    fn slots(&self) -> Vec<Option<(u32, bool)>> {
        self.reads.iter().map(|read| read.slot).collect()
    }

    /// Where its value comes from: its reads' addresses.
    fn source(&self) -> Vec<i64> {
        self.reads.iter().map(|read| read.at).collect()
    }
}

/// A placement's parameter in emitted code: a constant, or (a nest's
/// member) an expression of the nest's counters with the values it takes.
#[derive(Clone)]
enum Parameter {
    Constant(i64),
    Affine(String, Vec<i64>),
}

impl Parameter {
    fn text(&self) -> String {
        match self {
            Self::Constant(value) => format!("{}u", *value as u32),
            Self::Affine(expression, _) => format!("({expression})"),
        }
    }

    fn values(&self) -> Vec<i64> {
        match self {
            Self::Constant(value) => vec![*value],
            Self::Affine(_, values) => values.clone(),
        }
    }
}

/// A value evaluated (`Code::values`): its record, its runs' slots (where a
/// unit moves them) and where it comes from.
type ValueKey = (u32, Vec<Option<(u32, bool)>>, Vec<i64>);

/// The staged units a lane loads before storing any (CUDA): together in
/// flight, bounded for the registers holding them. Loading and storing unit
/// by unit serializes the loads (measured against A7, which batches 16:
/// q4_k mma16 1.19x and q6_k rows16 1.18x its time, within 2% batched).
const STAGE_BATCH: usize = 16;

/// Blocks' code: its lines. Vulkan's is straight-line, each source word and
/// value once (NVIDIA's driver leaves `[[unroll]]` loops rolled). Metal's
/// and CUDA's place pieces (`placed_block`): those placed alike whose record
/// bits and reads are affine in their order as nested loops, which
/// their compilers unroll, so their front ends compile one member's code,
/// not each's.
struct Code<'a> {
    words: &'a WordTable,
    name: &'a str,
    reads: Reads<'a>,
    /// Each lookup table's width (`render_lookups`).
    lookups: &'a [(WordLookup, u32)],
    /// A source packet's alignment and bytes.
    alignment: u32,
    packet_size: u32,
    moves: Option<Moves<'a>>,
    /// The runs visited so far (`WordSpan`'s order).
    visited: usize,
    /// Straight-line code's record words' terms.
    terms: Vec<(u32, String)>,
    lines: Vec<String>,
    /// The values evaluated, each once: each its record, the slots its runs
    /// move by, and where it comes from (straight-line code: its member's
    /// displacements of the value's runs; placed code:
    /// `Placement::source`). With each, the width of its bits (those above
    /// are zero).
    values: Vec<(ValueKey, u32)>,
    /// The declarations of the moved slots' displacements, before every
    /// other line (a stage moves by them; a loop reads them).
    displacements: Vec<String>,
    /// Locals and nests named so far.
    locals: u32,
    declared: Vec<String>,
}

impl<'a> Code<'a> {
    fn new(
        words: &'a WordTable,
        name: &'a str,
        reads: Reads<'a>,
        lookups: &'a [(WordLookup, u32)],
        alignment: u32,
        packet_size: u32,
    ) -> Self {
        Self {
            words,
            name,
            reads,
            lookups,
            alignment,
            packet_size,
            moves: None,
            visited: 0,
            terms: Vec::new(),
            lines: Vec::new(),
            values: Vec::new(),
            displacements: Vec::new(),
            locals: 0,
            declared: Vec::new(),
        }
    }

    /// How the code's dialect spells it.
    fn spelling(&self) -> Spelling {
        match self.reads {
            Reads::Packet(dialect) => dialect.spelling(),
            Reads::Stage(_) => Spelling::Cuda,
        }
    }

    /// The words of block `block`, at their record words (a unit's from
    /// `words[0]`, a fixed block's at their record words): Vulkan's
    /// straight-line (`straight_block`), the others' placed pieces
    /// (`placed_block`).
    fn block(&mut self, block: u32) {
        match self.reads {
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Reads::Packet(PacketDialect::Glsl) => self.straight_block(block),
            _ => self.placed_block(block),
        }
    }

    /// Block `block` as straight-line code: the OR of its groups' members.
    #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
    fn straight_block(&mut self, block: u32) {
        let words = self.words;
        let record = words.block_words(block);
        let count = record.len();
        let unit = self.moves.is_some();
        let at = move |word: u32| if unit { word } else { record[word as usize] };
        let mut terms: Vec<Vec<String>> = vec![Vec::new(); count];
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
            self.terms.push((at(word as u32), value));
        }
    }

    /// The code's lines: the stage's segments staged (CUDA), its source
    /// words and values, then its record words; and apart from them (the
    /// fixed plan's) the stage's loads, each into `loaded[i]`.
    fn finish(self) -> (Vec<String>, Vec<String>) {
        let mut lines = Vec::new();
        let mut loads = Vec::new();
        if let Reads::Stage(plan) = self.reads {
            // Each lane's staged units of the plan's segments: a unit plan's
            // in batches of STAGE_BATCH, a batch's loads, then its stores; the
            // fixed plan's loaded apart (into `loaded`, the kernel's), then
            // stored.
            let units = plan
                .segments()
                .iter()
                .flat_map(|&[part, first, length, position, unit]| {
                    let moved = match part {
                        0 => String::from("0"),
                        part => {
                            let slot = part - 1;
                            assert!(
                                self.declared.contains(&format!("d{slot}")),
                                "a moved part's slot is read"
                            );
                            format!("d{slot}b")
                        }
                    };
                    (0..(TILE_ROWS * (length / unit)).div_ceil(LANES)).map(move |index| {
                        (
                            format!("{first}u, {length}u, {unit}u, {moved}, {index}u"),
                            format!("{length}u, {position}u, {unit}u, {index}u"),
                        )
                    })
                })
                .collect::<Vec<_>>();
            match plan {
                Plan::Unit(..) => {
                    for (batch, units) in units.chunks(STAGE_BATCH).enumerate() {
                        for (index, (load, _)) in units.iter().enumerate() {
                            lines.push(format!(
                                "const unsigned s{batch}_{index} = stage.load({load});"
                            ));
                        }
                        for (index, (_, store)) in units.iter().enumerate() {
                            lines.push(format!("stage.store(s{batch}_{index}, {store});"));
                        }
                    }
                }
                Plan::Fixed(_) => {
                    for (index, (load, store)) in units.iter().enumerate() {
                        loads.push(format!("loaded[{index}] = stage.load({load});"));
                        lines.push(format!("stage.store(loaded[{index}], {store});"));
                    }
                }
            }
            lines.push(String::from("const unsigned *row = stage.row();"));
        }
        lines.splice(0..0, self.displacements);
        lines.extend(self.lines);
        for (word, value) in self.terms {
            lines.push(format!("words[{word}] = {value};"));
        }
        (loads, lines)
    }

    /// Member `m` of `group`, placed (straight-line code).
    #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
    fn member(&mut self, group: Group, m: u32) -> String {
        let piece = group.piece;
        let shift = piece.shift.wrapping_add(group.displacement(m, 1) as u32);
        match piece.bits {
            PieceBits::Source(bit) => {
                let bit = bit.wrapping_add(group.displacement(m, 2) as u32);
                let bits = self.source_run(bit, piece.length);
                placed(bits, piece.length, 0, piece.length, shift)
            }
            PieceBits::Value { record, bit } => {
                let (value, width) = self.value(group, record, m);
                placed(value, width, bit, piece.length, shift)
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

    /// The slot of run `run` (`WordSpan`'s order) when a unit moves it, and
    /// whether by whole bytes.
    fn slot(&self, run: usize) -> Option<(u32, bool)> {
        let entry = self.moves.as_ref()?.class.entry(run);
        (!entry.still).then_some((entry.slot, entry.whole))
    }

    /// Slot `slot`'s movement (by whole bytes: `whole`), its displacements
    /// declared once: the unit's, selected by the unit.
    fn moved(&mut self, (slot, whole): (u32, bool)) -> Moved {
        let moves = self.moves.as_ref().expect("a moved slot is a unit's");
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
        if !self.declared.contains(&format!("d{slot}")) {
            self.declared.push(format!("d{slot}"));
            self.displacements.push(line);
        }
        Moved {
            slot,
            whole,
            bytes,
            alignment,
        }
    }

    /// The `length` bits of the next run visited from source bit `bit` (a
    /// member's), in the low bits of a word (straight-line code).
    #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
    fn source_run(&mut self, bit: u32, length: u32) -> String {
        let moved = self.slot(self.visited).map(|slot| self.moved(slot));
        self.visited += 1;
        self.packet_run(bit, length, moved)
    }

    /// A run from a source packet in memory (straight-line code): the
    /// aligned source words holding it (of one size, the widest within the
    /// packet wherever a unit moves them), little-endian, shifted down and
    /// masked unless the words end with the run. A moved run's words are
    /// read from the packet moved by its slot's byte displacement, and
    /// shifted by its bit displacement more.
    fn packet_run(&mut self, bit: u32, length: u32, moved: Option<Moved>) -> String {
        let spelling = self.spelling();
        let (low_byte, high_byte) = moved.as_ref().map_or((0, 0), |moved| {
            (
                *moved.bytes.iter().min().unwrap(),
                *moved.bytes.iter().max().unwrap(),
            )
        });
        // Alignment the moved packets keep.
        let alignment = moved
            .as_ref()
            .map_or(self.alignment, |moved| self.alignment.min(moved.alignment));
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
        let words = (start / size..end / size)
            .map(|index| {
                let load = self.packet_word(moved.as_ref(), size, alignment, index);
                let load = if wide { spelling.wide(&load) } else { load };
                match 8 * (index * size - start) {
                    0 => load,
                    offset => format!("({load} << {offset}u)"),
                }
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let low = bit - 8 * start;
        let shift = match &moved {
            Some(Moved {
                slot, whole: false, ..
            }) => Some(format!("{low}u + {}", spelling.word(&format!("d{slot}s")))),
            _ => (low != 0).then(|| format!("{low}u")),
        };
        let shifted = match (shift, wide) {
            (None, false) => words,
            (None, true) => spelling.word(&words),
            (Some(shift), false) => format!("(({words}) >> ({shift}))"),
            (Some(shift), true) => spelling.word(&format!("({words}) >> ({shift})")),
        };
        let exact = matches!(moved, None | Some(Moved { whole: true, .. })) && low + length == span;
        if exact || length == 32 {
            shifted
        } else {
            format!("({shifted} & {:#x}u)", u32::MAX >> (32 - length))
        }
    }

    /// Word `index` of `size` bytes (aligned to `alignment`) of a source
    /// packet, moved by `moved`'s slot: a local, read once.
    fn packet_word(&mut self, moved: Option<&Moved>, size: u32, alignment: u32, index: u32) -> String {
        let spelling = self.spelling();
        let view = match moved {
            None => format!("source{size}_{alignment}"),
            Some(moved) => format!("moved{}_{size}_{alignment}", moved.slot),
        };
        let declaration = match (spelling, moved) {
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            (Spelling::Glsl, None) => format!("{name}_{size}_{alignment} {view} = {name}_{size}_{alignment}(packet);", name = self.name),
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            (Spelling::Glsl, Some(moved)) => format!(
                "{name}_{size}_{alignment} {view} = {name}_{size}_{alignment}(packet + uint64_t(int64_t(d{}b)));",
                moved.slot,
                name = self.name
            ),
            (_, moved) => {
                let base = moved.map_or_else(|| String::from("packet"), |moved| format!("packet + d{}b", moved.slot));
                let word = ["uchar", "ushort", "", "uint"][size as usize - 1];
                format!("device const {word} *{view} = (device const {word} *)({base});")
            }
        };
        self.declare(view.clone(), declaration);
        let local = format!("{view}_{index}");
        let load = match spelling {
            #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
            Spelling::Glsl => spelling.word(&format!("{view}.v[{index}]")),
            _ => spelling.word(&format!("{view}[{index}]")),
        };
        self.declare(local.clone(), format!("const uint {local} = {load};"));
        local
    }

    /// A local holding member `m` of `group`'s value whose record is at
    /// `record` (straight-line code): its postfix ops, visiting its runs,
    /// its runs displaced; evaluated on first use, once.
    #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
    fn value(&mut self, group: Group, record: u32, m: u32) -> (String, u32) {
        let runs = group.runs().len() as u32;
        let key = (
            record,
            (0..runs as usize).map(|run| self.slot(self.visited + run)).collect(),
            (2..2 + runs)
                .map(|parameter| i64::from(group.displacement(m, parameter)))
                .collect::<Vec<_>>(),
        );
        if let Some(index) = self.values.iter().position(|(known, _)| *known == key) {
            return (format!("v{index}"), self.values[index].1);
        }
        let mut parameter = 2;
        let result = self.evaluate(record, |code, bit, length| {
            parameter += 1;
            let delta = group.displacement(m, parameter - 1);
            (
                code.source_run(bit.wrapping_add(delta as u32), length),
                u32::MAX >> (32 - length),
            )
        });
        (self.local_value(key, &result), result.1)
    }

    /// `value` named `v{n}` for `key`, evaluated once.
    fn local_value(&mut self, key: ValueKey, (value, width): &(String, u32)) -> String {
        let name = format!("v{}", self.values.len());
        self.values.push((key, *width));
        self.lines.push(format!("const {} {name} = {value};", self.spelling().word_type()));
        name
    }

    /// The value whose record is at `record`: its postfix ops, each source
    /// run (its representative's first bit and length) by `operand` (in op
    /// order: its text and the largest value it can be); with the width of
    /// its bits (those above are zero). Placed code takes its members' runs
    /// from their placements; only straight-line code (Vulkan's) reads the
    /// representative's.
    fn evaluate(&mut self, record: u32, mut operand: impl FnMut(&mut Self, u32, u32) -> (String, u32)) -> (String, u32) {
        let spelling = self.spelling();
        let mut stack: Vec<(String, u32)> = Vec::new();
        for step in self.words.steps(record) {
            let pushed = match step {
                WordStep::Field(pieces) => {
                    let pieces = pieces.into_iter().collect::<Vec<_>>();
                    (
                        format!(
                            "({})",
                            pieces
                                .iter()
                                .map(|piece| {
                                    let (bits, _) = operand(self, piece.bit, piece.length);
                                    placed(bits, piece.length, 0, piece.length, piece.shift)
                                })
                                .collect::<Vec<_>>()
                                .join(" | ")
                        ),
                        pieces.iter().map(|piece| piece.shift + piece.length).max().unwrap_or(0),
                    )
                }
                WordStep::Or => {
                    let (right, right_width) = stack.pop().expect("an operand");
                    let (left, left_width) = stack.pop().expect("an operand");
                    (format!("({left} | {right})"), left_width.max(right_width))
                }
                WordStep::ShiftLeft(bits) => {
                    let (value, width) = stack.pop().expect("an operand");
                    (format!("({value} << {bits}u)"), 32.min(width + bits))
                }
                // Digit `digit` of the base-3 fraction in the low byte, in
                // 0..=2 (`seismic_lang::repack::base3_digit`).
                WordStep::Base3Digit(digit) => {
                    let (value, _) = stack.pop().expect("an operand");
                    (format!("(((({value} * {}u) & 0xffu) * 3u) >> 8u)", 3u32.pow(digit)), 2)
                }
                WordStep::F16ToF32 => (spelling.f16_to_f32(&stack.pop().expect("an operand").0), 32),
                // Times 2^exponent, a normal float built from its exponent
                // bits.
                WordStep::ScaleF32(exponent) => (
                    spelling.f32_multiply(
                        &stack.pop().expect("an operand").0,
                        &format!("{:#x}u", ((127 + exponent) as u32) << 23),
                    ),
                    32,
                ),
                WordStep::Lookup(lookup) => (
                    format!(
                        "{}_lookup_{}({})",
                        self.name,
                        lookup.first,
                        stack.pop().expect("an operand").0
                    ),
                    self.lookups
                        .iter()
                        .find(|(known, _)| *known == lookup)
                        .expect("a value's lookup is rendered")
                        .1,
                ),
            };
            stack.push(pushed);
        }
        let result = stack.pop().expect("a value");
        assert!(stack.is_empty(), "a value's ops leave one word");
        result
    }

    /// Block `block` as placed pieces (Metal, CUDA): its members' pieces,
    /// each continued by the next bits of its value or source run at the
    /// next record bits merged into one (a piece packed across two record
    /// words is one), those placed alike (`Placement::shape`) gathered in
    /// record order, those of one value at the same sources computing it
    /// once; each gathering as nested loops where its record bits and
    /// sources are affine in its order (`nest`), else piece by piece
    /// (measured: cheaper to compile than runs of short loops).
    fn placed_block(&mut self, block: u32) {
        let words = self.words;
        let record = words.block_words(block);
        let count = record.len();
        let unit = self.moves.is_some();
        let at = move |word: u32| if unit { word } else { record[word as usize] };
        for word in 0..count as u32 {
            self.lines.push(format!("words[{}] = 0u;", at(word)));
        }
        let mut pieces: Vec<Placement> = Vec::new();
        let mut records: Vec<(u32, Vec<String>)> = Vec::new();
        for group in words.groups(block) {
            let runs = group.runs();
            let (record, from) = match group.piece.bits {
                PieceBits::Source(_) => (None, 0),
                PieceBits::Value { record, bit } => (
                    Some({
                        let ops = value_ops(words, record);
                        match records.iter().find(|(_, known)| *known == ops) {
                            Some((first, _)) => *first,
                            None => {
                                records.push((record, ops));
                                record
                            }
                        }
                    }),
                    bit,
                ),
            };
            for m in 0..group.count {
                let word = at(group.word.wrapping_add(group.displacement(m, 0) as u32));
                let shift = group.piece.shift.wrapping_add(group.displacement(m, 1) as u32);
                pieces.push(Placement {
                    record,
                    bit: 32 * i64::from(word) + i64::from(shift),
                    from,
                    length: group.piece.length,
                    reads: runs
                        .iter()
                        .enumerate()
                        .map(|(index, (run, parameter))| {
                            let run_index = self.visited + index;
                            let bit = run.bit.wrapping_add(group.displacement(m, *parameter) as u32);
                            Read {
                                slot: self.slot(run_index),
                                at: self.address(run_index, run.bit, bit),
                                length: run.length,
                            }
                        })
                        .collect(),
                });
            }
            self.visited += runs.len();
        }
        // Pieces continued by others, merged.
        let mut merged: Vec<Placement> = Vec::new();
        let mut used = vec![false; pieces.len()];
        for first in 0..pieces.len() {
            if used[first] {
                continue;
            }
            used[first] = true;
            let mut piece = pieces[first].clone();
            while let Some(next) =
                (0..pieces.len()).find(|next| !used[*next] && piece.continued_by(&pieces[*next]))
            {
                used[next] = true;
                piece.length += pieces[next].length;
                if piece.record.is_none() {
                    piece.reads[0].length += pieces[next].length;
                }
            }
            merged.push(piece);
        }
        // Gathered by shape, in record order.
        let mut gatherings: Vec<Vec<Placement>> = Vec::new();
        for piece in merged {
            match gatherings.iter_mut().find(|gathering| gathering[0].shape() == piece.shape()) {
                Some(gathering) => gathering.push(piece),
                None => gatherings.push(vec![piece]),
            }
        }
        for gathering in &mut gatherings {
            gathering.sort_by_key(|piece| piece.bit);
        }
        // Gatherings of one value at the same sources, together.
        let mut done = vec![false; gatherings.len()];
        for first in 0..gatherings.len() {
            if done[first] {
                continue;
            }
            let together = (first..gatherings.len())
                .filter(|other| {
                    *other == first
                        || (gatherings[first][0].record.is_some()
                            && gatherings[*other][0].record == gatherings[first][0].record
                            && gatherings[*other][0].slots() == gatherings[first][0].slots()
                            && gatherings[*other].len() == gatherings[first].len()
                            && gatherings[*other]
                                .iter()
                                .zip(&gatherings[first])
                                .all(|(other, piece)| other.source() == piece.source()))
                })
                .collect::<Vec<_>>();
            for other in &together {
                done[*other] = true;
            }
            let together = together.iter().map(|index| &gatherings[*index]).collect::<Vec<_>>();
            self.place(&together);
        }
    }

    /// Where run `run` (a representative's from source bit `representative`)
    /// is read for a member's source bit `bit`: the packet's bit, or the
    /// stage's (CUDA): a unit's run's members staged together from its
    /// representative's staged bit, the fixed words' bytes each at its own.
    fn address(&self, run: usize, representative: u32, bit: u32) -> i64 {
        i64::from(match self.reads {
            Reads::Packet(_) => bit,
            Reads::Stage(Plan::Unit(plans, class)) => {
                plans.positions[class as usize][run].wrapping_add(bit.wrapping_sub(representative))
            }
            Reads::Stage(Plan::Fixed(plans)) => 8 * plans.fixed[bit as usize / 8] + bit % 8,
        })
    }

    /// Gatherings of one value at the same sources (`placed_block`), their
    /// members together: as a nest, or member by member.
    fn place(&mut self, together: &[&Vec<Placement>]) {
        // Each member's record bit in each gathering, then its sources.
        let members = (0..together[0].len())
            .map(|m| {
                together
                    .iter()
                    .map(|gathering| gathering[m].bit)
                    .chain(together[0][m].source())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        match nest(&members).filter(|loops| !loops.is_empty()) {
            Some(loops) => {
                let index = self.locals;
                self.locals += 1;
                let counter = |level: usize| format!("m{index}_{level}");
                // Each parameter: the first member's plus each loop's
                // counter times its stride (wrapping, as a 32-bit word).
                let parameters = (0..members[0].len())
                    .map(|parameter| {
                        let strides = loops.iter().enumerate().filter(|(_, (_, strides))| strides[parameter] != 0);
                        if strides.clone().count() == 0 {
                            return Parameter::Constant(members[0][parameter]);
                        }
                        Parameter::Affine(
                            strides.fold(format!("{}u", members[0][parameter] as u32), |sum, (level, (_, strides))| {
                                format!("{sum} + {}u * {}", strides[parameter] as u32, counter(level))
                            }),
                            members.iter().map(|member| member[parameter]).collect(),
                        )
                    })
                    .collect::<Vec<_>>();
                let outer = std::mem::take(&mut self.lines);
                self.place_member(together, &parameters, true);
                let body = std::mem::replace(&mut self.lines, outer);
                let unroll = match self.spelling() {
                    Spelling::Msl => "_Pragma(\"clang loop unroll(full)\")",
                    _ => "_Pragma(\"unroll\")",
                };
                let word_type = self.spelling().word_type();
                let depth = loops.len();
                for (level, (extent, _)) in loops.iter().enumerate().rev() {
                    let indent = "    ".repeat(depth - 1 - level);
                    self.lines.push(format!("{indent}{unroll}"));
                    self.lines.push(format!(
                        "{indent}for ({word_type} {m} = 0u; {m} < {extent}u; ++{m}) {{",
                        m = counter(level)
                    ));
                }
                let indent = "    ".repeat(depth);
                for line in body {
                    self.lines.push(format!("{indent}{line}"));
                }
                for level in 0..depth {
                    self.lines.push(format!("{}}}", "    ".repeat(depth - 1 - level)));
                }
            }
            _ => {
                for member in members {
                    let parameters = member.into_iter().map(Parameter::Constant).collect::<Vec<_>>();
                    self.place_member(together, &parameters, false);
                }
            }
        }
    }

    /// A member of `together` (its record bits in each gathering, then its
    /// sources: `parameters`) placed: its value (or source run) once, each
    /// gathering's bits of it ORed into the record at its record bit (and
    /// the next word, where they cross into it). `looped`: a loop's body,
    /// whose reads are its own.
    fn place_member(&mut self, together: &[&Vec<Placement>], parameters: &[Parameter], looped: bool) {
        let piece = &together[0][0];
        let sources = &parameters[together.len()..];
        let value = match piece.record {
            None => (self.read(piece.reads[0], &sources[0], looped), piece.length),
            Some(record) => {
                // (A loop's body's value is its own.)
                let key = (
                    record,
                    piece.slots(),
                    sources.iter().flat_map(Parameter::values).collect::<Vec<_>>(),
                );
                match self.values.iter().position(|(known, _)| !looped && *known == key) {
                    Some(index) => (format!("v{index}"), self.values[index].1),
                    None => {
                        // Runs come in op order.
                        let mut reads = piece.reads.iter().copied().zip(sources.iter().cloned());
                        let value = self.evaluate(record, |code, _, length| {
                            let (read, at) = reads.next().expect("a value's runs are read");
                            (code.read(read, &at, looped), u32::MAX >> (32 - length))
                        });
                        match looped {
                            true => {
                                let local = format!("u{}", self.locals);
                                self.locals += 1;
                                self.lines.push(format!("const {} {local} = {};", self.spelling().word_type(), value.0));
                                (local, value.1)
                            }
                            false => (self.local_value(key, &value), value.1),
                        }
                    }
                }
            }
        };
        for (gathering, bit) in together.iter().zip(parameters) {
            let piece = &gathering[0];
            // (A source run's read is its bits.)
            let bits = match piece.record {
                None => value.0.clone(),
                Some(_) => placed(value.0.clone(), value.1, piece.from, piece.length, 0),
            };
            let crosses = bit.values().iter().any(|bit| bit % 32 + i64::from(piece.length) > 32);
            match bit {
                Parameter::Constant(bit) => {
                    let (word, shift) = (bit / 32, *bit as u32 % 32);
                    self.lines.push(format!(
                        "words[{word}] |= {};",
                        placed(bits.clone(), piece.length, 0, piece.length, shift)
                    ));
                    if crosses {
                        self.lines.push(format!("words[{}] |= ({bits} >> {}u);", word + 1, 32 - shift));
                    }
                }
                Parameter::Affine(..) => {
                    let bit = bit.text();
                    let local = format!("p{}", self.locals);
                    self.locals += 1;
                    self.lines.push(format!("const {} {local} = {bits};", self.spelling().word_type()));
                    self.lines.push(format!("words[{bit} >> 5u] |= {local} << ({bit} & 31u);"));
                    if crosses {
                        self.lines.push(format!(
                            "if (({bit} & 31u) + {}u > 32u) words[({bit} >> 5u) + 1u] |= {local} >> (32u - ({bit} & 31u));",
                            piece.length
                        ));
                    }
                }
            }
        }
    }

    /// `read`'s bits at `at` (a member's address), in the low bits of a
    /// word: a stage's row's (CUDA), or a source packet's (Metal: within a
    /// loop's body read there, else from the shared words).
    fn read(&mut self, read: Read, at: &Parameter, looped: bool) -> String {
        let moved = read.slot.map(|slot| self.moved(slot));
        let length = read.length;
        match (self.reads, at) {
            (Reads::Stage(_), at) => {
                let text = at.text();
                match &moved {
                    Some(Moved {
                        slot, whole: false, ..
                    }) => {
                        let bytes = at
                            .values()
                            .iter()
                            .map(|at| (*at as u32 % 8 + length).div_ceil(8))
                            .max()
                            .expect("a read has members");
                        format!(
                            "{}_staged_bytes(row, {text} >> 3u, ({text} & 7u) + (unsigned)d{slot}s, {length}u, {bytes}u)",
                            self.name
                        )
                    }
                    _ => format!("{}_staged(row, {text}, {length}u)", self.name),
                }
            }
            (Reads::Packet(_), Parameter::Constant(bit)) if !looped => self.packet_run(*bit as u32, length, moved),
            (Reads::Packet(_), at) => self.packet_bits(at, length, moved),
        }
    }

    /// A run from a source packet in memory in a loop's body (Metal): the
    /// same number of aligned words for every member the loop visits (the
    /// most any needs), of the widest size the packet's (and a moved
    /// slot's) alignment allows that keeps every member's words within the
    /// packet, from the word holding the member's first bit, shifted down
    /// by its bit within it (and a bit-moved slot's displacement more: it
    /// stays within the bytes the member's run spans), masked.
    fn packet_bits(&mut self, at: &Parameter, length: u32, moved: Option<Moved>) -> String {
        let firsts = at.values();
        let (low, high) = moved.as_ref().map_or((0, 0), |moved| {
            (
                i64::from(*moved.bytes.iter().min().unwrap()),
                i64::from(*moved.bytes.iter().max().unwrap()),
            )
        });
        let alignment = moved
            .as_ref()
            .map_or(self.alignment, |moved| self.alignment.min(moved.alignment));
        let (size, count) = [4u32, 2, 1]
            .into_iter()
            .filter(|size| *size <= alignment)
            .map(|size| {
                let unit = i64::from(8 * size);
                let count = firsts
                    .iter()
                    .map(|first| (first % unit + i64::from(length) + unit - 1) / unit)
                    .max()
                    .expect("a run has members");
                (size, count)
            })
            .find(|(size, count)| {
                let unit = i64::from(8 * size);
                firsts.iter().all(|first| {
                    let start = first / unit * i64::from(*size);
                    start + low >= 0 && start + count * i64::from(*size) + high <= i64::from(self.packet_size)
                })
            })
            .expect("a byte's words stay within the packet");
        let base = match &moved {
            None => String::from("packet"),
            Some(moved) => format!("packet + d{}b", moved.slot),
        };
        let local = format!("r{}", self.locals);
        self.locals += 1;
        self.lines.push(format!("const uint {local} = {};", at.text()));
        let unit = 8 * size;
        let wide = count * i64::from(unit) > 32;
        let word = ["uchar", "ushort", "", "uint"][size as usize - 1];
        let words = (0..count)
            .map(|index| {
                let load = format!(
                    "((device const {word} *)({base}))[({local} >> {}u) + {index}u]",
                    unit.trailing_zeros()
                );
                let load = if wide { self.spelling().wide(&load) } else { self.spelling().word(&load) };
                match index as u32 * unit {
                    0 => load,
                    offset => format!("({load} << {offset}u)"),
                }
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let shift = match &moved {
            Some(Moved {
                slot, whole: false, ..
            }) => format!("({local} & {}u) + {}", unit - 1, self.spelling().word(&format!("d{slot}s"))),
            _ => format!("({local} & {}u)", unit - 1),
        };
        let shifted = self.spelling().word(&format!("({words}) >> ({shift})"));
        if length == 32 {
            shifted
        } else {
            format!("({shifted} & {:#x}u)", u32::MAX >> (32 - length))
        }
    }
}

/// The ops of the value whose record is at `record`, but for its runs'
/// first bits.
fn value_ops(words: &WordTable, record: u32) -> Vec<String> {
    words
        .steps(record)
        .map(|step| match step {
            WordStep::Field(pieces) => format!(
                "field {:?}",
                pieces.into_iter().map(|piece| (piece.length, piece.shift)).collect::<Vec<_>>()
            ),
            step => format!("{step:?}"),
        })
        .collect()
}

/// `members` (each member's parameters, in order) as nested loops, from
/// the innermost: each its extent and its stride of each parameter, every
/// member's parameters the first's plus each loop's counter times its
/// stride. The innermost loop is the longest whose members are affine in
/// it alike in every block of it; `None` where no loop of more than one
/// member is (a prime count of members, not affine).
fn nest(members: &[Vec<i64>]) -> Option<Vec<(usize, Vec<i64>)>> {
    let count = members.len();
    if count == 1 {
        return Some(Vec::new());
    }
    (2..=count).rev().filter(|extent| count % extent == 0).find_map(|extent| {
        let strides = members[1]
            .iter()
            .zip(&members[0])
            .map(|(second, first)| second - first)
            .collect::<Vec<_>>();
        let affine = members.chunks(extent).all(|block| {
            block.iter().enumerate().all(|(step, member)| {
                member
                    .iter()
                    .zip(&block[0])
                    .zip(&strides)
                    .all(|((value, first), stride)| *value == first + step as i64 * stride)
            })
        });
        if !affine {
            return None;
        }
        let starts = members.iter().step_by(extent).cloned().collect::<Vec<_>>();
        let outer = nest(&starts)?;
        Some(std::iter::once((extent, strides)).chain(outer).collect())
    })
}
