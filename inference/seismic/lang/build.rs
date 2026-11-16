//! Each backend's tables header (`<seismic/tables.*>`), rendered from ggml's
//! tables (`magnitude_ggml::tables`): the ids of its grids and codebooks
//! (`SEISMIC_GRID_<NAME>`, `SEISMIC_CODEBOOK_<NAME>`: position + 1, so that 0
//! names none, which a bound element's decode axes name), each grid's points
//! and each codebook's values (on Metal and Vulkan, which read them), and
//! their bindings by id: `grids::table<ID>::points()` and
//! `codebooks::table<ID>::values()` (Vulkan `GRIDS_TABLE_<ID>`,
//! `CODEBOOKS_TABLE_<ID>`). A
//! `u32` grid point is one word and a `u64` point its (low, high) words;
//! codebook values are exact floats. A 16-value codebook also has a dialect's own forms of its values:
//! on Metal each code byte's two values + the zero its lane producer enters
//! codebook values at (`metal/producers.decl`; `codebooks::pairs<ID>`: the
//! low nibble's in the low byte), on CUDA its values as little-endian s8
//! words (`codebooks::words<ID>`: `W0`..`W3`, values 4w..4w + 3 in word w)
//! and whether they are E2M1 doubled (`codebooks::e2m1<ID>`). On CUDA a grid
//! whose lane magnitudes are few also has the operands of its magnitudes
//! (`grids::magnitudes<ID>`, see `magnitudes`). On Vulkan each table's
//! `u32` words (`GRIDS_WORDS_<ID>`, `CODEBOOKS_WORDS_<ID>`, 0 for id 0) size
//! the copy its library stages in shared memory. A dialect declares some
//! tables only where its includer reads them (`Dialect::guarded`), each
//! under `SEISMIC_TABLES_READ_<TABLE>(ID)`, which the includer defines.

use magnitude_ggml::tables::{CODEBOOKS, GRIDS, GridPoints};
use std::fmt::Write as _;

#[path = "src/native_name.rs"]
mod native_name;
#[path = "src/producer_decl.rs"]
mod producer_decl;
use native_name::native_macro;

/// A table space: its namespace and the accessor of a table's elements.
#[derive(Clone, Copy)]
enum Space {
    Grids,
    Codebooks,
}

impl Space {
    fn name(self) -> &'static str {
        match self {
            Self::Grids => "grids",
            Self::Codebooks => "codebooks",
        }
    }

    fn accessor(self) -> &'static str {
        match self {
            Self::Grids => "points",
            Self::Codebooks => "values",
        }
    }
}

/// A kind of table: a grid's points, a codebook's values, a grid's
/// magnitude operands.
#[derive(Clone, Copy, PartialEq)]
enum Table {
    Grid,
    Codebook,
    Magnitudes,
}

impl Table {
    /// The condition its table `id` is declared under where a dialect
    /// declares it only where read (`Dialect::guarded`).
    fn read(self, id: usize) -> String {
        let table = match self {
            Self::Grid => "GRID",
            Self::Codebook => "CODEBOOK",
            Self::Magnitudes => "MAGNITUDES",
        };
        format!("SEISMIC_TABLES_READ_{table}({id})")
    }
}

/// A table element: a four-lane grid point (one word), an eight-lane grid
/// point (two words) or a codebook value.
#[derive(Clone, Copy)]
enum Element {
    Word,
    Pair,
    Float,
}

struct Dialect {
    file: &'static str,
    begin: &'static str,
    end: &'static str,
    /// The dialect's type of an element.
    ty: fn(Element) -> &'static str,
    /// Array `space::name` of `len` elements of type `ty`: its opening and
    /// its closing.
    open: fn(space: Space, name: &str, ty: &str, len: usize) -> String,
    close: &'static str,
    /// A `u64` grid point as its (low, high) words.
    pair: fn(u32, u32) -> String,
    /// The binding of table `id` of `space` to array `space::name` of type
    /// `ty`.
    bind: fn(space: Space, id: usize, name: &str, ty: &str) -> String,
    /// A 16-value codebook's dialect forms, where `zero` is the code its
    /// values enter Metal's lanes at (`metal/producers.decl`).
    sixteen: fn(id: usize, name: &str, values: [i8; 16], zero: i32) -> String,
    /// A grid's dialect forms.
    grid: fn(id: usize, points: GridPoints) -> String,
    /// Whether the dialect reads codebooks' values themselves (CUDA reads
    /// only their dialect forms).
    values: bool,
    /// The tables the dialect declares only where its includer reads them
    /// (`Table::read`).
    guarded: &'static [Table],
    /// Whether the dialect states each table's `u32` words
    /// (`GRIDS_WORDS_<ID>`, `CODEBOOKS_WORDS_<ID>`, 0 for id 0), which its
    /// library stages in shared memory.
    words: bool,
}

impl Dialect {
    /// Table `id` of kind `table`, declared by `declaration`: under its read
    /// condition where the dialect guards it.
    fn declare(&self, table: Table, id: usize, declaration: String) -> String {
        if declaration.is_empty() || !self.guarded.contains(&table) {
            declaration
        } else {
            format!("#if {}\n{declaration}#endif\n", table.read(id))
        }
    }
}

const DIALECTS: [Dialect; 3] = [
    Dialect {
        file: "tables.h",
        begin: "namespace grids { template <uint ID> struct table; }\nnamespace codebooks { template <uint ID> struct table; template <uint ID> struct pairs; }\n\n",
        end: "",
        ty: |element| match element {
            Element::Word => "uint",
            Element::Pair => "uint2",
            Element::Float => "float",
        },
        open: |space, name, ty, len| {
            format!("namespace {} {{ constant {ty} {name}[{len}] = {{\n", space.name())
        },
        close: "}; }\n\n",
        pair: |low, high| format!("uint2(0x{low:08x}u, 0x{high:08x}u)"),
        bind: |space, id, name, ty| {
            format!(
                "namespace {} {{ template <> struct table<{id}> {{ static constant {ty} *{}() {{ return {name}; }} }}; }}\n",
                space.name(),
                space.accessor()
            )
        },
        grid: |_, _| String::new(),
        values: true,
        guarded: &[],
        words: false,
        sixteen: |id, name, values, zero| {
            let pairs: Vec<String> = (0..256)
                .map(|byte| {
                    let value = |code: usize| (i32::from(values[code]) + zero) as u16;
                    format!("0x{:04x}", value(byte & 15) | (value(byte >> 4) << 8))
                })
                .collect();
            let mut text = format!("namespace codebooks {{ constant ushort {name}_pairs[256] = {{\n");
            body(&mut text, &pairs);
            text += &format!(
                "}}; template <> struct pairs<{id}> {{ static constant ushort *values() {{ return {name}_pairs; }} }}; }}\n"
            );
            text
        },
    },
    Dialect {
        file: "tables.cuh",
        begin: "#pragma once\n\nnamespace grids { template <unsigned ID> struct table; template <unsigned ID> struct magnitudes; }\nnamespace codebooks { template <unsigned ID> struct words; template <unsigned ID> struct e2m1; }\n\n",
        end: "",
        // NVRTC defines `uint2` but not `uint`.
        ty: |element| match element {
            Element::Word => "unsigned",
            Element::Pair => "uint2",
            Element::Float => "float",
        },
        open: |space, name, ty, len| {
            format!("namespace {} {{ __device__ const {ty} {name}[{len}] = {{\n", space.name())
        },
        close: "}; }\n\n",
        pair: |low, high| format!("{{0x{low:08x}u, 0x{high:08x}u}}"),
        bind: |space, id, name, ty| {
            format!(
                "namespace {} {{ template <> struct table<{id}> {{ __device__ static const {ty} *{}() {{ return {name}; }} }}; }}\n",
                space.name(),
                space.accessor()
            )
        },
        sixteen: |id, _, values, _| {
            let word = |w: usize| u32::from_le_bytes(std::array::from_fn(|b| values[4 * w + b] as u8));
            format!(
                "namespace codebooks {{ template <> struct words<{id}> {{ static constexpr unsigned W0 = 0x{:08x}u, W1 = 0x{:08x}u, W2 = 0x{:08x}u, W3 = 0x{:08x}u; }}; template <> struct e2m1<{id}> {{ static constexpr bool value = {}; }}; }}\n",
                word(0),
                word(1),
                word(2),
                word(3),
                e2m1_doubled(values)
            )
        },
        grid: magnitudes,
        values: false,
        // A grid's magnitude operands' only reader is the records GEMV.
        guarded: &[Table::Magnitudes],
        words: false,
    },
    Dialect {
        file: "tables.glsl",
        begin: "#ifndef SEISMIC_TABLES_GLSL\n#define SEISMIC_TABLES_GLSL\n\n",
        end: "#endif\n",
        ty: |element| match element {
            Element::Word => "uint",
            Element::Pair => "uvec2",
            Element::Float => "float",
        },
        open: |space, name, ty, len| format!("const {ty} {}_{name}[{len}] = {ty}[](\n", space.name()),
        close: ");\n\n",
        pair: |low, high| format!("uvec2(0x{low:08x}u, 0x{high:08x}u)"),
        bind: |space, id, name, _| {
            let space = space.name();
            format!("#define {}_TABLE_{id} {space}_{name}\n", native_macro(space))
        },
        sixteen: |_, _, _, _| String::new(),
        grid: |_, _| String::new(),
        values: true,
        // glslang emits every declared constant array, read or not: a
        // module holds only the tables its includer reads.
        guarded: &[Table::Grid, Table::Codebook],
        words: true,
    },
];

/// Whether a 16-value codebook is E2M1 doubled: code `c`'s magnitude bits
/// (e1 e0 m) read `m` for e = 0 and `2^e (1 + m / 2)` otherwise, doubled,
/// negated by its sign bit, -0 as +0.
fn e2m1_doubled(values: [i8; 16]) -> bool {
    (0..16).all(|code: i32| {
        let (exponent, mantissa) = ((code >> 1) & 3, code & 1);
        let magnitude = if exponent == 0 { mantissa } else { (2 + mantissa) << (exponent - 1) };
        i32::from(values[code as usize]) == if code & 8 != 0 { -magnitude } else { magnitude }
    })
}

/// A grid's magnitude operands, for a grid of at most four distinct lane
/// magnitudes, each `m >> SHIFT` (`SHIFT` the least that tells them apart
/// below 4: its magnitude code): per 16-bit operand type, eight operand bytes
/// (`LOW`, `HIGH`) and per magnitude code the PRMT selector byte (low nibble
/// its low byte, high nibble its high byte) of its positive operand
/// (`SELECTORS`). bf16: the distinct high bytes first, each negation (sign
/// bit set) four bytes on, then the low bytes, so a negative value's
/// selector is the positive's plus 0x40. f16: the low bytes, then the high
/// bytes; a negative value flips its sign bit. Grids of more magnitudes have
/// none.
///
/// Their only reader is the records GEMV (`packets::Records`).
fn magnitudes(id: usize, points: GridPoints) -> String {
    let mut distinct: Vec<u32> = (0..points.len())
        .flat_map(|index| (0..points.lanes()).map(move |lane| u32::from(points.byte(index, lane))))
        .collect();
    distinct.sort_unstable();
    distinct.dedup();
    // A reader shifts four packed magnitude bytes at once, so a code's two
    // bits stay inside its byte: shifts below 7.
    let Some(shift) = (0..7).find(|shift| {
        let codes: Vec<u32> = distinct.iter().map(|m| m >> shift).collect();
        codes.iter().all(|code| *code < 4) && codes.windows(2).all(|pair| pair[0] != pair[1])
    }) else {
        return String::new();
    };
    let bf16 = |m: u32| ((m as f32).to_bits() >> 16) as u16;
    let f16 = |m: u32| {
        // An integer magnitude below 2^11: exponent and 10-bit mantissa.
        if m == 0 {
            return 0;
        }
        let exponent = 31 - m.leading_zeros();
        (((exponent + 15) << 10) | ((m << (10 - exponent)) & 0x3ff)) as u16
    };
    let slots: [Option<u8>; 8] = [None; 8];
    let place = |slots: &mut [Option<u8>; 8], byte: u8, mut allowed: std::ops::Range<usize>| {
        if let Some(at) = slots.iter().position(|slot| *slot == Some(byte)) {
            return Some(at);
        }
        let at = allowed.find(|at| slots[*at].is_none())?;
        slots[at] = Some(byte);
        Some(at)
    };
    let operands = |bits: &dyn Fn(u32) -> u16, signed: bool| -> Option<String> {
        let mut slots = slots;
        let mut selectors = 0u32;
        let mut highs = Vec::new();
        if signed {
            for m in &distinct {
                let high = (bits(*m) >> 8) as u8;
                let at = place(&mut slots, high, 0..4)?;
                slots[at + 4] = Some(high | 0x80);
                highs.push(at);
            }
        }
        let lows = distinct
            .iter()
            .map(|m| place(&mut slots, bits(*m) as u8, 0..8))
            .collect::<Option<Vec<_>>>()?;
        if !signed {
            highs = distinct
                .iter()
                .map(|m| place(&mut slots, (bits(*m) >> 8) as u8, 0..8))
                .collect::<Option<Vec<_>>>()?;
        }
        for ((m, low), high) in distinct.iter().zip(lows).zip(highs) {
            selectors |= ((low | (high << 4)) as u32) << (8 * ((m >> shift) as usize));
        }
        let word = |half: usize| {
            u32::from_le_bytes(std::array::from_fn(|byte| slots[4 * half + byte].unwrap_or(0)))
        };
        Some(format!(
            "static constexpr unsigned LOW = 0x{:08x}u, HIGH = 0x{:08x}u, SELECTORS = 0x{selectors:08x}u;",
            word(0),
            word(1)
        ))
    };
    let (Some(bf16), Some(f16)) = (operands(&bf16, true), operands(&f16, false)) else {
        panic!("grid {id}'s four magnitudes have no eight operand bytes");
    };
    format!(
        "namespace grids {{ template <> struct magnitudes<{id}> {{ static constexpr unsigned SHIFT = {shift}; struct bf16 {{ {bf16} }}; struct f16 {{ {f16} }}; }}; }}\n"
    )
}

/// `words` as an array's lines, four a line, with no trailing comma (a GLSL
/// array constructor takes none).
fn body(text: &mut String, words: &[String]) {
    let lines: Vec<String> = words.chunks(4).map(|line| line.join(", ")).collect();
    writeln!(text, "    {}", lines.join(",\n    ")).unwrap();
}

/// A backend's producer predicates (`<seismic/producers.*>`), rendered from
/// its `native-library/<backend>/producers.decl`: one constexpr predicate per
/// producer over `(code, low, high)` and per form over `(code, low, high,
/// locals)`, true exactly where the declaration has a pattern for the axis
/// values. `code` is one of `producers::UNSIGNED`, `TWOS`, `OFFSET`, `ODD`,
/// `TABLE`, `GRID`, `E4M3` (the kinds of `CodeKind`), `locals` one
/// of `producers::DIRECT`, `PLANE_UNSIGNED` (a scales plane of plain
/// unsigned local codes), `PLANE` (any other plane's local codes) or
/// `CARRIED` (in the grid entries). The library's decode functions are
/// dispatched by these predicates, so the declaration is the one statement
/// of reach. Each producer's concrete (code, low, high) values are also
/// listed as `PRODUCERS_<NAME>(X)`, which the library checks
/// against its decode functions: a declared pattern without one fails the
/// build. A producer whose patterns state how its codes enter its lanes
/// also has `<name>_bytes(code, low, high)`, whether they enter as bytes,
/// and `<name>_zero(code, low, high, stored)`, the zero they enter at (the
/// declared one, else the code's own `stored`), which its decode functions
/// are held to.
fn producers_header(backend: &str, decl: &producer_decl::Declaration, device: &str, constant: &str) -> String {
    use producer_decl::Pattern;
    assert!(
        decl.producers.iter().all(|producer| producer.requires.is_empty())
            && decl.forms.iter().all(|form| form.requires.is_empty()),
        "the {backend} library's producers state no requirements"
    );
    let kind = |name: &str| name.to_uppercase();
    let uint = if backend == "CUDA" { "unsigned" } else { "uint" };
    let locals = |pattern: &Pattern| -> String {
        if pattern.locals.is_empty() {
            return "true".to_owned();
        }
        let terms: Vec<String> = pattern
            .locals
            .iter()
            .map(|(placement, kind)| match (placement.as_str(), kind.as_deref()) {
                ("direct", None) => "locals == DIRECT".to_owned(),
                ("carried", None) => "locals == CARRIED".to_owned(),
                ("plane", None) => "(locals == PLANE_UNSIGNED || locals == PLANE)".to_owned(),
                ("plane", Some("unsigned")) => "locals == PLANE_UNSIGNED".to_owned(),
                other => panic!("the {backend} library distinguishes no local placement {other:?}"),
            })
            .collect();
        format!("({})", terms.join(" || "))
    };
    let codes = |pattern: &Pattern| -> String {
        let mut terms = Vec::new();
        if !pattern.kinds.is_empty() {
            let kinds: Vec<String> = pattern.kinds.iter().map(|name| format!("code == {}", kind(name))).collect();
            terms.push(format!("({})", kinds.join(" || ")));
        }
        if !pattern.planes.is_empty() {
            let planes: Vec<String> =
                pattern.planes.iter().map(|(low, high)| format!("(low == {low}u && high == {high}u)")).collect();
            terms.push(format!("({})", planes.join(" || ")));
        }
        if terms.is_empty() { "true".to_owned() } else { terms.join(" && ") }
    };
    let any = |expressions: Vec<String>| if expressions.is_empty() { "false".to_owned() } else { expressions.join("\n        || ") };
    let mut text = format!(
        "// Seismic native library ({backend}): the producer predicates, rendered from\n\
         // `{backend}/producers.decl` by seismic/lang/build.rs.\n\nnamespace producers {{\n\
         {constant} {uint} DIRECT = 0u, PLANE_UNSIGNED = 1u, PLANE = 2u, CARRIED = 3u;\n\
         {constant} {uint} UNSIGNED = 0u, TWOS = 1u, OFFSET = 2u, ODD = 3u, TABLE = 4u, GRID = 5u, E4M3 = 6u;\n\n"
    );
    for producer in &decl.producers {
        let body = any(producer.patterns.iter().map(&codes).collect());
        writeln!(text, "{device}constexpr bool {}({uint} code, {uint} low, {uint} high) {{\n    return {body};\n}}\n", producer.name).unwrap();
        let entries: Vec<(&Pattern, producer_decl::Entry)> =
            producer.patterns.iter().filter_map(|pattern| pattern.entry.map(|entry| (pattern, entry))).collect();
        if entries.is_empty() {
            continue;
        }
        let bytes = any(entries.iter().filter(|(_, entry)| entry.bytes).map(|&(pattern, _)| codes(pattern)).collect());
        writeln!(text, "{device}constexpr bool {}_bytes({uint} code, {uint} low, {uint} high) {{\n    return {bytes};\n}}\n", producer.name).unwrap();
        let zero = entries.iter().filter_map(|&(pattern, entry)| entry.zero.map(|zero| (codes(pattern), zero))).fold(
            "stored".to_owned(),
            |rest, (held, zero)| format!("({held}) ? {zero} : {rest}"),
        );
        writeln!(
            text,
            "{device}constexpr int {}_zero({uint} code, {uint} low, {uint} high, int stored) {{\n    return {zero};\n}}\n",
            producer.name
        )
        .unwrap();
    }
    for form in &decl.forms {
        let body = any(
            form.patterns.iter().map(|pattern| format!("({} && {})", codes(pattern), locals(pattern))).collect(),
        );
        writeln!(
            text,
            "{device}constexpr bool {}({uint} code, {uint} low, {uint} high, {uint} locals) {{\n    return {body};\n}}\n",
            form.name
        )
        .unwrap();
    }
    text += "}\n\n";
    for producer in &decl.producers {
        let mut entries = Vec::new();
        for pattern in &producer.patterns {
            if pattern.kinds.is_empty() || pattern.planes.is_empty() {
                continue;
            }
            for name in &pattern.kinds {
                for (low, high) in &pattern.planes {
                    entries.push(format!("X(producers::{}, {low}u, {high}u)", kind(name)));
                }
            }
        }
        writeln!(text, "#define PRODUCERS_{}(X) \\", producer.name.to_uppercase()).unwrap();
        text += &entries.join(" \\\n");
        text += "\n";
    }
    text
}

/// Vulkan's producer predicates (`<seismic/producers.glsl>`), rendered from
/// `native-library/vulkan/producers.decl`: one preprocessor predicate per
/// producer over a bound slot's axes tuple (`PRODUCERS_<NAME>(AXES)`, read
/// with its selectors `SEISMIC_AXIS_*`), true exactly where the declaration
/// has a pattern for the slot's code kind and planes and its requirements
/// hold. GLSL has no templates: the library reaches a bound slot's axes in
/// `#if` alone, so the predicates are macros, not functions. Its forms
/// (`DECODED`) are admission's (`forms`), which no library reads.
fn producers_glsl(decl: &producer_decl::Declaration) -> String {
    const KINDS: [&str; 7] = ["UNSIGNED", "TWOS", "OFFSET", "ODD", "TABLE", "GRID", "E4M3"];
    let axis = |name: producer_decl::Axis| format!("AXES(SEISMIC_AXIS_{})", name.name());
    let constant = |domain: producer_decl::Domain, name: &str| format!("SEISMIC_{}_{name}", domain.name());
    let mut text = "// Seismic native library (Vulkan): the producer predicates, rendered from\n\
                    // `vulkan/producers.decl` by seismic/lang/build.rs.\n\n"
        .to_owned();
    for producer in &decl.producers {
        let patterns: Vec<String> = producer
            .patterns
            .iter()
            .map(|pattern| {
                assert!(
                    pattern.locals.is_empty() && pattern.entry.is_none() && !pattern.signed_bytes,
                    "a Vulkan producer's pattern names no locals, lane entry or values"
                );
                let mut terms = Vec::new();
                if !pattern.kinds.is_empty() {
                    let kinds: Vec<String> = pattern
                        .kinds
                        .iter()
                        .map(|name| {
                            let kind = name.to_uppercase();
                            assert!(KINDS.contains(&kind.as_str()), "no code kind `{name}`");
                            format!("AXES(SEISMIC_AXIS_CODES) == SEISMIC_CODES_{kind}")
                        })
                        .collect();
                    terms.push(format!("({})", kinds.join(" || ")));
                }
                if !pattern.planes.is_empty() {
                    let planes: Vec<String> = pattern
                        .planes
                        .iter()
                        .map(|(low, high)| {
                            format!("(AXES(SEISMIC_AXIS_CODE_LOW_BITS) == {low} && AXES(SEISMIC_AXIS_CODE_HIGH_BITS) == {high})")
                        })
                        .collect();
                    terms.push(format!("({})", planes.join(" || ")));
                }
                if terms.is_empty() { "1".to_owned() } else { format!("({})", terms.join(" && ")) }
            })
            .collect();
        let patterns = if patterns.is_empty() { "0".to_owned() } else { patterns.join(" \\\n    || ") };
        let requires: Vec<String> =
            producer.requires.iter().map(|requirement| requirement.render(&axis, &constant)).collect();
        let body = std::iter::once(format!("({patterns})")).chain(requires).collect::<Vec<_>>().join(" \\\n    && ");
        writeln!(text, "#define PRODUCERS_{}(AXES) ({body})", producer.name.to_uppercase()).unwrap();
    }
    text
}

fn main() {
    // The tables come from the build dependency `magnitude_ggml`, whose
    // changes Cargo tracks itself; the script reads no other file than the
    // producer declarations.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/producer_decl.rs");
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    let declaration = |directory: &str| {
        let path = format!("native-library/{directory}/producers.decl");
        println!("cargo:rerun-if-changed={path}");
        producer_decl::parse(&std::fs::read_to_string(&path).expect("producer declaration"))
    };
    let metal = declaration("metal");
    for (backend, decl, file, device, constant) in [
        ("Metal", &metal, "producers.h", "", "constant constexpr"),
        ("CUDA", &declaration("cuda"), "producers.cuh", "__host__ __device__ ", "constexpr"),
    ] {
        std::fs::write(out.join(file), producers_header(backend, decl, device, constant)).expect("producers header");
    }
    std::fs::write(out.join("producers.glsl"), producers_glsl(&declaration("vulkan"))).expect("producers header");
    // The zero Metal's lane producer enters a codebook's values at
    // (`codebooks::pairs`).
    let table_zero = metal
        .producers
        .iter()
        .flat_map(|producer| &producer.patterns)
        .find(|pattern| pattern.kinds.iter().any(|kind| kind == "table"))
        .and_then(|pattern| pattern.entry?.zero)
        .expect("Metal's lane producer states the zero codebook values enter at");
    let table_zero = i32::try_from(table_zero).expect("a lane zero of i32");
    for dialect in DIALECTS {
        let mut text = format!(
            "// Seismic native library: ggml's grids and codebooks (`ggml-common.h`),\n\
             // rendered from `magnitude_ggml::tables` by seismic/lang/build.rs. Byte `j`\n\
             // of a grid point is its lane `j`; an eight-lane point's lanes 0..3 are its\n\
             // first word, lanes 4..7 its second.\n\n{}",
            dialect.begin
        );
        for (index, grid) in GRIDS.iter().enumerate() {
            writeln!(text, "#define SEISMIC_GRID_{} {}", native_macro(grid.name), index + 1).unwrap();
        }
        for (index, codebook) in CODEBOOKS.iter().enumerate() {
            writeln!(text, "#define SEISMIC_CODEBOOK_{} {}", native_macro(codebook.name), index + 1).unwrap();
        }
        if dialect.words {
            text += "#define GRIDS_WORDS_0 0u\n#define CODEBOOKS_WORDS_0 0u\n";
            for (index, grid) in GRIDS.iter().enumerate() {
                writeln!(text, "#define GRIDS_WORDS_{} {}u", index + 1, grid.points.words()).unwrap();
            }
            for (index, codebook) in CODEBOOKS.iter().enumerate() {
                writeln!(text, "#define CODEBOOKS_WORDS_{} {}u", index + 1, codebook.values.len()).unwrap();
            }
        }
        text += "\n";
        for (index, grid) in GRIDS.iter().enumerate() {
            let (element, words): (_, Vec<String>) = match grid.points {
                GridPoints::Lanes4(points) => {
                    (Element::Word, points.iter().map(|point| format!("0x{point:08x}u")).collect())
                }
                GridPoints::Lanes8(points) => (
                    Element::Pair,
                    points
                        .iter()
                        .map(|point| (dialect.pair)(*point as u32, (*point >> 32) as u32))
                        .collect(),
                ),
            };
            let ty = (dialect.ty)(element);
            let mut points = (dialect.open)(Space::Grids, grid.name, ty, words.len());
            body(&mut points, &words);
            points += dialect.close;
            points += &(dialect.bind)(Space::Grids, index + 1, grid.name, ty);
            text += &dialect.declare(Table::Grid, index + 1, points);
            text += &dialect.declare(Table::Magnitudes, index + 1, (dialect.grid)(index + 1, grid.points));
            text += "\n";
        }
        for (index, codebook) in CODEBOOKS.iter().enumerate() {
            if dialect.values {
                let words: Vec<String> = codebook.values.iter().map(|value| format!("{value}.0f")).collect();
                let ty = (dialect.ty)(Element::Float);
                let mut values = (dialect.open)(Space::Codebooks, codebook.name, ty, words.len());
                body(&mut values, &words);
                values += dialect.close;
                values += &(dialect.bind)(Space::Codebooks, index + 1, codebook.name, ty);
                text += &dialect.declare(Table::Codebook, index + 1, values);
            }
            if let Ok(values) = <[i8; 16]>::try_from(codebook.values) {
                text += &(dialect.sixteen)(index + 1, codebook.name, values, table_zero);
            }
            text += "\n";
        }
        text += dialect.end;
        std::fs::write(out.join(dialect.file), text).expect("tables header");
    }
}
