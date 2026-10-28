//! Each backend's tables header (`<seismic/tables.*>`), rendered from ggml's
//! tables (`magnitude_ggml::tables`): the ids of its grids and codebooks
//! (`SEISMIC_GRID_<NAME>`, `SEISMIC_CODEBOOK_<NAME>`: position + 1, so that 0
//! names none, which a bound element's decode axes name), each grid's points
//! and each codebook's values (on Metal and Vulkan, which read them), and
//! their bindings by id: `grids::table<ID>::points()` and
//! `codebooks::table<ID>::values()` (Vulkan `GRIDS_TABLE_<ID>`,
//! `CODEBOOKS_TABLE_<ID>`). A `u32` grid point is one
//! word and a `u64` point its (low, high) words; codebook values are exact
//! floats. A codebook also has a dialect's own forms of its values over its
//! sixteen codes (fewer padded with zeros):
//! on Metal each code byte's two values + 128 (`codebooks::pairs<ID>`: the
//! low nibble's in the low byte), on CUDA its values as little-endian s8
//! words (`codebooks::words<ID>`: `W0`..`W3`, values 4w..4w + 3 in word w)
//! and whether they are E2M1 doubled (`codebooks::e2m1<ID>`). On CUDA a grid
//! whose lane magnitudes are few also has the operands of its magnitudes
//! (`grids::magnitudes<ID>`, see `magnitudes`).

use magnitude_ggml::tables::{CODEBOOKS, GRIDS, GridPoints};
use std::fmt::Write as _;

#[path = "src/native_name.rs"]
mod native_name;
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
    /// A codebook's dialect forms, its values padded to sixteen.
    sixteen: fn(id: usize, name: &str, values: [i8; 16]) -> String,
    /// A grid's dialect forms.
    grid: fn(id: usize, points: GridPoints) -> String,
    /// Whether the dialect reads codebooks' values themselves (CUDA reads
    /// only their dialect forms).
    values: bool,
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
        sixteen: |id, name, values| {
            let pairs: Vec<String> = (0..256)
                .map(|byte| {
                    let value = |code: usize| (i32::from(values[code]) + 128) as u16;
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
        sixteen: |id, _, values| {
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
        sixteen: |_, _, _| String::new(),
        grid: |_, _| String::new(),
        values: true,
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

fn main() {
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
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
            text += &(dialect.open)(Space::Grids, grid.name, ty, words.len());
            body(&mut text, &words);
            text += dialect.close;
            text += &(dialect.bind)(Space::Grids, index + 1, grid.name, ty);
            text += &(dialect.grid)(index + 1, grid.points);
            text += "\n";
        }
        for (index, codebook) in CODEBOOKS.iter().enumerate() {
            if dialect.values {
                let words: Vec<String> = codebook.values.iter().map(|value| format!("{value}.0f")).collect();
                let ty = (dialect.ty)(Element::Float);
                text += &(dialect.open)(Space::Codebooks, codebook.name, ty, words.len());
                body(&mut text, &words);
                text += dialect.close;
                text += &(dialect.bind)(Space::Codebooks, index + 1, codebook.name, ty);
            }
            // Codes index only their codebook's values, so a codebook of
            // fewer than 16 (Q1_0's two) takes the forms padded with zeros.
            assert!(codebook.values.len() <= 16, "a codebook of at most 4-bit codes");
            let mut values = [0i8; 16];
            values[..codebook.values.len()].copy_from_slice(codebook.values);
            text += &(dialect.sixteen)(index + 1, codebook.name, values);
            text += "\n";
        }
        text += dialect.end;
        std::fs::write(out.join(dialect.file), text).expect("tables header");
    }
}
