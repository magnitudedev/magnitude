//! The GGUF formats the registry imports, and the test modules the build
//! script generates for them (`gguf_format_tests!`, one module a format, so
//! a run selects formats by name, `gguf::iq1s::`, and libtest runs the
//! formats in parallel). A suite that generates no tests a format uses
//! neither the macro nor its export.
#![allow(dead_code, unused_macros)]

use magnitude_ggml::Encoding;
use seismic::{Element, GgufScales, GridScale, Layout, OperandRecipe};
use seismic_lang::registry::f16_bits;

include!("widths.rs");
include!(concat!(env!("OUT_DIR"), "/gguf_format_tests.rs"));
#[allow(unused_imports)]
pub(crate) use gguf_format_tests;

#[path = "../random/mod.rs"]
mod random;
pub use random::Rng;

/// A GGUF-imported element: weights stored as `encoding`, their scales
/// where `scales` says, converted by the registry into the element whose
/// products take `operands`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub encoding: Encoding,
    pub scales: GgufScales,
    pub operands: OperandRecipe,
}

impl std::fmt::Debug for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

impl Format {
    pub const Q4K: Format = Format::rounded(Encoding::Q4K);
    pub const Q5K: Format = Format::rounded(Encoding::Q5K);
    pub const Q6K: Format = Format::rounded(Encoding::Q6K);
    pub const Q8: Format = Format::rounded(Encoding::Q8_0);
    pub const IQ4_XS: Format = Format::rounded(Encoding::Iq4Xs);
    pub const NVFP4: Format = Format::rounded(Encoding::Nvfp4);
    /// GGUF NVFP4 calibrated for FP4 activations (W4A4): nvfp4g16's storage,
    /// NVFP4 operands.
    pub const NVFP4_A4: Format = Format {
        encoding: Encoding::Nvfp4,
        scales: GgufScales::Own,
        operands: OperandRecipe::Nvfp4Rows,
    };
    /// GGUF F8_E4M3 under its grid of E8M0 block scales: mxfp8g32.
    pub const F8_E4M3: Format = Format {
        encoding: Encoding::F8E4M3,
        scales: GgufScales::Grid(GridScale::E8m0),
        operands: OperandRecipe::Rounded,
    };
    /// GGUF F8_E4M3 under its grid of F32 tile scales: fp8g128.
    pub const F8_E4M3_F32: Format = Format {
        encoding: Encoding::F8E4M3,
        scales: GgufScales::Grid(GridScale::F32),
        operands: OperandRecipe::Rounded,
    };

    /// The element of `encoding`, with its own scales, products take
    /// rounded activations into.
    pub const fn rounded(encoding: Encoding) -> Self {
        Self {
            encoding,
            scales: GgufScales::Own,
            operands: OperandRecipe::Rounded,
        }
    }
    /// Every format the registry converts into `layout`, in ggml id order.
    pub fn all(layout: Layout) -> Vec<Format> {
        Self::pairs()
            .filter(|format| format.converted(layout).is_some())
            .collect()
    }
    /// The registry's source of this format's GGUF blocks (assembled
    /// blocks, under a scale grid).
    pub fn source(self) -> Element {
        Element::gguf_packet_source(self.encoding, self.scales)
            .unwrap_or_else(|| panic!("{:?} {:?}: no registered GGUF source", self.encoding, self.scales))
    }
    /// The values of one source packet: a block of the type, or an
    /// assembled grid block.
    fn packet_values(self) -> u64 {
        self.source().logical_group().expect("a source has packets")
    }
    /// This format's element in `layout`.
    pub fn element(self, layout: Layout) -> Element {
        self.converted(layout)
            .unwrap_or_else(|| panic!("{self:?}: no conversion into {layout:?}"))
    }
    /// The storage name: the representation, then the operand recipe's
    /// suffix.
    pub fn name(self) -> String {
        let element = self.stored().unwrap_or_else(|| {
            panic!(
                "{:?} {:?}: no registered conversion",
                self.encoding, self.operands
            )
        });
        format!("{}{}", element.representation(), self.operands.suffix())
    }
    /// The `PARTIAL_BLOCK_WIDTHS` whole blocks of this format fill.
    pub fn partial_block_widths(self) -> Vec<usize> {
        partial_block_widths(self.block_values()).collect()
    }
    pub fn block_values(self) -> usize {
        self.packet_values() as usize
    }
    pub fn block_bytes(self) -> usize {
        self.source()
            .canonical_byte_len(&[1, self.packet_values()])
            .expect("a source packet's bytes") as usize
    }
    /// One random GGUF block whose values reach about `scale`: random codes
    /// and sub-scales, with the scale fields set so the values reach a
    /// random 0.5 to 1.5 times `scale`.
    pub fn block(self, rng: &mut Rng, scale: f32) -> Vec<u8> {
        let mut block = (0..self.block_bytes())
            .map(|_| rng.byte())
            .collect::<Vec<_>>();
        let factor = scale * (0.5 + rng.fraction());
        let mut f16 = |at: usize, value: f32| {
            block[at..at + 2].copy_from_slice(&f16_bits(value).to_le_bytes())
        };
        match self.encoding {
            // d: codes - 8 reach 8.
            Encoding::Q4_0 => f16(0, factor / 8.0),
            // d and the minimum m: values d code + m span [-factor, factor).
            Encoding::Q4_1 => {
                f16(0, factor / 8.0);
                f16(2, -factor);
            }
            // d: codes - 16 reach 16.
            Encoding::Q5_0 => f16(0, factor / 16.0),
            Encoding::Q5_1 => {
                f16(0, factor / 16.0);
                f16(2, -factor);
            }
            // d: values reach d 127.
            Encoding::Q8_0 => f16(0, factor / 127.0),
            // d: the table's values reach 127 d.
            Encoding::Iq4Nl => f16(0, factor / 128.0),
            // d: values reach d 32 127.
            Encoding::Iq4Xs => f16(0, factor / 4096.0),
            // d and dmin: values reach d 63 15 and dmin 63.
            Encoding::Q4K => {
                f16(0, factor / 945.0);
                f16(2, factor / 63.0);
            }
            // d and dmin: values reach d 63 31 and dmin 63.
            Encoding::Q5K => {
                f16(0, factor / 1953.0);
                f16(2, factor / 63.0);
            }
            // d: values reach d 128 32.
            Encoding::Q6K => f16(208, factor / 4096.0),
            // d: values reach d (2 s + 1) 43 / 8.
            Encoding::Iq2Xs | Encoding::Iq2Xxs | Encoding::Iq2S => f16(0, factor / 40.0),
            // d: values reach d (2 s + 1) 62 / 4.
            Encoding::Iq3Xxs => f16(0, factor / 120.0),
            // d: values reach d (2 s + 1) 15.
            Encoding::Iq3S => f16(0, factor / 480.0),
            // d and dmin: values reach d 15 3 and dmin 15.
            Encoding::Q2K => {
                f16(80, factor / 45.0);
                f16(82, factor / 30.0);
            }
            // d: values reach d 32 4.
            Encoding::Q3K => f16(108, factor / 128.0),
            // d: codes - 1 reach 2.
            Encoding::Tq2_0 => f16(64, factor / 2.0),
            // d: digits - 1 reach 1.
            Encoding::Tq1_0 => f16(52, factor),
            // d: codes - 1 reach 2.
            Encoding::Q2_0 => f16(0, factor / 2.0),
            // d: values are ±d.
            Encoding::Q1_0 => f16(0, factor),
            // d: values reach d (2 s + 1) 9 / 8.
            Encoding::Iq1S => f16(0, factor / 16.0),
            // d, as the top nibbles of the four scale words: values reach
            // d (2 s + 1) 9 / 8.
            Encoding::Iq1M => {
                let d = f16_bits(factor / 16.0);
                for nibble in 0..4 {
                    let byte = &mut block[49 + 2 * nibble];
                    *byte = (*byte & 0x0f) | (((d >> (4 * nibble)) as u8 & 0x0f) << 4);
                }
            }
            // E8M0: the power of two nearest factor / 6 (the E2M1 range).
            Encoding::Mxfp4 => block[0] = (127.0 + (factor / 6.0).log2().round()) as u8,
            // UE4M3 exponent of factor / 6 (normal), random mantissas.
            Encoding::Nvfp4 => {
                for field in &mut block[..4] {
                    let exponent = ((factor / 6.0).log2().floor() as i32 + 7).clamp(1, 14) as u8;
                    *field = (exponent << 3) | (rng.next() % 8) as u8;
                }
            }
            // An assembled block: its scale, of factor / 448 (the E4M3
            // range), then random finite E4M3 codes (a NaN code,
            // S.1111.111, loses its exponent's top bit). An E8M0 scale is
            // the power of two nearest it (32 codes follow), an F32 one it
            // (128 codes follow).
            Encoding::F8E4M3 => {
                let codes = match self.scales {
                    GgufScales::Grid(GridScale::E8m0) => {
                        block[0] = (127.0 + (factor / 448.0).log2().round()) as u8;
                        1
                    }
                    GgufScales::Grid(GridScale::F32) => {
                        block[..4].copy_from_slice(&(factor / 448.0).to_le_bytes());
                        4
                    }
                    GgufScales::Own => panic!("F8_E4M3 blocks are scaled by a grid"),
                };
                for code in &mut block[codes..] {
                    if *code & 0x7f == 0x7f {
                        *code &= !0x40;
                    }
                }
            }
            other => panic!("{other:?}: no GGUF block fixture"),
        }
        block
    }
    /// Every encoding, its scales wherever they are, under every operand
    /// recipe, in ggml id order.
    fn pairs() -> impl Iterator<Item = Format> {
        Encoding::ALL.into_iter().flat_map(|encoding| {
            GgufScales::ALL
                .into_iter()
                .flat_map(move |scales| {
                    OperandRecipe::ALL.into_iter().map(move |operands| Format {
                        encoding,
                        scales,
                        operands,
                    })
                })
        })
    }
    fn converted(self, layout: Layout) -> Option<Element> {
        Element::gguf_packet_source(self.encoding, self.scales)?.converted(layout, self.operands)
    }
    /// This format's element in a layout the registry converts it into.
    fn stored(self) -> Option<Element> {
        Layout::ALL
            .into_iter()
            .find_map(|layout| self.converted(layout))
    }
}
