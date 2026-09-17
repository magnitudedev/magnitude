//! Packed representations. A representation is a property of data: at portable
//! scope an element read is its decoded value; at backend scope the packet
//! structure is exposed through the accessors here.

use crate::sym::Sym;
use crate::types::DType;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repr {
    pub name: &'static str,
    /// values per quantization group
    pub group: u32,
    /// bits per code
    pub bits: u32,
    /// dtype of the per-group scale and bias
    pub coefficient: DType,
    pub has_bias: bool,
    pub code: CodeInterpretation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodeInterpretation {
    Unsigned,
    TwosComplement,
    Offset(i32),
    Table(&'static [i32]),
}
impl Repr {
    pub fn decode_code(&self, raw: u32) -> i32 {
        match self.code {
            CodeInterpretation::Unsigned => raw as i32,
            CodeInterpretation::TwosComplement => {
                ((raw << (32 - self.bits)) as i32) >> (32 - self.bits)
            }
            CodeInterpretation::Offset(zero) => raw as i32 - zero,
            CodeInterpretation::Table(table) => table[raw as usize],
        }
    }
}

pub const REPRS: &[Repr] = &[
    // MLX affine 4-bit, group 64: bf16 scale and bias per group.
    Repr {
        name: "q4g64",
        group: 64,
        bits: 4,
        coefficient: DType::BF16,
        has_bias: true,
        code: CodeInterpretation::Unsigned,
    },
    Repr {
        name: "q4g32",
        group: 32,
        bits: 4,
        coefficient: DType::F32,
        has_bias: true,
        code: CodeInterpretation::Unsigned,
    },
    // Canonical resident planes for GGUF import. q5 codes occupy byte lanes;
    // q6 uses byte lanes with an explicit offset, preserving every source code.
    Repr {
        name: "q8g32a",
        group: 32,
        bits: 8,
        coefficient: DType::F32,
        has_bias: true,
        code: CodeInterpretation::Unsigned,
    },
    Repr {
        name: "q8g16z32",
        group: 16,
        bits: 8,
        coefficient: DType::F32,
        has_bias: false,
        code: CodeInterpretation::Offset(32),
    },
    Repr {
        name: "q8g32s",
        group: 32,
        bits: 8,
        coefficient: DType::F16,
        has_bias: false,
        code: CodeInterpretation::TwosComplement,
    },
    Repr {
        name: "iq4g32",
        group: 32,
        bits: 4,
        coefficient: DType::F32,
        has_bias: false,
        code: CodeInterpretation::Table(&[
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ]),
    },
    Repr {
        name: "q8g32",
        group: 32,
        bits: 8,
        coefficient: DType::F32,
        has_bias: false,
        code: CodeInterpretation::Unsigned,
    },
];

pub fn lookup(name: &str) -> Option<&'static Repr> {
    REPRS.iter().find(|r| r.name == name)
}

impl Repr {
    pub fn codes_per_word(&self) -> u32 {
        32 / self.bits
    }

    /// Storage bits per value, including coefficients.
    pub fn bits_per_value(&self) -> f64 {
        let coeff = self.coefficient.bytes() as f64 * 8.0 * if self.has_bias { 2.0 } else { 1.0 };
        self.bits as f64 + coeff / self.group as f64
    }

    /// Shapes of the packet accessors for a packed tile whose last axis has extent `k`.
    pub fn words_extent(&self, k: &Sym) -> Sym {
        k.quot(&Sym::constant(self.codes_per_word() as i64))
    }

    pub fn groups_extent(&self, k: &Sym) -> Sym {
        k.quot(&Sym::constant(self.group as i64))
    }
}
