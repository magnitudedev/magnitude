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
}

pub const REPRS: &[Repr] = &[
    // MLX affine 4-bit, group 64: bf16 scale and bias per group.
    Repr { name: "q4g64", group: 64, bits: 4, coefficient: DType::BF16, has_bias: true },
    Repr { name: "q4g32", group: 32, bits: 4, coefficient: DType::F32, has_bias: true },
    Repr { name: "q8g32", group: 32, bits: 8, coefficient: DType::F32, has_bias: false },
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
