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
    pub coefficients: Coefficients,
    pub code: CodeInterpretation,
}

/// Equivalent source-IR covers for a bounded packet decode owner. Specialized
/// decoding retains fixed code ranges and word reuse; indexed decoding uses a
/// bounded element loop with runtime word/bit coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PacketDecoder { Specialized, Indexed }

/// Private packet rows large enough to retain any logical prefix within a
/// representation group. The same physical plane geometry owns native storage
/// declarations and raw snapshot copies on every backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotLayout {
    pub physical_width: u64,
    pub strides: Vec<u64>,
    pub planes: Vec<SnapshotPlane>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotPlane {
    pub plane: Plane,
    pub elements_per_row: u64,
    pub elements: u64,
}
impl Repr {
    pub fn snapshot_layout(&self, capacities: &[u64]) -> Option<SnapshotLayout> {
        let (&width, outer) = capacities.split_last()?;
        let group = u64::from(self.storage_group());
        let physical_width = if width == 0 {
            0
        } else {
            width
                .checked_add(group - 1)?
                .div_ceil(group)
                .checked_mul(group)?
        };
        let rows = if outer.contains(&0) {
            0
        } else {
            outer.iter().try_fold(1u64, |n, &d| n.checked_mul(d))?
        };
        let mut strides = vec![1; capacities.len()];
        let mut stride = physical_width;
        for axis in (0..outer.len()).rev() {
            strides[axis] = stride;
            stride = stride.checked_mul(capacities[axis])?;
        }
        let planes = self
            .planes()
            .into_iter()
            .map(|plane| {
                let elements_per_row = plane.storage_elements(physical_width)?;
                Some(SnapshotPlane {
                    elements: rows.checked_mul(elements_per_row)?,
                    elements_per_row,
                    plane,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(SnapshotLayout {
            physical_width,
            strides,
            planes,
        })
    }
}
/// Physical coefficient encoding. Hierarchical fields are interleaved scale,
/// bias (when present); factors are shared by a larger group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Coefficients {
    Direct { dtype: DType, bias: bool },
    Hierarchical { factor_group: u32, factor_dtype: DType, bits: u32,
        interpretation: CodeInterpretation, bias: bool, bias_sign: i32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaneEncoding {
    Dense(DType),
    Packed { bits: u32, interpretation: CodeInterpretation },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plane {
    pub name: &'static str,
    /// Logical values sharing `fields` entries in this plane.
    pub group: u32,
    pub fields: u32,
    pub encoding: PlaneEncoding,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Coefficient {
    Direct { plane: Plane },
    Product { factor: Plane, coefficients: Plane, field: u32, sign: i32 },
}
impl Plane {
    pub fn dtype(&self) -> DType {
        match self.encoding { PlaneEncoding::Dense(dtype) => dtype, PlaneEncoding::Packed { .. } => DType::U32 }
    }
    pub fn entry_bits(&self) -> u32 {
        match self.encoding { PlaneEncoding::Dense(dtype) => dtype.bytes() * 8, PlaneEncoding::Packed { bits, .. } => bits }
    }
    pub fn entries(&self, values: u64) -> Option<u64> {
        values.div_ceil(u64::from(self.group)).checked_mul(u64::from(self.fields))
    }
    pub fn storage_elements(&self, values: u64) -> Option<u64> {
        let entries = self.entries(values)?;
        match self.encoding { PlaneEncoding::Dense(_) => Some(entries), PlaneEncoding::Packed { bits, .. } => entries.checked_mul(u64::from(bits)).map(|n| n.div_ceil(32)) }
    }
    pub fn bytes(&self, values: u64) -> Option<u64> {
        self.storage_elements(values)?.checked_mul(u64::from(self.dtype().bytes()))
    }
    /// Raw accessor extent. Owning packed rows are complete storage groups.
    pub fn extent(&self, values: &Sym) -> Sym {
        let entries = values.quot(&Sym::constant(i64::from(self.group))).scale(i64::from(self.fields));
        match self.encoding {
            PlaneEncoding::Dense(_) => entries,
            PlaneEncoding::Packed { bits, .. } => entries.scale(i64::from(bits)).add(&Sym::constant(31)).quot(&Sym::constant(32)),
        }
    }
    pub fn byte_offset(&self, logical: u64) -> Option<u64> {
        if !logical.is_multiple_of(u64::from(self.group)) { return None; }
        let bits = (logical / u64::from(self.group)).checked_mul(u64::from(self.fields))?.checked_mul(u64::from(self.entry_bits()))?;
        let alignment = u64::from(self.dtype().bytes()) * 8;
        bits.is_multiple_of(alignment).then_some(bits / 8)
    }
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
        self.code.decode(raw, self.bits)
    }
}

pub const REPRS: &[Repr] = &[
    // MLX affine 4-bit, group 64: bf16 scale and bias per group.
    Repr {
        name: "q4g64",
        group: 64,
        bits: 4,
        coefficients: Coefficients::Direct { dtype: DType::BF16, bias: true },
        code: CodeInterpretation::Unsigned,
    },
    Repr {
        name: "q4g32",
        group: 32,
        bits: 4,
        coefficients: Coefficients::Direct { dtype: DType::F32, bias: true },
        code: CodeInterpretation::Unsigned,
    },
    Repr {
        name: "q4k", group: 32, bits: 4, code: CodeInterpretation::Unsigned,
        coefficients: Coefficients::Hierarchical { factor_group: 256, factor_dtype: DType::F16,
            bits: 6, interpretation: CodeInterpretation::Unsigned, bias: true, bias_sign: -1 },
    },
    Repr {
        name: "q5k", group: 32, bits: 5, code: CodeInterpretation::Unsigned,
        coefficients: Coefficients::Hierarchical { factor_group: 256, factor_dtype: DType::F16,
            bits: 6, interpretation: CodeInterpretation::Unsigned, bias: true, bias_sign: -1 },
    },
    Repr {
        name: "q6k", group: 16, bits: 6, code: CodeInterpretation::Offset(32),
        coefficients: Coefficients::Hierarchical { factor_group: 256, factor_dtype: DType::F16,
            bits: 8, interpretation: CodeInterpretation::TwosComplement, bias: false, bias_sign: 1 },
    },
    Repr {
        name: "q8g32s",
        group: 32,
        bits: 8,
        coefficients: Coefficients::Direct { dtype: DType::F16, bias: false },
        code: CodeInterpretation::TwosComplement,
    },
    Repr {
        name: "iq4g32",
        group: 32,
        bits: 4,
        coefficients: Coefficients::Direct { dtype: DType::F32, bias: false },
        code: CodeInterpretation::Table(&[
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ]),
    },
    Repr {
        name: "q8g32",
        group: 32,
        bits: 8,
        coefficients: Coefficients::Direct { dtype: DType::F32, bias: false },
        code: CodeInterpretation::Unsigned,
    },
];

pub fn lookup(name: &str) -> Option<&'static Repr> {
    REPRS.iter().find(|r| r.name == name)
}

impl CodeInterpretation {
    pub fn decode(&self, raw: u32, bits: u32) -> i32 {
        match self {
            Self::Unsigned => raw as i32,
            Self::TwosComplement => ((raw << (32 - bits)) as i32) >> (32 - bits),
            Self::Offset(zero) => raw as i32 - zero,
            Self::Table(table) => table[raw as usize],
        }
    }
}

impl Repr {
    pub fn has_bias(&self) -> bool {
        match self.coefficients { Coefficients::Direct { bias, .. } | Coefficients::Hierarchical { bias, .. } => bias }
    }
    pub fn coefficient_dtype(&self) -> DType {
        match self.coefficients { Coefficients::Direct { dtype, .. } => dtype, Coefficients::Hierarchical { .. } => DType::F32 }
    }
    pub fn storage_group(&self) -> u32 {
        match self.coefficients { Coefficients::Direct { .. } => self.group, Coefficients::Hierarchical { factor_group, .. } => factor_group }
    }
    /// Ordered physical ABI planes; logical scale/bias accessors may decode several planes.
    pub fn planes(&self) -> Vec<Plane> {
        let mut result = vec![Plane { name: "words", group: 1, fields: 1,
            encoding: PlaneEncoding::Packed { bits: self.bits, interpretation: self.code.clone() } }];
        match &self.coefficients {
            Coefficients::Direct { dtype, bias } => {
                result.push(Plane { name: "scale", group: self.group, fields: 1, encoding: PlaneEncoding::Dense(*dtype) });
                if *bias { result.push(Plane { name: "bias", group: self.group, fields: 1, encoding: PlaneEncoding::Dense(*dtype) }); }
            }
            Coefficients::Hierarchical { factor_group, factor_dtype, bits, interpretation, bias, .. } => {
                result.push(Plane { name: "coefficients", group: self.group, fields: if *bias { 2 } else { 1 },
                    encoding: PlaneEncoding::Packed { bits: *bits, interpretation: interpretation.clone() } });
                result.push(Plane { name: "scale_factor", group: *factor_group, fields: 1, encoding: PlaneEncoding::Dense(*factor_dtype) });
                if *bias { result.push(Plane { name: "bias_factor", group: *factor_group, fields: 1, encoding: PlaneEncoding::Dense(*factor_dtype) }); }
            }
        }
        result
    }
    pub fn plane(&self, name: &str) -> Option<Plane> { self.planes().into_iter().find(|p| p.name == name) }
    pub fn plane_index(&self, name: &str) -> Option<usize> { self.planes().iter().position(|p| p.name == name) }
    pub fn coefficient(&self, bias: bool) -> Option<Coefficient> {
        if bias && !self.has_bias() { return None; }
        Some(match self.coefficients {
            Coefficients::Direct { .. } => Coefficient::Direct { plane: self.plane(if bias { "bias" } else { "scale" }).unwrap() },
            Coefficients::Hierarchical { bias_sign, .. } => Coefficient::Product {
                factor: self.plane(if bias { "bias_factor" } else { "scale_factor" }).unwrap(),
                coefficients: self.plane("coefficients").unwrap(), field: u32::from(bias), sign: if bias { bias_sign } else { 1 },
            },
        })
    }
    pub fn bits_per_value(&self) -> f64 {
        self.planes().iter().map(|p| p.entry_bits() as f64 * p.fields as f64 / p.group as f64).sum()
    }
    pub fn words_extent(&self, k: &Sym) -> Sym { self.plane("words").unwrap().extent(k) }
    pub fn groups_extent(&self, k: &Sym) -> Sym { k.quot(&Sym::constant(self.group as i64)) }
}

/// Little-endian contiguous packed entry; reads only bytes containing the entry.
pub fn read_packed(bytes: &[u8], entry: usize, bits: u32) -> u32 {
    let first = entry * bits as usize;
    let mut value = 0;
    for bit in 0..bits as usize { value |= u32::from((bytes[(first + bit) / 8] >> ((first + bit) % 8)) & 1) << bit; }
    value
}
pub fn write_packed(bytes: &mut [u8], entry: usize, bits: u32, value: u32) {
    let first = entry * bits as usize;
    for bit in 0..bits as usize {
        let index = (first + bit) / 8; let shift = (first + bit) % 8;
        bytes[index] = (bytes[index] & !(1 << shift)) | (((value >> bit) as u8 & 1) << shift);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_planes_match_payload_and_cross_word_entries() {
        for (name, expected) in [("q4k",144),("q5k",176),("q6k",210)] {
            let r=lookup(name).unwrap();
            assert_eq!(r.planes().iter().map(|p|p.bytes(256).unwrap()).sum::<u64>(),expected);
            for plane in r.planes() {
                if let PlaneEncoding::Packed { bits, .. }=plane.encoding {
                    let n=plane.entries(512).unwrap() as usize;
                    let mut bytes=vec![0;plane.bytes(512).unwrap()as usize];
                    for i in 0..n { write_packed(&mut bytes,i,bits,(i as u32).wrapping_mul(31)); }
                    for i in 0..n { assert_eq!(read_packed(&bytes,i,bits),(i as u32).wrapping_mul(31)&((1<<bits)-1)); }
                    assert_eq!(plane.byte_offset(256),plane.bytes(256));
                }
            }
        }
    }
}
