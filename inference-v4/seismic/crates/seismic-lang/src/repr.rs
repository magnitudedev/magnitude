//! Packed representations. A representation is a property of data: at portable
//! scope an element read is its decoded value; at backend scope the packet
//! structure is exposed through the accessors here.

use crate::intrinsics::PlaneField;
use crate::sym::Sym;
use crate::types::DType;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repr {
    pub name: &'static str,
    pub packing_axis: PackingAxisRule,
    /// values per quantization group
    pub group: u32,
    /// bits per code
    pub bits: u32,
    pub coefficients: Coefficients,
    pub code: CodeInterpretation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackingAxisRule {
    Last,
}

impl PackingAxisRule {
    pub fn resolve(self, rank: usize) -> Option<usize> {
        match self {
            Self::Last => rank.checked_sub(1),
        }
    }
}

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
    Direct {
        dtype: DType,
        bias: bool,
    },
    Hierarchical {
        factor_group: u32,
        factor_dtype: DType,
        bits: u32,
        interpretation: CodeInterpretation,
        bias: bool,
        bias_sign: i32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaneEncoding {
    Dense(DType),
    Packed {
        bits: u32,
        interpretation: CodeInterpretation,
    },
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
    Direct {
        plane: Plane,
    },
    Product {
        factor: Plane,
        coefficients: Plane,
        field: u32,
        sign: i32,
    },
}
impl Plane {
    pub fn dtype(&self) -> DType {
        match self.encoding {
            PlaneEncoding::Dense(dtype) => dtype,
            PlaneEncoding::Packed { .. } => DType::U32,
        }
    }
    pub fn entry_bits(&self) -> u32 {
        match self.encoding {
            PlaneEncoding::Dense(dtype) => dtype.bytes() * 8,
            PlaneEncoding::Packed { bits, .. } => bits,
        }
    }
    pub fn entries(&self, values: u64) -> Option<u64> {
        values
            .div_ceil(u64::from(self.group))
            .checked_mul(u64::from(self.fields))
    }
    pub fn storage_elements(&self, values: u64) -> Option<u64> {
        let entries = self.entries(values)?;
        match self.encoding {
            PlaneEncoding::Dense(_) => Some(entries),
            PlaneEncoding::Packed { bits, .. } => {
                entries.checked_mul(u64::from(bits)).map(|n| n.div_ceil(32))
            }
        }
    }
    pub fn bytes(&self, values: u64) -> Option<u64> {
        self.storage_elements(values)?
            .checked_mul(u64::from(self.dtype().bytes()))
    }
    /// Raw accessor extent. Owning packed rows are complete storage groups.
    pub fn extent(&self, values: &Sym) -> Sym {
        let entries = values
            .quot(&Sym::constant(i64::from(self.group)))
            .scale(i64::from(self.fields));
        match self.encoding {
            PlaneEncoding::Dense(_) => entries,
            PlaneEncoding::Packed { bits, .. } => entries
                .scale(i64::from(bits))
                .add(&Sym::constant(31))
                .quot(&Sym::constant(32)),
        }
    }
    pub fn byte_offset(&self, logical: u64) -> Option<u64> {
        if !logical.is_multiple_of(u64::from(self.group)) {
            return None;
        }
        let bits = (logical / u64::from(self.group))
            .checked_mul(u64::from(self.fields))?
            .checked_mul(u64::from(self.entry_bits()))?;
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

/// The complete description of one physical plane of a representation: its
/// typed field identity, ABI ordinal, encoding, grouping, and entry
/// addressing. Backends declare storage and address entries from this
/// schema; none reconstructs it from names.
///
/// Entry addressing: the logical value at position `v` along the packing
/// axis shares group `v / group`, whose `fields` entries are consecutive;
/// entry `(v / group) * fields + field` starts at bit
/// `entry * entry_bits` of the plane row, little-endian within
/// `storage_dtype` words (see `entry` and `bit_offset`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneSchema {
    pub field: PlaneField,
    /// Position in `Repr::planes()` (the ABI plane order).
    pub ordinal: u32,
    pub encoding: PlaneEncoding,
    /// Logical values sharing one group of entries.
    pub group: u32,
    /// Entries per group.
    pub fields: u32,
    /// Bits of one entry: the packed code width, or the dense dtype width.
    pub entry_bits: u32,
    /// The storage element dtype: `u32` words for a packed plane, the dense
    /// dtype otherwise.
    pub storage_dtype: DType,
}

impl PlaneSchema {
    /// The entry holding field `field` of the group of logical value `value`.
    pub fn entry(&self, value: u64, field: u32) -> u64 {
        value / u64::from(self.group) * u64::from(self.fields) + u64::from(field)
    }

    /// The first bit of `entry` within the plane row, little-endian within
    /// `storage_dtype` words.
    pub fn bit_offset(&self, entry: u64) -> u64 {
        entry * u64::from(self.entry_bits)
    }
}

/// A named temporary of a decode recipe; its dtype is
/// `DecodeRecipe::temporaries[index]`. Every temporary is defined by exactly
/// one step before any use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodeTemp(pub u32);

/// One typed step of a decode recipe. Arithmetic steps operate on `f32`
/// temporaries and round once at `f32`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeStep {
    /// Read entry `planes[plane].entry(value, field)` of the plane at
    /// ordinal `plane` for the logical value being decoded. A packed plane
    /// yields the raw `entry_bits`-wide code zero-extended to `u32`; a dense
    /// plane yields the element in its `storage_dtype`.
    ReadPlaneField {
        into: DecodeTemp,
        plane: u32,
        field: u32,
    },
    /// `into: i32 := interpretation.decode(raw, bits)` (unsigned, two's
    /// complement, zero-point offset, or table).
    InterpretCode {
        into: DecodeTemp,
        raw: DecodeTemp,
        bits: u32,
        interpretation: CodeInterpretation,
    },
    /// `into: f32 := from` converted by value (exact for every `i32` code
    /// and every `f16`/`bf16` coefficient).
    ConvertToF32 {
        into: DecodeTemp,
        from: DecodeTemp,
    },
    /// `into: f32 := left * right`, rounded once.
    Multiply {
        into: DecodeTemp,
        left: DecodeTemp,
        right: DecodeTemp,
    },
    /// `into: f32 := -from`.
    Negate {
        into: DecodeTemp,
        from: DecodeTemp,
    },
    /// `into: f32 := factor * multiplicand + addend`, rounded once (the
    /// reference evaluates the product and sum exactly and rounds to `f32`).
    MultiplyAdd {
        into: DecodeTemp,
        factor: DecodeTemp,
        multiplicand: DecodeTemp,
        addend: DecodeTemp,
    },
    /// `into: to := from` converted by value with the registry cast
    /// rounding (the final cast to a non-`f32` output).
    Cast {
        into: DecodeTemp,
        from: DecodeTemp,
        to: DType,
    },
}

impl DecodeStep {
    /// The temporary this step defines.
    pub fn defines(&self) -> DecodeTemp {
        match self {
            DecodeStep::ReadPlaneField { into, .. }
            | DecodeStep::InterpretCode { into, .. }
            | DecodeStep::ConvertToF32 { into, .. }
            | DecodeStep::Multiply { into, .. }
            | DecodeStep::Negate { into, .. }
            | DecodeStep::MultiplyAdd { into, .. }
            | DecodeStep::Cast { into, .. } => *into,
        }
    }

    /// The temporaries this step reads, in operand order.
    pub fn uses(&self) -> Vec<DecodeTemp> {
        match self {
            DecodeStep::ReadPlaneField { .. } => Vec::new(),
            DecodeStep::InterpretCode { raw, .. } => vec![*raw],
            DecodeStep::ConvertToF32 { from, .. }
            | DecodeStep::Negate { from, .. }
            | DecodeStep::Cast { from, .. } => vec![*from],
            DecodeStep::Multiply { left, right, .. } => vec![*left, *right],
            DecodeStep::MultiplyAdd {
                factor,
                multiplicand,
                addend,
                ..
            } => vec![*factor, *multiplicand, *addend],
        }
    }
}

/// The registry-provided decode of one logical value of a representation:
/// the planes it reads and the typed steps producing `output`. Encoders
/// emit the steps mechanically; none reconstructs coefficient structure,
/// code interpretation, or bias sign.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeRecipe {
    pub planes: Vec<PlaneSchema>,
    /// The dtype of every temporary, indexed by `DecodeTemp`.
    pub temporaries: Vec<DType>,
    pub steps: Vec<DecodeStep>,
    /// The temporary holding the decoded value.
    pub output: DecodeTemp,
}

impl DecodeRecipe {
    /// The dtype of one temporary of this recipe.
    pub fn dtype(&self, temp: DecodeTemp) -> DType {
        self.temporaries[temp.0 as usize]
    }
}

/// Builds a decode recipe: allocates typed temporaries and appends steps.
struct RecipeBuilder<'a> {
    name: &'static str,
    planes: &'a [PlaneSchema],
    temporaries: Vec<DType>,
    steps: Vec<DecodeStep>,
}

impl RecipeBuilder<'_> {
    fn temp(&mut self, dtype: DType) -> DecodeTemp {
        self.temporaries.push(dtype);
        DecodeTemp(self.temporaries.len() as u32 - 1)
    }

    fn ordinal(&self, field: PlaneField) -> u32 {
        match self.planes.iter().position(|plane| plane.field == field) {
            Some(ordinal) => ordinal as u32,
            None => panic!("`{}` has no `{}` plane", self.name, field.name()),
        }
    }

    fn read(&mut self, field: PlaneField, field_index: u32) -> DecodeTemp {
        let plane = self.ordinal(field);
        let dtype = self.planes[plane as usize].storage_dtype;
        let into = self.temp(dtype);
        self.steps.push(DecodeStep::ReadPlaneField {
            into,
            plane,
            field: field_index,
        });
        into
    }

    fn interpret(
        &mut self,
        raw: DecodeTemp,
        bits: u32,
        interpretation: CodeInterpretation,
    ) -> DecodeTemp {
        let into = self.temp(DType::I32);
        self.steps.push(DecodeStep::InterpretCode {
            into,
            raw,
            bits,
            interpretation,
        });
        into
    }

    fn to_f32(&mut self, from: DecodeTemp) -> DecodeTemp {
        let into = self.temp(DType::F32);
        self.steps.push(DecodeStep::ConvertToF32 { into, from });
        into
    }

    fn multiply(&mut self, left: DecodeTemp, right: DecodeTemp) -> DecodeTemp {
        let into = self.temp(DType::F32);
        self.steps.push(DecodeStep::Multiply { into, left, right });
        into
    }

    fn negate(&mut self, from: DecodeTemp) -> DecodeTemp {
        let into = self.temp(DType::F32);
        self.steps.push(DecodeStep::Negate { into, from });
        into
    }

    fn multiply_add(
        &mut self,
        factor: DecodeTemp,
        multiplicand: DecodeTemp,
        addend: DecodeTemp,
    ) -> DecodeTemp {
        let into = self.temp(DType::F32);
        self.steps.push(DecodeStep::MultiplyAdd {
            into,
            factor,
            multiplicand,
            addend,
        });
        into
    }

    fn cast(&mut self, from: DecodeTemp, to: DType) -> DecodeTemp {
        let into = self.temp(to);
        self.steps.push(DecodeStep::Cast { into, from, to });
        into
    }

    /// `factor_plane[value / factor_group] * decode(coefficients[field])`
    /// in `f32`: one hierarchical coefficient before its sign.
    fn hierarchical_coefficient(
        &mut self,
        factor_plane: PlaneField,
        field: u32,
        bits: u32,
        interpretation: CodeInterpretation,
    ) -> DecodeTemp {
        let raw = self.read(PlaneField::Coefficients, field);
        let code = self.interpret(raw, bits, interpretation);
        let code = self.to_f32(code);
        let factor = self.read(factor_plane, 0);
        let factor = self.to_f32(factor);
        self.multiply(factor, code)
    }
}

pub const REPRS: &[Repr] = &[
    // MLX affine 4-bit, group 64: bf16 scale and bias per group.
    Repr {
        name: "q4g64",
        packing_axis: PackingAxisRule::Last,
        group: 64,
        bits: 4,
        coefficients: Coefficients::Direct {
            dtype: DType::BF16,
            bias: true,
        },
        code: CodeInterpretation::Unsigned,
    },
    Repr {
        name: "q4g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 4,
        coefficients: Coefficients::Direct {
            dtype: DType::F32,
            bias: true,
        },
        code: CodeInterpretation::Unsigned,
    },
    Repr {
        name: "q4k",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 4,
        code: CodeInterpretation::Unsigned,
        coefficients: Coefficients::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            bits: 6,
            interpretation: CodeInterpretation::Unsigned,
            bias: true,
            bias_sign: -1,
        },
    },
    Repr {
        name: "q5k",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 5,
        code: CodeInterpretation::Unsigned,
        coefficients: Coefficients::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            bits: 6,
            interpretation: CodeInterpretation::Unsigned,
            bias: true,
            bias_sign: -1,
        },
    },
    Repr {
        name: "q6k",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        bits: 6,
        code: CodeInterpretation::Offset(32),
        coefficients: Coefficients::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            bits: 8,
            interpretation: CodeInterpretation::TwosComplement,
            bias: false,
            bias_sign: 1,
        },
    },
    Repr {
        name: "q8g32s",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 8,
        coefficients: Coefficients::Direct {
            dtype: DType::F16,
            bias: false,
        },
        code: CodeInterpretation::TwosComplement,
    },
    Repr {
        name: "iq4g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 4,
        coefficients: Coefficients::Direct {
            dtype: DType::F32,
            bias: false,
        },
        code: CodeInterpretation::Table(&[
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ]),
    },
    Repr {
        name: "q8g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 8,
        coefficients: Coefficients::Direct {
            dtype: DType::F32,
            bias: false,
        },
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
        match self.coefficients {
            Coefficients::Direct { bias, .. } | Coefficients::Hierarchical { bias, .. } => bias,
        }
    }
    pub fn coefficient_dtype(&self) -> DType {
        match self.coefficients {
            Coefficients::Direct { dtype, .. } => dtype,
            Coefficients::Hierarchical { .. } => DType::F32,
        }
    }
    pub fn storage_group(&self) -> u32 {
        match self.coefficients {
            Coefficients::Direct { .. } => self.group,
            Coefficients::Hierarchical { factor_group, .. } => factor_group,
        }
    }
    /// Ordered physical ABI planes with their typed field identities.
    fn plane_table(&self) -> Vec<(PlaneField, Plane)> {
        let mut result = vec![(
            PlaneField::Words,
            Plane {
                name: "words",
                group: 1,
                fields: 1,
                encoding: PlaneEncoding::Packed {
                    bits: self.bits,
                    interpretation: self.code.clone(),
                },
            },
        )];
        match &self.coefficients {
            Coefficients::Direct { dtype, bias } => {
                result.push((
                    PlaneField::Scale,
                    Plane {
                        name: "scale",
                        group: self.group,
                        fields: 1,
                        encoding: PlaneEncoding::Dense(*dtype),
                    },
                ));
                if *bias {
                    result.push((
                        PlaneField::Bias,
                        Plane {
                            name: "bias",
                            group: self.group,
                            fields: 1,
                            encoding: PlaneEncoding::Dense(*dtype),
                        },
                    ));
                }
            }
            Coefficients::Hierarchical {
                factor_group,
                factor_dtype,
                bits,
                interpretation,
                bias,
                ..
            } => {
                result.push((
                    PlaneField::Coefficients,
                    Plane {
                        name: "coefficients",
                        group: self.group,
                        fields: if *bias { 2 } else { 1 },
                        encoding: PlaneEncoding::Packed {
                            bits: *bits,
                            interpretation: interpretation.clone(),
                        },
                    },
                ));
                result.push((
                    PlaneField::ScaleFactor,
                    Plane {
                        name: "scale_factor",
                        group: *factor_group,
                        fields: 1,
                        encoding: PlaneEncoding::Dense(*factor_dtype),
                    },
                ));
                if *bias {
                    result.push((
                        PlaneField::BiasFactor,
                        Plane {
                            name: "bias_factor",
                            group: *factor_group,
                            fields: 1,
                            encoding: PlaneEncoding::Dense(*factor_dtype),
                        },
                    ));
                }
            }
        }
        result
    }

    /// Ordered physical ABI planes; logical scale/bias accessors may decode several planes.
    pub fn planes(&self) -> Vec<Plane> {
        self.plane_table()
            .into_iter()
            .map(|(_, plane)| plane)
            .collect()
    }

    /// The complete typed schema of every plane, in ABI order.
    pub fn plane_schemas(&self) -> Vec<PlaneSchema> {
        self.plane_table()
            .into_iter()
            .enumerate()
            .map(|(ordinal, (field, plane))| PlaneSchema {
                field,
                ordinal: ordinal as u32,
                group: plane.group,
                fields: plane.fields,
                entry_bits: plane.entry_bits(),
                storage_dtype: plane.dtype(),
                encoding: plane.encoding,
            })
            .collect()
    }

    /// The typed decode recipe of this representation producing `f32`
    /// (the portable `decode` result).
    pub fn decode_recipe(&self) -> DecodeRecipe {
        self.decode_recipe_to(DType::F32)
    }

    /// The typed decode recipe of this representation producing `output`
    /// (a `cast` of a packed value). The recipe is the registry's single
    /// statement of the decode: `scale * code + bias` rounded once to `f32`,
    /// where a direct coefficient is the plane value and a hierarchical
    /// coefficient is `factor * coefficient_code` (with the bias sign
    /// applied), exactly as the reference interpreter evaluates it.
    pub fn decode_recipe_to(&self, output: DType) -> DecodeRecipe {
        let planes = self.plane_schemas();
        let mut recipe = RecipeBuilder {
            name: self.name,
            planes: &planes,
            temporaries: Vec::new(),
            steps: Vec::new(),
        };
        let raw = recipe.read(PlaneField::Words, 0);
        let code = recipe.interpret(raw, self.bits, self.code.clone());
        let code_value = recipe.to_f32(code);
        let (scale, bias) = match &self.coefficients {
            Coefficients::Direct { bias, .. } => {
                let scale = recipe.read(PlaneField::Scale, 0);
                let scale = recipe.to_f32(scale);
                let bias = if *bias {
                    let bias = recipe.read(PlaneField::Bias, 0);
                    Some(recipe.to_f32(bias))
                } else {
                    None
                };
                (scale, bias)
            }
            Coefficients::Hierarchical {
                bits,
                interpretation,
                bias,
                bias_sign,
                ..
            } => {
                let scale = recipe.hierarchical_coefficient(
                    PlaneField::ScaleFactor,
                    0,
                    *bits,
                    interpretation.clone(),
                );
                let bias = if *bias {
                    let product = recipe.hierarchical_coefficient(
                        PlaneField::BiasFactor,
                        1,
                        *bits,
                        interpretation.clone(),
                    );
                    Some(match *bias_sign {
                        1 => product,
                        -1 => recipe.negate(product),
                        other => panic!("`{}` bias sign {other} is not a sign", self.name),
                    })
                } else {
                    None
                };
                (scale, bias)
            }
        };
        let value = match bias {
            Some(bias) => recipe.multiply_add(scale, code_value, bias),
            None => recipe.multiply(scale, code_value),
        };
        let output_temp = if output == DType::F32 {
            value
        } else {
            recipe.cast(value, output)
        };
        let RecipeBuilder {
            temporaries, steps, ..
        } = recipe;
        DecodeRecipe {
            planes,
            temporaries,
            steps,
            output: output_temp,
        }
    }
    pub fn plane(&self, name: &str) -> Option<Plane> {
        self.planes().into_iter().find(|p| p.name == name)
    }
    pub fn plane_index(&self, name: &str) -> Option<usize> {
        self.planes().iter().position(|p| p.name == name)
    }
    pub fn coefficient(&self, bias: bool) -> Option<Coefficient> {
        if bias && !self.has_bias() {
            return None;
        }
        Some(match self.coefficients {
            Coefficients::Direct { .. } => Coefficient::Direct {
                plane: self.plane(if bias { "bias" } else { "scale" }).unwrap(),
            },
            Coefficients::Hierarchical { bias_sign, .. } => Coefficient::Product {
                factor: self
                    .plane(if bias { "bias_factor" } else { "scale_factor" })
                    .unwrap(),
                coefficients: self.plane("coefficients").unwrap(),
                field: u32::from(bias),
                sign: if bias { bias_sign } else { 1 },
            },
        })
    }
    pub fn bits_per_value(&self) -> f64 {
        self.planes()
            .iter()
            .map(|p| p.entry_bits() as f64 * p.fields as f64 / p.group as f64)
            .sum()
    }
    pub fn groups_extent(&self, k: &Sym) -> Sym {
        k.quot(&Sym::constant(self.group as i64))
    }
}

/// Little-endian contiguous packed entry; reads only bytes containing the entry.
pub fn read_packed(bytes: &[u8], entry: usize, bits: u32) -> u32 {
    let first = entry * bits as usize;
    let mut value = 0;
    for bit in 0..bits as usize {
        value |= u32::from((bytes[(first + bit) / 8] >> ((first + bit) % 8)) & 1) << bit;
    }
    value
}
pub fn write_packed(bytes: &mut [u8], entry: usize, bits: u32, value: u32) {
    let first = entry * bits as usize;
    for bit in 0..bits as usize {
        let index = (first + bit) / 8;
        let shift = (first + bit) % 8;
        bytes[index] = (bytes[index] & !(1 << shift)) | (((value >> bit) as u8 & 1) << shift);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny reference evaluator of a decode recipe over temporaries held
    /// as `f64`, with plane contents supplied per (ordinal, entry): a packed
    /// entry is its raw code, a dense entry its value.
    fn evaluate(recipe: &DecodeRecipe, value: u64, plane_entry: &dyn Fn(u32, u64) -> f64) -> f64 {
        let mut temporaries: Vec<Option<f64>> = vec![None; recipe.temporaries.len()];
        let get = |temporaries: &[Option<f64>], temp: DecodeTemp| {
            temporaries[temp.0 as usize].expect("temporary defined before use")
        };
        for step in &recipe.steps {
            let result = match step {
                DecodeStep::ReadPlaneField { plane, field, .. } => {
                    let schema = &recipe.planes[*plane as usize];
                    plane_entry(*plane, schema.entry(value, *field))
                }
                DecodeStep::InterpretCode {
                    raw,
                    bits,
                    interpretation,
                    ..
                } => f64::from(interpretation.decode(get(&temporaries, *raw) as u32, *bits)),
                DecodeStep::ConvertToF32 { from, .. } => get(&temporaries, *from) as f32 as f64,
                DecodeStep::Multiply { left, right, .. } => {
                    (get(&temporaries, *left) * get(&temporaries, *right)) as f32 as f64
                }
                DecodeStep::Negate { from, .. } => -get(&temporaries, *from),
                DecodeStep::MultiplyAdd {
                    factor,
                    multiplicand,
                    addend,
                    ..
                } => (get(&temporaries, *factor) * get(&temporaries, *multiplicand)
                    + get(&temporaries, *addend)) as f32 as f64,
                DecodeStep::Cast { from, to, .. } => {
                    assert_eq!(*to, DType::F32, "the test evaluator casts to f32 only");
                    get(&temporaries, *from) as f32 as f64
                }
            };
            let into = step.defines();
            assert!(
                temporaries[into.0 as usize].is_none(),
                "temporary {into:?} defined twice"
            );
            temporaries[into.0 as usize] = Some(result);
        }
        get(&temporaries, recipe.output)
    }

    /// Deterministic plane contents for a representation: packed entries
    /// are raw codes below `2^bits`; dense entries are `f32`-representable
    /// values distinct per plane and entry.
    fn synthetic_plane_entry(repr: &Repr) -> impl Fn(u32, u64) -> f64 + '_ {
        move |ordinal, entry| {
            let planes = repr.planes();
            let plane = &planes[ordinal as usize];
            match plane.encoding {
                PlaneEncoding::Packed { bits, .. } => f64::from(
                    (entry as u32)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(ordinal.wrapping_mul(97))
                        & ((1u32 << bits) - 1),
                ),
                PlaneEncoding::Dense(_) => 0.375 + entry as f64 * 0.125 - f64::from(ordinal),
            }
        }
    }

    #[test]
    fn decode_recipes_match_reference_coefficient_semantics() {
        for repr in REPRS {
            let recipe = repr.decode_recipe();
            let plane_entry = synthetic_plane_entry(repr);
            let planes = repr.planes();
            assert_eq!(recipe.planes.len(), planes.len());
            for (schema, plane) in recipe.planes.iter().zip(&planes) {
                assert_eq!(schema.field.name(), plane.name);
                assert_eq!(schema.ordinal as usize, repr.plane_index(plane.name).unwrap());
                assert_eq!(schema.encoding, plane.encoding);
                assert_eq!(schema.group, plane.group);
                assert_eq!(schema.fields, plane.fields);
                assert_eq!(schema.entry_bits, plane.entry_bits());
                assert_eq!(schema.storage_dtype, plane.dtype());
            }
            let plane_value = |plane: &Plane, entry: usize| -> f32 {
                let ordinal = repr.plane_index(plane.name).unwrap() as u32;
                let raw = plane_entry(ordinal, entry as u64);
                match &plane.encoding {
                    PlaneEncoding::Packed {
                        bits,
                        interpretation,
                    } => interpretation.decode(raw as u32, *bits) as f32,
                    PlaneEncoding::Dense(_) => raw as f32,
                }
            };
            for flat in [0usize, 1, 5, 31, 32, 63, 64, 255, 256, 300, 1023] {
                // The existing `coefficient`/`decode_code` semantics as the
                // reference interpreter evaluates them.
                let coefficient = |bias: bool| -> f32 {
                    match repr.coefficient(bias) {
                        None => 0.0,
                        Some(Coefficient::Direct { plane }) => {
                            plane_value(&plane, flat / plane.group as usize)
                        }
                        Some(Coefficient::Product {
                            factor,
                            coefficients,
                            field,
                            sign,
                        }) => {
                            let code = plane_value(
                                &coefficients,
                                flat / coefficients.group as usize * coefficients.fields as usize
                                    + field as usize,
                            );
                            (plane_value(&factor, flat / factor.group as usize) * code)
                                * sign as f32
                        }
                    }
                };
                let code = plane_entry(0, flat as u64) as u32;
                let expected = (coefficient(false) as f64 * repr.decode_code(code) as f64
                    + coefficient(true) as f64) as f32 as f64;
                let actual = evaluate(&recipe, flat as u64, &plane_entry);
                assert_eq!(actual, expected, "`{}` at {flat}", repr.name);
            }
        }
    }

    #[test]
    fn decode_recipes_are_well_typed_and_single_assignment() {
        for repr in REPRS {
            for output in [DType::F32, DType::BF16] {
                let recipe = repr.decode_recipe_to(output);
                let mut defined = vec![false; recipe.temporaries.len()];
                for step in &recipe.steps {
                    for used in step.uses() {
                        assert!(defined[used.0 as usize], "`{}`: {step:?} uses an undefined temporary", repr.name);
                    }
                    let into = step.defines();
                    assert!(!defined[into.0 as usize], "`{}`: {step:?} redefines a temporary", repr.name);
                    defined[into.0 as usize] = true;
                    match step {
                        DecodeStep::ReadPlaneField { into, plane, field } => {
                            let schema = &recipe.planes[*plane as usize];
                            assert!(*field < schema.fields);
                            assert_eq!(recipe.dtype(*into), schema.storage_dtype);
                        }
                        DecodeStep::InterpretCode { into, raw, bits, .. } => {
                            assert_eq!(recipe.dtype(*raw), DType::U32);
                            assert_eq!(recipe.dtype(*into), DType::I32);
                            assert!((1..=32).contains(bits));
                        }
                        DecodeStep::ConvertToF32 { into, .. } => {
                            assert_eq!(recipe.dtype(*into), DType::F32);
                        }
                        DecodeStep::Multiply { into, left, right } => {
                            for temp in [into, left, right] {
                                assert_eq!(recipe.dtype(*temp), DType::F32);
                            }
                        }
                        DecodeStep::Negate { into, from } => {
                            assert_eq!(recipe.dtype(*into), DType::F32);
                            assert_eq!(recipe.dtype(*from), DType::F32);
                        }
                        DecodeStep::MultiplyAdd {
                            into,
                            factor,
                            multiplicand,
                            addend,
                        } => {
                            for temp in [into, factor, multiplicand, addend] {
                                assert_eq!(recipe.dtype(*temp), DType::F32);
                            }
                        }
                        DecodeStep::Cast { into, from, to } => {
                            assert_eq!(recipe.dtype(*from), DType::F32);
                            assert_eq!(recipe.dtype(*into), *to);
                        }
                    }
                }
                assert!(defined.iter().all(|defined| *defined));
                assert_eq!(recipe.dtype(recipe.output), output);
                assert_eq!(recipe.steps.last().unwrap().defines(), recipe.output);
                assert_eq!(
                    recipe.steps.iter().filter(|step| matches!(step, DecodeStep::Cast { .. })).count(),
                    usize::from(output != DType::F32)
                );
            }
        }
    }

    #[test]
    fn hierarchical_recipes_apply_the_bias_sign_and_direct_recipes_do_not_negate() {
        for repr in REPRS {
            let recipe = repr.decode_recipe();
            let negations = recipe
                .steps
                .iter()
                .filter(|step| matches!(step, DecodeStep::Negate { .. }))
                .count();
            let expected = match repr.coefficients {
                Coefficients::Hierarchical {
                    bias: true,
                    bias_sign: -1,
                    ..
                } => 1,
                Coefficients::Hierarchical { .. } | Coefficients::Direct { .. } => 0,
            };
            assert_eq!(negations, expected, "`{}`", repr.name);
            let has_add = recipe
                .steps
                .iter()
                .any(|step| matches!(step, DecodeStep::MultiplyAdd { .. }));
            assert_eq!(has_add, repr.has_bias(), "`{}`", repr.name);
        }
    }

    #[test]
    fn compact_planes_match_payload_and_cross_word_entries() {
        for (name, expected) in [("q4k", 144), ("q5k", 176), ("q6k", 210)] {
            let r = lookup(name).unwrap();
            assert_eq!(
                r.planes()
                    .iter()
                    .map(|p| p.bytes(256).unwrap())
                    .sum::<u64>(),
                expected
            );
            for plane in r.planes() {
                if let PlaneEncoding::Packed { bits, .. } = plane.encoding {
                    let n = plane.entries(512).unwrap() as usize;
                    let mut bytes = vec![0; plane.bytes(512).unwrap() as usize];
                    for i in 0..n {
                        write_packed(&mut bytes, i, bits, (i as u32).wrapping_mul(31));
                    }
                    for i in 0..n {
                        assert_eq!(
                            read_packed(&bytes, i, bits),
                            (i as u32).wrapping_mul(31) & ((1 << bits) - 1)
                        );
                    }
                    assert_eq!(plane.byte_offset(256), plane.bytes(256));
                }
            }
        }
    }
}
