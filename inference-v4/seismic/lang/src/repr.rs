//! Packed representations (crate-private tables behind `registry`). A
//! representation is a property of data: at portable scope an element read is
//! its decoded value; the physical plane structure here feeds the registry's
//! `RepresentationInfo`, the reference decoder, and the decode recipe.

use crate::types::DType;

/// One readable physical plane of a packed representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlaneField {
    Words,
    Scale,
    Bias,
    Coefficients,
    ScaleFactor,
    BiasFactor,
    BlockScale,
}

impl PlaneField {
    pub fn name(self) -> &'static str {
        match self {
            PlaneField::Words => "words",
            PlaneField::Scale => "scale",
            PlaneField::Bias => "bias",
            PlaneField::Coefficients => "coefficients",
            PlaneField::ScaleFactor => "scale_factor",
            PlaneField::BiasFactor => "bias_factor",
            PlaneField::BlockScale => "block_scale",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Repr {
    pub(crate) name: &'static str,
    pub(crate) packing_axis: PackingAxisRule,
    /// values per quantization group
    pub(crate) group: u32,
    /// bits per code
    pub(crate) bits: u32,
    pub(crate) coefficients: Coefficients,
    pub(crate) code: CodeInterpretation,
    /// Floating code carried by the value plane. Integer quantized formats
    /// leave this absent and use `code`.
    pub(crate) float_code: Option<FloatCodeFormat>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PackingAxisRule {
    Last,
}

/// Physical coefficient encoding. Hierarchical fields are interleaved scale,
/// bias (when present); factors are shared by a larger group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Coefficients {
    Direct {
        dtype: DType,
        bias: bool,
        /// Logical values carried by one physical packet. This may contain
        /// several independently scaled quantization groups.
        packet_group: u32,
    },
    Hierarchical {
        factor_group: u32,
        factor_dtype: DType,
        bits: u32,
        interpretation: CodeInterpretation,
        bias: bool,
        bias_sign: i32,
    },
    /// One floating scale code shared by the representation group.
    BlockFloat { format: FloatCodeFormat },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaneEncoding {
    Dense(DType),
    Packed {
        bits: u32,
        interpretation: CodeInterpretation,
    },
    /// A closed floating code format packed bytewise, never a scalar dtype.
    FloatCode {
        format: FloatCodeFormat,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatCodeFormat {
    E2M1,
    E4M3,
    /// NVIDIA unsigned E4M3 scale code. Its physical byte is E4M3FN with
    /// the sign bit ignored by NVFP4 matrix hardware.
    UE4M3,
}

impl FloatCodeFormat {
    pub const fn bits(self) -> u32 {
        match self {
            Self::E2M1 => 4,
            Self::E4M3 | Self::UE4M3 => 8,
        }
    }

    /// Exact value of the closed NVIDIA floating-code format.
    pub fn decode(self, raw: u32) -> f32 {
        let bits = self.bits();
        let raw = raw & ((1u32 << bits) - 1);
        let sign = if raw & (1 << (bits - 1)) == 0 {
            1.0
        } else {
            -1.0
        };
        match self {
            Self::E2M1 => {
                let exponent = (raw >> 1) & 0x3;
                let mantissa = raw & 0x1;
                if exponent == 0 {
                    sign * (mantissa as f32 * 0.5)
                } else {
                    sign * (1.0 + mantissa as f32 * 0.5) * 2.0f32.powi(exponent as i32 - 1)
                }
            }
            Self::E4M3 | Self::UE4M3 => {
                let sign = if self == Self::UE4M3 { 1.0 } else { sign };
                let raw = if self == Self::UE4M3 { raw & 0x7f } else { raw };
                let exponent = (raw >> 3) & 0xf;
                let mantissa = raw & 0x7;
                if exponent == 0 {
                    sign * (mantissa as f32 / 8.0) * 2.0f32.powi(-6)
                } else if exponent == 0xf && mantissa == 0x7 {
                    f32::NAN.copysign(sign)
                } else {
                    sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
                }
            }
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Plane {
    pub(crate) name: &'static str,
    /// Logical values sharing `fields` entries in this plane.
    pub(crate) group: u32,
    pub(crate) fields: u32,
    pub(crate) encoding: PlaneEncoding,
}
/// Reference coefficient structure; the decode-recipe tests check every
/// recipe against it.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Coefficient {
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
    pub(crate) fn dtype(&self) -> DType {
        match self.encoding {
            PlaneEncoding::Dense(dtype) => dtype,
            PlaneEncoding::Packed { .. } => DType::U32,
            PlaneEncoding::FloatCode { .. } => DType::U32,
        }
    }
    pub(crate) fn entry_bits(&self) -> u32 {
        match self.encoding {
            PlaneEncoding::Dense(dtype) => dtype.bytes() * 8,
            PlaneEncoding::Packed { bits, .. } => bits,
            PlaneEncoding::FloatCode { format } => format.bits(),
        }
    }
    /// Plane bytes for `values` logical values; the tests check the plane
    /// tables against the published payload sizes with it.
    #[cfg(test)]
    pub(crate) fn bytes(&self, values: u64) -> Option<u64> {
        let entries = values
            .div_ceil(u64::from(self.group))
            .checked_mul(u64::from(self.fields))?;
        match self.encoding {
            PlaneEncoding::Dense(dtype) => entries.checked_mul(u64::from(dtype.bytes())),
            PlaneEncoding::Packed { bits, .. } => entries
                .checked_mul(u64::from(bits))
                .map(|n| n.div_ceil(32) * u64::from(DType::U32.bytes())),
            PlaneEncoding::FloatCode { format } => entries
                .checked_mul(u64::from(format.bits()))
                .map(|bits| bits.div_ceil(8)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CodeInterpretation {
    Unsigned,
    TwosComplement,
    Offset(i32),
    Table(&'static [i32]),
}
#[cfg(test)]
impl Repr {
    pub(crate) fn decode_code(&self, raw: u32) -> i32 {
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

/// A typed temporary of one sealed decode recipe. Handles are created only by
/// `RecipeBuilder`; the representation/output pair prevents a temporary from
/// one recipe being used with another recipe that happens to have the same
/// ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodeTemp {
    representation: &'static str,
    output: DType,
    ordinal: u32,
    dtype: DType,
}

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
    /// Decode one closed floating code to its exact `f32` value.
    DecodeFloatCode {
        into: DecodeTemp,
        raw: DecodeTemp,
        format: FloatCodeFormat,
    },
    /// `into: f32 := from` converted by value (exact for every `i32` code
    /// and every `f16`/`bf16` coefficient).
    ConvertToF32 { into: DecodeTemp, from: DecodeTemp },
    /// `into: f32 := left * right`, rounded once.
    Multiply {
        into: DecodeTemp,
        left: DecodeTemp,
        right: DecodeTemp,
    },
    /// `into: f32 := -from`.
    Negate { into: DecodeTemp, from: DecodeTemp },
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
            | DecodeStep::DecodeFloatCode { into, .. }
            | DecodeStep::ConvertToF32 { into, .. }
            | DecodeStep::Multiply { into, .. }
            | DecodeStep::Negate { into, .. }
            | DecodeStep::MultiplyAdd { into, .. }
            | DecodeStep::Cast { into, .. } => *into,
        }
    }

    /// The temporaries this step reads, in operand order.
    pub(crate) fn uses(&self) -> Vec<DecodeTemp> {
        match self {
            DecodeStep::ReadPlaneField { .. } => Vec::new(),
            DecodeStep::InterpretCode { raw, .. } => vec![*raw],
            DecodeStep::DecodeFloatCode { raw, .. } => vec![*raw],
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
    representation: &'static str,
    output_dtype: DType,
    planes: Vec<PlaneSchema>,
    temporary_count: u32,
    steps: Vec<DecodeStep>,
    /// The temporary holding the decoded value.
    output: DecodeTemp,
}

impl DecodeRecipe {
    pub fn planes(&self) -> &[PlaneSchema] {
        &self.planes
    }

    pub fn steps(&self) -> &[DecodeStep] {
        &self.steps
    }

    pub fn temporary_count(&self) -> usize {
        self.temporary_count as usize
    }

    pub fn output(&self) -> DecodeTemp {
        self.output
    }

    /// The dtype of one temporary. A foreign handle is an internal registry
    /// programming error, not a runtime decoding condition.
    pub fn dtype(&self, temp: DecodeTemp) -> DType {
        self.assert_owns(temp);
        temp.dtype
    }

    /// Dense ordinal used only to index emitter-local values. The recipe has
    /// already proved that every operand precedes its defining step.
    pub fn ordinal(&self, temp: DecodeTemp) -> usize {
        self.assert_owns(temp);
        temp.ordinal as usize
    }

    fn assert_owns(&self, temp: DecodeTemp) {
        assert_eq!(
            (temp.representation, temp.output),
            (self.representation, self.output_dtype),
            "decode temporary belongs to another sealed recipe"
        );
        assert!(
            temp.ordinal < self.temporary_count,
            "decode temporary is outside its sealed recipe"
        );
    }
}

/// Builds a decode recipe: allocates typed temporaries and appends steps.
struct RecipeBuilder<'a> {
    name: &'static str,
    output: DType,
    planes: &'a [PlaneSchema],
    temporaries: Vec<DType>,
    steps: Vec<DecodeStep>,
}

impl RecipeBuilder<'_> {
    fn temp(&mut self, dtype: DType) -> DecodeTemp {
        self.temporaries.push(dtype);
        DecodeTemp {
            representation: self.name,
            output: self.output,
            ordinal: self.temporaries.len() as u32 - 1,
            dtype,
        }
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

    fn float_code(&mut self, raw: DecodeTemp, format: FloatCodeFormat) -> DecodeTemp {
        let into = self.temp(DType::F32);
        self.steps
            .push(DecodeStep::DecodeFloatCode { into, raw, format });
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

pub(crate) const REPRS: &[Repr] = &[
    // MLX affine 4-bit, group 64: bf16 scale and bias per group.
    Repr {
        name: "q4g64",
        packing_axis: PackingAxisRule::Last,
        group: 64,
        bits: 4,
        coefficients: Coefficients::Direct {
            dtype: DType::BF16,
            bias: true,
            packet_group: 64,
        },
        code: CodeInterpretation::Unsigned,
        float_code: None,
    },
    Repr {
        name: "q4g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 4,
        coefficients: Coefficients::Direct {
            dtype: DType::F32,
            bias: true,
            packet_group: 32,
        },
        code: CodeInterpretation::Unsigned,
        float_code: None,
    },
    Repr {
        name: "q4k",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 4,
        code: CodeInterpretation::Unsigned,
        float_code: None,
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
        float_code: None,
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
        float_code: None,
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
            packet_group: 32,
        },
        code: CodeInterpretation::TwosComplement,
        float_code: None,
    },
    Repr {
        name: "iq4g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 4,
        coefficients: Coefficients::Direct {
            dtype: DType::F32,
            bias: false,
            packet_group: 256,
        },
        code: CodeInterpretation::Table(&[
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ]),
        float_code: None,
    },
    Repr {
        name: "q8g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 8,
        coefficients: Coefficients::Direct {
            dtype: DType::F32,
            bias: false,
            packet_group: 32,
        },
        code: CodeInterpretation::Unsigned,
        float_code: None,
    },
    Repr {
        name: "nvfp4_e2m1_block16",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        bits: 4,
        coefficients: Coefficients::BlockFloat {
            format: FloatCodeFormat::UE4M3,
        },
        // Physical extraction is unsigned; interpretation is the explicit
        // floating-code decode below.
        code: CodeInterpretation::Unsigned,
        float_code: Some(FloatCodeFormat::E2M1),
    },
];

pub(crate) fn lookup(name: &str) -> Option<&'static Repr> {
    REPRS.iter().find(|r| r.name == name)
}

impl CodeInterpretation {
    pub(crate) fn decode(&self, raw: u32, bits: u32) -> i32 {
        let recipe = crate::reference_math::code_recipe(self, bits);
        let value = crate::reference_math::evaluate(
            &recipe,
            &[crate::reference_math::ReferenceScalar::U32(raw)],
        ).expect("registry code interpretation is total");
        let crate::reference_math::ReferenceScalar::I32(value) = value else { unreachable!() };
        value
    }
}

#[cfg(test)]
impl Repr {
    pub(crate) fn has_bias(&self) -> bool {
        match self.coefficients {
            Coefficients::Direct { bias, .. } | Coefficients::Hierarchical { bias, .. } => bias,
            Coefficients::BlockFloat { .. } => false,
        }
    }
}

impl Repr {
    pub(crate) fn storage_group(&self) -> u32 {
        match self.coefficients {
            Coefficients::Direct { packet_group, .. } => packet_group,
            Coefficients::Hierarchical { factor_group, .. } => factor_group,
            Coefficients::BlockFloat { .. } => self.group,
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
                encoding: match self.float_code {
                    Some(format) => PlaneEncoding::FloatCode { format },
                    None => PlaneEncoding::Packed {
                        bits: self.bits,
                        interpretation: self.code.clone(),
                    },
                },
            },
        )];
        match &self.coefficients {
            Coefficients::Direct { dtype, bias, .. } => {
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
            Coefficients::BlockFloat { format } => result.push((
                PlaneField::BlockScale,
                Plane {
                    name: "block_scale",
                    group: self.group,
                    fields: 1,
                    encoding: PlaneEncoding::FloatCode { format: *format },
                },
            )),
        }
        result
    }

    /// Ordered physical ABI planes; logical scale/bias accessors may decode several planes.
    pub(crate) fn planes(&self) -> Vec<Plane> {
        self.plane_table()
            .into_iter()
            .map(|(_, plane)| plane)
            .collect()
    }

    /// The complete typed schema of every plane, in ABI order.
    pub(crate) fn plane_schemas(&self) -> Vec<PlaneSchema> {
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

    /// The typed decode recipe of this representation producing `output`
    /// (a `cast` of a packed value). The recipe is the registry's single
    /// statement of the decode: `scale * code + bias` rounded once to `f32`,
    /// where a direct coefficient is the plane value and a hierarchical
    /// coefficient is `factor * coefficient_code` (with the bias sign
    /// applied), exactly as the reference interpreter evaluates it.
    pub(crate) fn decode_recipe_to(&self, output: DType) -> DecodeRecipe {
        let planes = self.plane_schemas();
        let mut recipe = RecipeBuilder {
            name: self.name,
            output,
            planes: &planes,
            temporaries: Vec::new(),
            steps: Vec::new(),
        };
        let raw = recipe.read(PlaneField::Words, 0);
        let code_value = match self.float_code {
            Some(format) => recipe.float_code(raw, format),
            None => {
                let code = recipe.interpret(raw, self.bits, self.code.clone());
                recipe.to_f32(code)
            }
        };
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
                    Some(if *bias_sign < 0 {
                        recipe.negate(product)
                    } else {
                        product
                    })
                } else {
                    None
                };
                (scale, bias)
            }
            Coefficients::BlockFloat { format } => {
                let raw = recipe.read(PlaneField::BlockScale, 0);
                (recipe.float_code(raw, *format), None)
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
            name,
            output,
            temporaries,
            steps,
            ..
        } = recipe;
        assert_eq!(
            output_temp.dtype, output,
            "decode recipe output type differs from its requested type"
        );
        assert_eq!(
            steps.len(),
            temporaries.len(),
            "decode recipe must define exactly one new temporary per step"
        );
        for (ordinal, step) in steps.iter().enumerate() {
            let defined = step.defines();
            assert_eq!(
                defined.ordinal as usize, ordinal,
                "decode recipe temporary definitions are not canonical"
            );
            for used in step.uses() {
                assert_eq!(
                    (used.representation, used.output),
                    (name, output),
                    "decode step uses a temporary from another recipe"
                );
                assert!(
                    used.ordinal < defined.ordinal,
                    "decode recipe reads a temporary before its definition"
                );
            }
        }
        DecodeRecipe {
            representation: name,
            output_dtype: output,
            planes,
            temporary_count: u32::try_from(temporaries.len())
                .expect("decode recipe has more than u32::MAX temporaries"),
            steps,
            output: output_temp,
        }
    }
}

#[cfg(test)]
impl Repr {
    pub(crate) fn plane_index(&self, name: &str) -> Option<usize> {
        self.planes().iter().position(|p| p.name == name)
    }
    /// The logical scale (`bias == false`) or bias coefficient structure.
    /// `None` when the representation has no bias.
    pub(crate) fn coefficient(&self, bias: bool) -> Option<Coefficient> {
        if bias && !self.has_bias() {
            return None;
        }
        let table = self.plane_table();
        let find = |field: PlaneField| {
            table
                .iter()
                .find(|(candidate, _)| *candidate == field)
                .map(|(_, plane)| plane.clone())
        };
        match self.coefficients {
            Coefficients::Direct { .. } => Some(Coefficient::Direct {
                plane: find(if bias {
                    PlaneField::Bias
                } else {
                    PlaneField::Scale
                })?,
            }),
            Coefficients::Hierarchical { bias_sign, .. } => Some(Coefficient::Product {
                factor: find(if bias {
                    PlaneField::BiasFactor
                } else {
                    PlaneField::ScaleFactor
                })?,
                coefficients: find(PlaneField::Coefficients)?,
                field: u32::from(bias),
                sign: if bias { bias_sign } else { 1 },
            }),
            Coefficients::BlockFloat { .. } => Some(Coefficient::Direct {
                plane: find(PlaneField::BlockScale)?,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny reference evaluator of a decode recipe over temporaries held
    /// as `f64`, with plane contents supplied per (ordinal, entry): a packed
    /// entry is its raw code, a dense entry its value.
    fn evaluate(recipe: &DecodeRecipe, value: u64, plane_entry: &dyn Fn(u32, u64) -> f64) -> f64 {
        let mut temporaries: Vec<f64> = Vec::with_capacity(recipe.temporary_count());
        let get = |temporaries: &[f64], temp: DecodeTemp| temporaries[recipe.ordinal(temp)];
        for step in recipe.steps() {
            let result = match step {
                DecodeStep::ReadPlaneField { plane, field, .. } => {
                    let schema = &recipe.planes()[*plane as usize];
                    plane_entry(*plane, schema.entry(value, *field))
                }
                DecodeStep::InterpretCode {
                    raw,
                    bits,
                    interpretation,
                    ..
                } => f64::from(interpretation.decode(get(&temporaries, *raw) as u32, *bits)),
                DecodeStep::DecodeFloatCode { raw, format, .. } => {
                    f64::from(format.decode(get(&temporaries, *raw) as u32))
                }
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
                } => {
                    (get(&temporaries, *factor) * get(&temporaries, *multiplicand)
                        + get(&temporaries, *addend)) as f32 as f64
                }
                DecodeStep::Cast { from, to, .. } => {
                    assert_eq!(*to, DType::F32, "the test evaluator casts to f32 only");
                    get(&temporaries, *from) as f32 as f64
                }
            };
            let into = step.defines();
            assert!(
                recipe.ordinal(into) == temporaries.len(),
                "sealed recipe definition order changed"
            );
            temporaries.push(result);
        }
        get(&temporaries, recipe.output())
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
                PlaneEncoding::FloatCode { format } => f64::from(
                    (entry as u32)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(ordinal.wrapping_mul(97))
                        & ((1u32 << format.bits()) - 1),
                ),
                PlaneEncoding::Dense(_) => 0.375 + entry as f64 * 0.125 - f64::from(ordinal),
            }
        }
    }

    #[test]
    fn decode_recipes_match_reference_coefficient_semantics() {
        for repr in REPRS {
            let recipe = repr.decode_recipe_to(DType::F32);
            let plane_entry = synthetic_plane_entry(repr);
            let planes = repr.planes();
            assert_eq!(recipe.planes().len(), planes.len());
            for (schema, plane) in recipe.planes().iter().zip(&planes) {
                assert_eq!(schema.field.name(), plane.name);
                assert_eq!(
                    schema.ordinal as usize,
                    repr.plane_index(plane.name).unwrap()
                );
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
                    PlaneEncoding::FloatCode { format } => format.decode(raw as u32),
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
                let decoded_code = match repr.float_code {
                    Some(format) => format.decode(code),
                    None => repr.decode_code(code) as f32,
                };
                let expected = (coefficient(false) as f64 * f64::from(decoded_code)
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
                let mut defined = vec![false; recipe.temporary_count()];
                for step in recipe.steps() {
                    for used in step.uses() {
                        assert!(
                            defined[recipe.ordinal(used)],
                            "`{}`: {step:?} uses an undefined temporary",
                            repr.name
                        );
                    }
                    let into = step.defines();
                    assert!(
                        !defined[recipe.ordinal(into)],
                        "`{}`: {step:?} redefines a temporary",
                        repr.name
                    );
                    defined[recipe.ordinal(into)] = true;
                    match step {
                        DecodeStep::ReadPlaneField { into, plane, field } => {
                            let schema = &recipe.planes()[*plane as usize];
                            assert!(*field < schema.fields);
                            assert_eq!(recipe.dtype(*into), schema.storage_dtype);
                        }
                        DecodeStep::InterpretCode {
                            into, raw, bits, ..
                        } => {
                            assert_eq!(recipe.dtype(*raw), DType::U32);
                            assert_eq!(recipe.dtype(*into), DType::I32);
                            assert!((1..=32).contains(bits));
                        }
                        DecodeStep::DecodeFloatCode {
                            into, raw, format, ..
                        } => {
                            assert_eq!(recipe.dtype(*raw), DType::U32);
                            assert_eq!(recipe.dtype(*into), DType::F32);
                            assert!(matches!(
                                format,
                                FloatCodeFormat::E2M1
                                    | FloatCodeFormat::E4M3
                                    | FloatCodeFormat::UE4M3
                            ));
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
                assert_eq!(recipe.dtype(recipe.output()), output);
                assert_eq!(recipe.steps().last().unwrap().defines(), recipe.output());
                assert_eq!(
                    recipe
                        .steps
                        .iter()
                        .filter(|step| matches!(step, DecodeStep::Cast { .. }))
                        .count(),
                    usize::from(output != DType::F32)
                );
            }
        }
    }

    #[test]
    fn hierarchical_recipes_apply_the_bias_sign_and_direct_recipes_do_not_negate() {
        for repr in REPRS {
            let recipe = repr.decode_recipe_to(DType::F32);
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
                Coefficients::Hierarchical { .. }
                | Coefficients::Direct { .. }
                | Coefficients::BlockFloat { .. } => 0,
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
    fn compact_planes_match_payload() {
        for (name, expected) in [("q4k", 144), ("q5k", 176), ("q6k", 210)] {
            let r = lookup(name).unwrap();
            assert_eq!(
                r.planes()
                    .iter()
                    .map(|p| p.bytes(256).unwrap())
                    .sum::<u64>(),
                expected
            );
        }
    }
}
