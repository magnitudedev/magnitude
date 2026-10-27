//! Packed representations (crate-private tables behind `registry`). A
//! representation is a property of data: at portable scope an element read is
//! its decoded value; the physical plane structure here feeds the registry's
//! `RepresentationInfo`, the reference decoder, and the decode recipe.

use crate::external_tables::RepackTable;
use crate::types::DType;
use magnitude_ggml::tables;

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
        /// The coefficient is the factor times the code times
        /// `2^factor_exponent` (an exact power-of-two scaling).
        factor_exponent: i32,
        bits: u32,
        interpretation: CodeInterpretation,
        /// The minimum term after each scale code, and how it enters a value.
        minimum: Option<MinimumSign>,
        /// Row layouts carry the coefficient codes in the grid entries
        /// (`GridCodes::carried_bit`) instead of a `scales` plane.
        carried: bool,
    },
    /// One floating scale code shared by the representation group.
    BlockFloat {
        format: FloatCodeFormat,
        /// Logical values carried by one physical packet (a multiple of
        /// the representation group).
        packet_group: u32,
    },
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
    /// OCP microscaling E8M0 scale code: the power of two `2^(raw - 127)`,
    /// with no sign and no mantissa; 0xff is NaN. `2^-127` (raw 0) is an F32
    /// subnormal, so every code decodes exactly.
    E8M0,
}

impl FloatCodeFormat {
    /// A value code's ggml codebook and the exponent its values are scaled
    /// by: E2M1 is ggml's `kvalues_mxfp4` (E2M1 doubled, −0 as +0) times
    /// `2^-1`. Scale codes have none.
    pub const fn codebook(self) -> Option<(&'static tables::Codebook, i32)> {
        match self {
            Self::E2M1 => Some((&tables::MXFP4, -1)),
            Self::E4M3 | Self::UE4M3 | Self::E8M0 => None,
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::E2M1 => 4,
            Self::E4M3 | Self::UE4M3 | Self::E8M0 => 8,
        }
    }

    /// Exact value of the closed NVIDIA / OCP floating-code format.
    pub fn decode(self, raw: u32) -> f32 {
        let bits = self.bits();
        let raw = raw & ((1u32 << bits) - 1);
        let sign = if raw & (1 << (bits - 1)) == 0 {
            1.0
        } else {
            -1.0
        };
        match self {
            Self::E8M0 => match raw {
                0 => f32::from_bits(0x0040_0000),
                0xff => f32::NAN,
                exponent => f32::from_bits(exponent << 23),
            },
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
        exponent: i32,
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
    /// The odd code `2c + 1` (ggml's IQ sub-block scales `0.5 + c`, doubled).
    Odd,
    /// Code `c` names `values[c]` of a ggml codebook.
    Table(&'static tables::Codebook),
    /// One entry codes eight values through a grid (`GridCodes`); the code
    /// of a value depends on its lane within the eight.
    Grid(&'static GridCodes),
}

/// A code interpretation without its parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodeKind {
    Unsigned,
    TwosComplement,
    Offset,
    Odd,
    Table,
    Grid,
}

impl CodeKind {
    pub const ALL: [Self; 6] = [
        Self::Unsigned,
        Self::TwosComplement,
        Self::Offset,
        Self::Odd,
        Self::Table,
        Self::Grid,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Unsigned => "UNSIGNED",
            Self::TwosComplement => "TWOS",
            Self::Offset => "OFFSET",
            Self::Odd => "ODD",
            Self::Table => "TABLE",
            Self::Grid => "GRID",
        }
    }
}

impl CodeInterpretation {
    pub const fn kind(&self) -> CodeKind {
        match self {
            Self::Unsigned => CodeKind::Unsigned,
            Self::TwosComplement => CodeKind::TwosComplement,
            Self::Offset(_) => CodeKind::Offset,
            Self::Odd => CodeKind::Odd,
            Self::Table(_) => CodeKind::Table,
            Self::Grid(_) => CodeKind::Grid,
        }
    }
}

/// Vector codes (the GGUF IQ grids): one entry codes `GRID_VALUES`
/// consecutive values. It holds `GRID_VALUES / lanes` grid indices of
/// `index_bits` each, first index lowest, then `GRID_SIGN_BITS` sign bits;
/// an eighth sign bit makes the count of set signs even (ggml's
/// `ksigns_iq2xs`). The code of the value at `lane` is the grid magnitude
/// for that lane, negated when its sign bit is set: byte `lane % lanes` of
/// `points[index lane / lanes]`. An entry is `entry_bits` wide, whole bytes
/// (so every layout addresses entries bytewise); bits past the signs are
/// zero.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct GridCodes {
    pub entry_bits: u32,
    /// The ggml grid: byte `j` of point `i` is the magnitude of its lane `j`.
    pub grid: &'static tables::Grid,
}

/// A constant table a kernel's code looks up: a conversion recipe's
/// registered table (`RepackExpr::Lookup`), or a grid's magnitudes
/// ([`GridCodes::magnitudes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LookupTable {
    Repack(RepackTable),
    Grid(&'static GridCodes),
}

impl LookupTable {
    pub fn values(self) -> &'static [i32] {
        match self {
            Self::Repack(table) => table.values(),
            Self::Grid(grid) => grid.magnitudes(),
        }
    }
}

/// Values one grid entry codes.
pub const GRID_VALUES: u32 = 8;
/// Explicit sign bits of one grid entry.
pub const GRID_SIGN_BITS: u32 = 7;

impl GridCodes {
    /// The grid's points.
    pub const fn points(&self) -> tables::GridPoints {
        self.grid.points
    }

    /// Bits of one grid index: the grid's points are a power of two.
    pub const fn index_bits(&self) -> u32 {
        let points = self.points().len();
        assert!(points.is_power_of_two(), "a grid of a power of two points");
        points.trailing_zeros()
    }

    /// Values one grid point codes.
    pub const fn lanes(&self) -> u32 {
        self.points().lanes()
    }

    /// Grid indices one entry holds.
    pub const fn indices(&self) -> u32 {
        GRID_VALUES / self.lanes()
    }

    /// Bits of one entry.
    pub const fn entry_bits(&self) -> u32 {
        assert!(
            self.entry_bits % 8 == 0
                && self.entry_bits >= self.indices() * self.index_bits() + GRID_SIGN_BITS,
            "a grid entry is whole bytes holding its indices and signs"
        );
        self.entry_bits
    }

    /// The first entry bit past its indices and signs. A representation
    /// whose coefficients are carried (`Coefficients::Hierarchical::carried`)
    /// places, in its row layouts, bit `j` of a group's coefficient code at
    /// this bit of the group's entry `j`; the packet form keeps it zero.
    pub const fn carried_bit(&self) -> u32 {
        self.indices() * self.index_bits() + GRID_SIGN_BITS
    }

    /// The code of the value at `lane` (< `GRID_VALUES`) of `entry`.
    pub fn code(&self, entry: u32, lane: u32) -> i32 {
        let index_mask = (1u32 << self.index_bits()) - 1;
        let index = (entry >> (self.index_bits() * (lane / self.lanes()))) & index_mask;
        let magnitude = self.points().byte(index as usize, lane % self.lanes());
        let signs = (entry >> (self.index_bits() * self.indices())) & ((1 << GRID_SIGN_BITS) - 1);
        let signs = signs | ((signs.count_ones() & 1) << GRID_SIGN_BITS);
        if (signs >> lane) & 1 == 0 {
            i32::from(magnitude)
        } else {
            -i32::from(magnitude)
        }
    }

    /// The grid's magnitudes as one table: point `i`'s lane `j` at
    /// `i * lanes + j`. Built once per grid, for code generators that read a
    /// table.
    pub fn magnitudes(&'static self) -> &'static [i32] {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};
        static TABLES: OnceLock<Mutex<HashMap<usize, &'static [i32]>>> = OnceLock::new();
        let mut tables = TABLES
            .get_or_init(Default::default)
            .lock()
            .expect("grid tables lock");
        tables
            .entry(self as *const Self as usize)
            .or_insert_with(|| {
                (0..self.points().len())
                    .flat_map(|point| {
                        (0..self.lanes())
                            .map(move |lane| i32::from(self.points().byte(point, lane)))
                    })
                    .collect::<Vec<_>>()
                    .leak()
            })
    }

    /// Every code the grid gives, ascending.
    pub fn codes(&self) -> Vec<i32> {
        let mut codes = (0..self.points().len())
            .flat_map(|point| (0..self.lanes()).map(move |lane| self.points().byte(point, lane)))
            .flat_map(|magnitude| [i32::from(magnitude), -i32::from(magnitude)])
            .collect::<Vec<_>>();
        codes.sort_unstable();
        codes.dedup();
        codes
    }
}

impl CodeInterpretation {
    /// Every code the interpretation gives a `bits`-wide entry, ascending.
    pub fn codes(&self, bits: u32) -> Vec<i32> {
        match self {
            Self::Grid(grid) => grid.codes(),
            _ => {
                let mut codes = (0..1u32 << bits)
                    .map(|raw| self.decode(raw, bits))
                    .collect::<Vec<_>>();
                codes.sort_unstable();
                codes.dedup();
                codes
            }
        }
    }
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

/// The entry a decode reads: field `field` of plane `plane`
/// (`DecodeStep::ReadPlaneField`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneRead {
    pub plane: u32,
    pub field: u32,
}

/// The reads of a decode whose value is its integer code times one stored
/// scale (`DecodeRecipe::scaled_codes`): the code is the low `bits` bits of
/// entry `code`, and its offset from the least code is those bits XOR
/// `flip`; the scale is entry `scale` as stored, of dtype `scale_dtype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaledCodes {
    pub code: PlaneRead,
    pub bits: u32,
    pub flip: u32,
    pub scale: PlaneRead,
    pub scale_dtype: DType,
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
    /// `into: u32 :=` the position of the logical value being decoded among
    /// the `planes[plane].group` values one entry of the plane at ordinal
    /// `plane` covers: a vector code's lane.
    ReadPlaneLane { into: DecodeTemp, plane: u32 },
    /// `into: i32 := interpretation.decode(raw, bits)` (unsigned, two's
    /// complement, zero-point offset, or table).
    InterpretCode {
        into: DecodeTemp,
        raw: DecodeTemp,
        bits: u32,
        interpretation: CodeInterpretation,
    },
    /// `into: i32 := grid.code(raw, lane)`: a vector code's value at its
    /// lane.
    InterpretGridCode {
        into: DecodeTemp,
        raw: DecodeTemp,
        lane: DecodeTemp,
        grid: &'static GridCodes,
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
    /// `into := value`, a constant of the representation (the power of two
    /// a hierarchical factor is scaled by).
    Constant {
        into: DecodeTemp,
        value: crate::reference_math::ReferenceScalar,
    },
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
            | DecodeStep::ReadPlaneLane { into, .. }
            | DecodeStep::InterpretCode { into, .. }
            | DecodeStep::InterpretGridCode { into, .. }
            | DecodeStep::DecodeFloatCode { into, .. }
            | DecodeStep::ConvertToF32 { into, .. }
            | DecodeStep::Constant { into, .. }
            | DecodeStep::Multiply { into, .. }
            | DecodeStep::Negate { into, .. }
            | DecodeStep::MultiplyAdd { into, .. }
            | DecodeStep::Cast { into, .. } => *into,
        }
    }

    /// The temporaries this step reads, in operand order.
    pub(crate) fn uses(&self) -> Vec<DecodeTemp> {
        match self {
            DecodeStep::ReadPlaneField { .. }
            | DecodeStep::ReadPlaneLane { .. }
            | DecodeStep::Constant { .. } => Vec::new(),
            DecodeStep::InterpretCode { raw, .. } => vec![*raw],
            DecodeStep::InterpretGridCode { raw, lane, .. } => vec![*raw, *lane],
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

/// Semantic coefficient roles in the registry-owned affine decode. These do
/// not name a physical plane or permit a caller to reinterpret storage bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QuantizedComponent {
    Code,
    CodeValue,
    ScaleCode,
    BiasCode,
    Scale,
    Bias,
    GroupSize,
    CodeMin,
    CodeMax,
    ScaleFactor,
    BiasFactor,
    CoefficientCodeMax,
    BiasPresent,
}
impl QuantizedComponent {
    pub const ALL: [Self; 13] = [
        Self::Code,
        Self::CodeValue,
        Self::ScaleCode,
        Self::BiasCode,
        Self::Scale,
        Self::Bias,
        Self::GroupSize,
        Self::CodeMin,
        Self::CodeMax,
        Self::ScaleFactor,
        Self::BiasFactor,
        Self::CoefficientCodeMax,
        Self::BiasPresent,
    ];

    pub fn requires_integer_code(self) -> bool {
        matches!(self, Self::Code | Self::CodeMin | Self::CodeMax)
    }
    pub fn dtype(self) -> DType {
        match self {
            Self::Code
            | Self::ScaleCode
            | Self::BiasCode
            | Self::GroupSize
            | Self::CodeMin
            | Self::CodeMax
            | Self::CoefficientCodeMax
            | Self::BiasPresent => DType::I32,
            Self::CodeValue | Self::Scale | Self::Bias | Self::ScaleFactor | Self::BiasFactor => {
                DType::F32
            }
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::CodeValue => "code_value",
            Self::ScaleCode => "scale_code",
            Self::BiasCode => "bias_code",
            Self::Scale => "scale",
            Self::Bias => "bias",
            Self::GroupSize => "group_size",
            Self::CodeMin => "code_min",
            Self::CodeMax => "code_max",
            Self::ScaleFactor => "scale_factor",
            Self::BiasFactor => "bias_factor",
            Self::CoefficientCodeMax => "coefficient_code_max",
            Self::BiasPresent => "bias_present",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeComponent {
    Temporary(DecodeTemp),
    Constant(crate::reference_math::ReferenceScalar),
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
    integer_code: Option<DecodeTemp>,
    code_value: DecodeTemp,
    scale_code: Option<DecodeTemp>,
    bias_code: Option<DecodeTemp>,
    scale: DecodeTemp,
    bias: Option<DecodeTemp>,
    group: u32,
    scale_factor: DecodeTemp,
    bias_factor: Option<DecodeTemp>,
    coefficient_code_max: i32,
}

impl DecodeRecipe {
    /// Resolve a semantic role from the same recipe that decodes this value.
    /// Floating-code formats intentionally have no integer-code role.
    pub fn component(&self, role: QuantizedComponent) -> Option<DecodeComponent> {
        match role {
            QuantizedComponent::Code => self.integer_code.map(DecodeComponent::Temporary),
            QuantizedComponent::CodeValue => Some(DecodeComponent::Temporary(self.code_value)),
            QuantizedComponent::ScaleCode => Some(match self.scale_code {
                Some(code) => DecodeComponent::Temporary(code),
                None => DecodeComponent::Constant(crate::reference_math::ReferenceScalar::I32(1)),
            }),
            QuantizedComponent::BiasCode => Some(match self.bias_code {
                Some(code) => DecodeComponent::Temporary(code),
                None => DecodeComponent::Constant(crate::reference_math::ReferenceScalar::I32(
                    i32::from(self.bias.is_some()),
                )),
            }),
            QuantizedComponent::Scale => Some(DecodeComponent::Temporary(self.scale)),
            QuantizedComponent::BiasPresent => Some(DecodeComponent::Constant(
                crate::reference_math::ReferenceScalar::I32(i32::from(self.bias.is_some())),
            )),
            QuantizedComponent::ScaleFactor => Some(DecodeComponent::Temporary(self.scale_factor)),
            QuantizedComponent::BiasFactor => Some(match self.bias_factor {
                Some(temp) => DecodeComponent::Temporary(temp),
                None => DecodeComponent::Constant(crate::reference_math::ReferenceScalar::F32(0)),
            }),
            QuantizedComponent::CoefficientCodeMax => Some(DecodeComponent::Constant(
                crate::reference_math::ReferenceScalar::I32(self.coefficient_code_max),
            )),
            QuantizedComponent::Bias => Some(match self.bias {
                Some(temp) => DecodeComponent::Temporary(temp),
                None => DecodeComponent::Constant(crate::reference_math::ReferenceScalar::F32(0)),
            }),
            QuantizedComponent::GroupSize => Some(DecodeComponent::Constant(
                crate::reference_math::ReferenceScalar::I32(
                    self.group.try_into().expect("sealed group fits i32"),
                ),
            )),
            QuantizedComponent::CodeMin | QuantizedComponent::CodeMax => {
                let code = self.integer_code?;
                let codes = self
                    .steps
                    .iter()
                    .find_map(|step| match step {
                        DecodeStep::InterpretCode {
                            into,
                            bits,
                            interpretation,
                            ..
                        } if *into == code => Some(interpretation.codes(*bits)),
                        DecodeStep::InterpretGridCode { into, grid, .. } if *into == code => {
                            Some(grid.codes())
                        }
                        _ => None,
                    })
                    .expect("integer code owns its interpretation");
                let value = if role == QuantizedComponent::CodeMin {
                    codes.first()
                } else {
                    codes.last()
                }
                .copied()
                .expect("code domain nonempty");
                Some(DecodeComponent::Constant(
                    crate::reference_math::ReferenceScalar::I32(value),
                ))
            }
        }
    }

    /// Where a value that is exactly its integer code times one stored
    /// scale (no bias, factor or code exponent) reads them, when the code's
    /// offset from the least code is its raw bits XOR a constant; `None`
    /// for any other decode.
    pub fn scaled_codes(&self) -> Option<ScaledCodes> {
        let defining = |temp| self.steps.iter().find(|step| step.defines() == temp);
        let read = |temp| match defining(temp)? {
            DecodeStep::ReadPlaneField { plane, field, .. } => Some(PlaneRead {
                plane: *plane,
                field: *field,
            }),
            _ => None,
        };
        let code = self.integer_code?;
        let DecodeStep::InterpretCode {
            raw,
            bits,
            interpretation,
            ..
        } = defining(code)?
        else {
            return None;
        };
        let DecodeStep::ConvertToF32 { from: stored, .. } = defining(self.scale)? else {
            return None;
        };
        let plain = matches!(
            defining(self.code_value)?, DecodeStep::ConvertToF32 { from, .. } if *from == code
        );
        let product = matches!(
            defining(self.output)?,
            DecodeStep::Multiply { left, right, .. }
                if [*left, *right] == [self.scale, self.code_value]
                    || [*left, *right] == [self.code_value, self.scale]
        );
        if !(plain && product) {
            return None;
        }
        let codes = (0..1u32 << bits)
            .map(|raw| interpretation.decode(raw, *bits))
            .collect::<Vec<_>>();
        let least = *codes.iter().min().expect("a code has values");
        let flip = (codes[0] - least) as u32;
        if !codes
            .iter()
            .enumerate()
            .all(|(raw, code)| (code - least) as u32 == raw as u32 ^ flip)
        {
            return None;
        }
        Some(ScaledCodes {
            code: read(*raw)?,
            bits: *bits,
            flip,
            scale: read(*stored)?,
            scale_dtype: self.dtype(*stored),
        })
    }

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
        if let CodeInterpretation::Grid(grid) = interpretation {
            let plane = self.ordinal(PlaneField::Words);
            let lane = self.temp(DType::U32);
            self.steps
                .push(DecodeStep::ReadPlaneLane { into: lane, plane });
            let into = self.temp(DType::I32);
            self.steps.push(DecodeStep::InterpretGridCode {
                into,
                raw,
                lane,
                grid,
            });
            return into;
        }
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

    fn constant(&mut self, value: crate::reference_math::ReferenceScalar) -> DecodeTemp {
        let into = self.temp(value.dtype());
        self.steps.push(DecodeStep::Constant { into, value });
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

    /// `factor_plane[value / factor_group] * decode(coefficients[field]) *
    /// 2^exponent` in `f32`: one hierarchical coefficient before its sign.
    /// The power-of-two scaling is exact.
    fn hierarchical_coefficient(
        &mut self,
        factor_plane: PlaneField,
        exponent: i32,
        field: u32,
        bits: u32,
        interpretation: CodeInterpretation,
    ) -> (DecodeTemp, DecodeTemp, DecodeTemp) {
        let raw = self.read(PlaneField::Coefficients, field);
        let code = self.interpret(raw, bits, interpretation);
        let converted = self.to_f32(code);
        let factor = self.read(factor_plane, 0);
        let factor = self.to_f32(factor);
        let mut product = self.multiply(factor, converted);
        if exponent != 0 {
            let power = self.constant(crate::reference_math::ReferenceScalar::F32(
                2.0f32.powi(exponent).to_bits(),
            ));
            product = self.multiply(product, power);
        }
        (product, factor, code)
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
            factor_exponent: 0,
            bits: 6,
            interpretation: CodeInterpretation::Unsigned,
            minimum: Some(MinimumSign::Subtract),
            carried: false,
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
            factor_exponent: 0,
            bits: 6,
            interpretation: CodeInterpretation::Unsigned,
            minimum: Some(MinimumSign::Subtract),
            carried: false,
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
            factor_exponent: 0,
            bits: 8,
            interpretation: CodeInterpretation::TwosComplement,
            minimum: None,
            carried: false,
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
        code: CodeInterpretation::Table(&tables::IQ4NL),
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
            packet_group: 16,
        },
        // Physical extraction is unsigned; interpretation is the explicit
        // floating-code decode below.
        code: CodeInterpretation::Unsigned,
        float_code: Some(FloatCodeFormat::E2M1),
    },
    // The 4-bit coded families (plan: c4g{16,32}) are a codebook, a group
    // and a scale kind; `iq4g32` above is the IQ4 codebook with f32 scales.
    // GGUF Q4_0: codes offset by 8, one f16 scale per 32.
    Repr {
        name: "q4g32s",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 4,
        coefficients: Coefficients::Direct {
            dtype: DType::F16,
            bias: false,
            packet_group: 32,
        },
        code: CodeInterpretation::Offset(8),
        float_code: None,
    },
    // A GGUF IQ4_NL block: the IQ4 codebook, one f16 scale per 32, packets
    // of 32 values (`iq4g32` above keeps eight blocks per packet with their
    // scales widened).
    Repr {
        name: "iq4g32s",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 4,
        coefficients: Coefficients::Direct {
            dtype: DType::F16,
            bias: false,
            packet_group: 32,
        },
        code: CodeInterpretation::Table(&tables::IQ4NL),
        float_code: None,
    },
    // GGUF MXFP4 (OCP MX): E2M1 codes, one E8M0 power-of-two scale per 32.
    Repr {
        name: "mxfp4g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 4,
        coefficients: Coefficients::BlockFloat {
            format: FloatCodeFormat::E8M0,
            packet_group: 32,
        },
        code: CodeInterpretation::Unsigned,
        float_code: Some(FloatCodeFormat::E2M1),
    },
    // GGUF NVFP4: E2M1 codes, one UE4M3 scale per 16, packets of 64 values
    // (the GGUF block).
    Repr {
        name: "nvfp4g16",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        bits: 4,
        coefficients: Coefficients::BlockFloat {
            format: FloatCodeFormat::UE4M3,
            packet_group: 64,
        },
        code: CodeInterpretation::Unsigned,
        float_code: Some(FloatCodeFormat::E2M1),
    },
    // GGUF Q5_0: 5-bit codes offset by 16, one f16 scale per 32.
    Repr {
        name: "q5g32s",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 5,
        coefficients: Coefficients::Direct {
            dtype: DType::F16,
            bias: false,
            packet_group: 32,
        },
        code: CodeInterpretation::Offset(16),
        float_code: None,
    },
    // GGUF Q5_1: unsigned 5-bit codes, one f16 scale and one f16 min per 32.
    Repr {
        name: "q5g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: 5,
        coefficients: Coefficients::Direct {
            dtype: DType::F16,
            bias: true,
            packet_group: 32,
        },
        code: CodeInterpretation::Unsigned,
        float_code: None,
    },
    // GGUF IQ2_XS: vector codes of eight values (a 9-bit grid index and 7
    // sign bits, ggml's `qs` word), an odd 4-bit scale `2 s + 1` per 16 and
    // one f32 factor `d / 8` per 256: ggml's `d (0.5 + s) / 4` exactly.
    Repr {
        name: "iq2xs",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        bits: IQ2XS_CODES.entry_bits(),
        code: CodeInterpretation::Grid(&IQ2XS_CODES),
        float_code: None,
        coefficients: Coefficients::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F32,
            factor_exponent: 0,
            bits: 4,
            interpretation: CodeInterpretation::Odd,
            minimum: None,
            carried: false,
        },
    },
    // GGUF IQ3_XXS: vector codes of eight values (two 8-bit grid indices of
    // four values and 7 sign bits, in a 24-bit entry), an odd 4-bit scale `2 s + 1` per 32 and
    // one f32 factor `d / 4` per 256: ggml's `d (0.5 + s) / 2` exactly.
    Repr {
        name: "iq3xxs",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: IQ3XXS_CODES.entry_bits(),
        code: CodeInterpretation::Grid(&IQ3XXS_CODES),
        float_code: None,
        coefficients: Coefficients::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F32,
            factor_exponent: 0,
            bits: 4,
            interpretation: CodeInterpretation::Odd,
            minimum: None,
            carried: false,
        },
    },
    // GGUF IQ2_XXS: vector codes of eight values (an 8-bit grid index and 7
    // sign bits, in a 16-bit entry), an odd 4-bit scale `2 s + 1` per 32 and
    // one f16 factor `d` per 256 scaled by 2^-3: ggml's `d (0.5 + s) / 4`
    // exactly. Row layouts carry the scale codes in the entries' sixteenth
    // bits, 66 bytes per 256 values as in GGUF.
    Repr {
        name: "iq2xxs",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        bits: IQ2XXS_CODES.entry_bits(),
        code: CodeInterpretation::Grid(&IQ2XXS_CODES),
        float_code: None,
        coefficients: Coefficients::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: -3,
            bits: 4,
            interpretation: CodeInterpretation::Odd,
            minimum: None,
            carried: true,
        },
    },
    // GGUF Q2_K: unsigned 2-bit codes, a 4-bit scale and a 4-bit minimum
    // per 16 under one f16 factor `d` and one f16 minimum factor `dmin` per
    // 256: ggml's `d s q - dmin m` exactly.
    Repr {
        name: "q2k",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        bits: 2,
        code: CodeInterpretation::Unsigned,
        float_code: None,
        coefficients: Coefficients::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: 0,
            bits: 4,
            interpretation: CodeInterpretation::Unsigned,
            minimum: Some(MinimumSign::Subtract),
            carried: false,
        },
    },
];

pub(crate) static IQ2XS_CODES: GridCodes = GridCodes {
    entry_bits: 16,
    grid: &tables::IQ2XS,
};

pub(crate) static IQ3XXS_CODES: GridCodes = GridCodes {
    entry_bits: 24,
    grid: &tables::IQ3XXS,
};

pub(crate) static IQ2XXS_CODES: GridCodes = GridCodes {
    entry_bits: 16,
    grid: &tables::IQ2XXS,
};

pub(crate) fn lookup(name: &str) -> Option<&'static Repr> {
    REPRS.iter().find(|r| r.name == name)
}

impl CodeInterpretation {
    pub(crate) fn decode(&self, raw: u32, bits: u32) -> i32 {
        let recipe = crate::reference_math::code_recipe(self, bits);
        let value = crate::reference_math::evaluate(
            &recipe,
            &[crate::reference_math::ReferenceScalar::U32(raw)],
        )
        .expect("registry code interpretation is total");
        let crate::reference_math::ReferenceScalar::I32(value) = value else {
            unreachable!()
        };
        value
    }
}

#[cfg(test)]
impl Repr {
    pub(crate) fn has_bias(&self) -> bool {
        match self.coefficients {
            Coefficients::Direct { bias, .. } => bias,
            Coefficients::Hierarchical { minimum, .. } => minimum.is_some(),
            Coefficients::BlockFloat { .. } => false,
        }
    }
}

/// How a matrix product over a weight element takes its activation operands
/// (docs/precision.md, "NVFP4 operands"). It is the weight's, as its
/// producer calibrated it: NVFP4 weights are W4A16 or W4A4 (NVIDIA's
/// `W4A16_NVFP4` and `NVFP4`), one storage with two elements, so every
/// backend computes the element's one recipe and FP4 tensor operations
/// compute W4A4 natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperandRecipe {
    /// The activation element as it is rounded to `A`.
    Rounded,
    /// Each row in 16-value blocks of E2M1 codes with UE4M3 block scales
    /// under the row's power-of-two global scale.
    Nvfp4Rows,
}

impl OperandRecipe {
    /// Every recipe, in order of its code.
    pub const ALL: [Self; 2] = [Self::Rounded, Self::Nvfp4Rows];

    /// The recipe as native declarations and sources read it
    /// (`operands(W)`): 0 rounded, 1 NVFP4 rows.
    pub const fn code(self) -> u64 {
        match self {
            Self::Rounded => 0,
            Self::Nvfp4Rows => 1,
        }
    }

    /// What names an element of this recipe after its storage: nothing for
    /// the rounded activation, `a4` for NVFP4 rows (`nvfp4g16a4`).
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Rounded => "",
            Self::Nvfp4Rows => "a4",
        }
    }
}

/// How a decoder reads a packed representation's values, the axes native
/// libraries compose their decoders from: code planes of `low` and `high`
/// bits (a grid code's entry is its one low plane), read through `code`,
/// their values scaled by `2^code_exponent` (a floating value code is its
/// codebook, `FloatCodeFormat::codebook`), and the `coefficients` that scale
/// them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeAxes {
    pub low: u32,
    pub high: u32,
    pub code: CodeInterpretation,
    pub code_exponent: i32,
    pub coefficients: CoefficientAxes,
}

/// A decoder's coefficients: a stored scale (and bias) per `group` values,
/// or per `group` local codes (`LocalCodes`) under a factor (and minimum
/// factor) per `factor_group` scaled by `2^exponent`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoefficientAxes {
    Direct {
        scale: ScaleEncoding,
        bias: bool,
        group: u32,
    },
    Hierarchical {
        factor: ScaleEncoding,
        factor_group: u32,
        exponent: i32,
        group: u32,
        locals: LocalCodes,
    },
}

/// A hierarchical decoder's local codes per group: a `bits`-bit scale code
/// read through `interpretation`, either in a scales plane (then with a
/// minimum code, its term added or subtracted by `minimum`) or carried by the
/// group's grid entries, bit `j` of the code at bit `bit` of entry `j`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalCodes {
    Plane {
        bits: u32,
        interpretation: CodeInterpretation,
        minimum: Option<MinimumSign>,
    },
    Carried {
        bit: u32,
        bits: u32,
        interpretation: CodeInterpretation,
    },
}

/// How a hierarchical decoder's minimum term enters a value: added (`+1`) or
/// subtracted (`-1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MinimumSign {
    Add,
    Subtract,
}

impl MinimumSign {
    pub const fn value(self) -> i32 {
        match self {
            Self::Add => 1,
            Self::Subtract => -1,
        }
    }
}

/// A decoder's coefficients without their parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoefficientKind {
    Direct,
    Hierarchical,
}

impl CoefficientKind {
    pub const ALL: [Self; 2] = [Self::Direct, Self::Hierarchical];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Direct => "DIRECT",
            Self::Hierarchical => "HIER",
        }
    }
}

impl CoefficientAxes {
    pub const fn kind(&self) -> CoefficientKind {
        match self {
            Self::Direct { .. } => CoefficientKind::Direct,
            Self::Hierarchical { .. } => CoefficientKind::Hierarchical,
        }
    }
}

/// The encoding of a stored scale, bias or factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScaleEncoding {
    F16,
    BF16,
    F32,
    E8M0,
    UE4M3,
}

impl ScaleEncoding {
    pub const ALL: [Self; 5] = [Self::F16, Self::BF16, Self::F32, Self::E8M0, Self::UE4M3];

    fn of_dtype(dtype: DType) -> Self {
        match dtype {
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            DType::F32 => Self::F32,
            other => panic!("a stored scale is floating, not {other:?}"),
        }
    }

    fn of_code(format: FloatCodeFormat) -> Self {
        match format {
            FloatCodeFormat::E8M0 => Self::E8M0,
            FloatCodeFormat::UE4M3 => Self::UE4M3,
            other => panic!("{other:?} is not a scale code"),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
            Self::E8M0 => "E8M0",
            Self::UE4M3 => "UE4M3",
        }
    }
}

impl Repr {
    pub(crate) fn decode_axes(&self) -> DecodeAxes {
        let (code, code_exponent) = match self.float_code {
            Some(format) => {
                let (codebook, exponent) = format
                    .codebook()
                    .expect("a value floating code has a codebook");
                (CodeInterpretation::Table(codebook), exponent)
            }
            None => (self.code.clone(), 0),
        };
        let (low, high) = match code {
            CodeInterpretation::Grid(_) => (self.bits, 0),
            _ => crate::registry::code_planes(self.bits).expect("a packed code width"),
        };
        let coefficients = match &self.coefficients {
            Coefficients::Direct { dtype, bias, .. } => CoefficientAxes::Direct {
                scale: ScaleEncoding::of_dtype(*dtype),
                bias: *bias,
                group: self.group,
            },
            Coefficients::BlockFloat { format, .. } => CoefficientAxes::Direct {
                scale: ScaleEncoding::of_code(*format),
                bias: false,
                group: self.group,
            },
            Coefficients::Hierarchical {
                factor_group,
                factor_dtype,
                factor_exponent,
                bits,
                interpretation,
                minimum,
                carried,
            } => CoefficientAxes::Hierarchical {
                factor: ScaleEncoding::of_dtype(*factor_dtype),
                factor_group: *factor_group,
                exponent: *factor_exponent,
                group: self.group,
                locals: match (carried, &code) {
                    (false, _) => LocalCodes::Plane {
                        bits: *bits,
                        interpretation: interpretation.clone(),
                        minimum: *minimum,
                    },
                    (true, CodeInterpretation::Grid(grid)) if minimum.is_none() => LocalCodes::Carried {
                        bit: grid.carried_bit(),
                        bits: *bits,
                        interpretation: interpretation.clone(),
                    },
                    (true, _) => {
                        panic!("only grid entries carry coefficient codes, with no minimum")
                    }
                },
            },
        };
        DecodeAxes {
            low,
            high,
            code,
            code_exponent,
            coefficients,
        }
    }

    /// The operand recipes of this representation's elements: every
    /// representation multiplies the rounded activation, and NVFP4 (E2M1
    /// codes in 16-value groups under UE4M3 scales) also has its W4A4
    /// element, whose activation side is the same format.
    pub(crate) fn operand_recipes(&self) -> &'static [OperandRecipe] {
        match (self.float_code, &self.coefficients, self.group) {
            (
                Some(FloatCodeFormat::E2M1),
                Coefficients::BlockFloat {
                    format: FloatCodeFormat::UE4M3,
                    ..
                },
                16,
            ) => &[OperandRecipe::Rounded, OperandRecipe::Nvfp4Rows],
            _ => &[OperandRecipe::Rounded],
        }
    }

    pub(crate) fn storage_group(&self) -> u32 {
        match self.coefficients {
            Coefficients::Direct { packet_group, .. } => packet_group,
            Coefficients::Hierarchical { factor_group, .. } => factor_group,
            Coefficients::BlockFloat { packet_group, .. } => packet_group,
        }
    }
    /// Ordered physical ABI planes with their typed field identities.
    fn plane_table(&self) -> Vec<(PlaneField, Plane)> {
        let mut result = vec![(
            PlaneField::Words,
            Plane {
                name: "words",
                // A vector code's entry is shared by its eight values.
                group: match self.code {
                    CodeInterpretation::Grid(_) => GRID_VALUES,
                    _ => 1,
                },
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
                minimum,
                ..
            } => {
                result.push((
                    PlaneField::Coefficients,
                    Plane {
                        name: "coefficients",
                        group: self.group,
                        fields: if minimum.is_some() { 2 } else { 1 },
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
                if minimum.is_some() {
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
            Coefficients::BlockFloat { format, .. } => result.push((
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
    /// coefficient is `factor * coefficient_code * 2^factor_exponent` (with
    /// the bias sign applied), exactly as the reference interpreter
    /// evaluates it.
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
        let (code_value, integer_code) = match self.float_code {
            Some(format) => (recipe.float_code(raw, format), None),
            None => {
                let code = recipe.interpret(raw, self.bits, self.code.clone());
                (recipe.to_f32(code), Some(code))
            }
        };
        let mut scale_code = None;
        let mut bias_code = None;
        let (scale, bias, scale_factor, bias_factor, coefficient_code_max) =
            match &self.coefficients {
                Coefficients::Direct { bias, .. } => {
                    let scale = recipe.read(PlaneField::Scale, 0);
                    let scale = recipe.to_f32(scale);
                    let bias = if *bias {
                        let bias = recipe.read(PlaneField::Bias, 0);
                        Some(recipe.to_f32(bias))
                    } else {
                        None
                    };
                    (scale, bias, scale, bias, 1)
                }
                Coefficients::Hierarchical {
                    factor_exponent,
                    bits,
                    interpretation,
                    minimum,
                    ..
                } => {
                    let (scale, scale_factor, code) = recipe.hierarchical_coefficient(
                        PlaneField::ScaleFactor,
                        *factor_exponent,
                        0,
                        *bits,
                        interpretation.clone(),
                    );
                    scale_code = Some(code);
                    let mut bias_factor = None;
                    let bias = if let Some(sign) = minimum {
                        let (product, factor, code) = recipe.hierarchical_coefficient(
                            PlaneField::BiasFactor,
                            *factor_exponent,
                            1,
                            *bits,
                            interpretation.clone(),
                        );
                        bias_factor = Some(factor);
                        bias_code = Some(code);
                        Some(match sign {
                            MinimumSign::Subtract => recipe.negate(product),
                            MinimumSign::Add => product,
                        })
                    } else {
                        None
                    };
                    let maximum = (0..(1u32 << bits))
                        .map(|raw| interpretation.decode(raw, *bits))
                        .max()
                        .expect("sealed coefficient domain nonempty");
                    (scale, bias, scale_factor, bias_factor, maximum)
                }
                Coefficients::BlockFloat { format, .. } => {
                    let raw = recipe.read(PlaneField::BlockScale, 0);
                    let scale = recipe.float_code(raw, *format);
                    (scale, None, scale, None, 1)
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
            integer_code,
            code_value,
            scale_code,
            bias_code,
            scale,
            bias,
            group: self.group,
            scale_factor,
            bias_factor,
            coefficient_code_max,
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
            Coefficients::Hierarchical {
                minimum,
                factor_exponent,
                ..
            } => Some(Coefficient::Product {
                factor: find(if bias {
                    PlaneField::BiasFactor
                } else {
                    PlaneField::ScaleFactor
                })?,
                exponent: factor_exponent,
                coefficients: find(PlaneField::Coefficients)?,
                field: u32::from(bias),
                sign: if bias { minimum.map_or(1, MinimumSign::value) } else { 1 },
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
                DecodeStep::ReadPlaneLane { plane, .. } => {
                    (value % u64::from(recipe.planes()[*plane as usize].group)) as f64
                }
                DecodeStep::InterpretGridCode {
                    raw, lane, grid, ..
                } => f64::from(grid.code(
                    get(&temporaries, *raw) as u32,
                    get(&temporaries, *lane) as u32,
                )),
                DecodeStep::DecodeFloatCode { raw, format, .. } => {
                    f64::from(format.decode(get(&temporaries, *raw) as u32))
                }
                DecodeStep::ConvertToF32 { from, .. } => get(&temporaries, *from) as f32 as f64,
                DecodeStep::Constant { value, .. } => match value {
                    crate::reference_math::ReferenceScalar::F32(bits) => {
                        f64::from(f32::from_bits(*bits))
                    }
                    other => panic!("the test evaluator has no {other:?} constant"),
                },
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
                            exponent,
                            coefficients,
                            field,
                            sign,
                        }) => {
                            let code = plane_value(
                                &coefficients,
                                flat / coefficients.group as usize * coefficients.fields as usize
                                    + field as usize,
                            );
                            let factor = plane_value(&factor, flat / factor.group as usize);
                            (factor * code * 2.0f32.powi(exponent)) * sign as f32
                        }
                    }
                };
                let code = plane_entry(0, flat as u64 / u64::from(planes[0].group)) as u32;
                let decoded_code = match (repr.float_code, &repr.code) {
                    (Some(format), _) => format.decode(code),
                    (None, CodeInterpretation::Grid(grid)) => {
                        grid.code(code, flat as u32 % GRID_VALUES) as f32
                    }
                    (None, _) => repr.decode_code(code) as f32,
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
                        DecodeStep::ReadPlaneLane { into, plane } => {
                            assert!((*plane as usize) < recipe.planes().len());
                            assert_eq!(recipe.dtype(*into), DType::U32);
                        }
                        DecodeStep::InterpretGridCode {
                            into, raw, lane, ..
                        } => {
                            assert_eq!(recipe.dtype(*raw), DType::U32);
                            assert_eq!(recipe.dtype(*lane), DType::U32);
                            assert_eq!(recipe.dtype(*into), DType::I32);
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
                                    | FloatCodeFormat::E8M0
                            ));
                        }
                        DecodeStep::ConvertToF32 { into, .. } => {
                            assert_eq!(recipe.dtype(*into), DType::F32);
                        }
                        DecodeStep::Constant { into, value } => {
                            assert_eq!(recipe.dtype(*into), value.dtype());
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
    fn e8m0_codes_are_exact_powers_of_two() {
        for raw in 1u32..255 {
            assert_eq!(
                FloatCodeFormat::E8M0.decode(raw),
                2.0f32.powi(raw as i32 - 127),
                "{raw}"
            );
        }
        // 2^-127 is an F32 subnormal; 0xff is NaN.
        assert_eq!(FloatCodeFormat::E8M0.decode(0).to_bits(), 0x0040_0000);
        assert!(FloatCodeFormat::E8M0.decode(0xff).is_nan());
    }

    #[test]
    fn hierarchical_recipes_apply_the_minimum_sign_and_direct_recipes_do_not_negate() {
        for repr in REPRS {
            let recipe = repr.decode_recipe_to(DType::F32);
            let negations = recipe
                .steps
                .iter()
                .filter(|step| matches!(step, DecodeStep::Negate { .. }))
                .count();
            let expected = match repr.coefficients {
                Coefficients::Hierarchical {
                    minimum: Some(MinimumSign::Subtract),
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
        for (name, expected) in [("q2k", 84), ("q4k", 144), ("q5k", 176), ("q6k", 210)] {
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
