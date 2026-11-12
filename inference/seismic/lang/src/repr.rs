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
    pub(crate) coding: Coding,
}

/// How a representation stores its codes in its words plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CodeStorage {
    /// `bits`-bit fields, each read through `interpretation`.
    Fields {
        bits: u32,
        interpretation: CodeInterpretation,
    },
    /// Floating codes of a closed format, decoded through its codebook.
    Float(FloatCodeFormat),
    /// Grid entries, each coding eight values (`GRID_VALUES`).
    Grid(&'static GridCodes),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PackingAxisRule {
    Last,
}

/// A representation's codes and the coefficients that scale them, in the
/// pairings it can store: codes under direct scales; or under hierarchical
/// local codes, which sit with the codes their placement admits
/// (`LocalCodes`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Coding {
    Direct {
        codes: CodeStorage,
        scales: Scales,
    },
    /// A `bits`-bit local code per group, read through `interpretation` (a
    /// signed bias's sign bit apart: `LocalCodes::Carried`), under a factor per
    /// `factor_group`: the coefficient is the factor times the code times
    /// `2^factor_exponent` (an exact power-of-two scaling).
    Hierarchical {
        factor_group: u32,
        factor_dtype: DType,
        factor_exponent: i32,
        bits: u32,
        interpretation: CodeInterpretation,
        locals: LocalCodes<CodeStorage>,
    },
}

/// Direct coefficients: a stored scale (and bias) per group, or one
/// floating scale code per group. `packet_group` is the logical values one
/// physical packet carries (several independently scaled groups).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Scales {
    Stored {
        dtype: DType,
        bias: bool,
        packet_group: u32,
    },
    BlockFloat {
        format: FloatCodeFormat,
        packet_group: u32,
    },
}

impl Coding {
    /// What the words plane holds (`LocalCodes::codes` under hierarchical
    /// coefficients).
    pub(crate) fn codes(&self) -> CodeStorage {
        match self {
            Self::Direct { codes, .. } => codes.clone(),
            Self::Hierarchical { locals, .. } => locals.codes(),
        }
    }
}

impl From<&'static GridCodes> for CodeStorage {
    fn from(grid: &'static GridCodes) -> Self {
        Self::Grid(grid)
    }
}

/// A hierarchical group's bias term: a minimum code after its scale code (a
/// second field) under the minimum factor, its term added or subtracted; or
/// `±2^exponent` times the group's scale, signed by its local code's own
/// top bit (ggml's IQ1_S `±IQ1S_DELTA`: `+` when clear), the code's low
/// bits the scale's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LocalBias {
    Minimum(MinimumSign),
    Signed { exponent: i32 },
}

impl LocalBias {
    pub const fn kind(self) -> BiasKind {
        match self {
            Self::Minimum(_) => BiasKind::Minimum,
            Self::Signed { .. } => BiasKind::Signed,
        }
    }
}

/// A coefficient group's bias term without its parameters: a minimum,
/// stored per group (direct coefficients' bias) or a code under a minimum
/// factor (`LocalBias::Minimum`), or a signed bias (`LocalBias::Signed`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BiasKind {
    Minimum,
    Signed,
}

impl BiasKind {
    pub const ALL: [Self; 2] = [Self::Minimum, Self::Signed];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Minimum => "MINIMUM",
            Self::Signed => "SIGNED",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaneEncoding {
    Dense(DType),
    Packed {
        bits: u32,
        interpretation: CodeInterpretation,
    },
    /// Grid entries of `entry_bits()` each (`GridCodes`), one entry several
    /// values' codes.
    Grid(&'static GridCodes),
    /// Local codes of a signed bias (`LocalBias::Signed`): the low
    /// `scale_bits` bits a group's scale code read through `interpretation`,
    /// the bit above them the sign of its bias. An entry's value is its scale
    /// code, negated where the sign is set: the bias's code.
    SignedLocal {
        scale_bits: u32,
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
    /// `sign * factor * 2^exponent * code`, the code read from field
    /// `field` of the `coefficients` plane as `bits` bits: an entry's value
    /// as its plane stores it where `bits` is its width, else its low `bits`
    /// bits read through the plane's interpretation (a scale code under a
    /// signed bias, `scale_bits`).
    Product {
        factor: Plane,
        exponent: i32,
        coefficients: Plane,
        field: u32,
        bits: u32,
        sign: i32,
    },
}
impl PlaneEncoding {
    /// Bits of one stored entry.
    pub(crate) fn entry_bits(&self) -> u32 {
        match *self {
            PlaneEncoding::Dense(dtype) => dtype.bytes() * 8,
            PlaneEncoding::Packed { bits, .. } => bits,
            PlaneEncoding::SignedLocal { scale_bits, .. } => scale_bits + 1,
            PlaneEncoding::Grid(grid) => grid.entry_bits(),
            PlaneEncoding::FloatCode { format } => format.bits(),
        }
    }

    /// The bit of an entry ([`Self::entry_bits`]) whose clearing leaves any
    /// entry a finite number, in a plane of floating-point numbers: its
    /// exponent's highest. None for a plane whose every entry is finite.
    pub(crate) fn finite_bit(&self) -> Option<u32> {
        match *self {
            PlaneEncoding::Dense(DType::F16 | DType::BF16) => Some(14),
            PlaneEncoding::Dense(DType::F32) => Some(30),
            PlaneEncoding::FloatCode { format: FloatCodeFormat::E4M3 | FloatCodeFormat::UE4M3 } => Some(6),
            PlaneEncoding::FloatCode { format: FloatCodeFormat::E8M0 } => Some(7),
            PlaneEncoding::Dense(_)
            | PlaneEncoding::FloatCode { format: FloatCodeFormat::E2M1 }
            | PlaneEncoding::Packed { .. }
            | PlaneEncoding::SignedLocal { .. }
            | PlaneEncoding::Grid(_) => None,
        }
    }
}
impl Plane {
    pub(crate) fn dtype(&self) -> DType {
        match self.encoding {
            PlaneEncoding::Dense(dtype) => dtype,
            PlaneEncoding::Packed { .. }
            | PlaneEncoding::Grid(_)
            | PlaneEncoding::SignedLocal { .. } => DType::U32,
            PlaneEncoding::FloatCode { .. } => DType::U32,
        }
    }
    pub(crate) fn entry_bits(&self) -> u32 {
        self.encoding.entry_bits()
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
            PlaneEncoding::Packed { .. }
            | PlaneEncoding::Grid(_)
            | PlaneEncoding::SignedLocal { .. } => entries
                .checked_mul(u64::from(self.entry_bits()))
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
    /// The odd code `2 (c - k) + 1` of `Odd(k)` (`Odd(0)`: ggml's IQ
    /// sub-block scales `0.5 + c`, doubled; `Odd(1)`: Q1_0's `2 c - 1`).
    Odd(i32),
    /// Code `c` names `values[c]` of a ggml codebook.
    Table(&'static tables::Codebook),
}

/// How a decoder's value codes read, without their parameters: a code
/// interpretation's kind (`CodeInterpretation::kind`), grid entries
/// (`ValueCodes::Grid`) or E4M3 bytes (`ValueCodes::E4m3`). The native
/// `CODES` axis takes its values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodeKind {
    Unsigned,
    TwosComplement,
    Offset,
    Odd,
    Table,
    Grid,
    E4m3,
}

impl CodeKind {
    pub const ALL: [Self; 7] = [
        Self::Unsigned,
        Self::TwosComplement,
        Self::Offset,
        Self::Odd,
        Self::Table,
        Self::Grid,
        Self::E4m3,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Unsigned => "UNSIGNED",
            Self::TwosComplement => "TWOS",
            Self::Offset => "OFFSET",
            Self::Odd => "ODD",
            Self::Table => "TABLE",
            Self::Grid => "GRID",
            Self::E4m3 => "E4M3",
        }
    }
}

impl CodeInterpretation {
    pub const fn kind(&self) -> CodeKind {
        match self {
            Self::Unsigned => CodeKind::Unsigned,
            Self::TwosComplement => CodeKind::TwosComplement,
            Self::Offset(_) => CodeKind::Offset,
            Self::Odd(_) => CodeKind::Odd,
            Self::Table(_) => CodeKind::Table,
        }
    }
}

/// Vector codes (the GGUF IQ grids): one entry codes `GRID_VALUES`
/// consecutive values. It holds `GRID_VALUES / lanes` grid indices of
/// `index_bits` each, first index lowest, then the sign bits of its `signs`,
/// then, with a `delta`, one delta bit. The value at `lane` reads lane
/// `lane % lanes` of point `index lane / lanes`: a magnitude, negated when
/// its sign is set, or a signed value where the entry has no signs. With
/// `delta = Some(e)` the code is `2^e v + 1`, or `2^e v - 1` when the delta
/// bit is set (ggml's IQ1_M `±IQ1M_DELTA`, scaled by `2^e`). An entry is
/// `entry_bits` wide; bits past the delta are zero, or carry coefficient
/// codes (`carried_bit`). A row layout keeps an entry's whole bytes in its
/// `grids` plane and the rest in its `grid_high` plane.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct GridCodes {
    pub entry_bits: u32,
    /// The ggml grid: byte `j` of point `i` is its lane `j`.
    pub grid: &'static tables::Grid,
    pub signs: GridSigns,
    pub delta: Option<u32>,
}

/// How a grid entry signs its eight values: sign `j` negates lane `j`, whose
/// point bytes are magnitudes; or no sign bits, the point bytes being two's
/// complement values (ggml's `iq1s_grid`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GridSigns {
    /// Seven sign bits and an eighth that makes the count of set signs even
    /// (ggml's `ksigns_iq2xs`).
    ParityPlus7,
    /// Eight sign bits (ggml's IQ2_S sign bytes).
    Explicit8,
    None,
}

impl GridSigns {
    pub const ALL: [Self; 3] = [Self::ParityPlus7, Self::Explicit8, Self::None];

    pub const fn name(self) -> &'static str {
        match self {
            Self::ParityPlus7 => "PARITY_PLUS7",
            Self::Explicit8 => "EXPLICIT8",
            Self::None => "NONE",
        }
    }

    /// Sign bits an entry holds.
    pub const fn bits(self) -> u32 {
        match self {
            Self::ParityPlus7 => 7,
            Self::Explicit8 => 8,
            Self::None => 0,
        }
    }

    /// The eight lane signs of an entry's `bits()` sign bits `signs`.
    pub const fn lanes(self, signs: u32) -> u32 {
        match self {
            Self::ParityPlus7 => signs | ((signs.count_ones() & 1) << 7),
            Self::Explicit8 => signs,
            Self::None => 0,
        }
    }
}

/// A constant table a kernel's code looks up: a conversion recipe's
/// registered table (`RepackExpr::Lookup`), or a grid's point values
/// ([`GridCodes::point_values`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LookupTable {
    Repack(RepackTable),
    Grid(&'static GridCodes),
}

impl LookupTable {
    pub fn values(self) -> &'static [i32] {
        match self {
            Self::Repack(table) => table.values(),
            Self::Grid(grid) => grid.point_values(),
        }
    }
}

/// Values one grid entry codes.
pub const GRID_VALUES: u32 = 8;

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

    /// The first entry bit past its indices: its signs.
    pub const fn sign_bit(&self) -> u32 {
        self.indices() * self.index_bits()
    }

    /// The entry bit past its signs: its delta, where it has one.
    pub const fn delta_bit(&self) -> u32 {
        self.sign_bit() + self.signs.bits()
    }

    /// Bits of one entry.
    pub const fn entry_bits(&self) -> u32 {
        assert!(
            self.entry_bits >= self.carried_bit(),
            "a grid entry holds its indices, signs and delta"
        );
        self.entry_bits
    }

    /// The first entry bit past its indices, signs and delta. A
    /// representation whose coefficients are carried
    /// (`LocalCodes::Carried`) places, in its row layouts,
    /// bit `j` of a group's coefficient code at this bit of the group's
    /// entry `j`; the packet form keeps it zero.
    pub const fn carried_bit(&self) -> u32 {
        self.delta_bit() + if self.delta.is_some() { 1 } else { 0 }
    }

    /// The value of lane `lane` of point `index`: its byte, a magnitude, or
    /// two's complement where the entry has no signs.
    pub const fn point_value(&self, index: usize, lane: u32) -> i32 {
        let byte = self.points().byte(index, lane);
        match self.signs {
            GridSigns::None => byte as i8 as i32,
            GridSigns::ParityPlus7 | GridSigns::Explicit8 => byte as i32,
        }
    }

    /// The code of the value at `lane` (< `GRID_VALUES`) of `entry`.
    pub fn code(&self, entry: u32, lane: u32) -> i32 {
        let index_mask = (1u32 << self.index_bits()) - 1;
        let index = (entry >> (self.index_bits() * (lane / self.lanes()))) & index_mask;
        let value = self.point_value(index as usize, lane % self.lanes());
        let signs = (entry >> self.sign_bit()) & ((1 << self.signs.bits()) - 1);
        let value = if (self.signs.lanes(signs) >> lane) & 1 == 0 {
            value
        } else {
            -value
        };
        match self.delta {
            Some(exponent) if (entry >> self.delta_bit()) & 1 == 0 => (value << exponent) + 1,
            Some(exponent) => (value << exponent) - 1,
            None => value,
        }
    }

    /// The grid's point values (`point_value`) as one table: point `i`'s
    /// lane `j` at `i * lanes + j`. Built once per grid, for code generators
    /// that read a table.
    pub fn point_values(&'static self) -> &'static [i32] {
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
                        (0..self.lanes()).map(move |lane| self.point_value(point, lane))
                    })
                    .collect::<Vec<_>>()
                    .leak()
            })
    }

    /// Every code the grid gives, ascending.
    pub fn codes(&self) -> Vec<i32> {
        let signed = |value: i32| match self.signs {
            GridSigns::ParityPlus7 | GridSigns::Explicit8 => vec![value, -value],
            GridSigns::None => vec![value],
        };
        let delta = |value: i32| match self.delta {
            Some(exponent) => vec![(value << exponent) + 1, (value << exponent) - 1],
            None => vec![value],
        };
        let mut codes = (0..self.points().len())
            .flat_map(|point| (0..self.lanes()).map(move |lane| self.point_value(point, lane)))
            .flat_map(signed)
            .flat_map(delta)
            .collect::<Vec<_>>();
        codes.sort_unstable();
        codes.dedup();
        codes
    }
}

impl CodeInterpretation {
    /// Every code the interpretation gives a `bits`-wide entry, ascending.
    pub fn codes(&self, bits: u32) -> Vec<i32> {
        let mut codes = (0..1u32 << bits)
            .map(|raw| self.decode(raw, bits))
            .collect::<Vec<_>>();
        codes.sort_unstable();
        codes.dedup();
        codes
    }
}
#[cfg(test)]
impl Repr {
    /// A field's code from its raw bits.
    pub(crate) fn decode_code(&self, raw: u32) -> i32 {
        match &self.coding.codes() {
            CodeStorage::Fields {
                bits,
                interpretation,
            } => interpretation.decode(raw, *bits),
            CodeStorage::Float(_) | CodeStorage::Grid(_) => {
                panic!("`{}` stores no integer fields", self.name)
            }
        }
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

    /// The position of logical value `value` among the values its entry
    /// covers: a vector code's lane.
    pub fn lane(&self, value: u64) -> u32 {
        (value % u64::from(self.group)) as u32
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
    /// the values its entry of the plane at ordinal `plane` covers
    /// (`PlaneSchema::lane`): a vector code's lane.
    ReadPlaneLane { into: DecodeTemp, plane: u32 },
    /// `into: i32 := interpretation.decode(raw, bits)` (unsigned, two's
    /// complement, zero-point offset, or table) of the low `bits` bits of
    /// `raw` (`DecodeRecipe::read_bits`: all of a read but a signed local
    /// code's).
    InterpretCode {
        into: DecodeTemp,
        raw: DecodeTemp,
        bits: u32,
        interpretation: CodeInterpretation,
    },
    /// `into: i32 := code`, negated where bit `bit` of `raw` is set: a
    /// signed bias's code, its local code's scale code signed by its top
    /// bit (`LocalBias::Signed`).
    SignedCode {
        into: DecodeTemp,
        code: DecodeTemp,
        raw: DecodeTemp,
        bit: u32,
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
            | DecodeStep::SignedCode { into, .. }
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
            DecodeStep::SignedCode { code, raw, .. } => vec![*code, *raw],
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

    /// The bits `u32` temporary `raw` holds: a plane read's entry bits, else
    /// all 32. An `InterpretCode` of fewer bits reads the low ones.
    pub fn read_bits(&self, raw: DecodeTemp) -> u32 {
        self.assert_owns(raw);
        match self.steps.iter().find(|step| step.defines() == raw) {
            Some(DecodeStep::ReadPlaneField { plane, .. }) => {
                self.planes[*plane as usize].entry_bits
            }
            _ => u32::BITS,
        }
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

    /// The code of the value being decoded at its lane of its grid entry
    /// `raw` of the words plane.
    fn grid_code(&mut self, raw: DecodeTemp, grid: &'static GridCodes) -> DecodeTemp {
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

    fn signed_code(&mut self, code: DecodeTemp, raw: DecodeTemp, bit: u32) -> DecodeTemp {
        let into = self.temp(DType::I32);
        self.steps.push(DecodeStep::SignedCode {
            into,
            code,
            raw,
            bit,
        });
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
    /// 2^exponent` in `f32`: one hierarchical coefficient before its sign,
    /// with its factor and its code. The power-of-two scaling is exact.
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
        let product = self.scaled(factor, converted, exponent);
        (product, factor, code)
    }

    /// `factor * code * 2^exponent` in `f32`, the power-of-two scaling
    /// exact.
    fn scaled(&mut self, factor: DecodeTemp, code: DecodeTemp, exponent: i32) -> DecodeTemp {
        let mut product = self.multiply(factor, code);
        if exponent != 0 {
            let power = self.constant(crate::reference_math::ReferenceScalar::F32(
                2.0f32.powi(exponent).to_bits(),
            ));
            product = self.multiply(product, power);
        }
        product
    }
}

pub(crate) const REPRS: &[Repr] = &[
    // MLX affine 4-bit, group 64: bf16 scale and bias per group.
    Repr {
        name: "q4g64",
        packing_axis: PackingAxisRule::Last,
        group: 64,
        coding: Coding::Direct {
            codes: CodeStorage::Fields {
                bits: 4,
                interpretation: CodeInterpretation::Unsigned,
            },
            scales: Scales::Stored {
                dtype: DType::BF16,
                bias: true,
                packet_group: 64,
            },
        },
    },
    // GGUF Q4_1: unsigned 4-bit codes, one f16 scale and one f16 minimum
    // per 32: ggml's `d q + m` exactly, 20 bytes per 32 values as in GGUF.
    Repr {
        name: "q4g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Direct {
            codes: CodeStorage::Fields {
                bits: 4,
                interpretation: CodeInterpretation::Unsigned,
            },
            scales: Scales::Stored {
                dtype: DType::F16,
                bias: true,
                packet_group: 32,
            },
        },
    },
    Repr {
        name: "q4k",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: 0,
            bits: 6,
            interpretation: CodeInterpretation::Unsigned,
            locals: LocalCodes::Plane {
                codes: CodeStorage::Fields {
                    bits: 4,
                    interpretation: CodeInterpretation::Unsigned,
                },
                minimum: Some(MinimumSign::Subtract),
            },
        },
    },
    Repr {
        name: "q5k",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: 0,
            bits: 6,
            interpretation: CodeInterpretation::Unsigned,
            locals: LocalCodes::Plane {
                codes: CodeStorage::Fields {
                    bits: 5,
                    interpretation: CodeInterpretation::Unsigned,
                },
                minimum: Some(MinimumSign::Subtract),
            },
        },
    },
    Repr {
        name: "q6k",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: 0,
            bits: 8,
            interpretation: CodeInterpretation::TwosComplement,
            locals: LocalCodes::Plane {
                codes: CodeStorage::Fields {
                    bits: 6,
                    interpretation: CodeInterpretation::Offset(32),
                },
                minimum: None,
            },
        },
    },
    Repr {
        name: "q8g32s",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Direct {
            codes: CodeStorage::Fields {
                bits: 8,
                interpretation: CodeInterpretation::TwosComplement,
            },
            scales: Scales::Stored {
                dtype: DType::F16,
                bias: false,
                packet_group: 32,
            },
        },
    },
    Repr {
        name: "q8g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Direct {
            codes: CodeStorage::Fields {
                bits: 8,
                interpretation: CodeInterpretation::Unsigned,
            },
            scales: Scales::Stored {
                dtype: DType::F32,
                bias: false,
                packet_group: 32,
            },
        },
    },
    Repr {
        name: "nvfp4_e2m1_block16",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        coding: Coding::Direct {
            codes: CodeStorage::Float(FloatCodeFormat::E2M1),
            scales: Scales::BlockFloat {
                format: FloatCodeFormat::UE4M3,
                packet_group: 16,
            },
        },
    },
    // GGUF Q4_0: codes offset by 8, one f16 scale per 32.
    Repr {
        name: "q4g32s",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Direct {
            codes: CodeStorage::Fields {
                bits: 4,
                interpretation: CodeInterpretation::Offset(8),
            },
            scales: Scales::Stored {
                dtype: DType::F16,
                bias: false,
                packet_group: 32,
            },
        },
    },
    // GGUF IQ4_NL: the IQ4 codebook, one f16 scale per 32: ggml's `d v`
    // exactly, 18 bytes per 32 values as in GGUF.
    Repr {
        name: "iq4g32s",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Direct {
            codes: CodeStorage::Fields {
                bits: 4,
                interpretation: CodeInterpretation::Table(&tables::IQ4NL),
            },
            scales: Scales::Stored {
                dtype: DType::F16,
                bias: false,
                packet_group: 32,
            },
        },
    },
    // GGUF MXFP4 (OCP MX): E2M1 codes, one E8M0 power-of-two scale per 32.
    Repr {
        name: "mxfp4g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Direct {
            codes: CodeStorage::Float(FloatCodeFormat::E2M1),
            scales: Scales::BlockFloat {
                format: FloatCodeFormat::E8M0,
                packet_group: 32,
            },
        },
    },
    // OCP MXFP8: E4M3 codes, one E8M0 power-of-two scale per 32. A GGUF
    // F8_E4M3 weight imports as it under its companion grid of E8M0 block
    // scales, each tile's scale on every 32 values of its rows.
    Repr {
        name: "mxfp8g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Direct {
            codes: CodeStorage::Float(FloatCodeFormat::E4M3),
            scales: Scales::BlockFloat {
                format: FloatCodeFormat::E8M0,
                packet_group: 32,
            },
        },
    },
    // E4M3 codes under one stored F32 scale per 128: a GGUF F8_E4M3
    // weight imports as it under its companion grid of F32 tile scales (an
    // FP8 checkpoint's `weight_scale_inv`, 128 columns a tile), each tile's
    // scale on every 128 values of its rows. A value is the code times its
    // scale, one F32 product.
    Repr {
        name: "fp8g128",
        packing_axis: PackingAxisRule::Last,
        group: 128,
        coding: Coding::Direct {
            codes: CodeStorage::Float(FloatCodeFormat::E4M3),
            scales: Scales::Stored {
                dtype: DType::F32,
                bias: false,
                packet_group: 128,
            },
        },
    },
    // GGUF NVFP4: E2M1 codes, one UE4M3 scale per 16, packets of 64 values
    // (the GGUF block).
    Repr {
        name: "nvfp4g16",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        coding: Coding::Direct {
            codes: CodeStorage::Float(FloatCodeFormat::E2M1),
            scales: Scales::BlockFloat {
                format: FloatCodeFormat::UE4M3,
                packet_group: 64,
            },
        },
    },
    // GGUF Q5_0: 5-bit codes offset by 16, one f16 scale per 32.
    Repr {
        name: "q5g32s",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Direct {
            codes: CodeStorage::Fields {
                bits: 5,
                interpretation: CodeInterpretation::Offset(16),
            },
            scales: Scales::Stored {
                dtype: DType::F16,
                bias: false,
                packet_group: 32,
            },
        },
    },
    // GGUF Q5_1: unsigned 5-bit codes, one f16 scale and one f16 min per 32.
    Repr {
        name: "q5g32",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Direct {
            codes: CodeStorage::Fields {
                bits: 5,
                interpretation: CodeInterpretation::Unsigned,
            },
            scales: Scales::Stored {
                dtype: DType::F16,
                bias: true,
                packet_group: 32,
            },
        },
    },
    // GGUF IQ2_XS: vector codes of eight values (a 9-bit grid index and 7
    // sign bits, ggml's `qs` word), an odd 4-bit scale `2 s + 1` per 16 and
    // one f32 factor `d / 8` per 256: ggml's `d (0.5 + s) / 4` exactly.
    Repr {
        name: "iq2xs",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F32,
            factor_exponent: 0,
            bits: 4,
            interpretation: CodeInterpretation::Odd(0),
            locals: LocalCodes::Plane {
                codes: CodeStorage::Grid(&IQ2XS_CODES),
                minimum: None,
            },
        },
    },
    // GGUF IQ3_XXS: vector codes of eight values (two 8-bit grid indices of
    // four values and 7 sign bits, in a 24-bit entry), an odd 4-bit scale `2 s + 1` per 32 and
    // one f32 factor `d / 4` per 256: ggml's `d (0.5 + s) / 2` exactly.
    Repr {
        name: "iq3xxs",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F32,
            factor_exponent: 0,
            bits: 4,
            interpretation: CodeInterpretation::Odd(0),
            locals: LocalCodes::Plane {
                codes: CodeStorage::Grid(&IQ3XXS_CODES),
                minimum: None,
            },
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
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: -3,
            bits: 4,
            interpretation: CodeInterpretation::Odd(0),
            locals: LocalCodes::Carried {
                grid: &IQ2XXS_CODES,
                signed_bias: None,
            },
        },
    },
    // GGUF Q2_K: unsigned 2-bit codes, a 4-bit scale and a 4-bit minimum
    // per 16 values under one f16 factor `d` and one f16 minimum factor
    // `dmin` per 256: ggml's `d s q - dmin m` exactly, 84 bytes per 256
    // values as in GGUF.
    Repr {
        name: "q2k",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: 0,
            bits: 4,
            interpretation: CodeInterpretation::Unsigned,
            locals: LocalCodes::Plane {
                codes: CodeStorage::Fields {
                    bits: 2,
                    interpretation: CodeInterpretation::Unsigned,
                },
                minimum: Some(MinimumSign::Subtract),
            },
        },
    },
    // GGUF TQ2_0: ternary codes stored as 2-bit codes offset by 1, one f16
    // scale per 256: ggml's `(q - 1) d` exactly, 66 bytes per 256 values as
    // in GGUF.
    Repr {
        name: "q2g256",
        packing_axis: PackingAxisRule::Last,
        group: 256,
        coding: Coding::Direct {
            codes: CodeStorage::Fields {
                bits: 2,
                interpretation: CodeInterpretation::Offset(1),
            },
            scales: Scales::Stored {
                dtype: DType::F16,
                bias: false,
                packet_group: 256,
            },
        },
    },
    // GGUF Q2_0: 2-bit codes offset by 1 (-1, 0, +1 or +2), one f16 scale
    // per 64: ggml's `(q - 1) d` exactly, 18 bytes per 64 values as in GGUF.
    Repr {
        name: "q2g64",
        packing_axis: PackingAxisRule::Last,
        group: 64,
        coding: Coding::Direct {
            codes: CodeStorage::Fields {
                bits: 2,
                interpretation: CodeInterpretation::Offset(1),
            },
            scales: Scales::Stored {
                dtype: DType::F16,
                bias: false,
                packet_group: 64,
            },
        },
    },
    // GGUF Q1_0: one-bit codes `2 c - 1`, -1 or +1, one f16 scale per 128:
    // ggml's `±d` exactly, 18 bytes per 128 values as in GGUF.
    Repr {
        name: "q1g128",
        packing_axis: PackingAxisRule::Last,
        group: 128,
        coding: Coding::Direct {
            codes: CodeStorage::Fields {
                bits: 1,
                interpretation: CodeInterpretation::Odd(1),
            },
            scales: Scales::Stored {
                dtype: DType::F16,
                bias: false,
                packet_group: 128,
            },
        },
    },
    // GGUF IQ2_S: vector codes of eight values (a 10-bit index of the
    // 1024-point iq2s grid and 8 explicit sign bits, an 18-bit entry), an
    // odd 4-bit scale `2 s + 1` per 16 and one f16 factor `d` per 256
    // scaled by 2^-3: ggml's `d (0.5 + s) / 4` exactly, 82 bytes per 256
    // values as in GGUF.
    Repr {
        name: "iq2s",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: -3,
            bits: 4,
            interpretation: CodeInterpretation::Odd(0),
            locals: LocalCodes::Plane {
                codes: CodeStorage::Grid(&IQ2S_CODES),
                minimum: None,
            },
        },
    },
    // GGUF IQ1_S: vector codes of eight signed values -1, 0 or 1 (an 11-bit
    // index of the iq1s grid), per 32 an odd 3-bit scale `2 s + 1` whose
    // sign bit signs the bias `±2^-3` times the scale, one f16 factor `d`
    // per 256: ggml's `d (2 s + 1) (g ± 1/8)` exactly. Row layouts carry
    // each group's scale and sign in its entries' twelfth bits, 50 bytes per
    // 256 values as in GGUF.
    Repr {
        name: "iq1s",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: 0,
            bits: 4,
            interpretation: CodeInterpretation::Odd(0),
            locals: LocalCodes::Carried {
                grid: &IQ1S_CODES,
                signed_bias: Some(-3),
            },
        },
    },
    // GGUF IQ1_M: vector codes of eight values (an 11-bit index of the iq1s
    // grid and a delta bit) coding `8 g ± 1`, an odd 3-bit scale `2 s + 1`
    // per 16 and one f16 factor `d` (gathered from the block's scale
    // nibbles) per 256 scaled by 2^-3: ggml's `d (2 s + 1) (g ± 1/8)`
    // exactly, 56 bytes per 256 values as in GGUF.
    Repr {
        name: "iq1m",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: -3,
            bits: 3,
            interpretation: CodeInterpretation::Odd(0),
            locals: LocalCodes::Plane {
                codes: CodeStorage::Grid(&IQ1M_CODES),
                minimum: None,
            },
        },
    },
    // GGUF Q3_K: 3-bit codes offset by 4 (a 2-bit low part and a 1-bit high
    // part), a 6-bit scale code offset by 32 per 16 values and one f16
    // factor `d` per 256: ggml's `d (s - 32) q` exactly, 110 bytes per 256
    // values as in GGUF.
    Repr {
        name: "q3k",
        packing_axis: PackingAxisRule::Last,
        group: 16,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: 0,
            bits: 6,
            interpretation: CodeInterpretation::Offset(32),
            locals: LocalCodes::Plane {
                codes: CodeStorage::Fields {
                    bits: 3,
                    interpretation: CodeInterpretation::Offset(4),
                },
                minimum: None,
            },
        },
    },
    // GGUF IQ4_XS: the IQ4 codebook, a 6-bit scale code offset by 32 per 32
    // values and one f16 factor `d` per 256: ggml's `d (s - 32) v` exactly,
    // 136 bytes per 256 values as in GGUF.
    Repr {
        name: "iq4xs",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: 0,
            bits: 6,
            interpretation: CodeInterpretation::Offset(32),
            locals: LocalCodes::Plane {
                codes: CodeStorage::Fields {
                    bits: 4,
                    interpretation: CodeInterpretation::Table(&tables::IQ4NL),
                },
                minimum: None,
            },
        },
    },
    // GGUF IQ3_S: vector codes of eight values (two 9-bit indices of the
    // 512-point iq3s grid of four values and 8 explicit sign bits, a 26-bit
    // entry), an odd 4-bit scale `2 s + 1` per 32 and one f16 factor `d`
    // per 256: ggml's `d (1 + 2 s) g` exactly, 110 bytes per 256 values as
    // in GGUF.
    Repr {
        name: "iq3s",
        packing_axis: PackingAxisRule::Last,
        group: 32,
        coding: Coding::Hierarchical {
            factor_group: 256,
            factor_dtype: DType::F16,
            factor_exponent: 0,
            bits: 4,
            interpretation: CodeInterpretation::Odd(0),
            locals: LocalCodes::Plane {
                codes: CodeStorage::Grid(&IQ3S_CODES),
                minimum: None,
            },
        },
    },
];

pub(crate) static IQ2XS_CODES: GridCodes = GridCodes {
    entry_bits: 16,
    grid: &tables::IQ2XS,
    signs: GridSigns::ParityPlus7,
    delta: None,
};

pub(crate) static IQ3XXS_CODES: GridCodes = GridCodes {
    entry_bits: 24,
    grid: &tables::IQ3XXS,
    signs: GridSigns::ParityPlus7,
    delta: None,
};

pub(crate) static IQ2XXS_CODES: GridCodes = GridCodes {
    entry_bits: 16,
    grid: &tables::IQ2XXS,
    signs: GridSigns::ParityPlus7,
    delta: None,
};

pub(crate) static IQ2S_CODES: GridCodes = GridCodes {
    entry_bits: 18,
    grid: &tables::IQ2S,
    signs: GridSigns::Explicit8,
    delta: None,
};

/// An 11-bit index and the carried bit.
pub(crate) static IQ1S_CODES: GridCodes = GridCodes {
    entry_bits: 12,
    grid: &tables::IQ1S,
    signs: GridSigns::None,
    delta: None,
};

/// An 11-bit index and a delta bit.
pub(crate) static IQ1M_CODES: GridCodes = GridCodes {
    entry_bits: 12,
    grid: &tables::IQ1S,
    signs: GridSigns::None,
    delta: Some(3),
};

/// Two 9-bit indices and eight sign bits.
pub(crate) static IQ3S_CODES: GridCodes = GridCodes {
    entry_bits: 26,
    grid: &tables::IQ3S,
    signs: GridSigns::Explicit8,
    delta: None,
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
        match &self.coding {
            Coding::Direct { scales, .. } => matches!(scales, Scales::Stored { bias: true, .. }),
            Coding::Hierarchical { locals, .. } => locals.bias().is_some(),
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
/// bits (a grid code's entry is its whole bytes, `low`, and the rest,
/// `high`), their values scaled by
/// `2^code_exponent` (a floating value code is its codebook,
/// `FloatCodeFormat::codebook`, or an E4M3 code its exact value, under
/// exponent 0), and the `coding`: the codes with the coefficients that
/// scale them. A decoder folds `2^code_exponent` into its scales (and a
/// factor's), which every scale's values take exactly
/// (`ScaleEncoding::scales_exactly`, which `decode_axes` asserts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeAxes {
    pub low: u32,
    pub high: u32,
    pub code_exponent: i32,
    pub coding: CodingAxes,
}

/// What a decoder's code planes hold: fields read through an
/// interpretation, grid entries each coding `GRID_VALUES` values, or E4M3
/// bytes, each code its exact value (`2^E4M3_EXPONENT` times an integer for
/// every finite code, the profile's) and both NaN codes the registry's one
/// NaN (`f32::NAN`, the canonical NaN its arithmetic gives).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueCodes {
    Fields(CodeInterpretation),
    Grid(&'static GridCodes),
    E4m3,
}

impl ValueCodes {
    /// The exponent of E4M3 values' integers: each finite value is
    /// `2^E4M3_EXPONENT` times an integer (its least subnormal is 2^-9).
    pub const E4M3_EXPONENT: i32 = -9;

    /// The integer of every finite E4M3 code (`ValueCodes::E4m3`),
    /// ascending: its value times `2^-E4M3_EXPONENT`.
    fn e4m3_integers() -> Vec<i32> {
        let mut codes = (0..=u8::MAX)
            .map(|raw| FloatCodeFormat::E4M3.decode(u32::from(raw)))
            .filter(|value| !value.is_nan())
            .map(|value| (value * 2f32.powi(-Self::E4M3_EXPONENT)) as i32)
            .collect::<Vec<_>>();
        codes.sort_unstable();
        codes.dedup();
        codes
    }
}

impl From<&'static GridCodes> for ValueCodes {
    fn from(grid: &'static GridCodes) -> Self {
        Self::Grid(grid)
    }
}

impl DecodeAxes {
    /// The weight's arithmetic (`Profile`): what its values are as integers
    /// and the levels that scale them, without how they are stored.
    pub fn profile(&self) -> Profile {
        let (local, upper) = match &self.coding {
            CodingAxes::Direct {
                scale, bias, group, ..
            } => (
                Local {
                    group: *group,
                    scale: LocalScale::Stored(*scale),
                    bias: if *bias { Bias::Stored } else { Bias::None },
                },
                None,
            ),
            CodingAxes::Hierarchical {
                factor,
                factor_group,
                exponent,
                group,
                bits,
                interpretation,
                locals,
            } => {
                let codes = Domain::of(&interpretation.codes(locals.scale_bits(*bits)));
                (
                    Local {
                        group: *group,
                        scale: LocalScale::Code(codes),
                        bias: match locals.bias() {
                            None => Bias::None,
                            Some(LocalBias::Minimum(sign)) => Bias::Minimum { codes, sign },
                            Some(LocalBias::Signed { exponent }) => Bias::Signed { exponent },
                        },
                    },
                    Some(Upper {
                        group: *factor_group,
                        factor: *factor,
                        exponent: *exponent,
                    }),
                )
            }
        };
        let values = self.coding.codes();
        // The integer codes and the power of two scaling them: E4M3's
        // exact values are integers under their own exponent.
        let (codes, code_exponent) = match &values {
            ValueCodes::Fields(interpretation) => {
                (interpretation.codes(self.low + self.high), self.code_exponent)
            }
            ValueCodes::Grid(grid) => (grid.codes(), self.code_exponent),
            ValueCodes::E4m3 => (
                ValueCodes::e4m3_integers(),
                self.code_exponent + ValueCodes::E4M3_EXPONENT,
            ),
        };
        Profile {
            codes: Domain::of(&codes),
            code_exponent,
            map: match &values {
                ValueCodes::Grid(_) => ValueMap::Grid,
                ValueCodes::Fields(CodeInterpretation::Table(_)) => ValueMap::Table,
                ValueCodes::Fields(_) => ValueMap::Linear,
                ValueCodes::E4m3 => ValueMap::Float,
            },
            local,
            upper,
        }
    }
}

/// A weight's arithmetic, derived from its decode axes alone
/// (`DecodeAxes::profile`): value `2^code_exponent q` times its local
/// level's coefficients, under its upper level's factor where it has one,
/// for an integer code value `q` in `codes`. Consumer forms state what they
/// need over it; how the codes and coefficients are stored stays in the
/// decode axes, which only producers read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    pub codes: Domain,
    pub code_exponent: i32,
    pub map: ValueMap,
    pub local: Local,
    pub upper: Option<Upper>,
}

impl Profile {
    /// Whether a group's `Σ q ℓ x` is an exact integer for integer `x` up
    /// to the upper level: its local scales are integer codes, and its
    /// codes not scaled by a power of two of their own.
    pub fn integer_through_upper(&self) -> bool {
        matches!(self.local.scale, LocalScale::Code(_)) && self.code_exponent == 0
    }
}

/// The least and greatest of a set of integers (`Profile::codes`, a local
/// code's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Domain {
    pub least: i32,
    pub greatest: i32,
}

impl Domain {
    fn of(values: &[i32]) -> Self {
        let least = *values.iter().min().expect("a code takes some value");
        let greatest = *values.iter().max().expect("a code takes some value");
        Self { least, greatest }
    }
    /// Whether every value is an `i8`.
    pub fn fits_i8(self) -> bool {
        i32::from(i8::MIN) <= self.least && self.greatest <= i32::from(i8::MAX)
    }
}

/// How a code becomes its integer `q`: read linearly (unsigned, two's
/// complement, offset or odd), as an entry of a codebook, as a lane of a
/// vector grid's entry, or decoded as a floating code (`ValueCodes::E4m3`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueMap {
    Linear,
    Table,
    Grid,
    Float,
}

/// A weight's local level: a scale per `group` values, stored, or an
/// integer code under the upper level's factor, and its bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Local {
    pub group: u32,
    pub scale: LocalScale,
    pub bias: Bias,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalScale {
    Stored(ScaleEncoding),
    Code(Domain),
}

/// A local level's bias: none, a stored float per group, a minimum code (of
/// `codes`) under the upper level's minimum factor, added or subtracted, or
/// `±2^exponent` times the group's scale (`LocalBias::Signed`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bias {
    None,
    Stored,
    Minimum { codes: Domain, sign: MinimumSign },
    Signed { exponent: i32 },
}

/// A weight's upper level: a factor (and minimum factor) per `group`
/// values, scaled by `2^exponent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Upper {
    pub group: u32,
    pub factor: ScaleEncoding,
    pub exponent: i32,
}

/// A decoder's codes and the coefficients that scale them: a stored scale
/// (and bias) per `group` values; or per `group` a `bits`-bit local code
/// read through `interpretation`, placed with the codes its placement
/// admits (`LocalCodes`), under a factor (and minimum factor) per
/// `factor_group` scaled by `2^exponent`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodingAxes {
    Direct {
        codes: ValueCodes,
        scale: ScaleEncoding,
        bias: bool,
        group: u32,
    },
    Hierarchical {
        factor: ScaleEncoding,
        factor_group: u32,
        exponent: i32,
        group: u32,
        bits: u32,
        interpretation: CodeInterpretation,
        locals: LocalCodes,
    },
}

/// Where a hierarchical decoder's local codes are, with the codes and the
/// bias terms each placement admits: in a scales plane in every layout,
/// under codes of any storage, a minimum code its second field where the
/// group has a minimum (`LocalBias::Minimum`); or, in row layouts, carried
/// by the group's entries of `grid`, bit `j` of the code at bit
/// `grid.carried_bit()` of entry `j`, with no minimum, its top bit signing
/// a signed bias (`LocalBias::Signed`) of `2^signed_bias` times the group's
/// scale where it has one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalCodes<Codes = ValueCodes> {
    Plane {
        codes: Codes,
        minimum: Option<MinimumSign>,
    },
    Carried {
        grid: &'static GridCodes,
        signed_bias: Option<i32>,
    },
}

impl CodingAxes {
    /// What the code planes hold (`LocalCodes::codes` under hierarchical
    /// coefficients).
    pub fn codes(&self) -> ValueCodes {
        match self {
            Self::Direct { codes, .. } => codes.clone(),
            Self::Hierarchical { locals, .. } => locals.codes(),
        }
    }

    pub const fn kind(&self) -> CoefficientKind {
        match self {
            Self::Direct { .. } => CoefficientKind::Direct,
            Self::Hierarchical { .. } => CoefficientKind::Hierarchical,
        }
    }
}

impl<Codes: Clone + From<&'static GridCodes>> LocalCodes<Codes> {
    /// What the code planes of a hierarchical coding with these local codes
    /// hold: the codes beside a scales plane, or the grid entries carrying
    /// the local codes.
    pub fn codes(&self) -> Codes {
        match self {
            Self::Plane { codes, .. } => codes.clone(),
            Self::Carried { grid, .. } => Codes::from(grid),
        }
    }
}

impl<Codes> LocalCodes<Codes> {
    /// The group's bias term.
    pub fn bias(&self) -> Option<LocalBias> {
        match self {
            Self::Plane { minimum, .. } => minimum.map(LocalBias::Minimum),
            Self::Carried { signed_bias, .. } => {
                signed_bias.map(|exponent| LocalBias::Signed { exponent })
            }
        }
    }

    pub(crate) fn minimum(&self) -> Option<MinimumSign> {
        match self {
            Self::Plane { minimum, .. } => *minimum,
            Self::Carried { .. } => None,
        }
    }

    /// The bits of a `bits`-bit local code read as the group's scale code:
    /// all but a signed bias's sign bit, its top.
    pub fn scale_bits(&self, bits: u32) -> u32 {
        match self.bias() {
            Some(LocalBias::Signed { .. }) => bits - 1,
            None | Some(LocalBias::Minimum(_)) => bits,
        }
    }
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

    /// Whether `2^exponent` times each of the encoding's finite values is
    /// exact in F32: its least bit (`2^-24` for F16, `2^-133` for BF16,
    /// `2^-149` for F32, `2^-127` for E8M0, `2^-9` for UE4M3) stays at or
    /// above F32's, `2^-149`, and its greatest value below `2^128`.
    const fn scales_exactly(self, exponent: i32) -> bool {
        let (least, greatest) = match self {
            Self::F16 => (-24, 15),
            Self::BF16 => (-133, 127),
            Self::F32 => (-149, 127),
            Self::E8M0 => (-127, 127),
            Self::UE4M3 => (-9, 8),
        };
        least + exponent >= -149 && greatest + exponent <= 127
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
        let codes = |codes: &CodeStorage| match codes {
            CodeStorage::Fields {
                bits,
                interpretation,
            } => (
                ValueCodes::Fields(interpretation.clone()),
                0,
                crate::registry::code_planes(*bits).expect("a packed code width"),
            ),
            CodeStorage::Float(format) => {
                let planes =
                    crate::registry::code_planes(format.bits()).expect("a packed code width");
                match (format, format.codebook()) {
                    (_, Some((codebook, exponent))) => (
                        ValueCodes::Fields(CodeInterpretation::Table(codebook)),
                        exponent,
                        planes,
                    ),
                    // An E4M3 code is its exact value, so a scale times it
                    // is one product under every scale.
                    (FloatCodeFormat::E4M3, None) => (ValueCodes::E4m3, 0, planes),
                    (_, None) => panic!("{format:?} codes are scale codes, not value codes"),
                }
            }
            CodeStorage::Grid(grid) => {
                let bits = grid.entry_bits();
                (ValueCodes::Grid(grid), 0, (bits / 8 * 8, bits % 8))
            }
        };
        let (values, code_exponent, (low, high)) = codes(&self.coding.codes());
        let coding = match &self.coding {
            Coding::Direct { scales, .. } => {
                let (scale, bias) = match scales {
                    Scales::Stored { dtype, bias, .. } => (ScaleEncoding::of_dtype(*dtype), *bias),
                    Scales::BlockFloat { format, .. } => (ScaleEncoding::of_code(*format), false),
                };
                CodingAxes::Direct {
                    codes: values,
                    scale,
                    bias,
                    group: self.group,
                }
            }
            Coding::Hierarchical {
                factor_group,
                factor_dtype,
                factor_exponent,
                bits,
                interpretation,
                locals,
            } => CodingAxes::Hierarchical {
                factor: ScaleEncoding::of_dtype(*factor_dtype),
                factor_group: *factor_group,
                exponent: *factor_exponent,
                group: self.group,
                bits: *bits,
                interpretation: interpretation.clone(),
                locals: match locals {
                    LocalCodes::Plane { minimum, .. } => LocalCodes::Plane {
                        codes: values,
                        minimum: *minimum,
                    },
                    LocalCodes::Carried { grid, signed_bias } => LocalCodes::Carried {
                        grid,
                        signed_bias: *signed_bias,
                    },
                },
            },
        };
        // A decoder folds the code exponent into the scales (a factor's with
        // its own exponent, and a minimum factor's own exponent alone), each
        // of whose values every fold must scale exactly.
        let folds = match &coding {
            CodingAxes::Direct { scale, .. } => vec![(*scale, code_exponent)],
            CodingAxes::Hierarchical {
                factor, exponent, ..
            } => vec![(*factor, exponent + code_exponent), (*factor, *exponent)],
        };
        for (folded, fold) in folds {
            assert!(
                folded.scales_exactly(fold),
                "2^{fold} does not scale every {} value exactly",
                folded.name()
            );
        }
        DecodeAxes {
            low,
            high,
            code_exponent,
            coding,
        }
    }

    /// The operand recipes of this representation's elements: every
    /// representation multiplies the rounded activation, and NVFP4 (E2M1
    /// codes in 16-value groups under UE4M3 scales) also has its W4A4
    /// element, whose activation side is the same format.
    pub(crate) fn operand_recipes(&self) -> &'static [OperandRecipe] {
        match (&self.coding, self.group) {
            (
                Coding::Direct {
                    codes: CodeStorage::Float(FloatCodeFormat::E2M1),
                    scales:
                        Scales::BlockFloat {
                            format: FloatCodeFormat::UE4M3,
                            ..
                        },
                },
                16,
            ) => &[OperandRecipe::Rounded, OperandRecipe::Nvfp4Rows],
            _ => &[OperandRecipe::Rounded],
        }
    }

    pub(crate) fn storage_group(&self) -> u32 {
        match &self.coding {
            Coding::Direct {
                scales:
                    Scales::Stored { packet_group, .. } | Scales::BlockFloat { packet_group, .. },
                ..
            } => *packet_group,
            Coding::Hierarchical { factor_group, .. } => *factor_group,
        }
    }
    /// Ordered physical ABI planes with their typed field identities.
    fn plane_table(&self) -> Vec<(PlaneField, Plane)> {
        let codes = self.coding.codes();
        let mut result = vec![(
            PlaneField::Words,
            Plane {
                name: "words",
                // A vector code's entry is shared by its eight values.
                group: match &codes {
                    CodeStorage::Grid(_) => GRID_VALUES,
                    CodeStorage::Fields { .. } | CodeStorage::Float(_) => 1,
                },
                fields: 1,
                encoding: match codes {
                    CodeStorage::Float(format) => PlaneEncoding::FloatCode { format },
                    CodeStorage::Fields {
                        bits,
                        interpretation,
                    } => PlaneEncoding::Packed {
                        bits,
                        interpretation,
                    },
                    CodeStorage::Grid(grid) => PlaneEncoding::Grid(grid),
                },
            },
        )];
        match &self.coding {
            Coding::Direct {
                scales: Scales::Stored { dtype, bias, .. },
                ..
            } => {
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
            Coding::Direct {
                scales: Scales::BlockFloat { format, .. },
                ..
            } => result.push((
                PlaneField::BlockScale,
                Plane {
                    name: "block_scale",
                    group: self.group,
                    fields: 1,
                    encoding: PlaneEncoding::FloatCode { format: *format },
                },
            )),
            Coding::Hierarchical {
                factor_group,
                factor_dtype,
                bits,
                interpretation,
                locals,
                ..
            } => {
                let minimum = locals.minimum().is_some();
                let (bits, interpretation) = (*bits, interpretation.clone());
                result.push((
                    PlaneField::Coefficients,
                    Plane {
                        name: "coefficients",
                        group: self.group,
                        fields: if minimum { 2 } else { 1 },
                        encoding: match locals.bias() {
                            Some(LocalBias::Signed { .. }) => PlaneEncoding::SignedLocal {
                                scale_bits: locals.scale_bits(bits),
                                interpretation,
                            },
                            None | Some(LocalBias::Minimum(_)) => PlaneEncoding::Packed {
                                bits,
                                interpretation,
                            },
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
                if minimum {
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
        let (code_value, integer_code) = match self.coding.codes() {
            CodeStorage::Float(format) => (recipe.float_code(raw, format), None),
            CodeStorage::Fields {
                bits,
                interpretation,
            } => {
                let code = recipe.interpret(raw, bits, interpretation);
                (recipe.to_f32(code), Some(code))
            }
            CodeStorage::Grid(grid) => {
                let code = recipe.grid_code(raw, grid);
                (recipe.to_f32(code), Some(code))
            }
        };
        let mut scale_code = None;
        let mut bias_code = None;
        let (scale, bias, scale_factor, bias_factor, coefficient_code_max) = match &self.coding {
            Coding::Direct {
                scales: Scales::Stored { bias, .. },
                ..
            } => {
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
            Coding::Hierarchical {
                factor_exponent,
                bits,
                interpretation,
                locals,
                ..
            } => {
                let bias = locals.bias();
                let (scale_bits, scale_interpretation) =
                    (locals.scale_bits(*bits), interpretation.clone());
                let mut bias_factor = None;
                let (scale, scale_factor, bias) = match bias {
                    // The scale's code is the low bits; the bias's code
                    // is it signed by the top bit, under the one factor.
                    Some(LocalBias::Signed { exponent }) => {
                        let raw = recipe.read(PlaneField::Coefficients, 0);
                        let magnitude =
                            recipe.interpret(raw, scale_bits, scale_interpretation.clone());
                        let signed = recipe.signed_code(magnitude, raw, scale_bits);
                        let factor = recipe.read(PlaneField::ScaleFactor, 0);
                        let factor = recipe.to_f32(factor);
                        let converted = recipe.to_f32(magnitude);
                        let scale = recipe.scaled(factor, converted, *factor_exponent);
                        let converted = recipe.to_f32(signed);
                        let bias = recipe.scaled(factor, converted, factor_exponent + exponent);
                        scale_code = Some(magnitude);
                        bias_code = Some(signed);
                        bias_factor = Some(factor);
                        (scale, factor, Some(bias))
                    }
                    Some(LocalBias::Minimum(_)) | None => {
                        let (scale, scale_factor, code) = recipe.hierarchical_coefficient(
                            PlaneField::ScaleFactor,
                            *factor_exponent,
                            0,
                            scale_bits,
                            scale_interpretation.clone(),
                        );
                        scale_code = Some(code);
                        let bias = locals.minimum().map(|sign| {
                            let (product, factor, code) = recipe.hierarchical_coefficient(
                                PlaneField::BiasFactor,
                                *factor_exponent,
                                1,
                                *bits,
                                interpretation.clone(),
                            );
                            bias_factor = Some(factor);
                            bias_code = Some(code);
                            match sign {
                                MinimumSign::Subtract => recipe.negate(product),
                                MinimumSign::Add => product,
                            }
                        });
                        (scale, scale_factor, bias)
                    }
                };
                let maximum = *scale_interpretation
                    .codes(scale_bits)
                    .last()
                    .expect("sealed coefficient domain nonempty");
                (scale, bias, scale_factor, bias_factor, maximum)
            }
            Coding::Direct {
                scales: Scales::BlockFloat { format, .. },
                ..
            } => {
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
        match &self.coding {
            Coding::Direct {
                scales: Scales::Stored { .. },
                ..
            } => Some(Coefficient::Direct {
                plane: find(if bias {
                    PlaneField::Bias
                } else {
                    PlaneField::Scale
                })?,
            }),
            Coding::Hierarchical {
                locals,
                factor_exponent,
                bits,
                ..
            } => Some(match (bias, locals.bias()) {
                (true, Some(LocalBias::Minimum(sign))) => Coefficient::Product {
                    factor: find(PlaneField::BiasFactor)?,
                    exponent: *factor_exponent,
                    coefficients: find(PlaneField::Coefficients)?,
                    field: 1,
                    bits: *bits,
                    sign: sign.value(),
                },
                (true, Some(LocalBias::Signed { exponent })) => Coefficient::Product {
                    factor: find(PlaneField::ScaleFactor)?,
                    exponent: *factor_exponent + exponent,
                    coefficients: find(PlaneField::Coefficients)?,
                    field: 0,
                    bits: *bits,
                    sign: 1,
                },
                _ => Coefficient::Product {
                    factor: find(PlaneField::ScaleFactor)?,
                    exponent: *factor_exponent,
                    coefficients: find(PlaneField::Coefficients)?,
                    field: 0,
                    bits: locals.scale_bits(*bits),
                    sign: 1,
                },
            }),
            Coding::Direct {
                scales: Scales::BlockFloat { .. },
                ..
            } => Some(Coefficient::Direct {
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
                DecodeStep::SignedCode { code, raw, bit, .. } => {
                    let code = get(&temporaries, *code);
                    if (get(&temporaries, *raw) as u32 >> bit) & 1 == 1 {
                        -code
                    } else {
                        code
                    }
                }
                DecodeStep::ReadPlaneLane { plane, .. } => {
                    f64::from(recipe.planes()[*plane as usize].lane(value))
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
                PlaneEncoding::Packed { .. }
                | PlaneEncoding::Grid(_)
                | PlaneEncoding::SignedLocal { .. } => f64::from(
                    (entry as u32)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(ordinal.wrapping_mul(97))
                        & ((1u32 << plane.entry_bits()) - 1),
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
                    // Its scale code, negated where its top bit is set.
                    PlaneEncoding::SignedLocal {
                        scale_bits,
                        interpretation,
                    } => {
                        let code = interpretation
                            .decode(raw as u32 & ((1 << *scale_bits) - 1), *scale_bits)
                            as f32;
                        if (raw as u32 >> *scale_bits) & 1 == 1 {
                            -code
                        } else {
                            code
                        }
                    }
                    PlaneEncoding::Grid(_) => {
                        unreachable!("grid entries are codes, not coefficients")
                    }
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
                            bits,
                            sign,
                        }) => {
                            let entry = flat / coefficients.group as usize
                                * coefficients.fields as usize
                                + field as usize;
                            let code = match &coefficients.encoding {
                                PlaneEncoding::Packed { interpretation, .. }
                                | PlaneEncoding::SignedLocal { interpretation, .. }
                                    if bits < coefficients.entry_bits() =>
                                {
                                    let ordinal =
                                        repr.plane_index(coefficients.name).unwrap() as u32;
                                    let raw = plane_entry(ordinal, entry as u64) as u32;
                                    interpretation.decode(raw & ((1 << bits) - 1), bits) as f32
                                }
                                _ => plane_value(&coefficients, entry),
                            };
                            let factor = plane_value(&factor, flat / factor.group as usize);
                            (factor * code * 2.0f32.powi(exponent)) * sign as f32
                        }
                    }
                };
                let words = &recipe.planes()[0];
                let code = plane_entry(0, words.entry(flat as u64, 0)) as u32;
                let decoded_code = match &repr.coding.codes() {
                    CodeStorage::Float(format) => format.decode(code),
                    CodeStorage::Grid(grid) => grid.code(code, flat as u32 % GRID_VALUES) as f32,
                    CodeStorage::Fields { .. } => repr.decode_code(code) as f32,
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
                        DecodeStep::SignedCode {
                            into,
                            code,
                            raw,
                            bit,
                        } => {
                            assert_eq!(recipe.dtype(*code), DType::I32);
                            assert_eq!(recipe.dtype(*raw), DType::U32);
                            assert_eq!(recipe.dtype(*into), DType::I32);
                            assert!(*bit < recipe.read_bits(*raw));
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
            let expected = match &repr.coding {
                Coding::Hierarchical { locals, .. } => {
                    usize::from(locals.minimum() == Some(MinimumSign::Subtract))
                }
                Coding::Direct { .. } => 0,
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
        for (name, expected) in [
            ("q4k", 144),
            ("q5k", 176),
            ("q6k", 210),
            ("q1g128", 18 * 2),
            ("q2g64", 18 * 4),
            ("iq2s", 82),
        ] {
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
