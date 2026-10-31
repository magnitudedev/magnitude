//! Consumer-form admission. Which of a backend's consumer forms a packed
//! row-layout weight takes is decided here, once, and emitted as
//! `<prefix>_FORM_<NAME>`: 0 or 1, and the values an admitted form reads its
//! codes with. Native libraries select on these and assert they implement
//! what is admitted, so a weight never falls back to another form by its
//! decoder's shape; host planning reads the same admission
//! (`metal_packing_of`).
//!
//! A form admits a weight when two independent facts hold:
//! - what the form's arithmetic requires of the weight's profile
//!   (`registry::profile::Profile`: its integer codes and levels), and
//! - that its library's producer delivers the form's operand from how the
//!   weight is stored (its decode axes: code planes, codebooks, grids and
//!   where its local codes live).
//!
//! A form's arithmetic requirements change only with the form, and its
//! producer's reach only with the library's producers.
//!
//! Native declarations read the same admission (`form(W, NAME)`, a
//! [`Form`]), so a configuration names only forms its weights take.
//!
//! The GPUs' forms are of packed rows, a dense weight having one path; the
//! CPU's are of every weight, whose dense rows take its quantized
//! activations too.

use crate::ids::RepresentationId;
use crate::registry::{self, BackendName, OperandRecipe, RepresentationInfo};
use crate::registry::profile::{Bias, LocalScale, Profile, ValueMap};
use crate::types::DType;

/// CUDA's forms, in emission order: the folded GEMM recipe (`GEMM_FOLD`)
/// and the int8 GEMM (`INT8`). [`cuda_forms`] values them in this order.
const CUDA_FORMS: [&str; 2] = ["GEMM_FOLD", "INT8"];

/// Metal's forms, in emission order (`packets::Forms` reads those its
/// library selects on). [`metal_forms`] values them in this order.
const METAL_FORMS: [&str; 10] = [
    "PACK",
    "PACK_CENTRE",
    "PACK_OFFSET",
    "PACK_RESIDUE",
    "PACK_FOLDS",
    "INT8",
    "INT8_INTERLEAVED",
    "MATRIX_CODES",
    "MATRIX_RUNS",
    "MATRIX_SYMMETRIC",
];

/// The CPU's forms: the quantized dot (`INT8`). [`cpu_int8`] values it.
const CPU_FORMS: [&str; 1] = ["INT8"];

/// The names of `backend`'s forms, in emission order.
pub fn names(backend: BackendName) -> &'static [&'static str] {
    match backend {
        BackendName::Cuda => &CUDA_FORMS,
        BackendName::Metal => &METAL_FORMS,
        BackendName::Cpu => &CPU_FORMS,
        BackendName::Vulkan => &[],
    }
}

/// The values of `backend`'s forms for representation `id`, in the order of
/// [`names`]; `None` where the backend chooses no form of it: on a GPU,
/// storage other than packed rows.
fn values(backend: BackendName, id: RepresentationId) -> Option<Vec<u32>> {
    let info = registry::representation_info(id);
    if backend == BackendName::Cpu {
        return Some(vec![u32::from(cpu_int8(info))]);
    }
    let (registry::RepresentationKind::PackedRows(rows), Some(axes)) = (&info.kind, registry::decode_axes(id))
    else {
        return None;
    };
    Some(match backend {
        BackendName::Cuda => cuda_forms(&axes, rows, info.operands).to_vec(),
        BackendName::Metal => metal_forms(&axes).to_vec(),
        BackendName::Cpu | BackendName::Vulkan => Vec::new(),
    })
}

/// The forms admitted for representation `id` on `backend`, in emission
/// order, with their values; none where the backend chooses no form of it
/// (`values`).
pub fn forms(backend: BackendName, id: RepresentationId) -> Vec<(&'static str, u32)> {
    values(backend, id).map_or_else(Vec::new, |values| names(backend).iter().copied().zip(values).collect())
}

/// One of a backend's forms, as a native declaration of that backend names
/// it (`form(W, NAME)`): only a name of [`names`] is represented, so its
/// value is defined for every representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "(BackendName, String)", into = "(BackendName, String)")]
pub struct Form {
    backend: BackendName,
    /// Its position in `names(backend)`.
    index: usize,
}

impl Form {
    /// Form `name` of `backend`, or `None` when the backend has no form of
    /// that name.
    pub fn named(backend: BackendName, name: &str) -> Option<Self> {
        let index = names(backend).iter().position(|candidate| *candidate == name)?;
        Some(Self { backend, index })
    }

    pub fn name(self) -> &'static str {
        names(self.backend)[self.index]
    }

    /// The form's admitted value for representation `id` (`forms`): 0 where
    /// the backend chooses no form of it.
    pub fn of(self, id: RepresentationId) -> u32 {
        values(self.backend, id).map_or(0, |values| values[self.index])
    }

    /// Every value the form takes over the registry's representations, in
    /// increasing order.
    pub fn values(self) -> Vec<u32> {
        let values = registry::representations()
            .iter()
            .map(|info| self.of(info.id))
            .collect::<std::collections::BTreeSet<_>>();
        values.into_iter().collect()
    }
}

impl TryFrom<(BackendName, String)> for Form {
    type Error = String;

    fn try_from((backend, name): (BackendName, String)) -> Result<Self, Self::Error> {
        Self::named(backend, &name).ok_or_else(|| format!("`{}` has no form `{name}`", backend.as_str()))
    }
}

impl From<Form> for (BackendName, String) {
    fn from(form: Form) -> Self {
        (form.backend, form.name().to_owned())
    }
}

/// CUDA's forms of a packed-rows weight, in the order of [`CUDA_FORMS`].
fn cuda_forms(
    axes: &registry::DecodeAxes,
    rows: &registry::PackedRowLayout,
    operands: OperandRecipe,
) -> [u32; CUDA_FORMS.len()] {
    [u32::from(cuda_gemm_fold(axes, rows)), u32::from(cuda_int8(axes, operands))]
}

/// Whether CUDA's GEMM multiplies the weight's exact codes and applies each
/// group's coefficients to its F32 partial (`fold`), rather than rounding
/// each value to the operand type first.
///
/// The fold recipe requires groups that split a 32-code half (16 or 32
/// codes) and codes its producer reads as exact operand integers (`q` of at
/// most 8 bits, which BF16 and F16 hold). Of the weights it admits, the fold
/// is chosen for grid codes and for linear codes below 8 bits under direct
/// coefficients, except whole 256-code rows (iq4g32, whose rounded GEMM its
/// Qwen results were qualified with): today's choice, which tuning owns once
/// both recipes are qualified for each weight.
fn cuda_gemm_fold(axes: &registry::DecodeAxes, rows: &registry::PackedRowLayout) -> bool {
    let profile = axes.profile();
    let admitted = matches!(profile.local.group, 16 | 32) && profile.codes.fits_i8();
    let chosen = match profile.map {
        ValueMap::Grid => true,
        ValueMap::Linear | ValueMap::Table => {
            matches!(profile.local.scale, LocalScale::Stored(_)) && axes.low != 8 && rows.group() % 256 != 0
        }
    };
    admitted && chosen
}

/// Whether CUDA's GEMM folds representation `id`'s codes
/// (`cuda_gemm_fold`); false for storage other than packed rows.
pub fn cuda_gemm_fold_of(id: RepresentationId) -> bool {
    match (&registry::representation_info(id).kind, registry::decode_axes(id)) {
        (registry::RepresentationKind::PackedRows(rows), Some(axes)) => cuda_gemm_fold(&axes, rows),
        _ => false,
    }
}

/// Whether CUDA's int8 GEMM takes representation `id` (`cuda_int8`, the
/// `INT8` form); false for storage other than packed rows.
pub fn cuda_int8_of(id: RepresentationId) -> bool {
    let info = registry::representation_info(id);
    match (&info.kind, registry::decode_axes(id)) {
        (registry::RepresentationKind::PackedRows(_), Some(axes)) => cuda_int8(&axes, info.operands),
        _ => false,
    }
}

/// Whether CUDA's int8 GEMM (`gemm_tile_s8`, a declaration's `INT8 = 1`)
/// takes the weight: it multiplies the weight's codes as s8 against the
/// activations it quantizes per 32 codes, one m16n8k32 MMA per 32 codes
/// into an exact int32 product, and folds each product under its group's
/// coefficients.
///
/// Its arithmetic requires integer codes of `i8`; groups of 16 codes (half
/// an MMA each) or of 32 and more (whole MMAs); and activations it
/// quantizes itself, so not a weight whose products take NVFP4 operand
/// rows, which its recipe defines (`OperandRecipe`).
///
/// Its producer delivers every packed-rows weight's codes as s8
/// (`packets::Packed::s8`, `packets::Records::s8`).
fn cuda_int8(axes: &registry::DecodeAxes, operands: OperandRecipe) -> bool {
    let profile = axes.profile();
    let group = profile.local.group;
    profile.codes.fits_i8() && (group == 16 || group >= 32) && operands != OperandRecipe::Nvfp4Rows
}

/// Metal's forms, in the order of [`METAL_FORMS`]: the token-packing GEMM
/// with its codes' entry and its operand's layout (`PACK`, `PACK_CENTRE`,
/// `PACK_OFFSET`, `PACK_RESIDUE`, `PACK_FOLDS`; 0 where not admitted), the
/// int8 GEMM with its operand's order (`INT8`, `INT8_INTERLEAVED`) and the
/// matrix GEMV, by one of its three products or none (`MATRIX_CODES`,
/// `MATRIX_RUNS`, `MATRIX_SYMMETRIC`). Each admits the weight on any device;
/// whether a device runs the form (tensor operations) is the library's
/// condition. One operand serves several weights of a declaration only
/// where their operand layout (`PACK_FOLDS`, `PACK_RESIDUE`) or order
/// (`INT8_INTERLEAVED`) agree.
fn metal_forms(axes: &registry::DecodeAxes) -> [u32; METAL_FORMS.len()] {
    let packing = metal_packing(axes);
    let int8 = metal_int8(axes);
    let matrix = metal_matrix(axes);
    [
        u32::from(packing.is_some()),
        packing.map_or(0, |packing| packing.centre),
        packing.map_or(0, |packing| packing.offset),
        packing.map_or(0, |packing| u32::from(packing.residue)),
        packing.map_or(0, |packing| packing.folds),
        u32::from(int8.is_some()),
        u32::from(int8 == Some(MetalInt8::Nibbles)),
        u32::from(matrix == Some(MetalMatrix::Codes)),
        u32::from(matrix == Some(MetalMatrix::Runs)),
        u32::from(matrix == Some(MetalMatrix::Symmetric)),
    ]
}

/// Columns of a Metal packet, whose coefficient groups a decoder holds
/// (`packets::Direct::groups`, `packets::Hier::groups`).
const METAL_PACKET: u32 = 32;

/// The coefficient groups of a Metal packet: two of 16 columns, else one.
fn metal_packet_groups(profile: &Profile) -> u32 {
    let group = profile.local.group;
    if group < METAL_PACKET { METAL_PACKET / group } else { 1 }
}

/// Whether Metal's code planes read a weight's codes as
/// `packets::Offset<ZERO>` (`Unsigned` is `Offset<0>`): linear codes of a
/// plane of low nibbles and, with `high`, a plane of that many high bits.
fn metal_offset_planes(axes: &registry::DecodeAxes, high: &[u32]) -> bool {
    matches!(axes.code, registry::CodeInterpretation::Unsigned | registry::CodeInterpretation::Offset(_))
        && axes.low == 4
        && high.contains(&axes.high)
}

/// Whether a Metal producer reads the weight's local codes from its scales
/// plane, of plain unsigned codes (`packets::LocalUnsigned`), rather than
/// from its grid entries.
fn metal_plane_locals(axes: &registry::DecodeAxes, unsigned: bool) -> bool {
    match &axes.coefficients {
        registry::CoefficientAxes::Direct { .. } => true,
        registry::CoefficientAxes::Hierarchical { locals, .. } => match locals {
            registry::LocalCodes::Plane { interpretation, .. } => {
                !unsigned || *interpretation == registry::CodeInterpretation::Unsigned
            }
            registry::LocalCodes::Carried { .. } => false,
        },
    }
}

/// A weight's admission to Metal's token-packing GEMM (`PACK`,
/// `kernels/metal/lib/projection/packing.h`), which multiplies its codes
/// minus their centre against two tokens' packed operand and folds each
/// run's sums under the run's scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalPacking {
    /// Runs a 32-column block: the coefficient groups of a packet.
    pub folds: u32,
    /// A code enters as its value + `offset`, an unsigned integer of the
    /// entry's width, centred at `centre`, half the entry's range.
    pub centre: u32,
    pub offset: u32,
    /// Whether the packed sums take residue bits beside their 16-bit fields:
    /// the entries are values of 8 bits, whose sums 16 bits a token do not
    /// hold at a useful gain.
    pub residue: bool,
}

impl MetalPacking {
    /// The columns of one run, which share a scale: the packing bound holds
    /// one value per run (`packing_bound`).
    pub fn run(self) -> u32 {
        METAL_PACKET / self.folds
    }
}

/// Metal's token-packing admission of representation `id`
/// (`MetalPacking`); `None` for storage other than packed rows, or which
/// the form does not serve.
pub fn metal_packing_of(id: RepresentationId) -> Option<MetalPacking> {
    metal_packing(&registry::decode_axes(id)?)
}

/// The token-packing GEMM (`packing_layout`, `packing_factors`).
///
/// Its arithmetic requires integer values, not scaled by a power of two of
/// their own, whose coefficients fold per run of a 32-column block: a factor
/// per 256 columns over local codes, or direct groups of whole runs in 256
/// columns; and a block of two runs only without a bias term,
/// (centre - offset) scale + bias.
///
/// Its producer packs entries of 4, 5 or 8 bits in runs of 32 columns or of
/// 6 bits in runs of 16: linear codes enter as themselves, a codebook's
/// 4-bit codes as its values of 8 bits + 128 (`codebooks::pairs`), and local
/// codes come from the scales plane.
fn metal_packing(axes: &registry::DecodeAxes) -> Option<MetalPacking> {
    let profile = axes.profile();
    let (bits, offset) = match profile.map {
        ValueMap::Linear if axes.code != registry::CodeInterpretation::Odd => {
            (axes.low + axes.high, u32::try_from(-profile.codes.least).ok()?)
        }
        ValueMap::Table if (axes.low, axes.high) == (4, 0) && profile.codes.fits_i8() => (8, 128),
        ValueMap::Linear | ValueMap::Table | ValueMap::Grid => return None,
    };
    let folds = metal_packet_groups(&profile);
    let entries = matches!((bits, folds), (4 | 5 | 8, 1) | (6, 2)) && metal_plane_locals(axes, false);
    let centre = 1u32 << (bits - 1);
    let biased = profile.local.bias != Bias::None || centre != offset;
    let factors = match profile.upper {
        Some(upper) => upper.group == 256,
        None => 256 % profile.local.group == 0,
    };
    let requires = profile.code_exponent == 0 && factors && (folds == 1 || !biased);
    (requires && entries).then_some(MetalPacking {
        folds,
        centre,
        offset,
        residue: bits == 8,
    })
}

/// The codes of Metal's int8 GEMM (`int8_codes`), by the order a lane's
/// fragment takes them in, which the quantized operand matches
/// (`int8_codes::interleaved`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetalInt8 {
    /// Unsigned 4-bit codes, a byte two of them: a lane's fragment takes
    /// its even columns, then its odd ones (`INT8_INTERLEAVED`).
    Nibbles,
    /// Two's complement 8-bit codes, in column order.
    Bytes,
}

/// Metal's int8 GEMM (`int8_codes`, on tensor operations), whose operands
/// are the weight's codes as int8.
///
/// Its arithmetic requires integer codes of `i8` under groups of 32
/// columns, not scaled by a power of two of their own: under a factor per
/// 256 columns of at most 16 bits over local codes of `u8`, so a factor
/// times a local code is exact in F32, or under a stored scale without a
/// bias.
///
/// Its producer reads unsigned 4-bit codes of one plane with plain unsigned
/// local codes of the scales plane (`Nibbles`), or 8-bit two's complement
/// codes (`Bytes`).
fn metal_int8(axes: &registry::DecodeAxes) -> Option<MetalInt8> {
    let profile = axes.profile();
    let codes = profile.codes.fits_i8() && profile.code_exponent == 0 && profile.local.group == 32;
    let coefficients = match (profile.upper, profile.local.scale, profile.local.bias) {
        (Some(upper), LocalScale::Code(locals), _) => {
            let narrow = upper.factor != registry::ScaleEncoding::F32;
            upper.group == 256 && narrow && 0 <= locals.least && locals.greatest <= i32::from(u8::MAX)
        }
        (None, LocalScale::Stored(_), Bias::None) => true,
        _ => false,
    };
    let produced = match profile.upper {
        Some(_) => (axes.code == registry::CodeInterpretation::Unsigned
            && (axes.low, axes.high) == (4, 0)
            && metal_plane_locals(axes, true))
        .then_some(MetalInt8::Nibbles),
        None => (axes.code == registry::CodeInterpretation::TwosComplement && (axes.low, axes.high) == (8, 0))
            .then_some(MetalInt8::Bytes),
    };
    produced.filter(|_| codes && coefficients)
}

/// Whether the CPU's quantized dot (`weights::Format::dot_q8`, a
/// declaration's `INT8 = 1`) takes the weight: it multiplies activations
/// quantized to Q8 blocks (`quant::Q8Block`: i8 codes under a scale per
/// block, with their sums per 16 values) by the weight's values, packed
/// rows in exact integer products a packet at a time (AVX2 pair sums, VNNI
/// or SDOT) folded under each coefficient group's scale and bias, dense
/// rows in F32.
///
/// Its arithmetic requires, of packed rows, integer codes of `i8`, so that
/// every tier's products are exact (two products of an AVX2 pair sum are at
/// most 2 · 128 · 127, inside i16; non-negative codes enter the unsigned
/// byte products as themselves, others as magnitudes against sign-adjusted
/// activations); coefficient groups of 16 values (half a
/// packet) or of whole packets; and activations it quantizes itself, so not
/// a weight whose products take NVFP4 operand rows, which its recipe
/// defines (`OperandRecipe`).
///
/// Its producer reads the codes of every packed-rows decode
/// (`weights::Codes`) and widens the floating dense elements
/// (`weights::DenseRows`).
pub fn cpu_int8_of(id: RepresentationId) -> bool {
    cpu_int8(registry::representation_info(id))
}

/// [`cpu_int8_of`]'s rule over the representation's registry record.
fn cpu_int8(info: &RepresentationInfo) -> bool {
    if info.operands == OperandRecipe::Nvfp4Rows {
        return false;
    }
    match (&info.kind, registry::decode_axes(info.id)) {
        (registry::RepresentationKind::Dense(dtype), _) => matches!(dtype, DType::F32 | DType::BF16 | DType::F16),
        (registry::RepresentationKind::PackedRows(_), Some(axes)) => {
            let profile = axes.profile();
            let group = profile.local.group;
            profile.codes.fits_i8() && (group == 16 || group % 32 == 0)
        }
        _ => false,
    }
}

/// The products of Metal's matrix GEMV (`matrix_form`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetalMatrix {
    /// Unsigned codes under hierarchical coefficients with a minimum, an
    /// unsigned local scale code per 32 columns and factors per 256, where
    /// a code times a scale code stays below 2048, an exact F16 integer of
    /// the operand's 2^-24 (`matrix_codes`).
    Codes,
    /// Codes under hierarchical coefficients without a minimum, a local
    /// scale code per 16 columns and factors per 256 (`matrix_runs`).
    Runs,
    /// Codes under a direct scale per 32 columns without a bias
    /// (`matrix_symmetric`).
    Symmetric,
}

/// The matrix GEMV product serving a weight.
///
/// Each product's arithmetic requires integer codes not scaled by a power
/// of two of their own, under its levels (`MetalMatrix`). Its producer reads
/// them as `packets::Offset` codes of a plane of low nibbles: with a plane
/// of at most one high bit and plain unsigned local codes of the scales
/// plane (`Codes`), with a plane of two high bits and local codes of the
/// scales plane (`Runs`), or alone (`Symmetric`).
fn metal_matrix(axes: &registry::DecodeAxes) -> Option<MetalMatrix> {
    let profile = axes.profile();
    if profile.code_exponent != 0 {
        return None;
    }
    let factors = profile.upper.is_some_and(|upper| upper.group == 256);
    match (profile.local.group, profile.local.scale, profile.local.bias) {
        (32, LocalScale::Code(locals), Bias::Minimum { .. })
            if factors
                && profile.codes.least == 0
                && metal_offset_planes(axes, &[0, 1])
                && metal_plane_locals(axes, true) =>
        {
            (profile.codes.greatest * locals.greatest < 2048).then_some(MetalMatrix::Codes)
        }
        (16, LocalScale::Code(_), Bias::None)
            if factors && metal_offset_planes(axes, &[2]) && metal_plane_locals(axes, false) =>
        {
            Some(MetalMatrix::Runs)
        }
        (32, LocalScale::Stored(_), Bias::None) if metal_offset_planes(axes, &[0]) => Some(MetalMatrix::Symmetric),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CUDA's GEMM recipe of every representation it imports, as main's
    /// library derived it (`ScaledGemm`), and for q4g32, q2k and q2g256 as
    /// the same choice takes them.
    #[test]
    fn cuda_folds_are_mains() {
        let folded = [
            "q4g32s", "iq4g32s", "q5g32s", "q5g32", "mxfp4g32", "nvfp4g16", "nvfp4g16a4", "iq2xs", "iq3xxs", "iq2xxs",
            "q4g32",
        ];
        let rounded = ["q4k", "q5k", "q6k", "q8g32s", "iq4g32", "q2k", "q2g256"];
        let expected = folded.map(|storage| (storage, true)).into_iter().chain(rounded.map(|storage| (storage, false)));
        for (storage, fold) in expected {
            let id = registry::storage(storage, registry::Layout::Mma16).unwrap();
            assert_eq!(cuda_gemm_fold_of(id), fold, "{storage}");
        }
    }

    /// CUDA's int8 GEMM admission of every representation it imports: main's
    /// library took the INT8 path for every packed weight whose products do
    /// not take NVFP4 operand rows (`projection::quantizable`), and so does
    /// the admission, of every packed-rows representation of the registry.
    #[test]
    fn cuda_int8_is_mains() {
        let admitted = [
            "q4k", "q5k", "q6k", "q8g32s", "iq4g32", "q4g32s", "iq4g32s", "q5g32s", "q5g32", "mxfp4g32", "nvfp4g16",
            "iq2xs", "iq3xxs", "iq2xxs", "q4g32", "q2k", "q2g256",
        ];
        let excluded = ["nvfp4g16a4"];
        let int8 = Form::named(BackendName::Cuda, "INT8").unwrap();
        let expected = admitted.map(|storage| (storage, 1)).into_iter().chain(excluded.map(|storage| (storage, 0)));
        for (storage, value) in expected {
            let id = registry::storage(storage, registry::Layout::Mma16).unwrap();
            assert_eq!(int8.of(id), value, "{storage}");
        }
        for info in registry::representations() {
            let main = match info.kind {
                registry::RepresentationKind::PackedRows(_) => u32::from(info.operands != OperandRecipe::Nvfp4Rows),
                _ => 0,
            };
            assert_eq!(int8.of(info.id), main, "{}", info.name);
        }
    }

    /// The CPU's quantized dot admission of every representation: main's
    /// projections quantized their activations for every weight with a CPU
    /// component, the floating dense elements and every packed-rows decode,
    /// except one whose products take NVFP4 operand rows, whose quantized
    /// rows they dropped for the F32 dot (`projection::project_arithmetic`).
    #[test]
    fn cpu_int8_is_mains() {
        let admitted = [
            "f32", "bf16", "f16", "q4k@rows16", "q5k@rows16", "q6k@rows16", "q8g32s@rows16", "iq4g32@rows16",
            "q4g32s@rows16", "iq4g32s@rows16", "q5g32s@rows16", "q5g32@rows16", "mxfp4g32@rows16",
            "nvfp4g16@rows16", "iq2xs@rows16", "iq3xxs@rows16", "iq2xxs@rows16", "q4g32@rows16", "q2k@rows16",
            "q2g256@rows16", "q4k@rows8",
        ];
        let excluded = ["nvfp4g16a4@rows16"];
        let int8 = Form::named(BackendName::Cpu, "INT8").unwrap();
        let expected = admitted.map(|name| (name, 1)).into_iter().chain(excluded.map(|name| (name, 0)));
        for (name, value) in expected {
            let id = registry::representation(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(int8.of(id), value, "{name}");
        }
        for info in registry::representations() {
            let main = match info.kind {
                registry::RepresentationKind::Dense(dtype) => matches!(dtype, DType::F32 | DType::BF16 | DType::F16),
                registry::RepresentationKind::PackedRows(_) => info.operands != OperandRecipe::Nvfp4Rows,
                _ => false,
            };
            assert_eq!(int8.of(info.id), u32::from(main), "{}", info.name);
        }
    }

    /// Every backend's forms are its names, valued on a GPU for packed rows
    /// only and on the CPU for every representation.
    #[test]
    fn forms_are_named_by_their_backend() {
        for backend in BackendName::ALL {
            for info in registry::representations() {
                let forms = forms(backend, info.id);
                let valued = backend == BackendName::Cpu
                    || matches!(info.kind, registry::RepresentationKind::PackedRows(_));
                let named = forms.iter().map(|(name, _)| *name).collect::<Vec<_>>();
                assert_eq!(named, if valued { names(backend).to_vec() } else { Vec::new() }, "{}", info.name);
                for (name, value) in forms {
                    assert_eq!(Form::named(backend, name).unwrap().of(info.id), value, "{} {name}", info.name);
                }
            }
            assert_eq!(Form::named(backend, "INT9"), None);
        }
    }

    /// The operand layout and order Metal's library takes for every
    /// representation of the registry, as main's library read them from an
    /// admitted weight's decoder: the token-packing operand's runs a block,
    /// the coefficient groups of a packet (`packing_codes::folds`,
    /// `Packet::groups`), and whether the int8 operand is interleaved, its
    /// codes unsigned nibbles (`int8_codes::interleaved`); 0 for weights not
    /// admitted.
    #[test]
    fn metal_operand_layouts_are_mains() {
        let name = |form| Form::named(BackendName::Metal, form).unwrap();
        let (pack, folds, int8, interleaved) = (name("PACK"), name("PACK_FOLDS"), name("INT8"), name("INT8_INTERLEAVED"));
        for info in registry::representations() {
            let groups = registry::decode_axes(info.id).map_or(0, |axes| metal_packet_groups(&axes.profile()));
            let nibbles = registry::decode_axes(info.id)
                .is_some_and(|axes| axes.code == registry::CodeInterpretation::Unsigned && (axes.low, axes.high) == (4, 0));
            assert_eq!(folds.of(info.id), pack.of(info.id) * groups, "{}", info.name);
            assert_eq!(interleaved.of(info.id), int8.of(info.id) * u32::from(nibbles), "{}", info.name);
        }
        let expected = [("q4k", 1, 1), ("q6k", 2, 0), ("q8g32s", 1, 0), ("q4g32s", 1, 0), ("mxfp4g32", 0, 0)];
        for (storage, runs, nibbles) in expected {
            let id = registry::storage(storage, registry::Layout::Rows32).unwrap();
            assert_eq!((folds.of(id), interleaved.of(id)), (runs, nibbles), "{storage}");
        }
    }

    /// Metal's admissions of every representation it imports, as main's
    /// libraries matched their decoders' shapes (and, for q4g32, q2k and
    /// q2g256, as the same rules admit them): PACK (folds, centre, offset,
    /// residue), INT8 and the matrix GEMV product.
    #[test]
    fn metal_admissions_are_mains() {
        let pack = |folds, centre, offset, residue| {
            Some(MetalPacking {
                folds,
                centre,
                offset,
                residue,
            })
        };
        let expected = [
            ("q4k", pack(1, 8, 0, false), Some(MetalInt8::Nibbles), Some(MetalMatrix::Codes)),
            ("q5k", pack(1, 16, 0, false), None, Some(MetalMatrix::Codes)),
            ("q6k", pack(2, 32, 32, false), None, Some(MetalMatrix::Runs)),
            ("q8g32s", pack(1, 128, 128, true), Some(MetalInt8::Bytes), None),
            ("iq4g32", pack(1, 128, 128, true), None, None),
            ("q4g32s", pack(1, 8, 8, false), None, Some(MetalMatrix::Symmetric)),
            ("iq4g32s", pack(1, 128, 128, true), None, None),
            ("q5g32s", pack(1, 16, 16, false), None, None),
            ("q5g32", pack(1, 16, 0, false), None, None),
            ("mxfp4g32", None, None, None),
            ("nvfp4g16", None, None, None),
            ("nvfp4g16a4", None, None, None),
            ("iq2xs", None, None, None),
            ("iq3xxs", None, None, None),
            ("iq2xxs", None, None, None),
            ("q4g32", pack(1, 8, 0, false), None, None),
            ("q2k", None, None, None),
            ("q2g256", None, None, None),
        ];
        for (storage, packing, int8, matrix) in expected {
            let id = registry::storage(storage, registry::Layout::Rows32).unwrap();
            let axes = registry::decode_axes(id).unwrap();
            assert_eq!(
                (metal_packing(&axes), metal_int8(&axes), metal_matrix(&axes)),
                (packing, int8, matrix),
                "{storage}"
            );
        }
    }
}
