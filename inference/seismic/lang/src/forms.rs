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

use crate::ids::RepresentationId;
use crate::registry::{self, BackendName, OperandRecipe};
use crate::registry::profile::{Bias, LocalScale, Profile, ValueMap};

/// CUDA's forms, in emission order: the folded GEMM recipe (`GEMM_FOLD`)
/// and the int8 GEMM (`INT8`). [`cuda_forms`] values them in this order.
const CUDA_FORMS: [&str; 2] = ["GEMM_FOLD", "INT8"];

/// Metal's forms, in emission order (`packets::Forms`). [`metal_forms`]
/// values them in this order.
const METAL_FORMS: [&str; 8] = [
    "PACK",
    "PACK_CENTRE",
    "PACK_OFFSET",
    "PACK_RESIDUE",
    "INT8",
    "MATRIX_CODES",
    "MATRIX_RUNS",
    "MATRIX_SYMMETRIC",
];

/// The names of `backend`'s forms, in emission order.
pub fn names(backend: BackendName) -> &'static [&'static str] {
    match backend {
        BackendName::Cuda => &CUDA_FORMS,
        BackendName::Metal => &METAL_FORMS,
        BackendName::Cpu | BackendName::Vulkan => &[],
    }
}

/// The values of `backend`'s forms for representation `id`, in the order of
/// [`names`]; `None` for storage other than packed rows, which has none.
fn values(backend: BackendName, id: RepresentationId) -> Option<Vec<u32>> {
    let info = registry::representation_info(id);
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
/// order, with their values; none for storage other than packed rows.
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

    /// The form's admitted value for representation `id` (`forms`): 0 for
    /// storage other than packed rows, which has no forms.
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

/// Metal's forms (`packets::Forms`), in the order of [`METAL_FORMS`]: the token-packing GEMM with its codes'
/// entry (`PACK`, `PACK_CENTRE`, `PACK_OFFSET`, `PACK_RESIDUE`; 0 where not
/// admitted), the int8 GEMM (`INT8`) and the matrix GEMV, by one of its
/// three products or none (`MATRIX_CODES`, `MATRIX_RUNS`,
/// `MATRIX_SYMMETRIC`). Each admits the weight on any device; whether a
/// device runs the form (tensor operations) is the library's condition.
fn metal_forms(axes: &registry::DecodeAxes) -> [u32; METAL_FORMS.len()] {
    let packing = metal_packing(axes);
    let matrix = metal_matrix(axes);
    [
        u32::from(packing.is_some()),
        packing.map_or(0, |packing| packing.centre),
        packing.map_or(0, |packing| packing.offset),
        packing.map_or(0, |packing| u32::from(packing.residue)),
        u32::from(metal_int8(axes)),
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
/// local codes of the scales plane, or 8-bit two's complement codes.
fn metal_int8(axes: &registry::DecodeAxes) -> bool {
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
        Some(_) => {
            axes.code == registry::CodeInterpretation::Unsigned
                && (axes.low, axes.high) == (4, 0)
                && metal_plane_locals(axes, true)
        }
        None => axes.code == registry::CodeInterpretation::TwosComplement && (axes.low, axes.high) == (8, 0),
    };
    codes && coefficients && produced
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

    /// Every backend's forms are its names, valued for packed rows only.
    #[test]
    fn forms_are_named_by_their_backend() {
        for backend in BackendName::ALL {
            for info in registry::representations() {
                let forms = forms(backend, info.id);
                let packed = matches!(info.kind, registry::RepresentationKind::PackedRows(_));
                let named = forms.iter().map(|(name, _)| *name).collect::<Vec<_>>();
                assert_eq!(named, if packed { names(backend).to_vec() } else { Vec::new() }, "{}", info.name);
                for (name, value) in forms {
                    assert_eq!(Form::named(backend, name).unwrap().of(info.id), value, "{} {name}", info.name);
                }
            }
            assert_eq!(Form::named(backend, "INT9"), None);
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
            ("q4k", pack(1, 8, 0, false), true, Some(MetalMatrix::Codes)),
            ("q5k", pack(1, 16, 0, false), false, Some(MetalMatrix::Codes)),
            ("q6k", pack(2, 32, 32, false), false, Some(MetalMatrix::Runs)),
            ("q8g32s", pack(1, 128, 128, true), true, None),
            ("iq4g32", pack(1, 128, 128, true), false, None),
            ("q4g32s", pack(1, 8, 8, false), false, Some(MetalMatrix::Symmetric)),
            ("iq4g32s", pack(1, 128, 128, true), false, None),
            ("q5g32s", pack(1, 16, 16, false), false, None),
            ("q5g32", pack(1, 16, 0, false), false, None),
            ("mxfp4g32", None, false, None),
            ("nvfp4g16", None, false, None),
            ("nvfp4g16a4", None, false, None),
            ("iq2xs", None, false, None),
            ("iq3xxs", None, false, None),
            ("iq2xxs", None, false, None),
            ("q4g32", pack(1, 8, 0, false), false, None),
            ("q2k", None, false, None),
            ("q2g256", None, false, None),
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
