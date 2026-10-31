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
//! Native declarations read the same admission (`form(W, NAME)`), so a
//! configuration names only forms its weights take.

use crate::ids::RepresentationId;
use crate::registry::{self, BackendName};
use crate::registry::profile::{Bias, LocalScale, Profile, ValueMap};

/// The forms admitted for representation `id` on `backend`, in emission
/// order, with their values; none for storage other than packed rows.
pub fn forms(backend: BackendName, id: RepresentationId) -> Vec<(&'static str, u32)> {
    let (registry::RepresentationKind::PackedRows(rows), Some(axes)) =
        (&registry::representation_info(id).kind, registry::decode_axes(id))
    else {
        return Vec::new();
    };
    match backend {
        BackendName::Cuda => vec![("GEMM_FOLD", u32::from(cuda_gemm_fold(&axes, rows)))],
        BackendName::Metal => metal_forms(&axes),
        BackendName::Cpu | BackendName::Vulkan => Vec::new(),
    }
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

/// Metal's forms (`packets::Forms`): the token-packing GEMM with its codes'
/// entry (`PACK`, `PACK_CENTRE`, `PACK_OFFSET`, `PACK_RESIDUE`; 0 where not
/// admitted), the int8 GEMM (`INT8`) and the matrix GEMV, by one of its
/// three products or none (`MATRIX_CODES`, `MATRIX_RUNS`,
/// `MATRIX_SYMMETRIC`). Each admits the weight on any device; whether a
/// device runs the form (tensor operations) is the library's condition.
fn metal_forms(axes: &registry::DecodeAxes) -> Vec<(&'static str, u32)> {
    let packing = metal_packing(axes);
    let matrix = metal_matrix(axes);
    vec![
        ("PACK", u32::from(packing.is_some())),
        ("PACK_CENTRE", packing.map_or(0, |packing| packing.centre)),
        ("PACK_OFFSET", packing.map_or(0, |packing| packing.offset)),
        ("PACK_RESIDUE", packing.map_or(0, |packing| u32::from(packing.residue))),
        ("INT8", u32::from(metal_int8(axes))),
        ("MATRIX_CODES", u32::from(matrix == Some(MetalMatrix::Codes))),
        ("MATRIX_RUNS", u32::from(matrix == Some(MetalMatrix::Runs))),
        ("MATRIX_SYMMETRIC", u32::from(matrix == Some(MetalMatrix::Symmetric))),
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
