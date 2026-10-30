//! Consumer-form admission. Which of a backend's consumer forms a packed
//! row-layout weight takes is decided here, once, from its decode axes
//! (`registry::DecodeAxes`, with its code and local-code domains) and its
//! layout, and emitted as `<prefix>_FORM_<NAME>`: 0 or 1, and the values an
//! admitted form reads its codes with. Native libraries select on these and
//! assert they implement what is admitted, so a weight never falls back to
//! another form by its decoder's shape; host planning reads the same
//! admission (`metal_packing_of`).

use super::Dialect;
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry;

/// The forms admitted for a packed row-layout weight under `dialect`, in
/// emission order, with their values.
pub(super) fn forms(
    dialect: Dialect,
    axes: &registry::DecodeAxes,
    rows: &registry::PackedRowLayout,
) -> Vec<(&'static str, u32)> {
    match dialect {
        Dialect::Cuda(_) => vec![("GEMM_FOLD", u32::from(cuda_gemm_fold(axes, rows)))],
        Dialect::Metal(_) => metal_forms(axes),
        #[cfg(any(not(target_os = "macos"), test, feature = "coverage"))]
        Dialect::Vulkan(_) => Vec::new(),
    }
}

/// CUDA's scaled GEMM, which multiplies exact codes and applies each group's
/// coefficient to its F32 partial, serves grid codes and direct coefficients
/// below 8-bit codes, except whole 256-code rows (iq4g32, whose dequantized
/// GEMM its Qwen results were qualified with). Today's forms' choice, which
/// tuning owns once both GEMMs are qualified for each.
fn cuda_gemm_fold(axes: &registry::DecodeAxes, rows: &registry::PackedRowLayout) -> bool {
    match (&axes.code, &axes.coefficients) {
        (registry::CodeInterpretation::Grid(_), _) => true,
        (_, registry::CoefficientAxes::Direct { .. }) => axes.low != 8 && rows.group() % 256 != 0,
        (_, registry::CoefficientAxes::Hierarchical { .. }) => false,
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
fn metal_packet_groups(coefficients: &registry::CoefficientAxes) -> u32 {
    let group = match coefficients {
        registry::CoefficientAxes::Direct { group, .. }
        | registry::CoefficientAxes::Hierarchical { group, .. } => *group,
    };
    if group < METAL_PACKET { METAL_PACKET / group } else { 1 }
}

/// The zero code of codes Metal reads as `packets::Offset<ZERO>` (`Unsigned`
/// is `Offset<0>`); `None` for two's complement, codebook and grid codes.
fn metal_offset(code: &registry::CodeInterpretation) -> Option<i32> {
    match code {
        registry::CodeInterpretation::Unsigned => Some(0),
        registry::CodeInterpretation::Offset(zero) => Some(*zero),
        _ => None,
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

/// The weights the token-packing GEMM serves (`packing_layout`,
/// `packing_factors`): integer codes, not scaled by a power of two of their
/// own, as 4-, 5- or 8-bit entries in runs of 32 columns or 6-bit entries in
/// runs of 16, where plain codes enter as themselves and a codebook's 4-bit
/// codes as its values of 8 bits + 128 (`codebooks::pairs`); hierarchical
/// coefficients with a factor per 256 columns over local codes of the
/// scales plane, or direct ones of whole groups in 256 columns; and a block
/// of two runs without a min term, (centre - offset) scale + bias.
fn metal_packing(axes: &registry::DecodeAxes) -> Option<MetalPacking> {
    if axes.code_exponent != 0 {
        return None;
    }
    let codes = axes.codes();
    let (bits, offset) = match &axes.code {
        registry::CodeInterpretation::Unsigned
        | registry::CodeInterpretation::Offset(_)
        | registry::CodeInterpretation::TwosComplement => {
            (axes.low + axes.high, u32::try_from(-codes[0]).ok()?)
        }
        registry::CodeInterpretation::Table(_) => {
            let bytes = i32::from(i8::MIN) <= codes[0] && codes[codes.len() - 1] <= i32::from(i8::MAX);
            ((axes.low, axes.high) == (4, 0) && bytes).then_some((8, 128))?
        }
        registry::CodeInterpretation::Odd | registry::CodeInterpretation::Grid(_) => return None,
    };
    let folds = metal_packet_groups(&axes.coefficients);
    if !matches!((bits, folds), (4 | 5 | 8, 1) | (6, 2)) {
        return None;
    }
    let (bias, factors) = match &axes.coefficients {
        registry::CoefficientAxes::Direct { bias, group, .. } => (*bias, 256 % group == 0),
        registry::CoefficientAxes::Hierarchical {
            factor_group,
            locals: registry::LocalCodes::Plane { minimum, .. },
            ..
        } => (minimum.is_some(), *factor_group == 256),
        registry::CoefficientAxes::Hierarchical {
            locals: registry::LocalCodes::Carried { .. },
            ..
        } => return None,
    };
    let centre = 1u32 << (bits - 1);
    let biased = bias || centre != offset;
    (factors && (folds == 1 || !biased)).then_some(MetalPacking {
        folds,
        centre,
        offset,
        residue: bits == 8,
    })
}

/// The weights Metal's int8 GEMM (`int8_codes`, on tensor operations)
/// serves with their stored codes as its operands: unsigned 4-bit codes
/// under hierarchical coefficients, an unsigned local scale code of at most
/// 8 bits per 32 columns under a factor of at most 16 bits per 256 (so their
/// product is exact in F32), or two's complement 8-bit codes under a direct
/// scale per 32 columns without a bias.
fn metal_int8(axes: &registry::DecodeAxes) -> bool {
    if axes.code_exponent != 0 {
        return false;
    }
    match (&axes.coefficients, axes.low, axes.high) {
        (
            registry::CoefficientAxes::Hierarchical {
                factor,
                factor_group: 256,
                group: 32,
                locals:
                    registry::LocalCodes::Plane {
                        bits,
                        interpretation: registry::CodeInterpretation::Unsigned,
                        ..
                    },
                ..
            },
            4,
            0,
        ) => {
            let narrow = match factor {
                registry::ScaleEncoding::F16
                | registry::ScaleEncoding::BF16
                | registry::ScaleEncoding::E8M0
                | registry::ScaleEncoding::UE4M3 => true,
                registry::ScaleEncoding::F32 => false,
            };
            metal_offset(&axes.code) == Some(0) && *bits <= 8 && narrow
        }
        (registry::CoefficientAxes::Direct { bias: false, group: 32, .. }, 8, 0) => {
            axes.code == registry::CodeInterpretation::TwosComplement
        }
        _ => false,
    }
}

/// The products of Metal's matrix GEMV (`matrix_form`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetalMatrix {
    /// Unsigned codes of a plane of low nibbles (and a plane of high bits)
    /// under hierarchical coefficients with a minimum, an unsigned local
    /// scale code per 32 columns and factors per 256, where a code times a
    /// scale code stays below 2048, an exact F16 integer of the operand's
    /// 2^-24 (`matrix_codes`).
    Codes,
    /// Codes of a plane of low nibbles and a plane of two high bits under
    /// hierarchical coefficients without a minimum, a local scale code per
    /// 16 columns and factors per 256 (`matrix_runs`).
    Runs,
    /// Codes of a plane of low nibbles under a direct scale per 32 columns
    /// without a bias (`matrix_symmetric`).
    Symmetric,
}

/// The matrix GEMV product serving a weight, of codes Metal reads as
/// `packets::Offset` that are not scaled by a power of two of their own.
fn metal_matrix(axes: &registry::DecodeAxes) -> Option<MetalMatrix> {
    let zero = metal_offset(&axes.code)?;
    if axes.code_exponent != 0 {
        return None;
    }
    let greatest = |codes: Vec<i32>| codes[codes.len() - 1];
    match (&axes.coefficients, axes.low, axes.high) {
        (
            registry::CoefficientAxes::Hierarchical {
                factor_group: 256,
                group: 32,
                locals:
                    registry::LocalCodes::Plane {
                        interpretation: registry::CodeInterpretation::Unsigned,
                        minimum: Some(_),
                        ..
                    },
                ..
            },
            4,
            0 | 1,
        ) if zero == 0 => {
            let locals = greatest(axes.coefficients.local_codes()?);
            (greatest(axes.codes()) * locals < 2048).then_some(MetalMatrix::Codes)
        }
        (
            registry::CoefficientAxes::Hierarchical {
                factor_group: 256,
                group: 16,
                locals: registry::LocalCodes::Plane { minimum: None, .. },
                ..
            },
            4,
            2,
        ) => Some(MetalMatrix::Runs),
        (registry::CoefficientAxes::Direct { bias: false, group: 32, .. }, 4, 0) => {
            Some(MetalMatrix::Symmetric)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metal's admissions of every representation it imports, as main's
    /// libraries matched their decoders' shapes: PACK (folds, centre,
    /// offset, residue), INT8 and the matrix GEMV product.
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
