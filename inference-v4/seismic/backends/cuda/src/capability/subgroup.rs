//! `cuda.subgroup` registry inventory.
//!
//! Authored calls have one construction path: the compiler-owned semantic
//! walker invokes `Cuda::lower_semantic_intrinsic`. This module therefore
//! records only the signatures the native emitter implements; it does not
//! expose a second `TypedIntrinsic` lowering path.

use super::{signature_id, SUBGROUP};
use seismic_lang::ids::IntrinsicId;
use seismic_lang::registry::OperandCategory;
use seismic_lang::types::DType;

/// The subgroup width assumed by CUDA shuffle and butterfly emission.
pub const WARP_LANES: u32 = 32;

const LANE_INDEX: &str = "lane_index";
const SHUFFLE: &str = "shuffle";
const SIMD_SUM: &str = "simd_sum";
const SIMD_MAX: &str = "simd_max";
const SIMD_MIN: &str = "simd_min";

/// The dtypes with native subgroup emission.
pub const ELEMENT_DTYPES: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

/// Every signature implemented by the CUDA subgroup emitter. Registry
/// lookup is exact and an absent row is a static registry inconsistency.
pub fn implemented() -> Vec<(Option<DType>, IntrinsicId)> {
    let mut out = vec![(None, signature_id(SUBGROUP, LANE_INDEX, &[]))];
    for dtype in ELEMENT_DTYPES {
        out.push((
            Some(dtype),
            signature_id(
                SUBGROUP,
                SHUFFLE,
                &[
                    OperandCategory::Scalar(dtype),
                    OperandCategory::Scalar(DType::I32),
                ],
            ),
        ));
        for name in [SIMD_SUM, SIMD_MAX, SIMD_MIN] {
            out.push((
                Some(dtype),
                signature_id(SUBGROUP, name, &[OperandCategory::Scalar(dtype)]),
            ));
        }
    }
    out
}
