//! `cuda.matrix` registry inventory.
//!
//! Matrix calls are constructed only by the compiler-owned semantic walker.
//! This module names the exact registry rows backed by native CUDA emission;
//! there is no parallel backend `TypedIntrinsic` lowering.

use super::{capability_id, MATRIX};
use seismic_lang::ids::IntrinsicId;
use seismic_lang::registry;

/// Every matrix signature implemented by the native emitter.
pub fn implemented() -> Vec<IntrinsicId> {
    registry::intrinsics(capability_id(MATRIX))
        .iter()
        .map(|signature| signature.id)
        .collect()
}
