//! Effective Metal target profile: the exact intersection of observed
//! device/compiler support with the source intrinsic registry and the
//! signatures this backend can actually emit.
//!
//! A signature is registered only when typing, reference meaning, detection,
//! legalization, resources, numerics, and emission are complete:
//! `metal.subgroup.{lane_index,shuffle,simd_sum,simd_max,simd_min}` are
//! registered. `metal.matrix` is observed by the device probes but is NOT
//! registered until fragment emission is complete; authored uses of an
//! unregistered signature are removed before planning with reasons, and the
//! portable bodies remain.

use seismic_lang::{
    intrinsics::{self, CapabilitySignature, IntrinsicId},
    sir::IntrinsicUse,
    types::{DType, Elem, ValueType},
};
use std::collections::BTreeSet;

pub const BACKEND_IMPLEMENTATION_REVISION: &str = "seismic-metal-realizations-v3";
pub const COMPILER_PROBE_REVISION: &str = "seismic-metal-compile-probes-v1";
/// Identity of this backend's cost model: the probe-calibrated estimate
/// model, carried with its provenance. Uncalibrated costs affect ranking
/// only, never legality.
pub const COST_MODEL_IDENTITY: &str = crate::mapping::estimate::IDENTITY;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixCombination {
    pub accumulator: DType,
    pub left: DType,
    pub right: DType,
}

/// The exact effective signature set of one observed device: registry
/// signatures this backend registers (subgroup collective dtypes admitted by
/// native compile probes). Matrix signatures stay absent until fragment
/// emission is complete.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetProfile {
    signatures: Vec<CapabilitySignature>,
    fingerprint: String,
}

impl TargetProfile {
    #[allow(clippy::too_many_arguments)]
    pub fn from_evidence(
        observation: &str,
        scalar_dtypes: &[DType],
        matrix_dtypes: &[DType],
        _matrix_combinations: &[MatrixCombination],
        max_threads: u64,
        max_threadgroup_bytes: u64,
        max_buffer_bytes: u64,
        private_budget_bytes: u64,
    ) -> Self {
        // Registered signatures: the registry's metal.subgroup entries whose
        // dtype survived the native compile probe, and the logical
        // metal.matrix matmul entries whose dtypes survived it.
        let mut signatures = Vec::new();
        for entry in intrinsics::capabilities() {
            if entry.id.capability.backend != "metal" {
                continue;
            }
            let admitted = match entry.id.capability.name.as_str() {
                "subgroup" => {
                    entry.arguments.iter().all(|argument| match argument {
                        ValueType::Scalar(dtype) => scalar_dtypes.contains(dtype),
                        _ => false,
                    }) && match &entry.result {
                        ValueType::Scalar(dtype) => scalar_dtypes.contains(dtype),
                        _ => false,
                    }
                }
                "matrix" => {
                    // The logical matmul signatures: every dtype the entry
                    // mentions must have survived the matrix probe.
                    let dtypes_of = |ty: &ValueType| match ty {
                        ValueType::Tensor(tensor) => match &tensor.elem {
                            Elem::Dtype(dtype) => Some(*dtype),
                            _ => None,
                        },
                        _ => None,
                    };
                    entry
                        .arguments
                        .iter()
                        .chain([&entry.result])
                        .filter_map(dtypes_of)
                        .all(|dtype| matrix_dtypes.contains(&dtype))
                }
                _ => continue,
            };
            if admitted {
                signatures.push(entry);
            }
        }
        signatures.sort_by(|a, b| a.id.path().cmp(&b.id.path()));
        signatures.dedup_by(|a, b| a.id == b.id);
        let identity = signatures
            .iter()
            .map(|signature| {
                format!(
                    "{}({})->{}",
                    signature.id.path(),
                    signature
                        .arguments
                        .iter()
                        .map(|argument| argument_type_name(argument))
                        .collect::<Vec<_>>()
                        .join(","),
                    argument_type_name(&signature.result)
                )
            })
            .collect::<Vec<_>>()
            .join("|");
        let fingerprint = format!(
            "seismic-metal-target-v3;observation={observation};probe={COMPILER_PROBE_REVISION};\
             registry={};backend={BACKEND_IMPLEMENTATION_REVISION};cost={COST_MODEL_IDENTITY};\
             threads={max_threads};threadgroup={max_threadgroup_bytes};buffer={max_buffer_bytes};\
             private={private_budget_bytes};signatures={identity}",
            intrinsics::REGISTRY_REVISION,
        );
        Self {
            signatures,
            fingerprint,
        }
    }

    pub fn synthetic(
        max_threads: u64,
        max_threadgroup_bytes: u64,
        max_buffer_bytes: u64,
        private_budget_bytes: u64,
    ) -> Self {
        Self::from_evidence(
            "synthetic-metal-baseline",
            &[DType::F16, DType::F32],
            &[DType::F16, DType::F32],
            &[],
            max_threads,
            max_threadgroup_bytes,
            max_buffer_bytes,
            private_budget_bytes,
        )
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// The exact effective signature ids (planning capability coverage).
    pub fn effective_signatures(&self) -> BTreeSet<IntrinsicId> {
        self.signatures.iter().map(|s| s.id.clone()).collect()
    }

    /// Exact intrinsic admission: the use must match a registered signature's
    /// id and concrete argument/result types.
    pub fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String> {
        if intrinsic.id.capability.backend != "metal" {
            return Err(format!(
                "intrinsic `{}` belongs to backend `{}`, not `metal`",
                intrinsic.id.path(),
                intrinsic.id.capability.backend
            ));
        }
        let registered = self.signatures.iter().any(|signature| {
            signature.id == intrinsic.id
                && signature.arguments.len() == intrinsic.arguments.len()
                && signature
                    .arguments
                    .iter()
                    .zip(&intrinsic.arguments)
                    .all(|(a, b)| concrete_matches(a, b))
                && concrete_matches(&signature.result, &intrinsic.result)
        });
        if registered {
            Ok(())
        } else {
            Err(format!(
                "effective Metal target profile does not support exact intrinsic signature \
                 `{}` (registered signatures: {})",
                intrinsic.id.path(),
                if self.signatures.is_empty() {
                    "none".into()
                } else {
                    self.signatures
                        .iter()
                        .map(|s| s.id.path())
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            ))
        }
    }
}

fn argument_type_name(ty: &ValueType) -> String {
    match ty {
        ValueType::Scalar(d) => d.name().into(),
        ValueType::Index { .. } => "i32".into(),
        ValueType::Range { .. } => "range".into(),
        ValueType::Tensor(s) => match &s.elem {
            seismic_lang::types::Elem::Dtype(d) => format!("tensor2<{}>", d.name()),
            _ => "tensor2<repr>".into(),
        },
        ValueType::Tuple(_) => "tuple".into(),
        ValueType::CapabilityValue(n) => format!("capability.{}.{}", n.target, n.name),
        ValueType::Void => "void".into(),
    }
}

/// Concrete type match against a registry signature type: scalars match
/// exactly; symbolic tensor patterns match by rank and element.
fn concrete_matches(pattern: &ValueType, concrete: &ValueType) -> bool {
    match (pattern, concrete) {
        (ValueType::Scalar(a), ValueType::Scalar(b)) => a == b,
        (ValueType::Tensor(pattern_tensor), ValueType::Tensor(concrete_tensor)) => {
            pattern_tensor.rank() == concrete_tensor.rank()
                && pattern_tensor.elem == concrete_tensor.elem
                || (matches!(pattern_tensor.elem, seismic_lang::types::Elem::Param(_))
                    && pattern_tensor.rank() == concrete_tensor.rank()
                    && matches!(concrete_tensor.elem, seismic_lang::types::Elem::Dtype(_)))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::intrinsics::CapabilityId;
    use seismic_lang::sym::Sym;
    use seismic_lang::types::{ExtentExpr, TensorType};

    fn intrinsic(
        family: &str,
        name: &str,
        arguments: Vec<ValueType>,
        result: ValueType,
    ) -> IntrinsicUse {
        IntrinsicUse {
            id: IntrinsicId {
                capability: CapabilityId::new("metal", family),
                name: name.into(),
            },
            arguments,
            result,
        }
    }

    #[test]
    fn fingerprint_tracks_exact_signature_intersection() {
        let base = TargetProfile::from_evidence(
            "msl=3.2@probe;families=metal3,apple9@query",
            &[DType::F16],
            &[DType::F16],
            &[MatrixCombination {
                accumulator: DType::F16,
                left: DType::F16,
                right: DType::F16,
            }],
            1024,
            32 * 1024,
            1 << 30,
            128 * 1024,
        );
        let wider = TargetProfile::from_evidence(
            "msl=3.2@probe;families=metal3,apple9@query",
            &[DType::F16, DType::F32],
            &[DType::F16],
            &[],
            1024,
            32 * 1024,
            1 << 30,
            128 * 1024,
        );
        assert_ne!(base.fingerprint(), wider.fingerprint());
        for identity in [
            intrinsics::REGISTRY_REVISION,
            BACKEND_IMPLEMENTATION_REVISION,
            COMPILER_PROBE_REVISION,
            COST_MODEL_IDENTITY,
        ] {
            assert!(base.fingerprint().contains(identity));
        }
        assert!(base.fingerprint().contains("signatures="));
    }

    #[test]
    fn admission_uses_the_exact_probed_signature_set() {
        let profile = TargetProfile::from_evidence(
            "synthetic-observation",
            &[DType::F32],
            &[],
            &[],
            1024,
            32 * 1024,
            1 << 30,
            128 * 1024,
        );
        assert!(profile
            .supports_intrinsic(&intrinsic(
                "subgroup",
                "simd_sum",
                vec![ValueType::Scalar(DType::F32)],
                ValueType::Scalar(DType::F32),
            ))
            .is_ok());
        assert!(profile
            .supports_intrinsic(&intrinsic(
                "subgroup",
                "simd_sum",
                vec![ValueType::Scalar(DType::BF16)],
                ValueType::Scalar(DType::BF16),
            ))
            .unwrap_err()
            .contains("does not support exact intrinsic signature"));
    }

    #[test]
    fn matrix_signatures_register_with_their_probed_dtypes() {
        let profile = TargetProfile::from_evidence(
            "synthetic-observation",
            &[DType::F32],
            &[DType::F16, DType::F32],
            &[MatrixCombination {
                accumulator: DType::F32,
                left: DType::F16,
                right: DType::F16,
            }],
            1024,
            32 * 1024,
            1 << 30,
            128 * 1024,
        );
        let matrix = |elem| {
            ValueType::Tensor(TensorType::new(
                vec![
                    ExtentExpr::Sym(Sym::param("rows")),
                    ExtentExpr::Sym(Sym::param("columns")),
                ],
                elem,
            ))
        };
        let inner = |elem| {
            ValueType::Tensor(TensorType::new(
                vec![
                    ExtentExpr::Sym(Sym::param("rows")),
                    ExtentExpr::Sym(Sym::param("inner")),
                ],
                elem,
            ))
        };
        let columns = |elem| {
            ValueType::Tensor(TensorType::new(
                vec![
                    ExtentExpr::Sym(Sym::param("inner")),
                    ExtentExpr::Sym(Sym::param("columns")),
                ],
                elem,
            ))
        };
        let f16 = seismic_lang::types::Elem::Dtype(DType::F16);
        let f32 = seismic_lang::types::Elem::Dtype(DType::F32);
        assert!(profile
            .supports_intrinsic(&intrinsic(
                "matrix",
                "matmul",
                vec![inner(f16.clone()), columns(f16.clone())],
                matrix(f32.clone()),
            ))
            .is_ok());
        // A dtype the probe did not admit stays unsupported.
        assert!(profile
            .supports_intrinsic(&intrinsic(
                "matrix",
                "matmul",
                vec![
                    inner(seismic_lang::types::Elem::Dtype(DType::BF16)),
                    columns(seismic_lang::types::Elem::Dtype(DType::BF16))
                ],
                matrix(f32.clone()),
            ))
            .unwrap_err()
            .contains("does not support exact intrinsic signature"));
    }
}
