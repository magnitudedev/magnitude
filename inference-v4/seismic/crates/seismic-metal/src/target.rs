//! Effective Metal target profile: the exact intersection of observed
//! device/compiler support with the source intrinsic registry and the
//! signatures this backend can actually emit, plus the fully filled
//! `TargetLimits` of the realization layer.
//!
//! A signature is registered only when typing, reference meaning, detection,
//! legalization, resources, numerics, and emission are complete:
//! `metal.subgroup.{lane_index,shuffle,simd_sum,simd_max,simd_min}` and the
//! logical `metal.matrix.{matmul,matmul_add}` entries whose dtypes survived
//! the native compile probes are registered.
//!
//! Metal has no cooperative-grid facility: `cooperative_grid` is `None`, so
//! a grid-cooperative proposal is never made on this target — never a
//! fallback.

use seismic_lang::{
    intrinsics::{self, CapabilitySignature, IntrinsicId},
    sir::IntrinsicUse,
    types::{DType, Elem, ValueType},
};
use seismic_realization::target::{CooperativeGrid, EffectiveTargetProfile, TargetLimits};
use std::collections::BTreeSet;

pub const BACKEND_IMPLEMENTATION_REVISION: &str = "seismic-metal-realizations-v4";
pub const COMPILER_PROBE_REVISION: &str = "seismic-metal-compile-probes-v1";
/// Identity of this backend's cost model: the probe-calibrated estimate
/// model, carried with its provenance. Uncalibrated costs affect ranking
/// only, never legality.
pub const COST_MODEL_IDENTITY: &str = crate::estimate::IDENTITY;

pub const TARGET: &str = "metal";
/// Metal's per-axis workgroup-count limit (a driver constant below the
/// documented 2^32-1 so geometry arithmetic stays comfortable in u32).
pub const MAX_GROUPS: u64 = 65_535;
/// The fixed lane topology of every subgroup collective this backend emits.
pub const SUBGROUP: u32 = crate::intrinsics::SUBGROUP_WIDTH;
/// The argument-table limit of one Metal compute pipeline (31 buffers).
pub const MAX_KERNEL_BUFFERS: u32 = 31;
/// Metal exposes no private-stack limit; a conservative compiler budget.
pub const CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES: u64 = 128 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixCombination {
    pub accumulator: DType,
    pub left: DType,
    pub right: DType,
}

/// Hard limits the Metal target offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_bytes: u64,
    pub max_private_bytes: u64,
    pub max_device_bytes: u64,
}

impl Limits {
    #[cfg(target_os = "macos")]
    pub fn from_device(device: &crate::runtime::DeviceInfo) -> Self {
        Self {
            max_threads_per_threadgroup: device.max_threads_per_threadgroup,
            max_threadgroup_bytes: device.max_threadgroup_bytes,
            max_private_bytes: device.profile.private_storage_budget_bytes.value,
            max_device_bytes: device.max_buffer_bytes,
        }
    }

    pub fn synthetic() -> Self {
        Self {
            max_threads_per_threadgroup: 1024,
            max_threadgroup_bytes: 32 * 1024,
            max_private_bytes: CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES,
            max_device_bytes: u64::MAX,
        }
    }
}

/// The exact effective signature set of one observed device: registry
/// signatures this backend registers (collective and matrix dtypes admitted
/// by native compile probes).
#[derive(Clone, Debug, PartialEq)]
pub struct TargetProfile {
    signatures: Vec<CapabilitySignature>,
    fingerprint: String,
    limits: Limits,
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
        // metal.matrix entries whose dtypes survived it.
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
            "seismic-metal-target-v4;observation={observation};probe={COMPILER_PROBE_REVISION};\
             registry={};backend={BACKEND_IMPLEMENTATION_REVISION};cost={COST_MODEL_IDENTITY};\
             threads={max_threads};threadgroup={max_threadgroup_bytes};buffer={max_buffer_bytes};\
             private={private_budget_bytes};signatures={identity}",
            intrinsics::REGISTRY_REVISION,
        );
        Self {
            signatures,
            fingerprint,
            limits: Limits {
                max_threads_per_threadgroup: max_threads,
                max_threadgroup_bytes,
                max_private_bytes: private_budget_bytes,
                max_device_bytes: max_buffer_bytes,
            },
        }
    }

    pub fn synthetic(limits: Limits) -> Self {
        Self::from_evidence(
            "synthetic-metal-baseline",
            &[DType::F16, DType::F32],
            &[DType::F16, DType::F32],
            &[],
            limits.max_threads_per_threadgroup,
            limits.max_threadgroup_bytes,
            limits.max_device_bytes,
            limits.max_private_bytes,
        )
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// The exact effective signature ids (planning capability coverage).
    pub fn effective_signatures(&self) -> BTreeSet<IntrinsicId> {
        self.signatures.iter().map(|s| s.id.clone()).collect()
    }

    /// Exact intrinsic admission: the use must match a registered
    /// signature's id and concrete argument/result types.
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

    /// The fully filled effective target profile of the realization layer.
    /// `cooperative_grid` is `None`: Metal has no grid-wide barrier
    /// facility, so no grid-cooperative proposal is ever made.
    pub fn effective_profile(&self) -> EffectiveTargetProfile {
        let limits = self.limits;
        if limits.max_threads_per_threadgroup < u64::from(SUBGROUP) {
            panic!(
                "compiler defect ({:?}): Metal needs at least {SUBGROUP} threads per \
                 threadgroup; the target offers {}",
                seismic_realization::failure::Package::B1Metal,
                limits.max_threads_per_threadgroup
            );
        }
        EffectiveTargetProfile {
            backend: TARGET.into(),
            capability_fingerprint: self.fingerprint.clone(),
            toolchain_fingerprint: format!(
                "{};{}",
                BACKEND_IMPLEMENTATION_REVISION, COMPILER_PROBE_REVISION
            ),
            effective_signatures: self.effective_signatures(),
            limits: TargetLimits {
                max_participants: limits.max_threads_per_threadgroup,
                max_workgroups_axis: [MAX_GROUPS; 3],
                max_workgroup_bytes: limits.max_threadgroup_bytes,
                max_explicit_private_bytes: limits.max_private_bytes,
                max_direct_bindings: MAX_KERNEL_BUFFERS,
                max_device_bytes: limits.max_device_bytes,
                cooperative_grid: None::<CooperativeGrid>,
            },
        }
    }
}

fn argument_type_name(ty: &ValueType) -> String {
    match ty {
        ValueType::Scalar(d) => d.name().into(),
        ValueType::Index { .. } => "i32".into(),
        ValueType::Range { .. } => "range".into(),
        ValueType::Tensor(s) => match &s.elem {
            Elem::Dtype(d) => format!("tensor2<{}>", d.name()),
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
                || (matches!(pattern_tensor.elem, Elem::Param(_))
                    && pattern_tensor.rank() == concrete_tensor.rank()
                    && matches!(concrete_tensor.elem, Elem::Dtype(_)))
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
        let f16 = Elem::Dtype(DType::F16);
        let f32 = Elem::Dtype(DType::F32);
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
                    inner(Elem::Dtype(DType::BF16)),
                    columns(Elem::Dtype(DType::BF16))
                ],
                matrix(f32.clone()),
            ))
            .unwrap_err()
            .contains("does not support exact intrinsic signature"));
    }

    #[test]
    fn the_effective_profile_fills_every_limit_and_has_no_cooperative_grid() {
        let profile = TargetProfile::synthetic(Limits::synthetic());
        let effective = profile.effective_profile();
        assert_eq!(effective.backend, "metal");
        assert_eq!(effective.limits.max_participants, 1024);
        assert_eq!(effective.limits.max_workgroups_axis, [MAX_GROUPS; 3]);
        assert_eq!(effective.limits.max_workgroup_bytes, 32 * 1024);
        assert_eq!(
            effective.limits.max_explicit_private_bytes,
            CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES
        );
        assert_eq!(effective.limits.max_direct_bindings, MAX_KERNEL_BUFFERS);
        assert_eq!(effective.limits.max_device_bytes, u64::MAX);
        assert!(effective.limits.cooperative_grid.is_none());
    }
}
