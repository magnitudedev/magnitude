//! CUDA target observation and PTX target selection.
//!
//! Source programs never reason about CUDA versions. This module turns the facts reported by
//! the installed CUDA driver into the most conservative PTX target that implements the selected
//! backend operation. The scalar backend currently implements one such operation set: the PTX
//! 7.0, SM 8.0 baseline. Newer family- and architecture-specific targets are represented here but
//! are not claimed until their instruction sets and driver requirements are implemented.

use std::fmt;

use seismic_lang::intrinsics::Operation;
use seismic_lang::sir::IntrinsicUse;
use seismic_lang::types::{DType, Ty};

/// Revision of the CUDA realization contract, independent of the source intrinsic registry.
pub const BACKEND_IMPLEMENTATION_REVISION: &str = "seismic-cuda-realization-v2";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TargetLimits {
    pub max_threads_per_block: u32,
    pub max_grid_x: u32,
    pub warp_size: u32,
    pub max_scratch_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SupportedIntrinsic {
    LaneIndexI32,
    ShuffleF32I32ToF32,
    SimdSumF32,
}

impl SupportedIntrinsic {
    const CURRENT: &'static [Self] = &[
        Self::LaneIndexI32,
        Self::ShuffleF32I32ToF32,
        Self::SimdSumF32,
    ];

    fn identity(self) -> &'static str {
        match self {
            Self::LaneIndexI32 => "cuda.subgroup.lane_index:()->i32",
            Self::ShuffleF32I32ToF32 => "cuda.subgroup.shuffle:(f32,i32)->f32",
            Self::SimdSumF32 => "cuda.subgroup.simd_sum:(f32)->f32",
        }
    }

    fn matches(self, use_: &IntrinsicUse) -> bool {
        let f32 = Ty::Scalar(DType::F32);
        let i32 = Ty::Scalar(DType::I32);
        match self {
            Self::LaneIndexI32 => {
                use_.operation == Operation::LaneIndex
                    && use_.id.path() == "cuda.subgroup.lane_index"
                    && use_.arguments.is_empty()
                    && use_.result == i32
            }
            Self::ShuffleF32I32ToF32 => {
                use_.operation == Operation::ShuffleIndex
                    && use_.id.path() == "cuda.subgroup.shuffle"
                    && use_.arguments == [f32.clone(), i32]
                    && use_.result == f32
            }
            Self::SimdSumF32 => {
                use_.operation == Operation::SimdSum
                    && use_.id.path() == "cuda.subgroup.simd_sum"
                    && use_.arguments == [f32.clone()]
                    && use_.result == f32
            }
        }
    }
}

/// Where a target fact came from. Synthetic facts are useful for deterministic selection tests;
/// production devices are always observed through the CUDA Driver API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FactSource {
    DriverApi,
    Synthetic,
}

/// A CUDA compute capability, kept separate from PTX target spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    pub major: u16,
    pub minor: u16,
}

impl ComputeCapability {
    pub const SM80: Self = Self { major: 8, minor: 0 };

    pub fn new(major: i32, minor: i32) -> Result<Self, TargetError> {
        let major = u16::try_from(major).map_err(|_| TargetError::InvalidObservation {
            reason: format!("negative CUDA compute capability major {major}"),
        })?;
        let minor = u16::try_from(minor).map_err(|_| TargetError::InvalidObservation {
            reason: format!("negative CUDA compute capability minor {minor}"),
        })?;
        if major == 0 || minor > 9 {
            return Err(TargetError::InvalidObservation {
                reason: format!("invalid CUDA compute capability {major}.{minor}"),
            });
        }
        Ok(Self { major, minor })
    }

    fn ptx_architecture(self) -> Result<u16, TargetError> {
        self.major
            .checked_mul(10)
            .and_then(|major| major.checked_add(self.minor))
            .ok_or_else(|| TargetError::InvalidObservation {
                reason: format!("CUDA compute capability {self} cannot be encoded as an SM target"),
            })
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// The integer returned by `cuDriverGetVersion`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DriverApiVersion(pub u32);

impl DriverApiVersion {
    pub fn new(raw: i32) -> Result<Self, TargetError> {
        let raw = u32::try_from(raw).map_err(|_| TargetError::InvalidObservation {
            reason: format!("negative CUDA driver API version {raw}"),
        })?;
        if raw < 1000 {
            return Err(TargetError::InvalidObservation {
                reason: format!("invalid CUDA driver API version {raw}"),
            });
        }
        Ok(Self(raw))
    }

    pub fn major(self) -> u32 {
        self.0 / 1000
    }

    pub fn minor(self) -> u32 {
        (self.0 % 1000) / 10
    }
}

impl fmt::Display for DriverApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major(), self.minor())
    }
}

/// Target-relevant facts. Quantitative launch and memory limits remain in `mapping::Limits`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TargetObservation {
    pub compute_capability: ComputeCapability,
    pub driver_api: DriverApiVersion,
    pub source: FactSource,
}

impl TargetObservation {
    pub fn driver(compute_capability: (i32, i32), driver_api: i32) -> Result<Self, TargetError> {
        Ok(Self {
            compute_capability: ComputeCapability::new(compute_capability.0, compute_capability.1)?,
            driver_api: DriverApiVersion::new(driver_api)?,
            source: FactSource::DriverApi,
        })
    }

    pub fn synthetic(compute_capability: (i32, i32), driver_api: i32) -> Result<Self, TargetError> {
        Ok(Self {
            compute_capability: ComputeCapability::new(compute_capability.0, compute_capability.1)?,
            driver_api: DriverApiVersion::new(driver_api)?,
            source: FactSource::Synthetic,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PtxVersion {
    pub major: u16,
    pub minor: u16,
}

impl PtxVersion {
    pub const V7_0: Self = Self { major: 7, minor: 0 };
}

impl fmt::Display for PtxVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// NVIDIA distinguishes forward-compatible baseline targets from family- and exact-
/// architecture-specific targets. The latter two are modeled now so future intrinsic families
/// cannot accidentally be emitted as if they were baseline-compatible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetTier {
    Baseline,
    Family,
    Architecture,
}

impl TargetTier {
    fn suffix(self) -> &'static str {
        match self {
            Self::Baseline => "",
            Self::Family => "f",
            Self::Architecture => "a",
        }
    }
}

/// A complete PTX virtual-ISA target retained by terminal code, native images, and caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PtxTarget {
    pub isa: PtxVersion,
    /// Numeric SM spelling: 80 is `sm_80`, 121 is `sm_121`.
    pub architecture: u16,
    pub tier: TargetTier,
}

impl PtxTarget {
    pub const SCALAR_BASELINE: Self = Self {
        isa: PtxVersion::V7_0,
        architecture: 80,
        tier: TargetTier::Baseline,
    };

    pub fn architecture_spelling(self) -> String {
        format!("sm_{}{}", self.architecture, self.tier.suffix())
    }

    pub fn header(self) -> String {
        format!(
            ".version {}\n.target {}\n.address_size 64\n\n",
            self.isa,
            self.architecture_spelling()
        )
    }
}

/// The backend operation set for which a PTX target is requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetRequirement {
    /// Current scalar realization, including its 32-lane subgroup helpers.
    ScalarBaseline,
    /// Reserved for operations whose PTX form is shared by one architecture family.
    FamilySpecific,
    /// Reserved for operations restricted to one exact compute capability.
    ArchitectureSpecific,
}

/// Normalized target capability, independent of device marketing names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetProfile {
    observation: TargetObservation,
    limits: TargetLimits,
    scalar_target: PtxTarget,
    supported_intrinsics: Vec<SupportedIntrinsic>,
    fingerprint: String,
}

impl TargetProfile {
    /// PTX 7.0 was introduced with CUDA 11.0. The scalar emitter deliberately uses the SM 8.0
    /// baseline on newer devices because it does not yet emit any newer ISA operation.
    const PTX_70_MINIMUM_DRIVER: DriverApiVersion = DriverApiVersion(11_000);

    pub fn from_observation(
        observation: TargetObservation,
        limits: TargetLimits,
    ) -> Result<Self, TargetError> {
        if observation.compute_capability < ComputeCapability::SM80 {
            return Err(TargetError::UnsupportedHardware {
                observed: observation.compute_capability,
                required: ComputeCapability::SM80,
                operation_set: "CUDA scalar baseline",
            });
        }
        if observation.driver_api < Self::PTX_70_MINIMUM_DRIVER {
            return Err(TargetError::UnsupportedDriver {
                observed: observation.driver_api,
                required: Self::PTX_70_MINIMUM_DRIVER,
                ptx: PtxVersion::V7_0,
            });
        }
        if limits.max_threads_per_block == 0
            || limits.max_grid_x == 0
            || limits.warp_size == 0
            || limits.max_scratch_bytes == 0
        {
            return Err(TargetError::InvalidObservation {
                reason: "CUDA target limits must be positive".into(),
            });
        }
        Ok(Self::build(
            observation,
            limits,
            SupportedIntrinsic::CURRENT.to_vec(),
            seismic_lang::intrinsics::REGISTRY_REVISION,
            BACKEND_IMPLEMENTATION_REVISION,
        ))
    }

    fn build(
        observation: TargetObservation,
        limits: TargetLimits,
        mut supported_intrinsics: Vec<SupportedIntrinsic>,
        registry_revision: &str,
        backend_revision: &str,
    ) -> Self {
        let scalar_target = PtxTarget::SCALAR_BASELINE;
        supported_intrinsics.sort_unstable_by_key(|signature| signature.identity());
        supported_intrinsics.dedup();
        let signatures = supported_intrinsics
            .iter()
            .map(|signature| signature.identity())
            .collect::<Vec<_>>()
            .join(",");
        let fingerprint = format!(
            "seismic-cuda-target-v2:registry={registry_revision}:backend={backend_revision}:cc={}:driver-api={}:codegen=ptx{}-{}-{:?}:limits=threads{},grid{},warp{},scratch{}:intrinsics=[{signatures}]",
            observation.compute_capability,
            observation.driver_api.0,
            scalar_target.isa,
            scalar_target.architecture_spelling(),
            scalar_target.tier,
            limits.max_threads_per_block,
            limits.max_grid_x,
            limits.warp_size,
            limits.max_scratch_bytes,
        );
        Self {
            observation,
            limits,
            scalar_target,
            supported_intrinsics,
            fingerprint,
        }
    }

    pub fn synthetic_baseline(limits: TargetLimits) -> Self {
        Self::from_observation(
            TargetObservation::synthetic((8, 0), 11_000)
                .expect("the built-in CUDA baseline observation is valid"),
            limits,
        )
        .expect("the built-in CUDA baseline profile is supported")
    }

    pub fn observation(&self) -> &TargetObservation {
        &self.observation
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn limits(&self) -> TargetLimits {
        self.limits
    }

    pub fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String> {
        if intrinsic.id.capability.backend != "cuda" {
            return Err(format!(
                "intrinsic `{}` belongs to backend `{}`, not `cuda`",
                intrinsic.id.path(),
                intrinsic.id.capability.backend
            ));
        }
        if matches!(
            intrinsic.operation,
            Operation::MatrixMatmul | Operation::MatrixMatmulAdd
        ) {
            return Err(format!(
                "BackendNotImplemented: logical CUDA matrix intrinsic `{}` has a preserved owned result ABI, but no CUDA realization and numerical contract",
                intrinsic.id.path()
            ));
        }
        self.supported_intrinsics
            .iter()
            .copied()
            .any(|signature| signature.matches(intrinsic))
            .then_some(())
            .ok_or_else(|| {
                format!(
                    "the CUDA target profile has no implemented realization for exact intrinsic use `{}` with arguments {:?} and result {}",
                    intrinsic.id.path(), intrinsic.arguments, intrinsic.result
                )
            })
    }

    pub fn plan(&self, requirement: TargetRequirement) -> Result<PtxTarget, TargetError> {
        match requirement {
            TargetRequirement::ScalarBaseline => Ok(self.scalar_target),
            TargetRequirement::FamilySpecific => Err(TargetError::BackendNotImplemented {
                tier: TargetTier::Family,
                observed: self.observation.compute_capability,
            }),
            TargetRequirement::ArchitectureSpecific => Err(TargetError::BackendNotImplemented {
                tier: TargetTier::Architecture,
                observed: self.observation.compute_capability,
            }),
        }
    }

    /// Validate retained target code before handing it to the installed driver JIT.
    pub fn admits(&self, target: PtxTarget) -> Result<(), TargetError> {
        if target == self.scalar_target {
            return Ok(());
        }
        match target.tier {
            TargetTier::Baseline => Err(TargetError::BackendNotImplemented {
                tier: TargetTier::Baseline,
                observed: self.observation.compute_capability,
            }),
            TargetTier::Family | TargetTier::Architecture => {
                Err(TargetError::BackendNotImplemented {
                    tier: target.tier,
                    observed: self.observation.compute_capability,
                })
            }
        }
    }

    pub fn observed_architecture(&self) -> Result<u16, TargetError> {
        self.observation.compute_capability.ptx_architecture()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetError {
    InvalidObservation {
        reason: String,
    },
    UnsupportedHardware {
        observed: ComputeCapability,
        required: ComputeCapability,
        operation_set: &'static str,
    },
    UnsupportedDriver {
        observed: DriverApiVersion,
        required: DriverApiVersion,
        ptx: PtxVersion,
    },
    BackendNotImplemented {
        tier: TargetTier,
        observed: ComputeCapability,
    },
}

impl fmt::Display for TargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidObservation { reason } => f.write_str(reason),
            Self::UnsupportedHardware {
                observed,
                required,
                operation_set,
            } => write!(
                f,
                "{operation_set} requires compute capability {required} or newer; device reports {observed}"
            ),
            Self::UnsupportedDriver {
                observed,
                required,
                ptx,
            } => write!(
                f,
                "PTX {ptx} requires CUDA driver API {required} or newer; installed driver reports {observed}"
            ),
            Self::BackendNotImplemented { tier, observed } => write!(
                f,
                "Seismic CUDA does not yet implement {tier:?} PTX emission for compute capability {observed}"
            ),
        }
    }
}

impl std::error::Error for TargetError {}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: TargetLimits = TargetLimits {
        max_threads_per_block: 1024,
        max_grid_x: i32::MAX as u32,
        warp_size: 32,
        max_scratch_bytes: 1 << 30,
    };

    fn profile(cc: (i32, i32), driver: i32) -> Result<TargetProfile, TargetError> {
        TargetProfile::from_observation(TargetObservation::synthetic(cc, driver)?, LIMITS)
    }

    #[test]
    fn scalar_baseline_is_selected_from_oldest_supported_matrix() {
        let profile = profile((8, 0), 11_000).unwrap();
        assert_eq!(
            profile.plan(TargetRequirement::ScalarBaseline),
            Ok(PtxTarget::SCALAR_BASELINE)
        );
        assert_eq!(
            PtxTarget::SCALAR_BASELINE.header(),
            ".version 7.0\n.target sm_80\n.address_size 64\n\n"
        );
    }

    #[test]
    fn newer_hardware_retains_the_implemented_forward_compatible_baseline() {
        let profile = profile((12, 1), 13_000).unwrap();
        assert_eq!(profile.observed_architecture(), Ok(121));
        assert_eq!(
            profile.plan(TargetRequirement::ScalarBaseline),
            Ok(PtxTarget::SCALAR_BASELINE)
        );
        assert!(profile.fingerprint().contains("cc=12.1"));
        assert!(profile.fingerprint().contains("driver-api=13000"));
    }

    #[test]
    fn supported_hardware_driver_matrix_has_one_honest_emission_target() {
        for (cc, driver) in [
            ((8, 0), 11_000),
            ((8, 9), 12_000),
            ((9, 0), 12_000),
            ((10, 0), 12_080),
            ((12, 1), 13_000),
        ] {
            let profile = profile(cc, driver).unwrap();
            assert_eq!(
                profile.plan(TargetRequirement::ScalarBaseline),
                Ok(PtxTarget::SCALAR_BASELINE),
                "synthetic cc={}.{} driver={driver}",
                cc.0,
                cc.1
            );
        }
    }

    #[test]
    fn hardware_below_the_emitted_baseline_is_rejected() {
        assert!(matches!(
            profile((7, 5), 12_000),
            Err(TargetError::UnsupportedHardware { .. })
        ));
    }

    #[test]
    fn driver_too_old_for_the_emitted_ptx_is_rejected() {
        assert!(matches!(
            profile((8, 0), 10_020),
            Err(TargetError::UnsupportedDriver { .. })
        ));
    }

    #[test]
    fn invalid_observations_are_not_guessed() {
        assert!(matches!(
            TargetObservation::synthetic((-1, 0), 13_000),
            Err(TargetError::InvalidObservation { .. })
        ));
        assert!(matches!(
            TargetObservation::synthetic((8, 0), 0),
            Err(TargetError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn specialized_tiers_are_explicitly_unimplemented() {
        let profile = profile((12, 1), 13_000).unwrap();
        assert!(matches!(
            profile.plan(TargetRequirement::FamilySpecific),
            Err(TargetError::BackendNotImplemented {
                tier: TargetTier::Family,
                ..
            })
        ));
        assert!(matches!(
            profile.plan(TargetRequirement::ArchitectureSpecific),
            Err(TargetError::BackendNotImplemented {
                tier: TargetTier::Architecture,
                ..
            })
        ));
    }

    #[test]
    fn fingerprint_changes_with_every_availability_or_identity_input() {
        let observation = TargetObservation::synthetic((8, 0), 11_000).unwrap();
        let baseline = TargetProfile::build(
            observation.clone(),
            LIMITS,
            SupportedIntrinsic::CURRENT.to_vec(),
            "registry-a",
            "backend-a",
        );
        let variants = [
            TargetProfile::build(
                TargetObservation::synthetic((8, 9), 12_000).unwrap(),
                LIMITS,
                SupportedIntrinsic::CURRENT.to_vec(),
                "registry-a",
                "backend-a",
            ),
            TargetProfile::build(
                observation.clone(),
                TargetLimits {
                    max_scratch_bytes: LIMITS.max_scratch_bytes - 1,
                    ..LIMITS
                },
                SupportedIntrinsic::CURRENT.to_vec(),
                "registry-a",
                "backend-a",
            ),
            TargetProfile::build(
                observation.clone(),
                LIMITS,
                vec![SupportedIntrinsic::LaneIndexI32],
                "registry-a",
                "backend-a",
            ),
            TargetProfile::build(
                observation.clone(),
                LIMITS,
                SupportedIntrinsic::CURRENT.to_vec(),
                "registry-b",
                "backend-a",
            ),
            TargetProfile::build(
                observation,
                LIMITS,
                SupportedIntrinsic::CURRENT.to_vec(),
                "registry-a",
                "backend-b",
            ),
        ];
        for variant in variants {
            assert_ne!(baseline.fingerprint(), variant.fingerprint());
        }
    }
}
