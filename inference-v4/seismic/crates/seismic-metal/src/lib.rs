//! The Metal backend (spec §2.3, §4, §11, §12.3, §24.1 R9).
//!
//! Owns exactly: Metal target discovery and profile parts, the Metal
//! capability registrations (`metal.subgroup`, `metal.matrix`) with their
//! typed intrinsic lowerings and resource rules, the Metal structural
//! implementation factories, native compilation of core-owned closed
//! kernels, and the device service and executor the core-owned executable
//! schedule drives.
//!
//! There is no schedule mirror, fallback, retry, or source accommodation.
//! Device-wide legality lives in `MetalFacts`/`TargetLimits`; concrete
//! pipeline width, launch domain, and static threadgroup storage enter
//! planning only through authoritative native reflection. Every command the
//! executor runs is derived once from the frozen plan's `ExecutableStep`
//! tree.
//!
//! Metal exists on macOS only; the crate is empty elsewhere.
#![cfg(target_os = "macos")]

pub mod device;
pub mod direct;
pub mod executor;
pub mod factories;
pub mod facts;
pub mod intrinsic;
pub mod profile;

mod command;
mod compile;
mod render;
mod services;

use objc2_metal::MTLComputePipelineState;
use seismic_compiler::errors::NativeCompilationError;
use seismic_compiler::target::{
    Backend, DeviceContract, IntrinsicIdentityBuilder, NativeArtifactMetrics, NativeKernelIdentity,
    NativeKernelReflection, NativeKernelResources, NativeLaunchDomain, NativeNumericalModeIdentity,
    NativeResourceUsage,
};
use seismic_lang::registry::BackendName;
use sha2::{Digest, Sha256};
use std::time::Instant;

pub use command::Pipeline;
pub use compile::NativeCandidate;
pub use device::{DeviceHandle, MetalBuffer, MetalDevice};
pub use direct::{DirectBatch, DirectPipeline};
pub use executor::MetalExecutor;
pub use facts::MetalFacts;
pub use intrinsic::MetalIntrinsic;

/// Revision of this backend's implementation: every change to a lowering,
/// an emission rule, a factory, a resource rule, or an execution-service rule
/// changes this string and with it every cache identity (§15.1).
pub const BACKEND_REVISION: &str = "seismic-metal-v5";

/// The Metal backend.
#[derive(Debug)]
pub struct Metal;

/// Metal has only independent compute dispatches; cooperative mode is
/// unrepresentable in compiled commands for this backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MetalLaunchMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MetalNumericalMode;

impl Backend for Metal {
    const NAME: BackendName = BackendName::Metal;

    type Intrinsic = MetalIntrinsic;
    type Facts = MetalFacts;
    type KernelAbi = profile::MetalKernelAbi;
    type NativeLaunchMode = MetalLaunchMode;
    type RawNativeCandidate = NativeCandidate;
    type NativeKernelHandle = Pipeline;
    type NativeNumericalMode = MetalNumericalMode;
    type Executor = MetalExecutor;

    fn independent_launch_mode() -> Self::NativeLaunchMode {
        MetalLaunchMode
    }

    fn cooperative_launch_mode(_facts: &Self::Facts) -> Option<Self::NativeLaunchMode> {
        None
    }

    fn native_launch_constraints(
        _target: &DeviceContract<Self>,
        arena: &mut seismic_lang::expr::ExprArena,
        launch: &seismic_compiler::schedule::Launch,
        _locals: &seismic_compiler::storage::LaunchLocalLayout,
        kernel: &seismic_compiler::kernel::Kernel<Self>,
        native: &seismic_compiler::target::NativeKernel<Self>,
    ) -> Vec<seismic_lang::expr::BoolExpr> {
        fn contains_matrix(
            kernel: &seismic_compiler::kernel::Kernel<Metal>,
            block: seismic_compiler::kernel::BlockId,
        ) -> bool {
            kernel.block(block).ops.iter().any(|op| match op {
                seismic_compiler::kernel::ops::Op::Intrinsic { op, .. } => {
                    matches!(op, MetalIntrinsic::Matrix { .. })
                }
                seismic_compiler::kernel::ops::Op::Branch {
                    then, otherwise, ..
                } => contains_matrix(kernel, *then) || contains_matrix(kernel, *otherwise),
                seismic_compiler::kernel::ops::Op::Repeat { body, .. } => {
                    contains_matrix(kernel, *body)
                }
                _ => false,
            })
        }

        if !contains_matrix(kernel, kernel.root()) {
            return Vec::new();
        }
        let width =
            arena.nat(u64::from(native.contract().launch.subgroup_width.expect(
                "Metal matrix pipeline closed without a reflected SIMD width",
            )));
        let one = arena.nat(1);
        vec![
            arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, launch.workgroup[0], width),
            arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, launch.workgroup[1], one),
            arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, launch.workgroup[2], one),
            arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, launch.grid[1], one),
            arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, launch.grid[2], one),
        ]
    }

    fn write_intrinsic_identity(
        intrinsic: &MetalIntrinsic,
        identity: &mut IntrinsicIdentityBuilder,
    ) {
        intrinsic::write_identity(intrinsic, identity)
    }

    fn intrinsic_addressable_resources(
        _intrinsic: &Self::Intrinsic,
    ) -> Vec<seismic_compiler::kernel::ops::AddressableResourceHandle> {
        Vec::new()
    }

    fn emitted_intrinsics() -> std::collections::BTreeSet<seismic_lang::ids::IntrinsicId> {
        intrinsic::emitted_intrinsics()
    }

    fn required_service_classes(
        facts: &Self::Facts,
        supported_intrinsics: &std::collections::BTreeSet<seismic_lang::ids::IntrinsicId>,
    ) -> std::collections::BTreeSet<seismic_compiler::target::ServiceClassId> {
        services::required_service_classes(facts, supported_intrinsics)
    }

    fn execution_demand(
        target: &DeviceContract<Self>,
        arena: &mut seismic_lang::expr::ExprArena,
        kernel: &seismic_compiler::kernel::Kernel<Self>,
        emission: &seismic_compiler::target::KernelEmissionLayout,
        op: seismic_compiler::kernel::ops::ClosedOpView<'_, Self>,
    ) -> Vec<seismic_compiler::target::ExecutionDemand> {
        services::execution_demand(target, arena, kernel, emission, op)
    }

    fn refine_execution_demand(
        _target: &DeviceContract<Self>,
        arena: &mut seismic_lang::expr::ExprArena,
        _launch: &seismic_compiler::schedule::Launch,
        _locals: &seismic_compiler::storage::LaunchLocalLayout,
        native: &seismic_compiler::target::NativeKernel<Self>,
        mut demand: seismic_compiler::target::ExecutionDemand,
    ) -> Vec<seismic_compiler::target::ExecutionDemand> {
        // Matrix and its internal barriers are measured in lane-issues by
        // the fixed probe suite. Their structural demand is aggregate tile
        // work, so close that unit only with this pipeline's reflected SIMD
        // width. Ordinary per-participant demands are multiplied by core.
        if demand.scope == seismic_compiler::target::DemandScope::PerLaunch
            && matches!(demand.class, services::MATRIX | services::BARRIER)
        {
            let width = native
                .contract()
                .launch
                .subgroup_width
                .expect("Metal subgroup demand closed without reflected pipeline width");
            let width = arena.nat(u64::from(width));
            demand.units = arena.nat_mul(demand.units, width);
        }
        vec![demand]
    }

    fn intrinsic_numerics(
        _facts: &Self::Facts,
        signature: &seismic_lang::registry::IntrinsicSignature,
        intrinsic: &Self::Intrinsic,
    ) -> seismic_compiler::target::IntrinsicNumericalSemantics {
        intrinsic::numerical_semantics(signature, intrinsic)
    }

    fn semantic_intrinsic_requirements(
        target: &DeviceContract<Self>,
        arena: &mut seismic_lang::expr::ExprArena,
        signature: &seismic_lang::registry::IntrinsicSignature,
        parallel_extent: seismic_lang::expr::NatExpr,
    ) -> seismic_compiler::kernel::ops::SemanticIntrinsicLaunchRequirements {
        intrinsic::semantic_requirements(target, arena, signature, parallel_extent)
    }

    fn lower_semantic_intrinsic(
        target: &DeviceContract<Self>,
        domain: &seismic_compiler::kernel::ops::SegmentLaunchDomain,
        call: seismic_compiler::kernel::ops::SemanticIntrinsicCall<'_>,
        sink: &mut seismic_compiler::kernel::ops::SemanticIntrinsicSink<'_, '_, Self>,
    ) {
        intrinsic::lower_semantic(target, domain, call, sink)
    }

    fn form_native_kernel_candidate(
        target: &DeviceContract<Self>,
        kernel: &seismic_compiler::kernel::Kernel<Self>,
        layout: &seismic_compiler::target::KernelEmissionLayout,
    ) -> Result<Self::RawNativeCandidate, NativeCompilationError> {
        compile::compile_kernel(target, kernel, layout)
    }

    fn reflect_native_kernel(
        target: &DeviceContract<Self>,
        kernel: &seismic_compiler::kernel::Kernel<Self>,
        _layout: &seismic_compiler::target::KernelEmissionLayout,
        candidate: Self::RawNativeCandidate,
    ) -> Result<NativeKernelReflection<Self>, NativeCompilationError> {
        let reflection_started = Instant::now();
        let pipeline_threads = u64::try_from(
            candidate.pipeline.state.maxTotalThreadsPerThreadgroup(),
        )
        .map_err(|_| {
            NativeCompilationError::MalformedToolchainOutput(
                "Metal pipeline thread limit exceeds u64".into(),
            )
        })?;
        let static_local = u64::try_from(candidate.pipeline.state.staticThreadgroupMemoryLength())
            .map_err(|_| {
                NativeCompilationError::MalformedToolchainOutput(
                    "Metal pipeline static threadgroup memory exceeds u64".into(),
                )
            })?;
        if pipeline_threads == 0 {
            return Err(NativeCompilationError::MalformedToolchainOutput(
                "Metal pipeline reflects an empty threadgroup domain".into(),
            ));
        }
        let subgroup_width = u32::try_from(candidate.pipeline.state.threadExecutionWidth())
            .map_err(|_| {
                NativeCompilationError::MalformedToolchainOutput(
                    "Metal pipeline SIMD width exceeds u32".into(),
                )
            })?;
        if subgroup_width == 0 {
            return Err(NativeCompilationError::MalformedToolchainOutput(
                "Metal pipeline reflects a zero SIMD width".into(),
            ));
        }
        let numerical_identity = NativeNumericalModeIdentity {
            fingerprint: Sha256::digest(b"seismic-metal-precise-no-fast-math-v1").into(),
        };
        Ok(NativeKernelReflection {
            handle: candidate.pipeline,
            identity: NativeKernelIdentity {
                compatibility: target.compatibility_identity().clone(),
                artifact_digest: candidate.artifact_digest,
            },
            abi: target.kernel_abi_layout(kernel),
            launch: NativeLaunchDomain {
                modes: vec![MetalLaunchMode],
                subgroup_width: Some(subgroup_width),
                cluster: seismic_compiler::target::NativeClusterDomain::NotApplicable,
                max_grid: target.limits().max_grid,
                max_workgroup_size: target.limits().max_workgroup_size,
                max_workgroup_threads: pipeline_threads,
                max_dynamic_local_bytes: target
                    .limits()
                    .max_workgroup_bytes
                    .checked_sub(static_local)
                    .ok_or_else(|| {
                        NativeCompilationError::MalformedToolchainOutput(
                            "Metal pipeline static threadgroup memory exceeds the device limit"
                                .into(),
                        )
                    })?,
            },
            resources: NativeKernelResources {
                registers_per_participant: NativeResourceUsage::NotApplicable,
                spill_bytes_per_participant: NativeResourceUsage::NotApplicable,
                static_local_bytes: NativeResourceUsage::Exact(static_local),
            },
            numerics: MetalNumericalMode,
            numerical_identity,
            artifact: NativeArtifactMetrics {
                compilation_ns: candidate
                    .compilation_ns
                    .checked_add(
                        u64::try_from(reflection_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    )
                    .unwrap_or(u64::MAX),
                // Metal does not expose a pipeline binary. This is the exact
                // accepted MSL source artifact size whose digest identifies
                // the native candidate.
                code_bytes: candidate.source_bytes,
                metadata_bytes: candidate.metadata_bytes,
            },
        })
    }
}
