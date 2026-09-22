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

pub mod model;

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
use seismic_ir::target::IntrinsicIdentityBuilder;
use seismic_lang::registry::BackendName;
use seismic_target::{
    DeviceDescription, NativeArtifactMetrics, NativeCompilationError, NativeKernelDescription,
    NativeKernelIdentity, NativeKernelReflection, NativeLaunchDomain, NativeNumericalModeIdentity,
    NativeResourceUsage, NativeResources,
};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub use command::Pipeline;
pub use compile::NativeCandidate;
pub use device::{DeviceHandle, MetalBuffer, MetalDevice};
pub use direct::DirectPipeline;
pub use executor::MetalExecutor;
pub use facts::MetalFacts;
use seismic_ir::metal::MetalIntrinsic;

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

impl seismic_ir::target::KernelDialect for Metal {
    const NAME: BackendName = BackendName::Metal;

    type Intrinsic = MetalIntrinsic;
    type Facts = MetalFacts;

    fn write_intrinsic_identity(
        intrinsic: &MetalIntrinsic,
        identity: &mut IntrinsicIdentityBuilder,
    ) {
        intrinsic::write_identity(intrinsic, identity)
    }

    fn intrinsic_addressable_resources(
        _intrinsic: &Self::Intrinsic,
    ) -> Vec<seismic_ir::kernel::ops::AddressableResourceHandle> {
        Vec::new()
    }

    fn intrinsic_numerics(
        _facts: &Self::Facts,
        signature: &seismic_lang::registry::IntrinsicSignature,
        intrinsic: &Self::Intrinsic,
    ) -> seismic_ir::target::IntrinsicNumericalSemantics {
        intrinsic::numerical_semantics(signature, intrinsic)
    }
}

impl seismic_target::TargetFamily for Metal {
    type KernelAbi = profile::MetalKernelAbi;
    type NativeLaunchMode = MetalLaunchMode;
    type NativeNumericalMode = MetalNumericalMode;
    type NativeProperties = ();
}

pub struct MetalNativeCompiler;
static METAL_NATIVE_COMPILER: MetalNativeCompiler = MetalNativeCompiler;

pub fn native_compiler() -> &'static MetalNativeCompiler {
    &METAL_NATIVE_COMPILER
}

pub(crate) fn native_launch_constraints(
    _target: &DeviceDescription<Metal>,
    arena: &mut seismic_lang::expr::ExprArena,
    launch: &seismic_ir::schedule::Launch,
    _locals: &seismic_ir::storage::LaunchLocalLayout,
    kernel: &seismic_ir::kernel::Kernel<Metal>,
    native: &seismic_target::NativeKernelDescription<Metal>,
) -> Vec<seismic_lang::expr::BoolExpr> {
    fn contains_matrix(
        kernel: &seismic_ir::kernel::Kernel<Metal>,
        block: seismic_ir::kernel::BlockId,
    ) -> bool {
        kernel.block(block).ops.iter().any(|op| match op {
            seismic_ir::kernel::ops::Op::Intrinsic { op, .. } => {
                matches!(op, MetalIntrinsic::Matrix { .. })
            }
            seismic_ir::kernel::ops::Op::Branch {
                then, otherwise, ..
            } => contains_matrix(kernel, *then) || contains_matrix(kernel, *otherwise),
            seismic_ir::kernel::ops::Op::Repeat { body, .. } => contains_matrix(kernel, *body),
            _ => false,
        })
    }

    if !contains_matrix(kernel, kernel.root()) {
        return Vec::new();
    }
    let width =
        arena.nat(u64::from(native.launch.subgroup_width.expect(
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

impl seismic_target::NativeCompiler<Metal> for MetalNativeCompiler {
    type Context = DeviceHandle;
    type Candidate = NativeCandidate;
    type Handle = Pipeline;
    fn form(
        &self,
        context: &Self::Context,
        target: &DeviceDescription<Metal>,
        kernel: &seismic_ir::kernel::Kernel<Metal>,
        layout: &seismic_ir::target::KernelEmissionLayout,
    ) -> Result<Self::Candidate, NativeCompilationError> {
        compile::compile_kernel(context, target, kernel, layout)
    }
    fn reflect(
        &self,
        target: &DeviceDescription<Metal>,
        kernel: &seismic_ir::kernel::Kernel<Metal>,
        _layout: &seismic_ir::target::KernelEmissionLayout,
        candidate: Self::Candidate,
    ) -> Result<NativeKernelReflection<Metal, Self::Handle>, NativeCompilationError> {
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
        let description = NativeKernelDescription {
            identity: NativeKernelIdentity {
                compatibility: target.compatibility_identity().clone(),
                artifact_digest: candidate.artifact_digest,
            },
            abi: target.kernel_abi_layout(kernel),
            launch: NativeLaunchDomain {
                modes: vec![MetalLaunchMode],
                subgroup_width: Some(subgroup_width),
                cluster: seismic_target::NativeClusterDomain::NotApplicable,
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
            resources: NativeResources {
                registers_per_participant: NativeResourceUsage::NotApplicable,
                spill_bytes_per_participant: NativeResourceUsage::NotApplicable,
                static_local_bytes: NativeResourceUsage::Exact(static_local),
            },
            numerics: MetalNumericalMode,
            numerical_identity,
            properties: (),
        };
        let metrics = NativeArtifactMetrics {
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
        };
        Ok(NativeKernelReflection::new(
            candidate.pipeline,
            description,
            metrics,
        ))
    }
}

#[cfg(test)]
mod isolation;
