//! The explicitly selected direct-native route
//! (`docs/seismic/language/functions-and-capabilities.md`).
//!
//! A checked entry's authored implementation for the opened device's backend
//! is formed under one [`NativeSpecialization`] and executed without a
//! compiler plan. Metal, CUDA and Vulkan sources receive the generated
//! prefix of [`abi`]; CPU implementations are Rust compiled into the binary
//! and reached through [`cpu`].
//!
//! Submission is asynchronous. Launches of one submission share one serial
//! encoder (Metal), one stream (CUDA), one command buffer on the device's
//! queue (Vulkan) or the worker pool (CPU); the device queue orders
//! submissions, and allocation fences order host access after them.

pub(crate) mod abi;
pub mod cpu;
mod cuda;
pub mod graph;
mod graph_replays;
#[cfg(not(target_os = "macos"))]
mod vulkan;
pub mod replay;
pub mod search;
mod timing;
pub mod trace;
pub mod tune;

use crate::api::device::DeviceInner;
use crate::api::kernel::{DecodedResults, DecodedValue, EncodedArgs, EncodedOutputs, PrepareError};
use crate::api::tensor::TensorInner;
use crate::api::{CallError, OutputError};
use crate::backends::{CpuOpened, CudaOpened, OpenedKind};
use crate::driver::{
    collect_native_access, typed_buffer, write_zeros, Allocation, DeviceCompletion,
};
use seismic_compiler::errors::{ExecutionError, InvocationError, PreparationError};
use seismic_compiler::prepared::{
    validate_invocation, ArgumentValue, DeviceIdentity, InvocationContract,
};
use seismic_lang::checked::{
    CheckedModule, NativeCondition, NativeImplementation, NativeNatExpr, NativeSpecialization,
};
use seismic_lang::entry::{CallSchema, ElementBindings, LogicalEntry, ParameterKind, ResultKind};
use seismic_lang::expr::compiled::{CompiledNat, InvocationValues};
use seismic_lang::expr::{SymbolId, SymbolValue};
use seismic_lang::ids::{EntryId, RepresentationId};
use seismic_lang::registry::BackendName;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub use cpu::{CpuInvocation, CpuKernelFn, CpuLaunchVariants, CpuNativeKernels, CpuTensor};

#[cfg(target_os = "macos")]
use crate::backends::MetalOpened;

#[derive(Clone)]
pub(crate) enum NativeResult {
    Tensor {
        representation: RepresentationId,
        axes: Vec<CompiledNat>,
    },
    Scalar(seismic_lang::types::DType),
    Index,
    Range,
}

#[derive(Clone)]
pub(crate) struct NativeTensorSpec {
    pub(crate) representation: RepresentationId,
    pub(crate) extents: Vec<u64>,
    pub(crate) strides: Vec<u64>,
    pub(crate) byte_len: u64,
}

/// Evaluated geometry of one launch of one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaunchGeometry {
    pub(crate) groups: [u64; 3],
    pub(crate) threads: [u64; 3],
    pub(crate) shared_bytes: u64,
}

/// Geometry of every launch of one call, by declaration ordinal; `None` for
/// a launch whose `when` condition does not hold (not encoded).
pub(crate) type CallLaunches = Vec<Option<LaunchGeometry>>;

/// Alignment of the base of every buffer the route places: standalone
/// scratch, and every graph result, local, host-written input, export and
/// scratch buffer. Device allocations are at least this aligned, so kernels
/// may rely on it for vector access to any placed buffer, as they do for a
/// standalone call's.
pub(crate) const BUFFER_ALIGNMENT: u64 = 256;

/// Per-device state of native submission.
#[derive(Default)]
pub(crate) struct NativeQueue {
    /// Held from encoding a submission to recording its allocation fences.
    order: Mutex<()>,
    /// CUDA graphs of sealed-plan submissions (CUDA devices only).
    cuda_replays: cuda::CudaReplays,
    /// Recorded Vulkan graphs of sealed-plan submissions (Vulkan devices
    /// only).
    #[cfg(not(target_os = "macos"))]
    vulkan_replays: vulkan::VulkanReplays,
}

/// Bytes charged to a scratch buffer that is empty or inactive: it keeps
/// its ABI slot.
const MINIMUM_SCRATCH_BYTES: u64 = 1;

enum NativeRoute {
    Cpu {
        opened: Arc<CpuOpened>,
        launches: Vec<CpuKernelFn>,
    },
    #[cfg(target_os = "macos")]
    Metal {
        opened: Arc<MetalOpened>,
        pipelines: Vec<seismic_metal::DirectPipeline>,
    },
    Cuda {
        opened: Arc<CudaOpened>,
        module: seismic_cuda::direct::DirectModule,
    },
    #[cfg(not(target_os = "macos"))]
    Vulkan {
        opened: Arc<crate::backends::VulkanOpened>,
        module: seismic_vulkan::formation::DirectModule,
    },
}

/// What was formed, for tuning records and measurement attribution: the
/// backend, entry, element bindings, specialization, rendered-source digest
/// and toolchain.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeArtifactIdentity(pub String);

/// One authored native implementation formed for one device, element
/// bindings and specialization.
pub struct NativePrepared {
    name: String,
    device: DeviceIdentity,
    public_device: Arc<DeviceInner>,
    logical: LogicalEntry,
    invocation: InvocationContract,
    implementation: NativeImplementation,
    specialization: NativeSpecialization,
    /// Static dimensions: name, invocation symbol and fixed value.
    statics: Vec<(String, SymbolId, u64)>,
    /// Tensor parameters (by schema ordinal) whose static extents render
    /// constant canonical strides (S10), with those strides.
    canonical_parameters: Vec<(usize, Vec<u64>)>,
    results: Vec<NativeResult>,
    scalar_words: usize,
    /// Scalar-result slots. Graph nodes publish no scalars, so only
    /// standalone calls write them; `standalone` serializes those calls.
    scalars: Arc<Allocation>,
    standalone: Mutex<Standalone>,
    scalar_bytes: u64,
    artifact: NativeArtifactIdentity,
    route: NativeRoute,
}

/// The pipeline geometry of every launch of a Vulkan implementation: its
/// workgroup size and group-memory view lengths. Both read only static
/// dimensions and parameters (the checker's Vulkan rule), so every pipeline
/// is formed, and checked against the device, at preparation.
#[cfg(not(target_os = "macos"))]
fn vulkan_geometry(
    opened: &crate::backends::VulkanOpened,
    name: &str,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
) -> Result<Vec<([u32; 3], Vec<u32>)>, PrepareError> {
    let limits = opened.service().facts().limits;
    let narrow = |value: u64, what: &str| {
        u32::try_from(value).map_err(|_| preparation(format!("`{name}`: {what} {value} exceeds the Vulkan ABI")))
    };
    implementation
        .launches
        .iter()
        .map(|launch| {
            let kernel = &launch.kernel;
            let evaluate = |expression: &NativeNatExpr| {
                expression
                    .evaluate(&|dimension| specialization.static_value(dimension), &|parameter| {
                        specialization.param(parameter)
                    })
                    .map_err(|error| preparation(format!("`{name}` launch `{kernel}`: {error}")))
            };
            let threads = [
                evaluate(&launch.group_extent[0])?,
                evaluate(&launch.group_extent[1])?,
                evaluate(&launch.group_extent[2])?,
            ];
            let invocations = threads.iter().try_fold(1u64, |product, value| product.checked_mul(*value));
            if threads.contains(&0)
                || invocations.is_none_or(|invocations| invocations > limits.max_invocations)
                || threads.iter().zip(limits.max_group_size).any(|(threads, limit)| *threads > limit)
            {
                return Err(preparation(format!(
                    "launch `{kernel}` requests {threads:?} threads per workgroup; the device allows {:?} and {} in total",
                    limits.max_group_size, limits.max_invocations
                )));
            }
            let shared_bytes = evaluate(&launch.shared_bytes)?;
            let footprint = abi::vulkan::shared_footprint(shared_bytes);
            if footprint > limits.max_shared_bytes {
                return Err(preparation(format!(
                    "launch `{kernel}` needs {footprint} shared bytes; the device allows {}",
                    limits.max_shared_bytes
                )));
            }
            Ok((
                [
                    narrow(threads[0], "a workgroup extent")?,
                    narrow(threads[1], "a workgroup extent")?,
                    narrow(threads[2], "a workgroup extent")?,
                ],
                abi::vulkan::shared_view_lengths(shared_bytes)
                    .into_iter()
                    .map(|length| narrow(length, "a shared view length"))
                    .collect::<Result<_, _>>()?,
            ))
        })
        .collect()
}

fn entry_name(module: &CheckedModule, entry: EntryId) -> String {
    module
        .entries()
        .iter()
        .find(|candidate| candidate.id == entry)
        .expect("entry belongs to its module")
        .name
        .clone()
}

fn preparation(message: impl Into<String>) -> PrepareError {
    PrepareError::Preparation(PreparationError::NativeSpecialization(message.into()))
}

pub(crate) fn backend_name(kind: &OpenedKind) -> BackendName {
    match kind {
        OpenedKind::Cpu(_) => BackendName::Cpu,
        #[cfg(target_os = "macos")]
        OpenedKind::Metal(_) => BackendName::Metal,
        OpenedKind::Cuda(_) => BackendName::Cuda,
        #[cfg(not(target_os = "macos"))]
        OpenedKind::Vulkan(_) => BackendName::Vulkan,
    }
}

impl NativePrepared {
    /// Form the entry's native implementation for `device`'s backend.
    /// `cpu` carries the compiled CPU launch functions when the build
    /// generated them.
    pub(crate) fn prepare(
        device: &Arc<DeviceInner>,
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        specialization: NativeSpecialization,
        cpu: Option<&'static CpuNativeKernels>,
    ) -> Result<Arc<Self>, PrepareError> {
        let backend = backend_name(&device.kind);
        let implementation = module
            .native_implementation(entry, backend)
            .cloned()
            .ok_or_else(|| {
                preparation(format!(
                    "`{}` has no native implementation for `{}`",
                    entry_name(module, entry),
                    backend.as_str()
                ))
            })?;
        Self::prepare_implementation(
            device,
            module,
            entry,
            bindings,
            specialization,
            cpu,
            implementation,
        )
    }

    /// Form `implementation` of the entry, which may differ from the
    /// module's declaration in its parameter domains only (a tuning
    /// survey's widened domains).
    pub(crate) fn prepare_implementation(
        device: &Arc<DeviceInner>,
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        specialization: NativeSpecialization,
        cpu: Option<&'static CpuNativeKernels>,
        implementation: NativeImplementation,
    ) -> Result<Arc<Self>, PrepareError> {
        let backend = backend_name(&device.kind);
        let name = entry_name(module, entry);
        implementation
            .validate(&specialization)
            .map_err(|error| preparation(format!("`{name}`: {error}")))?;
        let logical = module
            .entry(entry, &bindings)
            .map_err(PrepareError::Source)?;
        let schema = logical.schema();
        let statics = implementation
            .statics
            .iter()
            .map(|dimension| {
                let symbol = schema
                    .dimensions()
                    .iter()
                    .find(|candidate| &candidate.name == dimension)
                    .expect("checked static names an entry dimension")
                    .symbol;
                let value = specialization
                    .static_value(dimension)
                    .expect("validated specialization fixes every static dimension");
                (dimension.clone(), symbol, value)
            })
            .collect::<Vec<_>>();
        let invocation = InvocationContract::compile_entry(&logical);
        let results = schema
            .results()
            .iter()
            .map(|result| match &result.kind {
                ResultKind::Tensor {
                    representation,
                    axes,
                } => NativeResult::Tensor {
                    representation: *representation,
                    axes: axes
                        .iter()
                        .map(|axis| logical.arena().compile_nat(*axis))
                        .collect(),
                },
                ResultKind::Scalar(dtype) => NativeResult::Scalar(*dtype),
                ResultKind::Index { .. } => NativeResult::Index,
                ResultKind::Range { .. } => NativeResult::Range,
            })
            .collect::<Vec<_>>();
        let words = abi::word_count(schema);
        let scalar_words = abi::scalar_word_count(schema);
        let kernels = implementation
            .launches
            .iter()
            .map(|launch| launch.kernel.as_str())
            .collect::<Vec<_>>();
        let compilation =
            |error| PrepareError::Preparation(PreparationError::NativeCompilation(error));
        let asset = |backend| {
            module.native_asset(entry, backend).ok_or_else(|| {
                preparation(format!(
                    "native asset of `{name}` is absent from the module"
                ))
            })
        };
        let mut digest = Sha256::new();
        let (route, toolchain) = match &device.kind {
            OpenedKind::Cpu(opened) => {
                let kernels = cpu.ok_or_else(|| {
                    preparation(format!(
                        "`{name}` has no compiled CPU native functions in this build"
                    ))
                })?;
                let configuration = implementation
                    .params
                    .iter()
                    .map(|parameter| {
                        specialization
                            .param(&parameter.name)
                            .expect("validated specialization values every parameter")
                    })
                    .collect::<Vec<_>>();
                let launches = (0..implementation.launches.len())
                    .map(|launch| {
                        kernels.function(launch, &configuration).ok_or_else(|| {
                            preparation(format!(
                                "`{name}` launch {launch} has no compiled CPU variant for {configuration:?}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                digest.update(format!(
                    "{:?}",
                    launches.iter().map(|f| *f as usize).collect::<Vec<_>>()
                ));
                (
                    NativeRoute::Cpu {
                        opened: opened.clone(),
                        launches,
                    },
                    format!("cpu;{}", std::env::consts::ARCH),
                )
            }
            #[cfg(target_os = "macos")]
            OpenedKind::Metal(opened) => {
                let slots = abi::buffer_slots(schema, &implementation);
                if slots > seismic_metal::DIRECT_BUFFER_SLOTS {
                    return Err(PrepareError::Preparation(PreparationError::NativeBufferSlots {
                        entry: name,
                        slots,
                        limit: seismic_metal::DIRECT_BUFFER_SLOTS,
                    }));
                }
                if words * 8 > seismic_metal::DIRECT_WORD_BYTES_LIMIT {
                    return Err(preparation(format!(
                        "`{name}` needs {} argument bytes, beyond Metal's setBytes limit",
                        words * 8
                    )));
                }
                let source = abi::render_source(
                    abi::Dialect::Metal,
                    &logical,
                    &bindings,
                    &implementation,
                    &specialization,
                    asset(BackendName::Metal)?,
                );
                digest.update(source.as_bytes());
                let pipelines =
                    seismic_metal::DirectPipeline::compile_all(opened.service(), &source, &kernels)
                        .map_err(compilation)?;
                (
                    NativeRoute::Metal {
                        opened: opened.clone(),
                        pipelines,
                    },
                    format!(
                        "metal;{}",
                        opened.device_description().facts().operating_system()
                    ),
                )
            }
            OpenedKind::Cuda(opened) => {
                let source = abi::render_source(
                    abi::Dialect::Cuda,
                    &logical,
                    &bindings,
                    &implementation,
                    &specialization,
                    asset(BackendName::Cuda)?,
                );
                digest.update(source.as_bytes());
                let facts = opened.device_description().facts();
                let architecture = u32::from(facts.compute_capability.major) * 10
                    + u32::from(facts.compute_capability.minor);
                // A CUBIN is determined by its source and formation (NVRTC
                // release, architecture, options): the store's key.
                let key = |formation: &seismic_cuda::nvrtc::Formation| {
                    crate::artifacts::ArtifactKey::of(&[
                        source.as_bytes(),
                        formation.to_string().as_bytes(),
                    ])
                };
                let kind = crate::artifacts::ArtifactKind::CudaImage;
                let store = device.artifacts.as_deref();
                let module = seismic_cuda::direct::DirectModule::form(
                    opened.service(),
                    &source,
                    &format!("{name}.cu"),
                    architecture,
                    &kernels,
                    |formation| store.and_then(|store| store.get(kind, &key(formation))),
                    |formation, image| {
                        if let Some(store) = store {
                            store.put(kind, &key(formation), image);
                        }
                    },
                )
                .map_err(compilation)?;
                let toolchain = format!("cuda;{};driver {}", module.formation(), facts.driver_api.0);
                (
                    NativeRoute::Cuda {
                        opened: opened.clone(),
                        module,
                    },
                    toolchain,
                )
            }
            #[cfg(not(target_os = "macos"))]
            OpenedKind::Vulkan(opened) => {
                let source = abi::render_source(
                    abi::Dialect::Vulkan(opened.features()),
                    &logical,
                    &bindings,
                    &implementation,
                    &specialization,
                    asset(BackendName::Vulkan)?,
                );
                digest.update(source.as_bytes());
                let geometry = vulkan_geometry(opened, &name, &implementation, &specialization)?;
                let formation = seismic_vulkan::formation::formation(opened.service());
                // A launch's SPIR-V is determined by the source, its kernel
                // and the formation: the store's key.
                let key = |kernel: &str| {
                    crate::artifacts::ArtifactKey::of(&[
                        source.as_bytes(),
                        kernel.as_bytes(),
                        formation.as_bytes(),
                    ])
                };
                let kind = crate::artifacts::ArtifactKind::SpirV;
                let store = device.artifacts.as_deref();
                let formed = geometry
                    .iter()
                    .zip(&kernels)
                    .map(|((threads, views), kernel)| seismic_vulkan::formation::Kernel {
                        name: kernel,
                        threads: *threads,
                        constants: views,
                    })
                    .collect::<Vec<_>>();
                let module = seismic_vulkan::formation::DirectModule::form(
                    opened.service(),
                    &source,
                    &formed,
                    |kernel| store.and_then(|store| store.get(kind, &key(kernel))),
                    |kernel, spirv| {
                        if let Some(store) = store {
                            store.put(kind, &key(kernel), spirv);
                        }
                    },
                )
                .map_err(compilation)?;
                (
                    NativeRoute::Vulkan {
                        opened: opened.clone(),
                        module,
                    },
                    formation,
                )
            }
        };
        let bindings_text = bindings
            .iter()
            .map(|(parameter, representation)| {
                format!(
                    "{parameter}={}",
                    seismic_lang::registry::representation_info(representation).name
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let artifact = NativeArtifactIdentity(format!(
            "{};{name};{bindings_text};{specialization:?};{};{toolchain}",
            backend.as_str(),
            crate::telemetry::hex(&digest.finalize())
        ));
        // S10: tensor parameters whose extents are static render constant
        // canonical strides, so binding them requires those strides.
        let canonical_parameters = abi::static_geometry(&logical, &specialization)
            .parameters
            .into_iter()
            .enumerate()
            .filter_map(|(ordinal, fixed)| {
                fixed
                    .and_then(|fixed| fixed.strides)
                    .map(|strides| (ordinal, strides))
            })
            .collect();
        let scalar_bytes = (scalar_words as u64 * 8).max(1);
        let scalars = device.allocate(scalar_bytes, 8).map_err(|error| {
            PrepareError::Preparation(PreparationError::NativeWorkspaceAllocation(
                error.to_string(),
            ))
        })?;
        Ok(Arc::new(Self {
            name,
            device: device.kind.identity(),
            public_device: device.clone(),
            logical,
            invocation,
            implementation,
            specialization,
            statics,
            canonical_parameters,
            results,
            scalar_words,
            scalars,
            standalone: Mutex::new(Standalone::default()),
            scalar_bytes,
            artifact,
            route,
        }))
    }

    pub(crate) fn implementation(&self) -> &NativeImplementation {
        &self.implementation
    }
    pub(crate) fn specialization(&self) -> &NativeSpecialization {
        &self.specialization
    }
    pub(crate) fn artifact(&self) -> &NativeArtifactIdentity {
        &self.artifact
    }
    pub(crate) fn schema(&self) -> &CallSchema {
        self.logical.schema()
    }
    /// Device storage this prepared implementation holds for its calls:
    /// the scalar-result slots, fixed at preparation, and the standalone
    /// scratch arena, which grows to the largest standalone call's scratch.
    pub(crate) fn invocation_workspace_bytes(&self) -> u64 {
        self.scalar_bytes
            + self
                .standalone
                .lock()
                .expect("native standalone-call lock poisoned")
                .bytes()
    }
    pub(crate) fn result_count(&self) -> u32 {
        u32::try_from(self.results.len()).expect("native result ordinal space exhausted")
    }

    /// Validate arguments against the checked contract and the fixed static
    /// dimensions.
    pub(crate) fn validate(
        &self,
        arguments: &[ArgumentValue],
    ) -> Result<InvocationValues, CallError> {
        let values = validate_invocation(&self.invocation, self.device, arguments)
            .map_err(CallError::Invocation)?;
        for (dimension, symbol, expected) in &self.statics {
            match values.get(*symbol) {
                Some(SymbolValue::Nat(actual)) if actual == (*expected).into() => {}
                Some(SymbolValue::Nat(actual)) => {
                    return Err(CallError::Invocation(InvocationError::StaticDimension {
                        dimension: dimension.clone(),
                        expected: *expected,
                        actual,
                    }));
                }
                _ => panic!("validated invocation omitted a native static dimension"),
            }
        }
        for (ordinal, strides) in &self.canonical_parameters {
            let ArgumentValue::Tensor(tensor) = &arguments[*ordinal] else {
                panic!("validated invocation bound a non-tensor to a tensor parameter")
            };
            if &tensor.strides != strides {
                return Err(CallError::Invocation(
                    InvocationError::NoncanonicalStaticTensor {
                        parameter: self.schema().parameters()[*ordinal].name.clone(),
                    },
                ));
            }
        }
        Ok(values)
    }

    fn dimension(&self, values: &InvocationValues, name: &str) -> Option<u64> {
        let symbol = self
            .schema()
            .dimensions()
            .iter()
            .find(|dimension| dimension.name == name)?
            .symbol;
        match values.get(symbol) {
            Some(SymbolValue::Nat(value)) => u64::try_from(value).ok(),
            _ => None,
        }
    }

    fn evaluation_error(&self, error: seismic_lang::checked::NativeEvalError) -> CallError {
        CallError::Execution(ExecutionError::SubmissionFailed(format!(
            "native expression of `{}`: {error}",
            self.name
        )))
    }

    fn evaluate(
        &self,
        expression: &NativeNatExpr,
        values: &InvocationValues,
    ) -> Result<u64, CallError> {
        expression
            .evaluate(&|name| self.dimension(values, name), &|name| {
                self.specialization.param(name)
            })
            .map_err(|error| self.evaluation_error(error))
    }

    /// Whether a launch or scratch buffer with this `when` condition is
    /// active for one invocation.
    fn active(
        &self,
        when: &Option<NativeCondition>,
        values: &InvocationValues,
    ) -> Result<bool, CallError> {
        match when {
            None => Ok(true),
            Some(condition) => condition
                .holds(&|name| self.dimension(values, name), &|name| {
                    self.specialization.param(name)
                })
                .map_err(|error| self.evaluation_error(error)),
        }
    }

    /// Geometry of every launch for one invocation, checked against the
    /// formed functions and device limits. An inactive launch is `None`:
    /// its geometry is neither evaluated nor checked.
    pub(crate) fn launches(&self, values: &InvocationValues) -> Result<CallLaunches, CallError> {
        let mut geometry = Vec::with_capacity(self.implementation.launches.len());
        for (ordinal, launch) in self.implementation.launches.iter().enumerate() {
            if !self.active(&launch.when, values)? {
                geometry.push(None);
                continue;
            }
            let axes = |expressions: &[NativeNatExpr; 3]| -> Result<[u64; 3], CallError> {
                Ok([
                    self.evaluate(&expressions[0], values)?,
                    self.evaluate(&expressions[1], values)?,
                    self.evaluate(&expressions[2], values)?,
                ])
            };
            let groups = axes(&launch.groups)?;
            let threads = axes(&launch.group_extent)?;
            let shared_bytes = self.evaluate(&launch.shared_bytes, values)?;
            let limit =
                |message: String| CallError::Execution(ExecutionError::SubmissionFailed(message));
            let participants = threads
                .iter()
                .try_fold(1u64, |product, value| product.checked_mul(*value))
                .ok_or_else(|| {
                    limit(format!("launch `{}` thread count overflows", launch.kernel))
                })?;
            match &self.route {
                NativeRoute::Cpu { .. } => {}
                #[cfg(target_os = "macos")]
                NativeRoute::Metal { opened, pipelines } => {
                    let pipeline = &pipelines[ordinal];
                    if participants > pipeline.max_threads_per_threadgroup() {
                        return Err(limit(format!(
                            "launch `{}` requests {participants} threads per threadgroup; the pipeline allows {}",
                            launch.kernel,
                            pipeline.max_threads_per_threadgroup()
                        )));
                    }
                    let available = opened.device_description().limits().max_workgroup_bytes;
                    if shared_bytes + pipeline.static_threadgroup_bytes() > available {
                        return Err(limit(format!(
                            "launch `{}` needs {} threadgroup bytes; the device allows {available}",
                            launch.kernel,
                            shared_bytes + pipeline.static_threadgroup_bytes()
                        )));
                    }
                }
                NativeRoute::Cuda { opened, module } => {
                    if participants > module.max_threads_per_block(ordinal) {
                        return Err(limit(format!(
                            "launch `{}` requests {participants} threads per block; the function allows {}",
                            launch.kernel,
                            module.max_threads_per_block(ordinal)
                        )));
                    }
                    let available = opened.device_description().limits().max_workgroup_bytes;
                    if shared_bytes + module.static_shared_bytes(ordinal) > available {
                        return Err(limit(format!(
                            "launch `{}` needs {} shared bytes; the device allows {available}",
                            launch.kernel,
                            shared_bytes + module.static_shared_bytes(ordinal)
                        )));
                    }
                }
                // The workgroup size and group memory are the pipeline's,
                // checked at preparation; only the grid varies per call.
                #[cfg(not(target_os = "macos"))]
                NativeRoute::Vulkan { opened, .. } => {
                    let available = opened.service().facts().limits.max_group_count;
                    if let Some(axis) = (0..3).find(|axis| groups[*axis] > available[*axis]) {
                        return Err(limit(format!(
                            "launch `{}` requests {} workgroups on axis {axis}; the device allows {}",
                            launch.kernel, groups[axis], available[axis]
                        )));
                    }
                }
            }
            geometry.push(Some(LaunchGeometry {
                groups,
                threads,
                shared_bytes,
            }));
        }
        Ok(geometry)
    }

    /// Bytes of every scratch buffer for one invocation. An inactive buffer
    /// is charged the minimum without evaluating its size.
    pub(crate) fn scratch_bytes(&self, values: &InvocationValues) -> Result<Vec<u64>, CallError> {
        self.implementation
            .scratch
            .iter()
            .map(|scratch| {
                if !self.active(&scratch.when, values)? {
                    return Ok(MINIMUM_SCRATCH_BYTES);
                }
                self.evaluate(&scratch.bytes, values)
                    .map(|bytes| bytes.max(MINIMUM_SCRATCH_BYTES))
            })
            .collect()
    }

    fn result_extents(
        &self,
        values: &InvocationValues,
    ) -> Result<Vec<Option<(RepresentationId, Vec<u64>)>>, CallError> {
        self.results
            .iter()
            .map(|result| match result {
                NativeResult::Tensor {
                    representation,
                    axes,
                } => Ok(Some((
                    *representation,
                    axes.iter()
                        .map(|axis| evaluate_compiled(axis, values))
                        .collect::<Result<Vec<_>, _>>()?,
                ))),
                NativeResult::Scalar(_) | NativeResult::Index | NativeResult::Range => Ok(None),
            })
            .collect()
    }

    /// The parts of one call its arguments' geometry and scalar values fix,
    /// after full contract validation: result specs, scratch sizes, argument
    /// words and launch geometry. A standalone call evaluates it per call; a
    /// sealed graph evaluates it once per node, at seal.
    pub(crate) fn shape(&self, arguments: &[ArgumentValue]) -> Result<CallShape, CallError> {
        let values = self.validate(arguments)?;
        let launches = self.launches(&values)?;
        let results = self
            .result_extents(&values)?
            .into_iter()
            .map(|result| {
                result
                    .map(|(representation, extents)| {
                        let layout = crate::layout::canonical(representation, &extents)
                            .map_err(CallError::Execution)?;
                        Ok(NativeTensorSpec {
                            representation,
                            extents,
                            strides: layout.strides,
                            byte_len: layout.byte_len,
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, CallError>>()?;
        let words = native_words(self.schema(), arguments, &results, &values)?;
        Ok(CallShape {
            scratch: self.scratch_bytes(&values)?,
            results,
            words,
            launches,
        })
    }

    pub(crate) fn tensor_parameter_spec(
        &self,
        name: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativeTensorSpec, CallError> {
        let schema = self.schema();
        let failure =
            |message: String| CallError::Execution(ExecutionError::SubmissionFailed(message));
        let parameter = schema
            .parameters()
            .iter()
            .find(|parameter| parameter.name == name)
            .ok_or_else(|| {
                failure(format!(
                    "checked native entry has no tensor parameter `{name}`"
                ))
            })?;
        let ParameterKind::Tensor {
            representation,
            axes,
            ..
        } = &parameter.kind
        else {
            return Err(failure(format!(
                "checked native parameter `{name}` is not a tensor"
            )));
        };
        let mut values = InvocationValues::new();
        for dimension in schema.dimensions() {
            let value = dimensions
                .iter()
                .find(|(candidate, _)| *candidate == dimension.name)
                .map(|(_, value)| *value)
                .ok_or_else(|| {
                    failure(format!(
                        "native graph omitted dimension `{}` for `{name}`",
                        dimension.name
                    ))
                })?;
            values.bind(dimension.symbol, SymbolValue::Nat(value.into()));
        }
        let extents = axes
            .iter()
            .map(|axis| evaluate_compiled(&self.logical.arena().compile_nat(*axis), &values))
            .collect::<Result<Vec<_>, _>>()?;
        let layout =
            crate::layout::canonical(*representation, &extents).map_err(CallError::Execution)?;
        Ok(NativeTensorSpec {
            representation: *representation,
            extents,
            strides: layout.strides,
            byte_len: layout.byte_len,
        })
    }

    /// Graph nodes cannot publish scalar results: a scalar crosses a host
    /// boundary.
    pub(crate) fn validate_graph_node(&self, device: DeviceIdentity) -> Result<(), CallError> {
        if self.device != device {
            return Err(CallError::Workflow(
                crate::api::WorkflowError::NativeGraphSlotMismatch,
            ));
        }
        if self
            .results
            .iter()
            .any(|result| !matches!(result, NativeResult::Tensor { .. }))
        {
            return Err(CallError::Workflow(
                crate::api::WorkflowError::HostBoundaryRequired,
            ));
        }
        Ok(())
    }

    pub(crate) fn call(self: &Arc<Self>, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        self.call_with(args, None, || {})
    }

    pub(crate) fn call_into(
        self: &Arc<Self>,
        args: EncodedArgs,
        outputs: EncodedOutputs,
    ) -> Result<DecodedResults, CallError> {
        self.call_with(args, Some(outputs), || {})
    }

    pub(crate) fn call_with_commit(
        self: &Arc<Self>,
        args: EncodedArgs,
        commit: impl FnOnce(),
    ) -> Result<DecodedResults, CallError> {
        self.call_with(args, None, commit)
    }

    /// Validate a standalone call, place its results and its scratch (in the
    /// standalone scratch arena), and fix its executable form.
    fn prepare_call(
        &self,
        standalone: &mut Standalone,
        args: EncodedArgs,
        outputs: Option<EncodedOutputs>,
    ) -> Result<NativeBoundCall, CallError> {
        let shape = self.shape(&args.values())?;
        let specs = shape.results.iter().flatten().collect::<Vec<_>>();
        let supplied = outputs.map(EncodedOutputs::into_tensors);
        if let Some(outputs) = &supplied {
            if outputs.len() != specs.len() {
                return Err(CallError::Output(OutputError::Count {
                    expected: specs.len(),
                    actual: outputs.len(),
                }));
            }
        }
        let mut results = Vec::with_capacity(specs.len());
        for (index, spec) in specs.into_iter().enumerate() {
            let tensor = match &supplied {
                Some(outputs) => {
                    let tensor = outputs[index].clone();
                    validate_output(
                        index,
                        &tensor,
                        self.device,
                        spec.representation,
                        &spec.extents,
                        &args,
                        &results,
                    )?;
                    tensor
                }
                None => Arc::new(
                    TensorInner::zeros(&self.public_device, spec.representation, &spec.extents)
                        .map_err(tensor_error)?,
                ),
            };
            results.push(tensor);
        }
        let mut scratch = Vec::with_capacity(shape.scratch.len());
        let mut scratch_end = 0u64;
        for bytes in &shape.scratch {
            let offset = scratch_end.next_multiple_of(BUFFER_ALIGNMENT);
            scratch.push(offset);
            scratch_end = offset + bytes;
        }
        let arena = standalone.scratch(&self.public_device, scratch_end)?;

        let schema = self.schema();
        let mut buffers = Vec::new();
        let mut representations = Vec::new();
        let mut access = collect_native_access(schema, &args);
        for (ordinal, parameter) in schema.parameters().iter().enumerate() {
            if matches!(parameter.kind, ParameterKind::Tensor { .. }) {
                let tensor = args
                    .tensor(ordinal)
                    .expect("validated native tensor argument disappeared");
                buffers.push((tensor.allocation().clone(), tensor.byte_offset()));
                representations.push(representation_name(tensor.representation()));
            }
        }
        for tensor in &results {
            buffers.push((tensor.allocation().clone(), tensor.byte_offset()));
            representations.push(representation_name(tensor.representation()));
            access.push((tensor.allocation().clone(), true));
        }
        if let Some(arena) = arena {
            for offset in scratch {
                buffers.push((arena.clone(), offset));
                representations.push("bytes");
            }
            access.push((arena, true));
        }
        access.push((self.scalars.clone(), true));
        Ok(NativeBoundCall {
            results,
            buffers,
            representations,
            word_bytes: word_bytes(&shape.words),
            words: shape.words,
            launches: shape.launches,
            access,
        })
    }

    fn call_with(
        self: &Arc<Self>,
        args: EncodedArgs,
        outputs: Option<EncodedOutputs>,
        commit: impl FnOnce(),
    ) -> Result<DecodedResults, CallError> {
        // Standalone calls of this implementation share its scalar-result
        // slots and scratch arena; the guard covers their placement, reset,
        // execution and read.
        let mut standalone = self
            .standalone
            .lock()
            .expect("native standalone-call lock poisoned");
        let call = self.prepare_call(&mut standalone, args, outputs)?;
        let scalars = &self.scalars;
        {
            let _host = scalars.acquire(true);
            write_zeros(scalars.storage(), self.scalar_bytes).map_err(CallError::Execution)?;
        }
        commit();
        let calls = [call];
        StandaloneCalls {
            kernel: self,
            calls: &calls,
        }
        .submit(1)?
        .wait()?;
        let mut scalar_bytes = vec![0u8; self.scalar_words * 8];
        scalars
            .acquire(false)
            .read(scalars, 0, &mut scalar_bytes)
            .map_err(CallError::Execution)?;
        let [call] = calls;
        let mut offset = 0usize;
        let mut tensors = call.results.into_iter();
        let mut decoded = Vec::with_capacity(self.results.len());
        for result in &self.results {
            match result {
                NativeResult::Tensor { .. } => decoded.push(DecodedValue::Tensor(
                    tensors.next().expect("native result tensor count changed"),
                )),
                NativeResult::Scalar(dtype) => {
                    decoded.push(DecodedValue::Scalar(scalar_value(
                        *dtype,
                        read_word(&scalar_bytes, offset),
                    )));
                    offset += 1;
                }
                NativeResult::Index => {
                    decoded.push(DecodedValue::Scalar(ArgumentValue::Index(
                        read_word(&scalar_bytes, offset).into(),
                    )));
                    offset += 1;
                }
                NativeResult::Range => {
                    decoded.push(DecodedValue::Scalar(ArgumentValue::Range {
                        start: read_word(&scalar_bytes, offset).into(),
                        end: read_word(&scalar_bytes, offset + 1).into(),
                    }));
                    offset += 2;
                }
            }
        }
        Ok(DecodedResults::new(decoded))
    }
}

/// Device storage of standalone calls beyond the scalar-result slots: one
/// scratch arena, grown to the largest standalone call's scratch and reused
/// by every later one (the device queue orders its users).
#[derive(Default)]
struct Standalone {
    scratch: Option<Arc<Allocation>>,
}

impl Standalone {
    fn bytes(&self) -> u64 {
        self.scratch.as_ref().map_or(0, |arena| arena.bytes())
    }

    /// The arena, holding at least `bytes`; `None` when the call has no
    /// scratch.
    fn scratch(
        &mut self,
        device: &Arc<DeviceInner>,
        bytes: u64,
    ) -> Result<Option<Arc<Allocation>>, CallError> {
        if bytes == 0 {
            return Ok(None);
        }
        if let Some(arena) = self.scratch.as_ref().filter(|arena| arena.bytes() >= bytes) {
            return Ok(Some(arena.clone()));
        }
        let arena = device
            .allocate(bytes, BUFFER_ALIGNMENT)
            .map_err(CallError::Execution)?;
        write_zeros(arena.storage(), bytes).map_err(CallError::Execution)?;
        self.scratch = Some(arena.clone());
        Ok(Some(arena))
    }
}

/// What a call's argument geometry and scalar values fix
/// ([`NativePrepared::shape`]).
pub(crate) struct CallShape {
    /// Per result ordinal; `None` for a scalar result.
    pub(crate) results: Vec<Option<NativeTensorSpec>>,
    /// Bytes of each scratch buffer.
    pub(crate) scratch: Vec<u64>,
    pub(crate) words: Vec<u64>,
    pub(crate) launches: CallLaunches,
}

pub(crate) fn word_bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn representation_name(representation: RepresentationId) -> &'static str {
    seismic_lang::registry::representation_info(representation).name
}

/// One validated standalone call ready for submission: its arguments,
/// results, buffers, argument words and launch geometry are fixed.
struct NativeBoundCall {
    results: Vec<Arc<TensorInner>>,
    /// Buffers in ABI order with their byte offsets.
    buffers: Vec<(Arc<Allocation>, u64)>,
    representations: Vec<&'static str>,
    words: Vec<u64>,
    word_bytes: Vec<u8>,
    launches: CallLaunches,
    /// Every allocation the call touches, with whether it writes it.
    access: Vec<(Arc<Allocation>, bool)>,
}

/// Standalone calls of one prepared implementation, in submission order.
struct StandaloneCalls<'k> {
    kernel: &'k Arc<NativePrepared>,
    calls: &'k [NativeBoundCall],
}

impl StandaloneCalls<'_> {
    fn submit(&self, repetitions: usize) -> Result<NativeSubmission, CallError> {
        let access = merge_access(
            self.calls
                .iter()
                .flat_map(|call| call.access.iter().cloned()),
        );
        submit(self, &access, self.kernel.clone(), repetitions)
    }
}

impl DispatchList for StandaloneCalls<'_> {
    fn count(&self) -> usize {
        self.calls.len()
    }

    fn dispatch<'s>(
        &'s self,
        index: usize,
        buffers: &mut Vec<(&'s Allocation, u64)>,
    ) -> Dispatch<'s> {
        let call = &self.calls[index];
        buffers.extend(
            call.buffers
                .iter()
                .map(|(allocation, offset)| (&**allocation, *offset)),
        );
        Dispatch {
            kernel: self.kernel,
            words: &call.words,
            word_bytes: &call.word_bytes,
            launches: &call.launches,
            representations: &call.representations,
        }
    }

    fn plans(&self) -> Option<Vec<u64>> {
        None
    }
}

/// One call as the encoder consumes it: everything but its buffers is
/// fixed.
pub(crate) struct Dispatch<'a> {
    pub(crate) kernel: &'a NativePrepared,
    pub(crate) words: &'a [u64],
    pub(crate) word_bytes: &'a [u8],
    /// By declaration ordinal; `None` for an inactive launch.
    pub(crate) launches: &'a [Option<LaunchGeometry>],
    /// Registry name of each buffer's representation, in ABI order.
    pub(crate) representations: &'a [&'static str],
}

/// The calls of one submission, in order. Every call belongs to one device.
pub(crate) trait DispatchList {
    fn count(&self) -> usize;
    /// Call `index`; its buffers, in ABI order with byte offsets, are
    /// appended to `buffers`.
    fn dispatch<'s>(&'s self, index: usize, buffers: &mut Vec<(&'s Allocation, u64)>)
        -> Dispatch<'s>;
    /// Identities of the sealed plans whose runs make up the list, in
    /// order: with the buffer addresses the list binds they fix every launch
    /// argument, so CUDA replays the list as one graph. `None` for lists
    /// launched call by call.
    fn plans(&self) -> Option<Vec<u64>>;
}

/// One entry per allocation with merged write access, in identity order:
/// the order device permits are taken in, so concurrent submitters cannot
/// deadlock.
pub(crate) fn merge_access(
    access: impl IntoIterator<Item = (Arc<Allocation>, bool)>,
) -> Vec<(Arc<Allocation>, bool)> {
    let mut merged: BTreeMap<u64, (Arc<Allocation>, bool)> = BTreeMap::new();
    for (allocation, write) in access {
        merged
            .entry(allocation.identity())
            .and_modify(|(_, current)| *current |= write)
            .or_insert((allocation, write));
    }
    merged.into_values().collect()
}

enum RouteSubmission {
    Cpu {
        outcome: Result<(), ExecutionError>,
        /// Execution interval on the [`trace::host_seconds`] clock.
        interval: (f64, f64),
    },
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::DirectSubmission),
    Cuda(seismic_cuda::direct::DirectSubmission),
    #[cfg(not(target_os = "macos"))]
    Vulkan(seismic_vulkan::direct::DirectSubmission),
}

impl DeviceCompletion for RouteSubmission {
    fn is_complete(&self) -> bool {
        match self {
            Self::Cpu { .. } => true,
            #[cfg(target_os = "macos")]
            Self::Metal(submission) => submission.is_complete(),
            Self::Cuda(submission) => submission.is_complete(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(submission) => submission.is_complete(),
        }
    }
    fn wait_complete(&self) {
        match self {
            Self::Cpu { .. } => {}
            #[cfg(target_os = "macos")]
            Self::Metal(submission) => submission.wait_complete(),
            Self::Cuda(submission) => submission.wait_complete(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(submission) => submission.wait_complete(),
        }
    }
}

/// Submitted native work. It retains the formed functions it executes;
/// allocation fences keep every touched storage alive and order host access
/// until the work completes.
pub(crate) struct NativeSubmission {
    route: Arc<RouteSubmission>,
    _retained: Arc<dyn std::any::Any + Send + Sync>,
}

impl NativeSubmission {
    pub(crate) fn is_complete(&self) -> bool {
        self.route.is_complete()
    }

    /// Wait for completion and report the work's outcome.
    pub(crate) fn wait(&self) -> Result<(), CallError> {
        match &*self.route {
            RouteSubmission::Cpu { outcome, .. } => outcome.clone(),
            #[cfg(target_os = "macos")]
            RouteSubmission::Metal(submission) => submission.finish(),
            RouteSubmission::Cuda(submission) => submission.finish(),
            #[cfg(not(target_os = "macos"))]
            RouteSubmission::Vulkan(submission) => submission.finish(),
        }
        .map_err(CallError::Execution)
    }

    /// Device execution time of the completed submission.
    pub(crate) fn device_seconds(&self) -> Result<f64, CallError> {
        self.wait()?;
        match &*self.route {
            RouteSubmission::Cpu { interval, .. } => Ok(interval.1 - interval.0),
            #[cfg(target_os = "macos")]
            RouteSubmission::Metal(submission) => Ok(submission.device_seconds()),
            RouteSubmission::Cuda(submission) => {
                submission.device_seconds().map_err(CallError::Execution)
            }
            #[cfg(not(target_os = "macos"))]
            RouteSubmission::Vulkan(submission) => {
                submission.device_seconds().map_err(CallError::Execution)
            }
        }
    }
}

/// Encode `list` `repetitions` times as one unit of device work and commit
/// it. `access` names every allocation the work touches exactly once
/// ([`merge_access`]); `retained` keeps the formed functions alive with the
/// submission.
pub(crate) fn submit(
    list: &impl DispatchList,
    access: &[(Arc<Allocation>, bool)],
    retained: Arc<dyn std::any::Any + Send + Sync>,
    repetitions: usize,
) -> Result<NativeSubmission, CallError> {
    if list.count() == 0 {
        return Err(CallError::Workflow(crate::api::WorkflowError::Empty));
    }
    // Host access is excluded while the work is encoded.
    let permits = access
        .iter()
        .map(|(allocation, write)| allocation.acquire_for_device(*write))
        .collect::<Vec<_>>();
    let first = list.dispatch(0, &mut Vec::new()).kernel;
    // Commit and fence recording in one order: allocation fences rely on
    // being recorded in the queue's execution order.
    let _order = first
        .public_device
        .native
        .order
        .lock()
        .expect("native submission order lock is never poisoned");
    let trace = first.public_device.active_trace();
    let encode_start = trace::host_seconds();
    let labels = trace.as_ref().map(|_| launch_labels(list, repetitions));
    let timed = trace
        .as_deref()
        .filter(|sink| sink.detail() == trace::TraceDetail::Launches)
        .map(|sink| (sink, labels.as_ref().map_or(0, Vec::len)));
    let route = Arc::new(encode(first, list, repetitions, timed, &retained)?);
    if let (Some(sink), Some(labels)) = (trace, labels) {
        sink.record(route.clone(), labels, encode_start, trace::host_seconds());
    }
    let completion: Arc<dyn DeviceCompletion> = route.clone();
    for (allocation, write) in access {
        allocation.record_device_use(completion.clone(), *write);
    }
    drop(permits);
    Ok(NativeSubmission {
        route,
        _retained: retained,
    })
}

/// Entry name and launch index of every launch, in encode order.
fn launch_labels(list: &impl DispatchList, repetitions: usize) -> Vec<(String, usize)> {
    let mut labels = Vec::new();
    for _ in 0..repetitions {
        for index in 0..list.count() {
            let dispatch = list.dispatch(index, &mut Vec::new());
            labels.extend(
                (0..dispatch.launches.len()).map(|launch| (dispatch.kernel.name.clone(), launch)),
            );
        }
    }
    labels
}

/// Encode and commit the calls. `timed` (a launch-detail trace and the
/// launch count) encodes every launch in its own timed unit.
fn encode(
    first: &NativePrepared,
    list: &impl DispatchList,
    repetitions: usize,
    timed: Option<(&trace::TraceSink, usize)>,
    retained: &Arc<dyn std::any::Any + Send + Sync>,
) -> Result<RouteSubmission, CallError> {
    let mut buffers = Vec::new();
    match &first.route {
        NativeRoute::Cpu { opened, .. } => {
            let started = trace::host_seconds();
            let mut pointers = Vec::new();
            let outcome = (|| {
                for _ in 0..repetitions {
                    for index in 0..list.count() {
                        buffers.clear();
                        let dispatch = list.dispatch(index, &mut buffers);
                        run_cpu(opened, &dispatch, &buffers, &mut pointers)?;
                    }
                }
                Ok(())
            })();
            Ok(RouteSubmission::Cpu {
                outcome,
                interval: (started, trace::host_seconds()),
            })
        }
        #[cfg(target_os = "macos")]
        NativeRoute::Metal { opened, .. } => {
            type Metal = seismic_metal::Metal;
            type Executor = seismic_metal::MetalExecutor;
            let mut batch = match timed {
                Some((sink, launches)) => seismic_metal::DirectBatch::timed(
                    opened.service(),
                    sink.metal_timestamps(),
                    launches,
                ),
                None => seismic_metal::DirectBatch::new(opened.service()),
            }
            .map_err(CallError::Execution)?;
            let mut typed = Vec::new();
            for _ in 0..repetitions {
                for index in 0..list.count() {
                    buffers.clear();
                    let dispatch = list.dispatch(index, &mut buffers);
                    let NativeRoute::Metal { pipelines, .. } = &dispatch.kernel.route else {
                        unreachable!("one device has one native route");
                    };
                    typed.clear();
                    typed.extend(buffers.iter().map(|(allocation, offset)| {
                        (typed_buffer::<Metal, Executor>(allocation), *offset)
                    }));
                    let scalars = typed_buffer::<Metal, Executor>(&dispatch.kernel.scalars);
                    for (pipeline, launch) in pipelines.iter().zip(dispatch.launches) {
                        let Some(launch) = launch else {
                            batch.skip();
                            continue;
                        };
                        batch
                            .encode(&seismic_metal::DirectLaunch {
                                pipeline,
                                buffers: &typed,
                                words: dispatch.word_bytes,
                                scalar_results: (scalars, 0),
                                threadgroups: launch.groups,
                                threads_per_threadgroup: launch.threads,
                                threadgroup_bytes: launch.shared_bytes,
                            })
                            .map_err(CallError::Execution)?;
                    }
                }
            }
            Ok(RouteSubmission::Metal(batch.commit()))
        }
        NativeRoute::Cuda { opened, .. } => {
            cuda::encode(
                opened.service(),
                &first.public_device.native.cuda_replays,
                list,
                repetitions,
                timed.is_some(),
                retained,
            )
        }
        #[cfg(not(target_os = "macos"))]
        NativeRoute::Vulkan { opened, .. } => vulkan::encode(
            opened.service(),
            &first.public_device.native.vulkan_replays,
            list,
            repetitions,
            timed.map(|(_, launches)| launches),
            retained,
        ),
    }
}

fn run_cpu(
    opened: &Arc<CpuOpened>,
    dispatch: &Dispatch<'_>,
    buffers: &[(&Allocation, u64)],
    pointers: &mut Vec<*mut u8>,
) -> Result<(), ExecutionError> {
    type Cpu = seismic_cpu::Cpu;
    type Executor = seismic_cpu::Executor;
    let NativeRoute::Cpu { launches, .. } = &dispatch.kernel.route else {
        unreachable!("one device has one native route");
    };
    pointers.clear();
    pointers.extend(buffers.iter().map(|(allocation, offset)| {
        let base = typed_buffer::<Cpu, Executor>(allocation).data_pointer();
        // SAFETY: tensor views and scratch placements lie inside their
        // allocations (checked when the views were formed).
        unsafe { base.add(*offset as usize) }
    }));
    let scalars = typed_buffer::<Cpu, Executor>(&dispatch.kernel.scalars)
        .data_pointer()
        .cast::<u64>();
    for (function, launch) in launches.iter().zip(dispatch.launches) {
        let Some(launch) = launch else { continue };
        let invocation = CpuInvocation {
            buffers: pointers,
            representations: dispatch.representations,
            words: dispatch.words,
            scalar_results: scalars,
            groups: launch.groups,
            threads: launch.threads,
        };
        let [x, y, z] = launch.groups;
        let items = x
            .checked_mul(y)
            .and_then(|items| items.checked_mul(z))
            .ok_or_else(|| ExecutionError::SubmissionFailed("native CPU grid overflows".into()))?;
        let body = |index: u64, shared: &mut [u8]| {
            let coordinate = [index % x, (index / x) % y, index / (x * y)];
            function(&invocation, coordinate, shared);
        };
        opened
            .executor()
            .run_native_items(items, launch.shared_bytes, &body)?;
    }
    Ok(())
}

fn validate_output(
    result: usize,
    tensor: &Arc<TensorInner>,
    device: DeviceIdentity,
    representation: RepresentationId,
    extents: &[u64],
    args: &EncodedArgs,
    prior_outputs: &[Arc<TensorInner>],
) -> Result<(), CallError> {
    let descriptor = tensor.descriptor();
    if descriptor.device != device {
        return Err(CallError::Output(OutputError::WrongDevice { result }));
    }
    if descriptor.representation != representation {
        return Err(CallError::Output(OutputError::WrongRepresentation {
            result,
        }));
    }
    if descriptor.extents.len() != extents.len() {
        return Err(CallError::Output(OutputError::ShapeMismatch {
            result,
            axis: descriptor.extents.len().min(extents.len()),
        }));
    }
    for (axis, (actual, expected)) in descriptor.extents.iter().zip(extents).enumerate() {
        if actual != expected {
            return Err(CallError::Output(OutputError::ShapeMismatch {
                result,
                axis,
            }));
        }
    }
    let layout = crate::layout::canonical(representation, extents).map_err(CallError::Execution)?;
    if descriptor.strides != layout.strides || descriptor.byte_len != layout.byte_len {
        return Err(CallError::Output(OutputError::NoncanonicalLayout {
            result,
        }));
    }
    let overlaps = |other: &Arc<TensorInner>| {
        let other = other.descriptor();
        descriptor.allocation == other.allocation
            && descriptor.byte_offset < other.byte_offset.saturating_add(other.byte_len)
            && other.byte_offset < descriptor.byte_offset.saturating_add(descriptor.byte_len)
    };
    if args.tensors().flatten().any(|other| overlaps(other))
        || prior_outputs.iter().any(|other| overlaps(other))
    {
        return Err(CallError::Output(OutputError::IllegalAliasing { result }));
    }
    Ok(())
}

fn tensor_error(error: crate::api::TensorError) -> CallError {
    match error {
        crate::api::TensorError::Execution(error) => CallError::Execution(error),
        other => CallError::Execution(ExecutionError::AllocationFailed(other.to_string())),
    }
}

fn evaluate_compiled(
    expression: &CompiledNat,
    values: &InvocationValues,
) -> Result<u64, CallError> {
    expression.evaluate_u64(values).map_err(|error| {
        CallError::Execution(ExecutionError::SubmissionFailed(format!(
            "native ABI expression failed after invocation validation: {error:?}"
        )))
    })
}

fn native_words(
    schema: &CallSchema,
    arguments: &[ArgumentValue],
    results: &[Option<NativeTensorSpec>],
    values: &InvocationValues,
) -> Result<Vec<u64>, CallError> {
    let mut words = Vec::with_capacity(abi::word_count(schema));
    for dimension in schema.dimensions() {
        match values.get(dimension.symbol) {
            Some(SymbolValue::Nat(value)) => words.push(word(SymbolValue::Nat(value))?),
            _ => panic!("validated invocation omitted a native ABI dimension"),
        }
    }
    for (parameter, argument) in schema.parameters().iter().zip(arguments) {
        match (&parameter.kind, argument) {
            (ParameterKind::Tensor { .. }, ArgumentValue::Tensor(tensor)) => {
                words.extend_from_slice(&tensor.extents);
                words.extend_from_slice(&tensor.strides);
            }
            (ParameterKind::Tensor { .. }, _) => {
                panic!("validated native tensor argument is not a tensor")
            }
            (ParameterKind::Scalar { symbol, .. } | ParameterKind::Index { symbol, .. }, _) => {
                words.push(word(
                    values.get(*symbol).expect("validated scalar disappeared"),
                )?);
            }
            (ParameterKind::Range { start, end, .. }, _) => {
                words.push(word(
                    values
                        .get(*start)
                        .expect("validated range start disappeared"),
                )?);
                words.push(word(
                    values.get(*end).expect("validated range end disappeared"),
                )?);
            }
        }
    }
    for spec in results.iter().flatten() {
        words.extend_from_slice(&spec.extents);
        words.extend_from_slice(&spec.strides);
    }
    Ok(words)
}

fn word(value: SymbolValue) -> Result<u64, CallError> {
    value.try_word64().map_err(|error| {
        CallError::Execution(ExecutionError::ConstructionContradiction(format!(
            "native ABI quantity does not fit its word: {error:?}"
        )))
    })
}

fn scalar_value(dtype: seismic_lang::types::DType, value: u64) -> ArgumentValue {
    match dtype {
        seismic_lang::types::DType::F32 => ArgumentValue::F32(f32::from_bits(value as u32)),
        seismic_lang::types::DType::F16 => ArgumentValue::F16(value as u16),
        seismic_lang::types::DType::BF16 => ArgumentValue::BF16(value as u16),
        seismic_lang::types::DType::I32 => ArgumentValue::I32(value as u32 as i32),
        seismic_lang::types::DType::U32 => ArgumentValue::U32(value as u32),
        seismic_lang::types::DType::Bool => ArgumentValue::Bool(value != 0),
    }
}

fn read_word(bytes: &[u8], word: usize) -> u64 {
    let start = word * 8;
    u64::from_le_bytes(
        bytes[start..start + 8]
            .try_into()
            .expect("native scalar word"),
    )
}

/// Timing of one prepared native implementation over rotated argument sets.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    /// Per-call device seconds of every sample.
    pub samples: Vec<f64>,
    pub median: f64,
    /// Median absolute deviation of the samples.
    pub deviation: f64,
    /// Calls per sample submission.
    pub repetitions: usize,
    /// Distinct bytes the rotation reads and writes.
    pub rotation_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasureOptions {
    /// Measured submissions.
    pub samples: usize,
    /// Minimum device time of one sample; sets the repetition count.
    pub min_sample_seconds: f64,
}

impl Default for MeasureOptions {
    fn default() -> Self {
        Self {
            samples: 7,
            min_sample_seconds: 0.002,
        }
    }
}

fn median(samples: &[f64]) -> f64 {
    median_of(samples.to_vec())
}

fn median_of(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};

    /// A module built without seismic-build (which rejects this at build
    /// time) reaches preparation, which rejects it before compiling.
    #[test]
    #[ignore = "requires Metal device"]
    fn metal_preparation_rejects_implementations_beyond_the_buffer_table() {
        let parameters = (0..29)
            .map(|index| format!("x{index}: &tensor[N] f32"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut module = check_source(SourceSet::new(vec![SourceFile {
            path: "wide.seismic".into(),
            text: format!("fn wide[N]({parameters}) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x0[i]\n    return output\n\nnative wide for metal from \"wide.metal\":\n    launch wide:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n"),
        }]))
        .expect("wide source checks");
        let entry = module.entry_named("wide").expect("wide entry");
        module
            .capture_native_asset(entry, BackendName::Metal, "kernel void wide() {}\n".into())
            .expect("asset capture");
        let device = crate::devices::Catalog::discover()
            .expect("device discovery")
            .open_backend(BackendName::Metal)
            .expect("Metal device");
        match NativePrepared::prepare(
            &device,
            &module,
            entry,
            ElementBindings::default(),
            NativeSpecialization::new(),
            None,
        ) {
            Err(PrepareError::Preparation(PreparationError::NativeBufferSlots {
                entry,
                slots,
                limit,
            })) => {
                assert_eq!(entry, "wide");
                assert_eq!(slots, 32);
                assert_eq!(limit, 31);
            }
            Err(other) => panic!("unexpected preparation error: {other:?}"),
            Ok(_) => panic!("32 Metal buffer slots were prepared"),
        }
    }
}
