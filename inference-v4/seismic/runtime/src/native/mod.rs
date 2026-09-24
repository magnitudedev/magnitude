//! The explicitly selected direct-native route
//! (`docs/seismic/language/functions-and-capabilities.md`).
//!
//! A checked entry's authored implementation for the opened device's backend
//! is formed under one [`NativeSpecialization`] and executed without a
//! compiler plan. Metal and CUDA sources receive the generated prefix of
//! [`abi`]; CPU implementations are Rust compiled into the binary and reached
//! through [`cpu`].
//!
//! Submission is asynchronous. Launches of one submission share one serial
//! encoder (Metal), one stream (CUDA) or the worker pool (CPU); the device
//! queue orders submissions, and allocation fences order host access after
//! them.

pub(crate) mod abi;
pub mod cpu;
pub mod graph;
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
    CheckedModule, NativeImplementation, NativeNatExpr, NativeSpecialization,
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
    pub(crate) alignment: u64,
}

/// Evaluated geometry of one launch of one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaunchGeometry {
    pub(crate) groups: [u64; 3],
    pub(crate) threads: [u64; 3],
    pub(crate) shared_bytes: u64,
}

/// Alignment of every scratch buffer.
pub(crate) const SCRATCH_ALIGNMENT: u64 = 256;

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
    results: Vec<NativeResult>,
    scalar_words: usize,
    /// Scalar-result slots. Graph nodes publish no scalars, so only
    /// standalone calls write them; `standalone` serializes those calls.
    scalars: Arc<Allocation>,
    standalone: Mutex<()>,
    scalar_bytes: u64,
    artifact: NativeArtifactIdentity,
    route: NativeRoute,
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
        let name = module
            .entries()
            .iter()
            .find(|candidate| candidate.id == entry)
            .expect("entry belongs to its module")
            .name
            .clone();
        let implementation = module
            .native_implementation(entry, backend)
            .cloned()
            .ok_or_else(|| {
                preparation(format!(
                    "`{name}` has no native implementation for `{}`",
                    backend.as_str()
                ))
            })?;
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
                if words * 8 > seismic_metal::DIRECT_WORD_BYTES_LIMIT {
                    return Err(preparation(format!(
                        "`{name}` needs {} argument bytes, beyond Metal's setBytes limit",
                        words * 8
                    )));
                }
                let source = abi::render_source(
                    abi::Dialect::Metal,
                    schema,
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
                    schema,
                    &bindings,
                    &implementation,
                    &specialization,
                    asset(BackendName::Cuda)?,
                );
                digest.update(source.as_bytes());
                let facts = opened.device_description().facts();
                let architecture = u32::from(facts.compute_capability.major) * 10
                    + u32::from(facts.compute_capability.minor);
                let module = seismic_cuda::direct::DirectModule::compile(
                    opened.service(),
                    &source,
                    &format!("{name}.cu"),
                    architecture,
                    &kernels,
                )
                .map_err(compilation)?;
                let (major, minor) = seismic_cuda::direct::nvrtc_version().map_err(compilation)?;
                (
                    NativeRoute::Cuda {
                        opened: opened.clone(),
                        module,
                    },
                    format!(
                        "cuda;nvrtc {major}.{minor};sm_{architecture};driver {}",
                        facts.driver_api.0
                    ),
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
            results,
            scalar_words,
            scalars,
            standalone: Mutex::new(()),
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
    /// Fixed device storage of this prepared implementation: its
    /// scalar-result slots.
    pub(crate) fn invocation_workspace_bytes(&self) -> u64 {
        self.scalar_bytes
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

    fn evaluate(
        &self,
        expression: &NativeNatExpr,
        values: &InvocationValues,
    ) -> Result<u64, CallError> {
        expression
            .evaluate(&|name| self.dimension(values, name), &|name| {
                self.specialization.param(name)
            })
            .map_err(|error| {
                CallError::Execution(ExecutionError::SubmissionFailed(format!(
                    "native expression of `{}`: {error}",
                    self.name
                )))
            })
    }

    /// Geometry of every launch for one invocation, checked against the
    /// formed functions and device limits.
    pub(crate) fn launches(
        &self,
        values: &InvocationValues,
    ) -> Result<Vec<LaunchGeometry>, CallError> {
        let mut geometry = Vec::with_capacity(self.implementation.launches.len());
        for (ordinal, launch) in self.implementation.launches.iter().enumerate() {
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
            }
            geometry.push(LaunchGeometry {
                groups,
                threads,
                shared_bytes,
            });
        }
        Ok(geometry)
    }

    /// Bytes of every scratch buffer for one invocation.
    pub(crate) fn scratch_bytes(&self, values: &InvocationValues) -> Result<Vec<u64>, CallError> {
        self.implementation
            .scratch
            .iter()
            .map(|scratch| {
                self.evaluate(&scratch.bytes, values)
                    .map(|bytes| bytes.max(1))
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

    /// Describe a node without allocating: result specs and scratch sizes.
    /// Graph planning feeds descriptors, including virtual result edges,
    /// through the same contract as a direct call.
    pub(crate) fn describe(
        &self,
        arguments: &[ArgumentValue],
    ) -> Result<(Vec<Option<NativeTensorSpec>>, Vec<u64>), CallError> {
        let values = self.validate(arguments)?;
        self.launches(&values)?;
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
                            alignment: layout.alignment,
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, CallError>>()?;
        Ok((results, self.scratch_bytes(&values)?))
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
            alignment: layout.alignment,
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

    /// Check an attached node and produce its executable call.
    pub(crate) fn bind(
        self: &Arc<Self>,
        args: EncodedArgs,
        outputs: Vec<Arc<TensorInner>>,
        scratch: Vec<ScratchView>,
    ) -> Result<NativeBoundCall, CallError> {
        let values = self.validate(&args.values())?;
        let expected = self
            .result_extents(&values)?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if expected.len() != outputs.len() {
            return Err(CallError::Output(OutputError::Count {
                expected: expected.len(),
                actual: outputs.len(),
            }));
        }
        for (index, ((representation, extents), tensor)) in
            expected.iter().zip(&outputs).enumerate()
        {
            validate_output(
                index,
                tensor,
                self.device,
                *representation,
                extents,
                &args,
                &outputs[..index],
            )?;
        }
        let sizes = self.scratch_bytes(&values)?;
        if sizes.len() != scratch.len()
            || sizes
                .iter()
                .zip(&scratch)
                .any(|(bytes, view)| view.offset + bytes > view.allocation.bytes())
        {
            return Err(CallError::Execution(ExecutionError::SubmissionFailed(
                "native scratch placement does not cover the call's scratch".into(),
            )));
        }
        self.seal(args, outputs, scratch, &values)
    }

    fn seal(
        self: &Arc<Self>,
        args: EncodedArgs,
        results: Vec<Arc<TensorInner>>,
        scratch: Vec<ScratchView>,
        values: &InvocationValues,
    ) -> Result<NativeBoundCall, CallError> {
        let words = native_words(self.schema(), &args, &results, values)?;
        Ok(NativeBoundCall {
            launches: self.launches(values)?,
            kernel: self.clone(),
            args,
            results,
            scratch,
            words,
        })
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

    /// Allocate a call's results and scratch without submitting it.
    pub(crate) fn prepare_call(
        self: &Arc<Self>,
        args: EncodedArgs,
        outputs: Option<EncodedOutputs>,
    ) -> Result<NativeBoundCall, CallError> {
        let values = self.validate(&args.values())?;
        let expected = self
            .result_extents(&values)?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let supplied = outputs.map(EncodedOutputs::into_tensors);
        if let Some(outputs) = &supplied {
            if outputs.len() != expected.len() {
                return Err(CallError::Output(OutputError::Count {
                    expected: expected.len(),
                    actual: outputs.len(),
                }));
            }
        }
        let mut results = Vec::with_capacity(expected.len());
        for (index, (representation, extents)) in expected.into_iter().enumerate() {
            let tensor = match &supplied {
                Some(outputs) => {
                    let tensor = outputs[index].clone();
                    validate_output(
                        index,
                        &tensor,
                        self.device,
                        representation,
                        &extents,
                        &args,
                        &results,
                    )?;
                    tensor
                }
                None => Arc::new(
                    TensorInner::zeros(&self.public_device, representation, &extents)
                        .map_err(tensor_error)?,
                ),
            };
            results.push(tensor);
        }
        let scratch = self
            .scratch_bytes(&values)?
            .into_iter()
            .map(|bytes| {
                self.public_device
                    .allocate(bytes, SCRATCH_ALIGNMENT)
                    .map(|allocation| ScratchView {
                        allocation,
                        offset: 0,
                    })
                    .map_err(CallError::Execution)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.seal(args, results, scratch, &values)
    }

    fn call_with(
        self: &Arc<Self>,
        args: EncodedArgs,
        outputs: Option<EncodedOutputs>,
        commit: impl FnOnce(),
    ) -> Result<DecodedResults, CallError> {
        let call = self.prepare_call(args, outputs)?;
        // The scalar-result slots are shared by standalone calls of this
        // implementation; the guard covers their reset, execution and read.
        let _standalone = self
            .standalone
            .lock()
            .expect("native standalone-call lock poisoned");
        let scalars = &self.scalars;
        {
            let _host = scalars.acquire(true);
            write_zeros(scalars.storage(), self.scalar_bytes).map_err(CallError::Execution)?;
        }
        commit();
        let call = Arc::new(call);
        let submission = submit(std::slice::from_ref(&call), 1)?;
        submission.wait()?;
        let mut scalar_bytes = vec![0u8; self.scalar_words * 8];
        scalars
            .acquire(false)
            .read(&scalars, 0, &mut scalar_bytes)
            .map_err(CallError::Execution)?;
        let call = Arc::try_unwrap(call)
            .unwrap_or_else(|_| panic!("completed native submission retained its call"));
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

/// A scratch buffer's placement.
#[derive(Clone)]
pub(crate) struct ScratchView {
    pub(crate) allocation: Arc<Allocation>,
    pub(crate) offset: u64,
}

/// One validated call ready for submission: arguments, results, scratch,
/// argument words and launch geometry are fixed.
pub(crate) struct NativeBoundCall {
    kernel: Arc<NativePrepared>,
    args: EncodedArgs,
    results: Vec<Arc<TensorInner>>,
    scratch: Vec<ScratchView>,
    words: Vec<u64>,
    launches: Vec<LaunchGeometry>,
}

impl NativeBoundCall {
    /// Every allocation the call touches, with whether it writes it.
    fn access(&self) -> Vec<(Arc<Allocation>, bool)> {
        let mut access = collect_native_access(self.kernel.schema(), &self.args);
        access.extend(
            self.results
                .iter()
                .map(|tensor| (tensor.allocation().clone(), true)),
        );
        access.extend(
            self.scratch
                .iter()
                .map(|view| (view.allocation.clone(), true)),
        );
        access.push((self.kernel.scalars.clone(), true));
        access
    }

    /// Buffers in ABI order with their byte offsets.
    fn buffers(&self) -> Vec<(&Arc<Allocation>, u64)> {
        let mut buffers = Vec::new();
        for (ordinal, parameter) in self.kernel.schema().parameters().iter().enumerate() {
            if matches!(parameter.kind, ParameterKind::Tensor { .. }) {
                let tensor = self
                    .args
                    .tensor(ordinal)
                    .expect("validated native tensor argument disappeared");
                buffers.push((tensor.allocation(), tensor.byte_offset()));
            }
        }
        for tensor in &self.results {
            buffers.push((tensor.allocation(), tensor.byte_offset()));
        }
        for view in &self.scratch {
            buffers.push((&view.allocation, view.offset));
        }
        buffers
    }

    /// Registry name of each buffer's representation, in ABI order.
    fn representations(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        for (ordinal, parameter) in self.kernel.schema().parameters().iter().enumerate() {
            if matches!(parameter.kind, ParameterKind::Tensor { .. }) {
                let tensor = self
                    .args
                    .tensor(ordinal)
                    .expect("validated native tensor argument disappeared");
                names.push(
                    seismic_lang::registry::representation_info(tensor.representation()).name,
                );
            }
        }
        for tensor in &self.results {
            names.push(seismic_lang::registry::representation_info(tensor.representation()).name);
        }
        names.extend(self.scratch.iter().map(|_| "bytes"));
        names
    }

    fn word_bytes(&self) -> Vec<u8> {
        self.words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }
}

enum RouteSubmission {
    Cpu {
        outcome: Result<(), ExecutionError>,
        seconds: f64,
    },
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::DirectSubmission),
    Cuda(seismic_cuda::direct::DirectSubmission),
}

impl DeviceCompletion for RouteSubmission {
    fn is_complete(&self) -> bool {
        match self {
            Self::Cpu { .. } => true,
            #[cfg(target_os = "macos")]
            Self::Metal(submission) => submission.is_complete(),
            Self::Cuda(submission) => submission.is_complete(),
        }
    }
    fn wait_complete(&self) {
        match self {
            Self::Cpu { .. } => {}
            #[cfg(target_os = "macos")]
            Self::Metal(submission) => submission.wait_complete(),
            Self::Cuda(submission) => submission.wait_complete(),
        }
    }
}

/// Submitted native work. It retains the formed functions it executes;
/// allocation fences keep every touched storage alive and order host access
/// until the work completes.
pub(crate) struct NativeSubmission {
    route: Arc<RouteSubmission>,
    _kernels: Vec<Arc<NativePrepared>>,
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
        }
        .map_err(CallError::Execution)
    }

    /// Device execution time of the completed submission.
    pub(crate) fn device_seconds(&self) -> Result<f64, CallError> {
        self.wait()?;
        match &*self.route {
            RouteSubmission::Cpu { seconds, .. } => Ok(*seconds),
            #[cfg(target_os = "macos")]
            RouteSubmission::Metal(submission) => Ok(submission.device_seconds()),
            RouteSubmission::Cuda(submission) => {
                submission.device_seconds().map_err(CallError::Execution)
            }
        }
    }
}

/// Submit calls, in order, `repetitions` times, as one unit of device work.
/// Every call must belong to one device.
pub(crate) fn submit(
    calls: &[Arc<NativeBoundCall>],
    repetitions: usize,
) -> Result<NativeSubmission, CallError> {
    let first = calls
        .first()
        .ok_or(CallError::Workflow(crate::api::WorkflowError::Empty))?;
    if calls
        .iter()
        .any(|call| call.kernel.device != first.kernel.device)
    {
        return Err(CallError::Workflow(
            crate::api::WorkflowError::NativeGraphSlotMismatch,
        ));
    }
    let mut access: BTreeMap<u64, (Arc<Allocation>, bool)> = BTreeMap::new();
    for call in calls {
        for (allocation, write) in call.access() {
            access
                .entry(allocation.identity())
                .and_modify(|(_, current)| *current |= write)
                .or_insert((allocation, write));
        }
    }
    // Host access is excluded while the work is encoded; ordered by identity
    // so concurrent submitters cannot deadlock.
    let permits = access
        .values()
        .map(|(allocation, write)| allocation.acquire_for_device(*write))
        .collect::<Vec<_>>();
    let route = Arc::new(encode(first, calls, repetitions)?);
    let completion: Arc<dyn DeviceCompletion> = route.clone();
    for (allocation, write) in access.values() {
        allocation.record_device_use(completion.clone(), *write);
    }
    drop(permits);
    Ok(NativeSubmission {
        route,
        _kernels: calls.iter().map(|call| call.kernel.clone()).collect(),
    })
}

fn encode(
    first: &Arc<NativeBoundCall>,
    calls: &[Arc<NativeBoundCall>],
    repetitions: usize,
) -> Result<RouteSubmission, CallError> {
    match &first.kernel.route {
        NativeRoute::Cpu { opened, .. } => {
            let started = std::time::Instant::now();
            let outcome = (|| {
                for _ in 0..repetitions {
                    for call in calls {
                        run_cpu(opened, call)?;
                    }
                }
                Ok(())
            })();
            Ok(RouteSubmission::Cpu {
                outcome,
                seconds: started.elapsed().as_secs_f64(),
            })
        }
        #[cfg(target_os = "macos")]
        NativeRoute::Metal { opened, .. } => {
            type Metal = seismic_metal::Metal;
            type Executor = seismic_metal::MetalExecutor;
            let mut batch =
                seismic_metal::DirectBatch::new(opened.service()).map_err(CallError::Execution)?;
            for _ in 0..repetitions {
                for call in calls {
                    let NativeRoute::Metal { pipelines, .. } = &call.kernel.route else {
                        unreachable!("one device has one native route");
                    };
                    let owned = call
                        .buffers()
                        .into_iter()
                        .map(|(allocation, offset)| {
                            (typed_buffer::<Metal, Executor>(allocation), offset)
                        })
                        .collect::<Vec<_>>();
                    let buffers = owned
                        .iter()
                        .map(|(buffer, offset)| (buffer, *offset))
                        .collect::<Vec<_>>();
                    let scalars = typed_buffer::<Metal, Executor>(&call.kernel.scalars);
                    let words = call.word_bytes();
                    for (pipeline, launch) in pipelines.iter().zip(&call.launches) {
                        batch
                            .encode(&seismic_metal::DirectLaunch {
                                pipeline,
                                buffers: &buffers,
                                words: &words,
                                scalar_results: (&scalars, 0),
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
            type Cuda = seismic_cuda::Cuda;
            type Executor = seismic_cuda::Executor;
            let mut batch = seismic_cuda::direct::DirectBatch::new(opened.service())
                .map_err(CallError::Execution)?;
            for _ in 0..repetitions {
                for call in calls {
                    let NativeRoute::Cuda { module, .. } = &call.kernel.route else {
                        unreachable!("one device has one native route");
                    };
                    let owned = call
                        .buffers()
                        .into_iter()
                        .map(|(allocation, offset)| {
                            (typed_buffer::<Cuda, Executor>(allocation), offset)
                        })
                        .collect::<Vec<_>>();
                    let buffers = owned
                        .iter()
                        .map(|(buffer, offset)| (buffer, *offset))
                        .collect::<Vec<_>>();
                    let scalars = typed_buffer::<Cuda, Executor>(&call.kernel.scalars);
                    let words = call.word_bytes();
                    for (function, launch) in call.launches.iter().enumerate() {
                        batch
                            .launch(&seismic_cuda::direct::DirectLaunch {
                                module,
                                function,
                                buffers: &buffers,
                                words: &words,
                                scalar_results: (&scalars, 0),
                                grid: launch.groups,
                                block: launch.threads,
                                shared_bytes: launch.shared_bytes,
                            })
                            .map_err(CallError::Execution)?;
                    }
                }
            }
            batch
                .commit()
                .map(RouteSubmission::Cuda)
                .map_err(CallError::Execution)
        }
    }
}

fn run_cpu(opened: &Arc<CpuOpened>, call: &NativeBoundCall) -> Result<(), ExecutionError> {
    type Cpu = seismic_cpu::Cpu;
    type Executor = seismic_cpu::Executor;
    let NativeRoute::Cpu { launches, .. } = &call.kernel.route else {
        unreachable!("one device has one native route");
    };
    let pointers = call
        .buffers()
        .into_iter()
        .map(|(allocation, offset)| {
            let base = typed_buffer::<Cpu, Executor>(allocation).data_pointer();
            // SAFETY: tensor views and scratch placements lie inside their
            // allocations (checked when the views were formed).
            unsafe { base.add(offset as usize) }
        })
        .collect::<Vec<_>>();
    let representations = call.representations();
    let scalars = typed_buffer::<Cpu, Executor>(&call.kernel.scalars)
        .data_pointer()
        .cast::<u64>();
    for (function, launch) in launches.iter().zip(&call.launches) {
        let invocation = CpuInvocation {
            buffers: &pointers,
            representations: &representations,
            words: &call.words,
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
    args: &EncodedArgs,
    results: &[Arc<TensorInner>],
    values: &InvocationValues,
) -> Result<Vec<u64>, CallError> {
    let mut words = Vec::with_capacity(abi::word_count(schema));
    for dimension in schema.dimensions() {
        match values.get(dimension.symbol) {
            Some(SymbolValue::Nat(value)) => words.push(word(SymbolValue::Nat(value))?),
            _ => panic!("validated invocation omitted a native ABI dimension"),
        }
    }
    for (ordinal, parameter) in schema.parameters().iter().enumerate() {
        match &parameter.kind {
            ParameterKind::Tensor { .. } => {
                let tensor = args.tensor(ordinal).expect("validated tensor disappeared");
                words.extend_from_slice(tensor.extents());
                words.extend_from_slice(tensor.strides());
            }
            ParameterKind::Scalar { symbol, .. } | ParameterKind::Index { symbol, .. } => {
                words.push(word(
                    values.get(*symbol).expect("validated scalar disappeared"),
                )?);
            }
            ParameterKind::Range { start, end, .. } => {
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
    for tensor in results {
        words.extend_from_slice(tensor.extents());
        words.extend_from_slice(tensor.strides());
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

impl NativePrepared {
    /// Measure calls cycling through `rotation`. Results and scratch of each
    /// argument set are allocated once and reused by every repetition.
    pub(crate) fn measure(
        self: &Arc<Self>,
        rotation: Vec<EncodedArgs>,
        options: &MeasureOptions,
    ) -> Result<Measurement, CallError> {
        if rotation.is_empty() || options.samples == 0 {
            return Err(CallError::Workflow(crate::api::WorkflowError::Empty));
        }
        let calls = rotation
            .into_iter()
            .map(|args| self.prepare_call(args, None).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let mut distinct = BTreeMap::new();
        for call in &calls {
            for (allocation, _) in call.access() {
                distinct.insert(allocation.identity(), allocation.bytes());
            }
        }
        let rotation_bytes = distinct.values().sum();
        // Warm-up, which also calibrates the repetition count.
        let warm = submit(&calls, 1)?.device_seconds()? / calls.len() as f64;
        let repetitions = if warm > 0.0 {
            ((options.min_sample_seconds / warm).ceil() as usize)
                .div_ceil(calls.len())
                .max(1)
        } else {
            1
        };
        let mut samples = Vec::with_capacity(options.samples);
        for _ in 0..options.samples {
            let seconds = submit(&calls, repetitions)?.device_seconds()?;
            samples.push(seconds / (repetitions * calls.len()) as f64);
        }
        let median = median(&samples);
        let deviation = median_of(
            samples
                .iter()
                .map(|sample| (sample - median).abs())
                .collect(),
        );
        Ok(Measurement {
            samples,
            median,
            deviation,
            repetitions: repetitions * calls.len(),
            rotation_bytes,
        })
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
