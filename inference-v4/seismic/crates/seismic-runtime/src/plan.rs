//! Explicit compiled-plan preparation (package R1).
//!
//! One entry compiles once per `(entry, specialization domain, precision,
//! evidence catalog)`. `compile_entry` runs the whole semantic -> logical ->
//! plan-space -> physical-plan -> native-artifact pipeline eagerly and stores
//! a non-optional sealed artifact; there is no first-invocation compile path
//! and no optional kernel cache. Compilation identity is the specialization
//! domain's identity bytes, so repeated requests for the same domain share
//! one sealed artifact — dispatch among valid compiled artifacts, never
//! recompilation.

use crate::invocation::{self, Bindings, PreparedInvocation};
use crate::{Device, DeviceHandle};
use seismic_compiler::pipeline::{self, CompileFailure, SystemReport};
use seismic_compiler::planning::Budget;
use seismic_lang::abi::RangeEndpoint;
use seismic_lang::logical::LogicalProgram;
use seismic_lang::logical::specialization::SpecializationDomain;
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::sir::Program;
use seismic_lang::types::{DType, ValuePath};
use seismic_realization::failure::ExecutionFailure;
use seismic_realization::ids::NativeFactIx;
use seismic_realization::invocation::InvocationContract;
use seismic_realization::numerics::NumericalEvidence;
use seismic_realization::physical::PhysicalPlan;
use seismic_cpu::CpuDialect;
use std::rc::Rc;
use std::sync::Arc;

/// One compiler-owned scalar result. Range endpoints share a semantic path
/// and remain explicitly distinguished by `endpoint`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarResult {
    pub path: ValuePath,
    pub endpoint: Option<RangeEndpoint>,
    pub dtype: DType,
    pub value: f64,
}

/// One owned result plane of an invocation, in ABI result order: the buffer
/// preparation allocated and the kernels wrote through the executor's
/// binding path.
#[derive(Clone)]
pub struct ResultPlane {
    pub path: Vec<u32>,
    pub plane: String,
    pub buffer: crate::Buffer,
}

/// The outputs of one executed invocation.
#[derive(Clone, Default)]
pub struct InvocationResults {
    pub planes: Vec<ResultPlane>,
    pub scalars: Vec<ScalarResult>,
}

/// The sealed compiled artifact. Fields are private and construction happens
/// at exactly one point — `PlanCompiler::compile_entry` wrapping the compiler
/// pipeline's `Compiled{logical, physical, native}` result — so the physical
/// plan and native artifact of every arm are the one-to-one pairing the
/// pipeline produced, and no caller can assemble mismatched halves. The CUDA
/// native artifact is thread-affine, so no arm requires `Send` or `Sync`.
pub struct CompiledArtifact {
    backend: BackendArtifact,
}

/// The per-backend payload of a sealed artifact: the device the pipeline ran
/// on (where execution owns state: the CPU worker pool, the CUDA driver
/// context) and the paired plan/native halves. Private to this module.
enum BackendArtifact {
    Cpu {
        device: Rc<crate::CpuDevice>,
        physical: Arc<PhysicalPlan<CpuDialect>>,
        native: seismic_cpu::NativeArtifact,
    },
    #[cfg(target_os = "macos")]
    Metal {
        physical: Arc<PhysicalPlan<seismic_metal::intrinsics::MetalDialect>>,
        native: Arc<seismic_metal::native::NativeArtifact>,
    },
    Cuda {
        device: Rc<seismic_cuda::runtime::Device>,
        physical: Arc<PhysicalPlan<seismic_cuda::Dialect>>,
        native: Arc<seismic_cuda::native::NativeArtifact>,
    },
}

/// Everything one backend's executor needs, retrieved as one paired bundle
/// from the sealed artifact: the device handle, the physical plan, and the
/// native artifact of the same compilation. Total — one arm per backend, no
/// mismatchable halves.
pub(crate) enum SealedBackend<'a> {
    Cpu {
        device: &'a Rc<crate::CpuDevice>,
        physical: &'a Arc<PhysicalPlan<CpuDialect>>,
        native: &'a seismic_cpu::NativeArtifact,
    },
    #[cfg(target_os = "macos")]
    Metal {
        native: &'a Arc<seismic_metal::native::NativeArtifact>,
    },
    Cuda {
        device: &'a Rc<seismic_cuda::runtime::Device>,
        physical: &'a Arc<PhysicalPlan<seismic_cuda::Dialect>>,
        native: &'a Arc<seismic_cuda::native::NativeArtifact>,
    },
}

impl CompiledArtifact {
    /// The only constructor, private to this module and called only where
    /// `compile_entry` wraps the pipeline result: the pairing of device,
    /// physical plan, and native artifact is established here and nowhere
    /// else.
    fn seal(backend: BackendArtifact) -> CompiledArtifact {
        CompiledArtifact { backend }
    }

    /// The invocation contract of the sealed physical plan.
    pub fn contract(&self) -> &InvocationContract {
        match &self.backend {
            BackendArtifact::Cpu { physical, .. } => physical.contract(),
            #[cfg(target_os = "macos")]
            BackendArtifact::Metal { physical, .. } => physical.contract(),
            BackendArtifact::Cuda { physical, .. } => physical.contract(),
        }
    }

    /// Launches in the sealed plan; constant for a sealed artifact.
    pub fn kernel_count(&self) -> usize {
        match &self.backend {
            BackendArtifact::Cpu { physical, .. } => physical.launches().len(),
            #[cfg(target_os = "macos")]
            BackendArtifact::Metal { physical, .. } => physical.launches().len(),
            BackendArtifact::Cuda { physical, .. } => physical.launches().len(),
        }
    }

    /// One reflected native fact, folded and validated at assembly; dense and
    /// in-bounds by construction. This is the `native` closure source of
    /// `InvocationContract::evaluate`.
    pub fn native_fact(&self, index: NativeFactIx) -> u64 {
        match &self.backend {
            BackendArtifact::Cpu { native, .. } => native.native_fact(index),
            #[cfg(target_os = "macos")]
            BackendArtifact::Metal { native, .. } => native.native_fact(index),
            BackendArtifact::Cuda { native, .. } => native.native_fact(index),
        }
    }

    /// The sealed physical plan when this artifact compiled for the CPU.
    pub fn physical_cpu(&self) -> Option<&Arc<PhysicalPlan<CpuDialect>>> {
        match &self.backend {
            BackendArtifact::Cpu { physical, .. } => Some(physical),
            #[cfg(target_os = "macos")]
            BackendArtifact::Metal { .. } => None,
            BackendArtifact::Cuda { .. } => None,
        }
    }

    /// The sealed physical plan when this artifact compiled for Metal.
    #[cfg(target_os = "macos")]
    pub fn physical_metal(
        &self,
    ) -> Option<&Arc<PhysicalPlan<seismic_metal::intrinsics::MetalDialect>>> {
        match &self.backend {
            BackendArtifact::Metal { physical, .. } => Some(physical),
            BackendArtifact::Cpu { .. } | BackendArtifact::Cuda { .. } => None,
        }
    }

    /// The sealed physical plan when this artifact compiled for CUDA.
    pub fn physical_cuda(&self) -> Option<&Arc<PhysicalPlan<seismic_cuda::Dialect>>> {
        match &self.backend {
            BackendArtifact::Cuda { physical, .. } => Some(physical),
            BackendArtifact::Cpu { .. } => None,
            #[cfg(target_os = "macos")]
            BackendArtifact::Metal { .. } => None,
        }
    }

    /// The paired executor bundle of this artifact's backend. The only
    /// execution-side retrieval; every arm carries its own device, physical
    /// plan, and native artifact together.
    pub(crate) fn backend(&self) -> SealedBackend<'_> {
        match &self.backend {
            BackendArtifact::Cpu {
                device,
                physical,
                native,
            } => SealedBackend::Cpu {
                device,
                physical,
                native,
            },
            #[cfg(target_os = "macos")]
            BackendArtifact::Metal { native, .. } => SealedBackend::Metal { native },
            BackendArtifact::Cuda {
                device,
                physical,
                native,
            } => SealedBackend::Cuda {
                device,
                physical,
                native,
            },
        }
    }
}

/// One eagerly compiled entry: the sealed artifact, the interface parameter
/// names its ABI buffer ordinals resolve to, and the device it compiled for.
/// Clones share one sealed compilation.
#[derive(Clone)]
pub struct CompiledPlan {
    artifact: Arc<CompiledArtifact>,
    /// Interface parameter names in ABI parameter-ordinal order (the root
    /// names `Bindings::buffer` receives).
    parameters: Rc<Vec<String>>,
    device: Device,
}

impl CompiledPlan {
    /// Validate one invocation against the sealed contract and allocate its
    /// result planes. The only public constructor of `PreparedInvocation`.
    pub fn prepare(
        &self,
        bindings: &dyn Bindings,
    ) -> Result<PreparedInvocation, ExecutionFailure> {
        invocation::prepare(
            &self.device,
            Arc::clone(&self.artifact),
            &self.parameters,
            bindings,
        )
    }

    pub fn contract(&self) -> &InvocationContract {
        self.artifact.contract()
    }

    /// Launches in the sealed plan; constant for this compiled plan.
    pub fn kernel_count(&self) -> usize {
        self.artifact.kernel_count()
    }

    /// The solver's estimated cost of the selected assignment.
    pub fn estimated_cost(&self) -> u64 {
        match self.artifact.backend() {
            SealedBackend::Cpu { physical, .. } => physical.estimated_cost(),
            #[cfg(target_os = "macos")]
            SealedBackend::Metal { native, .. } => native.estimated_cost(),
            SealedBackend::Cuda { physical, .. } => physical.estimated_cost(),
        }
    }

    /// Whether the solver proved the selected assignment optimal within its
    /// budget.
    pub fn optimal(&self) -> bool {
        match self.artifact.backend() {
            SealedBackend::Cpu { physical, .. } => physical.optimal(),
            #[cfg(target_os = "macos")]
            SealedBackend::Metal { native, .. } => native.optimal(),
            SealedBackend::Cuda { physical, .. } => physical.optimal(),
        }
    }

    pub fn shares_compilation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.artifact, &other.artifact)
    }

    pub fn artifact(&self) -> &CompiledArtifact {
        &self.artifact
    }

    pub fn parameters(&self) -> &[String] {
        &self.parameters
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Explicit search budget. Implementation choices are compiler-owned; the
/// backend and its capacities come from the device the plan compiles for.
#[derive(Clone, Debug)]
pub struct Settings {
    pub budget: Budget,
    /// Observable numerical contract; part of every compiled entry's workload
    /// identity.
    pub precision: PrecisionPolicy,
    /// Whole-program numerical evidence keyed to complete physical
    /// assignments.
    pub numerical_evidence: Vec<NumericalEvidence>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            budget: Budget::default(),
            precision: PrecisionPolicy::default(),
            numerical_evidence: Vec::new(),
        }
    }
}

/// One eagerly compiled entry of one compiler.
struct CompiledEntry {
    identity: Vec<u8>,
    artifact: Arc<CompiledArtifact>,
    parameters: Rc<Vec<String>>,
}

/// Compiles linked entries through the unified pipeline only. One compiler
/// owns an immutable settings snapshot, so cached entries cannot observe a
/// catalog change.
pub struct PlanCompiler<'a> {
    device: &'a Device,
    program: Rc<Program>,
    settings: Settings,
    entries: Vec<CompiledEntry>,
}

impl<'a> PlanCompiler<'a> {
    pub fn new(device: &'a Device, program: &'a Program, settings: Settings) -> Self {
        Self {
            device,
            program: Rc::new(program.clone()),
            settings,
            entries: Vec::new(),
        }
    }

    pub fn settings(&self) -> Settings {
        self.settings.clone()
    }

    pub fn program(&self) -> &Program {
        &self.program
    }

    pub fn device(&self) -> &Device {
        self.device
    }

    /// Entries compiled so far (each eagerly sealed at its first request).
    pub fn kernel_count(&self) -> usize {
        self.entries.len()
    }

    /// Compile one entry under one complete specialization domain, eagerly:
    /// specialization, physical planning, and native compilation complete
    /// before the plan is returned. A repeated domain returns the same sealed
    /// artifact.
    pub fn compile_entry(
        &mut self,
        domain: &SpecializationDomain,
    ) -> Result<CompiledPlan, CompileFailure> {
        let identity = domain.identity_bytes();
        if let Some(entry) = self.entries.iter().find(|entry| entry.identity == identity) {
            return Ok(CompiledPlan {
                artifact: Arc::clone(&entry.artifact),
                parameters: Rc::clone(&entry.parameters),
                device: self.device.clone(),
            });
        }
        let (artifact, parameters) =
            compile_artifact(self.device, &self.program, domain, &self.settings)?;
        let plan = CompiledPlan {
            artifact: Arc::clone(&artifact),
            parameters: Rc::clone(&parameters),
            device: self.device.clone(),
        };
        self.entries.push(CompiledEntry {
            identity,
            artifact,
            parameters,
        });
        Ok(plan)
    }
}

/// Run the sole logical -> plan-space -> physical-plan -> native pipeline for
/// this device's backend and seal the result. The one construction point of
/// `CompiledArtifact`: each arm is assembled from the single `Compiled`
/// result of the pipeline run on this device, so the plan/native pairing (and
/// the device that will execute it) is established here and nowhere else. No
/// selected or partially lowered artifact is exposed at the runtime boundary.
fn compile_artifact(
    device: &Device,
    program: &Program,
    domain: &SpecializationDomain,
    settings: &Settings,
) -> Result<(Arc<CompiledArtifact>, Rc<Vec<String>>), CompileFailure> {
    let system =
        |reason: String| CompileFailure::SystemFailure(SystemReport(reason));
    match device.backend_handle() {
        crate::DeviceHandle::Cpu(cpu) => {
            let workers = cpu.workers.borrow().count() as u64;
            let backend = seismic_cpu::Cpu::host(workers).map_err(system)?;
            let compiled = pipeline::compile(
                program,
                domain,
                &settings.precision,
                &backend,
                &settings.numerical_evidence,
                settings.budget,
            )?;
            let parameters = parameter_names(&compiled.logical);
            let artifact = CompiledArtifact::seal(BackendArtifact::Cpu {
                device: Rc::clone(cpu),
                physical: compiled.physical,
                native: compiled.native,
            });
            Ok((Arc::new(artifact), parameters))
        }
        #[cfg(target_os = "macos")]
        crate::DeviceHandle::Metal(metal) => {
            let backend = seismic_metal::catalog::MetalCompiler::from_device(metal);
            let compiled = pipeline::compile(
                program,
                domain,
                &settings.precision,
                &backend,
                &settings.numerical_evidence,
                settings.budget,
            )?;
            let parameters = parameter_names(&compiled.logical);
            let artifact = CompiledArtifact::seal(BackendArtifact::Metal {
                physical: compiled.physical,
                native: Arc::new(compiled.native),
            });
            Ok((Arc::new(artifact), parameters))
        }
        crate::DeviceHandle::Cuda(cuda) => {
            let backend = seismic_cuda::CudaCompiler::new(cuda)
                .map_err(|error| system(error.to_string()))?;
            let compiled = pipeline::compile(
                program,
                domain,
                &settings.precision,
                &backend,
                &settings.numerical_evidence,
                settings.budget,
            )?;
            let parameters = parameter_names(&compiled.logical);
            let artifact = CompiledArtifact::seal(BackendArtifact::Cuda {
                device: Rc::clone(cuda),
                physical: compiled.physical,
                native: Arc::new(compiled.native),
            });
            Ok((Arc::new(artifact), parameters))
        }
    }
}

/// Interface parameter names in interface order; ABI parameter ordinals are
/// interface param ordinals.
fn parameter_names(logical: &LogicalProgram) -> Rc<Vec<String>> {
    Rc::new(
        logical
            .choice(logical.entry_choice)
            .interface
            .params
            .iter()
            .map(|param| param.name.clone())
            .collect(),
    )
}
