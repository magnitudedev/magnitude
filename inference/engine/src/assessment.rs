//! Metadata-only model assessment from a package path.
//!
//! Opens only the GGUF headers of the target and optional projector, derives
//! the family definition, resolves the execution manifest exactly as engine
//! configuration does, plans it through [`crate::planning::plan_execution`]
//! (the production planning inputs) and assesses the draft against stable
//! memory capacity and the selected device's memory bandwidth. Chat
//! capabilities come from the engine's own tokenizer validation and
//! template and reasoning inspection over header metadata. No device is
//! opened, no weight payload is read and nothing is decoded.

use crate::error::{
    classify_catalog, classify_graph, classify_plan, PlanOutcome, UnsupportedModel,
};
use crate::options::{ExecutionManifest, ModelMethod, ModelPolicy};
use crate::planning::{plan_backend, plan_execution, ExecutionPlanningError};
use magnitude_artifacts::PackageHeaders;
use magnitude_chat::{
    artifacts::{gguf_templates, gguf_tokenizer_vocabulary},
    TemplateInspection,
};
use magnitude_executor::{
    assessment::{
        finish_execution_assessment, prepare_execution_assessment, resolve_bandwidth,
        AssessmentError, AssessmentRequest, DeviceBandwidth, ExecutionAssessment,
        PreparedExecutionAssessment,
    },
    platform::{self, DeviceRequest, MemoryReserves, PlatformError, SelectedDevice},
    ExecutionPath, ExecutionPlanDraft,
};
use magnitude_family_contracts::ModelDefinition;
use magnitude_scheduler::ServiceLimits;
use seismic::{DeviceCatalog, DeviceTopology, HostMemoryStatus};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

/// The package components to assess and the generation method the bundle
/// declares. Only their headers are read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPackagePaths {
    /// The target GGUF, or the first shard of a split GGUF (the remaining
    /// shards are found beside it by the split naming convention).
    pub target: PathBuf,
    pub projector: Option<PathBuf>,
    /// A separate draft model (DFlash, DSpark, DFlash2).
    pub draft: Option<PathBuf>,
    /// The bundle's method: `Auto` for a standalone package, the declared
    /// method otherwise. A declared separate-draft method the draft does
    /// not implement, or the executor cannot run, makes the bundle
    /// unsupported or incompatible; it is never assessed as plain.
    pub method: ModelMethod,
}

/// The execution configuration every model is assessed on: the device a
/// native load selects, its memory bandwidth, the host and the serving
/// policy.
pub struct AssessmentSetup {
    pub topology: Arc<DeviceTopology>,
    pub host: HostMemoryStatus,
    pub device: DeviceRequest,
    pub selected: SelectedDevice,
    pub bandwidth: DeviceBandwidth,
    pub reserves: MemoryReserves,
    pub policy: ModelPolicy,
    pub service: ServiceLimits,
}

impl AssessmentSetup {
    /// Select the device a native load would select, resolve its bandwidth
    /// and observe the host.
    pub fn discover(
        catalog: &DeviceCatalog,
        device: DeviceRequest,
        reserves: MemoryReserves,
        policy: ModelPolicy,
        service: ServiceLimits,
    ) -> Result<Self, ModelAssessmentError> {
        let selected = platform::select_device(catalog, ExecutionPath::Native, device, &reserves)
            .map_err(ModelAssessmentError::Platform)?;
        let host = catalog
            .host_memory_status()
            .map_err(|error| ModelAssessmentError::Platform(PlatformError::Observation(error)))?;
        let topology = catalog.topology();
        // Apple silicon, the only chips whose bins the core count selects,
        // has no simultaneous multithreading.
        let cpu_cores = std::thread::available_parallelism()
            .map_or(1, |cores| u32::try_from(cores.get()).unwrap_or(u32::MAX));
        let bandwidth = resolve_bandwidth(&topology, &selected.info, cpu_cores);
        Ok(Self {
            topology,
            host,
            device,
            selected,
            bandwidth,
            reserves,
            policy,
            service,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningCapabilities {
    /// Normalized reasoning efforts the template distinguishes, `none`
    /// included when reasoning can be disabled.
    pub efforts: Vec<String>,
    /// The effort an omitted control renders as, when it is unambiguous.
    pub default_effort: Option<String>,
}

impl ReasoningCapabilities {
    /// Reasoning is controllable when some effort other than `none` exists.
    pub fn supported(&self) -> bool {
        self.efforts.iter().any(|effort| effort != "none")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub vision: bool,
    pub tools: bool,
    pub structured_output: bool,
    pub reasoning: ReasoningCapabilities,
}

/// Host-side model facts established from headers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFacts {
    /// Why an optional separate drafter was disabled; target admission still succeeded.
    pub draft_unavailable: Option<String>,
    pub capabilities: ModelCapabilities,
    /// [`magnitude_chat::TemplateInspection::fingerprint`] of the package's
    /// chat templates and reasoning profile.
    pub template_fingerprint: String,
    /// The model's supported context, which assessment serves.
    pub context_limit: u32,
}

/// A complete assessment of one package on one execution environment.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelAssessment {
    Assessed {
        facts: ModelFacts,
        execution: ExecutionAssessment,
    },
    /// The engine cannot execute the package: no family recognizes it, its
    /// definition, tokenizer or templates are outside what the engine
    /// executes, the planner refuses it on the backend, or its program calls
    /// a kernel outside the kernel's domain. The same classification a load
    /// makes.
    Unsupported(UnsupportedModel),
}

/// An operational failure: no result is produced for the package.
#[derive(Debug)]
pub enum ModelAssessmentError {
    /// A header could not be read or is not a valid GGUF component.
    Artifact(magnitude_artifacts::Error),
    /// The model's context limit is outside the assessment domain.
    ContextLimit(u64),
    /// The engine configuration could not be resolved for the model.
    Configuration(String),
    /// Planning or graph construction failed in the engine itself.
    Planning(String),
    Platform(PlatformError),
    Assessment(AssessmentError),
}

impl fmt::Display for ModelAssessmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => write!(formatter, "artifact header: {error}"),
            Self::ContextLimit(limit) => {
                write!(
                    formatter,
                    "model context limit {limit} exceeds the assessment domain"
                )
            }
            Self::Configuration(error) => write!(formatter, "engine configuration: {error}"),
            Self::Planning(error) => write!(formatter, "execution planning: {error}"),
            Self::Platform(error) => write!(formatter, "platform: {error}"),
            Self::Assessment(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ModelAssessmentError {}

/// Evidence from the exact package and selected execution configuration.
pub enum PreparedModelAssessment {
    Unsupported(UnsupportedModel),
    Planned {
        facts: ModelFacts,
        draft: ExecutionPlanDraft,
        execution: PreparedExecutionAssessment,
    },
}

/// Complete a prepared model against stable capacity and the device's
/// bandwidth.
pub fn finish_model_assessment(
    prepared: &PreparedModelAssessment,
    setup: &AssessmentSetup,
    performance_depths: &[u32],
) -> Result<ModelAssessment, ModelAssessmentError> {
    let (facts, draft, preparation) = match prepared {
        PreparedModelAssessment::Unsupported(unsupported) => {
            return Ok(ModelAssessment::Unsupported(unsupported.clone()));
        }
        PreparedModelAssessment::Planned {
            facts,
            draft,
            execution,
        } => (facts, draft, execution),
    };
    let execution = finish_execution_assessment(
        preparation,
        draft,
        &setup.topology,
        &setup.host,
        setup.bandwidth,
        &AssessmentRequest {
            context_limit: facts.context_limit,
            performance_depths: performance_depths.to_vec(),
            reserves: setup.reserves,
        },
    )
    .map_err(ModelAssessmentError::Assessment)?;
    Ok(ModelAssessment::Assessed {
        facts: facts.clone(),
        execution,
    })
}

/// Assess one package on the setup's selected device.
pub fn assess_model(
    package: &ModelPackagePaths,
    setup: &AssessmentSetup,
    performance_depths: &[u32],
) -> Result<ModelAssessment, ModelAssessmentError> {
    let prepared = prepare_model_assessment(package, setup)?;
    finish_model_assessment(&prepared, setup, performance_depths)
}

/// Prepare one package from its headers on the selected execution
/// configuration: recognition, the
/// family definition, capabilities from the engine's own chat inspection,
/// the execution manifest and its allocation-free plan, then decode demand
/// and the checked memory charge. A tokenizer or template the engine cannot
/// prepare makes the package unsupported whatever its plan.
pub fn prepare_model_assessment(
    package: &ModelPackagePaths,
    setup: &AssessmentSetup,
) -> Result<PreparedModelAssessment, ModelAssessmentError> {
    let (facts, manifest) = match resolve_package(
        package,
        &setup.policy,
        &setup.service,
        setup.device,
        setup.reserves,
    )? {
        ResolvedPackage::Unsupported(unsupported) => {
            return Ok(PreparedModelAssessment::Unsupported(unsupported))
        }
        ResolvedPackage::Resolved { facts, manifest } => (facts, manifest),
    };
    // Planner refusals and kernel domains classify exactly as a load's do.
    let refused = |outcome: PlanOutcome| match outcome {
        PlanOutcome::Unsupported(unsupported) => {
            Ok(PreparedModelAssessment::Unsupported(unsupported))
        }
        PlanOutcome::Internal(reason) => Err(ModelAssessmentError::Planning(reason)),
    };
    let draft = match plan_execution(&manifest, &setup.selected) {
        Ok(draft) => draft,
        Err(ExecutionPlanningError::Plan(error)) => {
            return refused(classify_plan(error, setup.selected.info.backend));
        }
        Err(error) => return Err(ModelAssessmentError::Planning(error.to_string())),
    };
    let execution =
        match prepare_execution_assessment(&manifest.definition, &draft, facts.context_limit) {
            Ok(execution) => execution,
            Err(AssessmentError::Graph(error)) => return refused(classify_graph(error)),
            Err(error) => return Err(ModelAssessmentError::Assessment(error)),
        };
    Ok(PreparedModelAssessment::Planned {
        facts,
        draft,
        execution,
    })
}

/// The native kernels a load of one package prepares on a backend, or why
/// the engine cannot execute the package there: the classification
/// assessment makes.
pub enum KernelInventory {
    Unsupported(UnsupportedModel),
    Requests(Vec<seismic::KernelRequest>),
}

/// The native kernels a load of `package` under `policy` prepares on
/// `backend`, derived from its headers without a device.
pub fn kernel_inventory(
    package: &ModelPackagePaths,
    policy: &ModelPolicy,
    service: &ServiceLimits,
    backend: seismic::BackendName,
) -> Result<KernelInventory, ModelAssessmentError> {
    let manifest = match resolve_package(
        package,
        policy,
        service,
        DeviceRequest::Backend(backend),
        MemoryReserves::standard(),
    )? {
        ResolvedPackage::Unsupported(unsupported) => {
            return Ok(KernelInventory::Unsupported(unsupported))
        }
        ResolvedPackage::Resolved { manifest, .. } => manifest,
    };
    let refused = |outcome: PlanOutcome| match outcome {
        PlanOutcome::Unsupported(unsupported) => Ok(KernelInventory::Unsupported(unsupported)),
        PlanOutcome::Internal(reason) => Err(ModelAssessmentError::Planning(reason)),
    };
    let plan = match plan_backend(&manifest, backend) {
        Ok(plan) => plan,
        Err(ExecutionPlanningError::Plan(error)) => {
            return refused(classify_plan(error, backend));
        }
        Err(error) => return Err(ModelAssessmentError::Planning(error.to_string())),
    };
    match magnitude_executor::kernel_inventory(&manifest.definition, &plan) {
        Ok(requests) => Ok(KernelInventory::Requests(requests)),
        Err(error) => refused(classify_catalog(error)),
    }
}

/// A package resolved from its headers: the manifest a load of it executes
/// and its host-side facts, or why the engine cannot execute it.
enum ResolvedPackage {
    Unsupported(UnsupportedModel),
    Resolved {
        facts: ModelFacts,
        manifest: ExecutionManifest,
    },
}

/// Recognition, the family definition, capabilities from the engine's own
/// chat inspection and the execution manifest of `package` under `policy`
/// (its method from the package).
fn resolve_package(
    package: &ModelPackagePaths,
    policy: &ModelPolicy,
    service: &ServiceLimits,
    device: DeviceRequest,
    reserves: MemoryReserves,
) -> Result<ResolvedPackage, ModelAssessmentError> {
    let headers = PackageHeaders::open(&package.target, package.projector.as_deref())
        .and_then(|headers| match &package.draft {
            Some(draft) => headers.with_draft(draft),
            None => Ok(headers),
        })
        .map_err(ModelAssessmentError::Artifact)?;
    let family = match crate::families::recognize(headers.target()) {
        Ok(family) => family,
        Err(unsupported) => return Ok(ResolvedPackage::Unsupported(unsupported)),
    };
    let admitted = match family
        .inspect(headers.target(), headers.projector(), headers.identity())
        .map_err(|error| error.0)
        .map(|declared| crate::host::bind_draft(family, declared, headers.draft()))
    {
        Ok(definition) => definition,
        Err(reason) => {
            return Ok(ResolvedPackage::Unsupported(
                UnsupportedModel::Representation { reason },
            ));
        }
    };
    let definition = admitted.definition;
    let context_limit = u32::try_from(definition.decoder.context_limit)
        .map_err(|_| ModelAssessmentError::ContextLimit(definition.decoder.context_limit))?;
    let mut facts = match model_facts(&headers, package, &definition, context_limit) {
        Ok(facts) => facts,
        Err(unsupported) => return Ok(ResolvedPackage::Unsupported(unsupported)),
    };
    facts.draft_unavailable = admitted.draft_unavailable.as_ref().map(ToString::to_string);
    let policy = ModelPolicy {
        method: package.method,
        ..policy.clone()
    };
    // A method the package cannot run (a declared draft of another variant)
    // is an invalid configuration of the bundle, as it is for a load.
    let model = policy
        .resolve_admitted(&definition, admitted.draft_unavailable.as_ref())
        .map_err(ModelAssessmentError::Configuration)?;
    let manifest = ExecutionManifest::new(
        headers.manifest(),
        definition,
        model,
        service.clone(),
        ExecutionPath::Native,
        device,
        None,
        reserves,
    )
    .map_err(ModelAssessmentError::Configuration)?;
    Ok(ResolvedPackage::Resolved { facts, manifest })
}

/// The package's host-side facts: capabilities and template fingerprint
/// from the engine's own chat inspection over header metadata.
fn model_facts(
    headers: &PackageHeaders,
    package: &ModelPackagePaths,
    definition: &ModelDefinition,
    context_limit: u32,
) -> Result<ModelFacts, UnsupportedModel> {
    let chat = inspect_chat(headers, package, definition)?;
    Ok(ModelFacts {
        draft_unavailable: None,
        capabilities: ModelCapabilities {
            vision: definition.vision.is_some(),
            tools: chat.tools,
            structured_output: chat.structured_output,
            reasoning: ReasoningCapabilities {
                efforts: chat
                    .reasoning
                    .mappings
                    .iter()
                    .map(|mapping| mapping.effort.clone())
                    .collect(),
                default_effort: chat.reasoning.default_effort.clone(),
            },
        },
        template_fingerprint: chat.fingerprint,
        context_limit,
    })
}

/// The engine's own tokenizer and template construction over header
/// metadata, then the bundle's capability and reasoning inspection. A
/// tokenizer or template the engine cannot prepare is an unsupported
/// representation, as it is when a load resolves the package.
fn inspect_chat(
    headers: &PackageHeaders,
    package: &ModelPackagePaths,
    definition: &magnitude_family_contracts::ModelDefinition,
) -> Result<TemplateInspection, UnsupportedModel> {
    let unsupported = |reason: String| UnsupportedModel::Representation { reason };
    let tokenizer_payload = magnitude_artifacts::TokenizerPayload::from_directory(headers.target());
    // Support and vocabulary are header facts; a load builds the tokenizer.
    let vocabulary = gguf_tokenizer_vocabulary(&tokenizer_payload)
        .map_err(|error| unsupported(error.to_string()))?;
    if u64::try_from(vocabulary).ok() != Some(definition.decoder.vocabulary) {
        return Err(unsupported(
            "tokenizer vocabulary differs from model vocabulary".into(),
        ));
    }
    let templates = magnitude_artifacts::TemplatePayload::from_directory(
        headers.target(),
        &package.target.display().to_string(),
    )
    .map_err(|error| unsupported(error.to_string()))?;
    gguf_templates(&templates, &tokenizer_payload)
        .and_then(|bundle| bundle.inspect_cached())
        .map_err(unsupported)
}
