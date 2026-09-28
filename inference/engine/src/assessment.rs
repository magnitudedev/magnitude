//! Metadata-only model assessment from a package path.
//!
//! Opens only the GGUF headers of the target and optional projector, derives
//! the family definition, resolves the execution manifest exactly as engine
//! configuration does, plans it through [`crate::planning::plan_execution`]
//! (the production planning inputs) and assesses the draft against the
//! environment's measurement basis and stable memory capacity. Chat
//! capabilities come from the engine's own tokenizer validation and
//! template and reasoning inspection over header metadata. No device is
//! opened, no weight payload is read and nothing is decoded.

use crate::error::UnsupportedModel;
use crate::options::{ExecutionManifest, ModelMethod, ModelPolicy};
use crate::planning::{ExecutionPlanningError, plan_execution};
use magnitude_artifacts::PackageHeaders;
use magnitude_chat::{
    TemplateInspection,
    artifacts::{gguf_templates, gguf_tokenizer_vocabulary},
};
use magnitude_executor::{
    ExecutionPath, ExecutionPlanDraft, PlanError,
    assessment::{
        AssessmentError, AssessmentRequest, ExecutionAssessment, IncompatibleReason,
        MeasurementBasis, PreparedExecutionAssessment, finish_execution_assessment,
        prepare_execution_assessment,
    },
    platform::{self, DeviceRequest, MemoryReserves, PlatformError, SelectedDevice},
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

/// The selected execution configuration paired with its generic measurement basis.
pub struct AssessmentEnvironment {
    pub setup: Arc<AssessmentSetup>,
    pub basis: MeasurementBasis,
}

/// Device and serving facts available before generic measurements finish.
pub struct AssessmentSetup {
    pub topology: Arc<DeviceTopology>,
    pub host: HostMemoryStatus,
    pub device: DeviceRequest,
    pub selected: SelectedDevice,
    pub reserves: MemoryReserves,
    pub policy: ModelPolicy,
    pub service: ServiceLimits,
}

impl AssessmentSetup {
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
        Ok(Self {
            topology: catalog.topology(),
            host,
            device,
            selected,
            reserves,
            policy,
            service,
        })
    }

    pub fn with_basis(
        self: Arc<Self>,
        basis: MeasurementBasis,
    ) -> Result<AssessmentEnvironment, ModelAssessmentError> {
        if basis.identity.backend != self.selected.info.backend.as_str() {
            return Err(ModelAssessmentError::BasisBackend {
                basis: basis.identity.backend.clone(),
                selected: self.selected.info.backend.as_str().to_owned(),
            });
        }
        Ok(AssessmentEnvironment { setup: self, basis })
    }
}

impl AssessmentEnvironment {
    /// Select the device a native load would select and observe the host.
    pub fn discover(
        catalog: &DeviceCatalog,
        device: DeviceRequest,
        reserves: MemoryReserves,
        basis: MeasurementBasis,
        policy: ModelPolicy,
        service: ServiceLimits,
    ) -> Result<Self, ModelAssessmentError> {
        Arc::new(AssessmentSetup::discover(
            catalog, device, reserves, policy, service,
        )?)
        .with_basis(basis)
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
    /// The engine cannot serve the package at all, before any device fact:
    /// no family recognizes it, or its definition, tokenizer or templates
    /// are outside what the engine executes (the same classification a
    /// load's resolution makes).
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
    Planning(ExecutionPlanningError),
    Platform(PlatformError),
    /// The basis was measured on a different backend than the selected device.
    BasisBackend {
        basis: String,
        selected: String,
    },
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
            Self::BasisBackend { basis, selected } => write!(
                formatter,
                "measurement basis backend {basis} differs from the selected {selected} device"
            ),
            Self::Assessment(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ModelAssessmentError {}

/// Basis-independent evidence from the exact package and selected execution configuration.
pub enum PreparedModelAssessment {
    Unsupported(UnsupportedModel),
    Incompatible {
        facts: ModelFacts,
        reason: String,
    },
    Planned {
        facts: ModelFacts,
        draft: ExecutionPlanDraft,
        execution: PreparedExecutionAssessment,
    },
}

/// Complete a prepared model using only the fixed measurement basis and stable capacity.
pub fn finish_model_assessment(
    prepared: &PreparedModelAssessment,
    environment: &AssessmentEnvironment,
    performance_depths: &[u32],
) -> Result<ModelAssessment, ModelAssessmentError> {
    let (facts, draft, preparation) = match prepared {
        PreparedModelAssessment::Unsupported(unsupported) => {
            return Ok(ModelAssessment::Unsupported(unsupported.clone()));
        }
        PreparedModelAssessment::Incompatible { facts, reason } => {
            return Ok(ModelAssessment::Assessed {
                facts: facts.clone(),
                execution: ExecutionAssessment::Incompatible {
                    reason: IncompatibleReason::Unsupported {
                        reason: reason.clone(),
                    },
                },
            });
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
        &environment.setup.topology,
        &environment.setup.host,
        &environment.basis,
        &AssessmentRequest {
            context_limit: facts.context_limit,
            performance_depths: performance_depths.to_vec(),
            reserves: environment.setup.reserves,
        },
    )
    .map_err(ModelAssessmentError::Assessment)?;
    Ok(ModelAssessment::Assessed {
        facts: facts.clone(),
        execution,
    })
}


/// Assess one package on the environment's selected device.
pub fn assess_model(
    package: &ModelPackagePaths,
    environment: &AssessmentEnvironment,
    performance_depths: &[u32],
) -> Result<ModelAssessment, ModelAssessmentError> {
    let prepared = prepare_model_assessment(package, &environment.setup)?;
    finish_model_assessment(&prepared, environment, performance_depths)
}

/// Prepare one package from its headers on the selected execution
/// configuration, independent of the measurement basis: recognition, the
/// family definition, capabilities from the engine's own chat inspection,
/// the execution manifest and its allocation-free plan, then decode demand
/// and the checked memory charge. A tokenizer or template the engine cannot
/// prepare makes the package unsupported whatever its plan.
pub fn prepare_model_assessment(
    package: &ModelPackagePaths,
    setup: &AssessmentSetup,
) -> Result<PreparedModelAssessment, ModelAssessmentError> {
    let headers = PackageHeaders::open(&package.target, package.projector.as_deref())
        .and_then(|headers| match &package.draft {
            Some(draft) => headers.with_draft(draft),
            None => Ok(headers),
        })
        .map_err(ModelAssessmentError::Artifact)?;
    let family = match crate::families::recognize(headers.target()) {
        Ok(family) => family,
        Err(unsupported) => return Ok(PreparedModelAssessment::Unsupported(unsupported)),
    };
    let definition = match family
        .inspect(headers.target(), headers.projector(), headers.identity())
        .map_err(|error| error.0)
        .and_then(|declared| crate::host::bind_draft(family, declared, headers.draft()))
    {
        Ok(definition) => definition,
        Err(reason) => {
            return Ok(PreparedModelAssessment::Unsupported(
                UnsupportedModel::Representation { reason },
            ));
        }
    };
    let context_limit = u32::try_from(definition.decoder.context_limit)
        .map_err(|_| ModelAssessmentError::ContextLimit(definition.decoder.context_limit))?;
    let facts = match model_facts(&headers, package, &definition, context_limit) {
        Ok(facts) => facts,
        Err(unsupported) => return Ok(PreparedModelAssessment::Unsupported(unsupported)),
    };
    let policy = ModelPolicy {
        method: package.method,
        ..setup.policy.clone()
    };
    // A method the package cannot run (a declared draft of another variant)
    // is a property of the bundle, not an assessment failure.
    let model = match policy.resolve(&definition) {
        Ok(model) => model,
        Err(reason) => return Ok(PreparedModelAssessment::Incompatible { facts, reason }),
    };
    let manifest = ExecutionManifest::new(
        headers.manifest(),
        definition,
        model,
        setup.service.clone(),
        ExecutionPath::Native,
        setup.device,
        None,
        setup.reserves,
    )
    .map_err(ModelAssessmentError::Configuration)?;
    let draft = match plan_execution(&manifest, &setup.selected) {
        Ok(draft) => draft,
        Err(ExecutionPlanningError::Plan(error)) if is_unsupported(&error) => {
            return Ok(PreparedModelAssessment::Incompatible {
                facts,
                reason: error.to_string(),
            });
        }
        Err(error) => return Err(ModelAssessmentError::Planning(error)),
    };
    let execution = prepare_execution_assessment(&manifest.definition, &draft, context_limit)
        .map_err(ModelAssessmentError::Assessment)?;
    Ok(PreparedModelAssessment::Planned {
        facts,
        draft,
        execution,
    })
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

/// Planner rejections of a recognized, validated definition are properties
/// of the artifact's representation or topology on this execution path; the
/// remaining variants are arithmetic or resource failures.
fn is_unsupported(error: &PlanError) -> bool {
    match error {
        PlanError::Unsupported(_)
        | PlanError::UnsupportedOperator { .. }
        | PlanError::Deferred(_)
        | PlanError::UnportedScale(_)
        | PlanError::InvalidDefinition(_)
        | PlanError::Topology(_) => true,
        PlanError::Arithmetic(_) | PlanError::ResourcePlanning(_) | PlanError::Resource(_) => false,
    }
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
