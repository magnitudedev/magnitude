//! Metadata-only model assessment from a package path.
//!
//! Opens only the GGUF headers of the target and optional projector, derives
//! the family definition, resolves the execution manifest exactly as engine
//! configuration does, plans it through [`crate::planning::plan_execution`]
//! (the production planning inputs) and assesses the draft against the
//! environment's measurement basis and stable memory capacity. Chat
//! capabilities come from the engine's own tokenizer, template and reasoning
//! inspection over header metadata. No device is opened, no weight payload
//! is read and nothing is decoded.

use crate::error::UnsupportedModel;
use crate::options::{ExecutionManifest, ModelPolicy};
use crate::planning::{plan_execution, ExecutionPlanningError};
use magnitude_artifacts::PackageHeaders;
use magnitude_chat::{
    artifacts::{gguf_byte_bpe, gguf_templates},
    ByteBpeTokenizer, TemplateInspection,
};
use magnitude_executor::{
    assessment::{
        assess_execution, AssessmentError, AssessmentRequest, ExecutionAssessment,
        IncompatibleReason, MeasurementBasis,
    },
    platform::{self, DeviceRequest, MemoryReserves, PlatformError, SelectedDevice},
    ExecutionPath, PlanError,
};
use magnitude_scheduler::ServiceLimits;
use seismic::{DeviceCatalog, DeviceTopology, HostMemoryStatus};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

/// The package components to assess. Only their headers are read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPackagePaths {
    /// The target GGUF, or the first shard of a split GGUF (the remaining
    /// shards are found beside it by the split naming convention).
    pub target: PathBuf,
    pub projector: Option<PathBuf>,
}

/// Everything one execution environment contributes to every assessment:
/// the selected device, its topology and host observation, the host's
/// reserve policy, the environment's measurement basis, and the engine
/// configuration a load of any model would use.
pub struct AssessmentEnvironment {
    pub topology: Arc<DeviceTopology>,
    pub host: HostMemoryStatus,
    pub device: DeviceRequest,
    pub selected: SelectedDevice,
    pub reserves: MemoryReserves,
    pub basis: MeasurementBasis,
    pub policy: ModelPolicy,
    pub service: ServiceLimits,
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
        let selected = platform::select_device(catalog, ExecutionPath::Native, device, &reserves)
            .map_err(ModelAssessmentError::Platform)?;
        if basis.identity.backend != selected.info.backend.as_str() {
            return Err(ModelAssessmentError::BasisBackend {
                basis: basis.identity.backend.clone(),
                selected: selected.info.backend.as_str().to_owned(),
            });
        }
        let host = catalog
            .host_memory_status()
            .map_err(|error| ModelAssessmentError::Platform(PlatformError::Observation(error)))?;
        Ok(Self {
            topology: catalog.topology(),
            host,
            device,
            selected,
            reserves,
            basis,
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
    BasisBackend { basis: String, selected: String },
    Assessment(AssessmentError),
}

impl fmt::Display for ModelAssessmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => write!(formatter, "artifact header: {error}"),
            Self::ContextLimit(limit) => {
                write!(formatter, "model context limit {limit} exceeds the assessment domain")
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

/// Assess one package on the environment's selected device.
pub fn assess_model(
    package: &ModelPackagePaths,
    environment: &AssessmentEnvironment,
    performance_depths: &[u32],
) -> Result<ModelAssessment, ModelAssessmentError> {
    let headers = PackageHeaders::open(&package.target, package.projector.as_deref())
        .map_err(ModelAssessmentError::Artifact)?;
    let family = match crate::families::recognize(headers.target()) {
        Ok(family) => family,
        Err(unsupported) => return Ok(ModelAssessment::Unsupported(unsupported)),
    };
    let definition =
        match family.inspect(headers.target(), headers.projector(), headers.identity()) {
            Ok(definition) => definition,
            Err(error) => {
                return Ok(ModelAssessment::Unsupported(
                    UnsupportedModel::Representation { reason: error.0 },
                ))
            }
        };
    let context_limit = u32::try_from(definition.geometry.context_limit)
        .map_err(|_| ModelAssessmentError::ContextLimit(definition.geometry.context_limit))?;
    let chat = match inspect_chat(&headers, package, &definition) {
        Ok(chat) => chat,
        Err(unsupported) => return Ok(ModelAssessment::Unsupported(unsupported)),
    };
    let facts = ModelFacts {
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
    };
    let model = environment
        .policy
        .resolve(&definition)
        .map_err(ModelAssessmentError::Configuration)?;
    let manifest = ExecutionManifest::new(
        headers.manifest(),
        definition,
        model,
        environment.service.clone(),
        ExecutionPath::Native,
        environment.device,
        None,
        environment.reserves,
    )
    .map_err(ModelAssessmentError::Configuration)?;
    let draft = match plan_execution(&manifest, &environment.selected) {
        Ok(draft) => draft,
        Err(ExecutionPlanningError::Plan(error)) if is_unsupported(&error) => {
            return Ok(ModelAssessment::Assessed {
                facts,
                execution: ExecutionAssessment::Incompatible {
                    reason: IncompatibleReason::Unsupported {
                        reason: error.to_string(),
                    },
                },
            });
        }
        Err(error) => return Err(ModelAssessmentError::Planning(error)),
    };
    let execution = assess_execution(
        &manifest.definition,
        &draft,
        &environment.topology,
        &environment.host,
        &environment.basis,
        &AssessmentRequest {
            context_limit,
            performance_depths: performance_depths.to_vec(),
            reserves: environment.reserves,
        },
    )
    .map_err(ModelAssessmentError::Assessment)?;
    Ok(ModelAssessment::Assessed { facts, execution })
}

/// Planner rejections of a recognized, validated definition are properties
/// of the artifact's representation or topology on this execution path; the
/// remaining variants are arithmetic or resource failures.
fn is_unsupported(error: &PlanError) -> bool {
    match error {
        PlanError::Unsupported(_) | PlanError::InvalidDefinition(_) | PlanError::Topology(_) => {
            true
        }
        PlanError::Arithmetic(_) | PlanError::ResourcePlanning(_) | PlanError::Resource(_) => {
            false
        }
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
    let tokenizer = gguf_byte_bpe(
        &tokenizer_payload,
        definition.artifact_identity.target.to_string(),
    )
    .and_then(ByteBpeTokenizer::new)
    .map_err(unsupported)?;
    if u64::try_from(tokenizer.vocabulary()).ok() != Some(definition.geometry.vocabulary) {
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
        .and_then(|bundle| bundle.inspect())
        .map_err(unsupported)
}
