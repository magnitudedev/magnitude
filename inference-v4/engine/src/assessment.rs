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

use crate::options::{ExecutionManifest, ModelPolicy};
use crate::planning::{plan_execution, ExecutionPlanningError};
use magnitude_artifacts::{ComponentManifest, PackageHeaders, PackageManifest};
use magnitude_chat::{
    artifacts::{gguf_byte_bpe, gguf_templates},
    ByteBpeTokenizer, TemplateInspection,
};
use magnitude_model_executor::{
    assessment::{
        assess_execution, AssessmentError, AssessmentRequest, ExecutionAssessment,
        IncompatibleReason, MeasurementBasis,
    },
    platform::{self, DeviceRequest, MemoryReserves, PlatformError, SelectedDevice},
    ExecutionPath, PlanError,
};
use magnitude_service::ServiceLimits;
use seismic::{DeviceCatalog, DeviceTopology, HostMemoryStatus};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The package components to assess. Only their headers are read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPackagePaths {
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

/// Why the engine cannot serve a package at all, before any device fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnsupportedModel {
    /// No supported model family recognizes the target.
    Family(String),
    /// The package's tokenizer profile is outside the engine's tokenizers.
    Tokenizer(String),
    /// The package's chat templates cannot be prepared by the engine.
    Template(String),
}

impl fmt::Display for UnsupportedModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Family(reason) => write!(formatter, "unsupported model family: {reason}"),
            Self::Tokenizer(reason) => write!(formatter, "unsupported tokenizer: {reason}"),
            Self::Template(reason) => write!(formatter, "unsupported chat template: {reason}"),
        }
    }
}

/// A complete assessment of one package on one execution environment.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelAssessment {
    Assessed {
        facts: ModelFacts,
        execution: ExecutionAssessment,
    },
    Unsupported(UnsupportedModel),
}

/// An operational failure: no result is produced for the package.
#[derive(Debug)]
pub enum ModelAssessmentError {
    /// A header could not be read or is not a valid GGUF component.
    Artifact(magnitude_artifacts::Error),
    /// A recognized family rejected the artifact's metadata.
    Definition(magnitude_model_qwen35::Error),
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
            Self::Definition(error) => write!(formatter, "model definition: {error}"),
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
    if let Err(error) = magnitude_model_qwen35::recognize(headers.target()) {
        return Ok(ModelAssessment::Unsupported(UnsupportedModel::Family(
            error.to_string(),
        )));
    }
    let definition = magnitude_model_qwen35::inspect_components(
        headers.target(),
        headers.projector(),
        headers.identity(),
    )
    .map_err(ModelAssessmentError::Definition)?;
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
        header_manifest(&headers, package)?,
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

/// The package manifest a payload-backed open would admit, from headers:
/// the planner reads only component identities and tensor directories.
fn header_manifest(
    headers: &PackageHeaders,
    package: &ModelPackagePaths,
) -> Result<PackageManifest, ModelAssessmentError> {
    let identity = headers.identity();
    let component = |path: &Path, identity, directory: &magnitude_artifacts::gguf::Directory| {
        Ok(ComponentManifest {
            path: path.to_path_buf(),
            identity,
            size: std::fs::metadata(path)
                .map_err(|error| {
                    ModelAssessmentError::Artifact(magnitude_artifacts::Error::Io(error))
                })?
                .len(),
            tensors: directory.tensors.clone(),
        })
    };
    let projector = match (&package.projector, headers.projector(), identity.projector) {
        (Some(path), Some(directory), Some(projector)) => {
            Some(component(path, projector, directory)?)
        }
        (None, None, None) => None,
        _ => {
            return Err(ModelAssessmentError::Artifact(
                magnitude_artifacts::Error::Invalid(
                    "projector header and identity disagree".into(),
                ),
            ))
        }
    };
    Ok(PackageManifest {
        identity,
        target: component(&package.target, identity.target, headers.target())?,
        projector,
    })
}

/// The engine's own tokenizer and template construction over header
/// metadata, then the bundle's capability and reasoning inspection.
fn inspect_chat(
    headers: &PackageHeaders,
    package: &ModelPackagePaths,
    definition: &magnitude_model_contracts::ModelDefinition,
) -> Result<TemplateInspection, UnsupportedModel> {
    let tokenizer_payload = magnitude_artifacts::TokenizerPayload::from_directory(headers.target());
    let tokenizer = gguf_byte_bpe(
        &tokenizer_payload,
        definition.artifact_identity.target.to_string(),
    )
    .and_then(ByteBpeTokenizer::new)
    .map_err(UnsupportedModel::Tokenizer)?;
    if u64::try_from(tokenizer.vocabulary()).ok() != Some(definition.geometry.vocabulary) {
        return Err(UnsupportedModel::Tokenizer(
            "tokenizer vocabulary differs from model vocabulary".into(),
        ));
    }
    let templates = magnitude_artifacts::TemplatePayload::from_directory(
        headers.target(),
        &package.target.display().to_string(),
    )
    .map_err(|error| UnsupportedModel::Template(error.to_string()))?;
    gguf_templates(&templates, &tokenizer_payload)
        .and_then(|bundle| bundle.inspect())
        .map_err(UnsupportedModel::Template)
}
