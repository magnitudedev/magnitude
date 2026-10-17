//! Metadata-only model assessment from a package path.
//!
//! Opens only the GGUF headers of the target, its optional projector and
//! drafter. Independent header checks (weight encodings, the family's
//! interpretation, the tokenizer and the chat templates) each run and their
//! refusals are reported together. An admitted target is planned plain
//! through [`crate::planning::plan_execution`] (the production planning
//! inputs), which decides whether it runs here; its drafter is then planned
//! with it, and a drafter that does not bind or plan is `Rejected` without
//! affecting the target. The selected plan is assessed against stable memory
//! capacity and the selected device's memory bandwidth. No device is opened,
//! no weight payload is read and nothing is decoded.

use crate::error::{
    classify_catalog, classify_graph, classify_plan, PlanOutcome, UnsupportedModel,
    UnsupportedReason, WeightTypeCount,
};
use crate::fingerprint::ModelFingerprint;
use crate::host::{bind_draft, vocabulary_mismatch};
use crate::options::{ExecutionManifest, ModelMethod, ModelPolicy, ResolvedMethod};
use crate::planning::{plan_backend, plan_execution, ExecutionPlanningError};
use magnitude_artifacts::{
    gguf::TensorDescriptor, PackageHeaders, PackageManifest, TemplatePayload, TokenizerPayload,
};
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
    resident_element, resident_layout, source_element, ExecutionPath, ExecutionPlanDraft,
};
use magnitude_family_contracts::{DraftVariant, ModelDefinition, ModelFamily};
use magnitude_scheduler::ServiceLimits;
use seismic::{DType, DeviceCatalog, DeviceTopology, HostMemoryStatus, Layout};
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

/// The package components to assess and the generation method the bundle
/// declares (`Auto` for a standalone package). Only their headers are read.
/// `target` is the target GGUF or the first shard of a split GGUF (the other
/// shards are found beside it by the split naming convention); `draft` is a
/// separate drafter (DFlash, DSpark, DFlash2, a Gemma assistant or a
/// prediction head). A drafter the method cannot draft with is `Rejected`
/// and the target is assessed without it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPackagePaths {
    pub target: PathBuf,
    pub projector: Option<PathBuf>,
    pub draft: Option<PathBuf>,
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

/// The speculative method a bound drafter runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeculativeMethod {
    Mtp,
    DFlash,
    DSpark,
    DFlash2,
}

/// Whether the engine binds the package's drafter: none was declared or
/// present (`None`), it binds and plans with the target (`Bound`, with the
/// device memory it adds to the target served alone), or it does not
/// (`Rejected`, with every reason; the target is assessed and loads without
/// it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrafterVerdict {
    None,
    Bound {
        method: SpeculativeMethod,
        memory_bytes: u64,
    },
    Rejected(UnsupportedModel),
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
    pub fingerprint: Option<ModelFingerprint>,
    pub drafter: DrafterVerdict,
}

/// A complete assessment of one package on one execution environment.
/// `Assessed` assesses the drafted plan when the drafter is `Bound`, the
/// plain one otherwise. `Unsupported` is a target the engine cannot execute,
/// with every reason its header checks found, or the planner's or a kernel
/// domain's first refusal of an admitted definition, and the target's
/// fingerprint when its header has one.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelAssessment {
    Assessed {
        facts: ModelFacts,
        execution: ExecutionAssessment,
    },
    Unsupported {
        unsupported: UnsupportedModel,
        fingerprint: Option<ModelFingerprint>,
    },
}

/// An operational failure: no result is produced for the package. A
/// truncated or invalid GGUF is not one (it is `Unsupported` with
/// [`UnsupportedReason::Unreadable`]); a file that cannot be opened is.
#[derive(Debug)]
pub enum ModelAssessmentError {
    Artifact(magnitude_artifacts::Error),
    ContextLimit(u64),
    Configuration(String),
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
    Unsupported {
        unsupported: UnsupportedModel,
        fingerprint: Option<ModelFingerprint>,
    },
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
        PreparedModelAssessment::Unsupported {
            unsupported,
            fingerprint,
        } => {
            return Ok(ModelAssessment::Unsupported {
                unsupported: unsupported.clone(),
                fingerprint: fingerprint.clone(),
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
/// configuration: admission (every header refusal at once), the target's
/// plain plan and its decode demand and checked memory charge, then the
/// drafter's plan when one is admitted. A drafter is `Bound` with what its
/// plan adds to the plain plan's memory at the fit depth: its weights, state
/// and prepared resources on the device
/// ([`magnitude_executor::AssessmentMemoryCharge::serving_bytes`]) and the
/// host memory of its kernels and graphs
/// ([`magnitude_executor::AssessmentMemoryCharge::host_bytes`]); the drafted
/// plan is the one assessed.
pub fn prepare_model_assessment(
    package: &ModelPackagePaths,
    setup: &AssessmentSetup,
) -> Result<PreparedModelAssessment, ModelAssessmentError> {
    let layout = resident_layout(ExecutionPath::Native, setup.selected.info.backend);
    let admitted = match admit(package, layout)? {
        Admission::Unsupported {
            unsupported,
            fingerprint,
        } => {
            return Ok(PreparedModelAssessment::Unsupported {
                unsupported,
                fingerprint,
            });
        }
        Admission::Admitted(admitted) => *admitted,
    };
    let environment = Environment {
        policy: &setup.policy,
        service: &setup.service,
        device: setup.device,
        reserves: setup.reserves,
    };
    let plain = environment
        .plain(&admitted)
        .map_err(ModelAssessmentError::Configuration)?;
    let (plain_draft, plain_execution) =
        match prepare_plan(&plain, &setup.selected, admitted.context_limit) {
            Ok(planned) => planned,
            Err(Refusal::Unsupported(unsupported)) => {
                return Ok(PreparedModelAssessment::Unsupported {
                    unsupported,
                    fingerprint: admitted.fingerprint,
                });
            }
            Err(Refusal::Failed(error)) => return Err(error),
        };
    let drafted = match &admitted.drafter {
        DrafterAdmission::None => Err(DrafterVerdict::None),
        DrafterAdmission::Rejected(unsupported) => {
            Err(DrafterVerdict::Rejected(unsupported.clone()))
        }
        DrafterAdmission::Candidate { definition, method } => environment
            .drafted(&admitted, definition, *method)
            .and_then(|(manifest, method)| {
                match prepare_plan(&manifest, &setup.selected, admitted.context_limit) {
                    Ok((draft, execution)) => Ok((method, draft, execution)),
                    Err(Refusal::Unsupported(unsupported)) => Err(unsupported),
                    Err(Refusal::Failed(error)) => {
                        Err(UnsupportedModel::mismatch(error.to_string()))
                    }
                }
            })
            .map_err(DrafterVerdict::Rejected),
    };
    let (drafter, draft, execution) = match drafted {
        Ok((method, draft, execution)) => {
            let held = |charge: &magnitude_executor::AssessmentMemoryCharge| {
                charge
                    .host_bytes()
                    .map(|host| charge.serving_bytes.saturating_add(host))
                    .map_err(ModelAssessmentError::Planning)
            };
            let memory_bytes = held(execution.memory_charge())?
                .saturating_sub(held(plain_execution.memory_charge())?);
            (
                DrafterVerdict::Bound {
                    method,
                    memory_bytes,
                },
                draft,
                execution,
            )
        }
        Err(verdict) => (verdict, plain_draft, plain_execution),
    };
    Ok(PreparedModelAssessment::Planned {
        facts: admitted.facts(drafter),
        draft,
        execution,
    })
}

/// The native kernels a load of one package prepares on a backend, or why
/// the engine cannot execute the package there: the classification
/// assessment makes. A declared drafter the engine rejects makes the
/// package's inventory unsupported, since the load it describes drafts.
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
    let layout = resident_layout(ExecutionPath::Native, backend);
    let admitted = match admit(package, layout)? {
        Admission::Unsupported { unsupported, .. } => {
            return Ok(KernelInventory::Unsupported(unsupported));
        }
        Admission::Admitted(admitted) => *admitted,
    };
    let environment = Environment {
        policy,
        service,
        device: DeviceRequest::Backend(backend),
        reserves: MemoryReserves::standard(),
    };
    let plain = environment
        .plain(&admitted)
        .map_err(ModelAssessmentError::Configuration)?;
    let requests = match inventory(&plain, backend)? {
        Ok(requests) => requests,
        Err(unsupported) => return Ok(KernelInventory::Unsupported(unsupported)),
    };
    let drafted = match &admitted.drafter {
        DrafterAdmission::None => return Ok(KernelInventory::Requests(requests)),
        DrafterAdmission::Rejected(unsupported) => Err(unsupported.clone()),
        DrafterAdmission::Candidate { definition, method } => environment
            .drafted(&admitted, definition, *method)
            .and_then(|(manifest, _)| {
                inventory(&manifest, backend)
                    .unwrap_or_else(|error| Err(UnsupportedModel::mismatch(error.to_string())))
            }),
    };
    Ok(match drafted {
        Ok(requests) => KernelInventory::Requests(requests),
        Err(unsupported) => KernelInventory::Unsupported(drafter_refusal(unsupported)),
    })
}

/// The kernels `manifest`'s load prepares on `backend`, or the planner's or
/// a kernel domain's refusal.
fn inventory(
    manifest: &ExecutionManifest,
    backend: seismic::BackendName,
) -> Result<Result<Vec<seismic::KernelRequest>, UnsupportedModel>, ModelAssessmentError> {
    let refused = |outcome: PlanOutcome| match outcome {
        PlanOutcome::Unsupported(unsupported) => Ok(Err(unsupported)),
        PlanOutcome::Internal(reason) => Err(ModelAssessmentError::Planning(reason)),
    };
    let plan = match plan_backend(manifest, backend) {
        Ok(plan) => plan,
        Err(ExecutionPlanningError::Plan(error)) => return refused(classify_plan(error, backend)),
        Err(error) => return Err(ModelAssessmentError::Planning(error.to_string())),
    };
    match magnitude_executor::kernel_inventory(&manifest.definition, &plan) {
        Ok(requests) => Ok(Ok(requests)),
        Err(error) => refused(classify_catalog(error)),
    }
}

/// A rejected drafter, reported as a package refusal.
fn drafter_refusal(unsupported: UnsupportedModel) -> UnsupportedModel {
    UnsupportedModel::new(
        unsupported.reasons,
        format!("the drafter is rejected: {}", unsupported.detail),
    )
}

/// Why a manifest has no plan here: the material is refused on this
/// execution path, or the engine itself failed.
enum Refusal {
    Unsupported(UnsupportedModel),
    Failed(ModelAssessmentError),
}

/// Plan `manifest` on the selected device, then its decode demand and
/// checked memory charge, which builds every graph a load prepares. Planner
/// refusals and kernel domains classify exactly as a load's do.
fn prepare_plan(
    manifest: &ExecutionManifest,
    selected: &SelectedDevice,
    context_limit: u32,
) -> Result<(ExecutionPlanDraft, PreparedExecutionAssessment), Refusal> {
    let refused = |outcome: PlanOutcome| match outcome {
        PlanOutcome::Unsupported(unsupported) => Refusal::Unsupported(unsupported),
        PlanOutcome::Internal(reason) => Refusal::Failed(ModelAssessmentError::Planning(reason)),
    };
    let draft = match plan_execution(manifest, selected) {
        Ok(draft) => draft,
        Err(ExecutionPlanningError::Plan(error)) => {
            return Err(refused(classify_plan(error, selected.info.backend)));
        }
        Err(error) => {
            return Err(Refusal::Failed(ModelAssessmentError::Planning(
                error.to_string(),
            )));
        }
    };
    match prepare_execution_assessment(&manifest.definition, &draft, context_limit) {
        Ok(execution) => Ok((draft, execution)),
        Err(AssessmentError::Graph(error)) => Err(refused(classify_graph(error))),
        Err(error) => Err(Refusal::Failed(ModelAssessmentError::Assessment(error))),
    }
}

/// The serving configuration manifests are resolved under.
struct Environment<'a> {
    policy: &'a ModelPolicy,
    service: &'a ServiceLimits,
    device: DeviceRequest,
    reserves: MemoryReserves,
}

impl Environment<'_> {
    /// The target served alone: plain generation, without the drafter's
    /// component.
    fn plain(&self, admitted: &AdmittedPackage) -> Result<ExecutionManifest, String> {
        let policy = ModelPolicy {
            method: ModelMethod::Plain,
            draft_policy: None,
            mtp_proposals: None,
            ..self.policy.clone()
        };
        let model = policy.resolve(&admitted.definition)?;
        let package = PackageManifest {
            draft: None,
            ..admitted.headers.manifest()
        };
        self.manifest(package, admitted.definition.clone(), model)
    }

    /// The target with its admitted drafter, under the method that drafts
    /// with it, and that method.
    fn drafted(
        &self,
        admitted: &AdmittedPackage,
        definition: &ModelDefinition,
        method: ModelMethod,
    ) -> Result<(ExecutionManifest, SpeculativeMethod), UnsupportedModel> {
        let policy = ModelPolicy {
            method,
            ..self.policy.clone()
        };
        let model = policy
            .resolve(definition)
            .map_err(UnsupportedModel::mismatch)?;
        let method = match model.method {
            ResolvedMethod::Mtp { .. } => SpeculativeMethod::Mtp,
            ResolvedMethod::DFlash { .. } => {
                match definition
                    .draft
                    .as_ref()
                    .map(|draft| draft.method.variant())
                {
                    Some(DraftVariant::DSpark) => SpeculativeMethod::DSpark,
                    Some(DraftVariant::DFlash2) => SpeculativeMethod::DFlash2,
                    Some(DraftVariant::DFlash) | None => SpeculativeMethod::DFlash,
                }
            }
            ResolvedMethod::Plain => {
                return Err(UnsupportedModel::mismatch(
                    "the drafter resolves to plain generation",
                ));
            }
        };
        let manifest = self
            .manifest(admitted.headers.manifest(), definition.clone(), model)
            .map_err(UnsupportedModel::mismatch)?;
        Ok((manifest, method))
    }

    fn manifest(
        &self,
        package: PackageManifest,
        definition: ModelDefinition,
        model: crate::options::ResolvedModelPolicy,
    ) -> Result<ExecutionManifest, String> {
        ExecutionManifest::new(
            package,
            definition,
            model,
            self.service.clone(),
            ExecutionPath::Native,
            self.device,
            None,
            self.reserves,
        )
    }
}

enum Admission {
    Unsupported {
        unsupported: UnsupportedModel,
        fingerprint: Option<ModelFingerprint>,
    },
    Admitted(Box<AdmittedPackage>),
}

/// A target the engine admits from its headers, with what its drafter's
/// headers decided.
struct AdmittedPackage {
    headers: PackageHeaders,
    definition: ModelDefinition,
    chat: TemplateInspection,
    context_limit: u32,
    fingerprint: Option<ModelFingerprint>,
    drafter: DrafterAdmission,
}

impl AdmittedPackage {
    fn facts(self, drafter: DrafterVerdict) -> ModelFacts {
        ModelFacts {
            capabilities: ModelCapabilities {
                vision: self.definition.vision.is_some(),
                tools: self.chat.tools,
                structured_output: self.chat.structured_output,
                reasoning: ReasoningCapabilities {
                    efforts: self
                        .chat
                        .reasoning
                        .mappings
                        .iter()
                        .map(|mapping| mapping.effort.clone())
                        .collect(),
                    default_effort: self.chat.reasoning.default_effort.clone(),
                },
            },
            template_fingerprint: self.chat.fingerprint,
            context_limit: self.context_limit,
            fingerprint: self.fingerprint,
            drafter,
        }
    }
}

/// What a drafter's headers decide: nothing to draft with, a definition
/// with the drafter bound and the method that drafts with it, or why it
/// cannot draft.
enum DrafterAdmission {
    None,
    Candidate {
        definition: Box<ModelDefinition>,
        method: ModelMethod,
    },
    Rejected(UnsupportedModel),
}

/// A header that is truncated or not a valid GGUF container.
fn unreadable(error: &magnitude_artifacts::Error) -> UnsupportedModel {
    UnsupportedModel::unreadable(error.to_string())
}

/// Admit `package` from its headers: the target's weight encodings, family
/// interpretation, tokenizer and chat templates are independent checks,
/// and every refusal among them is reported together. An admitted target's
/// drafter is admitted the same way, against the target's definition.
fn admit(package: &ModelPackagePaths, layout: Layout) -> Result<Admission, ModelAssessmentError> {
    let mut headers = match PackageHeaders::open(&package.target, package.projector.as_deref()) {
        Ok(headers) => headers,
        Err(error) if error.unreadable() => {
            return Ok(Admission::Unsupported {
                unsupported: unreadable(&error),
                fingerprint: None,
            });
        }
        Err(error) => return Err(ModelAssessmentError::Artifact(error)),
    };
    let target = headers.target();
    let fingerprint = ModelFingerprint::from_directory(target).ok();
    let mut refusals = Vec::new();
    let weights = target
        .tensors
        .iter()
        .map(|tensor| (tensor, Residency::Packed))
        .chain(
            headers
                .projector()
                .into_iter()
                .flat_map(|projector| &projector.tensors)
                .map(|tensor| (tensor, Residency::Dequantized)),
        );
    refusals.extend(unsupported_weights(weights, layout));
    let interpreted = crate::families::recognize(target).and_then(|family| {
        family
            .inspect(target, headers.projector(), headers.identity())
            .map(|definition| (family, definition))
            .map_err(UnsupportedModel::from)
    });
    let tokenizer = TokenizerPayload::from_directory(target);
    // Support and vocabulary are header facts; a load builds the tokenizer.
    let vocabulary = gguf_tokenizer_vocabulary(&tokenizer).map_err(UnsupportedModel::from);
    let chat = TemplatePayload::from_directory(target, &package.target.display().to_string())
        .map_err(|error| error.to_string())
        .and_then(|templates| gguf_templates(&templates, &tokenizer))
        .and_then(|bundle| bundle.inspect_cached())
        .map_err(UnsupportedModel::chat_template);
    if let (Ok((_, definition)), Ok(vocabulary)) = (&interpreted, &vocabulary) {
        if u64::try_from(*vocabulary).ok() != Some(definition.decoder.vocabulary) {
            refusals.push(vocabulary_mismatch(&tokenizer));
        }
    }
    let (interpreted, chat) = match (interpreted, vocabulary, chat) {
        (Ok(interpreted), Ok(_), Ok(chat)) if refusals.is_empty() => (interpreted, chat),
        (interpreted, vocabulary, chat) => {
            refusals.extend(interpreted.err());
            refusals.extend(vocabulary.err());
            refusals.extend(chat.err());
            return Ok(Admission::Unsupported {
                unsupported: UnsupportedModel::union(refusals)
                    .expect("a refused admission has a refusal"),
                fingerprint,
            });
        }
    };
    let (family, definition) = interpreted;
    let context_limit = u32::try_from(definition.decoder.context_limit)
        .map_err(|_| ModelAssessmentError::ContextLimit(definition.decoder.context_limit))?;
    let drafter = admit_drafter(package, &mut headers, family, &definition, layout)?;
    Ok(Admission::Admitted(Box::new(AdmittedPackage {
        headers,
        definition,
        chat,
        context_limit,
        fingerprint,
        drafter,
    })))
}

/// The package's drafter: its separate draft bound against the target
/// definition (its weight encodings and binding checked together), or the
/// target's own draft head. A plain package drafts with nothing; a method
/// that needs a drafter the package does not have is rejected.
fn admit_drafter(
    package: &ModelPackagePaths,
    headers: &mut PackageHeaders,
    family: &dyn ModelFamily,
    definition: &ModelDefinition,
    layout: Layout,
) -> Result<DrafterAdmission, ModelAssessmentError> {
    let rejected =
        |detail: &str| Ok(DrafterAdmission::Rejected(UnsupportedModel::mismatch(detail)));
    let Some(path) = &package.draft else {
        return match package.method {
            ModelMethod::Plain => Ok(DrafterAdmission::None),
            ModelMethod::Auto | ModelMethod::Mtp if definition.head.is_some() => {
                Ok(DrafterAdmission::Candidate {
                    definition: Box::new(definition.clone()),
                    method: ModelMethod::Mtp,
                })
            }
            ModelMethod::Auto => Ok(DrafterAdmission::None),
            ModelMethod::Mtp => rejected("MTP was requested but the target has no draft head"),
            ModelMethod::DFlash | ModelMethod::DSpark | ModelMethod::DFlash2 => {
                rejected("a separate-draft method was requested but the package has no draft")
            }
        };
    };
    if package.method == ModelMethod::Plain {
        return Ok(DrafterAdmission::None);
    }
    match headers.add_draft(path) {
        Ok(()) => {}
        Err(error) if error.unreadable() => {
            return Ok(DrafterAdmission::Rejected(unreadable(&error)));
        }
        Err(error) => return Err(ModelAssessmentError::Artifact(error)),
    }
    let draft = headers.draft().expect("the draft component was just added");
    let mut refusals = Vec::new();
    refusals.extend(unsupported_weights(
        draft
            .tensors
            .iter()
            .map(|tensor| (tensor, Residency::Packed)),
        layout,
    ));
    let bound = bind_draft(family, definition.clone(), Some(draft));
    refusals.extend(bound.draft_unavailable.map(|reason| reason.unsupported()));
    if let Some(unsupported) = UnsupportedModel::union(refusals) {
        return Ok(DrafterAdmission::Rejected(unsupported));
    }
    // An unspecified method drafts with whatever the draft bound as: a
    // separate block drafter, or a draft head.
    let method = match package.method {
        ModelMethod::Auto if bound.definition.draft.is_none() => ModelMethod::Mtp,
        method => method,
    };
    Ok(DrafterAdmission::Candidate {
        definition: Box::new(bound.definition),
        method,
    })
}

/// How a stored tensor is held: packed in the backend's resident layout, or
/// (a projector's) dequantized on the host into a dense form.
#[derive(Clone, Copy)]
enum Residency {
    Packed,
    Dequantized,
}

/// Every tensor whose stored encoding has no import here, counted per GGML
/// type: the same element maps weight planning applies
/// ([`source_element`], and [`resident_element`] in `layout`), so a type
/// added to them is admitted here with no other change.
fn unsupported_weights<'a>(
    tensors: impl IntoIterator<Item = (&'a TensorDescriptor, Residency)>,
    layout: Layout,
) -> Option<UnsupportedModel> {
    let mut counts = BTreeMap::<&'static str, u32>::new();
    for (tensor, residency) in tensors {
        let width = tensor.shape.last().copied().unwrap_or(1);
        let imported = source_element(tensor.encoding, width).is_some()
            && match residency {
                Residency::Packed => {
                    resident_element(tensor.encoding, width, DType::F32, layout).is_some()
                }
                Residency::Dequantized => true,
            };
        if !imported {
            *counts.entry(tensor.encoding.ggml_name()).or_default() += 1;
        }
    }
    if counts.is_empty() {
        return None;
    }
    let reason = UnsupportedReason::WeightTypes {
        types: counts
            .into_iter()
            .map(|(ggml_type, tensor_count)| WeightTypeCount {
                ggml_type: ggml_type.to_owned(),
                tensor_count,
            })
            .collect(),
    };
    let detail = format!(
        "{reason} have no import into the `{}` layout",
        layout.as_str()
    );
    Some(UnsupportedModel::reason(reason, detail))
}
