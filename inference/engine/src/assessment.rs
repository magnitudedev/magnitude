//! Metadata-only model assessment from a package path.
//!
//! Opens only the GGUF headers of the target, its optional projector and
//! drafter. Independent header checks (weight encodings, the family's
//! interpretation, the tokenizer and the chat templates) each run and their
//! refusals are reported together. An admitted target is planned through
//! [`crate::planning::plan_ceiling`] (the production planning inputs)
//! with its drafter, and the target's plan alone is charged beside it from
//! its headers, which decides whether it runs here; a drafter that does not
//! bind or plan is `Rejected` without affecting the target, which is then
//! planned alone. A projector the family cannot bind is `Rejected`
//! the same way: the target is interpreted and planned without it, taking
//! text input only. The selected plan is assessed against stable memory
//! capacity and, once the measurement job has stored them, the selected
//! executor's measured costs. No device is opened, no weight payload is
//! read, nothing is decoded and nothing is measured.

use crate::error::{
    classify_catalog, classify_graph, classify_plan, PlanOutcome, UnsupportedModel,
    UnsupportedReason, WeightTypeCount,
};
use crate::fingerprint::ModelFingerprint;
use crate::host::{bind_draft, interpret_target, vocabulary_mismatch, InterpretedTarget};
use crate::options::{ExecutionManifest, ModelMethod, ModelPolicy, ResolvedMethod};
use crate::planning::{plan_backend, plan_ceiling, plan_resident, ExecutionPlanningError};
use magnitude_artifacts::{
    gguf::{Directory, FileHeaders, HeaderSource, TensorDescriptor},
    ArtifactIdentity, PackageHeaders, PackageIdentity, PackageManifest, TemplatePayload,
    TokenizerPayload,
};
use magnitude_chat::{
    artifacts::{gguf_templates, gguf_tokenizer_vocabulary},
    TemplateInspection, TokenizerError,
};
use magnitude_executor::{
    assessment::{
        assess_execution_fit, decode_pricing, decode_step, estimate_execution, performance_depths,
        prepare_execution_assessment, undrafted_memory_charge, AssessmentError,
        AssessmentRequest,
        DomainFit, ExecutionAssessment, ExecutionShortfall,
        ExecutorCosts, ExpertShares, PerformanceEstimate, PreparedExecutionAssessment,
        PointRequests, Priced, Residency as FitResidency, StoredCosts, StreamedFigures, StreamedLayers,
    },
    platform::{self, DeviceRequest, MemoryReserves, PlatformError, SelectedDevice},
    held_storage, resident_layout, scaled_source_element, stored_weights, AssessmentMemoryCharge,
    ChargedDraft, DraftedCharge, ExpertBankPlan, ExecutionPath, ExecutionPlanDraft, HeldCharge, KernelCache, PlanPrices,
};
use magnitude_family_contracts::{DraftVariant, ModelDefinition, ModelFamily, ScaledActivations};
use magnitude_scheduler::ServiceLimits;
use seismic::{DType, DeviceTopology, Devices, Layout, Status};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The package components to assess and the generation method the bundle
/// declares (`Auto` for a standalone package). Only their headers are read.
/// `target` is the target GGUF or the first shard of a split GGUF (the other
/// shards are found beside it by the split naming convention); `draft` is a
/// separate drafter (DFlash, DSpark, DFlash2, LiLiCorr, a Gemma assistant or a
/// prediction head). A drafter the method cannot draft with is `Rejected`
/// and the target is assessed without it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPackagePaths {
    pub target: PathBuf,
    pub projector: Option<PathBuf>,
    pub draft: Option<PathBuf>,
    pub method: ModelMethod,
}

/// The structural part of the configuration every model is assessed on:
/// the device a native load selects, the topology, one status sample and
/// the serving policy. Discovery establishes it measuring nothing, so the
/// service has it at once; what of it the measurement job records (its
/// graphs' cost) is read again as the records land ([`Self::stored`]).
pub struct AssessmentStructure {
    pub topology: Arc<DeviceTopology>,
    pub status: Status,
    pub device: DeviceRequest,
    pub selected: SelectedDevice,
    pub reserves: MemoryReserves,
    pub policy: ModelPolicy,
    pub service: ServiceLimits,
    /// The context every model is planned at, as a load serving it at that
    /// context holds it (`host::resolve_served_context`): its declared
    /// limit when none.
    pub served_context: Option<usize>,
}

impl AssessmentStructure {
    /// Select the device a native load would select, its graphs priced as
    /// a load prices them from the kernel cache at `kernels`
    /// (`platform::select_recorded`), and take one status sample. Opens no
    /// device and measures nothing.
    pub fn discover(
        catalog: &Devices,
        device: DeviceRequest,
        kernels: Option<&Path>,
        reserves: MemoryReserves,
        policy: ModelPolicy,
        service: ServiceLimits,
    ) -> Result<Self, ModelAssessmentError> {
        let selected = platform::select_recorded(catalog, ExecutionPath::Native, device, &reserves, kernels)
            .map_err(ModelAssessmentError::Platform)?;
        let topology = catalog.topology();
        Ok(Self {
            status: catalog.status(&topology, selected.info.id()),
            topology,
            device,
            selected,
            reserves,
            policy,
            service,
            served_context: None,
        })
    }

    /// What this machine's stored records give assessment now: this
    /// structure with what it reads of the records priced from them as they
    /// stand (its graphs, [`platform::graph_pricing`]; itself while they
    /// price them as it does), under its measurements
    /// ([`Self::stored_measurements`]). A record the job stores after
    /// discovery so revises every input of assessment it enters.
    pub fn stored(self: &Arc<Self>, kernels: &Path, storage: &Path) -> AssessmentSetup {
        let structure = match platform::graph_pricing(&self.topology, &self.selected.info, Some(kernels)) {
            Ok(pricing) if pricing == self.selected.graph_cost => Arc::clone(self),
            Ok(pricing) => Arc::new(Self {
                topology: Arc::clone(&self.topology),
                status: self.status.clone(),
                device: self.device,
                selected: SelectedDevice {
                    graph_cost: pricing,
                    ..self.selected.clone()
                },
                reserves: self.reserves,
                policy: self.policy.clone(),
                service: self.service.clone(),
                served_context: self.served_context,
            }),
            Err(error) => {
                return AssessmentSetup {
                    structure: Arc::clone(self),
                    measurements: Measurements::Unavailable {
                        reason: format!("the graph memory records: {error}"),
                    },
                };
            }
        };
        let measurements = structure.stored_measurements(kernels, storage);
        AssessmentSetup { structure, measurements }
    }

    /// What this machine's stored records give assessment now: the selected
    /// executor's costs, derived from the records of its cost points in the
    /// kernel cache at `kernels`, and the streamed figures of the model store
    /// at `storage` ([`StreamedFigures`]), opening no device and measuring
    /// nothing.
    /// The costs are `Measured` once the probe and the figures no other
    /// prices are stored (`Required`), pending while a
    /// point request is outstanding (`PointRequests`: a load's, or a later
    /// point of the costs with no record yet, whose figure is priced from
    /// those read until the job takes it); before then they are `Pending`. Costs that
    /// cannot be derived from what is stored (a record store that cannot be
    /// read, or a required figure recorded unobtainable) are `Unavailable`.
    pub fn stored_measurements(&self, kernels: &Path, storage: &Path) -> Measurements {
        stored_measurements(&self.topology, &self.selected.info, kernels, storage)
    }

    /// The room a measurement job on the selected executor takes of its
    /// device's storage at most, its worker's own bytes among it
    /// (`ExecutorCosts::room`): what a job beside a resident model claims of
    /// it first.
    pub fn measurement_room(&self) -> Result<u64, String> {
        ExecutorCosts::room(&self.topology, &self.selected.info).map_err(|error| error.to_string())
    }
}

/// What the stored records give `info` of `topology` now
/// ([`AssessmentStructure::stored_measurements`]): its costs from the
/// kernel cache at `kernels` and the streamed figures of the model store at
/// `storage`, opening no device and measuring nothing.
pub fn stored_measurements(topology: &DeviceTopology, info: &seismic::Executor, kernels: &Path, storage: &Path) -> Measurements {
    let records = KernelCache::measurement_records(kernels);
    let walked = ExecutorCosts::stored(topology, info, &records);
    let costs_outstanding = walked.as_ref().ok().and_then(StoredCosts::outstanding);
    match walked {
        Ok(StoredCosts::Complete {
            costs,
            records: mut cost_records,
            pending,
        }) => {
            let (streamed, streamed_records) = StreamedFigures::stored(topology, info, storage, &records);
            cost_records.extend(streamed_records);
            // The job has points left to measure: a later point of the
            // standing requests, or a load's runtime request. Neither
            // enters an estimate.
            let pending = pending
                || match PointRequests::new(kernels).outstanding(topology, info, storage, &records, costs_outstanding) {
                    Ok(outstanding) => !outstanding.is_empty(),
                    Err(error) => {
                        return Measurements::Unavailable {
                            reason: format!("the point requests: {error}"),
                        };
                    }
                };
            Measurements::Measured {
                costs: Arc::new(costs),
                records: Arc::new(cost_records),
                pending,
                streamed,
            }
        }
        Ok(StoredCosts::Missing(_)) => Measurements::Pending,
        Err(error) => Measurements::Unavailable {
            reason: error.to_string(),
        },
    }
}

/// The prices a plan of `definition` as `plan` (its plan before any bank)
/// settles its holdings on (`PlanPrices`): the executor's costs and the
/// streamed figures `measurements` hold, and its decode on those costs in
/// each form, `plan`'s and the banked plan's streamed form's
/// (`assessment::decode_step`). Pending or unavailable as the measurements
/// or their streamed figures are: a plan that holds a bank waits for them.
pub fn plan_prices(measurements: &Measurements, plan: &ExecutionPlanDraft) -> Priced<PlanPrices> {
    match measurements {
        Measurements::Pending => Priced::Pending("the executor's costs are not measured yet".into()),
        Measurements::Unavailable { reason } => Priced::Unavailable(reason.clone()),
        Measurements::Measured { costs, streamed, .. } => streamed.as_ref().and_then(|figures| {
            let (resident, steps) = (plan.clone(), ExecutorCosts::clone(costs));
            Priced::Measured(PlanPrices::new(
                ExecutorCosts::clone(costs),
                *figures,
                Box::new(move |definition, banked| decode_step(definition, &resident, banked, steps.clone())),
            ))
        }),
    }
}

/// What this machine's measurements give assessment. Only the measurement
/// job takes them (device topology spec §2.15); assessment reads what it
/// stored.
#[derive(Clone)]
pub enum Measurements {
    /// The job has not stored the probe or a figure no other prices yet:
    /// fit is assessed and speed is pending.
    Pending,
    /// The stored records give no costs, and why: the probe is recorded
    /// unobtainable (the executor cannot stream its own weights), or the
    /// records cannot be read, and every fitting model's speed is
    /// unavailable.
    Unavailable { reason: String },
    /// The selected executor's costs and the streamed figures, and the
    /// stored records they were derived from, which identify them. Every
    /// model's speed is estimated from them; while `pending`, the job has a
    /// later point to measure, whose figure is priced until it is.
    Measured {
        costs: Arc<ExecutorCosts>,
        records: Arc<Vec<(seismic::PointKey, seismic::measure::Record)>>,
        pending: bool,
        streamed: Priced<StreamedFigures>,
    },
}

/// The configuration every model is assessed on: its structure, and what
/// is measured of this machine.
#[derive(Clone)]
pub struct AssessmentSetup {
    pub structure: Arc<AssessmentStructure>,
    pub measurements: Measurements,
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
    LiLiCorr,
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

/// Whether the engine binds the package's projector: none was present
/// (`None`), it binds and the model takes image input (`Bound`), or it does
/// not (`Rejected`, with every reason; the target is assessed and loads
/// without it, taking text input only).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectorVerdict {
    None,
    Bound,
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
    pub projector: ProjectorVerdict,
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
        execution: ModelExecution,
    },
    Unsupported {
        unsupported: UnsupportedModel,
        fingerprint: Option<ModelFingerprint>,
    },
}

/// How a package executes on the setup's selected device: a fitting one
/// with how it holds its weights and its decode speed held so, its plan's
/// expert bank among them, or why it does not fit.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelExecution {
    Fits {
        domains: Vec<DomainFit>,
        residency: ModelResidency,
        /// The plan's expert bank, whole or streamed: its slots, ceiling,
        /// host slots, prompt places and pass. None for a plan that banks
        /// no experts.
        bank: Option<magnitude_executor::ExpertBankPlan>,
    },
    DoesNotFit(ExecutionShortfall),
}

/// How a fitting package holds its weights, and its decode speed held so.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelResidency {
    /// Every weight in memory.
    Resident { performance: Performance },
    /// The routed experts streamed through an expert bank of `shares`:
    /// `performance` at its share, `floor_performance` at its floor.
    Streamed {
        shares: ExpertShares,
        performance: Performance,
        floor_performance: Performance,
    },
}

/// A fitting model's decode speed, resident or streamed, as this machine's
/// measurements give it.
#[derive(Clone, Debug, PartialEq)]
pub enum Performance {
    /// The measurements it is estimated from are not stored yet.
    Pending,
    /// A measurement it is estimated from cannot be taken on this machine.
    Unavailable { reason: String },
    /// One estimate per requested depth, ascending.
    Estimated(Vec<PerformanceEstimate>),
}

/// An operational failure: no result is produced for the package. A
/// truncated or invalid GGUF is not one (it is `Unsupported` with
/// [`UnsupportedReason::Unreadable`]); a file that cannot be opened is.
#[derive(Debug)]
pub enum ModelAssessmentError {
    Artifact(magnitude_artifacts::Error),
    ContextLimit(u64),
    Configuration(String),
    /// The weight cap is below the load's least weights.
    WeightBound(crate::error::WeightBoundBelowWeights),
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
            Self::WeightBound(error) => error.fmt(formatter),
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
        /// The definition `draft` plans.
        definition: ModelDefinition,
        draft: ExecutionPlanDraft,
        execution: PreparedExecutionAssessment,
    },
}

/// Complete a prepared model against stable capacity and, when it fits,
/// estimate its decode speed with `sequences` conversations decoding at
/// once from the setup's measurements: arithmetic over the prepared demand.
pub fn finish_model_assessment(
    prepared: &PreparedModelAssessment,
    setup: &AssessmentSetup,
    performance_depths: &[u32],
    sequences: u32,
) -> Result<ModelAssessment, ModelAssessmentError> {
    let (facts, definition, draft, preparation) = match prepared {
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
            definition,
            draft,
            execution,
        } => (facts, definition, draft, execution),
    };
    let structure = &setup.structure;
    let request = AssessmentRequest {
        context_limit: facts.context_limit,
        performance_depths: performance_depths.to_vec(),
        reserves: structure.reserves,
        sequences,
    };
    let fit = assess_execution_fit(
        definition,
        preparation,
        draft,
        &structure.topology,
        &structure.status,
        &request,
        &plan_prices(&setup.measurements, draft),
    )
    .map_err(ModelAssessmentError::Assessment)?;
    let execution = match fit {
        ExecutionAssessment::DoesNotFit(shortfall) => ModelExecution::DoesNotFit(shortfall),
        ExecutionAssessment::Fits(fit) => ModelExecution::Fits {
            residency: match fit.residency() {
                FitResidency::Streamed(bank) => {
                    let shares = ExpertShares::of(&bank);
                    let at = |held| streamed_performance(&setup.measurements, preparation, draft, &bank, &request, held);
                    ModelResidency::Streamed {
                        performance: at(shares.held(shares.slots()))?,
                        floor_performance: at(shares.held(shares.floor()))?,
                        shares,
                    }
                }
                FitResidency::Resident => ModelResidency::Resident {
                    performance: resident_performance(
                        &setup.measurements,
                        preparation,
                        draft,
                        &request,
                    )?,
                },
            },
            domains: fit.domains,
            bank: fit.bank,
        },
    };
    Ok(ModelAssessment::Assessed {
        facts: facts.clone(),
        execution,
    })
}

/// A resident fit's decode speed from the setup's measurements.
fn resident_performance(
    measurements: &Measurements,
    preparation: &PreparedExecutionAssessment,
    draft: &ExecutionPlanDraft,
    request: &AssessmentRequest,
) -> Result<Performance, ModelAssessmentError> {
    Ok(match measurements {
        Measurements::Pending => Performance::Pending,
        Measurements::Unavailable { reason } => Performance::Unavailable {
            reason: reason.clone(),
        },
        Measurements::Measured { costs, .. } => Performance::Estimated(
            estimate_execution(preparation, draft, costs, request)
                .map_err(ModelAssessmentError::Assessment)?,
        ),
    })
}

/// A streamed fit's decode speed holding `held` of each routed layer's
/// experts: pending or unavailable as the setup's measurements or their
/// streamed figures are, and once both are measured, [`streamed_estimate`].
fn streamed_performance(
    measurements: &Measurements,
    preparation: &PreparedExecutionAssessment,
    draft: &ExecutionPlanDraft,
    bank: &ExpertBankPlan,
    request: &AssessmentRequest,
    held: magnitude_executor::assessment::HeldSlots,
) -> Result<Performance, ModelAssessmentError> {
    match measurements {
        Measurements::Pending => Ok(Performance::Pending),
        Measurements::Unavailable { reason } => Ok(Performance::Unavailable { reason: reason.clone() }),
        Measurements::Measured { costs, streamed, .. } => match streamed {
            Priced::Pending(_) => Ok(Performance::Pending),
            Priced::Unavailable(reason) => Ok(Performance::Unavailable { reason: reason.clone() }),
            Priced::Measured(figures) => streamed_estimate(costs, figures, preparation, draft, bank, request, held),
        },
    }
}

/// A streamed decode's speed holding `held` of each routed layer's experts
/// (its bank's slots and its host tier's), at each depth `request` names:
/// the plan's streamed layers priced by the generic miss model and the one
/// step model ([`StreamedLayers::step_seconds`]) at the request's rows (a
/// row a sequence), their file reads along the model store's read line, on
/// `C` the step's estimate there in each form (`preparation` is the plan's,
/// `draft` the plan, `PreparedExecutionAssessment::form_seconds`), `bank`
/// the fit's: the plan's structure and the machine's stored figures alone.
pub fn streamed_estimate(
    costs: &ExecutorCosts,
    figures: &StreamedFigures,
    preparation: &PreparedExecutionAssessment,
    draft: &ExecutionPlanDraft,
    bank: &ExpertBankPlan,
    request: &AssessmentRequest,
    held: magnitude_executor::assessment::HeldSlots,
) -> Result<Performance, ModelAssessmentError> {
    let layers = match StreamedLayers::of_bank(bank, costs, figures) {
        Ok(layers) => layers,
        Err(reason) => return Ok(Performance::Unavailable { reason: format!("the streamed layers: {reason}") }),
    };
    let pricing = decode_pricing(draft, costs);
    let mut estimates = Vec::new();
    for depth in performance_depths(request.context_limit, &request.performance_depths) {
        let compute = preparation
            .form_seconds(&pricing, u64::from(request.sequences), depth)
            .map_err(ModelAssessmentError::Assessment)?;
        let step = layers.step_seconds(None, figures.read, held, f64::from(request.sequences), compute);
        estimates.push(PerformanceEstimate { context_tokens: depth, tokens_per_second: 1. / step.total_seconds });
    }
    Ok(Performance::Estimated(estimates))
}

/// The executor's costs as a walk of the stored records found them
/// (`ExecutorCosts::stored`), which a streamed load's measures price on:
/// pending until the job stores the probe, unavailable where they cannot
/// be derived.
pub fn priced_costs(
    walked: Result<StoredCosts, magnitude_executor::assessment::ExecutorCostsError>,
) -> magnitude_executor::assessment::Priced<ExecutorCosts> {
    use magnitude_executor::assessment::Priced;
    match walked {
        Ok(StoredCosts::Complete { costs, .. }) => Priced::Measured(costs),
        Ok(StoredCosts::Missing(point)) => Priced::Pending(format!("the executor's costs are not measured yet ({point:?})")),
        Err(error) => Priced::Unavailable(error.to_string()),
    }
}

/// Assess one package on the setup's selected device.
pub fn assess_model(
    package: &ModelPackagePaths,
    setup: &AssessmentSetup,
    performance_depths: &[u32],
    sequences: u32,
) -> Result<ModelAssessment, ModelAssessmentError> {
    let prepared = prepare_model_assessment(package, &setup.structure)?;
    finish_model_assessment(&prepared, setup, performance_depths, sequences)
}

/// Prepare one package from its headers on the selected execution
/// configuration: admission (every header refusal at once), then the plan
/// assessed, with its decode demand and checked memory charge: the drafted
/// plan when a drafter is admitted, the target's plan alone otherwise. A
/// drafter is `Bound` with what its plan adds to the target's alone at the
/// fit depth and the drafted plan's holding of experts: its weights, state
/// and prepared resources on the device
/// ([`magnitude_executor::AssessmentMemoryCharge::serving_bytes`]) and the
/// host memory of its kernels and graphs
/// ([`magnitude_executor::AssessmentMemoryCharge::host_bytes`]), the target
/// alone charged from its headers beside the drafted plan's certification
/// (`drafter_memory_bytes`). Reads only the structure, so a preparation
/// holds under every measurement of it.
pub fn prepare_model_assessment(
    package: &ModelPackagePaths,
    structure: &AssessmentStructure,
) -> Result<PreparedModelAssessment, ModelAssessmentError> {
    prepare_model_assessment_from(package, &FileHeaders, structure)
}

/// [`prepare_model_assessment`] with the package's headers from `headers`,
/// so a host that already read them never has them read again.
pub fn prepare_model_assessment_from(
    package: &ModelPackagePaths,
    headers: &dyn HeaderSource,
    structure: &AssessmentStructure,
) -> Result<PreparedModelAssessment, ModelAssessmentError> {
    let layout = resident_layout(ExecutionPath::Native, structure.selected.info.backend());
    let admitted = match admit(package, headers, layout)? {
        Admission::Unsupported {
            unsupported,
            fingerprint,
        } => {
            return Ok(PreparedModelAssessment::Unsupported {
                unsupported,
                fingerprint,
            });
        }
        Admission::Admitted(admitted) => admitted.served_at(structure.served_context)?,
    };
    let environment = Environment {
        policy: &structure.policy,
        service: &structure.service,
        device: structure.device,
        reserves: structure.reserves,
    };
    let plain = environment
        .plain(&admitted)
        .map_err(ModelAssessmentError::Configuration)?;
    let drafted = match &admitted.drafter {
        DrafterAdmission::None => Err(DrafterVerdict::None),
        DrafterAdmission::Rejected(unsupported) => {
            Err(DrafterVerdict::Rejected(unsupported.clone()))
        }
        DrafterAdmission::Candidate { definition, method } => environment
            .drafted(&admitted, definition, *method)
            .and_then(|(manifest, method)| {
                match prepare_plan::<DraftedCharge>(manifest, &structure.selected, admitted.context_limit) {
                    Ok(planned) => Ok((method, planned)),
                    Err(Refusal::Unsupported(unsupported)) => Err(unsupported),
                    Err(Refusal::Failed(error)) => {
                        Err(UnsupportedModel::mismatch(error.to_string()))
                    }
                }
            })
            .map_err(DrafterVerdict::Rejected),
    };
    // The target's plan alone is a package's refusal: a drafter's own is
    // the drafter's.
    let refused = |refusal| match refusal {
        Refusal::Unsupported(unsupported) => Ok(PreparedModelAssessment::Unsupported {
            unsupported,
            fingerprint: admitted.fingerprint.clone(),
        }),
        Refusal::Failed(error) => Err(error),
    };
    let (drafter, definition, draft, execution) = match drafted {
        Ok((method, (definition, planned, execution))) => {
            match drafter_memory_bytes(plain, &structure.selected, &planned) {
                Ok(memory_bytes) => {
                    (DrafterVerdict::Bound { method, memory_bytes }, definition, planned.into_draft(), execution)
                }
                Err(refusal) => return refused(refusal),
            }
        }
        Err(verdict) => {
            match prepare_plan::<AssessmentMemoryCharge>(plain, &structure.selected, admitted.context_limit) {
                Ok((definition, planned, execution)) => (verdict, definition, planned.into_draft(), execution),
                Err(refusal) => return refused(refusal),
            }
        }
    };
    Ok(PreparedModelAssessment::Planned {
        facts: admitted.facts(drafter),
        definition,
        draft,
        execution,
    })
}

/// What a drafter adds to the target's plan alone (`plain`) at the drafted
/// plan's holding of experts (`ExecutionPlanDraft::holding_as`): the
/// device's serving charge and the host's, of `drafted` less of the target
/// alone. The target alone's target family is the drafted plan's without its
/// drafter's writes, certified with it (`ChargedDraft::undrafted`); every
/// other term is the target's own, from its headers.
fn drafter_memory_bytes(
    plain: ExecutionManifest,
    selected: &SelectedDevice,
    drafted: &ChargedDraft<DraftedCharge>,
) -> Result<u64, Refusal> {
    let backend = selected.info.backend();
    let resident = plan_resident(&plain, selected).map_err(|error| match error {
        ExecutionPlanningError::Plan(error) => Refusal::of(classify_plan(error, backend)),
        error => Refusal::Failed(ModelAssessmentError::Planning(error.to_string())),
    })?;
    let alone = resident
        .holding_as(&plain.definition, drafted.draft())
        .map_err(|error| Refusal::of(classify_plan(error, backend)))?;
    let charge = undrafted_memory_charge(&plain.definition, &alone, drafted.undrafted().clone()).map_err(|error| match error {
        AssessmentError::Graph(error) => Refusal::of(classify_graph(error)),
        error => Refusal::Failed(ModelAssessmentError::Assessment(error)),
    })?;
    let held = |charge: &AssessmentMemoryCharge| {
        charge
            .host_bytes()
            .and_then(|host| charge.serving_bytes.checked_add(host).ok_or_else(|| "held charge overflows".to_owned()))
            .map_err(|error| Refusal::Failed(ModelAssessmentError::Planning(error)))
    };
    let (with, without) = (held(drafted.charge())?, held(&charge)?);
    assert!(
        with >= without,
        "a drafted plan holds {with} bytes, less than its target's plan alone ({without}) at the same holding of experts"
    );
    Ok(with - without)
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
    let admitted = match admit(package, &FileHeaders, layout)? {
        Admission::Unsupported { unsupported, .. } => {
            return Ok(KernelInventory::Unsupported(unsupported));
        }
        Admission::Admitted(admitted) => *admitted,
    };
    let environment = Environment {
        policy,
        service,
        device: DeviceRequest::Backend(backend),
        reserves: MemoryReserves::of_machine(None),
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
        PlanOutcome::WeightBound(error) => Err(ModelAssessmentError::WeightBound(error)),
        PlanOutcome::Internal(reason) => Err(ModelAssessmentError::Planning(reason)),
        PlanOutcome::PricesPending(reason) => unpriced(&reason),
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

impl Refusal {
    /// A planner's or kernel domain's outcome as a refusal.
    fn of(outcome: PlanOutcome) -> Self {
        match outcome {
            PlanOutcome::Unsupported(unsupported) => Self::Unsupported(unsupported),
            PlanOutcome::WeightBound(error) => Self::Failed(ModelAssessmentError::WeightBound(error)),
            PlanOutcome::Internal(reason) => Self::Failed(ModelAssessmentError::Planning(reason)),
            PlanOutcome::PricesPending(reason) => unpriced(&reason),
        }
    }
}

/// Assessment plans its structure and ceilings, never a priced settle (`with_bank`): no plan of
/// its waits on the machine's prices.
fn unpriced(reason: &str) -> ! {
    unreachable!("an assessment plan settled on the machine's prices, which are pending: {reason}")
}

/// Plan `manifest` on the selected device, then its decode demand and
/// checked memory charge, which builds every graph a load prepares. Planner
/// refusals and kernel domains classify exactly as a load's do.
fn prepare_plan<C: HeldCharge>(
    manifest: ExecutionManifest,
    selected: &SelectedDevice,
    context_limit: u32,
) -> Result<(ModelDefinition, ChargedDraft<C>, PreparedExecutionAssessment), Refusal> {
    let planned = match plan_ceiling(&manifest, selected) {
        Ok(planned) => planned,
        Err(ExecutionPlanningError::Plan(error)) => {
            return Err(Refusal::of(classify_plan(error, selected.info.backend())));
        }
        Err(error) => {
            return Err(Refusal::Failed(ModelAssessmentError::Planning(
                error.to_string(),
            )));
        }
    };
    match prepare_execution_assessment(&manifest.definition, &planned, context_limit) {
        Ok(execution) => Ok((manifest.definition, planned, execution)),
        Err(AssessmentError::Graph(error)) => Err(Refusal::of(classify_graph(error))),
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
                    Some(DraftVariant::LiLiCorr) => SpeculativeMethod::LiLiCorr,
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
    projector: ProjectorVerdict,
    drafter: DrafterAdmission,
}

impl AdmittedPackage {
    /// The package planned at `served` tokens of context, as a load serving
    /// it at that context holds it; at its declared limit when none.
    fn served_at(mut self: Box<Self>, served: Option<usize>) -> Result<Self, ModelAssessmentError> {
        let limit = crate::host::resolve_served_context(served, self.definition.decoder.context_limit)
            .map_err(|error| ModelAssessmentError::Configuration(error.to_string()))?;
        self.definition.decoder.context_limit = limit;
        // A drafter candidate drafts for the target's own definition, so it
        // serves the same context.
        if let DrafterAdmission::Candidate { definition, .. } = &mut self.drafter {
            definition.decoder.context_limit = limit;
        }
        self.context_limit = u32::try_from(limit).map_err(|_| ModelAssessmentError::ContextLimit(limit))?;
        Ok(*self)
    }

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
            projector: self.projector,
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

/// The verdict an artifact error is: weight types the parser doesn't know are unsupported weight
/// types, and a header that is truncated or not a valid GGUF container is unreadable. Any other
/// error (a source that couldn't be read) is no verdict.
fn artifact_refusal(error: &magnitude_artifacts::Error) -> Option<UnsupportedModel> {
    if let Some(types) = error.unknown_weight_types() {
        return Some(UnsupportedModel::reason(
            UnsupportedReason::WeightTypes { types },
            error.to_string(),
        ));
    }
    error
        .unreadable()
        .then(|| UnsupportedModel::unreadable(error.to_string()))
}

/// Admit `package` from its headers: the target's weight encodings, family
/// interpretation, tokenizer and chat templates are independent checks,
/// and every refusal among them is reported together. An admitted target's
/// drafter is admitted the same way, against the target's definition.
fn admit(
    package: &ModelPackagePaths,
    source: &dyn HeaderSource,
    layout: Layout,
) -> Result<Admission, ModelAssessmentError> {
    let mut headers = match PackageHeaders::open_from(
        &package.target,
        package.projector.as_deref(),
        source,
    ) {
        Ok(headers) => headers,
        Err(error) => match artifact_refusal(&error) {
            Some(unsupported) => {
                return Ok(Admission::Unsupported {
                    unsupported,
                    fingerprint: None,
                });
            }
            None => return Err(ModelAssessmentError::Artifact(error)),
        },
    };
    let target = headers.target();
    let fingerprint = ModelFingerprint::from_directory(target).ok();
    let mut refusals = Vec::new();
    refusals.extend(unsupported_weights(&target.tensors, Residency::Packed, layout));
    let interpreted = interpret_header(
        target,
        headers.projector(),
        headers.identity(),
        &package.target.display().to_string(),
    );
    // A bound projector loads its weights; one that does not bind loads none.
    if let (Ok((_, interpreted, _)), Some(projector)) = (&interpreted, headers.projector()) {
        if interpreted.projector_unavailable.is_none() {
            refusals.extend(unsupported_weights(
                &projector.tensors,
                Residency::Dequantized,
                layout,
            ));
        }
    }
    let (
        family,
        InterpretedTarget {
            definition,
            projector_unavailable,
        },
        chat,
    ) = match interpreted {
        Ok(interpreted) if refusals.is_empty() => interpreted,
        interpreted => {
            refusals.extend(interpreted.err().into_iter().flatten());
            return Ok(Admission::Unsupported {
                unsupported: UnsupportedModel::union(refusals)
                    .expect("a refused admission has a refusal"),
                fingerprint,
            });
        }
    };
    let projector = match (headers.projector(), projector_unavailable) {
        (_, Some(unsupported)) => ProjectorVerdict::Rejected(unsupported),
        (Some(_), None) => ProjectorVerdict::Bound,
        (None, None) => ProjectorVerdict::None,
    };
    let context_limit = u32::try_from(definition.decoder.context_limit)
        .map_err(|_| ModelAssessmentError::ContextLimit(definition.decoder.context_limit))?;
    let drafter = admit_drafter(package, source, &mut headers, family, &definition, layout)?;
    Ok(Admission::Admitted(Box::new(AdmittedPackage {
        headers,
        definition,
        chat,
        context_limit,
        fingerprint,
        projector,
        drafter,
    })))
}

/// A target header's interpretation: its family's definition with its
/// projector bound when the family binds it ([`interpret_target`]), the
/// tokenizer it declares (checked against the definition's vocabulary) and
/// its chat templates, as independent checks whose refusals are reported
/// together. It never reads weight encodings, a drafter or a device.
fn interpret_header(
    target: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
    name: &str,
) -> Result<(&'static dyn ModelFamily, InterpretedTarget, TemplateInspection), Vec<UnsupportedModel>>
{
    let interpreted = crate::families::recognize(target).and_then(|family| {
        interpret_target(family, target, projector, identity)
            .map(|interpreted| (family, interpreted))
    });
    let tokenizer = TokenizerPayload::from_directory(target);
    // Support and vocabulary are header facts; a load builds the tokenizer.
    let vocabulary = gguf_tokenizer_vocabulary(&tokenizer).map_err(UnsupportedModel::from);
    let chat = TemplatePayload::from_directory(target, name)
        .map_err(|error| error.to_string())
        .and_then(|templates| gguf_templates(&templates, &tokenizer))
        .and_then(|bundle| bundle.inspect_cached())
        .map_err(UnsupportedModel::chat_template);
    let mut refusals = Vec::new();
    if let (Ok((_, interpreted)), Ok(vocabulary)) = (&interpreted, &vocabulary) {
        if u64::try_from(*vocabulary).ok() != Some(interpreted.definition.decoder.vocabulary) {
            refusals.push(vocabulary_mismatch(&tokenizer));
        }
    }
    match (interpreted, vocabulary, chat) {
        (Ok((family, interpreted)), Ok(_), Ok(chat)) if refusals.is_empty() => {
            Ok((family, interpreted, chat))
        }
        (interpreted, vocabulary, chat) => {
            refusals.extend(interpreted.err());
            refusals.extend(vocabulary.err());
            refusals.extend(chat.err());
            Err(refusals)
        }
    }
}

/// How much of a GGUF header a judgment has: all of it through its tensor
/// directory, or only its leading metadata entries (a prefix that ends
/// before the tokenizer's arrays, say).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderExtent {
    Whole,
    Leading,
}

/// Why no file with `target`'s header can run, whatever its weight types
/// and wherever it runs: the refusals of the header interpretation an
/// assessment admits a target by, without its weight encodings, a projector,
/// a drafter or a device. With the `Whole` header that is the family's
/// interpretation, the tokenizer and the chat templates. With `Leading`
/// metadata only, it is what those entries settle by themselves: the
/// architecture and the tokenizer scheme and flags they declare. Entries a
/// prefix doesn't reach refuse nothing, so it never refuses what the whole
/// header would admit. `None` when nothing is refused.
pub fn header_refusal(target: &Directory, extent: HeaderExtent) -> Option<UnsupportedModel> {
    let refusals = match extent {
        HeaderExtent::Whole => {
            let identity = PackageIdentity {
                target: ArtifactIdentity([0; 32]),
                projector: None,
            };
            interpret_header(target, None, identity, "header").err()?
        }
        HeaderExtent::Leading => {
            let tokenizer = match gguf_tokenizer_vocabulary(&TokenizerPayload::from_directory(target)) {
                Err(error @ TokenizerError::Unsupported(_)) => Some(UnsupportedModel::from(error)),
                Ok(_) | Err(TokenizerError::Invalid(_)) => None,
            };
            crate::families::recognize(target)
                .err()
                .into_iter()
                .chain(tokenizer)
                .collect()
        }
    };
    UnsupportedModel::union(refusals)
}

/// The package's drafter: its separate draft bound against the target
/// definition (its weight encodings and binding checked together), or the
/// target's own draft head. A plain package drafts with nothing; a method
/// that needs a drafter the package does not have is rejected.
fn admit_drafter(
    package: &ModelPackagePaths,
    source: &dyn HeaderSource,
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
            ModelMethod::DFlash
            | ModelMethod::DSpark
            | ModelMethod::DFlash2
            | ModelMethod::LiLiCorr => {
                rejected("a separate-draft method was requested but the package has no draft")
            }
        };
    };
    if package.method == ModelMethod::Plain {
        return Ok(DrafterAdmission::None);
    }
    match headers.add_draft_from(path, source) {
        Ok(()) => {}
        Err(error) => match artifact_refusal(&error) {
            Some(unsupported) => return Ok(DrafterAdmission::Rejected(unsupported)),
            None => return Err(ModelAssessmentError::Artifact(error)),
        },
    }
    let draft = headers.draft().expect("the draft component was just added");
    let mut refusals = Vec::new();
    refusals.extend(unsupported_weights(&draft.tensors, Residency::Packed, layout));
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

/// Every tensor of a component (`tensors`) whose stored encoding has no
/// import here, counted per GGML type: the same classification weight
/// planning applies ([`stored_weights`], each weight with its scales and
/// a grid companion with its weight, then [`scaled_source_element`] and,
/// for a packed one, [`held_storage`] in `layout`), so a
/// type added to them is admitted here with no other change.
fn unsupported_weights(
    tensors: &[TensorDescriptor],
    residency: Residency,
    layout: Layout,
) -> Option<UnsupportedModel> {
    let mut counts = BTreeMap::<&'static str, u32>::new();
    for (tensor, scales) in stored_weights(tensors) {
        let width = tensor.shape.last().copied().unwrap_or(1);
        let imported = match residency {
            // Every encoding with a resident element has a W4A16 one.
            Residency::Packed => held_storage(
                tensor.encoding,
                scales,
                width,
                ScaledActivations::Rounded,
                DType::F32,
                layout,
            )
            .is_some(),
            Residency::Dequantized => {
                scaled_source_element(tensor.encoding, scales, width).is_some()
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An F8_E4M3 weight with its I8 `.scale` grid is admitted as planning
    /// holds it, the grid counted with its weight, on every backend's
    /// layout.
    #[test]
    fn a_grid_scaled_f8_weight_is_admitted_where_planning_holds_it() {
        use magnitude_artifacts::gguf::Encoding;
        let tensor = |name: &str, shape: Vec<u64>, encoding| TensorDescriptor {
            name: name.into(),
            shape,
            encoding,
            offset: 0,
            nbytes: 0,
        };
        let tensors = [
            tensor("blk.0.attn_q_b.weight", vec![64, 256], Encoding::F8E4M3),
            tensor("blk.0.attn_q_b.scale", vec![2, 8], Encoding::I8),
        ];
        for backend in seismic::BackendName::ALL {
            let layout = backend.resident_layout();
            let refused = unsupported_weights(&tensors, Residency::Packed, layout);
            assert!(refused.is_none(), "{backend:?} {refused:?}");
        }
    }
}
