//! Metadata-only model assessment from a package path.
//!
//! Opens only the GGUF headers of the target, its optional projector and
//! drafter. Independent header checks (weight encodings, the family's
//! interpretation, the tokenizer and the chat templates) each run and their
//! refusals are reported together. An admitted target is planned through
//! [`crate::planning::plan_resident`] (the production planning inputs)
//! with its drafter, each way of holding its experts outlined once at its
//! floor (`ExecutionPlanDraft::floor_forms`), and the target's plan alone
//! is charged beside it from its headers; every served tier's fit, bank and
//! verdict is arithmetic over those outlines. A drafter that does not
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
use crate::planning::{plan_backend, plan_resident, ExecutionPlanningError};
use magnitude_artifacts::{
    gguf::{Directory, FileHeaders, HeaderSource, TensorDescriptor},
    ArtifactIdentity, PackageHeaders, PackageIdentity, PackageManifest, TemplatePayload,
    TokenizerPayload,
};
use magnitude_chat::{
    artifacts::{gguf_templates, gguf_tokenizer_vocabulary},
    TemplateBundle, TemplateInspection, TokenizerError,
};
use magnitude_executor::{
    assessment::{
        assess_execution_fit_with_priced_bank, estimate_execution, lookahead, performance_depths,
        pool, streamed_layers, stage_timer, AssessmentError, DecodeDemand, PricedBankContext,
        SizingScope, StructuralCharge, AssessmentRequest,
        ExecutionAssessment, ExecutionFit, ExecutionShortfall,
        ExecutorCosts, ExpertShares, PerformanceEstimate, PreparedExecutionAssessment,
        PointRequests, Priced, StoredCosts, StreamedFigures,
    },
    platform::{self, DeviceRequest, GraphPricing, ReleaseRate, PlatformError, SelectedDevice},
    held_storage, resident_layout, scaled_source_element, stored_weights,
    ExpertBankPlan, ExecutionPath, ExecutionPlanDraft, FloorForms, HeldForm, KernelCache, UndraftedBounds,
};
use magnitude_family_contracts::{
    DraftVariant, ModelDefinition, ModelFamily, Operator, ScaledActivations,
};
use magnitude_scheduler::ServiceLimits;
use seismic::{DType, DeviceTopology, Devices, Layout, Status};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The package components to assess and the generation method the bundle
/// declares (`Auto` for a standalone package). Only their headers are read.
/// `target_shards` holds the target GGUF file(s) in shard order: one file for
/// an unsplit GGUF or the first shard of a standard-named split GGUF, or every
/// shard explicitly when shards have non-standard names or live across
/// directories; `draft` is a separate drafter (DFlash, DSpark, DFlash2,
/// LiLiCorr, a Gemma assistant or a prediction head). A drafter the method
/// cannot draft with is `Rejected` and the target is assessed without it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPackagePaths {
    pub target_shards: Vec<PathBuf>,
    pub projector: Option<PathBuf>,
    pub draft: Option<PathBuf>,
    pub method: ModelMethod,
}

impl ModelPackagePaths {
    pub fn single(
        target: PathBuf,
        projector: Option<PathBuf>,
        draft: Option<PathBuf>,
        method: ModelMethod,
    ) -> Self {
        Self {
            target_shards: vec![target],
            projector,
            draft,
            method,
        }
    }

    pub fn target(&self) -> &Path {
        self.target_shards
            .first()
            .map(PathBuf::as_path)
            .unwrap_or_else(|| Path::new(""))
    }
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
    pub release: ReleaseRate,
    pub policy: ModelPolicy,
    pub service: ServiceLimits,
    /// The one context every model is planned at, as a load serving it at
    /// that context holds it (`host::resolve_served_context`): when none,
    /// each is planned at its served tiers ([`served_contexts`]) from its
    /// declared limit down.
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
        release: ReleaseRate,
        policy: ModelPolicy,
        service: ServiceLimits,
    ) -> Result<Self, ModelAssessmentError> {
        let selected = platform::select_recorded(catalog, ExecutionPath::Native, device, &release, kernels)
            .map_err(ModelAssessmentError::Platform)?;
        let topology = catalog.topology();
        Ok(Self {
            status: catalog.status(&topology, selected.info.id()),
            topology,
            device,
            selected,
            release,
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
                release: self.release,
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
    let costs_outstanding = StoredCosts::outstanding_of(&walked);
    let outstanding = match PointRequests::new(kernels).outstanding(topology, info, storage, &records, costs_outstanding) {
        Ok(outstanding) => outstanding,
        Err(error) => {
            return Measurements::Unavailable {
                reason: format!("the point requests: {error}"),
            };
        }
    };
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
            let pending = pending || !outstanding.is_empty();
            Measurements::Measured {
                costs: Arc::new(costs),
                records: Arc::new(cost_records),
                pending,
                streamed,
            }
        }
        Ok(StoredCosts::Missing(_)) => Measurements::Pending(
            magnitude_executor::assessment::PendingPrices::new(outstanding)
                .expect("missing executor costs must have outstanding requests"),
        ),
        Err(error) => Measurements::Unavailable {
            reason: error.to_string(),
        },
    }
}

/// What this machine's measurements give assessment. Only the measurement
/// job takes them (device topology spec §2.15); assessment reads what it
/// stored.
#[derive(Clone)]
pub enum Measurements {
    /// The job has not stored the probe or a figure no other prices yet:
    /// fit is assessed and speed is pending, carrying the non-empty
    /// outstanding requests.
    Pending(magnitude_executor::assessment::PendingPrices),
    /// The stored records give no costs, and why: the probe is recorded
    /// unobtainable (the executor cannot stream its own weights), the
    /// records cannot be read, or the measurement job could not open the
    /// device in two jobs in a row this service's run (it keeps retrying),
    /// and every fitting model's speed is unavailable.
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

/// Host-side model facts established from headers, with the drafter's
/// verdict as `D`: a prepared model's drafter ([`PreparedDrafter`]) until
/// its assessment is finished.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFacts<D = DrafterVerdict> {
    pub capabilities: ModelCapabilities,
    /// [`magnitude_chat::TemplateInspection::fingerprint`] of the package's
    /// chat templates and reasoning profile.
    pub template_fingerprint: String,
    /// The model's supported context, or the one the structure fixes
    /// ([`AssessmentStructure::served_context`]): the largest it is served
    /// at ([`served_contexts`]).
    pub context_limit: u32,
    pub fingerprint: Option<ModelFingerprint>,
    pub projector: ProjectorVerdict,
    pub drafter: D,
}

impl<D> ModelFacts<D> {
    /// These facts with `drafter`'s verdict.
    fn with_drafter<E>(&self, drafter: E) -> ModelFacts<E> {
        ModelFacts {
            capabilities: self.capabilities.clone(),
            template_fingerprint: self.template_fingerprint.clone(),
            context_limit: self.context_limit,
            fingerprint: self.fingerprint.clone(),
            projector: self.projector.clone(),
            drafter,
        }
    }
}

/// A prepared model's drafter: what [`DrafterVerdict`] its finished
/// assessment reports, a bound drafter's memory charged at the served tier
/// on the machine's graph pricing ([`finish_model_assessment`]).
pub enum PreparedDrafter {
    None,
    /// It binds and plans with the target: the target's plan alone at the
    /// drafted plan's holding of experts, which its memory is the drafted
    /// plan's charge less of.
    Bound {
        method: SpeculativeMethod,
        alone: Box<TargetAlone>,
    },
    Rejected(UnsupportedModel),
}

/// The target's plan alone at a drafted plan's holding of experts
/// (`ExecutionPlanDraft::holding_as`) and its charge's structure, whose
/// target and vision families are the drafted plan's without its drafter's
/// writes (`StructuralCharge::undrafted`). Boxed where it is made
/// ([`target_alone`]): it carries a whole definition, plan draft and charge
/// structure, and a preparation is moved by value through deep frames.
pub struct TargetAlone {
    draft: ExecutionPlanDraft,
    structure: StructuralCharge,
}

impl PreparedDrafter {
    /// The drafter's verdict at `context` tokens of the served context, its
    /// graphs priced at `pricing`: a bound drafter's memory is the drafted
    /// plan's serving and host charge (`execution`'s primary form of its
    /// definition) less the target's alone at the same holding of experts.
    fn verdict(
        &self,
        execution: &PreparedExecutionAssessment,
        context: u32,
        pricing: GraphPricing,
    ) -> Result<DrafterVerdict, ModelAssessmentError> {
        let (method, alone) = match self {
            Self::None => return Ok(DrafterVerdict::None),
            Self::Rejected(unsupported) => return Ok(DrafterVerdict::Rejected(unsupported.clone())),
            Self::Bound { method, alone } => (*method, alone),
        };
        let context = u64::from(context);
        let held = |draft: &ExecutionPlanDraft, structure: &StructuralCharge| {
            let charge = structure
                .charge_at_context_with_pricing(context, draft, pricing)
                .map_err(ModelAssessmentError::Assessment)?;
            charge
                .host_bytes()
                .and_then(|host| charge.serving_bytes.checked_add(host).ok_or_else(|| "held charge overflows".to_owned()))
                .map_err(ModelAssessmentError::Planning)
        };
        let primary = execution.primary();
        let with = held(&primary.draft, &primary.structure)?;
        let without = held(&alone.draft, &alone.structure)?;
        assert!(
            with >= without,
            "a drafted plan holds {with} bytes, less than its target's plan alone ({without}) at the same holding of experts"
        );
        Ok(DrafterVerdict::Bound { method, memory_bytes: with - without })
    }
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
    /// It fits at the served context `fit.fit_context_tokens`, the largest
    /// of its served tiers ([`served_contexts`]) that fits, with what it
    /// requires there and its plan's expert bank, whole or streamed.
    Fits {
        fit: ExecutionFit,
        residency: ModelResidency,
    },
    /// It fits at none of its served tiers: the shortfall at the smallest.
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
    /// The package planned at its largest served tier, `facts.context_limit`
    /// (`execution` with the definition it plans, its drafter in `facts`):
    /// every served tier's fit and its drafter's memory are arithmetic over
    /// it ([`finish_model_assessment`]).
    ///
    /// Its facts and execution are boxed: each carries whole definitions and
    /// plan drafts, and a preparation is returned by value through deep
    /// frames, whose stacks hold every by-value copy
    /// (`seismic_threads::STACK_BYTES`).
    Planned {
        facts: Box<ModelFacts<PreparedDrafter>>,
        execution: Box<PreparedExecutionAssessment>,
        /// The package's served tiers, largest first: every one of
        /// [`served_contexts`], or the one
        /// [`AssessmentStructure::served_context`] fixes.
        tiers: Vec<u32>,
    },
}

/// The smallest context a model is served at when its trained limit allows
/// more: a coding harness's prompt and tools alone fill much of 16K.
pub const SERVED_CONTEXT_FLOOR: u32 = 32_768;

/// The contexts a model trained for `context_limit` tokens may be served at,
/// largest first: the limit, then every power of two below it down to
/// [`SERVED_CONTEXT_FLOOR`]. A model trained for less than the floor is
/// served only at its limit. Tiers rather than the exact largest context
/// keep the served context stable when a build's footprint shifts slightly.
pub fn served_contexts(context_limit: u32) -> Vec<u32> {
    let mut contexts = vec![context_limit];
    let mut tier = SERVED_CONTEXT_FLOOR;
    let mut below = Vec::new();
    while tier < context_limit {
        below.push(tier);
        match tier.checked_mul(2) {
            Some(next) => tier = next,
            None => break,
        }
    }
    contexts.extend(below.into_iter().rev());
    contexts
}

/// Complete a prepared model against stable capacity at the largest of its
/// served tiers it fits ([`served_contexts`], [`served_tier`]), and, when it
/// fits, estimate its decode speed at the requested depths within that
/// context with `sequences` conversations decoding at once from the setup's
/// measurements: arithmetic over the prepared structure and demand. A
/// smaller tier's charge is the prepared structure with its state planned
/// at that context; the graphs outlined at the largest tier bound every
/// smaller one's. A streamed model's tier does not trade its bank for
/// context: the bank is sized against the room the tier's state leaves.
pub fn finish_model_assessment(
    prepared: &PreparedModelAssessment,
    setup: &AssessmentSetup,
    performance_depths: &[u32],
    sequences: u32,
) -> Result<ModelAssessment, ModelAssessmentError> {
    let (facts, execution, tiers) = match prepared {
        PreparedModelAssessment::Unsupported {
            unsupported,
            fingerprint,
        } => {
            return Ok(ModelAssessment::Unsupported {
                unsupported: unsupported.clone(),
                fingerprint: fingerprint.clone(),
            });
        }
        PreparedModelAssessment::Planned { facts, execution, tiers } => (&**facts, &**execution, tiers),
    };
    let _scope = SizingScope::enter();
    let definition = execution.definition();
    let structure = &setup.structure;
    let pricing = structure.selected.graph_cost;
    let priced_bank = match &setup.measurements {
        Measurements::Measured {
            costs,
            streamed: Priced::Measured(figures),
            ..
        } => PricedBankContext::new(execution, costs, figures)
            .map_err(ModelAssessmentError::Assessment)?,
        _ => None,
    };
    let served = served_tier(tiers.len(), |index| {
        let tier = tiers[index];
        let fit = assess_execution_fit_with_priced_bank(
            u64::from(tier),
            execution,
            pricing,
            &structure.topology,
            &structure.status,
            priced_bank.as_ref(),
        )
        .map_err(ModelAssessmentError::Assessment)?;
        Ok::<_, ModelAssessmentError>(match fit {
            ExecutionAssessment::Fits(fit) => Ok((tier, fit)),
            ExecutionAssessment::DoesNotFit(shortfall) => Err((tier, shortfall)),
        })
    })?;
    let (tier, fit) = match served {
        Ok((tier, fit)) => (tier, fit),
        Err((tier, shortfall)) => {
            let facts = facts.with_drafter(facts.drafter.verdict(execution, tier, pricing)?);
            return Ok(ModelAssessment::Assessed {
                facts,
                execution: ModelExecution::DoesNotFit(shortfall),
            });
        }
    };
    let request = AssessmentRequest {
        context_limit: fit.fit_context_tokens,
        performance_depths: performance_depths.to_vec(),
        release: structure.release,
        sequences,
    };
    let compute_residency = || -> Result<ModelResidency, AssessmentError> {
        Ok(match fit.streamed_bank() {
            Some(bank) => {
                let shares = ExpertShares::of(bank);
                let (performance, floor_performance) = streamed_performance_pair(
                    &setup.measurements,
                    execution,
                    bank,
                    &request,
                    shares.held(shares.slots()),
                    shares.held(shares.floor()),
                    priced_bank.as_ref(),
                )?;
                ModelResidency::Streamed {
                    performance,
                    floor_performance,
                    shares,
                }
            }
            None => ModelResidency::Resident {
                performance: resident_performance(&setup.measurements, execution, &request)?,
            },
        })
    };
    let (drafter, residency) = if matches!(facts.drafter, PreparedDrafter::Bound { .. })
        && u64::from(tier) != definition.decoder.context_limit
    {
        let (residency, drafter) = pool::join(compute_residency, || {
            facts.drafter.verdict(execution, tier, pricing)
        });
        (drafter?, residency.map_err(ModelAssessmentError::Assessment)?)
    } else {
        let drafter = facts.drafter.verdict(execution, tier, pricing)?;
        let residency = compute_residency().map_err(ModelAssessmentError::Assessment)?;
        (drafter, residency)
    };
    let facts = facts.with_drafter(drafter);
    Ok(ModelAssessment::Assessed {
        facts,
        execution: ModelExecution::Fits { fit, residency },
    })
}

/// The largest of `count` served tiers, largest first, at which `assess`
/// finds a fit, or the shortfall at the smallest: the largest is assessed
/// first, then the smallest, then the tiers between by bisection, a smaller
/// context holding less.
fn served_tier<F, S, E>(
    count: usize,
    mut assess: impl FnMut(usize) -> Result<Result<F, S>, E>,
) -> Result<Result<F, S>, E> {
    assert!(count > 0, "a package is served at one tier at least");
    let shortfall = match assess(0)? {
        Ok(fit) => return Ok(Ok(fit)),
        Err(shortfall) => shortfall,
    };
    if count == 1 {
        return Ok(Err(shortfall));
    }
    let mut fit = match assess(count - 1)? {
        Ok(fit) => fit,
        Err(shortfall) => return Ok(Err(shortfall)),
    };
    let (mut short, mut fits) = (0, count - 1);
    while fits - short > 1 {
        let middle = short + (fits - short) / 2;
        match assess(middle)? {
            Ok(found) => (fit, fits) = (found, middle),
            Err(_) => short = middle,
        }
    }
    Ok(Ok(fit))
}

/// A resident fit's decode speed from the setup's measurements.
fn resident_performance(
    measurements: &Measurements,
    preparation: &PreparedExecutionAssessment,
    request: &AssessmentRequest,
) -> Result<Performance, AssessmentError> {
    Ok(match measurements {
        Measurements::Pending(_) => Performance::Pending,
        Measurements::Unavailable { reason } => Performance::Unavailable {
            reason: reason.clone(),
        },
        Measurements::Measured { costs, .. } => {
            Performance::Estimated(estimate_execution(preparation, costs, request)?)
        }
    })
}

/// Evaluate streamed decode performance for `N` slot holdings (`holdings`),
/// sharing `streamed_layers`, `costs.figures`, and the `held`-independent
/// compute step time across all holdings, and evaluating independent depths
/// concurrently on the assessment pool.
fn streamed_estimates<const N: usize>(
    costs: &ExecutorCosts,
    figures: &StreamedFigures,
    preparation: &PreparedExecutionAssessment,
    bank: &ExpertBankPlan,
    request: &AssessmentRequest,
    holdings: [magnitude_cost_model::HeldSlots; N],
    priced_bank: Option<&PricedBankContext>,
) -> Result<[Performance; N], AssessmentError> {
    let target = preparation.target_census();
    let owned;
    let (layers, priced) = match priced_bank {
        Some(ctx) => (&ctx.layers, &ctx.figures),
        None => {
            let (layers, priced) = pool::join(
                || streamed_layers(bank, costs, figures),
                || costs.figures(&target.censuses()),
            );
            let layers = match layers {
                Priced::Measured(layers) => layers,
                Priced::Pending(_) => return Ok(std::array::from_fn(|_| Performance::Pending)),
                Priced::Unavailable(reason) => {
                    let msg = format!("the streamed layers: {reason}");
                    return Ok(std::array::from_fn(|_| Performance::Unavailable {
                        reason: msg.clone(),
                    }));
                }
            };
            owned = (layers, priced);
            (&owned.0, &owned.1)
        }
    };
    let pricing = seismic::measure::pricing::Pricing {
        figures: priced,
        lookahead: lookahead(preparation.draft()),
    };
    let depths = performance_depths(request.context_limit, &request.performance_depths);
    let seq_u64 = u64::from(request.sequences);
    let seq_f64 = f64::from(request.sequences);
    let read = figures.read;
    let mut periods = Vec::with_capacity(N);
    let mut holding_index = [0usize; N];
    for i in 0..N {
        holding_index[i] = if let Some(prev) = holdings[..i].iter().position(|&h| h == holdings[i]) {
            holding_index[prev]
        } else {
            let idx = periods.len();
            periods.push(layers.prepare(None, holdings[i], seq_f64));
            idx
        };
    }
    let rows = pool::par_for_results(&depths, |&depth| {
        let compute = target.each(|census| {
            magnitude_cost_model::step_seconds(census, &pricing, seq_u64, depth)
        })?;
        let mut tps = [0.0f64; N];
        for (slot, period) in tps.iter_mut().zip(&periods) {
            *slot = 1. / period.step_seconds(read, compute).total_seconds;
        }
        Ok::<_, seismic::measure::pricing::PriceError>(std::array::from_fn::<_, N, _>(|i| {
            PerformanceEstimate {
                context_tokens: depth,
                tokens_per_second: tps[holding_index[i]],
            }
        }))
    })?;
    let mut estimates: [Vec<PerformanceEstimate>; N] =
        std::array::from_fn(|_| Vec::with_capacity(rows.len()));
    for row in rows {
        for (dst, est) in estimates.iter_mut().zip(row) {
            dst.push(est);
        }
    }
    Ok(estimates.map(Performance::Estimated))
}

/// A streamed fit's decode speeds at two slot holdings (`held_a`, `held_b`),
/// sharing `streamed_layers`, `costs.figures`, and the `held`-independent
/// compute step time across both evaluations, and evaluating independent
/// depths and slot holdings concurrently on the assessment pool.
fn streamed_performance_pair(
    measurements: &Measurements,
    preparation: &PreparedExecutionAssessment,
    bank: &ExpertBankPlan,
    request: &AssessmentRequest,
    held_a: magnitude_cost_model::HeldSlots,
    held_b: magnitude_cost_model::HeldSlots,
    priced_bank: Option<&PricedBankContext>,
) -> Result<(Performance, Performance), AssessmentError> {
    match measurements {
        Measurements::Pending(_) => Ok((Performance::Pending, Performance::Pending)),
        Measurements::Unavailable { reason } => Ok((
            Performance::Unavailable { reason: reason.clone() },
            Performance::Unavailable { reason: reason.clone() },
        )),
        Measurements::Measured { costs, streamed, .. } => match streamed {
            Priced::Pending(_) => Ok((Performance::Pending, Performance::Pending)),
            Priced::Unavailable(reason) => Ok((
                Performance::Unavailable { reason: reason.clone() },
                Performance::Unavailable { reason: reason.clone() },
            )),
            Priced::Measured(figures) => {
                let [a, b] = streamed_estimates(
                    costs,
                    figures,
                    preparation,
                    bank,
                    request,
                    [held_a, held_b],
                    priced_bank,
                )?;
                Ok((a, b))
            }
        },
    }
}

/// A streamed decode's speed holding `held` of each routed layer's experts
/// (its bank's slots and its host tier's), at each depth `request` names:
/// the plan's streamed layers priced by the generic miss model and the one
/// step model ([`StreamedLayers::step_seconds`]) at the request's rows (a
/// row a sequence), their file reads along the model store's read line, on
/// `C` the step's estimate there in each form (`preparation` is the plan's,
/// `draft` the plan, `magnitude_cost_model::step_seconds` of each of its
/// censuses, `PreparedExecutionAssessment::censuses`), `bank`
/// the fit's: the plan's structure and the machine's stored figures alone.
pub fn streamed_estimate(
    costs: &ExecutorCosts,
    figures: &StreamedFigures,
    preparation: &PreparedExecutionAssessment,
    bank: &ExpertBankPlan,
    request: &AssessmentRequest,
    held: magnitude_cost_model::HeldSlots,
) -> Result<Performance, ModelAssessmentError> {
    let [performance] = streamed_estimates(costs, figures, preparation, bank, request, [held], None)
        .map_err(ModelAssessmentError::Assessment)?;
    Ok(performance)
}

/// The executor's costs as a walk of the stored records found them
/// (`ExecutorCosts::stored`), which a streamed load's measures price on:
/// pending until the job stores the probe, unavailable where they cannot
/// be derived.
pub fn priced_costs(
    walked: Result<StoredCosts, magnitude_executor::assessment::ExecutorCostsError>,
) -> magnitude_executor::assessment::Priced<ExecutorCosts> {
    StoredCosts::priced(walked)
}

/// Assess one package on the setup's selected device.
pub fn assess_model(
    package: &ModelPackagePaths,
    setup: &AssessmentSetup,
    performance_depths: &[u32],
    sequences: u32,
) -> Result<ModelAssessment, ModelAssessmentError> {
    let _scope = SizingScope::enter();
    let prepared = prepare_model_assessment(package, &setup.structure)?;
    finish_model_assessment(&prepared, setup, performance_depths, sequences)
}

/// Prepare one package from its headers on the selected execution
/// configuration: admission (every header refusal at once), then the plan
/// assessed, with its decode demand and each way of holding its experts
/// outlined at its floor: the drafted plan when a drafter is admitted, the
/// target's plan alone otherwise, beside which a bound drafter keeps the
/// target's plan alone at the drafted plan's holding of experts
/// ([`PreparedDrafter`]). Reads only the structure, and nothing of it the
/// measurement records price (its graphs' cost, which finishing applies),
/// so a preparation holds under every measurement of it.
pub fn prepare_model_assessment(
    package: &ModelPackagePaths,
    structure: &AssessmentStructure,
) -> Result<PreparedModelAssessment, ModelAssessmentError> {
    prepare_model_assessment_with(package, HeaderInput::Files, structure)
}

/// [`prepare_model_assessment`] with the package's headers from `headers`,
/// so a host that already read them never has them read again.
pub fn prepare_model_assessment_from(
    package: &ModelPackagePaths,
    headers: &dyn HeaderSource,
    structure: &AssessmentStructure,
) -> Result<PreparedModelAssessment, ModelAssessmentError> {
    prepare_model_assessment_with(package, HeaderInput::Source(headers), structure)
}

fn prepare_model_assessment_with(
    package: &ModelPackagePaths,
    headers: HeaderInput<'_>,
    structure: &AssessmentStructure,
) -> Result<PreparedModelAssessment, ModelAssessmentError> {
    let layout = resident_layout(ExecutionPath::Native, structure.selected.info.backend());
    let source = headers.as_source();
    let draft_prefetch_path = match headers {
        HeaderInput::Files if package.method != ModelMethod::Plain => package.draft.as_deref(),
        _ => None,
    };
    let (vision, projector, context_limit, fingerprint, chat, drafter, execution) =
        if let Some(draft_path) = draft_prefetch_path {
            // Overlap draft header inspection (`PackageHeaders::inspect_component`)
            // with target header admission, chat inspection, fingerprinting, and
            // (for dense models) plain target planning so target work starts
            // immediately instead of idling on draft I/O and populates the
            // target-family layout cache reused by drafted certification (and
            // `TargetAlone` / plain fallback). Routed MoE models skip plain
            // pre-planning because `can_reuse_plain_as_target_alone` requires
            // `expert_bank().is_none()`, so pre-planning would duplicate banked
            // certification before `admit_drafter` can start.
            let (prefetched, target_branch) = pool::join(
                move || prefetch_draft(draft_path, layout),
                || -> Result<PrefetchedTargetBranch, ModelAssessmentError> {
                    let target = match admit_target_headers_timed(package, source, layout)? {
                        TargetHeadersAdmission::Unsupported {
                            unsupported,
                            fingerprint,
                        } => {
                            return Ok(PrefetchedTargetBranch::Unsupported {
                                unsupported,
                                fingerprint,
                            });
                        }
                        TargetHeadersAdmission::Admitted(target) => *target,
                    };
                    let AdmittedTarget {
                        headers: pkg_headers,
                        family,
                        definition,
                        bundle,
                        projector,
                    } = target;
                    let recording = stage_timer::is_recording();
                    if has_routed_experts(&definition) {
                        // Defer chat inspection and fingerprinting until after
                        // `admit_drafter` so they overlap with `plan_package`
                        // instead of serializing cold template probing ahead of
                        // banked MoE planning.
                        return Ok(PrefetchedTargetBranch::Admitted(Box::new(PreplannedTarget {
                            headers: pkg_headers,
                            family,
                            definition,
                            projector,
                            chat: DeferredChat::Pending(bundle),
                            preplanned: None,
                        })));
                    }
                    let target_dir = pkg_headers.target();
                    let (((chat_res, chat_ns), (fingerprint, fp_ns)), (definition, preplanned)) =
                        pool::join(
                            move || inspect_chat_and_fingerprint(bundle, target_dir, recording),
                            || preplan_plain(&pkg_headers, definition, structure),
                        );
                    if recording {
                        stage_timer::record_stage(|t| {
                            t.chat_inspect_ns += chat_ns;
                            t.admit_fingerprint_ns += fp_ns;
                        });
                    }
                    Ok(PrefetchedTargetBranch::Admitted(Box::new(PreplannedTarget {
                        headers: pkg_headers,
                        family,
                        definition,
                        projector,
                        chat: DeferredChat::Done { chat_res, fingerprint },
                        preplanned,
                    })))
                },
            );
            let PreplannedTarget {
                headers: mut pkg_headers,
                family,
                mut definition,
                projector,
                chat,
                preplanned,
            } = match target_branch? {
                PrefetchedTargetBranch::Unsupported {
                    unsupported,
                    fingerprint,
                } => {
                    return Ok(PreparedModelAssessment::Unsupported {
                        unsupported,
                        fingerprint,
                    });
                }
                PrefetchedTargetBranch::Admitted(target) => *target,
            };
            let vision = definition.vision.is_some();
            let _ = u32::try_from(definition.decoder.context_limit)
                .map_err(|_| ModelAssessmentError::ContextLimit(definition.decoder.context_limit))?;
            let mut drafter = stage_timer::time_stage(
                |t, ns| t.admit_drafter_ns += ns,
                || {
                    admit_drafter(
                        package,
                        source,
                        &mut pkg_headers,
                        Some(prefetched),
                        family,
                        &definition,
                        layout,
                    )
                },
            )?;
            let context_limit =
                apply_served_context(&mut definition, &mut drafter, structure.served_context)?;
            let (chat_res, fingerprint, planned_res) = match chat {
                DeferredChat::Done { chat_res, fingerprint } => {
                    let chat = match chat_res {
                        Ok(chat) => chat,
                        Err(error) => {
                            return Ok(PreparedModelAssessment::Unsupported {
                                unsupported: UnsupportedModel::chat_template(error),
                                fingerprint,
                            });
                        }
                    };
                    let planned_res =
                        plan_package(&pkg_headers, definition, drafter, structure, preplanned);
                    (Ok(chat), fingerprint, planned_res)
                }
                DeferredChat::Pending(bundle) => {
                    let recording = stage_timer::is_recording();
                    let target_dir = pkg_headers.target();
                    let (((chat_res, chat_ns), (fingerprint, fp_ns)), planned_res) = pool::join(
                        move || inspect_chat_and_fingerprint(bundle, target_dir, recording),
                        || plan_package(&pkg_headers, definition, drafter, structure, preplanned),
                    );
                    if recording {
                        stage_timer::record_stage(|t| {
                            t.chat_inspect_ns += chat_ns;
                            t.admit_fingerprint_ns += fp_ns;
                        });
                    }
                    (chat_res, fingerprint, planned_res)
                }
            };
            let chat = match chat_res {
                Ok(chat) => chat,
                Err(error) => {
                    return Ok(PreparedModelAssessment::Unsupported {
                        unsupported: UnsupportedModel::chat_template(error),
                        fingerprint,
                    });
                }
            };
            let (drafter, execution) = match planned_res? {
                Ok(planned) => planned,
                Err(unsupported) => {
                    return Ok(PreparedModelAssessment::Unsupported {
                        unsupported,
                        fingerprint,
                    });
                }
            };
            (vision, projector, context_limit, fingerprint, chat, drafter, execution)
        } else {
            let target = match admit_target_headers_timed(package, source, layout)? {
                TargetHeadersAdmission::Unsupported {
                    unsupported,
                    fingerprint,
                } => {
                    return Ok(PreparedModelAssessment::Unsupported {
                        unsupported,
                        fingerprint,
                    });
                }
                TargetHeadersAdmission::Admitted(target) => target,
            };
            let vision = target.definition.vision.is_some();
            let projector = target.projector;
            let bundle = target.bundle;
            let mut pkg_headers = target.headers;
            let family = target.family;
            let mut definition = target.definition;
            let _ = u32::try_from(definition.decoder.context_limit)
                .map_err(|_| ModelAssessmentError::ContextLimit(definition.decoder.context_limit))?;
            let mut drafter = stage_timer::time_stage(
                |t, ns| t.admit_drafter_ns += ns,
                || {
                    admit_drafter(
                        package,
                        source,
                        &mut pkg_headers,
                        None,
                        family,
                        &definition,
                        layout,
                    )
                },
            )?;
            let context_limit =
                apply_served_context(&mut definition, &mut drafter, structure.served_context)?;
            let recording = stage_timer::is_recording();
            let target_dir = pkg_headers.target();
            // `bundle.inspect_cached()` and `ModelFingerprint::from_directory` are
            // independent of each other and of execution planning; overlap both with
            // each other and with `plan_package`.
            let (((chat_res, chat_ns), (fingerprint, fp_ns)), planned_res) = pool::join(
                move || inspect_chat_and_fingerprint(bundle, target_dir, recording),
                || plan_package(&pkg_headers, definition, drafter, structure, None),
            );
            if recording {
                stage_timer::record_stage(|t| {
                    t.chat_inspect_ns += chat_ns;
                    t.admit_fingerprint_ns += fp_ns;
                });
            }
            let chat = match chat_res {
                Ok(chat) => chat,
                Err(error) => {
                    return Ok(PreparedModelAssessment::Unsupported {
                        unsupported: UnsupportedModel::chat_template(error),
                        fingerprint,
                    });
                }
            };
            let (drafter, execution) = match planned_res? {
                Ok(planned) => planned,
                Err(unsupported) => {
                    return Ok(PreparedModelAssessment::Unsupported {
                        unsupported,
                        fingerprint,
                    });
                }
            };
            (vision, projector, context_limit, fingerprint, chat, drafter, execution)
        };
    Ok(PreparedModelAssessment::Planned {
        facts: Box::new(ModelFacts {
            capabilities: ModelCapabilities {
                vision,
                tools: chat.tools,
                structured_output: chat.structured_output,
                reasoning: ReasoningCapabilities {
                    efforts: chat
                        .reasoning
                        .mappings
                        .into_iter()
                        .map(|mapping| mapping.effort)
                        .collect(),
                    default_effort: chat.reasoning.default_effort,
                },
            },
            template_fingerprint: chat.fingerprint,
            context_limit,
            fingerprint,
            projector,
            drafter,
        }),
        execution,
        tiers: match structure.served_context {
            Some(_) => vec![context_limit],
            None => served_contexts(context_limit),
        },
    })
}

enum PrefetchedTargetBranch {
    Unsupported {
        unsupported: UnsupportedModel,
        fingerprint: Option<ModelFingerprint>,
    },
    Admitted(Box<PreplannedTarget>),
}

enum DeferredChat {
    Pending(TemplateBundle),
    Done {
        chat_res: Result<TemplateInspection, String>,
        fingerprint: Option<ModelFingerprint>,
    },
}

struct PreplannedTarget {
    headers: PackageHeaders,
    family: &'static dyn ModelFamily,
    definition: ModelDefinition,
    projector: ProjectorVerdict,
    chat: DeferredChat,
    preplanned: Option<Result<FloorForms<()>, Refusal>>,
}

fn inspect_chat_and_fingerprint(
    bundle: TemplateBundle,
    target_dir: &Directory,
    recording: bool,
) -> (
    (Result<TemplateInspection, String>, u64),
    (Option<ModelFingerprint>, u64),
) {
    pool::join(
        move || {
            let chat_start = recording.then(std::time::Instant::now);
            let chat_res = bundle.inspect_cached();
            let chat_ns = chat_start.map_or(0, |s| {
                u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
            });
            (chat_res, chat_ns)
        },
        move || {
            let fp_start = recording.then(std::time::Instant::now);
            let fp = ModelFingerprint::from_directory(target_dir).ok();
            let fp_ns = fp_start.map_or(0, |s| {
                u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
            });
            (fp, fp_ns)
        },
    )
}

/// Whether `definition`'s decoder contains any routed expert sublayer (directly
/// or inside a parallel branch).
fn has_routed_experts(definition: &ModelDefinition) -> bool {
    fn op_is_routed(op: &Operator) -> bool {
        match op {
            Operator::RoutedFfn(_) => true,
            Operator::Parallel(branches) => branches.iter().any(|branch| op_is_routed(&branch.op)),
            _ => false,
        }
    }
    definition
        .decoder
        .sublayers()
        .any(|(_, layer)| op_is_routed(&layer.op))
}

/// Pre-plan the plain target while a separate draft component is being
/// prefetched, returning `definition` with its declared `context_limit`
/// restored alongside the plain target's [`FloorForms`].
fn preplan_plain(
    headers: &PackageHeaders,
    mut definition: ModelDefinition,
    structure: &AssessmentStructure,
) -> (ModelDefinition, Option<Result<FloorForms<()>, Refusal>>) {
    let declared_context_limit = definition.decoder.context_limit;
    if u32::try_from(declared_context_limit).is_err() {
        return (definition, None);
    }
    let served_limit = match crate::host::resolve_served_context(
        structure.served_context,
        declared_context_limit,
    ) {
        Ok(limit) if u32::try_from(limit).is_ok() => limit,
        _ => return (definition, None),
    };
    let environment = Environment {
        policy: &structure.policy,
        service: &structure.service,
        device: structure.device,
        release: structure.release,
    };
    definition.decoder.context_limit = served_limit;
    let mut package = headers.manifest();
    package.draft = None;
    let plain = match environment.plain_with_package_or_return(package, definition) {
        Ok(plain) => plain,
        Err((_error, mut definition)) => {
            definition.decoder.context_limit = declared_context_limit;
            return (definition, None);
        }
    };
    let _scope = SizingScope::enter();
    let outline = |definition: &ModelDefinition, draft: &ExecutionPlanDraft| {
        StructuralCharge::of(definition, draft).map(|charge| (charge, ()))
    };
    let forms = prepare_forms(&plain, &structure.selected, outline, |_| ())
        .map(|(forms, ())| forms);
    let mut definition = plain.definition;
    definition.decoder.context_limit = declared_context_limit;
    (definition, Some(forms))
}

/// Apply `served` context to the target definition and any drafter candidate
/// (which serves the same context as the target), returning the resolved limit.
fn apply_served_context(
    definition: &mut ModelDefinition,
    drafter: &mut DrafterAdmission,
    served: Option<usize>,
) -> Result<u32, ModelAssessmentError> {
    let limit = crate::host::resolve_served_context(served, definition.decoder.context_limit)
        .map_err(|error| ModelAssessmentError::Configuration(error.to_string()))?;
    definition.decoder.context_limit = limit;
    if let DrafterAdmission::Candidate { definition, .. } = drafter {
        definition.decoder.context_limit = limit;
    }
    u32::try_from(limit).map_err(|_| ModelAssessmentError::ContextLimit(limit))
}

/// `admitted` planned at its served context, as a load serving it there
/// plans it, each way of holding its experts outlined once
/// ([`prepare_forms`]): the drafted plan when its drafter plans, the
/// target's alone otherwise; or the target's refusal.
fn plan_package(
    headers: &PackageHeaders,
    plain_definition: ModelDefinition,
    drafter: DrafterAdmission,
    structure: &AssessmentStructure,
    mut preplanned_plain: Option<Result<FloorForms<()>, Refusal>>,
) -> Result<Result<(PreparedDrafter, Box<PreparedExecutionAssessment>), UnsupportedModel>, ModelAssessmentError> {
    let _scope = SizingScope::enter();
    let environment = Environment {
        policy: &structure.policy,
        service: &structure.service,
        device: structure.device,
        release: structure.release,
    };
    // The target's plan alone is a package's refusal: a drafter's own is
    // the drafter's.
    let refused = |refusal| match refusal {
        Refusal::Unsupported(unsupported) => Ok(Err(unsupported)),
        Refusal::Failed(error) => Err(error),
    };
    let manifest_package = headers.manifest();
    let plain_forms_ref = preplanned_plain.as_ref().and_then(|res| res.as_ref().ok());
    match drafted_plan(
        &environment,
        manifest_package,
        drafter,
        &plain_definition,
        &structure.selected,
        plain_forms_ref,
    ) {
        Ok(planned) => {
            let DraftedPlan {
                method,
                manifest,
                forms,
                demands,
                decode_demand_ns,
                alone,
            } = *planned;
            if stage_timer::is_recording() {
                stage_timer::record_stage(|t| t.decode_demand_ns += decode_demand_ns);
            }
            let (draft, mut structure) = match alone {
                Some(Ok(alone)) => alone,
                Some(Err(refusal)) => return refused(refusal),
                None => {
                    let HeldForm { draft, structure, .. } = preplanned_plain
                        .take()
                        .expect("reused plain target_alone requires preplanned plain forms")
                        .ok()
                        .expect("reused plain target_alone came from Ok plain forms")
                        .into_primary();
                    (draft, structure)
                }
            };
            structure.release_target_cache();
            let (demand, whole) = match demands {
                Ok(demands) => demands,
                Err(AssessmentError::Graph(error)) => return refused(Refusal::of(classify_graph(error))),
                Err(error) => return Err(ModelAssessmentError::Assessment(error)),
            };
            let alone = Box::new(TargetAlone {
                draft,
                structure,
            });
            let execution = PreparedExecutionAssessment::from_demands(
                manifest.definition,
                forms.structures(),
                demand,
                whole,
            );
            Ok(Ok((PreparedDrafter::Bound { method, alone }, execution)))
        }
        Err((verdict, mut package)) => {
            let recording = stage_timer::is_recording();
            let (plain_definition, forms, demands_res, dd_ns) = match preplanned_plain {
                Some(Ok(forms)) => {
                    let start = recording.then(std::time::Instant::now);
                    let demands_res = PreparedExecutionAssessment::compute_decode_demands(
                        &plain_definition,
                        forms.primary(),
                    );
                    let dd_ns = start.map_or(0, |s| {
                        u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
                    });
                    (plain_definition, forms, demands_res, dd_ns)
                }
                Some(Err(refusal)) => return refused(refusal),
                None => {
                    package.draft = None;
                    let plain = environment
                        .plain_with_package(package, plain_definition)
                        .map_err(ModelAssessmentError::Configuration)?;
                    let outline = |definition: &ModelDefinition, draft: &ExecutionPlanDraft| {
                        StructuralCharge::of(definition, draft).map(|charge| (charge, ()))
                    };
                    let def_ref = &plain.definition;
                    let (forms, (demands_res, dd_ns)) = match prepare_forms(
                        &plain,
                        &structure.selected,
                        outline,
                        move |primary| {
                            let start = recording.then(std::time::Instant::now);
                            let res = PreparedExecutionAssessment::compute_decode_demands(def_ref, primary);
                            let ns = start.map_or(0, |s| {
                                u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
                            });
                            (res, ns)
                        },
                    ) {
                        Ok(planned) => planned,
                        Err(refusal) => return refused(refusal),
                    };
                    (plain.definition, forms, demands_res, dd_ns)
                }
            };
            if recording {
                stage_timer::record_stage(|t| t.decode_demand_ns += dd_ns);
            }
            let (demand, whole) = match demands_res {
                Ok(demands) => demands,
                Err(AssessmentError::Graph(error)) => return refused(Refusal::of(classify_graph(error))),
                Err(error) => return Err(ModelAssessmentError::Assessment(error)),
            };
            let execution = PreparedExecutionAssessment::from_demands(
                plain_definition,
                forms,
                demand,
                whole,
            );
            Ok(Ok((verdict, execution)))
        }
    }
}

/// Output of [`drafted_plan`], boxed so its manifest and forms stay off
/// [`plan_package`]'s stack frame.
struct DraftedPlan {
    method: SpeculativeMethod,
    manifest: ExecutionManifest,
    forms: FloorForms<Option<UndraftedBounds>>,
    demands: Result<(DecodeDemand, Option<DecodeDemand>), AssessmentError>,
    decode_demand_ns: u64,
    /// `None` when the pre-planned plain primary form already equals
    /// `target_alone` and is reused directly.
    alone: Option<Result<(ExecutionPlanDraft, StructuralCharge), Refusal>>,
}

/// Whether `plain`'s primary form already has the exact same target holding,
/// limits, codec, and pass as `drafted.without_drafter(plain_definition)`,
/// so it can be reused directly as `TargetAlone` without certifying undrafted
/// target/readout families.
fn can_reuse_plain_as_target_alone(
    drafted: &ExecutionPlanDraft,
    plain: &ExecutionPlanDraft,
) -> bool {
    drafted.load().expert_bank().is_none()
        && plain.load().expert_bank().is_none()
        && drafted.policy().limits() == plain.policy().limits()
        && drafted.policy().codec() == plain.policy().codec()
        && drafted.load().pass(drafted.policy().limits())
            == plain.load().pass(plain.policy().limits())
}

/// `admitted` drafting with its drafter, planned on `selected` with each
/// way of holding its experts outlined once (reusing any pre-planned plain
/// target-family layout cache, and reusing the pre-planned plain primary form
/// directly as `TargetAlone` when its holding matches), overlapping decode
/// demand and [`target_alone`] with any secondary resident form certification;
/// or the drafter's verdict (returning `package` for the plain fallback)
/// where there is no drafter, or it does not plan.
/// A frame of its own: its manifests are each a whole definition, which
/// [`plan_package`]'s frame would hold beside the target's.
fn drafted_plan(
    environment: &Environment<'_>,
    package: PackageManifest,
    drafter: DrafterAdmission,
    plain_definition: &ModelDefinition,
    selected: &SelectedDevice,
    plain_forms: Option<&FloorForms<()>>,
) -> Result<Box<DraftedPlan>, (PreparedDrafter, PackageManifest)> {
    match drafter {
        DrafterAdmission::None => Err((PreparedDrafter::None, package)),
        DrafterAdmission::Rejected(unsupported) => {
            Err((PreparedDrafter::Rejected(unsupported), package))
        }
        DrafterAdmission::Candidate { definition, method } => {
            let (manifest, method) = match environment.drafted_with_package(package, *definition, method) {
                Ok(resolved) => resolved,
                Err((unsupported, package)) => {
                    return Err((PreparedDrafter::Rejected(unsupported), package));
                }
            };
            let recording = stage_timer::is_recording();
            let def_ref = &manifest.definition;
            let plain_primary = plain_forms.map(FloorForms::primary);
            let plain_cache = plain_primary.and_then(|form| form.structure.target_cache());
            let outline = |definition: &ModelDefinition, draft: &ExecutionPlanDraft| {
                if plain_primary.is_some_and(|plain| can_reuse_plain_as_target_alone(draft, &plain.draft)) {
                    StructuralCharge::of_with_cache(definition, draft, plain_cache.as_ref())
                        .map(|charge| (charge, None))
                } else {
                    StructuralCharge::with_undrafted_with_cache(definition, draft, plain_cache.as_ref())
                        .map(|(charge, undrafted)| (charge, Some(undrafted)))
                }
            };
            match prepare_forms(
                &manifest,
                selected,
                outline,
                move |primary: &HeldForm<Option<UndraftedBounds>>| {
                    match primary.more.as_ref() {
                        Some(undrafted) => {
                            let (demands, alone) = pool::join(
                                move || {
                                    let start = recording.then(std::time::Instant::now);
                                    let res = PreparedExecutionAssessment::compute_decode_demands(def_ref, primary);
                                    let ns = start.map_or(0, |s| {
                                        u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
                                    });
                                    (res, ns)
                                },
                                || target_alone(plain_definition, selected, &primary.draft, undrafted.clone()),
                            );
                            (demands, Some(alone))
                        }
                        None => {
                            let start = recording.then(std::time::Instant::now);
                            let res = PreparedExecutionAssessment::compute_decode_demands(def_ref, primary);
                            let ns = start.map_or(0, |s| {
                                u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
                            });
                            ((res, ns), None)
                        }
                    }
                },
            ) {
                Ok((forms, ((demands, decode_demand_ns), alone))) => Ok(Box::new(DraftedPlan {
                    method,
                    manifest,
                    forms,
                    demands,
                    decode_demand_ns,
                    alone,
                })),
                Err(Refusal::Unsupported(unsupported)) => {
                    Err((PreparedDrafter::Rejected(unsupported), manifest.package))
                }
                Err(Refusal::Failed(error)) => Err((
                    PreparedDrafter::Rejected(UnsupportedModel::mismatch(error.to_string())),
                    manifest.package,
                )),
            }
        }
    }
}

/// The target's plan alone (`plain_definition`) at the drafted plan's holding
/// of experts (`ExecutionPlanDraft::without_drafter`), at the drafted plan's
/// primary form (`FloorForms::primary`), and its charge's structure: its
/// target and vision families are the drafted plan's without its drafter's
/// writes, outlined with it (`StructuralCharge::with_undrafted`); every other
/// term is the target's own, from its headers.
fn target_alone(
    plain_definition: &ModelDefinition,
    selected: &SelectedDevice,
    drafted_primary: &ExecutionPlanDraft,
    undrafted: UndraftedBounds,
) -> Result<(ExecutionPlanDraft, StructuralCharge), Refusal> {
    let backend = selected.info.backend();
    let draft = drafted_primary
        .without_drafter(plain_definition)
        .map_err(|error| Refusal::of(classify_plan(error, backend)))?;
    let structure = StructuralCharge::undrafted(plain_definition, &draft, undrafted).map_err(|error| match error {
        AssessmentError::Graph(error) => Refusal::of(classify_graph(error)),
        error => Refusal::Failed(ModelAssessmentError::Assessment(error)),
    })?;
    Ok((draft, structure))
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
    let admitted = match admit(package, HeaderInput::Files, layout)? {
        Admission::Unsupported(unsupported) => {
            return Ok(KernelInventory::Unsupported(unsupported));
        }
        Admission::Admitted(admitted) => admitted,
    };
    let environment = Environment {
        policy,
        service,
        device: DeviceRequest::Backend(backend),
        release: ReleaseRate::of_machine(None),
    };
    let plain = environment
        .plain(&admitted, admitted.definition.clone())
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
        PlanOutcome::PricesPending(_) => unreachable!("assessment plans without banking"),
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

impl Refusal {
    /// A planner's or kernel domain's outcome as a refusal.
    fn of(outcome: PlanOutcome) -> Self {
        match outcome {
            PlanOutcome::Unsupported(unsupported) => Self::Unsupported(unsupported),
            PlanOutcome::WeightBound(error) => Self::Failed(ModelAssessmentError::WeightBound(error)),
            PlanOutcome::PricesPending(_) => unreachable!("assessment plans without banking"),
            PlanOutcome::Internal(reason) => Self::Failed(ModelAssessmentError::Planning(reason)),
        }
    }
}

/// `manifest` planned on the selected device, each way of holding its
/// experts outlined once by `structure` (`ExecutionPlanDraft::floor_forms_with_primary`),
/// running `after_primary` against the primary form as soon as it is ready
/// (overlapping with any secondary resident form certification).
/// Planner refusals and kernel domains classify exactly as a load's do.
fn prepare_forms<T: Sync, U: Send>(
    manifest: &ExecutionManifest,
    selected: &SelectedDevice,
    structure: impl Fn(&ModelDefinition, &ExecutionPlanDraft) -> Result<(StructuralCharge, T), AssessmentError>,
    after_primary: impl FnOnce(&HeldForm<T>) -> U + Send,
) -> Result<(FloorForms<T>, U), Refusal> {
    let backend = selected.info.backend();
    let resident = stage_timer::time_stage(|t, ns| t.plan_resident_ns += ns, || plan_resident(manifest, selected))
        .map_err(|error| planning_refusal(error, backend))?;
    resident
        .floor_forms_with_primary(
            &manifest.definition,
            manifest.model.weight_bound,
            |draft| structure(&manifest.definition, draft).map_err(FormError::Assessment),
            after_primary,
        )
        .map_err(|error| match error {
            FormError::Plan(error) => Refusal::of(classify_plan(error, backend)),
            FormError::Assessment(AssessmentError::Graph(error)) => Refusal::of(classify_graph(error)),
            FormError::Assessment(error) => Refusal::Failed(ModelAssessmentError::Assessment(error)),
        })
}

/// Why a form of a plan has no outline: its planning's refusal, or its
/// charge's.
enum FormError {
    Plan(magnitude_executor::PlanError),
    Assessment(AssessmentError),
}

impl From<magnitude_executor::PlanError> for FormError {
    fn from(error: magnitude_executor::PlanError) -> Self {
        Self::Plan(error)
    }
}

/// A planning failure as a refusal: the planner's refusal classified as a
/// load's, any other the engine's.
fn planning_refusal(error: ExecutionPlanningError, backend: seismic::BackendName) -> Refusal {
    match error {
        ExecutionPlanningError::Plan(error) => Refusal::of(classify_plan(error, backend)),
        error => Refusal::Failed(ModelAssessmentError::Planning(error.to_string())),
    }
}

/// The serving configuration manifests are resolved under.
struct Environment<'a> {
    policy: &'a ModelPolicy,
    service: &'a ServiceLimits,
    device: DeviceRequest,
    release: ReleaseRate,
}

impl Environment<'_> {
    /// The target served alone: plain generation, without the drafter's
    /// component.
    fn plain(&self, admitted: &AdmittedPackage, definition: ModelDefinition) -> Result<ExecutionManifest, String> {
        let package = PackageManifest {
            draft: None,
            ..admitted.headers.manifest()
        };
        self.plain_with_package(package, definition)
    }

    fn plain_with_package(
        &self,
        package: PackageManifest,
        definition: ModelDefinition,
    ) -> Result<ExecutionManifest, String> {
        self.plain_with_package_or_return(package, definition)
            .map_err(|(error, _)| error)
    }

    fn plain_with_package_or_return(
        &self,
        package: PackageManifest,
        definition: ModelDefinition,
    ) -> Result<ExecutionManifest, (String, ModelDefinition)> {
        let policy = ModelPolicy {
            method: ModelMethod::Plain,
            draft_policy: None,
            mtp_proposals: None,
            ..self.policy.clone()
        };
        let model = match policy.resolve(&definition) {
            Ok(model) => model,
            Err(error) => return Err((error, definition)),
        };
        if let Err(error) = self.service.validate() {
            return Err((error, definition));
        }
        if package.identity != definition.artifact_identity {
            return Err((
                "package manifest and model definition identities differ".to_owned(),
                definition,
            ));
        }
        match self.manifest(package, definition, model) {
            Ok(manifest) => Ok(manifest),
            Err(_) => unreachable!("validated policy, service, and identity construct a manifest"),
        }
    }

    /// The target with its admitted drafter, under the method that drafts
    /// with it, and that method.
    fn drafted(
        &self,
        admitted: &AdmittedPackage,
        definition: &ModelDefinition,
        method: ModelMethod,
    ) -> Result<(ExecutionManifest, SpeculativeMethod), UnsupportedModel> {
        self.drafted_with_package(admitted.headers.manifest(), definition.clone(), method)
            .map_err(|(unsupported, _)| unsupported)
    }

    fn drafted_with_package(
        &self,
        package: PackageManifest,
        definition: ModelDefinition,
        method: ModelMethod,
    ) -> Result<(ExecutionManifest, SpeculativeMethod), (UnsupportedModel, PackageManifest)> {
        let policy = ModelPolicy {
            method,
            ..self.policy.clone()
        };
        let model = match policy.resolve(&definition) {
            Ok(model) => model,
            Err(error) => return Err((UnsupportedModel::mismatch(error), package)),
        };
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
                return Err((
                    UnsupportedModel::mismatch("the drafter resolves to plain generation"),
                    package,
                ));
            }
        };
        if let Err(error) = self.service.validate() {
            return Err((UnsupportedModel::mismatch(error), package));
        }
        if package.identity != definition.artifact_identity {
            return Err((
                UnsupportedModel::mismatch("package manifest and model definition identities differ"),
                package,
            ));
        }
        match self.manifest(package, definition, model) {
            Ok(manifest) => Ok((manifest, method)),
            Err(_) => unreachable!("validated policy, service, and identity construct a manifest"),
        }
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
            self.release,
        )
    }
}

enum Admission {
    Unsupported(UnsupportedModel),
    Admitted(Box<AdmittedPackage>),
}

/// A target the engine admits from its headers, with what its drafter's
/// headers decided.
struct AdmittedPackage {
    headers: PackageHeaders,
    definition: ModelDefinition,
    drafter: DrafterAdmission,
}

enum TargetAdmission {
    Unsupported {
        unsupported: UnsupportedModel,
        fingerprint: Option<ModelFingerprint>,
    },
    Admitted {
        target: Box<AdmittedTarget>,
        prefetched: Option<PrefetchedDraft>,
    },
}

enum TargetHeadersAdmission {
    Unsupported {
        unsupported: UnsupportedModel,
        fingerprint: Option<ModelFingerprint>,
    },
    Admitted(Box<AdmittedTarget>),
}

/// A target header admitted before chat template inspection, target
/// fingerprinting and drafter admission so callers can overlap
/// `bundle.inspect_cached()` and `ModelFingerprint::from_directory` with
/// execution planning.
struct AdmittedTarget {
    headers: PackageHeaders,
    family: &'static dyn ModelFamily,
    definition: ModelDefinition,
    bundle: TemplateBundle,
    projector: ProjectorVerdict,
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

/// Where `admit` reads GGUF headers: directly from files (allowing draft
/// header inspection to overlap with target header inspection on the
/// assessment pool), or from a caller-provided [`HeaderSource`].
#[derive(Clone, Copy)]
enum HeaderInput<'a> {
    Files,
    Source(&'a dyn HeaderSource),
}

impl<'a> HeaderInput<'a> {
    fn as_source(&self) -> &dyn HeaderSource {
        match self {
            Self::Files => &FileHeaders,
            Self::Source(source) => *source,
        }
    }
}

type PrefetchedDraft = Result<
    (Directory, Vec<magnitude_artifacts::ComponentFile>, Option<UnsupportedModel>),
    magnitude_artifacts::Error,
>;

/// Admit `package` from its headers: the target's weight encodings, family
/// interpretation, tokenizer and chat templates are independent checks,
/// and every refusal among them is reported together. An admitted target's
/// drafter is admitted the same way, against the target's definition.
fn admit(
    package: &ModelPackagePaths,
    input: HeaderInput<'_>,
    layout: Layout,
) -> Result<Admission, ModelAssessmentError> {
    let (mut target, prefetched) = match admit_target(package, input, layout)? {
        TargetAdmission::Unsupported { unsupported, .. } => {
            return Ok(Admission::Unsupported(unsupported));
        }
        TargetAdmission::Admitted { target, prefetched } => (target, prefetched),
    };
    if let Err(error) = stage_timer::time_stage(
        |t, ns| t.chat_inspect_ns += ns,
        || target.bundle.inspect_cached(),
    ) {
        return Ok(Admission::Unsupported(UnsupportedModel::chat_template(
            error,
        )));
    }
    let _ = u32::try_from(target.definition.decoder.context_limit)
        .map_err(|_| ModelAssessmentError::ContextLimit(target.definition.decoder.context_limit))?;
    let drafter = stage_timer::time_stage(|t, ns| t.admit_drafter_ns += ns, || {
        admit_drafter(
            package,
            input.as_source(),
            &mut target.headers,
            prefetched,
            target.family,
            &target.definition,
            layout,
        )
    })?;
    Ok(Admission::Admitted(Box::new(AdmittedPackage {
        headers: target.headers,
        definition: target.definition,
        drafter,
    })))
}

fn prefetch_draft(path: &Path, layout: Layout) -> PrefetchedDraft {
    let (directory, files) = PackageHeaders::inspect_component(path)?;
    let weight_refusal = unsupported_weights(&directory.tensors, Residency::Packed, layout);
    Ok((directory, files, weight_refusal))
}

fn admit_target_headers_timed(
    package: &ModelPackagePaths,
    source: &dyn HeaderSource,
    layout: Layout,
) -> Result<TargetHeadersAdmission, ModelAssessmentError> {
    let recording = stage_timer::is_recording();
    let start = recording.then(std::time::Instant::now);
    let before_sub = if recording {
        let mut sub = 0u64;
        stage_timer::record_stage(|t| sub = t.header_open_ns + t.chat_inspect_ns);
        sub
    } else {
        0
    };
    let target_res = admit_target_headers(package, source, layout);
    if let Some(start) = start {
        let total = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        stage_timer::record_stage(|t| {
            let sub = (t.header_open_ns + t.chat_inspect_ns).saturating_sub(before_sub);
            t.admit_other_ns += total.saturating_sub(sub);
        });
    }
    target_res
}

/// Admit `package`'s target headers while prefetching its separate draft
/// component (if any) concurrently. When target headers have no refusals,
/// returns the parsed [`TemplateBundle`] uninspected and defers target
/// fingerprinting so callers can overlap `bundle.inspect_cached()` and
/// `ModelFingerprint::from_directory` with execution planning; when target
/// headers already have refusals, runs both immediately to preserve the
/// exact refusal union and unsupported fingerprint.
fn admit_target(
    package: &ModelPackagePaths,
    input: HeaderInput<'_>,
    layout: Layout,
) -> Result<TargetAdmission, ModelAssessmentError> {
    let source = input.as_source();
    let draft_prefetch_path = match input {
        HeaderInput::Files if package.method != ModelMethod::Plain => package.draft.as_deref(),
        _ => None,
    };
    let (prefetched, target_res) = if let Some(path) = draft_prefetch_path {
        let (prefetched, target_res) = pool::join(
            move || prefetch_draft(path, layout),
            || admit_target_headers_timed(package, source, layout),
        );
        (Some(prefetched), target_res)
    } else {
        (None, admit_target_headers_timed(package, source, layout))
    };
    match target_res? {
        TargetHeadersAdmission::Unsupported {
            unsupported,
            fingerprint,
        } => Ok(TargetAdmission::Unsupported {
            unsupported,
            fingerprint,
        }),
        TargetHeadersAdmission::Admitted(target) => Ok(TargetAdmission::Admitted {
            target,
            prefetched,
        }),
    }
}

fn admit_target_headers(
    package: &ModelPackagePaths,
    source: &dyn HeaderSource,
    layout: Layout,
) -> Result<TargetHeadersAdmission, ModelAssessmentError> {
    let headers = match stage_timer::time_stage(|t, ns| t.header_open_ns += ns, || {
        PackageHeaders::open_shards_from(&package.target_shards, package.projector.as_deref(), source)
    }) {
        Ok(headers) => headers,
        Err(error) => match artifact_refusal(&error) {
            Some(unsupported) => {
                return Ok(TargetHeadersAdmission::Unsupported {
                    unsupported,
                    fingerprint: None,
                });
            }
            None => return Err(ModelAssessmentError::Artifact(error)),
        },
    };
    let target_name = package.target().display().to_string();
    let (mut refusals, header_bundle) = {
        let target = headers.target();
        let projector_dir = headers.projector();
        let identity = headers.identity();
        let recording = stage_timer::is_recording();
        let (((weight_refusal, uw_ns), (tokenizer, vocabulary, bundle, tv_ns, gt_ns)), interpreted) =
            pool::join(
                || {
                    pool::join(
                        || {
                            let start = recording.then(std::time::Instant::now);
                            let res = unsupported_weights(&target.tensors, Residency::Packed, layout);
                            let ns = start.map_or(0, |s| {
                                u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
                            });
                            (res, ns)
                        },
                        || parse_tokenizer_and_templates(target, &target_name, recording),
                    )
                },
                || {
                    stage_timer::time_stage(|t, ns| t.admit_interpret_target_ns += ns, || {
                        crate::families::recognize(target).and_then(|family| {
                            interpret_target(family, target, projector_dir, identity)
                                .map(|interpreted| (family, interpreted))
                        })
                    })
                },
            );
        if recording {
            stage_timer::record_stage(|t| {
                t.admit_unsupported_weights_ns += uw_ns;
                t.admit_tokenizer_vocab_ns += tv_ns;
                t.admit_gguf_templates_ns += gt_ns;
            });
        }
        let mut refusals = Vec::new();
        refusals.extend(weight_refusal);
        let header_bundle = assemble_header_bundle(interpreted, tokenizer, vocabulary, bundle);
        // A bound projector loads its weights; one that does not bind loads none.
        if let (HeaderBundle::Ready { interpreted, .. }, Some(projector)) =
            (&header_bundle, projector_dir)
        {
            if interpreted.projector_unavailable.is_none() {
                refusals.extend(stage_timer::time_stage(
                    |t, ns| t.admit_unsupported_weights_ns += ns,
                    || {
                        unsupported_weights(
                            &projector.tensors,
                            Residency::Dequantized,
                            layout,
                        )
                    },
                ));
            }
        }
        (refusals, header_bundle)
    };
    let finalize_refused = |mut refusals: Vec<UnsupportedModel>, bundle: Option<TemplateBundle>| {
        let target_dir = headers.target();
        let fingerprint = if let Some(bundle) = bundle {
            let recording = stage_timer::is_recording();
            let ((chat_res, chat_ns), (fp, fp_ns)) = pool::join(
                move || {
                    let chat_start = recording.then(std::time::Instant::now);
                    let chat_res = bundle.inspect_cached();
                    let chat_ns = chat_start.map_or(0, |s| {
                        u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
                    });
                    (chat_res, chat_ns)
                },
                || {
                    let fp_start = recording.then(std::time::Instant::now);
                    let fp = ModelFingerprint::from_directory(target_dir).ok();
                    let fp_ns = fp_start.map_or(0, |s| {
                        u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
                    });
                    (fp, fp_ns)
                },
            );
            if recording {
                stage_timer::record_stage(|t| {
                    t.chat_inspect_ns += chat_ns;
                    t.admit_fingerprint_ns += fp_ns;
                });
            }
            if let Err(error) = chat_res {
                refusals.push(UnsupportedModel::chat_template(error));
            }
            fp
        } else {
            stage_timer::time_stage(
                |t, ns| t.admit_fingerprint_ns += ns,
                || ModelFingerprint::from_directory(target_dir).ok(),
            )
        };
        TargetHeadersAdmission::Unsupported {
            unsupported: UnsupportedModel::union(refusals)
                .expect("a refused admission has a refusal"),
            fingerprint,
        }
    };
    let (
        family,
        InterpretedTarget {
            definition,
            projector_unavailable,
        },
        bundle,
    ) = match header_bundle {
        HeaderBundle::Ready {
            family,
            interpreted,
            bundle,
        } if refusals.is_empty() => (family, interpreted, bundle),
        HeaderBundle::Ready { bundle, .. } => {
            return Ok(finalize_refused(refusals, Some(bundle)));
        }
        HeaderBundle::Refused {
            refusals: header_refusals,
            bundle,
        } => {
            refusals.extend(header_refusals);
            return Ok(finalize_refused(refusals, bundle));
        }
    };
    let projector = match (headers.projector(), projector_unavailable) {
        (_, Some(unsupported)) => ProjectorVerdict::Rejected(unsupported),
        (Some(_), None) => ProjectorVerdict::Bound,
        (None, None) => ProjectorVerdict::None,
    };
    Ok(TargetHeadersAdmission::Admitted(Box::new(AdmittedTarget {
        headers,
        family,
        definition,
        bundle,
        projector,
    })))
}

/// Result of interpreting a target header before [`TemplateBundle::inspect_cached`].
enum HeaderBundle {
    Ready {
        family: &'static dyn ModelFamily,
        interpreted: InterpretedTarget,
        bundle: TemplateBundle,
    },
    Refused {
        refusals: Vec<UnsupportedModel>,
        bundle: Option<TemplateBundle>,
    },
}

/// Parse a target header's tokenizer payload, vocabulary size, and chat template
/// bundle, timing each step when `recording` is enabled.
fn parse_tokenizer_and_templates(
    target: &Directory,
    name: &str,
    recording: bool,
) -> (
    TokenizerPayload,
    Result<usize, UnsupportedModel>,
    Result<TemplateBundle, UnsupportedModel>,
    u64,
    u64,
) {
    let tv_start = recording.then(std::time::Instant::now);
    let tokenizer = TokenizerPayload::from_directory(target);
    // Support and vocabulary are header facts; a load builds the tokenizer.
    let vocabulary = gguf_tokenizer_vocabulary(&tokenizer).map_err(UnsupportedModel::from);
    let tv_ns = tv_start.map_or(0, |s| {
        u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
    });
    let gt_start = recording.then(std::time::Instant::now);
    let bundle = TemplatePayload::from_directory(target, name)
        .map_err(|error| error.to_string())
        .and_then(|templates| gguf_templates(&templates, &tokenizer))
        .map_err(UnsupportedModel::chat_template);
    let gt_ns = gt_start.map_or(0, |s| {
        u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX)
    });
    (tokenizer, vocabulary, bundle, tv_ns, gt_ns)
}

/// Assemble a [`HeaderBundle`] from the independent family interpretation,
/// tokenizer vocabulary check, and parsed template bundle.
fn assemble_header_bundle(
    interpreted: Result<(&'static dyn ModelFamily, InterpretedTarget), UnsupportedModel>,
    tokenizer: TokenizerPayload,
    vocabulary: Result<usize, UnsupportedModel>,
    bundle: Result<TemplateBundle, UnsupportedModel>,
) -> HeaderBundle {
    let mut refusals = Vec::new();
    if let (Ok((_, interpreted)), Ok(vocabulary)) = (&interpreted, &vocabulary) {
        if u64::try_from(*vocabulary).ok() != Some(interpreted.definition.decoder.vocabulary) {
            refusals.push(vocabulary_mismatch(&tokenizer));
        }
    }
    match (interpreted, vocabulary, bundle) {
        (Ok((family, interpreted)), Ok(_), Ok(bundle)) if refusals.is_empty() => {
            HeaderBundle::Ready {
                family,
                interpreted,
                bundle,
            }
        }
        (interpreted, vocabulary, bundle) => {
            refusals.extend(interpreted.err());
            refusals.extend(vocabulary.err());
            let bundle = match bundle {
                Ok(bundle) => Some(bundle),
                Err(error) => {
                    refusals.push(error);
                    None
                }
            };
            HeaderBundle::Refused { refusals, bundle }
        }
    }
}

/// A target header's interpretation up to template bundle parsing: its family's
/// definition with its projector bound when the family binds it
/// ([`interpret_target`]), the tokenizer it declares (checked against the
/// definition's vocabulary) and its parsed [`TemplateBundle`].
fn interpret_header_bundle(
    target: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
    name: &str,
) -> HeaderBundle {
    let interpreted = stage_timer::time_stage(|t, ns| t.admit_interpret_target_ns += ns, || {
        crate::families::recognize(target).and_then(|family| {
            interpret_target(family, target, projector, identity)
                .map(|interpreted| (family, interpreted))
        })
    });
    let recording = stage_timer::is_recording();
    let (tokenizer, vocabulary, bundle, tv_ns, gt_ns) =
        parse_tokenizer_and_templates(target, name, recording);
    if recording {
        stage_timer::record_stage(|t| {
            t.admit_tokenizer_vocab_ns += tv_ns;
            t.admit_gguf_templates_ns += gt_ns;
        });
    }
    assemble_header_bundle(interpreted, tokenizer, vocabulary, bundle)
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
    match interpret_header_bundle(target, projector, identity, name) {
        HeaderBundle::Ready {
            family,
            interpreted,
            bundle,
        } => {
            let chat = stage_timer::time_stage(
                |t, ns| t.chat_inspect_ns += ns,
                || bundle.inspect_cached(),
            )
            .map_err(UnsupportedModel::chat_template);
            match chat {
                Ok(chat) => Ok((family, interpreted, chat)),
                Err(error) => Err(vec![error]),
            }
        }
        HeaderBundle::Refused {
            mut refusals,
            bundle,
        } => {
            if let Some(bundle) = bundle {
                if let Err(error) = stage_timer::time_stage(
                    |t, ns| t.chat_inspect_ns += ns,
                    || bundle.inspect_cached(),
                ) {
                    refusals.push(UnsupportedModel::chat_template(error));
                }
            }
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
    prefetched: Option<PrefetchedDraft>,
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
    let weight_refusal = match prefetched {
        Some(Ok((directory, files, weight_refusal))) => {
            match headers.add_preopened_draft(path, directory, files) {
                Ok(()) => weight_refusal,
                Err(error) => match artifact_refusal(&error) {
                    Some(unsupported) => return Ok(DrafterAdmission::Rejected(unsupported)),
                    None => return Err(ModelAssessmentError::Artifact(error)),
                },
            }
        }
        Some(Err(error)) => match artifact_refusal(&error) {
            Some(unsupported) => return Ok(DrafterAdmission::Rejected(unsupported)),
            None => return Err(ModelAssessmentError::Artifact(error)),
        },
        None => {
            match headers.add_draft_from(path, source) {
                Ok(()) => {}
                Err(error) => match artifact_refusal(&error) {
                    Some(unsupported) => return Ok(DrafterAdmission::Rejected(unsupported)),
                    None => return Err(ModelAssessmentError::Artifact(error)),
                },
            }
            let draft = headers.draft().expect("the draft component was just added");
            unsupported_weights(&draft.tensors, Residency::Packed, layout)
        }
    };
    let draft = headers.draft().expect("the draft component was just added");
    let mut refusals = Vec::new();
    refusals.extend(weight_refusal);
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
        // A weight whose scales are read nowhere is imported nowhere.
        let imported = scales.is_some_and(|scales| match residency {
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
        });
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

    #[test]
    fn served_contexts_are_the_limit_then_powers_of_two_to_the_floor() {
        assert_eq!(
            served_contexts(262_144),
            vec![262_144, 131_072, 65_536, 32_768]
        );
        assert_eq!(
            served_contexts(1_048_576),
            vec![1_048_576, 524_288, 262_144, 131_072, 65_536, 32_768]
        );
        assert_eq!(served_contexts(40_960), vec![40_960, 32_768]);
        assert_eq!(served_contexts(131_000), vec![131_000, 65_536, 32_768]);
        assert_eq!(served_contexts(SERVED_CONTEXT_FLOOR), vec![SERVED_CONTEXT_FLOOR]);
        assert_eq!(served_contexts(8_192), vec![8_192]);
        assert_eq!(served_contexts(u32::MAX).last(), Some(&SERVED_CONTEXT_FLOOR));
        assert!(served_contexts(u32::MAX).windows(2).all(|pair| pair[0] > pair[1]));
    }

    /// The served tier is the largest that fits, found with the largest
    /// assessed first, then the smallest, then by bisection; a shortfall is
    /// the smallest tier's.
    #[test]
    fn the_served_tier_is_the_largest_that_fits() {
        for count in 1..=7usize {
            for first_fit in 0..=count {
                let mut assessed = Vec::new();
                let served = served_tier(count, |index| {
                    assessed.push(index);
                    Ok::<_, ()>(if index >= first_fit { Ok(index) } else { Err(index) })
                })
                .unwrap();
                match first_fit {
                    _ if first_fit == count => assert_eq!(served, Err(count - 1)),
                    _ => assert_eq!(served, Ok(first_fit)),
                }
                let mut once = assessed.clone();
                once.sort();
                once.dedup();
                assert_eq!(once.len(), assessed.len(), "no tier is assessed twice: {assessed:?}");
                assert!(assessed.len() <= 2 + count.next_power_of_two().trailing_zeros() as usize);
                match first_fit {
                    0 => assert_eq!(assessed, vec![0]),
                    _ if first_fit == count => assert_eq!(assessed, (0..count).filter(|i| *i == 0 || *i == count - 1).collect::<Vec<_>>()),
                    _ => {}
                }
            }
        }
    }
}
