//! The engine loaded in process and driven through its executor domain,
//! below the scheduler: every step's operations are chosen by the harness,
//! and every step can be traced per device submission.

use crate::artifacts::Artifacts;
use magnitude_engine::composition::{
    EngineConfiguration, ReadyEngine, ResolvedEngineConfiguration, load_in_process,
    start_in_process,
};
use magnitude_engine::load_native_domain;
use magnitude_engine::options::{
    ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection, ResolvedMethod,
};
use magnitude_engine::worker::protocol::LoadProgress;
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
use magnitude_executor::{
    Demand, DomainError, ExecutionPath, ExecutorDomain, NumericalProfile, Operation, Outcome,
    PhysicalDecision, RequestId, ReservedResources, Sampling, SelectSpec, Shaping, StateBindings,
    TokenId, TuningSettings, WorkKind,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_scheduler::ServiceLimits;
use magnitude_state::KvCodec;
use seismic::{SubmissionTrace, TracedSubmission, host_seconds};
use std::cell::Cell;
use std::path::PathBuf;

/// Rows of one prefill forward, as the service bounds them.
pub const PREFILL_ROWS: usize = 512;

/// How every load of one run is made.
pub struct EngineOptions {
    pub cache_dir: Option<PathBuf>,
    pub device: DeviceRequest,
    pub kv_codec: KvCodec,
    pub numerical_profile: Option<NumericalProfile>,
    /// Bytes every weight of a load may hold (`ModelPolicy::weight_cap`).
    pub weight_cap: Option<u64>,
    /// The requested prompt group (`ModelPolicy::prefill_group`).
    pub prefill_group: magnitude_executor::PrefillGroup,
    pub budget: std::time::Duration,
    /// Search afresh in the Prepare of the run's first load. Later loads of
    /// the run reuse what it stored, as any later load would.
    pub fresh: Cell<bool>,
    /// The run's tuning trace; every Prepare and every load appends to it.
    pub tuning_trace: PathBuf,
    /// Arguments of this executable's worker role besides its subcommand:
    /// the development tools that belong to a model's Prepare.
    pub worker_arguments: Vec<std::ffi::OsString>,
}

impl EngineOptions {
    /// The tuning settings of the run's next load.
    fn tuning(&self) -> TuningSettings {
        TuningSettings {
            budget: self.budget,
            fresh: self.fresh.replace(false),
        }
    }
}

/// What one load serves.
pub struct Load {
    /// Load the catalog drafter (its own method) rather than the target alone.
    pub drafter: bool,
    pub proposals: Option<u8>,
    pub context_tokens: usize,
    /// Rows one decode step may carry: concurrent sequences times their width.
    pub decode_rows: usize,
    /// Rows one launch exports full logits for; zero loads as served.
    pub exported_logits_rows: usize,
    pub lookahead: bool,
}

/// Resolve the catalog model for `load`, as the engine's host would.
pub fn configuration(
    artifacts: &Artifacts,
    options: &EngineOptions,
    load: &Load,
) -> Result<ResolvedEngineConfiguration, String> {
    let draft = if load.drafter {
        Some(
            artifacts
                .drafter
                .as_ref()
                .ok_or_else(|| format!("{} declares no drafter", artifacts.model))?
                .path
                .clone(),
        )
    } else {
        None
    };
    let mut resolved = EngineConfiguration {
        package: PackageOptions {
            target: artifacts.target.clone(),
            projector: ProjectorSelection::Disabled,
            draft,
        },
        model: ModelPolicy {
            draft_policy: None,
            method: if load.drafter {
                ModelMethod::Auto
            } else {
                ModelMethod::Plain
            },
            mtp_proposals: load.proposals,
            kv_codec: options.kv_codec,
            lookahead: load.lookahead,
            exported_logits_rows: load.exported_logits_rows,
            numerical_profile: options.numerical_profile.clone(),
            weight_cap: options.weight_cap,
            prefill_group: options.prefill_group,
        },
        context_tokens: Some(load.context_tokens),
        service: ServiceLimits {
            prefill_tokens: PREFILL_ROWS,
            decode_tokens: load.decode_rows,
            decode_share: 0.5,
            locality_seconds: 1.0,
        },
        path: ExecutionPath::Native,
        device: options.device,
        kernel_cache: options.cache_dir.clone(),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .map_err(|error| error.to_string())?;
    resolved.manifest.tuning = options.tuning();
    resolved.manifest.tuning_trace = Some(options.tuning_trace.clone());
    Ok(resolved)
}

/// The subcommand this executable runs the engine's worker under.
pub const WORKER_COMMAND: &str = "worker";

/// This executable in its worker role: what a load starts for the model's
/// Prepare when it finds tuning units without a stored record.
pub fn worker(options: &EngineOptions) -> Result<std::process::Command, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("locating the harness executable: {error}"))?;
    let mut worker = std::process::Command::new(executable);
    worker.arg(WORKER_COMMAND).args(&options.worker_arguments);
    Ok(worker)
}

/// Load `resolved` on a worker thread of this process behind the engine's
/// worker protocol, as the engine's own hosts load: the model's Prepare
/// first, in a worker process of this executable, when tuning records are
/// missing. `progress` sees each step.
pub fn start(
    resolved: ResolvedEngineConfiguration,
    options: &EngineOptions,
    progress: impl FnMut(LoadProgress),
) -> Result<ReadyEngine, String> {
    start_in_process(resolved, worker(options)?, progress).map_err(|error| error.to_string())
}

pub struct Engine {
    pub(crate) domain: ExecutorDomain,
    /// The domain's state bindings while no flight carries them.
    pub(crate) bindings: Option<StateBindings>,
    device: seismic::Device,
    pub vocabulary: usize,
    /// The loaded generation method: which drafter, if any, and its width.
    pub method: ResolvedMethod,
    next_request: u64,
    /// Under cross-step pipelining a step's submissions are traced during
    /// the previous step, so traces are collected per window.
    pub lookahead: bool,
}

/// One open request and its accepted position.
pub struct Sequence {
    pub request: RequestId,
    pub position: usize,
    /// The token the last step selected.
    pub selected: Option<TokenId>,
}

/// Host and device times of one step. All times are `seismic::host_seconds`.
pub struct Step {
    pub start: f64,
    pub submitted: f64,
    pub finished: f64,
    pub end: f64,
    pub submissions: Vec<TracedSubmission>,
}

pub fn text(error: DomainError) -> String {
    error.to_string()
}

impl Engine {
    pub fn open(
        artifacts: &Artifacts,
        options: &EngineOptions,
        load: &Load,
    ) -> Result<Self, String> {
        let resolved = configuration(artifacts, options, load)?;
        let vocabulary = usize::try_from(resolved.manifest.definition.decoder.vocabulary)
            .map_err(|_| "vocabulary exceeds the host domain")?;
        let method = resolved.manifest.model.method;
        let (domain, bindings, _) = load_in_process(
            &resolved.manifest,
            |unprepared| {
                load_native_domain(
                    &resolved.manifest,
                    resolved.host.shared_package(),
                    unprepared,
                )
            },
            worker(options)?,
            |_| {},
        )
        .map_err(|error| error.to_string())?;
        let device = domain.resources().device().clone();
        Ok(Self {
            domain,
            bindings: Some(bindings),
            device,
            vocabulary,
            method,
            next_request: 1,
            lookahead: load.lookahead,
        })
    }

    pub fn device(&self) -> &seismic::Device {
        &self.device
    }

    pub(crate) fn bindings(&mut self) -> StateBindings {
        self.bindings
            .take()
            .expect("no flight holds the state bindings")
    }

    pub fn open_sequence(&mut self) -> Result<Sequence, String> {
        let request = RequestId(self.next_request);
        self.next_request += 1;
        self.domain
            .install_input(request, PreparedModelInput::continuation_only())?;
        let bindings = self
            .bindings
            .as_mut()
            .expect("no flight holds the state bindings");
        self.domain
            .open_state(bindings, request, None)
            .map_err(text)?;
        Ok(Sequence {
            request,
            position: 0,
            selected: None,
        })
    }

    pub fn close(&mut self, sequence: Sequence) -> Result<(), String> {
        self.domain.close(sequence.request)
    }

    pub fn greedy(position: usize) -> SelectSpec {
        SelectSpec {
            sampling: Sampling::Greedy,
            seed: 0,
            position,
            domain: 0,
            mask: None,
            shaping: Shaping::default(),
            history: None,
        }
    }

    /// A forward of `tokens` at the sequence's position. `SELECT` selects
    /// greedily from the last row (a decode, or a finishing prefill).
    pub fn forward(
        sequence: &Sequence,
        kind: WorkKind,
        tokens: Vec<TokenId>,
        demand: Demand,
    ) -> Operation {
        Self::forward_at(sequence, sequence.position, kind, tokens, demand)
    }

    /// The forward of `tokens` from `position` of the sequence.
    pub fn forward_at(
        sequence: &Sequence,
        position: usize,
        kind: WorkKind,
        tokens: Vec<TokenId>,
        demand: Demand,
    ) -> Operation {
        let last = position + tokens.len() - 1;
        let select = if demand.contains(Demand::SELECT) {
            vec![Self::greedy(last)]
        } else {
            Vec::new()
        };
        let committed = tokens.len();
        Operation::Forward {
            request: sequence.request,
            kind,
            tokens,
            position,
            conditioning: None,
            demand,
            select,
            committed,
            prime: None,
        }
    }

    /// Submit one group of target forwards, wait, commit every row and
    /// advance the sequences. Returns the step's timings (submissions from
    /// `trace`, when given) and the outcomes in operation order.
    pub fn step(
        &mut self,
        operations: Vec<Operation>,
        sequences: &mut [&mut Sequence],
        trace: Option<&SubmissionTrace>,
    ) -> Result<(Step, Vec<Outcome>), String> {
        let mut bindings = self.bindings();
        let start = host_seconds();
        let ReservedResources::Target(reservation) = self
            .domain
            .reserve(&mut bindings, &operations)
            .map_err(text)?
            .into_resources()
        else {
            return Err("a forward step runs on the target lane".into());
        };
        let flight = self
            .domain
            .submit_target(bindings, &operations, reservation)
            .map_err(|failure| failure.error().to_string())?;
        let submitted = host_seconds();
        let (pending, bindings) = self.domain.finish_target(flight).map_err(text)?;
        self.bindings = Some(bindings);
        let finished = host_seconds();
        if pending.len() != sequences.len() {
            return Err("step outcomes differ from its sequences".into());
        }
        let mut outcomes = Vec::with_capacity(pending.len());
        for (pending, sequence) in pending.into_iter().zip(sequences.iter_mut()) {
            if pending.request() != sequence.request {
                return Err("step outcome order differs from its sequences".into());
            }
            if let Outcome::Forward { rows } = pending.outcome() {
                sequence.selected = rows
                    .last()
                    .and_then(|row| row.selected)
                    .map(|row| row.token);
            }
            outcomes.push(pending.outcome().clone());
            let rows = pending.rows();
            self.domain
                .reconcile(
                    pending,
                    PhysicalDecision {
                        accepted_rows: rows,
                    },
                )
                .map_err(text)?;
            sequence.position += rows;
        }
        let end = host_seconds();
        let submissions = trace
            .map(|trace| trace.collect().map_err(|error| error.to_string()))
            .transpose()?
            .unwrap_or_default();
        Ok((
            Step {
                start,
                submitted,
                finished,
                end,
                submissions,
            },
            outcomes,
        ))
    }

    /// The prompt chunks that follow the next step, as the service's owner
    /// plans them (`ExecutorDomain::plan_successors`).
    pub fn plan_successors(&mut self, planned: Vec<Operation>) {
        self.domain.plan_successors(planned);
    }

    /// The prefill chunks of `tokens` from `position` of the sequence, as
    /// the service's owner plans a prompt: `PREFILL_ROWS` rows each, the
    /// last selecting when `select`.
    pub fn prompt_chunks(
        sequence: &Sequence,
        position: usize,
        tokens: &[TokenId],
        select: bool,
    ) -> Vec<Operation> {
        let chunks = tokens.chunks(PREFILL_ROWS).collect::<Vec<_>>();
        let mut position = position;
        chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                let demand = if select && index + 1 == chunks.len() {
                    Demand::SELECT
                } else {
                    Demand::NONE
                };
                let operation = Self::forward_at(
                    sequence,
                    position,
                    WorkKind::Prefill,
                    chunk.to_vec(),
                    demand,
                );
                position += chunk.len();
                operation
            })
            .collect()
    }

    /// Accept `tokens` from the sequence's position in prefill chunks, as the
    /// service runs a prompt. The last chunk selects when `select`.
    ///
    /// Before each chunk the domain is told the chunks after it, as the
    /// service's owner plans them: a load that prefills layer-major submits a
    /// group of them together and later chunks claim their queued launches;
    /// otherwise each chunk queues the next behind it (lookahead).
    pub fn prefill(
        &mut self,
        sequence: &mut Sequence,
        tokens: &[TokenId],
        select: bool,
    ) -> Result<(), String> {
        let operations = Self::prompt_chunks(sequence, sequence.position, tokens, select);
        for (index, operation) in operations.iter().enumerate() {
            self.plan_successors(operations[index + 1..].to_vec());
            self.step(vec![operation.clone()], &mut [&mut *sequence], None)?;
        }
        Ok(())
    }
}

pub fn tokens(ids: &[u32]) -> Vec<TokenId> {
    ids.iter().copied().map(TokenId).collect()
}
