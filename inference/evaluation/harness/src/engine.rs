//! The engine loaded in process and driven through its executor domain,
//! below the scheduler: every step's operations are chosen by the harness,
//! and every step can be traced per device submission.

use crate::artifacts::Artifacts;
use magnitude_artifacts::InputLayout;
use magnitude_engine::composition::{
    EngineConfiguration, ResolvedEngineConfiguration, load_in_process,
};
use magnitude_engine::load_native_domain;
use magnitude_engine::options::{
    ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection, ResolvedMethod,
};
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
use magnitude_executor::{
    Demand, DomainError, ExecutionPath, ExecutorDomain, NumericalProfile, Operation, Outcome,
    PhysicalDecision, ProgramFamily, RequestId, ReservedResources, Sampling, SelectSpec, Shaping,
    StateBindings, TokenId, TuningSettings, WorkKind,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_scheduler::owner::{BetweenFlights, MemoryHost, between_flights};
use magnitude_scheduler::ServiceLimits;
use magnitude_state::KvCodec;
use seismic::{SubmissionTrace, TracedSubmission, host_seconds};
use std::cell::Cell;
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Rows of one prefill forward, as the service bounds them.
pub const PREFILL_ROWS: usize = 512;

/// How every load of one run is made.
pub struct EngineOptions {
    pub cache_dir: Option<PathBuf>,
    pub device: DeviceRequest,
    pub kv_codec: KvCodec,
    pub numerical_profile: Option<NumericalProfile>,
    /// A ceiling on every weight byte each load holds.
    pub weight_cap: Option<u64>,
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
    /// Sequences the load holds at once, each up to `context_tokens`: the
    /// plan charges their state (`ServiceLimits::conversations`).
    pub sequences: std::num::NonZeroUsize,
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
        },
        context_tokens: Some(load.context_tokens),
        service: ServiceLimits {
            prefill_tokens: PREFILL_ROWS,
            decode_tokens: load.decode_rows,
            decode_share: 0.5,
            locality_seconds: 1.0,
            conversations: load.sequences,
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

/// This executable in its worker role, with `arguments`: what a load starts
/// for the model's Prepare when it finds tuning units without a stored
/// record, and what `memory` loads in.
pub fn worker(arguments: &[std::ffi::OsString]) -> Result<std::process::Command, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("locating the harness executable: {error}"))?;
    let mut worker = std::process::Command::new(executable);
    worker.arg(WORKER_COMMAND).args(arguments);
    Ok(worker)
}

/// The model's Prepare alone, as a load that found tuning records missing
/// runs it: its search, in a worker process of this executable, publishing
/// records to the kernel cache, with nothing imported or loaded here.
pub fn prepare(artifacts: &Artifacts, options: &EngineOptions, load: &Load) -> Result<(), String> {
    use magnitude_engine::worker::{Preparation, prepare, start_worker_process};
    let resolved = configuration(artifacts, options, load)?;
    let command = worker(&options.worker_arguments)?;
    match prepare(&resolved.manifest, || start_worker_process(command), |_| {})
        .map_err(|error| error.to_string())?
    {
        Preparation::Prepared | Preparation::Unneeded => Ok(()),
        Preparation::Unshared => Err("a Prepare needs a kernel cache (--cache-dir)".into()),
        Preparation::Incomplete(error) => Err(format!("the Prepare did not finish: {error}")),
    }
}

/// Run the measurement job, by its own code (`worker::measure`), until the
/// points a load of `resolved` waits on and the executor's costs are
/// recorded: no job runs beside the harness, and a cell times what a served
/// load is once its job has run. It stops early only where a job leaves
/// them as they were, reporting why, and the load then serves what is
/// pending as any load does. Nothing to do without a kernel cache.
fn settle_measurements(artifacts: &Artifacts, resolved: &magnitude_engine::composition::ResolvedEngineConfiguration) -> Result<(), String> {
    use magnitude_engine::worker::measure::{measure_machine, MeasureEvent, MeasureJob, MeasureRequest};
    use magnitude_executor::assessment::{PointRequests, MEASUREMENT_SECONDS};
    let manifest = &resolved.manifest;
    let Some(root) = &manifest.kernel_cache else {
        return Ok(());
    };
    let directory = std::path::absolute(&artifacts.target)
        .map_err(|error| error.to_string())?
        .parent()
        .ok_or_else(|| format!("{} has no directory", artifacts.target.display()))?
        .to_path_buf();
    let paths = vec![directory];
    let catalog = seismic::Devices::discover(&paths).map_err(|error| error.to_string())?;
    magnitude_engine::request_load_points(&catalog, manifest, resolved.host.package()).map_err(|error| error.to_string())?;
    let selected = magnitude_executor::platform::select_recorded(&catalog, manifest.path, manifest.device, &manifest.reserves, root)
        .map_err(|error| error.to_string())?;
    let topology = catalog.topology();
    let records = magnitude_executor::KernelCache::measurement_records(root);
    let requests = PointRequests::new(root);
    let outstanding = || requests.outstanding(&topology, &selected.info, &records);
    let mut left = outstanding()?;
    let mut job = 0;
    while !left.is_empty() {
        job += 1;
        let request = MeasureRequest {
            paths: paths.clone(),
            kernels: root.clone(),
            executor: selected.info.selector(),
            job: MeasureJob(job),
            time: std::time::Duration::from_secs_f64(MEASUREMENT_SECONDS),
        };
        let mut recorded = false;
        let end = measure_machine(&request, &mut |event| recorded |= matches!(event, MeasureEvent::Recorded(..)));
        drop(end.held);
        let after = outstanding()?;
        if !recorded && after == left {
            for failure in &end.failures {
                eprintln!("magnitude-harness: not measured: {failure}");
            }
            eprintln!("magnitude-harness: the job left {} requests outstanding: {after:?}", after.len());
            break;
        }
        left = after;
    }
    Ok(())
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
    /// Where a streamed load's experts are stored in the served file
    /// ([`expert_ranges`]); none of a load that holds its experts whole.
    expert_ranges: Vec<FileRange>,
    /// The load's memory between steps, which the scheduler's one memory
    /// step owns ([`magnitude_scheduler::owner::between_flights`]), on the
    /// clock from its open.
    memory: BetweenFlights,
    opened: std::time::Instant,
}

/// What the scheduler's memory step runs on below it: a domain, its memory
/// between steps and the clock then, and no request the step would touch:
/// no wait retries on an availability change, no prefix is retained, no
/// request is preempted or refused, and an unload ends the run.
struct Unrequested<'a, F: ProgramFamily> {
    domain: &'a mut magnitude_executor::ExecutorDomain<F>,
    memory: &'a mut BetweenFlights,
    now: u64,
}

impl<F: ProgramFamily> MemoryHost for Unrequested<'_, F> {
    type Family = F;

    fn domain(&mut self) -> &mut magnitude_executor::ExecutorDomain<F> {
        self.domain
    }

    fn memory(&mut self) -> &mut BetweenFlights {
        self.memory
    }

    fn now(&self) -> u64 {
        self.now
    }

    /// The harness drives each step to its end itself: a resize's wake
    /// needs no waiter, as the next step's boundary advances it.
    fn completion(&self) -> magnitude_executor::CompletionWake {
        magnitude_executor::CompletionWake::new(|| {})
    }

    fn advance_epoch(&mut self) {}

    fn held_prefixes(&self) -> BTreeSet<u64> {
        BTreeSet::new()
    }

    fn evict_prefix(&mut self, _: &BTreeSet<u64>) -> bool {
        false
    }

    fn evict_victims(&mut self, _: &[RequestId], _: bool) -> Result<bool, DomainError> {
        Ok(false)
    }

    fn refuse_memory_waiters(&mut self) {}

    fn unload(&mut self) {}
}

/// The boundary after a step, as the scheduler's owner runs it after
/// reconciling a flight: its one memory step at `now` on the service clock
/// (memory observed, the expert bank's resize advanced, shrinks and
/// releases while memory is short, a growth step where owed), on the
/// bindings the step returned.
fn between_steps<F: ProgramFamily>(
    domain: &mut magnitude_executor::ExecutorDomain<F>,
    memory: &mut BetweenFlights,
    bindings: &mut StateBindings,
    now: u64,
) -> Result<(), String> {
    let mut host = Unrequested { domain, memory, now };
    match between_flights(&mut host, Some(bindings)).map_err(|error| error.to_string())? {
        true => Err("memory stayed below the reclaim reserve: the model would unload".into()),
        false => Ok(()),
    }
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
    /// When its outcomes were reconciled, before the work between steps.
    pub reconciled: f64,
    pub end: f64,
    pub submissions: Vec<TracedSubmission>,
}

/// A domain error as the harness reports it. A claim refused in the Reclaim
/// band is also written as a memory record (`admission refused`) with the
/// reading it was judged on, which `mrig` reports as the cell's outcome.
pub fn text(error: DomainError) -> String {
    if let DomainError::Reclaim(reading) = &error {
        eprintln!(
            "magnitude-harness: memory {}",
            serde_json::json!({
                "moment": "admission refused",
                "store": reading.identity.to_string(),
                "constraint": reading.constraint.to_string(),
                "headroom_bytes": reading.headroom_bytes,
                "planning_bytes": reading.thresholds.planning_bytes,
                "distress": reading.distress.map(|distress| distress.to_string()),
                "wired_bytes": reading.held.map(|held| held.wired_bytes),
                "compressed_bytes": reading.held.map(|held| held.compressed_bytes),
                // Most recent first, each window's pages.
                "windows": reading.displacement.as_ref().map(|displacement| {
                    displacement
                        .windows
                        .iter()
                        .map(|window| {
                            serde_json::json!({
                                "ms": window.duration.as_secs_f64() * 1e3,
                                "compressed_pages": window.compressed_pages,
                                "decompressed_pages": window.decompressed_pages,
                                "swapped_out_pages": window.swapped_out_pages,
                                "swapped_in_pages": window.swapped_in_pages,
                            })
                        })
                        .collect::<Vec<_>>()
                }),
            })
        );
    }
    error.to_string()
}

impl Engine {
    /// What the device's graphs cost serving so far
    /// (`ExecutorDomain::graph_replays`).
    pub fn graph_replays(&self) -> seismic::GraphReplays {
        self.domain.graph_replays()
    }

    pub fn open(
        artifacts: &Artifacts,
        options: &EngineOptions,
        load: &Load,
    ) -> Result<Self, String> {
        let resolved = configuration(artifacts, options, load)?;
        settle_measurements(artifacts, &resolved)?;
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
            worker(&options.worker_arguments)?,
            |_| {},
        )
        .map_err(|error| error.to_string())?;
        let device = domain.resources().device().clone();
        let expert_ranges = expert_ranges(domain.expert_bank_plan(), resolved.host.package().target())?;
        Ok(Self {
            domain,
            bindings: Some(bindings),
            device,
            vocabulary,
            method,
            next_request: 1,
            lookahead: load.lookahead,
            expert_ranges,
            memory: BetweenFlights::new(),
            opened: std::time::Instant::now(),
        })
    }

    /// How many pages of the load's expert stacks the file cache holds now,
    /// of how many (the OS's own answer, `mincore`): a streamed load reads
    /// its experts past the cache, so any it holds served a read from it.
    pub fn cached_expert_pages(&self) -> Result<(u64, u64), String> {
        cached_pages(&self.expert_ranges)
    }

    pub fn device(&self) -> &seismic::Device {
        &self.device
    }

    /// [`between_steps`] after a step, on the clock from the load's open.
    pub(crate) fn between_steps(&mut self) -> Result<(), String> {
        let now = u64::try_from(self.opened.elapsed().as_nanos()).map_err(|_| "the run outlasts the clock")?;
        let bindings = self.bindings.as_mut().expect("no flight holds the state bindings");
        between_steps(&mut self.domain, &mut self.memory, bindings, now)
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

    /// [`Self::forward`] at `position`.
    fn forward_at(
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

    /// Plan the prompt chunks after the sequence's next one as the owner
    /// does before submitting a lone prompt chunk
    /// (`ExecutorDomain::plan_prompt`, no decode waiting here), the
    /// prompt's grouping (`InputLayout::group_chunks`) forming them. `rest`
    /// is the prompt from the sequence's position; its last chunk selects
    /// when `select`. Nothing is planned while a chain is queued: its chunks
    /// are claimed.
    pub fn plan_prompt(
        &mut self,
        sequence: &Sequence,
        rest: &[TokenId],
        select: bool,
    ) -> Result<(), String> {
        if self.domain.chain_holder().is_some() {
            return Ok(());
        }
        self.domain.plan_prompt(None, true, |chunks| {
            let grouped = InputLayout::new(rest.len(), Vec::new())?
                .group_chunks(PREFILL_ROWS.min(rest.len()), PREFILL_ROWS, None, chunks)?;
            Ok(grouped
                .into_iter()
                .map(|chunk| {
                    let demand = match select && chunk.end == rest.len() {
                        true => Demand::SELECT,
                        false => Demand::NONE,
                    };
                    let position = sequence.position + chunk.start;
                    let tokens = rest[chunk].to_vec();
                    Self::forward_at(sequence, position, WorkKind::Prefill, tokens, demand)
                })
                .collect())
        })
    }

    /// Submit one group of target forwards, wait, commit every row and
    /// advance the sequences. Returns the step's timings (submissions from
    /// `trace`, when given) and the outcomes in operation order.
    /// A step a late pulse interrupted is aborted and run again from its
    /// accepted state, as the scheduler's rounds are.
    pub fn step(
        &mut self,
        operations: Vec<Operation>,
        sequences: &mut [&mut Sequence],
        trace: Option<&SubmissionTrace>,
    ) -> Result<(Step, Vec<Outcome>), String> {
        loop {
            if let Some(step) = self.attempt(operations.clone(), sequences, trace)? {
                return Ok(step);
            }
        }
    }

    /// One submission of `step`'s forwards: none where the flight was
    /// interrupted, its outcomes aborted.
    fn attempt(
        &mut self,
        operations: Vec<Operation>,
        sequences: &mut [&mut Sequence],
        trace: Option<&SubmissionTrace>,
    ) -> Result<Option<(Step, Vec<Outcome>)>, String> {
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
        // A flight is interrupted whole, and a forward step's state stays
        // resident: only a prefill chain's chunk releases it.
        if pending.iter().any(|pending| matches!(pending.outcome(), Outcome::Interrupted { .. })) {
            for pending in pending {
                if !matches!(pending.outcome(), Outcome::Interrupted { resident: true, .. }) {
                    return Err("a step's flight was interrupted in part, or released its state".into());
                }
                self.domain.abort(pending).map_err(text)?;
            }
            return Ok(None);
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
        let reconciled = host_seconds();
        self.between_steps()?;
        let end = host_seconds();
        let submissions = trace
            .map(|trace| trace.collect().map_err(|error| error.to_string()))
            .transpose()?
            .unwrap_or_default();
        Ok(Some((
            Step {
                start,
                submitted,
                finished,
                reconciled,
                end,
                submissions,
            },
            outcomes,
        )))
    }

    /// Accept `tokens` from the sequence's position in prefill chunks, as the
    /// service runs a prompt. The last chunk selects when `select`.
    pub fn prefill(
        &mut self,
        sequence: &mut Sequence,
        tokens: &[TokenId],
        select: bool,
    ) -> Result<(), String> {
        let chunks = tokens.chunks(PREFILL_ROWS).collect::<Vec<_>>();
        for (index, chunk) in chunks.iter().enumerate() {
            let demand = if select && index + 1 == chunks.len() {
                Demand::SELECT
            } else {
                Demand::NONE
            };
            self.plan_prompt(sequence, &tokens[index * PREFILL_ROWS..], select)?;
            let operation = Self::forward(sequence, WorkKind::Prefill, chunk.to_vec(), demand);
            self.step(vec![operation], &mut [&mut *sequence], None)?;
        }
        Ok(())
    }
}

/// A range of a file: its path, and the offset and bytes of the range.
pub type FileRange = (PathBuf, u64, u64);

/// The distinct ranges of the served file a streamed load's experts are
/// stored in under its bank `bank` (none for a load that streams nothing),
/// in order: two stacks of row ranges of one stored tensor lie in one.
pub fn expert_ranges(
    bank: Option<&magnitude_executor::ExpertBankPlan>,
    artifact: &magnitude_artifacts::gguf::GgufArtifact,
) -> Result<Vec<FileRange>, String> {
    let mut ranges = std::collections::BTreeSet::new();
    for layer in bank.map(|bank| bank.stored_stacks(artifact)).transpose()?.into_iter().flatten() {
        for stack in layer {
            let bytes = stack.bytes();
            ranges.insert((stack.path, stack.offset, bytes));
        }
    }
    Ok(ranges.into_iter().collect())
}

/// The pages of `ranges` the file cache holds, and all their pages.
pub fn cached_pages(ranges: &[FileRange]) -> Result<(u64, u64), String> {
    ranges.iter().try_fold((0, 0), |(cached, pages), (path, offset, bytes)| {
        let (held, all) = magnitude_artifacts::cached_pages(path, *offset, *bytes)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Ok((cached + held, pages + all))
    })
}

pub fn tokens(ids: &[u32]) -> Vec<TokenId> {
    ids.iter().copied().map(TokenId).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::fixtures::{bank_backends, banked, host_tier, misses_at, record_misses};

    /// A capped load on a discrete-shaped device, its bank whole on the
    /// device and a host tier beside it, grows that tier across the
    /// harness's step boundaries as the owner's do: once memory has stayed
    /// free, one step to every expert (the tier is the inclusive copy every
    /// miss reaches, priced by no record), and no further. On every backend
    /// discovery reports available.
    #[test]
    fn a_capped_load_grows_its_host_tier_across_step_boundaries() {
        for backend in bank_backends() {
            let (mut domain, mut bindings) = banked(backend, 2, 2);
            host_tier(&mut domain, 4);
            record_misses(&mut domain, misses_at(4, 100, 4));
            domain.measure_experts(magnitude_executor::ExpertMeasures {
                lines: magnitude_executor::assessment::Priced::Measured(std::sync::Arc::from([])),
                store: None,
                resident: magnitude_executor::assessment::Priced::Measured(Box::new(|_, _| Ok(0.015))),
                pass: magnitude_executor::assessment::Priced::Pending("unpriced".into()),
                moves: magnitude_executor::assessment::Priced::Pending("unmeasured".into()),
                tier_copy: magnitude_executor::assessment::Priced::Measured(None),
                convert: magnitude_executor::assessment::Priced::Measured(None),
            }).unwrap();
            let host_slots = |domain: &magnitude_executor::ExecutorDomain<_>| {
                domain.expert_residency().unwrap().unwrap().host.unwrap().slots
            };
            let mut memory = BetweenFlights::new();
            // A boundary every 200 ms of the service clock, past the memory
            // observation interval, for 12 s: past the 5 s memory must stay
            // free before the bank grows.
            let slots = (0..60u64)
                .map(|boundary| {
                    between_steps(&mut domain, &mut memory, &mut bindings, boundary * 200_000_000).unwrap();
                    host_slots(&domain)
                })
                .collect::<Vec<_>>();
            assert_eq!(slots[0], 0, "nothing grows before memory has stayed free");
            assert_eq!(slots.last(), Some(&4), "grown to every expert: {slots:?}");
            let grown = slots.windows(2).filter(|pair| pair[1] != pair[0]).count();
            assert_eq!(grown, 1, "one step: {slots:?}");
        }
    }
}
