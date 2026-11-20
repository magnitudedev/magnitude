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
use magnitude_executor::platform::{DeviceRequest, ReleaseRate};
use magnitude_executor::{
    ChainEntry, Demand, DomainError, DraftForm, ExecutionPath, ExecutorDomain, FeatureReader, FeatureRows, FeatureSpan, FrontGrant,
    FrontPass, HeadPhase, NumericalProfile, Operation, Outcome, PhysicalDecision, Priming, ProgramFamily, RequestId,
    ReservedResources, ResumeState, Sampling, SelectSpec, Selected, Shaping, StateBindings, TokenId, TuningSettings, WorkKind,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{RoundState, drafted_chain};
use magnitude_scheduler::owner::{BetweenFlights, MemoryHost, between_flights, relieve};
use magnitude_scheduler::ServiceLimits;
use magnitude_state::KvCodec;
use seismic::{SubmissionTrace, TracedSubmission, host_seconds};
use std::cell::Cell;
use std::path::PathBuf;

/// Rows of one prefill forward, as the service bounds them.
pub const PREFILL_ROWS: usize = magnitude_executor::EVALUATION_PREFILL_ROWS;

/// How every load of one run is made.
pub struct EngineOptions {
    pub cache_dir: Option<PathBuf>,
    pub device: DeviceRequest,
    /// The host's KV policy, when named (`ModelPolicy::kv_codec`).
    pub kv_codec: Option<KvCodec>,
    pub numerical_profile: Option<NumericalProfile>,
    /// The bound on each load's weights: a cap on every weight byte, or the
    /// device's room alone (`ModelPolicy::weight_bound`).
    pub weight_bound: Option<magnitude_executor::WeightBound>,
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
    /// How the load generates: `Plain` the target alone, else the engine's own selection of a drafter.
    pub method: ModelMethod,
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
    /// The load records its routes (`ModelPolicy::record_routes`): a
    /// streamed load, whose decode cells write them (`time --record-routes`).
    pub record_routes: bool,
}

/// Resolve the catalog model for `load`, as the engine's host would.
pub fn configuration(
    artifacts: &Artifacts,
    options: &EngineOptions,
    load: &Load,
) -> Result<ResolvedEngineConfiguration, String> {
    let separate = |named: bool| match (&artifacts.drafter, named) {
        (Some(drafter), _) => Ok(Some(drafter.path.clone())),
        (None, false) => Ok(None),
        (None, true) => Err(format!("{} declares no drafter", artifacts.model)),
    };
    // A separate draft is loaded where the method takes one: `Auto` takes the catalog's when it has one (else its
    // draft head), a method naming a separate draft's variant requires it, and the others take none.
    let draft = match (load.method, load.method.draft_variant()) {
        (ModelMethod::Auto, _) => separate(false)?,
        (_, Some(_)) => separate(true)?,
        (_, None) => None,
    };
    // The machine's figures live in the kernel cache, keyed by its host.
    let topology = match &options.cache_dir {
        Some(_) => Some(seismic::Devices::discover(&[]).map_err(|error| error.to_string())?.topology()),
        None => None,
    };
    let release = ReleaseRate::of_machine(options.cache_dir.as_deref().zip(topology.as_deref()));
    let mut resolved = EngineConfiguration {
        package: PackageOptions::single(artifacts.target.clone(), ProjectorSelection::Disabled, draft),
        model: ModelPolicy {
            draft_policy: None,
            method: load.method,
            mtp_proposals: load.proposals,
            kv_codec: options.kv_codec,
            lookahead: load.lookahead,
            exported_logits_rows: load.exported_logits_rows,
            numerical_profile: options.numerical_profile.clone(),
            weight_bound: options.weight_bound,
            record_routes: load.record_routes,
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
        release,
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

/// The paths a load of `artifacts` serves: the target's directory.
fn served_paths(artifacts: &Artifacts) -> Result<Vec<PathBuf>, String> {
    let directory = std::path::absolute(&artifacts.target)
        .map_err(|error| error.to_string())?
        .parent()
        .ok_or_else(|| format!("{} has no directory", artifacts.target.display()))?
        .to_path_buf();
    Ok(vec![directory])
}

/// The machine a load of `resolved` serves on, discovered with the paths
/// of `artifacts`, and the executor the load selects, by the records of the
/// kernel cache at `root`.
fn selected_machine(
    artifacts: &Artifacts,
    resolved: &magnitude_engine::composition::ResolvedEngineConfiguration,
    root: &std::path::Path,
) -> Result<(Vec<PathBuf>, seismic::Devices, magnitude_executor::platform::SelectedDevice), String> {
    let manifest = &resolved.manifest;
    let paths = served_paths(artifacts)?;
    let catalog = seismic::Devices::discover(&paths).map_err(|error| error.to_string())?;
    let selected = magnitude_executor::platform::select_recorded(&catalog, manifest.path, manifest.device, &manifest.release, Some(root))
        .map_err(|error| error.to_string())?;
    Ok((paths, catalog, selected))
}

/// Run the measurement job's probes of the device a load of `load` selects
/// (`worker::measure::probe_machine`), each in a probe process of this
/// executable, so the device's probed properties are measured before a
/// check reads them; reports each result and each probe that kept none.
/// Needs a kernel cache, whose records select the device; the results are
/// kept in the process's program store (`seismic::ProgramStore::user`).
pub fn probe(artifacts: &Artifacts, options: &EngineOptions, load: &Load) -> Result<(), String> {
    let resolved = configuration(artifacts, options, load)?;
    let root = resolved.manifest.kernel_cache.as_ref().ok_or("probing needs a kernel cache (--cache-dir)")?;
    let (paths, _, selected) = selected_machine(artifacts, &resolved, root)?;
    let run = magnitude_engine::worker::measure::probe_machine(
        &paths,
        selected.info.selector(),
        std::time::Duration::from_secs_f64(magnitude_executor::assessment::MEASUREMENT_SECONDS),
    )?;
    for kept in &run.kept {
        eprintln!("magnitude-harness: probe {}: {}\n{}", kept.probe, kept.result.line(), kept.report);
    }
    for failure in &run.failures {
        eprintln!("magnitude-harness: {failure}");
    }
    Ok(())
}

/// The model's Prepare alone, as a load that found tuning records missing
/// runs it: its search, in a worker process of this executable, publishing
/// records to the kernel cache, with nothing imported or loaded here.
pub fn prepare(artifacts: &Artifacts, options: &EngineOptions, load: &Load) -> Result<(), String> {
    use magnitude_engine::worker::Preparation;
    let resolved = configuration(artifacts, options, load)?;
    let command = worker(&options.worker_arguments)?;
    match magnitude_engine::composition::prepare_in_process(&resolved.manifest, command, |_| {})
        .map_err(|error| error.to_string())?
    {
        Preparation::Prepared | Preparation::Unneeded => Ok(()),
        Preparation::Unshared => Err("a Prepare needs a kernel cache (--cache-dir)".into()),
        Preparation::Incomplete(error) => Err(format!("the Prepare did not finish: {error}")),
    }
}

pub struct Engine {
    pub(crate) domain: Box<ExecutorDomain>,
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

    fn unload(&mut self) {}

    fn evict_owed_prefixes(&mut self) -> bool {
        false
    }

    fn prefix_value(&self) -> Option<(u64, u64)> {
        None
    }

    /// The run's sequences are its own: none is evicted, so no relief
    /// names one.
    fn evictable(&self) -> std::collections::BTreeSet<RequestId> {
        std::collections::BTreeSet::new()
    }

    fn preempt(&mut self, sequences: &[RequestId]) -> Result<(), magnitude_scheduler::publication::RequestError> {
        panic!("invariant violated: a relief names the harness's sequences {sequences:?}, none of which is evictable")
    }
}

/// The boundary after a step, as the scheduler's owner runs it after
/// reconciling a flight: its one memory step at `now` on the service clock
/// (memory read, the expert bank's resize advanced and its derived target
/// followed), on the bindings the step returned.
fn between_steps<F: ProgramFamily>(
    domain: &mut magnitude_executor::ExecutorDomain<F>,
    memory: &mut BetweenFlights,
    bindings: &mut StateBindings,
    now: u64,
) -> Result<(), String> {
    let mut host = Unrequested { domain, memory, now };
    match between_flights(&mut host, Some(bindings)).map_err(|error| error.to_string())? {
        true => Err("the memory guarantee could not be kept: the model would unload".into()),
        false => Ok(()),
    }
}

/// One open request and its accepted position.
pub struct Sequence {
    pub request: RequestId,
    pub position: usize,
    /// The token the last step selected.
    pub selected: Option<TokenId>,
    /// Its front-row decode: the chain its last step's outcome guessed, the
    /// next pass's front tokens ([`Engine::two_row_step`]).
    pub two_row: magnitude_executor::TwoRowDecode,
    /// The form and guessed rows of its next pass: its pricing's grant, as
    /// [`Engine::two_row_step`] takes it.
    pub grant: FrontGrant,
    /// Whether its steps enter the MTP head, which makes its chain: behind
    /// a front pass on the device (`magnitude_executor::ChainEntry`), or
    /// after a verify pass in a host head transaction ([`Trailing`]). A
    /// two-row sequence of a load that fronts, which opens with its whole
    /// prompt, closing it selecting ([`Engine::open_two_row_sequence`]).
    /// That closing chunk enters its selection in the head right past the
    /// prompt's, so the head's position is its own from then on; no other
    /// prefill follows.
    chains: bool,
    /// The pairs its head lacks, after a verify pass: none where the head
    /// entered every pair up to its selection.
    trailing: Option<Trailing>,
    /// The two-row pass of the step under way, whose round accepts its
    /// rows ([`Engine::two_row_step`]): none of any other step.
    pass: Option<RoundState>,
}

/// A verify pass's accepted rows as pairs the head enters: each committed
/// row's successor token with that row's feature, read to the host. The
/// sequence's next pass enters them first, in one head transaction drafting
/// that pass's chain from the last, as generation's does.
#[derive(Clone)]
struct Trailing {
    tokens: Vec<TokenId>,
    features: FeatureRows,
}

impl Sequence {
    /// Its next pass from its selection, in the form and as deep as its
    /// grant holds it, the round generation runs (`RoundState::pass`),
    /// which also accepts its rows; every row selects greedily. A chaining
    /// sequence's demands its rows' features: a front pass enters the head
    /// behind it from its position, and a verify pass's accepted pairs
    /// enter it after.
    pub fn two_row_round(&self) -> Result<RoundState, String> {
        let pending = self.selected.ok_or("a two-row decode follows a selection")?;
        let rows = self.two_row.pass(self.grant);
        let (demand, chain) = match (self.chains, self.grant) {
            (true, FrontGrant::Verify(_)) => (Demand::FEATURES, None),
            (true, FrontGrant::UpTo(_) | FrontGrant::Exactly(_)) => {
                (Demand::DEVICE_FEATURES, Some(ChainEntry { position: self.position }))
            }
            (false, _) => (Demand::NONE, None),
        };
        RoundState::pass(pending, rows, self.position, &[], None, Sampling::Greedy, Shaping::default(), 0, demand, chain)
    }
}

/// A sequence's accepted state and selection, which sequences open from
/// as it stood ([`Engine::open_from`]), as the service resumes a cached
/// prefix.
pub struct Checkpoint {
    state: ResumeState,
    selected: Option<TokenId>,
    two_row: magnitude_executor::TwoRowDecode,
    chains: bool,
    trailing: Option<Trailing>,
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

/// A domain error as the harness reports it.
pub fn text(error: DomainError) -> String {
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
        let vocabulary = usize::try_from(resolved.manifest.definition.decoder.vocabulary)
            .map_err(|_| "vocabulary exceeds the host domain")?;
        let method = resolved.manifest.model.method;
        let (domain, bindings, _) = load_in_process(
            &resolved.manifest,
            resolved.host.package(),
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
        Ok(Sequence {
            request: self.open_request(None)?,
            position: 0,
            selected: None,
            two_row: Default::default(),
            grant: FrontGrant::UpTo(0),
            chains: false,
            trailing: None,
            pass: None,
        })
    }

    /// A sequence of `prompt` that decodes in front passes
    /// ([`Self::two_row_step`]), chaining through the MTP head where the
    /// load fronts (`ExecutorDomain::fronts`): its prompt primes the head,
    /// its closing chunk selecting, and every pass enters it, as a
    /// generation of the MTP method does there.
    pub fn open_two_row_sequence(&mut self, prompt: &[TokenId]) -> Result<Sequence, String> {
        let mut sequence = Sequence { chains: self.domain.fronts(), ..self.open_sequence()? };
        self.run_prompt(&mut sequence, prompt, true)?;
        Ok(sequence)
    }

    /// A request opened with fresh state, or `from`'s.
    fn open_request(&mut self, from: Option<&ResumeState>) -> Result<RequestId, String> {
        let request = RequestId(self.next_request);
        self.next_request += 1;
        self.domain
            .install_input(request, PreparedModelInput::continuation_only())?;
        let bindings = self
            .bindings
            .as_mut()
            .expect("no flight holds the state bindings");
        // The harness runs no chain between its steps. A sequence opens
        // holding the rows it resumes and a launch's rows ahead, as the
        // service's requests do; its steps reserve ahead as they go
        // ([`Self::provision`]).
        let unchained = self.domain.unchained(bindings).expect("no chain is under way between the harness's steps");
        let rows = from.map_or(0, ResumeState::position) as u64;
        self.domain.open_state(unchained, request, from, rows).map_err(text)?;
        Ok(request)
    }

    /// Prefetch the stored-table rows the sequence reads as it feeds
    /// `tokens` from its position (`ExecutorDomain::prefetch_prompt`), and
    /// wait until the tables have taken them: what a step then reads after it
    /// is queued is what no prefetch held.
    pub fn prefetch(&self, sequence: &Sequence, tokens: &[TokenId]) -> Result<(), String> {
        for prefetched in self.domain.prefetch_prompt(sequence.request, sequence.position, tokens)? {
            prefetched.wait();
        }
        Ok(())
    }

    /// The sequence's accepted state, its head's among it, its selection
    /// and its chain ([`Checkpoint`]).
    pub fn checkpoint(&self, sequence: &Sequence) -> Result<Checkpoint, String> {
        Ok(Checkpoint {
            state: self.domain.resume_state(sequence.request)?,
            selected: sequence.selected,
            two_row: sequence.two_row.clone(),
            chains: sequence.chains,
            trailing: sequence.trailing.clone(),
        })
    }

    /// A sequence opened from `checkpoint`: its state, position, selection
    /// and chain as the checkpoint's sequence stood.
    pub fn open_from(&mut self, checkpoint: &Checkpoint) -> Result<Sequence, String> {
        Ok(Sequence {
            request: self.open_request(Some(&checkpoint.state))?,
            position: checkpoint.state.position(),
            selected: checkpoint.selected,
            two_row: checkpoint.two_row.clone(),
            grant: FrontGrant::UpTo(0),
            chains: checkpoint.chains,
            trailing: checkpoint.trailing.clone(),
            pass: None,
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
            front: Vec::new(),
            chain: None,
        }
    }

    /// A prompt chunk of `tokens` at `position`, the chunk closing the
    /// prompt (`closing`) selecting greedily from its last row. A chaining
    /// sequence's ([`Sequence::chains`]) primes the head across its
    /// boundary, and the closing chunk enters its selection right past that
    /// entry, as generation's chunks do.
    fn prompt_chunk(sequence: &Sequence, position: usize, tokens: Vec<TokenId>, closing: bool) -> Operation {
        let demand = match closing {
            true => Demand::SELECT,
            false => Demand::NONE,
        };
        let mut operation = Self::forward_at(sequence, position, WorkKind::Prefill, tokens, demand);
        if sequence.chains {
            let Operation::Forward { tokens, demand, prime, chain, .. } = &mut operation else {
                unreachable!("a prompt chunk is a forward");
            };
            let entry = Priming {
                boundary: true,
                tokens: match position {
                    0 => tokens[1..].to_vec(),
                    _ => tokens.clone(),
                },
                position: position.saturating_sub(1),
                draft_from: position + tokens.len(),
            };
            *chain = closing.then(|| ChainEntry { position: entry.position + entry.tokens.len() });
            *demand |= Demand::FEATURES;
            *prime = Some(entry);
        }
        operation
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
        self.domain.plan_prompt(None, true, true, |chunks| {
            let grouped = InputLayout::new(rest.len(), Vec::new())?
                .group_chunks(PREFILL_ROWS.min(rest.len()), PREFILL_ROWS, None, chunks)?;
            Ok(grouped
                .into_iter()
                .map(|chunk| {
                    let closing = select && chunk.end == rest.len();
                    Self::prompt_chunk(sequence, sequence.position + chunk.start, rest[chunk].to_vec(), closing)
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

    /// Provision `operations` as the service's owner does, so the step's
    /// reservation takes nothing more: a growth of a sequence's reservation
    /// short of memory is relieved through the scheduler's one relief path,
    /// taking cache alone (the run's sequences are never preempted), and
    /// waits at the next boundary for a shrink that relief owes it.
    pub(crate) fn provision(&mut self, bindings: &mut StateBindings, operations: &[Operation]) -> Result<(), String> {
        loop {
            let first = self.domain.provision(bindings, operations, None);
            let now = u64::try_from(self.opened.elapsed().as_nanos()).map_err(|_| "the run outlasts the clock")?;
            let mut host = Unrequested { domain: &mut self.domain, memory: &mut self.memory, now };
            let relieved = relieve(&mut host, bindings, first, false, |host, bindings| {
                Ok(host.domain().provision(bindings, operations, None))
            })
            .map_err(|error| error.to_string())?;
            match relieved.current {
                Ok(()) => return Ok(()),
                Err(magnitude_executor::DomainError::Capacity(_)) if relieved.pending => {
                    between_steps(&mut self.domain, &mut self.memory, bindings, now)?;
                }
                Err(error) => return Err(error.to_string()),
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
        self.provision(&mut bindings, &operations)?;
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
        // A flight is interrupted whole, and its states stay resident.
        if pending.iter().any(|pending| matches!(pending.outcome(), Outcome::Interrupted { .. })) {
            for pending in pending {
                if !matches!(pending.outcome(), Outcome::Interrupted { .. }) {
                    return Err("a step's flight was interrupted in part".into());
                }
                self.domain.abort(pending).map_err(text)?;
            }
            return Ok(None);
        }
        let mut outcomes = Vec::with_capacity(pending.len());
        for ((pending, sequence), operation) in pending.into_iter().zip(sequences.iter_mut()).zip(&operations) {
            if pending.request() != sequence.request {
                return Err("step outcome order differs from its sequences".into());
            }
            // A two-row pass accepts as its round does
            // (`Sequence::two_row_round`), its decision carrying the verdict
            // to the expert bank, and a chaining sequence's verify pass
            // leaves its accepted pairs trailing the head; any other forward
            // commits every row, selecting its last. The chain a forward's
            // outcome guessed advances the sequence's front-row decode.
            let decision = match (operation, sequence.pass.clone()) {
                (Operation::Forward { .. }, Some(round)) => {
                    let Outcome::Forward { rows, guesses } = pending.outcome() else {
                        return Err(format!("a two-row pass completed as {:?}", pending.outcome()));
                    };
                    let samples = round.forward().samples(rows)?;
                    let verifies = sequence.chains && round.forward().chain.is_none();
                    let acceptance = round.reconcile(&samples, &Default::default())?;
                    let next = *acceptance.emitted.last().expect("a pass emits its exact row's token");
                    if verifies {
                        // Each committed row's feature selected the token
                        // after it: the next accepted input, or the pass's
                        // successor after the last.
                        let committed = acceptance.committed_rows;
                        let features = rows
                            .iter()
                            .find_map(|row| row.features.clone())
                            .ok_or("a verify pass publishes its rows' features")?;
                        let features = self
                            .domain
                            .read(&FeatureSpan::new(features, 0, committed).map_err(|error| error.to_string())?)?;
                        let mut tokens = acceptance.inputs[1..committed].to_vec();
                        tokens.push(next);
                        sequence.trailing = Some(Trailing { tokens, features });
                    }
                    acceptance.advance(Some(&mut sequence.two_row), guesses.clone());
                    sequence.selected = Some(next);
                    acceptance.decision()
                }
                _ => {
                    if let Outcome::Forward { rows, guesses } = pending.outcome() {
                        sequence.selected = rows
                            .last()
                            .and_then(|row| row.selected)
                            .and_then(|selected| match selected {
                                Selected::Token(token) => Some(token),
                                _ => None,
                            });
                        sequence.two_row.guessed(guesses.clone());
                    }
                    PhysicalDecision::rows(pending.rows())
                }
            };
            outcomes.push(pending.outcome().clone());
            let rows = decision.accepted_rows;
            self.domain.reconcile(pending, decision).map_err(text)?;
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

    /// Decode `sequence` from its selection on in one pass
    /// (`magnitude_executor::TwoRowDecode`) in the form and at the depth its
    /// pricing grants (`ExecutorDomain::front_pass`, of one sequence: the
    /// queued pass's where one is queued, else the regime's; none before
    /// the regime is measured, or of a load whose bank is whole), every row
    /// selecting greedily. A front pass emits its exact row's token, and
    /// the pass after it is queued on the device before this one finishes,
    /// its tokens this pass's selections
    /// (`ExecutorDomain::queue_successor`), and the next call claims it, as
    /// plain decode's successor is. A verify pass emits its exact row's
    /// token and those of the guesses that stood, and is never queued
    /// behind: the next call first enters its accepted pairs in a head
    /// transaction drafting that pass's chain, as generation's next round
    /// does.
    pub fn two_row_step(&mut self, sequence: &mut Sequence, trace: Option<&SubmissionTrace>) -> Result<Step, String> {
        let request = [sequence.request];
        sequence.grant = match self.domain.front_pass(&request, &request).map_err(|error| error.to_string())? {
            // A claim's requests are exactly the launch's.
            FrontPass::Claim(claim) => FrontGrant::Exactly(claim[&sequence.request]),
            FrontPass::Built { form, depths } => form.grant(depths.get(&sequence.request).copied().unwrap_or(0)),
        };
        if let Some(Trailing { tokens, features }) = sequence.trailing.take() {
            // The head entered every pair before the pass's; its
            // proposals, at most its width, are the pass's chain.
            let entered = tokens.len();
            let drafts = sequence.grant.depth().min(self.method.proposals());
            let operation = Operation::Head {
                request: sequence.request,
                phase: HeadPhase::Generation,
                tokens,
                conditioning: features,
                position: sequence.position - entered,
                proposals: (0..drafts).map(|proposal| Self::greedy(sequence.position + proposal)).collect(),
                form: DraftForm::Chained,
            };
            let Outcome::Head { proposals, .. } = self.head(operation, entered)? else {
                return Err("a head transaction returns proposals".into());
            };
            sequence.two_row.guessed(drafted_chain(&proposals)?);
        }
        let round = sequence.two_row_round()?;
        let operation = round.forward().clone().into_operation(sequence.request, sequence.position, None)?;
        sequence.pass = Some(round);
        let step = self.step(vec![operation], &mut [&mut *sequence], trace);
        sequence.pass = None;
        step.map(|(step, _)| step)
    }

    /// Submit one head operation and reconcile `accepted` of its entry
    /// rows, as the scheduler reconciles a head transaction; its outcome.
    pub(crate) fn head(&mut self, operation: Operation, accepted: usize) -> Result<Outcome, String> {
        let operations = [operation];
        let mut bindings = self.bindings();
        self.provision(&mut bindings, &operations)?;
        let ReservedResources::Head(workspace, output, advances) = self
            .domain
            .reserve(&mut bindings, &operations)
            .map_err(text)?
            .into_resources()
        else {
            return Err("a head operation runs on the head lane".into());
        };
        let flight = self
            .domain
            .submit_head(bindings, &operations, workspace, output, advances)
            .map_err(|failure| failure.error().to_string())?;
        let (pending, bindings) = self.domain.finish_head(flight).map_err(text)?;
        self.bindings = Some(bindings);
        let mut outcome = None;
        for pending in pending {
            outcome = Some(pending.outcome().clone());
            self.domain.reconcile(pending, PhysicalDecision::rows(accepted)).map_err(text)?;
        }
        self.between_steps()?;
        outcome.ok_or_else(|| "a head operation returned no outcome".into())
    }

    /// Accept `tokens` from the sequence's position in prefill chunks, as the
    /// service runs a prompt. The last chunk selects when `select`. A
    /// chaining sequence's prompt is its opening's alone.
    pub fn prefill(
        &mut self,
        sequence: &mut Sequence,
        tokens: &[TokenId],
        select: bool,
    ) -> Result<(), String> {
        assert!(!sequence.chains, "invariant violated: a chaining sequence prefills only the prompt it opens with");
        self.run_prompt(sequence, tokens, select)
    }

    /// [`Self::prefill`], of any sequence.
    fn run_prompt(&mut self, sequence: &mut Sequence, tokens: &[TokenId], select: bool) -> Result<(), String> {
        let chunks = tokens.chunks(PREFILL_ROWS).collect::<Vec<_>>();
        for (index, chunk) in chunks.iter().enumerate() {
            let closing = select && index + 1 == chunks.len();
            self.plan_prompt(sequence, &tokens[index * PREFILL_ROWS..], select)?;
            let operation = Self::prompt_chunk(sequence, sequence.position, chunk.to_vec(), closing);
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
            ranges.extend(stack.file_ranges());
        }
    }
    Ok(ranges.into_iter().collect())
}

/// The pages of `ranges` the file cache holds, and all their pages.
pub fn cached_pages(ranges: &[FileRange]) -> Result<(u64, u64), String> {
    ranges.iter().try_fold((0, 0), |(cached, pages), (path, offset, bytes)| {
        let (held, all) = seismic_io::cached_pages(path, *offset, *bytes)
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
    use magnitude_executor::fixtures::{bank_backends, banked, count_decode, host_tier, measure_load, misses_at};

    /// A capped load on a discrete-shaped device, its bank whole on the
    /// device and a host tier beside it, grows that tier across the
    /// harness's step boundaries as the owner's do: the ledger derives the
    /// tier's target from the reading and the first boundary takes one step
    /// to every expert (the tier is the inclusive copy every miss reaches,
    /// priced by no record), and no further. No timer holds the growth back:
    /// room the reading shows is room it takes. On every backend discovery
    /// reports available.
    #[test]
    fn a_capped_load_grows_its_host_tier_across_step_boundaries() {
        for backend in bank_backends() {
            let (mut domain, mut bindings) = banked(backend, 2, 2);
            host_tier(&mut domain, 4);
            let decode = misses_at(4, 100);
            count_decode(&mut domain, decode);
            measure_load(&mut domain, &mut bindings);
            let host_slots = |domain: &magnitude_executor::ExecutorDomain<_>| {
                domain.expert_residency().unwrap().unwrap().host.unwrap().slots
            };
            let mut memory = BetweenFlights::new();
            // A boundary every 200 ms of the service clock, past the memory
            // observation interval, for 12 s.
            let slots = std::iter::once(host_slots(&domain))
                .chain((0..60u64).map(|boundary| {
                    between_steps(&mut domain, &mut memory, &mut bindings, boundary * 200_000_000).unwrap();
                    host_slots(&domain)
                }))
                .collect::<Vec<_>>();
            assert_eq!(slots[1], 4, "the first boundary grows to every expert: {slots:?}");
            assert_eq!(slots.last(), Some(&4), "grown to every expert: {slots:?}");
            let grown = slots.windows(2).filter(|pair| pair[1] != pair[0]).count();
            assert_eq!(grown, 1, "one step: {slots:?}");
        }
    }
}
