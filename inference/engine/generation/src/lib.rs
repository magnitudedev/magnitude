//! Family-neutral request-local progress. The executor prepares numerical rows;
//! generation accepts them independently after shared physical completion.
mod acceptance;
mod controls;
mod decode_pass_plan;
mod dflash;
mod draft_budget;
mod draft_control;
mod draft_transition;
mod front;
mod method;
mod mtp;
mod plain;
mod round;
mod shaping;
pub use acceptance::accept_prefix;
pub use controls::{EndOfGeneration, ReasoningBudget};
pub use decode_pass_plan::{
    AcceptanceProfile, BatchPlanCandidate, BranchPrediction, ComputePricing, DecodePassPlan,
    DecodePassScorer, DraftBudgetPlan, DrafterSpec, FrontCreditModel, HorizonLimits,
    PassCommitment, PassCostModel, PassId, PassOutcome, PassOutcomeTracker, PassPricing,
    PredictionJudgment, PredictionRole, RowMissPricing, SealedBatch, seal_batch_plans,
};
pub use dflash::DFlash;
pub use draft_budget::{DraftBudget, DraftPolicy};
pub use draft_control::{
    DraftController, DraftLoad, DraftOutlook, RoundChoice, TimedLaunch, share_rows,
};
pub use draft_transition::{
    CATCH_UP_ROWS, DraftHeadCall, DraftHistory, DraftRoundState, DraftTargetTransition,
};
pub use magnitude_artifacts::{BoundaryRule, InputLayout, InputSpan, PromptCoordinates};
pub use magnitude_executor::{
    Demand, FeatureReader, FeatureRef, FrontGrant, LaunchDuration, Operation, Priming, RequestId, Sampling,
    SelectSpec, Shaping, TokenId, WorkKind,
};
use magnitude_executor::{AheadVerdict, ChainEntry, Guess, GuessKind, PhysicalDecision, Rows, StreamedVerifyPrior, TwoRowDecode};
pub use method::{
    CheckpointCompatibility, DraftCheckpoint, FrontEntry, Method, MethodCheckpoint, MethodCheckpointError,
    MethodChoice, MethodEffects, MethodRequirements, MethodState, Propose, Verification,
};
pub use mtp::{drafted_chain, Mtp};
pub use plain::Plain;
pub use round::{MethodUpdate, RoundAcceptance, RoundForward, RoundState};
pub use shaping::{selection_history, verification_selects, HISTORY_WIDTH};
use std::{
    collections::{BTreeSet, VecDeque},
    sync::{Arc, Mutex},
};

/// Whether another request may reuse a generation's intermediate prefixes,
/// and with whom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrefixReuse {
    /// No other request reuses them.
    Transient,
    /// Requests of the same `partition` reuse them: with `None`, every
    /// unpartitioned request.
    Reusable { partition: Option<Arc<str>> },
}

/// Live runtime pricing observed from the executor domain for this generation.
#[derive(Default)]
struct ObservedPricing {
    streamed_prior: Option<StreamedVerifyPrior>,
    launch_prices: Option<magnitude_cost_model::LaunchPrices>,
}

/// Request-local grammar state. Staging returns an independent validated successor;
/// it cannot mutate the original state, and installing it cannot fail after commit.
pub trait Constraint: Send {
    fn position(&self) -> usize;
    /// Independent matcher at the same accepted position, including terminal state.
    fn fork(&self) -> Box<dyn Constraint>;
    fn stage(&self, tokens: &[TokenId]) -> Result<Box<dyn Constraint>, String>;
    fn forced(&self, limit: usize) -> Result<Vec<TokenId>, String>;
    fn mask(&self) -> Result<std::sync::Arc<[u32]>, String>;
}

/// Accepted token counts, including EOS and tokens suppressed by host stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftStats {
    pub proposed: usize,
    pub accepted: usize,
}

impl DraftStats {
    pub fn record(&mut self, proposed: usize, accepted: usize) -> Result<(), String> {
        if accepted > proposed {
            return Err("accepted draft count exceeds proposed draft count".into());
        }
        self.proposed = self
            .proposed
            .checked_add(proposed)
            .ok_or("draft proposal counter exhausted")?;
        self.accepted = self
            .accepted
            .checked_add(accepted)
            .ok_or("accepted draft counter exhausted")?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DetailedUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub draft_n: usize,
    pub draft_n_accepted: usize,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Options {
    pub max_tokens: usize,
    pub output_capacity: usize,
    pub context_limit: usize,
    pub vocabulary: usize,
    pub stop_tokens: BTreeSet<TokenId>,
    /// Tokens the model never selects (a tokenizer fact, such as a GGUF's
    /// `suppress_tokens`).
    pub suppressed_tokens: BTreeSet<TokenId>,
    pub sampling: Sampling,
    pub shaping: Shaping,
    pub seed: u64,
    pub forced_quantum: usize,
    pub method: MethodChoice,
    pub end_of_generation: EndOfGeneration,
    pub reasoning_budget: Option<ReasoningBudget>,
}

/// Device-free request intent. This is the only generation value allowed to
/// cross into the numerical worker; live method state is created there.
pub struct GenerationSeed {
    prompt: Vec<TokenId>,
    layout: InputLayout,
    coordinates: PromptCoordinates,
    options: Options,
    constraint: Option<Box<dyn Constraint>>,
}

impl GenerationSeed {
    pub fn new(
        prompt: Vec<TokenId>,
        layout: InputLayout,
        options: Options,
        constraint: Option<Box<dyn Constraint>>,
    ) -> Result<Self, String> {
        let coordinates = PromptCoordinates::linear(prompt.len())?;
        Self::with_coordinates(prompt, layout, coordinates, options, constraint)
    }

    pub fn with_coordinates(
        prompt: Vec<TokenId>,
        layout: InputLayout,
        coordinates: PromptCoordinates,
        options: Options,
        constraint: Option<Box<dyn Constraint>>,
    ) -> Result<Self, String> {
        validate_input(&prompt, &layout, &options, constraint.as_deref())?;
        if coordinates.len() != prompt.len() {
            return Err("prompt coordinates length must match prompt token count".into());
        }
        Ok(Self {
            prompt,
            layout,
            coordinates,
            options,
            constraint,
        })
    }

    pub const fn method(&self) -> MethodChoice {
        self.options.method
    }

    pub fn prompt(&self) -> &[TokenId] {
        &self.prompt
    }

    pub fn layout(&self) -> &InputLayout {
        &self.layout
    }

    pub fn coordinates(&self) -> &PromptCoordinates {
        &self.coordinates
    }

    pub fn constraint_position(&self) -> Option<usize> {
        self.constraint.as_ref().map(|value| value.position())
    }

    pub fn selection_mask(&self) -> Result<Option<std::sync::Arc<[u32]>>, String> {
        self.constraint
            .as_ref()
            .map(|value| value.mask())
            .transpose()
    }

    pub fn into_generation(self, method_factory: Arc<dyn Method>) -> Result<Generation, String> {
        Generation::from_seed(self, method_factory)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    Context,
    Cancelled,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalProgress {
    pub accepted_position: usize,
    pub resident_position: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingReconciliation {
    pub start: usize,
    pub end: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OutputToken {
    pub index: usize,
    pub token: TokenId,
}

/// The outcome of starting the next round. Method (drafter) work starts no
/// target round, so the generation comes back with it; callers start again
/// after reconciling that work.
pub enum RoundStart {
    Target(StartedRound),
    Method(Generation, Vec<Operation>),
}

/// The next target round of a resident generation, fixed before the method
/// proposes. `Generation::start_round` builds its round from this plan and
/// `Generation::publication_bound` bounds the round's publication from it, so
/// the two cannot disagree about the round's inputs.
enum RoundPlan {
    /// Replay of numerical rows through `end` held before eviction.
    Replay { end: usize },
    /// The prompt chunk ending at `end`.
    Prefill { end: usize },
    /// A constrained run accepted without selection.
    Forced(Vec<TokenId>),
    /// A decode or verification of at most `limit` proposals.
    Propose { limit: usize },
}

enum NextRound {
    Target(RoundState),
    Method(Vec<Operation>),
}

/// Every fallible logical change is performed against private state before
/// physical reconciliation. Dropping this value leaves the started round
/// untouched, including its generation's method state.
pub struct PreparedGenerationTransition {
    request: RequestId,
    decision: PhysicalDecision,
    commit: GenerationCommit,
}

/// What a prepared round transition commits into its generation, once the
/// executor reconciled its decision.
struct GenerationCommit {
    expected_resident_position: usize,
    expected_generated_len: usize,
    expected_finish: Option<FinishReason>,
    expected_output_len: usize,
    expected_published: usize,
    expected_forward: RoundForward,
    method: Box<dyn MethodState>,
    effects: MethodEffects,
    constraint: Option<Box<dyn Constraint>>,
    generated: Vec<TokenId>,
    output: VecDeque<OutputToken>,
    draft_stats: DraftStats,
    resident_position: usize,
    accepted_position: usize,
    finish: Option<FinishReason>,
    two_row: Option<TwoRowDecode>,
    settled_outcome: Option<PassOutcome>,
}

/// A method operation's outcome consumed against private method state.
pub struct PreparedMethodTransition {
    request: RequestId,
    expected_resident_position: usize,
    expected_generated_len: usize,
    expected_finish: Option<FinishReason>,
    method: Box<dyn MethodState>,
    /// The head entry rows it commits.
    accepted_rows: usize,
    /// Its generation's front-row decode after it: the chain a head
    /// transaction that drafted leaves the next pass.
    two_row: Option<TwoRowDecode>,
}

impl PreparedMethodTransition {
    /// The sole decision generation sends back to the executor for a head
    /// transaction: its entry rows to commit, guessing nothing.
    pub fn decision(&self) -> PhysicalDecision {
        PhysicalDecision::rows(self.accepted_rows)
    }
    pub const fn request(&self) -> RequestId {
        self.request
    }
}

impl PreparedGenerationTransition {
    /// The sole decision generation sends back to the executor: the target
    /// rows to commit and how the round's guessed rows fared
    /// (`RoundAcceptance::decision`). A verified prefix may need physical
    /// repair before the transition commits, but its accepted length cannot
    /// change during that repair. [`StartedRound::reconcile`] moves it into
    /// the executor's reconciliation.
    pub const fn decision(&self) -> &PhysicalDecision {
        &self.decision
    }
    pub const fn request(&self) -> RequestId {
        self.request
    }
}

pub struct Generation {
    prompt: Vec<TokenId>,
    layout: InputLayout,
    coordinates: PromptCoordinates,
    options: Options,
    constraint: Option<Box<dyn Constraint>>,
    generated: Vec<TokenId>,
    output: VecDeque<OutputToken>,
    published: usize,
    resident_position: usize,
    accepted_position: usize,
    reconciliation_target: usize,
    resident: bool,
    prefix_reuse: PrefixReuse,
    finish: Option<FinishReason>,
    method_factory: Arc<dyn Method>,
    method: Box<dyn MethodState>,
    cached_tokens: usize,
    draft_stats: DraftStats,
    /// Its front-row decode, where its method decodes in front passes
    /// (`Method::decodes_two_rows`): the chain the head made past the
    /// pending token, behind its last round on the device or in the host
    /// head transaction that drafted it, the next pass's guesses. A guess
    /// is only checked against the sequence's own tokens, so no state
    /// change invalidates it; a completed round that made none leaves none
    /// (`RoundAcceptance::advance`). Such a generation forces no token and
    /// proposes none: every decode round is a pass in the form its launch
    /// grants, a front pass entering the head behind it or a verify pass
    /// whose accepted pairs the next round's head transaction enters, and
    /// the chunk closing its prompt enters the head as a front pass does
    /// where the head enters its chunks on the device.
    two_row: Option<TwoRowDecode>,
    /// Monotonic request generation for [`PassId`] (incremented on eviction).
    pass_generation: u64,
    /// Monotonic per-request pass counter for [`PassId`].
    next_pass_id: u64,
    /// Single-owner single-settlement outcome tracker for decode passes.
    pass_outcomes: PassOutcomeTracker,
    /// Active in-flight sealed decode pass and its proposals (if any).
    active_pass: Option<(DecodePassPlan, Vec<TokenId>)>,
    /// Most recently settled decode pass outcome.
    last_pass_outcome: Option<PassOutcome>,
    /// Live runtime pricing observed from the executor domain.
    pricing: Mutex<ObservedPricing>,
}

/// A target round in progress. It owns its generation from start until the
/// round's reconciled transition commits, or until eviction, cancellation or
/// failure discards the round. Operations that require no round in progress
/// exist only on `Generation`.
pub struct StartedRound {
    generation: Generation,
    round: RoundState,
}

impl Generation {
    pub fn new(
        prompt: Vec<TokenId>,
        layout: InputLayout,
        options: Options,
        constraint: Option<Box<dyn Constraint>>,
    ) -> Result<Self, String> {
        GenerationSeed::new(prompt, layout, options, constraint)?.into_generation(Arc::new(Plain))
    }

    pub fn new_with_method(
        prompt: Vec<TokenId>,
        layout: InputLayout,
        options: Options,
        constraint: Option<Box<dyn Constraint>>,
        method_factory: Arc<dyn Method>,
    ) -> Result<Self, String> {
        GenerationSeed::new(prompt, layout, options, constraint)?.into_generation(method_factory)
    }

    fn from_seed(seed: GenerationSeed, method_factory: Arc<dyn Method>) -> Result<Self, String> {
        let GenerationSeed {
            prompt,
            layout,
            coordinates,
            mut options,
            constraint,
        } = seed;
        if method_factory.identity().is_empty()
            || !method_matches_choice(options.method, method_factory.as_ref())
        {
            return Err("generation method factory does not match request policy".into());
        }
        // Media rows enter the target from the projector under placeholder
        // tokens. A drafter reading the target's history sees them through
        // the target; one embedding every prompt token cannot enter them, so
        // such a request generates without drafting.
        let method_factory: Arc<dyn Method> = if method_factory.embeds_prompt_tokens()
            && layout.spans().iter().any(|span| !span.language_history)
        {
            options.method = MethodChoice::Plain;
            Arc::new(Plain)
        } else {
            method_factory
        };
        let finish = (options.max_tokens == 0).then_some(FinishReason::Length);
        let constraint = controls::Controlled::compose(
            constraint,
            options.end_of_generation,
            &options.stop_tokens,
            &options.suppressed_tokens,
            options.reasoning_budget.as_ref(),
            options.vocabulary,
        )?;
        let method = method_factory.create(None)?;
        let two_row = method_factory.decodes_two_rows().then(TwoRowDecode::default);
        Ok(Self {
            prompt,
            layout,
            coordinates,
            options,
            constraint,
            generated: Vec::new(),
            output: VecDeque::new(),
            published: 0,
            resident_position: 0,
            accepted_position: 0,
            reconciliation_target: 0,
            resident: false,
            prefix_reuse: PrefixReuse::Reusable { partition: None },
            finish,
            method_factory,
            method,
            cached_tokens: 0,
            draft_stats: DraftStats::default(),
            two_row,
            pass_generation: 0,
            next_pass_id: 0,
            pass_outcomes: PassOutcomeTracker::default(),
            active_pass: None,
            last_pass_outcome: None,
            pricing: Mutex::new(ObservedPricing::default()),
        })
    }
    pub fn prompt(&self) -> &[TokenId] {
        &self.prompt
    }
    pub fn layout(&self) -> &InputLayout {
        &self.layout
    }
    pub fn coordinates(&self) -> &PromptCoordinates {
        &self.coordinates
    }
    pub fn resident_position(&self) -> usize {
        self.resident_position
    }
    pub fn accepted_position(&self) -> usize {
        self.accepted_position
    }
    /// The numerical position the accepted path reconciles to: every accepted
    /// token except a successor sampled from the last context row, which no
    /// later row reads.
    pub fn reconciled_position(&self) -> usize {
        self.accepted_position.min(self.options.context_limit)
    }
    pub fn causal_progress(&self) -> CausalProgress {
        CausalProgress {
            accepted_position: self.accepted_position(),
            resident_position: self.resident_position(),
        }
    }
    pub fn generated(&self) -> &[TokenId] {
        &self.generated
    }
    pub fn usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.prompt.len(),
            completion_tokens: self.generated.len(),
        }
    }
    pub fn detailed_usage(&self) -> DetailedUsage {
        DetailedUsage {
            prompt_tokens: self.prompt.len(),
            completion_tokens: self.generated.len(),
            cached_tokens: self.cached_tokens,
            draft_n: self.draft_stats.proposed,
            draft_n_accepted: self.draft_stats.accepted,
        }
    }
    /// The method this request generates under: the load's, or plain for a
    /// request whose media rows its drafter cannot enter.
    pub fn method_choice(&self) -> MethodChoice {
        self.options.method
    }
    pub fn method_identity(&self) -> &str {
        self.method_factory.identity()
    }
    pub fn method_requirements(&self) -> MethodRequirements {
        self.method_factory.requires()
    }
    pub fn method_reclaimable(&self) -> u64 {
        self.method.reclaimable()
    }
    /// Declare before residency whether, and with whom, another request may
    /// reuse an intermediate prefix. Reusable histories must be complete at
    /// every boundary: a peer can introduce a branch point after this request
    /// has begun prefilling.
    pub fn set_prefix_reuse(&mut self, reuse: PrefixReuse) -> Result<(), String> {
        if self.resident || self.resident_position != 0 || !self.generated.is_empty() {
            return Err("prefix reuse must be declared before generation starts".into());
        }
        self.prefix_reuse = reuse;
        Ok(())
    }

    /// The partition this request shares prefixes within, if any.
    pub fn prefix_partition(&self) -> Option<&Arc<str>> {
        match &self.prefix_reuse {
            PrefixReuse::Reusable { partition } => partition.as_ref(),
            PrefixReuse::Transient => None,
        }
    }

    fn reusable_prefixes(&self) -> bool {
        matches!(self.prefix_reuse, PrefixReuse::Reusable { .. })
    }

    fn priming_horizon(&self, boundary: usize) -> usize {
        if self.reusable_prefixes() {
            boundary
        } else {
            self.prompt.len() + self.generated.len()
        }
    }
    /// Export owned method state for prefix retention.
    pub fn method_checkpoint(&self) -> Result<MethodCheckpoint, String> {
        if !self.reusable_prefixes() {
            return Err("request does not retain reusable prefixes".into());
        }
        if !self.resident {
            return Err("method checkpoint requires resident state".into());
        }
        self.method.checkpoint().map_err(|error| error.to_string())
    }

    /// Method state restored from a retained checkpoint of `position` rows,
    /// one this generation resumes from (`MethodCheckpoint::resumes`).
    fn method_at(
        &self,
        position: usize,
        checkpoint: &MethodCheckpoint,
    ) -> Result<Box<dyn MethodState>, String> {
        let successor = self.prompt.iter().chain(&self.generated).nth(position).copied();
        if !checkpoint.resumes(self.options.method, position, successor) {
            return Err("retained method checkpoint does not match the generation".into());
        }
        self.method_factory.create(Some(checkpoint))
    }
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish
    }
    pub fn output_len(&self) -> usize {
        self.output.len()
    }
    pub fn constraint_position(&self) -> Option<usize> {
        self.constraint.as_ref().map(|c| c.position())
    }
    /// Immutable request-local mask; host construction may overlap numerical work.
    /// The prepared executor binds this exact snapshot to device-side selection.
    pub fn selection_mask(&self) -> Result<Option<std::sync::Arc<[u32]>>, String> {
        self.constraint
            .as_ref()
            .map(|constraint| constraint.mask())
            .transpose()
    }

    /// Whether its next round is a pass in the form its launch grants
    /// (`magnitude_executor::FrontPass`): it decodes in front passes
    /// (`Method::decodes_two_rows`) and its next round is a decode. A front
    /// pass enters the head behind it (`Operation::Forward`'s `chain`) and
    /// a verify pass does not, and a launch's forwards chain all or none,
    /// so such rounds launch only beside each other, in one form.
    pub fn chains(&self) -> bool {
        self.two_row.is_some() && matches!(self.plan_round(1), Ok(RoundPlan::Propose { .. }))
    }

    /// The kind of the first guess its next pass runs where granted every
    /// guessed row it holds: the chain's first, or the head's where its
    /// head trails the target, so that the round drafts the chain first
    /// ([`MethodState::trails`]); none where it decodes in no front pass,
    /// where that round is no decode, or where no guessed row fits it.
    pub fn front_guess(&self) -> Option<GuessKind> {
        let two_row = self.two_row.as_ref()?;
        let RoundPlan::Propose { limit } = self.plan_round(1).ok()? else {
            return None;
        };
        let deepest = FrontGrant::UpTo(usize::MAX);
        match self.method.can_front() {
            true => (self.granted(limit, usize::MAX, deepest).min(self.method_factory.proposals()) > 0)
                .then_some(GuessKind::Head),
            false => Some(self.front_rows(two_row, usize::MAX, limit, deepest).guesses().first()?.kind),
        }
    }

    pub fn is_resident(&self) -> bool {
        self.resident
    }
    /// Start the next target round, which then owns this generation. Method
    /// work is returned as ordinary executor operations with the generation;
    /// callers start again after reconciling it. A refused start returns the
    /// generation with its error. `grant`: the form and guessed rows of a
    /// decode round of a method decoding in front passes (its launch grants
    /// them, `ExecutorDomain::front_pass`), as its rows hold.
    pub fn start_round(
        mut self,
        request: RequestId,
        allowance: usize,
        grant: FrontGrant,
    ) -> Result<RoundStart, (Self, String)> {
        match self.next_round(request, allowance, grant) {
            Ok(NextRound::Target(round)) => Ok(RoundStart::Target(StartedRound {
                generation: self,
                round,
            })),
            Ok(NextRound::Method(operations)) => Ok(RoundStart::Method(self, operations)),
            Err(error) => Err((self, error)),
        }
    }

    /// The most tokens the round `start_round(_, allowance)` would publish,
    /// computed without starting it, so publication permits can be reserved
    /// first. It is computed from the same plan `start_round` builds the round
    /// from, and is at least that round's `round_publication_bound()`:
    ///
    /// - replay publishes nothing;
    /// - a prompt chunk publishes one token when it ends the prompt (forced or
    ///   selected), otherwise none;
    /// - a forced run publishes exactly its forced tokens;
    /// - a decode or verification publishes at most `limit + 1` tokens, where
    ///   `limit` is the proposal width from the allowance, remaining
    ///   `max_tokens`, output credit, context limit and method proposals;
    ///   of a method decoding in front passes, the deepest verify pass
    ///   those allow, whatever form its launch grants (a front pass emits
    ///   its exact row's token alone).
    ///   The method may propose fewer, or return drafter work that starts no
    ///   target round and publishes nothing.
    ///
    /// It fails exactly when `start_round` would refuse before proposing.
    pub fn publication_bound(&self, allowance: usize) -> Result<usize, String> {
        Ok(match self.plan_round(allowance)? {
            RoundPlan::Replay { .. } => 0,
            RoundPlan::Prefill { end } => usize::from(end == self.prompt.len()),
            RoundPlan::Forced(tokens) => tokens.len(),
            RoundPlan::Propose { limit } => limit + 1,
        })
    }

    fn plan_round(&self, allowance: usize) -> Result<RoundPlan, String> {
        if allowance == 0 {
            return Err("service allowance must be positive".into());
        }
        if self.finish.is_some() {
            return Err("a finished generation cannot start a round".into());
        }
        if !self.resident {
            return Err("a non-resident generation cannot start a round".into());
        }
        if self.output.len() >= self.options.output_capacity {
            return Err("generation output is full".into());
        }
        let position = self.resident_position;
        if position < self.reconciliation_target {
            // A replay from inside the prompt ends at the prompt end: the executor admits
            // target rows only within, or wholly past, the prepared input.
            let target = if position < self.prompt.len() {
                self.reconciliation_target.min(self.prompt.len())
            } else {
                self.reconciliation_target
            };
            let end = self.layout.chunk_end(position, target, allowance)?;
            return Ok(RoundPlan::Replay { end });
        }
        if position < self.prompt.len() {
            let end = self
                .layout
                .chunk_end(position, self.prompt.len(), allowance)?;
            return Ok(RoundPlan::Prefill { end });
        }
        if self.generated.is_empty()
            || position != self.prompt.len() + self.generated.len() - 1
            || position >= self.options.context_limit
        {
            return Err("generation history and numerical continuation disagree".into());
        }
        let remaining = self.options.max_tokens - self.generated.len();
        let credit = self.options.output_capacity - self.output.len();
        let forced = self.forced_tokens(
            allowance
                .min(self.options.context_limit - position)
                .min(remaining)
                .min(credit),
        )?;
        if !forced.is_empty() {
            return Ok(RoundPlan::Forced(forced));
        }
        let limit = RoundState::proposal_limit(
            allowance.min(self.options.context_limit - position),
            remaining,
            credit,
            self.options.method == MethodChoice::Plain,
        );
        // A verify pass's acceptable rows are the chain's guesses, as deep
        // as its launch grants; a drafted proposal is the method's.
        let limit = match self.two_row {
            Some(_) => limit,
            None => limit.min(self.options.method.proposals()),
        };
        Ok(RoundPlan::Propose { limit })
    }

    fn next_round(&mut self, request: RequestId, allowance: usize, grant: FrontGrant) -> Result<NextRound, String> {
        let requirements = self.method_factory.requires();
        let round = match self.plan_round(allowance)? {
            RoundPlan::Replay { end } => {
                let position = self.resident_position;
                let tokens = self
                    .prompt
                    .iter()
                    .chain(&self.generated)
                    .skip(position)
                    .take(end - position)
                    .copied()
                    .collect();
                RoundState::progress(
                    WorkKind::Replay,
                    tokens,
                    position,
                    Vec::new(),
                    None,
                    None,
                    None,
                    requirements,
                )?
            }
            RoundPlan::Prefill { end } => {
                self.prefill_round(self.resident_position, end, self.method.priming_position())?
            }
            RoundPlan::Forced(forced) => {
                RoundState::forced(*self.generated.last().unwrap(), forced, requirements)?
            }
            RoundPlan::Propose { limit } => {
                let pending = *self.generated.last().unwrap();
                let remaining_after_exact = (self.options.max_tokens - self.generated.len()).saturating_sub(1);
                if let Some(two_row) = &self.two_row {
                    // Embedded `TwoRowDecode` supplies its existing chain via
                    // `DrafterSpec::from_two_row` (never forcing a new drafter).
                    let _drafter = DrafterSpec::from_two_row(two_row);
                    let pass_id = PassId {
                        request,
                        generation: self.pass_generation,
                        pass: self.next_pass_id,
                    };
                    let max_drafts = self
                        .granted(limit, allowance, grant)
                        .min(self.method_factory.proposals());
                    let drafts = match grant {
                        FrontGrant::Verify(_) | FrontGrant::Mixed { .. }
                            if self.method.trails() && max_drafts > 0 =>
                        {
                            let (costs, acceptance, has_real) =
                                self.pass_cost_and_acceptance(1 + max_drafts, max_drafts);
                            if has_real {
                                let (max_k, max_front) = match grant {
                                    FrontGrant::Mixed { verify, front } => {
                                        (verify.min(max_drafts), front.min(max_drafts))
                                    }
                                    _ => (max_drafts, 0),
                                };
                                DecodePassScorer::seal_unchained_plan_on_widths(
                                    pass_id,
                                    max_drafts,
                                    0.0,
                                    1 + max_drafts,
                                    0..=max_k,
                                    max_front,
                                    &acceptance,
                                    HorizonLimits {
                                        remaining_tokens: remaining_after_exact,
                                        eos_at: None,
                                    },
                                    &costs,
                                )
                                .map(|plan| (plan.verify + plan.front).min(max_drafts))
                                .unwrap_or(max_drafts)
                            } else {
                                max_drafts
                            }
                        }
                        _ => max_drafts,
                    };
                    let selects = self.proposal_selects(drafts)?;
                    let mut method = self.method.fork_transition();
                    let entry = method.front(request, pending, grant, &selects)?;
                    let (next, sealed_pass) = match entry {
                        FrontEntry::Pending(operations) => {
                            self.validate_method_operations(request, &operations)?;
                            (NextRound::Method(operations), None)
                        }
                        FrontEntry::Plain => {
                            let (rows, sealed_pass) = self.seal_front_pass(
                                pass_id,
                                two_row,
                                allowance,
                                limit,
                                grant,
                                remaining_after_exact,
                            )?;
                            let round = RoundState::pass(
                                pending,
                                rows,
                                self.generated.len(),
                                &self.generated,
                                self.constraint.as_deref(),
                                self.options.sampling,
                                self.options.shaping,
                                self.options.seed,
                                Demand::NONE,
                                None,
                            )?;
                            (NextRound::Target(round), Some(sealed_pass))
                        }
                        FrontEntry::Chain(chain) => {
                            let (rows, sealed_pass) = self.seal_front_pass(
                                pass_id,
                                two_row,
                                allowance,
                                limit,
                                grant,
                                remaining_after_exact,
                            )?;
                            let round = RoundState::pass(
                                pending,
                                rows,
                                self.generated.len(),
                                &self.generated,
                                self.constraint.as_deref(),
                                self.options.sampling,
                                self.options.shaping,
                                self.options.seed,
                                requirements.verify_demand.device_only_features(),
                                Some(chain),
                            )?;
                            (NextRound::Target(round), Some(sealed_pass))
                        }
                        FrontEntry::Verify => {
                            let (rows, sealed_pass) = self.seal_front_pass(
                                pass_id,
                                two_row,
                                allowance,
                                limit,
                                grant,
                                remaining_after_exact,
                            )?;
                            let round = RoundState::pass(
                                pending,
                                rows,
                                self.generated.len(),
                                &self.generated,
                                self.constraint.as_deref(),
                                self.options.sampling,
                                self.options.shaping,
                                self.options.seed,
                                requirements.verify_demand,
                                None,
                            )?;
                            (NextRound::Target(round), Some(sealed_pass))
                        }
                    };
                    self.method = method;
                    if let Some((plan, proposals)) = sealed_pass {
                        self.next_pass_id = self.next_pass_id.wrapping_add(1);
                        if let Some(settled) = self.pass_outcomes.feed(plan, proposals.clone()) {
                            self.last_pass_outcome = Some(settled);
                        } else {
                            self.active_pass = Some((plan, proposals));
                        }
                    }
                    return Ok(next);
                }
                let selects = self.proposal_selects(limit)?;
                let (proposal, front) = match self.method.propose(request, &selects) {
                    Propose::Tokens { acceptable, front } => (acceptable, front),
                    Propose::Pending(operations) if operations.is_empty() => {
                        return Err("method returned an empty pending operation set".into());
                    }
                    Propose::Pending(operations) => {
                        self.validate_method_operations(request, &operations)?;
                        return Ok(NextRound::Method(operations));
                    }
                };
                if proposal.len() > limit
                    || proposal.len() + front.len() > limit
                    || proposal
                        .iter()
                        .chain(front.iter().map(|guess| &guess.token))
                        .any(|token| token.0 as usize >= self.options.vocabulary)
                {
                    return Err("method returned an invalid proposal".into());
                }
                // A stop token is never accepted as a draft match, so rows
                // after it could only be rejected.
                let proposal_stopped = proposal
                    .iter()
                    .any(|token| self.options.stop_tokens.contains(token));
                let acceptable = proposal
                    .iter()
                    .take_while(|token| !self.options.stop_tokens.contains(token))
                    .copied()
                    .collect::<Vec<_>>();
                let front_tokens: Vec<TokenId> = if proposal_stopped {
                    Vec::new()
                } else {
                    front
                        .iter()
                        .map(|guess| guess.token)
                        .take_while(|token| !self.options.stop_tokens.contains(token))
                        .collect()
                };
                let mut all_tokens = acceptable.clone();
                all_tokens.extend_from_slice(&front_tokens);
                // Seal the decode pass through the single-owner `DecodePassScorer`
                // (reusing the real `DecodePassPlan` sealed by an adaptive drafter
                // when available, preserving measured costs, calibrated survival,
                // and any `F > 0` foresight tail) and register it in `PassOutcomeTracker`.
                let pass_id = PassId {
                    request,
                    generation: self.pass_generation,
                    pass: self.next_pass_id,
                };
                self.next_pass_id = self.next_pass_id.wrapping_add(1);
                let plan = if let Some(mut sealed) = self.method.pass_plan() {
                    let cap = 1 + limit;
                    let verify = sealed
                        .verify
                        .min(acceptable.len())
                        .min(limit)
                        .min(remaining_after_exact);
                    let front = if verify == 0 {
                        0
                    } else {
                        sealed
                            .front
                            .min(front_tokens.len())
                            .min(limit.saturating_sub(verify))
                            .min(magnitude_executor::CHAINED_ROWS.saturating_sub(1 + verify))
                    };
                    sealed.pass_id = pass_id;
                    sealed.generated = sealed.generated.max(all_tokens.len());
                    sealed.verify = verify;
                    sealed.front = front;
                    sealed.idle_front = 0;
                    sealed.reserved_capacity = cap;
                    sealed
                } else {
                    self.seal_verify_pass(
                        pass_id,
                        all_tokens.len(),
                        1 + limit,
                        remaining_after_exact,
                    )?
                };
                let rows = plan.build_rows(&all_tokens, GuessKind::Chained)?;
                // A round that verifies nothing publishes its feature only
                // where its method reads it.
                let verify_demand = match rows.acceptable().is_empty() && !self.method.taps() {
                    true => Demand::NONE,
                    false => requirements.verify_demand,
                };
                let round = RoundState::pass(
                    pending,
                    rows,
                    self.generated.len(),
                    &self.generated,
                    self.constraint.as_deref(),
                    self.options.sampling,
                    self.options.shaping,
                    self.options.seed,
                    verify_demand,
                    None,
                )?;
                if let Some(settled) = self.pass_outcomes.feed(plan, all_tokens.clone()) {
                    self.last_pass_outcome = Some(settled);
                } else {
                    self.active_pass = Some((plan, all_tokens));
                }
                round
            }
        };
        Ok(NextRound::Target(round))
    }

    /// The guessed rows `grant` gives a pass of this generation at most. A
    /// front pass the host builds runs as many as the rows `allowance` and
    /// the context leave past its exact row hold. A verify pass runs as
    /// many as its decode's `limit` holds ([`RoundPlan::Propose`]), every
    /// acceptable row a token it may emit. One claiming the pass queued
    /// behind its predecessor runs that pass's rows as they were queued,
    /// held to the context then
    /// (`magnitude_executor::ExecutorDomain::front_pass`): it is never
    /// reshaped.
    fn granted(&self, limit: usize, allowance: usize, grant: FrontGrant) -> usize {
        let context_extra = allowance
            .min(self.options.context_limit.saturating_sub(self.resident_position))
            .saturating_sub(1);
        match grant {
            FrontGrant::Plain => 0,
            FrontGrant::UpTo(depth) => depth.min(context_extra),
            FrontGrant::Exactly(depth) => depth,
            FrontGrant::Verify(depth) => depth.min(limit),
            FrontGrant::Mixed { verify, front } => {
                let k = verify.min(limit);
                let f = front.min(context_extra.saturating_sub(k));
                k + f
            }
        }
    }

    /// Build the single-owner [`PassCostModel`] and empirical [`AcceptanceProfile`]
    /// from observed runtime pricing (`StreamedVerifyPrior` / `LaunchPrices`) and
    /// settled [`PassOutcomeTracker`] conditional acceptance counts.
    fn pass_cost_and_acceptance(
        &self,
        max_rows: usize,
        proposal_len: usize,
    ) -> (PassCostModel, AcceptanceProfile, bool) {
        let observed = self
            .pricing
            .lock()
            .expect("generation pricing lock poisoned");
        let (costs, has_real_pricing) = if let Some(prior) = observed.streamed_prior.as_ref() {
            (PassCostModel::from_streamed_prior(prior), true)
        } else if let Some(prices) = observed.launch_prices.as_ref() {
            let depth = self.resident_position.max(1).next_power_of_two() as u32;
            match PassCostModel::from_launch_prices(prices, max_rows, depth) {
                Some(costs) => (costs, true),
                None => (PassCostModel::unit_compute(max_rows), false),
            }
        } else {
            (PassCostModel::unit_compute(max_rows), false)
        };
        let acceptance = AcceptanceProfile::from_empirical(
            proposal_len,
            observed
                .streamed_prior
                .as_ref()
                .map(|p| p.accepts.as_slice()),
            observed.streamed_prior.as_ref().and_then(|p| p.stand),
            &self.pass_outcomes,
        );
        (costs, acceptance, has_real_pricing)
    }

    /// Seal a verify pass through the single-owner [`DecodePassScorer`],
    /// using the real measured [`PassCostModel`] and empirical acceptance.
    fn seal_verify_pass(
        &self,
        pass_id: PassId,
        actual_generated: usize,
        reserved_capacity: usize,
        remaining_after_exact: usize,
    ) -> Result<DecodePassPlan, String> {
        let (costs, acceptance, has_real_pricing) =
            self.pass_cost_and_acceptance(reserved_capacity, actual_generated);
        let acceptance = if has_real_pricing {
            acceptance
        } else {
            AcceptanceProfile::Conditional(vec![1.0; actual_generated])
        };
        let fronts_target = self.method_factory.requires().head
            && self
                .pricing
                .lock()
                .expect("generation pricing lock poisoned")
                .streamed_prior
                .is_some();
        let max_front = if fronts_target { actual_generated } else { 0 };
        DecodePassScorer::seal_unchained_plan_on_widths(
            pass_id,
            actual_generated,
            0.0,
            reserved_capacity,
            0..=actual_generated,
            max_front,
            &acceptance,
            HorizonLimits {
                remaining_tokens: remaining_after_exact,
                eos_at: None,
            },
            &costs,
        )
    }

    /// Seal a `TwoRowDecode` pass through the single-owner [`DecodePassScorer`]
    /// using the real measured [`PassCostModel`] and empirical acceptance,
    /// preserving embedded chain semantics (never forcing a new drafter).
    fn seal_front_pass(
        &self,
        pass_id: PassId,
        two_row: &TwoRowDecode,
        allowance: usize,
        limit: usize,
        grant: FrontGrant,
        remaining_after_exact: usize,
    ) -> Result<(Rows, (DecodePassPlan, Vec<TokenId>)), String> {
        let raw_rows = self.front_rows(two_row, allowance, limit, grant);
        let reserved_capacity = raw_rows.row_count();
        match grant {
            FrontGrant::Verify(_) | FrontGrant::Mixed { .. } => {
                let raw_acceptable = raw_rows.acceptable();
                let raw_guesses = raw_rows.guesses();
                let candidate_guesses: Vec<Guess> = two_row
                    .chain()
                    .iter()
                    .take(raw_acceptable.len())
                    .copied()
                    .chain(raw_guesses.iter().copied())
                    .collect();
                let proposals: Vec<TokenId> = candidate_guesses.iter().map(|g| g.token).collect();
                let max_k = raw_acceptable.len();
                let max_front = match grant {
                    FrontGrant::Mixed { front, .. } => front.min(raw_guesses.len()),
                    _ => 0,
                };
                let (costs, acceptance, _has_real_pricing) =
                    self.pass_cost_and_acceptance(reserved_capacity, proposals.len());
                let mut plan = DecodePassScorer::seal_unchained_plan_on_widths(
                    pass_id,
                    proposals.len(),
                    0.0,
                    reserved_capacity,
                    0..=max_k,
                    max_front,
                    &acceptance,
                    HorizonLimits {
                        remaining_tokens: remaining_after_exact,
                        eos_at: None,
                    },
                    &costs,
                )?;
                plan.generated = two_row.chain_len().max(proposals.len());
                let acceptable = proposals[..plan.verify].to_vec();
                let front_guesses =
                    candidate_guesses[plan.verify..plan.verify + plan.front].to_vec();
                let rows = Rows::pass(acceptable, front_guesses, 0);
                Ok((rows, (plan, proposals)))
            }
            FrontGrant::Plain | FrontGrant::UpTo(_) | FrontGrant::Exactly(_) => {
                let front_guesses = raw_rows.guesses();
                let idle_front = raw_rows.front_rows().saturating_sub(front_guesses.len());
                let proposals: Vec<TokenId> = front_guesses.iter().map(|g| g.token).collect();
                let (costs, acceptance, _has_real_pricing) =
                    self.pass_cost_and_acceptance(reserved_capacity, proposals.len());
                let cum = acceptance.cumulative_survival();
                let pricing = DecodePassScorer::score_candidate_with_idle(
                    0.0,
                    0,
                    front_guesses.len(),
                    idle_front,
                    &cum,
                    HorizonLimits {
                        remaining_tokens: remaining_after_exact,
                        eos_at: None,
                    },
                    &costs,
                );
                let plan = DecodePassPlan {
                    pass_id,
                    generated: two_row.chain_len().max(proposals.len()),
                    verify: 0,
                    front: front_guesses.len(),
                    idle_front,
                    reserved_capacity,
                    pricing,
                };
                Ok((raw_rows, (plan, proposals)))
            }
        }
    }

    /// Single-owner decode pass outcome tracker for this generation.
    pub fn pass_outcomes(&self) -> &PassOutcomeTracker {
        &self.pass_outcomes
    }

    /// Most recently settled decode pass outcome, if any.
    pub fn last_pass_outcome(&self) -> Option<&PassOutcome> {
        self.last_pass_outcome.as_ref()
    }

    /// The rows of a pass of `two_row` where `grant`ed
    /// (`TwoRowDecode::pass`), held to what it gives at most
    /// ([`Self::granted`]): its exact row alone where none fits. A verify
    /// pass's acceptable rows end before a stop token, which is never
    /// accepted as a guess, so rows after it could only be rejected.
    fn front_rows(&self, two_row: &TwoRowDecode, allowance: usize, limit: usize, grant: FrontGrant) -> Rows {
        let depth = self.granted(limit, allowance, grant);
        match grant {
            FrontGrant::Plain => Rows::plain(),
            FrontGrant::UpTo(_) => two_row.pass(FrontGrant::UpTo(depth)),
            FrontGrant::Exactly(_) => two_row.pass(grant),
            FrontGrant::Verify(_) => {
                let pass = two_row.pass(FrontGrant::Verify(depth));
                let acceptable_stopped = pass
                    .acceptable()
                    .iter()
                    .any(|token| self.options.stop_tokens.contains(token));
                let acceptable: Vec<TokenId> = pass
                    .acceptable()
                    .iter()
                    .take_while(|token| !self.options.stop_tokens.contains(token))
                    .copied()
                    .collect();
                let front: Vec<Guess> = if acceptable_stopped || acceptable.is_empty() {
                    Vec::new()
                } else {
                    pass.guesses()
                        .iter()
                        .take_while(|guess| !self.options.stop_tokens.contains(&guess.token))
                        .copied()
                        .collect()
                };
                let row_budget = allowance
                    .min(self.options.context_limit.saturating_sub(self.resident_position))
                    .min(magnitude_executor::CHAINED_ROWS)
                    .max(1);
                Rows::verify_with_front(acceptable, front).fit_budget(row_budget)
            }
            FrontGrant::Mixed { verify, front } => {
                let k = verify.min(limit);
                let f = front.min(depth.saturating_sub(k));
                let raw = two_row.pass(FrontGrant::Mixed { verify: k, front: f });
                let acceptable: Vec<TokenId> = raw
                    .acceptable()
                    .iter()
                    .copied()
                    .take_while(|token| !self.options.stop_tokens.contains(token))
                    .collect();
                let tail = if acceptable.len() == raw.acceptable().len() {
                    raw.guesses().to_vec()
                } else {
                    Vec::new()
                };
                Rows::pass(acceptable, tail, 0)
            }
        }
    }

    /// The prompt chunk from `position` to `end`; the chunk ending the prompt
    /// selects (or forces) the first generated token. A drafter entering
    /// chunks on the device at `drafter` enters, behind the chunk, each
    /// following prompt token with the chunk row before it; of a sequence
    /// decoding in front passes, the chunk ending the prompt then enters its
    /// selection with its boundary feature, right past that entry
    /// ([`ChainEntry`]).
    fn prefill_round(
        &self,
        position: usize,
        end: usize,
        drafter: Option<usize>,
    ) -> Result<RoundState, String> {
        let requirements = self.method_factory.requires();
        let successor_dependent = self.method_factory.priming_requires_successor();
        let prime = drafter
            .filter(|_| requirements.head && requirements.prefill_demand.contains(Demand::FEATURES))
            .map(|drafter| Priming {
                tokens: if successor_dependent {
                    self.prompt[drafter + 1..end].to_vec()
                } else {
                    self.prompt[position + 1..(end + 1).min(self.prompt.len())].to_vec()
                },
                boundary: successor_dependent,
                position: drafter,
                draft_from: self.priming_horizon(end),
            })
            .filter(|prime| prime.boundary || !prime.tokens.is_empty());
        let finishing = end == self.prompt.len();
        let tokens = self.prompt[position..end].to_vec();
        let forced = if finishing {
            self.forced_tokens(1)?
        } else {
            Vec::new()
        };
        let select = if finishing && forced.is_empty() {
            Some(
                verification_selects(
                    self.generated.len(),
                    &self.generated,
                    &[],
                    self.constraint.as_deref(),
                    self.options.sampling,
                    self.options.shaping,
                    self.options.seed,
                )?
                .remove(0),
            )
        } else {
            None
        };
        let chain = prime
            .as_ref()
            .filter(|prime| self.two_row.is_some() && select.is_some() && prime.boundary)
            .map(|prime| ChainEntry { position: prime.position + prime.tokens.len() });
        RoundState::progress(WorkKind::Prefill, tokens, position, forced, select, prime, chain, requirements)
    }

    /// See `StartedRound::planned_prefill`.
    fn planned_prefill(
        &self,
        forward: &RoundForward,
        request: RequestId,
        chunk_rows: usize,
        stop: Option<usize>,
        chunks: usize,
    ) -> Result<Vec<Operation>, String> {
        if forward.kind != WorkKind::Prefill
            || !forward.selects.is_empty()
            || forward.committed != forward.tokens.len()
            || !self.layout.spans().is_empty()
            || self.finish.is_some()
        {
            return Ok(Vec::new());
        }
        let position = self.resident_position + forward.tokens.len();
        if position < self.reconciliation_target {
            return Ok(Vec::new());
        }
        let mut drafter = forward
            .prime
            .as_ref()
            .map(|prime| prime.position + prime.tokens.len());
        // The last chunk ends the prompt, the one that selects.
        let mut planned = Vec::new();
        for chunk in self.layout.group_chunks(position, chunk_rows, stop, chunks)? {
            let next = self.prefill_round(chunk.start, chunk.end, drafter)?.forward().clone();
            drafter = next
                .prime
                .as_ref()
                .map(|prime| prime.position + prime.tokens.len());
            planned.push(next.into_operation(request, chunk.start, None)?);
        }
        Ok(planned)
    }

    /// One selection per proposal, keyed like the target's selection of the
    /// same output position (so identical draft and target distributions
    /// select identical tokens). Proposal history is not previewed, and only
    /// the first proposal knows its grammar mask.
    fn proposal_selects(&self, count: usize) -> Result<Vec<SelectSpec>, String> {
        let history = self
            .options
            .shaping
            .uses_history()
            .then(|| selection_history(&self.generated, &[]))
            .transpose()?;
        let first_mask = self
            .constraint
            .as_ref()
            .map(|constraint| constraint.mask())
            .transpose()?;
        (0..count)
            .map(|index| {
                Ok(SelectSpec {
                    sampling: self.options.sampling,
                    seed: self.options.seed,
                    position: self
                        .generated
                        .len()
                        .checked_add(index)
                        .ok_or("proposal selection position exhausted")?,
                    domain: 0,
                    mask: if index == 0 { first_mask.clone() } else { None },
                    shaping: self.options.shaping,
                    history: history.clone(),
                })
            })
            .collect()
    }

    /// Numerical state may trail logically accepted tokens because selection
    /// commits the input row that produced a successor, not the successor row
    /// itself. This fact is independent of why a checkpoint is requested.
    pub fn pending_reconciliation(&self) -> Option<PendingReconciliation> {
        let end = self.reconciled_position();
        (self.resident && self.resident_position < end).then_some(PendingReconciliation {
            start: self.resident_position,
            end,
        })
    }

    /// Advance accepted successors without selecting or publishing another
    /// token. Callers use this before any operation requiring exact numerical
    /// state, including but not limited to retention. A refused start returns
    /// the generation unchanged with its error.
    pub fn start_reconciliation(self, allowance: usize) -> Result<StartedRound, (Self, String)> {
        match self.reconciliation_round(allowance) {
            Ok((target, round)) => {
                let mut generation = self;
                generation.reconciliation_target = target;
                Ok(StartedRound { generation, round })
            }
            Err(error) => Err((self, error)),
        }
    }

    /// The reconciliation round and the numerical boundary it advances toward.
    fn reconciliation_round(&self, allowance: usize) -> Result<(usize, RoundState), String> {
        let pending = self
            .pending_reconciliation()
            .ok_or("generation has no pending causal reconciliation")?;
        if allowance == 0 {
            return Err("reconciliation allowance must be positive".into());
        }
        let end = self
            .layout
            .chunk_end(pending.start, pending.end, allowance)?;
        let tokens = self
            .prompt
            .iter()
            .chain(&self.generated)
            .skip(pending.start)
            .take(end - pending.start)
            .copied()
            .collect();
        let round = RoundState::progress(
            WorkKind::Replay,
            tokens,
            pending.start,
            Vec::new(),
            None,
            None,
            None,
            self.method_factory.requires(),
        )?;
        Ok((pending.end, round))
    }

    /// A constrained run of at most `limit` tokens accepted without
    /// selection. A sequence decoding in front passes forces none: each row
    /// it commits enters the head behind a selecting step, and the
    /// constraint's mask selects what it would force.
    fn forced_tokens(&self, limit: usize) -> Result<Vec<TokenId>, String> {
        if self.two_row.is_some() {
            return Ok(Vec::new());
        }
        let limit = limit
            .min(self.options.forced_quantum)
            .min(self.options.max_tokens - self.generated.len())
            .min(self.options.output_capacity - self.output.len());
        // A zero bound forces nothing: quantum 0 disables forced runs, and the
        // output or token budget may be spent. Constraints serve positive
        // allowances only.
        let Some(constraint) = self.constraint.as_ref().filter(|_| limit > 0) else {
            return Ok(Vec::new());
        };
        let forced = constraint.forced(limit)?;
        if forced.len() > limit
            || forced
                .iter()
                .any(|token| token.0 as usize >= self.options.vocabulary)
        {
            return Err("constraint returned an invalid forced run".into());
        }
        let mut bounded = Vec::with_capacity(forced.len());
        for token in forced {
            bounded.push(token);
            if self.options.stop_tokens.contains(&token) {
                break;
            }
        }
        Ok(bounded)
    }
    pub fn take(&mut self, count: usize) -> Result<Vec<OutputToken>, String> {
        if count == 0 {
            return Err("output collection count must be positive".into());
        }
        let tokens = self
            .output
            .drain(..count.min(self.output.len()))
            .collect::<Vec<_>>();
        self.published += tokens.len();
        Ok(tokens)
    }

    /// Consume a method operation's outcome against private method state.
    /// The decision commits a head transaction's entry rows.
    pub fn prepare_method_transition(
        &self,
        operation: &Operation,
        outcome: &magnitude_executor::Outcome,
    ) -> Result<PreparedMethodTransition, String> {
        let Operation::Head {
            request, tokens, ..
        } = operation
        else {
            return Err("only a head transaction reconciles into method state".into());
        };
        if let magnitude_executor::Outcome::Head { proposals, .. } = outcome {
            for &selected in proposals {
                match selected {
                    magnitude_executor::Selected::Token(token)
                        if (token.0 as usize) >= self.options.vocabulary =>
                    {
                        return Err("head returned an out-of-vocabulary proposal".into());
                    }
                    magnitude_executor::Selected::NonFinite => {
                        return Err("draft sampling distribution is nonfinite".into());
                    }
                    magnitude_executor::Selected::Token(_)
                    | magnitude_executor::Selected::Empty
                    | magnitude_executor::Selected::Declined => {}
                }
            }
        }
        let mut method = self.method.fork_transition();
        method.reconcile(operation, outcome.clone())?;
        // A head transaction that drafted, of a generation decoding in
        // front passes, drafted the next pass's chain.
        let mut two_row = self.two_row.clone();
        if let (Some(decode), Operation::Head { proposals, .. }, magnitude_executor::Outcome::Head { proposals: selected, .. }) =
            (two_row.as_mut(), operation, outcome)
        {
            if !proposals.is_empty() {
                decode.guessed(drafted_chain(selected)?);
            }
        }
        Ok(PreparedMethodTransition {
            request: *request,
            expected_resident_position: self.resident_position,
            expected_generated_len: self.generated.len(),
            expected_finish: self.finish,
            method,
            accepted_rows: tokens.len(),
            two_row,
        })
    }

    /// A launch carrying this request's work finished in `duration`: `rows`
    /// rows of `sequences` requests. Call it before the launch's outcome is
    /// reconciled.
    pub fn timed(
        &mut self,
        launch: TimedLaunch,
        duration: Option<LaunchDuration>,
        sequences: usize,
        rows: usize,
    ) {
        self.method.timed(launch, duration, sequences, rows);
    }

    /// The drafted block this request's next target round verifies, when
    /// its method chooses how much of it to verify. Requests sharing a
    /// launch share its rows by these ([`share_rows`]).
    pub fn draft_outlook(&self) -> Option<DraftOutlook> {
        self.method.outlook()
    }

    /// Observe the streamed target verify prior for the upcoming decode
    /// round, storing it for [`DecodePassScorer`] and forwarding to this
    /// generation's method.
    pub fn observe_streamed_prior(&self, prior: Option<StreamedVerifyPrior>) {
        self.pricing
            .lock()
            .expect("generation pricing lock poisoned")
            .streamed_prior = prior.clone();
        self.method_factory.observe_streamed_prior(prior);
    }

    /// Observe the load's analytical [`magnitude_cost_model::LaunchPrices`]
    /// snapshot, storing it for [`DecodePassScorer`] and forwarding to this
    /// generation's method.
    pub fn observe_launch_prices(&self, prices: Option<magnitude_cost_model::LaunchPrices>) {
        self.pricing
            .lock()
            .expect("generation pricing lock poisoned")
            .launch_prices = prices.clone();
        self.method_factory.observe_launch_prices(prices);
    }

    pub fn commit_method_transition(&mut self, transition: PreparedMethodTransition) {
        assert!(
            self.resident_position == transition.expected_resident_position
                && self.generated.len() == transition.expected_generated_len
                && self.finish == transition.expected_finish,
            "generation changed between method preparation and physical reconciliation"
        );
        self.method = transition.method;
        self.two_row = transition.two_row;
    }

    /// See `StartedRound::prepare_round_transition`.
    fn prepare_round_transition(
        &self,
        round: &RoundState,
        request: RequestId,
        samples: &[TokenId],
        guesses: Vec<Guess>,
        features: Option<FeatureRef>,
        reader: &mut dyn FeatureReader,
    ) -> Result<PreparedGenerationTransition, String> {
        let causal_reconciliation = self.finish.is_some()
            && round.forward().kind == WorkKind::Replay
            && round.forward().selects.is_empty();
        let mut acceptance = round.clone().reconcile(samples, &self.options.stop_tokens)?;
        // The chain the round's outcome guessed is the next pass's, of a
        // sequence decoding in front passes, which alone runs front rows.
        let mut two_row = self.two_row.clone();
        acceptance.advance(two_row.as_mut(), guesses);
        // A finished generation accepts nothing of a round that ran past its
        // end, and its guesses decide nothing.
        if self.finish.is_some() && !causal_reconciliation {
            acceptance.emitted.clear();
            acceptance.committed_rows = 0;
            acceptance.proposed = 0;
            acceptance.accepted_proposals = 0;
            acceptance.method_update = MethodUpdate::None;
            acceptance.verdict = AheadVerdict::default();
        }
        if acceptance
            .emitted
            .iter()
            .any(|token| token.0 as usize >= self.options.vocabulary)
            || acceptance.emitted.len()
                > self
                    .options
                    .output_capacity
                    .saturating_sub(self.output.len())
            || acceptance.emitted.len()
                > self.options.max_tokens.saturating_sub(self.generated.len())
            || (!causal_reconciliation
                && acceptance.committed_rows
                    > self
                        .options
                        .context_limit
                        .saturating_sub(self.resident_position))
        {
            return Err("round returned invalid tokens or exceeded a generation bound".into());
        }
        let constraint = if acceptance.emitted.is_empty() {
            None
        } else {
            self.constraint
                .as_ref()
                .map(|constraint| -> Result<_, String> {
                    if constraint.position() != self.generated.len() {
                        return Err("constraint progress differs from accepted tokens".into());
                    }
                    let next = constraint.stage(&acceptance.emitted)?;
                    if next.position() != self.generated.len() + acceptance.emitted.len() {
                        return Err("constraint successor has incorrect position".into());
                    }
                    Ok(next)
                })
                .transpose()?
        };
        let mut method = self.method.fork_transition();
        let effects = match acceptance.method_update {
            MethodUpdate::None => MethodEffects::default(),
            MethodUpdate::Prime => {
                if round.forward().demand.contains(Demand::FEATURES) && features.is_none() {
                    return Err("prompt chunk demanded target features it did not return".into());
                }
                method.prime(
                    request,
                    &acceptance.inputs,
                    acceptance.emitted.first().copied(),
                    features,
                    self.priming_horizon(self.resident_position + acceptance.committed_rows),
                    round
                        .forward()
                        .prime
                        .as_ref()
                        .map(|prime| prime.tokens.len()),
                    reader,
                )?
            }
            MethodUpdate::Observe => {
                let next = *acceptance
                    .emitted
                    .last()
                    .ok_or("resolved verification emitted no successor token")?;
                method.observe(
                    request,
                    Verification {
                        inputs: &acceptance.inputs,
                        accepted: acceptance.committed_rows,
                        next,
                        features,
                    },
                    reader,
                )?
            }
        };
        self.validate_method_operations(request, &effects.operations)?;
        let mut draft_stats = self.draft_stats;
        draft_stats.record(acceptance.proposed, acceptance.accepted_proposals)?;
        let resident_position = self
            .resident_position
            .checked_add(acceptance.committed_rows)
            .ok_or("resident position exhausted")?;
        let mut generated = self.generated.clone();
        let mut output = self.output.clone();
        let mut finish = self.finish;
        for &token in &acceptance.emitted {
            generated.push(token);
            if self.options.stop_tokens.contains(&token) {
                finish = Some(FinishReason::Stop);
                break;
            }
            let index = self
                .published
                .checked_add(output.len())
                .ok_or("output index exhausted")?;
            output.push_back(OutputToken { index, token });
        }
        if finish.is_none() && generated.len() >= self.options.max_tokens {
            finish = Some(FinishReason::Length);
        }
        let accepted_position = self
            .prompt
            .len()
            .checked_add(generated.len())
            .ok_or("accepted position exhausted")?
            .max(resident_position);
        if finish.is_none() && resident_position >= self.options.context_limit {
            finish = Some(FinishReason::Context);
        }
        let settled_outcome = self
            .active_pass
            .as_ref()
            .map(|(plan, proposals)| {
                if self.finish.is_some() {
                    Ok::<_, String>(PassOutcome::discarded(plan.pass_id, Some(plan), proposals))
                } else {
                    let outcome =
                        PassOutcome::from_samples(plan, proposals, samples, &self.options.stop_tokens)?;
                    debug_assert!(
                        matches!(&outcome.commitment, PassCommitment::Committed(prefix) if *prefix == acceptance.emitted),
                        "single terminal truth: PassOutcome committed prefix must match acceptance.emitted"
                    );
                    Ok(outcome)
                }
            })
            .transpose()?;
        Ok(PreparedGenerationTransition {
            request,
            decision: acceptance.decision(),
            commit: GenerationCommit {
                expected_resident_position: self.resident_position,
                expected_generated_len: self.generated.len(),
                expected_finish: self.finish,
                expected_output_len: self.output.len(),
                expected_published: self.published,
                expected_forward: round.forward().clone(),
                method,
                effects,
                constraint,
                generated,
                output,
                draft_stats,
                resident_position,
                accepted_position,
                finish,
                two_row,
                settled_outcome,
            },
        })
    }

    /// See `StartedRound::commit`.
    fn commit_transition(&mut self, forward: &RoundForward, transition: GenerationCommit) -> MethodEffects {
        assert!(
            self.resident_position == transition.expected_resident_position
                && self.generated.len() == transition.expected_generated_len
                && self.finish == transition.expected_finish
                && self.output.len() == transition.expected_output_len
                && self.published == transition.expected_published
                && forward == &transition.expected_forward,
            "generation changed between preparation and physical reconciliation"
        );
        self.method = transition.method;
        if let Some(constraint) = transition.constraint {
            self.constraint = Some(constraint);
        }
        self.generated = transition.generated;
        self.output = transition.output;
        self.draft_stats = transition.draft_stats;
        self.resident_position = transition.resident_position;
        self.accepted_position = transition.accepted_position;
        self.finish = transition.finish;
        self.two_row = transition.two_row;
        self.active_pass = None;
        if let Some(outcome) = transition.settled_outcome {
            if let Some(settled) = self.pass_outcomes.record_verdict(outcome) {
                self.last_pass_outcome = Some(settled);
            }
        }
        transition.effects
    }

    /// Method work belongs to the method's own executor and request.
    fn validate_method_operations(
        &self,
        request: RequestId,
        operations: &[Operation],
    ) -> Result<(), String> {
        if operations.iter().any(|operation| {
            operation.request() != request
                || operation.executable() != magnitude_executor::ExecutableKind::Head
                || !self.method_factory.requires().head
        }) {
            return Err("method returned effects outside its executor ownership".into());
        }
        Ok(())
    }

    fn cancel_active_pass(&mut self) {
        if let Some((plan, _)) = self.active_pass.take() {
            if let Some(settled) = self.pass_outcomes.cancel(plan.pass_id) {
                self.last_pass_outcome = Some(settled);
            }
        }
    }

    /// Accepted output remains owned by the caller until drained or discarded.
    pub fn cancel(&mut self) {
        self.cancel_active_pass();
        if self.finish.is_none() {
            self.finish = Some(FinishReason::Cancelled);
        }
    }
    pub fn fail(&mut self) {
        self.cancel_active_pass();
        if self.finish.is_none() {
            self.finish = Some(FinishReason::Failed);
        }
    }
    pub fn discard_output(&mut self) {
        self.published += self.output.len();
        self.output.clear();
    }

    /// Called after the execution owner releases numerical state. Logical history,
    /// grammar, queued output, and terminal decisions remain available.
    pub fn evicted(&mut self) -> Result<(), String> {
        if let Some((plan, _)) = self.active_pass.take() {
            let req = plan.pass_id.request;
            if let Some(settled) = self.pass_outcomes.cancel(plan.pass_id) {
                self.last_pass_outcome = Some(settled);
            }
            self.pass_generation = self.pass_generation.wrapping_add(1);
            self.pass_outcomes.retire_before(req, self.pass_generation);
        } else {
            self.pass_generation = self.pass_generation.wrapping_add(1);
        }
        // Restore at least the numerical prefix that existed before eviction,
        // preserving an earlier target if eviction occurs again mid-replay.
        // A live decode normally has one accepted successor that has not yet
        // been fed through the model; replaying that token here would advance
        // past the point from which the next decode must start.
        self.reconciliation_target = self.reconciliation_target.max(self.resident_position);
        self.method.evict();
        self.resident = false;
        Ok(())
    }
    /// The exclusive bound on a resumable prefix: the request must still
    /// compute the row it next samples from (the last prompt row before its
    /// first token, the last accepted row after that).
    pub fn resume_bound(&self) -> usize {
        self.prompt.len().max(self.accepted_position)
    }

    /// Become resident on numerical state the executor has just installed:
    /// fresh state at zero, or a prefix of this request's path of `position`
    /// rows with its method checkpoint. Replay advances from there to the
    /// numerical boundary held before eviction; prompt rows the prefix covers
    /// beyond that boundary are accepted as cached.
    pub fn resume_at(&mut self, from: Option<(usize, &MethodCheckpoint)>) -> Result<(), String> {
        if self.resident {
            return Err("only a non-resident generation can resume".into());
        }
        let position = match from {
            None => 0,
            Some((position, checkpoint)) => {
                if position == 0
                    || position >= self.resume_bound()
                    || (position < self.prompt.len() && !self.layout.boundary(position))
                {
                    return Err(
                        "resumed prefix is not an exact boundary of the request path".into(),
                    );
                }
                self.method = self.method_at(position, checkpoint)?;
                position
            }
        };
        if position > self.reconciliation_target {
            self.reconciliation_target = position;
            self.accepted_position = self.accepted_position.max(position);
            self.cached_tokens = self.cached_tokens.max(position.min(self.prompt.len()));
        }
        self.resident_position = position;
        self.resident = true;
        Ok(())
    }
}

impl StartedRound {
    /// The generation as it was when the round started.
    pub fn generation(&self) -> &Generation {
        &self.generation
    }

    /// See [`Generation::timed`].
    pub fn timed(
        &mut self,
        launch: TimedLaunch,
        duration: Option<LaunchDuration>,
        sequences: usize,
        rows: usize,
    ) {
        self.generation.timed(launch, duration, sequences, rows);
    }

    pub fn round_forward(&self) -> &RoundForward {
        self.round.forward()
    }

    /// Whether the decode queued behind this round's target step publishes
    /// its feature (`MethodState::publishes_behind`).
    pub fn publishes_behind(&self) -> bool {
        self.generation.method.publishes_behind()
    }

    pub fn round_publication_bound(&self) -> usize {
        self.round.publication_bound()
    }

    /// The prompt chunks that follow this round once it commits, when they
    /// are known now: the round is a prompt chunk that selects nothing, and
    /// the prompt is plain text (a conditioned span's rows need its
    /// encoding). At most `chunks` of them, each of at most `chunk_rows` rows
    /// and none past `stop`: the chunks a lookahead or a chain queues behind
    /// it. Each chunk's drafter entry follows the one before it; planning
    /// ends with the chunk that selects the first token.
    pub fn planned_prefill(
        &self,
        request: RequestId,
        chunk_rows: usize,
        stop: Option<usize>,
        chunks: usize,
    ) -> Result<Vec<Operation>, String> {
        self.generation
            .planned_prefill(self.round.forward(), request, chunk_rows, stop, chunks)
    }

    /// Prepare acceptance, grammar, method effects, counters, and publication
    /// against private state. The round stays started until the executor has
    /// reconciled the returned decision.
    pub fn prepare_round_transition(
        &self,
        request: RequestId,
        samples: &[TokenId],
        guesses: Vec<Guess>,
        features: Option<FeatureRef>,
        reader: &mut dyn FeatureReader,
    ) -> Result<PreparedGenerationTransition, String> {
        self.generation
            .prepare_round_transition(&self.round, request, samples, guesses, features, reader)
    }

    /// Apply a fully checked transition after physical state reconciliation,
    /// ending the round. Any mismatch here is an executor/generation ordering
    /// bug.
    pub fn commit(self, transition: PreparedGenerationTransition) -> (Generation, MethodEffects) {
        self.commit_with(transition.commit)
    }

    fn commit_with(self, commit: GenerationCommit) -> (Generation, MethodEffects) {
        let Self {
            mut generation,
            round,
        } = self;
        let effects = generation.commit_transition(round.forward(), commit);
        (generation, effects)
    }

    /// Reconcile the round's outcome physically by `transition`'s decision,
    /// moved into `physical` (the executor's reconciliation), then commit
    /// `transition` where that succeeds ([`Self::commit`]). Where it fails,
    /// the round returns unended, beside the failure.
    pub fn reconcile<E>(
        self,
        transition: PreparedGenerationTransition,
        physical: impl FnOnce(PhysicalDecision) -> Result<(), E>,
    ) -> Result<(Generation, MethodEffects), (Box<Self>, E)> {
        let PreparedGenerationTransition { decision, commit, .. } = transition;
        match physical(decision) {
            Ok(()) => Ok(self.commit_with(commit)),
            Err(error) => Err((Box::new(self), error)),
        }
    }

    /// Discard the round after the execution owner releases numerical state.
    /// See `Generation::evicted`.
    pub fn evicted(self) -> Result<Generation, String> {
        let mut generation = self.generation;
        generation.evicted()?;
        Ok(generation)
    }

    /// Discard the round, its physical outcome aborted (its sequence's
    /// accepted state returned), to run it again: the generation is as it
    /// was before the round, unfinished, and starts it anew. Starting a
    /// round changes the generation only by its method's proposal, which a
    /// re-run proposes again; its sampling is seeded by position, so the
    /// round's output is the same.
    pub fn abandon(mut self) -> Generation {
        self.generation.cancel_active_pass();
        self.generation
    }

    /// Discard the round without committing or publishing it.
    pub fn cancel(self) -> Generation {
        let mut generation = self.generation;
        generation.cancel();
        generation
    }

    pub fn fail(self) -> Generation {
        let mut generation = self.generation;
        generation.fail();
        generation
    }
}

fn validate_input(
    prompt: &[TokenId],
    layout: &InputLayout,
    options: &Options,
    constraint: Option<&dyn Constraint>,
) -> Result<(), String> {
    if prompt.is_empty()
        || prompt.len() != layout.count()
        || prompt.len() > options.context_limit
        || options.context_limit > i32::MAX as usize
        || options.vocabulary == 0
        || options.vocabulary > i32::MAX as usize
        || options.output_capacity == 0
        || options.forced_quantum > 256
        || options.max_tokens > i32::MAX as usize
        || options.shaping.validate().is_err()
        || (options.shaping.temperature == 0.0) != (options.sampling == Sampling::Greedy)
        || prompt
            .iter()
            .chain(options.stop_tokens.iter())
            .chain(options.suppressed_tokens.iter())
            .any(|id| id.0 as usize >= options.vocabulary)
        || constraint.is_some_and(|value| value.position() != 0)
    {
        return Err("invalid generation input, options, or initial constraint position".into());
    }
    Ok(())
}

#[cfg(test)]
mod thread_boundary {
    use super::GenerationSeed;

    fn assert_send<T: Send>() {}

    #[test]
    fn generation_seed_is_transportable() {
        assert_send::<GenerationSeed>();
    }
}

fn method_matches_choice(choice: MethodChoice, method: &dyn Method) -> bool {
    let requirements = method.requires();
    match (choice, method.capability()) {
        (MethodChoice::Plain, MethodChoice::Plain) => {
            requirements
                == (MethodRequirements {
                    prefill_demand: Demand::NONE,
                    closing_demand: Demand::NONE,
                    verify_demand: Demand::NONE,
                    head: false,
                })
        }
        (
            MethodChoice::Mtp { proposals },
            MethodChoice::Mtp {
                proposals: capacity,
            },
        )
        | (
            MethodChoice::DFlash { proposals },
            MethodChoice::DFlash {
                proposals: capacity,
            },
        ) => {
            proposals > 0
                && proposals <= capacity
                && usize::from(proposals) <= method.proposals()
                && requirements.head
                // The first draft's anchor is selected by the closing
                // chunk's final feature, whichever demand returns it.
                && (requirements.prefill_demand | requirements.closing_demand)
                    .contains(Demand::FEATURES)
                && requirements.verify_demand.contains(Demand::FEATURES)
        }
        _ => false,
    }
}

#[cfg(test)]
mod method_choice_tests {
    use super::*;

    struct Descriptor {
        identity: &'static str,
        capability: MethodChoice,
        requirements: MethodRequirements,
        proposals: usize,
    }

    impl Method for Descriptor {
        fn identity(&self) -> &str {
            self.identity
        }
        fn capability(&self) -> MethodChoice {
            self.capability
        }
        fn requires(&self) -> MethodRequirements {
            self.requirements
        }
        fn embeds_prompt_tokens(&self) -> bool {
            false
        }
        fn proposals(&self) -> usize {
            self.proposals
        }
        fn create(
            &self,
            _checkpoint: Option<&MethodCheckpoint>,
        ) -> Result<Box<dyn MethodState>, String> {
            Err("descriptor-only validation fixture".into())
        }
    }

    #[test]
    fn choice_requires_matching_identity_requirements_and_mtp_width() {
        let plain = Descriptor {
            identity: "plain",
            capability: MethodChoice::Plain,
            requirements: MethodRequirements {
                prefill_demand: Demand::NONE,
                closing_demand: Demand::NONE,
                verify_demand: Demand::NONE,
                head: false,
            },
            proposals: 0,
        };
        assert!(method_matches_choice(MethodChoice::Plain, &plain));
        assert!(!method_matches_choice(
            MethodChoice::Mtp { proposals: 1 },
            &plain
        ));

        let mtp = Descriptor {
            identity: "mtp:fixture:3",
            capability: MethodChoice::Mtp { proposals: 3 },
            requirements: MethodRequirements {
                prefill_demand: Demand::FEATURES,
                closing_demand: Demand::NONE,
                verify_demand: Demand::FEATURES,
                head: true,
            },
            proposals: 3,
        };
        assert!(method_matches_choice(
            MethodChoice::Mtp { proposals: 3 },
            &mtp
        ));
        assert!(!method_matches_choice(
            MethodChoice::Mtp { proposals: 4 },
            &mtp
        ));
        assert!(!method_matches_choice(
            MethodChoice::Mtp { proposals: 0 },
            &mtp
        ));

        // A head reading the target's history demands only the closing
        // chunk's features; a head demanding no prompt features cannot draft.
        let reading = Descriptor {
            identity: "mtp:fixture:3",
            capability: MethodChoice::Mtp { proposals: 3 },
            requirements: MethodRequirements {
                prefill_demand: Demand::NONE,
                closing_demand: Demand::FEATURES,
                verify_demand: Demand::FEATURES,
                head: true,
            },
            proposals: 3,
        };
        assert!(method_matches_choice(MethodChoice::Mtp { proposals: 3 }, &reading));
        let blind = Descriptor {
            requirements: MethodRequirements {
                closing_demand: Demand::NONE,
                ..reading.requirements
            },
            ..reading
        };
        assert!(!method_matches_choice(MethodChoice::Mtp { proposals: 3 }, &blind));
    }
}

#[cfg(test)]
mod seal_front_pass_real_pricing_tests {
    use super::*;
    use magnitude_executor::StreamedVerifyPrior;

    fn fronting_generation(proposals: usize, chain: Vec<Guess>) -> Generation {
        let options = Options {
            max_tokens: 64,
            output_capacity: 64,
            context_limit: 512,
            vocabulary: 1024,
            stop_tokens: BTreeSet::new(),
            suppressed_tokens: BTreeSet::new(),
            sampling: Sampling::Greedy,
            shaping: Shaping {
                temperature: 0.0,
                ..Shaping::default()
            },
            seed: 0,
            forced_quantum: 0,
            method: MethodChoice::Mtp {
                proposals: proposals as u8,
            },
            end_of_generation: EndOfGeneration::default(),
            reasoning_budget: None,
        };
        let method = Arc::new(Mtp::new("glm53-test", proposals).unwrap().fronting().unwrap());
        let mut generation = Generation::new_with_method(
            vec![TokenId(1), TokenId(2)],
            InputLayout::new(2, Vec::new()).unwrap(),
            options,
            None,
            method,
        )
        .unwrap();
        generation.resident = true;
        generation.resident_position = 2;
        generation.accepted_position = 3;
        generation.generated.push(TokenId(10));
        let mut two_row = TwoRowDecode::default();
        two_row.guessed(chain);
        generation.two_row = Some(two_row);
        generation
    }

    fn make_chain(len: usize) -> Vec<Guess> {
        (0..len)
            .map(|i| Guess {
                token: TokenId(20 + i as u32),
                kind: if i + 1 == len {
                    GuessKind::Head
                } else {
                    GuessKind::Chained
                },
            })
            .collect()
    }

    /// Reproduces GLM-5.3 UD-IQ2_XXS + Embedded MTP (`f6ec8c` on RTX 5090):
    /// high streamed PCIe I/O (~15-20 ms/row marginal) + ~59% empirical
    /// acceptance MUST cap verify width tightly instead of blasting 14-16
    /// token verify batches.
    #[test]
    fn high_io_and_moderate_acceptance_caps_verify_width() {
        let chain = make_chain(15);
        let gen = fronting_generation(15, chain);
        // Streamed MoE prior: exact row I/O 40 ms, +18 ms/row marginal I/O,
        // resident compute 8 ms + 1 ms/row, 59% empirical conditional acceptance.
        let io_seconds: Vec<f64> = (0..=15).map(|k| 0.040 + 0.018 * k as f64).collect();
        let compute_seconds: Vec<f64> = (0..=15).map(|m| 0.008 + 0.001 * m as f64).collect();
        let verify_seconds: Vec<f64> = io_seconds
            .iter()
            .zip(&compute_seconds)
            .map(|(&io, &c)| io.max(c))
            .collect();
        gen.observe_streamed_prior(Some(StreamedVerifyPrior {
            regime_epoch: 1,
            verify_seconds,
            compute_seconds,
            io_seconds,
            front_stall_saved_per_row: 0.010,
            front_waste_per_row: vec![0.015; 15],
            verify_host_overhead_seconds: 0.002,
            verify_host_overhead_per_step_seconds: 0.0,
            accepts: vec![Some(0.59)],
            stand: Some(0.59),
            ..Default::default()
        }));

        let two_row = gen.two_row.as_ref().unwrap();
        let (rows, (plan, _proposals)) = gen
            .seal_front_pass(
                PassId {
                    request: RequestId(1),
                    generation: 0,
                    pass: 0,
                },
                two_row,
                16,
                15,
                FrontGrant::Verify(15),
                63,
            )
            .unwrap();

        assert!(
            plan.verify <= 3,
            "high-I/O + 59% acceptance must cap verify width tightly (never 14-16), got K = {}, F = {}, pricing = {:?}",
            plan.verify,
            plan.front,
            plan.pricing
        );
        assert_eq!(rows.acceptable().len(), plan.verify);
        assert_eq!(rows.guesses().len(), plan.front);
        assert!(
            plan.pricing.compulsory_io_seconds >= 0.040,
            "sealed plan must reflect real compulsory I/O, got {}",
            plan.pricing.compulsory_io_seconds
        );
    }

    /// High-acceptance (95%+) VRAM-resident / low-I/O (DSpark or Qwen3.8):
    /// `seal_front_pass` must still select optimal wide verify.
    #[test]
    fn high_acceptance_resident_keeps_wide_verify() {
        let chain = make_chain(7);
        let gen = fronting_generation(7, chain);
        // VRAM-resident: zero I/O, gentle compute slope (10 ms + 1 ms/row),
        // 95% conditional acceptance across depths.
        let compute_seconds: Vec<f64> = (0..=7).map(|m| 0.010 + 0.001 * m as f64).collect();
        gen.observe_streamed_prior(Some(StreamedVerifyPrior {
            regime_epoch: 1,
            verify_seconds: compute_seconds.clone(),
            compute_seconds,
            io_seconds: vec![0.0; 8],
            front_stall_saved_per_row: 0.0,
            front_waste_per_row: vec![0.0; 7],
            verify_host_overhead_seconds: 0.001,
            verify_host_overhead_per_step_seconds: 0.0,
            accepts: vec![Some(0.95); 7],
            stand: Some(0.95),
            ..Default::default()
        }));

        let two_row = gen.two_row.as_ref().unwrap();
        let (rows, (plan, _)) = gen
            .seal_front_pass(
                PassId {
                    request: RequestId(1),
                    generation: 0,
                    pass: 0,
                },
                two_row,
                8,
                7,
                FrontGrant::Verify(7),
                63,
            )
            .unwrap();

        assert!(
            plan.verify >= 6,
            "95% acceptance VRAM-resident must keep wide verify (K >= 6), got K = {}, pricing = {:?}",
            plan.verify,
            plan.pricing
        );
        assert_eq!(plan.front, 0, "resident pass has zero stall so F must be 0");
        assert_eq!(rows.acceptable().len(), plan.verify);
    }

    /// Empirical acceptance accumulated in `PassOutcomeTracker` (even when
    /// watcher `accepts` is empty after a regime reset) caps verify width
    /// under high I/O.
    #[test]
    fn tracker_empirical_acceptance_caps_verify_when_watcher_accepts_empty() {
        let chain = make_chain(15);
        let mut gen = fronting_generation(15, chain);
        // Settle 10 passes with 59% (6/10) match rate at depth 1.
        for pass in 0..10u64 {
            let pid = PassId {
                request: RequestId(1),
                generation: 0,
                pass,
            };
            let proposals = vec![TokenId(20)];
            let plan = DecodePassPlan {
                pass_id: pid,
                generated: 1,
                verify: 1,
                front: 0,
                idle_front: 0,
                reserved_capacity: 2,
                pricing: DecodePassScorer::score_candidate(
                    0.0,
                    1,
                    0,
                    &[1.0],
                    HorizonLimits::UNBOUNDED,
                    &PassCostModel::unit_compute(2),
                ),
            };
            gen.pass_outcomes.feed(plan, proposals.clone());
            let samples = if pass < 6 {
                vec![TokenId(20), TokenId(99)]
            } else {
                vec![TokenId(88), TokenId(99)]
            };
            let outcome = PassOutcome::from_samples(&plan, &proposals, &samples, &BTreeSet::new()).unwrap();
            gen.pass_outcomes.record_verdict(outcome);
        }
        assert!(
            (gen.pass_outcomes.conditional_rate(1).unwrap() - 0.6).abs() < 1e-12
        );

        let io_seconds: Vec<f64> = (0..=15).map(|k| 0.040 + 0.018 * k as f64).collect();
        let compute_seconds: Vec<f64> = (0..=15).map(|m| 0.008 + 0.001 * m as f64).collect();
        gen.observe_streamed_prior(Some(StreamedVerifyPrior {
            regime_epoch: 2,
            verify_seconds: io_seconds.clone(),
            compute_seconds,
            io_seconds,
            front_stall_saved_per_row: 0.010,
            front_waste_per_row: vec![0.015; 15],
            verify_host_overhead_seconds: 0.002,
            verify_host_overhead_per_step_seconds: 0.0,
            accepts: Vec::new(),
            stand: None,
            ..Default::default()
        }));

        let two_row = gen.two_row.as_ref().unwrap();
        let (_rows, (plan, _)) = gen
            .seal_front_pass(
                PassId {
                    request: RequestId(1),
                    generation: 0,
                    pass: 10,
                },
                two_row,
                16,
                15,
                FrontGrant::Verify(15),
                63,
            )
            .unwrap();

        assert!(
            plan.verify <= 3,
            "tracker empirical 60% acceptance must cap verify width under high I/O, got K = {}",
            plan.verify
        );
    }

    /// `FrontGrant::Exactly` (queued pass claim) preserves exact queued rows
    /// and scores with real pricing rather than dummy zero-I/O values.
    #[test]
    fn exact_front_grant_preserves_queued_rows_with_real_pricing() {
        let chain = make_chain(2);
        let gen = fronting_generation(4, chain);
        gen.observe_streamed_prior(Some(StreamedVerifyPrior {
            regime_epoch: 1,
            verify_seconds: vec![0.040, 0.055, 0.070, 0.085, 0.100],
            compute_seconds: vec![0.008, 0.009, 0.010, 0.011, 0.012],
            io_seconds: vec![0.040, 0.055, 0.070, 0.085, 0.100],
            front_stall_saved_per_row: 0.012,
            front_waste_per_row: vec![0.005; 4],
            verify_host_overhead_seconds: 0.002,
            verify_host_overhead_per_step_seconds: 0.0,
            accepts: vec![Some(0.75)],
            stand: Some(0.75),
            ..Default::default()
        }));

        let two_row = gen.two_row.as_ref().unwrap();
        let (rows, (plan, _)) = gen
            .seal_front_pass(
                PassId {
                    request: RequestId(1),
                    generation: 0,
                    pass: 0,
                },
                two_row,
                5,
                4,
                FrontGrant::Exactly(4),
                63,
            )
            .unwrap();

        assert_eq!(plan.verify, 0);
        assert_eq!(plan.front, 2);
        assert_eq!(plan.idle_front, 2);
        assert_eq!(rows.row_count(), 5);
        assert!((plan.pricing.compulsory_io_seconds - 0.040).abs() < 1e-12);
        assert!((plan.pricing.compute_seconds - 0.012).abs() < 1e-12);
    }

    /// Reproduces `e13e3b` (`rtx3090-64g-gen4` Qwen3.8 path-bound `TwoRowDecode`):
    /// when `FrontGrant::Verify` runs an unchained pass (`chain == None`) under
    /// path-bound stall + verify host overhead where `(K = 0, F = 1)` would
    /// otherwise outscore `(K = 0, F = 0)` and `(K >= 1, F)`, `seal_front_pass`
    /// MUST never emit `(K == 0, F > 0)` and the resulting `RoundState::pass`
    /// MUST validate cleanly via `into_operation` without `OperationError::FrontRows`.
    #[test]
    fn unchained_verify_grant_never_seals_k0_with_front_rows() {
        let chain = make_chain(4);
        let gen = fronting_generation(4, chain);
        // High verify host overhead + high stall saved with low front waste:
        // unrestricted scorer would pick (K = 0, F = 1), which is illegal for
        // `WorkKind::Decode` with `chain == None`.
        gen.observe_streamed_prior(Some(StreamedVerifyPrior {
            regime_epoch: 1,
            verify_seconds: vec![0.040, 0.080, 0.120, 0.160, 0.200],
            compute_seconds: vec![0.008, 0.009, 0.010, 0.011, 0.012],
            io_seconds: vec![0.040, 0.080, 0.120, 0.160, 0.200],
            front_stall_saved_per_row: 0.025,
            front_waste_per_row: vec![0.002; 4],
            verify_host_overhead_seconds: 0.020,
            verify_host_overhead_per_step_seconds: 0.0,
            accepts: vec![Some(0.60)],
            stand: Some(0.60),
            ..Default::default()
        }));

        let two_row = gen.two_row.as_ref().unwrap();
        let (rows, (plan, _)) = gen
            .seal_front_pass(
                PassId {
                    request: RequestId(1),
                    generation: 0,
                    pass: 0,
                },
                two_row,
                5,
                4,
                FrontGrant::Verify(4),
                63,
            )
            .unwrap();

        assert!(
            !(plan.verify == 0 && plan.front > 0),
            "unchained Verify grant must never seal K == 0 with F > 0: {plan:?}"
        );
        let round = RoundState::pass(
            *gen.generated.last().unwrap(),
            rows,
            gen.generated.len(),
            &gen.generated,
            None,
            gen.options.sampling,
            gen.options.shaping,
            gen.options.seed,
            Demand::FEATURES,
            None,
        )
        .unwrap();
        round
            .forward()
            .clone()
            .into_operation(RequestId(1), gen.resident_position, None)
            .expect("sealed unchained pass must pass Operation::validate without FrontRows error");
    }

    /// Reproduces `5e0635` vs `7079d0` (Qwen3.8 on 5090-64g, 88% VRAM-resident,
    /// `FrontGrant::Verify(15)`): a verify-only grant must NOT inject `F = 1`
    /// front rows or apply phantom `front_credit` that traps `seal_front_pass`
    /// at `K = 6, F = 1` (`M = 8` class = 7 tok/step) when `K = 15, F = 0`
    /// (`16` tok/step) has higher throughput.
    #[test]
    fn verify_only_grant_does_not_trap_wide_resident_verify_at_k6_f1() {
        let chain = make_chain(15);
        let gen = fronting_generation(15, chain);
        // Power-of-two compute classes (M = 1..=8 at 15 ms, M = 9..=16 at 24 ms)
        // with low PCIe I/O (12 ms + 1.5 ms/row) and unmeasured stall fallback
        // (`front_stall_saved_per_row = 0.015`).
        let compute_seconds: Vec<f64> = (1..=16)
            .map(|rows| if rows <= 8 { 0.015 } else { 0.024 })
            .collect();
        let io_seconds: Vec<f64> = (0..=15).map(|k| 0.012 + 0.0015 * k as f64).collect();
        let verify_seconds: Vec<f64> = compute_seconds
            .iter()
            .zip(&io_seconds)
            .map(|(&c, &io)| c.max(io))
            .collect();
        gen.observe_streamed_prior(Some(StreamedVerifyPrior {
            regime_epoch: 1,
            verify_seconds,
            compute_seconds,
            io_seconds,
            front_stall_saved_per_row: 0.015,
            front_waste_per_row: vec![0.002; 15],
            verify_host_overhead_seconds: 0.002,
            verify_host_overhead_per_step_seconds: 0.0,
            accepts: vec![Some(0.95); 15],
            stand: Some(0.95),
            ..Default::default()
        }));

        let two_row = gen.two_row.as_ref().unwrap();
        let (rows, (plan, _)) = gen
            .seal_front_pass(
                PassId {
                    request: RequestId(1),
                    generation: 0,
                    pass: 0,
                },
                two_row,
                16,
                15,
                FrontGrant::Verify(15),
                63,
            )
            .unwrap();

        assert_eq!(plan.front, 0, "FrontGrant::Verify must keep F == 0, got {plan:?}");
        assert!(
            plan.verify >= 12,
            "mostly VRAM-resident high-acceptance Verify(15) must widen past K=6 (got K = {})",
            plan.verify
        );
        assert_eq!(rows.acceptable().len(), plan.verify);
        assert_eq!(rows.guesses().len(), 0);
    }
}
