//! Family-neutral request-local progress. The executor prepares numerical rows;
//! generation accepts them independently after shared physical completion.
mod acceptance;
mod controls;
mod dflash;
mod method;
mod mtp;
mod plain;
mod round;
mod shaping;
pub use acceptance::accept_prefix;
pub use controls::{EndOfGeneration, ReasoningBudget};
pub use dflash::DFlash;
pub use magnitude_artifacts::{BoundaryRule, InputLayout, InputSpan};
pub use magnitude_executor::{
    Demand, FeatureReader, FeatureRef, Operation, Priming, RequestId, Sampling, SelectSpec,
    Shaping, TokenId, WorkKind,
};
pub use method::{
    DraftCheckpoint, Method, MethodCheckpoint, MethodCheckpointError, MethodChoice, MethodEffects,
    MethodRequirements, MethodState, Propose, Verification,
};
pub use mtp::Mtp;
pub use plain::Plain;
pub use round::{MethodUpdate, RoundAcceptance, RoundForward, RoundState};
pub use shaping::{selection_history, verification_selects, HISTORY_WIDTH};
use std::{
    collections::{BTreeSet, VecDeque},
    sync::Arc,
};

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
        validate_input(&prompt, &layout, &options, constraint.as_deref())?;
        Ok(Self {
            prompt,
            layout,
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

/// The sole numerical decision generation sends back to the executor: the
/// target rows (or, for a head transaction, the head entry rows) to commit.
/// A verified prefix may need physical repair before the prepared transition
/// is committed, but its accepted length cannot change during that repair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReconcileDecision {
    pub accepted_rows: usize,
}

/// Every fallible logical change is performed against private state before
/// physical reconciliation. Dropping this value leaves the started round
/// untouched, including its generation's method state.
pub struct PreparedGenerationTransition {
    request: RequestId,
    expected_resident_position: usize,
    expected_generated_len: usize,
    expected_finish: Option<FinishReason>,
    expected_output_len: usize,
    expected_published: usize,
    expected_forward: RoundForward,
    decision: ReconcileDecision,
    method: Box<dyn MethodState>,
    effects: MethodEffects,
    constraint: Option<Box<dyn Constraint>>,
    generated: Vec<TokenId>,
    output: VecDeque<OutputToken>,
    draft_stats: DraftStats,
    resident_position: usize,
    accepted_position: usize,
    finish: Option<FinishReason>,
}

/// A method operation's outcome consumed against private method state.
pub struct PreparedMethodTransition {
    request: RequestId,
    expected_resident_position: usize,
    expected_generated_len: usize,
    expected_finish: Option<FinishReason>,
    method: Box<dyn MethodState>,
    decision: ReconcileDecision,
}

impl PreparedMethodTransition {
    pub const fn decision(&self) -> ReconcileDecision {
        self.decision
    }
    pub const fn request(&self) -> RequestId {
        self.request
    }
}

impl PreparedGenerationTransition {
    pub const fn decision(&self) -> ReconcileDecision {
        self.decision
    }
    pub const fn request(&self) -> RequestId {
        self.request
    }
}

pub struct Generation {
    prompt: Vec<TokenId>,
    layout: InputLayout,
    options: Options,
    constraint: Option<Box<dyn Constraint>>,
    generated: Vec<TokenId>,
    output: VecDeque<OutputToken>,
    published: usize,
    resident_position: usize,
    accepted_position: usize,
    reconciliation_target: usize,
    resident: bool,
    finish: Option<FinishReason>,
    method_factory: Arc<dyn Method>,
    method: Box<dyn MethodState>,
    cached_tokens: usize,
    draft_stats: DraftStats,
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
            options,
            constraint,
        } = seed;
        if method_factory.identity().is_empty()
            || !method_matches_choice(options.method, method_factory.as_ref())
        {
            return Err("generation method factory does not match request policy".into());
        }
        let finish = (options.max_tokens == 0).then_some(FinishReason::Length);
        let method = method_factory.create(None)?;
        let constraint = controls::Controlled::compose(
            constraint,
            options.end_of_generation,
            &options.stop_tokens,
            &options.suppressed_tokens,
            options.reasoning_budget.as_ref(),
            options.vocabulary,
        )?;
        Ok(Self {
            prompt,
            layout,
            options,
            constraint,
            generated: Vec::new(),
            output: VecDeque::new(),
            published: 0,
            resident_position: 0,
            accepted_position: 0,
            reconciliation_target: 0,
            resident: false,
            finish,
            method_factory,
            method,
            cached_tokens: 0,
            draft_stats: DraftStats::default(),
        })
    }
    pub fn prompt(&self) -> &[TokenId] {
        &self.prompt
    }
    pub fn layout(&self) -> &InputLayout {
        &self.layout
    }
    pub fn resident_position(&self) -> usize {
        self.resident_position
    }
    pub fn accepted_position(&self) -> usize {
        self.accepted_position
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
    pub fn method_identity(&self) -> &str {
        self.method_factory.identity()
    }
    pub fn method_requirements(&self) -> MethodRequirements {
        self.method_factory.requires()
    }
    pub fn method_reclaimable(&self) -> u64 {
        self.method.reclaimable()
    }
    /// Export owned method state for prefix retention.
    pub fn method_checkpoint(&self) -> Result<MethodCheckpoint, String> {
        if !self.resident {
            return Err("method checkpoint requires resident state".into());
        }
        self.method.checkpoint().map_err(|error| error.to_string())
    }

    /// Method state restored from a retained checkpoint of `position` rows.
    fn method_at(
        &self,
        position: usize,
        checkpoint: &MethodCheckpoint,
    ) -> Result<Box<dyn MethodState>, String> {
        match (self.options.method, checkpoint) {
            (MethodChoice::Plain, MethodCheckpoint::Plain) => {}
            (MethodChoice::Mtp { .. }, MethodCheckpoint::Mtp(state))
            | (MethodChoice::DFlash { .. }, MethodCheckpoint::DFlash(state))
                if state.target_rows() == position => {}
            _ => {
                return Err("retained method checkpoint does not match the generation".into());
            }
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

    pub fn is_resident(&self) -> bool {
        self.resident
    }
    /// Start the next target round, which then owns this generation. Method
    /// work is returned as ordinary executor operations with the generation;
    /// callers start again after reconciling it. A refused start returns the
    /// generation with its error.
    pub fn start_round(
        mut self,
        request: RequestId,
        allowance: usize,
    ) -> Result<RoundStart, (Self, String)> {
        match self.next_round(request, allowance) {
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
    ///   `max_tokens`, output credit, context limit and method proposals. The
    ///   method may propose fewer, or return drafter work that starts no
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
            let end = self
                .layout
                .chunk_end(position, self.reconciliation_target, allowance)?;
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
        )
        .min(self.options.method.proposals());
        Ok(RoundPlan::Propose { limit })
    }

    fn next_round(&mut self, request: RequestId, allowance: usize) -> Result<NextRound, String> {
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
                    Vec::new(),
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
                let selects = self.proposal_selects(limit)?;
                let proposal = match self.method.propose(request, &selects) {
                    Propose::Tokens(tokens) => tokens,
                    Propose::Pending(operations) if operations.is_empty() => {
                        return Err("method returned an empty pending operation set".into());
                    }
                    Propose::Pending(operations) => {
                        self.validate_method_operations(request, &operations)?;
                        return Ok(NextRound::Method(operations));
                    }
                };
                if proposal.len() > limit
                    || proposal
                        .iter()
                        .any(|token| token.0 as usize >= self.options.vocabulary)
                {
                    return Err("method returned an invalid proposal".into());
                }
                // A stop token is never accepted as a draft match, so rows
                // after it could only be rejected.
                let proposal = proposal
                    .iter()
                    .take_while(|token| !self.options.stop_tokens.contains(token))
                    .copied()
                    .collect();
                RoundState::verification(
                    *self.generated.last().unwrap(),
                    proposal,
                    limit,
                    self.generated.len(),
                    &self.generated,
                    self.constraint.as_deref(),
                    self.options.sampling,
                    self.options.shaping,
                    self.options.seed,
                    requirements,
                )?
            }
        };
        Ok(NextRound::Target(round))
    }

    /// The prompt chunk from `position` to `end`; the chunk ending the prompt
    /// selects (or forces) the first generated token. A drafter entering
    /// chunks on the device at `drafter` enters, behind the chunk, each
    /// following prompt token with the chunk row before it.
    fn prefill_round(
        &self,
        position: usize,
        end: usize,
        drafter: Option<usize>,
    ) -> Result<RoundState, String> {
        let requirements = self.method_factory.requires();
        let prime = drafter
            .filter(|_| requirements.head && requirements.prefill_demand.contains(Demand::FEATURES))
            .map(|drafter| Priming {
                tokens: self.prompt[position + 1..(end + 1).min(self.prompt.len())].to_vec(),
                position: drafter,
                draft_from: self.prompt.len() + self.generated.len(),
            })
            .filter(|prime| !prime.tokens.is_empty());
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
        RoundState::progress(WorkKind::Prefill, tokens, forced, select, prime, requirements)
    }

    /// See `StartedRound::planned_prefill`.
    fn planned_prefill(
        &self,
        forward: &RoundForward,
        request: RequestId,
        allowance: usize,
    ) -> Result<Option<Operation>, String> {
        if forward.kind != WorkKind::Prefill
            || !forward.selects.is_empty()
            || forward.committed != forward.tokens.len()
            || !self.layout.spans().is_empty()
            || self.finish.is_some()
            || allowance == 0
        {
            return Ok(None);
        }
        let position = self.resident_position + forward.tokens.len();
        if position < self.reconciliation_target || position >= self.prompt.len() {
            return Ok(None);
        }
        let drafter = forward
            .prime
            .as_ref()
            .map(|prime| prime.position + prime.tokens.len());
        let end = self
            .layout
            .chunk_end(position, self.prompt.len(), allowance)?;
        self.prefill_round(position, end, drafter)?
            .forward()
            .clone()
            .into_operation(request, position, None)
            .map(Some)
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
        let progress = self.causal_progress();
        (self.resident && progress.resident_position < progress.accepted_position).then_some(
            PendingReconciliation {
                start: progress.resident_position,
                end: progress.accepted_position,
            },
        )
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
            Vec::new(),
            None,
            None,
            self.method_factory.requires(),
        )?;
        Ok((pending.end, round))
    }

    fn forced_tokens(&self, limit: usize) -> Result<Vec<TokenId>, String> {
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
        if let magnitude_executor::Outcome::Head { proposals } = outcome {
            if proposals.iter().any(|selected| {
                selected.status > 2
                    || (selected.status == 0
                        && selected.token.0 as usize >= self.options.vocabulary)
            }) {
                return Err("head returned an invalid proposal".into());
            }
        }
        let mut method = self.method.fork_transition();
        method.reconcile(operation, outcome.clone())?;
        Ok(PreparedMethodTransition {
            request: *request,
            expected_resident_position: self.resident_position,
            expected_generated_len: self.generated.len(),
            expected_finish: self.finish,
            method,
            decision: ReconcileDecision {
                accepted_rows: tokens.len(),
            },
        })
    }

    pub fn commit_method_transition(&mut self, transition: PreparedMethodTransition) {
        assert!(
            self.resident_position == transition.expected_resident_position
                && self.generated.len() == transition.expected_generated_len
                && self.finish == transition.expected_finish,
            "generation changed between method preparation and physical reconciliation"
        );
        self.method = transition.method;
    }

    /// See `StartedRound::prepare_round_transition`.
    fn prepare_round_transition(
        &self,
        round: &RoundState,
        request: RequestId,
        samples: &[TokenId],
        features: Option<FeatureRef>,
        reader: &mut dyn FeatureReader,
    ) -> Result<PreparedGenerationTransition, String> {
        let causal_reconciliation = self.finish.is_some()
            && round.forward().kind == WorkKind::Replay
            && round.forward().selects.is_empty();
        let mut acceptance = round
            .clone()
            .reconcile(samples, &self.options.stop_tokens)?;
        if self.finish.is_some() && !causal_reconciliation {
            acceptance.emitted.clear();
            acceptance.committed_rows = 0;
            acceptance.proposed = 0;
            acceptance.accepted_proposals = 0;
            acceptance.method_update = MethodUpdate::None;
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
                if self
                    .method_factory
                    .requires()
                    .prefill_demand
                    .contains(Demand::FEATURES)
                    && features.is_none()
                {
                    return Err("prefill method requires target features".into());
                }
                match features {
                    Some(features) => method.prime(
                        request,
                        &acceptance.inputs,
                        acceptance.emitted.first().copied(),
                        features,
                        self.prompt.len() + self.generated.len(),
                        round
                            .forward()
                            .prime
                            .as_ref()
                            .map_or(0, |prime| prime.tokens.len()),
                        reader,
                    )?,
                    None => MethodEffects::default(),
                }
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
        for token in acceptance.emitted {
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
        Ok(PreparedGenerationTransition {
            request,
            expected_resident_position: self.resident_position,
            expected_generated_len: self.generated.len(),
            expected_finish: self.finish,
            expected_output_len: self.output.len(),
            expected_published: self.published,
            expected_forward: round.forward().clone(),
            decision: ReconcileDecision {
                accepted_rows: acceptance.committed_rows,
            },
            method,
            effects,
            constraint,
            generated,
            output,
            draft_stats,
            resident_position,
            accepted_position,
            finish,
        })
    }

    /// See `StartedRound::commit`.
    fn commit_transition(
        &mut self,
        forward: &RoundForward,
        transition: PreparedGenerationTransition,
    ) -> MethodEffects {
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

    /// Accepted output remains owned by the caller until drained or discarded.
    pub fn cancel(&mut self) {
        if self.finish.is_none() {
            self.finish = Some(FinishReason::Cancelled);
        }
    }
    pub fn fail(&mut self) {
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
        // Restore exactly the numerical prefix that existed before eviction.
        // A live decode normally has one accepted successor that has not yet
        // been fed through the model; replaying that token here would advance
        // past the point from which the next decode must start.
        self.reconciliation_target = self.resident_position;
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
            self.cached_tokens = self.cached_tokens.max(position);
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

    pub fn round_forward(&self) -> &RoundForward {
        self.round.forward()
    }

    pub fn round_publication_bound(&self) -> usize {
        self.round.publication_bound()
    }

    /// The prompt chunk of at most `allowance` rows that follows this round
    /// once it commits, when that is known now: the round is a prompt chunk
    /// that selects nothing, and the prompt is plain text (a conditioned
    /// span's rows need its encoding).
    pub fn planned_prefill(
        &self,
        request: RequestId,
        allowance: usize,
    ) -> Result<Option<Operation>, String> {
        self.generation
            .planned_prefill(self.round.forward(), request, allowance)
    }

    /// Prepare acceptance, grammar, method effects, counters, and publication
    /// against private state. The round stays started until the executor has
    /// reconciled the returned decision.
    pub fn prepare_round_transition(
        &self,
        request: RequestId,
        samples: &[TokenId],
        features: Option<FeatureRef>,
        reader: &mut dyn FeatureReader,
    ) -> Result<PreparedGenerationTransition, String> {
        self.generation
            .prepare_round_transition(&self.round, request, samples, features, reader)
    }

    /// Apply a fully checked transition after physical state reconciliation,
    /// ending the round. Any mismatch here is an executor/generation ordering
    /// bug.
    pub fn commit(self, transition: PreparedGenerationTransition) -> (Generation, MethodEffects) {
        let Self {
            mut generation,
            round,
        } = self;
        let effects = generation.commit_transition(round.forward(), transition);
        (generation, effects)
    }

    /// Discard the round after the execution owner releases numerical state.
    /// See `Generation::evicted`.
    pub fn evicted(self) -> Result<Generation, String> {
        let mut generation = self.generation;
        generation.evicted()?;
        Ok(generation)
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
    match choice {
        MethodChoice::Plain => {
            method.identity() == "plain"
                && requirements
                    == (MethodRequirements {
                        prefill_demand: Demand::NONE,
                        verify_demand: Demand::NONE,
                        head: false,
                    })
        }
        MethodChoice::Mtp { proposals } | MethodChoice::DFlash { proposals } => {
            let prefix = match choice {
                MethodChoice::Mtp { .. } => "mtp:",
                _ => "dflash:",
            };
            proposals > 0
                && method.identity().starts_with(prefix)
                && requirements.head
                && requirements.prefill_demand.contains(Demand::FEATURES)
                && requirements.verify_demand.contains(Demand::FEATURES)
                && usize::from(proposals) <= method.proposals()
        }
    }
}

#[cfg(test)]
mod method_choice_tests {
    use super::*;

    struct Descriptor {
        identity: &'static str,
        requirements: MethodRequirements,
        proposals: usize,
    }

    impl Method for Descriptor {
        fn identity(&self) -> &str {
            self.identity
        }
        fn requires(&self) -> MethodRequirements {
            self.requirements
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
            requirements: MethodRequirements {
                prefill_demand: Demand::NONE,
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
            requirements: MethodRequirements {
                prefill_demand: Demand::FEATURES,
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
    }
}
