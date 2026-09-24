//! Family-neutral request-local progress. The executor prepares numerical rows;
//! generation accepts them independently after shared physical completion.
mod acceptance;
pub mod grammar;
mod method;
mod mtp;
mod plain;
mod round;
mod shaping;
pub use acceptance::accept_prefix;
pub use magnitude_artifacts::{BoundaryRule, InputLayout, InputSpan};
pub use magnitude_model_executor::{
    Demand, FeatureRef, FeatureRetainer, Operation, RequestId, Sampling, SelectSpec, Shaping,
    TokenId, WorkKind,
};
pub use method::{
    Method, MethodCheckpoint, MethodCheckpointError, MethodChoice, MethodEffects,
    MethodRequirements, MethodState, MtpCheckpoint, Propose, Verification,
};
pub use mtp::Mtp;
pub use plain::Plain;
pub use round::{MethodUpdate, RoundAcceptance, RoundForward, RoundState};
pub use shaping::{HISTORY_WIDTH, selection_history, verification_selects};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetailedUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub draft_n: usize,
    pub draft_n_accepted: usize,
}
#[derive(Clone, Debug)]
pub struct Options {
    pub max_tokens: usize,
    pub output_capacity: usize,
    pub context_limit: usize,
    pub vocabulary: usize,
    pub stop_tokens: BTreeSet<TokenId>,
    pub sampling: Sampling,
    pub shaping: Shaping,
    pub seed: u64,
    pub forced_quantum: usize,
    pub method: MethodChoice,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Context,
    Cancelled,
    Failed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitReason {
    Completion,
    Output,
    Finished,
    Residency,
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputToken {
    pub index: usize,
    pub token: TokenId,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RoundStart {
    Target,
    Method(Vec<Operation>),
}

struct ResolvedRound {
    acceptance: RoundAcceptance,
    constraint: Option<Box<dyn Constraint>>,
}

/// The sole numerical decision generation sends back to the executor. A
/// verified prefix may need physical repair before the prepared transition is
/// committed, but its accepted length cannot change during that repair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReconcileDecision {
    pub accepted_rows: usize,
    pub head_prefix: Option<usize>,
}

/// Every fallible logical change is performed against private state before
/// physical reconciliation. Dropping this value leaves the live generation
/// untouched, including its suspended round and method state.
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

pub struct PreparedMethodTransition {
    request: RequestId,
    expected_preview_len: Option<usize>,
    expected_resident_position: usize,
    expected_generated_len: usize,
    expected_finish: Option<FinishReason>,
    method: Box<dyn MethodState>,
    preview: Option<MethodPreview>,
    effects: MethodEffects,
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

enum SuspendedRound {
    Awaiting(RoundState),
    Resolved(ResolvedRound),
}

struct MethodPreview {
    request: RequestId,
    limit: usize,
    tokens: Vec<TokenId>,
    constraint: Option<Box<dyn Constraint>>,
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
    round: Option<SuspendedRound>,
    method_preview: Option<MethodPreview>,
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
        let method = method_factory.create(None);
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
            resident: true,
            finish,
            method_factory,
            method,
            cached_tokens: 0,
            draft_stats: DraftStats::default(),
            round: None,
            method_preview: None,
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
    /// Export owned method state for prefix retention. Transient features are
    /// copied through the executor-domain retainer before the checkpoint leaves
    /// the live request.
    pub fn method_checkpoint(
        &self,
        retainer: &mut dyn FeatureRetainer,
    ) -> Result<MethodCheckpoint, String> {
        if self.round.is_some() || self.method_preview.is_some() || !self.resident {
            return Err("method checkpoint requires reconciled resident state".into());
        }
        self.method
            .checkpoint(retainer)
            .map_err(|error| error.to_string())
    }

    /// Restore a fresh generation at an exact retained prompt boundary. Output,
    /// grammar, and usage remain those of this fresh request; only numerical
    /// progress and owned method state are restored.
    pub fn restore_prefix(
        &mut self,
        position: usize,
        checkpoint: &MethodCheckpoint,
    ) -> Result<(), String> {
        if self.resident_position != 0
            || self.accepted_position != 0
            || self.reconciliation_target != 0
            || !self.generated.is_empty()
            || !self.output.is_empty()
            || self.round.is_some()
            || self.method_preview.is_some()
            || position > self.prompt.len()
            || !self.layout.boundary(position)
        {
            return Err("retained prefix does not match a fresh exact input boundary".into());
        }
        match (self.options.method, checkpoint) {
            (MethodChoice::Plain, MethodCheckpoint::Plain) => {}
            (MethodChoice::Mtp { .. }, MethodCheckpoint::Mtp(state))
                if state.position() == position => {}
            _ => {
                return Err(
                    "retained method checkpoint does not match the fresh generation".into(),
                );
            }
        }
        self.method = self.method_factory.create(Some(checkpoint));
        self.resident_position = position;
        self.accepted_position = position;
        self.reconciliation_target = position;
        self.cached_tokens = position;
        Ok(())
    }
    pub fn stabilize_method_features(
        &mut self,
        retainer: &mut dyn FeatureRetainer,
    ) -> Result<(), String> {
        self.method.stabilize(retainer)
    }
    fn propose_method(&mut self, request: RequestId, limit: usize, select: SelectSpec) -> Propose {
        let context = self
            .prompt
            .iter()
            .chain(&self.generated)
            .copied()
            .collect::<Vec<_>>();
        self.method.propose(request, &context, limit, select)
    }
    pub fn credit_cached_tokens(&mut self, count: usize) -> Result<(), String> {
        if count > self.prompt.len() || count < self.cached_tokens {
            return Err("cached-token credit must be monotonic and within the prompt".into());
        }
        self.cached_tokens = count;
        Ok(())
    }
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish
    }
    pub fn output_len(&self) -> usize {
        self.output.len()
    }
    pub fn awaiting_completion(&self) -> bool {
        self.round.is_some()
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
    pub fn wait_reason(&self) -> Option<WaitReason> {
        if self.round.is_some() {
            Some(WaitReason::Completion)
        } else if self.finish.is_some() {
            Some(WaitReason::Finished)
        } else if self.output.len() >= self.options.output_capacity {
            Some(WaitReason::Output)
        } else if !self.resident {
            Some(WaitReason::Residency)
        } else {
            None
        }
    }

    /// Build and suspend the next target round. Method work is returned as
    /// ordinary executor operations; callers retry after reconciling it.
    pub fn start_round(
        &mut self,
        request: RequestId,
        allowance: usize,
    ) -> Result<RoundStart, String> {
        if allowance == 0 {
            return Err("service allowance must be positive".into());
        }
        if let Some(reason) = self.wait_reason() {
            return Err(format!("generation cannot start a round while {reason:?}"));
        }
        let position = self.resident_position;
        let requirements = self.method_factory.requires();
        let round = if position < self.reconciliation_target {
            let end = self
                .layout
                .chunk_end(position, self.reconciliation_target, allowance)?;
            let tokens = self
                .prompt
                .iter()
                .chain(&self.generated)
                .skip(position)
                .take(end - position)
                .copied()
                .collect();
            RoundState::progress(WorkKind::Replay, tokens, Vec::new(), None, requirements)?
        } else if position < self.prompt.len() {
            let end = self
                .layout
                .chunk_end(position, self.prompt.len(), allowance)?;
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
            RoundState::progress(WorkKind::Prefill, tokens, forced, select, requirements)?
        } else {
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
                RoundState::forced(*self.generated.last().unwrap(), forced, requirements)?
            } else {
                let limit = RoundState::proposal_limit(
                    allowance.min(self.options.context_limit - position),
                    remaining,
                    credit,
                    self.options.method == MethodChoice::Plain,
                );
                let (method_limit, select) = match &self.method_preview {
                    Some(preview) if preview.request != request => {
                        return Err("method preview belongs to another request".into());
                    }
                    Some(preview) => (
                        preview.limit,
                        method_select(
                            &self.options,
                            &self.generated,
                            &preview.tokens,
                            preview.constraint.as_deref(),
                        )?,
                    ),
                    None => (
                        limit,
                        method_select(
                            &self.options,
                            &self.generated,
                            &[],
                            self.constraint.as_deref(),
                        )?,
                    ),
                };
                let proposal = match self.propose_method(request, method_limit, select) {
                    Propose::Tokens(tokens) => {
                        if let Some(preview) = self.method_preview.take() {
                            if tokens != preview.tokens {
                                return Err(
                                    "method proposal differs from reconciled preview".into()
                                );
                            }
                        }
                        tokens
                    }
                    Propose::Pending(operations) if operations.is_empty() => {
                        return Err("method returned an empty pending operation set".into());
                    }
                    Propose::Pending(operations) => {
                        if self.method_preview.is_none() {
                            self.method_preview = Some(MethodPreview {
                                request,
                                limit: method_limit,
                                tokens: Vec::new(),
                                constraint: self
                                    .constraint
                                    .as_ref()
                                    .map(|constraint| constraint.fork()),
                            });
                        }
                        return Ok(RoundStart::Method(operations));
                    }
                };
                if proposal.len() > method_limit
                    || proposal
                        .iter()
                        .any(|token| token.0 as usize >= self.options.vocabulary)
                {
                    return Err("method returned an invalid proposal".into());
                }
                RoundState::verification(
                    *self.generated.last().unwrap(),
                    proposal,
                    method_limit,
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
        self.round = Some(SuspendedRound::Awaiting(round));
        Ok(RoundStart::Target)
    }

    /// Numerical state may trail logically accepted tokens because selection
    /// commits the input row that produced a successor, not the successor row
    /// itself. This fact is independent of why a checkpoint is requested.
    pub fn pending_reconciliation(&self) -> Option<PendingReconciliation> {
        let progress = self.causal_progress();
        (self.resident
            && self.round.is_none()
            && self.method_preview.is_none()
            && progress.resident_position < progress.accepted_position)
            .then_some(PendingReconciliation {
                start: progress.resident_position,
                end: progress.accepted_position,
            })
    }

    /// Advance accepted successors without selecting or publishing another
    /// token. Callers use this before any operation requiring exact numerical
    /// state, including but not limited to retention.
    pub fn start_reconciliation(&mut self, allowance: usize) -> Result<(), String> {
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
        self.reconciliation_target = pending.end;
        let round = RoundState::progress(
            WorkKind::Replay,
            tokens,
            Vec::new(),
            None,
            self.method_factory.requires(),
        )?;
        self.round = Some(SuspendedRound::Awaiting(round));
        Ok(())
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

    pub fn round_forward(&self) -> Option<&RoundForward> {
        match self.round.as_ref() {
            Some(SuspendedRound::Awaiting(round)) => Some(round.forward()),
            Some(SuspendedRound::Resolved(_)) | None => None,
        }
    }

    pub fn prepare_method_transition(
        &self,
        operation: &Operation,
        outcome: &magnitude_model_executor::Outcome,
        retainer: &mut dyn FeatureRetainer,
    ) -> Result<PreparedMethodTransition, String> {
        if self.round.is_some() {
            return Err("method work cannot reconcile while a target round is suspended".into());
        }
        let request = operation.request();
        let mut method = self.method.fork_transition()?;
        let mut preview = self.method_preview.as_ref().map(|preview| MethodPreview {
            request: preview.request,
            limit: preview.limit,
            tokens: preview.tokens.clone(),
            constraint: preview
                .constraint
                .as_ref()
                .map(|constraint| constraint.fork()),
        });
        let next_select = if let Some(preview) = preview.as_mut() {
            if request != preview.request {
                return Err("method outcome belongs to another request".into());
            }
            let staged_token = match (operation, outcome) {
                (Operation::Head { .. }, magnitude_model_executor::Outcome::Head { .. }) => None,
                (
                    Operation::Project { .. },
                    magnitude_model_executor::Outcome::Project { selected },
                ) => {
                    let [selected] = selected.as_slice() else {
                        return Err(
                            "method projection must return exactly one selected token".into()
                        );
                    };
                    match selected.status {
                        0 => {}
                        1 => return Err("method projection distribution is empty".into()),
                        2 => return Err("method projection distribution is nonfinite".into()),
                        _ => return Err("method projection returned an unknown status".into()),
                    }
                    if selected.token.0 as usize >= self.options.vocabulary
                        || preview.tokens.len() >= preview.limit
                    {
                        return Err("method projection returned an invalid proposal token".into());
                    }
                    if let Some(constraint) = preview.constraint.as_ref() {
                        let next = constraint.stage(&[selected.token])?;
                        if next.position() != constraint.position() + 1 {
                            return Err("method preview advanced to an invalid position".into());
                        }
                        preview.constraint = Some(next);
                    }
                    preview.tokens.push(selected.token);
                    Some(selected.token)
                }
                _ => return Err("method operation returned an outcome of the wrong kind".into()),
            };
            (preview.tokens.len() < preview.limit
                && staged_token.is_none_or(|token| !self.options.stop_tokens.contains(&token)))
            .then(|| {
                method_select(
                    &self.options,
                    &self.generated,
                    &preview.tokens,
                    preview.constraint.as_deref(),
                )
            })
            .transpose()?
        } else {
            if !matches!(
                (operation, outcome),
                (
                    Operation::Head { .. },
                    magnitude_model_executor::Outcome::Head { .. }
                )
            ) {
                return Err(
                    "only a priming head outcome may reconcile without a proposal preview".into(),
                );
            }
            None
        };
        let effects = method.reconcile(operation, outcome.clone(), next_select)?;
        self.validate_method_effects(&effects)?;
        if effects
            .operations
            .iter()
            .any(|operation| operation.request() != request)
        {
            return Err("method returned a follow-up operation for another request".into());
        }
        if let Some(prefix) = effects.head_prefix {
            if !matches!(operation, Operation::Head { .. }) || prefix > operation.row_count() {
                return Err("method accepted an invalid head prefix".into());
            }
        }
        method.stabilize(retainer)?;
        Ok(PreparedMethodTransition {
            request,
            expected_preview_len: self
                .method_preview
                .as_ref()
                .map(|preview| preview.tokens.len()),
            expected_resident_position: self.resident_position,
            expected_generated_len: self.generated.len(),
            expected_finish: self.finish,
            method,
            preview,
            decision: ReconcileDecision {
                accepted_rows: 0,
                head_prefix: effects.head_prefix,
            },
            effects,
        })
    }

    pub fn commit_method_transition(
        &mut self,
        transition: PreparedMethodTransition,
    ) -> MethodEffects {
        assert!(
            self.round.is_none()
                && self
                    .method_preview
                    .as_ref()
                    .map(|preview| preview.tokens.len())
                    == transition.expected_preview_len
                && self.resident_position == transition.expected_resident_position
                && self.generated.len() == transition.expected_generated_len
                && self.finish == transition.expected_finish,
            "method preview changed between preparation and physical reconciliation"
        );
        self.method = transition.method;
        self.method_preview = transition.preview;
        transition.effects
    }

    pub fn reconcile_method(
        &mut self,
        operation: &Operation,
        outcome: magnitude_model_executor::Outcome,
    ) -> Result<MethodEffects, String> {
        match self.reconcile_method_inner(operation, outcome) {
            Ok(effects) => Ok(effects),
            Err(error) => {
                self.method_preview = None;
                self.finish = Some(FinishReason::Failed);
                Err(error)
            }
        }
    }

    fn reconcile_method_inner(
        &mut self,
        operation: &Operation,
        outcome: magnitude_model_executor::Outcome,
    ) -> Result<MethodEffects, String> {
        if self.round.is_some() {
            return Err("method work cannot reconcile while a target round is suspended".into());
        }
        let Some(preview) = self.method_preview.as_ref() else {
            if !matches!(
                (&operation, &outcome),
                (
                    Operation::Head { .. },
                    magnitude_model_executor::Outcome::Head { .. }
                )
            ) {
                return Err(
                    "only a priming head outcome may reconcile without a proposal preview".into(),
                );
            }
            let request = operation.request();
            let effects = self.method.reconcile(operation, outcome, None)?;
            if effects
                .operations
                .iter()
                .any(|operation| operation.request() != request)
            {
                return Err("method returned a follow-up operation for another request".into());
            }
            self.validate_method_effects(&effects)?;
            return Ok(effects);
        };
        if operation.request() != preview.request {
            return Err("method outcome belongs to another request".into());
        }
        let request = preview.request;

        let mut staged = None;
        let mut staged_token = None;
        match (operation, &outcome) {
            (Operation::Head { .. }, magnitude_model_executor::Outcome::Head { .. }) => {}
            (
                Operation::Project { .. },
                magnitude_model_executor::Outcome::Project { selected },
            ) => {
                let [selected] = selected.as_slice() else {
                    return Err("method projection must return exactly one selected token".into());
                };
                match selected.status {
                    0 => {}
                    1 => return Err("method projection distribution is empty".into()),
                    2 => return Err("method projection distribution is nonfinite".into()),
                    _ => return Err("method projection returned an unknown status".into()),
                }
                if selected.token.0 as usize >= self.options.vocabulary
                    || preview.tokens.len() >= preview.limit
                {
                    return Err("method projection returned an invalid proposal token".into());
                }
                if let Some(constraint) = preview.constraint.as_ref() {
                    let next = constraint.stage(&[selected.token])?;
                    if next.position() != constraint.position() + 1 {
                        return Err("method preview advanced to an invalid position".into());
                    }
                    staged = Some(next);
                }
                staged_token = Some(selected.token);
            }
            _ => return Err("method operation returned an outcome of the wrong kind".into()),
        }

        let mut next_tokens = preview.tokens.clone();
        if let Some(token) = staged_token {
            next_tokens.push(token);
        }
        let next_constraint = staged.as_deref().or(preview.constraint.as_deref());
        let next_select = (next_tokens.len() < preview.limit
            && staged_token.is_none_or(|token| !self.options.stop_tokens.contains(&token)))
        .then(|| {
            method_select(
                &self.options,
                &self.generated,
                &next_tokens,
                next_constraint,
            )
        })
        .transpose()?;
        let effects = self.method.reconcile(operation, outcome, next_select)?;
        if effects
            .operations
            .iter()
            .any(|operation| operation.request() != request)
        {
            return Err("method returned a follow-up operation for another request".into());
        }
        let preview = self.method_preview.as_mut().unwrap();
        if let Some(token) = staged_token {
            preview.tokens.push(token);
        }
        if staged.is_some() {
            preview.constraint = staged;
        }
        self.validate_method_effects(&effects)?;
        Ok(effects)
    }

    /// Prepare acceptance, grammar, method effects, counters, and publication
    /// against private state. The live round remains suspended until the
    /// executor has reconciled the returned decision.
    pub fn prepare_round_transition(
        &self,
        request: RequestId,
        samples: &[TokenId],
        features: Option<FeatureRef>,
        retainer: &mut dyn FeatureRetainer,
    ) -> Result<PreparedGenerationTransition, String> {
        let Some(SuspendedRound::Awaiting(round)) = self.round.as_ref() else {
            return Err("generation has no target round awaiting selections".into());
        };
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
        let mut method = self.method.fork_transition()?;
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
                    Some(features) => method.prime(request, &acceptance.inputs, features)?,
                    None => MethodEffects::default(),
                }
            }
            MethodUpdate::Observe => {
                let next = *acceptance
                    .emitted
                    .last()
                    .ok_or("resolved verification emitted no successor token")?;
                method.observe(Verification {
                    inputs: &acceptance.inputs,
                    accepted: acceptance.committed_rows,
                    next,
                    features,
                })?
            }
        };
        self.validate_method_effects(&effects)?;
        if effects
            .operations
            .iter()
            .any(|operation| operation.request() != request)
        {
            return Err("method returned an operation for another request".into());
        }
        method.stabilize(retainer)?;
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
                head_prefix: effects.head_prefix,
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

    /// Apply a fully checked transition after physical state reconciliation.
    /// Any mismatch here is an executor/generation ordering bug.
    pub fn commit_transition(&mut self, transition: PreparedGenerationTransition) -> MethodEffects {
        assert!(
            matches!(self.round.as_ref(), Some(SuspendedRound::Awaiting(_)))
                && self.resident_position == transition.expected_resident_position
                && self.generated.len() == transition.expected_generated_len
                && self.finish == transition.expected_finish
                && self.output.len() == transition.expected_output_len
                && self.published == transition.expected_published
                && self.round_forward() == Some(&transition.expected_forward),
            "generation changed between preparation and physical reconciliation"
        );
        self.round = None;
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

    /// Reconcile device selections and stage grammar on a private matcher. No
    /// logical token is published until the caller has committed the numerical
    /// prefix and invokes `commit_round`.
    pub fn resolve_round(&mut self, samples: &[TokenId]) -> Result<&RoundAcceptance, String> {
        let Some(SuspendedRound::Awaiting(round)) = self.round.take() else {
            return Err("generation has no target round awaiting selections".into());
        };
        // Reconciliation started after a finish decision commits already
        // accepted tokens into numerical and method state without selecting.
        let causal_reconciliation = self.finish.is_some()
            && round.forward().kind == WorkKind::Replay
            && round.forward().selects.is_empty();
        let mut acceptance = match round.reconcile(samples, &self.options.stop_tokens) {
            Ok(acceptance) => acceptance,
            Err(error) => {
                self.finish = Some(FinishReason::Failed);
                return Err(error);
            }
        };
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
            || acceptance.emitted.len() > self.options.output_capacity - self.output.len()
            || acceptance.emitted.len() > self.options.max_tokens - self.generated.len()
            || (!causal_reconciliation
                && acceptance.committed_rows
                    > self
                        .options
                        .context_limit
                        .saturating_sub(self.resident_position))
        {
            self.finish = Some(FinishReason::Failed);
            return Err("round returned invalid tokens or exceeded a generation bound".into());
        }
        let constraint = if acceptance.emitted.is_empty() {
            Ok(None)
        } else {
            self.constraint
                .as_ref()
                .map(|constraint| {
                    if constraint.position() != self.generated.len() {
                        return Err("constraint progress differs from accepted tokens".into());
                    }
                    let next = constraint.stage(&acceptance.emitted)?;
                    if next.position() != self.generated.len() + acceptance.emitted.len() {
                        return Err("constraint successor has incorrect position".into());
                    }
                    Ok(next)
                })
                .transpose()
        };
        let constraint = match constraint {
            Ok(constraint) => constraint,
            Err(error) => {
                self.finish = Some(FinishReason::Failed);
                return Err(error);
            }
        };
        self.round = Some(SuspendedRound::Resolved(ResolvedRound {
            acceptance,
            constraint,
        }));
        let Some(SuspendedRound::Resolved(resolved)) = self.round.as_ref() else {
            unreachable!()
        };
        Ok(&resolved.acceptance)
    }

    /// Install a staged round after its numerical prefix (and any repair) has
    /// committed. This is the only round path that publishes tokens or usage.
    pub fn commit_round(
        &mut self,
        request: RequestId,
        features: Option<FeatureRef>,
    ) -> Result<MethodEffects, String> {
        let Some(SuspendedRound::Resolved(resolved)) = self.round.take() else {
            return Err("generation has no resolved round to commit".into());
        };
        let method_effects = match resolved.acceptance.method_update {
            MethodUpdate::None => MethodEffects::default(),
            MethodUpdate::Prime => {
                if self
                    .method_factory
                    .requires()
                    .prefill_demand
                    .contains(Demand::FEATURES)
                    && features.is_none()
                {
                    self.finish = Some(FinishReason::Failed);
                    return Err("prefill method requires target features".into());
                }
                match features {
                    Some(features) => {
                        self.method
                            .prime(request, &resolved.acceptance.inputs, features)?
                    }
                    None => MethodEffects::default(),
                }
            }
            MethodUpdate::Observe => {
                let next = *resolved
                    .acceptance
                    .emitted
                    .last()
                    .ok_or("resolved verification emitted no successor token")?;
                match self.method.observe(Verification {
                    inputs: &resolved.acceptance.inputs,
                    accepted: resolved.acceptance.committed_rows,
                    next,
                    features,
                }) {
                    Ok(effects) => effects,
                    Err(error) => {
                        self.finish = Some(FinishReason::Failed);
                        return Err(error);
                    }
                }
            }
        };
        self.validate_method_effects(&method_effects)?;
        if method_effects
            .operations
            .iter()
            .any(|operation| operation.request() != request)
        {
            return Err("method returned an operation for another request".into());
        }
        self.draft_stats.record(
            resolved.acceptance.proposed,
            resolved.acceptance.accepted_proposals,
        )?;
        self.resident_position = self
            .resident_position
            .checked_add(resolved.acceptance.committed_rows)
            .ok_or("resident position exhausted")?;
        if let Some(constraint) = resolved.constraint {
            self.constraint = Some(constraint);
        }
        for token in resolved.acceptance.emitted {
            self.generated.push(token);
            if self.options.stop_tokens.contains(&token) {
                self.finish = Some(FinishReason::Stop);
                break;
            }
            self.output.push_back(OutputToken {
                index: self.published + self.output.len(),
                token,
            });
        }
        if self.finish.is_none() && self.generated.len() >= self.options.max_tokens {
            self.finish = Some(FinishReason::Length);
        }
        self.accepted_position = self
            .resident_position
            .max(self.prompt.len() + self.generated.len());
        if self.finish.is_none() && self.resident_position >= self.options.context_limit {
            self.finish = Some(FinishReason::Context);
        }
        Ok(method_effects)
    }

    fn validate_method_effects(&self, effects: &MethodEffects) -> Result<(), String> {
        if effects.operations.iter().any(|operation| {
            operation.executable() == magnitude_model_executor::ExecutableKind::Target
        }) || (effects.head_prefix.is_some() && !self.method_factory.requires().head)
        {
            return Err("method returned effects outside its executor ownership".into());
        }
        Ok(())
    }

    /// Accepted output remains owned by the caller until drained or discarded.
    pub fn cancel(&mut self) {
        self.round = None;
        self.method_preview = None;
        if self.finish.is_none() {
            self.finish = Some(FinishReason::Cancelled);
        }
    }
    pub fn fail(&mut self) {
        self.round = None;
        self.method_preview = None;
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
        self.reconciliation_target = self.accepted_position;
        self.round = None;
        self.method_preview = None;
        self.method.evict();
        self.resident = false;
        Ok(())
    }
    /// Called after fresh numerical state for the same retained input is installed.
    /// Replay advances to the retained acceptance boundary without sampling.
    pub fn restored(&mut self) -> Result<(), String> {
        if self.resident || self.finish.is_some() {
            return Err("only an evicted live request can restore".into());
        }
        self.resident_position = 0;
        self.method.restore();
        self.resident = true;
        Ok(())
    }

    /// Fork reconciled logical state at the executor's independently checked
    /// numerical checkpoint position.
    pub fn fork_at(
        &self,
        numerical_position: usize,
        retainer: &mut dyn FeatureRetainer,
    ) -> Result<Self, String> {
        if self.round.is_some()
            || self.method_preview.is_some()
            || !self.resident
            || numerical_position != self.resident_position
        {
            return Err("checkpoint requires matching reconciled resident state".into());
        }
        self.fork_reconciled(retainer)
    }

    fn fork_reconciled(&self, retainer: &mut dyn FeatureRetainer) -> Result<Self, String> {
        let constraint = self
            .constraint
            .as_ref()
            .map(|constraint| -> Result<_, String> {
                let fork = constraint.fork();
                if fork.position() != constraint.position() {
                    return Err("checkpoint constraint fork changed its position".into());
                }
                Ok(fork)
            })
            .transpose()?;
        let method_checkpoint = self
            .method
            .checkpoint(retainer)
            .map_err(|error| error.to_string())?;
        let method = self.method_factory.create(Some(&method_checkpoint));
        Ok(Self {
            prompt: self.prompt.clone(),
            layout: self.layout.clone(),
            options: self.options.clone(),
            constraint,
            generated: self.generated.clone(),
            output: self.output.clone(),
            published: self.published,
            resident_position: self.resident_position,
            accepted_position: self.accepted_position,
            reconciliation_target: self.reconciliation_target,
            resident: self.resident,
            finish: self.finish,
            method_factory: self.method_factory.clone(),
            method,
            cached_tokens: self.cached_tokens,
            draft_stats: self.draft_stats,
            round: None,
            method_preview: None,
        })
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

fn method_select(
    options: &Options,
    generated: &[TokenId],
    preview: &[TokenId],
    constraint: Option<&dyn Constraint>,
) -> Result<SelectSpec, String> {
    Ok(SelectSpec {
        sampling: options.sampling,
        seed: options.seed,
        position: generated
            .len()
            .checked_add(preview.len())
            .ok_or("method selection position exhausted")?,
        domain: 1,
        mask: constraint.map(Constraint::mask).transpose()?,
        shaping: options.shaping,
        history: options
            .shaping
            .uses_history()
            .then(|| selection_history(generated, preview))
            .transpose()?,
    })
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
        MethodChoice::Mtp { proposals } => {
            let capacity = method
                .identity()
                .strip_prefix("mtp:")
                .and_then(|identity| identity.rsplit_once(':'))
                .and_then(|(_, capacity)| capacity.parse::<usize>().ok());
            proposals > 0
                && requirements.head
                && requirements.prefill_demand.contains(Demand::FEATURES)
                && requirements.verify_demand.contains(Demand::FEATURES)
                && capacity.is_some_and(|capacity| usize::from(proposals) <= capacity)
        }
    }
}

#[cfg(test)]
mod method_choice_tests {
    use super::*;

    struct Descriptor {
        identity: &'static str,
        requirements: MethodRequirements,
    }

    impl Method for Descriptor {
        fn identity(&self) -> &str {
            self.identity
        }
        fn requires(&self) -> MethodRequirements {
            self.requirements
        }
        fn create(&self, _checkpoint: Option<&MethodCheckpoint>) -> Box<dyn MethodState> {
            panic!("descriptor-only validation fixture")
        }
    }

    #[test]
    fn choice_requires_matching_identity_requirements_and_mtp_capacity() {
        let plain = Descriptor {
            identity: "plain",
            requirements: MethodRequirements {
                prefill_demand: Demand::NONE,
                verify_demand: Demand::NONE,
                head: false,
            },
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
