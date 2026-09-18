//! Request-local logical progress. The execution owner prepares numerical rows;
//! generation accepts them independently after shared physical completion.
pub mod constraints;
mod checkpoint;
pub mod grammar;
pub mod sampling;
pub use checkpoint::Checkpoint;
use crate::inputs::{InputLayout, TokenId};
use crate::models::sequence::Advance;
use std::{
    collections::{BTreeSet, VecDeque},
    sync::atomic::{AtomicU64, Ordering},
};

fn next_identity() -> Result<u64, String> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| "generation identity exhausted".into())
}

/// Request-local grammar state. Staging returns an independent validated successor;
/// it cannot mutate the original state, and installing it cannot fail after commit.
pub trait Constraint {
    fn position(&self) -> usize;
    /// Independent matcher at the same accepted position, including terminal state.
    fn fork(&self) -> Box<dyn Constraint>;
    fn stage(&self, tokens: &[TokenId]) -> Result<Box<dyn Constraint>, String>;
    fn forced(&self, limit: usize) -> Result<Vec<TokenId>, String>;
    fn mask(&self) -> Result<std::sync::Arc<[u32]>, String>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampling {
    Greedy,
    Categorical,
}
/// Accepted token counts, including EOS and tokens suppressed by host stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}
#[derive(Clone, Debug)]
pub struct Options {
    pub max_tokens: usize,
    pub output_capacity: usize,
    pub context_limit: usize,
    pub vocabulary: usize,
    pub stop_tokens: BTreeSet<TokenId>,
    pub sampling: Sampling,
    pub seed: u64,
    pub forced_quantum: usize,
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
pub enum WorkKind {
    Prefill,
    Decode,
    Replay,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitReason {
    Completion,
    Output,
    Finished,
    Residency,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputToken {
    pub index: usize,
    pub token: TokenId,
}

/// Pure logical proposal. Its private owner and revision prevent stale attachment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposal {
    owner: u64,
    revision: u64,
    allowance: usize,
    kind: WorkKind,
    tokens: Vec<TokenId>,
    position: usize,
    sample_position: usize,
    sample: bool,
    forced: Vec<TokenId>,
    sampling: Sampling,
    seed: u64,
}
impl Proposal {
    pub fn kind(&self) -> WorkKind {
        self.kind
    }
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn sample_position(&self) -> usize {
        self.sample_position
    }
    pub fn needs_sample(&self) -> bool {
        self.sample
    }
    pub fn sampling(&self) -> Sampling {
        self.sampling
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn forced(&self) -> &[TokenId] {
        &self.forced
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Readiness {
    Ready(Proposal),
    Wait(WaitReason),
}

struct Pending {
    proposal: Proposal,
    advance: Box<dyn Advance>,
}
pub struct Generation {
    id: u64,
    revision: u64,
    prompt: Vec<TokenId>,
    layout: InputLayout,
    options: Options,
    constraint: Option<Box<dyn Constraint>>,
    generated: Vec<TokenId>,
    output: VecDeque<OutputToken>,
    published: usize,
    processed: usize,
    recovery_position: usize,
    resident: bool,
    finish: Option<FinishReason>,
    pending: Option<Pending>,
}
impl Generation {
    pub fn new(
        prompt: Vec<TokenId>,
        layout: InputLayout,
        options: Options,
        constraint: Option<Box<dyn Constraint>>,
    ) -> Result<Self, String> {
        if prompt.is_empty()
            || prompt.len() != layout.count()
            || prompt.len() > options.context_limit
            || options.context_limit > i32::MAX as usize
            || options.vocabulary == 0
            || options.vocabulary > u32::MAX as usize
            || options.output_capacity == 0
            || options.forced_quantum > 256
            || options.max_tokens > i32::MAX as usize
            || prompt
                .iter()
                .chain(options.stop_tokens.iter())
                .any(|id| id.0 as usize >= options.vocabulary)
            || constraint.as_ref().is_some_and(|c| c.position() != 0)
        {
            return Err("invalid generation input, options, or initial constraint position".into());
        }
        let id = next_identity()?;
        let finish = (options.max_tokens == 0).then_some(FinishReason::Length);
        Ok(Self {
            id,
            revision: 0,
            prompt,
            layout,
            options,
            constraint,
            generated: Vec::new(),
            output: VecDeque::new(),
            published: 0,
            processed: 0,
            recovery_position: 0,
            resident: true,
            finish,
            pending: None,
        })
    }
    pub fn prompt(&self) -> &[TokenId] { &self.prompt }
    pub fn layout(&self) -> &InputLayout { &self.layout }
    pub fn processed(&self) -> usize {
        self.processed
    }
    pub fn accepted_position(&self) -> usize {
        self.processed.max(self.recovery_position)
    }
    pub fn generated(&self) -> &[TokenId] {
        &self.generated
    }
    pub fn usage(&self) -> Usage {
        Usage { prompt_tokens: self.prompt.len(), completion_tokens: self.generated.len() }
    }
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish
    }
    pub fn output_len(&self) -> usize {
        self.output.len()
    }
    pub fn awaiting_completion(&self) -> bool {
        self.pending.is_some()
    }
    pub fn completion_ready(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| pending.advance.is_complete())
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

    pub fn ready(&self, allowance: usize) -> Result<Readiness, String> {
        self.ready_impl(allowance, false)
    }
    pub fn is_resident(&self) -> bool {
        self.resident
    }
    /// Pure work preview for a fresh numerical state. The executor must restore
    /// it atomically with preparation; `restored` then makes this proposal live.
    pub fn ready_for_recovery(&self, allowance: usize) -> Result<Readiness, String> {
        if self.resident {
            return Err("recovery preview requires an evicted sequence".into());
        }
        self.ready_impl(allowance, true)
    }
    fn ready_impl(&self, allowance: usize, recovery: bool) -> Result<Readiness, String> {
        if allowance == 0 {
            return Err("service allowance must be positive".into());
        }
        let waiting = if self.pending.is_some() {
            Some(WaitReason::Completion)
        } else if self.finish.is_some() {
            Some(WaitReason::Finished)
        } else if self.output.len() >= self.options.output_capacity {
            Some(WaitReason::Output)
        } else if !self.resident && !recovery {
            Some(WaitReason::Residency)
        } else {
            None
        };
        if let Some(reason) = waiting {
            return Ok(Readiness::Wait(reason));
        }
        let position = if recovery { 0 } else { self.processed };
        let (kind, mut tokens, mut sample, forced_limit) = if position < self.recovery_position {
            let end = self
                .layout
                .chunk_end(position, self.recovery_position, allowance)?;
            let tokens = self
                .prompt
                .iter()
                .chain(&self.generated)
                .skip(position)
                .take(end - position)
                .copied()
                .collect();
            (WorkKind::Replay, tokens, false, 0)
        } else if position < self.prompt.len() {
            let end = self
                .layout
                .chunk_end(position, self.prompt.len(), allowance)?;
            let sample = end == self.prompt.len();
            (
                WorkKind::Prefill,
                self.prompt[position..end].to_vec(),
                sample,
                usize::from(sample),
            )
        } else {
            if self.generated.is_empty()
                || position != self.prompt.len() + self.generated.len() - 1
                || position >= self.options.context_limit
            {
                return Err("generation history and numerical continuation disagree".into());
            }
            (
                WorkKind::Decode,
                vec![*self.generated.last().unwrap()],
                true,
                allowance.min(self.options.context_limit - position),
            )
        };
        let limit = forced_limit
            .min(self.options.forced_quantum)
            .min(self.options.max_tokens - self.generated.len())
            .min(self.options.output_capacity - self.output.len());
        let forced = if limit > 0 {
            match &self.constraint {
                Some(constraint) => {
                    let forced = constraint.forced(limit)?;
                    if forced.len() > limit
                        || forced
                            .iter()
                            .any(|t| t.0 as usize >= self.options.vocabulary)
                    {
                        return Err("constraint returned an invalid forced run".into());
                    }
                    forced
                        .into_iter()
                        .take_while(|t| !self.options.stop_tokens.contains(t))
                        .collect::<Vec<_>>()
                }
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        if !forced.is_empty() {
            sample = false;
            if kind == WorkKind::Decode {
                tokens.extend_from_slice(&forced[..forced.len() - 1]);
            }
        }
        Ok(Readiness::Ready(Proposal {
            owner: self.id,
            revision: if recovery {
                self.revision
                    .checked_add(1)
                    .ok_or("generation revision exhausted")?
            } else {
                self.revision
            },
            allowance,
            kind,
            tokens,
            position,
            sample_position: self.generated.len(),
            sample,
            forced,
            sampling: self.options.sampling,
            seed: self.options.seed,
        }))
    }
    /// Call after whole-batch preparation succeeds. Failure drops only the supplied
    /// tentative row; a pure proposal itself never acquired capacity or state.
    pub fn attach(&mut self, proposal: Proposal, advance: Box<dyn Advance>) -> Result<(), String> {
        if self.ready(proposal.allowance)? != Readiness::Ready(proposal.clone()) {
            return Err("generation proposal is no longer ready".into());
        }
        self.pending = Some(Pending { proposal, advance });
        Ok(())
    }
    /// Returns false while physical work is outstanding. Completion reconciliation
    /// remains necessary after cancellation, even though no new output is accepted.
    pub fn reconcile(&mut self) -> Result<bool, String> {
        let pending = self
            .pending
            .as_ref()
            .ok_or("generation has no pending work")?;
        if !pending.advance.is_complete() {
            return Ok(false);
        }
        let mut pending = self.pending.take().unwrap();
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or("generation revision exhausted")?;
        if self.finish.is_some() {
            return Ok(true);
        }
        let result = (|| {
            let selected = pending.advance.selected()?;
            let proposal = &pending.proposal;
            if proposal.sample != selected.is_some() {
                return Err("model selection differs from requested readout".into());
            }
            let accepted = if proposal.forced.is_empty() {
                selected.into_iter().collect::<Vec<_>>()
            } else {
                proposal.forced.clone()
            };
            if accepted
                .iter()
                .any(|t| t.0 as usize >= self.options.vocabulary)
            {
                return Err("selected token is outside vocabulary".into());
            }
            let staged = if accepted.is_empty() {
                None
            } else {
                self.constraint
                    .as_ref()
                    .map(|constraint| -> Result<Box<dyn Constraint>, String> {
                        if constraint.position() != self.generated.len() {
                            return Err("constraint progress differs from accepted tokens".into());
                        }
                        let next = constraint.stage(&accepted)?;
                        if next.position() != self.generated.len() + accepted.len() {
                            return Err("constraint successor has incorrect position".into());
                        }
                        Ok(next)
                    })
                    .transpose()?
            };
            pending.advance.commit()?;
            // Everything after numerical commitment is an infallible logical install.
            self.processed += proposal.tokens.len();
            if let Some(constraint) = staged {
                self.constraint = Some(constraint);
            }
            for token in accepted {
                self.generated.push(token);
                if self.options.stop_tokens.contains(&token) {
                    self.finish = Some(FinishReason::Stop);
                } else {
                    self.output.push_back(OutputToken {
                        index: self.published + self.output.len(),
                        token,
                    });
                }
            }
            if self.finish.is_none() && self.generated.len() >= self.options.max_tokens {
                self.finish = Some(FinishReason::Length);
            }
            if self.finish.is_none() && self.processed >= self.options.context_limit {
                self.finish = Some(FinishReason::Context);
            }
            Ok(())
        })();
        if result.is_err() {
            self.finish = Some(FinishReason::Failed);
        }
        result.map(|_| true)
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
        if self.pending.is_some() {
            return Err("eviction requires reconciled work".into());
        }
        self.recovery_position = self.accepted_position();
        self.resident = false;
        Ok(())
    }
    /// Called after fresh numerical state for the same retained input is installed.
    /// Replay advances to the retained acceptance boundary without sampling.
    pub fn restored(&mut self) -> Result<(), String> {
        if self.resident || self.pending.is_some() || self.finish.is_some() {
            return Err("only an evicted live request can restore".into());
        }
        self.processed = 0;
        self.resident = true;
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or("generation revision exhausted")?;
        Ok(())
    }
}
