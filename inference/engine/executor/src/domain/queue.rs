//! Steps queued on the device behind a submitted target step, before the
//! owner asks for them. What runs behind a step is planned in one place
//! (`ExecutorDomain::behind`):
//!
//! - A chain: a lone prompt chunk takes the planned chunks continuing it,
//!   within the load's chain capacity, submitted with it as one chain
//!   (`TargetProgram::submit`) that runs every block over every chunk in
//!   chain order, its recurrent state alternating between two banks
//!   (`OwnedChainedAdvance`). The chain runs on the device as one and
//!   commits as one: each chunk finishes into it (`HeldChain`) and the last
//!   publishes it whole. Its holder's next chunks claim it before any other
//!   work (`chain_holder`), and it is never orphaned: a holder that will
//!   not claim it (it ended) releases it (`release_chain`, `close`).
//! - Cross-step pipelining ("lookahead"): right after a continuable step N
//!   is submitted, its successor N+1 is submitted behind it. A decode
//!   continuation's input token is N's selection read on the device, and its
//!   selection is N's with the position advanced by one. A prompt's next
//!   prefill chunk is known before N finishes: the owner plans it
//!   (`plan_successors`) and it is queued with its own tokens. The device
//!   then runs N+1 without waiting for the host to observe N and submit the
//!   next step.
//!
//! Every queued step follows the one before it: each slot's successor advance
//! follows its predecessor's, and a chunk that primes its drafter enters it
//! behind its own step, chained on the entry before. Queued steps live in the
//! domain's one queue (`Queue`), with the chain they continue: they are the
//! only work that holds bindings between flights, so every binding change
//! resolves them first (`orphan_queue`). The owner never sees them. When
//! the operations it next submits equal the front step's prediction (token
//! included, known once its predecessor finished) and the accepted states are
//! the ones the successors follow, the group claims the queued step: nothing
//! is reserved or launched, and its flight is returned. Any other target
//! group, and any binding change, orphans a pipelined successor: the domain
//! waits for it and drops it, so its rows, bank and leases return first.
//! Every device write of a queued step goes to rows and banks its own
//! successors own, so an orphan never changes accepted state.

use super::*;
use crate::programs::{SubmittedHead, SubmittedTarget};
use magnitude_state::{OwnedSuccessorAdvance, TentativeAdvance};
use std::collections::VecDeque;

/// A prompt group's chain under way: its holder, whose target state it is,
/// the chunks that finished, joined in order, and the rest of it, queued.
/// The last chunk's outcome publishes it whole (`Advance::Chain`). Its
/// queued chunks read the chain's bindings and are resolved with it, never
/// orphaned.
pub(super) struct HeldChain<F: ProgramFamily> {
    holder: RequestId,
    joined: OwnedChainAdvance,
    /// The rest of the chain, each step behind the one before; empty once
    /// its last chunk was claimed, until that chunk finishes.
    steps: VecDeque<QueuedStep<F>>,
}

/// What runs behind a submitted target step.
pub(super) enum Behind {
    /// Nothing: the owner submits the next step once this one finishes.
    Nothing,
    /// The rest of a chain: the planned chunks of the step's lone request,
    /// in order, submitted with it. A chain opens only where a step
    /// launches.
    Chain { chunks: Vec<Operation> },
    /// One pipelined successor, submitted right after the step, of every
    /// slot's decode continuation: its tokens are the step's selections,
    /// read on the device.
    Continuation { operations: Vec<Operation> },
    /// One pipelined successor, submitted right after the step, of the
    /// planned prompt chunk after each slot's: its tokens are its own.
    NextChunk { operations: Vec<Operation> },
}

impl Behind {
    /// The queued operations of `request`.
    pub(super) fn of(&self, request: RequestId) -> impl Iterator<Item = &Operation> {
        let operations = match self {
            Self::Nothing => &[][..],
            Self::Chain { chunks: operations }
            | Self::Continuation { operations }
            | Self::NextChunk { operations } => operations,
        };
        operations
            .iter()
            .filter(move |operation| operation.request() == request)
    }
}

/// How a queued step's advances follow the step before it.
enum Follow {
    /// As pipelined successors, their tokens read from `TargetTokens`: a
    /// decode continuation's, the predecessor's selection rows.
    Successor(TargetTokens),
    /// As the next chunk of a chain, of its own tokens.
    Chained,
}

/// A queued step's drafter entry, secured before the step is submitted: its
/// advance, following the entry of the step before, and its launch leases.
pub(super) struct QueuedPriming {
    advance: OwnedSuccessorAdvance,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_output: NativeGraphOutputLease,
}

/// A step built to queue behind another, not yet submitted.
pub(super) struct Pending {
    operations: Vec<Operation>,
    priming: Option<QueuedPriming>,
    launch_trace: Option<seismic::SubmissionTrace>,
}

pub(super) struct QueuedStep<F: ProgramFamily> {
    flight: TargetWork<F>,
    /// The flight whose selections are this one's tokens.
    predecessor: u64,
    /// Per slot, the operation that claims it (a decode continuation with a
    /// placeholder token, which its launch reads from its predecessor's
    /// selections: `TargetTokens::Selected`).
    predicted: Vec<Operation>,
    /// Per slot, the predecessor's selection, once it finished.
    selected: Option<Vec<Selected>>,
}

/// The steps queued behind a submitted target step.
pub(super) enum Queue<F: ProgramFamily> {
    Empty,
    /// The rest of a chain whose first chunk is in flight, never empty, each
    /// step behind the one before: that chunk's finish opens it
    /// (`open_chain`).
    Opening(VecDeque<QueuedStep<F>>),
    /// A chain under way, until its last chunk finishes. It runs on the
    /// device as one, so its holder's claims come before any other work
    /// (`chain_holder`).
    Chain(HeldChain<F>),
    /// One pipelined successor: speculative, any other work orphans it.
    Successor(QueuedStep<F>),
}

impl<F: ProgramFamily> Queue<F> {
    /// Whether flight `flight` opens the chain queued behind it: it is the
    /// chain's first chunk.
    pub(super) fn opens_chain(&self, flight: u64) -> bool {
        match self {
            Self::Opening(steps) => steps.front().is_some_and(|step| step.predecessor == flight),
            Self::Empty | Self::Chain(_) | Self::Successor(_) => false,
        }
    }

    /// The holder of the chain under way.
    fn holder(&self) -> Option<RequestId> {
        match self {
            Self::Chain(chain) => Some(chain.holder),
            Self::Empty | Self::Opening(_) | Self::Successor(_) => None,
        }
    }

    fn front(&self) -> Option<&QueuedStep<F>> {
        match self {
            Self::Empty => None,
            Self::Opening(steps) | Self::Chain(HeldChain { steps, .. }) => steps.front(),
            Self::Successor(step) => Some(step),
        }
    }

    /// The front step, claimed: a chain stays under way until its last
    /// chunk finishes.
    fn pop_front(&mut self) -> Option<QueuedStep<F>> {
        match std::mem::replace(self, Self::Empty) {
            Self::Empty => None,
            Self::Successor(step) => Some(step),
            Self::Opening(mut steps) => {
                let front = steps.pop_front();
                if !steps.is_empty() {
                    *self = Self::Opening(steps);
                }
                front
            }
            Self::Chain(mut chain) => {
                let front = chain.steps.pop_front();
                *self = Self::Chain(chain);
                front
            }
        }
    }

    fn steps_mut(&mut self) -> impl Iterator<Item = &mut QueuedStep<F>> {
        let (chain, successor) = match self {
            Self::Empty => (None, None),
            Self::Opening(steps) | Self::Chain(HeldChain { steps, .. }) => (Some(steps.iter_mut()), None),
            Self::Successor(step) => (None, Some(step)),
        };
        chain.into_iter().flatten().chain(successor)
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Opening(steps) | Self::Chain(HeldChain { steps, .. }) => steps.len(),
            Self::Successor(_) => 1,
        }
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Wait for the queued steps nobody will claim and drop them, in queue
    /// order: their successors release their rows and banks, their leases
    /// return to their pools. A chain under way is not among them: its
    /// holder claims it to its end or releases it, and the owner changes no
    /// bindings until then (`chain_holder`). The binding right proves no
    /// flight is on the device: every queued step waited for is behind none.
    pub(super) fn orphan(
        &mut self,
        experts: Option<&ExpertMeasures>,
        _between_flights: &mut StateBindings,
    ) -> Result<(), DomainError> {
        if self.holder().is_some() {
            return Err(DomainError::invariant("a chain under way is orphaned"));
        }
        self.drain(experts)
    }

    /// Wait for every queued step and drop it, in queue order, and empty
    /// the queue; a chain under way goes with them. The first failure is
    /// reported once every step was waited for.
    fn drain(&mut self, experts: Option<&ExpertMeasures>) -> Result<(), DomainError> {
        let steps = match std::mem::replace(self, Self::Empty) {
            Self::Empty => VecDeque::new(),
            Self::Successor(step) => VecDeque::from([step]),
            Self::Opening(steps) | Self::Chain(HeldChain { steps, .. }) => steps,
        };
        drain_steps(steps, experts)
    }
}

/// Wait for each of `steps` and drop it, in order. The first failure is
/// reported once every step was waited for.
fn drain_steps<F: ProgramFamily>(
    steps: VecDeque<QueuedStep<F>>,
    experts: Option<&ExpertMeasures>,
) -> Result<(), DomainError> {
    steps.into_iter().fold(Ok(()), |outcome, step| {
        let orphaned = step.orphan(experts);
        outcome.and(orphaned)
    })
}

impl<F: ProgramFamily> QueuedStep<F> {
    /// Wait for the step nobody claims and drop it.
    fn orphan(self, experts: Option<&ExpertMeasures>) -> Result<(), DomainError> {
        // Its drafter entry runs behind it on the device and finishes too,
        // so no lease returns while the device still uses it.
        let priming = self
            .flight
            .priming
            .map(|priming| priming.submission.finish().map(drop));
        // The wait stays in the host's turn: a request that stops decoding
        // plainly pays for the step it discarded in its next launch. Its
        // output completes as a used step's does (its reads land, its expert
        // bank's step completes, releasing what it held); the slots' read
        // failures fail nothing, as no sequence takes its outputs.
        let finished = self.flight.submission.finish().map(|completed| {
            let (core, output) = completed.into_parts();
            output.complete(core.batch(), super::recording(experts, core.work())).map(|_| ())
        });
        match (finished, priming) {
            (Ok(Ok(())), None | Some(Ok(()))) => Ok(()),
            (Ok(Err(error)), _) => Err(DomainError::Invariant(error)),
            (Err(error), _) | (_, Some(Err(error))) => Err(DomainError::Device(error)),
        }
    }
}

/// A queued step's state that fits now (`Some`), or not until the step
/// provisions as usual (`None`, `Error::until_provisioned`: it then runs
/// unpipelined); any other refusal is a fault.
fn fitting<T>(result: Result<T, magnitude_state::Error>) -> Result<Option<T>, DomainError> {
    match result {
        Ok(fitted) => Ok(Some(fitted)),
        Err(error) if error.until_provisioned() => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Every slot's decode continuation within a context of `context` rows
/// (`continuation_of`), or `None` when a slot has none. A round whose
/// drafter proposed (a request in `proposed`) has none whatever its rows, as
/// a verification of several rows has none: a decode queued behind a
/// one-row verification that declined every proposal is what the next
/// round's drafter waits for and discards. On a load whose features are a
/// separate draft's fused taps (`ModelLoadPlan::draft_fusion`) a round
/// drafting nothing publishes no feature: a decode that publishes one
/// prepares the next round's draft, and has none either; and a decode whose
/// next round prepares (a request in `publishing`) has one that publishes
/// its feature, so that round is the queued decode, discarding nothing.
pub(super) fn continuations(
    operations: &[Operation],
    proposed: &BTreeSet<RequestId>,
    publishing: &BTreeSet<RequestId>,
    fuses_draft_taps: bool,
    context: usize,
) -> Option<Vec<Operation>> {
    operations
        .iter()
        .map(|operation| {
            let request = operation.request();
            let drafts_next = proposed.contains(&request)
                || (fuses_draft_taps && operation.demand().contains(crate::batching::Demand::FEATURES));
            match drafts_next {
                true => None,
                false => continuation_of(operation, publishing.contains(&request), context),
            }
        })
        .collect()
}

/// The operation continuing `operation` by one decode row: the same request,
/// selection and output demand, one position later. A decode that also
/// publishes its row's feature for a drafter continues like a plain one: the
/// queued step publishes its own feature, read by the host only after it
/// finishes. `None` when the next step depends on the host (grammar masks,
/// penalty histories, logits demanded, conditioned rows, more than one row),
/// or when `operation` fills the last of the `context` rows a sequence
/// holds: its request ends there, and no step follows it. The decode
/// continuing `operation` selects its next row as its own was, publishing
/// its feature where `publishes`.
pub(super) fn continuation_of(operation: &Operation, publishes: bool, context: usize) -> Option<Operation> {
    let Operation::Forward {
        request,
        kind: WorkKind::Decode,
        tokens,
        position,
        conditioning: None,
        demand,
        select,
        committed: 1,
        prime: None,
    } = operation
    else {
        return None;
    };
    let [spec] = select.as_slice() else {
        return None;
    };
    let select_only = crate::batching::Demand::SELECT;
    if tokens.len() != 1
        || (*demand != select_only && *demand != select_only | crate::batching::Demand::FEATURES)
        || spec.mask.is_some()
        || spec.history.is_some()
    {
        return None;
    }
    let following = position.checked_add(1)?;
    if following >= context {
        return None;
    }
    let mut next = spec.clone();
    next.position = spec.position.checked_add(1)?;
    Some(Operation::Forward {
        request: *request,
        kind: WorkKind::Decode,
        tokens: vec![crate::TokenId(0)],
        position: following,
        conditioning: None,
        demand: match publishes {
            true => *demand | crate::batching::Demand::FEATURES,
            false => *demand,
        },
        select: vec![next],
        committed: 1,
        prime: None,
    })
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Plan the operations that follow the next submitted target group, when
    /// they are known before it finishes. Its provisioning makes room for
    /// them and its launch queues them; any other group discards them.
    pub(crate) fn plan_successors(&mut self, planned: Vec<Operation>) {
        self.planned = planned;
    }

    /// Plan the prompt chunks behind the next submitted prompt chunk: the
    /// one planning of a prompt's successors, for the scheduler's owner and
    /// the evaluation harness alike. `decode_owed` is the decode seconds
    /// owed each second they run while decode waits (none when it doesn't),
    /// which sizes their group ([`Self::group_size`]). `chunks` forms
    /// the operations of at most the count it is given: the rest of a prompt
    /// group of [`Self::group_size`] chunks (decode waiting as there), or
    /// without chains the one chunk a pipelined successor runs. That
    /// successor is claimed only when the next group submitted is that chunk,
    /// and any other group waits for it and discards it, so it is planned
    /// only when the device serves the request `alone`. A chain's holder
    /// alone continues it, and the scheduler selects it first; but a group
    /// of its own round submitted after it would orphan it, so one opens
    /// only behind the round's `last` group.
    pub fn plan_prompt<E>(
        &mut self,
        decode_owed: Option<f64>,
        alone: bool,
        last: bool,
        chunks: impl FnOnce(usize) -> Result<Vec<Operation>, E>,
    ) -> Result<(), E> {
        // A load that chains runs the rest of the group behind the chunk
        // (`behind_within`); one that doesn't, a pipelined successor.
        let count = match self.chain_capacity() > std::num::NonZeroUsize::MIN {
            true if last => self.group_size(decode_owed.is_some()).saturating_sub(1),
            true => 0,
            false => usize::from(alone),
        };
        let planned = chunks(count)?;
        self.plan_successors(planned);
        Ok(())
    }

    /// The most prompt chunks one chain runs (`ResourceLimits::chain_capacity`).
    /// The scheduler chooses each round's chain within it by the chunks it
    /// plans.
    pub fn chain_capacity(&self) -> std::num::NonZeroUsize {
        self.execution.policy().limits().chain_capacity
    }

    /// The chunks one prompt group runs (a chain's launches): with decode
    /// waiting, the load's size rule (`ProgramFamily::waiting_group`), which
    /// bounds how long decode waits behind the group; otherwise the chain
    /// capacity. The scheduler plans that many chunks.
    pub fn group_size(&self, decode_waiting: bool) -> usize {
        match decode_waiting {
            true => self.family.waiting_group(self.chain_capacity().get()),
            false => self.chain_capacity().get(),
        }
    }

    /// The request whose chain is under way: its target state is the
    /// chain's until the last chunk publishes it. The chain runs and commits
    /// as one, so the owner submits that request's claims before any other
    /// group, and releases nothing (every release changes bindings) until
    /// it ends; a holder that ended releases it (`release_chain`).
    pub fn chain_holder(&self) -> Option<RequestId> {
        self.queue.holder()
    }

    /// Where `request`'s target state ends: its chain's while one is under
    /// way (its holder's state is the chain's), else its accepted state's.
    pub(super) fn target_end(&self, request: RequestId) -> Option<usize> {
        match &self.queue {
            Queue::Chain(chain) if chain.holder == request => Some(chain.joined.end()),
            Queue::Empty | Queue::Opening(_) | Queue::Chain(_) | Queue::Successor(_) => {
                self.target.get(&request).map(SequenceState::position)
            }
        }
    }

    /// The steps queued behind the last flight.
    #[cfg(test)]
    pub(crate) fn queued_steps(&self) -> usize {
        self.queue.len()
    }

    /// Whether `request`'s work may be claimed now: while a chain is under
    /// way, its holder's alone (`chain_holder`), which continues it.
    pub fn claims(&self, request: RequestId) -> bool {
        self.chain_holder().is_none_or(|holder| holder == request)
    }

    /// Release the chain of a holder that will not claim the rest of it (it
    /// ended): its queued chunks are waited for and dropped, and its holder's
    /// state goes with it, as an evicted request's (`release_state`): the
    /// state the chain started from would trail what its holder took.
    pub fn release_chain(&mut self) -> Result<(), DomainError> {
        let HeldChain { holder, joined, steps } = match std::mem::replace(&mut self.queue, Queue::Empty) {
            Queue::Chain(chain) => chain,
            other => {
                self.queue = other;
                return Ok(());
            }
        };
        let drained = drain_steps(steps, self.measures.as_ref().and_then(|measures| measures.experts.as_ref()));
        self.target.insert(holder, joined.abort());
        let released = self.release_state(&[holder]).map(|_bytes| ()).map_err(DomainError::invariant);
        // The first failure of both: the drained steps' fault, else the
        // release's.
        drained.and(released)
    }

    /// Join `request`'s finished chained chunk to its chain: held while the
    /// rest of it is queued, published whole with the last chunk.
    pub(super) fn join_chain(
        &mut self,
        request: RequestId,
        chunk: magnitude_state::OwnedChainedAdvance,
    ) -> Result<Advance, DomainError> {
        let mut chain = match std::mem::replace(&mut self.queue, Queue::Empty) {
            Queue::Chain(chain) if chain.holder == request => chain,
            other => {
                self.queue = other;
                return Err(DomainError::invariant("a chained chunk finished outside its chain"));
            }
        };
        let joined = chain.joined.join(chunk).map_err(|(_, error)| DomainError::from(error));
        if joined.is_err() || !chain.steps.is_empty() {
            self.queue = Queue::Chain(chain);
            return joined.map(|()| Advance::Chained);
        }
        Ok(Advance::Chain(chain.joined))
    }

    /// Hold the chain `first` opens, the first chunk of `request`'s prompt
    /// group, whose flight just finished with the rest queued behind it
    /// (`Queue::opens_chain`).
    pub(super) fn open_chain(
        &mut self,
        request: RequestId,
        first: OwnedStateAdvance,
    ) -> Result<Advance, DomainError> {
        let steps = match std::mem::replace(&mut self.queue, Queue::Empty) {
            Queue::Opening(steps) => steps,
            other => {
                self.queue = other;
                return Err(DomainError::invariant("a chain opens where none is queued"));
            }
        };
        match OwnedChainAdvance::begin(first) {
            Ok(joined) => {
                self.queue = Queue::Chain(HeldChain { holder: request, joined, steps });
                Ok(Advance::Chained)
            }
            Err((first, error)) => {
                self.queue = Queue::Opening(steps);
                self.target.insert(request, first.abort());
                Err(DomainError::from(error))
            }
        }
    }

    /// What runs behind `operations` once they are submitted or claimed, a
    /// function of the round's operations, the chunks the scheduler planned
    /// for it and the load. A lone prompt chunk opens a chain of the planned
    /// chunks continuing it, within the places of the class the group holds
    /// (`grant_group`), that the family runs in a chain; otherwise, with
    /// pipelining, every slot's successor of one kind queues behind it.
    pub(super) fn behind(&self, operations: &[Operation]) -> Behind {
        self.behind_within(operations, self.resources.group_places().get())
    }

    /// What runs behind `operations` (`behind`) with prompt groups of up to
    /// `places` places.
    pub(super) fn behind_within(&self, operations: &[Operation], places: usize) -> Behind {
        let chains = |operation: &Operation| self.family.chains(operation.row_count());
        if let [operation] = operations {
            let mut chunks = Vec::new();
            let mut previous = operation;
            while chains(operation) && chunks.len() + 1 < places {
                let Some(next) = self.planned_after(previous).filter(|next| chains(next)) else {
                    break;
                };
                chunks.push(next.clone());
                previous = next;
            }
            if !chunks.is_empty() {
                return Behind::Chain { chunks };
            }
        }
        if !self.execution.policy().limits().flights().pipelines() {
            return Behind::Nothing;
        }
        // Every slot's successor is of one kind: all decode continuations, or
        // all planned chunks (a decode has no planned chunk, a prefill no
        // continuation). A planned chunk of a load that chains opens the next
        // chain rather than queueing alone behind this one.
        let context = self.execution.resources().target_state().context_rows;
        let fuses_draft_taps = self.execution.load().draft_fusion().is_some();
        if let Some(operations) = continuations(operations, &self.proposed, &self.publishing, fuses_draft_taps, context) {
            return Behind::Continuation { operations };
        }
        if self.chain_capacity() > std::num::NonZeroUsize::MIN {
            return Behind::Nothing;
        }
        operations
            .iter()
            .map(|operation| self.planned_after(operation).cloned())
            .collect::<Option<Vec<_>>>()
            .map_or(Behind::Nothing, |operations| Behind::NextChunk { operations })
    }

    /// The planned prompt chunk after `operation`, a prefill chunk that
    /// selects nothing.
    pub(super) fn planned_after(&self, operation: &Operation) -> Option<&Operation> {
        let Operation::Forward {
            request,
            kind: WorkKind::Prefill,
            tokens,
            position,
            conditioning: None,
            select,
            ..
        } = operation
        else {
            return None;
        };
        if !select.is_empty() {
            return None;
        }
        let end = position.checked_add(tokens.len())?;
        self.planned.iter().find(|planned| {
            matches!(planned, Operation::Forward {
                request: next,
                kind: WorkKind::Prefill,
                position: start,
                conditioning: None,
                ..
            } if next == request && *start == end)
        })
    }

    /// Build the rest of a chain behind its first launch, that of
    /// `operations`: each planned chunk following the one before, with its
    /// drafter entry following the one before (the first's is `priming`).
    /// Stops at the first chunk that does not fit; the owner launches it
    /// later.
    pub(super) fn build_chain(
        &mut self,
        operations: &[Operation],
        first: &ValidatedTargetLaunch,
        priming: Option<&PrimingReservation>,
        chunks: Vec<Operation>,
    ) -> Result<(Vec<ValidatedTargetLaunch>, Vec<Pending>), DomainError> {
        let mut launches: Vec<ValidatedTargetLaunch> = Vec::new();
        let mut pending: Vec<Pending> = Vec::new();
        let mut ahead = operations.to_vec();
        for chunk in chunks {
            let operations = vec![chunk];
            let previous = pending.last();
            let entry = self.queued_priming(&operations, |count| match previous {
                None => priming.map_or(Ok(None), |priming| fitting(priming.advance.successor(count))),
                Some(previous) => previous
                    .priming
                    .as_ref()
                    .map_or(Ok(None), |priming| fitting(priming.advance.successor(count))),
            })?;
            let Some(entry) = entry else {
                break;
            };
            let predecessor = launches.last().unwrap_or(first).advances();
            let Some(launch) =
                self.build_successor(predecessor, &operations, Follow::Chained, &ahead)?
            else {
                break;
            };
            ahead.extend(operations.iter().cloned());
            launches.push(launch);
            pending.push(Pending {
                operations,
                priming: entry,
                launch_trace: None,
            });
        }
        Ok((launches, pending))
    }

    /// Queue the successor of `flight` (just submitted or claimed for
    /// `operations`, see `behind`) when a decode continuation's flight has a
    /// selection tensor to read tokens from, and the successors and leases
    /// fit without growing anything. Otherwise the next step is submitted by
    /// the owner as usual.
    pub(super) fn queue_successor(
        &mut self,
        flight: &TargetWork<F>,
        operations: &[Operation],
    ) -> Result<(), DomainError> {
        let behind = self.behind(operations);
        self.planned.clear();
        // Behind a claimed chain step runs the rest of its chain: only an
        // empty queue takes a successor.
        if !matches!(self.queue, Queue::Empty) {
            return Ok(());
        }
        // Tokens are the flight's selection rows, one per slot in slot
        // order, so the continuation keeps every slot in the same order.
        if operations.len() != flight.requests.len()
            || operations
                .iter()
                .zip(&flight.requests)
                .any(|(operation, (request, ..))| operation.request() != *request)
        {
            return Ok(());
        }
        let (predicted, tokens) = match behind {
            Behind::Nothing | Behind::Chain { .. } => return Ok(()),
            Behind::Continuation { operations } => {
                let Some(selected) = flight
                    .submission
                    .output()
                    .readout
                    .as_ref()
                    .and_then(|readout| readout.selected.clone())
                else {
                    return Ok(());
                };
                (operations, TargetTokens::Selected(selected))
            }
            Behind::NextChunk { operations } => (operations, TargetTokens::Host),
        };
        // A planned prompt chunk's drafter entry follows the in-flight entry
        // of the chunk before it.
        let before = flight
            .priming
            .as_ref()
            .filter(|priming| matches!(predicted.as_slice(), [operation] if operation.request() == priming.request))
            .and_then(|priming| priming.submission.launch().advances().first());
        let entry = self.queued_priming(&predicted, |count| {
            before.map_or(Ok(None), |advance| fitting(advance.successor(count)))
        })?;
        let Some(entry) = entry else {
            return Ok(());
        };
        // The successor is a chain of one: its output and the spare.
        let Ok(spare) = self.resources.target_graph().acquire_output() else {
            return Ok(());
        };
        let Some(launch) = self.build_successor(
            flight.submission.launch().advances(),
            &predicted,
            Follow::Successor(tokens),
            &[],
        )?
        else {
            return Ok(());
        };
        let pending = Pending {
            launch_trace: self.trace_target_submission(&predicted),
            operations: predicted,
            priming: entry,
        };
        let started = Instant::now();
        let epoch = self.launch_epoch();
        let (submission, _) = match self.submit_chain(Chain::one(launch), spare) {
            Ok(submitted) => submitted.split_first(),
            Err(refusal) => return Err(DomainError::from(refusal.error)),
        };
        self.queue = Queue::Successor(self.queued_step(
            flight,
            pending,
            submission,
            started,
            epoch,
        )?);
        Ok(())
    }

    /// The drafter entry of a step queued behind another: none when the
    /// step primes nothing, or its advance (`successor` of the entry before,
    /// given its rows) and its draft launch leases. `None` when the step
    /// primes and its entry does not fit.
    fn queued_priming(
        &self,
        operations: &[Operation],
        successor: impl FnOnce(usize) -> Result<Option<OwnedSuccessorAdvance>, DomainError>,
    ) -> Result<Option<Option<QueuedPriming>>, DomainError> {
        let [Operation::Forward {
            prime: Some(prime), ..
        }] = operations
        else {
            return Ok(Some(None));
        };
        let Some(advance) = successor(prime.tokens.len())? else {
            return Ok(None);
        };
        let leases = self
            .resources
            .head_graph()
            .and_then(|head| Some((head.acquire_workspace().ok()?, head.acquire_output().ok()?)));
        Ok(leases.map(|(graph_workspace, graph_output)| {
            Some(QueuedPriming {
                advance,
                graph_workspace,
                graph_output,
            })
        }))
    }

    /// Queue the rest of the chain submitted behind `first`, each step
    /// behind the one before. A step that fails leaves the ones before it
    /// queued, to be waited for and dropped.
    pub(super) fn queue_chain(
        &mut self,
        first: &TargetWork<F>,
        queued: impl IntoIterator<Item = (Pending, F::TargetSubmission)>,
        started: Instant,
        epoch: Option<LaunchEpoch>,
    ) -> Result<(), DomainError> {
        let mut steps = VecDeque::new();
        let mut outcome = Ok(());
        for (pending, submission) in queued {
            let predecessor = steps
                .back()
                .map_or(first, |step: &QueuedStep<F>| &step.flight);
            match self.queued_step(predecessor, pending, submission, started, epoch) {
                Ok(step) => steps.push_back(step),
                Err(error) => {
                    outcome = Err(error);
                    break;
                }
            }
        }
        if !steps.is_empty() {
            self.queue = Queue::Opening(steps);
        }
        outcome
    }

    /// A submitted step queued behind `predecessor`, with its drafter entry.
    /// `started` is when it was submitted, and `epoch` the launch epoch then.
    fn queued_step(
        &mut self,
        predecessor: &TargetWork<F>,
        pending: Pending,
        submission: F::TargetSubmission,
        started: Instant,
        epoch: Option<LaunchEpoch>,
    ) -> Result<QueuedStep<F>, DomainError> {
        let Pending {
            operations,
            priming,
            launch_trace,
        } = pending;
        let requests = operations
            .iter()
            .map(|operation| {
                let Operation::Forward {
                    request,
                    kind,
                    committed,
                    ..
                } = operation
                else {
                    unreachable!("a queued step is a forward operation")
                };
                (*request, operation.row_count(), None, *kind, *committed)
            })
            .collect();
        let id = self.flight_id();
        let mut flight = TargetWork {
            launch_trace,
            requests,
            submission,
            priming: None,
            started,
            epoch,
            runnable: started,
            previous_selection: None,
            id,
            continuation: None,
        };
        if let (Some(priming), [Operation::Forward { request, prime: Some(prime), .. }]) =
            (priming, operations.as_slice())
        {
            flight = self.launch_priming(
                flight,
                *request,
                prime,
                TentativeAdvance::Successor(priming.advance),
                predecessor.submission.output().boundary.clone(),
                priming.graph_workspace,
                priming.graph_output,
            )?;
        }
        Ok(QueuedStep {
            flight,
            predecessor: predecessor.id,
            predicted: operations,
            selected: None,
        })
    }

    pub(super) fn flight_id(&mut self) -> u64 {
        self.next_flight += 1;
        self.next_flight
    }

    /// The front queued step's slot each operation claims, in operation
    /// order, when the group is exactly that step's continuation of
    /// accepted states.
    pub(super) fn claim_slots(&self, operations: &[Operation]) -> Option<Vec<usize>> {
        let step = self.queue.front()?;
        let selected = step.selected.as_ref()?;
        let advances = step.flight.submission.launch().advances();
        let mut slots = Vec::with_capacity(operations.len());
        for operation in operations {
            let slot = step
                .predicted
                .iter()
                .position(|predicted| predicted.request() == operation.request())?;
            let mut expected = step.predicted[slot].clone();
            if let TargetTokens::Selected(_) = step.flight.submission.launch().tokens() {
                // Every table row the queued step read for this token is the
                // token's (a landed read, `DeviceTable::read_selected`). A
                // slot whose rows could not be read is claimed all the same:
                // its sequence fails when the step finishes, and the others
                // continue (`finish_target_work`).
                let choice = selected.get(slot).filter(|choice| choice.status == 0)?;
                if let Operation::Forward { tokens, .. } = &mut expected {
                    tokens[0] = choice.token;
                }
            }
            let advance = advances.get(slot)?;
            // A chained chunk follows its chain; any other slot the
            // request's accepted state.
            let follows = match advance {
                TentativeAdvance::Chained(_) => match &self.queue {
                    Queue::Chain(chain) => {
                        chain.holder == operation.request() && chain.joined.end() == advance.position()
                    }
                    Queue::Empty | Queue::Opening(_) | Queue::Successor(_) => false,
                },
                TentativeAdvance::Accepted(_) | TentativeAdvance::Successor(_) => {
                    let state = self.target.get(&operation.request())?;
                    state.position() == advance.position()
                        && state.bank_index() == advance.bindings().previous_bank
                        && state.tape_rows() == advance.bindings().previous_tape
                        && state.domain_ranges() == advance.domain_ranges()
                }
            };
            let drafter_follows = match operation {
                Operation::Forward {
                    prime: Some(prime), ..
                } => self
                    .head
                    .get(&operation.request())
                    .is_some_and(|head| head.position() == prime.position),
                _ => true,
            };
            if slots.contains(&slot) || !follows || !drafter_follows || operation != &expected {
                return None;
            }
            slots.push(slot);
        }
        Some(slots)
    }

    /// The launch class a claimable group occupies.
    pub(super) fn claim_class(&self, operations: &[Operation]) -> Option<crate::LaunchClass> {
        self.claim_slots(operations)?;
        let step = self.queue.front()?;
        Some(step.flight.submission.launch().batch().class())
    }

    /// Hand the front queued step to the owner as the flight of
    /// `operations`. Each claimed slot takes its request's accepted state,
    /// which its successor attaches to at finish; unclaimed slots are
    /// discarded then.
    pub(super) fn claim_queued(
        &mut self,
        operations: &[Operation],
        slots: Vec<usize>,
    ) -> Result<TargetWork<F>, DomainError> {
        if self.claim_slots(operations).as_ref() != Some(&slots) {
            return Err(DomainError::invariant(
                "claimed queued step changed after reservation",
            ));
        }
        let Some(QueuedStep { mut flight, .. }) = self.queue.pop_front() else {
            return Err(DomainError::invariant("claimed queued step is absent"));
        };
        let mut sources = (0..flight.requests.len()).map(|_| None).collect::<Vec<_>>();
        // `claim_slots` found every claimed request's accepted state.
        for (operation, slot) in operations.iter().zip(slots) {
            sources[slot] = self
                .target
                .remove(&operation.request())
                .map(InFlightState::new);
        }
        flight.continuation = Some(sources);
        if let Some(priming) = &mut flight.priming {
            priming.continuation = Some(self.head.remove(&priming.request).map(InFlightState::new));
        }
        Ok(flight)
    }

    /// Wait for the queued steps nobody will claim and drop them
    /// (`Queue::orphan`): every binding change does so first.
    pub(super) fn orphan_queue(&mut self, bindings: &mut StateBindings) -> Result<(), DomainError> {
        self.queue.orphan(self.measures.as_ref().and_then(|measures| measures.experts.as_ref()), bindings)
    }

    /// Wait for every queued step and drop it, a chain under way with them
    /// (`Queue::drain`).
    pub(super) fn drain_queue(&mut self) -> Result<(), DomainError> {
        self.queue.drain(self.measures.as_ref().and_then(|measures| measures.experts.as_ref()))
    }

    /// The launch of `predicted` queued behind a step whose advances are
    /// `predecessor` (in flight or about to be): each slot's advance follows
    /// its predecessor's as `follow` says. `ahead` are the operations of its
    /// chain before it, whose tokens are not yet fed (`target_preceding`).
    /// `None` when the advances or their leases do not fit without growing
    /// anything, or a slot needs input slices: the step is then submitted as
    /// usual.
    fn build_successor(
        &mut self,
        predecessor: &[TentativeAdvance],
        predicted: &[Operation],
        follow: Follow,
        ahead: &[Operation],
    ) -> Result<Option<ValidatedTargetLaunch>, DomainError> {
        // The step's rows are admitted and placed as its own provisioning
        // would place them once its predecessors commit.
        let slots = predecessor
            .iter()
            .zip(predicted)
            .map(|(advance, operation)| (advance, operation.row_count()));
        let (advances, tokens) = match follow {
            Follow::Successor(tokens) => (
                TentativeAdvance::successors(slots).map(|successors| {
                    successors
                        .into_iter()
                        .map(TentativeAdvance::Successor)
                        .collect::<Vec<_>>()
                }),
                tokens,
            ),
            Follow::Chained => (
                TentativeAdvance::chained(slots).map(|chunks| {
                    chunks
                        .into_iter()
                        .map(TentativeAdvance::Chained)
                        .collect::<Vec<_>>()
                }),
                TargetTokens::Host,
            ),
        };
        let advances = match advances {
            Ok(advances) => advances,
            // No room without growing the backing (or a limit): the next
            // step runs unpipelined and provisions as usual.
            Err(error) if error.until_provisioned() => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let mut slots = Vec::with_capacity(predicted.len());
        let mut segments = 1usize;
        for (operation, advance) in predicted.iter().zip(&advances) {
            let (slot, slices) = self
                .target_slot(operation, advance)
                .map_err(DomainError::Input)?;
            if !slices.is_empty() {
                return Ok(None);
            }
            segments = segments.max(super::target::reserved_segments(
                &self.target_store,
                advance.span_count(),
                operation.row_count(),
            ));
            slots.push(slot);
        }
        // The continuation follows the tokens fed so far.
        let preceding = self.target_preceding(predicted, ahead).map_err(DomainError::Input)?;
        let class_limits = self.target_class_limits();
        let vocabulary = self.definition.decoder.vocabulary as usize;
        let batch = match tokens {
            TargetTokens::Selected(_) => ValidatedTargetBatch::from_slots(&slots, vocabulary, class_limits),
            TargetTokens::Host => ValidatedTargetBatch::covering(&slots, vocabulary, class_limits, segments),
        }
        .map_err(|error| DomainError::Input(error.to_string()))?;
        let tokens = match tokens {
            TargetTokens::Selected(selected) => {
                let rows = batch.class().rows() as u64;
                if selected
                    .tensor()
                    .extents()
                    .first()
                    .is_none_or(|extent| *extent < rows)
                {
                    return Ok(None);
                }
                TargetTokens::Selected(
                    selected
                        .slice_leading(0, rows)
                        .map_err(|error| DomainError::Input(error.to_string()))?,
                )
            }
            TargetTokens::Host => TargetTokens::Host,
        };
        let graph = self.resources.target_graph();
        let readout = self.resources.target_readout_graph(super::projected_rows(predicted));
        let (Ok(graph_workspace), Ok(graph_output), Ok(readout_workspace), Ok(readout_output)) = (
            graph.acquire_workspace(),
            graph.acquire_output(),
            readout.acquire_workspace(),
            readout.acquire_output(),
        ) else {
            return Ok(None);
        };
        let boundary = if predicted.iter().any(|op| {
            matches!(
                op,
                Operation::Forward {
                    prime: Some(Priming { boundary: true, .. }),
                    ..
                }
            )
        }) {
            match self.allocate_boundary() {
                Ok(boundary) => Some(boundary),
                Err(DomainError::Capacity(_)) => return Ok(None),
                Err(error) => return Err(error),
            }
        } else {
            None
        };
        let count = predicted.len();
        let work = crate::operation::LaunchWork::of(predicted.iter().filter_map(|operation| match operation {
            Operation::Forward { kind, .. } => Some(*kind),
            _ => None,
        }));
        let inputs = TargetLaunchInputs::new(
            batch,
            work,
            tokens,
            advances,
            vec![None; count],
            vec![Vec::new(); count],
            graph_workspace,
            graph_output,
            readout_workspace,
            readout_output,
        )
        .with_boundary(boundary)
        .with_preceding(preceding);
        match ValidatedTargetLaunch::new(
            inputs,
            &self.target_store,
            self.domain.id(),
            self.definition.decoder.hidden as usize,
            self.definition.head.as_ref().map_or(self.definition.decoder.hidden, |head| head.conditioning_width(&self.definition.decoder)) as usize,
        ) {
            Ok(launch) => Ok(Some(launch)),
            Err((_, error)) => Err(DomainError::Invariant(error)),
        }
    }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Record the selections of a finished flight for the queued step it
    /// feeds. The predecessor's service ended at `completed`, its results
    /// read: the queued step's selections are known, and its service runs
    /// from then on.
    pub(super) fn predecessor_selected(&mut self, flight: u64, selected: &[Selected], completed: Instant) {
        if let Some(step) = self.queue.steps_mut().find(|step| step.predecessor == flight) {
            step.selected = Some(selected.to_vec());
            step.flight.runnable = completed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{continuation_of, continuations, fitting};

    /// A queued step's state that does not fit until it provisions runs it
    /// unpipelined; any other refusal of the store is a fault.
    #[test]
    fn only_a_refusal_until_provisioned_runs_a_step_unpipelined() {
        use magnitude_state::Error;
        assert!(matches!(fitting(Ok::<_, Error>(3)), Ok(Some(3))));
        assert!(matches!(fitting::<()>(Err(Error::Unbacked)), Ok(None)));
        assert!(matches!(fitting::<()>(Err(Error::BanksExhausted { capacity: 2 })), Ok(None)));
        let invariant = Error::Request("a chain follows an advance whose rows all commit".into());
        assert!(fitting::<()>(Err(invariant)).is_err());
    }
    use crate::batching::Demand;
    use crate::{Operation, RequestId, Sampling, SelectSpec, Shaping, TokenId, WorkKind};

    fn decode(demand: Demand, tokens: Vec<TokenId>) -> Operation {
        Operation::Forward {
            request: RequestId(3),
            kind: if tokens.len() == 1 { WorkKind::Decode } else { WorkKind::Verify },
            position: 7,
            conditioning: None,
            demand,
            select: (0..tokens.len())
                .map(|row| SelectSpec {
                    sampling: Sampling::Greedy,
                    seed: 1,
                    position: 4 + row,
                    domain: 0,
                    mask: None,
                    shaping: Shaping::default(),
                    history: None,
                })
                .collect(),
            tokens,
            committed: 1,
            prime: None,
        }
    }

    /// A drafting load's round that drafts nothing is a one-row decode that
    /// also publishes its feature: it is queued behind like a plain decode,
    /// keeping its demand, so the step that claims it equals it.
    #[test]
    fn a_decode_feeding_a_drafter_continues_like_a_plain_decode() {
        for demand in [Demand::SELECT, Demand::SELECT | Demand::FEATURES] {
            let next = continuation_of(&decode(demand, vec![TokenId(5)]), false, 9)
                .unwrap_or_else(|| panic!("{demand:?} decode has a continuation"));
            let Operation::Forward {
                position,
                demand: next_demand,
                select,
                ..
            } = &next
            else {
                unreachable!()
            };
            assert_eq!((*position, *next_demand, select[0].position), (8, demand, 5));
        }
        // Logits go to the host, and a verification's rows are the method's.
        assert!(continuation_of(&decode(Demand::SELECT | Demand::LOGITS, vec![TokenId(5)]), false, 9).is_none());
        assert!(continuation_of(
            &decode(Demand::SELECT | Demand::FEATURES, vec![TokenId(5), TokenId(6)]),
            false,
            9
        )
        .is_none());
    }

    /// A round whose drafter proposed verifies them in its target launch,
    /// and one that declined every proposal verifies none in a one-row
    /// decode: neither queues a decode behind it, which the next round's
    /// drafter would wait for and discard (13.6 ms on an M4 Pro serving
    /// Qwen3.6-35B-A3B q4 with its MTP head, charged to that drafter).
    #[test]
    fn a_round_whose_drafter_proposed_queues_no_decode_behind_it() {
        let declined = decode(Demand::SELECT | Demand::FEATURES, vec![TokenId(5)]);
        let plain = std::collections::BTreeSet::new();
        let none = std::collections::BTreeSet::new();
        assert!(continuations(std::slice::from_ref(&declined), &plain, &none, false, 9).is_some());
        let proposed = std::collections::BTreeSet::from([RequestId(3)]);
        assert!(continuations(std::slice::from_ref(&declined), &proposed, &none, false, 9).is_none());
    }

    /// On a load whose features are a separate draft's fused taps, a round
    /// drafting nothing publishes no feature: a decode publishing one
    /// prepares the next round's draft, and queues no decode behind it; a
    /// plain decode queues one as anywhere, which publishes its feature
    /// where the next round prepares.
    #[test]
    fn a_decode_preparing_a_fused_draft_queues_no_decode_behind_it() {
        let none = std::collections::BTreeSet::new();
        let preparing = decode(Demand::SELECT | Demand::FEATURES, vec![TokenId(5)]);
        assert!(continuations(std::slice::from_ref(&preparing), &none, &none, true, 9).is_none());
        let plain = decode(Demand::SELECT, vec![TokenId(5)]);
        let queued = continuations(std::slice::from_ref(&plain), &none, &none, true, 9).unwrap();
        assert_eq!(queued[0].demand(), Demand::SELECT);
        let publishing = std::collections::BTreeSet::from([plain.request()]);
        let queued = continuations(std::slice::from_ref(&plain), &none, &publishing, true, 9).unwrap();
        assert_eq!(queued[0].demand(), Demand::SELECT | Demand::FEATURES);
    }
}
