//! Steps queued on the device behind a submitted target step, before the
//! owner asks for them. What runs behind a step is planned in one place
//! (`ExecutorDomain::behind`):
//!
//! - A chain: a lone prompt chunk takes the planned chunks continuing it,
//!   within the load's chain capacity, submitted with it as one chain
//!   (`TargetProgram::submit`) that runs every block over every chunk in
//!   chain order, its recurrent state alternating between two banks
//!   (`OwnedChainedAdvance`). The chain runs on the device as one: each
//!   chunk joins it as its outcome reconciles (`HeldChain`), so it holds
//!   exactly the rows its holder accepted. A claimed chunk carries the
//!   chain's state from its claim until its outcome returns it (`Joined`),
//!   so that state has one owner, and its outcome is the only way back.
//!   Its holder's next chunks claim it before any other work
//!   (`chain_holder`). It ends one way, concluded (`Queue::conclude`): its
//!   unclaimed chunks are waited for and dropped and what it joined
//!   commits, its holder's state again (or, carried by a chunk, once that
//!   chunk's outcome returns it), whether its last chunk reconciled, a
//!   chunk was aborted (cancelled, interrupted, failed), its holder ended,
//!   or other work changed bindings.
//! - Cross-step pipelining ("lookahead"): right after a continuable step N
//!   is submitted, its successor N+1 is submitted behind it. A decode
//!   continuation's input tokens are N's selections read on the device: of
//!   a chained pass, the rows the front head launch right behind N makes
//!   (`guess_chain`'s `next_selected`, `continuation_of`), its exact token
//!   and the chain's guesses; and its selections are N's with the position
//!   advanced by one. The front head launch behind a queued step is secured
//!   as the step is queued and submitted as it is claimed, so the device
//!   runs N, N's head launch, N+1, N+1's head launch, N+2. A prompt's next
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
//! resolves them first (`conclude_chain`); the owner defers one during a
//! chain to the chain's end (`Unchained`), so as not to cut it short. The
//! owner never sees them. When
//! the operations it next submits equal the front step's prediction (token
//! included, known once its predecessor finished) and the accepted states are
//! the ones the successors follow, the group claims the queued step: nothing
//! is reserved or launched, and its flight is returned. Any other target
//! group, and any binding change, orphans a pipelined successor: the domain
//! waits for it and drops it, so its rows, bank and leases return first.
//! Every device write of a queued step goes to rows and banks its own
//! successors own, so an orphan never changes accepted state.

use super::*;
use crate::batching::FrontContinuation;
use crate::programs::{SubmittedHead, SubmittedTarget};
use magnitude_state::{OwnedSuccessorAdvance, TentativeAdvance};
use std::collections::VecDeque;

/// A prompt group's chain under way: its holder, whose target state it is,
/// the chunks whose outcomes reconciled, joined in order, and the rest of
/// it, queued. Its queued chunks read the chain's bindings and end with it:
/// it is only ever concluded ([`Queue::conclude`]), which commits what it
/// joined.
pub(super) struct HeldChain<F: ProgramFamily> {
    holder: RequestId,
    joined: Joined,
    /// The rest of the chain, each step behind the one before; empty once
    /// its last chunk was claimed, until that chunk reconciles.
    steps: VecDeque<QueuedStep<F>>,
}

/// Where a chain's joined chunks are: the chain's state has one owner.
enum Joined {
    /// In the chain, between its chunks.
    Here(OwnedChainAdvance),
    /// With its chunk of flight `.0`, which claimed it: on the device, then
    /// in that chunk's outcome, which returns it as it reconciles or aborts
    /// (`ExecutorDomain::return_chain`). A chain concluded meanwhile commits
    /// nothing; its holder's state is the one that outcome returns.
    Out(u64),
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

/// Where a decode continuation's tokens are read on the device.
enum TokenSource {
    /// The rows the front head launch behind its chained predecessor makes
    /// (`guess_chain`'s `next_selected` and `next_guesses`).
    Chain,
    /// Its predecessor's selections, of a pass that chains nothing.
    Selected(crate::GraphOutputTensor),
    /// The host's: a planned prompt chunk's own tokens.
    Host,
}

/// Token source for a queued successor before its per-slot advances are built.
enum SuccessorTokens {
    Host,
    Selected {
        selected: crate::GraphOutputTensor,
        guesses: Option<crate::GraphOutputTensor>,
    },
}

/// How a queued step's advances follow the step before it.
enum Follow {
    /// As pipelined successors, their tokens read from `SuccessorTokens`: a
    /// decode continuation's, the predecessor's selection rows.
    Successor(SuccessorTokens),
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
    /// Its front head launch, of a step whose forwards chain.
    front: Option<FrontHead<OwnedSuccessorAdvance>>,
    launch_trace: Option<seismic::SubmissionTrace>,
}

pub(super) struct QueuedStep<F: ProgramFamily> {
    flight: TargetWork<F>,
    /// The flight whose selections are this one's tokens.
    predecessor: u64,
    /// Per slot, the first of its predecessor's batch rows: its exact
    /// selection's, where the chain made behind it begins (`claimed`).
    reads: Vec<usize>,
    /// Per slot, the operation that claims it (a decode continuation with
    /// placeholder tokens and no front row's guess kind, which its launch
    /// reads on the device: `TargetTokens::Selected`).
    predicted: Vec<Operation>,
    /// Per row, the predecessor's selection and the guess of the chain made
    /// behind it there, once it finished.
    selected: Option<Vec<(Selected, Option<Guess>)>>,
    /// Its front head launch, of a step whose forwards chain: secured as it
    /// was queued, due as it is claimed (`claim_queued`), launched behind it
    /// then (`queue_successor`); an orphan's never launches, and its
    /// advances and leases return as it drops.
    front: Option<Box<FrontHead<OwnedSuccessorAdvance>>>,
}

/// The steps queued behind a submitted target step.
pub(super) enum Queue<F: ProgramFamily> {
    Empty,
    /// The rest of a chain whose first chunk is in flight, never empty, each
    /// step behind the one before: that chunk's reconcile opens it
    /// (`open_chain`).
    Opening(VecDeque<QueuedStep<F>>),
    /// A chain under way, until its last chunk reconciles. It runs on the
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
            Self::Opening(steps) => opened_by(steps, flight),
            Self::Empty | Self::Chain(_) | Self::Successor(_) => false,
        }
    }

    /// The rest of the chain flight `flight` opens ([`Self::opens_chain`]),
    /// taken from the queue; none, the queue unchanged, where it opens none.
    fn take_opening(&mut self, flight: u64) -> Option<VecDeque<QueuedStep<F>>> {
        match std::mem::replace(self, Self::Empty) {
            Self::Opening(steps) if opened_by(&steps, flight) => Some(steps),
            other => {
                *self = other;
                None
            }
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

    /// The pass of front passes queued behind the last flight, a decode
    /// continuation reading its tokens on the device: each slot's request
    /// and the front rows it runs (`ExecutorDomain::front_pass`).
    pub(super) fn queued_fronts(&self) -> Option<Vec<(RequestId, usize)>> {
        let Self::Successor(step) = self else {
            return None;
        };
        matches!(step.flight.submission.launch().tokens(), TargetTokens::Selected { .. })
            .then(|| step.predicted.iter().map(|operation| (operation.request(), operation.front_rows())).collect())
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

    /// Conclude what the queue holds: the one way a queued step or a chain
    /// leaves it unclaimed. Each queued step is waited for and dropped, in
    /// order, so its rows, banks and leases return; a chain under way then
    /// commits the chunks joined to it, every one its holder reconciled,
    /// into `target`, its holder's state the one they reached, unless a
    /// chunk carries them, whose outcome returns that state
    /// ([`ExecutorDomain::return_chain`]). `bank` is told of the passes
    /// discarded with the dropped steps. The first failure of the dropped
    /// steps is reported once the holder's state is back.
    pub(super) fn conclude(
        &mut self,
        requests: &mut BTreeMap<RequestId, RequestEntry>,
        experts: Option<&ExpertMeasures>,
        bank: Option<&dyn ElasticBank>,
    ) -> Result<(), DomainError> {
        let (steps, chain) = match std::mem::replace(self, Self::Empty) {
            Self::Empty => (VecDeque::new(), None),
            Self::Successor(step) => (VecDeque::from([step]), None),
            Self::Opening(steps) => (steps, None),
            Self::Chain(HeldChain { holder, joined: Joined::Here(joined), steps }) => (steps, Some((holder, joined))),
            Self::Chain(HeldChain { joined: Joined::Out(_), steps, .. }) => (steps, None),
        };
        let drained = steps.into_iter().fold(Ok(()), |outcome, step| {
            let orphaned = step.orphan(experts, bank);
            outcome.and(orphaned)
        });
        if let Some((holder, joined)) = chain {
            requests
                .get_mut(&holder)
                .and_then(RequestEntry::resident_mut)
                .expect("invariant violated: a concluded chain's holder is not resident")
                .put_target(joined.commit());
        }
        drained
    }
}

/// Whether flight `flight` opens the chain of `steps`, the rest of a chain
/// queued behind its first chunk: that chunk is their front's predecessor.
fn opened_by<F: ProgramFamily>(steps: &VecDeque<QueuedStep<F>>, flight: u64) -> bool {
    steps.front().is_some_and(|step| step.predecessor == flight)
}

/// The binding right while no chain is under way
/// ([`ExecutorDomain::unchained`]): what changes bindings for other work
/// than a chain's holder (a request's open, a shrink) takes it, so it waits
/// for the chain's end rather than cutting the chain short.
pub struct Unchained<'b> {
    bindings: &'b mut StateBindings,
}

impl Unchained<'_> {
    /// The binding right it holds.
    pub(super) fn bindings(&mut self) -> &mut StateBindings {
        self.bindings
    }
}

impl<F: ProgramFamily> QueuedStep<F> {
    /// Wait for the step nobody claims and drop it: each of its passes
    /// that ran rows ahead in the bank's step is discarded, no verdict
    /// judging it (`ElasticBank::discarded`), once its step completed.
    fn orphan(self, experts: Option<&ExpertMeasures>, bank: Option<&dyn ElasticBank>) -> Result<(), DomainError> {
        // Its front head launch never ran: what it secured drops with it.
        drop(self.front);
        let requests = self.flight.requests;
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
            let completed = output.complete(core.batch(), super::recording(experts, core.work()))?;
            if let (Some(bank), Some(step)) = (bank, completed.expert_step) {
                let discarded = requests.iter().filter(|&&(_, rows, .., committed_rows)| rows > committed_rows);
                for &(request, ..) in discarded {
                    bank.discarded(step, request);
                }
            }
            Ok(())
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
/// (`continuation_of`), or `None` when a slot has none. A round that
/// verifies (a request in `verifying`: its drafter proposed, or its pass is
/// a verify pass) has none whatever its rows, as a verification of several
/// rows has none: a decode queued behind a one-row verification (every
/// proposal declined, or a verify pass whose chain was cut empty) is what
/// the next round waits for and discards. On a load whose features are a
/// separate draft's fused taps (`ModelLoadPlan::draft_fusion`) a round
/// drafting nothing publishes no feature: a decode that publishes one
/// prepares the next round's draft, and has none either; and a decode whose
/// next round prepares (a request in `publishing`) has one that publishes
/// its feature, so that round is the queued decode, discarding nothing.
pub(super) fn continuations(
    operations: &[Operation],
    verifying: impl Fn(RequestId) -> bool,
    publishing: impl Fn(RequestId) -> bool,
    fuses_draft_taps: bool,
    context: usize,
) -> Option<Vec<Operation>> {
    operations
        .iter()
        .map(|operation| {
            let request = operation.request();
            let drafts_next = verifying(request)
                || (fuses_draft_taps && operation.demand().computes_features());
            match drafts_next {
                true => None,
                false => continuation_of(operation, publishing(request), context),
            }
        })
        .collect()
}

/// The operation continuing `operation` by one decode row: the same request,
/// selection and output demand, one position later. A decode that also
/// publishes its row's feature for a drafter continues like a plain one: the
/// queued step publishes its own feature, read by the host only after it
/// finishes. A chained pass (its forward's `chain`) continues as the next
/// front pass as deep as its chain can reach, one past its own front rows
/// (the head lengthens the chain by one per pass), within the decode form's
/// rows and the context, its head entry one position later; the regime
/// holds it shallower as it queues (`ExecutorDomain::at_chain_depths`). Its
/// exact token is this pass's exact row's selection and each front token
/// the guess of the chain made behind this pass at that depth, the pending
/// token past the chain (`guess_chain`'s `next_selected`); all of them, and
/// the guesses' kinds ([`crate::FrontRow`]), are read on the device. A
/// one-row pass that chains nothing continues as a one-row pass. `None`
/// when the next step depends on the host (grammar masks, penalty
/// histories, logits demanded, conditioned rows, rows a decision may accept
/// beyond the exact row), or when its exact row
/// would pass the last of the `context` rows a sequence holds: its request
/// ends there, and no step follows it. The decode continuing `operation`
/// selects its next rows as its own were, publishing its feature where
/// `publishes`.
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
        front,
        chain,
    } = operation
    else {
        return None;
    };
    let [spec] = select.as_slice() else {
        return None;
    };
    let select_only = crate::batching::Demand::SELECT;
    let host_bound = |spec: &crate::SelectSpec| spec.mask.is_some() || spec.history.is_some();
    if tokens.len() != 1 + front.len()
        || (*demand != select_only
            && *demand != select_only | crate::batching::Demand::FEATURES
            && *demand != select_only | crate::batching::Demand::DEVICE_FEATURES)
        || host_bound(spec)
        || front.iter().any(|front| host_bound(&front.select))
    {
        return None;
    }
    let following = position.checked_add(1)?;
    let room = context.checked_sub(following)?.checked_sub(1)?;
    let reach = match chain {
        Some(_) => (front.len() + 1).min(room).min(crate::CHAINED_ROWS - 1),
        None => 0,
    };
    let next = crate::SelectSpec { position: spec.position.checked_add(1)?, ..spec.clone() };
    let front = (1..=reach)
        .map(|depth| {
            Some(crate::FrontRow {
                select: crate::SelectSpec { position: next.position.checked_add(depth)?, ..next.clone() },
                kind: crate::FrontKind::Device,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Operation::Forward {
        request: *request,
        kind: WorkKind::Decode,
        tokens: vec![crate::TokenId(0); 1 + reach],
        position: following,
        conditioning: None,
        demand: match publishes {
            true => *demand | crate::batching::Demand::FEATURES,
            false => *demand,
        },
        select: vec![next],
        committed: 1,
        prime: None,
        front,
        chain: match chain {
            Some(chain) => Some(crate::ChainEntry { position: chain.position.checked_add(1)? }),
            None => None,
        },
    })
}

/// `operation`, a chained decode continuation, held to `depth` front rows
/// (its exact row and its leading front rows).
fn at_depth(operation: &Operation, depth: usize) -> Operation {
    let mut operation = operation.clone();
    if let Operation::Forward { tokens, front, .. } = &mut operation {
        front.truncate(depth);
        tokens.truncate(1 + front.len());
    }
    operation
}

/// The first batch row of each slot of a launch whose slots run `rows`
/// rows each, in slot order: a launch's slots take their rows one after
/// another.
fn batch_rows(rows: impl IntoIterator<Item = usize>) -> Vec<usize> {
    rows.into_iter()
        .scan(0, |row, rows| {
            let first = *row;
            *row += rows;
            Some(first)
        })
        .collect()
}

/// The token rows of the step queued behind a front head launch, each the
/// slot of its predecessor it continues and its depth there
/// (`guess_chain`'s `continuing`): `predicted`'s rows in batch order, every
/// slot continuing its predecessor's slot of the same index.
fn continuing(predicted: &[Operation]) -> Vec<Option<FrontContinuation>> {
    predicted
        .iter()
        .enumerate()
        .flat_map(|(slot, operation)| (0..operation.row_count()).map(move |depth| Some(FrontContinuation { slot, depth })))
        .collect()
}

/// The operation that claims the queued slot `predicted`, once its
/// predecessor selected `selected` (per predecessor row, its selection and
/// the guess of the chain made behind it there), its predecessor's slot
/// beginning at row `first`: its exact token the selection of that row,
/// each front row's token and kind the guess of the chain at its depth
/// (held at row `first + depth - 1`), and past the chain an idle row on the
/// exact token ([`crate::FrontKind::Idle`]), as `guess_chain` writes its
/// `next_selected` rows. `None` where its predecessor selected no token for
/// that row (a failed selection's status): the group is no continuation.
fn claimed(predicted: &Operation, first: usize, selected: &[(Selected, Option<Guess>)]) -> Option<Operation> {
    let mut expected = predicted.clone();
    if let Operation::Forward { tokens, front, .. } = &mut expected {
        let (Selected::Token(exact), _) = selected.get(first).copied()? else {
            return None;
        };
        let chain = selected[first..].iter().map_while(|(_, guess)| *guess).take(front.len()).collect::<Vec<_>>();
        for (depth, (token, front)) in tokens[1..].iter_mut().zip(front.iter_mut()).enumerate() {
            (*token, front.kind) = match chain.get(depth) {
                Some(guess) => (guess.token, crate::FrontKind::Host(guess.kind)),
                None => (exact, crate::FrontKind::Idle),
            };
        }
        tokens[0] = exact;
    }
    Some(expected)
}

/// Whether `state` is the accepted state an advance at `position` of
/// `bindings` over `ranges` follows: its position, bank, tape and history.
fn follows_state(
    state: &SequenceState,
    position: usize,
    bindings: magnitude_state::OwnedAdvanceBindings<'_>,
    ranges: Vec<Vec<(usize, usize)>>,
) -> bool {
    state.position() == position
        && state.bank_index() == bindings.previous_bank
        && state.tape_rows() == bindings.previous_tape
        && state.domain_ranges() == ranges
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
    /// chain's until the chain concludes. The chain runs as one, so the
    /// owner submits that request's claims before any other group, and
    /// releases nothing (every release changes bindings, concluding it)
    /// until it ends; a holder that ended concludes it (`conclude_chain`).
    pub fn chain_holder(&self) -> Option<RequestId> {
        self.queue.holder()
    }

    /// Where `request`'s target state ends: its chain's while one is under
    /// way (its holder's state is the chain's), else its accepted state's.
    pub(super) fn target_end(&self, request: RequestId) -> Option<usize> {
        match &self.queue {
            Queue::Chain(HeldChain { holder, joined: Joined::Here(joined), .. }) if *holder == request => {
                Some(joined.end())
            }
            Queue::Empty | Queue::Opening(_) | Queue::Chain(_) | Queue::Successor(_) => {
                self.target_state(request).map(SequenceState::position)
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

    /// Conclude what the queue holds ([`Queue::conclude`]), the chain under
    /// way with it, whose holder keeps the state its reconciled chunks
    /// reached, and close the group it ran ([`Self::close_group`]). Every
    /// binding change does so first, and so does a holder that will not
    /// claim the rest of its chain (it ended).
    pub fn conclude_chain(&mut self) -> Result<(), DomainError> {
        let experts = self.measures.as_ref().and_then(|measures| measures.experts.as_ref());
        let concluded = self.queue.conclude(&mut self.requests, experts, self.family.expert_bank());
        concluded.and_then(|()| self.close_group())
    }

    /// Join `request`'s reconciled chained chunk of flight `flight` to the
    /// chain `joined` it carried, which returns ([`Self::return_chain`]);
    /// the chain's last chunk concludes it.
    pub(super) fn join_chain(
        &mut self,
        request: RequestId,
        flight: u64,
        mut joined: OwnedChainAdvance,
        chunk: magnitude_state::OwnedChainedAdvance,
    ) -> Result<(), DomainError> {
        let joins = joined.join(chunk).map_err(|(_, error)| DomainError::from(error));
        let whole = self.return_chain(request, flight, joined)
            && matches!(&self.queue, Queue::Chain(chain) if chain.steps.is_empty());
        // Its last chunk was claimed: the chain is whole.
        let concluded = match whole {
            true => self.conclude_chain(),
            false => Ok(()),
        };
        joins.and(concluded)
    }

    /// Return the chain state `joined` that `request`'s chunk of flight
    /// `flight` carried: to its chain, still under way (`true`); otherwise
    /// that chain was concluded meanwhile, and its holder's state is the one
    /// `joined` reached.
    pub(super) fn return_chain(&mut self, request: RequestId, flight: u64, joined: OwnedChainAdvance) -> bool {
        match &mut self.queue {
            Queue::Chain(chain) if chain.holder == request && matches!(chain.joined, Joined::Out(out) if out == flight) => {
                chain.joined = Joined::Here(joined);
                true
            }
            Queue::Empty | Queue::Opening(_) | Queue::Chain(_) | Queue::Successor(_) => {
                self.put_target_state(request, joined.commit());
                false
            }
        }
    }

    /// Open the chain queued behind `first`, `request`'s reconciled first
    /// chunk of its prompt group, of flight `flight` (`Queue::opens_chain`).
    /// Where that queue was concluded meanwhile, the chunk publishes as a
    /// step.
    pub(super) fn open_chain(
        &mut self,
        request: RequestId,
        flight: u64,
        first: OwnedStateAdvance,
    ) -> Result<(), DomainError> {
        let Some(steps) = self.queue.take_opening(flight) else {
            let rows = first.rows();
            let (OwnedAdvanceResolution::Aborted(state) | OwnedAdvanceResolution::Committed(state)) =
                first.commit(rows).map_err(|(state, error)| {
                    self.put_target_state(request, state);
                    DomainError::from(error)
                })?;
            self.put_target_state(request, state);
            return Ok(());
        };
        match OwnedChainAdvance::begin(first) {
            Ok(joined) => {
                self.queue = Queue::Chain(HeldChain { holder: request, joined: Joined::Here(joined), steps });
                Ok(())
            }
            Err((first, error)) => {
                // The queued chunks follow its rows: they go first.
                self.queue = Queue::Opening(steps);
                let concluded = self.conclude_chain();
                self.put_target_state(request, first.abort());
                concluded.and(Err(DomainError::from(error)))
            }
        }
    }

    /// What runs behind `operations` once they are submitted or claimed, a
    /// function of the round's operations, the chunks the scheduler planned
    /// for it and the load. A lone prompt chunk opens a chain of the planned
    /// chunks continuing it, a place each of the class its group holds
    /// (`group_places`, claimed as it opened, `open_group`), that the family
    /// runs in a chain; otherwise, with pipelining, every slot's successor
    /// of one kind queues behind it.
    pub(super) fn behind(&self, operations: &[Operation]) -> Behind {
        self.behind_within(operations, self.group_places().get())
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
        if let Some(operations) = continuations(
            operations,
            |request| self.resident(request).is_some_and(|resident| resident.verifying),
            |request| self.resident(request).is_some_and(|resident| resident.publishing),
            fuses_draft_taps,
            context,
        ) {
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
            let entry_before = |count| match previous {
                None => priming.map_or(Ok(None), |priming| fitting(priming.advance.successor(count))),
                Some(previous) => previous
                    .priming
                    .as_ref()
                    .map_or(Ok(None), |priming| fitting(priming.advance.successor(count))),
            };
            let entry = self.queued_priming(&operations, entry_before)?;
            let Some(entry) = entry else {
                break;
            };
            let Some(front) = self.queued_front(&operations, None, entry.as_ref())? else {
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
                front,
                launch_trace: None,
            });
        }
        Ok((launches, pending))
    }

    /// Queue the successor of `flight` (just submitted or claimed for
    /// `operations`, see `behind`) when a decode continuation's flight has a
    /// selection tensor to read tokens from, and the successors and leases
    /// fit without growing anything. Otherwise the next step is submitted by
    /// the owner as usual. The front head launch due behind a flight whose
    /// forwards chain (`front`) launches here, whatever queues: right behind
    /// it and before its successor, whose rows it continues (none where no
    /// chained successor is queued).
    pub(super) fn queue_successor(
        &mut self,
        flight: &mut TargetWork<F>,
        front: Option<DueFront>,
        operations: &[Operation],
    ) -> Result<(), DomainError> {
        let behind = self.behind(operations);
        self.planned.clear();
        let next = self.queueable(flight, operations, behind);
        if let Some(front) = front {
            let continuing = match &next {
                Ok(Some((predicted, TokenSource::Chain))) => continuing(predicted),
                Ok(_) | Err(_) => Vec::new(),
            };
            flight.front = Some(self.launch_front(&flight.submission, front, &continuing)?);
        }
        let Some((predicted, source)) = next? else {
            return Ok(());
        };
        let (tokens, before) = match source {
            TokenSource::Chain => {
                let front = flight
                    .front
                    .as_ref()
                    .expect("invariant violated: a chained continuation's predecessor ran no front head launch");
                let chain = front.submission.output().front().expect("invariant violated: a front head launch makes no chain");
                let HeadState::Owned { advances, .. } = front.submission.launch().state() else {
                    panic!("invariant violated: a front head launch owns no advances");
                };
                (
                    SuccessorTokens::Selected { selected: chain.next_selected.clone(), guesses: Some(chain.next_guesses.clone()) },
                    Some(advances.as_slice()),
                )
            }
            TokenSource::Selected(selected) => (SuccessorTokens::Selected { selected, guesses: None }, None),
            TokenSource::Host => (SuccessorTokens::Host, None),
        };
        // A planned prompt chunk's drafter entry follows the in-flight entry
        // of the chunk before it.
        let entry_before = flight
            .priming
            .as_ref()
            .filter(|priming| matches!(predicted.as_slice(), [operation] if operation.request() == priming.request))
            .and_then(|priming| match priming.submission.launch().state() {
                HeadState::Owned { advances, .. } => advances.first(),
                HeadState::TargetReads { .. } => {
                    panic!("invariant violated: a drafter entry owns no advances")
                }
            });
        let entry_before = |count| entry_before.map_or(Ok(None), |advance| fitting(advance.successor(count)));
        let entry = self.queued_priming(&predicted, entry_before)?;
        let Some(entry) = entry else {
            return Ok(());
        };
        let Some(front) = self.queued_front(&predicted, before, entry.as_ref())? else {
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
            front,
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

    /// The step that may queue behind `flight` (`behind`) and where its
    /// tokens are read: none while steps are queued, the expert bank wants
    /// a gap, the slots differ from the flight's, nothing pipelines behind
    /// it, or a continuation of no chain has no selection tensor to read.
    fn queueable(
        &self,
        flight: &TargetWork<F>,
        operations: &[Operation],
        behind: Behind,
    ) -> Result<Option<(Vec<Operation>, TokenSource)>, DomainError> {
        // Behind a claimed chain step runs the rest of its chain: only an
        // empty queue takes a successor.
        if !matches!(self.queue, Queue::Empty) {
            return Ok(None);
        }
        // The expert bank waits for a break: what it staged is written only
        // where no step runs (`ElasticBank::wants_gap`), so the next step
        // begins after this one ends.
        if self.family.expert_bank().is_some_and(|bank| bank.wants_gap()) {
            return Ok(None);
        }
        // Tokens are the flight's selection rows, one per slot in slot
        // order, so the continuation keeps every slot in the same order.
        if operations.len() != flight.requests.len()
            || operations
                .iter()
                .zip(&flight.requests)
                .any(|(operation, (request, ..))| operation.request() != *request)
        {
            return Ok(None);
        }
        Ok(match behind {
            Behind::Nothing | Behind::Chain { .. } => None,
            // A launch's forwards chain all or none (`requirements`).
            Behind::Continuation { operations } if operations.iter().any(|operation| operation.chain().is_some()) => {
                self.at_chain_depths(operations)?.map(|operations| (operations, TokenSource::Chain))
            }
            Behind::Continuation { operations } => flight
                .submission
                .output()
                .readout
                .as_ref()
                .and_then(|readout| readout.selected.clone())
                .map(|selected| (operations, TokenSource::Selected(selected))),
            Behind::NextChunk { operations } => Some((operations, TokenSource::Host)),
        })
    }

    /// The chained decode continuation `operations` at the depths it runs,
    /// none where the regime wants a verify pass of the launch
    /// (`magnitude_cost_model::Regime::pass`): a verify pass is never
    /// queued, so the pass after the flight is the host's to build, in the
    /// form the regime wants then (`ExecutorDomain::front_pass`). Else, of
    /// the slots whose chain reaches a front row (`continuation_of`,
    /// within the context), each as deep as its chain reaches and the regime
    /// wants of the launch (`TwoRowPricing::regime`), as many fronting,
    /// first first, as a chained launch's rows hold beside every exact row
    /// at that depth (`crate::two_row::fronted_requests`); every other slot
    /// one row. Its rows so fit a chained launch
    /// (`crate::ResourceLimits::chained_rows`), the budget the owner claims
    /// it within, and each slot's context. A row the chain does not reach
    /// runs idle on its exact token (`guess_chain`'s `next_selected`), so no
    /// depth refuses a continuation, and the host builds the same pass as
    /// it claims it (`ExecutorDomain::front_pass`,
    /// `crate::FrontGrant::Exactly`).
    fn at_chain_depths(&self, operations: Vec<Operation>) -> Result<Option<Vec<Operation>>, DomainError> {
        let requests = operations.len();
        let reaching = operations.iter().filter(|operation| operation.front_rows() > 0).count();
        let regime = if reaching == 0 {
            None
        } else {
            match self.two_row_pricing()? {
                Some(pricing) => pricing.regime(requests, reaching)?,
                None => None,
            }
        };
        let depth = match regime {
            Some(FrontRegime { pass: magnitude_cost_model::Form::Verify(_), .. }) => return Ok(None),
            Some(FrontRegime { depth, .. }) => depth,
            None => 0,
        };
        let mut fronting = crate::two_row::fronted_requests(self.chained_rows(), requests, depth).min(reaching);
        Ok(Some(operations
            .iter()
            .map(|operation| match operation.front_rows() > 0 && fronting > 0 {
                true => {
                    fronting -= 1;
                    at_depth(operation, depth.min(operation.front_rows()))
                }
                false => at_depth(operation, 0),
            })
            .collect()))
    }

    /// The front head launch of `predicted`, a step queued behind another,
    /// secured as it queues: none where its forwards chain nothing; where
    /// they do, each slot's head advance (one committed entry row and a
    /// scratch row per front row) following the head advance right before
    /// it: of a decode continuation, its predecessor's in the front head
    /// launch behind it (`before`, in slot order); of the prompt's closing
    /// chunk, its drafter entry `priming`, which a queued chunk always
    /// enters rows by (it follows a chunk, so its prime pairs every token
    /// with the feature before it). And its launch leases. `None` when they
    /// do not fit without growing anything.
    fn queued_front(
        &self,
        predicted: &[Operation],
        before: Option<&[TentativeAdvance]>,
        priming: Option<&QueuedPriming>,
    ) -> Result<Option<Option<FrontHead<OwnedSuccessorAdvance>>>, DomainError> {
        if predicted.iter().all(|operation| operation.chain().is_none()) {
            return Ok(Some(None));
        }
        let advances = match (predicted, before, priming) {
            ([Operation::Forward { prime: Some(_), .. }], None, Some(priming)) => {
                fitting(priming.advance.successor(1))?.map(|advance| vec![advance])
            }
            (_, Some(before), None) if before.len() == predicted.len() => fitting(TentativeAdvance::successors(
                before.iter().zip(predicted).map(|(advance, operation)| (advance, 1, operation.front_rows())),
            ))?,
            _ => panic!(
                "invariant violated: a queued chained step follows neither a front head launch nor its prompt chunk's drafter entry"
            ),
        };
        let Some(advances) = advances else {
            return Ok(None);
        };
        let Some((graph_workspace, graph_output)) = self
            .resources
            .head_graph()
            .and_then(|head| Some((head.acquire_workspace().ok()?, head.acquire_output().ok()?)))
        else {
            return Ok(None);
        };
        let slots = predicted
            .iter()
            .zip(advances)
            .map(|(operation, advance)| SecuredFrontSlot {
                request: operation.request(),
                advance,
                selects: super::front_selects(operation),
            })
            .collect();
        Ok(Some(Some(FrontHead { slots, graph_workspace, graph_output })))
    }

    /// The drafter entry of a step queued behind another: none when the
    /// step primes no row (as a launched step's, `reserve`), or its advance
    /// (`successor` of the entry before, given its rows) and its draft
    /// launch leases. `None` when the step primes and its entry does not
    /// fit.
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
        if prime.tokens.is_empty() {
            return Ok(Some(None));
        }
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
            front,
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
            verifies: false,
            requests,
            submission,
            priming: None,
            front: None,
            started,
            epoch,
            runnable: started,
            previous_selection: None,
            id,
            continuation: None,
        };
        if let (Some(priming), [chunk]) = (priming, operations.as_slice()) {
            flight = self.launch_priming(
                flight,
                chunk,
                TentativeAdvance::Successor(priming.advance),
                predecessor.submission.output().boundary.clone(),
                priming.graph_workspace,
                priming.graph_output,
            )?;
        }
        Ok(QueuedStep {
            flight,
            predecessor: predecessor.id,
            reads: batch_rows(predecessor.requests.iter().map(|&(_, rows, ..)| rows)),
            predicted: operations,
            selected: None,
            front: front.map(Box::new),
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
        let launch = step.flight.submission.launch();
        let mut slots = Vec::with_capacity(operations.len());
        for operation in operations {
            let slot = step
                .predicted
                .iter()
                .position(|predicted| predicted.request() == operation.request())?;
            let expected = match launch.tokens() {
                // Each of the slot's rows read its predecessor's exact
                // selection, or the guess of the chain made behind its
                // predecessor at its depth, or idle the exact selection
                // past the chain (`claimed`). A predecessor's exact row that
                // selected no token refuses the whole claim, and the queued
                // step is orphaned. Every table row the queued step read for
                // a token is the token's (a landed read,
                // `DeviceTable::read_selected`). A slot whose table rows
                // could not be read is claimed all the same: its sequence
                // fails when the step finishes, and the others continue
                // (`finish_target_work`).
                TargetTokens::Selected { .. } => claimed(&step.predicted[slot], step.reads[slot], selected)?,
                TargetTokens::Host(..) => step.predicted[slot].clone(),
            };
            let advance = &launch.slot(slot)?.advance;
            // A chained chunk follows its chain; any other slot the
            // request's accepted state.
            let follows = match advance {
                TentativeAdvance::Chained(_) => match &self.queue {
                    Queue::Chain(HeldChain { holder, joined: Joined::Here(joined), .. }) => {
                        *holder == operation.request() && joined.end() == advance.position()
                    }
                    Queue::Empty | Queue::Opening(_) | Queue::Chain(_) | Queue::Successor(_) => false,
                },
                TentativeAdvance::Accepted(_) | TentativeAdvance::Successor(_) => follows_state(
                    self.target_state(operation.request())?,
                    advance.position(),
                    advance.bindings(),
                    advance.domain_ranges(),
                ),
            };
            // A chained decode's head advance follows the request's
            // accepted head state; the closing chunk's, its drafter entry.
            let head_follows = match (&step.front, operation) {
                (Some(front), Operation::Forward { prime: None, .. }) => {
                    let advance = &front.slots[slot].advance;
                    self.head_state(operation.request()).is_some_and(|state| {
                        follows_state(state, advance.position(), advance.bindings(), advance.domain_ranges())
                    })
                }
                _ => true,
            };
            let drafter_follows = match operation {
                Operation::Forward {
                    prime: Some(prime), ..
                } => self
                    .head_state(operation.request())
                    .is_some_and(|head| head.position() == prime.position),
                _ => true,
            };
            if slots.contains(&slot) || !follows || !drafter_follows || !head_follows || operation != &expected {
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
    /// `operations`, the slots `slots` its reservation found
    /// ([`Self::claim_slots`]), which nothing changes before the claim. Each
    /// claimed slot takes its request's accepted state, which its successor
    /// attaches to at finish, or, a chained chunk, the chain it joins;
    /// unclaimed slots are discarded then. With it, the front head launch
    /// due behind it, of a step whose forwards chain.
    pub(super) fn claim_queued(&mut self, operations: &[Operation], slots: Vec<usize>) -> (TargetWork<F>, Option<DueFront>) {
        assert_eq!(
            self.claim_slots(operations).as_ref(),
            Some(&slots),
            "invariant violated: a claimed queued step changed after its reservation"
        );
        let QueuedStep { mut flight, front, predicted, .. } =
            self.queue.pop_front().expect("invariant violated: a claimable queued step is absent");
        let mut sources = (0..flight.requests.len()).map(|_| Source::Unclaimed).collect::<Vec<_>>();
        // Of a chained decode, each claimed slot's head advance attaches to
        // its request's accepted head state; the closing chunk's follows the
        // head advance before it, its drafter entry's, which its outcome
        // joins (`FrontAdvance::Entry`).
        let chunk = matches!(predicted.as_slice(), [Operation::Forward { prime: Some(_), .. }]);
        let mut heads = (front.is_some() && !chunk).then(|| (0..flight.requests.len()).map(|_| None).collect::<Vec<_>>());
        // `claim_slots` found every claimed request's accepted states, or the
        // chain a chained chunk joins, which the chunk takes with it.
        // Take-outs: the target state returns as the flight finishes
        // (`finish_target_work` attaches it); the head state as its front
        // head launch finishes (`finish_front`).
        let launch = flight.submission.launch();
        for (operation, &slot) in operations.iter().zip(&slots) {
            sources[slot] = match &launch
                .slot(slot)
                .expect("invariant violated: claimed slot is absent")
                .advance
            {
                TentativeAdvance::Chained(_) => {
                    let Queue::Chain(chain) = &mut self.queue else {
                        panic!("invariant violated: a claimed chained chunk without its chain");
                    };
                    let Joined::Here(joined) = std::mem::replace(&mut chain.joined, Joined::Out(flight.id)) else {
                        panic!("invariant violated: a chain claimed by two chunks at once");
                    };
                    Source::Chain(joined)
                }
                TentativeAdvance::Accepted(_) | TentativeAdvance::Successor(_) => {
                    let state = self
                        .take_target_state(operation.request())
                        .expect("invariant violated: a claimed slot's accepted state is absent");
                    Source::Accepted(state)
                }
            };
            if let Some(heads) = &mut heads {
                heads[slot] = self.take_head_state(operation.request());
            }
        }
        flight.continuation = Some(sources);
        // Take-out: the drafter entry's accepted head state, returned as it
        // finishes (`finish_priming_work` attaches it).
        if let Some(priming) = &mut flight.priming {
            priming.continuation = match self.take_head_state(priming.request) {
                Some(state) => PrimingContinuation::Accepted(state),
                None => PrimingContinuation::Unclaimed,
            };
        }
        let front = front.map(|head| match heads {
            Some(states) => DueFront::Claimed(*head, states),
            None => DueFront::Held(HeldFront::Entry(*head)),
        });
        (flight, front)
    }

    /// The binding right for work other than a chain's holder's
    /// ([`Unchained`]); none while a chain is under way, whose holder claims
    /// it to its end or concludes it: what changes bindings waits for that.
    pub fn unchained<'b>(&self, bindings: &'b mut StateBindings) -> Option<Unchained<'b>> {
        self.queue.holder().is_none().then_some(Unchained { bindings })
    }

    /// The launch of `predicted` queued behind a step whose advances are
    /// `predecessor` (in flight or about to be): each slot's advance follows
    /// its predecessor's as `follow` says. `ahead` are the operations of its
    /// chain before it, whose tokens are not yet fed (`target_preceding`).
    /// `None` when the advances or their leases do not fit without growing
    /// anything, or a slot needs input slices: the step is then submitted as
    /// usual.
    fn build_successor<'a>(
        &mut self,
        predecessor: impl Iterator<Item = &'a TentativeAdvance>,
        predicted: &[Operation],
        follow: Follow,
        ahead: &[Operation],
    ) -> Result<Option<ValidatedTargetLaunch>, DomainError> {
        // The step's rows are admitted and placed as its own provisioning
        // would place them once its predecessors commit.
        let (advances, tokens) = match follow {
            Follow::Successor(tokens) => (
                TentativeAdvance::successors(
                    predecessor
                        .zip(predicted)
                        .map(|(advance, operation)| (advance, operation.committed_rows(), operation.front_rows())),
                )
                .map(|successors| {
                    successors
                        .into_iter()
                        .map(TentativeAdvance::Successor)
                        .collect::<Vec<_>>()
                }),
                tokens,
            ),
            Follow::Chained => (
                TentativeAdvance::chained(
                    predecessor.zip(predicted).map(|(advance, operation)| (advance, operation.row_count())),
                )
                .map(|chunks| {
                    chunks
                        .into_iter()
                        .map(TentativeAdvance::Chained)
                        .collect::<Vec<_>>()
                }),
                SuccessorTokens::Host,
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
                |domain| advance.domain_span_count(domain),
                operation.row_count(),
            ));
            slots.push(slot);
        }
        // The continuation follows the tokens fed so far.
        let preceding = self.target_preceding(predicted, ahead).map_err(DomainError::Input)?;
        let class_limits = self.target_class_limits();
        let vocabulary = self.definition.decoder.vocabulary as usize;
        let batch = match tokens {
            SuccessorTokens::Selected { .. } => ValidatedTargetBatch::from_slots(&slots, vocabulary, class_limits),
            SuccessorTokens::Host => ValidatedTargetBatch::covering(&slots, vocabulary, class_limits, segments),
        }
        .map_err(|error| DomainError::Input(error.to_string()))?;
        let tokens = match tokens {
            SuccessorTokens::Selected { selected, guesses } => {
                let rows = batch.class().rows() as u64;
                if selected
                    .tensor()
                    .extents()
                    .first()
                    .is_none_or(|extent| *extent < rows)
                {
                    return Ok(None);
                }
                // The guesses are the selections' rows'.
                let leading = |tensor: crate::GraphOutputTensor| {
                    tensor.slice_leading(0, rows).map_err(|error| DomainError::Input(error.to_string()))
                };
                let target_slots = predicted
                    .iter()
                    .zip(advances)
                    .zip(preceding)
                    .map(|((operation, advance), preceding)| {
                        assert!(
                            matches!(
                                operation,
                                Operation::Forward { front, .. }
                                    if front.iter().all(|row| matches!(row.kind, crate::operation::FrontKind::Device))
                            ),
                            "a selected-token launch slot must run a forward with Device front rows"
                        );
                        TargetSlot {
                            request: operation.request(),
                            advance,
                            preceding,
                        }
                    })
                    .collect();
                TargetTokens::Selected {
                    selected: leading(selected)?,
                    guesses: guesses.map(leading).transpose()?,
                    slots: target_slots,
                }
            }
            SuccessorTokens::Host => TargetTokens::Host(
                predicted
                    .iter()
                    .zip(advances)
                    .zip(preceding)
                    .map(|((operation, advance), preceding)| TargetHostSlot {
                        slot: TargetSlot {
                            request: operation.request(),
                            advance,
                            preceding,
                        },
                        conditioning: None,
                        conditioning_slices: Vec::new(),
                        guesses: SlotGuess::of(operation),
                    })
                    .collect(),
            ),
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
        let work = crate::operation::LaunchWork::of(predicted.iter().filter_map(|operation| match operation {
            Operation::Forward { kind, .. } => Some(*kind),
            _ => None,
        }));
        let inputs = TargetLaunchInputs::new(
            batch,
            work,
            tokens,
            graph_workspace,
            graph_output,
            readout_workspace,
            readout_output,
        )
        .with_boundary(boundary);
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
    pub(super) fn predecessor_selected(&mut self, flight: u64, selected: &[(Selected, Option<Guess>)], completed: Instant) {
        if let Some(step) = self.queue.steps_mut().find(|step| step.predecessor == flight) {
            step.selected = Some(selected.to_vec());
            step.flight.runnable = completed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{at_depth, batch_rows, claimed, continuation_of, continuations, continuing, fitting};

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
            front: Vec::new(),
            chain: None,
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
        assert!(continuations(std::slice::from_ref(&declined), |_| false, |_| false, false, 9).is_some());
        assert!(continuations(std::slice::from_ref(&declined), |id| id == RequestId(3), |_| false, false, 9).is_none());
    }

    /// On a load whose features are a separate draft's fused taps, a round
    /// drafting nothing publishes no feature: a decode publishing one
    /// prepares the next round's draft, and queues no decode behind it; a
    /// plain decode queues one as anywhere, which publishes its feature
    /// where the next round prepares.
    #[test]
    fn a_decode_preparing_a_fused_draft_queues_no_decode_behind_it() {
        let preparing = decode(Demand::SELECT | Demand::FEATURES, vec![TokenId(5)]);
        assert!(continuations(std::slice::from_ref(&preparing), |_| false, |_| false, true, 9).is_none());
        let plain = decode(Demand::SELECT, vec![TokenId(5)]);
        let queued = continuations(std::slice::from_ref(&plain), |_| false, |_| false, true, 9).unwrap();
        assert_eq!(queued[0].demand(), Demand::SELECT);
        let queued = continuations(std::slice::from_ref(&plain), |_| false, |id| id == plain.request(), true, 9).unwrap();
        assert_eq!(queued[0].demand(), Demand::SELECT | Demand::FEATURES);
    }

    fn spec(position: usize) -> SelectSpec {
        SelectSpec { sampling: Sampling::Greedy, seed: 1, position, domain: 0, mask: None, shaping: Shaping::default(), history: None }
    }

    /// A chained front pass of `request` at `position`, its head entry at
    /// `position - 1`, its exact row selecting at `select`, of `tokens` (its
    /// exact token, then its front rows'), each front row of kind `kind`.
    fn pass(request: RequestId, position: usize, select: usize, tokens: &[u32], kind: crate::FrontKind) -> Operation {
        Operation::Forward {
            request,
            kind: WorkKind::Decode,
            tokens: tokens.iter().copied().map(TokenId).collect(),
            position,
            conditioning: None,
            demand: Demand::SELECT | Demand::DEVICE_FEATURES,
            select: vec![spec(select)],
            committed: 1,
            prime: None,
            front: (1..tokens.len()).map(|depth| crate::FrontRow { select: spec(select + depth), kind }).collect(),
            chain: Some(crate::ChainEntry { position: position - 1 }),
        }
    }

    /// A predecessor's selections, row by row from `first`: each row's
    /// token the one after the row before's, and the chain made behind it
    /// there, its guess `chain[row]`, none past it.
    fn selections(rows: usize, first: u32, chain: &[Option<u32>]) -> Vec<(crate::Selected, Option<crate::Guess>)> {
        (0..rows)
            .map(|row| {
                let token = TokenId(first + row as u32);
                let guess = chain.get(row).copied().flatten().map(|token| crate::Guess { token: TokenId(token), kind: crate::GuessKind::Chained });
                (crate::Selected::Token(token), guess)
            })
            .collect()
    }

    /// A chained pass continues one front row deeper than it ran (the head
    /// lengthens the chain by one a pass), its head entry one position on,
    /// within a chained launch's rows and the context; a decode of no chain
    /// continues as one row of no chain.
    #[test]
    fn a_chained_pass_continues_one_deeper_than_it_ran() {
        let request = RequestId(3);
        let next = continuation_of(&pass(request, 7, 4, &[1, 2], crate::FrontKind::Device), false, 64).unwrap();
        assert_eq!((next.front_rows(), next.chain()), (2, Some(crate::ChainEntry { position: 7 })));
        let Operation::Forward { select, front, .. } = &next else { unreachable!() };
        assert_eq!(std::iter::once(&select[0]).chain(front.iter().map(|front| &front.select)).map(|spec| spec.position).collect::<Vec<_>>(), [5, 6, 7]);
        let rows = crate::CHAINED_ROWS;
        let deepest = pass(request, 7, 4, &vec![1; rows], crate::FrontKind::Device);
        assert_eq!(continuation_of(&deepest, false, 64).unwrap().row_count(), rows, "no deeper than a chained launch");
        assert_eq!(continuation_of(&pass(request, 7, 4, &[1, 2], crate::FrontKind::Device), false, 10).unwrap().front_rows(), 1, "within the context");
        let mut unchained = pass(request, 7, 4, &[1], crate::FrontKind::Device);
        if let Operation::Forward { chain, .. } = &mut unchained {
            *chain = None;
        }
        let next = continuation_of(&unchained, false, 64).unwrap();
        assert_eq!((next.row_count(), next.chain()), (1, None));
    }

    /// Two chained passes queued behind one launch, held to the regime's
    /// depth: each reads its own predecessor's rows, wherever its own run,
    /// as the front head launch writes them (`continuing`). Its claim takes
    /// the exact selection, the chain's guesses as deep as the chain goes,
    /// and idle rows on the exact token past it.
    #[test]
    fn a_queued_pass_claims_its_predecessors_chain_and_idles_past_it() {
        let (short, long) = (RequestId(1), RequestId(2));
        let ran = [pass(short, 7, 4, &[1, 2], crate::FrontKind::Device), pass(long, 9, 6, &[1, 2, 3, 4], crate::FrontKind::Device)];
        let reads = batch_rows(ran.iter().map(Operation::row_count));
        assert_eq!(reads, [0, 2]);
        let next = ran.iter().map(|operation| continuation_of(operation, false, 64).unwrap()).collect::<Vec<_>>();
        let within = [at_depth(&next[0], 1), at_depth(&next[1], 4)];
        assert_eq!(within.iter().map(Operation::front_rows).collect::<Vec<_>>(), [1, 4]);
        let rows = continuing(&within);
        let continued = |slot, depth| Some(crate::batching::FrontContinuation { slot, depth });
        assert_eq!(rows, [continued(0, 0), continued(0, 1), continued(1, 0), continued(1, 1), continued(1, 2), continued(1, 3), continued(1, 4)]);
        // The short pass's chain holds both its depths; the long one's
        // fell past its second.
        let selected = selections(6, 20, &[Some(40), Some(41), Some(50), Some(51), None, None]);
        let claim = |slot: usize| match claimed(&within[slot], reads[slot], &selected).unwrap() {
            Operation::Forward { tokens, front, .. } => (tokens, front.into_iter().map(|front| front.kind).collect::<Vec<_>>()),
            _ => unreachable!("a claimed pass is a forward"),
        };
        let chained = crate::FrontKind::Host(crate::GuessKind::Chained);
        assert_eq!(claim(0), (vec![TokenId(20), TokenId(40)], vec![chained]));
        assert_eq!(
            claim(1),
            (vec![TokenId(22), TokenId(50), TokenId(51), TokenId(22), TokenId(22)], vec![chained, chained, crate::FrontKind::Idle, crate::FrontKind::Idle]),
        );
    }

    /// A pass queued as the regime turns GPU-bound (its depth 0) runs one
    /// row, the exact token its predecessor selected, and the host's pass
    /// granted exactly that depth claims it; a deeper one would orphan it.
    #[test]
    fn a_pass_queued_gpu_bound_runs_one_row_and_is_claimed() {
        let request = RequestId(3);
        let ran = pass(request, 7, 4, &[1, 2, 3, 4], crate::FrontKind::Host(crate::GuessKind::Chained));
        let next = continuation_of(&ran, false, 64).unwrap();
        assert_eq!(next.front_rows(), 4, "it continues as deep as its chain reaches");
        let within = at_depth(&next, 0);
        assert_eq!((within.front_rows(), within.row_count()), (0, 1));
        let selected = selections(5, 30, &[Some(31), Some(32), None, None, None]);
        let built = pass(request, 8, 5, &[30], crate::FrontKind::Device);
        assert_eq!(claimed(&within, 0, &selected), Some(built), "the host's pass at depth 0 claims it");
        let deeper = pass(request, 8, 5, &[30, 31], crate::FrontKind::Host(crate::GuessKind::Chained));
        assert_ne!(claimed(&within, 0, &selected), Some(deeper), "a deeper pass would orphan it");
    }
}
