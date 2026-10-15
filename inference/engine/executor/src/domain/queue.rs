//! Steps queued on the device behind a submitted target step, before the
//! owner asks for them. What runs behind a step is planned in one place
//! (`ExecutorDomain::behind`):
//!
//! - A layer-major prompt group: a lone prompt chunk of a load that groups
//!   takes the planned chunks continuing it, within the load's capacity,
//!   submitted as one chain (`TargetProgram::submit`), so each streamed layer
//!   is filled once for the group. The group runs on the device as one: the
//!   scheduler claims its chunks before any other work (`group_holder`).
//! - Cross-step pipelining ("lookahead"): right after a continuable step N is
//!   submitted, its successor N+1 is submitted behind it. A decode
//!   continuation's input token is N's selection read on the device, and its
//!   selection is N's with the position advanced by one; a prompt's next
//!   chunk is known before N finishes, planned by the owner
//!   (`plan_successors`). The device then runs N+1 without waiting for the
//!   host to observe N and submit the next step.
//!
//! Every queued step follows the one before it: each slot's successor advance
//! follows its predecessor's, and a chunk that primes its drafter enters it
//! behind its own step, chained on the entry before. Queued steps live in the
//! domain's `StateBindings`: they are the only work that holds bindings
//! between flights. The owner carries them there but never sees them. When
//! the operations it next submits equal the front step's prediction (token
//! included, known once its predecessor finished) and the accepted states are
//! the ones the successors follow, the group claims the queued step: nothing
//! is reserved or launched, and its flight is returned. Any other target
//! group, and any binding change, orphans them: the domain waits for them and
//! drops them, so their rows, banks and leases return first. Every device
//! write of a queued step goes to rows and banks its own successors own, so
//! an orphan never changes accepted state.

use super::*;
use crate::programs::{SubmittedHead, SubmittedTarget};
use magnitude_state::{OwnedSuccessorAdvance, TentativeAdvance};

/// What runs behind a submitted target step.
pub(super) enum Behind {
    /// Nothing: the owner submits the next step once this one finishes.
    Nothing,
    /// The rest of a layer-major prompt group: the step's planned chunks, in
    /// order, submitted with it as one chain. A group opens only where a
    /// step launches.
    Group(Vec<Operation>),
    /// One pipelined successor, submitted right after the step: per slot, its
    /// operation, and whether its tokens are the step's selections (a decode
    /// continuation) rather than its operations' own (a planned prompt
    /// chunk).
    Successor {
        operations: Vec<Operation>,
        selected_tokens: bool,
    },
}

impl Behind {
    /// The queued operations of `request`.
    pub(super) fn of(&self, request: RequestId) -> impl Iterator<Item = &Operation> {
        let operations = match self {
            Self::Nothing => &[][..],
            Self::Group(operations) | Self::Successor { operations, .. } => operations,
        };
        operations
            .iter()
            .filter(move |operation| operation.request() == request)
    }
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
    selected_tokens: bool,
    priming: Option<QueuedPriming>,
    launch_trace: Option<seismic::SubmissionTrace>,
}

pub(super) struct QueuedStep<F: ProgramFamily> {
    flight: TargetWork<F>,
    /// The flight whose selections are this one's tokens.
    predecessor: u64,
    /// Per slot, the operation that claims it (a decode continuation with a
    /// placeholder token).
    predicted: Vec<Operation>,
    /// Whether the queued step's tokens are its predecessor's selections (a
    /// decode continuation) rather than its operations' own (a planned
    /// prefill chunk).
    selected_tokens: bool,
    /// Per slot, the predecessor's selection, once it finished.
    selected: Option<Vec<Selected>>,
    /// A chunk of a layer-major prompt group: it runs on the device as one
    /// unit with its group, so its request's claims come before any other
    /// work (`group_holder`). A pipelined successor is speculative instead:
    /// any other work orphans it.
    in_group: bool,
}

/// The operation continuing `operation` by one decode row: the same request
/// and selection, one position later. `None` when the next step depends on
/// the host (grammar masks, penalty histories, logits or features demanded,
/// conditioned rows, more than one row).
pub(super) fn continuation_of(operation: &Operation) -> Option<Operation> {
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
    if tokens.len() != 1
        || *demand != crate::batching::Demand::SELECT
        || spec.mask.is_some()
        || spec.history.is_some()
    {
        return None;
    }
    let mut next = spec.clone();
    next.position = spec.position.checked_add(1)?;
    Some(Operation::Forward {
        request: *request,
        kind: WorkKind::Decode,
        tokens: vec![crate::TokenId(0)],
        position: position.checked_add(1)?,
        conditioning: None,
        demand: *demand,
        select: vec![next],
        committed: 1,
        prime: None,
    })
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Plan the operations that follow the next submitted target group, when
    /// they are known before it finishes. Its provisioning makes room for
    /// them and its launch queues them; any other group discards them.
    pub fn plan_successors(&mut self, planned: Vec<Operation>) {
        self.planned = planned;
    }

    /// The most chunks of a prompt one chain runs as a layer-major group
    /// (`ResourceLimits::group_capacity`). The scheduler chooses each round's
    /// group within it by the chunks it plans.
    pub fn group_capacity(&self) -> usize {
        self.execution.policy().limits().group_capacity.get()
    }

    /// The request whose layer-major prompt group still has chunks queued.
    /// The group runs on the device as one, so the owner submits that
    /// request's claims before any other group, which would orphan them and
    /// discard their finished work.
    pub fn group_holder(&self, bindings: &StateBindings<F>) -> Option<RequestId> {
        let queued = bindings.queue.front().filter(|queued| queued.in_group)?;
        match queued.predicted.as_slice() {
            [operation] => Some(operation.request()),
            _ => None,
        }
    }

    /// What runs behind `operations` once they are submitted or claimed, a
    /// function of the round's operations, the chunks the scheduler planned
    /// for it and the load. A lone prompt chunk of a load that groups opens a
    /// group of the planned chunks continuing it, within the capacity;
    /// otherwise, with pipelining, every slot's successor of one kind queues
    /// behind it.
    pub(super) fn behind(&self, operations: &[Operation]) -> Behind {
        if let ([operation], true) = (operations, self.group_capacity() > 1) {
            let mut chunks = Vec::new();
            let mut previous = operation;
            while chunks.len() + 1 < self.group_capacity() {
                let Some(next) = self.planned_after(previous) else {
                    break;
                };
                chunks.push(next.clone());
                previous = next;
            }
            if !chunks.is_empty() {
                return Behind::Group(chunks);
            }
        }
        if !self.execution.policy().limits().lookahead {
            return Behind::Nothing;
        }
        let successors = operations
            .iter()
            .map(|operation| match continuation_of(operation) {
                Some(next) => Some((next, true)),
                None => self.planned_after(operation).map(|next| (next.clone(), false)),
            })
            .collect::<Option<Vec<_>>>();
        let Some(successors) = successors else {
            return Behind::Nothing;
        };
        // Every slot's successor is of one kind: all decode continuations, or
        // all planned chunks. A planned chunk of a layer-major load opens the
        // next group rather than queueing alone behind this one, where it
        // would read its experts again.
        let selected_tokens = successors.iter().all(|(_, selected)| *selected);
        let planned = successors.iter().all(|(_, selected)| !*selected);
        if !selected_tokens && (!planned || self.group_capacity() > 1) {
            return Behind::Nothing;
        }
        Behind::Successor {
            operations: successors.into_iter().map(|(operation, _)| operation).collect(),
            selected_tokens,
        }
    }

    /// The planned prompt chunk after `operation`, a prefill chunk that
    /// selects nothing.
    fn planned_after(&self, operation: &Operation) -> Option<&Operation> {
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

    /// Build the rest of a layer-major group behind its first launch: each
    /// planned chunk chained on the one before, with its drafter entry
    /// following the one before (the first's is `priming`). Stops at the
    /// first chunk that does not fit; the owner launches it later.
    pub(super) fn build_group(
        &mut self,
        first: &ValidatedTargetLaunch,
        priming: Option<&PrimingReservation>,
        chunks: Vec<Operation>,
    ) -> Result<(Vec<ValidatedTargetLaunch>, Vec<Pending>), DomainError> {
        let mut launches: Vec<ValidatedTargetLaunch> = Vec::new();
        let mut pending: Vec<Pending> = Vec::new();
        for chunk in chunks {
            let operations = vec![chunk];
            let entry = self.queued_priming(&operations, |count| match pending.last() {
                None => priming?.advance.successor(count).ok(),
                Some(previous) => previous.priming.as_ref()?.advance.successor(count).ok(),
            });
            let Some(entry) = entry else {
                break;
            };
            let predecessor = launches.last().unwrap_or(first).advances();
            let Some(launch) = self.build_successor(predecessor, &operations, None, 0)? else {
                break;
            };
            launches.push(launch);
            pending.push(Pending {
                operations,
                selected_tokens: false,
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
        bindings: &mut StateBindings<F>,
        flight: &TargetWork<F>,
        operations: &[Operation],
    ) -> Result<(), DomainError> {
        let behind = self.behind(operations);
        self.planned.clear();
        let Behind::Successor {
            operations: predicted,
            selected_tokens,
        } = behind
        else {
            return Ok(());
        };
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
        let selected = match selected_tokens {
            true => {
                let Some(selected) = flight
                    .submission
                    .output()
                    .readout
                    .as_ref()
                    .and_then(|readout| readout.selected.clone())
                else {
                    return Ok(());
                };
                Some(selected)
            }
            false => None,
        };
        // A planned prompt chunk's drafter entry follows the in-flight entry
        // of the chunk before it.
        let entry = self.queued_priming(&predicted, |count| {
            let priming = flight.priming.as_ref()?;
            let [operation] = predicted.as_slice() else {
                return None;
            };
            (priming.request == operation.request())
                .then(|| priming.submission.launch().advances().first())
                .flatten()?
                .successor(count)
                .ok()
        });
        let Some(entry) = entry else {
            if self.trace_lookahead {
                eprintln!(
                    "lookahead skipped after={}: no drafter entry fits",
                    flight.id
                );
            }
            return Ok(());
        };
        let Some(launch) = self.build_successor(
            flight.submission.launch().advances(),
            &predicted,
            selected,
            flight.id,
        )?
        else {
            return Ok(());
        };
        let pending = Pending {
            launch_trace: self.trace_target_submission(&predicted),
            operations: predicted,
            selected_tokens,
            priming: entry,
        };
        let started = Instant::now();
        let submissions = self
            .family
            .submit_target(vec![launch])
            .map_err(|(error, _)| DomainError::from(error))?;
        self.enqueue(bindings, flight, std::iter::once(pending).zip(submissions), started, false)
    }

    /// The drafter entry of a step queued behind another: none when the
    /// step primes nothing, or its advance (`successor` of the entry before,
    /// given its rows) and its draft launch leases. `None` when the step
    /// primes and its entry does not fit.
    fn queued_priming(
        &self,
        operations: &[Operation],
        successor: impl FnOnce(usize) -> Option<OwnedSuccessorAdvance>,
    ) -> Option<Option<QueuedPriming>> {
        let [Operation::Forward {
            prime: Some(prime), ..
        }] = operations
        else {
            return Some(None);
        };
        let advance = successor(prime.tokens.len())?;
        let head = self
            .resources
            .head_graph()
            .filter(|head| head.available_workspace() > 0 && head.available_output() > 0)?;
        let graph_workspace = head.acquire_workspace().ok()?;
        let graph_output = head.acquire_output().ok()?;
        Some(Some(QueuedPriming {
            advance,
            graph_workspace,
            graph_output,
        }))
    }

    /// Queue submitted steps behind `predecessor`, each behind the one
    /// before, with their drafter entries; `in_group` when they are the rest
    /// of its layer-major group. `started` is when their chain was submitted.
    pub(super) fn enqueue(
        &mut self,
        bindings: &mut StateBindings<F>,
        predecessor: &TargetWork<F>,
        queued: impl IntoIterator<Item = (Pending, F::TargetSubmission)>,
        started: Instant,
        in_group: bool,
    ) -> Result<(), DomainError> {
        let mut previous = (predecessor.id, predecessor.submission.output().boundary.clone());
        for (pending, submission) in queued {
            let Pending {
                operations,
                selected_tokens,
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
            if self.trace_lookahead {
                eprintln!(
                    "lookahead queued flight={id} after={} slots={} rows={}",
                    previous.0,
                    operations.len(),
                    operations.iter().map(Operation::row_count).sum::<usize>()
                );
            }
            let mut flight =
                TargetWork::submitted(id, requests, submission, launch_trace, started, None);
            if let (Some(priming), [Operation::Forward { request, prime: Some(prime), .. }]) =
                (priming, operations.as_slice())
            {
                flight = self.launch_priming(
                    flight,
                    *request,
                    prime,
                    TentativeAdvance::Successor(priming.advance),
                    previous.1.clone(),
                    priming.graph_workspace,
                    priming.graph_output,
                )?;
            }
            let boundary = flight.submission.output().boundary.clone();
            bindings.queue.push_back(QueuedStep {
                flight,
                predecessor: previous.0,
                predicted: operations,
                selected_tokens,
                selected: None,
                in_group,
            });
            previous = (id, boundary);
        }
        Ok(())
    }

    pub(super) fn flight_id(&mut self) -> u64 {
        self.next_flight += 1;
        self.next_flight
    }

    /// The lookahead slot each operation claims, in operation order, when
    /// the group is exactly the queued continuation of accepted states.
    pub(super) fn claim_slots(
        &self,
        bindings: &StateBindings<F>,
        operations: &[Operation],
    ) -> Option<Vec<usize>> {
        let step =bindings.queue.front()?;
        let selected = step.selected.as_ref()?;
        let advances = step.flight.submission.launch().advances();
        let mut slots = Vec::with_capacity(operations.len());
        for operation in operations {
            let slot = step
                .predicted
                .iter()
                .position(|predicted| predicted.request() == operation.request())?;
            let mut expected = step.predicted[slot].clone();
            if step.selected_tokens {
                let choice = selected.get(slot).filter(|choice| choice.status == 0)?;
                // A queued step that read a table row the device did not
                // hold for this token is not the continuation. (A slot whose
                // stored rows could not be read is claimed all the same: its
                // sequence fails when the step finishes, and the others
                // continue, `finish_target_work`.)
                if !step.flight.submission.output().found_selected(choice.token.0) {
                    return None;
                }
                if let Operation::Forward { tokens, .. } = &mut expected {
                    tokens[0] = choice.token;
                }
            }
            let state = self.target.get(&operation.request())?;
            let advance = advances.get(slot)?;
            let drafter_follows = match operation {
                Operation::Forward {
                    prime: Some(prime), ..
                } => self
                    .head
                    .get(&operation.request())
                    .is_some_and(|head| head.position() == prime.position),
                _ => true,
            };
            if slots.contains(&slot)
                || !drafter_follows
                || operation != &expected
                || state.position() != advance.position()
                || state.bank_index() != advance.bindings().previous_bank
                || state.tape_rows() != advance.bindings().previous_tape
                || state.domain_ranges() != advance.domain_ranges()
            {
                return None;
            }
            slots.push(slot);
        }
        Some(slots)
    }

    /// The launch class a claimable group occupies.
    pub(super) fn claim_class(
        &self,
        bindings: &StateBindings<F>,
        operations: &[Operation],
    ) -> Option<crate::LaunchClass> {
        self.claim_slots(bindings, operations)?;
        let step =bindings.queue.front()?;
        Some(step.flight.submission.launch().batch().class())
    }

    /// Hand the queued step to the owner as the flight of `operations`. Each
    /// claimed slot takes its request's accepted state, which its successor
    /// attaches to at finish; unclaimed slots are discarded then.
    pub(super) fn claim_queued(
        &mut self,
        bindings: &mut StateBindings<F>,
        operations: &[Operation],
        slots: Vec<usize>,
    ) -> Result<TargetWork<F>, DomainError> {
        if self.claim_slots(bindings, operations).as_ref() != Some(&slots) {
            return Err(DomainError::invariant(
                "claimed lookahead changed after reservation",
            ));
        }
        let Some(QueuedStep { mut flight, .. }) = bindings.queue.pop_front() else {
            return Err(DomainError::invariant("claimed lookahead is absent"));
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
        if self.trace_lookahead {
            eprintln!(
                "lookahead claimed flight={} slots={}",
                flight.id,
                operations.len()
            );
        }
        Ok(flight)
    }

    /// Wait for the queued steps nobody will claim and drop them, in queue
    /// order: their successors release their rows and banks, their leases
    /// return to their pools. The first failure is reported once every step
    /// was waited for.
    pub(super) fn orphan_queue(
        &self,
        bindings: &mut StateBindings<F>,
    ) -> Result<(), DomainError> {
        let mut outcome = Ok(());
        while let Some(step) = bindings.queue.pop_front() {
            let orphaned = self.orphan_one(step);
            if outcome.is_ok() {
                outcome = orphaned;
            }
        }
        outcome
    }

    fn orphan_one(&self, step: QueuedStep<F>) -> Result<(), DomainError> {
        if self.trace_lookahead {
            eprintln!(
                "lookahead orphaned flight={} predecessor_finished={}",
                step.flight.id,
                step.selected.is_some()
            );
        }
        // Its drafter entry runs behind it on the device and finishes too,
        // so no lease returns while the device still uses it.
        let priming = step
            .flight
            .priming
            .map(|priming| priming.submission.finish().map(drop));
        match (step.flight.submission.finish().map(drop), priming) {
            (Ok(()), None | Some(Ok(()))) => Ok(()),
            (Err(error), _) | (_, Some(Err(error))) => Err(DomainError::Device(error)),
        }
    }

    /// The launch of `predicted` queued behind a step whose advances are
    /// `predecessor` (in flight or about to be): each slot's successor
    /// advance follows its predecessor's, and a decode continuation reads its
    /// tokens from `selected`, the predecessor's selection rows. `None` when
    /// the successors or their leases do not fit without growing anything,
    /// or a slot needs input slices: the step is then submitted as usual.
    fn build_successor(
        &mut self,
        predecessor: &[TentativeAdvance],
        predicted: &[Operation],
        selected: Option<GraphOutputTensor>,
        after: u64,
    ) -> Result<Option<ValidatedTargetLaunch>, DomainError> {
        let mut advances = Vec::with_capacity(predicted.len());
        for (advance, operation) in predecessor.iter().zip(predicted) {
            match advance.successor(operation.row_count()) {
                Ok(successor) => advances.push(TentativeAdvance::Successor(successor)),
                // No room without growing the backing (or a limit): the next
                // step runs unpipelined and provisions as usual.
                Err(error) => {
                    if self.trace_lookahead {
                        eprintln!("lookahead skipped after={after}: {error}");
                    }
                    return Ok(None);
                }
            }
        }
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
        let Ok(preceding) = self.target_preceding(predicted) else {
            return Ok(None);
        };
        let class_limits = self.target_class_limits();
        let vocabulary = self.definition.decoder.vocabulary as usize;
        let batch = if selected.is_some() {
            ValidatedTargetBatch::from_slots(&slots, vocabulary, class_limits)
        } else {
            ValidatedTargetBatch::covering(&slots, vocabulary, class_limits, segments)
        }
        .map_err(|error| DomainError::Input(error.to_string()))?;
        let tokens = match selected {
            Some(selected) => {
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
            None => TargetTokens::Host,
        };
        let graph = self.resources.target_graph();
        let readout = self.resources.target_readout_graph();
        if graph.available_workspace() == 0
            || graph.available_output() < 2
            || readout.available_workspace() == 0
            || readout.available_output() == 0
        {
            if self.trace_lookahead {
                eprintln!("lookahead skipped after={after}: no free launch leases");
            }
            return Ok(None);
        }
        let invariant = |error: CapacityError| {
            DomainError::invariant(format!("lookahead lease changed after its check: {error}"))
        };
        let graph_workspace = graph.acquire_workspace().map_err(invariant)?;
        let graph_outputs = [
            graph.acquire_output().map_err(invariant)?,
            graph.acquire_output().map_err(invariant)?,
        ];
        let readout_workspace = readout.acquire_workspace().map_err(invariant)?;
        let readout_output = readout.acquire_output().map_err(invariant)?;
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
        let inputs = TargetLaunchInputs::new(
            batch,
            tokens,
            advances,
            vec![None; count],
            vec![Vec::new(); count],
            graph_workspace,
            graph_outputs,
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

impl<F: ProgramFamily> StateBindings<F> {
    /// Record the selections of a finished flight for the lookahead it feeds.
    /// The predecessor finished at `completed`: the queued step's selections
    /// are known, and the device runs it from then on.
    pub(super) fn predecessor_selected(
        &mut self,
        flight: u64,
        selected: &[Selected],
        completed: Instant,
    ) {
        if let Some(step) = self
            .queue
            .iter_mut()
            .find(|step| step.predecessor == flight)
        {
            step.selected = Some(selected.to_vec());
            step.flight.runnable = completed;
        }
    }
}
