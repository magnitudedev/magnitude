//! Cross-step pipelining ("lookahead"). Right after a continuable target
//! step N is submitted, the domain queues step N+1 behind it on the device:
//! each slot's successor advance follows N's in-flight advance. A decode
//! continuation's input token is N's selection read on the device, and its
//! selection is N's with the position advanced by one. A prompt's next
//! prefill chunk is known before N finishes: the owner plans it
//! (`plan_successors`) and it is queued with its own tokens. The device then
//! runs N+1 without waiting for the host to observe N and submit the next
//! step.
//!
//! The lookahead lives in the domain's `StateBindings`: it is the only work
//! that holds bindings between flights. The owner carries it there but never
//! sees it. When the operations it next submits equal the prediction (token
//! included, known once N finished) and the accepted states are the ones the
//! successors follow, the group claims the queued step: nothing is reserved
//! or launched, and its flight is returned. Any other target group, and any
//! binding change, orphans it: the domain waits for it and drops it, so its
//! rows, banks and leases return first. Every device write of a lookahead
//! goes to rows and banks its own successors own, so an orphan never changes
//! accepted state.

use super::*;
use crate::programs::{SubmittedHead, SubmittedTarget};
use magnitude_state::TentativeAdvance;

pub(super) struct Lookahead<F: ProgramFamily> {
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
    /// them and its lookahead queues them; any other group discards them.
    pub fn plan_successors(&mut self, planned: Vec<Operation>) {
        self.planned = planned;
    }

    /// The step to queue behind `operation`, and whether its tokens are
    /// `operation`'s selections: the decode continuation, or the planned
    /// prompt chunk after a prefill chunk that selects nothing.
    pub(super) fn successor_of(&self, operation: &Operation) -> Option<(Operation, bool)> {
        if let Some(next) = continuation_of(operation) {
            return Some((next, true));
        }
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
        self.planned
            .iter()
            .find(|planned| {
                matches!(planned, Operation::Forward {
                    request: next,
                    kind: WorkKind::Prefill,
                    position: start,
                    conditioning: None,
                    ..
                } if next == request && *start == end)
            })
            .map(|planned| (planned.clone(), false))
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
        let lookahead = bindings.lookahead.as_ref()?;
        let selected = lookahead.selected.as_ref()?;
        let advances = lookahead.flight.submission.launch().advances();
        let mut slots = Vec::with_capacity(operations.len());
        for operation in operations {
            let slot = lookahead
                .predicted
                .iter()
                .position(|predicted| predicted.request() == operation.request())?;
            let mut expected = lookahead.predicted[slot].clone();
            if lookahead.selected_tokens {
                let choice = selected.get(slot).filter(|choice| choice.status == 0)?;
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
        let lookahead = bindings.lookahead.as_ref()?;
        Some(lookahead.flight.submission.launch().batch().class())
    }

    /// Hand the queued step to the owner as the flight of `operations`. Each
    /// claimed slot takes its request's accepted state, which its successor
    /// attaches to at finish; unclaimed slots are discarded then.
    pub(super) fn claim_lookahead(
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
        let Some(Lookahead { mut flight, .. }) = bindings.lookahead.take() else {
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

    /// Wait for a queued step nobody will claim and drop it: its successors
    /// release their rows and banks, its leases return to their pools.
    pub(super) fn orphan_lookahead(
        &self,
        bindings: &mut StateBindings<F>,
    ) -> Result<(), DomainError> {
        let Some(lookahead) = bindings.lookahead.take() else {
            return Ok(());
        };
        if self.trace_lookahead {
            eprintln!(
                "lookahead orphaned flight={} predecessor_finished={}",
                lookahead.flight.id,
                lookahead.selected.is_some()
            );
        }
        // Its drafter entry runs behind it on the device and finishes too,
        // so no lease returns while the device still uses it.
        let priming = lookahead
            .flight
            .priming
            .map(|priming| priming.submission.finish().map(drop));
        match (lookahead.flight.submission.finish().map(drop), priming) {
            (Ok(()), None | Some(Ok(()))) => Ok(()),
            (Err(error), _) | (_, Some(Err(error))) => Err(DomainError::Device(error)),
        }
    }

    /// Queue the successor of `flight` (just submitted or claimed for
    /// `operations`) when every slot has one of the same kind (see
    /// `successor_of`), a decode continuation's flight has a selection tensor
    /// to read tokens from, and the successors and leases fit without growing
    /// anything. Otherwise the next step is submitted by the owner as usual.
    pub(super) fn queue_lookahead(
        &mut self,
        bindings: &mut StateBindings<F>,
        flight: &TargetWork<F>,
        operations: &[Operation],
    ) -> Result<(), DomainError> {
        let successors = operations
            .iter()
            .map(|operation| self.successor_of(operation))
            .collect::<Option<Vec<_>>>();
        self.planned.clear();
        if !self.execution.policy().limits().lookahead {
            return Ok(());
        }
        if bindings.lookahead.is_some() {
            return Err(DomainError::invariant("a lookahead was queued over another"));
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
        let Some(successors) = successors else {
            return Ok(());
        };
        let selected_tokens = successors.iter().all(|(_, selected)| *selected);
        if !selected_tokens && successors.iter().any(|(_, selected)| *selected) {
            return Ok(());
        }
        let predicted = successors
            .into_iter()
            .map(|(operation, _)| operation)
            .collect::<Vec<_>>();
        let selected = if selected_tokens {
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
        } else {
            None
        };
        let mut advances = Vec::with_capacity(operations.len());
        for (advance, operation) in flight.submission.launch().advances().iter().zip(&predicted) {
            match advance.successor(operation.row_count()) {
                Ok(successor) => advances.push(TentativeAdvance::Successor(successor)),
                // No room without growing the backing (or a limit): the next
                // step runs unpipelined and provisions as usual.
                Err(error) => {
                    if self.trace_lookahead {
                        eprintln!("lookahead skipped after={}: {error}", flight.id);
                    }
                    return Ok(());
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
                return Ok(());
            }
            segments = segments.max(super::target::reserved_segments(
                &self.target_store,
                advance.span_count(),
                operation.row_count(),
            ));
            slots.push(slot);
        }
        let class_limits = self.target_class_limits();
        let vocabulary = self.definition.decoder.vocabulary as usize;
        let batch = if selected_tokens {
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
                    return Ok(());
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
                eprintln!(
                    "lookahead skipped after={}: no free launch leases",
                    flight.id
                );
            }
            return Ok(());
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
        let count = operations.len();
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
        );
        let launch = match ValidatedTargetLaunch::new(
            inputs,
            &self.target_store,
            self.domain.id(),
            self.definition.decoder.hidden as usize,
        ) {
            Ok(launch) => launch,
            Err((_, error)) => return Err(DomainError::Invariant(error)),
        };
        // A planned prompt chunk's drafter entry follows the in-flight entry
        // of the chunk before it, with its own draft launch leases.
        let priming = match &predicted[..] {
            [Operation::Forward {
                request,
                prime: Some(prime),
                ..
            }] => {
                let Some(successor) = flight
                    .priming
                    .as_ref()
                    .filter(|priming| priming.request == *request)
                    .and_then(|priming| priming.submission.launch().advances().first())
                    .and_then(|advance| advance.successor(prime.tokens.len()).ok())
                else {
                    if self.trace_lookahead {
                        eprintln!(
                            "lookahead skipped after={}: drafter entry has no room",
                            flight.id
                        );
                    }
                    return Ok(());
                };
                let Some(head) = self
                    .resources
                    .head_graph()
                    .filter(|head| head.available_workspace() > 0 && head.available_output() > 0)
                else {
                    if self.trace_lookahead {
                        eprintln!(
                            "lookahead skipped after={}: no free drafter launch leases",
                            flight.id
                        );
                    }
                    return Ok(());
                };
                Some((
                    *request,
                    prime.clone(),
                    successor,
                    head.acquire_workspace().map_err(invariant)?,
                    head.acquire_output().map_err(invariant)?,
                ))
            }
            _ => None,
        };
        let started = Instant::now();
        let submission = self
            .family
            .submit_target(launch)
            .map_err(|(error, _)| DomainError::from(error))?;
        let requests = predicted
            .iter()
            .map(|operation| {
                let Operation::Forward {
                    request,
                    kind,
                    committed,
                    ..
                } = operation
                else {
                    unreachable!("a successor is a forward operation")
                };
                (*request, operation.row_count(), None, *kind, *committed)
            })
            .collect();
        let id = self.flight_id();
        if self.trace_lookahead {
            eprintln!(
                "lookahead queued flight={id} after={} slots={count} rows={}",
                flight.id,
                predicted.iter().map(Operation::row_count).sum::<usize>()
            );
        }
        let mut queued = TargetWork {
            launch_trace: None,
            requests,
            submission,
            priming: None,
            started,
            runnable: started,
            previous_selection: None,
            id,
            continuation: None,
        };
        if let Some((request, prime, successor, workspace, output)) = priming {
            queued = self.launch_priming(
                queued,
                request,
                &prime,
                TentativeAdvance::Successor(successor),
                workspace,
                output,
            )?;
        }
        bindings.lookahead = Some(Lookahead {
            flight: queued,
            predecessor: flight.id,
            predicted,
            selected_tokens,
            selected: None,
        });
        Ok(())
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
        if let Some(lookahead) = self
            .lookahead
            .as_mut()
            .filter(|lookahead| lookahead.predecessor == flight)
        {
            lookahead.selected = Some(selected.to_vec());
            lookahead.flight.runnable = completed;
        }
    }
}
