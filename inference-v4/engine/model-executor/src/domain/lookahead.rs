//! Cross-step pipelining ("lookahead"). Right after a continuable target
//! step N is submitted, the domain queues step N+1 behind it on the device:
//! each slot's successor advance follows N's in-flight advance, its input
//! token is N's selection read on the device, and its selection is N's with
//! the position advanced by one. The device then runs N+1 without waiting for
//! the host to observe N and submit the next step.
//!
//! The owner never sees the lookahead. When the operations it next submits
//! equal the prediction (token included, known once N finished) and the
//! accepted states are the ones the successors follow, the group claims the
//! queued step: nothing is reserved or launched, and its flight is returned.
//! Anything else orphans it: the domain waits for it and drops it, so its
//! rows, banks and leases return before the other group reserves. Every
//! device write of a lookahead goes to rows and banks its own successors
//! own, so an orphan never changes accepted state.

use super::*;
use crate::programs::SubmittedTarget;
use magnitude_model_state::TentativeAdvance;

pub(super) struct Lookahead<S: ProgramSubmission<CompletedWork = crate::CompletedTargetWork>> {
    flight: TargetFlight<S>,
    /// The flight whose selections are this one's tokens.
    predecessor: u64,
    /// Per slot, the operation that claims it, with a placeholder token.
    predicted: Vec<Operation>,
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
    })
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    pub(super) fn flight_id(&mut self) -> u64 {
        self.next_flight += 1;
        self.next_flight
    }

    /// The lookahead slot each operation claims, in operation order, when
    /// the group is exactly the queued continuation of accepted states.
    pub(super) fn claim_slots(&self, operations: &[Operation]) -> Option<Vec<usize>> {
        let lookahead = self.lookahead.as_ref()?;
        let selected = lookahead.selected.as_ref()?;
        let advances = lookahead.flight.submission.launch().advances();
        let mut slots = Vec::with_capacity(operations.len());
        for operation in operations {
            let slot = lookahead
                .predicted
                .iter()
                .position(|predicted| predicted.request() == operation.request())?;
            let choice = selected.get(slot).filter(|choice| choice.status == 0)?;
            let mut expected = lookahead.predicted[slot].clone();
            if let Operation::Forward { tokens, .. } = &mut expected {
                tokens[0] = choice.token;
            }
            let state = self.target.get(&operation.request())?;
            let advance = advances.get(slot)?;
            if slots.contains(&slot)
                || operation != &expected
                || state.position() != advance.position()
                || state.bank_index() != advance.bindings().previous_bank
                || state.tape_rows() != advance.bindings().previous_tape
                || state.history_ranges() != advance.history_ranges()
            {
                return None;
            }
            slots.push(slot);
        }
        Some(slots)
    }

    /// The launch class a claimable group occupies.
    pub(super) fn claim_class(&self, operations: &[Operation]) -> Option<crate::LaunchClass> {
        self.claim_slots(operations)?;
        let lookahead = self.lookahead.as_ref()?;
        Some(lookahead.flight.submission.launch().batch().class())
    }

    /// Hand the queued step to the owner as the flight of `operations`. Each
    /// claimed slot takes its request's accepted state, which its successor
    /// attaches to at finish; unclaimed slots are discarded then.
    pub(super) fn claim_lookahead(
        &mut self,
        operations: &[Operation],
        slots: Vec<usize>,
    ) -> Result<TargetFlight<F::TargetSubmission>, DomainError> {
        self.healthy()?;
        if self.claim_slots(operations).as_ref() != Some(&slots) {
            return Err(self.fatal_invariant("claimed lookahead changed after reservation"));
        }
        let Some(Lookahead { mut flight, .. }) = self.lookahead.take() else {
            return Err(self.fatal_invariant("claimed lookahead is absent"));
        };
        let mut sources = (0..flight.requests.len()).map(|_| None).collect::<Vec<_>>();
        // `claim_slots` found every claimed request's accepted state.
        for (operation, slot) in operations.iter().zip(slots) {
            sources[slot] = self.target.remove(&operation.request());
        }
        flight.continuation = Some(sources);
        if self.trace_lookahead {
            eprintln!("lookahead claimed flight={} slots={}", flight.id, operations.len());
        }
        Ok(flight)
    }

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

    /// Wait for a queued step nobody will claim and drop it: its successors
    /// release their rows and banks, its leases return to their pools.
    pub(super) fn orphan_lookahead(&mut self) -> Result<(), DomainError> {
        let Some(lookahead) = self.lookahead.take() else {
            return Ok(());
        };
        if self.trace_lookahead {
            eprintln!(
                "lookahead orphaned flight={} predecessor_finished={}",
                lookahead.flight.id,
                lookahead.selected.is_some()
            );
        }
        match lookahead.flight.submission.finish() {
            Ok(completed) => {
                drop(completed);
                Ok(())
            }
            Err(error) => {
                let failure = DomainError::Device(error);
                self.fatal = Some(failure.clone());
                Err(failure)
            }
        }
    }

    /// Queue the continuation of `flight` (just submitted or claimed for
    /// `operations`) when every slot continues by one decode row, the flight
    /// has a selection tensor to read tokens from, and the successors and
    /// leases fit without growing anything. Otherwise the next step is
    /// submitted by the owner as usual.
    pub(super) fn queue_lookahead(
        &mut self,
        flight: &TargetFlight<F::TargetSubmission>,
        operations: &[Operation],
    ) -> Result<(), DomainError> {
        if !self.execution.policy().limits().lookahead {
            return Ok(());
        }
        if self.lookahead.is_some() {
            return Err(self.fatal_invariant("a lookahead was queued over another"));
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
        let Some(predicted) = operations
            .iter()
            .map(continuation_of)
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(());
        };
        let Some(selected) = flight
            .submission
            .output()
            .readout
            .as_ref()
            .and_then(|readout| readout.selected.clone())
        else {
            return Ok(());
        };
        let mut advances = Vec::with_capacity(operations.len());
        for advance in flight.submission.launch().advances() {
            match advance.successor(1) {
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
        for (operation, advance) in predicted.iter().zip(&advances) {
            let (slot, slices) = self.target_slot(operation, advance).map_err(DomainError::Input)?;
            if !slices.is_empty() {
                return Ok(());
            }
            slots.push(slot);
        }
        let limits = self.execution.policy().limits();
        let batch = ValidatedTargetBatch::from_slots(
            &slots,
            self.definition.geometry.vocabulary as usize,
            limits.max_batch_rows,
        )
        .map_err(|error| DomainError::Input(error.to_string()))?;
        let rows = batch.class().rows() as u64;
        if selected.tensor().extents().first().is_none_or(|extent| *extent < rows) {
            return Ok(());
        }
        let tokens = selected
            .slice_leading(0, rows)
            .map_err(|error| DomainError::Input(error.to_string()))?;
        let graph = self.resources.target_graph();
        let readout = self.resources.target_readout_graph();
        if graph.available_workspace() == 0
            || graph.available_output() < 2
            || readout.available_workspace() == 0
            || readout.available_output() == 0
        {
            if self.trace_lookahead {
                eprintln!("lookahead skipped after={}: no free launch leases", flight.id);
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
            TargetTokens::Selected(tokens),
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
            self.definition.geometry.hidden as usize,
        ) {
            Ok(launch) => launch,
            Err((_, error)) => {
                let failure = DomainError::Invariant(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        let started = Instant::now();
        let submission = match self.family.submit_target(launch) {
            Ok(submission) => submission,
            Err((error, _)) => {
                let failure = DomainError::from(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        let requests = predicted
            .iter()
            .map(|operation| (operation.request(), 1, None, WorkKind::Decode, 1))
            .collect();
        let id = self.flight_id();
        if self.trace_lookahead {
            eprintln!("lookahead queued flight={id} after={} slots={count}", flight.id);
        }
        self.lookahead = Some(Lookahead {
            flight: TargetFlight {
                requests,
                submission,
                started,
                runnable: started,
                previous_selection: None,
                id,
                continuation: None,
            },
            predecessor: flight.id,
            predicted,
            selected: None,
        });
        Ok(())
    }
}
