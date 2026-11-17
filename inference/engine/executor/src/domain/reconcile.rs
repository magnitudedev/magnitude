//! reconcile lifecycle for the executor domain.

use super::*;

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// The decision is relative to this request's submitted rows. Every
    /// accepted prefix within the row commitment publishes at once: recurrent
    /// state commits as a tape version of the advance's successor bank.
    pub fn reconcile(
        &mut self,
        mut pending: PendingOperationOutcome,
        decision: PhysicalDecision,
    ) -> Result<(), DomainError> {
        // A poisoned sequence's outputs are discarded like an orphan's: its
        // accepted state returns, and its sequence fails.
        if let Outcome::WeightReadFailed(error) = &pending.outcome {
            let error = error.clone();
            self.abort(pending)?;
            return Err(DomainError::WeightRead(error));
        }
        // An interrupted outcome is aborted and re-run, never reconciled.
        assert!(
            !matches!(pending.outcome, Outcome::Interrupted { .. }),
            "invariant violated: an interrupted outcome is reconciled"
        );
        if matches!(pending.outcome, Outcome::Head { .. }) {
            // A head commits exactly its entry rows; its chained proposal
            // rows never become visible.
            if decision.accepted_rows != pending.committed_rows {
                let _ = self.abort(pending);
                return Err("head decision differs from its committed entry rows".into());
            }
            // A head that keeps no history committed nothing: the target
            // state it read returned at physical completion.
            if pending.advance.is_none() && self.head_reads_target() {
                return Ok(());
            }
            // An outcome owns exactly its own rows: a drafter's, its step's.
            let Some(Advance::Step(advance)) = pending.advance.take() else {
                panic!("invariant violated: a drafter outcome owns no step of its own");
            };
            let (OwnedAdvanceResolution::Aborted(state) | OwnedAdvanceResolution::Committed(state)) =
                advance
                    .commit(decision.accepted_rows)
                    .map_err(|(state, error)| {
                        self.head.insert(pending.request, state);
                        DomainError::from(error)
                    })?;
            self.head.insert(pending.request, state);
            return Ok(());
        }
        if matches!(pending.outcome, Outcome::Encode { .. }) {
            if decision.accepted_rows != 0 {
                return Err("stateless result has a physical prefix decision".into());
            }
            if let (Outcome::Encode { features }, Some(image)) = (&pending.outcome, &pending.image)
            {
                let slot = self
                    .input
                    .get_mut(&pending.request)
                    .expect("invariant violated: a vision outcome's request has no admitted input")
                    .images
                    .values_mut()
                    .find(|slot| &slot.image == image)
                    .expect("invariant violated: an encoded image is absent from its admitted input");
                assert!(slot.features.is_none(), "invariant violated: an encoded image's features are already installed");
                slot.features = Some(features.clone());
            }
            return Ok(());
        }
        if decision.accepted_rows < pending.committed_rows
            || decision.accepted_rows > pending.rows()
        {
            let _ = self.abort(pending);
            return Err("accepted target prefix is outside the submitted row commitment".into());
        }
        let request = pending.request;
        // Its state is the outcome's alone until it reconciles, a chain it
        // carries with it: no other owner can hold it.
        assert!(
            !self.target.contains_key(&request) && !(pending.primed.is_some() && self.head.contains_key(&request)),
            "invariant violated: a reconciled request has another physical state owner"
        );
        let advance = pending.advance.take().expect("invariant violated: a target outcome owns no target advance");
        // A prompt chunk commits whole, and its drafter entry with it.
        if let Some(primed) = pending.primed.take() {
            let rows = primed.rows();
            match primed.commit(rows) {
                Ok(
                    OwnedAdvanceResolution::Aborted(state)
                    | OwnedAdvanceResolution::Committed(state),
                ) => {
                    self.head.insert(request, state);
                }
                Err((state, error)) => {
                    self.head.insert(request, state);
                    self.abort_target(request, advance)?;
                    return Err(error.into());
                }
            }
        }
        match advance {
            Advance::Step(advance) => match advance.commit(decision.accepted_rows) {
                Ok(
                    OwnedAdvanceResolution::Aborted(state)
                    | OwnedAdvanceResolution::Committed(state),
                ) => {
                    self.target.insert(request, state);
                }
                Err((state, error)) => {
                    self.target.insert(request, state);
                    return Err(error.into());
                }
            },
            // A chain holds its chunks' rows until it concludes.
            Advance::Opens { flight, first } => self.open_chain(request, flight, first)?,
            Advance::Chained { flight, joined, chunk } => self.join_chain(request, flight, joined, chunk)?,
        }
        let boundary = match &pending.outcome {
            Outcome::Forward { rows } => rows
                .last()
                .and_then(|row| row.features.as_ref())
                .and_then(FeatureRef::boundary)
                .cloned(),
            _ => None,
        };
        self.input
            .get_mut(&request)
            .expect("resident request has input")
            .boundary = boundary;
        Ok(())
    }

    /// Cancellation while completion is already available makes no physical
    /// successor visible and returns the original accepted sequence.
    pub fn abort(&mut self, pending: PendingOperationOutcome) -> Result<(), DomainError> {
        let conflicting_owner = match &pending.outcome {
            Outcome::Head { .. } => self.head.contains_key(&pending.request),
            Outcome::Forward { .. } | Outcome::WeightReadFailed(_) | Outcome::Interrupted { .. } => {
                self.target.contains_key(&pending.request)
                    || (pending.primed.is_some() && self.head.contains_key(&pending.request))
            }
            Outcome::Encode { .. } => false,
        };
        // Its state is the outcome's alone until it aborts, a chain it
        // carries with it: no other owner can hold it.
        assert!(!conflicting_owner, "invariant violated: an aborted request has another physical state owner");
        if let Some(primed) = pending.primed {
            self.head.insert(pending.request, primed.abort());
        }
        let Some(advance) = pending.advance else {
            return Ok(());
        };
        match (&pending.outcome, advance) {
            (Outcome::Head { .. }, Advance::Step(advance)) => {
                self.head.insert(pending.request, advance.abort());
                Ok(())
            }
            (Outcome::Forward { .. } | Outcome::WeightReadFailed(_) | Outcome::Interrupted { .. }, advance) => {
                self.abort_target(pending.request, advance)
            }
            // An outcome owns exactly its own rows: a drafter's never a
            // chain's, an encode's none.
            (Outcome::Head { .. }, Advance::Opens { .. } | Advance::Chained { .. }) => {
                panic!("invariant violated: a drafter outcome owns a chain's rows")
            }
            (Outcome::Encode { .. }, _) => panic!("invariant violated: a stateless outcome owns sequence state"),
        }
    }

    /// Return the accepted target state `request`'s aborted `advance` came
    /// from.
    fn abort_target(&mut self, request: RequestId, advance: Advance) -> Result<(), DomainError> {
        match advance {
            Advance::Step(advance) => {
                self.target.insert(request, advance.abort());
                Ok(())
            }
            // A chain's first chunk: the chunks queued behind it follow its
            // rows, so they are concluded before its state returns.
            Advance::Opens { flight, first } => {
                let concluded = match self.queue.opens_chain(flight) {
                    true => self.conclude_chain(),
                    false => Ok(()),
                };
                self.target.insert(request, first.abort());
                concluded
            }
            // A chunk of a chain: the chain cannot run past it, so it
            // concludes with the chunks reconciled before, its holder's
            // state the one they reached, and the chunk's rows are dropped.
            Advance::Chained { flight, joined, chunk } => {
                drop(chunk);
                match self.return_chain(request, flight, joined) {
                    true => self.conclude_chain(),
                    false => Ok(()),
                }
            }
        }
    }
}
