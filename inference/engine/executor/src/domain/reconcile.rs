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
                self.abort(pending)?;
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
            self.abort(pending)?;
            return Err("accepted target prefix is outside the submitted row commitment".into());
        }
        let request = pending.request;
        // Its state is the outcome's alone until it reconciles, a chain it
        // carries with it: no other owner can hold it.
        assert!(
            !self.target.contains_key(&request)
                && !((pending.primed.is_some() || matches!(pending.front, Some(FrontAdvance::Step(_))))
                    && self.head.contains_key(&request)),
            "invariant violated: a reconciled request has another physical state owner"
        );
        let advance = pending.advance.take().expect("invariant violated: a target outcome owns no target advance");
        // How the guesses of a pass that ran rows ahead in a bank's step
        // fared, which only a bank reading experts ahead follows, told
        // before its state is published.
        if let (Some(bank), Some(step)) = (self.family.expert_bank(), pending.awaited) {
            bank.decided(step, request, &decision.ahead);
        }
        // The target commits the decision's rows first: the only commit a
        // decision decides. A refusal leaves every head advance uncommitted,
        // returned with it. A chain's chunk is accepted whole; a failure
        // joining or concluding its chain is reported once the head advances
        // committed, as they commit whatever its chain does.
        let published = match advance {
            Advance::Step(advance) => match advance.commit(decision.accepted_rows) {
                Ok(
                    OwnedAdvanceResolution::Aborted(state)
                    | OwnedAdvanceResolution::Committed(state),
                ) => {
                    self.target.insert(request, state);
                    Ok(())
                }
                Err((state, error)) => {
                    self.target.insert(request, state);
                    self.return_head_advances(request, pending.primed.take(), pending.front.take());
                    return Err(error.into());
                }
            },
            // A chain holds its chunks' rows until it concludes.
            Advance::Opens { flight, first } => self.open_chain(request, flight, first),
            Advance::Chained { flight, joined, chunk } => self.join_chain(request, flight, joined, chunk),
        };
        // Then the head advances commit what they always commit, whatever
        // the decision: a prompt chunk's drafter entry its every row, and the
        // front head launch behind a chained forward its one entry row (the
        // closing chunk's following the drafter entry just committed). The
        // target accepted at least its committed rows, so only a broken
        // invariant refuses either.
        let broken = |error: magnitude_state::Error| -> ! {
            panic!("invariant violated: a head advance refused the commit its target's commit implies: {error}")
        };
        if let Some(primed) = pending.primed.take() {
            let rows = primed.rows();
            match primed.commit(rows) {
                Ok(OwnedAdvanceResolution::Aborted(state) | OwnedAdvanceResolution::Committed(state)) => {
                    self.head.insert(request, state);
                }
                Err((_, error)) => broken(error),
            }
        }
        if let Some(front) = pending.front.take() {
            let front = match front {
                FrontAdvance::Step(front) => front,
                FrontAdvance::Entry(front) => {
                    let state = self
                        .head
                        .remove(&request)
                        .expect("invariant violated: a closing chunk's front head advance follows no drafter entry");
                    front.attach(state).unwrap_or_else(|(_, error)| broken(error))
                }
            };
            match front.commit(1) {
                Ok(OwnedAdvanceResolution::Aborted(state) | OwnedAdvanceResolution::Committed(state)) => {
                    self.head.insert(request, state);
                }
                Err((_, error)) => broken(error),
            }
        }
        published?;
        let boundary = match &pending.outcome {
            Outcome::Forward { rows, .. } => rows
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
    /// successor visible and returns the original accepted sequence. A pass
    /// that ran rows ahead in a bank's step is discarded
    /// (`ElasticBank::discarded`).
    pub fn abort(&mut self, pending: PendingOperationOutcome) -> Result<(), DomainError> {
        let conflicting_owner = match &pending.outcome {
            Outcome::Head { .. } => self.head.contains_key(&pending.request),
            Outcome::Forward { .. } | Outcome::WeightReadFailed(_) | Outcome::Interrupted { .. } => {
                self.target.contains_key(&pending.request)
                    || ((pending.primed.is_some() || matches!(pending.front, Some(FrontAdvance::Step(_))))
                        && self.head.contains_key(&pending.request))
            }
            Outcome::Encode { .. } => false,
        };
        // Its state is the outcome's alone until it aborts, a chain it
        // carries with it: no other owner can hold it.
        assert!(!conflicting_owner, "invariant violated: an aborted request has another physical state owner");
        // No verdict will judge a pass that ran rows ahead in a bank's step.
        if let (Some(bank), Some(step)) = (self.family.expert_bank(), pending.awaited) {
            bank.discarded(step, pending.request);
        }
        self.return_head_advances(pending.request, pending.primed, pending.front);
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

    /// Return `request`'s uncommitted head advances: its drafter entry
    /// returns the head state it holds, and so does a front head advance
    /// over it; the closing chunk's follows the drafter entry just returned,
    /// and drops.
    fn return_head_advances(&mut self, request: RequestId, primed: Option<OwnedStateAdvance>, front: Option<FrontAdvance>) {
        if let Some(primed) = primed {
            self.head.insert(request, primed.abort());
        }
        match front {
            Some(FrontAdvance::Step(front)) => {
                self.head.insert(request, front.abort());
            }
            Some(FrontAdvance::Entry(front)) => drop(front),
            None => {}
        }
    }
}
