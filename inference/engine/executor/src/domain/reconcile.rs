//! reconcile lifecycle for the executor domain.

use super::*;

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// The decision is relative to this request's submitted rows. Every
    /// accepted prefix within the row commitment publishes at once: recurrent
    /// state commits as a tape version of the advance's successor bank.
    pub fn reconcile(
        &mut self,
        pending: PendingOperationOutcome,
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
        // Validate decision before consuming `pending` so a rejected decision
        // aborts through the single `abort` path.
        match &pending.payload {
            PendingPayload::Head { .. } => {
                // A head commits exactly its entry rows; its chained proposal
                // rows never become visible.
                if decision.accepted_rows != pending.committed_rows {
                    self.abort(pending)?;
                    return Err("head decision differs from its committed entry rows".into());
                }
            }
            PendingPayload::Encode { .. } => {
                if decision.accepted_rows != 0 {
                    return Err("stateless result has a physical prefix decision".into());
                }
            }
            PendingPayload::Target { .. } => {
                if decision.accepted_rows < pending.committed_rows
                    || decision.accepted_rows > pending.rows()
                {
                    self.abort(pending)?;
                    return Err(
                        "accepted target prefix is outside the submitted row commitment".into(),
                    );
                }
            }
        }
        let request = pending.request;
        match pending.payload {
            PendingPayload::Head { advance } => {
                // A head that keeps no history committed nothing: the target
                // state it read returned at physical completion.
                let Some(advance) = advance else {
                    assert!(
                        self.head_reads_target(),
                        "invariant violated: a head with its own store owns no advance"
                    );
                    return Ok(());
                };
                let (OwnedAdvanceResolution::Aborted(state)
                | OwnedAdvanceResolution::Committed(state)) = advance
                    .commit(decision.accepted_rows)
                    .map_err(|(state, error)| {
                        self.put_head_state(request, state);
                        DomainError::from(error)
                    })?;
                self.put_head_state(request, state);
                Ok(())
            }
            PendingPayload::Encode { image } => {
                if let Outcome::Encode { features } = &pending.outcome {
                    let entry = self
                        .requests
                        .get_mut(&request)
                        .expect("invariant violated: a vision outcome's request has no admitted input");
                    let identity = entry
                        .images
                        .iter()
                        .find_map(|(identity, candidate)| {
                            (candidate == &image).then(|| identity.clone())
                        })
                        .expect("invariant violated: an encoded image is absent from its admitted input");
                    let resident = entry
                        .resident_mut()
                        .expect("invariant violated: a vision outcome's request is not resident");
                    assert!(
                        resident.image_features.insert(identity, features.clone()).is_none(),
                        "invariant violated: an encoded image's features are already installed"
                    );
                }
                Ok(())
            }
            PendingPayload::Target {
                advance,
                head,
                awaited,
            } => {
                // Its state is the outcome's alone until it reconciles, a chain it
                // carries with it: no other owner can hold it.
                assert!(
                    self.target_state(request).is_none()
                        && !(head.owns_head_state() && self.head_state(request).is_some()),
                    "invariant violated: a reconciled request has another physical state owner"
                );
                // How the guesses of a pass that ran rows ahead in a bank's step
                // fared, which only a bank reading experts ahead follows, told
                // before its state is published.
                if let (Some(bank), Some(step)) = (self.family.expert_bank(), awaited) {
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
                            self.put_target_state(request, state);
                            Ok(())
                        }
                        Err((state, error)) => {
                            self.put_target_state(request, state);
                            self.return_head_advances(request, head);
                            return Err(error.into());
                        }
                    },
                    // A chain holds its chunks' rows until it concludes.
                    Advance::Opens { flight, first } => self.open_chain(request, flight, first),
                    Advance::Chained {
                        flight,
                        joined,
                        chunk,
                    } => self.join_chain(request, flight, joined, chunk),
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
                match head {
                    TargetHeadAdvances::None => {}
                    TargetHeadAdvances::Primed(primed) => {
                        let rows = primed.rows();
                        match primed.commit(rows) {
                            Ok(
                                OwnedAdvanceResolution::Aborted(state)
                                | OwnedAdvanceResolution::Committed(state),
                            ) => {
                                self.put_head_state(request, state);
                            }
                            Err((_, error)) => broken(error),
                        }
                    }
                    TargetHeadAdvances::Front(front) => match front.commit(1) {
                        Ok(
                            OwnedAdvanceResolution::Aborted(state)
                            | OwnedAdvanceResolution::Committed(state),
                        ) => {
                            self.put_head_state(request, state);
                        }
                        Err((_, error)) => broken(error),
                    },
                    TargetHeadAdvances::PrimedAndFront { primed, front } => {
                        let rows = primed.rows();
                        let state = match primed.commit(rows) {
                            Ok(
                                OwnedAdvanceResolution::Aborted(state)
                                | OwnedAdvanceResolution::Committed(state),
                            ) => state,
                            Err((_, error)) => broken(error),
                        };
                        let front = front.attach(state).unwrap_or_else(|(_, error)| broken(error));
                        match front.commit(1) {
                            Ok(
                                OwnedAdvanceResolution::Aborted(state)
                                | OwnedAdvanceResolution::Committed(state),
                            ) => {
                                self.put_head_state(request, state);
                            }
                            Err((_, error)) => broken(error),
                        }
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
                self.resident_mut(request)
                    .expect("resident request has entry")
                    .boundary = boundary;
                Ok(())
            }
        }
    }

    /// Cancellation while completion is already available makes no physical
    /// successor visible and returns the original accepted sequence. A pass
    /// that ran rows ahead in a bank's step is discarded
    /// (`ElasticBank::discarded`).
    pub fn abort(&mut self, pending: PendingOperationOutcome) -> Result<(), DomainError> {
        let request = pending.request;
        match pending.payload {
            PendingPayload::Head { advance } => {
                assert!(
                    self.head_state(request).is_none(),
                    "invariant violated: an aborted request has another physical state owner"
                );
                if let Some(advance) = advance {
                    self.put_head_state(request, advance.abort());
                }
                Ok(())
            }
            PendingPayload::Encode { .. } => Ok(()),
            PendingPayload::Target {
                advance,
                head,
                awaited,
            } => {
                // Its state is the outcome's alone until it aborts, a chain it
                // carries with it: no other owner can hold it.
                assert!(
                    self.target_state(request).is_none()
                        && !(head.owns_head_state() && self.head_state(request).is_some()),
                    "invariant violated: an aborted request has another physical state owner"
                );
                // No verdict will judge a pass that ran rows ahead in a bank's step.
                if let (Some(bank), Some(step)) = (self.family.expert_bank(), awaited) {
                    bank.discarded(step, request);
                }
                self.return_head_advances(request, head);
                self.abort_target(request, advance)
            }
        }
    }

    /// Return the accepted target state `request`'s aborted `advance` came
    /// from.
    fn abort_target(&mut self, request: RequestId, advance: Advance) -> Result<(), DomainError> {
        match advance {
            Advance::Step(advance) => {
                self.put_target_state(request, advance.abort());
                Ok(())
            }
            // A chain's first chunk: the chunks queued behind it follow its
            // rows, so they are concluded before its state returns.
            Advance::Opens { flight, first } => {
                let concluded = match self.queue.opens_chain(flight) {
                    true => self.conclude_chain(),
                    false => Ok(()),
                };
                self.put_target_state(request, first.abort());
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
    fn return_head_advances(&mut self, request: RequestId, head: TargetHeadAdvances) {
        match head {
            TargetHeadAdvances::None => {}
            TargetHeadAdvances::Primed(primed) => {
                self.put_head_state(request, primed.abort());
            }
            TargetHeadAdvances::Front(front) => {
                self.put_head_state(request, front.abort());
            }
            TargetHeadAdvances::PrimedAndFront { primed, front } => {
                drop(front);
                self.put_head_state(request, primed.abort());
            }
        }
    }
}
