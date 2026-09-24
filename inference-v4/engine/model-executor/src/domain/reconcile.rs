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
        self.healthy()?;
        if matches!(pending.outcome, Outcome::Head { .. }) {
            // A head commits exactly its entry rows; its chained proposal
            // rows never become visible.
            if decision.accepted_rows != pending.committed_rows {
                let _ = self.abort(pending);
                return Err("head decision differs from its committed entry rows".into());
            }
            let advance = pending
                .advance
                .take()
                .ok_or_else(|| self.fatal_invariant("head outcome has no owned advance"))?;
            let (OwnedAdvanceResolution::Aborted(state) | OwnedAdvanceResolution::Committed(state)) =
                advance
                    .commit(decision.accepted_rows)
                    .map_err(|(state, error)| {
                        self.head.insert(pending.request, state);
                        self.fatal_state(error)
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
                let Some(input) = self.input.get(&pending.request) else {
                    return Err(self.fatal_invariant("vision outcome has no admitted input"));
                };
                let Some(existing) = input.images.values().find(|slot| &slot.image == image) else {
                    return Err(self.fatal_invariant("encoded image is absent from admitted input"));
                };
                if existing.features.is_some() {
                    return Err(self.fatal_invariant("encoded image feature is already installed"));
                }
                let input = self
                    .input
                    .get_mut(&pending.request)
                    .expect("admitted input checked above");
                let slot = input
                    .images
                    .values_mut()
                    .find(|slot| &slot.image == image)
                    .expect("admitted image checked above");
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
        if self.target.contains_key(&request) {
            return Err(self.fatal_invariant("request already has another physical state owner"));
        }
        let Some(advance) = pending.advance.take() else {
            return Err(self.fatal_invariant("target outcome has no owned target advance"));
        };
        match advance.commit(decision.accepted_rows) {
            Ok(OwnedAdvanceResolution::Aborted(state) | OwnedAdvanceResolution::Committed(state)) => {
                self.target.insert(request, state);
                Ok(())
            }
            Err((state, error)) => {
                self.target.insert(request, state);
                Err(self.fatal_state(error))
            }
        }
    }

    /// Cancellation while completion is already available makes no physical
    /// successor visible and returns the original accepted sequence.
    pub fn abort(&mut self, pending: PendingOperationOutcome) -> Result<(), DomainError> {
        let conflicting_owner = match &pending.outcome {
            Outcome::Head { .. } => self.head.contains_key(&pending.request),
            Outcome::Forward { .. } => self.target.contains_key(&pending.request),
            Outcome::Encode { .. } => false,
        };
        if conflicting_owner {
            return Err(self.fatal_invariant("request already has another physical state owner"));
        }
        if let Some(advance) = pending.advance {
            match pending.outcome {
                Outcome::Head { .. } => {
                    self.head.insert(pending.request, advance.abort());
                }
                Outcome::Forward { .. } => {
                    self.target.insert(pending.request, advance.abort());
                }
                Outcome::Encode { .. } => {
                    return Err(
                        self.fatal_invariant("stateless outcome unexpectedly owns sequence state")
                    );
                }
            }
        }
        Ok(())
    }
}
