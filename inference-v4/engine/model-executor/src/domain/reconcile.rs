//! reconcile lifecycle for the executor domain.

use super::*;

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// The decision is relative to this request's submitted rows. An interior
    /// recurrent prefix stays owned by `repairs` until replay completes.
    pub fn reconcile(
        &mut self,
        mut pending: PendingOperationOutcome,
        decision: PhysicalDecision,
    ) -> Result<PhysicalResolution, DomainError> {
        self.healthy()?;
        if matches!(pending.outcome, Outcome::Head { .. }) {
            if decision.accepted_rows != 0
                || decision
                    .head_prefix
                    .is_some_and(|rows| rows > pending.rows())
            {
                let _ = self.abort(pending);
                return Err("head decision exceeds its completed numerical rows".into());
            }
            let advance = pending
                .advance
                .take()
                .ok_or_else(|| self.fatal_invariant("head outcome has no owned advance"))?;
            if let Some(rows) = decision.head_prefix {
                let resolved = advance.commit(rows).map_err(|(state, error)| {
                    self.head.insert(pending.request, state);
                    self.fatal_state(error)
                })?;
                let state = match resolved {
                    OwnedAdvanceResolution::Aborted(state)
                    | OwnedAdvanceResolution::Committed(state) => state,
                    OwnedAdvanceResolution::Repair(repair) => {
                        let error = self.fatal_invariant(
                            "head method decision needs unsupported recurrent repair",
                        );
                        drop(repair);
                        return Err(error);
                    }
                };
                self.head.insert(pending.request, state);
                return Ok(PhysicalResolution::Committed);
            }
            if self.head_pending.insert(pending.request, advance).is_some() {
                return Err(self.fatal_invariant("head request has two suspended advances"));
            }
            return Ok(PhysicalResolution::Committed);
        }
        if matches!(
            pending.outcome,
            Outcome::Project { .. } | Outcome::Encode { .. }
        ) {
            if decision.accepted_rows != 0 || decision.head_prefix.is_some() {
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
            return Ok(PhysicalResolution::Committed);
        }
        if decision.accepted_rows < pending.committed_rows
            || decision.accepted_rows > pending.rows()
        {
            let _ = self.abort(pending);
            return Err("accepted target prefix is outside the submitted row commitment".into());
        }
        let request = pending.request;
        if decision.head_prefix.is_some_and(|rows| {
            self.head_pending
                .get(&request)
                .is_none_or(|advance| rows > advance.rows())
        }) {
            let _ = self.abort(pending);
            return Err("accepted head prefix has no matching suspended head advance".into());
        }
        if self.target.contains_key(&request) || self.repairs.contains_key(&request) {
            return Err(self.fatal_invariant("request already has another physical state owner"));
        }
        let Some(advance) = pending.advance.take() else {
            return Err(self.fatal_invariant("target outcome has no owned target advance"));
        };
        let resolution = match advance.commit(decision.accepted_rows) {
            Ok(resolution) => resolution,
            Err((state, error)) => {
                self.target.insert(request, state);
                return Err(self.fatal_state(error));
            }
        };
        match resolution {
            OwnedAdvanceResolution::Aborted(state) | OwnedAdvanceResolution::Committed(state) => {
                self.target.insert(request, state);
                if let Some(head_prefix) = decision.head_prefix {
                    self.publish_head_prefix(request, head_prefix)?;
                }
                Ok(PhysicalResolution::Committed)
            }
            OwnedAdvanceResolution::Repair(repair) => {
                let rows = repair.rows();
                let mut slot = pending.slot.ok_or_else(|| {
                    self.fatal_invariant("recurrent repair has no original target rows")
                })?;
                slot.rows.truncate(rows);
                self.repairs.insert(
                    request,
                    PendingRepair {
                        advance: repair,
                        slot,
                        conditioning: pending.conditioning,
                        conditioning_slices: pending
                            .conditioning_slices
                            .into_iter()
                            .filter(|slice| slice.destination < rows)
                            .map(|mut slice| {
                                slice.source.count =
                                    slice.source.count.min(rows - slice.destination);
                                slice
                            })
                            .collect(),
                        head_prefix: decision.head_prefix,
                    },
                );
                Ok(PhysicalResolution::Repair { request, rows })
            }
        }
    }

    /// Cancellation while completion is already available makes no physical
    /// successor visible and returns the original accepted sequence.
    pub fn abort(&mut self, pending: PendingOperationOutcome) -> Result<(), DomainError> {
        let conflicting_owner = match &pending.outcome {
            Outcome::Head { .. } => self.head.contains_key(&pending.request),
            Outcome::Forward { .. } => {
                self.target.contains_key(&pending.request)
                    || self.repairs.contains_key(&pending.request)
            }
            Outcome::Project { .. } | Outcome::Encode { .. } | Outcome::Repair => false,
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
                _ => {
                    return Err(
                        self.fatal_invariant("stateless outcome unexpectedly owns sequence state")
                    );
                }
            }
        }
        Ok(())
    }

    /// Cancel a staged head successor before a target decision consumes it.
    pub fn abort_head_pending(&mut self, request: RequestId) -> Result<(), String> {
        if let Some(advance) = self.head_pending.remove(&request) {
            self.head.insert(request, advance.abort());
        }
        Ok(())
    }

    pub(super) fn publish_head_prefix(
        &mut self,
        request: RequestId,
        accepted: usize,
    ) -> Result<(), DomainError> {
        let advance = self
            .head_pending
            .remove(&request)
            .ok_or_else(|| self.fatal_invariant("head prefix has no suspended advance"))?;
        let resolution = match advance.commit(accepted) {
            Ok(value) => value,
            Err((state, error)) => {
                self.head.insert(request, state);
                return Err(self.fatal_state(error));
            }
        };
        let state = match resolution {
            OwnedAdvanceResolution::Aborted(state) | OwnedAdvanceResolution::Committed(state) => {
                state
            }
            OwnedAdvanceResolution::Repair(repair) => {
                let error =
                    self.fatal_invariant("head prefix requires unsupported recurrent repair");
                drop(repair);
                return Err(error);
            }
        };
        self.head.insert(request, state);
        Ok(())
    }
}
