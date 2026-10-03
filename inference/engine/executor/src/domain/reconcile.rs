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
        #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
        if self.pipeline_owner.is_some() && pending.pipeline_prefix.is_none() {
            return Err(DomainError::invariant(
                "paired domain cannot publish suffix-only state",
            ));
        }
        #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
        if pending.pipeline_prefix.is_some()
            && (!matches!(pending.outcome, Outcome::Forward { .. }) || pending.primed.is_some())
        {
            return Err(DomainError::invariant(
                "pipeline state cannot publish through a head, vision or primed result",
            ));
        }
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
                .ok_or_else(|| DomainError::invariant("head outcome has no owned advance"))?;
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
                let Some(input) = self.input.get(&pending.request) else {
                    return Err(DomainError::invariant("vision outcome has no admitted input"));
                };
                let Some(existing) = input.images.values().find(|slot| &slot.image == image) else {
                    return Err(DomainError::invariant(
                        "encoded image is absent from admitted input",
                    ));
                };
                if existing.features.is_some() {
                    return Err(DomainError::invariant(
                        "encoded image feature is already installed",
                    ));
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
        if self.target.contains_key(&request)
            || (pending.primed.is_some() && self.head.contains_key(&request))
        {
            return Err(DomainError::invariant(
                "request already has another physical state owner",
            ));
        }
        let Some(advance) = pending.advance.take() else {
            return Err(DomainError::invariant(
                "target outcome has no owned target advance",
            ));
        };
        #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
        if let Some(prefix) = pending.pipeline_prefix.take() {
            if pending.primed.is_some() || self.pipeline_prefix.contains_key(&request) {
                return Err(DomainError::invariant(
                    "pipeline result has another state owner or primed work",
                ));
            }
            let transaction = magnitude_state::PipelineStateAdvance::new(vec![prefix, advance])
                .map_err(|failure| {
                    let (advances, error) = *failure;
                    self.restore_pipeline_sources(
                        request,
                        advances.into_iter().map(OwnedStateAdvance::abort).collect(),
                    );
                    DomainError::from(error)
                })?;
            // The complete physical pair must already have finished. Preflight
            // both stores before either publication; releasing claims does not
            // undo GPU writes. An error returns unchanged logical source states.
            return match transaction.commit(decision.accepted_rows) {
                Ok(state) => {
                    self.restore_pipeline_sources(request, state.into_stages());
                    Ok(())
                }
                Err(failure) => {
                    let (state, error) = *failure;
                    self.restore_pipeline_sources(request, state.into_stages());
                    Err(error.into())
                }
            };
        }
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
                    self.target.insert(request, advance.abort());
                    return Err(error.into());
                }
            }
        }
        match advance.commit(decision.accepted_rows) {
            Ok(
                OwnedAdvanceResolution::Aborted(state) | OwnedAdvanceResolution::Committed(state),
            ) => {
                self.target.insert(request, state);
                Ok(())
            }
            Err((state, error)) => {
                self.target.insert(request, state);
                Err(error.into())
            }
        }
    }

    /// Cancellation while completion is already available makes no physical
    /// successor visible and returns the original accepted sequence.
    pub fn abort(&mut self, pending: PendingOperationOutcome) -> Result<(), DomainError> {
        #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
        let mut pending = pending;
        #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
        if let Some(prefix) = pending.pipeline_prefix.take() {
            if !matches!(pending.outcome, Outcome::Forward { .. })
                || pending.primed.is_some()
                || self.pipeline_prefix.contains_key(&pending.request)
                || self.target.contains_key(&pending.request)
            {
                return Err(DomainError::invariant(
                    "pipeline abort has incompatible output or another owner",
                ));
            }
            let suffix = pending
                .advance
                .take()
                .ok_or_else(|| DomainError::invariant("pipeline abort has no suffix advance"))?;
            self.restore_pipeline_sources(pending.request, vec![prefix.abort(), suffix.abort()]);
            return Ok(());
        }
        let conflicting_owner = match &pending.outcome {
            Outcome::Head { .. } => self.head.contains_key(&pending.request),
            Outcome::Forward { .. } => {
                self.target.contains_key(&pending.request)
                    || (pending.primed.is_some() && self.head.contains_key(&pending.request))
            }
            Outcome::Encode { .. } => false,
        };
        if conflicting_owner {
            return Err(DomainError::invariant(
                "request already has another physical state owner",
            ));
        }
        if let Some(primed) = pending.primed {
            self.head.insert(pending.request, primed.abort());
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
                    return Err(DomainError::invariant(
                        "stateless outcome unexpectedly owns sequence state",
                    ));
                }
            }
        }
        Ok(())
    }
    #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
    pub(super) fn restore_pipeline_sources(
        &mut self,
        request: RequestId,
        stages: Vec<SequenceState>,
    ) {
        let [prefix, suffix]: [SequenceState; 2] = stages
            .try_into()
            .unwrap_or_else(|_| unreachable!("two-stage transaction retains exactly two sources"));
        self.pipeline_prefix.insert(request, prefix);
        self.target.insert(request, suffix);
    }
}
