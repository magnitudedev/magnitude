//! state lifecycle for the executor domain.

use super::*;

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Commit state backing for a launch before its capacity check: the rows
    /// and successor banks its operations will claim, placed so histories
    /// keep growing in place. A device limit leaves the backing unchanged and
    /// the capacity check reports the shortage.
    /// Histories at the segment limit are repacked first.
    pub fn provision(&mut self, operations: &[Operation]) -> Result<(), DomainError> {
        // A group claiming the queued lookahead needs nothing; any other
        // group first waits for the lookahead and releases it.
        if self.claim_slots(operations).is_some() {
            return Ok(());
        }
        self.orphan_lookahead()?;
        for operation in operations {
            if let Operation::Forward { request, .. } = operation {
                self.compact_target(*request)?;
            }
        }
        let mut target = Vec::new();
        let mut head = Vec::new();
        // A continuable step also queues its lookahead, which follows it by
        // one row and one bank.
        let lookahead = self.execution.policy().limits().lookahead;
        let mut target_banks = 0usize;
        for operation in operations {
            let rows = operation.row_count();
            match operation {
                Operation::Forward { request, .. } => {
                    if let Some(state) = self.target.get(request) {
                        let ahead = usize::from(
                            lookahead && lookahead::continuation_of(operation).is_some(),
                        );
                        target.push(state.demand(rows + ahead));
                        target_banks += 1 + ahead;
                    }
                }
                Operation::Head { request, .. } => {
                    if let Some(state) = self.head.get(request) {
                        head.push(state.demand(rows));
                    }
                }
                _ => {}
            }
        }
        if !target.is_empty() {
            self.target_store.provision(&target, target_banks)?;
        }
        if let (Some(store), false) = (&self.head_store, head.is_empty()) {
            store.provision(&head, head.len())?;
        }
        Ok(())
    }

    /// Repack the target history of a request at the visible segment limit
    /// into one contiguous run with the bit-exact state copy program, so its
    /// next launch cannot exceed the limit. Rare by construction (placement
    /// keeps histories in few runs), so the copy completes before the state
    /// publishes the new run and the launch proceeds.
    pub(super) fn compact_target(&mut self, request: RequestId) -> Result<(), DomainError> {
        let Some(state) = self.target.get(&request) else {
            return Ok(());
        };
        if !state.compaction_needed() || !self.family.state_is_bound() {
            return Ok(());
        }
        let rows = state
            .history_ranges()
            .iter()
            .map(|(_, count)| count)
            .sum::<usize>();
        self.target_store.provision(
            &[magnitude_model_state::RowDemand { after: None, rows }],
            0,
        )?;
        let state = self.target.remove(&request).expect("state checked above");
        // One copy launch of at most the largest prepared copy class.
        let max_rows = self.execution.policy().limits().max_batch_rows;
        let compaction = match OwnedCompaction::prepare(state, max_rows) {
            Ok(OwnedCompactionPreparation::Ready(compaction)) => compaction,
            Ok(
                OwnedCompactionPreparation::NotNeeded(state)
                | OwnedCompactionPreparation::Deferred { state, .. },
            ) => {
                self.target.insert(request, state);
                return Ok(());
            }
            Err((state, error)) => {
                self.target.insert(request, state);
                return Err(error.into());
            }
        };
        let workspace = match self.resources.state_graph().acquire_workspace() {
            Ok(workspace) => workspace,
            Err(error) => {
                self.target.insert(request, compaction.abort());
                return Err(DomainError::Capacity(error));
            }
        };
        let Some(class_rows) = crate::batching::row_class(compaction.rows()) else {
            self.target.insert(request, compaction.abort());
            return Err(self.fatal_invariant("a copy within the batch bound has no row class"));
        };
        let batch = match crate::batching::ValidatedStateBatch::copy(
            compaction.copies().to_vec(),
            self.target_store.history_capacity(),
            class_rows,
        ) {
            Ok(batch) => batch,
            Err(error) => {
                self.target.insert(request, compaction.abort());
                return Err(DomainError::invariant(error.to_string()));
            }
        };
        let inputs = StateLaunchInputs::new(batch, StateWork::Copy(compaction), workspace);
        let launch = match ValidatedStateLaunch::new(
            inputs,
            &self.target_store,
            None,
            self.domain.id(),
        ) {
            Ok(launch) => launch,
            Err((_, error)) => return Err(self.fatal_invariant(error.to_string())),
        };
        let submission = match self.family.submit_state(launch) {
            Ok(submission) => submission,
            Err((error, _)) => {
                let failure = DomainError::from(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        let completed = match submission.finish() {
            Ok(completed) => completed,
            Err(error) => {
                let failure = DomainError::Device(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        let (core, _) = completed.into_parts();
        let (_, work) = core.into_parts();
        let StateWork::Copy(compaction) = work else {
            return Err(self.fatal_invariant("state copy returned another maintenance kind"));
        };
        self.target.insert(request, compaction.commit());
        Ok(())
    }

    /// Commit the successor banks a newly opened request needs.
    pub fn provision_open(&self) -> Result<(), DomainError> {
        let requirements = self.open_requirements();
        self.target_store.provision(&[], requirements.target_banks())?;
        if let Some(store) = &self.head_store {
            store.provision(&[], requirements.head_banks())?;
        }
        Ok(())
    }

    /// Release committed state backing beyond what the stores need (see
    /// [`StateStore::shrink`]): at idle with hysteresis, under memory
    /// pressure fully. Returns the physical bytes released.
    /// A queued lookahead is released first: it holds a state transaction,
    /// under which nothing shrinks.
    pub fn shrink_state(
        &mut self,
        policy: magnitude_model_state::ShrinkPolicy,
    ) -> Result<u64, DomainError> {
        self.orphan_lookahead()?;
        let mut released = self.target_store.shrink(policy)?;
        if let Some(store) = &self.head_store {
            released += store.shrink(policy)?;
        }
        Ok(released)
    }

}
