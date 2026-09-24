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
            self.definition.geometry.hidden as usize,
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

    /// Replay a recurrent interior prefix while the accepted source remains
    /// pinned. The original token and coordinate rows are reused exactly;
    /// demand is cleared because repair publishes no user-facing result.
    pub fn submit_repair(
        &mut self,
        request: RequestId,
        reservation: ReservedRepair,
    ) -> Result<StateFlight<F::StateSubmission>, DomainError> {
        if !self.family.state_is_bound() {
            return Err("state program is unavailable".into());
        }
        let ReservedRepair {
            class_rows,
            state_graph_workspace,
            graph_workspace,
            graph_outputs,
            pending,
        } = reservation;
        let rows = pending.advance.rows();
        let mut slot = pending.slot.clone();
        let visible = pending
            .advance
            .history_ranges()
            .into_iter()
            .map(|(start, count)| {
                Ok([
                    i32::try_from(start).map_err(|_| "repair history start exceeds i32")?,
                    i32::try_from(
                        start
                            .checked_add(count)
                            .ok_or("repair history end overflow")?,
                    )
                    .map_err(|_| "repair history end exceeds i32")?,
                ])
            })
            .collect::<Result<Vec<_>, String>>()?;
        for row in &mut slot.rows {
            row.visible = visible.clone();
            row.demand = crate::batching::Demand::NONE;
            row.select = None;
        }
        slot.bank = i32::try_from(pending.advance.previous_bank())
            .map_err(|_| "repair bank exceeds i32")?;
        slot.following_bank = i32::try_from(pending.advance.following_bank())
            .map_err(|_| "repair successor bank exceeds i32")?;
        let replay = ValidatedTargetBatch::from_slots(
            &[slot],
            self.definition.geometry.vocabulary as usize,
            self.execution.policy().limits().max_batch_rows,
        )
        .map_err(|error| error.to_string())?;
        let batch = crate::batching::ValidatedStateBatch::repair(replay, rows, class_rows)
            .map_err(|error| error.to_string())?;
        let original_slot = pending.slot;
        let work = StateWork::RecurrentRepair {
            advance: pending.advance,
            conditioning: pending.conditioning,
            conditioning_slices: pending.conditioning_slices,
            graph_workspace,
            graph_outputs,
        };
        let inputs = StateLaunchInputs::new(batch, work, state_graph_workspace);
        let launch = match ValidatedStateLaunch::new(
            inputs,
            &self.target_store,
            None,
            self.domain.id(),
            self.definition.geometry.hidden as usize,
        ) {
            Ok(launch) => launch,
            Err((inputs, error)) => {
                let (_, work, _) = inputs.into_parts();
                if let StateWork::RecurrentRepair {
                    advance,
                    conditioning,
                    conditioning_slices,
                    ..
                } = work
                {
                    self.repairs.insert(
                        request,
                        PendingRepair {
                            advance,
                            slot: original_slot,
                            conditioning,
                            conditioning_slices,
                        },
                    );
                }
                let failure = DomainError::Invariant(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        let started = Instant::now();
        let submission = match self.family.submit_state(launch) {
            Ok(submission) => submission,
            Err((error, launch)) => {
                let (core, _) = launch.into_submission_parts();
                let (_, work) = core.into_parts();
                let StateWork::RecurrentRepair {
                    advance,
                    conditioning,
                    conditioning_slices,
                    ..
                } = work
                else {
                    unreachable!("a recurrent repair launch retains recurrent repair work")
                };
                self.repairs.insert(
                    request,
                    PendingRepair {
                        advance,
                        slot: original_slot,
                        conditioning,
                        conditioning_slices,
                    },
                );
                let failure = DomainError::from(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        Ok(StateFlight {
            request,
            submission,
            started,
        })
    }

    /// The numerical replay already completed; this is the sole publication
    /// of its accepted prefix and recurrent successor bank.
    pub fn finish_repair(
        &mut self,
        flight: StateFlight<F::StateSubmission>,
    ) -> Result<Duration, DomainError> {
        let result = self.finish_repair_inner(flight);
        if let Err(error) = &result {
            self.fatal = Some(error.clone());
        }
        result
    }

    fn finish_repair_inner(
        &mut self,
        flight: StateFlight<F::StateSubmission>,
    ) -> Result<Duration, DomainError> {
        let completed = match flight.submission.finish() {
            Ok(completed) => completed,
            Err(error) => {
                return Err(DomainError::Device(error));
            }
        };
        let duration = flight.started.elapsed();
        let (core, _) = completed.into_parts();
        let (_, work) = core.into_parts();
        let StateWork::RecurrentRepair { advance, .. } = work else {
            return Err(DomainError::invariant(
                "state flight returned another maintenance kind",
            ));
        };
        if self.target.contains_key(&flight.request) || self.repairs.contains_key(&flight.request) {
            return Err(DomainError::invariant(
                "repair request has another physical state owner",
            ));
        }
        self.target.insert(flight.request, advance.commit());
        Ok(duration)
    }

    pub fn abort_repair(&mut self, request: RequestId) -> Result<(), String> {
        let pending = self
            .repairs
            .remove(&request)
            .ok_or_else(|| "request has no queued recurrent repair".to_owned())?;
        self.target.insert(request, pending.advance.abort());
        Ok(())
    }
}
