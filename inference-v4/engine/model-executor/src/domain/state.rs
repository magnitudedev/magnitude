//! state lifecycle for the executor domain.

use super::*;

impl<F: ProgramFamily> ExecutorDomain<F> {
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
                            head_prefix: pending.head_prefix,
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
                        head_prefix: pending.head_prefix,
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
            head_prefix: pending.head_prefix,
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
        if let Some(head_prefix) = flight.head_prefix {
            self.publish_head_prefix(flight.request, head_prefix)?;
        }
        Ok(duration)
    }

    pub fn abort_repair(&mut self, request: RequestId) -> Result<(), String> {
        let pending = self
            .repairs
            .remove(&request)
            .ok_or_else(|| "request has no queued recurrent repair".to_owned())?;
        self.target.insert(request, pending.advance.abort());
        if let Some(head) = self.head_pending.remove(&request) {
            self.head.insert(request, head.abort());
        }
        Ok(())
    }
}
