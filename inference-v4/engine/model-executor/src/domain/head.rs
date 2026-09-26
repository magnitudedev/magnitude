//! head lifecycle for the executor domain.

use super::*;
use crate::batching::HeadSlot;

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Bytes of one head conditioning row: the target's normalized output
    /// feature in the activation representation.
    pub(super) fn head_conditioning_bytes(&self) -> usize {
        self.definition.geometry.hidden as usize * self.definition.geometry.activation_dtype.bytes()
    }

    /// Submit one batch of head transactions. Each commits its entry rows to
    /// head state when it completes; its chained proposal rows are scratch.
    pub fn submit_head(
        &mut self,
        operations: &[Operation],
        graph_workspace: NativeGraphWorkspaceLease,
        graph_output: NativeGraphOutputLease,
        advances: Vec<OwnedStateAdvance>,
    ) -> Result<HeadFlight<F::HeadSubmission>, DomainError> {
        self.healthy()?;
        let store = self
            .head_store
            .as_ref()
            .ok_or("head state is disabled")?
            .clone();
        if operations.is_empty() {
            return Err("head group is empty".into());
        }
        if operations.len() != advances.len() {
            return Err(
                self.fatal_invariant("head reservation advance count differs from operations")
            );
        }
        let mut seen = BTreeSet::new();
        let mut steps = 0usize;
        for (operation, advance) in operations.iter().zip(&advances) {
            operation.validate().map_err(|error| error.to_string())?;
            let Operation::Head {
                request,
                position,
                proposals,
                ..
            } = operation
            else {
                return Err("head group contains another operation kind".into());
            };
            if !seen.insert(*request) {
                return Err("head request is repeated".into());
            }
            if advance.position() != *position || advance.rows() != operation.row_count() {
                return Err("head position or rows differ from accepted state".into());
            }
            steps = steps.max(proposals.len());
        }
        let metadata = operations
            .iter()
            .map(|operation| match operation {
                Operation::Head {
                    request,
                    tokens,
                    proposals,
                    ..
                } => (*request, tokens.len(), proposals.len()),
                _ => unreachable!("validated head group"),
            })
            .collect::<Vec<_>>();
        let slots = operations
            .iter()
            .zip(&advances)
            .map(|(operation, advance)| self.head_slot(operation, advance, steps))
            .collect::<Result<Vec<_>, _>>();
        let slots = match slots {
            Ok(slots) => slots,
            Err(error) => {
                self.restore_head_advances(&metadata, advances);
                return Err(error.into());
            }
        };
        let batch = match crate::batching::ValidatedHeadBatch::from_slots(
            &slots,
            self.definition.geometry.vocabulary as usize,
            self.execution.policy().limits().max_batch_rows,
        ) {
            Ok(batch) => batch,
            Err(error) => {
                self.restore_head_advances(&metadata, advances);
                return Err(error.to_string().into());
            }
        };
        let conditioning = operations
            .iter()
            .map(|operation| match operation {
                Operation::Head { conditioning, .. } => conditioning.clone(),
                _ => unreachable!("validated head group"),
            })
            .collect();
        let inputs =
            HeadLaunchInputs::new(batch, advances, conditioning, graph_workspace, graph_output);
        let launch = match ValidatedHeadLaunch::new(
            inputs,
            &store,
            self.domain.id(),
            self.head_conditioning_bytes(),
        ) {
            Ok(launch) => launch,
            Err((inputs, error)) => {
                let (_, advances, _, _, _) = inputs.into_parts();
                self.restore_head_advances(&metadata, advances);
                let failure = DomainError::Invariant(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        let started = Instant::now();
        let submission = match self.family.submit_head(launch) {
            Ok(submission) => submission,
            Err((error, launch)) => {
                let (core, _, _) = launch.into_submission_parts();
                let (_, advances, _) = core.into_parts();
                self.restore_head_advances(&metadata, advances);
                let failure = DomainError::from(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        Ok(HeadFlight {
            requests: metadata,
            steps,
            submission,
            started,
        })
    }

    fn restore_head_advances(
        &mut self,
        metadata: &[(RequestId, usize, usize)],
        advances: Vec<OwnedStateAdvance>,
    ) {
        for ((request, _, _), advance) in metadata.iter().zip(advances) {
            self.head.insert(*request, advance.abort());
        }
    }

    /// The head rows of one transaction padded to `steps` selections: entry
    /// rows at the accepted head position, then one chained row per further
    /// step. Chained rows past the request's own proposals append nowhere and
    /// select at the request's last proposal position; their selections are
    /// discarded.
    fn head_slot(
        &self,
        operation: &Operation,
        advance: &OwnedStateAdvance,
        steps: usize,
    ) -> Result<HeadSlot, String> {
        let Operation::Head {
            request,
            tokens,
            position,
            proposals,
            ..
        } = operation
        else {
            return Err("non-head operation".into());
        };
        let binding = advance.bindings();
        let i32_of = |value: usize, what: &str| {
            i32::try_from(value).map_err(|_| format!("head {what} exceeds i32"))
        };
        let history = advance
            .history_ranges()
            .into_iter()
            .map(|(start, count)| {
                Ok([
                    i32_of(start, "history start")?,
                    i32_of(
                        start
                            .checked_add(count)
                            .ok_or("head history end overflow")?,
                        "history end",
                    )?,
                ])
            })
            .collect::<Result<Vec<[i32; 2]>, String>>()?;
        let destination = |row: usize| -> Result<i32, String> {
            binding
                .destinations
                .get(row)
                .map_or(Ok(-1), |value| i32_of(*value, "destination"))
        };
        // A head row at head position p pairs the token after target row p
        // with that row's feature, so it takes target row p's coordinates.
        let chained = steps.saturating_sub(1);
        let coordinates = self.input_coordinates(*request, *position, tokens.len() + chained)?;
        let row = |index: usize, token: i32, visible: Vec<[i32; 2]>| -> Result<Row, String> {
            Ok(Row {
                token,
                coordinates: *coordinates
                    .get(index)
                    .ok_or("prepared input coordinate count differs from head rows")?,
                visible,
                destination: destination(index)?,
                demand: crate::batching::Demand::NONE,
                select: None,
            })
        };
        let entry = tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                row(
                    index,
                    i32::try_from(token.0).map_err(|_| "head token exceeds i32")?,
                    history.clone(),
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        // Chained row j sees the history, the entry rows, and the chained
        // rows before it; the entry rows were appended by the entry pass.
        let mut visible = history.clone();
        for index in 0..tokens.len() {
            let appended = destination(index)?;
            visible.push([appended, appended + 1]);
        }
        let mut chain = Vec::with_capacity(chained);
        for step in 1..steps {
            let index = tokens.len() + step - 1;
            chain.push(row(index, 0, coalesce(&visible))?);
            let appended = destination(index)?;
            if appended >= 0 {
                visible.push([appended, appended + 1]);
            }
        }
        let last = proposals.last();
        let selections = (0..steps)
            .map(|step| {
                let spec = proposals
                    .get(step)
                    .or(last)
                    .ok_or("a causal head in a drafting batch has no selection")?;
                Ok(super::target::select_row(spec))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(HeadSlot {
            entry: Slot {
                rows: entry,
                bank: i32_of(binding.previous_bank, "bank")?,
                previous_tape: i32_of(binding.previous_tape, "tape rows")?,
                following_bank: i32_of(binding.following_bank, "successor bank")?,
                stop: i32_of(binding.stop, "committed rows")?,
            },
            chain,
            proposals: selections,
        })
    }

    /// Complete a head batch: read every drafted selection once, then hold
    /// each transaction for its owner's reconciliation.
    pub fn finish_head(
        &mut self,
        flight: HeadFlight<F::HeadSubmission>,
    ) -> Result<Vec<PendingOperationOutcome>, DomainError> {
        let result = self.finish_head_inner(flight);
        if let Err(error) = &result {
            self.fatal = Some(error.clone());
        }
        result
    }

    fn finish_head_inner(
        &mut self,
        flight: HeadFlight<F::HeadSubmission>,
    ) -> Result<Vec<PendingOperationOutcome>, DomainError> {
        let completed = flight.submission.finish().map_err(DomainError::Device)?;
        let duration = flight.started.elapsed();
        let (core, selections) = completed.into_parts();
        let slots = core.batch().actual_slots();
        if slots != flight.requests.len() {
            return Err(DomainError::invariant(
                "head slot count differs from request count",
            ));
        }
        let selected = match selections {
            Some(selections) => {
                decode_selected(&selections.tensor().read_to_host().map_err(|error| {
                    DomainError::Device(crate::DeviceError::Transfer(error.to_string()))
                })?)
                .map_err(DomainError::invariant)?
            }
            None => Vec::new(),
        };
        let slot_class = selected.len().checked_div(flight.steps).unwrap_or(0);
        if flight.steps != 0 && (slot_class < slots || selected.len() != flight.steps * slot_class)
        {
            return Err(DomainError::invariant(
                "head selections differ from the batch's steps and slots",
            ));
        }
        let (_, advances, _) = core.into_parts();
        Ok(flight
            .requests
            .into_iter()
            .zip(advances)
            .enumerate()
            .map(
                |(slot, ((request, rows, proposals), advance))| PendingOperationOutcome {
                    request,
                    outcome: Outcome::Head {
                        proposals: (0..proposals)
                            .map(|step| selected[step * slot_class + slot])
                            .collect(),
                    },
                    advance: Some(advance),
                    rows: advance_rows(rows, proposals),
                    committed_rows: rows,
                    kind: WorkKind::Decode,
                    physical_duration: duration,
                    image: None,
                },
            )
            .collect())
    }
}

fn advance_rows(entry: usize, proposals: usize) -> usize {
    entry + proposals.saturating_sub(1)
}

/// Ascending visible spans with adjacent spans merged.
fn coalesce(spans: &[[i32; 2]]) -> Vec<[i32; 2]> {
    let mut merged: Vec<[i32; 2]> = Vec::with_capacity(spans.len());
    for &span in spans {
        match merged.last_mut() {
            Some(previous) if previous[1] == span[0] => previous[1] = span[1],
            _ => merged.push(span),
        }
    }
    merged
}
