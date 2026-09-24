//! Target submission, completion, and row construction.

use super::*;

/// Host timing of one finished target step. Device execution overlaps the
/// encode span; the selection gap is time the device may idle between steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetHostTiming {
    /// From entry into `submit_target` to the step's first commit.
    pub launch: Duration,
    /// From the step's first to its last command-buffer commit.
    pub encode: Duration,
    /// From the previous selection read to this step's first commit, when a
    /// selection was read since the previous step was submitted.
    pub selection_to_commit: Option<Duration>,
    /// From entry into `finish_target` to its return.
    pub finish: Duration,
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Submit one target group with state advances already reserved, or
    /// claim the queued lookahead step it equals. A continuable group queues
    /// its own lookahead behind it.
    pub fn submit_target(
        &mut self,
        operations: &[Operation],
        reservation: TargetGraphReservation,
    ) -> Result<TargetFlight<F::TargetSubmission>, DomainError> {
        let flight = match reservation {
            TargetGraphReservation::Claim(slots) => self.claim_lookahead(operations, slots)?,
            TargetGraphReservation::Launch(reservation) => {
                self.launch_target(operations, reservation)?
            }
        };
        self.queue_lookahead(&flight, operations)?;
        Ok(flight)
    }

    fn launch_target(
        &mut self,
        operations: &[Operation],
        reservation: TargetLaunchReservation,
    ) -> Result<TargetFlight<F::TargetSubmission>, DomainError> {
        let TargetLaunchReservation {
            advances,
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output,
        } = reservation;
        let advances = advances
            .into_iter()
            .map(TentativeAdvance::Accepted)
            .collect::<Vec<_>>();
        self.healthy()?;
        if operations.is_empty() {
            return Err("target group is empty".into());
        }
        let mut seen = BTreeSet::new();
        let mut rows = 0usize;
        let mut demand = crate::batching::Demand::NONE;
        let mut segments = 1usize;
        if operations.len() != advances.len() {
            return Err(
                self.fatal_invariant("target reservation advance count differs from operations")
            );
        }
        for (operation, advance) in operations.iter().zip(&advances) {
            operation.validate().map_err(|error| error.to_string())?;
            let Operation::Forward {
                request,
                position,
                tokens,
                conditioning,
                demand: next,
                ..
            } = operation
            else {
                return Err("target group contains another operation kind".into());
            };
            if !seen.insert(*request) {
                return Err("target group repeats a request".into());
            }
            if advance.position() != *position || advance.rows() != tokens.len() {
                return Err("target position differs from accepted state".into());
            }
            if conditioning.as_ref().is_some_and(|lease| {
                lease.domain() != self.domain.id() || lease.allocation().rows() != tokens.len()
            }) {
                return Err(
                    "target conditioning differs from physical rows or resource domain".into(),
                );
            }
            rows = rows
                .checked_add(tokens.len())
                .ok_or("target row count overflow")?;
            demand |= *next;
            segments = segments.max(advance.history_ranges().len());
        }
        let limits = self.execution.policy().limits();
        let class = crate::LaunchClass::covering(rows, segments, demand, limits.max_batch_rows)
            .map_err(|error| error.to_string())?;
        let mut metadata = Vec::with_capacity(operations.len());
        for operation in operations {
            let request = operation.request();
            let Operation::Forward {
                conditioning,
                kind,
                committed,
                ..
            } = operation
            else {
                unreachable!()
            };
            metadata.push((
                request,
                operation.row_count(),
                conditioning.clone(),
                *kind,
                *committed,
            ));
        }
        let prepared = operations
            .iter()
            .zip(&advances)
            .map(|(operation, advance)| self.target_slot(operation, advance))
            .collect::<Result<Vec<_>, _>>();
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                self.restore_advances(metadata, advances);
                return Err(error.into());
            }
        };
        let (slots, conditioning_slices): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
        let batch = match ValidatedTargetBatch::from_slots(
            &slots,
            self.definition.geometry.vocabulary as usize,
            limits.max_batch_rows,
        ) {
            Ok(batch) if batch.class() == class => batch,
            Ok(_) => {
                self.restore_advances(metadata, advances);
                return Err(self.fatal_invariant("target packed class changed after reservation"));
            }
            Err(error) => {
                self.restore_advances(metadata, advances);
                return Err(error.to_string().into());
            }
        };
        let conditioning = metadata
            .iter()
            .map(|(_, _, lease, _, _)| lease.clone())
            .collect();
        let inputs = TargetLaunchInputs::new(
            batch,
            TargetTokens::Host,
            advances,
            conditioning,
            conditioning_slices.clone(),
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output,
        );
        let launch = match ValidatedTargetLaunch::new(
            inputs,
            &self.target_store,
            self.domain.id(),
            self.definition.geometry.hidden as usize,
        ) {
            Ok(launch) => launch,
            Err((inputs, error)) => {
                let (_, advances, _, _) = inputs.into_parts();
                self.restore_advances(metadata, advances);
                let failure = DomainError::Invariant(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        let started = Instant::now();
        let submission = match self.family.submit_target(launch) {
            Ok(submission) => submission,
            Err((error, launch)) => {
                let (core, _) = launch.into_submission_parts();
                let (_, advances, _, _) = core.into_parts();
                self.restore_advances(metadata, advances);
                let failure = DomainError::from(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        Ok(TargetFlight {
            requests: metadata,
            submission,
            started,
            previous_selection: self.selection_read.take(),
            slots,
            conditioning_slices,
            id: self.flight_id(),
            continuation: None,
        })
    }

    /// Completion exposes immutable outcomes while every owned advance stays
    /// suspended. The caller can stage grammar and method changes from these
    /// views before choosing physical prefixes.
    pub fn finish_target(
        &mut self,
        flight: TargetFlight<F::TargetSubmission>,
    ) -> Result<Vec<PendingOperationOutcome>, DomainError> {
        let result = self.finish_target_inner(flight);
        if let Err(error) = &result {
            self.fatal = Some(error.clone());
        }
        result
    }

    fn finish_target_inner(
        &mut self,
        flight: TargetFlight<F::TargetSubmission>,
    ) -> Result<Vec<PendingOperationOutcome>, DomainError> {
        let completed = match flight.submission.finish() {
            Ok(completed) => completed,
            Err(error) => {
                return Err(DomainError::Device(error));
            }
        };
        let finish_started = Instant::now();
        let physical_duration = flight.started.elapsed();
        let (core, output) = completed.into_parts();
        let crate::TargetOutput {
            readout: output,
            commits,
        } = output;
        let batch = core.batch();
        if batch.actual_slots() != flight.requests.len() {
            return Err("completed target slot count differs from submitted requests".into());
        }
        let upload = batch.upload();
        let selected = if upload.select_rows.is_empty() {
            Vec::new()
        } else {
            let tensor = output
                .as_ref()
                .and_then(|result| result.selected.as_ref())
                .ok_or("completed target has no selection tensor")?
                .slice_leading(0, upload.select_rows.len() as u64)
                .map_err(|error| error.to_string())?;
            let bytes = tensor.tensor().read_to_host().map_err(|error| {
                DomainError::Device(crate::DeviceError::Transfer(error.to_string()))
            })?;
            self.selection_read = Some(Instant::now());
            decode_selected(&bytes)?
        };
        self.predecessor_selected(flight.id, &selected);
        let mut physical = vec![RowResult::default(); batch.actual_rows()];
        for (projected_index, &output_index) in output
            .as_ref()
            .map(|result| result.projected_output_rows.as_slice())
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            let row = *upload
                .out_rows
                .get(output_index)
                .ok_or("projected output row exceeds output rows")?;
            let row = usize::try_from(row).map_err(|_| "negative target output row")?;
            let result = physical
                .get_mut(row)
                .ok_or("target output row exceeds actual batch")?;
            let demand = crate::batching::Demand::from_bits(upload.demand[row])
                .ok_or("target output demand is invalid")?;
            if demand.contains(crate::batching::Demand::LOGITS) {
                let view = output
                    .as_ref()
                    .and_then(|result| result.logits.as_ref())
                    .ok_or("target logits output is absent")?
                    .slice_leading(projected_index as u64, projected_index as u64 + 1)
                    .map_err(|error| error.to_string())?;
                result.logits = Some(
                    self.domain
                        .publish_graph_logits(view)
                        .map_err(|error| error.to_string())?,
                );
            }
        }
        for (index, &output_index) in upload.select_rows.iter().enumerate() {
            let output_index = usize::try_from(output_index)
                .map_err(|_| "negative target selection output index")?;
            let row = *upload
                .out_rows
                .get(output_index)
                .ok_or("target selection output index exceeds output rows")?;
            let row = usize::try_from(row).map_err(|_| "negative target selection row")?;
            let result = physical
                .get_mut(row)
                .ok_or("target selection row exceeds actual batch")?;
            result.selected = Some(
                *selected
                    .get(index)
                    .ok_or("target selection count differs")?,
            );
        }
        let mut request_start = 0usize;
        for (_, request_rows, _, _, _) in &flight.requests {
            let request_end = request_start
                .checked_add(*request_rows)
                .ok_or("request row range overflows")?;
            let feature_outputs = upload
                .out_rows
                .iter()
                .enumerate()
                .filter_map(|(output_index, row)| {
                    let row = usize::try_from(*row).ok()?;
                    let demand = crate::batching::Demand::from_bits(*upload.demand.get(row)?)?;
                    (row >= request_start
                        && row < request_end
                        && demand.contains(crate::batching::Demand::FEATURES))
                    .then_some((output_index, row))
                })
                .collect::<Vec<_>>();
            if let Some(&(first, _)) = feature_outputs.first() {
                if feature_outputs
                    .iter()
                    .enumerate()
                    .any(|(offset, (index, _))| *index != first + offset)
                {
                    return Err("request feature rows are not contiguous in physical output".into());
                }
                let view = output
                    .as_ref()
                    .map(|result| &result.features)
                    .ok_or("target feature output is absent")?
                    .slice_leading(first as u64, (first + feature_outputs.len()) as u64)
                    .map_err(|error| error.to_string())?;
                let aggregate = self
                    .domain
                    .publish_graph_features(view)
                    .map_err(|error| error.to_string())?;
                for (_, row) in feature_outputs {
                    physical[row].features = Some(aggregate.clone());
                }
            }
            request_start = request_end;
        }
        let (_, advances, _, _) = core.into_parts();
        let mut pending = Vec::with_capacity(flight.requests.len());
        let mut offset = 0usize;
        let mut continuation = flight.continuation;
        for (index, ((((request, rows, conditioning, kind, committed_rows), advance), slot), slices)) in
            flight
                .requests
                .into_iter()
                .zip(advances)
                .zip(flight.slots)
                .zip(flight.conditioning_slices)
                .enumerate()
        {
            let end = offset
                .checked_add(rows)
                .ok_or("target request row count overflow")?;
            let result = physical
                .get(offset..end)
                .ok_or("target request row slice exceeds batch")?
                .to_vec();
            offset = end;
            let source = continuation.as_mut().map(|sources| sources[index].take());
            let advance = match (advance, source) {
                (TentativeAdvance::Accepted(advance), None) => advance,
                (TentativeAdvance::Successor(successor), Some(Some(state))) => {
                    match successor.attach(state) {
                        Ok(advance) => advance,
                        Err((state, error)) => {
                            self.target.insert(request, state);
                            return Err(DomainError::invariant(format!(
                                "claimed lookahead slot no longer follows its state: {error}"
                            )));
                        }
                    }
                }
                // Nobody claimed this slot: its rows are discarded.
                (TentativeAdvance::Successor(_), Some(None)) => continue,
                _ => {
                    return Err(DomainError::invariant(
                        "target flight advances differ from its continuation",
                    ))
                }
            };
            pending.push(PendingOperationOutcome {
                request,
                outcome: Outcome::Forward { rows: result },
                advance: Some(advance),
                rows,
                committed_rows,
                kind,
                physical_duration,
                slot: Some(slot),
                conditioning,
                conditioning_slices: slices,
                image: None,
            });
        }
        let timing = TargetHostTiming {
            launch: commits.first.saturating_duration_since(flight.started),
            encode: commits.last.saturating_duration_since(commits.first),
            selection_to_commit: flight
                .previous_selection
                .map(|read| commits.first.saturating_duration_since(read)),
            finish: finish_started.elapsed(),
        };
        if self.trace_host_steps {
            eprintln!(
                "target host step rows={} launch_us={} encode_us={} selection_to_commit_us={} finish_us={}",
                physical.len(),
                timing.launch.as_micros(),
                timing.encode.as_micros(),
                timing
                    .selection_to_commit
                    .map_or_else(|| "-".to_owned(), |gap| gap.as_micros().to_string()),
                timing.finish.as_micros(),
            );
        }
        self.target_timing = Some(timing);
        Ok(pending)
    }

    fn restore_advances(
        &mut self,
        metadata: Vec<(
            RequestId,
            usize,
            Option<crate::ConditioningRef>,
            WorkKind,
            usize,
        )>,
        advances: Vec<TentativeAdvance>,
    ) {
        for ((request, _, _, _, _), advance) in metadata.into_iter().zip(advances) {
            match advance {
                TentativeAdvance::Accepted(advance) => {
                    self.target.insert(request, advance.abort());
                }
                // A successor's source state stays with its predecessor;
                // dropping it releases its rows and bank.
                TentativeAdvance::Successor(successor) => drop(successor),
            }
        }
    }

    pub(super) fn target_slot(
        &self,
        operation: &Operation,
        advance: &TentativeAdvance,
    ) -> Result<(Slot, Vec<crate::ConditioningSlice>), String> {
        let Operation::Forward {
            request,
            tokens,
            position,
            ..
        } = operation
        else {
            return Err("non-forward target operation".into());
        };
        let (coordinates, slices) = self.input_rows(*request, *position, tokens)?;
        let visible = advance
            .history_ranges()
            .into_iter()
            .map(|(start, count)| {
                Ok([
                    i32::try_from(start).map_err(|_| "history start exceeds i32")?,
                    i32::try_from(start.checked_add(count).ok_or("history end overflow")?)
                        .map_err(|_| "history end exceeds i32")?,
                ])
            })
            .collect::<Result<Vec<_>, String>>()?;
        let binding = advance.bindings();
        let mut rows = Vec::with_capacity(tokens.len());
        for (index, token) in tokens.iter().enumerate() {
            let coordinates = *coordinates
                .get(index)
                .ok_or("prepared input coordinate count differs from tokens")?;
            let destination = binding.destinations.get(index).map_or(Ok(-1), |value| {
                i32::try_from(*value).map_err(|_| "target destination exceeds i32")
            })?;
            rows.push(Row {
                token: i32::try_from(token.0).map_err(|_| "token exceeds i32")?,
                coordinates,
                visible: visible.clone(),
                destination,
                demand: row_demand(operation, index),
                select: selection(operation, index),
            });
        }
        Ok((
            Slot {
                rows,
                bank: i32::try_from(binding.previous_bank).map_err(|_| "bank exceeds i32")?,
                following_bank: i32::try_from(binding.following_bank)
                    .map_err(|_| "successor bank exceeds i32")?,
            },
            slices,
        ))
    }
}

fn row_demand(operation: &Operation, row: usize) -> crate::batching::Demand {
    let Operation::Forward {
        tokens,
        kind,
        demand,
        ..
    } = operation
    else {
        return crate::batching::Demand::NONE;
    };
    match kind {
        WorkKind::Decode | WorkKind::Verify => *demand,
        WorkKind::Prefill | WorkKind::Replay if row + 1 == tokens.len() => *demand,
        WorkKind::Prefill | WorkKind::Replay => {
            *demand & (crate::batching::Demand::FEATURES | crate::batching::Demand::TAPS)
        }
    }
}

fn selection(operation: &Operation, row: usize) -> Option<Select> {
    operation.selection_for_row(row).map(select_row)
}

/// The packed selection controls of one selection spec.
pub(super) fn select_row(spec: &crate::SelectSpec) -> Select {
    Select {
        draw: Draw {
            kind: match spec.sampling {
                crate::Sampling::Greedy => DrawKind::Greedy,
                crate::Sampling::Categorical => DrawKind::Categorical,
            },
            seed: spec.seed,
            position: spec.position as u64,
            domain: spec.domain,
        },
        mask: spec.mask.clone(),
        shaping: RowShaping {
            temperature: spec.shaping.temperature,
            top_p: spec.shaping.top_p,
            top_k: spec.shaping.top_k,
            min_p: spec.shaping.min_p,
            repetition_penalty: spec.shaping.repetition_penalty,
            presence_penalty: spec.shaping.presence_penalty,
            frequency_penalty: spec.shaping.frequency_penalty,
            flags: 0,
        },
        history: spec
            .history
            .as_ref()
            .map_or_else(Vec::new, |value| value.to_vec()),
    }
}
