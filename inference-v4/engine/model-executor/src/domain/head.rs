//! head lifecycle for the executor domain.

use super::*;

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Submit one batch of speculative head rows. The head successor is held
    /// separately from the accepted head state until a later target decision.
    pub fn submit_head(
        &mut self,
        operations: Vec<Operation>,
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
        let mut seen = BTreeSet::new();
        if operations.len() != advances.len() {
            return Err(
                self.fatal_invariant("head reservation advance count differs from operations")
            );
        }
        for (operation, advance) in operations.iter().zip(&advances) {
            operation.validate().map_err(|error| error.to_string())?;
            let Operation::Head {
                request,
                position,
                tokens,
                conditioning,
                ..
            } = operation
            else {
                return Err("head group contains another operation kind".into());
            };
            if !seen.insert(*request) || self.head_pending.contains_key(request) {
                return Err("head request is repeated or has a suspended advance".into());
            }
            if advance.position() != *position
                || advance.rows() != tokens.len()
                || conditioning.count != tokens.len()
                || conditioning.features.domain() != self.domain.id()
            {
                return Err("head position or conditioning differs from accepted state".into());
            }
        }
        let mut metadata = Vec::new();
        for operation in &operations {
            let request = operation.request();
            metadata.push((request, operation.row_count()));
        }
        let slots = operations
            .iter()
            .zip(&advances)
            .map(|(operation, advance)| self.head_slot(operation, advance))
            .collect::<Result<Vec<_>, _>>();
        let slots = match slots {
            Ok(slots) => slots,
            Err(error) => {
                self.restore_head_advances(metadata, advances);
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
                self.restore_head_advances(metadata, advances);
                return Err(error.to_string().into());
            }
        };
        let conditioning = operations
            .iter()
            .map(|operation| match operation {
                Operation::Head { conditioning, .. } => conditioning.clone(),
                _ => unreachable!(),
            })
            .collect();
        let inputs =
            HeadLaunchInputs::new(batch, advances, conditioning, graph_workspace, graph_output);
        let launch = match ValidatedHeadLaunch::new(
            inputs,
            &store,
            self.domain.id(),
            self.definition.geometry.hidden as usize,
        ) {
            Ok(launch) => launch,
            Err((inputs, error)) => {
                let (_, advances, _, _, _) = inputs.into_parts();
                self.restore_head_advances(metadata, advances);
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
                self.restore_head_advances(metadata, advances);
                let failure = DomainError::from(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        Ok(HeadFlight {
            requests: metadata,
            submission,
            started,
        })
    }

    fn restore_head_advances(
        &mut self,
        metadata: Vec<(RequestId, usize)>,
        advances: Vec<OwnedStateAdvance>,
    ) {
        for ((request, _), advance) in metadata.into_iter().zip(advances) {
            self.head.insert(request, advance.abort());
        }
    }

    fn head_slot(
        &self,
        operation: &Operation,
        advance: &OwnedStateAdvance,
    ) -> Result<Slot, String> {
        let Operation::Head {
            tokens, position, ..
        } = operation
        else {
            return Err("non-head operation".into());
        };
        let binding = advance.bindings();
        let visible = advance
            .history_ranges()
            .into_iter()
            .map(|(start, count)| {
                Ok([
                    i32::try_from(start).map_err(|_| "head history start exceeds i32")?,
                    i32::try_from(
                        start
                            .checked_add(count)
                            .ok_or("head history end overflow")?,
                    )
                    .map_err(|_| "head history end exceeds i32")?,
                ])
            })
            .collect::<Result<Vec<_>, String>>()?;
        let rows = tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let coordinate = i32::try_from(
                    position
                        .checked_add(index)
                        .ok_or("head position overflow")?,
                )
                .map_err(|_| "head position exceeds i32")?;
                let mut coordinates = [0; 4];
                coordinates[..usize::from(self.definition.inputs.coordinate_axes)].fill(coordinate);
                Ok(Row {
                    token: i32::try_from(token.0).map_err(|_| "head token exceeds i32")?,
                    coordinates,
                    visible: visible.clone(),
                    destination: binding.destinations.get(index).map_or(Ok(-1), |value| {
                        i32::try_from(*value).map_err(|_| "head destination exceeds i32")
                    })?,
                    demand: if index + 1 == tokens.len() {
                        crate::batching::Demand::FEATURES
                    } else {
                        crate::batching::Demand::NONE
                    },
                    select: None,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Slot {
            rows,
            bank: i32::try_from(binding.previous_bank).map_err(|_| "head bank exceeds i32")?,
        })
    }

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
        let completed = match flight.submission.finish() {
            Ok(completed) => completed,
            Err(error) => {
                return Err(DomainError::Device(error));
            }
        };
        let duration = flight.started.elapsed();
        let (core, output) = completed.into_parts();
        if core.batch().upload().out_rows.len() != flight.requests.len() {
            return Err(DomainError::invariant(
                "head readout count differs from request count",
            ));
        }
        let (_, advances, _) = core.into_parts();
        let mut pending = Vec::new();
        for (index, ((request, rows), advance)) in
            flight.requests.into_iter().zip(advances).enumerate()
        {
            let view = output
                .slice_leading(index as u64, index as u64 + 1)
                .map_err(|error| DomainError::invariant(error.to_string()))?;
            let features = self
                .domain
                .publish_graph_features(view)
                .map_err(|error| DomainError::invariant(error.to_string()))?;
            pending.push(PendingOperationOutcome {
                request,
                outcome: Outcome::Head { features },
                advance: Some(advance),
                rows,
                committed_rows: 0,
                kind: WorkKind::Decode,
                physical_duration: duration,
                slot: None,
                conditioning: None,
                conditioning_slices: Vec::new(),
                image: None,
            });
        }
        Ok(pending)
    }
}
