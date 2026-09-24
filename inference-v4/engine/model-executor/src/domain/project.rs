//! project lifecycle for the executor domain.

use super::*;

impl<F: ProgramFamily> ExecutorDomain<F> {
    pub fn submit_project(
        &mut self,
        operations: &[Operation],
        graph_workspace: NativeGraphWorkspaceLease,
        graph_output: NativeGraphOutputLease,
    ) -> Result<ProjectFlight<F::ProjectSubmission>, DomainError> {
        self.healthy()?;
        if operations.is_empty() {
            return Err("projection group is empty".into());
        }
        let mut seen = BTreeSet::new();
        let mut rows = 0usize;
        let mut requests = Vec::new();
        let mut projections = Vec::new();
        for operation in operations {
            operation.validate().map_err(|error| error.to_string())?;
            let Operation::Project {
                request,
                features,
                select,
            } = operation
            else {
                return Err("projection group contains another operation kind".into());
            };
            if !seen.insert(*request)
                || !self.head.contains_key(request) && !self.head_pending.contains_key(request)
            {
                return Err("projection request is repeated or not open".into());
            }
            rows = rows
                .checked_add(features.allocation().rows())
                .ok_or("projection row count overflow")?;
            requests.push(*request);
            projections.push(ProjectionRequest::new(
                *request,
                features.clone(),
                select.clone(),
            ));
        }
        let class = crate::LaunchClass::covering(
            rows,
            requests.len(),
            crate::batching::Demand::SELECT,
            self.execution.policy().limits().max_batch_rows,
        )
        .map_err(|error| error.to_string())?;
        let launch = ValidatedProjectionLaunch::new(
            ProjectionLaunchInputs::new(projections, class, graph_workspace, graph_output),
            self.domain.id(),
            self.definition.geometry.hidden as usize,
            self.definition.geometry.vocabulary as usize,
        )
        .map_err(|(_, error)| error.to_string())?;
        let started = Instant::now();
        let submission = match self.family.submit_project(launch) {
            Ok(submission) => submission,
            Err((error, _launch)) => {
                let failure = DomainError::from(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        Ok(ProjectFlight {
            requests,
            submission,
            started,
        })
    }

    pub fn finish_project(
        &mut self,
        flight: ProjectFlight<F::ProjectSubmission>,
    ) -> Result<Vec<PendingOperationOutcome>, DomainError> {
        let result = self.finish_project_inner(flight);
        if let Err(DomainError::Device(error)) = &result {
            self.fatal = Some(DomainError::Device(error.clone()));
        }
        result
    }

    fn finish_project_inner(
        &mut self,
        flight: ProjectFlight<F::ProjectSubmission>,
    ) -> Result<Vec<PendingOperationOutcome>, DomainError> {
        let completed = flight.submission.finish().map_err(DomainError::Device)?;
        let duration = flight.started.elapsed();
        let (_, output) = completed.into_parts();
        let selected = output
            .selected
            .slice_leading(0, flight.requests.len() as u64)
            .map_err(|error| error.to_string())?;
        let selected = decode_selected(&selected.tensor().read_to_host().map_err(|error| {
            DomainError::Device(crate::DeviceError::Transfer(error.to_string()))
        })?)?;
        if selected.len() != flight.requests.len() {
            return Err("projection result count differs from submitted requests".into());
        }
        Ok(flight
            .requests
            .into_iter()
            .zip(selected)
            .map(|(request, selected)| PendingOperationOutcome {
                request,
                outcome: Outcome::Project {
                    selected: vec![selected],
                },
                advance: None,
                rows: 0,
                committed_rows: 0,
                kind: WorkKind::Decode,
                physical_duration: duration,
                slot: None,
                conditioning: None,
                conditioning_slices: Vec::new(),
                image: None,
            })
            .collect())
    }
}
