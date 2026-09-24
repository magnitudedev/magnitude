//! vision lifecycle for the executor domain.

use super::*;

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Vision launches are one prepared image each. The physical feature
    /// remains a proposal until `reconcile` installs it in admitted input.
    pub fn submit_vision(
        &mut self,
        operation: &Operation,
        workspace: NativeGraphWorkspaceLease,
        output: NativeGraphOutputLease,
    ) -> Result<VisionFlight<F::VisionSubmission>, DomainError> {
        self.healthy()?;
        let Operation::Encode { request, image } = operation else {
            return Err("vision submission requires one Encode operation".into());
        };
        let input = self
            .input
            .get(&request)
            .ok_or("vision request has no admitted input")?;
        let slot = input
            .images
            .values()
            .find(|slot| slot.image == *image)
            .ok_or("image is not part of admitted input")?;
        if slot.features.is_some() {
            return Err("image is already encoded".into());
        }
        let patch_rows = image
            .prepared_input()
            .map_err(|error| error.to_string())?
            .spatial()
            .rows();
        let batch = crate::batching::ValidatedVisionBatch::new(
            image.prepared_input().map_err(|error| error.to_string())?,
            patch_rows,
        )
        .map_err(|error| error.to_string())?;
        let launch = ValidatedVisionLaunch::new(
            VisionLaunchInputs::new(batch, workspace, output),
            self.domain.id(),
        )
        .map_err(|(_, error)| {
            let failure = DomainError::Invariant(error);
            self.fatal = Some(failure.clone());
            failure
        })?;
        let started = Instant::now();
        let submission = match self.family.submit_vision(launch) {
            Ok(submission) => submission,
            Err((error, _launch)) => {
                let failure = DomainError::from(error);
                self.fatal = Some(failure.clone());
                return Err(failure);
            }
        };
        Ok(VisionFlight {
            request: *request,
            image: image.clone(),
            submission,
            started,
        })
    }

    pub fn finish_vision(
        &mut self,
        flight: VisionFlight<F::VisionSubmission>,
    ) -> Result<PendingOperationOutcome, DomainError> {
        let result = self.finish_vision_inner(flight);
        if let Err(error) = &result {
            self.fatal = Some(error.clone());
        }
        result
    }

    fn finish_vision_inner(
        &mut self,
        flight: VisionFlight<F::VisionSubmission>,
    ) -> Result<PendingOperationOutcome, DomainError> {
        let completed = flight.submission.finish().map_err(DomainError::Device)?;
        let duration = flight.started.elapsed();
        let (core, output) = completed.into_parts();
        let patch_rows = core.batch().patch_rows();
        let merge = self
            .definition
            .vision
            .as_ref()
            .ok_or_else(|| DomainError::invariant("vision definition is absent"))?
            .geometry
            .merge;
        let merge = usize::try_from(
            merge
                .checked_mul(merge)
                .ok_or_else(|| DomainError::invariant("vision merge area overflow"))?,
        )
        .map_err(|_| DomainError::invariant("vision merge area exceeds host domain"))?;
        if merge == 0 || !patch_rows.is_multiple_of(merge) {
            return Err(DomainError::invariant(
                "vision patch rows differ from merge geometry",
            ));
        }
        let output_rows = patch_rows / merge;
        let view = output
            .slice_leading(0, output_rows as u64)
            .map_err(|error| DomainError::invariant(error.to_string()))?;
        let features = self
            .domain
            .publish_graph_features(view)
            .map_err(|error| DomainError::invariant(error.to_string()))?;
        Ok(PendingOperationOutcome {
            request: flight.request,
            outcome: Outcome::Encode { features },
            advance: None,
            rows: 0,
            committed_rows: 0,
            kind: WorkKind::Prefill,
            physical_duration: duration,
            slot: None,
            conditioning: None,
            conditioning_slices: Vec::new(),
            image: Some(flight.image),
        })
    }
}
