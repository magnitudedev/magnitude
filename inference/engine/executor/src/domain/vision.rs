//! vision lifecycle for the executor domain.

use super::*;

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Vision launches are one prepared image each. The physical feature
    /// remains a proposal until `reconcile` installs it in admitted input.
    /// The flight holds `bindings` until `finish_vision` returns them.
    pub fn submit_vision(
        &mut self,
        bindings: StateBindings<F>,
        operation: &Operation,
        workspace: NativeGraphWorkspaceLease,
        output: NativeGraphOutputLease,
    ) -> Result<VisionFlight<F>, SubmitFailure<F>> {
        let (request, image, batch) = match self.vision_batch(operation) {
            Ok(checked) => checked,
            Err(error) => return Err(SubmitFailure::Refused(error, bindings)),
        };
        let launch = ValidatedVisionLaunch::new(
            VisionLaunchInputs::new(batch, workspace, output),
            self.domain.id(),
        )
        .map_err(|(_, error)| SubmitFailure::Failed(DomainError::Invariant(error)))?;
        let started = Instant::now();
        let submission = self
            .family
            .submit_vision(launch)
            .map_err(|(error, _)| SubmitFailure::Failed(error.into()))?;
        Ok(VisionFlight {
            request,
            image,
            submission,
            started,
            bindings,
        })
    }

    /// The one image `operation` encodes, checked against admitted input.
    fn vision_batch(
        &self,
        operation: &Operation,
    ) -> Result<(RequestId, ImageRef, crate::batching::ValidatedVisionBatch), DomainError> {
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
        let prepared = image.prepared_input().map_err(|error| error.to_string())?;
        let class = self.image_class(prepared.spatial().rows())?;
        let batch = crate::batching::ValidatedVisionBatch::new(prepared, class)
            .map_err(|error| error.to_string())?;
        Ok((*request, image.clone(), batch))
    }

    /// The patch rows of the class an image of `patch_rows` encodes in: the
    /// cell class covering its cells, within the load's image cell limit.
    pub(super) fn image_class(&self, patch_rows: usize) -> Result<usize, String> {
        let cell = self
            .definition
            .vision
            .as_ref()
            .ok_or("vision definition is absent")?
            .cell_rows();
        let cell = usize::try_from(cell).map_err(|_| "vision merge area exceeds host domain")?;
        let limit = self.execution.policy().limits().max_image_cells;
        if cell == 0 || !patch_rows.is_multiple_of(cell) || patch_rows / cell > limit {
            return Err(format!(
                "image encodes {patch_rows} vision patch rows but the load prepares at most {}",
                limit.saturating_mul(cell)
            ));
        }
        let cells = magnitude_batching::image_cell_class(patch_rows / cell)
            .ok_or("image encodes no vision cells")?;
        Ok(cells * cell)
    }

    /// Complete an encode and return the bindings its flight held. An error
    /// consumes them: the domain cannot continue.
    pub fn finish_vision(
        &mut self,
        flight: VisionFlight<F>,
    ) -> Result<(PendingOperationOutcome, StateBindings<F>), DomainError> {
        let VisionFlight {
            request,
            image,
            submission,
            started,
            bindings,
        } = flight;
        let completed = submission.finish().map_err(DomainError::Device)?;
        let duration = started.elapsed();
        let (core, output) = completed.into_parts();
        let patch_rows = core.batch().patch_rows();
        let merge = usize::try_from(
            self.definition
                .vision
                .as_ref()
                .ok_or_else(|| DomainError::invariant("vision definition is absent"))?
                .cell_rows(),
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
        let pending = PendingOperationOutcome {
            request,
            outcome: Outcome::Encode { features },
            advance: None,
            primed: None,
            rows: 0,
            committed_rows: 0,
            kind: WorkKind::Prefill,
            physical_duration: duration,
            image: Some(image),
        };
        Ok((pending, bindings))
    }
}
