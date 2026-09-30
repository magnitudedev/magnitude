//! input lifecycle for the executor domain.

use super::*;

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Install a request's prepared input for its lifetime. It is not
    /// resident until [`Self::open_state`]; its images are encoded then.
    pub fn install_input(
        &mut self,
        request: RequestId,
        input: PreparedModelInput,
    ) -> Result<(), String> {
        if self.input.contains_key(&request)
            || self.target.contains_key(&request)
            || self.head.contains_key(&request)
        {
            return Err(format!("request {} is already open", request.0));
        }
        if input.vision().len() > self.execution.policy().limits().max_images_per_request {
            return Err("input exceeds planned image capacity".into());
        }
        let images = input
            .vision()
            .iter()
            .map(|(identity, prepared)| {
                let image = ImageRef::prepared(self.domain.id().clone(), prepared.clone())
                    .map_err(|error| error.to_string())?;
                Ok((
                    identity.clone(),
                    InputImage {
                        image,
                        features: None,
                    },
                ))
            })
            .collect::<Result<_, String>>()?;
        self.input.insert(
            request,
            RequestInput {
                input,
                images,
                resident: false,
            },
        );
        Ok(())
    }

    fn installed_input(&self, request: RequestId) -> Result<&RequestInput, String> {
        self.input
            .get(&request)
            .ok_or_else(|| format!("request {} lowers rows without installed input", request.0))
    }

    /// Rotary coordinates of the target rows `position..position + count`,
    /// including continuation rows past the admitted input.
    pub(super) fn input_coordinates(
        &self,
        request: RequestId,
        position: usize,
        count: usize,
    ) -> Result<Vec<[i32; 4]>, String> {
        Ok(self
            .installed_input(request)?
            .input
            .coordinates_at(position, count)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|[a, b, c]| [a, b, c, 0])
            .collect())
    }

    pub(super) fn input_rows(
        &self,
        request: RequestId,
        position: usize,
        tokens: &[crate::TokenId],
    ) -> Result<(Vec<[i32; 4]>, Vec<crate::ConditioningSlice>), String> {
        let coordinates = self.input_coordinates(request, position, tokens.len())?;
        let state = self.installed_input(request)?;
        let end = position
            .checked_add(tokens.len())
            .ok_or("input end overflows")?;
        if !state.input.layout().boundary(position)
            || !state.input.layout().boundary(end)
            || (position < state.input.tokens().len() && end > state.input.tokens().len())
            || tokens.iter().enumerate().any(|(index, token)| {
                state
                    .input
                    .tokens()
                    .get(position + index)
                    .is_some_and(|expected| expected != token)
            })
        {
            return Err("target rows differ from admitted prepared input".into());
        }
        let mut slices = Vec::new();
        for span in state
            .input
            .layout()
            .spans()
            .iter()
            .filter(|span| span.start < end && position < span.end)
        {
            let image = state
                .images
                .get(&span.identity)
                .ok_or("prepared input image is absent")?;
            let features = image
                .features
                .clone()
                .ok_or("prepared input image has not been encoded")?;
            let start = position.max(span.start);
            let stop = end.min(span.end);
            slices.push(crate::ConditioningSlice {
                source: FeatureSpan::new(features, start - span.start, stop - start)
                    .map_err(|error| error.to_string())?,
                destination: start - position,
            });
        }
        Ok((coordinates, slices))
    }
}
