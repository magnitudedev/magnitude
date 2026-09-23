//! input lifecycle for the executor domain.

use super::*;

impl<F: ProgramFamily> ExecutorDomain<F> {
    pub fn install_input(
        &mut self,
        request: RequestId,
        input: PreparedModelInput,
    ) -> Result<Vec<Operation>, String> {
        let state = self.target.get(&request).ok_or("request is not idle")?;
        if state.position() != 0 || self.input.contains_key(&request) {
            return Err("input must be installed once before target progress".into());
        }
        if input.vision().len() > self.execution.policy().limits().max_images_per_request {
            return Err("input exceeds planned image capacity".into());
        }
        let mut images = BTreeMap::new();
        for prepared in input.vision() {
            let image = ImageRef::prepared(self.domain.id().clone(), prepared.clone())
                .map_err(|error| error.to_string())?;
            if images
                .insert(
                    prepared.identity().to_owned(),
                    InputImage {
                        image,
                        features: None,
                    },
                )
                .is_some()
            {
                return Err("prepared input repeats a vision identity".into());
            }
        }
        if images.len() != input.layout().spans().len()
            || input
                .layout()
                .spans()
                .iter()
                .any(|span| !images.contains_key(&span.identity))
        {
            return Err("prepared vision inputs differ from conditioned layout".into());
        }
        let operations = images
            .values()
            .map(|slot| Operation::Encode {
                request,
                image: slot.image.clone(),
            })
            .collect();
        self.input.insert(request, RequestInput { input, images });
        Ok(operations)
    }

    pub fn install_retained_input(
        &mut self,
        request: RequestId,
        input: PreparedModelInput,
    ) -> Result<Vec<Operation>, String> {
        let position = self
            .target
            .get(&request)
            .ok_or("request is not idle")?
            .position();
        let old = self
            .input
            .get(&request)
            .ok_or("request has no retained input")?;
        if position == 0
            || !input.layout().boundary(position)
            || position > input.tokens().len()
            || position > old.input.tokens().len()
            || input.tokens()[..position] != old.input.tokens()[..position]
        {
            return Err("retained input differs before accepted position".into());
        }
        if input.vision().len() > self.execution.policy().limits().max_images_per_request {
            return Err("retained input exceeds planned image capacity".into());
        }
        let mut images = BTreeMap::new();
        for prepared in input.vision() {
            let image = ImageRef::prepared(self.domain.id().clone(), prepared.clone())
                .map_err(|error| error.to_string())?;
            images.insert(
                prepared.identity().to_owned(),
                InputImage {
                    image,
                    features: None,
                },
            );
        }
        for span in input
            .layout()
            .spans()
            .iter()
            .filter(|span| span.start < position)
        {
            let old_slot = old
                .images
                .get(&span.identity)
                .ok_or("retained input lacks previously accepted vision feature")?;
            images.insert(
                span.identity.clone(),
                InputImage {
                    image: old_slot.image.clone(),
                    features: old_slot.features.clone(),
                },
            );
        }
        let operations = images
            .values()
            .filter(|slot| slot.features.is_none())
            .map(|slot| Operation::Encode {
                request,
                image: slot.image.clone(),
            })
            .collect();
        self.input.insert(request, RequestInput { input, images });
        Ok(operations)
    }

    pub(super) fn input_rows(
        &self,
        request: RequestId,
        position: usize,
        tokens: &[crate::TokenId],
    ) -> Result<(Vec<[i32; 4]>, Vec<crate::ConditioningSlice>), String> {
        let Some(state) = self.input.get(&request) else {
            let end = position
                .checked_add(tokens.len())
                .ok_or("target input end overflows")?;
            let coordinates = (position..end)
                .map(|position| {
                    let position =
                        i32::try_from(position).map_err(|_| "target position exceeds i32")?;
                    let mut values = [0; 4];
                    match self.definition.inputs.text_coordinates {
                        TextCoordinateSemantics::ReplicatedPosition => values
                            [..usize::from(self.definition.inputs.coordinate_axes)]
                            .fill(position),
                    }
                    Ok(values)
                })
                .collect::<Result<Vec<_>, String>>()?;
            return Ok((coordinates, Vec::new()));
        };
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
        let coordinates = state
            .input
            .coordinates_at(position, tokens.len())
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|[a, b, c]| [a, b, c, 0])
            .collect();
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
