//! Qwen token, prompt, and vision-coordinate preparation. The generic media
//! processor supplies pixels and grids; this adapter assigns numerical meaning.

use magnitude_artifacts::{
    BoundaryRule, InputLayout, InputSpan, MAX_IMAGES_PER_REQUEST, TokenId,
    media::{MAX_PREPARED_BYTES, PreparedMedia},
};
use magnitude_family_contracts::{
    InputPreparationError, ModelDefinition, ModelInputAdapter, PositionSampling, PreparedModelInput,
    PreparedVisionInput, TokenPlan, VisionPositionBlend, VisionSpatialControls,
    image_crop_identity, image_crops,
};
use std::collections::{BTreeMap, btree_map::Entry};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QwenImageTokens {
    image: TokenId,
    start: TokenId,
    end: TokenId,
}

impl QwenImageTokens {
    pub fn new(
        image: TokenId,
        start: TokenId,
        end: TokenId,
    ) -> Result<Self, InputPreparationError> {
        if [image, start, end]
            .iter()
            .any(|token| token.0 > i32::MAX as u32)
            || image == start
            || image == end
            || start == end
        {
            return Err(InputPreparationError::TokenDomain);
        }
        Ok(Self { image, start, end })
    }
}

#[derive(Clone, Debug)]
pub struct QwenInputAdapter {
    /// The family ids whose definitions this adapter prepares input for.
    families: &'static [&'static str],
    image_tokens: Option<QwenImageTokens>,
}

impl QwenInputAdapter {
    pub fn new(families: &'static [&'static str], image_tokens: Option<QwenImageTokens>) -> Self {
        Self {
            families,
            image_tokens,
        }
    }
}

impl ModelInputAdapter for QwenInputAdapter {
    fn prepare(
        &self,
        definition: &ModelDefinition,
        token_plan: TokenPlan,
        media: &[PreparedMedia],
    ) -> Result<PreparedModelInput, InputPreparationError> {
        if !self.families.contains(&definition.family.0.as_str()) {
            return Err(InputPreparationError::InputAlignment);
        }
        if !token_plan.layout().spans().is_empty() {
            return Err(InputPreparationError::InputAlignment);
        }
        let (tokens, _) = token_plan.into_parts();
        if media.is_empty() {
            if self.image_tokens.is_some_and(|markers| {
                tokens
                    .iter()
                    .any(|token| [markers.image, markers.start, markers.end].contains(token))
            }) {
                return Err(InputPreparationError::InputAlignment);
            }
            let continuation = tokens.len();
            let coordinates = (0..continuation).map(|value| [value as i32; 3]).collect();
            let layout = InputLayout::new(continuation, Vec::new())
                .map_err(|_| InputPreparationError::InputAlignment)?;
            return PreparedModelInput::from_text_coordinates(
                TokenPlan::new(tokens, layout)?,
                coordinates,
            );
        }
        let markers = self
            .image_tokens
            .ok_or(InputPreparationError::UnsupportedMedia)?;
        let vision = definition
            .vision
            .as_ref()
            .ok_or(InputPreparationError::UnsupportedMedia)?;
        let [prepared] = media else {
            return Err(InputPreparationError::UnsupportedMedia);
        };
        let (processor, images) = image_crops(definition, prepared)?;
        if images.len() > MAX_IMAGES_PER_REQUEST {
            return Err(InputPreparationError::VisionGeometry);
        }
        let merge = usize::try_from(vision.preprocessing.merge)
            .ok()
            .filter(|value| *value > 0)
            .ok_or(InputPreparationError::VisionGeometry)?;
        let PositionSampling::AlignedCorners { side } = vision
            .stem
            .positions()
            .ok_or(InputPreparationError::VisionGeometry)?
            .sampling
        else {
            return Err(InputPreparationError::VisionGeometry);
        };
        let table_side = usize::try_from(side)
            .ok()
            .filter(|value| *value > 0)
            .ok_or(InputPreparationError::VisionGeometry)?;
        let merge_area = merge
            .checked_mul(merge)
            .ok_or(InputPreparationError::VisionGeometry)?;
        let mut expanded = Vec::with_capacity(tokens.len());
        let mut source_boundaries = Vec::with_capacity(tokens.len() + 1);
        let (mut source, mut image) = (0usize, 0usize);
        while source < tokens.len() {
            let token = tokens[source];
            if token == markers.start {
                let [t, h, w] = images
                    .get(image)
                    .ok_or(InputPreparationError::InputAlignment)?
                    .grid;
                if tokens.get(source + 1) != Some(&markers.image)
                    || tokens.get(source + 2) != Some(&markers.end)
                {
                    return Err(InputPreparationError::InputAlignment);
                }
                let count = t
                    .checked_mul(h)
                    .and_then(|value| value.checked_mul(w))
                    .map(|value| value / merge_area)
                    .ok_or(InputPreparationError::VisionGeometry)?;
                source_boundaries.push(expanded.len());
                expanded.push(markers.start);
                source_boundaries.push(expanded.len());
                expanded.extend(std::iter::repeat_n(markers.image, count));
                source_boundaries.push(expanded.len());
                expanded.push(markers.end);
                source += 3;
                image += 1;
            } else if token == markers.image || token == markers.end {
                return Err(InputPreparationError::InputAlignment);
            } else {
                source_boundaries.push(expanded.len());
                expanded.push(token);
                source += 1;
            }
        }
        source_boundaries.push(expanded.len());
        if image != images.len() || expanded.len() > i32::MAX as usize {
            return Err(InputPreparationError::InputAlignment);
        }
        let mut coordinates = Vec::with_capacity(expanded.len());
        let mut vision_inputs = BTreeMap::new();
        let mut spans = Vec::with_capacity(images.len());
        let (mut cursor, mut rotary) = (0usize, 0usize);
        for crop in images {
            let [t, h, w] = crop.grid;
            let patches = t
                .checked_mul(h)
                .and_then(|value| value.checked_mul(w))
                .ok_or(InputPreparationError::VisionGeometry)?;
            let count = patches / merge_area;
            let start = expanded[cursor..]
                .iter()
                .position(|token| *token == markers.image)
                .map(|offset| cursor + offset)
                .ok_or(InputPreparationError::InputAlignment)?;
            let end = start
                .checked_add(count)
                .filter(|end| *end < expanded.len())
                .ok_or(InputPreparationError::InputAlignment)?;
            if start == 0
                || expanded[start - 1] != markers.start
                || expanded[end] != markers.end
                || expanded[start..end]
                    .iter()
                    .any(|token| *token != markers.image)
            {
                return Err(InputPreparationError::InputAlignment);
            }
            let base = rotary
                .checked_add(start - cursor)
                .ok_or(InputPreparationError::InputAlignment)?;
            let next_rotary = base
                .checked_add(t.max(h / merge).max(w / merge))
                .filter(|value| *value <= i32::MAX as usize)
                .ok_or(InputPreparationError::InputAlignment)?;
            coordinates.extend((rotary..base).map(|value| [value as i32; 3]));
            for row in 0..h / merge {
                for col in 0..w / merge {
                    coordinates.push([base as i32, (base + row) as i32, (base + col) as i32]);
                }
            }
            let identity = image_crop_identity(&processor, &crop);
            spans.push(InputSpan {
                start,
                end,
                identity: identity.clone(),
                boundaries: BoundaryRule::Causal,
                language_history: false,
            });
            if let Entry::Vacant(entry) = vision_inputs.entry(identity) {
                entry.insert(PreparedVisionInput::new(
                    definition,
                    crop.grid,
                    crop.pixels,
                    spatial_controls(crop.grid, merge, table_side)?,
                )?);
            }
            rotary = next_rotary;
            cursor = end;
        }
        if expanded[cursor..].contains(&markers.image) {
            return Err(InputPreparationError::InputAlignment);
        }
        let continuation = rotary
            .checked_add(expanded.len() - cursor)
            .filter(|value| *value <= i32::MAX as usize)
            .ok_or(InputPreparationError::InputAlignment)?;
        coordinates.extend((rotary..continuation).map(|value| [value as i32; 3]));
        let layout = InputLayout::new(expanded.len(), spans)
            .map_err(|_| InputPreparationError::InputAlignment)?;
        PreparedModelInput::new(
            definition,
            TokenPlan::new(expanded, layout)?,
            coordinates,
            continuation,
            source_boundaries,
            vision_inputs,
        )
    }
}

/// Qwen merge-group patch order and align-corners position-table interpolation.
pub fn spatial_controls(
    grid: [usize; 3],
    merge: usize,
    table_side: usize,
) -> Result<VisionSpatialControls, InputPreparationError> {
    let [t, h, w] = grid;
    if t != 1
        || merge == 0
        || table_side == 0
        || h < merge
        || w < merge
        || h % merge != 0
        || w % merge != 0
    {
        return Err(InputPreparationError::VisionGeometry);
    }
    table_side
        .checked_mul(table_side)
        .filter(|rows| *rows <= i32::MAX as usize)
        .ok_or(InputPreparationError::VisionGeometry)?;
    let rows = h
        .checked_mul(w)
        .filter(|rows| *rows <= MAX_PREPARED_BYTES / 40)
        .ok_or(InputPreparationError::VisionGeometry)?;
    let mut coordinates = Vec::with_capacity(rows);
    let mut indices: [Vec<i32>; 4] = std::array::from_fn(|_| Vec::with_capacity(rows));
    let mut coefficients: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(rows));
    let axis = |position: usize, extent: usize| {
        let value = if extent == 1 {
            0.0
        } else {
            (position as f64 * ((table_side - 1) as f64 / (extent - 1) as f64)) as f32
        };
        let low = (value as usize).min(table_side - 1);
        (
            low,
            (low + 1).min(table_side - 1),
            (value as f64 - low as f64).clamp(0.0, 1.0),
        )
    };
    for block_row in 0..h / merge {
        for block_col in 0..w / merge {
            for inner_row in 0..merge {
                for inner_col in 0..merge {
                    let row = block_row * merge + inner_row;
                    let col = block_col * merge + inner_col;
                    coordinates.push([row as i32, col as i32]);
                    let (y0, y1, dy) = axis(row, h);
                    let (x0, x1, dx) = axis(col, w);
                    for (index, (y, x, coefficient)) in [
                        (y0, x0, (1.0 - dy) * (1.0 - dx)),
                        (y0, x1, (1.0 - dy) * dx),
                        (y1, x0, dy * (1.0 - dx)),
                        (y1, x1, dy * dx),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        indices[index].push((y * table_side + x) as i32);
                        coefficients[index].push(coefficient as f32);
                    }
                }
            }
        }
    }
    VisionSpatialControls::new(
        (0..rows).collect(),
        coordinates,
        Some(VisionPositionBlend {
            indices,
            coefficients,
        }),
        Vec::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::spatial_controls;

    #[test]
    fn spatial_controls_clamps_boundary_indices_within_table() {
        for (grid, merge, table_side) in [
            ([1, 2, 2], 2, 1),
            ([1, 4, 6], 2, 2),
            ([1, 50, 98], 2, 48),
        ] {
            let controls = spatial_controls(grid, merge, table_side).unwrap();
            let blend = controls.position_blend().expect("aligned-corners blend");
            let limit = (table_side * table_side) as i32;
            let rows = grid[1] * grid[2];
            for corner in 0..4 {
                assert_eq!(blend.indices[corner].len(), rows);
                assert!(blend.indices[corner].iter().all(|&index| (0..limit).contains(&index)));
            }
            // The last raster patch in the last merge cell is (h - 1, w - 1) and must
            // land on the bottom-right table entry without exceeding `limit - 1`.
            let last = rows - 1;
            let expected_corner = limit - 1;
            for corner in 0..4 {
                assert_eq!(blend.indices[corner][last], expected_corner);
            }
            let sum: f32 = (0..4).map(|corner| blend.coefficients[corner][last]).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
