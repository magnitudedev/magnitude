//! Qwen token, prompt, and vision-coordinate preparation. The generic media
//! processor supplies pixels and grids; this adapter assigns numerical meaning.

use magnitude_artifacts::{
    media::{DType, PreparedMedia, PreparedTensor, MAX_PREPARED_BYTES},
    BoundaryRule, ImageProcessor, InputLayout, InputSpan, TokenId,
};
use magnitude_model_contracts::{
    InputPreparationError, ModelDefinition, ModelInputAdapter, PreparedModelInput,
    PreparedVisionInput, TokenPlan, VisionSpatialControls,
};
use sha2::{Digest, Sha256};

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
    image_tokens: Option<QwenImageTokens>,
}

impl QwenInputAdapter {
    pub fn new(image_tokens: Option<QwenImageTokens>) -> Self {
        Self { image_tokens }
    }
}

impl ModelInputAdapter for QwenInputAdapter {
    fn prepare(
        &self,
        definition: &ModelDefinition,
        token_plan: TokenPlan,
        media: &[PreparedMedia],
    ) -> Result<PreparedModelInput, InputPreparationError> {
        if !matches!(definition.family.0.as_str(), "qwen35" | "qwen35moe") {
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
        let processor = ImageProcessor::new(
            vision
                .image_processor_config()
                .map_err(|_| InputPreparationError::VisionGeometry)?,
        )
        .map_err(|_| InputPreparationError::VisionGeometry)?;
        if prepared.processor() != processor.identity() || prepared.tensors().len() != 2 {
            return Err(InputPreparationError::UnsupportedMedia);
        }
        let width = [
            vision.geometry.channels,
            vision.geometry.temporal_patch,
            vision.geometry.patch,
            vision.geometry.patch,
        ]
        .into_iter()
        .try_fold(1u64, |value, extent| value.checked_mul(extent))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(InputPreparationError::VisionGeometry)?;
        let merge = usize::try_from(vision.geometry.merge)
            .ok()
            .filter(|value| *value > 0)
            .ok_or(InputPreparationError::VisionGeometry)?;
        let table_side = usize::try_from(vision.geometry.table_side)
            .ok()
            .filter(|value| *value > 0)
            .ok_or(InputPreparationError::VisionGeometry)?;
        let pixels = prepared
            .tensors()
            .iter()
            .find(|tensor| tensor.name() == "pixel_values")
            .ok_or(InputPreparationError::VisionGeometry)?;
        let grids = prepared
            .tensors()
            .iter()
            .find(|tensor| tensor.name() == "image_grid_thw")
            .ok_or(InputPreparationError::VisionGeometry)?;
        if pixels.dtype() != DType::F32
            || pixels.shape().len() != 2
            || pixels.shape()[1] != width
            || grids.dtype() != DType::I64
            || grids.shape().len() != 2
            || grids.shape()[1] != 3
            || grids.shape()[0] > 16
            || pixels
                .data()
                .chunks_exact(4)
                .any(|bytes| !f32::from_le_bytes(bytes.try_into().expect("f32 chunk")).is_finite())
        {
            return Err(InputPreparationError::VisionGeometry);
        }
        let sizes = grids
            .data()
            .chunks_exact(24)
            .map(|row| {
                let mut grid = [0usize; 3];
                for (index, bytes) in row.chunks_exact(8).enumerate() {
                    grid[index] =
                        usize::try_from(i64::from_le_bytes(bytes.try_into().expect("i64 chunk")))
                            .map_err(|_| InputPreparationError::VisionGeometry)?;
                }
                let [t, h, w] = grid;
                if t != 1 || h < merge || w < merge || h % merge != 0 || w % merge != 0 {
                    return Err(InputPreparationError::VisionGeometry);
                }
                Ok(grid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let total = sizes
            .iter()
            .try_fold(0usize, |count, &[t, h, w]| {
                t.checked_mul(h)
                    .and_then(|rows| rows.checked_mul(w))
                    .and_then(|rows| count.checked_add(rows))
            })
            .ok_or(InputPreparationError::VisionGeometry)?;
        if total != pixels.shape()[0] {
            return Err(InputPreparationError::VisionGeometry);
        }
        let merge_area = merge
            .checked_mul(merge)
            .ok_or(InputPreparationError::VisionGeometry)?;
        let mut expanded = Vec::with_capacity(tokens.len());
        let (mut source, mut image) = (0usize, 0usize);
        while source < tokens.len() {
            let token = tokens[source];
            if token == markers.start {
                let [t, h, w] = *sizes
                    .get(image)
                    .ok_or(InputPreparationError::InputAlignment)?;
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
                expanded.push(markers.start);
                expanded.extend(std::iter::repeat_n(markers.image, count));
                expanded.push(markers.end);
                source += 3;
                image += 1;
            } else if token == markers.image || token == markers.end {
                return Err(InputPreparationError::InputAlignment);
            } else {
                expanded.push(token);
                source += 1;
            }
        }
        if image != sizes.len() || expanded.len() > i32::MAX as usize {
            return Err(InputPreparationError::InputAlignment);
        }
        let mut coordinates = Vec::with_capacity(expanded.len());
        let mut vision_inputs = Vec::with_capacity(sizes.len());
        let mut spans = Vec::with_capacity(sizes.len());
        let (mut cursor, mut pixel_start, mut rotary) = (0usize, 0usize, 0usize);
        for grid in sizes {
            let [t, h, w] = grid;
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
            let byte_start = pixel_start
                .checked_mul(width)
                .and_then(|value| value.checked_mul(4))
                .ok_or(InputPreparationError::VisionGeometry)?;
            let byte_end = pixel_start
                .checked_add(patches)
                .and_then(|value| value.checked_mul(width))
                .and_then(|value| value.checked_mul(4))
                .ok_or(InputPreparationError::VisionGeometry)?;
            let values = PreparedTensor::new(
                "pixel_values".into(),
                DType::F32,
                vec![patches, width],
                pixels
                    .data()
                    .get(byte_start..byte_end)
                    .ok_or(InputPreparationError::VisionGeometry)?
                    .to_vec(),
            )
            .map_err(|_| InputPreparationError::VisionGeometry)?;
            let mut digest = Sha256::new();
            digest.update(processor.identity().as_bytes());
            for value in grid {
                digest.update((value as i64).to_le_bytes());
            }
            digest.update(values.data());
            let identity: String = digest
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            spans.push(InputSpan {
                start,
                end,
                identity: identity.clone(),
                boundaries: BoundaryRule::Causal,
                language_history: false,
            });
            vision_inputs.push(PreparedVisionInput::new(
                definition,
                identity,
                grid,
                values,
                spatial_controls(grid, merge, table_side)?,
            )?);
            rotary = next_rotary;
            cursor = end;
            pixel_start += patches;
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
        let low = value as usize;
        (
            low,
            (low + 1).min(table_side - 1),
            value as f64 - low as f64,
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
    VisionSpatialControls::new((0..rows).collect(), coordinates, indices, coefficients)
}
