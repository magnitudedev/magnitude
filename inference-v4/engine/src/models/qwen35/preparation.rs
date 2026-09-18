//! Qwen image tensor interpretation. This validates processor output and prompt
//! spans without creating device resources or running an image encoder.
use crate::inputs::{
    media::{DType, PreparedMedia, PreparedTensor},
    BoundaryRule, InputLayout, InputSpan, TokenId,
};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct ImageGeometry {
    pub channels: usize,
    pub temporal_patch: usize,
    pub patch: usize,
    pub merge: usize,
    pub image_token: TokenId,
    pub start_token: TokenId,
    pub end_token: TokenId,
}
impl ImageGeometry {
    pub fn patch_width(&self) -> Result<usize, String> {
        if [self.channels, self.temporal_patch, self.patch, self.merge].contains(&0)
            || [self.image_token, self.start_token, self.end_token]
                .iter()
                .any(|id| id.0 > i32::MAX as u32)
            || self.image_token == self.start_token
            || self.image_token == self.end_token
            || self.start_token == self.end_token
        {
            return Err("invalid Qwen image geometry".into());
        }
        self.channels
            .checked_mul(self.temporal_patch)
            .and_then(|n| n.checked_mul(self.patch))
            .and_then(|n| n.checked_mul(self.patch))
            .ok_or("image patch width overflows".into())
    }
}
#[derive(Clone, Debug)]
pub struct InputPlan {
    tokens: Vec<TokenId>,
    layout: InputLayout,
    coordinates: Vec<[i32; 3]>,
    continuation: usize,
}
impl InputPlan {
    pub fn text(tokens: Vec<TokenId>) -> Result<Self, String> {
        if tokens.iter().any(|id| id.0 > i32::MAX as u32) {
            return Err("input token exceeds int32 domain".into());
        }
        let layout = InputLayout::new(tokens.len(), vec![])?;
        Ok(Self {
            continuation: tokens.len(),
            tokens,
            layout,
            coordinates: vec![],
        })
    }
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }
    pub fn layout(&self) -> &InputLayout {
        &self.layout
    }
    pub fn continuation(&self) -> usize {
        self.continuation
    }
    pub fn rotary(&self, position: usize, count: usize) -> Result<Vec<[i32; 3]>, String> {
        let end = position
            .checked_add(count)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or("rotary range exceeds int32 domain")?;
        (position..end)
            .map(|index| {
                if index < self.tokens.len() {
                    Ok(self
                        .coordinates
                        .get(index)
                        .copied()
                        .unwrap_or([index as i32; 3]))
                } else {
                    let value = self
                        .continuation
                        .checked_add(index - self.tokens.len())
                        .and_then(|n| i32::try_from(n).ok())
                        .ok_or("rotary continuation exceeds int32 domain")?;
                    Ok([value; 3])
                }
            })
            .collect()
    }
}
#[derive(Clone, Debug)]
pub struct ImagePatches {
    pub identity: String,
    pub grid: [usize; 3],
    pub pixels: PreparedTensor,
}
#[derive(Clone, Debug)]
pub struct PreparedInput {
    pub plan: InputPlan,
    pub images: Vec<ImagePatches>,
}
pub fn interpret(
    tokens: Vec<TokenId>,
    media: &PreparedMedia,
    processor: &str,
    geometry: &ImageGeometry,
) -> Result<PreparedInput, String> {
    let width = geometry.patch_width()?;
    let mut plan = InputPlan::text(tokens)?;
    if media.processor() != processor {
        return Err("image processor differs from the bound artifact".into());
    }
    if media.tensors().len() != 2 {
        return Err("Qwen prepared tensors differ from the image contract".into());
    }
    let pixels = media
        .tensors()
        .iter()
        .find(|t| t.name() == "pixel_values")
        .ok_or("missing image pixels")?;
    let grids = media
        .tensors()
        .iter()
        .find(|t| t.name() == "image_grid_thw")
        .ok_or("missing image grids")?;
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
            .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
    {
        return Err("invalid Qwen image tensor geometry or values".into());
    }
    let merge = geometry.merge;
    let merge_area = merge
        .checked_mul(merge)
        .ok_or("image merge area overflows")?;
    let sizes = grids
        .data()
        .chunks_exact(24)
        .map(|row| {
            let mut grid = [0usize; 3];
            for (index, bytes) in row.chunks_exact(8).enumerate() {
                grid[index] = usize::try_from(i64::from_le_bytes(bytes.try_into().unwrap()))
                    .map_err(|_| "negative image grid")?;
            }
            let [t, h, w] = grid;
            if t != 1 || h < merge || w < merge || h % merge != 0 || w % merge != 0 {
                return Err("image grids require one frame and merge-aligned dimensions");
            }
            Ok(grid)
        })
        .collect::<Result<Vec<_>, &str>>()?;
    let total = sizes
        .iter()
        .try_fold(0usize, |n, &[_, h, w]| {
            h.checked_mul(w).and_then(|rows| n.checked_add(rows))
        })
        .ok_or("image grid extent overflows")?;
    if total != pixels.shape()[0] {
        return Err("image grids must cover pixel patches exactly".into());
    }
    let mut images = Vec::new();
    let mut spans = Vec::new();
    let (mut cursor, mut pixel_start, mut rotary) = (0usize, 0usize, 0usize);
    for grid in sizes {
        let [t, h, w] = grid;
        let patches = h * w; // checked by the aggregate extent above
        let count = patches / merge_area;
        let start = plan.tokens[cursor..]
            .iter()
            .position(|&token| token == geometry.image_token)
            .map(|i| cursor + i)
            .ok_or("image has no corresponding prompt span")?;
        let end = start
            .checked_add(count)
            .filter(|&end| end < plan.tokens.len())
            .ok_or("image prompt span exceeds input")?;
        if start == 0
            || plan.tokens[start - 1] != geometry.start_token
            || plan.tokens[end] != geometry.end_token
            || plan.tokens[start..end]
                .iter()
                .any(|&token| token != geometry.image_token)
        {
            return Err("image prompt span differs from feature geometry".into());
        }
        let base = rotary
            .checked_add(start - cursor)
            .ok_or("rotary extent overflows")?;
        let continuation = base
            .checked_add(t.max(h / merge).max(w / merge))
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or("image rotary extent exceeds int32 domain")?;
        plan.coordinates
            .extend((rotary..base).map(|p| [p as i32; 3]));
        for row in 0..h / merge {
            for col in 0..w / merge {
                plan.coordinates
                    .push([base as i32, (base + row) as i32, (base + col) as i32]);
            }
        }
        let byte_start = pixel_start * width * 4;
        let byte_end = (pixel_start + patches) * width * 4;
        let values = PreparedTensor::new(
            "pixel_values".into(),
            DType::F32,
            vec![patches, width],
            pixels.data()[byte_start..byte_end].to_vec(),
        )?;
        let mut digest = Sha256::new();
        digest.update(processor.as_bytes());
        for &value in &grid {
            digest.update((value as i64).to_le_bytes());
        }
        digest.update(values.data());
        let identity: String = digest
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        spans.push(InputSpan {
            start,
            end,
            identity: identity.clone(),
            boundaries: BoundaryRule::Causal,
            language_history: false,
        });
        images.push(ImagePatches {
            identity,
            grid,
            pixels: values,
        });
        rotary = continuation;
        cursor = end;
        pixel_start += patches;
    }
    if plan.tokens[cursor..].contains(&geometry.image_token) {
        return Err("image prompt span has no source pixels".into());
    }
    plan.continuation = rotary
        .checked_add(plan.tokens.len() - cursor)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or("rotary continuation exceeds int32 domain")?;
    plan.coordinates
        .extend((rotary..plan.continuation).map(|p| [p as i32; 3]));
    plan.layout = InputLayout::new(plan.tokens.len(), spans)?;
    Ok(PreparedInput { plan, images })
}

/// Geometry-only encoder controls in merge-group patch order. Position-table
/// interpolation uses align_corners, matching the Qwen processor/model contract.
pub struct SpatialControls {
    pub coordinates: Vec<[i32; 2]>,
    pub indices: [Vec<i32>; 4],
    pub coefficients: [Vec<f32>; 4],
}
pub fn spatial_controls(
    grid: [usize; 3],
    merge: usize,
    table_side: usize,
) -> Result<SpatialControls, String> {
    let [t, h, w] = grid;
    if t != 1
        || merge == 0
        || table_side == 0
        || h < merge
        || w < merge
        || h % merge != 0
        || w % merge != 0
    {
        return Err("invalid image geometry for spatial controls".into());
    }
    table_side
        .checked_mul(table_side)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or("position table exceeds int32 addressability")?;
    let rows = h
        .checked_mul(w)
        .filter(|&rows| rows <= crate::inputs::media::MAX_PREPARED_BYTES / 40)
        .ok_or("spatial controls exceed prepared byte capacity")?;
    let mut result = SpatialControls {
        coordinates: Vec::with_capacity(rows),
        indices: std::array::from_fn(|_| Vec::with_capacity(rows)),
        coefficients: std::array::from_fn(|_| Vec::with_capacity(rows)),
    };
    let axis = |position: usize, extent: usize| {
        // np.linspace(..., dtype=float32) rounds before interpolation weights
        // are formed. Its endpoint is exact, including the singleton case.
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
                    result.coordinates.push([row as i32, col as i32]);
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
                        result.indices[index].push((y * table_side + x) as i32);
                        result.coefficients[index].push(coefficient as f32);
                    }
                }
            }
        }
    }
    Ok(result)
}
