//! Closed numerical input after family interpretation.

use crate::ModelDefinition;
use magnitude_artifacts::{
    media::{DType, PreparedMedia, PreparedTensor},
    InputLayout, TokenId,
};
use std::{error, fmt};

/// Tokenized host input before a model family assigns numerical coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenPlan {
    tokens: Vec<TokenId>,
    layout: InputLayout,
}

impl TokenPlan {
    pub fn new(tokens: Vec<TokenId>, layout: InputLayout) -> Result<Self, InputPreparationError> {
        if tokens.iter().any(|token| token.0 > i32::MAX as u32) {
            return Err(InputPreparationError::TokenDomain);
        }
        if layout.count() != tokens.len() {
            return Err(InputPreparationError::InputAlignment);
        }
        Ok(Self { tokens, layout })
    }

    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }

    pub fn layout(&self) -> &InputLayout {
        &self.layout
    }

    pub fn into_parts(self) -> (Vec<TokenId>, InputLayout) {
        (self.tokens, self.layout)
    }
}

/// Patch order and spatial controls supplied by the model family, not derived
/// by an encoder stage. Indices address the model's position table.
#[derive(Clone, Debug)]
pub struct VisionSpatialControls {
    patch_order: Vec<usize>,
    attention_coordinates: Vec<[i32; 2]>,
    interpolation_indices: [Vec<i32>; 4],
    interpolation_coefficients: [Vec<f32>; 4],
}

impl VisionSpatialControls {
    pub fn new(
        patch_order: Vec<usize>,
        attention_coordinates: Vec<[i32; 2]>,
        interpolation_indices: [Vec<i32>; 4],
        interpolation_coefficients: [Vec<f32>; 4],
    ) -> Result<Self, InputPreparationError> {
        let rows = patch_order.len();
        let mut seen = vec![false; rows];
        for &index in &patch_order {
            let Some(occupied) = seen.get_mut(index) else {
                return Err(InputPreparationError::VisionGeometry);
            };
            if *occupied {
                return Err(InputPreparationError::VisionGeometry);
            }
            *occupied = true;
        }
        if rows == 0
            || attention_coordinates.len() != rows
            || attention_coordinates
                .iter()
                .any(|coordinate| coordinate.iter().any(|value| *value < 0))
            || interpolation_indices
                .iter()
                .any(|indices| indices.len() != rows || indices.iter().any(|index| *index < 0))
            || interpolation_coefficients.iter().any(|coefficients| {
                coefficients.len() != rows
                    || coefficients
                        .iter()
                        .any(|coefficient| !coefficient.is_finite() || *coefficient < 0.0)
            })
        {
            return Err(InputPreparationError::VisionGeometry);
        }
        for row in 0..rows {
            let sum = interpolation_coefficients
                .iter()
                .map(|coefficients| coefficients[row])
                .sum::<f32>();
            if (sum - 1.0).abs() > 1e-5 {
                return Err(InputPreparationError::VisionGeometry);
            }
        }
        Ok(Self {
            patch_order,
            attention_coordinates,
            interpolation_indices,
            interpolation_coefficients,
        })
    }

    pub fn patch_order(&self) -> &[usize] {
        &self.patch_order
    }

    pub fn attention_coordinates(&self) -> &[[i32; 2]] {
        &self.attention_coordinates
    }

    pub fn interpolation_indices(&self) -> &[Vec<i32>; 4] {
        &self.interpolation_indices
    }

    pub fn interpolation_coefficients(&self) -> &[Vec<f32>; 4] {
        &self.interpolation_coefficients
    }

    pub fn rows(&self) -> usize {
        self.patch_order.len()
    }
}

#[derive(Clone, Debug)]
pub struct PreparedVisionInput {
    identity: String,
    grid: [usize; 3],
    pixels: PreparedTensor,
    spatial: VisionSpatialControls,
}

impl PreparedVisionInput {
    pub fn new(
        definition: &ModelDefinition,
        identity: String,
        grid: [usize; 3],
        pixels: PreparedTensor,
        spatial: VisionSpatialControls,
    ) -> Result<Self, InputPreparationError> {
        let geometry = &definition
            .vision
            .as_ref()
            .ok_or(InputPreparationError::UnsupportedMedia)?
            .geometry;
        let patch_width = [
            geometry.channels,
            geometry.temporal_patch,
            geometry.patch,
            geometry.patch,
        ]
        .into_iter()
        .try_fold(1u64, |product, extent| product.checked_mul(extent))
        .and_then(|width| usize::try_from(width).ok())
        .ok_or(InputPreparationError::VisionGeometry)?;
        let table_rows = geometry
            .table_side
            .checked_mul(geometry.table_side)
            .filter(|rows| *rows <= i32::MAX as u64)
            .ok_or(InputPreparationError::VisionGeometry)?;
        let merge = usize::try_from(geometry.merge)
            .ok()
            .filter(|merge| *merge > 0)
            .ok_or(InputPreparationError::VisionGeometry)?;
        let rows = grid
            .into_iter()
            .try_fold(1usize, |product, extent| product.checked_mul(extent))
            .ok_or(InputPreparationError::VisionGeometry)?;
        if identity.is_empty()
            || grid.contains(&0)
            || pixels.dtype() != DType::F32
            || pixels.shape().len() != 2
            || pixels.shape()[0] != rows
            || pixels.shape()[1] != patch_width
            || spatial.rows() != rows
            || grid[1] % merge != 0
            || grid[2] % merge != 0
            || spatial.attention_coordinates().iter().any(|coordinate| {
                coordinate[0] as usize >= grid[1] || coordinate[1] as usize >= grid[2]
            })
            || spatial
                .interpolation_indices()
                .iter()
                .any(|indices| indices.iter().any(|index| *index as u64 >= table_rows))
        {
            return Err(InputPreparationError::VisionGeometry);
        }
        Ok(Self {
            identity,
            grid,
            pixels,
            spatial,
        })
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn grid(&self) -> [usize; 3] {
        self.grid
    }

    pub fn pixels(&self) -> &PreparedTensor {
        &self.pixels
    }

    pub fn spatial(&self) -> &VisionSpatialControls {
        &self.spatial
    }
}

/// The only numerical input value accepted after model-family preparation.
#[derive(Clone, Debug)]
pub struct PreparedModelInput {
    tokens: Vec<TokenId>,
    layout: InputLayout,
    coordinates: Vec<[i32; 3]>,
    continuation: usize,
    vision: Vec<PreparedVisionInput>,
}

impl PreparedModelInput {
    /// Close a text-only token plan whose numerical coordinates have already
    /// been supplied by the model family. No coordinate semantics are derived
    /// at this boundary.
    pub fn from_text_coordinates(
        plan: TokenPlan,
        coordinates: Vec<[i32; 3]>,
    ) -> Result<Self, InputPreparationError> {
        let (tokens, layout) = plan.into_parts();
        if !layout.spans().is_empty()
            || coordinates.len() != tokens.len()
            || tokens.len() > i32::MAX as usize
            || coordinates
                .iter()
                .any(|axes| axes.iter().any(|value| *value < 0))
        {
            return Err(InputPreparationError::InputAlignment);
        }
        let continuation = tokens.len();
        Ok(Self {
            tokens,
            layout,
            coordinates,
            continuation,
            vision: Vec::new(),
        })
    }

    pub fn new(
        definition: &ModelDefinition,
        tokens: TokenPlan,
        coordinates: Vec<[i32; 3]>,
        continuation: usize,
        vision: Vec<PreparedVisionInput>,
    ) -> Result<Self, InputPreparationError> {
        let (tokens, layout) = tokens.into_parts();
        if coordinates.len() != tokens.len()
            || continuation > i32::MAX as usize
            || coordinates
                .iter()
                .any(|axes| axes.iter().any(|coordinate| *coordinate < 0))
            || layout.spans().len() != vision.len()
            || layout
                .spans()
                .iter()
                .zip(&vision)
                .any(|(span, image)| span.identity != image.identity)
        {
            return Err(InputPreparationError::InputAlignment);
        }
        if let Some(vision_geometry) = definition.vision.as_ref().map(|v| &v.geometry) {
            let merge = usize::try_from(vision_geometry.merge)
                .ok()
                .filter(|merge| *merge > 0)
                .ok_or(InputPreparationError::VisionGeometry)?;
            let merge_area = merge
                .checked_mul(merge)
                .ok_or(InputPreparationError::VisionGeometry)?;
            if layout.spans().iter().zip(&vision).any(|(span, image)| {
                let [t, h, w] = image.grid();
                t.checked_mul(h)
                    .and_then(|patches| patches.checked_mul(w))
                    .map(|patches| patches / merge_area != span.end - span.start)
                    .unwrap_or(true)
            }) {
                return Err(InputPreparationError::InputAlignment);
            }
        } else if !vision.is_empty() {
            return Err(InputPreparationError::UnsupportedMedia);
        }
        Ok(Self {
            tokens,
            layout,
            coordinates,
            continuation,
            vision,
        })
    }

    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }

    pub fn layout(&self) -> &InputLayout {
        &self.layout
    }

    pub fn coordinates(&self) -> &[[i32; 3]] {
        &self.coordinates
    }

    pub fn continuation(&self) -> usize {
        self.continuation
    }

    pub fn vision(&self) -> &[PreparedVisionInput] {
        &self.vision
    }

    /// Resolve prompt and continuation coordinates from the closed family input.
    pub fn coordinates_at(
        &self,
        position: usize,
        count: usize,
    ) -> Result<Vec<[i32; 3]>, InputPreparationError> {
        let end = position
            .checked_add(count)
            .filter(|end| *end <= i32::MAX as usize)
            .ok_or(InputPreparationError::InputAlignment)?;
        (position..end)
            .map(|index| {
                if let Some(coordinates) = self.coordinates.get(index) {
                    Ok(*coordinates)
                } else {
                    let continuation = self
                        .continuation
                        .checked_add(index - self.tokens.len())
                        .and_then(|value| i32::try_from(value).ok())
                        .ok_or(InputPreparationError::InputAlignment)?;
                    Ok([continuation; 3])
                }
            })
            .collect()
    }
}

pub trait ModelInputAdapter {
    fn prepare(
        &self,
        definition: &ModelDefinition,
        tokens: TokenPlan,
        media: &[PreparedMedia],
    ) -> Result<PreparedModelInput, InputPreparationError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputPreparationError {
    TokenDomain,
    InputAlignment,
    VisionGeometry,
    UnsupportedMedia,
}

impl fmt::Display for InputPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenDomain => formatter.write_str("tokens exceed the model input domain"),
            Self::InputAlignment => formatter.write_str("model input fields are misaligned"),
            Self::VisionGeometry => formatter.write_str("model vision controls are invalid"),
            Self::UnsupportedMedia => formatter.write_str("model does not support this media"),
        }
    }
}

impl error::Error for InputPreparationError {}
