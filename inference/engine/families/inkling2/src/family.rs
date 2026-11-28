//! Inkling 2 as a registered model family.

use crate::{ARCHITECTURE, ARCHITECTURE_LOOP, inspect_components};
use magnitude_artifacts::{BoundaryRule, PackageIdentity, gguf::Directory};
use magnitude_family_common::TextInput;
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, ImageMarkers, InputPreparationError, MarkerTokens,
    ModelDefinition, ModelFamily, PositionSampling, SequentialImageInput, VisionPositionBlend,
    VisionSpatialControls,
};

/// Image boundary markers in Inkling 2's chat template:
/// `<|content_image|><|unused_200054|><|end_message|>`.
/// `<|end_message|>` also closes ordinary text messages, so it is registered
/// with `ImageMarkers::new_with_shared_end`.
const IMAGE_MARKERS: [&str; 3] = ["<|content_image|>", "<|unused_200054|>", "<|end_message|>"];
const IMAGE_PLACEHOLDER: &str = "<|content_image|><|unused_200054|><|end_message|>";

/// Bilinear interpolation (`align_corners=False`, edge-clamped, matching
/// `F.interpolate(mode="bilinear", align_corners=False)` and
/// `GGML_SCALE_MODE_BILINEAR`) of one axis of a `side`-element position grid.
fn bilinear_axis(index: usize, size: usize, side: usize) -> (usize, usize, f32) {
    let source = ((index as f32 + 0.5) * side as f32 / size as f32 - 0.5)
        .clamp(0.0, (side - 1) as f32);
    let low = (source.floor() as usize).min(side - 1);
    let high = (low + 1).min(side - 1);
    let weight_high = (source - low as f32).clamp(0.0, 1.0);
    (low, high, weight_high)
}

/// Spatial controls for Inkling 2's ViT (`inkling2v`):
/// - Patches stay in cell-major order (matching `build_patch_merge_permute`).
/// - 2-D axial RoPE coordinates are 1-indexed `[col + 1, row + 1]` (first half
///   of each head rotates by width/column, second half by height/row; the CLS
///   register row gets `[0, 0]`).
/// - Learned position embeddings are bilinearly sampled (`PixelCenters`,
///   `align_corners=False`, edge-clamped).
fn inkling2_spatial_controls(
    definition: &ModelDefinition,
    grid: [usize; 3],
) -> Result<VisionSpatialControls, InputPreparationError> {
    let vision = definition
        .vision
        .as_ref()
        .ok_or(InputPreparationError::UnsupportedMedia)?;
    let PositionSampling::PixelCenters { side } = vision
        .stem
        .positions()
        .ok_or(InputPreparationError::VisionGeometry)?
        .sampling
    else {
        return Err(InputPreparationError::VisionGeometry);
    };
    let side = usize::try_from(side)
        .ok()
        .filter(|&s| s > 0)
        .ok_or(InputPreparationError::VisionGeometry)?;
    let merge = usize::try_from(vision.preprocessing.merge)
        .map_err(|_| InputPreparationError::VisionGeometry)?;
    let [t, h, w] = grid;
    if t != 1 || merge == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
        return Err(InputPreparationError::VisionGeometry);
    }
    side.checked_mul(side)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or(InputPreparationError::VisionGeometry)?;
    let rows = h * w;
    let mut coordinates = Vec::with_capacity(rows);
    let mut indices: [Vec<i32>; 4] = std::array::from_fn(|_| Vec::with_capacity(rows));
    let mut coefficients: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(rows));
    for cell_row in 0..h / merge {
        for cell_col in 0..w / merge {
            for inner_row in 0..merge {
                for inner_col in 0..merge {
                    let row = cell_row * merge + inner_row;
                    let col = cell_col * merge + inner_col;
                    coordinates.push([
                        i32::try_from(col + 1).map_err(|_| InputPreparationError::VisionGeometry)?,
                        i32::try_from(row + 1).map_err(|_| InputPreparationError::VisionGeometry)?,
                    ]);
                    let (y0, y1, dy) = bilinear_axis(row, h, side);
                    let (x0, x1, dx) = bilinear_axis(col, w, side);
                    for (plane, (y, x, coeff)) in [
                        (y0, x0, (1.0 - dy) * (1.0 - dx)),
                        (y0, x1, (1.0 - dy) * dx),
                        (y1, x0, dy * (1.0 - dx)),
                        (y1, x1, dy * dx),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        indices[plane].push((y * side + x) as i32);
                        coefficients[plane].push(coeff);
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

pub struct Inkling2Family;

impl ModelFamily for Inkling2Family {
    fn name(&self) -> &'static str {
        ARCHITECTURE
    }

    fn architectures(&self) -> &'static [&'static str] {
        &[ARCHITECTURE, ARCHITECTURE_LOOP]
    }

    fn inspect(
        &self,
        target: &Directory,
        projector: Option<&Directory>,
        identity: PackageIdentity,
    ) -> Result<ModelDefinition, FamilyError> {
        inspect_components(target, projector, identity)
    }

    fn input_adapter(
        &self,
        definition: &ModelDefinition,
        markers: &dyn MarkerTokens,
    ) -> Result<FamilyInputAdapter, FamilyError> {
        if definition.vision.is_none() {
            return Ok(Box::new(TextInput::new(definition)));
        }
        let [start, row, end] = IMAGE_MARKERS;
        let markers = ImageMarkers::new_with_shared_end(
            markers.marker(start)?,
            markers.marker(row)?,
            markers.marker(end)?,
        )
        .map_err(|error| FamilyError::invalid(error.to_string()))?;
        Ok(Box::new(SequentialImageInput::new(
            ARCHITECTURE,
            Some(markers),
            BoundaryRule::Causal,
            inkling2_spatial_controls,
        )))
    }

    fn media_placeholder(&self, definition: &ModelDefinition) -> Option<&'static str> {
        definition.vision.as_ref().map(|_| IMAGE_PLACEHOLDER)
    }
}
