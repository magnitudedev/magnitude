//! Nemotron Omni image input. Every row, text or media, takes its absolute
//! position (the decoder has no rotary embedding); an image expands between
//! its `<img>` and `</img>` markers to one `<image>` row per 2 × 2 cell, as
//! the released processor expands each image.

use crate::ARCHITECTURE;
use magnitude_artifacts::BoundaryRule;
use magnitude_family_contracts::{
    ImageMarkers, InputPreparationError, ModelDefinition, PositionSampling, SequentialImageInput,
    VisionSpatialControls,
};

/// The text one image renders as: the image row between its markers, which
/// input preparation expands to the image's cells.
pub const IMAGE_PLACEHOLDER: &str = "<img><image></img>";

/// The start, row and end markers of an image.
pub const IMAGE_MARKERS: [&str; 3] = ["<img>", "<image>", "</img>"];

pub fn input_adapter(markers: ImageMarkers) -> SequentialImageInput {
    // The decoder reads media rows causally, so a span may be split.
    SequentialImageInput::new(
        ARCHITECTURE,
        Some(markers),
        BoundaryRule::Causal,
        spatial_controls,
    )
}

/// One axis of PyTorch's bilinear `interpolate` (`align_corners=False`) from
/// `side` table rows to `size` samples, in F32 as PyTorch computes it: the
/// two taps and their weights. The source index is clamped at zero and the
/// upper tap at the last row.
fn taps(index: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    let scale = side as f32 / size as f32;
    let source = (scale * (index as f32 + 0.5) - 0.5).max(0.0);
    let first = (source.floor() as usize).min(side - 1);
    let next = if first < side - 1 { first + 1 } else { first };
    let fraction = (source - first as f32).clamp(0.0, 1.0);
    [(first, 1.0 - fraction), (next, fraction)]
}

/// Patch rows cell by cell (the `merge`² cells in raster order), each cell's
/// patches in raster order; the tower attends over all of them, so its rows
/// keep the processor's order. A patch's position value is the C-RADIO
/// cropped position embedding: the square table resampled to a square of
/// the grid's longer side, sampled at the patch's row and column. The tower
/// has no rotary embedding.
pub fn spatial_controls(
    definition: &ModelDefinition,
    grid: [usize; 3],
) -> Result<VisionSpatialControls, InputPreparationError> {
    let vision = definition
        .vision
        .as_ref()
        .ok_or(InputPreparationError::UnsupportedMedia)?;
    let PositionSampling::CroppedSquare { side } = vision.stem.positions().sampling else {
        return Err(InputPreparationError::VisionGeometry);
    };
    let side = usize::try_from(side).map_err(|_| InputPreparationError::VisionGeometry)?;
    let merge = usize::try_from(vision.preprocessing.merge)
        .map_err(|_| InputPreparationError::VisionGeometry)?;
    let [t, h, w] = grid;
    if t != 1 || side == 0 || merge == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
        return Err(InputPreparationError::VisionGeometry);
    }
    let square = h.max(w);
    let rows = h
        .checked_mul(w)
        .filter(|rows| *rows <= i32::MAX as usize)
        .ok_or(InputPreparationError::VisionGeometry)?;
    let mut indices: [Vec<i32>; 4] = std::array::from_fn(|_| Vec::with_capacity(rows));
    let mut coefficients: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(rows));
    for cell_row in 0..h / merge {
        for cell_col in 0..w / merge {
            for inner_row in 0..merge {
                for inner_col in 0..merge {
                    let row = cell_row * merge + inner_row;
                    let col = cell_col * merge + inner_col;
                    let (vertical, horizontal) = (taps(row, square, side), taps(col, square, side));
                    for (plane, ((y, wy), (x, wx))) in vertical
                        .iter()
                        .flat_map(|y| horizontal.iter().map(move |x| (*y, *x)))
                        .enumerate()
                    {
                        indices[plane].push(
                            i32::try_from(y * side + x)
                                .map_err(|_| InputPreparationError::VisionGeometry)?,
                        );
                        coefficients[plane].push(wy * wx);
                    }
                }
            }
        }
    }
    VisionSpatialControls::new(
        (0..rows).collect(),
        vec![[0, 0]; rows],
        indices,
        coefficients,
        Vec::new(),
    )
}
