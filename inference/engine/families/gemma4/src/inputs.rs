//! Gemma 4 input preparation. Every row, text or media, takes its absolute
//! position on the model's single rotary axis; an image expands between its
//! `<|image>` and `<image|>` markers to one `<|image|>` row per pooled cell.
//! Media rows (non-causal within their span, read unscaled, with the
//! per-layer table's media row) are described by the definition.

use magnitude_artifacts::BoundaryRule;
use magnitude_family_contracts::{
    ImageMarkers, InputPreparationError, ModelDefinition, PositionSampling, SequentialImageInput,
    VisionSpatialControls,
};

/// The text one image renders as: the image row between its markers, which
/// input preparation expands to the image's pooled cells.
pub const IMAGE_PLACEHOLDER: &str = "<|image><|image|><image|>";

pub fn input_adapter(markers: Option<ImageMarkers>) -> SequentialImageInput {
    // Image rows see every row of their image, so a span is never split.
    SequentialImageInput::new(
        crate::ARCHITECTURE,
        markers,
        BoundaryRule::Indivisible,
        spatial_controls,
    )
}

/// Patch rows cell by cell (the pooler's `merge`² cells in raster order),
/// each cell's patches in raster order. The tower's rotary coordinates are
/// `[column, row]` (HF's `pixel_position_ids` `(x, y)`); its position value
/// is row `column` of the first axis table plus row `row` of the second.
fn spatial_controls(
    definition: &ModelDefinition,
    grid: [usize; 3],
) -> Result<VisionSpatialControls, InputPreparationError> {
    let vision = definition
        .vision
        .as_ref()
        .ok_or(InputPreparationError::UnsupportedMedia)?;
    let PositionSampling::Axes { length } = vision.stem.positions().sampling else {
        return Err(InputPreparationError::VisionGeometry);
    };
    let merge = vision.preprocessing.merge as usize;
    let length = usize::try_from(length).map_err(|_| InputPreparationError::VisionGeometry)?;
    let [t, h, w] = grid;
    if t != 1 || merge == 0 || h % merge != 0 || w % merge != 0 || h > length || w > length {
        return Err(InputPreparationError::VisionGeometry);
    }
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
                    coordinates.push([col as i32, row as i32]);
                    for (plane, (index, coefficient)) in
                        [(col, 1.0), (length + row, 1.0), (0, 0.0), (0, 0.0)]
                            .into_iter()
                            .enumerate()
                    {
                        indices[plane].push(index as i32);
                        coefficients[plane].push(coefficient);
                    }
                }
            }
        }
    }
    VisionSpatialControls::new((0..rows).collect(), coordinates, indices, coefficients, Vec::new())
}
