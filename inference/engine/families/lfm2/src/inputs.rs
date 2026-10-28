//! LFM3-VL image input. Every row, text or media, takes its absolute
//! position on the decoder's one rotary axis; an image expands between its
//! `<|vision_start|>` and `<|vision_end|>` markers to one `<|image|>` row per
//! merged cell, as the released processor expands each image.

use crate::Architecture;
use magnitude_artifacts::BoundaryRule;
use magnitude_family_contracts::{
    ImageMarkers, InputPreparationError, ModelDefinition, SequentialImageInput,
    VisionSpatialControls,
};

/// Declares an image's start, row and end markers once, and the text one
/// image renders as from them.
macro_rules! image_markers {
    ($start:literal, $row:literal, $end:literal) => {
        /// The text one image renders as: the image row between its markers,
        /// which input preparation expands to the image's cells.
        pub const IMAGE_PLACEHOLDER: &str = concat!($start, $row, $end);

        /// The start, row and end markers of an image.
        pub const IMAGE_MARKERS: [&str; 3] = [$start, $row, $end];
    };
}

image_markers!("<|vision_start|>", "<|image|>", "<|vision_end|>");

pub fn input_adapter(markers: ImageMarkers) -> SequentialImageInput {
    // The decoder reads media rows causally, so a span may be split.
    SequentialImageInput::new(
        Architecture::Lfm3.family_id(),
        Some(markers),
        BoundaryRule::Causal,
        spatial_controls,
    )
}

/// Patch rows cell by cell (the `merge`² cells in raster order), each cell's
/// patches in raster order, as the processor flattens them; the tower holds
/// them in that order. A row's rotary coordinates are its patch's grid row
/// and column. The tower has no position table.
pub fn spatial_controls(
    definition: &ModelDefinition,
    grid: [usize; 3],
) -> Result<VisionSpatialControls, InputPreparationError> {
    let vision = definition
        .vision
        .as_ref()
        .ok_or(InputPreparationError::UnsupportedMedia)?;
    let merge = usize::try_from(vision.preprocessing.merge)
        .map_err(|_| InputPreparationError::VisionGeometry)?;
    let [t, h, w] = grid;
    if t != 1 || merge == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
        return Err(InputPreparationError::VisionGeometry);
    }
    let coordinate =
        |value: usize| i32::try_from(value).map_err(|_| InputPreparationError::VisionGeometry);
    let mut coordinates = Vec::with_capacity(h * w);
    for cell_row in 0..h / merge {
        for cell_col in 0..w / merge {
            for inner_row in 0..merge {
                for inner_col in 0..merge {
                    coordinates.push([
                        coordinate(cell_row * merge + inner_row)?,
                        coordinate(cell_col * merge + inner_col)?,
                    ]);
                }
            }
        }
    }
    VisionSpatialControls::new((0..h * w).collect(), coordinates, None, Vec::new())
}
