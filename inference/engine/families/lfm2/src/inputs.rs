//! LFM3-VL image input. Every row, text or media, takes its absolute
//! position on the decoder's one rotary axis; an image expands between its
//! `<|vision_start|>` and `<|vision_end|>` markers to one `<|image|>` row per
//! merged cell, as the released processor expands each image.

use crate::Architecture;
use magnitude_artifacts::BoundaryRule;
use magnitude_family_contracts::{cell_raster_controls, ImageMarkers, SequentialImageInput};

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
        cell_raster_controls,
    )
}
