//! One prepared vision unit and its planned physical patch class.

use crate::{PREFILL_ROW_QUANTUM, SMALL_CLASS_ROWS};
use magnitude_family_contracts::PreparedVisionInput;
use std::fmt;

/// The cell class an image of `cells` merged cells encodes in: the launch row
/// ladder (the next power of two up to [`SMALL_CLASS_ROWS`], otherwise the
/// next multiple of [`PREFILL_ROW_QUANTUM`]) without the launch row bound.
/// `None` for zero cells.
pub const fn image_cell_class(cells: usize) -> Option<usize> {
    if cells == 0 {
        return None;
    }
    if cells <= SMALL_CLASS_ROWS {
        return Some(cells.next_power_of_two());
    }
    Some(cells.div_ceil(PREFILL_ROW_QUANTUM) * PREFILL_ROW_QUANTUM)
}

/// Every cell class of images of at most `limit` cells, in ascending order:
/// the ladder up to and including the class covering `limit`.
pub fn image_cell_classes(limit: usize) -> Vec<usize> {
    let Some(last) = image_cell_class(limit) else {
        return Vec::new();
    };
    let mut classes = Vec::new();
    let mut cells = 1;
    while cells <= last {
        classes.push(cells);
        cells = image_cell_class(cells + 1).unwrap_or(cells + 1);
    }
    classes
}

#[derive(Clone, Debug)]
pub struct ValidatedVisionBatch {
    input: PreparedVisionInput,
    patch_rows: usize,
    class_patch_rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionBatchError {
    pub required_rows: usize,
    pub available_rows: usize,
}

impl fmt::Display for VisionBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "vision needs {} patch rows but its class has {}",
            self.required_rows, self.available_rows
        )
    }
}

impl std::error::Error for VisionBatchError {}

impl ValidatedVisionBatch {
    /// `PreparedVisionInput` already proves pixel and spatial alignment. This
    /// constructor adds the only remaining device-independent capacity fact.
    pub fn new(
        input: PreparedVisionInput,
        class_patch_rows: usize,
    ) -> Result<Self, VisionBatchError> {
        let patch_rows = input.spatial().rows();
        if patch_rows == 0 || patch_rows > class_patch_rows {
            return Err(VisionBatchError {
                required_rows: patch_rows,
                available_rows: class_patch_rows,
            });
        }
        Ok(Self {
            input,
            patch_rows,
            class_patch_rows,
        })
    }

    pub fn input(&self) -> &PreparedVisionInput {
        &self.input
    }
    pub fn patch_rows(&self) -> usize {
        self.patch_rows
    }
    pub fn class_patch_rows(&self) -> usize {
        self.class_patch_rows
    }
}

#[cfg(test)]
mod tests {
    use super::{image_cell_class, image_cell_classes};

    #[test]
    fn image_cell_classes_continue_the_launch_ladder_past_the_launch_bound() {
        assert_eq!(image_cell_class(0), None);
        assert_eq!(image_cell_class(3), Some(4));
        assert_eq!(image_cell_class(33), Some(64));
        assert_eq!(image_cell_class(4096), Some(4096));
        assert_eq!(image_cell_class(4097), Some(4160));
        let classes = image_cell_classes(4096);
        assert_eq!(&classes[..7], &[1, 2, 4, 8, 16, 32, 64]);
        assert_eq!(classes.len(), 6 + 4096 / 64);
        assert_eq!(classes.last(), Some(&4096));
        assert_eq!(image_cell_classes(280).last(), Some(&320));
        assert!(image_cell_classes(0).is_empty());
        for cells in 1..=4096 {
            let class = image_cell_class(cells).unwrap();
            assert!(classes.contains(&class) && class >= cells && class - cells < 64);
        }
    }
}
