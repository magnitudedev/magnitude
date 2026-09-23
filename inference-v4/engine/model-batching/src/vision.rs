//! One prepared vision unit and its planned physical patch class.

use magnitude_model_contracts::PreparedVisionInput;
use std::fmt;

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
