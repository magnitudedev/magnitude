//! Family-neutral row packing and physical launch-class contracts.

mod classes;
mod demand;
mod head;
mod rows;
mod state;
mod target;
mod vision;

pub use classes::{ClassError, LaunchClass, MAX_CLASS_ROWS, MAX_CLASS_SEGMENTS};
pub use demand::Demand;
pub use head::ValidatedHeadBatch;
pub use rows::{
    ControlField, ControlOffsets, Draw, DrawKind, PackError, PackedRowTables, Row, Select, Shaping,
    Slot, HISTORY_WIDTH, SHAPING_WIDTH,
};
pub use state::{StateBatchError, StateBatchKind, ValidatedStateBatch};
pub use target::{TargetBatchSlot, TargetBatchUpload, ValidatedTargetBatch};
pub use vision::{ValidatedVisionBatch, VisionBatchError};
