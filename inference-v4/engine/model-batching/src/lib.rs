//! Family-neutral row packing and physical launch-class contracts.

mod classes;
mod demand;
mod head;
mod rows;
mod state;
mod target;
mod vision;

pub use classes::{
    row_class, row_classes, ClassError, LaunchClass, MAX_CLASS_ROWS, MAX_CLASS_SEGMENTS,
    PREFILL_ROW_QUANTUM, SMALL_CLASS_ROWS,
};
pub use demand::Demand;
pub use head::{HeadSlot, ValidatedHeadBatch};
pub use rows::{
    Draw, DrawKind, PackError, PackedRowTables, Row, Select, Shaping, Slot, HISTORY_WIDTH,
    SHAPING_WIDTH,
};
pub use state::{StateBatchError, StateBatchKind, ValidatedStateBatch};
pub use target::{TargetBatchSlot, TargetBatchUpload, ValidatedTargetBatch};
pub use vision::{ValidatedVisionBatch, VisionBatchError};
