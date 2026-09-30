//! Family-neutral row packing and physical launch-class contracts.

mod classes;
mod demand;
mod head;
mod rows;
mod state;
mod target;
mod vision;

pub use classes::{
    row_class, row_classes, ClassError, ClassLimits, LaunchClass, MAX_CLASS_ROWS,
    PREFILL_ROW_QUANTUM, SMALL_CLASS_ROWS,
};
pub use demand::Demand;
pub use head::{BlockSlot, HeadPasses, HeadSlot, ValidatedHeadBatch};
pub use rows::{
    Draw, DrawKind, HistoryTables, PackError, PackedRowTables, Row, RowHistory, Select, Shaping,
    Slot, HISTORY_WIDTH, SHAPING_WIDTH,
};
pub use state::{StateBatchError, StateBatchKind, ValidatedStateBatch};
pub use target::{SlotHistory, TargetBatchSlot, TargetBatchUpload, ValidatedTargetBatch};
pub use vision::{ValidatedVisionBatch, VisionBatchError};
