//! Durable local model inventory, storage, and Hugging Face acquisition.

mod cache;
mod download;
pub mod gguf;
mod identity;
mod inventory;
mod legacy;
mod manifest;
mod preview;
mod service;
mod validation;

pub use cache::{ModelBlobKind, ModelCache, ModelCacheWorkspace, ModelIndexKind};
pub use inventory::{InventoryConfig, ModelManager};
pub use preview::{ModelPreviewService, PreparedPreview};
pub use validation::validate_download_request;
