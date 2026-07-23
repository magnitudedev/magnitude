//! Durable local model inventory, storage, and Hugging Face acquisition.

mod cache;
mod capabilities;
mod catalog;
mod download;
mod download_service;
pub mod gguf;
mod identity;
mod inventory;
mod manifest;
mod package_service;
mod preview;
mod service;
mod validation;

pub use cache::{ModelBlobKind, ModelCache, ModelCacheWorkspace, ModelIndexKind};
pub use catalog::NativeRecommendableCatalog;
pub use download_service::ManagedModelDownloads;
pub use inventory::{InventoryConfig, ModelManager};
pub use preview::{ModelPreviewService, PreparedPreview};
pub use validation::validate_download_request;
