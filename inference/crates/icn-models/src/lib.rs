//! Durable local model inventory, storage, and Hugging Face acquisition.

mod download;
pub mod gguf;
mod identity;
mod inventory;
mod legacy;
mod manifest;
mod service;
mod validation;

pub use inventory::{InventoryConfig, ModelManager};
pub use validation::validate_download_request;
