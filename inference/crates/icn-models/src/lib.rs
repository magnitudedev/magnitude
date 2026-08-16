//! Durable local model inventory, storage, and Hugging Face acquisition.

mod cache;
mod capabilities;
mod catalog;
mod catalog_affiliations;
mod catalog_models;
mod download;
mod download_service;
pub mod gguf;
mod hugging_face;
mod identity;
mod inventory;
mod package_service;
mod planner_stub;
mod preview;
mod service;
mod store_fs;
#[cfg(test)]
mod test_support;
mod validation;

pub use cache::{ModelBlobKind, ModelCache, ModelCacheWorkspace, ModelIndexKind};
pub use catalog::{
    GeneratedReleaseCatalog, ReleaseCatalog, ReleaseRecommendableCatalog,
    ResolvingRecommendableCatalog, advance_model_catalog_lock, load_release_catalog,
    model_catalog_lock,
};
pub use catalog_models::ManagedCatalogModels;
pub use download_service::ManagedModelDownloads;
pub use inventory::{InventoryConfig, ManagedModelStore};
pub use package_service::{
    canonical_package_id, servable_model_bundle_key, servable_model_bundle_key_for_bundle,
    serving_configuration_id, serving_configuration_identity_is_valid,
    speculative_servable_model_bundle_key,
};
pub use planner_stub::{
    PlannerStubComponent, PlannerStubContext, PlannerStubError, compact_planner_stub,
    planner_stub_context,
};
pub use preview::{ModelPreviewService, PreparedPreview, refresh_hugging_face_repository};
