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
mod planner_stub;
mod preview;
mod service;
#[cfg(test)]
mod test_support;
mod validation;

pub use cache::{ModelBlobKind, ModelCache, ModelCacheWorkspace, ModelIndexKind};
pub use catalog::{
    GeneratedReleaseCatalog, ReleaseCatalog, ReleaseCatalogManifest, ReleaseRecommendableCatalog,
    ResolvingRecommendableCatalog, catalog_source_digest, catalog_source_digest_from,
    encode_release_planner_bundle, encode_release_planner_bundle_with_progress,
    load_release_catalog, release_catalog_manifest,
};
pub use download_service::ManagedModelDownloads;
pub use inventory::{InventoryConfig, ModelManager};
pub use package_service::{canonical_package_id, offering_target_id};
pub use planner_stub::{
    PLANNER_STUB_FORMAT_IDENTITY, PlannerStubComponent, PlannerStubContext, PlannerStubError,
    compact_planner_stub, planner_stub_context,
};
pub use preview::{ModelPreviewService, PreparedPreview};
pub use validation::validate_download_request;
