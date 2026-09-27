//! Durable local model inventory, storage, and Hugging Face acquisition.

mod cache;
mod catalog;
mod catalog_affiliations;
mod catalog_installations;
mod catalog_models;
mod discovered_models;
mod download;
mod download_service;
pub mod gguf;
mod hugging_face;
mod identity;
mod inventory;
mod model_domains;
mod model_projection;
mod package_service;
mod planner_stub;
mod preview;
mod service;
mod store_fs;
#[cfg(test)]
mod test_support;
mod validation;

pub use cache::{
    CachedModelAssessment, ModelBlobKind, ModelCache, ModelCacheWorkspace, ModelIndexKind,
};
pub use catalog::{
    GeneratedReleaseCatalog, ReleaseCatalog, ReleaseRecommendableCatalog,
    ResolvingRecommendableCatalog, advance_model_catalog_lock, load_release_catalog,
    model_catalog_lock,
};
pub use catalog_installations::ManagedCatalogInstallations;
pub use catalog_models::{CatalogRemovalPlan, ManagedCatalogModels};
pub use discovered_models::ManagedDiscoveredModels;
pub use download_service::ManagedModelDownloads;
pub use inventory::{InventoryConfig, ManagedModelStore};
pub use model_domains::{ManagedModelServices, ModelDomainResolver, managed_model_services};
pub use package_service::{
    ServableModelBundleKey, canonical_package_id, servable_model_bundle_key,
    servable_model_bundle_key_for_bundle, serving_configuration_fingerprint,
    speculative_servable_model_bundle_key,
};
pub use planner_stub::{
    AssessmentMaterialComponent, AssessmentMaterialContext, AssessmentMaterialError,
    assessment_material_context, compact_assessment_material,
};
pub use preview::{ModelPreviewService, PreparedPreview, refresh_hugging_face_repository};
