//! Durable local model inventory, storage, and Hugging Face acquisition.

mod cache;
pub mod candidate_files;
mod catalog;
mod catalog_acquisition;
mod catalog_affiliations;
mod catalog_installations;
mod catalog_models;
mod custom_models;
mod download;
mod download_service;
pub mod gguf;
mod header_cache;
mod header_layout;
mod hugging_face;
mod identity;
mod inventory;
mod local_roots;
mod local_watch;
pub mod lineage;
mod model_domains;
mod model_projection;
mod requested_downloads;
mod package_service;
mod preview;
mod remote_header;
mod search_shape;
mod service;
mod snapshot_blobs;
mod store_fs;
#[cfg(test)]
mod test_gguf;
mod upstream_updates;
mod validation;
mod weight_lineage;

pub use cache::{
    CachedModelAssessment, ModelCache, ModelCacheWorkspace, ModelIndexKind,
};
pub use catalog::{
    CatalogLockAdvance, GeneratedReleaseCatalog, ReleaseCatalog, ReleaseComponentHeader,
    ReleasePackageRole, ReleaseRecommendableCatalog,
    ResolvingRecommendableCatalog, advance_model_catalog_lock, load_release_catalog,
    model_catalog_lock,
};
pub use catalog_acquisition::{AcquiredCatalogModel, AcquiredDrafter, AcquiredFile};
pub use catalog_installations::ManagedCatalogInstallations;
pub use catalog_models::{CatalogRemovalPlan, ManagedCatalogModels};
pub use custom_models::{
    CatalogDrafterFit, CustomModelEnrichment, CustomModelEnrichmentResult, CustomModelFacts,
    DrafterContent, ManagedLocalModels,
};
pub use download_service::ManagedModelDownloads;
pub use inventory::{DownloadSpace, InventoryConfig, ManagedModelStore};
pub use local_roots::llama_cpp_flat_prefix;
pub use model_domains::{ManagedModelServices, ModelDomainResolver, managed_model_services};
pub use package_service::{
    ServableModelBundleKey, canonical_package_id, servable_model_bundle_key,
    servable_model_bundle_key_for_bundle, serving_configuration_fingerprint,
    speculative_servable_model_bundle_key,
};
pub use search_shape::{
    CatalogFingerprints, HeaderExtent, HeaderVerdicts, ModelShape, SHAPE_PREFIX_BYTES,
};
pub use weight_lineage::{SampleRead, WeightSample};
pub use preview::{
    HuggingFaceDiscovery, PreparedPreview, PreparedPreviewHeader, RemoteGgufHeader,
    cached_hugging_face_repository, pinned_hugging_face_repository,
    refresh_hugging_face_repository,
};
