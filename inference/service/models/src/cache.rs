use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{fs, io};

use getrandom::fill;
use magnitude_service_contracts::models::{ModelAssessment, ModelCapabilities, ModelId, ParsedModelId};
use seismic_io::{Durability, Privacy, Store};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::header_cache::{HEADER_CAPACITY_BYTES, HEADER_EVICTION_GRACE, HeaderCache};

/// One completed profile assessment. Its evidence key carries the assessment's identity:
/// environment, bundle, profile and depths.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct CachedModelAssessment {
    pub capabilities: ModelCapabilities,
    pub template_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<magnitude_service_contracts::models::ModelFingerprint>,
    pub profile: ModelAssessment,
}

/// Presentation already established by candidate inspection, before the weights are installed.
/// This is derived engine data, never a declaration supplied by the client.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct CachedModelPresentation {
    pub display_name: String,
    pub variant_label: String,
    pub variant_of: Option<ModelId>,
    pub license: Option<String>,
}

const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;

/// The inventory index under the cache root ([`ModelCache::update_inventory`]).
const INVENTORY: &str = "indexes/inventory.json";

#[derive(Clone, Copy, Debug)]
pub enum ModelIndexKind {
    Artifact,
    InventoryMetadata,
    GgufInspection,
    HuggingFaceRepositorySnapshot,
    HuggingFaceBaseModels,
    HuggingFaceGgufHeader,
    HuggingFaceSearchShape,
    HuggingFaceWeightSample,
    RecommendableModelCatalog,
    ModelAssessment,
    CandidateClassification,
    CandidatePresentation,
    OllamaHubFile,
}

impl ModelIndexKind {
    fn relative(self) -> &'static str {
        match self {
            Self::Artifact => "artifacts",
            Self::InventoryMetadata => "inventory/model-metadata",
            Self::GgufInspection => "inspections/gguf",
            Self::HuggingFaceRepositorySnapshot => "sources/hugging-face/repositories",
            Self::HuggingFaceBaseModels => "sources/hugging-face/base-models",
            Self::HuggingFaceGgufHeader => "sources/hugging-face/gguf-headers",
            Self::HuggingFaceSearchShape => "sources/hugging-face/search-shapes",
            Self::HuggingFaceWeightSample => "sources/hugging-face/weight-samples",
            Self::RecommendableModelCatalog => "catalogs/recommendable-models",
            Self::ModelAssessment => "assessments/model-configurations",
            Self::CandidateClassification => "inspections/candidate-classification",
            Self::CandidatePresentation => "inspections/candidate-presentation",
            Self::OllamaHubFile => "sources/hugging-face/ollama-files",
        }
    }
}

/// The sole filesystem namespace owner for recomputable model data: one store
/// ([`seismic_io::Store`]) under the cache root, private to its owner, which every process that
/// names the root shares.
///
/// Every file it keeps is derived from what it is keyed by, so a read that fails in any way is a
/// miss its caller computes again, and every write is volatile: a crash that tears one leaves a
/// file that does not decode, a miss. An index is keyed by the digest of its evidence and holds
/// what that evidence determines, so each writer writes a whole value any reader accepts: it is
/// published without a lock, the last to land holding it. The inventory index gathers what
/// several writers learn, so it changes only under its record's lock, from what it holds then
/// ([`ModelCache::update_inventory`]).
#[derive(Clone, Debug)]
pub struct ModelCache {
    store: Arc<Store>,
    headers: HeaderCache,
}

pub struct ModelCacheWorkspace {
    path: PathBuf,
    temporary: Option<tempfile::TempDir>,
}

impl ModelCacheWorkspace {
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ModelCacheWorkspace {
    fn drop(&mut self) {
        if self.temporary.is_none() {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

impl ModelCache {
    pub fn new(cache_root: &Path) -> Result<Self, seismic_io::StoreError> {
        Self::with_header_capacity(cache_root, HEADER_CAPACITY_BYTES)
    }

    /// A cache whose GGUF headers are bounded by `capacity` bytes: a build that must hold every header
    /// it reads until it ends (the release catalog's bundle) bounds them by none.
    pub fn with_header_capacity(cache_root: &Path, capacity: u64) -> Result<Self, seismic_io::StoreError> {
        let store = Arc::new(Store::open(cache_root.to_path_buf(), Privacy::Owner)?);
        Ok(Self {
            headers: HeaderCache::new(Arc::clone(&store), capacity, HEADER_EVICTION_GRACE),
            store,
        })
    }

    /// The inventory index as it is on disk, none where it is absent or is no JSON object.
    pub(crate) fn read_inventory(&self) -> Option<Map<String, Value>> {
        decode_object(self.store.read(Path::new(INVENTORY), MAX_INDEX_BYTES).ok().flatten()?)
    }

    /// Replace the inventory index by `change` of what it holds now (none where it is absent or
    /// does not decode), its record's lock held from the read to the write, so a change another
    /// process made meanwhile is never lost to this one. A failure leaves the index as it was.
    pub(crate) fn update_inventory<T: Serialize>(&self, change: impl FnOnce(Option<Map<String, Value>>) -> T) {
        let Ok(record) = self.store.lock(Path::new(INVENTORY)) else {
            return;
        };
        let held = record.read(MAX_INDEX_BYTES).ok().flatten().and_then(decode_object);
        if let Some(bytes) = encoded(&change(held)) {
            let _ = record.write(&bytes, Durability::Volatile);
        }
    }

    pub fn read_index<T: DeserializeOwned>(
        &self,
        kind: ModelIndexKind,
        evidence: &str,
    ) -> Option<T> {
        let bytes = self.store.read(&index_path(kind, &evidence_digest(evidence)), MAX_INDEX_BYTES).ok().flatten()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Publish what `evidence` determines of `kind` (see [`ModelCache`]). A
    /// failure, or a value larger than any index is read to, leaves what was there.
    pub fn write_index<T: Serialize>(&self, kind: ModelIndexKind, evidence: &str, value: &T) {
        if let Some(bytes) = encoded(value) {
            let _ = self.store.shared().and_then(|shared| {
                shared.publish(&index_path(kind, &evidence_digest(evidence)), Durability::Volatile, |file| {
                    io::Write::write_all(file, &bytes)
                })
            });
        }
    }

    pub fn read_model_assessment(&self, evidence: &str) -> Option<CachedModelAssessment> {
        self.read_index(ModelIndexKind::ModelAssessment, evidence)
    }

    pub fn write_model_assessment(&self, evidence: &str, assessment: &CachedModelAssessment) {
        self.write_index(ModelIndexKind::ModelAssessment, evidence, assessment);
    }

    /// A version's inspected identity at an immutable Hub commit. Changing the release catalog
    /// invalidates its family mapping; no repository name is used to guess a family on a miss.
    pub fn read_model_presentation(
        &self,
        model_id: &ModelId,
        commit: &str,
    ) -> Option<CachedModelPresentation> {
        self.read_index::<CachedModelPresentation>(
            ModelIndexKind::CandidatePresentation,
            &presentation_evidence(model_id, commit),
        )
        .filter(|presentation| {
            !presentation.display_name.trim().is_empty()
                && !presentation.variant_label.trim().is_empty()
                && presentation.variant_of.as_ref().is_none_or(|id| {
                    matches!(id.parsed(), ParsedModelId::Catalog { .. })
                })
        })
    }

    pub fn write_model_presentation(
        &self,
        model_id: &ModelId,
        commit: &str,
        presentation: &CachedModelPresentation,
    ) {
        self.write_index(
            ModelIndexKind::CandidatePresentation,
            &presentation_evidence(model_id, commit),
            presentation,
        );
    }

    /// The engine's compiled-kernel store, shared by loads and tuning.
    #[must_use]
    pub fn kernel_directory(&self) -> PathBuf {
        self.store.root().join("kernels")
    }

    /// The GGUF header with this sha256, or `None` when the header cache doesn't hold it (see
    /// `header_cache`): its caller fetches it again.
    #[must_use]
    pub fn read_header(&self, digest: &str) -> Option<Vec<u8>> {
        self.headers.read(digest)
    }

    /// The skeleton of the GGUF header with this sha256 (its large arrays' payloads absent), or
    /// `None` when the header cache doesn't hold it.
    #[must_use]
    pub fn read_header_skeleton(
        &self,
        digest: &str,
    ) -> Option<magnitude_artifacts::gguf::Skeleton> {
        self.headers.read_skeleton(digest)
    }

    /// Discards the header with this sha256 from the header cache, a payload of its skeleton having
    /// failed to read: it is stored again when next written.
    pub fn discard_header(&self, digest: &str) {
        self.headers.discard_header(digest);
    }

    /// Cached headers holding a large entry whose fixed part is `head` (see `header_cache`).
    #[must_use]
    pub(crate) fn sharing_headers(&self, head: &[u8]) -> Vec<Vec<u8>> {
        self.headers.sharing(head)
    }

    /// Cached headers holding that entry, after any read fetching it stores its header; or the
    /// claim to fetch it (see `header_cache`).
    pub(crate) async fn lend_or_claim(&self, head: &[u8]) -> crate::header_cache::Lending {
        self.headers.lend_or_claim(head).await
    }

    /// Whether the header cache holds this header, marking it used for the read that follows;
    /// `read_header` still verifies it.
    #[must_use]
    pub fn has_header(&self, digest: &str) -> bool {
        self.headers.contains(digest)
    }

    /// Stores a GGUF header and returns its sha256, evicting what no live cache pins once the
    /// header cache has outgrown its capacity.
    pub fn write_header(&self, header: &[u8]) -> String {
        self.headers.write(header)
    }

    /// Pin `pinned`, the headers this process needs kept, for every process's eviction while
    /// this cache lives: pinned again whenever they change.
    pub fn pin_headers(&self, pinned: BTreeSet<String>) {
        self.headers.pin(pinned);
    }

    pub fn workspace(&self) -> io::Result<ModelCacheWorkspace> {
        let mut random = [0_u8; 16];
        fill(&mut random).map_err(io::Error::other)?;
        let name = random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let path = self.store.root().join(".work").join(name);
        // A workspace is its owner's alone, as the store is: where it cannot be made so, it is
        // made in the system's temporary directory.
        match Privacy::Owner.directory(&path) {
            Ok(()) => return Ok(ModelCacheWorkspace { path, temporary: None }),
            Err(_) => {
                let _ = fs::remove_dir_all(&path);
            }
        }
        let temporary = tempfile::Builder::new()
            .prefix("magnitude-model-preview-")
            .tempdir()?;
        Ok(ModelCacheWorkspace {
            path: temporary.path().to_path_buf(),
            temporary: Some(temporary),
        })
    }
}

/// Where the index of `kind` keyed by the evidence digest `digest` lies under the cache root.
fn index_path(kind: ModelIndexKind, digest: &str) -> PathBuf {
    Path::new("indexes").join(kind.relative()).join(format!("{digest}.json"))
}

/// The JSON object `bytes` hold, none where they hold none.
fn decode_object(bytes: Vec<u8>) -> Option<Map<String, Value>> {
    match serde_json::from_slice::<Value>(&bytes).ok()? {
        Value::Object(object) => Some(object),
        _ => None,
    }
}

/// `value` as an index holds it, none where it does not encode or is larger than any index is
/// read to.
fn encoded<T: Serialize>(value: &T) -> Option<Vec<u8>> {
    serde_json::to_vec_pretty(value).ok().filter(|bytes| bytes.len() as u64 <= MAX_INDEX_BYTES)
}

fn presentation_evidence(model_id: &ModelId, commit: &str) -> String {
    static CATALOG: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let catalog = CATALOG.get_or_init(|| {
        format!("{}:{}", crate::catalog::catalog_source_digest(), crate::catalog::catalog_lock_digest())
    });
    format!(
        "candidate-presentation-v2:{}:{catalog}:{model_id}@{commit}",
        crate::gguf::RULES
    )
}

fn evidence_digest(evidence: &str) -> String {
    hex_sha256(evidence.as_bytes())
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_service_contracts::MemoryDomainId;
    use magnitude_service_contracts::models::{
        MemoryAssessment, MemoryBreakdown, ModelAssessmentId, ModelPlacement, ModelCapabilities, ModelPerformance,
        ModelReasoningCapabilities, ModelResidency, PerformanceEvidence, ServingProfile,
    };

    fn profile(context_length: u32) -> ServingProfile {
        ServingProfile { context_length }
    }

    #[test]
    fn inspected_presentation_is_pinned_to_commit_and_selected_file_across_reopen() {
        let directory = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(directory.path()).unwrap();
        let q4: ModelId = "hf:publisher/Qwen-Finetune-GGUF/model-Q4_K_M.gguf".parse().unwrap();
        let q8: ModelId = "hf:publisher/Qwen-Finetune-GGUF/model-Q8_0.gguf".parse().unwrap();
        let presentation = CachedModelPresentation {
            display_name: "Qwen Finetune".to_owned(),
            variant_label: "Q4_K_M".to_owned(),
            variant_of: Some("qwen3.6-35b-a3b:gguf:q4".parse().unwrap()),
            license: None,
        };
        let commit = "c".repeat(40);
        cache.write_model_presentation(&q4, &commit, &presentation);
        let reopened = ModelCache::new(directory.path()).unwrap();
        assert_eq!(reopened.read_model_presentation(&q4, &commit), Some(presentation.clone()));
        assert_eq!(reopened.read_model_presentation(&q8, &commit), None);
        assert_eq!(reopened.read_model_presentation(&q4, &"d".repeat(40)), None);
        cache.write_model_presentation(&q4, &commit, &CachedModelPresentation {
            variant_of: Some(q8),
            ..presentation
        });
        assert_eq!(reopened.read_model_presentation(&q4, &commit), None, "a family must be a validated catalog identity");
    }

    #[test]
    fn namespaces_are_typed_and_fail_as_misses() {
        let directory = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(directory.path()).unwrap();

        cache.write_index(ModelIndexKind::InventoryMetadata, "model evidence", &42_u64);
        assert_eq!(
            cache.read_index::<u64>(ModelIndexKind::InventoryMetadata, "model evidence"),
            Some(42)
        );
        let bytes = b"header";
        let digest = cache.write_header(bytes);
        assert_eq!(digest, hex_sha256(bytes));
        assert_eq!(cache.read_header(&digest).as_deref(), Some(bytes.as_slice()));
        assert!(cache.read_header(&"0".repeat(64)).is_none());

        let index_path = directory
            .path()
            .join(index_path(ModelIndexKind::InventoryMetadata, &evidence_digest("model evidence")));
        fs::write(&index_path, b"corrupt").unwrap();
        assert_eq!(
            cache.read_index::<u64>(ModelIndexKind::InventoryMetadata, "model evidence"),
            None
        );
        cache.write_index(ModelIndexKind::InventoryMetadata, "model evidence", &7_u64);
        assert_eq!(
            cache.read_index::<u64>(ModelIndexKind::InventoryMetadata, "model evidence"),
            Some(7)
        );

        fs::write(directory.path().join("blobs/gguf-headers").join(&digest), b"corrupt").unwrap();
        assert!(cache.read_header(&digest).is_none());
    }

    #[test]
    fn workspaces_are_private_and_removed_on_drop() {
        let directory = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(directory.path()).unwrap();
        let path = {
            let workspace = cache.workspace().unwrap();
            let path = workspace.path().to_path_buf();
            assert!(path.is_dir());
            path
        };
        assert!(!path.exists());
    }

    #[test]
    fn fitting_target_with_non_fitting_sibling_profiles_survives_cache_reopen() {
        let directory = tempfile::tempdir().unwrap();
        let fits = ModelAssessment::Fits {
            profile: profile(50_000),
            served: profile(50_000),
            assessment_id: ModelAssessmentId("fits-assessment".to_owned()),
            memory: vec![MemoryAssessment {
                memory_domain_id: MemoryDomainId::system(),
                capacity_bytes: 100,
                required_bytes: 50,
                remaining_bytes: 50,
            }],
            memory_breakdown: MemoryBreakdown {
                weights_bytes: 30,
                context_bytes: 10,
                overhead_bytes: 10,
            },
            placement: ModelPlacement {
                device_bytes: 50,
                host_bytes: 0,
                disk_bytes: 0,
                lookup_disk_bytes: 0,
            },
            residency: ModelResidency::Resident {
                performance: ModelPerformance::Estimated {
                    samples: vec![PerformanceEvidence {
                        context_tokens: 50_000,
                        estimated_tokens_per_second: 2.0,
                    }],
                },
            },
            projector: magnitude_service_contracts::models::ProjectorVerdict::None,
            drafter: magnitude_service_contracts::models::DrafterVerdict::None,
        };
        let does_not_fit = ModelAssessment::DoesNotFit {
            profile: profile(100_000),
            shortfall: profile(32_768),
            assessment_id: ModelAssessmentId("non-fit-assessment".to_owned()),
            memory: vec![MemoryAssessment {
                memory_domain_id: MemoryDomainId::system(),
                capacity_bytes: 100,
                required_bytes: 101,
                remaining_bytes: -1,
            }],
            limiting_resource: "system memory".to_owned(),
            deficit_bytes: 1,
            projector: magnitude_service_contracts::models::ProjectorVerdict::None,
            drafter: magnitude_service_contracts::models::DrafterVerdict::None,
        };

        let initial = ModelCache::new(directory.path()).unwrap();
        let capabilities = ModelCapabilities {
            vision: false,
            tools: true,
            structured_output: true,
            reasoning: ModelReasoningCapabilities {
                supported: false,
                efforts: Vec::new(),
                default_effort: None,
            },
        };
        let fits = CachedModelAssessment {
            capabilities: capabilities.clone(),
            template_fingerprint: "template".to_owned(),
            fingerprint: None,
            profile: fits,
        };
        let does_not_fit = CachedModelAssessment {
            capabilities,
            template_fingerprint: fits.template_fingerprint.clone(),
            fingerprint: None,
            profile: does_not_fit,
        };
        initial.write_model_assessment("fits evidence", &fits);
        initial.write_model_assessment("non-fit evidence", &does_not_fit);

        let reopened = ModelCache::new(directory.path()).unwrap();

        assert_eq!(reopened.read_model_assessment("fits evidence"), Some(fits));
        assert_eq!(
            reopened.read_model_assessment("non-fit evidence"),
            Some(does_not_fit),
        );
    }

    #[test]
    fn workspace_falls_back_when_the_cache_root_is_unusable() {
        let directory = tempfile::tempdir().unwrap();
        let blocked = directory.path().join("blocked-cache");
        fs::write(&blocked, b"not a directory").unwrap();
        assert!(matches!(ModelCache::new(&blocked), Err(seismic_io::StoreError::Io { .. })));

        let cache_root = directory.path().join("cache");
        let cache = ModelCache::new(&cache_root).unwrap();
        fs::write(cache_root.join(".work"), b"not a directory").unwrap();
        let path = {
            let workspace = cache.workspace().unwrap();
            let path = workspace.path().to_path_buf();
            assert!(path.is_dir());
            assert!(!path.starts_with(directory.path()));
            path
        };
        assert!(!path.exists());
    }
}
