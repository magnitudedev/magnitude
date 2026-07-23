use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use futures_util::{StreamExt, stream};
use icn_contracts::models::{
    CatalogDiagnostic, ModelFailure, ModelOfferingTarget, RecommendableModel,
    RecommendableModelCatalog, RecommendableModelCatalogProvider, RecommendableModelId,
    ServingProfile,
};
use icn_contracts::{
    HuggingFaceModelCatalog, HuggingFaceRepositoryRequest, HuggingFaceRepositorySnapshot,
    InventoryError, ModelPreviewSource,
};
use serde::Serialize;

use crate::cache::ModelIndexKind;
use crate::capabilities::model_capabilities;
use crate::inventory::ModelManager;
use crate::package_service::{offering_target_id, package_from_resolved};

const CATALOG_RESOLUTION_REVISION: &str = "recommendable-model-catalog-v1";

#[derive(Clone, Copy, Serialize)]
struct CatalogModel {
    id: &'static str,
    display_name: &'static str,
    description: &'static str,
    repository: &'static str,
    formats: &'static [&'static str],
    contexts: &'static [u32],
    license: &'static str,
    quality_score: f64,
    quality_score_provenance: &'static str,
    quality_evidence: &'static [&'static str],
}

const QWEN_FORMATS: &[&str] = &["UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL"];
const ONE_HUNDRED_K: &[u32] = &[100_000];
const PRODUCT_CONTEXTS: &[u32] = &[100_000, 200_000];
const QUANT_EVIDENCE: &[&str] = &[
    "Quantization fidelity is curated independently from hardware fit.",
    "https://arxiv.org/abs/2606.19558",
];
const LAGUNA_EVIDENCE: &[&str] = &[
    "Publisher-measured Terminal-Bench v2.1 checkpoint result; not an exact GGUF quantization measurement.",
    "https://huggingface.co/poolside/Laguna-S-2.1",
    "https://arxiv.org/abs/2606.19558",
];

const CATALOG_MODELS: &[CatalogModel] = &[
    CatalogModel {
        id: "qwen3.5-4b",
        display_name: "Qwen3.5 4B",
        description: "Compact dense model for machines where responsiveness and footprint matter most.",
        repository: "unsloth/Qwen3.5-4B-GGUF",
        formats: QWEN_FORMATS,
        contexts: PRODUCT_CONTEXTS,
        license: "apache-2.0",
        quality_score: 25.8,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "qwen3.5-9b",
        display_name: "Qwen3.5 9B",
        description: "Small dense model with a substantial capability gain over the 4B tier.",
        repository: "unsloth/Qwen3.5-9B-GGUF",
        formats: QWEN_FORMATS,
        contexts: PRODUCT_CONTEXTS,
        license: "apache-2.0",
        quality_score: 29.2,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "qwen3.6-27b",
        display_name: "Qwen3.6 27B",
        description: "Large dense coding model with strong agent and coding capability.",
        repository: "unsloth/Qwen3.6-27B-GGUF",
        formats: QWEN_FORMATS,
        contexts: PRODUCT_CONTEXTS,
        license: "apache-2.0",
        quality_score: 60.7,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "qwen3.6-35b-a3b",
        display_name: "Qwen3.6 35B-A3B",
        description: "Efficient MoE coding model with a large knowledge footprint and low active compute.",
        repository: "unsloth/Qwen3.6-35B-A3B-GGUF",
        formats: QWEN_FORMATS,
        contexts: PRODUCT_CONTEXTS,
        license: "apache-2.0",
        quality_score: 44.9,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "gemma-4-e2b-it-qat",
        display_name: "Gemma 4 E2B",
        description: "Very small dense model optimized for on-device use.",
        repository: "unsloth/gemma-4-E2B-it-qat-GGUF",
        formats: &["UD-Q4_K_XL"],
        contexts: ONE_HUNDRED_K,
        license: "gemma",
        quality_score: 15.0,
        quality_score_provenance: "estimated_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "gemma-4-12b-it-qat",
        display_name: "Gemma 4 12B",
        description: "Mid-size dense model with native tool use, reasoning, and multimodal capability.",
        repository: "unsloth/gemma-4-12B-it-qat-GGUF",
        formats: &["UD-Q4_K_XL"],
        contexts: PRODUCT_CONTEXTS,
        license: "gemma",
        quality_score: 21.0,
        quality_score_provenance: "estimated_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "gemma-4-26b-a4b-it-qat",
        display_name: "Gemma 4 26B-A4B",
        description: "Mid-size MoE model balancing a substantial weight footprint with low active compute.",
        repository: "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        formats: &["UD-Q4_K_XL"],
        contexts: PRODUCT_CONTEXTS,
        license: "gemma",
        quality_score: 39.0,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "gemma-4-31b-it-qat",
        display_name: "Gemma 4 31B",
        description: "Large dense Gemma model with the strongest measured coding score in its family.",
        repository: "unsloth/gemma-4-31B-it-qat-GGUF",
        formats: &["UD-Q4_K_XL"],
        contexts: PRODUCT_CONTEXTS,
        license: "gemma",
        quality_score: 43.4,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "laguna-s-2.1",
        display_name: "Laguna S 2.1 118B-A8B",
        description: "High-capability MoE model designed for agentic coding and long-horizon software work.",
        repository: "poolside/Laguna-S-2.1-GGUF",
        formats: &["Q4_K_M", "Q8_0"],
        contexts: PRODUCT_CONTEXTS,
        license: "openmdw-1.1",
        quality_score: 70.2,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: LAGUNA_EVIDENCE,
    },
    CatalogModel {
        id: "qwen3.5-122b-a10b",
        display_name: "Qwen3.5 122B-A10B",
        description: "Workstation-class MoE model with a large knowledge footprint and moderate active compute.",
        repository: "unsloth/Qwen3.5-122B-A10B-GGUF",
        formats: &["UD-Q4_K_XL"],
        contexts: PRODUCT_CONTEXTS,
        license: "apache-2.0",
        quality_score: 47.6,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "nemotron-3-super-120b-a12b",
        display_name: "NVIDIA Nemotron 3 Super 120B-A12B",
        description: "Workstation-class hybrid MoE model designed for agentic workflows.",
        repository: "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
        formats: &["MXFP4_MOE"],
        contexts: PRODUCT_CONTEXTS,
        license: "nvidia-nemotron-open-model-license",
        quality_score: 38.6,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "deepseek-v4-flash",
        display_name: "DeepSeek V4 Flash 284B-A13B",
        description: "Frontier MoE model with a very large weight footprint and low active compute.",
        repository: "unsloth/DeepSeek-V4-Flash-GGUF",
        formats: &["UD-Q8_K_XL"],
        contexts: PRODUCT_CONTEXTS,
        license: "mit",
        quality_score: 61.8,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "nemotron-3-ultra-550b-a55b",
        display_name: "NVIDIA Nemotron 3 Ultra 550B-A55B",
        description: "Frontier workstation/server MoE model for exceptionally high-memory systems.",
        repository: "unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF",
        formats: &["MXFP4_MOE"],
        contexts: PRODUCT_CONTEXTS,
        license: "openmdw-1.1",
        quality_score: 53.9,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
    CatalogModel {
        id: "glm-5.2",
        display_name: "GLM 5.2 753B-A40B",
        description: "Largest catalog tier, intended only for exceptionally high-memory systems.",
        repository: "unsloth/GLM-5.2-GGUF",
        formats: &["UD-Q4_K_XL"],
        contexts: PRODUCT_CONTEXTS,
        license: "mit",
        quality_score: 77.9,
        quality_score_provenance: "measured_terminal_bench_2.1",
        quality_evidence: QUANT_EVIDENCE,
    },
];

fn fidelity(declaration_id: &str, format: &str) -> (u32, bool) {
    if declaration_id.starts_with("gemma-4-") {
        return (58, true);
    }
    if declaration_id == "nemotron-3-super-120b-a12b"
        || declaration_id == "nemotron-3-ultra-550b-a55b"
    {
        return (58, true);
    }
    if declaration_id == "glm-5.2" {
        return (70, false);
    }
    let rank = if format.contains("Q8") {
        80
    } else if format.contains("Q6") {
        60
    } else if format.contains("Q5") {
        50
    } else {
        40
    };
    (rank, false)
}

pub struct NativeRecommendableCatalog {
    models: Arc<ModelManager>,
    hugging_face: Arc<dyn HuggingFaceModelCatalog>,
}

impl NativeRecommendableCatalog {
    #[must_use]
    pub fn new(models: Arc<ModelManager>, hugging_face: Arc<dyn HuggingFaceModelCatalog>) -> Self {
        Self {
            models,
            hugging_face,
        }
    }

    async fn resolve_model(
        &self,
        declaration: CatalogModel,
        format: &str,
        snapshot: &HuggingFaceRepositorySnapshot,
    ) -> Result<RecommendableModel, InventoryError> {
        let evidence = serde_json::to_string(&(
            CATALOG_RESOLUTION_REVISION,
            declaration,
            format,
            snapshot.repository.as_str(),
            snapshot.commit.as_str(),
        ))
        .map_err(|error| InventoryError::Internal(error.to_string()))?;
        if let Some(model) = self
            .models
            .cache
            .read_index(ModelIndexKind::RecommendableModelCatalog, &evidence)
        {
            return Ok(model);
        }
        let selector = format.to_ascii_lowercase();
        let mut matches = snapshot
            .gguf_files
            .iter()
            .filter(|file| {
                let path = file.path.to_string_lossy().to_ascii_lowercase();
                let basename = path.rsplit('/').next().unwrap_or(path.as_str());
                path.contains(&selector)
                    && !basename.starts_with("mmproj-")
                    && !basename.contains("imatrix")
                    && (!is_later_shard(basename) || is_first_shard(basename))
            })
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(InventoryError::Integrity(format!(
                "{} format {format} resolved to {} primary files",
                declaration.repository,
                matches.len()
            )));
        }
        let primary = matches.remove(0);
        let prepared = self
            .models
            .prepare_preview(&ModelPreviewSource {
                repository: snapshot.repository.clone(),
                revision: snapshot.commit.clone(),
                primary_gguf: primary.path.clone(),
                additional_components: Vec::new(),
            })
            .await?;
        let package = package_from_resolved(&prepared.model)?;
        let capabilities = model_capabilities(&prepared.model.model.properties);
        let (fidelity_rank, quantization_aware) = fidelity(declaration.id, format);
        let target_id = offering_target_id(&[&package.id]);
        let model = RecommendableModel {
            id: RecommendableModelId(format!("{}:{format}", declaration.id)),
            checkpoint_id: declaration.id.to_owned(),
            target_id,
            target: ModelOfferingTarget::Package { package },
            eligible_serving_profiles: declaration
                .contexts
                .iter()
                .map(|context_length| ServingProfile {
                    context_length: *context_length,
                    parallel_sequences: 1,
                })
                .collect(),
            display_name: declaration.display_name.to_owned(),
            description: declaration.description.to_owned(),
            license: snapshot
                .license
                .clone()
                .filter(|license| license != "other")
                .unwrap_or_else(|| declaration.license.to_owned()),
            capabilities,
            quality_score: declaration.quality_score,
            quality_score_provenance: declaration.quality_score_provenance.to_owned(),
            fidelity_rank,
            quantization_aware,
            quality_evidence: declaration
                .quality_evidence
                .iter()
                .map(|evidence| (*evidence).to_owned())
                .collect(),
        };
        self.models
            .cache
            .write_index(ModelIndexKind::RecommendableModelCatalog, &evidence, &model);
        Ok(model)
    }
}

fn is_first_shard(name: &str) -> bool {
    name.rsplit_once("-00001-of-")
        .is_some_and(|(_, suffix)| suffix.ends_with(".gguf"))
}

fn is_later_shard(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".gguf") else {
        return false;
    };
    stem.rsplit_once("-of-")
        .and_then(|(prefix, count)| prefix.rsplit_once('-').map(|(_, index)| (index, count)))
        .is_some_and(|(index, count)| {
            index.len() == 5
                && count.len() == 5
                && index.bytes().all(|byte| byte.is_ascii_digit())
                && count.bytes().all(|byte| byte.is_ascii_digit())
                && index != "00001"
        })
}

impl RecommendableModelCatalogProvider for NativeRecommendableCatalog {
    fn catalog(&self) -> BoxFuture<'_, Result<RecommendableModelCatalog, InventoryError>> {
        Box::pin(async move {
            let repositories = CATALOG_MODELS
                .iter()
                .map(|declaration| declaration.repository.to_owned())
                .fold(Vec::new(), |mut unique, repository| {
                    if !unique.contains(&repository) {
                        unique.push(repository);
                    }
                    unique
                });
            let snapshots = stream::iter(repositories)
                .map(|repository| async move {
                    let result = self
                        .hugging_face
                        .resolve(HuggingFaceRepositoryRequest {
                            repository: repository.clone(),
                            revision: "main".to_owned(),
                        })
                        .await;
                    (repository, result)
                })
                .buffer_unordered(12)
                .collect::<Vec<_>>()
                .await;
            let mut resolved_snapshots = BTreeMap::new();
            let mut snapshot_failures = BTreeMap::new();
            for (repository, result) in snapshots {
                match result {
                    Ok(snapshot) => {
                        resolved_snapshots.insert(repository, snapshot);
                    }
                    Err(error) => {
                        snapshot_failures.insert(repository, error.to_string());
                    }
                }
            }
            let requests = CATALOG_MODELS
                .iter()
                .flat_map(|declaration| {
                    declaration
                        .formats
                        .iter()
                        .map(move |format| (*declaration, (*format).to_owned()))
                })
                .enumerate()
                .collect::<Vec<_>>();
            let resolved_snapshots = &resolved_snapshots;
            let snapshot_failures = &snapshot_failures;
            let mut resolved = stream::iter(requests)
                .map(|(index, (declaration, format))| async move {
                    let result = match resolved_snapshots.get(declaration.repository) {
                        Some(snapshot) => self.resolve_model(declaration, &format, snapshot).await,
                        None => Err(InventoryError::Io(
                            snapshot_failures
                                .get(declaration.repository)
                                .cloned()
                                .unwrap_or_else(|| {
                                    format!(
                                        "repository {} was not resolved",
                                        declaration.repository
                                    )
                                }),
                        )),
                    };
                    (index, declaration, format, result)
                })
                .buffer_unordered(12)
                .collect::<Vec<_>>()
                .await;
            resolved.sort_by_key(|(index, ..)| *index);
            let mut models = Vec::new();
            let mut diagnostics = Vec::new();
            for (_, declaration, format, result) in resolved {
                match result {
                    Ok(model) => models.push(model),
                    Err(error) => diagnostics.push(CatalogDiagnostic {
                        entry_id: Some(RecommendableModelId(format!(
                            "{}:{format}",
                            declaration.id
                        ))),
                        failure: ModelFailure {
                            code: "catalog_resolution_failed".to_owned(),
                            message: error.to_string(),
                            retryable: true,
                        },
                    }),
                }
            }
            Ok(RecommendableModelCatalog {
                models,
                diagnostics,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_selector_distinguishes_first_and_later_shards() {
        assert!(is_first_shard("model-00001-of-00003.gguf"));
        assert!(!is_later_shard("model-00001-of-00003.gguf"));
        assert!(is_later_shard("model-00002-of-00003.gguf"));
        assert!(!is_later_shard("model.gguf"));
    }
}
