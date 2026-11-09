//! Which lab develops a model, from the catalog's lab table. A catalog model names its lab. Any
//! other model is attributed by what it carries: the organization of its Hugging Face repository,
//! else the organization of a repository its GGUF declares as a base model, else the lab of the
//! catalog model whose shape it has. Without any of them its developer is unknown.

use std::collections::BTreeMap;

use magnitude_service_contracts::models::{
    CatalogBaseId, LabId, ModelDeveloper, ModelId, ParsedModelId,
};

pub struct ModelDevelopers {
    labs: Vec<Lab>,
    catalog_models: BTreeMap<CatalogBaseId, LabId>,
}

struct Lab {
    developer: ModelDeveloper,
    organizations: Vec<String>,
}

/// What a model that is not a catalog model shows of its origin. `base_repositories` are the
/// `owner/repository` names its GGUF declares as base models.
#[derive(Clone, Copy, Default)]
pub struct DeveloperEvidence<'a> {
    pub repository: Option<&'a str>,
    pub base_repositories: &'a [String],
    pub variant_of: Option<&'a ModelId>,
}

impl ModelDevelopers {
    /// `labs` pairs each developer with its Hugging Face organizations; `catalog_models` names each
    /// catalog model's lab, which `labs` holds.
    pub(crate) fn new(
        labs: impl IntoIterator<Item = (ModelDeveloper, Vec<String>)>,
        catalog_models: BTreeMap<CatalogBaseId, LabId>,
    ) -> Self {
        Self {
            labs: labs
                .into_iter()
                .map(|(developer, organizations)| Lab {
                    developer,
                    organizations: organizations
                        .iter()
                        .map(|organization| organization.to_lowercase())
                        .collect(),
                })
                .collect(),
            catalog_models,
        }
    }

    #[must_use]
    pub fn catalog_model(&self, model_id: &CatalogBaseId) -> Option<ModelDeveloper> {
        let lab = self.catalog_models.get(model_id)?;
        self.labs
            .iter()
            .find(|candidate| candidate.developer.lab == *lab)
            .map(|lab| lab.developer.clone())
    }

    #[must_use]
    pub fn resolve(&self, evidence: DeveloperEvidence<'_>) -> Option<ModelDeveloper> {
        evidence
            .repository
            .and_then(|repository| self.repository(repository))
            .or_else(|| {
                evidence
                    .base_repositories
                    .iter()
                    .find_map(|repository| self.repository(repository))
            })
            .or_else(|| match evidence.variant_of?.parsed() {
                ParsedModelId::Catalog { base_id, .. } => self.catalog_model(&base_id),
                _ => None,
            })
    }

    fn repository(&self, repository: &str) -> Option<ModelDeveloper> {
        let (organization, _) = repository.split_once('/')?;
        let organization = organization.to_lowercase();
        self.labs
            .iter()
            .find(|lab| lab.organizations.contains(&organization))
            .map(|lab| lab.developer.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn developers() -> &'static ModelDevelopers {
        crate::catalog::model_developers()
    }

    fn lab(developer: Option<ModelDeveloper>) -> Option<String> {
        developer.map(|developer| developer.lab.0)
    }

    #[test]
    fn every_catalog_model_has_its_lab() {
        let qwen = CatalogBaseId::new("qwen3.5-4b").unwrap();
        assert_eq!(
            developers().catalog_model(&qwen),
            Some(ModelDeveloper {
                lab: LabId("alibaba".to_owned()),
                name: "Alibaba".to_owned(),
                country: "CN".to_owned(),
            })
        );
    }

    #[test]
    fn a_repository_organization_decides_first_ignoring_case() {
        let gemma = "gemma-4-31b-it-qat:gguf:q4".parse::<ModelId>().unwrap();
        let bases = ["google/gemma-4-31b-it".to_owned()];
        assert_eq!(
            lab(developers().resolve(DeveloperEvidence {
                repository: Some("qwen/Qwen3.5-4B"),
                base_repositories: &bases,
                variant_of: Some(&gemma),
            })),
            Some("alibaba".to_owned())
        );
    }

    #[test]
    fn a_quant_uploader_falls_back_to_its_base_model_then_its_catalog_shape() {
        let gemma = "gemma-4-31b-it-qat:gguf:q4".parse::<ModelId>().unwrap();
        let bases = ["someone/merge".to_owned(), "nvidia/nemotron-3.5".to_owned()];
        assert_eq!(
            lab(developers().resolve(DeveloperEvidence {
                repository: Some("unsloth/Nemotron-GGUF"),
                base_repositories: &bases,
                variant_of: Some(&gemma),
            })),
            Some("nvidia".to_owned())
        );
        assert_eq!(
            lab(developers().resolve(DeveloperEvidence {
                repository: Some("unsloth/gemma-GGUF"),
                base_repositories: &[],
                variant_of: Some(&gemma),
            })),
            Some("google".to_owned())
        );
        assert_eq!(
            lab(developers().resolve(DeveloperEvidence {
                repository: Some("unsloth/unknown-GGUF"),
                ..DeveloperEvidence::default()
            })),
            None
        );
    }
}
