use std::fs;
use std::path::Path;

use magnitude_service_contracts::models::{
    CatalogComponent, CustomModelPart, ModelDownloadId, ModelId,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const FILE_NAME: &str = "requested-downloads.json";

/// A download the user asked for, as intent only: a model and which part of it. It is admitted
/// again by resolving that intent against what the service holds then (the catalog, the
/// local-models configuration, the files here), never from a stored wire contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct RequestedDownload {
    pub(crate) model: ModelId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) part: Option<RequestedPart>,
}

/// A catalog model's component, or a custom model's part (see [`CustomModelPart`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub(crate) enum RequestedPart {
    Projector,
    Drafter,
    Update,
}

impl From<CatalogComponent> for RequestedPart {
    fn from(component: CatalogComponent) -> Self {
        match component {
            CatalogComponent::Projector => Self::Projector,
            CatalogComponent::Drafter => Self::Drafter,
        }
    }
}

impl From<CustomModelPart> for RequestedPart {
    fn from(part: CustomModelPart) -> Self {
        match part {
            CustomModelPart::Drafter => Self::Drafter,
            CustomModelPart::Update => Self::Update,
        }
    }
}

impl RequestedDownload {
    fn concerns(&self, model: &ModelId) -> bool {
        self.model == *model
    }
}

#[derive(Debug, Clone)]
struct Entry {
    download: Option<ModelDownloadId>,
    request: RequestedDownload,
}

#[derive(Serialize)]
struct Stored<'a> {
    requests: Vec<&'a RequestedDownload>,
}

/// Every requested download that has not ended, persisted in the store so a restart resumes it.
/// A request ends when its download completes, is cancelled, or has its failure dismissed, or when
/// its model is removed. Requests loaded at startup have no download until they are admitted again.
#[derive(Debug, Default)]
pub(crate) struct RequestedDownloads {
    entries: Vec<Entry>,
}

impl RequestedDownloads {
    pub(crate) fn load(root: &Path) -> Self {
        let Ok(bytes) = fs::read(root.join(FILE_NAME)) else {
            return Self::default();
        };
        let Ok(Value::Object(document)) = serde_json::from_slice::<Value>(&bytes) else {
            tracing::warn!("discarding malformed requested downloads");
            return Self::default();
        };
        let Some(Value::Array(requests)) = document.get("requests") else {
            tracing::warn!("discarding requested downloads with an invalid shape");
            return Self::default();
        };
        let entries = requests
            .iter()
            .filter_map(|request| match serde_json::from_value(request.clone()) {
                Ok(request) => Some(Entry {
                    download: None,
                    request,
                }),
                Err(error) => {
                    tracing::warn!(%error, "discarding an unreadable requested download");
                    None
                }
            })
            .collect();
        Self { entries }
    }

    /// Requests loaded at startup that have not been admitted again.
    pub(crate) fn unresumed(&self) -> Vec<RequestedDownload> {
        self.entries
            .iter()
            .filter(|entry| entry.download.is_none())
            .map(|entry| entry.request.clone())
            .collect()
    }

    pub(crate) fn is_unresumed(&self, request: &RequestedDownload) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.download.is_none() && entry.request == *request)
    }

    /// Records `download` as the current attempt at `request`; a retry replaces the earlier one.
    pub(crate) fn admitted(&mut self, request: RequestedDownload, download: ModelDownloadId) {
        match self
            .entries
            .iter_mut()
            .find(|entry| entry.request == request)
        {
            Some(entry) => entry.download = Some(download),
            None => self.entries.push(Entry {
                download: Some(download),
                request,
            }),
        }
    }

    /// Forgets a request loaded at startup that was not admitted again; whether there was one.
    pub(crate) fn forget_unresumed(&mut self, request: &RequestedDownload) -> bool {
        self.retain(|entry| entry.download.is_some() || entry.request != *request)
    }

    /// Keeps only requests not yet admitted again and those whose download is `live`.
    pub(crate) fn retain_live(&mut self, live: impl Fn(&ModelDownloadId) -> bool) -> bool {
        self.retain(|entry| entry.download.as_ref().is_none_or(&live))
    }

    /// Forgets every request for a removed model; whether there was one.
    pub(crate) fn forget_model(&mut self, model: &ModelId) -> bool {
        self.retain(|entry| !entry.request.concerns(model))
    }

    fn retain(&mut self, keep: impl Fn(&Entry) -> bool) -> bool {
        let before = self.entries.len();
        self.entries.retain(keep);
        self.entries.len() != before
    }

    pub(crate) fn persist(&self, root: &Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(&Stored {
            requests: self.entries.iter().map(|entry| &entry.request).collect(),
        })
        .map_err(std::io::Error::other)?;
        crate::store_fs::replace_file_sync(root, FILE_NAME, &bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog(model: &str) -> RequestedDownload {
        RequestedDownload {
            model: model.parse().unwrap(),
            part: Some(CatalogComponent::Projector.into()),
        }
    }

    #[test]
    fn requests_are_stored_as_a_model_and_a_part() {
        let root = tempfile::tempdir().unwrap();
        let mut requests = RequestedDownloads::default();
        let added = "hf:unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf";
        for (index, request) in [
            RequestedDownload {
                model: "qwen3.5-4b:gguf:q4".parse().unwrap(),
                part: None,
            },
            RequestedDownload {
                model: added.parse().unwrap(),
                part: Some(CustomModelPart::Drafter.into()),
            },
        ]
        .into_iter()
        .enumerate()
        {
            requests.admitted(request, ModelDownloadId(format!("model_download_{index}")));
        }
        requests.persist(root.path()).unwrap();

        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(root.path().join(FILE_NAME)).unwrap())
                .unwrap(),
            serde_json::json!({ "requests": [
                { "model": "qwen3.5-4b:gguf:q4" },
                { "model": added, "part": "Drafter" },
            ] })
        );
    }

    #[test]
    fn requests_round_trip_through_the_store_and_await_admission() {
        let root = tempfile::tempdir().unwrap();
        let mut requests = RequestedDownloads::default();
        requests.admitted(
            catalog("qwen3.5-4b:gguf:q4"),
            ModelDownloadId("model_download_a".to_owned()),
        );
        requests.persist(root.path()).unwrap();

        let loaded = RequestedDownloads::load(root.path());
        assert_eq!(loaded.unresumed(), vec![catalog("qwen3.5-4b:gguf:q4")]);
    }

    #[test]
    fn an_unreadable_request_is_dropped_without_losing_the_others() {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join(FILE_NAME),
            serde_json::to_vec(&serde_json::json!({
                "requests": [
                    { "model": "qwen3.5-4b:gguf:q4", "part": "Unknown" },
                    serde_json::to_value(catalog("qwen3.5-4b:gguf:q4")).unwrap(),
                ],
            }))
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            RequestedDownloads::load(root.path()).unresumed(),
            vec![catalog("qwen3.5-4b:gguf:q4")]
        );
    }
}
