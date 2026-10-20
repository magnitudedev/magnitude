use std::fs;
use std::path::Path;

use magnitude_service_contracts::models::{
    CatalogComponent, ModelDownloadId, ModelId, StartModelDownloadRequest,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const FILE_NAME: &str = "requested-downloads.json";

/// A download the user asked for, as it was asked: a catalog installation or one of its
/// components is admitted again through the catalog, so it resolves against the current catalog
/// and files; any other download is its exact bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub(crate) enum RequestedDownload {
    CatalogModel {
        model: ModelId,
    },
    CatalogComponent {
        model: ModelId,
        component: CatalogComponent,
    },
    Bundle {
        request: StartModelDownloadRequest,
    },
}

impl RequestedDownload {
    fn concerns(&self, model: &ModelId) -> bool {
        match self {
            Self::CatalogModel { model: requested }
            | Self::CatalogComponent {
                model: requested, ..
            } => requested == model,
            Self::Bundle { request } => request.for_model.as_ref() == Some(model),
        }
    }

    pub(crate) fn bundle(request: &StartModelDownloadRequest) -> Self {
        Self::Bundle {
            request: request.clone(),
        }
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
        RequestedDownload::CatalogComponent {
            model: model.parse().unwrap(),
            component: CatalogComponent::Projector,
        }
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
                    { "_tag": "Unknown" },
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
