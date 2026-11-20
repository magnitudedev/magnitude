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

/// The most bytes the file is read as.
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;

/// What this process changed of the requests since its last persist, applied to what the file
/// holds then ([`RequestedDownloads::persist`]): a request it added, or one it ended.
#[derive(Debug, Default)]
struct Pending {
    added: Vec<RequestedDownload>,
    ended: Vec<RequestedDownload>,
}

/// Every requested download that has not ended, persisted in the store so a restart resumes it.
/// A request ends when its download completes, is cancelled, or has its failure dismissed, or when
/// its model is removed. Requests loaded at startup have no download until they are admitted again.
///
/// Every service process over one store keeps its own requests in memory, and each persist
/// applies only what this process changed to what the file holds, under the file's record lock
/// (`seismic_io::Store::lock`), so no process drops another's requests.
#[derive(Debug, Default)]
pub(crate) struct RequestedDownloads {
    entries: Vec<Entry>,
    pending: Pending,
}

/// The requests a file's `bytes` hold; a request that does not read is dropped, a file that does
/// not is none.
fn decode(bytes: &[u8]) -> Vec<RequestedDownload> {
    let Ok(Value::Object(document)) = serde_json::from_slice::<Value>(bytes) else {
        tracing::warn!("discarding malformed requested downloads");
        return Vec::new();
    };
    let Some(Value::Array(requests)) = document.get("requests") else {
        tracing::warn!("discarding requested downloads with an invalid shape");
        return Vec::new();
    };
    requests
        .iter()
        .filter_map(|request| match serde_json::from_value(request.clone()) {
            Ok(request) => Some(request),
            Err(error) => {
                tracing::warn!(%error, "discarding an unreadable requested download");
                None
            }
        })
        .collect()
}

impl RequestedDownloads {
    pub(crate) fn load(root: &Path) -> Self {
        let requests = match crate::store_fs::model_store(root).and_then(|store| {
            store.read(Path::new(FILE_NAME), MAX_FILE_BYTES).map_err(|error| {
                magnitude_service_contracts::InventoryError::Io(error.to_string())
            })
        }) {
            Ok(Some(bytes)) => decode(&bytes),
            Ok(None) => Vec::new(),
            Err(error) => {
                tracing::warn!(%error, "discarding unreadable requested downloads");
                Vec::new()
            }
        };
        Self {
            entries: requests
                .into_iter()
                .map(|request| Entry {
                    download: None,
                    request,
                })
                .collect(),
            pending: Pending::default(),
        }
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
    /// The request joins the file at the next persist, whatever this process knew of it: another
    /// process may have ended it there since this one loaded it.
    pub(crate) fn admitted(&mut self, request: RequestedDownload, download: ModelDownloadId) {
        self.pending.ended.retain(|ended| *ended != request);
        if !self.pending.added.contains(&request) {
            self.pending.added.push(request.clone());
        }
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

    /// Keeps the entries `keep` keeps, the rest ended; whether one ended.
    fn retain(&mut self, keep: impl Fn(&Entry) -> bool) -> bool {
        let (kept, ended): (Vec<_>, Vec<_>) = std::mem::take(&mut self.entries).into_iter().partition(&keep);
        self.entries = kept;
        for Entry { request, .. } in &ended {
            self.pending.added.retain(|added| added != request);
            self.pending.ended.push(request.clone());
        }
        !ended.is_empty()
    }

    /// Apply what this process changed since its last persist to what the file holds, under the
    /// file's record lock: the requests it ended leave it, and those it added join it. A failure
    /// keeps the change for the next persist.
    pub(crate) fn persist(&mut self, root: &Path) -> std::io::Result<()> {
        if self.pending.added.is_empty() && self.pending.ended.is_empty() {
            return Ok(());
        }
        let store = crate::store_fs::model_store(root).map_err(std::io::Error::other)?;
        let record = store.lock(Path::new(FILE_NAME)).map_err(std::io::Error::other)?;
        let mut requests = record.read(MAX_FILE_BYTES).map_err(std::io::Error::other)?.map(|bytes| decode(&bytes)).unwrap_or_default();
        requests.retain(|request| !self.pending.ended.contains(request));
        for added in &self.pending.added {
            if !requests.contains(added) {
                requests.push(added.clone());
            }
        }
        let bytes = serde_json::to_vec_pretty(&Stored {
            requests: requests.iter().collect(),
        })
        .map_err(std::io::Error::other)?;
        record.write(&bytes, seismic_io::Durability::Durable).map_err(std::io::Error::other)?;
        self.pending = Pending::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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

    /// Two service processes over one store, each adding and ending requests of its own: the file
    /// holds what both changed, neither dropping the other's.
    #[test]
    fn two_processes_keep_each_others_requests() {
        let root = tempfile::tempdir().unwrap();
        let (kept, ended, theirs) = (
            catalog("qwen3.5-4b:gguf:q4"),
            catalog("qwen3.5-9b:gguf:q4"),
            catalog("gemma4-4b:gguf:q4"),
        );
        let mut first = RequestedDownloads::default();
        first.admitted(kept.clone(), ModelDownloadId("model_download_a".to_owned()));
        first.admitted(ended.clone(), ModelDownloadId("model_download_b".to_owned()));
        first.persist(root.path()).unwrap();
        // Another process, started before the first persisted, knows none of its requests.
        let mut second = RequestedDownloads::default();
        second.admitted(theirs.clone(), ModelDownloadId("model_download_c".to_owned()));
        second.persist(root.path()).unwrap();
        assert!(first.forget_model(&ended.model));
        first.persist(root.path()).unwrap();
        assert_eq!(RequestedDownloads::load(root.path()).unresumed(), vec![kept.clone(), theirs.clone()]);
        // A persist that changes nothing leaves the file as the others left it.
        let mut idle = RequestedDownloads::default();
        idle.persist(root.path()).unwrap();
        assert_eq!(RequestedDownloads::load(root.path()).unresumed(), vec![kept, theirs]);
    }
}
