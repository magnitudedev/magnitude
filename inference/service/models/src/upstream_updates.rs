//! Updates for models added from Hugging Face. An added model is pinned to a repository commit;
//! it has an update when the repository's `main` has moved to a commit where its target file is
//! still present with other content. Checks run in the background, each result is kept for
//! [`CHECK_INTERVAL`], and a check that cannot reach the Hub keeps the last result and says
//! nothing. Once an update has landed and the configuration pins its commit, Magnitude's copy at
//! the commit it replaced is removed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use magnitude_service_contracts::{InventoryError, ModelSource};
use magnitude_service_contracts::models::{
    AddedModel, CustomModel, CustomModelOrigin, InstalledModelPackages as _, ModelFileRole,
    ModelPackageId, ModelPackageInstallationOrigin, PackageValidation,
};
use tokio::sync::Notify;

use crate::download::HubFileContent;
use crate::inventory::{InstalledPackageSnapshot, ManagedModelStore};
use crate::model_domains::ModelDomainResolver;

pub(crate) const CHECK_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);
const RETRY_INTERVAL: Duration = Duration::from_secs(30 * 60);
const SWEEP_INTERVAL: Duration = Duration::from_secs(10 * 60);
const CHECK_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct UpstreamKey {
    repository: String,
    commit: String,
    target: Vec<String>,
}

impl UpstreamKey {
    fn of(added: &AddedModel) -> Self {
        Self {
            repository: added.repository.as_str().to_owned(),
            commit: added.revision.clone(),
            target: crate::candidate_files::split_set(added.path.as_str()),
        }
    }
}

struct UpstreamEntry {
    next_check: Instant,
    update_commit: Option<String>,
    confirmed: bool,
}

/// The last upstream check of each added model, by repository, pinned commit and target.
pub(crate) struct UpstreamUpdates {
    entries: Mutex<BTreeMap<UpstreamKey, UpstreamEntry>>,
    wake: Arc<Notify>,
}

impl UpstreamUpdates {
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(BTreeMap::new()),
            wake: Arc::new(Notify::new()),
        }
    }

    /// Asks the background task to check due added models and remove replaced copies now.
    pub(crate) fn wake(&self) {
        self.wake.notify_one();
    }

    /// Sets each added model's `update_commit` from the last check of it.
    pub(crate) fn annotate<'a>(
        &self,
        models: impl IntoIterator<Item = &'a mut CustomModel>,
        added: &[AddedModel],
    ) {
        let Ok(entries) = self.entries.lock() else {
            return;
        };
        for model in models {
            if !matches!(model.origin, CustomModelOrigin::Added { .. }) {
                continue;
            }
            model.update_commit = added
                .iter()
                .find(|added| added.model_id() == model.model_id)
                .and_then(|added| entries.get(&UpstreamKey::of(added)))
                .and_then(|entry| entry.update_commit.clone());
        }
    }

    /// Checks every added model whose last result is missing or due, and forgets the ones no
    /// longer added. Returns whether any model's update changed.
    pub(crate) async fn check_due(&self, store: &ManagedModelStore, added: &[AddedModel]) -> bool {
        let keys = added.iter().map(UpstreamKey::of).collect::<BTreeSet<_>>();
        let due = {
            let Ok(mut entries) = self.entries.lock() else {
                return false;
            };
            entries.retain(|key, _| keys.contains(key));
            let now = Instant::now();
            keys.into_iter()
                .filter(|key| entries.get(key).is_none_or(|entry| entry.next_check <= now))
                .collect::<Vec<_>>()
        };
        let mut changed = false;
        for key in due {
            let checked = tokio::time::timeout(
                CHECK_TIMEOUT,
                upstream_update(store, &key.repository, &key.commit, &key.target),
            )
            .await
            .unwrap_or_else(|_| Err("the Hugging Face check timed out".to_owned()));
            let Ok(mut entries) = self.entries.lock() else {
                return changed;
            };
            let previous = entries
                .get(&key)
                .and_then(|entry| entry.update_commit.clone());
            let confirmed_before = entries.get(&key).is_some_and(|entry| entry.confirmed);
            let (next_check, update_commit, confirmed) = match checked {
                Ok(update_commit) => (Instant::now() + CHECK_INTERVAL, update_commit, true),
                Err(error) => {
                    tracing::debug!(repository = %key.repository, %error, "upstream update check skipped");
                    (
                        Instant::now() + RETRY_INTERVAL,
                        previous.clone(),
                        confirmed_before,
                    )
                }
            };
            changed |= update_commit != previous;
            entries.insert(
                key,
                UpstreamEntry {
                    next_check,
                    update_commit,
                    confirmed,
                },
            );
        }
        changed
    }

    /// For an added model whose repository has been reached at least once, the newer commit an
    /// update may be fetching, if any; nothing while it never has.
    fn confirmed_update(&self, added: &AddedModel) -> Option<Option<String>> {
        let entries = self.entries.lock().ok()?;
        let entry = entries.get(&UpstreamKey::of(added))?;
        entry.confirmed.then(|| entry.update_commit.clone())
    }
}

/// The commit `main` has moved to when the target of a model pinned at `commit` has other content
/// there. A target file that `main` no longer has is no update: the model cannot follow it.
pub(crate) async fn upstream_update(
    store: &ManagedModelStore,
    repository: &str,
    commit: &str,
    target: &[String],
) -> Result<Option<String>, String> {
    let current = store.hub_revision_files(repository, "main").await?;
    if current.commit == commit {
        return Ok(None);
    }
    let pinned = store.hub_revision_files(repository, commit).await?;
    Ok(target_changed(&pinned.files, &current.files, target).then_some(current.commit))
}

fn target_changed(
    pinned: &BTreeMap<String, HubFileContent>,
    current: &BTreeMap<String, HubFileContent>,
    target: &[String],
) -> bool {
    !target.is_empty()
        && target.iter().all(|path| current.contains_key(path))
        && target
            .iter()
            .any(|path| pinned.get(path) != current.get(path))
}

/// Magnitude's copies of added targets at a commit the configuration no longer pins, once the
/// copy at the pinned commit is installed and valid: the copies an update replaced. `known` gives
/// an added model's update commit as the Hub last confirmed it; a copy at that commit is an update
/// still waiting to be pinned and stays, and nothing is removed for a model the Hub has not
/// confirmed since the service started.
pub(crate) fn replaced_added_packages(
    snapshot: &InstalledPackageSnapshot,
    added: &[AddedModel],
    known: impl Fn(&AddedModel) -> Option<Option<String>>,
) -> Vec<ModelPackageId> {
    fn managed_target(
        record: &crate::inventory::InstalledPackageRecord,
    ) -> Option<(&str, &str, &Path)> {
        if record.installed.origin != ModelPackageInstallationOrigin::Magnitude {
            return None;
        }
        let ModelSource::HuggingFace {
            repository, commit, ..
        } = &record.model.source
        else {
            return None;
        };
        let primary = record
            .installed
            .package
            .files
            .iter()
            .filter(|file| file.role == ModelFileRole::Weights)
            .min_by(|left, right| left.path.cmp(&right.path))?;
        Some((repository.as_str(), commit.as_str(), primary.path.as_path()))
    }
    let mut kept = BTreeMap::<(&str, &Path), (&str, Option<String>)>::new();
    for added in added {
        let Some(update) = known(added) else {
            continue;
        };
        kept.insert(
            (added.repository.as_str(), Path::new(added.path.as_str())),
            (added.revision.as_str(), update),
        );
    }
    let landed = snapshot
        .records
        .values()
        .filter(|record| matches!(record.installed.validation, PackageValidation::Valid))
        .filter_map(managed_target)
        .filter(|(repository, commit, path)| {
            kept.get(&(*repository, *path))
                .is_some_and(|(pinned, _)| pinned == commit)
        })
        .map(|(repository, _, path)| (repository, path))
        .collect::<BTreeSet<_>>();
    snapshot
        .records
        .values()
        .filter(|record| {
            managed_target(record).is_some_and(|(repository, commit, path)| {
                landed.contains(&(repository, path))
                    && kept
                        .get(&(repository, path))
                        .is_some_and(|(pinned, update)| {
                            *pinned != commit && update.as_deref() != Some(commit)
                        })
            })
        })
        .map(|record| record.installed.package.id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

impl ModelDomainResolver {
    /// The commit an added model's update fetches: the one the last check confirmed, or, before
    /// any check has reached the Hub (just after a restart), the one the Hub reports now.
    pub async fn added_model_update(
        &self,
        added: &AddedModel,
    ) -> Result<Option<String>, InventoryError> {
        match self.upstream.confirmed_update(added) {
            Some(update) => Ok(update),
            None => upstream_update(
                &self.inventory,
                added.repository.as_str(),
                &added.revision,
                &UpstreamKey::of(added).target,
            )
            .await
            .map_err(InventoryError::Upstream),
        }
    }
}

/// Runs the checks and the removal of replaced copies in the background for as long as the
/// resolver lives: after every configuration or inventory change, and every few minutes so a
/// result is renewed once it is due.
pub(crate) fn start(resolver: &Arc<ModelDomainResolver>) {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let wake = Arc::clone(&resolver.upstream.wake);
    let resolver = Arc::downgrade(resolver);
    runtime.spawn(async move {
        while let Some(resolver) = Weak::upgrade(&resolver) {
            sweep(&resolver).await;
            drop(resolver);
            tokio::select! {
                () = wake.notified() => {}
                () = tokio::time::sleep(SWEEP_INTERVAL) => {}
            }
        }
    });
}

async fn sweep(resolver: &ModelDomainResolver) {
    let Ok(configuration) = resolver.inventory.local_models_configuration() else {
        return;
    };
    if resolver
        .upstream
        .check_due(&resolver.inventory, &configuration.added)
        .await
    {
        resolver.local_models_changed();
    }
    let replaced = resolver
        .inventory
        .installed_packages
        .read()
        .map(|snapshot| {
            replaced_added_packages(&snapshot, &configuration.added, |added| {
                resolver.upstream.confirmed_update(added)
            })
        })
        .unwrap_or_default();
    for package_id in replaced {
        match resolver.inventory.remove_installed(&package_id).await {
            Ok(removal) => tracing::info!(
                package_id = %package_id.0,
                freed_bytes = removal.freed_bytes,
                "removed the copy an added model's update replaced"
            ),
            Err(error) => tracing::warn!(
                package_id = %package_id.0,
                %error,
                "the copy an added model's update replaced was not removed"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::InventoryConfig;
    use magnitude_service_contracts::models::{
        CustomModelState, HuggingFaceArtifactSelector, HuggingFaceRepositoryId, LocalModels as _,
        LocalModelsConfiguration, RecommendableModelCatalog,
    };

    struct NoRemoval;

    impl magnitude_service_contracts::models::CatalogPackageRemover for NoRemoval {
        fn remove_catalog_packages(
            &self,
            _package_ids: Vec<ModelPackageId>,
        ) -> futures_util::future::BoxFuture<'_, Result<u64, magnitude_service_contracts::InventoryError>>
        {
            Box::pin(async { Ok(0) })
        }
    }

    const REPOSITORY: &str = "owner/Model-GGUF";
    const TARGET: &str = "Model-Q4_K_M.gguf";

    fn commit(fill: char) -> String {
        fill.to_string().repeat(40)
    }

    fn revision_json(commit: &str, target_sha: Option<&str>, projector_sha: &str) -> String {
        let mut siblings = vec![format!(
            r#"{{"rfilename":"mmproj.gguf","size":10,"lfs":{{"sha256":"{projector_sha}","size":10}}}}"#
        )];
        if let Some(sha) = target_sha {
            siblings.push(format!(
                r#"{{"rfilename":"{TARGET}","size":100,"lfs":{{"sha256":"{sha}","size":100}}}}"#
            ));
        }
        format!(
            r#"{{"sha":"{commit}","siblings":[{}]}}"#,
            siblings.join(",")
        )
    }

    /// A Hub whose repository metadata answers for `main` and each pinned commit in `revisions`.
    fn serve_hub(revisions: Vec<(String, String)>) -> String {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("hub listener");
        let endpoint = format!("http://{}", listener.local_addr().expect("hub address"));
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut connection) = connection else {
                    return;
                };
                let mut reader = BufReader::new(connection.try_clone().expect("hub stream"));
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).is_err() {
                    continue;
                }
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                }
                let path = request_line.split_whitespace().nth(1).unwrap_or_default();
                let path = path.split('?').next().unwrap_or_default();
                let body = revisions.iter().find_map(|(revision, body)| {
                    (path == format!("/api/models/{REPOSITORY}/revision/{revision}"))
                        .then_some(body.clone())
                });
                let _ = match body {
                    Some(body) => write!(
                        connection,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    ),
                    None => write!(
                        connection,
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    ),
                };
            }
        });
        endpoint
    }

    async fn store(endpoint: String) -> (tempfile::TempDir, Arc<ManagedModelStore>) {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let mut config = InventoryConfig::with_roots(
            temporary.path().join("store"),
            temporary.path().join("cache"),
        )
        .expect("config");
        config.hub_endpoint = Some(endpoint);
        let store = ManagedModelStore::open_unbounded(config).await.expect("store");
        (temporary, Arc::new(store))
    }

    fn declaration(pinned: &str) -> AddedModel {
        AddedModel {
            repository: HuggingFaceRepositoryId::new(REPOSITORY).unwrap(),
            path: HuggingFaceArtifactSelector::new(TARGET).unwrap(),
            revision: pinned.to_owned(),
            projector: Some("mmproj.gguf".to_owned()),
        }
    }

    #[tokio::test]
    async fn a_newer_commit_with_other_target_content_is_an_update_and_anything_else_is_not() {
        let (pinned, newer) = (commit('a'), commit('b'));
        let old = revision_json(&pinned, Some(&"1".repeat(64)), &"7".repeat(64));
        let target = [TARGET.to_owned()];

        let changed = serve_hub(vec![
            (
                "main".to_owned(),
                revision_json(&newer, Some(&"2".repeat(64)), &"7".repeat(64)),
            ),
            (pinned.clone(), old.clone()),
        ]);
        let (_directory, hub) = store(changed).await;
        assert_eq!(
            upstream_update(&hub, REPOSITORY, &pinned, &target).await,
            Ok(Some(newer.clone()))
        );

        // A commit that changes only other files leaves the target's blob as it was.
        let unchanged = serve_hub(vec![
            (
                "main".to_owned(),
                revision_json(&newer, Some(&"1".repeat(64)), &"8".repeat(64)),
            ),
            (pinned.clone(), old.clone()),
        ]);
        let (_directory, hub) = store(unchanged).await;
        assert_eq!(
            upstream_update(&hub, REPOSITORY, &pinned, &target).await,
            Ok(None)
        );

        let removed = serve_hub(vec![
            (
                "main".to_owned(),
                revision_json(&newer, None, &"7".repeat(64)),
            ),
            (pinned.clone(), old.clone()),
        ]);
        let (_directory, hub) = store(removed).await;
        assert_eq!(
            upstream_update(&hub, REPOSITORY, &pinned, &target).await,
            Ok(None)
        );

        let current = serve_hub(vec![("main".to_owned(), old)]);
        let (_directory, hub) = store(current).await;
        assert_eq!(
            upstream_update(&hub, REPOSITORY, &pinned, &target).await,
            Ok(None)
        );
    }

    #[tokio::test]
    async fn the_listed_added_model_carries_its_update_and_an_offline_check_keeps_quiet() {
        let (pinned, newer) = (commit('a'), commit('b'));
        let endpoint = serve_hub(vec![
            (
                "main".to_owned(),
                revision_json(&newer, Some(&"2".repeat(64)), &"7".repeat(64)),
            ),
            (
                pinned.clone(),
                revision_json(&pinned, Some(&"1".repeat(64)), &"7".repeat(64)),
            ),
        ]);
        let (_directory, inventory) = store(endpoint).await;
        let resolver = ModelDomainResolver::new(
            inventory,
            RecommendableModelCatalog {
                models: Vec::new(),
                diagnostics: Vec::new(),
            },
        );
        let local =
            crate::custom_models::ManagedLocalModels::new(resolver.clone(), Arc::new(NoRemoval));
        let configuration = LocalModelsConfiguration {
            added: vec![declaration(&pinned)],
            ..LocalModelsConfiguration::default()
        };
        local.configure(configuration.clone()).await.unwrap();
        let before = local.list_local().await.unwrap();
        assert_eq!(before.models[0].update_commit, None);
        assert_eq!(before.models[0].state, CustomModelState::NotInstalled);

        assert!(
            resolver
                .upstream
                .check_due(&resolver.inventory, &configuration.added)
                .await
        );
        let listed = local.list_local().await.unwrap();
        assert_eq!(
            listed.models[0].update_commit.as_deref(),
            Some(newer.as_str())
        );
        // A result is kept until it is due again, so listing makes no requests.
        assert!(
            !resolver
                .upstream
                .check_due(&resolver.inventory, &configuration.added)
                .await
        );

        // Pinned to the newer commit, the model is current, and an unreachable Hub says nothing.
        let (_offline_directory, offline) = store("http://127.0.0.1:9".to_owned()).await;
        let updated = vec![declaration(&newer)];
        let upstream = UpstreamUpdates::new();
        assert!(!upstream.check_due(&offline, &updated).await);
        let mut model = listed.models[0].clone();
        upstream.annotate([&mut model], &updated);
        assert_eq!(model.update_commit, None);
    }

    #[test]
    fn a_replaced_copy_is_removed_only_once_the_pinned_copy_is_valid() {
        use magnitude_service_contracts::models::{
            InstalledCatalogAttribution, InstalledModelPackage, ModelFile, ModelFileId,
            ModelPackage, ModelPackageProperties, ModelPackageSource,
        };
        use magnitude_service_contracts::{
            Integrity, InventoryEntryId, InventoryModel, InventoryProperties, ModelAvailability,
            ModelLocation,
        };
        let record = |commit: &str, validation: PackageValidation| {
            let package = ModelPackage {
                id: ModelPackageId(format!("package-{commit}")),
                source: ModelPackageSource::HuggingFace {
                    repository: REPOSITORY.to_owned(),
                    revision: commit.to_owned(),
                },
                files: vec![ModelFile {
                    id: ModelFileId(format!("file-{commit}")),
                    path: TARGET.into(),
                    role: ModelFileRole::Weights,
                    size_bytes: 100,
                    tensor_storage_bytes: None,
                    sha256: "1".repeat(64),
                }],
                relationships: Vec::new(),
                properties: ModelPackageProperties {
                    format: "gguf".to_owned(),
                    quantization: "q4_k_m".to_owned(),
                    quantization_name: "Q4_K_M".to_owned(),
                    architecture: "llama".to_owned(),
                    maximum_context_length: None,
                    intrinsic_model_id: None,
                    intrinsic_quality_id: None,
                },
            };
            crate::inventory::InstalledPackageRecord {
                installed: InstalledModelPackage {
                    package,
                    path: TARGET.into(),
                    origin: ModelPackageInstallationOrigin::Magnitude,
                    validation,
                    catalog_attribution: InstalledCatalogAttribution::NotCatalogTarget,
                },
                model: InventoryModel {
                    id: InventoryEntryId(format!("entry-{commit}")),
                    content_id: magnitude_service_contracts::ContentId(String::new()),
                    created: 0,
                    name: REPOSITORY.to_owned(),
                    supported_parameters: Vec::new(),
                    availability: ModelAvailability::Available { ready_at: 0 },
                    source: ModelSource::HuggingFace {
                        repository: REPOSITORY.to_owned(),
                        requested_revision: commit.to_owned(),
                        commit: commit.to_owned(),
                        metadata: None,
                    },
                    location: ModelLocation::MagnitudeCache {
                        components: Vec::new(),
                        total_bytes: 100,
                        integrity: Integrity::Verified {
                            method: "content_identity".to_owned(),
                        },
                    },
                    properties: InventoryProperties::Pending,
                    operations: Vec::new(),
                    updated_at: 0,
                },
                primary: None,
            }
        };
        let (old, new) = (commit('a'), commit('b'));
        let snapshot = |new_validation: PackageValidation| {
            let mut snapshot = InstalledPackageSnapshot::default();
            for record in [
                record(&old, PackageValidation::Valid),
                record(&new, new_validation),
            ] {
                snapshot.records.insert(record.model.id.clone(), record);
            }
            snapshot
        };
        let current = |_: &AddedModel| Some(None);
        let pinned_new = [declaration(&new)];
        assert_eq!(
            replaced_added_packages(&snapshot(PackageValidation::Valid), &pinned_new, current),
            [ModelPackageId(format!("package-{old}"))]
        );
        assert!(
            replaced_added_packages(&snapshot(PackageValidation::Pending), &pinned_new, current)
                .is_empty()
        );
        // Unconfirmed since the service started: nothing is removed.
        assert!(
            replaced_added_packages(&snapshot(PackageValidation::Valid), &pinned_new, |_| None)
                .is_empty()
        );
        // While still pinned to the old commit, the update that landed is waiting to be pinned.
        let pinned_old = [declaration(&old)];
        assert!(
            replaced_added_packages(&snapshot(PackageValidation::Valid), &pinned_old, |_| {
                Some(Some(new.clone()))
            })
            .is_empty()
        );
    }
}
