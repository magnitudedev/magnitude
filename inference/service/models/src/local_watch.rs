//! Filesystem watches over the model locations, so models appear and disappear without a restart.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf, Prefix};
use std::time::Duration;

use magnitude_service_contracts::models::{
    DrafterChoice, DrafterReference, LocalModelsConfiguration,
};
use notify::{EventKind, RecursiveMode, Watcher};

use crate::identity::stable_path;
use crate::local_roots::is_partial_download;

const CASE_INSENSITIVE_OS: bool = cfg!(any(target_os = "macos", windows));

/// How long a location must be quiet before a change is reconciled.
pub(crate) const WATCH_DEBOUNCE: Duration = Duration::from_millis(1_500);

/// How often every enabled location is checked for becoming readable (after a permission change,
/// say), so it is scanned as soon as it is, without a restart.
pub(crate) const ROOT_RECHECK: Duration = Duration::from_secs(10);

/// How often a location whose watch could not start is rescanned instead.
pub(crate) const UNWATCHED_RESCAN: Duration = Duration::from_secs(60);

/// One watch. `interest` is the configured path it stands for; events under an ancestor watched
/// in place of a missing root count only on the way to that root.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct WatchTarget {
    pub(crate) path: PathBuf,
    pub(crate) recursive: bool,
    pub(crate) interest: PathBuf,
}

/// Every enabled root, recursively (a root that is one file through its folder, which also holds
/// its other shards and its drafters), and the directories of drafters attached by local path. A
/// root that does not exist yet is watched through its nearest existing ancestor, so creating it
/// is noticed.
pub(crate) fn watch_targets(configuration: &LocalModelsConfiguration) -> Vec<WatchTarget> {
    let roots = configuration
        .roots
        .iter()
        .filter(|root| root.enabled)
        .map(|root| existing_target(&root.path, true));
    let files = configuration
        .drafters
        .iter()
        .filter_map(|attachment| match &attachment.drafter {
            DrafterChoice::Reference {
                reference: DrafterReference::Local { path },
            } => path.parent(),
            _ => None,
        })
        .map(|directory| existing_target(directory, false));
    let mut targets = roots.chain(files).flatten().collect::<Vec<_>>();
    targets.sort();
    targets.dedup();
    targets
}

fn existing_target(path: &Path, recursive: bool) -> Option<WatchTarget> {
    let interest = stable_path(path);
    if path.is_dir() {
        return Some(WatchTarget {
            path: path.to_path_buf(),
            recursive,
            interest,
        });
    }
    path.ancestors()
        .skip(1)
        .find(|ancestor| ancestor.is_dir())
        .map(|ancestor| WatchTarget {
            path: ancestor.to_path_buf(),
            recursive: false,
            interest,
        })
}

/// Pure in-memory normalized path components for prefix matching without filesystem syscalls.
/// Strips Windows verbatim prefixes (`\\?\`) so canonicalized roots match OS watcher paths, and
/// optionally folds case on case-insensitive filesystems (macOS/Windows).
#[derive(Debug, Clone, PartialEq, Eq)]
struct NormalizedPath {
    components: Vec<String>,
}

impl NormalizedPath {
    fn from_path(path: &Path, case_insensitive: bool) -> Self {
        let mut components = Vec::new();
        let mut root_len = 0_usize;
        for component in path.components() {
            match component {
                Component::Prefix(prefix) => {
                    let normalized = normalize_prefix(prefix);
                    components.push(if case_insensitive {
                        normalized.to_lowercase()
                    } else {
                        normalized
                    });
                    root_len = components.len();
                }
                Component::RootDir => {
                    components.push(String::from("/"));
                    root_len = components.len();
                }
                Component::CurDir => {}
                Component::ParentDir => {
                    if components.len() > root_len
                        && components.last().is_some_and(|last| last != "..")
                    {
                        components.pop();
                    } else if root_len == 0 {
                        components.push(String::from(".."));
                    }
                }
                Component::Normal(part) => {
                    let text = part.to_string_lossy();
                    components.push(if case_insensitive {
                        text.to_lowercase()
                    } else {
                        text.into_owned()
                    });
                }
            }
        }
        Self { components }
    }

    /// Whether `self` is an ancestor of, equal to, or descendant of `other`.
    fn overlaps(&self, other: &Self) -> bool {
        !self.components.is_empty()
            && !other.components.is_empty()
            && self
                .components
                .iter()
                .zip(other.components.iter())
                .all(|(left, right)| left == right)
    }
}

fn normalize_prefix(prefix: std::path::PrefixComponent<'_>) -> String {
    match prefix.kind() {
        Prefix::VerbatimDisk(drive) | Prefix::Disk(drive) => {
            format!("{}:", char::from(drive))
        }
        Prefix::VerbatimUNC(server, share) | Prefix::UNC(server, share) => {
            format!(
                "\\\\{}\\{}",
                server.to_string_lossy(),
                share.to_string_lossy()
            )
        }
        Prefix::Verbatim(rest) => rest.to_string_lossy().into_owned(),
        Prefix::DeviceNS(device) => format!("\\\\.\\{}", device.to_string_lossy()),
    }
}

/// Pre-canonicalized watch rule built once when starting a watcher so the OS callback performs
/// pure in-memory prefix matching with zero filesystem syscalls.
#[derive(Debug, Clone)]
struct WatchScope {
    watch_path: PathBuf,
    scope: NormalizedPath,
    interest: PathBuf,
}

impl WatchScope {
    fn from_target(target: &WatchTarget, case_insensitive: bool) -> Self {
        let watch_path = stable_path(&target.path);
        let canonical_interest = stable_path(&target.interest);
        // For a single-file root (non-directory target), widen the match scope to its watched
        // parent directory so sibling shards and sibling drafters are never dropped, while still
        // reporting the configured root's `interest` for `ScanScope::Roots`.
        let scope_path = if std::fs::metadata(&canonical_interest)
            .is_ok_and(|metadata| !metadata.is_dir())
        {
            watch_path.clone()
        } else {
            canonical_interest
        };
        Self {
            watch_path,
            scope: NormalizedPath::from_path(&scope_path, case_insensitive),
            interest: target.interest.clone(),
        }
    }
}

/// The configured paths a change concerns: the locations it happened in, or a missing location
/// it may have created. Pure in-memory matching (no `canonicalize()` / filesystem syscalls).
fn affected(
    event: &notify::Event,
    scopes: &[WatchScope],
    case_insensitive: bool,
) -> BTreeSet<PathBuf> {
    if matches!(event.kind, EventKind::Access(_)) {
        return BTreeSet::new();
    }
    event
        .paths
        .iter()
        .filter(|path| !is_partial_download(path))
        .flat_map(|path| {
            let normalized = NormalizedPath::from_path(path, case_insensitive);
            scopes
                .iter()
                .filter(move |scope| normalized.overlaps(&scope.scope))
                .map(|scope| scope.interest.clone())
        })
        .collect()
}

/// A set of watches that calls `on_change` with the configured paths a burst of events concerned,
/// once the burst has settled. Dropping it stops watching and reports a burst still settling.
pub(crate) struct RootWatcher {
    targets: Vec<WatchTarget>,
    unwatched: BTreeSet<PathBuf>,
    _watcher: notify::RecommendedWatcher,
}

impl RootWatcher {
    pub(crate) fn start(
        targets: Vec<WatchTarget>,
        on_change: impl Fn(BTreeSet<PathBuf>) + Send + 'static,
    ) -> Option<Self> {
        let runtime = tokio::runtime::Handle::try_current().ok()?;
        let (events, received) = tokio::sync::mpsc::unbounded_channel::<BTreeSet<PathBuf>>();
        let scopes = targets
            .iter()
            .map(|target| WatchScope::from_target(target, CASE_INSENSITIVE_OS))
            .collect::<Vec<_>>();
        let watch_paths = scopes
            .iter()
            .map(|scope| scope.watch_path.clone())
            .collect::<Vec<_>>();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                if let Ok(event) = event {
                    let affected = affected(&event, &scopes, CASE_INSENSITIVE_OS);
                    if !affected.is_empty() {
                        let _ = events.send(affected);
                    }
                }
            })
            .inspect_err(|error| tracing::warn!(%error, "model location watcher could not start"))
            .ok()?;
        let mut unwatched = BTreeSet::new();
        for (target, watch_path) in targets.iter().zip(&watch_paths) {
            let mode = if target.recursive {
                RecursiveMode::Recursive
            } else {
                RecursiveMode::NonRecursive
            };
            if let Err(error) = watcher.watch(watch_path, mode) {
                tracing::warn!(
                    path = %watch_path.display(),
                    %error,
                    "model location is not watched; it is rescanned periodically instead"
                );
                unwatched.insert(target.interest.clone());
            }
        }
        runtime.spawn(debounced(received, on_change));
        Some(Self {
            targets,
            unwatched,
            _watcher: watcher,
        })
    }

    pub(crate) fn targets(&self) -> &[WatchTarget] {
        &self.targets
    }

    /// The configured paths whose watch could not start, which are rescanned periodically.
    pub(crate) fn unwatched(&self) -> &BTreeSet<PathBuf> {
        &self.unwatched
    }
}

/// Reports each burst of changes once `WATCH_DEBOUNCE` passes without more. A burst still
/// settling when the watch stops (replaced because a location was added, say, or dropped) is
/// reported then, never lost: the watcher owns the only sender, so stopping it ends the channel.
pub(crate) async fn debounced(
    mut received: tokio::sync::mpsc::UnboundedReceiver<BTreeSet<PathBuf>>,
    on_change: impl Fn(BTreeSet<PathBuf>),
) {
    while let Some(mut affected) = received.recv().await {
        loop {
            match tokio::time::timeout(WATCH_DEBOUNCE, received.recv()).await {
                Ok(Some(more)) => affected.extend(more),
                Ok(None) => {
                    on_change(affected);
                    return;
                }
                Err(_) => break,
            }
        }
        on_change(affected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_service_contracts::models::{ModelRoot, ModelRootKind};
    use notify::event::{CreateKind, ModifyKind};

    fn create_event(path: PathBuf) -> notify::Event {
        notify::Event {
            kind: EventKind::Create(CreateKind::File),
            paths: vec![path],
            attrs: Default::default(),
        }
    }

    #[test]
    fn single_file_root_admits_sibling_shards_and_drafters() {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path().join("models");
        std::fs::create_dir_all(&dir).unwrap();
        let shard1 = dir.join("model-00001-of-00002.gguf");
        std::fs::write(&shard1, b"shard1").unwrap();

        let configuration = LocalModelsConfiguration {
            roots: vec![ModelRoot {
                kind: ModelRootKind::Folder,
                path: shard1.clone(),
                enabled: true,
            }],
            ..LocalModelsConfiguration::default()
        };
        let targets = watch_targets(&configuration);
        assert_eq!(targets.len(), 1);
        let scopes = targets
            .iter()
            .map(|target| WatchScope::from_target(target, CASE_INSENSITIVE_OS))
            .collect::<Vec<_>>();
        let expected_interest = stable_path(&shard1);
        let canonical_dir = stable_path(&dir);

        let shard2_event = create_event(canonical_dir.join("model-00002-of-00002.gguf"));
        assert_eq!(
            affected(&shard2_event, &scopes, CASE_INSENSITIVE_OS),
            BTreeSet::from([expected_interest.clone()])
        );

        let drafter_event = create_event(canonical_dir.join("drafter.gguf"));
        assert_eq!(
            affected(&drafter_event, &scopes, CASE_INSENSITIVE_OS),
            BTreeSet::from([expected_interest.clone()])
        );

        let unrelated_event = create_event(stable_path(temporary.path()).join("other.gguf"));
        assert!(affected(&unrelated_event, &scopes, CASE_INSENSITIVE_OS).is_empty());

        let partial_event = create_event(canonical_dir.join("model-00002-of-00002.gguf.part"));
        assert!(affected(&partial_event, &scopes, CASE_INSENSITIVE_OS).is_empty());
    }

    #[test]
    fn case_insensitive_matching_admits_missing_root_created_with_different_casing() {
        let temporary = tempfile::tempdir().unwrap();
        let missing = temporary.path().join("MissingDir").join("Models");
        let configuration = LocalModelsConfiguration {
            roots: vec![ModelRoot {
                kind: ModelRootKind::Folder,
                path: missing.clone(),
                enabled: true,
            }],
            ..LocalModelsConfiguration::default()
        };
        let targets = watch_targets(&configuration);
        assert_eq!(targets.len(), 1);
        let expected_interest = stable_path(&missing);
        let canonical_base = stable_path(temporary.path());

        let insensitive_scopes = targets
            .iter()
            .map(|target| WatchScope::from_target(target, true))
            .collect::<Vec<_>>();
        let created_different_case = notify::Event {
            kind: EventKind::Modify(ModifyKind::Name(notify::event::RenameMode::Any)),
            paths: vec![canonical_base.join("missingdir")],
            attrs: Default::default(),
        };
        assert_eq!(
            affected(&created_different_case, &insensitive_scopes, true),
            BTreeSet::from([expected_interest.clone()])
        );

        let unrelated_sibling = create_event(canonical_base.join("unrelated"));
        assert!(affected(&unrelated_sibling, &insensitive_scopes, true).is_empty());

        let sensitive_scopes = targets
            .iter()
            .map(|target| WatchScope::from_target(target, false))
            .collect::<Vec<_>>();
        assert!(affected(&created_different_case, &sensitive_scopes, false).is_empty());
    }
}
