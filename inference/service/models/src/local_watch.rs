//! Filesystem watches over the model locations, so models appear and disappear without a restart.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use magnitude_service_contracts::models::{
    CustomModelDeclarationOrigin, DrafterChoice, DrafterReference, LocalModelsConfiguration,
};
use notify::{EventKind, RecursiveMode, Watcher};

use crate::local_roots::is_partial_download;

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

/// Every enabled root, recursively, and the directories of declared files: imported models (whose
/// folder also holds their drafters) and drafters attached by local path. A root that does not
/// exist yet is watched through its nearest existing ancestor, so creating it is noticed.
pub(crate) fn watch_targets(configuration: &LocalModelsConfiguration) -> Vec<WatchTarget> {
    let roots = configuration
        .roots
        .iter()
        .filter(|root| root.enabled)
        .map(|root| existing_target(&root.path, true));
    let declared = configuration
        .declarations
        .iter()
        .filter(|declaration| {
            matches!(
                declaration.origin,
                CustomModelDeclarationOrigin::Imported { .. }
            )
        })
        .flat_map(|declaration| declaration.target.iter().chain(&declaration.projector))
        .map(Path::new);
    let drafters =
        configuration
            .drafters
            .iter()
            .filter_map(|attachment| match &attachment.drafter {
                DrafterChoice::Reference {
                    reference: DrafterReference::Local { path },
                    ..
                } => Some(path.as_path()),
                _ => None,
            });
    let files = declared
        .chain(drafters)
        .filter_map(Path::parent)
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

/// The canonical form of the deepest existing ancestor with the rest appended, which is how
/// watch events name paths.
fn stable_path(path: &Path) -> PathBuf {
    let mut missing = Vec::new();
    let mut existing = path;
    loop {
        if let Ok(canonical) = existing.canonicalize() {
            return missing
                .iter()
                .rev()
                .fold(canonical, |path, component| path.join(component));
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => {
                missing.push(name);
                existing = parent;
            }
            _ => return path.to_path_buf(),
        }
    }
}

/// Where an event happened, as `stable_path` names it but without following the path itself: a
/// link created in a location is a change there, wherever it points.
fn event_path(path: &Path) -> PathBuf {
    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => stable_path(parent).join(name),
        _ => stable_path(path),
    }
}

/// The configured paths a change concerns: the locations it happened in, or a missing location
/// it may have created.
fn affected(event: &notify::Event, interests: &[PathBuf]) -> BTreeSet<PathBuf> {
    if matches!(event.kind, EventKind::Access(_)) {
        return BTreeSet::new();
    }
    event
        .paths
        .iter()
        .map(|path| event_path(path))
        .filter(|path| !is_partial_download(path))
        .flat_map(|path| {
            interests
                .iter()
                .filter(move |interest| path.starts_with(interest) || interest.starts_with(&path))
                .cloned()
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
        let interests = targets
            .iter()
            .map(|target| target.interest.clone())
            .collect::<Vec<_>>();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                if let Ok(event) = event {
                    let affected = affected(&event, &interests);
                    if !affected.is_empty() {
                        let _ = events.send(affected);
                    }
                }
            })
            .inspect_err(|error| tracing::warn!(%error, "model location watcher could not start"))
            .ok()?;
        let mut unwatched = BTreeSet::new();
        for target in &targets {
            let mode = if target.recursive {
                RecursiveMode::Recursive
            } else {
                RecursiveMode::NonRecursive
            };
            if let Err(error) = watcher.watch(&target.path, mode) {
                tracing::warn!(
                    path = %target.path.display(),
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
