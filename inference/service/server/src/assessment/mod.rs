//! The automatic assessment pool: one revisioned snapshot over the current catalog and custom-model
//! sources in the service's current environment, with exact-work reconciliation, deduplication,
//! per-target deadlines, publication guards and one-attempt `Dropped` semantics. When the
//! measurement job stores this machine's measurements, every assessed model is finished again
//! under the new environment and published together in one revision.

pub mod assessor;
pub mod candidates;
pub mod catalog_kernels;
pub mod environment;
mod inspection_progress;
pub mod measurement;

use std::sync::{Arc, Mutex, Weak};

use futures_util::{StreamExt, future::BoxFuture, stream::BoxStream};
use magnitude_service_contracts::InventoryError;
use magnitude_service_contracts::models::{
    AssessmentEnvironmentId, CatalogModel, CatalogModelSelection, CatalogModelState, CatalogModels,
    CustomModel, CustomModelState, EffectiveModel, LocalModels, MeasurementState, ModelAssessment,
    ModelAssessmentDomainSnapshot, ModelAssessmentEntry, ModelAssessmentEntryState,
    ModelAssessmentSubject, ModelAssessments, ModelAssessmentsInvalidation,
    ModelAssessmentsSnapshot, ModelFailure as DomainModelFailure, ModelId,
    ModelServingConfiguration, ServingProfile,
};
use magnitude_service_models::{
    CachedModelAssessment, ServableModelBundleKey, servable_model_bundle_key_for_bundle,
};

use assessor::{
    AssessmentOutcome, AssessmentWork, AssessmentWorkKey, ModelAssessor, PreparationResult,
    inventory_model_failure,
};
use environment::AssessmentEnvironment;
use measurement::MeasurementStatus;

use crate::device_lock::{DeviceLock, DeviceShare};

/// Bounds one target's whole attempt: material resolution and the engine's arithmetic.
const MODEL_ASSESSMENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
const MODEL_ASSESSMENT_SOURCE_RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(1);
/// Concurrent targets, each holding at most one blocking-pool task of header arithmetic.
const MAX_ASSESSMENT_CONCURRENCY: usize = 8;

struct AssessmentTarget {
    entry: ModelAssessmentEntry,
    key: AssessmentWorkKey,
    work: Option<AssessmentWork>,
}

/// One preparation per bundle, shared by every profile and source slice that names it.
#[derive(Default)]
struct PreparationCoordinator {
    cells: Mutex<
        std::collections::BTreeMap<
            ServableModelBundleKey,
            Weak<tokio::sync::OnceCell<PreparationResult>>,
        >,
    >,
}

impl PreparationCoordinator {
    fn prepare(
        &self,
        configuration: &ModelServingConfiguration,
    ) -> Arc<tokio::sync::OnceCell<PreparationResult>> {
        let key = servable_model_bundle_key_for_bundle(&configuration.bundle);
        let mut cells = self.cells.lock().expect("preparation cells lock poisoned");
        cells.retain(|_, cell| cell.strong_count() > 0);
        if let Some(cell) = cells.get(&key).and_then(Weak::upgrade) {
            return cell;
        }
        let cell = Arc::new(tokio::sync::OnceCell::new());
        cells.insert(key, Arc::downgrade(&cell));
        cell
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum AssessmentDomain {
    Catalog,
    Custom,
}

fn catalog_candidates(models: Vec<CatalogModel>) -> Vec<(ModelAssessmentSubject, ServingProfile)> {
    models
        .into_iter()
        .filter_map(|model| {
            let (selection, profile) = match model.local_state {
                CatalogModelState::NotInstalled => {
                    (CatalogModelSelection::Desired, model.desired.profile)
                }
                CatalogModelState::Installed {
                    effective: EffectiveModel::Ready { model },
                    ..
                } => (CatalogModelSelection::Effective, model.profile),
                CatalogModelState::Installed {
                    effective: EffectiveModel::Unavailable { .. },
                    ..
                } => return None,
            };
            Some((
                ModelAssessmentSubject::Catalog {
                    model_id: model.id,
                    selection,
                },
                profile,
            ))
        })
        .collect()
}

fn custom_candidates(
    models: Vec<CustomModel>,
) -> Vec<(ModelAssessmentSubject, ServingProfile)> {
    models
        .into_iter()
        .filter_map(|model| {
            let CustomModelState::Ready { model: ready, .. } = model.state else {
                return None;
            };
            Some((
                ModelAssessmentSubject::Custom {
                    model_id: model.model_id,
                },
                ready.profile,
            ))
        })
        .collect()
}

struct AssessmentPoolCurrent {
    snapshot: ModelAssessmentsSnapshot,
    entry_keys: std::collections::BTreeMap<ModelAssessmentSubject, AssessmentWorkKey>,
    /// The work of every key an entry references, with its preparation: a new environment
    /// finishes each assessed model again from it.
    works: std::collections::BTreeMap<AssessmentWorkKey, AssessmentWork>,
    revalidating: std::collections::BTreeSet<ModelAssessmentSubject>,
    /// Sources whose last read failed. A failed read is the model inventory's failure, which the
    /// catalog reports from the same listing; the pool keeps the slice it has and reads again.
    failed_reads: std::collections::BTreeSet<AssessmentDomain>,
}

fn assessment_domain_entries(domain: &ModelAssessmentDomainSnapshot) -> &[ModelAssessmentEntry] {
    match domain {
        ModelAssessmentDomainSnapshot::Available { entries, .. } => entries,
        ModelAssessmentDomainSnapshot::Pending { .. } => &[],
    }
}

impl AssessmentPoolCurrent {
    fn domains(&self) -> [&ModelAssessmentDomainSnapshot; 2] {
        [&self.snapshot.catalog, &self.snapshot.custom]
    }

    fn domain(&self, domain: AssessmentDomain) -> &ModelAssessmentDomainSnapshot {
        match domain {
            AssessmentDomain::Catalog => &self.snapshot.catalog,
            AssessmentDomain::Custom => &self.snapshot.custom,
        }
    }

    fn domain_mut(&mut self, domain: AssessmentDomain) -> &mut ModelAssessmentDomainSnapshot {
        match domain {
            AssessmentDomain::Catalog => &mut self.snapshot.catalog,
            AssessmentDomain::Custom => &mut self.snapshot.custom,
        }
    }

    fn retained_terminal_states(
        &self,
    ) -> std::collections::BTreeMap<AssessmentWorkKey, ModelAssessmentEntryState> {
        self.domains()
            .into_iter()
            .flat_map(assessment_domain_entries)
            .filter(|entry| !self.revalidating.contains(&entry.subject))
            .filter_map(|entry| {
                self.entry_keys
                    .get(&entry.subject)
                    .map(|key| (key.clone(), entry.state.clone()))
            })
            .filter(|(_, state)| {
                matches!(
                    state,
                    ModelAssessmentEntryState::Assessed { .. } | ModelAssessmentEntryState::Dropped
                )
            })
            .collect()
    }

    fn replace_domain(
        &mut self,
        domain: AssessmentDomain,
        source_revision: u64,
        targets: &[AssessmentTarget],
    ) -> bool {
        let retained = self.retained_terminal_states();
        let work_by_entry = targets
            .iter()
            .map(|target| (target.entry.subject.clone(), target.key.clone()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let previous_results = assessment_domain_entries(self.domain(domain))
            .iter()
            .filter(|entry| matches!(entry.state, ModelAssessmentEntryState::Assessed { .. }))
            .map(|entry| (entry.subject.clone(), entry.state.clone()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut revalidating = std::collections::BTreeSet::new();
        let entries = targets
            .iter()
            .map(|target| {
                let mut entry = target.entry.clone();
                if let Some(state) = retained.get(&target.key) {
                    entry.state = state.clone();
                } else if matches!(entry.state, ModelAssessmentEntryState::Assessing)
                    && let Some(state) = previous_results.get(&entry.subject)
                {
                    entry.state = state.clone();
                    revalidating.insert(entry.subject.clone());
                }
                entry
            })
            .collect::<Vec<_>>();
        let next = ModelAssessmentDomainSnapshot::Available {
            source_revision,
            entries,
        };
        let previous = self.domain(domain);
        let old_subjects = assessment_domain_entries(previous)
            .iter()
            .map(|entry| entry.subject.clone())
            .collect::<Vec<_>>();
        let old_work = old_subjects
            .iter()
            .filter_map(|subject| {
                self.entry_keys
                    .get(subject)
                    .map(|key| (subject.clone(), key.clone()))
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        if previous == &next && old_work == work_by_entry {
            return false;
        }
        for subject in old_subjects {
            self.entry_keys.remove(&subject);
            self.revalidating.remove(&subject);
        }
        self.entry_keys.extend(work_by_entry);
        self.revalidating.extend(revalidating);
        for target in targets {
            if let Some(work) = &target.work {
                self.works
                    .entry(work.key.clone())
                    .or_insert_with(|| work.clone());
            }
        }
        self.retain_referenced_works();
        *self.domain_mut(domain) = next;
        true
    }

    fn retain_referenced_works(&mut self) {
        let referenced = self
            .entry_keys
            .values()
            .collect::<std::collections::BTreeSet<_>>();
        self.works.retain(|key, _| referenced.contains(key));
    }

    /// The keys of the entries assessed now, whose results a new environment replaces in one
    /// revision.
    fn assessed_keys(&self) -> std::collections::BTreeSet<AssessmentWorkKey> {
        self.domains()
            .into_iter()
            .flat_map(assessment_domain_entries)
            .filter(|entry| {
                matches!(entry.state, ModelAssessmentEntryState::Assessed { .. })
                    && !self.revalidating.contains(&entry.subject)
            })
            .filter_map(|entry| self.entry_keys.get(&entry.subject).cloned())
            .collect()
    }

    /// Move every entry not assessed now to its work in a new environment (`rekeyed`, by old
    /// key): its old work's outcome no longer applies. Its visible state is unchanged, so this
    /// publishes nothing.
    fn rekey_unassessed(
        &mut self,
        rekeyed: &std::collections::BTreeMap<AssessmentWorkKey, AssessmentWork>,
        assessed: &std::collections::BTreeSet<AssessmentWorkKey>,
    ) {
        for key in self.entry_keys.values_mut() {
            if assessed.contains(key) {
                continue;
            }
            if let Some(work) = rekeyed.get(key) {
                *key = work.key.clone();
            }
        }
        self.works.extend(
            rekeyed
                .iter()
                .filter(|(old, _)| !assessed.contains(*old))
                .map(|(_, work)| (work.key.clone(), work.clone())),
        );
        self.retain_referenced_works();
    }

    /// Replace every assessed entry's result with its result in `environment` (`outcomes`, the
    /// finishes of its work there, whose key `rekeyed` maps its old key to), with the snapshot's
    /// environment and measurement state, all as one change. No entry is seen passing through
    /// `Assessing`.
    fn republish(
        &mut self,
        environment: &AssessmentEnvironmentId,
        measurement: MeasurementState,
        rekeyed: &std::collections::BTreeMap<AssessmentWorkKey, AssessmentWorkKey>,
        outcomes: &[AssessmentOutcome],
    ) -> AppliedAssessmentOutcome {
        for (domain_kind, subject) in self
            .domains()
            .into_iter()
            .zip([AssessmentDomain::Catalog, AssessmentDomain::Custom])
            .flat_map(|(domain, kind)| {
                assessment_domain_entries(domain)
                    .iter()
                    .map(move |entry| (kind, entry.subject.clone()))
            })
            .collect::<Vec<_>>()
        {
            let Some(key) = self
                .entry_keys
                .get(&subject)
                .and_then(|key| rekeyed.get(key))
            else {
                continue;
            };
            self.entry_keys.insert(subject.clone(), key.clone());
            if let ModelAssessmentDomainSnapshot::Available { entries, .. } =
                self.domain_mut(domain_kind)
            {
                for entry in entries.iter_mut().filter(|entry| entry.subject == subject) {
                    entry.state = ModelAssessmentEntryState::Assessing;
                }
            }
        }
        self.retain_referenced_works();
        let mut applied = AppliedAssessmentOutcome::default();
        for outcome in outcomes {
            let mut one = self.apply_outcome(outcome);
            applied
                .dropped_catalog_models
                .append(&mut one.dropped_catalog_models);
        }
        applied.changed = true;
        self.snapshot.environment_id = environment.clone();
        self.snapshot.measurement = measurement;
        applied
    }

    fn set_domain_pending(&mut self, domain: AssessmentDomain, source_revision: u64) -> bool {
        let next = ModelAssessmentDomainSnapshot::Pending { source_revision };
        let previous = self.domain(domain);
        if previous == &next {
            return false;
        }
        let old_subjects = assessment_domain_entries(previous)
            .iter()
            .map(|entry| entry.subject.clone())
            .collect::<Vec<_>>();
        for subject in old_subjects {
            self.entry_keys.remove(&subject);
            self.revalidating.remove(&subject);
        }
        self.retain_referenced_works();
        *self.domain_mut(domain) = next;
        true
    }

    /// Whether every slice is read and no entry is being assessed.
    fn settled(&self) -> bool {
        self.domains().into_iter().all(|domain| {
            matches!(domain, ModelAssessmentDomainSnapshot::Available { .. })
                && assessment_domain_entries(domain)
                    .iter()
                    .all(|entry| !self.awaits_outcome(entry))
        })
    }

    /// Whether `entry` waits on its work's outcome: it is assessing, or shows its previous
    /// result while its new work is assessed.
    fn awaits_outcome(&self, entry: &ModelAssessmentEntry) -> bool {
        matches!(entry.state, ModelAssessmentEntryState::Assessing)
            || self.revalidating.contains(&entry.subject)
    }

    fn references_assessing(&self, key: &AssessmentWorkKey) -> bool {
        self.domains().into_iter().any(|domain| {
            assessment_domain_entries(domain).iter().any(|entry| {
                self.entry_keys.get(&entry.subject) == Some(key) && self.awaits_outcome(entry)
            })
        })
    }

    fn references_dropped(
        &self,
        subject: &ModelAssessmentSubject,
        key: &AssessmentWorkKey,
    ) -> bool {
        self.entry_keys.get(subject) == Some(key)
            && self.domains().into_iter().any(|domain| {
                assessment_domain_entries(domain).iter().any(|entry| {
                    &entry.subject == subject
                        && matches!(entry.state, ModelAssessmentEntryState::Dropped)
                })
            })
    }

    fn apply_outcome(&mut self, outcome: &AssessmentOutcome) -> AppliedAssessmentOutcome {
        let mut applied = AppliedAssessmentOutcome::default();
        for (domain_kind, domain) in [
            (AssessmentDomain::Catalog, &mut self.snapshot.catalog),
            (AssessmentDomain::Custom, &mut self.snapshot.custom),
        ] {
            let ModelAssessmentDomainSnapshot::Available { entries, .. } = domain else {
                continue;
            };
            for entry in entries.iter_mut() {
                if self.entry_keys.get(&entry.subject) == Some(&outcome.key)
                    && (matches!(entry.state, ModelAssessmentEntryState::Assessing)
                        || self.revalidating.remove(&entry.subject))
                {
                    let catalog = matches!(domain_kind, AssessmentDomain::Catalog);
                    let dropped = match &outcome.result {
                        // A catalog model the engine cannot execute is a release defect,
                        // never a verdict.
                        Ok(CachedModelAssessment {
                            profile: ModelAssessment::Unsupported { detail, .. },
                            ..
                        }) if catalog => Some(DomainModelFailure {
                            code: "unsupported".to_owned(),
                            message: detail.clone(),
                            retryable: false,
                        }),
                        Ok(_) => None,
                        Err(failure) => Some(failure.clone()),
                    };
                    entry.state = match (&outcome.result, &dropped) {
                        (Ok(assessment), None) => ModelAssessmentEntryState::Assessed {
                            capabilities: assessment.capabilities.clone(),
                            template_fingerprint: assessment.template_fingerprint.clone(),
                            fingerprint: assessment.fingerprint.clone(),
                            profiles: vec![assessment.profile.clone()],
                        },
                        _ => ModelAssessmentEntryState::Dropped,
                    };
                    applied.changed = true;
                    if let (true, Some(failure)) = (catalog, dropped) {
                        applied
                            .dropped_catalog_models
                            .push((entry.subject.model_id().clone(), failure));
                    }
                }
            }
        }
        applied
    }
}

#[derive(Default)]
struct AppliedAssessmentOutcome {
    changed: bool,
    /// Each dropped catalog model with its failure.
    dropped_catalog_models: Vec<(ModelId, DomainModelFailure)>,
}

pub struct ManagedModelAssessments {
    assessor: Arc<ModelAssessor>,
    catalog: Arc<dyn CatalogModels>,
    local: Arc<dyn LocalModels>,
    current: std::sync::RwLock<AssessmentPoolCurrent>,
    changes: tokio::sync::broadcast::Sender<ModelAssessmentsInvalidation>,
    active: Mutex<ActiveAssessments>,
    preparations: PreparationCoordinator,
    concurrency: Arc<tokio::sync::Semaphore>,
    host: HostDevice,
}

/// The device the measurement job times, where it is the host's own cores (a CPU executor): the
/// pool's work runs on them too. On a 4-vCPU EPYC 7763 the catalog's 73 models turned a 9.6 s
/// point into 226 s, and every record the job stored then would price this machine for good. So
/// a job starts only once the pool has settled (every slice read and no entry assessing, which
/// [`Self::settled`] reports), and each later attempt, the pool's or a candidate's
/// (`candidates::ManagedModelCandidates`, whose drafter fits are worked out for minutes after a
/// start), shares the device, waiting for a running job to end rather than preempting it: a job
/// is tens of seconds, and one ended at every arrival would be measured in pieces. A GPU's points run on the GPU: the pool shares nothing with them, and is always
/// settled for them.
#[derive(Clone)]
pub struct HostDevice {
    device: Option<DeviceLock>,
    settled: Arc<tokio::sync::watch::Sender<bool>>,
}

impl HostDevice {
    /// The measured device `device`, which is the host's cores where `host` holds.
    pub fn new(device: &DeviceLock, host: bool) -> Self {
        Self {
            device: host.then(|| device.clone()),
            settled: Arc::new(tokio::sync::watch::Sender::new(!host)),
        }
    }

    /// Whether the pool has settled, as it changes: the measurement job waits for it.
    pub fn settled(&self) -> tokio::sync::watch::Receiver<bool> {
        self.settled.subscribe()
    }

    /// A share of the device for one attempt, where it is the host's cores, once no job holds
    /// it.
    async fn share(&self) -> Option<DeviceShare> {
        match &self.device {
            Some(device) => Some(device.share_after_measurement().await),
            None => None,
        }
    }

    /// Note the pool's state `current`, where the device is the host's cores.
    fn observe(&self, current: &AssessmentPoolCurrent) {
        if self.device.is_some() {
            self.settled.send_if_modified(|settled| {
                let now = current.settled();
                std::mem::replace(settled, now) != now
            });
        }
    }
}

struct ActiveAssessment {
    id: u64,
    abort: Option<tokio::task::AbortHandle>,
}

struct ActiveAssessments {
    next_id: u64,
    entries: std::collections::BTreeMap<AssessmentWorkKey, ActiveAssessment>,
}

impl ActiveAssessments {
    fn new() -> Self {
        Self {
            next_id: 1,
            entries: std::collections::BTreeMap::new(),
        }
    }

    fn admit(&mut self, key: AssessmentWorkKey) -> Option<u64> {
        if self.entries.contains_key(&key) {
            return None;
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.entries
            .insert(key, ActiveAssessment { id, abort: None });
        Some(id)
    }

    fn attach(
        &mut self,
        key: &AssessmentWorkKey,
        id: u64,
        abort: tokio::task::AbortHandle,
    ) -> bool {
        let Some(entry) = self.entries.get_mut(key).filter(|entry| entry.id == id) else {
            return false;
        };
        entry.abort = Some(abort);
        true
    }

    fn finish(&mut self, key: &AssessmentWorkKey, id: u64) {
        if self.entries.get(key).is_some_and(|entry| entry.id == id) {
            self.entries.remove(key);
        }
    }

    fn cancel_unreferenced(&mut self, referenced: &std::collections::BTreeSet<AssessmentWorkKey>) {
        self.entries.retain(|key, entry| {
            let keep = referenced.contains(key);
            if !keep && let Some(handle) = &entry.abort {
                handle.abort();
            }
            keep
        });
    }
}

impl ManagedModelAssessments {
    /// Start the pool in the assessor's environment, adopting each new one `measurement`
    /// brings, each attempt sharing `host`. Both source slices are `Pending` until their first
    /// read.
    pub fn start(
        assessor: Arc<ModelAssessor>,
        catalog: Arc<dyn CatalogModels>,
        local: Arc<dyn LocalModels>,
        measurement: tokio::sync::watch::Receiver<MeasurementStatus>,
        host: HostDevice,
    ) -> Arc<Self> {
        let (changes, _) = tokio::sync::broadcast::channel(64);
        let concurrency = assessment_concurrency();
        let environment_id = assessor.environment().id.clone();
        let state = measurement.borrow().state.clone();
        let service = Arc::new(Self {
            assessor,
            catalog,
            local,
            current: std::sync::RwLock::new(AssessmentPoolCurrent {
                snapshot: ModelAssessmentsSnapshot {
                    revision: 0,
                    environment_id,
                    catalog: ModelAssessmentDomainSnapshot::Pending { source_revision: 0 },
                    custom: ModelAssessmentDomainSnapshot::Pending { source_revision: 0 },
                    measurement: state,
                },
                entry_keys: std::collections::BTreeMap::new(),
                works: std::collections::BTreeMap::new(),
                revalidating: std::collections::BTreeSet::new(),
                failed_reads: std::collections::BTreeSet::new(),
            }),
            changes,
            active: Mutex::new(ActiveAssessments::new()),
            preparations: PreparationCoordinator::default(),
            concurrency: Arc::new(tokio::sync::Semaphore::new(concurrency)),
            host,
        });
        let owner = Arc::clone(&service);
        tokio::spawn(async move { owner.run(measurement).await });
        service
    }

    fn publish(&self, mutate: impl FnOnce(&mut AssessmentPoolCurrent) -> bool) -> bool {
        let mut current = self
            .current
            .write()
            .expect("assessment state lock poisoned");
        if !mutate(&mut current) {
            return false;
        }
        current.snapshot.revision = current.snapshot.revision.saturating_add(1);
        let revision = current.snapshot.revision;
        self.host.observe(&current);
        drop(current);
        let _ = self.changes.send(ModelAssessmentsInvalidation { revision });
        true
    }

    fn target_entry(
        &self,
        domain: AssessmentDomain,
        subject: ModelAssessmentSubject,
        profile: ServingProfile,
    ) -> AssessmentTarget {
        // A target that cannot be resolved has no environment to be assessed in.
        let unresolved_key = AssessmentWorkKey(
            serde_json::to_string(&(&subject, &profile, "unresolved"))
                .expect("assessment target identity is serializable"),
        );
        if self
            .current
            .read()
            .expect("assessment state lock poisoned")
            .references_dropped(&subject, &unresolved_key)
        {
            return AssessmentTarget {
                entry: ModelAssessmentEntry {
                    subject,
                    state: ModelAssessmentEntryState::Dropped,
                },
                key: unresolved_key,
                work: None,
            };
        }
        let environment = self.assessor.environment();
        let work = self
            .assessor
            .configuration_for(&subject)
            .map(|configuration| {
                let preparation = self.preparations.prepare(&configuration);
                ModelAssessor::work_for(&environment, configuration, profile, preparation)
            });
        match work {
            Ok(work) => AssessmentTarget {
                entry: ModelAssessmentEntry {
                    subject,
                    state: ModelAssessmentEntryState::Assessing,
                },
                key: work.key.clone(),
                work: Some(work),
            },
            Err(error) => {
                let failure = inventory_model_failure(error);
                if matches!(domain, AssessmentDomain::Catalog) {
                    tracing::error!(
                        model.id = %subject.model_id().as_str(),
                        failure.code = %failure.code,
                        failure.message = %failure.message,
                        "catalog model assessment dropped"
                    );
                }
                AssessmentTarget {
                    entry: ModelAssessmentEntry {
                        subject,
                        state: ModelAssessmentEntryState::Dropped,
                    },
                    key: unresolved_key,
                    work: None,
                }
            }
        }
    }

    async fn reconcile_catalog(self: &Arc<Self>) -> Result<(), InventoryError> {
        let source = self.catalog.list_catalog().await?;
        let targets = catalog_candidates(source.models)
            .into_iter()
            .map(|(subject, profile)| {
                self.target_entry(AssessmentDomain::Catalog, subject, profile)
            })
            .collect::<Vec<_>>();
        self.replace_domain(AssessmentDomain::Catalog, source.revision, targets);
        Ok(())
    }

    async fn reconcile_custom(self: &Arc<Self>) -> Result<(), InventoryError> {
        let source = self.local.list_local().await?;
        if !source.reconciliation_complete {
            self.publish(|current| {
                current.set_domain_pending(AssessmentDomain::Custom, source.revision)
            });
            self.cancel_unreferenced();
            return Ok(());
        }
        let targets = custom_candidates(source.models)
            .into_iter()
            .map(|(subject, profile)| {
                self.target_entry(AssessmentDomain::Custom, subject, profile)
            })
            .collect::<Vec<_>>();
        self.replace_domain(AssessmentDomain::Custom, source.revision, targets);
        Ok(())
    }

    /// Read one source and reconcile its slice. A failed read keeps the slice as it is and marks
    /// the source for the next retry.
    async fn read_source(self: &Arc<Self>, domain: AssessmentDomain) {
        let read = match domain {
            AssessmentDomain::Catalog => self.reconcile_catalog().await,
            AssessmentDomain::Custom => self.reconcile_custom().await,
        };
        let mut current = self
            .current
            .write()
            .expect("assessment state lock poisoned");
        match read {
            Ok(()) => {
                current.failed_reads.remove(&domain);
            }
            Err(error) => {
                tracing::warn!(?domain, %error, "model assessment source read failed");
                current.failed_reads.insert(domain);
            }
        }
    }

    fn replace_domain(
        self: &Arc<Self>,
        domain: AssessmentDomain,
        source_revision: u64,
        targets: Vec<AssessmentTarget>,
    ) {
        let work = targets
            .iter()
            .filter_map(|target| target.work.clone())
            .collect::<Vec<_>>();
        self.publish(|current| current.replace_domain(domain, source_revision, &targets));
        self.cancel_unreferenced();
        self.schedule(work);
    }

    fn schedule(self: &Arc<Self>, work: Vec<AssessmentWork>) {
        for item in work {
            let current = self.current.read().expect("assessment state lock poisoned");
            if !current.references_assessing(&item.key) {
                continue;
            }
            let mut active = self.active.lock().expect("assessment owner lock poisoned");
            let Some(active_id) = active.admit(item.key.clone()) else {
                continue;
            };
            drop(active);
            drop(current);
            let service = Arc::clone(self);
            let task_key = item.key.clone();
            let runner_key = task_key.clone();
            let handle = tokio::spawn(async move {
                if let Some(outcome) = service.attempt(item).await {
                    service.publish_outcome(outcome);
                }
                let mut active = service
                    .active
                    .lock()
                    .expect("assessment owner lock poisoned");
                active.finish(&runner_key, active_id);
            });
            let abort = handle.abort_handle();
            let mut active = self.active.lock().expect("assessment owner lock poisoned");
            if !active.attach(&task_key, active_id, abort) {
                handle.abort();
            }
        }
    }

    /// One attempt at `item` within the pool's concurrency and the target deadline, sharing the
    /// host's device where it is measured; `None` when the pool is closing.
    async fn attempt(&self, item: AssessmentWork) -> Option<AssessmentOutcome> {
        let _permit = Arc::clone(&self.concurrency).acquire_owned().await.ok()?;
        let _share = self.host.share().await;
        let key = item.key.clone();
        Some(
            tokio::time::timeout(MODEL_ASSESSMENT_TIMEOUT, self.assessor.assess(item))
                .await
                .unwrap_or_else(|_| {
                    Err(InventoryError::ModelOperation {
                        code: "assessment_deadline".to_owned(),
                        message: "model assessment target deadline expired".to_owned(),
                        retryable: true,
                    })
                })
                .unwrap_or_else(|error| AssessmentOutcome {
                    key,
                    result: Err(inventory_model_failure(error)),
                }),
        )
    }

    /// Adopt what the measurement job reports (`measurement`'s current status): its state, and
    /// a new environment when its measurements change the current one.
    fn adopt_measurement(
        self: &Arc<Self>,
        measurement: &mut tokio::sync::watch::Receiver<MeasurementStatus>,
    ) {
        let status = measurement.borrow_and_update().clone();
        match self.assessor.adopt(status.measurements) {
            Some(environment) => self.adopt_environment(environment, measurement.clone()),
            None => {
                self.publish(|current| {
                    let changed = current.snapshot.measurement != status.state;
                    current.snapshot.measurement = status.state;
                    changed
                });
            }
        }
    }

    /// Move the pool to `environment`: work not yet assessed moves at once, and every assessed
    /// model is finished again in it (arithmetic over its prepared demand) on a task of its own,
    /// off the run loop. Its results are published together with `measurement`'s state then in
    /// one revision, never passing back through `Assessing`, unless the environment has moved
    /// on by then: the next one finishes them again.
    fn adopt_environment(
        self: &Arc<Self>,
        environment: Arc<AssessmentEnvironment>,
        measurement: tokio::sync::watch::Receiver<MeasurementStatus>,
    ) {
        let (rekeyed, assessed) = {
            let current = self.current.read().expect("assessment state lock poisoned");
            let rekeyed = current
                .works
                .iter()
                .map(|(key, work)| {
                    let next = ModelAssessor::work_for(
                        &environment,
                        work.configuration.clone(),
                        work.profile.profile.clone(),
                        Arc::clone(&work.preparation),
                    );
                    (key.clone(), next)
                })
                .collect::<std::collections::BTreeMap<_, _>>();
            (rekeyed, current.assessed_keys())
        };
        self.current
            .write()
            .expect("assessment state lock poisoned")
            .rekey_unassessed(&rekeyed, &assessed);
        self.cancel_unreferenced();
        self.schedule(
            rekeyed
                .iter()
                .filter(|(old, _)| !assessed.contains(*old))
                .map(|(_, work)| work.clone())
                .collect(),
        );
        let refinished = rekeyed
            .into_iter()
            .filter(|(old, _)| assessed.contains(old))
            .collect::<std::collections::BTreeMap<_, _>>();
        let service = Arc::clone(self);
        tokio::spawn(async move {
            let outcomes = futures_util::future::join_all(
                refinished
                    .values()
                    .map(|work| service.attempt(work.clone())),
            )
            .await;
            let Some(outcomes) = outcomes.into_iter().collect::<Option<Vec<_>>>() else {
                return;
            };
            let mut current = service
                .current
                .write()
                .expect("assessment state lock poisoned");
            // The environment moved on while these were finished: its own adoption finishes
            // the assessed models again.
            if service.assessor.environment().id != environment.id {
                return;
            }
            let keys = refinished
                .iter()
                .map(|(old, work)| (old.clone(), work.key.clone()))
                .collect();
            current.works.extend(
                refinished
                    .into_values()
                    .map(|work| (work.key.clone(), work)),
            );
            let state = measurement.borrow().state.clone();
            let applied = current.republish(&environment.id, state, &keys, &outcomes);
            current.snapshot.revision = current.snapshot.revision.saturating_add(1);
            let revision = current.snapshot.revision;
            service.host.observe(&current);
            drop(current);
            log_dropped_catalog_models(applied.dropped_catalog_models);
            let _ = service
                .changes
                .send(ModelAssessmentsInvalidation { revision });
        });
    }

    fn publish_outcome(&self, outcome: AssessmentOutcome) {
        let mut current = self
            .current
            .write()
            .expect("assessment state lock poisoned");
        let applied = current.apply_outcome(&outcome);
        if !applied.changed {
            return;
        }
        current.snapshot.revision = current.snapshot.revision.saturating_add(1);
        let revision = current.snapshot.revision;
        self.host.observe(&current);
        drop(current);
        log_dropped_catalog_models(applied.dropped_catalog_models);
        let _ = self.changes.send(ModelAssessmentsInvalidation { revision });
    }

    fn cancel_unreferenced(&self) {
        let referenced = {
            let current = self.current.read().expect("assessment state lock poisoned");
            current
                .entry_keys
                .values()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
        };
        let mut active = self.active.lock().expect("assessment owner lock poisoned");
        active.cancel_unreferenced(&referenced);
    }

    async fn run(
        self: Arc<Self>,
        mut measurement: tokio::sync::watch::Receiver<MeasurementStatus>,
    ) {
        // Subscribe before the initial reads so a source revision published between a read and
        // entering the select loop is buffered and reconciled rather than missed.
        let mut catalog = self.catalog.watch_catalog().skip(1);
        let mut discovered = self.local.watch_local().skip(1);
        self.adopt_measurement(&mut measurement);
        self.read_source(AssessmentDomain::Catalog).await;
        self.read_source(AssessmentDomain::Custom).await;
        let mut source_retry = tokio::time::interval(MODEL_ASSESSMENT_SOURCE_RETRY_DELAY);
        source_retry.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        source_retry.tick().await;
        let mut reporting = true;
        loop {
            tokio::select! {
                event = catalog.next() => match event {
                    Some(_) => self.read_source(AssessmentDomain::Catalog).await,
                    None => return,
                },
                event = discovered.next() => match event {
                    Some(_) => self.read_source(AssessmentDomain::Custom).await,
                    None => return,
                },
                changed = measurement.changed(), if reporting => match changed {
                    Ok(()) => self.adopt_measurement(&mut measurement),
                    // The job's owner is gone: its last state stands.
                    Err(_) => reporting = false,
                },
                _ = source_retry.tick() => {
                    let failed = self
                        .current
                        .read()
                        .expect("assessment state lock poisoned")
                        .failed_reads
                        .clone();
                    for domain in failed {
                        self.read_source(domain).await;
                    }
                },
            }
        }
    }
}

impl ModelAssessments for ManagedModelAssessments {
    fn snapshot(&self) -> BoxFuture<'_, Result<ModelAssessmentsSnapshot, InventoryError>> {
        Box::pin(async {
            Ok(self
                .current
                .read()
                .map_err(|_| InventoryError::Internal("assessment state lock poisoned".to_owned()))?
                .snapshot
                .clone())
        })
    }

    fn watch(&self) -> BoxStream<'static, ModelAssessmentsInvalidation> {
        let initial = self
            .current
            .read()
            .map(|state| state.snapshot.revision)
            .unwrap_or_default();
        let receiver = self.changes.subscribe();
        let changes = futures_util::stream::unfold(receiver, |mut receiver| async move {
            loop {
                match receiver.recv().await {
                    Ok(event) => return Some((event, receiver)),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
                }
            }
        });
        Box::pin(
            futures_util::stream::once(async move {
                ModelAssessmentsInvalidation { revision: initial }
            })
            .chain(changes),
        )
    }
}

fn log_dropped_catalog_models(dropped: Vec<(ModelId, DomainModelFailure)>) {
    for (model_id, failure) in dropped {
        tracing::error!(
            model.id = %model_id.as_str(),
            failure.code = %failure.code,
            failure.message = %failure.message,
            "catalog model assessment dropped"
        );
    }
}

fn assessment_concurrency() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |cores| cores.get().min(MAX_ASSESSMENT_CONCURRENCY))
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_service_contracts::models::{ModelPackageId, ServableModelBundle};
    use std::path::PathBuf;

    fn test_configuration(model_id: &str) -> ModelServingConfiguration {
        ModelServingConfiguration {
            bundle: ServableModelBundle::Standalone {
                package: magnitude_service_contracts::models::ModelPackage {
                    id: ModelPackageId(format!("package:{model_id}")),
                    source: magnitude_service_contracts::models::ModelPackageSource::Local {
                        path: PathBuf::from(format!("/test/{model_id}.gguf")),
                    },
                    files: Vec::new(),
                    relationships: Vec::new(),
                    properties: magnitude_service_contracts::models::ModelPackageProperties {
                        format: "gguf".to_owned(),
                        quantization: "f16".to_owned(),
                        quantization_name: "F16".to_owned(),
                        architecture: "test".to_owned(),
                        maximum_context_length: Some(8_192),
                        intrinsic_model_id: Some(model_id.to_owned()),
                        intrinsic_quality_id: None,
                    },
                },
            },
            profile: magnitude_service_contracts::models::ServingProfile {
                context_length: 8_192,
            },
        }
    }

    fn assessment_subject(
        base: &str,
        selection: CatalogModelSelection,
    ) -> magnitude_service_contracts::models::ModelAssessmentSubject {
        let base =
            magnitude_service_contracts::models::CatalogBaseId::new(base).expect("catalog base ID");
        let variant = magnitude_service_contracts::models::CatalogVariantId::new("gguf:q4")
            .expect("catalog variant ID");
        magnitude_service_contracts::models::ModelAssessmentSubject::Catalog {
            model_id: ModelId::catalog(&base, &variant),
            selection,
        }
    }

    /// A target being assessed under `key`. Its work is the pool's to run, which these tests
    /// do not.
    fn assessment_target(
        subject: magnitude_service_contracts::models::ModelAssessmentSubject,
        key: &str,
    ) -> AssessmentTarget {
        AssessmentTarget {
            entry: ModelAssessmentEntry {
                subject,
                state: ModelAssessmentEntryState::Assessing,
            },
            key: AssessmentWorkKey(key.to_owned()),
            work: None,
        }
    }

    /// Where the measured device is the host's cores, an attempt's share keeps a job from taking
    /// it, and waits for one that holds it to end without preempting it; where it is not, an
    /// attempt shares nothing.
    #[tokio::test(start_paused = true)]
    async fn attempts_share_the_host_cores_with_the_measurement_job() {
        let device = DeviceLock::default();
        let share = HostDevice::new(&device, true).share().await;
        assert!(share.is_some());
        assert!(device.try_job().is_none());
        drop(share);
        let mut job = device.try_job().unwrap();
        let host = HostDevice::new(&device, true);
        let waiting = tokio::spawn(async move { host.share().await.is_some() });
        tokio::select! {
            () = job.preempted() => panic!("an attempt preempted the job"),
            () = tokio::time::sleep(std::time::Duration::from_secs(60)) => {}
        }
        assert!(!waiting.is_finished());
        drop(job);
        assert!(waiting.await.unwrap());
        assert!(HostDevice::new(&device, false).share().await.is_none());
        assert!(device.try_job().is_some());
    }

    /// The pool settles for the job once every slice is read and no entry is assessing; a GPU's
    /// job never waits for it.
    #[test]
    fn the_pool_settles_once_nothing_is_assessing() {
        let device = DeviceLock::default();
        let host = HostDevice::new(&device, true);
        let settled = host.settled();
        let mut current = assessment_pool_current();
        host.observe(&current);
        assert!(!*settled.borrow(), "slices unread");
        let subject = assessment_subject("a", CatalogModelSelection::Desired);
        current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[assessment_target(subject.clone(), "work-a")],
        );
        current.replace_domain(AssessmentDomain::Custom, 1, &[]);
        host.observe(&current);
        assert!(!*settled.borrow(), "an entry assessing");
        current.apply_outcome(&AssessmentOutcome {
            key: AssessmentWorkKey("work-a".to_owned()),
            result: Ok(fits(
                &subject,
                magnitude_service_contracts::models::ModelPerformance::Pending,
            )),
        });
        host.observe(&current);
        assert!(*settled.borrow());
        assert!(*HostDevice::new(&device, false).settled().borrow());
    }

    fn assessment_pool_current() -> AssessmentPoolCurrent {
        AssessmentPoolCurrent {
            snapshot: ModelAssessmentsSnapshot {
                revision: 0,
                environment_id: AssessmentEnvironmentId("test-environment".to_owned()),
                catalog: ModelAssessmentDomainSnapshot::Pending { source_revision: 0 },
                custom: ModelAssessmentDomainSnapshot::Pending { source_revision: 0 },
                measurement: MeasurementState::Unmeasured,
            },
            entry_keys: std::collections::BTreeMap::new(),
            works: std::collections::BTreeMap::new(),
            revalidating: std::collections::BTreeSet::new(),
            failed_reads: std::collections::BTreeSet::new(),
        }
    }

    fn fits(
        subject: &ModelAssessmentSubject,
        performance: magnitude_service_contracts::models::ModelPerformance,
    ) -> CachedModelAssessment {
        let profile = test_configuration(subject.model_id().as_str()).profile;
        CachedModelAssessment {
            capabilities: magnitude_service_contracts::models::ModelCapabilities {
                vision: false,
                tools: false,
                structured_output: false,
                reasoning: magnitude_service_contracts::models::ModelReasoningCapabilities {
                    supported: false,
                    efforts: Vec::new(),
                    default_effort: None,
                },
            },
            template_fingerprint: String::new(),
            fingerprint: None,
            profile: ModelAssessment::Fits {
                profile,
                assessment_id: magnitude_service_contracts::models::ModelAssessmentId(
                    "assessment".to_owned(),
                ),
                memory: Vec::new(),
                performance,
                projector: magnitude_service_contracts::models::ProjectorVerdict::None,
                drafter: magnitude_service_contracts::models::DrafterVerdict::None,
            },
        }
    }

    /// Measurements arriving finish every assessed model again and publish them all, with the
    /// new environment and the job's state, as one change: no entry shows `Assessing`, and each
    /// carries its estimate.
    #[test]
    fn arriving_measurements_republish_every_assessed_model_at_once() {
        use magnitude_service_contracts::models::{ModelPerformance, PerformanceEvidence};
        let subjects = ["first", "second"]
            .map(|base| assessment_subject(base, CatalogModelSelection::Desired));
        let mut current = assessment_pool_current();
        current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[
                assessment_target(subjects[0].clone(), "first-pending"),
                assessment_target(subjects[1].clone(), "second-pending"),
            ],
        );
        for (subject, key) in subjects.iter().zip(["first-pending", "second-pending"]) {
            current.apply_outcome(&AssessmentOutcome {
                key: AssessmentWorkKey(key.to_owned()),
                result: Ok(fits(subject, ModelPerformance::Pending)),
            });
        }
        let assessed = current.assessed_keys();
        assert_eq!(assessed.len(), 2);
        let estimated = ModelPerformance::Estimated {
            samples: vec![PerformanceEvidence {
                context_tokens: 8_192,
                estimated_tokens_per_second: 40.0,
            }],
        };
        let key = |key: &str| AssessmentWorkKey(key.to_owned());
        let rekeyed = [
            (key("first-pending"), key("first-measured")),
            (key("second-pending"), key("second-measured")),
        ]
        .into_iter()
        .collect();
        let outcomes = subjects
            .iter()
            .zip(["first-measured", "second-measured"])
            .map(|(subject, key)| AssessmentOutcome {
                key: AssessmentWorkKey(key.to_owned()),
                result: Ok(fits(subject, estimated.clone())),
            })
            .collect::<Vec<_>>();
        let measured = AssessmentEnvironmentId("measured-environment".to_owned());
        let applied = current.republish(&measured, MeasurementState::Measured, &rekeyed, &outcomes);
        assert!(applied.changed);
        assert_eq!(current.snapshot.environment_id, measured);
        assert_eq!(current.snapshot.measurement, MeasurementState::Measured);
        let ModelAssessmentDomainSnapshot::Available { entries, .. } =
            current.domain(AssessmentDomain::Catalog)
        else {
            panic!("the catalog slice is available")
        };
        for entry in entries {
            assert!(matches!(
                &entry.state,
                ModelAssessmentEntryState::Assessed { profiles, .. }
                    if matches!(profiles.as_slice(), [ModelAssessment::Fits { performance, .. }]
                        if *performance == estimated)
            ));
        }
        assert!(
            current
                .assessed_keys()
                .iter()
                .all(|key| key.0.ends_with("-measured"))
        );
    }

    /// QA: once measurements arrived, a change to the local models read before the assessed
    /// models were finished again in the new environment sent every one of them back to
    /// `Assessing`, so a finished catalog showed its assessment progress again from the start.
    /// An assessed model keeps its result until its result in the new environment lands.
    #[test]
    fn a_source_read_while_measurements_land_keeps_every_assessed_model() {
        use magnitude_service_contracts::models::{ModelPerformance, PerformanceEvidence};
        let subject = assessment_subject("refinished", CatalogModelSelection::Desired);
        let mut current = assessment_pool_current();
        current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[assessment_target(subject.clone(), "pending")],
        );
        current.apply_outcome(&AssessmentOutcome {
            key: AssessmentWorkKey("pending".to_owned()),
            result: Ok(fits(&subject, ModelPerformance::Pending)),
        });
        let catalog = seismic::Devices::discover(&[]).expect("device catalog");
        let structure = magnitude_engine::assessment::AssessmentStructure::discover(
            &catalog,
            magnitude_executor::platform::DeviceRequest::Backend(seismic::BackendName::Cpu),
            magnitude_executor::platform::MemoryReserves::standard(),
            environment::serving_policy(),
            magnitude_engine::options::standard_service_limits(),
        )
        .expect("the CPU backend is selectable");
        let environment = Arc::new(
            AssessmentEnvironment::new(
                Arc::new(structure),
                magnitude_engine::assessment::Measurements::Pending,
            )
            .expect("the CPU environment"),
        );
        let configuration = test_configuration(subject.model_id().as_str());
        let profile = configuration.profile.clone();
        let work = ModelAssessor::work_for(
            &environment,
            configuration,
            profile,
            Arc::new(tokio::sync::OnceCell::new()),
        );
        let measured = work.key.clone();
        let pending = AssessmentWorkKey("pending".to_owned());
        let assessed = current.assessed_keys();
        current.rekey_unassessed(&[(pending.clone(), work)].into_iter().collect(), &assessed);
        current.replace_domain(
            AssessmentDomain::Catalog,
            2,
            &[assessment_target(subject.clone(), &measured.0)],
        );
        assert!(
            matches!(
                current.domain(AssessmentDomain::Catalog),
                ModelAssessmentDomainSnapshot::Available { entries, .. }
                    if matches!(entries.as_slice(), [ModelAssessmentEntry {
                        state: ModelAssessmentEntryState::Assessed { .. }, ..
                    }])
            ),
            "the read keeps the assessed result"
        );
        assert!(current.references_assessing(&measured));
        let estimated = ModelPerformance::Estimated {
            samples: vec![PerformanceEvidence {
                context_tokens: 8_192,
                estimated_tokens_per_second: 40.0,
            }],
        };
        current.republish(
            &environment.id,
            MeasurementState::Measured,
            &[(pending, measured.clone())].into_iter().collect(),
            &[AssessmentOutcome {
                key: measured.clone(),
                result: Ok(fits(&subject, estimated.clone())),
            }],
        );
        assert_eq!(current.entry_keys.get(&subject), Some(&measured));
        assert!(matches!(
            current.domain(AssessmentDomain::Catalog),
            ModelAssessmentDomainSnapshot::Available { entries, .. }
                if matches!(entries.as_slice(), [ModelAssessmentEntry {
                    state: ModelAssessmentEntryState::Assessed { profiles, .. }, ..
                }] if matches!(profiles.as_slice(), [ModelAssessment::Fits { performance, .. }]
                    if *performance == estimated))
        ));
    }

    /// QA: once the catalog learned that a model with a drafter fits only without it, its new
    /// configuration was a new target, and the model showed `Assessing` again (77 of 78) until
    /// that target was assessed. A model keeps its result while its new target is assessed.
    #[test]
    fn a_model_whose_target_changes_keeps_its_result_until_the_new_one_lands() {
        use magnitude_service_contracts::models::{ModelPerformance, PerformanceEvidence};
        let subject = assessment_subject("redrafted", CatalogModelSelection::Desired);
        let mut current = assessment_pool_current();
        current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[assessment_target(subject.clone(), "with-drafter")],
        );
        current.apply_outcome(&AssessmentOutcome {
            key: AssessmentWorkKey("with-drafter".to_owned()),
            result: Ok(fits(&subject, ModelPerformance::Pending)),
        });
        let plain = AssessmentWorkKey("plain".to_owned());
        assert!(current.replace_domain(
            AssessmentDomain::Catalog,
            2,
            &[assessment_target(subject.clone(), &plain.0)],
        ));
        let entry_state = |current: &AssessmentPoolCurrent| match current
            .domain(AssessmentDomain::Catalog)
        {
            ModelAssessmentDomainSnapshot::Available { entries, .. } => entries[0].state.clone(),
            ModelAssessmentDomainSnapshot::Pending { .. } => {
                panic!("the catalog slice is available")
            }
        };
        assert!(matches!(
            entry_state(&current),
            ModelAssessmentEntryState::Assessed { profiles, .. }
                if matches!(profiles.as_slice(), [ModelAssessment::Fits { performance: ModelPerformance::Pending, .. }])
        ));
        assert!(
            current.references_assessing(&plain),
            "the new target is scheduled"
        );
        assert!(
            current.assessed_keys().is_empty(),
            "no environment finishes the old result again"
        );
        let estimated = ModelPerformance::Estimated {
            samples: vec![PerformanceEvidence {
                context_tokens: 8_192,
                estimated_tokens_per_second: 40.0,
            }],
        };
        assert!(
            current
                .apply_outcome(&AssessmentOutcome {
                    key: plain.clone(),
                    result: Ok(fits(&subject, estimated.clone())),
                })
                .changed
        );
        assert!(matches!(
            entry_state(&current),
            ModelAssessmentEntryState::Assessed { profiles, .. }
                if matches!(profiles.as_slice(), [ModelAssessment::Fits { performance, .. }] if *performance == estimated)
        ));
        assert!(!current.references_assessing(&plain));
        assert_eq!(
            current.assessed_keys().into_iter().collect::<Vec<_>>(),
            vec![plain]
        );
    }

    fn assessment_failure() -> DomainModelFailure {
        DomainModelFailure {
            code: "assessment_failed".to_owned(),
            message: "assessment failure".to_owned(),
            retryable: true,
        }
    }

    #[test]
    fn assessment_pool_replaces_equal_visible_state_when_exact_work_changes() {
        let subject = assessment_subject("pool-identity", CatalogModelSelection::Desired);
        let mut current = assessment_pool_current();
        assert!(current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[assessment_target(subject.clone(), "old")],
        ));
        assert!(current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[assessment_target(subject.clone(), "new")],
        ));
        assert_eq!(
            current.entry_keys.get(&subject),
            Some(&AssessmentWorkKey("new".to_owned()))
        );
    }

    #[test]
    fn assessment_pool_ignores_completion_for_superseded_exact_work() {
        let subject = assessment_subject("pool-stale", CatalogModelSelection::Desired);
        let mut current = assessment_pool_current();
        current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[assessment_target(subject.clone(), "old")],
        );
        current.replace_domain(
            AssessmentDomain::Catalog,
            2,
            &[assessment_target(subject, "new")],
        );
        assert!(
            !current
                .apply_outcome(&AssessmentOutcome {
                    key: AssessmentWorkKey("old".to_owned()),
                    result: Err(assessment_failure()),
                })
                .changed
        );
        assert!(current.references_assessing(&AssessmentWorkKey("new".to_owned())));
    }

    #[test]
    fn assessment_pool_reuses_exact_terminal_evidence_across_source_slices() {
        let catalog = assessment_subject("pool-shared", CatalogModelSelection::Desired);
        let discovered = magnitude_service_contracts::models::ModelAssessmentSubject::Custom {
            model_id: catalog.model_id().clone(),
        };
        let key = AssessmentWorkKey("shared".to_owned());
        let mut current = assessment_pool_current();
        current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[assessment_target(catalog, "shared")],
        );
        assert!(
            current
                .apply_outcome(&AssessmentOutcome {
                    key: key.clone(),
                    result: Err(DomainModelFailure {
                        code: "invalid_artifact".to_owned(),
                        message: "invalid artifact".to_owned(),
                        retryable: false,
                    }),
                })
                .changed
        );
        current.replace_domain(
            AssessmentDomain::Custom,
            3,
            &[assessment_target(discovered, "shared")],
        );
        let discovered = current.domain(AssessmentDomain::Custom);
        assert!(matches!(
            discovered,
            ModelAssessmentDomainSnapshot::Available { entries, .. }
                if matches!(entries.as_slice(), [ModelAssessmentEntry {
                    state: ModelAssessmentEntryState::Dropped,
                    ..
                }])
        ));
    }

    #[test]
    fn assessment_pool_drops_failures_without_retrying_the_same_work() {
        let subject = assessment_subject("pool-drop", CatalogModelSelection::Desired);
        let key = AssessmentWorkKey("drop".to_owned());
        let mut current = assessment_pool_current();
        current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[assessment_target(subject.clone(), "drop")],
        );
        assert!(
            current
                .apply_outcome(&AssessmentOutcome {
                    key: key.clone(),
                    result: Err(assessment_failure()),
                })
                .changed
        );
        assert!(!current.references_assessing(&key));
        current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[assessment_target(subject, "drop")],
        );
        assert!(!current.references_assessing(&key));
    }

    #[test]
    fn an_unsupported_catalog_model_is_dropped_and_a_custom_one_is_assessed() {
        let catalog = assessment_subject("pool-unsupported", CatalogModelSelection::Desired);
        let discovered = magnitude_service_contracts::models::ModelAssessmentSubject::Custom {
            model_id: catalog.model_id().clone(),
        };
        let failure = DomainModelFailure {
            code: "unsupported".to_owned(),
            message: "`routed_output` has no admissible metal configuration".to_owned(),
            retryable: false,
        };
        let mut current = assessment_pool_current();
        current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[assessment_target(catalog.clone(), "unsupported")],
        );
        current.replace_domain(
            AssessmentDomain::Custom,
            1,
            &[assessment_target(discovered, "unsupported")],
        );
        let applied = current.apply_outcome(&AssessmentOutcome {
            key: AssessmentWorkKey("unsupported".to_owned()),
            result: Ok(CachedModelAssessment {
                capabilities: magnitude_service_contracts::models::ModelCapabilities {
                    vision: false,
                    tools: false,
                    structured_output: false,
                    reasoning: magnitude_service_contracts::models::ModelReasoningCapabilities {
                        supported: false,
                        efforts: Vec::new(),
                        default_effort: None,
                    },
                },
                template_fingerprint: String::new(),
                fingerprint: None,
                profile: ModelAssessment::Unsupported {
                    profile: Some(test_configuration(catalog.model_id().as_str()).profile),
                    reasons: Vec::new(),
                    detail: failure.message.clone(),
                },
            }),
        });
        assert_eq!(
            applied.dropped_catalog_models,
            vec![(catalog.model_id().clone(), failure)]
        );
        let state = |domain| match current.domain(domain) {
            ModelAssessmentDomainSnapshot::Available { entries, .. } => entries[0].state.clone(),
            other => panic!("expected an available slice, got {other:?}"),
        };
        assert!(matches!(
            state(AssessmentDomain::Catalog),
            ModelAssessmentEntryState::Dropped
        ));
        assert!(matches!(
            state(AssessmentDomain::Custom),
            ModelAssessmentEntryState::Assessed {
                profiles,
                ..
            } if matches!(profiles.as_slice(), [ModelAssessment::Unsupported { .. }])
        ));
    }

    #[test]
    fn stale_assessment_task_cannot_remove_its_replacement() {
        let key = AssessmentWorkKey("generation".to_owned());
        let mut active = ActiveAssessments::new();
        let first = active.admit(key.clone()).expect("first admission");
        active.cancel_unreferenced(&std::collections::BTreeSet::new());
        let replacement = active.admit(key.clone()).expect("replacement admission");
        assert_ne!(first, replacement);
        active.finish(&key, first);
        assert!(active.entries.contains_key(&key));
        active.finish(&key, replacement);
        assert!(!active.entries.contains_key(&key));
    }

    #[test]
    fn a_source_slice_updates_without_disturbing_the_other_slice() {
        let catalog = assessment_subject("pool-source", CatalogModelSelection::Desired);
        let discovered = magnitude_service_contracts::models::ModelAssessmentSubject::Custom {
            model_id: catalog.model_id().clone(),
        };
        let mut current = assessment_pool_current();
        current.replace_domain(
            AssessmentDomain::Catalog,
            2,
            &[assessment_target(catalog, "catalog")],
        );
        current.replace_domain(
            AssessmentDomain::Custom,
            3,
            &[assessment_target(discovered.clone(), "discovered")],
        );
        assert!(current.set_domain_pending(AssessmentDomain::Custom, 4));
        assert!(!current.set_domain_pending(AssessmentDomain::Custom, 4));
        assert!(current.references_assessing(&AssessmentWorkKey("catalog".to_owned())));
        assert!(!current.references_assessing(&AssessmentWorkKey("discovered".to_owned())));
        assert!(current.replace_domain(
            AssessmentDomain::Custom,
            5,
            &[assessment_target(discovered, "discovered")],
        ));
        assert!(current.references_assessing(&AssessmentWorkKey("discovered".to_owned())));
    }

    #[test]
    fn dropped_target_does_not_request_source_retry() {
        let subject = assessment_subject("pool-resolution", CatalogModelSelection::Desired);
        let key = AssessmentWorkKey("unresolved".to_owned());
        let mut current = assessment_pool_current();
        assert!(current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[AssessmentTarget {
                entry: ModelAssessmentEntry {
                    subject: subject.clone(),
                    state: ModelAssessmentEntryState::Dropped,
                },
                key: key.clone(),
                work: None,
            }],
        ));
        assert!(current.failed_reads.is_empty());
        assert!(current.references_dropped(&subject, &key));
        assert!(!current.replace_domain(
            AssessmentDomain::Catalog,
            1,
            &[AssessmentTarget {
                entry: ModelAssessmentEntry {
                    subject,
                    state: ModelAssessmentEntryState::Assessing,
                },
                key,
                work: None,
            }],
        ));
    }
}
