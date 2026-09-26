//! The automatic assessment pool: one revisioned snapshot over the current catalog and discovery
//! sources, with exact-work reconciliation, deduplication, per-target deadlines, publication
//! guards and one-attempt `Dropped` semantics.

pub mod assessor;
pub mod environment;
pub mod measurement;

use std::sync::{Arc, Mutex};

use futures_util::{StreamExt, future::BoxFuture, stream::BoxStream};
use magnitude_service_contracts::InventoryError;
use magnitude_service_contracts::models::{
    CatalogModelSelection, CatalogModelState, CatalogModels, DiscoveredModelState,
    DiscoveredModels, EffectiveModel, ModelAssessmentDomainSnapshot, ModelAssessmentEntry,
    ModelAssessmentEntryState, ModelAssessmentPoolState, ModelAssessments,
    ModelAssessmentsInvalidation, ModelAssessmentsSnapshot, ModelFailure as DomainModelFailure,
    ModelId,
};

use assessor::{
    AssessmentOutcome, AssessmentWork, AssessmentWorkKey, ModelAssessor, inventory_model_failure,
};
use environment::AssessmentEnvironment;

/// Bounds one target's whole attempt: material resolution and the engine's arithmetic.
const MODEL_ASSESSMENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
const MODEL_ASSESSMENT_SOURCE_RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(1);
const ENVIRONMENT_RETRY_INITIAL: std::time::Duration = std::time::Duration::from_secs(1);
const ENVIRONMENT_RETRY_MAXIMUM: std::time::Duration = std::time::Duration::from_secs(60);
/// Concurrent targets, each holding at most one blocking-pool task of header arithmetic.
const MAX_ASSESSMENT_CONCURRENCY: usize = 8;

struct AssessmentTarget {
    entry: ModelAssessmentEntry,
    key: AssessmentWorkKey,
    work: Option<AssessmentWork>,
}

#[derive(Clone, Copy)]
enum AssessmentDomain {
    Catalog,
    Discovered,
}

struct AssessmentPoolCurrent {
    snapshot: ModelAssessmentsSnapshot,
    entry_keys: std::collections::BTreeMap<
        magnitude_service_contracts::models::ModelAssessmentSubject,
        AssessmentWorkKey,
    >,
}

fn assessment_domain_entries(domain: &ModelAssessmentDomainSnapshot) -> &[ModelAssessmentEntry] {
    match domain {
        ModelAssessmentDomainSnapshot::Available { entries, .. } => entries,
        ModelAssessmentDomainSnapshot::Pending { .. }
        | ModelAssessmentDomainSnapshot::Failed { .. } => &[],
    }
}

fn assessment_domain_entries_mut(
    domain: &mut ModelAssessmentDomainSnapshot,
) -> Option<&mut Vec<ModelAssessmentEntry>> {
    match domain {
        ModelAssessmentDomainSnapshot::Available { entries, .. } => Some(entries),
        ModelAssessmentDomainSnapshot::Pending { .. }
        | ModelAssessmentDomainSnapshot::Failed { .. } => None,
    }
}

fn assessment_domain_source_revision(domain: &ModelAssessmentDomainSnapshot) -> u64 {
    match domain {
        ModelAssessmentDomainSnapshot::Pending { source_revision }
        | ModelAssessmentDomainSnapshot::Available {
            source_revision, ..
        }
        | ModelAssessmentDomainSnapshot::Failed {
            source_revision, ..
        } => *source_revision,
    }
}

impl AssessmentPoolCurrent {
    fn ready_domains(
        &self,
    ) -> Option<(
        &ModelAssessmentDomainSnapshot,
        &ModelAssessmentDomainSnapshot,
    )> {
        match &self.snapshot.state {
            ModelAssessmentPoolState::Ready {
                catalog,
                discovered,
                ..
            } => Some((catalog, discovered)),
            _ => None,
        }
    }

    fn domain(&self, domain: AssessmentDomain) -> Option<&ModelAssessmentDomainSnapshot> {
        let (catalog, discovered) = self.ready_domains()?;
        Some(match domain {
            AssessmentDomain::Catalog => catalog,
            AssessmentDomain::Discovered => discovered,
        })
    }

    fn domain_mut(
        &mut self,
        domain: AssessmentDomain,
    ) -> Option<&mut ModelAssessmentDomainSnapshot> {
        let ModelAssessmentPoolState::Ready {
            catalog,
            discovered,
            ..
        } = &mut self.snapshot.state
        else {
            return None;
        };
        Some(match domain {
            AssessmentDomain::Catalog => catalog,
            AssessmentDomain::Discovered => discovered,
        })
    }

    fn retained_terminal_states(
        &self,
    ) -> std::collections::BTreeMap<AssessmentWorkKey, ModelAssessmentEntryState> {
        self.ready_domains()
            .into_iter()
            .flat_map(|(catalog, discovered)| [catalog, discovered])
            .flat_map(assessment_domain_entries)
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
        let entries = targets
            .iter()
            .map(|target| {
                let mut entry = target.entry.clone();
                if let Some(state) = work_by_entry
                    .get(&entry.subject)
                    .and_then(|key| retained.get(key))
                {
                    entry.state = state.clone();
                }
                entry
            })
            .collect();
        let next = ModelAssessmentDomainSnapshot::Available {
            source_revision,
            entries,
        };
        let Some(previous) = self.domain(domain) else {
            return false;
        };
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
        }
        self.entry_keys.extend(work_by_entry);
        *self
            .domain_mut(domain)
            .expect("ready assessment domain was checked") = next;
        true
    }

    fn set_domain_pending(&mut self, domain: AssessmentDomain, source_revision: u64) -> bool {
        let next = ModelAssessmentDomainSnapshot::Pending { source_revision };
        let Some(previous) = self.domain(domain) else {
            return false;
        };
        if previous == &next {
            return false;
        }
        let old_subjects = assessment_domain_entries(previous)
            .iter()
            .map(|entry| entry.subject.clone())
            .collect::<Vec<_>>();
        for subject in old_subjects {
            self.entry_keys.remove(&subject);
        }
        *self
            .domain_mut(domain)
            .expect("ready assessment domain was checked") = next;
        true
    }

    fn set_domain_failure(
        &mut self,
        domain: AssessmentDomain,
        failure: DomainModelFailure,
    ) -> bool {
        let Some(previous) = self.domain(domain) else {
            return false;
        };
        let next = ModelAssessmentDomainSnapshot::Failed {
            source_revision: assessment_domain_source_revision(previous),
            failure,
        };
        if previous == &next {
            return false;
        }
        let old_subjects = assessment_domain_entries(previous)
            .iter()
            .map(|entry| entry.subject.clone())
            .collect::<Vec<_>>();
        for subject in old_subjects {
            self.entry_keys.remove(&subject);
        }
        *self
            .domain_mut(domain)
            .expect("ready assessment domain was checked") = next;
        true
    }

    fn domain_source_failed(&self, domain: AssessmentDomain) -> bool {
        self.domain(domain)
            .is_some_and(|state| matches!(state, ModelAssessmentDomainSnapshot::Failed { .. }))
    }

    fn references_assessing(&self, key: &AssessmentWorkKey) -> bool {
        self.ready_domains().is_some_and(|(catalog, discovered)| {
            [catalog, discovered].into_iter().any(|domain| {
                assessment_domain_entries(domain).iter().any(|entry| {
                    self.entry_keys.get(&entry.subject) == Some(key)
                        && matches!(entry.state, ModelAssessmentEntryState::Assessing)
                })
            })
        })
    }

    fn references_dropped(
        &self,
        subject: &magnitude_service_contracts::models::ModelAssessmentSubject,
        key: &AssessmentWorkKey,
    ) -> bool {
        self.entry_keys.get(subject) == Some(key)
            && self.ready_domains().is_some_and(|(catalog, discovered)| {
                [catalog, discovered].into_iter().any(|domain| {
                    assessment_domain_entries(domain).iter().any(|entry| {
                        &entry.subject == subject
                            && matches!(entry.state, ModelAssessmentEntryState::Dropped)
                    })
                })
            })
    }

    fn apply_outcome(&mut self, outcome: &AssessmentOutcome) -> AppliedAssessmentOutcome {
        let ModelAssessmentPoolState::Ready {
            catalog,
            discovered,
            ..
        } = &mut self.snapshot.state
        else {
            return AppliedAssessmentOutcome::default();
        };
        let mut applied = AppliedAssessmentOutcome::default();
        for (domain_kind, domain) in [
            (AssessmentDomain::Catalog, catalog),
            (AssessmentDomain::Discovered, discovered),
        ] {
            let Some(entries) = assessment_domain_entries_mut(domain) else {
                continue;
            };
            for entry in entries.iter_mut() {
                if self.entry_keys.get(&entry.subject) == Some(&outcome.key)
                    && matches!(entry.state, ModelAssessmentEntryState::Assessing)
                {
                    entry.state = match &outcome.result {
                        Ok(assessment) => ModelAssessmentEntryState::Assessed {
                            capabilities: assessment.capabilities.clone(),
                            template_fingerprint: assessment.template_fingerprint.clone(),
                            profiles: vec![assessment.profile.clone()],
                        },
                        Err(_) => ModelAssessmentEntryState::Dropped,
                    };
                    applied.changed = true;
                    if matches!(domain_kind, AssessmentDomain::Catalog) && outcome.result.is_err() {
                        applied
                            .dropped_catalog_models
                            .push(entry.subject.model_id().clone());
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
    dropped_catalog_models: Vec<ModelId>,
}

pub struct ManagedModelAssessments {
    assessor: Arc<ModelAssessor>,
    catalog: Arc<dyn CatalogModels>,
    discovery: Arc<dyn DiscoveredModels>,
    current: std::sync::RwLock<AssessmentPoolCurrent>,
    changes: tokio::sync::broadcast::Sender<ModelAssessmentsInvalidation>,
    active: Mutex<ActiveAssessments>,
    concurrency: Arc<tokio::sync::Semaphore>,
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
    /// Start the pool. It is `Preparing` until the assessment environment, with its measurement
    /// basis, is established.
    pub fn start(
        assessor: Arc<ModelAssessor>,
        catalog: Arc<dyn CatalogModels>,
        discovery: Arc<dyn DiscoveredModels>,
    ) -> Arc<Self> {
        let (changes, _) = tokio::sync::broadcast::channel(64);
        let concurrency = assessment_concurrency();
        let service = Arc::new(Self {
            assessor,
            catalog,
            discovery,
            current: std::sync::RwLock::new(AssessmentPoolCurrent {
                snapshot: ModelAssessmentsSnapshot {
                    revision: 0,
                    state: ModelAssessmentPoolState::Preparing,
                },
                entry_keys: std::collections::BTreeMap::new(),
            }),
            changes,
            active: Mutex::new(ActiveAssessments::new()),
            concurrency: Arc::new(tokio::sync::Semaphore::new(concurrency)),
        });
        let owner = Arc::clone(&service);
        tokio::spawn(async move { owner.run().await });
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
        drop(current);
        let _ = self.changes.send(ModelAssessmentsInvalidation { revision });
        true
    }

    fn publish_pool_failure(&self, error: InventoryError) {
        let failure = inventory_model_failure(error);
        self.publish(|current| {
            let next = ModelAssessmentPoolState::Failed { failure };
            if current.snapshot.state == next {
                return false;
            }
            current.entry_keys.clear();
            current.snapshot.state = next;
            true
        });
        self.cancel_unreferenced();
    }

    fn publish_domain_failure(&self, domain: AssessmentDomain, error: InventoryError) {
        let failure = inventory_model_failure(error);
        self.publish(|current| current.set_domain_failure(domain, failure));
        self.cancel_unreferenced();
    }

    fn target_entry(
        &self,
        domain: AssessmentDomain,
        environment: &AssessmentEnvironment,
        subject: magnitude_service_contracts::models::ModelAssessmentSubject,
        profile: magnitude_service_contracts::models::ServingProfile,
    ) -> AssessmentTarget {
        let unresolved_key = AssessmentWorkKey(
            serde_json::to_string(&(&environment.id, &subject, &profile, "unresolved"))
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
        match self.assessor.work_for(&environment.id, &subject, profile) {
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

    async fn reconcile_catalog(
        self: &Arc<Self>,
        environment: &Arc<AssessmentEnvironment>,
    ) -> Result<(), InventoryError> {
        let source = self.catalog.list_catalog().await?;
        let mut targets = Vec::new();
        for model in source.models {
            let selected = match model.local_state {
                CatalogModelState::NotInstalled => {
                    Some((CatalogModelSelection::Desired, model.desired.profile))
                }
                CatalogModelState::Installed {
                    effective: EffectiveModel::Ready { model },
                    ..
                } => Some((CatalogModelSelection::Effective, model.profile)),
                CatalogModelState::Installed {
                    effective: EffectiveModel::Unavailable { .. },
                    ..
                } => None,
            };
            let Some((selection, profile)) = selected else {
                continue;
            };
            let subject = magnitude_service_contracts::models::ModelAssessmentSubject::Catalog {
                model_id: model.id,
                selection,
            };
            targets.push(self.target_entry(
                AssessmentDomain::Catalog,
                environment,
                subject,
                profile,
            ));
        }
        let work = targets
            .iter()
            .filter_map(|target| target.work.clone())
            .collect::<Vec<_>>();
        self.replace_domain(AssessmentDomain::Catalog, source.revision, targets);
        self.schedule(work, environment);
        Ok(())
    }

    async fn reconcile_discovered(
        self: &Arc<Self>,
        environment: &Arc<AssessmentEnvironment>,
    ) -> Result<(), InventoryError> {
        let source = self.discovery.list_discovered().await?;
        if !source.reconciliation_complete {
            self.publish(|current| {
                current.set_domain_pending(AssessmentDomain::Discovered, source.revision)
            });
            self.cancel_unreferenced();
            return Ok(());
        }
        let mut targets = Vec::new();
        for model in source.models {
            let DiscoveredModelState::Ready { model: ready, .. } = model.state else {
                continue;
            };
            let subject = magnitude_service_contracts::models::ModelAssessmentSubject::Discovery {
                model_id: model.id,
            };
            targets.push(self.target_entry(
                AssessmentDomain::Discovered,
                environment,
                subject,
                ready.profile,
            ));
        }
        let work = targets
            .iter()
            .filter_map(|target| target.work.clone())
            .collect::<Vec<_>>();
        self.replace_domain(AssessmentDomain::Discovered, source.revision, targets);
        self.schedule(work, environment);
        Ok(())
    }

    fn replace_domain(
        &self,
        domain: AssessmentDomain,
        source_revision: u64,
        targets: Vec<AssessmentTarget>,
    ) {
        self.publish(|current| current.replace_domain(domain, source_revision, &targets));
        self.cancel_unreferenced();
    }

    fn schedule(
        self: &Arc<Self>,
        work: Vec<AssessmentWork>,
        environment: &Arc<AssessmentEnvironment>,
    ) {
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
            let task_environment = Arc::clone(environment);
            let handle = tokio::spawn(async move {
                let Ok(_permit) = Arc::clone(&service.concurrency).acquire_owned().await else {
                    let mut active = service
                        .active
                        .lock()
                        .expect("assessment owner lock poisoned");
                    active.finish(&runner_key, active_id);
                    return;
                };
                let outcome = tokio::time::timeout(
                    MODEL_ASSESSMENT_TIMEOUT,
                    service.assessor.assess(item, task_environment),
                )
                .await
                .unwrap_or_else(|_| {
                    Err(InventoryError::ModelOperation {
                        code: "assessment_deadline".to_owned(),
                        message: "model assessment target deadline expired".to_owned(),
                        retryable: true,
                    })
                })
                .unwrap_or_else(|error| AssessmentOutcome {
                    key: runner_key.clone(),
                    result: Err(inventory_model_failure(error)),
                });
                drop(_permit);
                service.publish_outcome(outcome);
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

    fn publish_outcome(&self, outcome: AssessmentOutcome) {
        let failure = outcome.result.as_ref().err().cloned();
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
        drop(current);
        if let Some(failure) = failure {
            for model_id in applied.dropped_catalog_models {
                tracing::error!(
                    model.id = %model_id.as_str(),
                    failure.code = %failure.code,
                    failure.message = %failure.message,
                    "catalog model assessment dropped"
                );
            }
        }
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

    async fn run(self: Arc<Self>) {
        let mut retry = ENVIRONMENT_RETRY_INITIAL;
        let environment = loop {
            match self.assessor.establish_environment().await {
                Ok(environment) => break Arc::new(environment),
                Err(error) => {
                    tracing::error!(%error, "assessment environment unavailable");
                    self.publish_pool_failure(InventoryError::ModelOperation {
                        code: "assessment_environment_unavailable".to_owned(),
                        message: error.to_string(),
                        retryable: true,
                    });
                    tokio::time::sleep(retry).await;
                    retry = (retry * 2).min(ENVIRONMENT_RETRY_MAXIMUM);
                }
            }
        };
        self.publish(|current| {
            current.snapshot.state = ModelAssessmentPoolState::Ready {
                environment_id: environment.id.clone(),
                catalog: ModelAssessmentDomainSnapshot::Pending { source_revision: 0 },
                discovered: ModelAssessmentDomainSnapshot::Pending { source_revision: 0 },
            };
            true
        });
        // Subscribe before the initial reads so a source revision published between a read and
        // entering the select loop is buffered and reconciled rather than missed.
        let mut catalog = self.catalog.watch_catalog().skip(1);
        let mut discovered = self.discovery.watch_discovery().skip(1);
        if let Err(error) = self.reconcile_catalog(&environment).await {
            tracing::warn!(%error, "catalog assessment reconciliation failed");
            self.publish_domain_failure(AssessmentDomain::Catalog, error);
        }
        if let Err(error) = self.reconcile_discovered(&environment).await {
            tracing::warn!(%error, "discovered-model assessment reconciliation failed");
            self.publish_domain_failure(AssessmentDomain::Discovered, error);
        }
        let mut source_retry = tokio::time::interval(MODEL_ASSESSMENT_SOURCE_RETRY_DELAY);
        source_retry.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        source_retry.tick().await;
        loop {
            tokio::select! {
                event = catalog.next() => match event {
                    Some(_) => if let Err(error) = self.reconcile_catalog(&environment).await {
                        tracing::warn!(%error, "catalog assessment reconciliation failed");
                        self.publish_domain_failure(AssessmentDomain::Catalog, error);
                    },
                    None => return,
                },
                event = discovered.next() => match event {
                    Some(_) => if let Err(error) = self.reconcile_discovered(&environment).await {
                        tracing::warn!(%error, "discovered-model assessment reconciliation failed");
                        self.publish_domain_failure(AssessmentDomain::Discovered, error);
                    },
                    None => return,
                },
                _ = source_retry.tick() => {
                    let (catalog_retry, discovered_retry) = {
                        let current = self.current.read().expect("assessment state lock poisoned");
                        (
                            current.domain_source_failed(AssessmentDomain::Catalog),
                            current.domain_source_failed(AssessmentDomain::Discovered),
                        )
                    };
                    if catalog_retry
                        && let Err(error) = self.reconcile_catalog(&environment).await
                    {
                        tracing::warn!(%error, "catalog assessment reconciliation retry failed");
                        self.publish_domain_failure(AssessmentDomain::Catalog, error);
                    }
                    if discovered_retry
                        && let Err(error) = self.reconcile_discovered(&environment).await
                    {
                        tracing::warn!(%error, "discovered-model assessment reconciliation retry failed");
                        self.publish_domain_failure(AssessmentDomain::Discovered, error);
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

fn assessment_concurrency() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |cores| cores.get().min(MAX_ASSESSMENT_CONCURRENCY))
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_service_contracts::models::{
        AssessmentEnvironmentId, ModelAssessmentProfile, ModelPackageId, ModelServingConfiguration,
        ServableModelBundle,
    };
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

    fn assessment_target(
        subject: magnitude_service_contracts::models::ModelAssessmentSubject,
        key: &str,
    ) -> AssessmentTarget {
        let configuration = test_configuration(subject.model_id().as_str());
        AssessmentTarget {
            entry: ModelAssessmentEntry {
                subject: subject.clone(),
                state: ModelAssessmentEntryState::Assessing,
            },
            key: AssessmentWorkKey(key.to_owned()),
            work: Some(AssessmentWork {
                key: AssessmentWorkKey(key.to_owned()),
                profile: ModelAssessmentProfile {
                    profile: configuration.profile.clone(),
                    performance_context_tokens: vec![configuration.profile.context_length],
                },
                configuration,
            }),
        }
    }

    fn assessment_pool_current() -> AssessmentPoolCurrent {
        AssessmentPoolCurrent {
            snapshot: ModelAssessmentsSnapshot {
                revision: 0,
                state: ModelAssessmentPoolState::Ready {
                    environment_id: AssessmentEnvironmentId("test-environment".to_owned()),
                    catalog: ModelAssessmentDomainSnapshot::Pending { source_revision: 0 },
                    discovered: ModelAssessmentDomainSnapshot::Pending { source_revision: 0 },
                },
            },
            entry_keys: std::collections::BTreeMap::new(),
        }
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
        let discovered = magnitude_service_contracts::models::ModelAssessmentSubject::Discovery {
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
            AssessmentDomain::Discovered,
            3,
            &[assessment_target(discovered, "shared")],
        );
        let discovered = current
            .domain(AssessmentDomain::Discovered)
            .expect("discovered assessment slice");
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
    fn failed_source_slice_recovers_without_disturbing_the_other_slice() {
        let catalog = assessment_subject("pool-source", CatalogModelSelection::Desired);
        let discovered = magnitude_service_contracts::models::ModelAssessmentSubject::Discovery {
            model_id: catalog.model_id().clone(),
        };
        let mut current = assessment_pool_current();
        current.replace_domain(
            AssessmentDomain::Catalog,
            2,
            &[assessment_target(catalog, "catalog")],
        );
        current.replace_domain(
            AssessmentDomain::Discovered,
            3,
            &[assessment_target(discovered.clone(), "discovered")],
        );
        let failure = DomainModelFailure {
            code: "source_unavailable".to_owned(),
            message: "source unavailable".to_owned(),
            retryable: true,
        };
        assert!(current.set_domain_failure(AssessmentDomain::Discovered, failure.clone()));
        assert!(!current.set_domain_failure(AssessmentDomain::Discovered, failure));
        assert!(current.references_assessing(&AssessmentWorkKey("catalog".to_owned())));
        assert!(current.replace_domain(
            AssessmentDomain::Discovered,
            4,
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
        assert!(!current.domain_source_failed(AssessmentDomain::Catalog));
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
