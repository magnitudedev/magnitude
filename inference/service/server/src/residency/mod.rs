//! Model residency: the singleton residency actor that owns instance identity, lifecycle,
//! leases, FIFO replacement, idle release, stop and retirement proof (integration spec §10).
//!
//! The actor is independent of what is resident. A [`ModelResidencyDriver`] loads, supervises
//! and releases workers and names the resident payload a lease hands out.

pub mod controller;
pub mod supervisor;
pub mod worker;

use std::sync::Arc;

use magnitude_service_contracts::InventoryError;
use magnitude_service_contracts::models::{
    ModelId, ModelInstance, ModelInstanceAllocation, ModelInstanceFailure, ModelInstanceId,
    ModelInstanceLifecycle, ModelInstancesInvalidation, ModelInstancesSnapshot, ModelLoadPlan,
    ModelLoadStage, ModelPackageId, ModelReleaseReason, ModelServingConfiguration,
    ModelStoppingAllocation,
};

/// The failure code of a load refused for lack of memory.
pub const LOW_MEMORY_FAILURE_CODE: &str = "low_memory";

/// Observes model loading as a completed fraction in `[0, 1]`.
pub type LoadingObserver = Arc<dyn Fn(f32) + Send + Sync>;

pub enum ResidencyAcquisition {
    Inference { progress: Option<LoadingObserver> },
    Warm,
}

pub enum ResidencyGrant<R> {
    Lease(ResidencyLease<R>),
    Ready(ModelInstance),
}

/// One hold on a ready instance. While any lease is held the instance is neither idle-released
/// nor drained; dropping the lease releases the hold.
pub struct ResidencyLease<R> {
    instance_id: ModelInstanceId,
    resident: R,
    release: LeaseRelease,
}

struct LeaseRelease {
    notifier: ResidencyNotifier,
    instance_id: ModelInstanceId,
    token: u64,
}

impl Drop for LeaseRelease {
    fn drop(&mut self) {
        self.notifier.send(ResidencyNotification::LeaseReleased {
            instance_id: self.instance_id.clone(),
            token: self.token,
        });
    }
}

impl<R> ResidencyLease<R> {
    pub fn instance_id(&self) -> &ModelInstanceId {
        &self.instance_id
    }

    pub fn resident(&self) -> &R {
        &self.resident
    }

    /// The resident payload and the guard whose drop releases the lease.
    pub fn into_parts(self) -> (R, impl Send + Sync + 'static) {
        (self.resident, self.release)
    }
}

struct AcquisitionWaiter<R> {
    target: ResolvedResidencyTarget,
    purpose: ResidencyAcquisition,
    reply: tokio::sync::oneshot::Sender<Result<ResidencyGrant<R>, InventoryError>>,
}

#[derive(Clone, PartialEq, Eq)]
pub struct ResolvedResidencyTarget {
    pub model_id: ModelId,
    pub configuration: ModelServingConfiguration,
    /// The serving configuration fingerprint: instance equality.
    pub configuration_key: String,
}

/// The process that holds a resident model.
pub trait ResidencyWorker: Send + Sync {
    fn pid(&self) -> Option<u32>;
    fn try_wait(&self) -> std::io::Result<Option<std::process::ExitStatus>>;
    /// End the worker now; attached requests fail.
    fn terminate(&self, code: &str, reason: &str);
    /// Ask the worker to unload and exit; attached requests end with the unload.
    fn shutdown(&self);
}

fn verify_worker_retirement(worker: &dyn ResidencyWorker) -> Result<(), ModelOperationFailure> {
    match worker.try_wait() {
        Ok(Some(_)) => Ok(()),
        Ok(None) => Err(ModelOperationFailure::new(
            "worker_retirement_failed",
            "inference worker retirement remains unproven",
            true,
        )),
        Err(error) => Err(ModelOperationFailure::new(
            "worker_retirement_failed",
            format!("cannot verify inference worker retirement: {error}"),
            true,
        )),
    }
}

pub struct PreparedResidency<R> {
    pub package_ids: Vec<ModelPackageId>,
    pub allocation: ModelInstanceAllocation,
    pub worker: Arc<dyn ResidencyWorker>,
    pub resident: R,
}

struct LoadingResidency<R> {
    instance: ModelInstance,
    configuration: ModelServingConfiguration,
    configuration_key: String,
    operation: tokio::task::JoinHandle<()>,
    worker: Option<Arc<dyn ResidencyWorker>>,
    waiters: Vec<AcquisitionWaiter<R>>,
}

struct ReadyResidency<R> {
    instance: ModelInstance,
    configuration_key: String,
    resources: ResidentResources<R>,
    idle_deadline: Option<tokio::time::Instant>,
}

struct ResidentResources<R> {
    package_ids: Vec<ModelPackageId>,
    worker: Arc<dyn ResidencyWorker>,
    resident: R,
    leases: std::collections::BTreeSet<u64>,
}

#[derive(Clone)]
pub struct ReleaseControl {
    interrupt: tokio::sync::watch::Sender<bool>,
    worker: Arc<dyn ResidencyWorker>,
}

impl ReleaseControl {
    pub fn new(
        interrupt: tokio::sync::watch::Sender<bool>,
        worker: Arc<dyn ResidencyWorker>,
    ) -> Self {
        Self { interrupt, worker }
    }

    fn interrupt(&self) {
        self.interrupt.send_replace(true);
    }
}

enum ReleaseOperation<R> {
    StoppingLoad {
        operation: tokio::task::JoinHandle<()>,
        worker: Option<Arc<dyn ResidencyWorker>>,
    },
    Draining {
        resources: Option<ResidentResources<R>>,
    },
    Stopping {
        control: ReleaseControl,
    },
    RetirementFailed {
        worker: Arc<dyn ResidencyWorker>,
        failure: ModelOperationFailure,
    },
}

struct ReleasingResidency<R> {
    instance: ModelInstance,
    reason: ModelReleaseReason,
    terminal_failure: Option<ModelOperationFailure>,
    operation: ReleaseOperation<R>,
    stop_replies: Vec<tokio::sync::oneshot::Sender<Result<(), InventoryError>>>,
}

enum ResidencyState<R> {
    Vacant,
    Loading(LoadingResidency<R>),
    Ready(ReadyResidency<R>),
    Releasing(ReleasingResidency<R>),
}

struct TerminalInstance {
    instance: ModelInstance,
    updated_revision: u64,
}

/// Actor inputs that carry no resident payload: every lease, permit and driver task sends
/// these, whatever is resident. They share the command channel so their order with load
/// completion is preserved.
pub enum ResidencyNotification {
    PackageRemovalFinished,
    LoadWorkerStarted {
        instance_id: ModelInstanceId,
        worker: Arc<dyn ResidencyWorker>,
    },
    LoadProgress {
        instance_id: ModelInstanceId,
        stage: ModelLoadStage,
        progress: Option<f32>,
        planned_allocation: Option<ModelLoadPlan>,
    },
    LoadingStopCompleted {
        instance_id: ModelInstanceId,
    },
    LeaseReleased {
        instance_id: ModelInstanceId,
        token: u64,
    },
    ReleaseFinished {
        instance_id: ModelInstanceId,
        result: Result<(), ModelOperationFailure>,
    },
    WorkerFailed {
        instance_id: ModelInstanceId,
        failure: ModelOperationFailure,
    },
    ReleaseRequested {
        instance_id: ModelInstanceId,
        reason: ModelReleaseReason,
    },
    /// The resident instance's latest memory standing.
    AllocationObserved {
        instance_id: ModelInstanceId,
        allocation: ModelInstanceAllocation,
    },
}

enum ResidencyCommand<R> {
    Acquire(AcquisitionWaiter<R>),
    Lease {
        instance_id: ModelInstanceId,
        reply: tokio::sync::oneshot::Sender<Result<ResidencyLease<R>, InventoryError>>,
    },
    Stop {
        instance_id: ModelInstanceId,
        reply: tokio::sync::oneshot::Sender<Result<(), InventoryError>>,
    },
    Snapshot {
        reply: tokio::sync::oneshot::Sender<ModelInstancesSnapshot>,
    },
    AcquirePackageRemoval {
        package_id: ModelPackageId,
        reply: tokio::sync::oneshot::Sender<Result<PackageRemovalPermit, InventoryError>>,
    },
    LoadFinished {
        instance_id: ModelInstanceId,
        result: Result<PreparedResidency<R>, ModelOperationFailure>,
    },
    Notification(ResidencyNotification),
}

/// Sends payload-free notifications to the actor, whatever its resident type.
#[derive(Clone)]
pub struct ResidencyNotifier {
    send: Arc<dyn Fn(ResidencyNotification) + Send + Sync>,
}

impl ResidencyNotifier {
    pub fn send(&self, notification: ResidencyNotification) {
        (self.send)(notification);
    }
}

/// Sends the result of a load to the actor.
pub struct LoadCompletion<R> {
    commands: tokio::sync::mpsc::UnboundedSender<ResidencyCommand<R>>,
}

impl<R> Clone for LoadCompletion<R> {
    fn clone(&self) -> Self {
        Self {
            commands: self.commands.clone(),
        }
    }
}

impl<R> LoadCompletion<R> {
    pub fn finish(
        &self,
        instance_id: ModelInstanceId,
        result: Result<PreparedResidency<R>, ModelOperationFailure>,
    ) {
        let _ = self.commands.send(ResidencyCommand::LoadFinished {
            instance_id,
            result,
        });
    }
}

/// The actor's inbound channel and change broadcast.
pub struct ResidencyClient<R> {
    commands: tokio::sync::mpsc::UnboundedSender<ResidencyCommand<R>>,
    changes: tokio::sync::broadcast::Sender<ModelInstancesInvalidation>,
}

impl<R> Clone for ResidencyClient<R> {
    fn clone(&self) -> Self {
        Self {
            commands: self.commands.clone(),
            changes: self.changes.clone(),
        }
    }
}

/// The receiving end of a [`ResidencyClient`], consumed by the actor.
pub struct ResidencyInbox<R> {
    commands: tokio::sync::mpsc::UnboundedReceiver<ResidencyCommand<R>>,
}

fn owner_stopped() -> InventoryError {
    InventoryError::Internal("model residency owner stopped".to_owned())
}

impl<R: Send + 'static> ResidencyClient<R> {
    pub fn channel() -> (Self, ResidencyInbox<R>) {
        let (commands, command_receiver) = tokio::sync::mpsc::unbounded_channel();
        let (changes, _) = tokio::sync::broadcast::channel(16);
        (
            Self { commands, changes },
            ResidencyInbox {
                commands: command_receiver,
            },
        )
    }

    pub fn notifier(&self) -> ResidencyNotifier {
        let commands = self.commands.clone();
        ResidencyNotifier {
            send: Arc::new(move |notification| {
                let _ = commands.send(ResidencyCommand::Notification(notification));
            }),
        }
    }

    pub fn load_completion(&self) -> LoadCompletion<R> {
        LoadCompletion {
            commands: self.commands.clone(),
        }
    }

    pub async fn acquire(
        &self,
        target: ResolvedResidencyTarget,
        purpose: ResidencyAcquisition,
    ) -> Result<ResidencyGrant<R>, InventoryError> {
        let (reply, response) = tokio::sync::oneshot::channel();
        self.commands
            .send(ResidencyCommand::Acquire(AcquisitionWaiter {
                target,
                purpose,
                reply,
            }))
            .map_err(|_| owner_stopped())?;
        response.await.map_err(|_| owner_stopped())?
    }

    pub async fn lease(
        &self,
        instance_id: ModelInstanceId,
    ) -> Result<ResidencyLease<R>, InventoryError> {
        let (reply, response) = tokio::sync::oneshot::channel();
        self.commands
            .send(ResidencyCommand::Lease { instance_id, reply })
            .map_err(|_| owner_stopped())?;
        response.await.map_err(|_| owner_stopped())?
    }

    pub async fn stop(&self, instance_id: ModelInstanceId) -> Result<(), InventoryError> {
        let (reply, response) = tokio::sync::oneshot::channel();
        self.commands
            .send(ResidencyCommand::Stop { instance_id, reply })
            .map_err(|_| owner_stopped())?;
        response.await.map_err(|_| owner_stopped())?
    }

    pub async fn snapshot(&self) -> Result<ModelInstancesSnapshot, InventoryError> {
        let (reply, response) = tokio::sync::oneshot::channel();
        self.commands
            .send(ResidencyCommand::Snapshot { reply })
            .map_err(|_| owner_stopped())?;
        response.await.map_err(|_| owner_stopped())
    }

    pub async fn acquire_package_removal(
        &self,
        package_id: ModelPackageId,
    ) -> Result<PackageRemovalPermit, InventoryError> {
        let (reply, response) = tokio::sync::oneshot::channel();
        self.commands
            .send(ResidencyCommand::AcquirePackageRemoval { package_id, reply })
            .map_err(|_| owner_stopped())?;
        response.await.map_err(|_| owner_stopped())?
    }

    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<ModelInstancesInvalidation> {
        self.changes.subscribe()
    }
}

/// Exclusive admission for removing installed packages; dropping it ends the removal.
pub struct PackageRemovalPermit {
    notifier: ResidencyNotifier,
}

impl Drop for PackageRemovalPermit {
    fn drop(&mut self) {
        self.notifier.send(ResidencyNotification::PackageRemovalFinished);
    }
}

#[derive(Clone, Debug)]
pub enum ModelOperationFailure {
    Operation {
        code: String,
        message: String,
        retryable: bool,
    },
    LowMemory(LowMemory),
}

/// A load refused because its limiting memory domain lacks room above its planning reserve.
#[derive(Clone, Debug)]
pub struct LowMemory {
    pub required_memory_bytes: u64,
    pub allocation_headroom_bytes: u64,
    /// The limiting domain's planning reserve.
    pub system_reserve_bytes: u64,
}

impl ModelOperationFailure {
    pub fn new(code: impl Into<String>, message: impl Into<String>, retryable: bool) -> Self {
        Self::Operation {
            code: code.into(),
            message: message.into(),
            retryable,
        }
    }

    pub fn code(&self) -> &str {
        match self {
            Self::Operation { code, .. } => code,
            Self::LowMemory(_) => LOW_MEMORY_FAILURE_CODE,
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::Operation { message, .. } => message.clone(),
            Self::LowMemory(low) => low.message(),
        }
    }

    pub fn retryable(&self) -> bool {
        match self {
            Self::Operation { retryable, .. } => *retryable,
            Self::LowMemory(_) => true,
        }
    }

    fn into_instance_failure(self) -> ModelInstanceFailure {
        match self {
            Self::Operation {
                code,
                message,
                retryable,
            } => ModelInstanceFailure::Operation {
                code,
                message,
                retryable,
            },
            Self::LowMemory(low) => ModelInstanceFailure::LowMemory {
                code: LOW_MEMORY_FAILURE_CODE.to_owned(),
                message: low.message(),
                retryable: true,
                required_memory_bytes: low.required_memory_bytes,
                allocation_headroom_bytes: low.allocation_headroom_bytes,
                system_reserve_bytes: low.system_reserve_bytes,
                load_boundary_bytes: low.load_boundary_bytes(),
                minimum_additional_available_bytes: low.minimum_additional_available_bytes(),
            },
        }
    }
}

impl LowMemory {
    pub fn load_boundary_bytes(&self) -> u64 {
        self.required_memory_bytes
            .saturating_add(self.system_reserve_bytes)
    }

    pub fn minimum_additional_available_bytes(&self) -> u64 {
        self.load_boundary_bytes()
            .saturating_sub(self.allocation_headroom_bytes)
            .saturating_add(1)
    }

    fn message(&self) -> String {
        let short = self.minimum_additional_available_bytes();
        format!(
            "not enough memory available: model requires {} bytes plus {} bytes reserved for the system; {} bytes are available ({short} {} short)",
            self.required_memory_bytes,
            self.system_reserve_bytes,
            self.allocation_headroom_bytes,
            if short == 1 { "byte" } else { "bytes" },
        )
    }
}

fn stopped_instance_preparation_error(reason: &ModelReleaseReason) -> InventoryError {
    if *reason == ModelReleaseReason::UserStop {
        return InventoryError::ModelOperation {
            code: "model_instance_stopped".to_owned(),
            message: "model instance was stopped".to_owned(),
            retryable: false,
        };
    }
    InventoryError::NotReady(format!(
        "model instance stopped during request preparation: {reason:?}"
    ))
}

impl From<InventoryError> for ModelOperationFailure {
    fn from(error: InventoryError) -> Self {
        match error {
            InventoryError::ModelOperation {
                code,
                message,
                retryable,
            } => Self::new(code, message, retryable),
            error => Self::new("model_transition_failed", error.to_string(), true),
        }
    }
}

/// Loads, supervises and releases workers for the actor.
pub trait ModelResidencyDriver: Clone + Send + Sync + 'static {
    /// What a lease on a ready instance hands out.
    type Resident: Clone + Send + Sync + 'static;
    fn client(&self) -> &ResidencyClient<Self::Resident>;
    fn next_instance_id(&self) -> ModelInstanceId;
    fn start_model_load(
        &self,
        instance_id: ModelInstanceId,
        target: ResolvedResidencyTarget,
    ) -> Result<tokio::task::JoinHandle<()>, InventoryError>;
    fn start_worker_release(
        &self,
        instance_id: ModelInstanceId,
        worker: Arc<dyn ResidencyWorker>,
        interrupt_immediately: bool,
    ) -> ReleaseControl;
    fn supervise_worker(
        &self,
        instance_id: ModelInstanceId,
        worker: Arc<dyn ResidencyWorker>,
        resident: Self::Resident,
    );
    fn idle_timeout(&self) -> std::time::Duration;
}

pub struct ModelResidency<D: ModelResidencyDriver> {
    controller: D,
    state: ResidencyState<D::Resident>,
    queue: std::collections::VecDeque<AcquisitionWaiter<D::Resident>>,
    terminals: std::collections::BTreeMap<ModelInstanceId, TerminalInstance>,
    revision: u64,
    next_lease_token: u64,
    package_removal_active: bool,
    package_removals: std::collections::VecDeque<(
        ModelPackageId,
        tokio::sync::oneshot::Sender<Result<PackageRemovalPermit, InventoryError>>,
    )>,
}

impl<D: ModelResidencyDriver> ModelResidency<D> {
    pub fn new(controller: D) -> Self {
        Self {
            controller,
            state: ResidencyState::Vacant,
            queue: std::collections::VecDeque::new(),
            terminals: std::collections::BTreeMap::new(),
            revision: 0,
            next_lease_token: 1,
            package_removal_active: false,
            package_removals: std::collections::VecDeque::new(),
        }
    }

    /// Serve the actor until its client and every load completion are gone.
    pub async fn run(mut self, mut inbox: ResidencyInbox<D::Resident>) {
        loop {
            let idle_deadline = match &self.state {
                ResidencyState::Ready(ready) => ready.idle_deadline,
                _ => None,
            };
            let idle = async move {
                match idle_deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                biased;
                command = inbox.commands.recv() => match command {
                    Some(command) => self.handle(command),
                    None => break,
                },
                () = idle => {}
            }
            self.idle_expired();
        }
        self.shutdown();
    }

    fn handle(&mut self, command: ResidencyCommand<D::Resident>) {
        match command {
            ResidencyCommand::Acquire(waiter) => self.acquire(waiter),
            ResidencyCommand::Lease { instance_id, reply } => {
                let result = self.lease_exact(&instance_id);
                let _ = reply.send(result);
            }
            ResidencyCommand::Stop { instance_id, reply } => self.stop(instance_id, reply),
            ResidencyCommand::Snapshot { reply } => {
                let _ = reply.send(self.snapshot());
            }
            ResidencyCommand::AcquirePackageRemoval { package_id, reply } => {
                self.package_removals.push_back((package_id, reply));
                self.admit_package_removal();
            }
            ResidencyCommand::LoadFinished {
                instance_id,
                result,
            } => self.load_finished(instance_id, result),
            ResidencyCommand::Notification(notification) => self.notify(notification),
        }
    }

    fn notify(&mut self, notification: ResidencyNotification) {
        match notification {
            ResidencyNotification::PackageRemovalFinished => {
                self.package_removal_active = false;
                self.resume_pending();
            }
            ResidencyNotification::LoadWorkerStarted {
                instance_id,
                worker,
            } => {
                if let ResidencyState::Loading(loading) = &mut self.state
                    && loading.instance.id == instance_id
                {
                    loading.worker = Some(worker);
                } else if let ResidencyState::Releasing(releasing) = &mut self.state
                    && releasing.instance.id == instance_id
                    && let ReleaseOperation::StoppingLoad {
                        worker: retained, ..
                    } = &mut releasing.operation
                {
                    // The start notification may already be queued when cancellation is admitted.
                    worker.terminate("model_instance_stopped", "model instance was stopped");
                    *retained = Some(worker);
                }
            }
            ResidencyNotification::LoadProgress {
                instance_id,
                stage,
                progress,
                planned_allocation,
            } => self.load_progress(instance_id, stage, progress, planned_allocation),
            ResidencyNotification::AllocationObserved {
                instance_id,
                allocation,
            } => self.allocation_observed(instance_id, allocation),
            ResidencyNotification::LoadingStopCompleted { instance_id } => {
                if let ResidencyState::Releasing(releasing) = &mut self.state
                    && releasing.instance.id == instance_id
                    && let ReleaseOperation::StoppingLoad { worker, .. } = &mut releasing.operation
                {
                    if let Some(worker) = worker.take() {
                        let control =
                            self.controller
                                .start_worker_release(instance_id, worker, true);
                        releasing.operation = ReleaseOperation::Stopping { control };
                    } else {
                        self.finish_stopped(instance_id);
                    }
                }
            }
            ResidencyNotification::LeaseReleased { instance_id, token } => {
                self.lease_released(instance_id, token);
            }
            ResidencyNotification::ReleaseFinished {
                instance_id,
                result,
            } => self.release_finished(instance_id, result),
            ResidencyNotification::WorkerFailed {
                instance_id,
                failure,
            } => self.worker_failed(instance_id, failure),
            ResidencyNotification::ReleaseRequested {
                instance_id,
                reason,
            } => self.release_requested(instance_id, reason),
        }
    }

    /// A newer standing replaces the published allocation of the exact ready instance.
    fn allocation_observed(
        &mut self,
        instance_id: ModelInstanceId,
        allocation: ModelInstanceAllocation,
    ) {
        let ResidencyState::Ready(ready) = &mut self.state else {
            return;
        };
        if ready.instance.id != instance_id {
            return;
        }
        let lifecycle = ModelInstanceLifecycle::Ready { allocation };
        if ready.instance.lifecycle == lifecycle {
            return;
        }
        ready.instance.lifecycle = lifecycle;
        self.publish_current();
    }

    fn acquire(&mut self, waiter: AcquisitionWaiter<D::Resident>) {
        if let Some(failure) = self.retirement_failure() {
            let _ = waiter.reply.send(Err(Self::inventory_failure(failure)));
            return;
        }
        if self.package_removal_active || !self.package_removals.is_empty() {
            self.queue.push_back(waiter);
            return;
        }
        match &mut self.state {
            ResidencyState::Vacant => self.start_loading(waiter),
            ResidencyState::Loading(loading)
                if loading.instance.model_id == waiter.target.model_id
                    && loading.configuration_key == waiter.target.configuration_key =>
            {
                Self::notify_progress(&waiter, &loading.instance.lifecycle);
                loading.waiters.push(waiter);
            }
            ResidencyState::Ready(ready)
                if ready.instance.model_id == waiter.target.model_id
                    && ready.configuration_key == waiter.target.configuration_key =>
            {
                self.grant_ready(waiter);
            }
            ResidencyState::Loading(_) | ResidencyState::Releasing(_) => {
                self.queue.push_back(waiter);
            }
            ResidencyState::Ready(_) => {
                self.queue.push_back(waiter);
                self.begin_ready_release(ModelReleaseReason::Replacement, Vec::new());
            }
        }
    }

    fn admit_package_removal(&mut self) {
        if let Some(failure) = self.retirement_failure() {
            let errors = self
                .package_removals
                .iter()
                .map(|_| Self::inventory_failure(failure))
                .collect::<Vec<_>>();
            for ((_, reply), error) in self.package_removals.drain(..).zip(errors) {
                let _ = reply.send(Err(error));
            }
            return;
        }
        if self.package_removal_active
            || matches!(
                self.state,
                ResidencyState::Loading(_) | ResidencyState::Releasing(_)
            )
        {
            return;
        }
        while let Some((package_id, reply)) = self.package_removals.pop_front() {
            if reply.is_closed() {
                continue;
            }
            if matches!(
                &self.state,
                ResidencyState::Ready(ready)
                    if ready.resources.package_ids.contains(&package_id)
            ) {
                let _ = reply.send(Err(InventoryError::Loaded(package_id.0)));
                continue;
            }
            self.package_removal_active = true;
            let _ = reply.send(Ok(PackageRemovalPermit {
                notifier: self.controller.client().notifier(),
            }));
            break;
        }
    }

    fn start_loading(&mut self, first: AcquisitionWaiter<D::Resident>) {
        let instance_id = self.controller.next_instance_id();
        let operation = match self
            .controller
            .start_model_load(instance_id.clone(), first.target.clone())
        {
            Ok(operation) => operation,
            Err(error) => {
                let _ = first.reply.send(Err(error));
                self.start_next_queued();
                return;
            }
        };
        let instance = ModelInstance {
            id: instance_id.clone(),
            model_id: first.target.model_id.clone(),
            lifecycle: ModelInstanceLifecycle::Loading {
                stage: ModelLoadStage::Queued,
                progress: None,
                planned_allocation: None,
            },
        };
        Self::notify_progress(&first, &instance.lifecycle);
        self.state = ResidencyState::Loading(LoadingResidency {
            instance,
            configuration: first.target.configuration.clone(),
            configuration_key: first.target.configuration_key.clone(),
            operation,
            worker: None,
            waiters: vec![first],
        });
        self.publish_current();

        let target = match &self.state {
            ResidencyState::Loading(loading) => ResolvedResidencyTarget {
                model_id: loading.instance.model_id.clone(),
                configuration: loading.configuration.clone(),
                configuration_key: loading.configuration_key.clone(),
            },
            _ => unreachable!(),
        };
        let mut retained = std::collections::VecDeque::new();
        while let Some(waiter) = self.queue.pop_front() {
            if waiter.target == target {
                if let ResidencyState::Loading(loading) = &mut self.state {
                    Self::notify_progress(&waiter, &loading.instance.lifecycle);
                    loading.waiters.push(waiter);
                }
            } else {
                retained.push_back(waiter);
            }
        }
        self.queue = retained;
    }

    fn grant_ready(&mut self, waiter: AcquisitionWaiter<D::Resident>) {
        let ResidencyState::Ready(mut ready) =
            std::mem::replace(&mut self.state, ResidencyState::Vacant)
        else {
            unreachable!("ready grant requires Ready residency")
        };
        match waiter.purpose {
            ResidencyAcquisition::Inference { .. } => {
                let lease = self.make_lease(&mut ready);
                let _ = waiter.reply.send(Ok(ResidencyGrant::Lease(lease)));
            }
            ResidencyAcquisition::Warm => {
                let _ = waiter
                    .reply
                    .send(Ok(ResidencyGrant::Ready(ready.instance.clone())));
            }
        }
        self.state = ResidencyState::Ready(ready);
        self.refresh_idle_deadline();
    }

    fn make_lease(
        &mut self,
        ready: &mut ReadyResidency<D::Resident>,
    ) -> ResidencyLease<D::Resident> {
        let token = self.next_lease_token;
        self.next_lease_token = self.next_lease_token.saturating_add(1);
        ready.resources.leases.insert(token);
        ready.idle_deadline = None;
        ResidencyLease {
            instance_id: ready.instance.id.clone(),
            resident: ready.resources.resident.clone(),
            release: LeaseRelease {
                notifier: self.controller.client().notifier(),
                instance_id: ready.instance.id.clone(),
                token,
            },
        }
    }

    fn lease_exact(
        &mut self,
        instance_id: &ModelInstanceId,
    ) -> Result<ResidencyLease<D::Resident>, InventoryError> {
        let state = std::mem::replace(&mut self.state, ResidencyState::Vacant);
        let ResidencyState::Ready(mut ready) = state else {
            self.state = state;
            return Err(InventoryError::NotReady(format!(
                "model instance {} is not loaded",
                instance_id.0
            )));
        };
        if ready.instance.id != *instance_id {
            self.state = ResidencyState::Ready(ready);
            return Err(InventoryError::NotReady(format!(
                "model instance {} is not loaded",
                instance_id.0
            )));
        }
        let lease = self.make_lease(&mut ready);
        self.state = ResidencyState::Ready(ready);
        Ok(lease)
    }

    fn stop(
        &mut self,
        instance_id: ModelInstanceId,
        reply: tokio::sync::oneshot::Sender<Result<(), InventoryError>>,
    ) {
        if self.terminals.contains_key(&instance_id) {
            let _ = reply.send(Ok(()));
            return;
        }
        match &self.state {
            ResidencyState::Loading(loading) if loading.instance.id == instance_id => {
                self.stop_loading(reply);
            }
            ResidencyState::Ready(ready) if ready.instance.id == instance_id => {
                self.begin_ready_release(ModelReleaseReason::UserStop, vec![reply]);
            }
            ResidencyState::Releasing(releasing) if releasing.instance.id == instance_id => {
                self.escalate_release(reply);
            }
            _ => {
                let _ = reply.send(Ok(()));
            }
        }
    }

    fn stop_loading(&mut self, reply: tokio::sync::oneshot::Sender<Result<(), InventoryError>>) {
        let ResidencyState::Loading(mut loading) =
            std::mem::replace(&mut self.state, ResidencyState::Vacant)
        else {
            unreachable!()
        };
        for waiter in loading.waiters.drain(..) {
            let _ = waiter.reply.send(Err(stopped_instance_preparation_error(
                &ModelReleaseReason::UserStop,
            )));
        }
        let worker = loading.worker.take();
        if let Some(worker) = &worker {
            worker.terminate("model_instance_stopped", "model instance was stopped");
        }
        loading.operation.abort();
        let notifier = self.controller.client().notifier();
        let instance_id = loading.instance.id.clone();
        let operation = loading.operation;
        let completion = tokio::spawn(async move {
            let _ = operation.await;
            notifier.send(ResidencyNotification::LoadingStopCompleted { instance_id });
        });
        let stopping = ModelInstance {
            id: loading.instance.id,
            model_id: loading.instance.model_id,
            lifecycle: ModelInstanceLifecycle::Stopping {
                reason: ModelReleaseReason::UserStop,
                allocation: ModelStoppingAllocation::Planned {
                    allocation: match loading.instance.lifecycle {
                        ModelInstanceLifecycle::Loading {
                            planned_allocation, ..
                        } => planned_allocation,
                        _ => None,
                    },
                },
            },
        };
        self.state = ResidencyState::Releasing(ReleasingResidency {
            instance: stopping,
            reason: ModelReleaseReason::UserStop,
            terminal_failure: None,
            operation: ReleaseOperation::StoppingLoad {
                operation: completion,
                worker,
            },
            stop_replies: vec![reply],
        });
        self.publish_current();
    }

    fn escalate_release(
        &mut self,
        reply: tokio::sync::oneshot::Sender<Result<(), InventoryError>>,
    ) {
        let ResidencyState::Releasing(releasing) = &mut self.state else {
            unreachable!()
        };
        releasing.reason = ModelReleaseReason::UserStop;
        releasing.stop_replies.push(reply);
        match &mut releasing.operation {
            ReleaseOperation::StoppingLoad { .. } => {}
            ReleaseOperation::Draining { resources } => {
                let Some(resources) = resources.take() else {
                    return;
                };
                let instance = releasing.instance.clone();
                let control = self.controller.start_worker_release(
                    instance.id.clone(),
                    resources.worker,
                    true,
                );
                releasing.operation = ReleaseOperation::Stopping { control };
            }
            ReleaseOperation::Stopping { control } => control.interrupt(),
            ReleaseOperation::RetirementFailed { worker, .. } => {
                let control = self.controller.start_worker_release(
                    releasing.instance.id.clone(),
                    Arc::clone(worker),
                    true,
                );
                releasing.operation = ReleaseOperation::Stopping { control };
            }
        }
        releasing.instance.lifecycle = ModelInstanceLifecycle::Stopping {
            reason: ModelReleaseReason::UserStop,
            allocation: match &releasing.instance.lifecycle {
                ModelInstanceLifecycle::Stopping { allocation, .. } => allocation.clone(),
                _ => ModelStoppingAllocation::Planned { allocation: None },
            },
        };
        self.publish_current();
    }

    fn load_progress(
        &mut self,
        instance_id: ModelInstanceId,
        stage: ModelLoadStage,
        progress: Option<f32>,
        planned_allocation: Option<ModelLoadPlan>,
    ) {
        let ResidencyState::Loading(loading) = &mut self.state else {
            return;
        };
        if loading.instance.id != instance_id {
            return;
        }
        let lifecycle = ModelInstanceLifecycle::Loading {
            stage,
            progress,
            planned_allocation,
        };
        if loading.instance.lifecycle == lifecycle {
            return;
        }
        loading.instance.lifecycle = lifecycle;
        for waiter in &loading.waiters {
            Self::notify_progress(waiter, &loading.instance.lifecycle);
        }
        self.publish_current();
    }

    fn load_finished(
        &mut self,
        instance_id: ModelInstanceId,
        result: Result<PreparedResidency<D::Resident>, ModelOperationFailure>,
    ) {
        let state = std::mem::replace(&mut self.state, ResidencyState::Vacant);
        let ResidencyState::Loading(mut loading) = state else {
            self.state = state;
            return;
        };
        if loading.instance.id != instance_id {
            self.state = ResidencyState::Loading(loading);
            return;
        }
        match result {
            Ok(prepared) => {
                let instance = ModelInstance {
                    id: instance_id.clone(),
                    model_id: loading.instance.model_id,
                    lifecycle: ModelInstanceLifecycle::Ready {
                        allocation: prepared.allocation.clone(),
                    },
                };
                let mut ready = ReadyResidency {
                    instance,
                    configuration_key: loading.configuration_key,
                    resources: ResidentResources {
                        package_ids: prepared.package_ids,
                        worker: prepared.worker,
                        resident: prepared.resident,
                        leases: std::collections::BTreeSet::new(),
                    },
                    idle_deadline: None,
                };
                for waiter in loading.waiters.drain(..) {
                    match waiter.purpose {
                        ResidencyAcquisition::Inference { .. } => {
                            let lease = self.make_lease(&mut ready);
                            let _ = waiter.reply.send(Ok(ResidencyGrant::Lease(lease)));
                        }
                        ResidencyAcquisition::Warm => {
                            let _ = waiter
                                .reply
                                .send(Ok(ResidencyGrant::Ready(ready.instance.clone())));
                        }
                    }
                }
                self.controller.supervise_worker(
                    instance_id,
                    ready.resources.worker.clone(),
                    ready.resources.resident.clone(),
                );
                self.state = ResidencyState::Ready(ready);
                self.publish_current();
                self.admit_package_removal();
                if !self.package_removal_active {
                    if self.queue.is_empty() {
                        self.refresh_idle_deadline();
                    } else {
                        self.begin_ready_release(ModelReleaseReason::Replacement, Vec::new());
                    }
                }
            }
            Err(failure) => {
                for waiter in loading.waiters.drain(..) {
                    let _ = waiter.reply.send(Err(Self::inventory_failure(&failure)));
                }
                if let Some(worker) = loading.worker {
                    self.begin_failed_release(loading.instance, worker, failure);
                } else {
                    self.finish_failed(instance_id, loading.instance.model_id, failure);
                }
            }
        }
    }

    fn begin_ready_release(
        &mut self,
        reason: ModelReleaseReason,
        stop_replies: Vec<tokio::sync::oneshot::Sender<Result<(), InventoryError>>>,
    ) {
        let state = std::mem::replace(&mut self.state, ResidencyState::Vacant);
        let ResidencyState::Ready(ready) = state else {
            self.state = state;
            return;
        };
        let stopping = ModelInstance {
            id: ready.instance.id.clone(),
            model_id: ready.instance.model_id.clone(),
            lifecycle: ModelInstanceLifecycle::Stopping {
                reason,
                allocation: ModelStoppingAllocation::Resident {
                    allocation: match &ready.instance.lifecycle {
                        ModelInstanceLifecycle::Ready { allocation } => allocation.clone(),
                        _ => unreachable!("Ready residency must project Ready lifecycle"),
                    },
                },
            },
        };
        if reason == ModelReleaseReason::UserStop || ready.resources.leases.is_empty() {
            let control = self.controller.start_worker_release(
                ready.instance.id.clone(),
                ready.resources.worker,
                reason == ModelReleaseReason::UserStop,
            );
            self.state = ResidencyState::Releasing(ReleasingResidency {
                instance: stopping,
                reason,
                terminal_failure: None,
                operation: ReleaseOperation::Stopping { control },
                stop_replies,
            });
        } else {
            self.state = ResidencyState::Releasing(ReleasingResidency {
                instance: stopping,
                reason,
                terminal_failure: None,
                operation: ReleaseOperation::Draining {
                    resources: Some(ready.resources),
                },
                stop_replies,
            });
        }
        self.publish_current();
    }

    fn lease_released(&mut self, instance_id: ModelInstanceId, token: u64) {
        let mut refresh_idle_deadline = false;
        match &mut self.state {
            ResidencyState::Ready(ready) if ready.instance.id == instance_id => {
                if ready.resources.leases.remove(&token) {
                    refresh_idle_deadline = ready.resources.leases.is_empty();
                }
            }
            ResidencyState::Releasing(ReleasingResidency {
                instance,
                operation:
                    ReleaseOperation::Draining {
                        resources: Some(resources),
                    },
                ..
            }) if instance.id == instance_id => {
                resources.leases.remove(&token);
                if resources.leases.is_empty() {
                    self.start_drained_release();
                }
            }
            _ => {}
        }
        if refresh_idle_deadline {
            self.refresh_idle_deadline();
        }
    }

    fn start_drained_release(&mut self) {
        let ResidencyState::Releasing(releasing) = &mut self.state else {
            return;
        };
        let ReleaseOperation::Draining { resources } = &mut releasing.operation else {
            return;
        };
        let Some(current) = resources.as_ref() else {
            return;
        };
        if !current.leases.is_empty() {
            return;
        }
        let resources = resources.take().expect("drained residency remains owned");
        let control = self.controller.start_worker_release(
            releasing.instance.id.clone(),
            resources.worker,
            false,
        );
        releasing.operation = ReleaseOperation::Stopping { control };
    }

    fn release_finished(
        &mut self,
        instance_id: ModelInstanceId,
        result: Result<(), ModelOperationFailure>,
    ) {
        let ResidencyState::Releasing(releasing) = &self.state else {
            return;
        };
        if releasing.instance.id != instance_id {
            return;
        }
        let ReleaseOperation::Stopping { control } = &releasing.operation else {
            return;
        };
        let worker = Arc::clone(&control.worker);
        // Release completion is a notification, not evidence that native resources are gone.
        let result = result.and_then(|()| verify_worker_retirement(worker.as_ref()));
        match result {
            Ok(()) => self.finish_stopped(instance_id),
            Err(failure) => {
                let ResidencyState::Releasing(releasing) = &mut self.state else {
                    unreachable!();
                };
                for reply in releasing.stop_replies.drain(..) {
                    let _ = reply.send(Err(Self::inventory_failure(&failure)));
                }
                for waiter in self.queue.drain(..) {
                    let _ = waiter.reply.send(Err(Self::inventory_failure(&failure)));
                }
                releasing.operation = ReleaseOperation::RetirementFailed { worker, failure };
                self.admit_package_removal();
                self.publish_current();
            }
        }
    }

    fn retirement_failure(&self) -> Option<&ModelOperationFailure> {
        match &self.state {
            ResidencyState::Releasing(ReleasingResidency {
                operation: ReleaseOperation::RetirementFailed { failure, .. },
                ..
            }) => Some(failure),
            _ => None,
        }
    }

    fn worker_failed(&mut self, instance_id: ModelInstanceId, failure: ModelOperationFailure) {
        let ResidencyState::Ready(ready) = &self.state else {
            return;
        };
        if ready.instance.id != instance_id {
            return;
        }
        let ResidencyState::Ready(ready) =
            std::mem::replace(&mut self.state, ResidencyState::Vacant)
        else {
            unreachable!()
        };
        self.begin_failed_release(ready.instance, ready.resources.worker, failure);
    }

    fn begin_failed_release(
        &mut self,
        mut instance: ModelInstance,
        worker: Arc<dyn ResidencyWorker>,
        failure: ModelOperationFailure,
    ) {
        let allocation = match instance.lifecycle {
            ModelInstanceLifecycle::Loading {
                planned_allocation, ..
            } => ModelStoppingAllocation::Planned {
                allocation: planned_allocation,
            },
            ModelInstanceLifecycle::Ready { allocation } => {
                ModelStoppingAllocation::Resident { allocation }
            }
            _ => unreachable!("only a loading or ready worker can fail"),
        };
        instance.lifecycle = ModelInstanceLifecycle::Stopping {
            reason: ModelReleaseReason::Failure,
            allocation,
        };
        let control = self
            .controller
            .start_worker_release(instance.id.clone(), worker, true);
        self.state = ResidencyState::Releasing(ReleasingResidency {
            instance,
            reason: ModelReleaseReason::Failure,
            terminal_failure: Some(failure),
            operation: ReleaseOperation::Stopping { control },
            stop_replies: Vec::new(),
        });
        self.publish_current();
    }

    fn release_requested(&mut self, instance_id: ModelInstanceId, reason: ModelReleaseReason) {
        if matches!(&self.state, ResidencyState::Ready(ready) if ready.instance.id == instance_id) {
            self.begin_ready_release(reason, Vec::new());
        }
    }

    fn idle_expired(&mut self) {
        let now = tokio::time::Instant::now();
        if matches!(
            &self.state,
            ResidencyState::Ready(ready)
                if ready.resources.leases.is_empty()
                    && ready.idle_deadline.is_some_and(|deadline| deadline <= now)
        ) {
            self.begin_ready_release(ModelReleaseReason::IdleTimeout, Vec::new());
        }
    }

    fn refresh_idle_deadline(&mut self) {
        let timeout = self.controller.idle_timeout();
        let ResidencyState::Ready(ready) = &mut self.state else {
            return;
        };
        ready.idle_deadline = ready
            .resources
            .leases
            .is_empty()
            .then(|| tokio::time::Instant::now() + timeout);
    }

    fn finish_stopped(&mut self, instance_id: ModelInstanceId) {
        let state = std::mem::replace(&mut self.state, ResidencyState::Vacant);
        let ResidencyState::Releasing(mut releasing) = state else {
            self.state = state;
            return;
        };
        if releasing.instance.id != instance_id {
            self.state = ResidencyState::Releasing(releasing);
            return;
        }
        let terminal = ModelInstance {
            id: releasing.instance.id,
            model_id: releasing.instance.model_id,
            lifecycle: match releasing.terminal_failure {
                Some(failure) => ModelInstanceLifecycle::Failed {
                    failure: failure.into_instance_failure(),
                },
                None => ModelInstanceLifecycle::Stopped {
                    reason: releasing.reason,
                },
            },
        };
        self.publish_terminal(terminal);
        for reply in releasing.stop_replies.drain(..) {
            let _ = reply.send(Ok(()));
        }
        self.start_next_queued();
    }

    fn finish_failed(
        &mut self,
        instance_id: ModelInstanceId,
        model_id: ModelId,
        failure: ModelOperationFailure,
    ) {
        let state = std::mem::replace(&mut self.state, ResidencyState::Vacant);
        let stop_replies = match state {
            ResidencyState::Releasing(releasing) if releasing.instance.id == instance_id => {
                releasing.stop_replies
            }
            _ => Vec::new(),
        };
        let error = Self::inventory_failure(&failure);
        self.publish_terminal(ModelInstance {
            id: instance_id,
            model_id,
            lifecycle: ModelInstanceLifecycle::Failed {
                failure: failure.into_instance_failure(),
            },
        });
        for reply in stop_replies {
            let _ = reply.send(Err(match &error {
                InventoryError::ModelOperation {
                    code,
                    message,
                    retryable,
                } => InventoryError::ModelOperation {
                    code: code.clone(),
                    message: message.clone(),
                    retryable: *retryable,
                },
                _ => InventoryError::Internal(error.to_string()),
            }));
        }
        self.start_next_queued();
    }

    fn start_next_queued(&mut self) {
        self.admit_package_removal();
        if self.package_removal_active
            || !self.package_removals.is_empty()
            || !matches!(self.state, ResidencyState::Vacant)
        {
            return;
        }
        while let Some(waiter) = self.queue.pop_front() {
            if waiter.reply.is_closed() {
                continue;
            }
            self.start_loading(waiter);
            break;
        }
    }

    fn resume_pending(&mut self) {
        self.admit_package_removal();
        if self.package_removal_active || !self.package_removals.is_empty() {
            return;
        }
        match &self.state {
            ResidencyState::Vacant => self.start_next_queued(),
            ResidencyState::Ready(_) if !self.queue.is_empty() => {
                self.begin_ready_release(ModelReleaseReason::Replacement, Vec::new());
            }
            ResidencyState::Ready(_) => {}
            ResidencyState::Loading(_) | ResidencyState::Releasing(_) => {}
        }
    }

    fn publish_current(&mut self) {
        self.revision = self.revision.saturating_add(1);
        match &self.state {
            ResidencyState::Loading(loading) => tracing::info!(
                model.id = %loading.instance.model_id,
                model.instance.id = %loading.instance.id.0,
                residency.state = "loading",
                residency.revision = self.revision,
                "model residency transitioned"
            ),
            ResidencyState::Ready(ready) => tracing::info!(
                model.id = %ready.instance.model_id,
                model.instance.id = %ready.instance.id.0,
                residency.state = "ready",
                residency.revision = self.revision,
                "model residency transitioned"
            ),
            ResidencyState::Releasing(releasing) => tracing::info!(
                model.id = %releasing.instance.model_id,
                model.instance.id = %releasing.instance.id.0,
                residency.state = "releasing",
                residency.release.reason = ?releasing.reason,
                residency.revision = self.revision,
                "model residency transitioned"
            ),
            ResidencyState::Vacant => tracing::info!(
                residency.state = "vacant",
                residency.revision = self.revision,
                "model residency transitioned"
            ),
        }
        let _ = self
            .controller
            .client()
            .changes
            .send(ModelInstancesInvalidation {
                revision: self.revision,
            });
    }

    fn publish_terminal(&mut self, instance: ModelInstance) {
        self.revision = self.revision.saturating_add(1);
        tracing::info!(
            model.id = %instance.model_id,
            model.instance.id = %instance.id.0,
            residency.state = "vacant",
            model.instance.lifecycle = ?instance.lifecycle,
            residency.revision = self.revision,
            "model residency terminalized instance"
        );
        self.terminals.insert(
            instance.id.clone(),
            TerminalInstance {
                instance,
                updated_revision: self.revision,
            },
        );
        let _ = self
            .controller
            .client()
            .changes
            .send(ModelInstancesInvalidation {
                revision: self.revision,
            });
    }

    fn snapshot(&self) -> ModelInstancesSnapshot {
        let mut instances = self
            .terminals
            .values()
            .map(|terminal| (terminal.updated_revision, terminal.instance.clone()))
            .collect::<Vec<_>>();
        let current = match &self.state {
            ResidencyState::Vacant => None,
            ResidencyState::Loading(loading) => Some(loading.instance.clone()),
            ResidencyState::Ready(ready) => Some(ready.instance.clone()),
            ResidencyState::Releasing(releasing) => Some(releasing.instance.clone()),
        };
        if let Some(current) = current {
            instances.push((self.revision, current));
        }
        instances.sort_by_key(|(revision, _)| *revision);
        ModelInstancesSnapshot {
            revision: self.revision,
            instances: instances
                .into_iter()
                .map(|(_, instance)| instance)
                .collect(),
        }
    }

    fn notify_progress(waiter: &AcquisitionWaiter<D::Resident>, lifecycle: &ModelInstanceLifecycle) {
        let ResidencyAcquisition::Inference {
            progress: Some(observer),
        } = &waiter.purpose
        else {
            return;
        };
        let fraction = match lifecycle {
            ModelInstanceLifecycle::Loading {
                stage: ModelLoadStage::Loading | ModelLoadStage::Verifying,
                progress,
                ..
            } => progress.unwrap_or(0.0).clamp(0.0, 1.0),
            ModelInstanceLifecycle::Loading { .. } => 0.0,
            ModelInstanceLifecycle::Ready { .. } => 1.0,
            _ => return,
        };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| observer(fraction)));
    }

    fn inventory_failure(failure: &ModelOperationFailure) -> InventoryError {
        InventoryError::ModelOperation {
            code: failure.code().to_owned(),
            message: failure.message(),
            retryable: failure.retryable(),
        }
    }

    fn shutdown(&mut self) {
        match std::mem::replace(&mut self.state, ResidencyState::Vacant) {
            ResidencyState::Loading(loading) => {
                loading.operation.abort();
                if let Some(worker) = loading.worker {
                    worker.terminate("service_shutdown", "the inference service stopped");
                }
            }
            ResidencyState::Ready(ready) => ready
                .resources
                .worker
                .terminate("service_shutdown", "the inference service stopped"),
            ResidencyState::Releasing(ReleasingResidency {
                operation:
                    ReleaseOperation::Draining {
                        resources: Some(resources),
                    },
                ..
            }) => resources
                .worker
                .terminate("service_shutdown", "the inference service stopped"),
            ResidencyState::Releasing(ReleasingResidency {
                operation: ReleaseOperation::Draining { resources: None },
                ..
            }) => {}
            ResidencyState::Releasing(ReleasingResidency {
                operation: ReleaseOperation::Stopping { control },
                ..
            }) => control.interrupt(),
            ResidencyState::Releasing(ReleasingResidency {
                operation: ReleaseOperation::StoppingLoad { operation, worker },
                ..
            }) => {
                operation.abort();
                if let Some(worker) = worker {
                    worker.terminate("service_shutdown", "the inference service stopped");
                }
            }
            ResidencyState::Releasing(ReleasingResidency {
                operation: ReleaseOperation::RetirementFailed { worker, .. },
                ..
            }) => worker.terminate("service_shutdown", "the inference service stopped"),
            ResidencyState::Vacant => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_service_contracts::models::ServableModelBundle;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

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

    #[test]
    fn model_transition_preserves_typed_model_operation_failure() {
        let failure = ModelOperationFailure::from(InventoryError::ModelOperation {
            code: "invalid_split_layout".to_owned(),
            message: "the shard layout is invalid".to_owned(),
            retryable: false,
        });

        assert_eq!(failure.code(), "invalid_split_layout");
        assert_eq!(failure.message(), "the shard layout is invalid");
        assert!(!failure.retryable());
    }

    #[test]
    fn low_memory_failure_preserves_boundary_arithmetic() {
        let failure = ModelOperationFailure::LowMemory(LowMemory {
            required_memory_bytes: 10,
            allocation_headroom_bytes: 11,
            system_reserve_bytes: 3,
        })
        .into_instance_failure();
        let ModelInstanceFailure::LowMemory {
            required_memory_bytes,
            load_boundary_bytes,
            minimum_additional_available_bytes,
            retryable,
            ..
        } = failure
        else {
            panic!("low memory failure changed kind");
        };
        assert_eq!(required_memory_bytes, 10);
        assert_eq!(load_boundary_bytes, 13);
        assert_eq!(minimum_additional_available_bytes, 3);
        assert!(retryable);
    }

    /// Test residents name the model their load was for.
    type TestResident = String;

    #[derive(Clone)]
    struct TestResidencyDriver {
        client: ResidencyClient<TestResident>,
        next_instance_id: Arc<AtomicU64>,
        loads: tokio::sync::mpsc::UnboundedSender<(ModelInstanceId, String)>,
        auto_ready: bool,
        workers: Arc<Mutex<Vec<Arc<TestResidencyWorker>>>>,
        idle_timeout: std::time::Duration,
    }

    #[derive(Default)]
    struct TestResidencyWorker {
        interrupted: AtomicU64,
        shutdown: AtomicU64,
        retired: std::sync::atomic::AtomicBool,
        fail_retirement: std::sync::atomic::AtomicBool,
    }

    impl ResidencyWorker for TestResidencyWorker {
        fn pid(&self) -> Option<u32> {
            None
        }

        fn try_wait(&self) -> std::io::Result<Option<std::process::ExitStatus>> {
            #[cfg(unix)]
            use std::os::unix::process::ExitStatusExt;
            #[cfg(windows)]
            use std::os::windows::process::ExitStatusExt;
            Ok(self
                .retired
                .load(Ordering::Acquire)
                .then(|| std::process::ExitStatus::from_raw(0)))
        }

        fn terminate(&self, _code: &str, _reason: &str) {
            self.interrupted.fetch_add(1, Ordering::AcqRel);
            self.retired.store(
                !self.fail_retirement.load(Ordering::Acquire),
                Ordering::Release,
            );
        }

        fn shutdown(&self) {
            self.shutdown.fetch_add(1, Ordering::AcqRel);
            self.retired.store(
                !self.fail_retirement.load(Ordering::Acquire),
                Ordering::Release,
            );
        }
    }

    impl ModelResidencyDriver for TestResidencyDriver {
        type Resident = TestResident;

        fn client(&self) -> &ResidencyClient<TestResident> {
            &self.client
        }

        fn next_instance_id(&self) -> ModelInstanceId {
            ModelInstanceId(format!(
                "test-instance-{}",
                self.next_instance_id.fetch_add(1, Ordering::Relaxed)
            ))
        }

        fn start_model_load(
            &self,
            instance_id: ModelInstanceId,
            target: ResolvedResidencyTarget,
        ) -> Result<tokio::task::JoinHandle<()>, InventoryError> {
            let model_id = target.model_id;
            self.loads
                .send((instance_id.clone(), model_id.to_string()))
                .expect("test load observer remains alive");
            if self.auto_ready {
                let worker = Arc::new(TestResidencyWorker::default());
                self.workers
                    .lock()
                    .expect("test worker lock")
                    .push(Arc::clone(&worker));
                let notifier = self.client.notifier();
                let completion = self.client.load_completion();
                return Ok(tokio::spawn(async move {
                    tokio::task::yield_now().await;
                    let worker: Arc<dyn ResidencyWorker> = worker;
                    notifier.send(ResidencyNotification::LoadWorkerStarted {
                        instance_id: instance_id.clone(),
                        worker: Arc::clone(&worker),
                    });
                    completion.finish(
                        instance_id,
                        Ok(PreparedResidency {
                            package_ids: Vec::new(),
                            allocation: ModelInstanceAllocation {
                                context_window_tokens: 1,
                                memory_domains: Vec::new(),
                            },
                            worker,
                            resident: model_id.to_string(),
                        }),
                    );
                }));
            }
            Ok(tokio::spawn(std::future::pending()))
        }

        fn start_worker_release(
            &self,
            instance_id: ModelInstanceId,
            worker: Arc<dyn ResidencyWorker>,
            interrupt_immediately: bool,
        ) -> ReleaseControl {
            let (interrupt, _) = tokio::sync::watch::channel(interrupt_immediately);
            if interrupt_immediately {
                worker.terminate("model_instance_stopped", "model instance was stopped");
            } else {
                worker.shutdown();
            }
            let notifier = self.client.notifier();
            tokio::spawn(async move {
                tokio::task::yield_now().await;
                notifier.send(ResidencyNotification::ReleaseFinished {
                    instance_id,
                    result: Ok(()),
                });
            });
            ReleaseControl { interrupt, worker }
        }

        fn supervise_worker(
            &self,
            _instance_id: ModelInstanceId,
            _worker: Arc<dyn ResidencyWorker>,
            _resident: TestResident,
        ) {
        }

        fn idle_timeout(&self) -> std::time::Duration {
            self.idle_timeout
        }
    }

    fn test_residency(
        auto_ready: bool,
    ) -> (
        ResidencyClient<TestResident>,
        tokio::sync::mpsc::UnboundedReceiver<(ModelInstanceId, String)>,
        Arc<Mutex<Vec<Arc<TestResidencyWorker>>>>,
        tokio::task::JoinHandle<()>,
    ) {
        test_residency_with_idle_timeout(auto_ready, std::time::Duration::from_secs(600))
    }

    fn test_residency_with_idle_timeout(
        auto_ready: bool,
        idle_timeout: std::time::Duration,
    ) -> (
        ResidencyClient<TestResident>,
        tokio::sync::mpsc::UnboundedReceiver<(ModelInstanceId, String)>,
        Arc<Mutex<Vec<Arc<TestResidencyWorker>>>>,
        tokio::task::JoinHandle<()>,
    ) {
        let (client, inbox) = ResidencyClient::channel();
        let (loads, load_events) = tokio::sync::mpsc::unbounded_channel();
        let workers = Arc::new(Mutex::new(Vec::new()));
        let driver = TestResidencyDriver {
            client: client.clone(),
            next_instance_id: Arc::new(AtomicU64::new(1)),
            loads,
            auto_ready,
            workers: Arc::clone(&workers),
            idle_timeout,
        };
        let actor = tokio::spawn(ModelResidency::new(driver).run(inbox));
        (client, load_events, workers, actor)
    }

    fn assert_explicit_stop(result: Result<ResidencyGrant<TestResident>, InventoryError>) {
        assert!(matches!(
            result,
            Err(InventoryError::ModelOperation {
                code,
                retryable: false,
                ..
            }) if code == "model_instance_stopped"
        ));
    }

    fn test_target(model_id: &str) -> ResolvedResidencyTarget {
        test_target_with_key(model_id, &format!("configuration:{model_id}"))
    }

    fn test_target_with_key(model_id: &str, configuration_key: &str) -> ResolvedResidencyTarget {
        ResolvedResidencyTarget {
            model_id: format!("{model_id}:gguf:test")
                .parse()
                .expect("valid test model ID"),
            configuration: test_configuration(model_id),
            configuration_key: configuration_key.to_owned(),
        }
    }

    #[tokio::test]
    async fn equivalent_loading_demand_joins_one_instance_and_stop_terminalizes_every_waiter() {
        let (client, mut loads, _, actor) = test_residency(false);
        let first = tokio::spawn({
            let client = client.clone();
            async move {
                client
                    .acquire(test_target("model-a"), ResidencyAcquisition::Warm)
                    .await
            }
        });
        let (instance_id, model_id) = loads.recv().await.expect("first load admitted");
        assert_eq!(model_id, "model-a:gguf:test");
        let second = tokio::spawn({
            let client = client.clone();
            async move {
                client
                    .acquire(test_target("model-a"), ResidencyAcquisition::Warm)
                    .await
            }
        });
        tokio::task::yield_now().await;

        assert!(loads.try_recv().is_err());
        let snapshot = client.snapshot().await.unwrap();
        assert_eq!(snapshot.instances.len(), 1);
        assert_eq!(snapshot.instances[0].id, instance_id);
        assert!(matches!(
            snapshot.instances[0].lifecycle,
            ModelInstanceLifecycle::Loading { .. }
        ));

        client
            .stop(instance_id.clone())
            .await
            .expect("stop completes");
        assert_explicit_stop(first.await.expect("first waiter task"));
        assert_explicit_stop(second.await.expect("second waiter task"));
        let snapshot = client.snapshot().await.unwrap();
        assert!(matches!(
            snapshot
                .instances
                .last()
                .map(|instance| &instance.lifecycle),
            Some(ModelInstanceLifecycle::Stopped {
                reason: ModelReleaseReason::UserStop
            })
        ));
        actor.abort();
    }

    #[tokio::test]
    async fn changed_configuration_replaces_residency_for_the_same_model_id() {
        let (client, mut loads, _, actor) = test_residency(true);
        let first = client
            .acquire(
                test_target_with_key("model-a", "configuration:v1"),
                ResidencyAcquisition::Warm,
            )
            .await
            .expect("first configuration becomes ready");
        let ResidencyGrant::Ready(first) = first else {
            panic!("warm acquisition returned an inference lease");
        };
        let _ = loads.recv().await.expect("first configuration load");

        let replacement = tokio::spawn({
            let client = client.clone();
            async move {
                client
                    .acquire(
                        test_target_with_key("model-a", "configuration:v2"),
                        ResidencyAcquisition::Warm,
                    )
                    .await
            }
        });
        let (replacement_id, replacement_model_id) =
            loads.recv().await.expect("replacement configuration load");
        assert_eq!(replacement_model_id, "model-a:gguf:test");
        assert_ne!(replacement_id, first.id);
        assert!(matches!(
            replacement.await.expect("replacement task"),
            Ok(ResidencyGrant::Ready(_))
        ));

        client.stop(replacement_id).await.expect("replacement stop");
        actor.abort();
    }

    #[tokio::test]
    async fn package_removal_admission_excludes_new_model_loads() {
        let (client, mut loads, _, actor) = test_residency(false);
        let permit = client
            .acquire_package_removal(ModelPackageId("package-a".to_owned()))
            .await
            .expect("package removal admitted");
        let request = tokio::spawn({
            let client = client.clone();
            async move {
                client
                    .acquire(test_target("model-a"), ResidencyAcquisition::Warm)
                    .await
            }
        });
        tokio::task::yield_now().await;
        assert!(loads.try_recv().is_err());

        drop(permit);
        let (instance_id, model_id) = loads.recv().await.expect("load admitted after removal");
        assert_eq!(model_id, "model-a:gguf:test");
        client.stop(instance_id).await.expect("load stopped");
        assert_explicit_stop(request.await.expect("request waiter"));
        actor.abort();
    }

    #[tokio::test]
    async fn conflicting_loading_demand_is_fifo_and_stale_stop_cannot_touch_the_successor() {
        let (client, mut loads, _, actor) = test_residency(false);
        let request = |model_id: &'static str| {
            let client = client.clone();
            tokio::spawn(async move {
                client
                    .acquire(test_target(model_id), ResidencyAcquisition::Warm)
                    .await
            })
        };
        let first = request("model-a");
        let (first_id, _) = loads.recv().await.expect("first load admitted");
        let second = request("model-b");
        let third = request("model-c");
        tokio::task::yield_now().await;
        assert!(loads.try_recv().is_err());

        client.stop(first_id.clone()).await.expect("first stop");
        let (second_id, second_model) = loads.recv().await.expect("second load admitted");
        assert_eq!(second_model, "model-b:gguf:test");
        client
            .stop(first_id)
            .await
            .expect("stale stop is idempotent");
        assert_eq!(
            client
                .snapshot()
                .await
                .unwrap()
                .instances
                .last()
                .map(|instance| &instance.id),
            Some(&second_id),
        );

        client.stop(second_id).await.expect("second stop");
        let (third_id, third_model) = loads.recv().await.expect("third load admitted");
        assert_eq!(third_model, "model-c:gguf:test");
        client.stop(third_id).await.expect("third stop");
        assert_explicit_stop(first.await.expect("first task"));
        assert_explicit_stop(second.await.expect("second task"));
        assert_explicit_stop(third.await.expect("third task"));
        actor.abort();
    }

    #[tokio::test]
    async fn cancelled_waiter_detaches_without_cancelling_the_shared_load() {
        let (client, mut loads, _, actor) = test_residency(false);
        let cancelled = tokio::spawn({
            let client = client.clone();
            async move {
                client
                    .acquire(test_target("model-a"), ResidencyAcquisition::Warm)
                    .await
            }
        });
        let (instance_id, _) = loads.recv().await.expect("load admitted");
        cancelled.abort();
        let joined = tokio::spawn({
            let client = client.clone();
            async move {
                client
                    .acquire(test_target("model-a"), ResidencyAcquisition::Warm)
                    .await
            }
        });
        tokio::task::yield_now().await;
        assert!(loads.try_recv().is_err());
        assert_eq!(client.snapshot().await.unwrap().instances.len(), 1);

        client.stop(instance_id).await.expect("stop completes");
        assert_explicit_stop(joined.await.expect("joined task"));
        actor.abort();
    }

    #[tokio::test]
    async fn ready_stop_interrupts_active_execution_and_a_later_request_gets_a_new_instance() {
        let (client, mut loads, workers, actor) = test_residency(true);
        let first_lease = match client
            .acquire(
                test_target("model-a"),
                ResidencyAcquisition::Inference { progress: None },
            )
            .await
            .expect("first acquisition")
        {
            ResidencyGrant::Lease(lease) => lease,
            ResidencyGrant::Ready(_) => panic!("inference acquisition returned warm result"),
        };
        let (first_id, _) = loads.recv().await.expect("first load");
        client.stop(first_id.clone()).await.expect("ready stop");
        assert_eq!(
            workers.lock().expect("test worker lock")[0]
                .interrupted
                .load(Ordering::Acquire),
            1,
        );
        drop(first_lease);

        let second_lease = match client
            .acquire(
                test_target("model-a"),
                ResidencyAcquisition::Inference { progress: None },
            )
            .await
            .expect("second acquisition")
        {
            ResidencyGrant::Lease(lease) => lease,
            ResidencyGrant::Ready(_) => panic!("inference acquisition returned warm result"),
        };
        let (second_id, _) = loads.recv().await.expect("second load");
        assert_ne!(first_id, second_id);
        drop(second_lease);
        client.stop(second_id).await.expect("second stop");
        actor.abort();
    }

    #[tokio::test]
    async fn canceled_load_retains_worker_before_or_after_stop_notification() {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            for start_after_stop in [false, true] {
                let (client, mut loads, _, actor) = test_residency(false);
                let loading = tokio::spawn({ let client = client.clone(); async move {
                    client.acquire(test_target("model-a"), ResidencyAcquisition::Warm).await
                }});
                let (instance_id, _) = loads.recv().await.unwrap();
                let worker = Arc::new(TestResidencyWorker::default());
                worker.fail_retirement.store(true, Ordering::Release);
                let started = ResidencyCommand::Notification(ResidencyNotification::LoadWorkerStarted { instance_id: instance_id.clone(), worker: worker.clone() });
                let (reply, stopped) = tokio::sync::oneshot::channel();
                let stop = ResidencyCommand::Stop { instance_id: instance_id.clone(), reply };
                let commands = if start_after_stop { [stop, started] } else { [started, stop] };
                for command in commands { client.commands.send(command).unwrap(); }
                assert!(matches!(stopped.await.unwrap(), Err(InventoryError::ModelOperation { code, .. }) if code == "worker_retirement_failed"));
                assert_explicit_stop(loading.await.unwrap());
                let snapshot = client.snapshot().await.unwrap();
                assert_eq!(snapshot.instances[0].id, instance_id);
                assert!(matches!(snapshot.instances[0].lifecycle, ModelInstanceLifecycle::Stopping { .. }));
                assert!(client.acquire(test_target("model-b"), ResidencyAcquisition::Warm).await.is_err());
                assert!(client.acquire_package_removal(ModelPackageId("test-package".into())).await.is_err());
                assert!(loads.try_recv().is_err());
                worker.fail_retirement.store(false, Ordering::Release);
                client.stop(instance_id).await.expect("retry retires canceled load worker");
                let next = tokio::spawn({ let client = client.clone(); async move {
                    client.acquire(test_target("model-b"), ResidencyAcquisition::Warm).await
                }});
                let (next_id, _) = loads.recv().await.unwrap();
                client.stop(next_id).await.unwrap();
                assert_explicit_stop(next.await.unwrap());
                actor.abort();
            }
        }).await.expect("cancellation lost worker ownership or a reply");
    }

    #[tokio::test]
    async fn failed_load_and_ready_worker_retain_ownership_until_cleanup_succeeds() {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            for ready in [false, true] {
                let (client, mut loads, workers, actor) = test_residency(ready);
                let mut acquisition = Some(tokio::spawn({ let client = client.clone(); async move {
                    client.acquire(test_target("model-a"), ResidencyAcquisition::Warm).await
                }}));
                let (instance_id, _) = loads.recv().await.unwrap();
                let worker = if ready {
                    acquisition.take().unwrap().await.unwrap().unwrap();
                    Arc::clone(&workers.lock().unwrap()[0])
                } else {
                    let worker = Arc::new(TestResidencyWorker::default());
                    client.notifier().send(ResidencyNotification::LoadWorkerStarted {
                        instance_id: instance_id.clone(), worker: worker.clone(),
                    });
                    worker
                };
                worker.fail_retirement.store(true, Ordering::Release);
                let failure = ModelOperationFailure::new("fixture_failure", "original worker failure", false);
                let command = if ready {
                    ResidencyCommand::Notification(ResidencyNotification::WorkerFailed { instance_id: instance_id.clone(), failure })
                } else {
                    ResidencyCommand::LoadFinished { instance_id: instance_id.clone(), result: Err(failure) }
                };
                client.commands.send(command).unwrap();
                if let Some(acquisition) = acquisition {
                    assert!(matches!(acquisition.await.unwrap(),
                        Err(InventoryError::ModelOperation { code, .. }) if code == "fixture_failure"));
                }
                // The next admission waits on cleanup and must receive its failure, not start a successor.
                assert!(matches!(client.acquire(test_target("model-b"), ResidencyAcquisition::Warm).await,
                    Err(InventoryError::ModelOperation { code, .. }) if code == "worker_retirement_failed"));
                let snapshot = client.snapshot().await.unwrap();
                assert_eq!(snapshot.instances.len(), 1);
                assert_eq!(snapshot.instances[0].id, instance_id);
                assert!(matches!(snapshot.instances[0].lifecycle,
                    ModelInstanceLifecycle::Stopping { reason: ModelReleaseReason::Failure, .. }));
                assert!(client.acquire_package_removal(ModelPackageId("test-package".into())).await.is_err());
                assert!(loads.try_recv().is_err());
                worker.fail_retirement.store(false, Ordering::Release);
                client.stop(instance_id.clone()).await.expect("cleanup retry retires failed worker");
                let snapshot = client.snapshot().await.unwrap();
                let terminal = snapshot.instances.iter().find(|instance| instance.id == instance_id).unwrap();
                assert!(matches!(&terminal.lifecycle, ModelInstanceLifecycle::Failed { failure }
                    if serde_json::to_string(failure).unwrap().contains("fixture_failure")));
                let next = tokio::spawn({ let client = client.clone(); async move {
                    client.acquire(test_target("model-b"), ResidencyAcquisition::Warm).await
                }});
                let (next_id, _) = loads.recv().await.unwrap();
                client.stop(next_id).await.unwrap();
                let _ = next.await.unwrap();
                actor.abort();
            }
        }).await.expect("failure cleanup lost ownership or admission reply");
    }

    #[tokio::test]
    async fn unproven_ready_release_blocks_loads_and_removal_until_exact_stop_retry() {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            for explicit_stop in [true, false] {
                let (client, mut loads, workers, actor) = test_residency(true);
                client.acquire(test_target("model-a"), ResidencyAcquisition::Warm).await.unwrap();
                let (first_id, _) = loads.recv().await.unwrap();
                let worker = Arc::clone(&workers.lock().unwrap()[0]);
                worker.fail_retirement.store(true, Ordering::Release);
                let failed = if explicit_stop {
                    client.stop(first_id.clone()).await
                } else {
                    client.acquire(test_target("model-b"), ResidencyAcquisition::Warm).await.map(|_| ())
                };
                assert!(matches!(failed, Err(InventoryError::ModelOperation { code, .. }) if code == "worker_retirement_failed"));
                let snapshot = client.snapshot().await.unwrap();
                assert_eq!(snapshot.instances.len(), 1);
                assert_eq!(snapshot.instances[0].id, first_id);
                assert!(matches!(snapshot.instances[0].lifecycle, ModelInstanceLifecycle::Stopping { .. }));
                assert!(client.acquire(test_target("model-b"), ResidencyAcquisition::Warm).await.is_err());
                assert!(client.acquire_package_removal(ModelPackageId("test-package".into())).await.is_err());
                assert!(loads.try_recv().is_err(), "unproven retirement admitted a new load");
                worker.fail_retirement.store(false, Ordering::Release);
                client.stop(first_id.clone()).await.expect("retry retires the same worker");
                client.acquire(test_target("model-b"), ResidencyAcquisition::Warm).await.unwrap();
                let (next_id, _) = loads.recv().await.unwrap();
                assert_ne!(first_id, next_id);
                client.stop(next_id).await.unwrap();
                actor.abort();
            }
        }).await.expect("retirement failure or retry lost a reply");
    }

    #[tokio::test]
    async fn replacement_drains_the_accepted_lease_before_starting_the_next_model() {
        let (client, mut loads, workers, actor) = test_residency(true);
        let lease = match client
            .acquire(
                test_target("model-a"),
                ResidencyAcquisition::Inference { progress: None },
            )
            .await
            .expect("first acquisition")
        {
            ResidencyGrant::Lease(lease) => lease,
            ResidencyGrant::Ready(_) => panic!("inference acquisition returned warm result"),
        };
        let _ = loads.recv().await.expect("first load");
        let replacement = tokio::spawn({
            let client = client.clone();
            async move {
                client
                    .acquire(test_target("model-b"), ResidencyAcquisition::Warm)
                    .await
            }
        });
        tokio::task::yield_now().await;
        assert!(loads.try_recv().is_err());
        assert_eq!(
            workers.lock().expect("test worker lock")[0]
                .shutdown
                .load(Ordering::Acquire),
            0,
        );

        drop(lease);
        let (_, model_id) = loads.recv().await.expect("replacement load");
        assert_eq!(model_id, "model-b:gguf:test");
        assert!(matches!(
            replacement.await.expect("replacement task"),
            Ok(ResidencyGrant::Ready(_))
        ));
        assert_eq!(
            workers.lock().expect("test worker lock")[0]
                .shutdown
                .load(Ordering::Acquire),
            1,
        );
        let current = client
            .snapshot()
            .await
            .unwrap()
            .instances
            .last()
            .expect("current instance")
            .id
            .clone();
        client.stop(current).await.expect("cleanup stop");
        actor.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn inference_lease_blocks_idle_release_and_final_release_starts_a_full_interval() {
        let timeout = std::time::Duration::from_secs(60);
        let (client, mut loads, workers, actor) = test_residency_with_idle_timeout(true, timeout);
        let lease = match client
            .acquire(
                test_target("model-a"),
                ResidencyAcquisition::Inference { progress: None },
            )
            .await
            .expect("inference acquisition")
        {
            ResidencyGrant::Lease(lease) => lease,
            ResidencyGrant::Ready(_) => panic!("inference acquisition returned warm result"),
        };
        let _ = loads.recv().await.expect("model load");

        tokio::time::advance(timeout * 2).await;
        assert!(matches!(
            client
                .snapshot()
                .await
                .unwrap()
                .instances
                .last()
                .map(|instance| &instance.lifecycle),
            Some(ModelInstanceLifecycle::Ready { .. })
        ));

        drop(lease);
        tokio::task::yield_now().await;
        tokio::time::advance(timeout - std::time::Duration::from_secs(1)).await;
        assert!(matches!(
            client
                .snapshot()
                .await
                .unwrap()
                .instances
                .last()
                .map(|instance| &instance.lifecycle),
            Some(ModelInstanceLifecycle::Ready { .. })
        ));
        tokio::time::advance(std::time::Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        assert!(matches!(
            client
                .snapshot()
                .await
                .unwrap()
                .instances
                .last()
                .map(|instance| &instance.lifecycle),
            Some(ModelInstanceLifecycle::Stopped {
                reason: ModelReleaseReason::IdleTimeout
            })
        ));
        assert_eq!(
            workers.lock().expect("test worker lock")[0]
                .shutdown
                .load(Ordering::Acquire),
            1,
        );
        actor.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn equivalent_warm_demand_restarts_the_idle_interval() {
        let timeout = std::time::Duration::from_secs(60);
        let (client, mut loads, _, actor) = test_residency_with_idle_timeout(true, timeout);
        let first = client
            .acquire(test_target("model-a"), ResidencyAcquisition::Warm)
            .await
            .expect("warm acquisition");
        assert!(matches!(first, ResidencyGrant::Ready(_)));
        let _ = loads.recv().await.expect("model load");

        tokio::time::advance(timeout - std::time::Duration::from_secs(1)).await;
        let second = client
            .acquire(test_target("model-a"), ResidencyAcquisition::Warm)
            .await
            .expect("equivalent warm acquisition");
        assert!(matches!(second, ResidencyGrant::Ready(_)));
        tokio::time::advance(timeout - std::time::Duration::from_secs(1)).await;
        assert!(matches!(
            client
                .snapshot()
                .await
                .unwrap()
                .instances
                .last()
                .map(|instance| &instance.lifecycle),
            Some(ModelInstanceLifecycle::Ready { .. })
        ));
        actor.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn unrelated_package_removal_does_not_restart_the_idle_interval() {
        let timeout = std::time::Duration::from_secs(60);
        let (client, mut loads, _, actor) = test_residency_with_idle_timeout(true, timeout);
        let ready = client
            .acquire(test_target("model-a"), ResidencyAcquisition::Warm)
            .await
            .expect("warm acquisition");
        assert!(matches!(ready, ResidencyGrant::Ready(_)));
        let _ = loads.recv().await.expect("model load");

        tokio::time::advance(timeout / 2).await;
        let permit = client
            .acquire_package_removal(ModelPackageId("unrelated-package".to_owned()))
            .await
            .expect("unrelated package removal admitted");
        drop(permit);
        tokio::task::yield_now().await;

        tokio::time::advance(timeout / 2).await;
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        assert!(matches!(
            client
                .snapshot()
                .await
                .unwrap()
                .instances
                .last()
                .map(|instance| &instance.lifecycle),
            Some(ModelInstanceLifecycle::Stopped {
                reason: ModelReleaseReason::IdleTimeout
            })
        ));
        actor.abort();
    }

    #[test]
    fn explicit_stop_is_a_canonical_non_retryable_preparation_result() {
        match stopped_instance_preparation_error(&ModelReleaseReason::UserStop) {
            InventoryError::ModelOperation {
                code,
                message,
                retryable,
            } => {
                assert_eq!(code, "model_instance_stopped");
                assert_eq!(message, "model instance was stopped");
                assert!(!retryable);
            }
            error => panic!("unexpected stopped-instance result: {error:?}"),
        }
    }

}