//! Concrete root facade over the family-neutral execution owner.

use magnitude_generation::{DetailedUsage, FinishReason, Method, OutputToken};
use magnitude_model_contracts::PreparedModelInput;
use magnitude_model_executor::{ExecutorDomain, ProgramFamily, RequestId, ResourcePlan};
use magnitude_service::{
    owner::{AdmissionError, Owner, Status},
    protocol::{AdmitRequest, CapacityStatus, WorkerCommand, WorkerReply},
    publication::{
        ModelUnloadCause, Publication, PublicationQueue, PublicationReceiver, PublicationWake,
    },
    retention::{RetentionKey, RetentionRequest},
    worker::{self, CompletionWake, Drive, Driven, Worker, WorkerWakeHandle},
};
use std::future::poll_fn;
use std::sync::Arc;

use crate::options::{ExecutionManifest, ReadyInfo, ResourcePlanSummary};

pub struct OutputBatch {
    pub tokens: Vec<OutputToken>,
    pub finish: Option<FinishReason>,
    pub usage: Option<DetailedUsage>,
    pub method: Option<String>,
    pub timings: Option<magnitude_chat::ExecutionTimings>,
    pub error: Option<String>,
}

struct ExecutionOwner<F: ProgramFamily> {
    owner: Option<Owner<F>>,
    method: Arc<dyn Method>,
    wakes: Option<WorkerWakeHandle>,
}
impl<F: ProgramFamily> ExecutionOwner<F> {
    fn shed_unloaded_owner(&mut self) {
        if self.owner.as_mut().is_some_and(Owner::memory_unload_ready) {
            self.owner.take();
        }
    }
}
impl<F: ProgramFamily> Driven for ExecutionOwner<F> {
    fn periodic(&mut self, now: u64) -> Result<(), String> {
        if let Some(owner) = self.owner.as_mut() {
            <Owner<F> as Driven>::periodic(owner, now)?;
        }
        self.shed_unloaded_owner();
        Ok(())
    }
    fn install_wake_handle(&mut self, wakes: WorkerWakeHandle) {
        self.wakes = Some(wakes);
    }
    fn publication_wake(
        &mut self,
        request: RequestId,
        wake: PublicationWake,
        _now: u64,
    ) -> Result<(), String> {
        match self.owner.as_mut() {
            Some(owner) => owner.publication_wake(request, wake),
            None => Ok(()),
        }
    }
    fn command(&mut self, command: WorkerCommand, now: u64) -> Result<WorkerReply, String> {
        self.shed_unloaded_owner();
        let Some(owner) = self.owner.as_mut() else {
            return match command {
                WorkerCommand::Admit(_) => Ok(WorkerReply::AdmissionRefused(
                    AdmissionError::ModelUnloaded {
                        cause: ModelUnloadCause::MemoryPressure,
                    },
                )),
                WorkerCommand::Capacity => Ok(WorkerReply::Capacity(CapacityStatus {
                    active: 0,
                    limit: 0,
                })),
                WorkerCommand::Cancel { .. } | WorkerCommand::Stop { .. } => {
                    Ok(WorkerReply::Acknowledged)
                }
                _ => Err("model unloaded due to memory pressure".into()),
            };
        };
        match command {
            WorkerCommand::Check => {
                if owner.memory_unloading() {
                    Err("model unloaded due to memory pressure".into())
                } else {
                    match owner.fatal_error() {
                        Some(error) => Err(error.to_owned()),
                        None => Ok(WorkerReply::Acknowledged),
                    }
                }
            }
            WorkerCommand::Admit(AdmitRequest {
                seed,
                input,
                retention,
                output_capacity,
            }) => {
                if output_capacity == 0 {
                    return Err("request output capacity must be positive".into());
                }
                if seed.prompt() != input.tokens() || seed.layout() != input.layout() {
                    return Err("generation seed does not match prepared input".into());
                }
                let generation = seed.into_generation(self.method.clone())?;
                let admitted = match retention {
                    Some(retention) => owner.admit_retained_with(
                        generation,
                        retention,
                        now,
                        move |domain, request, _, hit| match hit {
                            Some(_) => domain.install_retained_input(request, input),
                            None => domain.install_input(request, input),
                        },
                    ),
                    None => owner.admit_with(generation, now, move |domain, request, _| {
                        domain.install_input(request, input)
                    }),
                };
                let request = match admitted {
                    Ok(request) => request,
                    Err(error) => return Ok(WorkerReply::AdmissionRefused(error)),
                };
                let wakes = self
                    .wakes
                    .as_ref()
                    .ok_or("publication wake handle was not installed")?
                    .clone();
                let (sender, receiver) = PublicationQueue::bounded(output_capacity, move |wake| {
                    wakes.publication(request, wake);
                })?;
                owner.attach_publication(request, sender, output_capacity)?;
                Ok(WorkerReply::Admitted { request, receiver })
            }
            WorkerCommand::Stop { request } => {
                owner.stop(request)?;
                Ok(WorkerReply::Acknowledged)
            }
            WorkerCommand::Cancel { request } => {
                owner.release(request)?;
                Ok(WorkerReply::Acknowledged)
            }
            WorkerCommand::Status { request } => Ok(WorkerReply::Status(owner.status(request)?)),
            WorkerCommand::Capacity => {
                let (active, limit) = if owner.memory_unloading() {
                    (0, 0)
                } else {
                    owner.request_capacity()
                };
                Ok(WorkerReply::Capacity(CapacityStatus { active, limit }))
            }
            WorkerCommand::Close => {
                Err("worker lifecycle command reached the execution domain".into())
            }
        }
    }
    fn advance(&mut self, now: u64, wake: CompletionWake) -> Result<Drive, String> {
        self.shed_unloaded_owner();
        let result = match self.owner.as_mut() {
            Some(owner) => <Owner<F> as Driven>::advance(owner, now, wake),
            None => Ok(Drive::Idle),
        };
        self.shed_unloaded_owner();
        result
    }
    fn failed(&mut self, error: &str) {
        if let Some(owner) = self.owner.as_mut() {
            <Owner<F> as Driven>::failed(owner, error);
        }
    }
    fn failure(&self) -> Option<&str> {
        self.owner.as_ref().and_then(<Owner<F> as Driven>::failure)
    }
    fn shutdown(&mut self) -> Result<bool, String> {
        match self.owner.as_mut() {
            Some(owner) => <Owner<F> as Driven>::shutdown(owner),
            None => Ok(true),
        }
    }
}

/// Owns the numerical worker. Its executor types are erased at this boundary.
pub struct EngineService {
    worker: Worker,
    retention: Option<RetentionKey>,
    method_identity: String,
}
impl EngineService {
    /// Production construction seam. The same immutable plan is handed to the
    /// numerical factory for every physical allocation and to the service
    /// owner for retention/capacity limits before readiness is published.
    pub(crate) fn spawn_planned_domain<F: ProgramFamily + 'static, H>(
        factory: impl FnOnce(&ExecutionManifest) -> Result<(ExecutorDomain<F>, ResourcePlan), String>
            + Send
            + 'static,
        manifest: ExecutionManifest,
        method: Arc<dyn Method>,
        control_capacity: usize,
        host: impl FnOnce() -> Result<(H, Option<RetentionKey>), String>,
    ) -> Result<(Self, ReadyInfo, H), String> {
        let method_identity = method.identity().to_owned();
        let (worker, ready, (host, retention)) = Worker::spawn_ready_with(
            move || {
                let (domain, plan) = factory(&manifest)?;
                if domain.execution_path() != manifest.path {
                    return Err("executor domain differs from the planned execution path".into());
                }
                let backend = domain.execution_backend();
                let owner = Owner::with_resource_plan(domain, manifest.service.clone(), &plan)?;
                let ready = ReadyInfo {
                    package: manifest.package.identity.clone(),
                    model: manifest.model.clone(),
                    service: manifest.service.clone(),
                    resources: ResourcePlanSummary::from_plan(&plan)?,
                    path: manifest.path,
                    backend,
                };
                Ok((
                    Box::new(ExecutionOwner {
                        owner: Some(owner),
                        method,
                        wakes: None,
                    }) as Box<dyn Driven>,
                    ready,
                ))
            },
            control_capacity,
            host,
        )?;
        let service = Self {
            worker,
            retention,
            method_identity,
        };
        Ok((service, ready, host))
    }

    pub fn client(&self) -> EngineClient {
        EngineClient {
            worker: self.worker.client(),
            retention: self.retention.clone(),
            method_identity: self.method_identity.clone(),
        }
    }
    pub fn close(&mut self) {
        self.worker.close();
    }
}

#[derive(Clone)]
pub struct EngineClient {
    worker: worker::Client,
    retention: Option<RetentionKey>,
    method_identity: String,
}
impl EngineClient {
    pub fn method_identity(&self) -> &str {
        &self.method_identity
    }
    pub async fn check(&self) -> Result<(), String> {
        match self.worker.dispatch(WorkerCommand::Check)?.await? {
            WorkerReply::Acknowledged => Ok(()),
            _ => Err("execution worker returned an invalid health reply".into()),
        }
    }
    pub async fn capacity(&self) -> Result<CapacityStatus, String> {
        match self.worker.dispatch(WorkerCommand::Capacity)?.await? {
            WorkerReply::Capacity(capacity) => Ok(capacity),
            _ => Err("execution worker returned an invalid capacity reply".into()),
        }
    }
    pub async fn admit(
        &self,
        seed: magnitude_generation::GenerationSeed,
        input: PreparedModelInput,
        output_capacity: usize,
    ) -> Result<EngineRequest, AdmissionError> {
        let retention = self
            .retention
            .as_ref()
            .map(|key| RetentionRequest::new(key.clone(), &input))
            .transpose()?;
        self.admit_inner(seed, input, retention, output_capacity)
            .await
    }
    pub async fn admit_retained(
        &self,
        seed: magnitude_generation::GenerationSeed,
        input: PreparedModelInput,
        retention: RetentionRequest,
        output_capacity: usize,
    ) -> Result<EngineRequest, AdmissionError> {
        self.admit_inner(seed, input, Some(retention), output_capacity)
            .await
    }
    async fn admit_inner(
        &self,
        seed: magnitude_generation::GenerationSeed,
        input: PreparedModelInput,
        retention: Option<RetentionRequest>,
        output_capacity: usize,
    ) -> Result<EngineRequest, AdmissionError> {
        if output_capacity == 0 {
            return Err("request output capacity must be positive".into());
        }
        if seed.prompt() != input.tokens() || seed.layout() != input.layout() {
            return Err("generation seed does not match prepared input".into());
        }
        let reply = self
            .worker
            .dispatch(WorkerCommand::Admit(AdmitRequest {
                seed,
                input,
                retention,
                output_capacity,
            }))?
            .await?;
        let (id, receiver) = match reply {
            WorkerReply::Admitted { request, receiver } => (request, receiver),
            WorkerReply::AdmissionRefused(error) => return Err(error),
            _ => return Err("execution worker returned an invalid admission reply".into()),
        };
        Ok(EngineRequest {
            client: self.worker.clone(),
            id,
            output_capacity,
            receiver,
            released: false,
        })
    }
}

/// Unique host receiver for a bounded worker publication stream.
pub struct EngineRequest {
    client: worker::Client,
    id: RequestId,
    output_capacity: usize,
    receiver: PublicationReceiver,
    released: bool,
}

impl EngineRequest {
    pub const fn id(&self) -> RequestId {
        self.id
    }
    pub const fn output_capacity(&self) -> usize {
        self.output_capacity
    }

    pub async fn status(&self) -> Result<Status, String> {
        if self.released {
            return Err("request released".into());
        }
        match self
            .client
            .dispatch(WorkerCommand::Status { request: self.id })?
            .await?
        {
            WorkerReply::Status(status) => Ok(status),
            _ => Err("execution worker returned an invalid status reply".into()),
        }
    }

    pub async fn receive(&mut self) -> Result<OutputBatch, String> {
        if self.released {
            return Err("request released".into());
        }
        let publication = poll_fn(|cx| self.receiver.poll_next(cx))
            .await
            .ok_or("worker publication stream ended without a terminal event")?;
        let batch = match publication {
            Publication::Output(tokens) => OutputBatch {
                tokens,
                finish: None,
                usage: None,
                method: None,
                timings: None,
                error: None,
            },
            Publication::Completed {
                finish,
                usage,
                method,
                timings,
            } => {
                self.released = true;
                OutputBatch {
                    tokens: Vec::new(),
                    finish: Some(finish),
                    usage: Some(usage),
                    method: Some(method),
                    error: None,
                    timings: Some(magnitude_chat::ExecutionTimings {
                        prompt_ns: timings.prompt_ns,
                        predicted_ns: timings.predicted_ns,
                    }),
                }
            }
            Publication::Failed(error) => {
                self.released = true;
                OutputBatch {
                    tokens: Vec::new(),
                    finish: Some(FinishReason::Failed),
                    usage: None,
                    method: None,
                    timings: None,
                    error: Some(format!("{error:?}")),
                }
            }
        };
        Ok(batch)
    }

    /// Request a normal ordered stop, then drain any accepted output before
    /// returning its terminal metadata to the host parser.
    pub async fn stop(&mut self) -> Result<OutputBatch, String> {
        if self.released {
            return Err("request released".into());
        }
        match self
            .client
            .dispatch(WorkerCommand::Stop { request: self.id })?
            .await?
        {
            WorkerReply::Acknowledged => {}
            _ => return Err("execution worker returned an invalid stop reply".into()),
        }
        loop {
            let batch = self.receive().await?;
            if batch.finish.is_some() {
                return Ok(batch);
            }
        }
    }
}
