//! The worker's thread-confined execution owner over the family-neutral
//! scheduler. Only the closed scheduler command vocabulary reaches it.

use crate::error::UnloadCause;
use magnitude_executor::{ExecutorDomain, ProgramFamily, RequestId};
use magnitude_generation::Method;
use magnitude_scheduler::{
    owner::{AdmissionError, Owner},
    protocol::{AdmitRequest, RequestSnapshot, WorkerCommand, WorkerReply},
    publication::{ModelUnloadCause, PublicationQueue, PublicationWake, RequestError},
    worker::{CompletionWake, Drive, Driven, WorkerWakeHandle},
};
use std::sync::{Arc, Mutex};

/// Set once by the execution thread when the model stops serving; read by
/// the worker session.
#[derive(Default)]
pub(crate) struct UnloadNotice {
    cause: Mutex<Option<UnloadCause>>,
    wake: Mutex<Option<Box<dyn Fn() + Send>>>,
}

impl UnloadNotice {
    pub fn install_wake(&self, wake: impl Fn() + Send + 'static) {
        *self.wake.lock().unwrap() = Some(Box::new(wake));
    }

    fn record(&self, cause: UnloadCause) {
        let recorded = {
            let mut current = self.cause.lock().unwrap();
            if current.is_some() {
                false
            } else {
                *current = Some(cause);
                true
            }
        };
        if recorded {
            if let Some(wake) = self.wake.lock().unwrap().as_ref() {
                wake();
            }
        }
    }

    pub fn cause(&self) -> Option<UnloadCause> {
        self.cause.lock().unwrap().clone()
    }
}

pub(crate) struct ExecutionOwner<F: ProgramFamily> {
    pub owner: Option<Owner<F>>,
    pub method: Arc<dyn Method>,
    pub wakes: Option<WorkerWakeHandle>,
    pub notice: Arc<UnloadNotice>,
}

impl<F: ProgramFamily> ExecutionOwner<F> {
    fn shed_unloaded_owner(&mut self) {
        if self.owner.as_mut().is_some_and(Owner::memory_unload_ready) {
            self.owner.take();
            self.notice.record(UnloadCause::MemoryPressure);
        }
    }

    fn snapshot(owner: &Owner<F>, request: RequestId) -> Option<RequestSnapshot> {
        let usage = owner.usage(request).ok()?;
        Some(RequestSnapshot {
            status: owner.status(request).ok()?,
            prompt_tokens: usage.prompt_tokens,
            cached_tokens: usage.cached_tokens,
            resident_position: owner.resident_position(request).ok()?,
            output_tokens: owner.output_len(request).ok()?,
        })
    }

    /// A fresh reading of every domain the device uses, then the heap's
    /// standing as that reading left it. Fails only when an observation
    /// fails.
    fn observe(owner: &Owner<F>) -> Result<WorkerReply, String> {
        owner.inspect_domain(|domain: &ExecutorDomain<F>| {
            let readings = domain.refresh_memory().map_err(|error| error.to_string())?;
            let heap = domain.memory();
            Ok(WorkerReply::Observed {
                readings,
                standing: heap
                    .standing()
                    .map_err(|error| format!("memory standing is unavailable: {error:?}"))?,
                holdings: heap.holdings().collect(),
            })
        })
    }
}

impl<F: ProgramFamily> Driven for ExecutionOwner<F> {
    fn admission_ready(&self) -> bool {
        self.owner
            .as_ref()
            .is_none_or(<Owner<F> as Driven>::admission_ready)
    }

    fn settle_completion(&mut self, now: u64) -> Result<(), String> {
        if let Some(owner) = self.owner.as_mut() {
            <Owner<F> as Driven>::settle_completion(owner, now)?;
        }
        self.shed_unloaded_owner();
        Ok(())
    }

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
            return Ok(match command {
                WorkerCommand::Admit(_) => {
                    WorkerReply::AdmissionRefused(AdmissionError::ModelUnloaded {
                        cause: ModelUnloadCause::MemoryPressure,
                    })
                }
                WorkerCommand::Status { .. } => WorkerReply::Status(None),
                WorkerCommand::Cancel { .. } | WorkerCommand::Stop { .. } => {
                    WorkerReply::Acknowledged
                }
                // Nothing is resident; the session reports the unload.
                WorkerCommand::Observe => WorkerReply::Acknowledged,
                WorkerCommand::Close => {
                    return Err("worker lifecycle command reached the execution domain".into())
                }
            });
        };
        match command {
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
            WorkerCommand::Status { request } => {
                Ok(WorkerReply::Status(Self::snapshot(owner, request)))
            }
            WorkerCommand::Observe => Self::observe(owner),
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
        self.notice.record(UnloadCause::Internal {
            reason: error.to_owned(),
        });
    }
    fn failure(&self) -> Option<&RequestError> {
        let failure = self.owner.as_ref().and_then(<Owner<F> as Driven>::failure);
        if let Some(error) = failure {
            self.notice.record(UnloadCause::from(error));
        }
        failure
    }
    fn shutdown(&mut self) -> Result<bool, String> {
        match self.owner.as_mut() {
            Some(owner) => <Owner<F> as Driven>::shutdown(owner),
            None => Ok(true),
        }
    }
}
