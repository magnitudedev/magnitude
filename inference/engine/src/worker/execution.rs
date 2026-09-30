//! The worker's thread-confined execution owner over the family-neutral
//! scheduler. Only the closed scheduler command vocabulary reaches it.

use crate::error::UnloadCause;
use magnitude_executor::ProgramFamily;
use magnitude_generation::Method;
use magnitude_scheduler::{
    owner::{AdmissionError, Owner, Ran},
    protocol::{AdmitRequest, WorkerCommand, WorkerReply},
    publication::{ModelUnloadCause, RequestError},
    worker::{Driven, Event},
};
use std::sync::Arc;

/// The execution lifecycle. Every transition moves the owner by value.
enum Execution<F: ProgramFamily> {
    Serving(Owner<F>),
    /// Every request has been terminated with the cause; the in-flight group
    /// finishes, then `Stopped`.
    Draining(Owner<F>, UnloadCause),
    Stopped(UnloadCause),
}

pub(crate) struct ExecutionOwner<F: ProgramFamily> {
    execution: Execution<F>,
    method: Arc<dyn Method>,
}

impl<F: ProgramFamily> ExecutionOwner<F> {
    pub fn new(owner: Owner<F>, method: Arc<dyn Method>) -> Self {
        Self {
            execution: Execution::Serving(owner),
            method,
        }
    }

    /// Move the execution out for a transition. The placeholder is truthful:
    /// a transition that panics has stopped execution.
    fn take(&mut self) -> Execution<F> {
        std::mem::replace(
            &mut self.execution,
            Execution::Stopped(UnloadCause::Internal {
                reason: "execution transition interrupted".into(),
            }),
        )
    }
}

/// The admission refusal of an execution that is draining or stopped.
fn unloaded(cause: &UnloadCause) -> AdmissionError {
    match cause {
        UnloadCause::MemoryPressure => AdmissionError::ModelUnloaded {
            cause: ModelUnloadCause::MemoryPressure,
        },
        UnloadCause::Shutdown | UnloadCause::DeviceLost { .. } | UnloadCause::Internal { .. } => {
            AdmissionError::Refused(RequestError::WorkerClosed)
        }
    }
}

impl<F: ProgramFamily> Driven for ExecutionOwner<F> {
    type Stopped = UnloadCause;

    fn command(&mut self, command: WorkerCommand, now: u64) -> Result<WorkerReply, String> {
        match command {
            WorkerCommand::Admit(AdmitRequest {
                seed,
                input,
                prefix_cache,
                output_capacity,
            }) => match &mut self.execution {
                Execution::Serving(owner) => {
                    let generation = seed.into_generation(self.method.clone())?;
                    Ok(
                        match owner.admit(generation, input, prefix_cache, output_capacity, now) {
                            Ok((request, receiver)) => WorkerReply::Admitted { request, receiver },
                            Err(error) => WorkerReply::AdmissionRefused(error),
                        },
                    )
                }
                Execution::Draining(_, cause) | Execution::Stopped(cause) => {
                    Ok(WorkerReply::AdmissionRefused(unloaded(cause)))
                }
            },
            WorkerCommand::Stop { request } => {
                if let Execution::Serving(owner) | Execution::Draining(owner, _) =
                    &mut self.execution
                {
                    owner.stop(request);
                }
                Ok(WorkerReply::Acknowledged)
            }
            WorkerCommand::Status { request } => Ok(WorkerReply::Status(match &self.execution {
                Execution::Serving(owner) | Execution::Draining(owner, _) => {
                    owner.snapshot(request)
                }
                Execution::Stopped(_) => None,
            })),
            WorkerCommand::Observe => match &self.execution {
                Execution::Serving(owner) | Execution::Draining(owner, _) => owner.observe(),
                Execution::Stopped(cause) => Err(format!("execution stopped: {cause}")),
            },
        }
    }

    fn event(&mut self, event: Event, _: u64) {
        match event {
            // The owner reconciles its flight when it next runs.
            Event::Completion => {}
            Event::Publication(request, wake) => match &mut self.execution {
                Execution::Serving(owner) | Execution::Draining(owner, _) => {
                    owner.publication_wake(request, wake)
                }
                Execution::Stopped(_) => {}
            },
            Event::Cancel(request) => match &mut self.execution {
                Execution::Serving(owner) | Execution::Draining(owner, _) => owner.cancel(request),
                Execution::Stopped(_) => {}
            },
            Event::Close => {
                self.execution = match self.take() {
                    Execution::Serving(mut owner) => {
                        owner.terminate_all(RequestError::WorkerClosed);
                        Execution::Draining(owner, UnloadCause::Shutdown)
                    }
                    draining @ Execution::Draining(..) => draining,
                    stopped @ Execution::Stopped(_) => stopped,
                }
            }
        }
    }

    fn run(&mut self, now: u64) -> Result<u64, UnloadCause> {
        loop {
            match self.take() {
                Execution::Serving(owner) => match owner.run(now) {
                    Ok((owner, Ran { unload: true, .. })) => {
                        self.execution = Execution::Draining(owner, UnloadCause::MemoryPressure)
                    }
                    Ok((owner, Ran { due, .. })) => {
                        self.execution = Execution::Serving(owner);
                        return Ok(due);
                    }
                    Err(error) => self.execution = Execution::Stopped(UnloadCause::from(&error)),
                },
                Execution::Draining(owner, cause) => match owner.run(now) {
                    Ok((_, Ran { in_flight: false, .. })) => {
                        self.execution = Execution::Stopped(cause)
                    }
                    Ok((owner, Ran { due, .. })) => {
                        self.execution = Execution::Draining(owner, cause);
                        return Ok(due);
                    }
                    Err(error) => self.execution = Execution::Stopped(UnloadCause::from(&error)),
                },
                Execution::Stopped(cause) => {
                    self.execution = Execution::Stopped(cause.clone());
                    return Err(cause);
                }
            }
        }
    }
}
