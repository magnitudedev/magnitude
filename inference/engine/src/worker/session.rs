//! The worker side of one host connection.
//!
//! Three threads: a reader decodes host messages into the session inbox; the
//! execution thread owns the device, scheduler and publication queues; the
//! session thread binds admissions (constraint matcher, retention key),
//! dispatches them to the execution owner and forwards each request's
//! publications while the host holds credit for them.

use super::execution::UnloadNotice;
use super::protocol::{
    Admission, DomainHeadroom, ExecutionTimings, HostMessage, HostRequestId, MemoryObservation,
    RequestProgress, RequestState, RetentionPolicy, WorkerMessage,
};
use magnitude_executor::platform::DomainRole;
use super::transport::{MessageReceiver, MessageSender, TransportError};
use super::WorkerExit;
use crate::census::{AllocationCensus, MemoryDomain};
use crate::error::{RequestError, UnloadCause};
use magnitude_chat::Vocabulary;
use magnitude_executor::RequestId;
use magnitude_family_contracts::ModelDefinition;
use magnitude_scheduler::{
    owner::Status,
    protocol::{AdmitRequest, RequestSnapshot, WorkerCommand, WorkerReply},
    publication::{Publication, PublicationReceiver},
    retention::{RetentionKey, RetentionRequest},
    worker::{Call, Client, Worker},
};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    future::Future,
    pin::Pin,
    sync::{Arc, Condvar, Mutex},
    task::{Context, Poll, Wake, Waker},
};

pub(crate) type Outbound = Arc<Mutex<Box<dyn MessageSender<WorkerMessage>>>>;

enum Inbound {
    Message(HostMessage),
    Disconnected(Option<TransportError>),
}

/// The session's single wake source: inbound messages, execution replies,
/// publication readiness and unload notices all land here.
#[derive(Default)]
pub(crate) struct Signal {
    state: Mutex<SignalState>,
    changed: Condvar,
}

#[derive(Default)]
struct SignalState {
    inbox: VecDeque<Inbound>,
    woken: bool,
}

impl Signal {
    fn push(&self, inbound: Inbound) {
        self.state.lock().unwrap().inbox.push_back(inbound);
        self.changed.notify_one();
    }

    pub fn notify(&self) {
        self.state.lock().unwrap().woken = true;
        self.changed.notify_one();
    }

    fn wait(&self) -> VecDeque<Inbound> {
        let mut state = self.state.lock().unwrap();
        while state.inbox.is_empty() && !state.woken {
            state = self.changed.wait(state).unwrap();
        }
        state.woken = false;
        std::mem::take(&mut state.inbox)
    }
}

impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.notify();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.notify();
    }
}

/// Decode host messages until the stream ends.
pub(crate) fn spawn_reader(
    mut receiver: impl MessageReceiver<HostMessage>,
    signal: Arc<Signal>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("magnitude-worker-reader".into())
        .spawn(move || loop {
            match receiver.receive() {
                Ok(Some(message)) => signal.push(Inbound::Message(message)),
                Ok(None) => {
                    signal.push(Inbound::Disconnected(None));
                    return;
                }
                Err(error) => {
                    signal.push(Inbound::Disconnected(Some(error)));
                    return;
                }
            }
        })
        .expect("spawn the worker reader thread")
}

struct PendingAdmission {
    id: HostRequestId,
    call: Call,
    output_capacity: usize,
    stop: bool,
    cancelled: bool,
}

struct Stream {
    request: RequestId,
    receiver: PublicationReceiver,
    credit: u64,
}

/// What readiness fixed about the loaded model that the session needs.
pub(crate) struct Loaded {
    pub execution: Worker,
    pub vocabulary: Vocabulary,
    pub retention: RetentionKey,
    pub definition: ModelDefinition,
    pub compute_bytes: u64,
    pub domain: MemoryDomain,
    pub notice: Arc<UnloadNotice>,
}

pub(crate) struct Session {
    outbound: Outbound,
    signal: Arc<Signal>,
    waker: Waker,
    loaded: Loaded,
    client: Client,
    admissions: Vec<PendingAdmission>,
    streams: BTreeMap<HostRequestId, Stream>,
    stops: Vec<Call>,
    statuses: Vec<(HostRequestId, Call)>,
    observations: Vec<Call>,
    /// The model stopped serving; the session ends once every open request
    /// has delivered its terminal outcome.
    unloading: Option<UnloadCause>,
}

enum Flow {
    Continue,
    Exit(WorkerExit),
}

impl Session {
    pub fn new(outbound: Outbound, signal: Arc<Signal>, loaded: Loaded) -> Self {
        let waker = Waker::from(signal.clone());
        let client = loaded.execution.client();
        Self {
            outbound,
            signal,
            waker,
            loaded,
            client,
            admissions: Vec::new(),
            streams: BTreeMap::new(),
            stops: Vec::new(),
            statuses: Vec::new(),
            observations: Vec::new(),
            unloading: None,
        }
    }

    fn send(&self, message: WorkerMessage) -> Result<(), TransportError> {
        self.outbound.lock().unwrap().send(message)
    }

    /// Serve the connection until shutdown, host loss or unload.
    pub fn run(mut self) -> WorkerExit {
        loop {
            let inbound = self.signal.wait();
            for inbound in inbound {
                let flow = match inbound {
                    Inbound::Message(message) => self.handle(message),
                    Inbound::Disconnected(error) => Ok(Flow::Exit(WorkerExit::HostLost {
                        reason: error.map(|error| error.to_string()),
                    })),
                };
                match flow {
                    Ok(Flow::Continue) => {}
                    Ok(Flow::Exit(exit)) => return self.finish(exit),
                    Err(error) => return self.finish(WorkerExit::TransportFailed(error.to_string())),
                }
            }
            match self.progress() {
                Ok(Flow::Continue) => {}
                Ok(Flow::Exit(exit)) => return self.finish(exit),
                Err(error) => return self.finish(WorkerExit::TransportFailed(error.to_string())),
            }
        }
    }

    fn finish(mut self, exit: WorkerExit) -> WorkerExit {
        // Dropping receivers cancels their requests; closing the execution
        // owner then releases the device.
        self.admissions.clear();
        self.streams.clear();
        self.stops.clear();
        self.statuses.clear();
        self.observations.clear();
        self.loaded.execution.close();
        exit
    }

    fn dispatch(&self, command: WorkerCommand) -> Result<Call, String> {
        self.client.dispatch(command)
    }

    /// The cause a failed execution command reports.
    fn unload_cause(&self, reason: String) -> UnloadCause {
        self.loaded
            .notice
            .cause()
            .unwrap_or(UnloadCause::Internal { reason })
    }

    fn handle(&mut self, message: HostMessage) -> Result<Flow, TransportError> {
        match message {
            HostMessage::Hello { .. } | HostMessage::Load { .. } => {
                return Ok(Flow::Exit(WorkerExit::ProtocolViolation(
                    "handshake message after readiness".into(),
                )))
            }
            HostMessage::Admit(admission) => self.admit(admission)?,
            HostMessage::Credit {
                request_id,
                batches,
            } => {
                if let Some(stream) = self.streams.get_mut(&request_id) {
                    stream.credit = stream.credit.saturating_add(u64::from(batches));
                }
            }
            HostMessage::Stop { request_id } => {
                if let Some(pending) = self
                    .admissions
                    .iter_mut()
                    .find(|pending| pending.id == request_id)
                {
                    pending.stop = true;
                } else if let Some(stream) = self.streams.get(&request_id) {
                    match self.dispatch(WorkerCommand::Stop {
                        request: stream.request,
                    }) {
                        Ok(call) => self.stops.push(call),
                        Err(reason) => self.fail_stream(request_id, reason)?,
                    }
                }
            }
            HostMessage::Cancel { request_id } => {
                if let Some(pending) = self
                    .admissions
                    .iter_mut()
                    .find(|pending| pending.id == request_id)
                {
                    pending.cancelled = true;
                }
                // Dropping the receiver wakes the owner to release the request.
                self.streams.remove(&request_id);
            }
            HostMessage::Status { request_id } => match self.streams.get(&request_id) {
                Some(stream) => match self.dispatch(WorkerCommand::Status {
                    request: stream.request,
                }) {
                    Ok(call) => self.statuses.push((request_id, call)),
                    Err(_) => self.send(WorkerMessage::Status {
                        request_id,
                        progress: None,
                    })?,
                },
                None => self.send(WorkerMessage::Status {
                    request_id,
                    progress: None,
                })?,
            },
            HostMessage::Observe => {
                if let Ok(call) = self.dispatch(WorkerCommand::Observe) {
                    self.observations.push(call);
                }
            }
            HostMessage::Shutdown => {
                self.send(WorkerMessage::Unloaded {
                    cause: UnloadCause::Shutdown,
                })?;
                return Ok(Flow::Exit(WorkerExit::Shutdown));
            }
        }
        Ok(Flow::Continue)
    }

    fn admit(&mut self, admission: Admission) -> Result<(), TransportError> {
        let id = admission.request_id;
        match self.bind(admission) {
            Ok((command, output_capacity)) => match self.dispatch(command) {
                Ok(call) => self.admissions.push(PendingAdmission {
                    id,
                    call,
                    output_capacity,
                    stop: false,
                    cancelled: false,
                }),
                Err(reason) => self.send(WorkerMessage::AdmissionRefused {
                    request_id: id,
                    error: RequestError::ModelUnloaded {
                        cause: self.unload_cause(reason),
                    },
                })?,
            },
            Err(error) => self.send(WorkerMessage::AdmissionRefused {
                request_id: id,
                error,
            })?,
        }
        Ok(())
    }

    /// Validate the decoded input against the loaded definition, instantiate
    /// the constraint with this worker's vocabulary and derive retention.
    fn bind(&mut self, admission: Admission) -> Result<(WorkerCommand, usize), RequestError> {
        if let Some(cause) = &self.unloading {
            return Err(RequestError::ModelUnloaded {
                cause: cause.clone(),
            });
        }
        let Admission {
            request_id: _,
            input,
            options,
            constraint,
            retention,
            output_capacity,
        } = admission;
        let invalid = |reason: String| RequestError::InvalidRequest { reason };
        if output_capacity == 0 {
            return Err(invalid("request output capacity must be positive".into()));
        }
        let input = input
            .validated(&self.loaded.definition)
            .map_err(|error| invalid(error.to_string()))?;
        let limit = self.loaded.definition.geometry.context_limit;
        let required = input.tokens().len() as u64;
        if required >= limit {
            return Err(RequestError::ContextLengthExceeded { required, limit });
        }
        let seed = self
            .loaded
            .vocabulary
            .prepare_generation(constraint.as_ref(), options, input.tokens(), input.layout())
            .map_err(invalid)?;
        let retention = match retention {
            RetentionPolicy::Retain => Some(
                RetentionRequest::new(self.loaded.retention.clone(), &input).map_err(invalid)?,
            ),
            RetentionPolicy::Transient => None,
        };
        Ok((
            WorkerCommand::Admit(AdmitRequest {
                seed,
                input,
                retention,
                output_capacity,
            }),
            output_capacity,
        ))
    }

    fn fail_stream(&mut self, id: HostRequestId, reason: String) -> Result<(), TransportError> {
        if self.streams.remove(&id).is_some() {
            let cause = self.unload_cause(reason);
            self.send(WorkerMessage::Failed {
                request_id: id,
                error: RequestError::ModelUnloaded { cause },
            })?;
        }
        Ok(())
    }

    fn poll_call(waker: &Waker, call: &mut Call) -> Poll<Result<WorkerReply, String>> {
        Pin::new(call).poll(&mut Context::from_waker(waker))
    }

    /// Advance every pending reply and every stream holding credit.
    fn progress(&mut self) -> Result<Flow, TransportError> {
        if self.unloading.is_none() {
            if let Some(cause) = self.loaded.notice.cause() {
                self.unloading = Some(cause);
            } else if let Some(reason) = self.client.closed_reason() {
                self.unloading = Some(UnloadCause::Internal { reason });
            }
        }
        let waker = self.waker.clone();
        let mut index = 0;
        while index < self.admissions.len() {
            let Poll::Ready(reply) = Self::poll_call(&waker, &mut self.admissions[index].call)
            else {
                index += 1;
                continue;
            };
            let pending = self.admissions.swap_remove(index);
            self.admitted(pending, reply)?;
        }
        self.stops
            .retain_mut(|call| Self::poll_call(&waker, call).is_pending());
        let mut index = 0;
        while index < self.statuses.len() {
            let Poll::Ready(reply) = Self::poll_call(&waker, &mut self.statuses[index].1) else {
                index += 1;
                continue;
            };
            let (request_id, _) = self.statuses.swap_remove(index);
            let progress = match reply {
                Ok(WorkerReply::Status(snapshot)) => snapshot.map(progress),
                _ => None,
            };
            self.send(WorkerMessage::Status {
                request_id,
                progress,
            })?;
        }
        let mut index = 0;
        while index < self.observations.len() {
            let Poll::Ready(reply) = Self::poll_call(&waker, &mut self.observations[index]) else {
                index += 1;
                continue;
            };
            self.observations.swap_remove(index);
            match reply {
                Ok(WorkerReply::Observed {
                    standing,
                    holdings,
                    readings,
                }) => match AllocationCensus::classify(
                    &standing,
                    holdings,
                    self.loaded.compute_bytes,
                    self.loaded.domain,
                ) {
                    Ok(census) => {
                        let domains = readings
                            .iter()
                            .map(|reading| DomainHeadroom {
                                domain: match reading.role {
                                    DomainRole::Allocation => self.loaded.domain,
                                    DomainRole::Staging => MemoryDomain::HostRam,
                                },
                                headroom_bytes: reading.headroom_bytes,
                            })
                            .collect();
                        self.send(WorkerMessage::Observed {
                            observation: Ok(MemoryObservation { census, domains }),
                        })?
                    }
                    Err(reason) => {
                        return Ok(Flow::Exit(WorkerExit::Unloaded(UnloadCause::Internal {
                            reason,
                        })))
                    }
                },
                // Nothing is resident: the unload notice ends the session.
                Ok(_) => {}
                Err(reason) => self.send(WorkerMessage::Observed {
                    observation: Err(RequestError::MemoryObservationUnavailable { reason }),
                })?,
            }
        }
        self.forward_streams()?;
        if let Some(cause) = &self.unloading {
            if self.streams.is_empty() && self.admissions.is_empty() {
                let cause = cause.clone();
                self.send(WorkerMessage::Unloaded {
                    cause: cause.clone(),
                })?;
                return Ok(Flow::Exit(WorkerExit::Unloaded(cause)));
            }
        }
        Ok(Flow::Continue)
    }

    fn admitted(
        &mut self,
        pending: PendingAdmission,
        reply: Result<WorkerReply, String>,
    ) -> Result<(), TransportError> {
        let refused = |error| WorkerMessage::AdmissionRefused {
            request_id: pending.id,
            error,
        };
        match reply {
            Ok(WorkerReply::Admitted { request, receiver }) => {
                if pending.cancelled {
                    // Dropping the receiver releases the request.
                    return Ok(());
                }
                self.streams.insert(
                    pending.id,
                    Stream {
                        request,
                        receiver,
                        credit: pending.output_capacity as u64,
                    },
                );
                self.send(WorkerMessage::Admitted {
                    request_id: pending.id,
                })?;
                if pending.stop {
                    match self.dispatch(WorkerCommand::Stop { request }) {
                        Ok(call) => self.stops.push(call),
                        Err(reason) => self.fail_stream(pending.id, reason)?,
                    }
                }
                Ok(())
            }
            Ok(WorkerReply::AdmissionRefused(error)) => self.send(refused(error.into())),
            Ok(_) => self.send(refused(RequestError::internal(
                "execution owner returned an invalid admission reply",
            ))),
            Err(reason) => {
                let cause = self.unload_cause(reason);
                self.send(refused(RequestError::ModelUnloaded { cause }))
            }
        }
    }

    /// Forward publications of every stream the host has credit for. The
    /// host drains its buffer before a request's terminal outcome, so a
    /// terminal behind exhausted credit arrives with the next credit.
    fn forward_streams(&mut self) -> Result<(), TransportError> {
        let mut context = Context::from_waker(&self.waker);
        let mut finished = BTreeSet::new();
        let mut messages = Vec::new();
        for (&id, stream) in &mut self.streams {
            while stream.credit > 0 {
                match stream.receiver.poll_next(&mut context) {
                    Poll::Pending => break,
                    Poll::Ready(Some(Publication::Output(tokens))) => {
                        stream.credit -= 1;
                        messages.push(WorkerMessage::Output {
                            request_id: id,
                            tokens,
                        });
                    }
                    Poll::Ready(Some(Publication::Completed {
                        finish,
                        usage,
                        method,
                        timings,
                    })) => {
                        messages.push(WorkerMessage::Completed {
                            request_id: id,
                            finish,
                            usage,
                            method,
                            timings: ExecutionTimings {
                                prompt_ns: timings.prompt_ns,
                                predicted_ns: timings.predicted_ns,
                            },
                        });
                        finished.insert(id);
                        break;
                    }
                    Poll::Ready(Some(Publication::Failed(error))) => {
                        messages.push(WorkerMessage::Failed {
                            request_id: id,
                            error: error.into(),
                        });
                        finished.insert(id);
                        break;
                    }
                    Poll::Ready(None) => {
                        messages.push(WorkerMessage::Failed {
                            request_id: id,
                            error: RequestError::internal(
                                "publication stream ended without a terminal outcome",
                            ),
                        });
                        finished.insert(id);
                        break;
                    }
                }
            }
        }
        for id in finished {
            self.streams.remove(&id);
        }
        let mut outbound = self.outbound.lock().unwrap();
        for message in messages {
            outbound.send(message)?;
        }
        Ok(())
    }
}

fn progress(snapshot: RequestSnapshot) -> RequestProgress {
    RequestProgress {
        state: match snapshot.status {
            Status::Runnable => RequestState::Runnable,
            Status::AwaitingCompletion => RequestState::AwaitingCompletion,
            Status::OutputBlocked => RequestState::OutputBlocked,
            Status::Preempted => RequestState::Preempted,
            Status::CapacityBlocked {
                required,
                available,
            } => RequestState::CapacityBlocked {
                required,
                available,
            },
            Status::Terminal(finish) => RequestState::Finished(finish),
        },
        prompt_tokens: snapshot.prompt_tokens,
        cached_tokens: snapshot.cached_tokens,
        resident_tokens: snapshot.resident_position,
        output_tokens: snapshot.output_tokens,
    }
}
