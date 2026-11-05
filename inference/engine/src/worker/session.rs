//! The worker side of one host connection.
//!
//! Three threads: a reader decodes host messages into the session inbox; the
//! execution thread owns the device, scheduler and publication queues; the
//! session thread binds admissions (constraint matcher, prefix-cache opt-in),
//! dispatches them to the execution owner and forwards each request's
//! publications while the host holds credit for them.

use super::binding::GenerationBinding;
use super::protocol::{
    Admission, ExecutionTimings, HostMessage, HostRequestId, MemoryObservation, RequestProgress,
    RequestState, RoomClaimId, StoreHeadroom, WorkerMessage,
};
use super::transport::{MessageReceiver, MessageSender, TransportError};
use super::WorkerExit;
use crate::census::AllocationCensus;
use crate::error::{RequestError, UnloadCause};
use magnitude_executor::RequestId;
use magnitude_family_contracts::ModelDefinition;
use magnitude_scheduler::{
    owner::Status,
    protocol::{AdmitRequest, RequestSnapshot, RoomAnswer, RoomGrant, WorkerCommand, WorkerReply},
    publication::{Publication, PublicationReceiver},
    worker::{Call, CallError, Client, DispatchError, Worker},
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
/// publication readiness and the execution stopping all land here.
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

    fn notify(&self) {
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
    call: Call<UnloadCause>,
    output_capacity: usize,
    stop: bool,
    cancelled: bool,
}

/// The measurement room's claim: dispatched to the execution owner, then
/// held there until it answers.
enum RoomClaim {
    Dispatched(Call<UnloadCause>),
    Claimed(RoomGrant),
}

struct Stream {
    request: RequestId,
    receiver: PublicationReceiver,
    credit: u64,
    /// Its host was answered: admitted once it began, or refused for what
    /// it published before ([`Stream::answer`]).
    answered: bool,
}

impl Stream {
    /// Its host's answer, once it can be given: admitted once the request
    /// began, or once it published anything else first (`read`: what it
    /// published, `None` once its stream ended; `None` before a
    /// publication is read); refused when it failed before it began.
    fn answer(&mut self, id: HostRequestId, read: Option<Option<&Publication>>) -> Option<WorkerMessage> {
        if self.answered {
            return None;
        }
        // The request begins before anything it publishes, so this reads
        // whether it began before what was read.
        let answer = match (self.receiver.began(), read) {
            (true, _) => WorkerMessage::Admitted { request_id: id },
            (false, None) => return None,
            (false, Some(Some(Publication::Failed(error)))) => WorkerMessage::AdmissionRefused {
                request_id: id,
                error: error.clone().into(),
            },
            (false, Some(_)) => WorkerMessage::Admitted { request_id: id },
        };
        self.answered = true;
        Some(answer)
    }
}

/// What readiness fixed about the loaded model that the session needs.
pub(crate) struct Loaded {
    pub execution: Worker<UnloadCause>,
    pub binding: GenerationBinding,
    pub definition: ModelDefinition,
    /// The model's host-resident tables' bytes, which the census reports in
    /// host RAM (`AllocationCensus::classify`).
    pub host_table_bytes: u64,
}

pub(crate) struct Session {
    outbound: Outbound,
    signal: Arc<Signal>,
    waker: Waker,
    loaded: Loaded,
    client: Client<UnloadCause>,
    admissions: Vec<PendingAdmission>,
    streams: BTreeMap<HostRequestId, Stream>,
    /// Commands the host awaits no reply to (stops, the measurement room's
    /// release, repricing): only a refusal is reported.
    unanswered: Vec<Call<UnloadCause>>,
    /// The measurement room's claim, until its answer is sent.
    room: Option<(RoomClaimId, RoomClaim)>,
    statuses: Vec<(HostRequestId, Call<UnloadCause>)>,
    observations: Vec<Call<UnloadCause>>,
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
        // Watch for execution stopping from the start: the session sleeps
        // until a signal arrives.
        let unloading = client.poll_stopped(&waker);
        Self {
            outbound,
            signal,
            waker,
            loaded,
            client,
            admissions: Vec::new(),
            streams: BTreeMap::new(),
            unanswered: Vec::new(),
            room: None,
            statuses: Vec::new(),
            observations: Vec::new(),
            unloading,
        }
    }

    fn send(&self, message: WorkerMessage) -> Result<(), TransportError> {
        self.outbound.lock().unwrap().send(message)
    }

    /// Serve the connection until shutdown, host loss or unload; then the
    /// load, stopped, its durable effects made, still holding the device.
    pub fn run(mut self) -> (WorkerExit, Loaded) {
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

    /// Stop the load: dropping receivers cancels their requests, then
    /// execution stops with every durable effect of the load made
    /// (`magnitude_executor::settle`). Only then is a shutdown answered, so
    /// its answer means nothing remains but releasing what the load holds.
    fn finish(mut self, exit: WorkerExit) -> (WorkerExit, Loaded) {
        self.admissions.clear();
        self.streams.clear();
        self.unanswered.clear();
        self.room = None;
        self.statuses.clear();
        self.observations.clear();
        self.loaded.execution.stop();
        let exit = match exit {
            WorkerExit::Shutdown => match self.send(WorkerMessage::Unloaded { cause: UnloadCause::Shutdown }) {
                Ok(()) => WorkerExit::Shutdown,
                Err(error) => WorkerExit::TransportFailed(error.to_string()),
            },
            exit => exit,
        };
        (exit, self.loaded)
    }

    fn dispatch(
        &self,
        command: WorkerCommand,
    ) -> Result<Call<UnloadCause>, DispatchError<UnloadCause>> {
        self.client.dispatch(command)
    }

    fn handle(&mut self, message: HostMessage) -> Result<Flow, TransportError> {
        match message {
            HostMessage::Hello { .. }
            | HostMessage::Load { .. }
            | HostMessage::Prepare { .. }
            | HostMessage::Measure { .. } => {
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
                        Ok(call) => self.unanswered.push(call),
                        Err(error) => self.fail_stream(request_id, error)?,
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
            HostMessage::ClaimMeasurementRoom { claim, bytes } => {
                match self.dispatch(WorkerCommand::ClaimMeasurementRoom { bytes }) {
                    Ok(call) => self.room = Some((claim, RoomClaim::Dispatched(call))),
                    Err(error) => self.send(WorkerMessage::MeasurementRoom {
                        claim,
                        room: Err(undispatched(error)),
                    })?,
                }
            }
            HostMessage::ReleaseMeasurementRoom => {
                // A claim not yet answered is withdrawn with it.
                self.room = None;
                self.unanswer(WorkerCommand::ReleaseMeasurementRoom);
            }
            HostMessage::Reprice => self.unanswer(WorkerCommand::Reprice),
            // Answered once the load has stopped (`Session::finish`).
            HostMessage::Shutdown => return Ok(Flow::Exit(WorkerExit::Shutdown)),
        }
        Ok(Flow::Continue)
    }

    /// Dispatch a command the host awaits no reply to; one that cannot be
    /// dispatched is reported.
    fn unanswer(&mut self, command: WorkerCommand) {
        match self.dispatch(command) {
            Ok(call) => self.unanswered.push(call),
            Err(error) => tracing::warn!(
                "a command could not reach the execution owner: {}",
                undispatched(error)
            ),
        }
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
                Err(error) => self.send(WorkerMessage::AdmissionRefused {
                    request_id: id,
                    error: undispatched(error),
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
    /// the constraint with this worker's vocabulary and bind the prefix-cache
    /// opt-in.
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
        let limit = self.loaded.definition.decoder.context_limit;
        let required = input.tokens().len() as u64;
        if required >= limit {
            return Err(RequestError::ContextLengthExceeded { required, limit });
        }
        let seed = self
            .loaded
            .binding
            .seed(constraint.as_ref(), options, input.tokens(), input.layout())
            .map_err(invalid)?;
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

    fn fail_stream(
        &mut self,
        id: HostRequestId,
        error: DispatchError<UnloadCause>,
    ) -> Result<(), TransportError> {
        if self.streams.remove(&id).is_some() {
            self.send(WorkerMessage::Failed {
                request_id: id,
                error: undispatched(error),
            })?;
        }
        Ok(())
    }

    fn poll_call(
        waker: &Waker,
        call: &mut Call<UnloadCause>,
    ) -> Poll<Result<WorkerReply, CallError<UnloadCause>>> {
        Pin::new(call).poll(&mut Context::from_waker(waker))
    }

    /// Advance every pending reply and every stream holding credit.
    fn progress(&mut self) -> Result<Flow, TransportError> {
        if self.unloading.is_none() {
            self.unloading = self.client.poll_stopped(&self.waker);
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
        self.unanswered.retain_mut(|call| match Self::poll_call(&waker, call) {
            Poll::Pending => true,
            Poll::Ready(Err(CallError::Refused(reason))) => {
                tracing::warn!("the execution owner refused a command: {reason}");
                false
            }
            Poll::Ready(_) => false,
        });
        if let Some((id, claim)) = &mut self.room {
            let mut answer = None;
            if let RoomClaim::Dispatched(call) = claim {
                if let Poll::Ready(reply) = Self::poll_call(&waker, call) {
                    match reply {
                        Ok(WorkerReply::MeasurementRoom(grant)) => *claim = RoomClaim::Claimed(grant),
                        Ok(_) => unreachable!("the execution owner answers a room claim with its grant"),
                        Err(CallError::Refused(reason)) => {
                            unreachable!("the execution owner refuses no room claim: {reason}")
                        }
                        Err(CallError::Stopped(cause)) => answer = Some(Err(RequestError::ModelUnloaded { cause })),
                    }
                }
            }
            if let RoomClaim::Claimed(grant) = claim {
                if let Poll::Ready(room) = grant.poll_answer(&mut Context::from_waker(&waker)) {
                    answer = Some(Ok(room));
                }
            }
            match answer {
                // The session drops a claim the host released or replaced, so
                // one the owner released is released by its ending, which the
                // host learns from `Unloaded`.
                Some(Ok(RoomAnswer::Released)) => self.room = None,
                Some(room) => {
                    let claim = *id;
                    self.room = None;
                    self.send(WorkerMessage::MeasurementRoom { claim, room })?;
                }
                None => {}
            }
        }
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
                    reconciliation,
                    readings,
                    serving,
                }) => match AllocationCensus::classify(
                    &reconciliation,
                    self.loaded.host_table_bytes,
                    serving,
                ) {
                    Ok(census) => {
                        let stores = readings
                            .iter()
                            .map(|reading| StoreHeadroom {
                                store: reading.identity.clone(),
                                headroom_bytes: reading.headroom_bytes,
                            })
                            .collect();
                        self.send(WorkerMessage::Observed {
                            observation: Ok(MemoryObservation { census, stores }),
                        })?
                    }
                    Err(reason) => {
                        return Ok(Flow::Exit(WorkerExit::Unloaded(UnloadCause::Internal {
                            reason,
                        })))
                    }
                },
                Ok(_) => self.send(WorkerMessage::Observed {
                    observation: Err(RequestError::internal(
                        "execution owner returned an invalid observation reply",
                    )),
                })?,
                Err(CallError::Refused(reason)) => self.send(WorkerMessage::Observed {
                    observation: Err(RequestError::MemoryObservationUnavailable { reason }),
                })?,
                Err(CallError::Stopped(cause)) => self.send(WorkerMessage::Observed {
                    observation: Err(RequestError::ModelUnloaded { cause }),
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
        reply: Result<WorkerReply, CallError<UnloadCause>>,
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
                // Its host is answered once it begins.
                self.streams.insert(
                    pending.id,
                    Stream {
                        request,
                        receiver,
                        credit: pending.output_capacity as u64,
                        answered: false,
                    },
                );
                if pending.stop {
                    match self.dispatch(WorkerCommand::Stop { request }) {
                        Ok(call) => self.unanswered.push(call),
                        Err(error) => self.fail_stream(pending.id, error)?,
                    }
                }
                Ok(())
            }
            Ok(WorkerReply::AdmissionRefused(error)) => self.send(refused(error.into())),
            Ok(_) => self.send(refused(RequestError::internal(
                "execution owner returned an invalid admission reply",
            ))),
            Err(CallError::Refused(reason)) => self.send(refused(RequestError::internal(reason))),
            Err(CallError::Stopped(cause)) => {
                self.send(refused(RequestError::ModelUnloaded { cause }))
            }
        }
    }

    /// Answer each stream's host once it can be, and forward publications
    /// of every stream the host has credit for. A request is answered
    /// admitted once it began (became resident), or once it published
    /// anything else first; one that failed before it began is refused, so
    /// a request is never answered admitted and then refused for what kept
    /// it from beginning. The host drains its buffer before a request's
    /// terminal outcome, so a terminal behind exhausted credit arrives with
    /// the next credit.
    fn forward_streams(&mut self) -> Result<(), TransportError> {
        let mut context = Context::from_waker(&self.waker);
        let mut finished = BTreeSet::new();
        let mut messages = Vec::new();
        for (&id, stream) in &mut self.streams {
            // Read before the publication below: the request begins before
            // anything it publishes.
            messages.extend(stream.answer(id, None));
            while stream.credit > 0 {
                let publication = stream.receiver.poll_next(&mut context);
                if let Poll::Ready(publication) = &publication {
                    let answer = stream.answer(id, Some(publication.as_ref()));
                    if let Some(refused @ WorkerMessage::AdmissionRefused { .. }) = answer {
                        messages.push(refused);
                        finished.insert(id);
                        break;
                    }
                    messages.extend(answer);
                }
                match publication {
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

/// Why a command the session could not dispatch fails its request.
fn undispatched(error: DispatchError<UnloadCause>) -> RequestError {
    match error {
        DispatchError::Stopped(cause) => RequestError::ModelUnloaded { cause },
        DispatchError::Full => RequestError::Overloaded,
    }
}

fn progress(snapshot: RequestSnapshot) -> RequestProgress {
    RequestProgress {
        state: match snapshot.status {
            Status::Runnable => RequestState::Runnable,
            Status::AwaitingCompletion => RequestState::AwaitingCompletion,
            Status::OutputBlocked => RequestState::OutputBlocked,
            Status::Preempted => RequestState::Preempted,
            Status::MemoryBlocked => RequestState::MemoryBlocked,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::{BpeConfig, ByteBpeTokenizer, PieceKind, TokenId};
    use crate::worker::execution::ExecutionOwner;
    use magnitude_chat::{Normalization, PieceEncoding};
    use magnitude_executor::fixtures::{bank_backends, banked, tiny_definition};
    use magnitude_generation::Plain;
    use magnitude_scheduler::{
        owner::Owner,
        publication::{PublicationQueue, RequestError as Scheduled},
        worker::{Driven, Wakes},
        ServiceLimits,
    };
    use std::time::{Duration, Instant};

    fn stream() -> (magnitude_scheduler::publication::PublicationSender, Stream) {
        let (sender, receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
        (sender, Stream { request: RequestId(1), receiver, credit: 1, answered: false })
    }

    /// A request is answered admitted once it began, and only once; one
    /// that failed before it began is refused instead, never admitted and
    /// then failed; one that published anything else first is admitted.
    #[test]
    fn a_request_is_answered_admitted_once_it_began_and_refused_if_it_failed_before() {
        let id = HostRequestId(7);
        let (sender, mut began) = stream();
        assert!(began.answer(id, None).is_none(), "not answered before it began");
        sender.began();
        assert!(matches!(began.answer(id, None), Some(WorkerMessage::Admitted { request_id }) if request_id == id));
        assert!(began.answer(id, None).is_none(), "answered once");
        // Failed after it began: admitted already, the failure is its own.
        let failed = Publication::Failed(Scheduled::MemoryReclaim);
        assert!(began.answer(id, Some(Some(&failed))).is_none());

        let (_sender, mut refused) = stream();
        assert!(matches!(
            refused.answer(id, Some(Some(&failed))),
            Some(WorkerMessage::AdmissionRefused { error: RequestError::MemoryReclaim, .. })
        ));

        let (_sender, mut ended) = stream();
        assert!(matches!(ended.answer(id, Some(None)), Some(WorkerMessage::Admitted { .. })));
    }

    /// Records every message the session sends the host.
    struct Recorded(Arc<Mutex<Vec<WorkerMessage>>>);

    impl MessageSender<WorkerMessage> for Recorded {
        fn send(&mut self, message: WorkerMessage) -> Result<(), TransportError> {
            self.0.lock().unwrap().push(message);
            Ok(())
        }
    }

    /// The 256 byte-fallback pieces and a stop token: the least vocabulary
    /// a generation binding takes.
    fn binding() -> GenerationBinding {
        let mut pieces = (0..=255u8).map(|byte| format!("<0x{byte:02X}>")).collect::<Vec<_>>();
        pieces.push("<eos>".into());
        let mut kinds = vec![PieceKind::Byte; 256];
        kinds.push(PieceKind::Control);
        let projection = pieces.len();
        let tokenizer = ByteBpeTokenizer::new(BpeConfig {
            artifact_identity: "fixture".into(),
            pieces,
            kinds,
            merges: vec![],
            normalization: Normalization::None,
            splits: vec![],
            encoding: PieceEncoding::MetaspaceByteFallback,
            ignore_merges: false,
            implicit_bos: None,
            stop_tokens: BTreeSet::from([TokenId(256)]),
            suppressed_tokens: BTreeSet::new(),
        })
        .unwrap();
        GenerationBinding::new(Arc::new(tokenizer), projection, super::super::CONSTRAINT_CACHE).unwrap()
    }

    /// A session whose execution owner serves the controlled domain with a
    /// fake expert bank of four slots on `backend`'s device, and what it
    /// sends the host.
    fn session(backend: seismic::BackendName) -> (Session, Arc<Mutex<Vec<WorkerMessage>>>) {
        let limits = ServiceLimits {
            prefill_tokens: 2,
            decode_tokens: 2,
            decode_share: 0.5,
            locality_seconds: 1.0,
            conversations: std::num::NonZeroUsize::MIN,
        };
        let spawned = Worker::spawn(
            move |wakes| {
                let (domain, bindings) = banked(backend, 4, 4);
                let owner = Owner::new(domain, bindings, limits, Arc::new(wakes) as Arc<dyn Wakes>)?;
                Ok(Box::new(ExecutionOwner::new(owner, Arc::new(Plain), None)) as Box<dyn Driven<Stopped = UnloadCause>>)
            },
            usize::MAX,
        );
        let execution = spawned.unwrap_or_else(|error| panic!("{backend:?}: {error}"));
        let sent = Arc::new(Mutex::new(Vec::new()));
        let outbound: Outbound = Arc::new(Mutex::new(Box::new(Recorded(sent.clone()))));
        let loaded = Loaded {
            execution,
            binding: binding(),
            definition: tiny_definition(),
            host_table_bytes: 0,
        };
        (Session::new(outbound, Arc::new(Signal::default()), loaded), sent)
    }

    /// The room answers the session has sent the host, in order.
    fn answers(sent: &Mutex<Vec<WorkerMessage>>) -> Vec<(RoomClaimId, Result<RoomAnswer, RequestError>)> {
        sent.lock()
            .unwrap()
            .iter()
            .filter_map(|message| match message {
                WorkerMessage::MeasurementRoom { claim, room } => Some((*claim, room.clone())),
                _ => None,
            })
            .collect()
    }

    /// Progress the session until it has sent `count` room answers.
    fn progress_until(session: &mut Session, sent: &Mutex<Vec<WorkerMessage>>, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while answers(sent).len() < count {
            assert!(Instant::now() < deadline, "answers so far: {:?}", answers(sent));
            assert!(matches!(session.progress(), Ok(Flow::Continue)));
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// The session relays each room claim to the execution owner and the
    /// owner's answer back under the claim's id: one made is answered made;
    /// one released before its answer is withdrawn, so no answer names it,
    /// and the claim after it is answered under its own id; and one the
    /// stopped execution cannot take is answered refused at once.
    #[test]
    fn each_room_claim_is_answered_under_its_id() {
        for backend in bank_backends() {
            let (mut session, sent) = session(backend);
            let claim = |claim, bytes| HostMessage::ClaimMeasurementRoom { claim: RoomClaimId(claim), bytes };
            assert!(matches!(session.handle(claim(1, 0)), Ok(Flow::Continue)));
            progress_until(&mut session, &sent, 1);
            assert!(matches!(session.handle(claim(2, u64::MAX / 4)), Ok(Flow::Continue)));
            assert!(matches!(session.handle(HostMessage::ReleaseMeasurementRoom), Ok(Flow::Continue)));
            assert!(matches!(session.handle(claim(3, 0)), Ok(Flow::Continue)));
            progress_until(&mut session, &sent, 2);
            session.loaded.execution.close();
            assert!(matches!(session.handle(claim(4, 0)), Ok(Flow::Continue)));
            let answers = answers(&sent);
            let made = Ok(RoomAnswer::Made);
            assert_eq!(answers[..2], [(RoomClaimId(1), made.clone()), (RoomClaimId(3), made)], "{answers:?}");
            assert!(matches!(&answers[2..], [(RoomClaimId(4), Err(RequestError::ModelUnloaded { .. }))]), "{answers:?}");
        }
    }
}
