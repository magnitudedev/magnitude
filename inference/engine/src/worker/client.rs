//! The host side of a worker connection.
//!
//! One reader thread decodes worker messages into per-request state; request
//! handles consume their own events and return one batch of output credit to
//! the worker for each output batch they take. Everything is runtime-agnostic
//! (std synchronization and wakers), so any async executor can drive it.

use super::protocol::{
    Admission, EngineBuild, ExecutionTimings, HostMessage, HostRequestId,
    LoadProgress, MemoryObservation, RequestProgress, RoomClaimId, WorkerMessage,
};
use super::transport::{HostTransport, MessageReceiver, MessageSender};
use crate::error::{LoadError, RequestError};
use crate::execution::LoadAttempt;
use magnitude_executor::UnpreparedUnits;
use crate::options::{ExecutionManifest, ReadyInfo};
use magnitude_chat::ConstraintPlan;
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{DetailedUsage, FinishReason, Options, OutputToken};
use magnitude_scheduler::prefix_cache::PrefixRetention;
use magnitude_scheduler::protocol::RoomAnswer;
use std::{
    collections::{HashMap, VecDeque},
    future::poll_fn,
    sync::{Arc, Mutex},
    task::{Poll, Waker},
};

/// What a host asks the worker to generate.
pub struct RequestOptions {
    pub input: PreparedModelInput,
    pub options: Options,
    /// The constraint description; the worker instantiates the matcher.
    pub constraint: Option<ConstraintPlan>,
    pub retention: PrefixRetention,
    /// Output batches buffered for this request, on each side.
    pub output_capacity: usize,
}

/// One event of a request's ordered outcome stream. Exactly one terminal
/// event (`Completed` or `Failed`) ends it.
#[derive(Clone, Debug, PartialEq)]
pub enum RequestEvent {
    Output(Vec<OutputToken>),
    Completed {
        finish: FinishReason,
        usage: DetailedUsage,
        method: String,
        timings: ExecutionTimings,
    },
    Failed(RequestError),
}

impl RequestEvent {
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::Output(_))
    }
}

#[derive(Default)]
struct RequestSlot {
    /// `None` while admission is pending: until the request began, or was
    /// refused.
    admission: Option<Result<(), RequestError>>,
    events: VecDeque<RequestEvent>,
    terminal_received: bool,
    /// `Some` once the pending status query was answered.
    status: Option<Option<RequestProgress>>,
    waker: Option<Waker>,
}

impl RequestSlot {
    fn wake(&mut self) {
        if let Some(waker) = self.waker.take() {
            waker.wake();
        }
    }
}

/// One reply shared by every caller waiting for it.
struct SharedReply<T> {
    value: Option<Result<T, RequestError>>,
    wakers: Vec<Waker>,
}

impl<T> SharedReply<T> {
    fn fulfill(&mut self, value: Result<T, RequestError>) {
        self.value = Some(value);
        for waker in self.wakers.drain(..) {
            waker.wake();
        }
    }
}

type Shared<T> = Arc<Mutex<SharedReply<T>>>;

#[derive(Default)]
struct State {
    next_id: u64,
    requests: HashMap<HostRequestId, RequestSlot>,
    observation: Option<Shared<MemoryObservation>>,
    /// The measurement room's pending claim, and the identity the next
    /// claim takes.
    room: Option<(RoomClaimId, Shared<RoomAnswer>)>,
    next_claim: u64,
    /// Why the connection ended; every later operation fails with it.
    closed: Option<RequestError>,
    /// The host asked the worker to shut down ([`EngineClient::shutdown`]): a connection that
    /// ends after that ends for the stop, however the worker's end came (its own unload, or
    /// its kill once it overran its exit).
    shutting_down: bool,
}

impl State {
    /// Why a connection whose worker ended for `reason` ended: the host's stop, where the host
    /// asked for one, so a stopped model's requests end alike whenever its worker ends.
    fn lost(&self, reason: String) -> RequestError {
        match self.shutting_down {
            true => RequestError::ModelUnloaded { cause: crate::error::UnloadCause::Shutdown },
            false => RequestError::WorkerLost { reason },
        }
    }

    fn close(&mut self, error: RequestError) {
        if self.closed.is_some() {
            return;
        }
        for slot in self.requests.values_mut() {
            if slot.admission.is_none() {
                slot.admission = Some(Err(error.clone()));
            } else if !slot.terminal_received {
                slot.events.push_back(RequestEvent::Failed(error.clone()));
                slot.terminal_received = true;
            }
            slot.status.get_or_insert(None);
            slot.wake();
        }
        if let Some(reply) = self.observation.take() {
            reply.lock().unwrap().fulfill(Err(error.clone()));
        }
        if let Some((_, reply)) = self.room.take() {
            reply.lock().unwrap().fulfill(Err(error.clone()));
        }
        self.closed = Some(error);
    }

    /// The pending measurement room claim's reply, which a claim while one
    /// is pending joins; with the identity of a new claim, which its host
    /// sends.
    fn claim_room(&mut self) -> (Shared<RoomAnswer>, Option<RoomClaimId>) {
        if let Some((_, reply)) = &self.room {
            return (reply.clone(), None);
        }
        let claim = RoomClaimId(self.next_claim);
        self.next_claim += 1;
        let reply = Arc::new(Mutex::new(SharedReply { value: None, wakers: Vec::new() }));
        self.room = Some((claim, reply.clone()));
        (reply, Some(claim))
    }

    /// Withdraw the pending measurement room claim: it is answered as
    /// released.
    fn release_room(&mut self) {
        if let Some((_, reply)) = self.room.take() {
            reply.lock().unwrap().fulfill(Ok(RoomAnswer::Released));
        }
    }

    fn deliver(&mut self, message: WorkerMessage) -> Result<(), String> {
        match message {
            WorkerMessage::Admitted { request_id } => {
                if let Some(slot) = self.requests.get_mut(&request_id) {
                    slot.admission = Some(Ok(()));
                    slot.wake();
                }
            }
            WorkerMessage::AdmissionRefused { request_id, error } => {
                if let Some(slot) = self.requests.get_mut(&request_id) {
                    slot.admission = Some(Err(error));
                    slot.wake();
                }
            }
            WorkerMessage::Output { request_id, tokens } => {
                self.event(request_id, RequestEvent::Output(tokens))?
            }
            WorkerMessage::Completed {
                request_id,
                finish,
                usage,
                method,
                timings,
            } => self.event(
                request_id,
                RequestEvent::Completed {
                    finish,
                    usage,
                    method,
                    timings,
                },
            )?,
            WorkerMessage::Failed { request_id, error } => {
                self.event(request_id, RequestEvent::Failed(error))?
            }
            WorkerMessage::Status {
                request_id,
                progress,
            } => {
                if let Some(slot) = self.requests.get_mut(&request_id) {
                    slot.status = Some(progress);
                    slot.wake();
                }
            }
            WorkerMessage::Observed { observation } => {
                let reply = self
                    .observation
                    .take()
                    .ok_or("observation reply without a query")?;
                reply.lock().unwrap().fulfill(observation);
            }
            WorkerMessage::MeasurementRoom { claim, room } => {
                // A claim released before its answer has no one waiting, and
                // its answer answers no later claim.
                if self.room.as_ref().is_some_and(|(pending, _)| *pending == claim) {
                    let (_, reply) = self.room.take().expect("the pending claim");
                    reply.lock().unwrap().fulfill(room);
                }
            }
            WorkerMessage::Unloaded { cause } => {
                self.close(RequestError::ModelUnloaded { cause });
            }
            WorkerMessage::Hello { .. }
            | WorkerMessage::LoadProgress { .. }
            | WorkerMessage::Ready { .. }
            | WorkerMessage::Prepared
            | WorkerMessage::Unprepared
            | WorkerMessage::LoadFailed { .. } => return Err("load message after readiness".into()),
            WorkerMessage::Attempting { .. }
            | WorkerMessage::Recorded { .. }
            | WorkerMessage::MeasureFinished { .. } => {
                return Err("measurement message from a serving worker".into())
            }
        }
        Ok(())
    }

    /// Requests the host already abandoned are absent; their late events
    /// are dropped.
    fn event(&mut self, id: HostRequestId, event: RequestEvent) -> Result<(), String> {
        let Some(slot) = self.requests.get_mut(&id) else {
            return Ok(());
        };
        if slot.terminal_received {
            return Err(format!("request {id} received an event after its terminal"));
        }
        slot.terminal_received = event.is_terminal();
        slot.events.push_back(event);
        slot.wake();
        Ok(())
    }
}

/// The reader thread holds only `state`: dropping the last client closes
/// the sender, which ends the worker and then the reader.
struct Connection {
    sender: Mutex<Box<dyn MessageSender<HostMessage>>>,
    state: Arc<Mutex<State>>,
    ready: ReadyInfo,
}

impl Connection {
    fn send(&self, message: HostMessage) -> Result<(), RequestError> {
        self.sender
            .lock()
            .unwrap()
            .send(message)
            .map_err(|error| RequestError::WorkerLost {
                reason: error.to_string(),
            })
    }
}

/// A cloneable handle to one ready worker.
#[derive(Clone)]
pub struct EngineClient {
    connection: Arc<Connection>,
}

/// The worker's readiness and the client serving it.
pub struct WorkerConnection {
    pub client: EngineClient,
    pub ready: ReadyInfo,
}

/// Complete the load handshake over `transport` and start serving.
///
/// With `load`, this is a worker process that has not yet been told what to
/// load: the host sends `Hello` and `Load`. Without it, the worker was given
/// its manifest directly (in-process). Either way the worker's `Hello` must
/// carry this build, and its `LoadProgress` events reach `progress` until
/// `Ready`, or until `Unprepared`: a worker asked to report tuning units
/// without a stored record met one, placed nothing and is ending. Hosts
/// reach this through [`super::load_prepared`].
pub fn connect_worker(
    transport: impl HostTransport,
    load: Option<(ExecutionManifest, UnpreparedUnits)>,
    mut progress: impl FnMut(LoadProgress),
) -> Result<LoadAttempt<WorkerConnection>, LoadError> {
    let (mut receiver, mut sender) = transport.split();
    let lost = |reason: String| LoadError::WorkerLost { reason };
    let ended = "it was ready";
    let load = load.map(|(manifest, unprepared)| HostMessage::Load {
        manifest,
        unprepared,
    });
    handshake(&mut sender, &mut receiver, load, ended, lost)?;
    let ready = loop {
        match receive(&mut receiver, ended, lost)? {
            WorkerMessage::LoadProgress { progress: current } => progress(current),
            WorkerMessage::Ready { ready } => break ready,
            WorkerMessage::Unprepared => return Ok(LoadAttempt::Unprepared),
            WorkerMessage::LoadFailed { error } => return Err(error),
            _ => return Err(lost("unexpected message during load".into())),
        }
    };
    let state = Arc::new(Mutex::new(State::default()));
    let connection = Arc::new(Connection {
        sender: Mutex::new(Box::new(sender)),
        state: state.clone(),
        ready: ready.clone(),
    });
    let reading = state;
    std::thread::Builder::new()
        .name("magnitude-host-reader".into())
        .spawn(move || loop {
            let failure = match receiver.receive() {
                Ok(Some(message)) => {
                    let mut state = reading.lock().unwrap();
                    match state.deliver(message) {
                        Ok(()) if state.closed.is_none() => continue,
                        Ok(()) => return,
                        Err(reason) => reason,
                    }
                }
                Ok(None) => "the worker ended".to_owned(),
                Err(error) => error.to_string(),
            };
            let mut state = reading.lock().unwrap();
            let lost = state.lost(failure);
            state.close(lost);
            return;
        })
        .map_err(|error| LoadError::Internal {
            reason: error.to_string(),
        })?;
    Ok(LoadAttempt::Loaded(WorkerConnection {
        client: EngineClient { connection },
        ready,
    }))
}

/// Ask a worker process for a Prepare of `manifest` and wait for its
/// outcome, reporting its `LoadProgress` to `progress`. The worker exits once
/// prepared; it never loads the model. Hosts reach this through
/// [`super::load_prepared`] or [`super::prepare`], which also see the
/// process ended. The outer error is a worker that reported no outcome (it
/// was lost, or broke the protocol); the inner result is the outcome a
/// worker reported, after which it exits on its own.
pub(super) fn prepare_worker(
    transport: impl HostTransport,
    manifest: ExecutionManifest,
    mut progress: impl FnMut(LoadProgress),
) -> Result<Result<(), LoadError>, LoadError> {
    let (mut receiver, mut sender) = transport.split();
    let lost = |reason: String| LoadError::WorkerLost { reason };
    let ended = "it was prepared";
    let prepare = HostMessage::Prepare { manifest };
    handshake(&mut sender, &mut receiver, Some(prepare), ended, lost)?;
    loop {
        match receive(&mut receiver, ended, lost)? {
            WorkerMessage::LoadProgress { progress: current } => progress(current),
            WorkerMessage::Prepared => return Ok(Ok(())),
            WorkerMessage::LoadFailed { error } => return Ok(Err(error)),
            _ => return Err(lost("unexpected message during preparation".into())),
        }
    }
}

/// Open an exchange with a worker: send it this build's `Hello` and then
/// `first`, unless it was given its work directly (in-process, `None`), and
/// require its own `Hello` to carry this build. Every failure is the worker
/// lost, as `lost` names it; `ended` is what its stream ended before.
pub(super) fn handshake<E>(
    sender: &mut impl MessageSender<HostMessage>,
    receiver: &mut impl MessageReceiver<WorkerMessage>,
    first: Option<HostMessage>,
    ended: &str,
    lost: impl Fn(String) -> E,
) -> Result<(), E> {
    let build = EngineBuild::current();
    if let Some(first) = first {
        sender
            .send(HostMessage::Hello {
                build: build.clone(),
            })
            .and_then(|()| sender.send(first))
            .map_err(|error| lost(error.to_string()))?;
    }
    match receive(receiver, ended, &lost)? {
        WorkerMessage::Hello { build: worker } if worker == build => Ok(()),
        WorkerMessage::Hello { build: worker } => Err(lost(format!(
            "worker build {worker} differs from host build {build}"
        ))),
        _ => Err(lost("the worker did not greet the host".into())),
    }
}

/// The worker's next message. Its stream ending, before `ended`, or
/// failing is the worker lost, as `lost` names it.
pub(super) fn receive<E>(
    receiver: &mut impl MessageReceiver<WorkerMessage>,
    ended: &str,
    lost: impl Fn(String) -> E,
) -> Result<WorkerMessage, E> {
    match receiver.receive() {
        Ok(Some(message)) => Ok(message),
        Ok(None) => Err(lost(format!("the worker ended before {ended}"))),
        Err(error) => Err(lost(error.to_string())),
    }
}

impl EngineClient {
    pub fn ready_info(&self) -> &ReadyInfo {
        &self.connection.ready
    }

    /// `Ok` while the worker serves; otherwise why it stopped.
    pub fn check(&self) -> Result<(), RequestError> {
        match &self.connection.state.lock().unwrap().closed {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    /// Admit `request`: answered once it began (became resident), or
    /// published before it did, or with why it was refused, which a request
    /// that cannot begin (memory reclaimed until only the model's unload
    /// could make its room) is, before anything of it is answered.
    pub async fn admit(&self, request: RequestOptions) -> Result<EngineRequest, RequestError> {
        if request.output_capacity == 0 {
            return Err(RequestError::InvalidRequest {
                reason: "request output capacity must be positive".into(),
            });
        }
        let id = {
            let mut state = self.connection.state.lock().unwrap();
            if let Some(error) = &state.closed {
                return Err(error.clone());
            }
            let id = HostRequestId(state.next_id);
            state.next_id += 1;
            state.requests.insert(id, RequestSlot::default());
            id
        };
        // From here, dropping the handle cancels the request.
        let mut handle = EngineRequest {
            connection: self.connection.clone(),
            id,
            output_capacity: request.output_capacity,
            finished: false,
        };
        self.connection.send(HostMessage::Admit(Admission {
            request_id: id,
            input: request.input,
            options: request.options,
            constraint: request.constraint,
            retention: request.retention,
            output_capacity: request.output_capacity,
        }))?;
        let admission = poll_fn(|cx| {
            let mut state = self.connection.state.lock().unwrap();
            let slot = state.requests.get_mut(&id).expect("live request slot");
            match slot.admission.clone() {
                Some(result) => Poll::Ready(result),
                None => {
                    slot.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        })
        .await;
        match admission {
            Ok(()) => Ok(handle),
            Err(error) => {
                handle.finished = true;
                self.connection.state.lock().unwrap().requests.remove(&id);
                Err(error)
            }
        }
    }

    /// The memory heap's current standing, as the allocation census, and a
    /// fresh reading of every memory domain the loaded device uses.
    pub async fn observe(&self) -> Result<MemoryObservation, RequestError> {
        let reply = self.shared(
            |state| &mut state.observation,
            HostMessage::Observe,
        )?;
        Self::wait(reply).await
    }

    /// Make `bytes` of device memory free beside the model for the service's
    /// measurement job: answered once the worker's expert bank has shrunk
    /// to make it, with why it cannot, or released by
    /// [`Self::release_measurement_room`] first. The bank grows nothing
    /// until that release. One job claims at a time; a claim while one is
    /// pending joins it.
    pub async fn claim_measurement_room(&self, bytes: u64) -> Result<RoomAnswer, RequestError> {
        let reply = {
            let mut state = self.connection.state.lock().unwrap();
            if let Some(error) = &state.closed {
                return Err(error.clone());
            }
            let (reply, claim) = state.claim_room();
            if let Some(claim) = claim {
                // Sent under the state lock so the reply cannot arrive first.
                self.connection.send(HostMessage::ClaimMeasurementRoom { claim, bytes })?;
            }
            reply
        };
        Self::wait(reply).await
    }

    /// The measurement job ended: the bank may grow into its room again. A
    /// claim still pending is withdrawn.
    pub fn release_measurement_room(&self) -> Result<(), RequestError> {
        let mut state = self.connection.state.lock().unwrap();
        state.release_room();
        // Sent under the state lock, so a later claim follows it.
        self.connection.send(HostMessage::ReleaseMeasurementRoom)
    }

    /// The measurement job stored new records: the worker prices its
    /// experts again from what is stored.
    pub fn reprice(&self) -> Result<(), RequestError> {
        self.connection.send(HostMessage::Reprice)
    }

    /// Ask the worker to unload and exit. Open requests fail with
    /// `ModelUnloaded { cause: Shutdown }`.
    pub fn shutdown(&self) -> Result<(), RequestError> {
        self.connection.state.lock().unwrap().shutting_down = true;
        self.connection.send(HostMessage::Shutdown)
    }

    /// Join the pending query of this kind, or send one.
    fn shared<T>(
        &self,
        field: impl FnOnce(&mut State) -> &mut Option<Shared<T>>,
        query: HostMessage,
    ) -> Result<Shared<T>, RequestError> {
        let mut state = self.connection.state.lock().unwrap();
        if let Some(error) = &state.closed {
            return Err(error.clone());
        }
        let slot = field(&mut state);
        if let Some(reply) = slot {
            return Ok(reply.clone());
        }
        let reply = Arc::new(Mutex::new(SharedReply {
            value: None,
            wakers: Vec::new(),
        }));
        *slot = Some(reply.clone());
        // Sent under the state lock so the reply cannot arrive first.
        self.connection.send(query)?;
        Ok(reply)
    }

    async fn wait<T: Clone>(reply: Shared<T>) -> Result<T, RequestError> {
        poll_fn(|cx| {
            let mut reply = reply.lock().unwrap();
            match &reply.value {
                Some(value) => Poll::Ready(value.clone()),
                None => {
                    reply.wakers.push(cx.waker().clone());
                    Poll::Pending
                }
            }
        })
        .await
    }
}

/// The host's unique handle to one admitted request. Dropping it before its
/// terminal event cancels the request.
pub struct EngineRequest {
    connection: Arc<Connection>,
    id: HostRequestId,
    output_capacity: usize,
    finished: bool,
}

impl EngineRequest {
    pub const fn id(&self) -> HostRequestId {
        self.id
    }

    pub const fn output_capacity(&self) -> usize {
        self.output_capacity
    }

    /// The next event; `None` once the terminal event was returned.
    pub async fn receive(&mut self) -> Option<RequestEvent> {
        if self.finished {
            return None;
        }
        let id = self.id;
        let event = poll_fn(|cx| {
            let mut state = self.connection.state.lock().unwrap();
            let slot = state.requests.get_mut(&id).expect("live request slot");
            match slot.events.pop_front() {
                Some(event) => Poll::Ready(event),
                None => {
                    slot.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        })
        .await;
        if event.is_terminal() {
            self.finished = true;
            self.connection.state.lock().unwrap().requests.remove(&id);
        } else if self.connection.check_open() {
            // A lost worker already terminated the stream above.
            let _ = self.connection.send(HostMessage::Credit {
                request_id: id,
                batches: 1,
            });
        }
        Some(event)
    }

    /// Request an ordered stop and return the terminal event, discarding
    /// output accepted after the stop.
    pub async fn stop(&mut self) -> Option<RequestEvent> {
        if self.finished {
            return None;
        }
        // A failed send means the worker is gone: the reader terminates the
        // stream with that failure, which the loop below returns.
        let _ = self.connection.send(HostMessage::Stop { request_id: self.id });
        loop {
            let event = self.receive().await?;
            if event.is_terminal() {
                return Some(event);
            }
        }
    }

    /// The request's scheduling state and progress; `None` once it is no
    /// longer live in the worker.
    pub async fn status(&mut self) -> Option<RequestProgress> {
        if self.finished {
            return None;
        }
        let id = self.id;
        {
            let mut state = self.connection.state.lock().unwrap();
            if state.closed.is_some() {
                return None;
            }
            state
                .requests
                .get_mut(&id)
                .expect("live request slot")
                .status = None;
        }
        self.connection
            .send(HostMessage::Status { request_id: id })
            .ok()?;
        poll_fn(|cx| {
            let mut state = self.connection.state.lock().unwrap();
            let slot = state.requests.get_mut(&id).expect("live request slot");
            match slot.status.take() {
                Some(progress) => Poll::Ready(progress),
                None => {
                    slot.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        })
        .await
    }
}

impl Connection {
    fn check_open(&self) -> bool {
        self.state.lock().unwrap().closed.is_none()
    }
}

impl Drop for EngineRequest {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let open = {
            let mut state = self.connection.state.lock().unwrap();
            state.requests.remove(&self.id);
            state.closed.is_none()
        };
        if open {
            let _ = self.connection.send(HostMessage::Cancel {
                request_id: self.id,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_scheduler::protocol::RoomRefusal;

    /// The answer to a claim released before it arrived answers no later
    /// claim: only the answer naming the pending claim fulfils it.
    #[test]
    fn a_released_claims_answer_does_not_answer_a_later_claim() {
        let mut state = State::default();
        let (first, Some(released)) = state.claim_room() else { panic!("a new claim") };
        state.release_room();
        assert_eq!(first.lock().unwrap().value, Some(Ok(RoomAnswer::Released)));
        let (second, Some(claim)) = state.claim_room() else { panic!("a new claim") };
        assert_ne!(claim, released);
        state.deliver(WorkerMessage::MeasurementRoom { claim: released, room: Ok(RoomAnswer::Made) }).unwrap();
        assert_eq!(second.lock().unwrap().value, None, "the released claim's answer");
        let (joined, None) = state.claim_room() else { panic!("a claim while one is pending joins it") };
        let short = Ok(RoomAnswer::Refused(RoomRefusal::AtFloor { short: 1 }));
        state.deliver(WorkerMessage::MeasurementRoom { claim, room: short.clone() }).unwrap();
        assert_eq!(second.lock().unwrap().value, Some(short.clone()));
        assert_eq!(joined.lock().unwrap().value, Some(short));
    }

    /// A worker that ends after the host asked it to shut down ended for the stop, whether it
    /// unloaded or was killed for overrunning its exit; one the host never stopped was lost.
    #[test]
    fn a_stopped_workers_end_is_the_stop_however_it_ends() {
        let mut state = State::default();
        assert!(matches!(state.lost("the worker ended".into()), RequestError::WorkerLost { .. }));
        state.shutting_down = true;
        assert_eq!(
            state.lost("the worker ended".into()),
            RequestError::ModelUnloaded { cause: crate::error::UnloadCause::Shutdown }
        );
    }
}
