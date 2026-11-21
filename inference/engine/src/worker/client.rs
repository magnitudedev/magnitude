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
use crate::census::AllocationCensus;
use crate::error::{LoadError, RequestError, UnloadCause};
use crate::execution::LoadAttempt;
use crate::invocation::ReleaseGuard;
use magnitude_executor::UnpreparedUnits;
use crate::options::{ExecutionManifest, ReadyInfo};
use magnitude_chat::ConstraintPlan;
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{DetailedUsage, FinishReason, Options, OutputToken};
use magnitude_scheduler::prefix_cache::PrefixRetention;
use magnitude_scheduler::protocol::RoomAnswer;
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
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
enum RequestLifecycle {
    /// Until the request began, or was refused.
    #[default]
    PendingAdmission,
    Open(VecDeque<RequestEvent>),
    Terminal(VecDeque<RequestEvent>),
    Refused(RequestError),
}

#[derive(Default)]
struct RequestSlot {
    lifecycle: RequestLifecycle,
    /// `Some` once the pending status query was answered.
    status: Option<Option<RequestProgress>>,
    waker: Option<Waker>,
}

impl RequestSlot {
    fn admit(&mut self, id: HostRequestId) -> Result<(), String> {
        match &self.lifecycle {
            RequestLifecycle::PendingAdmission => {
                self.lifecycle = RequestLifecycle::Open(VecDeque::new());
                self.wake();
                Ok(())
            }
            RequestLifecycle::Open(_)
            | RequestLifecycle::Terminal(_)
            | RequestLifecycle::Refused(_) => Err(format!(
                "request {id} received admission after it was already decided"
            )),
        }
    }

    fn refuse(&mut self, id: HostRequestId, error: RequestError) -> Result<(), String> {
        match &self.lifecycle {
            RequestLifecycle::PendingAdmission => {
                self.lifecycle = RequestLifecycle::Refused(error);
                self.wake();
                Ok(())
            }
            RequestLifecycle::Open(_)
            | RequestLifecycle::Terminal(_)
            | RequestLifecycle::Refused(_) => Err(format!(
                "request {id} received admission refusal after it was already decided"
            )),
        }
    }

    fn push_event(&mut self, id: HostRequestId, event: RequestEvent) -> Result<(), String> {
        match &mut self.lifecycle {
            RequestLifecycle::Open(events) => {
                let terminal = event.is_terminal();
                events.push_back(event);
                if terminal {
                    let events = std::mem::take(events);
                    self.lifecycle = RequestLifecycle::Terminal(events);
                }
                self.wake();
                Ok(())
            }
            RequestLifecycle::Terminal(_) => {
                Err(format!("request {id} received an event after its terminal"))
            }
            RequestLifecycle::PendingAdmission => {
                Err(format!("request {id} received an event before admission"))
            }
            RequestLifecycle::Refused(_) => Err(format!(
                "request {id} received an event after admission was refused"
            )),
        }
    }

    fn close(&mut self, error: RequestError) {
        match &mut self.lifecycle {
            RequestLifecycle::PendingAdmission => {
                self.lifecycle = RequestLifecycle::Refused(error);
            }
            RequestLifecycle::Open(events) => {
                events.push_back(RequestEvent::Failed(error));
                let events = std::mem::take(events);
                self.lifecycle = RequestLifecycle::Terminal(events);
            }
            RequestLifecycle::Terminal(_) | RequestLifecycle::Refused(_) => {}
        }
        self.status.get_or_insert(None);
        self.wake();
    }

    fn wake(&mut self) {
        if let Some(waker) = self.waker.take() {
            waker.wake();
        }
    }
}

/// Keyed waiter registry: repeated `Poll::Pending` updates a waiter's slot in
/// place instead of growing a `Vec`, and dropping a pending future removes its
/// slot immediately.
#[derive(Default)]
struct WakerSet {
    next: u64,
    wakers: BTreeMap<u64, Waker>,
}

impl WakerSet {
    fn register(&mut self, slot: &mut Option<u64>, waker: &Waker) {
        let id = *slot.get_or_insert_with(|| {
            let id = self.next;
            self.next += 1;
            id
        });
        match self.wakers.get_mut(&id) {
            Some(existing) => {
                if !existing.will_wake(waker) {
                    existing.clone_from(waker);
                }
            }
            None => {
                self.wakers.insert(id, waker.clone());
            }
        }
    }

    fn remove(&mut self, slot: &mut Option<u64>) {
        if let Some(id) = slot.take() {
            self.wakers.remove(&id);
        }
    }

    fn wake_all(&mut self) {
        for (_, waker) in std::mem::take(&mut self.wakers) {
            waker.wake();
        }
    }
}

/// One reply shared by every caller waiting for it.
struct SharedReply<T> {
    value: Option<Result<T, RequestError>>,
    wakers: WakerSet,
}

impl<T> Default for SharedReply<T> {
    fn default() -> Self {
        Self {
            value: None,
            wakers: WakerSet::default(),
        }
    }
}

impl<T> SharedReply<T> {
    fn fulfill(&mut self, value: Result<T, RequestError>) {
        self.value = Some(value);
        self.wakers.wake_all();
    }

    fn fulfill_if_pending(&mut self, value: Result<T, RequestError>) {
        if self.value.is_none() {
            self.fulfill(value);
        }
    }
}

struct WatcherGuard<'a> {
    state: &'a Mutex<State>,
    slot: Option<u64>,
}

impl Drop for WatcherGuard<'_> {
    fn drop(&mut self) {
        if self.slot.is_some() {
            self.state.lock().unwrap().watchers.remove(&mut self.slot);
        }
    }
}

struct ReplyGuard<'a, T> {
    reply: &'a Mutex<SharedReply<T>>,
    slot: Option<u64>,
}

impl<T> Drop for ReplyGuard<'_, T> {
    fn drop(&mut self) {
        if self.slot.is_some() {
            self.reply.lock().unwrap().wakers.remove(&mut self.slot);
        }
    }
}

type Shared<T> = Arc<Mutex<SharedReply<T>>>;

#[derive(Default)]
enum ConnectionPhase {
    #[default]
    Open,
    /// The host asked the worker to shut down ([`EngineClient::shutdown`]): a connection that
    /// ends after that ends for the stop, however the worker's end came (its own unload, or
    /// its kill once it overran its exit).
    ShuttingDown,
    /// Why the connection ended; every later operation fails with it. `unloaded` is true when
    /// the worker itself answered that its model is unloaded: its process is ending by its own
    /// choice, unlike a connection that merely closed.
    Closed {
        error: RequestError,
        unloaded: bool,
    },
}

impl ConnectionPhase {
    fn is_open(&self) -> bool {
        match self {
            Self::Open | Self::ShuttingDown => true,
            Self::Closed { .. } => false,
        }
    }

    fn check_open(&self) -> Result<(), RequestError> {
        match self {
            Self::Open | Self::ShuttingDown => Ok(()),
            Self::Closed { error, .. } => Err(error.clone()),
        }
    }

    fn closed_error(&self) -> Option<&RequestError> {
        match self {
            Self::Open | Self::ShuttingDown => None,
            Self::Closed { error, .. } => Some(error),
        }
    }

    fn is_unloaded(&self) -> bool {
        match self {
            Self::Open | Self::ShuttingDown => false,
            Self::Closed { unloaded, .. } => *unloaded,
        }
    }

    fn begin_shutdown(&mut self) {
        match self {
            Self::Open => *self = Self::ShuttingDown,
            Self::ShuttingDown | Self::Closed { .. } => {}
        }
    }

    /// Why a connection whose worker ended for `reason` ended: the host's stop, where the host
    /// asked for one, so a stopped model's requests end alike whenever its worker ends.
    fn lost(&self, reason: String) -> RequestError {
        match self {
            Self::Open => RequestError::WorkerLost { reason },
            Self::ShuttingDown => RequestError::ModelUnloaded {
                cause: UnloadCause::Shutdown,
            },
            Self::Closed { error, .. } => error.clone(),
        }
    }
}

#[derive(Default)]
struct State {
    next_id: u64,
    requests: HashMap<HostRequestId, RequestSlot>,
    // Request handles and their output queues can disappear before executor state retires.
    leases: HashMap<HostRequestId, ReleaseGuard>,
    observation: Option<Shared<MemoryObservation>>,
    /// The measurement room's pending claim, and the identity the next
    /// claim takes.
    room: Option<(RoomClaimId, Shared<RoomAnswer>)>,
    next_claim: u64,
    phase: ConnectionPhase,
    /// The census the worker pushed last, and how many it pushed
    /// ([`EngineClient::census`]).
    census: Option<AllocationCensus>,
    censuses: u64,
    /// Waiters on the connection's end or the census's next change
    /// ([`EngineClient::closed`], [`EngineClient::census`]).
    watchers: WakerSet,
}

impl State {
    fn open_request(&mut self, lease: ReleaseGuard) -> Result<HostRequestId, RequestError> {
        self.phase.check_open()?;
        let id = HostRequestId(self.next_id);
        self.next_id += 1;
        self.requests.insert(id, RequestSlot::default());
        if lease.is_held() {
            self.leases.insert(id, lease);
        }
        Ok(id)
    }

    /// Why a connection whose worker ended for `reason` ended: the host's stop, where the host
    /// asked for one, so a stopped model's requests end alike whenever its worker ends.
    fn lost(&self, reason: String) -> RequestError {
        self.phase.lost(reason)
    }

    fn close(&mut self, error: RequestError) {
        self.finish_closing(error, false);
    }

    fn unload(&mut self, cause: UnloadCause) {
        self.finish_closing(RequestError::ModelUnloaded { cause }, true);
    }

    fn finish_closing(&mut self, error: RequestError, unloaded: bool) {
        if !self.phase.is_open() {
            return;
        }
        for slot in self.requests.values_mut() {
            slot.close(error.clone());
        }
        if let Some(reply) = self.observation.take() {
            reply.lock().unwrap().fulfill(Err(error.clone()));
        }
        if let Some((_, reply)) = self.room.take() {
            reply.lock().unwrap().fulfill(Err(error.clone()));
        }
        self.leases.clear();
        self.phase = ConnectionPhase::Closed { error, unloaded };
        self.watchers.wake_all();
    }

    /// Begin a measurement room claim, or join the pending one. Returns the
    /// shared reply and, if a new claim was opened, the claim ID to send.
    fn begin_room_claim(&mut self) -> (Shared<RoomAnswer>, Option<RoomClaimId>) {
        if let Some((_, reply)) = &self.room {
            return (reply.clone(), None);
        }
        let claim = RoomClaimId(self.next_claim);
        self.next_claim += 1;
        let reply = Arc::new(Mutex::new(SharedReply::default()));
        self.room = Some((claim, reply.clone()));
        (reply, Some(claim))
    }

    /// Clear a pending measurement room claim whose send failed, and fail its
    /// shared reply so any caller that joined it wakes with the error.
    fn abort_room_claim(
        &mut self,
        claim: RoomClaimId,
        reply: &Shared<RoomAnswer>,
        error: RequestError,
    ) {
        if self.room.as_ref().is_some_and(|(pending, pending_reply)| {
            *pending == claim && Arc::ptr_eq(pending_reply, reply)
        }) {
            self.room = None;
        }
        reply.lock().unwrap().fulfill_if_pending(Err(error));
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
            WorkerMessage::Retired { request_id } => {
                self.leases.remove(&request_id);
            }
            WorkerMessage::Admitted { request_id } => {
                if let Some(slot) = self.requests.get_mut(&request_id) {
                    slot.admit(request_id)?;
                }
            }
            WorkerMessage::AdmissionRefused { request_id, error } => {
                if let Some(slot) = self.requests.get_mut(&request_id) {
                    slot.refuse(request_id, error)?;
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
            WorkerMessage::Census { census } => {
                self.census = Some(census);
                self.censuses += 1;
                self.watchers.wake_all();
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
                self.unload(cause);
            }
            WorkerMessage::Hello { .. }
            | WorkerMessage::LoadProgress { .. }
            | WorkerMessage::Ready { .. }
            | WorkerMessage::Prepared
            | WorkerMessage::Unprepared
            | WorkerMessage::LoadFailed { .. } => return Err("load message after readiness".into()),
            WorkerMessage::Attempting { .. }
            | WorkerMessage::Opened
            | WorkerMessage::MeasureProgress { .. }
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
        slot.push_event(id, event)
    }
}

/// The reader thread holds only `state`: dropping the last client closes
/// the sender, which ends the worker and then the reader.
///
/// Lock order is strictly `sender -> state` (and `state` is never held across
/// `sender.send`), so every state transition that emits a host message
/// reaches the wire in the same order it mutated `State`.
struct Connection {
    sender: Mutex<Box<dyn MessageSender<HostMessage>>>,
    state: Arc<Mutex<State>>,
    ready: ReadyInfo,
}

impl Connection {
    fn send_locked(
        sender: &mut dyn MessageSender<HostMessage>,
        message: HostMessage,
    ) -> Result<(), RequestError> {
        sender.send(message).map_err(|error| RequestError::WorkerLost {
            reason: error.to_string(),
        })
    }

    fn send(&self, message: HostMessage) -> Result<(), RequestError> {
        let mut sender = self.sender.lock().unwrap();
        Self::send_locked(&mut **sender, message)
    }

    /// Run a state transition and its wire send under one `sender` lock,
    /// releasing `state` before performing I/O. `prepare` returns the value
    /// for the caller and an optional message to send; if sending fails,
    /// `on_error` rolls back `State` before returning the error.
    fn send_with_state<T>(
        &self,
        prepare: impl FnOnce(&mut State) -> Result<(T, Option<HostMessage>), RequestError>,
        on_error: impl FnOnce(&mut State, &T, &RequestError),
    ) -> Result<T, RequestError> {
        let mut sender = self.sender.lock().unwrap();
        let (value, message) = {
            let mut state = self.state.lock().unwrap();
            prepare(&mut state)?
        };
        if let Some(message) = message {
            if let Err(error) = Self::send_locked(&mut **sender, message) {
                let mut state = self.state.lock().unwrap();
                on_error(&mut state, &value, &error);
                return Err(error);
            }
        }
        Ok(value)
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
    let lost = |lost: Lost| LoadError::WorkerLost { reason: lost.reason };
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
            _ => return Err(lost(Lost::broke("unexpected message during load"))),
        }
    };
    let state = Arc::new(Mutex::new(State::default()));
    let connection = Arc::new(Connection {
        sender: Mutex::new(Box::new(sender)),
        state: state.clone(),
        ready: ready.clone(),
    });
    let reading = state;
    seismic::threads::spawn("magnitude-host-reader", move || loop {
        let failure = match receiver.receive() {
            Ok(Some(message)) => {
                let mut state = reading.lock().unwrap();
                match state.deliver(message) {
                    Ok(()) if state.phase.is_open() => continue,
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
    Ok(LoadAttempt::Loaded(Box::new(WorkerConnection {
        client: EngineClient { connection },
        ready,
    })))
}

/// Ask a worker process for a Prepare of `manifest` and wait for its
/// outcome, reporting its `LoadProgress` to `progress`. The worker exits once
/// prepared; it never loads the model. Hosts reach this through
/// [`super::load_prepared`] or [`super::prepare`], which also see the
/// process ended. The outer error is a worker that reported no outcome (its
/// stream ended, failed, or it broke the protocol); the inner result is the
/// outcome a worker reported, after which it exits on its own.
pub(super) fn prepare_worker(
    transport: impl HostTransport,
    manifest: ExecutionManifest,
    mut progress: impl FnMut(LoadProgress),
) -> Result<Result<(), LoadError>, Lost> {
    let (mut receiver, mut sender) = transport.split();
    let lost = |lost: Lost| lost;
    let ended = "it was prepared";
    let prepare = HostMessage::Prepare { manifest };
    handshake(&mut sender, &mut receiver, Some(prepare), ended, lost)?;
    loop {
        match receive(&mut receiver, ended, lost)? {
            WorkerMessage::LoadProgress { progress: current } => progress(current),
            WorkerMessage::Prepared => return Ok(Ok(())),
            WorkerMessage::LoadFailed { error } => return Ok(Err(error)),
            _ => return Err(Lost::broke("unexpected message during preparation")),
        }
    }
}

/// How a host lost its worker: why, and whether the worker's stream ended,
/// which a worker's ends only as its process ends, on its own.
#[derive(Debug)]
pub(super) struct Lost {
    pub(super) reason: String,
    pub(super) ended: bool,
}

impl Lost {
    /// A worker whose stream failed or that broke the protocol: it may
    /// still run.
    fn broke(reason: impl Into<String>) -> Self {
        Self { reason: reason.into(), ended: false }
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
    lost: impl Fn(Lost) -> E,
) -> Result<(), E> {
    let build = EngineBuild::current();
    if let Some(first) = first {
        sender
            .send(HostMessage::Hello {
                build: build.clone(),
            })
            .and_then(|()| sender.send(first))
            .map_err(|error| lost(Lost::broke(error.to_string())))?;
    }
    match receive(receiver, ended, &lost)? {
        WorkerMessage::Hello { build: worker } if worker == build => Ok(()),
        WorkerMessage::Hello { build: worker } => Err(lost(Lost::broke(format!(
            "worker build {worker} differs from host build {build}"
        )))),
        _ => Err(lost(Lost::broke("the worker did not greet the host"))),
    }
}

/// The worker's next message. Its stream ending, before `ended`, or
/// failing is the worker lost, as `lost` names it.
pub(super) fn receive<E>(
    receiver: &mut impl MessageReceiver<WorkerMessage>,
    ended: &str,
    lost: impl Fn(Lost) -> E,
) -> Result<WorkerMessage, E> {
    match receiver.receive() {
        Ok(Some(message)) => Ok(message),
        Ok(None) => Err(lost(Lost { reason: format!("the worker ended before {ended}"), ended: true })),
        Err(error) => Err(lost(Lost::broke(error.to_string()))),
    }
}

impl EngineClient {
    pub fn ready_info(&self) -> &ReadyInfo {
        &self.connection.ready
    }

    /// Whether the worker itself answered that its model is unloaded: its process
    /// is ending by its own choice, unlike a connection that merely closed.
    pub fn unloaded(&self) -> bool {
        self.connection.state.lock().unwrap().phase.is_unloaded()
    }

    /// `Ok` while the worker serves; otherwise why it stopped.
    pub fn check(&self) -> Result<(), RequestError> {
        self.connection.state.lock().unwrap().phase.check_open()
    }

    /// Why the worker stopped, once it has: the connection's end, awaited
    /// rather than polled.
    pub async fn closed(&self) -> RequestError {
        let mut guard = WatcherGuard {
            state: &self.connection.state,
            slot: None,
        };
        poll_fn(|context| {
            let mut state = self.connection.state.lock().unwrap();
            if let Some(error) = state.phase.closed_error().cloned() {
                state.watchers.remove(&mut guard.slot);
                Poll::Ready(error)
            } else {
                state.watchers.register(&mut guard.slot, context.waker());
                Poll::Pending
            }
        })
        .await
    }

    /// The census once the worker pushed more than `seen` of them, with
    /// how many it pushed: each is a change the worker counted, never a
    /// sample. Why the worker stopped, once it has.
    pub async fn census(&self, seen: u64) -> Result<(u64, AllocationCensus), RequestError> {
        let mut guard = WatcherGuard {
            state: &self.connection.state,
            slot: None,
        };
        poll_fn(|context| {
            let mut state = self.connection.state.lock().unwrap();
            if state.censuses > seen {
                state.watchers.remove(&mut guard.slot);
                let census = state.census.clone().expect("a pushed census is kept");
                return Poll::Ready(Ok((state.censuses, census)));
            }
            if let Some(error) = state.phase.closed_error().cloned() {
                state.watchers.remove(&mut guard.slot);
                Poll::Ready(Err(error))
            } else {
                state.watchers.register(&mut guard.slot, context.waker());
                Poll::Pending
            }
        })
        .await
    }

    /// Admit `request`: answered once it began (became resident), or
    /// published before it did, or with why it was refused, which a request
    /// that cannot begin (memory reclaimed until only the model's unload
    /// could make its room) is, before anything of it is answered.
    pub async fn admit(&self, request: RequestOptions) -> Result<EngineRequest, RequestError> {
        self.admit_with_lease(request, ReleaseGuard::none()).await
    }

    pub(crate) async fn admit_with_lease(&self, request: RequestOptions, lease: ReleaseGuard) -> Result<EngineRequest, RequestError> {
        if request.output_capacity == 0 {
            return Err(RequestError::InvalidRequest {
                reason: "request output capacity must be positive".into(),
            });
        }
        let output_capacity = request.output_capacity;
        let id = self.connection.send_with_state(
            |state| {
                let id = state.open_request(lease)?;
                Ok((
                    id,
                    Some(HostMessage::Admit(Admission {
                        request_id: id,
                        input: request.input,
                        options: request.options,
                        constraint: request.constraint,
                        retention: request.retention,
                        output_capacity,
                    })),
                ))
            },
            |state, &id, _| {
                state.requests.remove(&id);
                state.leases.remove(&id);
            },
        )?;
        // From here, dropping the handle cancels the request.
        let mut handle = EngineRequest {
            connection: self.connection.clone(),
            id,
            output_capacity,
            finished: false,
        };
        let admission = poll_fn(|cx| {
            let mut state = self.connection.state.lock().unwrap();
            let slot = state.requests.get_mut(&id).expect("live request slot");
            match &slot.lifecycle {
                RequestLifecycle::PendingAdmission => {
                    slot.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
                RequestLifecycle::Open(_) | RequestLifecycle::Terminal(_) => Poll::Ready(Ok(())),
                RequestLifecycle::Refused(error) => Poll::Ready(Err(error.clone())),
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
        let (reply, _) = self.connection.send_with_state(
            |state| {
                state.phase.check_open()?;
                let (reply, send_claim) = state.begin_room_claim();
                let message =
                    send_claim.map(|claim| HostMessage::ClaimMeasurementRoom { claim, bytes });
                Ok(((reply, send_claim), message))
            },
            |state, (reply, send_claim), error| {
                if let Some(claim) = *send_claim {
                    state.abort_room_claim(claim, reply, error.clone());
                }
            },
        )?;
        Self::wait(reply).await
    }

    /// The measurement job holding the room was started as `process`: the
    /// worker credits its footprint against the room from now.
    pub fn attach_measurement_process(&self, process: magnitude_executor::ProcessId) -> Result<(), RequestError> {
        self.connection.send_with_state(
            |state| {
                state.phase.check_open()?;
                Ok(((), Some(HostMessage::AttachMeasurementProcess { process })))
            },
            |_, _, _| {},
        )
    }

    /// The measurement job ended: the bank may grow into its room again. A
    /// claim still pending is withdrawn.
    pub fn release_measurement_room(&self) -> Result<(), RequestError> {
        self.connection.send_with_state(
            |state| {
                state.release_room();
                Ok(((), Some(HostMessage::ReleaseMeasurementRoom)))
            },
            |_, _, _| {},
        )
    }

    /// The measurement job stored new records: the worker prices its
    /// experts again from what is stored.
    pub fn reprice(&self) -> Result<(), RequestError> {
        self.connection.send(HostMessage::Reprice)
    }

    /// Ask the worker to unload and exit. Open requests fail with
    /// `ModelUnloaded { cause: Shutdown }`.
    pub fn shutdown(&self) -> Result<(), RequestError> {
        self.connection.send_with_state(
            |state| {
                state.phase.begin_shutdown();
                Ok(((), Some(HostMessage::Shutdown)))
            },
            |_, _, _| {},
        )
    }

    /// Join the pending query of this kind, or send one.
    fn shared<T>(
        &self,
        field: impl Fn(&mut State) -> &mut Option<Shared<T>>,
        query: HostMessage,
    ) -> Result<Shared<T>, RequestError> {
        self.connection.send_with_state(
            |state| {
                state.phase.check_open()?;
                let slot = field(state);
                if let Some(reply) = slot {
                    Ok((reply.clone(), None))
                } else {
                    let reply = Arc::new(Mutex::new(SharedReply::default()));
                    *slot = Some(reply.clone());
                    Ok((reply, Some(query)))
                }
            },
            |state, reply, error| {
                let slot = field(state);
                if slot
                    .as_ref()
                    .is_some_and(|pending| Arc::ptr_eq(pending, reply))
                {
                    *slot = None;
                }
                reply.lock().unwrap().fulfill_if_pending(Err(error.clone()));
            },
        )
    }

    async fn wait<T: Clone>(reply: Shared<T>) -> Result<T, RequestError> {
        let mut guard = ReplyGuard {
            reply: &reply,
            slot: None,
        };
        poll_fn(|cx| {
            let mut reply = reply.lock().unwrap();
            if let Some(value) = reply.value.clone() {
                reply.wakers.remove(&mut guard.slot);
                Poll::Ready(value)
            } else {
                reply.wakers.register(&mut guard.slot, cx.waker());
                Poll::Pending
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
            match &mut slot.lifecycle {
                RequestLifecycle::Open(events) => match events.pop_front() {
                    Some(event) => Poll::Ready(event),
                    None => {
                        slot.waker = Some(cx.waker().clone());
                        Poll::Pending
                    }
                },
                RequestLifecycle::Terminal(events) => Poll::Ready(
                    events
                        .pop_front()
                        .expect("terminal request has queued terminal event"),
                ),
                RequestLifecycle::PendingAdmission | RequestLifecycle::Refused(_) => {
                    unreachable!("EngineRequest only exists for admitted requests")
                }
            }
        })
        .await;
        if event.is_terminal() {
            self.finished = true;
            self.connection.state.lock().unwrap().requests.remove(&id);
        } else {
            // A lost worker already terminated the stream above.
            let _ = self.connection.send_with_state(
                |state| {
                    let message = state.phase.is_open().then_some(HostMessage::Credit {
                        request_id: id,
                        batches: 1,
                    });
                    Ok(((), message))
                },
                |_, _, _| {},
            );
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
        let sent = self.connection.send_with_state(
            |state| {
                if !state.phase.is_open() {
                    return Ok((false, None));
                }
                state
                    .requests
                    .get_mut(&id)
                    .expect("live request slot")
                    .status = None;
                Ok((true, Some(HostMessage::Status { request_id: id })))
            },
            |_, _, _| {},
        );
        if !matches!(sent, Ok(true)) {
            return None;
        }
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

impl Drop for EngineRequest {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let _ = self.connection.send_with_state(
            |state| {
                state.requests.remove(&self.id);
                let message = state.phase.is_open().then_some(HostMessage::Cancel {
                    request_id: self.id,
                });
                Ok(((), message))
            },
            |_, _, _| {},
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_scheduler::protocol::RoomRefusal;

    #[test]
    fn request_leases_survive_abandonment_until_retirement_or_worker_loss() {
        let mut state = State::default();
        let open = |state: &mut State| {
            let held = Arc::new(());
            let observe = Arc::downgrade(&held);
            let id = state.open_request(ReleaseGuard::new(held)).unwrap();
            (id, observe)
        };
        let (pending, pending_guard) = open(&mut state);
        let (active, active_guard) = open(&mut state);
        state.deliver(WorkerMessage::Admitted { request_id: active }).unwrap();
        state.deliver(WorkerMessage::Failed {
            request_id: active,
            error: RequestError::internal("failed with work in flight"),
        }).unwrap();
        // A disconnect before admission and a consumer that read its terminal
        // both remove their host slots while the worker still owns the requests.
        state.requests.remove(&pending);
        state.requests.remove(&active);
        assert!(pending_guard.upgrade().is_some());
        assert!(active_guard.upgrade().is_some());
        state.deliver(WorkerMessage::Retired { request_id: pending }).unwrap();
        assert!(pending_guard.upgrade().is_none());
        assert!(active_guard.upgrade().is_some());
        state.close(RequestError::WorkerLost { reason: "worker exited".into() });
        assert!(active_guard.upgrade().is_none());
    }

    /// The answer to a claim released before it arrived answers no later
    /// claim: only the answer naming the pending claim fulfils it.
    #[test]
    fn a_released_claims_answer_does_not_answer_a_later_claim() {
        let mut state = State::default();
        let (aborted, claim) = state.begin_room_claim();
        let claim = claim.expect("first claim sends");
        let (joined_aborted, send) = state.begin_room_claim();
        assert!(send.is_none());
        let send_error = RequestError::internal("send failed");
        state.abort_room_claim(claim, &aborted, send_error.clone());
        assert!(state.room.is_none(), "an unsent claim has no reply to join");
        assert_eq!(joined_aborted.lock().unwrap().value, Some(Err(send_error)));

        let (first, released) = state.begin_room_claim();
        let released = released.expect("new claim after abort");
        state.release_room();
        assert_eq!(first.lock().unwrap().value, Some(Ok(RoomAnswer::Released)));
        let (second, claim) = state.begin_room_claim();
        let claim = claim.expect("new claim after release");
        assert_ne!(claim, released);
        state.deliver(WorkerMessage::MeasurementRoom { claim: released, room: Ok(RoomAnswer::Made) }).unwrap();
        assert_eq!(second.lock().unwrap().value, None, "the released claim's answer");
        let (joined, send) = state.begin_room_claim();
        assert!(send.is_none(), "a claim while one is pending joins it");
        let short = Ok(RoomAnswer::Refused(RoomRefusal::Short { bytes: 1 }));
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
        state.phase.begin_shutdown();
        assert_eq!(
            state.lost("the worker ended".into()),
            RequestError::ModelUnloaded { cause: UnloadCause::Shutdown }
        );
    }

    #[test]
    fn request_lifecycle_rejects_out_of_order_transitions_and_closes_cleanly() {
        let mut state = State::default();
        let pending = state.open_request(ReleaseGuard::none()).unwrap();
        let open = state.open_request(ReleaseGuard::none()).unwrap();
        let refused = state.open_request(ReleaseGuard::none()).unwrap();
        let terminal = state.open_request(ReleaseGuard::none()).unwrap();

        // Event before admission is a protocol error.
        assert!(state
            .deliver(WorkerMessage::Output {
                request_id: pending,
                tokens: Vec::new(),
            })
            .is_err());

        // Admitting opens the request; duplicate admission or refusal is rejected.
        state.deliver(WorkerMessage::Admitted { request_id: open }).unwrap();
        assert!(state.deliver(WorkerMessage::Admitted { request_id: open }).is_err());
        assert!(state
            .deliver(WorkerMessage::AdmissionRefused {
                request_id: open,
                error: RequestError::Overloaded,
            })
            .is_err());

        // Refusing fixes the error; subsequent events are rejected.
        state
            .deliver(WorkerMessage::AdmissionRefused {
                request_id: refused,
                error: RequestError::Overloaded,
            })
            .unwrap();
        assert!(state
            .deliver(WorkerMessage::Output {
                request_id: refused,
                tokens: Vec::new(),
            })
            .is_err());

        // Terminal event closes the stream; subsequent events are rejected.
        state.deliver(WorkerMessage::Admitted { request_id: terminal }).unwrap();
        state
            .deliver(WorkerMessage::Failed {
                request_id: terminal,
                error: RequestError::internal("boom"),
            })
            .unwrap();
        assert!(state
            .deliver(WorkerMessage::Output {
                request_id: terminal,
                tokens: Vec::new(),
            })
            .is_err());

        // Closing the connection transitions PendingAdmission -> Refused and Open -> Terminal,
        // while preserving already-Refused and already-Terminal slots.
        let lost = RequestError::WorkerLost {
            reason: "gone".into(),
        };
        state.close(lost.clone());
        assert!(!state.phase.is_open());
        assert!(!state.phase.is_unloaded());
        assert!(matches!(
            &state.requests[&pending].lifecycle,
            RequestLifecycle::Refused(err) if *err == lost
        ));
        assert!(matches!(
            &state.requests[&open].lifecycle,
            RequestLifecycle::Terminal(events)
                if events.len() == 1 && matches!(&events[0], RequestEvent::Failed(err) if *err == lost)
        ));
        assert!(matches!(
            &state.requests[&refused].lifecycle,
            RequestLifecycle::Refused(RequestError::Overloaded)
        ));
        assert!(matches!(
            &state.requests[&terminal].lifecycle,
            RequestLifecycle::Terminal(events) if events.len() == 1
        ));
    }

    #[test]
    fn unloaded_closes_connection_and_marks_unloaded() {
        let mut state = State::default();
        assert!(state.phase.is_open());
        assert!(!state.phase.is_unloaded());
        state
            .deliver(WorkerMessage::Unloaded {
                cause: UnloadCause::Shutdown,
            })
            .unwrap();
        assert!(!state.phase.is_open());
        assert!(state.phase.is_unloaded());
        assert_eq!(
            state.phase.check_open(),
            Err(RequestError::ModelUnloaded {
                cause: UnloadCause::Shutdown
            })
        );
    }
}
