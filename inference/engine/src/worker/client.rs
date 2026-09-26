//! The host side of a worker connection.
//!
//! One reader thread decodes worker messages into per-request state; request
//! handles consume their own events and return one batch of output credit to
//! the worker for each output batch they take. Everything is runtime-agnostic
//! (std synchronization and wakers), so any async executor can drive it.

use super::protocol::{
    Admission, EngineBuild, ExecutionTimings, HostMessage, HostRequestId,
    LoadPhase, MemoryObservation, RequestProgress, RetentionPolicy, WorkerMessage,
};
use super::transport::{HostTransport, MessageReceiver, MessageSender};
use crate::error::{LoadError, RequestError};
use crate::options::{ExecutionManifest, ReadyInfo};
use magnitude_chat::ConstraintPlan;
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{DetailedUsage, FinishReason, Options, OutputToken};
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
    pub retention: RetentionPolicy,
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
    /// `None` while admission is pending.
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
    /// Why the connection ended; every later operation fails with it.
    closed: Option<RequestError>,
}

impl State {
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
        self.closed = Some(error);
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
            WorkerMessage::Unloaded { cause } => {
                self.close(RequestError::ModelUnloaded { cause });
            }
            WorkerMessage::Hello { .. }
            | WorkerMessage::LoadPhase { .. }
            | WorkerMessage::Ready { .. }
            | WorkerMessage::LoadFailed { .. } => {
                return Err("load message after readiness".into())
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
/// carry this build, and its `LoadPhase` events reach `phase` until `Ready`.
pub fn connect_worker(
    transport: impl HostTransport,
    load: Option<ExecutionManifest>,
    mut phase: impl FnMut(LoadPhase),
) -> Result<WorkerConnection, LoadError> {
    let (mut receiver, mut sender) = transport.split();
    let lost = |reason: String| LoadError::WorkerLost { reason };
    let build = EngineBuild::current();
    if let Some(manifest) = load {
        sender
            .send(HostMessage::Hello {
                build: build.clone(),
            })
            .and_then(|()| sender.send(HostMessage::Load { manifest }))
            .map_err(|error| lost(error.to_string()))?;
    }
    let mut next = || match receiver.receive() {
        Ok(Some(message)) => Ok(message),
        Ok(None) => Err(lost("the worker ended before it was ready".into())),
        Err(error) => Err(lost(error.to_string())),
    };
    match next()? {
        WorkerMessage::Hello { build: worker } if worker == build => {}
        WorkerMessage::Hello { build: worker } => {
            return Err(lost(format!(
                "worker build {worker} differs from host build {build}"
            )))
        }
        _ => return Err(lost("the worker did not greet the host".into())),
    }
    let ready = loop {
        match next()? {
            WorkerMessage::LoadPhase { phase: current } => phase(current),
            WorkerMessage::Ready { ready } => break ready,
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
            reading
                .lock()
                .unwrap()
                .close(RequestError::WorkerLost { reason: failure });
            return;
        })
        .map_err(|error| LoadError::Internal {
            reason: error.to_string(),
        })?;
    Ok(WorkerConnection {
        client: EngineClient { connection },
        ready,
    })
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

    /// Ask the worker to unload and exit. Open requests fail with
    /// `ModelUnloaded { cause: Shutdown }`.
    pub fn shutdown(&self) -> Result<(), RequestError> {
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
