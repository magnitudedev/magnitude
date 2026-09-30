//! Thread-confined execution behind one closed, device-free protocol.
//!
//! The worker is an event pump: it applies every command and event to the
//! execution owner as it arrives, lets the owner run every transition that is
//! possible, then sleeps until the next arrival or the owner's next due time.
//! It makes no scheduling, storage, or lifecycle decision of its own.

use crate::{
    protocol::{WorkerCommand, WorkerReply},
    publication::PublicationWake,
};
pub use magnitude_executor::CompletionWake;
use magnitude_executor::RequestId;
use std::{
    collections::VecDeque,
    future::Future,
    panic::{catch_unwind, AssertUnwindSafe},
    pin::Pin,
    sync::{Arc, Condvar, Mutex},
    task::{Context, Poll, Waker},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Everything the owner applies other than a command.
pub enum Event {
    /// The in-flight group finished on the device.
    Completion,
    Publication(RequestId, PublicationWake),
    /// A request whose admission reply was dropped, or whose host cancelled it.
    Cancel(RequestId),
    /// The worker is closing: terminate every request, let the flight finish, stop.
    Close,
}

/// Worker-local execution domain. Implementations and all values they own stay
/// on this thread; only the closed command/reply vocabulary crosses it.
pub trait Driven {
    /// Why execution stopped. Every later call fails with it.
    type Stopped;
    /// Executes immediately. No command is deferred or refused for "not now".
    fn command(&mut self, command: WorkerCommand, now: u64) -> Result<WorkerReply, String>;
    fn event(&mut self, event: Event, now: u64);
    /// Run every transition that is possible now. Returns the worker-clock
    /// time at which the owner must run again without an event (its next
    /// memory observation), or why execution stopped.
    fn run(&mut self, now: u64) -> Result<u64, Self::Stopped>;
}

/// The execution owner panicked: execution stopped without a cause of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Panicked;

/// Why a dispatched command has no reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallError<C> {
    /// The owner refused the command.
    Refused(String),
    /// Execution stopped before or while the command ran.
    Stopped(C),
}

/// Why a command could not be dispatched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchError<C> {
    Stopped(C),
    /// Every control slot holds an unconsumed call.
    Full,
}

struct ReplyState<C> {
    abandoned: bool,
    result: Option<Result<WorkerReply, CallError<C>>>,
    waker: Option<Waker>,
}
struct Reply<C> {
    state: Mutex<ReplyState<C>>,
    changed: Condvar,
}
struct Envelope<C> {
    command: WorkerCommand,
    reply: Arc<Reply<C>>,
}
enum Item<C> {
    Control(Envelope<C>),
    Event(Event),
}
struct Inbox<C> {
    /// Commands and events in arrival order.
    items: VecDeque<Item<C>>,
    stopped: Option<C>,
    /// The session's waker, woken when execution stops.
    stop_watcher: Option<Waker>,
    /// Dispatched calls not yet consumed or abandoned.
    outstanding: usize,
}
struct Mailbox<C> {
    state: Mutex<Inbox<C>>,
    changed: Condvar,
    capacity: usize,
}

impl<C: Clone> Mailbox<C> {
    fn push(&self, event: Event) {
        let mut state = self.state.lock().unwrap();
        if state.stopped.is_some() {
            return;
        }
        state.items.push_back(Item::Event(event));
        self.changed.notify_one();
    }
    fn release(&self) {
        self.state.lock().unwrap().outstanding -= 1;
    }
    /// Deliver one reply. Returns the request to cancel when the caller had
    /// abandoned an admission that the owner accepted.
    fn deliver(&self, reply: &Reply<C>, result: Result<WorkerReply, CallError<C>>) -> Option<RequestId> {
        let mut state = reply.state.lock().unwrap();
        if !state.abandoned {
            state.result = Some(result);
            let waker = state.waker.take();
            drop(state);
            reply.changed.notify_one();
            if let Some(waker) = waker {
                waker.wake();
            }
            return None;
        }
        drop(state);
        self.release();
        match result {
            Ok(WorkerReply::Admitted { request, .. }) => Some(request),
            _ => None,
        }
    }
    /// Stop accepting work and fail every pending command with `cause`.
    fn stop(&self, cause: C) {
        let (pending, watcher) = {
            let mut state = self.state.lock().unwrap();
            state.stopped = Some(cause.clone());
            (std::mem::take(&mut state.items), state.stop_watcher.take())
        };
        if let Some(watcher) = watcher {
            watcher.wake();
        }
        for item in pending {
            if let Item::Control(envelope) = item {
                self.deliver(&envelope.reply, Err(CallError::Stopped(cause.clone())));
            }
        }
    }
}

/// The wakes an execution owner raises: a submitted flight's completion and a
/// request's publication activity.
pub trait Wakes: Send + Sync {
    fn completion(&self) -> CompletionWake;
    fn publication(&self, request: RequestId, wake: PublicationWake);
}

impl<C: Clone + Send + 'static> Wakes for WorkerWakeHandle<C> {
    fn completion(&self) -> CompletionWake {
        WorkerWakeHandle::completion(self)
    }
    fn publication(&self, request: RequestId, wake: PublicationWake) {
        WorkerWakeHandle::publication(self, request, wake)
    }
}

/// Reserved wake capability of the owner. It never consumes command capacity
/// and carries no caller-controlled behavior.
pub struct WorkerWakeHandle<C> {
    mailbox: Arc<Mailbox<C>>,
}

impl<C> Clone for WorkerWakeHandle<C> {
    fn clone(&self) -> Self {
        Self {
            mailbox: self.mailbox.clone(),
        }
    }
}

impl<C: Clone + Send + 'static> WorkerWakeHandle<C> {
    pub fn publication(&self, request: RequestId, wake: PublicationWake) {
        self.mailbox.push(Event::Publication(request, wake));
    }
    /// The wake a submitted flight fires on completion.
    pub fn completion(&self) -> CompletionWake {
        let mailbox = self.mailbox.clone();
        CompletionWake::new(move || mailbox.push(Event::Completion))
    }
}

/// One response to one closed worker command.
pub struct Call<C: Clone> {
    reply: Arc<Reply<C>>,
    mailbox: Arc<Mailbox<C>>,
    consumed: bool,
}
impl<C: Clone> Call<C> {
    pub fn wait(mut self) -> Result<WorkerReply, CallError<C>> {
        let result = {
            let mut state = self.reply.state.lock().unwrap();
            loop {
                if let Some(result) = state.result.take() {
                    break result;
                }
                state = self.reply.changed.wait(state).unwrap();
            }
        };
        self.consumed = true;
        self.mailbox.release();
        result
    }
}
impl<C: Clone> Future for Call<C> {
    type Output = Result<WorkerReply, CallError<C>>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        assert!(!this.consumed, "worker reply polled after consumption");
        let result = {
            let mut state = this.reply.state.lock().unwrap();
            match state.result.take() {
                Some(result) => Some(result),
                None => {
                    state.waker = Some(cx.waker().clone());
                    None
                }
            }
        };
        match result {
            Some(result) => {
                this.consumed = true;
                this.mailbox.release();
                Poll::Ready(result)
            }
            None => Poll::Pending,
        }
    }
}
impl<C: Clone> Drop for Call<C> {
    fn drop(&mut self) {
        if self.consumed {
            return;
        }
        let completed = {
            let mut state = self.reply.state.lock().unwrap();
            state.abandoned = true;
            state.waker.take();
            state.result.take()
        };
        if let Some(result) = completed {
            if let Ok(WorkerReply::Admitted { request, .. }) = result {
                self.mailbox.push(Event::Cancel(request));
            }
            self.mailbox.release();
        }
    }
}

/// Why an execution owner could not be started.
#[derive(Debug)]
pub enum SpawnError<E> {
    /// The control capacity must be positive.
    Capacity,
    Thread(std::io::Error),
    /// Construction panicked.
    Panicked,
    /// The execution factory failed.
    Factory(E),
    /// Host-side construction failed.
    Host(E),
}

impl<E: std::fmt::Display> std::fmt::Display for SpawnError<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capacity => formatter.write_str("execution control capacity must be positive"),
            Self::Thread(error) => write!(formatter, "execution thread spawn failed: {error}"),
            Self::Panicked => formatter.write_str("execution factory panicked"),
            Self::Factory(error) | Self::Host(error) => error.fmt(formatter),
        }
    }
}

/// The execution owner a factory builds on the worker thread.
pub type Owner<C> = Box<dyn Driven<Stopped = C>>;

pub struct Worker<C: Clone + Send + 'static> {
    mailbox: Arc<Mailbox<C>>,
    thread: Option<JoinHandle<()>>,
}
impl<C: Clone + Send + From<Panicked> + 'static> Worker<C> {
    pub fn spawn(
        factory: impl FnOnce(WorkerWakeHandle<C>) -> Result<Owner<C>, String> + Send + 'static,
        capacity: usize,
    ) -> Result<Self, String> {
        Self::spawn_ready(move |wakes| factory(wakes).map(|owner| (owner, ())), capacity)
            .map(|(worker, ())| worker)
    }

    /// Start a thread-confined numerical owner and return device-free readiness
    /// evidence only after construction has completed on that worker.
    pub fn spawn_ready<R: Send + 'static>(
        factory: impl FnOnce(WorkerWakeHandle<C>) -> Result<(Owner<C>, R), String> + Send + 'static,
        capacity: usize,
    ) -> Result<(Self, R), String> {
        let (worker, ready, ()) = Self::spawn_ready_with(factory, capacity, || Ok(()))
            .map_err(|error| error.to_string())?;
        Ok((worker, ready))
    }

    /// Construct host artifacts on the caller while the execution thread
    /// builds its domain, then wait for both sides before publishing readiness.
    pub fn spawn_ready_with<R: Send + 'static, H, E: Send + 'static>(
        factory: impl FnOnce(WorkerWakeHandle<C>) -> Result<(Owner<C>, R), E> + Send + 'static,
        capacity: usize,
        host: impl FnOnce() -> Result<H, E>,
    ) -> Result<(Self, R, H), SpawnError<E>> {
        if capacity == 0 {
            return Err(SpawnError::Capacity);
        }
        let mailbox = Arc::new(Mailbox {
            state: Mutex::new(Inbox {
                items: VecDeque::new(),
                stopped: None,
                stop_watcher: None,
                outstanding: 0,
            }),
            changed: Condvar::new(),
            capacity,
        });
        let queue = mailbox.clone();
        let (ready, receive) = std::sync::mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("magnitude-execution".into())
            .spawn(move || {
                let wakes = WorkerWakeHandle {
                    mailbox: queue.clone(),
                };
                let made = catch_unwind(AssertUnwindSafe(|| factory(wakes)))
                    .map_err(|_| SpawnError::Panicked)
                    .and_then(|made| made.map_err(SpawnError::Factory));
                match made {
                    Ok((mut owner, readiness)) => {
                        let _ = ready.send(Ok(readiness));
                        let cause = catch_unwind(AssertUnwindSafe(|| run(owner.as_mut(), &queue)))
                            .unwrap_or_else(|_| C::from(Panicked));
                        queue.stop(cause);
                    }
                    Err(error) => {
                        let _ = ready.send(Err(error));
                    }
                }
            })
            .map_err(SpawnError::Thread)?;
        let worker = Self {
            mailbox,
            thread: Some(thread),
        };
        let host = host().map_err(SpawnError::Host)?;
        let readiness = receive.recv().map_err(|_| SpawnError::Panicked)??;
        Ok((worker, readiness, host))
    }
    pub fn client(&self) -> Client<C> {
        Client {
            mailbox: self.mailbox.clone(),
        }
    }
    /// Terminate every request, let the flight finish, and join the thread.
    pub fn close(&mut self) {
        if let Some(thread) = self.thread.take() {
            self.mailbox.push(Event::Close);
            let _ = thread.join();
        }
    }
}
impl<C: Clone + Send + 'static> Drop for Worker<C> {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            self.mailbox.push(Event::Close);
            let _ = thread.join();
        }
    }
}

/// Cloneable, concrete command capability.
pub struct Client<C> {
    mailbox: Arc<Mailbox<C>>,
}
impl<C> Clone for Client<C> {
    fn clone(&self) -> Self {
        Self {
            mailbox: self.mailbox.clone(),
        }
    }
}
impl<C: Clone + Send + 'static> Client<C> {
    /// Why execution stopped, once it has.
    pub fn stopped(&self) -> Option<C> {
        self.mailbox.state.lock().unwrap().stopped.clone()
    }
    /// Why execution stopped; otherwise `waker` replaces the watcher woken
    /// once it does.
    pub fn poll_stopped(&self, waker: &Waker) -> Option<C> {
        let mut state = self.mailbox.state.lock().unwrap();
        if state.stopped.is_none() {
            state.stop_watcher = Some(waker.clone());
        }
        state.stopped.clone()
    }
    pub fn dispatch(&self, command: WorkerCommand) -> Result<Call<C>, DispatchError<C>> {
        let reply = Arc::new(Reply {
            state: Mutex::new(ReplyState {
                abandoned: false,
                result: None,
                waker: None,
            }),
            changed: Condvar::new(),
        });
        let mut state = self.mailbox.state.lock().unwrap();
        if let Some(cause) = &state.stopped {
            return Err(DispatchError::Stopped(cause.clone()));
        }
        if state.outstanding >= self.mailbox.capacity {
            return Err(DispatchError::Full);
        }
        state.outstanding += 1;
        state.items.push_back(Item::Control(Envelope {
            command,
            reply: reply.clone(),
        }));
        self.mailbox.changed.notify_one();
        Ok(Call {
            reply,
            mailbox: self.mailbox.clone(),
            consumed: false,
        })
    }
    /// Reserved lifecycle path; it cannot carry arbitrary caller behavior.
    pub fn cancel(&self, request: RequestId) {
        self.mailbox.push(Event::Cancel(request));
    }
}

fn run<C: Clone>(owner: &mut dyn Driven<Stopped = C>, mailbox: &Mailbox<C>) -> C {
    let start = Instant::now();
    let now = || u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    loop {
        let items = std::mem::take(&mut mailbox.state.lock().unwrap().items);
        for item in items {
            match item {
                Item::Event(event) => owner.event(event, now()),
                Item::Control(envelope) => {
                    let result = owner
                        .command(envelope.command, now())
                        .map_err(CallError::Refused);
                    if let Some(request) = mailbox.deliver(&envelope.reply, result) {
                        owner.event(Event::Cancel(request), now());
                    }
                }
            }
        }
        let due = match owner.run(now()) {
            Ok(due) => due,
            Err(cause) => return cause,
        };
        let mut state = mailbox.state.lock().unwrap();
        while state.items.is_empty() {
            let remaining = due.saturating_sub(now());
            if remaining == 0 {
                break;
            }
            state = mailbox
                .changed
                .wait_timeout(state, Duration::from_nanos(remaining))
                .unwrap()
                .0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_family_contracts::TokenPlan;
    use magnitude_generation::{
        EndOfGeneration, GenerationSeed, InputLayout, MethodChoice, Options, Sampling, Shaping,
        TokenId,
    };
    use std::collections::BTreeSet;
    use std::sync::mpsc;

    const TIMEOUT: Duration = Duration::from_secs(2);
    const NEVER: u64 = u64::MAX;

    impl From<Panicked> for String {
        fn from(_: Panicked) -> Self {
            "execution owner panicked".into()
        }
    }

    fn admit_request() -> WorkerCommand {
        let tokens = vec![TokenId(1), TokenId(2)];
        let layout = InputLayout::new(tokens.len(), vec![]).unwrap();
        let seed = GenerationSeed::new(
            tokens.clone(),
            layout.clone(),
            Options {
                max_tokens: 1,
                output_capacity: 1,
                context_limit: 8,
                vocabulary: 16,
                stop_tokens: BTreeSet::new(),
                suppressed_tokens: BTreeSet::new(),
                sampling: Sampling::Greedy,
                shaping: Shaping {
                    temperature: 0.0,
                    ..Shaping::default()
                },
                seed: 1,
                forced_quantum: 1,
                method: MethodChoice::Plain,
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: None,
            },
            None,
        )
        .unwrap();
        let input = magnitude_family_contracts::PreparedModelInput::from_text_coordinates(
            TokenPlan::new(tokens, layout).unwrap(),
            vec![[0, 0, 0], [1, 0, 0]],
        )
        .unwrap();
        WorkerCommand::Admit(crate::protocol::AdmitRequest {
            seed,
            input,
            prefix_cache: false,
            output_capacity: 1,
        })
    }

    /// Reports every call; stops once closed.
    struct Recorder {
        calls: mpsc::Sender<&'static str>,
        due: u64,
        closed: bool,
    }
    impl Driven for Recorder {
        type Stopped = String;
        fn command(&mut self, command: WorkerCommand, _: u64) -> Result<WorkerReply, String> {
            self.calls.send("command").unwrap();
            Ok(match command {
                WorkerCommand::Status { .. } => WorkerReply::Status(None),
                _ => WorkerReply::Acknowledged,
            })
        }
        fn event(&mut self, event: Event, _: u64) {
            self.calls
                .send(match event {
                    Event::Completion => "completion",
                    Event::Publication(..) => "publication",
                    Event::Cancel(_) => "cancel",
                    Event::Close => {
                        self.closed = true;
                        "close"
                    }
                })
                .unwrap();
        }
        fn run(&mut self, now: u64) -> Result<u64, String> {
            self.calls.send("run").unwrap();
            if self.closed {
                return Err("closed".into());
            }
            Ok(self.due.saturating_add(now))
        }
    }

    fn recorder(due: u64) -> (Worker<String>, mpsc::Receiver<&'static str>) {
        let (calls, observed) = mpsc::channel();
        let worker = Worker::spawn(
            move |_| {
                Ok(Box::new(Recorder {
                    calls,
                    due,
                    closed: false,
                }) as Owner<String>)
            },
            4,
        )
        .unwrap();
        (worker, observed)
    }

    #[test]
    fn host_construction_runs_before_worker_readiness() {
        let (started, observe_start) = mpsc::sync_channel(1);
        let (finish, allow_finish) = mpsc::sync_channel(1);
        let (calls, _observed) = mpsc::channel();
        let (mut worker, ready, host) = Worker::<String>::spawn_ready_with::<_, _, String>(
            move |_| {
                started.send(()).unwrap();
                allow_finish.recv_timeout(TIMEOUT).map_err(|error| error.to_string())?;
                Ok((
                    Box::new(Recorder {
                        calls,
                        due: NEVER,
                        closed: false,
                    }) as Owner<String>,
                    7,
                ))
            },
            1,
            move || {
                observe_start.recv_timeout(TIMEOUT).map_err(|error| error.to_string())?;
                finish.send(()).unwrap();
                Ok(9)
            },
        )
        .unwrap();
        assert_eq!((ready, host), (7, 9));
        worker.close();
    }

    #[test]
    fn an_idle_owner_runs_once_and_the_worker_sleeps() {
        let (mut worker, observed) = recorder(NEVER);
        assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "run");
        assert!(observed.recv_timeout(Duration::from_millis(200)).is_err());
        worker.close();
    }

    #[test]
    fn the_owner_runs_again_when_its_observation_is_due() {
        let (mut worker, observed) = recorder(Duration::from_millis(5).as_nanos() as u64);
        for _ in 0..3 {
            assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "run");
        }
        worker.close();
    }

    /// The incident's worker half: commands never wait on the owner's state.
    /// An admission dispatched while a flight is in the air is applied at once,
    /// and the flight's completion arrives as an event.
    #[test]
    fn commands_and_completion_are_applied_while_a_flight_is_in_the_air() {
        struct InFlight {
            wakes: WorkerWakeHandle<String>,
            flight: mpsc::Sender<CompletionWake>,
            calls: mpsc::Sender<&'static str>,
            submitted: bool,
            closed: bool,
        }
        impl Driven for InFlight {
            type Stopped = String;
            fn command(&mut self, command: WorkerCommand, _: u64) -> Result<WorkerReply, String> {
                Ok(match command {
                    WorkerCommand::Admit(_) => {
                        self.calls.send("admit").unwrap();
                        WorkerReply::Acknowledged
                    }
                    _ => WorkerReply::Status(None),
                })
            }
            fn event(&mut self, event: Event, _: u64) {
                match event {
                    Event::Completion => self.calls.send("completion").unwrap(),
                    Event::Close => self.closed = true,
                    _ => {}
                }
            }
            fn run(&mut self, _: u64) -> Result<u64, String> {
                if self.closed {
                    return Err("closed".into());
                }
                if !self.submitted {
                    self.submitted = true;
                    self.flight.send(self.wakes.completion()).unwrap();
                }
                Ok(NEVER)
            }
        }
        let (flight, flights) = mpsc::channel();
        let (calls, observed) = mpsc::channel();
        let mut worker = Worker::spawn(
            move |wakes| {
                Ok(Box::new(InFlight {
                    wakes,
                    flight,
                    calls,
                    submitted: false,
                    closed: false,
                }) as Owner<String>)
            },
            2,
        )
        .unwrap();
        let completion = flights.recv_timeout(TIMEOUT).unwrap();
        let client = worker.client();
        assert!(matches!(
            client.dispatch(admit_request()).unwrap().wait(),
            Ok(WorkerReply::Acknowledged)
        ));
        assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "admit");
        completion.complete();
        assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "completion");
        worker.close();
    }

    #[test]
    fn a_stopped_owner_fails_pending_and_later_commands_with_its_cause() {
        struct Stopping;
        impl Driven for Stopping {
            type Stopped = String;
            fn command(&mut self, _: WorkerCommand, _: u64) -> Result<WorkerReply, String> {
                Ok(WorkerReply::Acknowledged)
            }
            fn event(&mut self, _: Event, _: u64) {}
            fn run(&mut self, _: u64) -> Result<u64, String> {
                Err("device lost".into())
            }
        }
        let mut worker =
            Worker::spawn(|_| Ok(Box::new(Stopping) as Owner<String>), 1).unwrap();
        let client = worker.client();
        let deadline = Instant::now() + TIMEOUT;
        while client.stopped().is_none() {
            assert!(Instant::now() < deadline);
            thread::yield_now();
        }
        assert_eq!(client.stopped(), Some("device lost".into()));
        assert!(matches!(
            client.dispatch(WorkerCommand::Observe),
            Err(DispatchError::Stopped(cause)) if cause == "device lost"
        ));
        worker.close();
    }

    #[test]
    fn an_abandoned_admission_is_cancelled_on_the_worker() {
        struct Admitting {
            cancelled: mpsc::Sender<RequestId>,
            closed: bool,
        }
        impl Driven for Admitting {
            type Stopped = String;
            fn command(&mut self, _: WorkerCommand, _: u64) -> Result<WorkerReply, String> {
                let (_, receiver) =
                    crate::publication::PublicationQueue::bounded(1, |_| {})?;
                Ok(WorkerReply::Admitted {
                    request: RequestId(3),
                    receiver,
                })
            }
            fn event(&mut self, event: Event, _: u64) {
                match event {
                    Event::Cancel(request) => self.cancelled.send(request).unwrap(),
                    Event::Close => self.closed = true,
                    _ => {}
                }
            }
            fn run(&mut self, _: u64) -> Result<u64, String> {
                if self.closed {
                    return Err("closed".into());
                }
                Ok(NEVER)
            }
        }
        let (cancelled, observed) = mpsc::channel();
        let mut worker = Worker::spawn(
            move |_| {
                Ok(Box::new(Admitting {
                    cancelled,
                    closed: false,
                }) as Owner<String>)
            },
            1,
        )
        .unwrap();
        drop(worker.client().dispatch(admit_request()).unwrap());
        assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), RequestId(3));
        worker.close();
    }

    #[test]
    fn events_bypass_a_saturated_control_queue() {
        let (mut worker, observed) = recorder(NEVER);
        assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "run");
        let client = worker.client();
        let held = client.dispatch(WorkerCommand::Observe).unwrap();
        let _held_too = client.dispatch(WorkerCommand::Observe).unwrap();
        let _third = client.dispatch(WorkerCommand::Observe).unwrap();
        let _fourth = client.dispatch(WorkerCommand::Observe).unwrap();
        assert!(matches!(
            client.dispatch(WorkerCommand::Observe),
            Err(DispatchError::Full)
        ));
        client.cancel(RequestId(1));
        let mut seen = Vec::new();
        while !seen.contains(&"cancel") {
            seen.push(observed.recv_timeout(TIMEOUT).unwrap());
        }
        drop(held);
        worker.close();
    }
}
