//! Thread-confined execution behind one closed, device-free protocol.
//!
//! The worker is an event pump: it applies every command and event to the
//! execution owner as it arrives, lets the owner run every transition that is
//! possible, then sleeps until the next arrival. An OS memory pressure event
//! arrives as an event like any other. The one wake no arrival brings is the
//! owner's timed re-read of memory ([`Reread`]), whose interval is a search
//! over the owner's own readings, never a set period. It makes no
//! scheduling, storage, or lifecycle decision of its own.

use crate::{
    protocol::{WorkerCommand, WorkerReply},
    publication::PublicationWake,
};
pub use magnitude_executor::CompletionWake;
use magnitude_executor::RequestId;
use std::{
    collections::VecDeque,
    future::Future,
    pin::Pin,
    sync::{Arc, Condvar, Mutex},
    task::{Context, Poll, Waker},
    thread::JoinHandle,
    time::{Duration, Instant},
};

/// Everything the owner applies other than a command.
pub enum Event {
    /// The in-flight group finished on the device.
    Completion,
    Publication(RequestId, PublicationWake),
    /// A request whose admission reply was dropped, or whose host cancelled it.
    Cancel(RequestId),
    /// The measurement job ended. Its grant and traffic hold are released in arrival order.
    ReleaseMeasurementRoom,
    /// The OS signalled memory pressure: the owner reads memory at once.
    MemoryPressure,
    /// The owner's timed re-read of memory is due ([`Reread`]): it reads
    /// memory at once.
    Reread,
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
    /// Run every transition that is possible now: when the owner reads
    /// memory again with no arrival ([`Reread`]), or why execution stopped,
    /// where it did. Otherwise the owner runs again only at the next
    /// arrival.
    fn run(&mut self, now: u64) -> Result<Reread, Self::Stopped>;
}

/// What a run says of the owner's next reading of memory that no arrival
/// brings.
///
/// The owner reads memory at every flight boundary, every OS pressure
/// event and every completion of a change of its own. Nothing on any OS
/// signals memory another program frees, so while something stands that
/// only a later reading can answer (a real shortage, a wait for memory, a
/// growth held back or the bank short of its saturation) and no flight is
/// in progress to bring a boundary, the worker wakes the owner to read
/// again ([`Event::Reread`]). The first wait is what the reading that found
/// it standing took to read; each later reading that changes nothing
/// doubles it, and one that changes anything sets it back to what that
/// reading took. So freed memory is noticed within the time already
/// waited, and idle readings cost a share of the time that halves with
/// each. The wait never grows past the shortest interval the load's
/// ledgers have seen between two readings across which others' memory
/// moved (`Ledger::others_interval`): the owner never reads slower than
/// others were seen to change. Until one is seen, the doubling's own bound
/// is the only one. The doubling is a search over the owner's own
/// readings and its ceiling a measurement, not a margin: no period is set.
/// Once nothing stands, no timed reading is owed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reread {
    /// No timed reading is owed: nothing stands, or a flight is in
    /// progress, whose boundary reads.
    Settled,
    /// The run read no memory: the timed reading stands as it was.
    Unread,
    /// The run read memory, `cost` the reading took, and something stands
    /// with no flight in progress; `changed` is whether the reading's
    /// response changed anything of the owner's, and `ceiling` the shortest
    /// interval across which others' memory was seen to move, none before
    /// one was.
    Stands { cost: Duration, changed: bool, ceiling: Option<Duration> },
}

/// The worker's timed reading ([`Reread`]): settled, or active with the
/// interval its search stands at and when the owner reads next (none once one
/// fired and no reading has answered it yet).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Rereading {
    #[default]
    Settled,
    Active {
        interval: Duration,
        due: Option<Instant>,
    },
}

impl Rereading {
    /// The timed reading after a run that returned `reread`, at `now`.
    /// A coarse monotonic clock may report a zero reading duration; it is
    /// floored to 1 ns so the interval stays positive and doubling makes
    /// progress.
    fn after(self, reread: Reread, now: Instant) -> Self {
        match reread {
            Reread::Settled => Self::Settled,
            Reread::Unread => self,
            Reread::Stands { cost, changed, ceiling } => {
                let cost = cost.max(Duration::from_nanos(1));
                let interval = match (changed, self) {
                    (false, Self::Active { interval, .. }) => interval.saturating_mul(2),
                    _ => cost,
                };
                let interval = ceiling.map_or(interval, |ceiling| interval.min(ceiling.max(Duration::from_nanos(1))));
                Self::Active { interval, due: now.checked_add(interval) }
            }
        }
    }
}

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

enum ReplyState<C> {
    Waiting(Option<Waker>),
    Ready(Result<WorkerReply, CallError<C>>),
    Abandoned,
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
enum InboxPhase<C> {
    Open {
        /// Commands and events in arrival order.
        items: VecDeque<Item<C>>,
        /// The session's waker, woken when execution stops.
        stop_watcher: Option<Waker>,
    },
    Stopped(C),
}
struct Inbox<C> {
    phase: InboxPhase<C>,
    /// Dispatched calls not yet consumed or abandoned.
    outstanding: usize,
}
struct Mailbox<C> {
    state: Mutex<Inbox<C>>,
    changed: Condvar,
    /// Notified once execution stops ([`Worker::stop`]).
    stopped: Condvar,
    capacity: usize,
}

impl<C: Clone> Mailbox<C> {
    fn push(&self, event: Event) {
        let mut state = self.state.lock().unwrap();
        let InboxPhase::Open { items, .. } = &mut state.phase else {
            return;
        };
        items.push_back(Item::Event(event));
        self.changed.notify_one();
    }
    fn release(&self) {
        self.state.lock().unwrap().outstanding -= 1;
    }
    /// Deliver one reply. Returns the request to cancel when the caller had
    /// abandoned an admission that the owner accepted.
    fn deliver(&self, reply: &Reply<C>, result: Result<WorkerReply, CallError<C>>) -> Option<RequestId> {
        let mut state = reply.state.lock().unwrap();
        match std::mem::replace(&mut *state, ReplyState::Abandoned) {
            ReplyState::Waiting(waker) => {
                *state = ReplyState::Ready(result);
                drop(state);
                reply.changed.notify_one();
                if let Some(waker) = waker {
                    waker.wake();
                }
                None
            }
            ReplyState::Abandoned => {
                drop(state);
                self.release();
                match result {
                    Ok(WorkerReply::Admitted { request, .. }) => Some(request),
                    _ => None,
                }
            }
            ReplyState::Ready(_) => unreachable!("invariant violated: a worker reply was delivered twice"),
        }
    }
    /// Stop accepting work and fail every pending command with `cause`.
    fn stop(&self, cause: C) {
        let (pending, watcher) = {
            let mut state = self.state.lock().unwrap();
            match std::mem::replace(&mut state.phase, InboxPhase::Stopped(cause.clone())) {
                InboxPhase::Open { items, stop_watcher } => (items, stop_watcher),
                InboxPhase::Stopped(_) => unreachable!("invariant violated: mailbox stopped twice"),
            }
        };
        self.stopped.notify_all();
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

/// The wakes an execution owner raises: a submitted flight's completion, a
/// request's publication activity, and the OS's memory pressure events.
pub trait Wakes: Send + Sync {
    fn completion(&self) -> CompletionWake;
    fn publication(&self, request: RequestId, wake: PublicationWake);
    /// The wake each OS memory pressure event raises, as often as it comes.
    fn pressure(&self) -> Box<dyn Fn() + Send + Sync>;
}

impl<C: Clone + Send + 'static> Wakes for WorkerWakeHandle<C> {
    fn completion(&self) -> CompletionWake {
        WorkerWakeHandle::completion(self)
    }
    fn publication(&self, request: RequestId, wake: PublicationWake) {
        WorkerWakeHandle::publication(self, request, wake)
    }
    fn pressure(&self) -> Box<dyn Fn() + Send + Sync> {
        let mailbox = self.mailbox.clone();
        Box::new(move || mailbox.push(Event::MemoryPressure))
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
                if matches!(&*state, ReplyState::Waiting(_)) {
                    state = self.reply.changed.wait(state).unwrap();
                    continue;
                }
                match std::mem::replace(&mut *state, ReplyState::Abandoned) {
                    ReplyState::Ready(result) => break result,
                    ReplyState::Waiting(_) => unreachable!(),
                    ReplyState::Abandoned => unreachable!("invariant violated: unconsumed call reply was abandoned"),
                }
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
            match &mut *state {
                ReplyState::Waiting(waker) => {
                    *waker = Some(cx.waker().clone());
                    None
                }
                ReplyState::Ready(_) => match std::mem::replace(&mut *state, ReplyState::Abandoned) {
                    ReplyState::Ready(result) => Some(result),
                    _ => unreachable!(),
                },
                ReplyState::Abandoned => unreachable!("invariant violated: unconsumed call reply was abandoned"),
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
            match std::mem::replace(&mut *state, ReplyState::Abandoned) {
                ReplyState::Ready(result) => Some(result),
                ReplyState::Waiting(_) => None,
                ReplyState::Abandoned => unreachable!("invariant violated: unconsumed call reply was already abandoned"),
            }
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
impl<C: Clone + Send + 'static> Worker<C> {
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
                phase: InboxPhase::Open {
                    items: VecDeque::new(),
                    stop_watcher: None,
                },
                outstanding: 0,
            }),
            changed: Condvar::new(),
            stopped: Condvar::new(),
            capacity,
        });
        let queue = mailbox.clone();
        let (ready, receive) = std::sync::mpsc::sync_channel(1);
        let thread = seismic_threads::spawn("magnitude-execution", move || {
            let wakes = WorkerWakeHandle {
                mailbox: queue.clone(),
            };
            // A panic of the owner, as it is built or as it runs, ends the
            // process (`seismic_threads::spawn`): nothing drains, drops or
            // answers from the state it left.
            match factory(wakes) {
                Ok((mut owner, readiness)) => {
                    let _ = ready.send(Ok(readiness));
                    let cause = run(owner.as_mut(), &queue);
                    queue.stop(cause);
                }
                Err(error) => {
                    let _ = ready.send(Err(SpawnError::Factory(error)));
                }
            }
        })
        .map_err(SpawnError::Thread)?;
        let worker = Self {
            mailbox,
            thread: Some(thread),
        };
        let host = host().map_err(SpawnError::Host)?;
        let readiness = receive
            .recv()
            .expect("invariant violated: the execution thread answers its readiness, since its panic ends the process")?;
        Ok((worker, readiness, host))
    }
    pub fn client(&self) -> Client<C> {
        Client {
            mailbox: self.mailbox.clone(),
        }
    }
    /// Terminate every request, let the flight finish, and return once
    /// execution has stopped, with why: its owner has settled what it owes,
    /// and its teardown runs on from here until [`Self::close`] (or the
    /// drop) joins it.
    pub fn stop(&self) -> C {
        self.mailbox.push(Event::Close);
        let state = self.mailbox.state.lock().unwrap();
        let state = self
            .mailbox
            .stopped
            .wait_while(state, |state| matches!(state.phase, InboxPhase::Open { .. }))
            .unwrap();
        match &state.phase {
            InboxPhase::Stopped(cause) => cause.clone(),
            InboxPhase::Open { .. } => unreachable!("execution stopped"),
        }
    }
    /// Terminate every request, let the flight finish, and join the thread.
    pub fn close(&mut self) {
        if let Some(thread) = self.thread.take() {
            self.mailbox.push(Event::Close);
            thread.join().expect("the execution thread's panic ends the process");
        }
    }
}
impl<C: Clone + Send + 'static> Drop for Worker<C> {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            self.mailbox.push(Event::Close);
            thread.join().expect("the execution thread's panic ends the process");
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
        match &self.mailbox.state.lock().unwrap().phase {
            InboxPhase::Open { .. } => None,
            InboxPhase::Stopped(cause) => Some(cause.clone()),
        }
    }
    /// Why execution stopped; otherwise `waker` replaces the watcher woken
    /// once it does.
    pub fn poll_stopped(&self, waker: &Waker) -> Option<C> {
        let mut state = self.mailbox.state.lock().unwrap();
        match &mut state.phase {
            InboxPhase::Open { stop_watcher, .. } => {
                *stop_watcher = Some(waker.clone());
                None
            }
            InboxPhase::Stopped(cause) => Some(cause.clone()),
        }
    }
    pub fn dispatch(&self, command: WorkerCommand) -> Result<Call<C>, DispatchError<C>> {
        let reply = Arc::new(Reply {
            state: Mutex::new(ReplyState::Waiting(None)),
            changed: Condvar::new(),
        });
        let mut guard = self.mailbox.state.lock().unwrap();
        let inbox = &mut *guard;
        match &mut inbox.phase {
            InboxPhase::Stopped(cause) => return Err(DispatchError::Stopped(cause.clone())),
            InboxPhase::Open { items, .. } => {
                if inbox.outstanding >= self.mailbox.capacity {
                    return Err(DispatchError::Full);
                }
                inbox.outstanding += 1;
                items.push_back(Item::Control(Envelope {
                    command,
                    reply: reply.clone(),
                }));
            }
        }
        self.mailbox.changed.notify_one();
        Ok(Call {
            reply,
            mailbox: self.mailbox.clone(),
            consumed: false,
        })
    }
    /// Reserved lifecycle path; it cannot carry arbitrary caller behavior.
    pub fn release_measurement_room(&self) {
        self.mailbox.push(Event::ReleaseMeasurementRoom);
    }
    pub fn cancel(&self, request: RequestId) {
        self.mailbox.push(Event::Cancel(request));
    }
}

/// Pump `mailbox` into `owner` until it stops: every arrival applied, the
/// owner run, then a sleep until the next arrival or the owner's timed
/// reading ([`Reread`]).
fn run<C: Clone>(owner: &mut dyn Driven<Stopped = C>, mailbox: &Mailbox<C>) -> C {
    let mut rereading = Rereading::default();
    let start = Instant::now();
    let now = || u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    loop {
        let items = {
            let mut state = mailbox.state.lock().unwrap();
            match &mut state.phase {
                InboxPhase::Open { items, .. } => std::mem::take(items),
                InboxPhase::Stopped(_) => unreachable!("invariant violated: mailbox stopped while worker is running"),
            }
        };
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
        match owner.run(now()) {
            Ok(reread) => rereading = rereading.after(reread, Instant::now()),
            Err(cause) => return cause,
        }
        let mut state = mailbox.state.lock().unwrap();
        loop {
            match &mut state.phase {
                InboxPhase::Open { items, .. } if !items.is_empty() => break,
                InboxPhase::Open { .. } => {}
                InboxPhase::Stopped(_) => unreachable!("invariant violated: mailbox stopped while worker is running"),
            }
            let Rereading::Active { interval, due: Some(due) } = rereading else {
                state = mailbox.changed.wait(state).unwrap();
                continue;
            };
            match due.checked_duration_since(Instant::now()).filter(|left| !left.is_zero()) {
                Some(left) => state = mailbox.changed.wait_timeout(state, left).unwrap().0,
                None => {
                    // Fired once: the reading it brings sets the next.
                    rereading = Rereading::Active { interval, due: None };
                    if let InboxPhase::Open { items, .. } = &mut state.phase {
                        items.push_back(Item::Event(Event::Reread));
                    }
                    break;
                }
            }
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
    use std::time::Duration;

    const TIMEOUT: Duration = Duration::from_secs(2);

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
            retention: crate::prefix_cache::PrefixRetention::Transient,
            output_capacity: 1,
        })
    }

    /// Reports every call; stops once closed.
    struct Recorder {
        calls: mpsc::Sender<&'static str>,
        closed: bool,
        /// What its runs return, in turn; settled once spent.
        reread: VecDeque<Reread>,
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
                    Event::ReleaseMeasurementRoom => "release room",
                    Event::MemoryPressure => "pressure",
                    Event::Reread => "reread",
                    Event::Close => {
                        self.closed = true;
                        "close"
                    }
                })
                .unwrap();
        }
        fn run(&mut self, _: u64) -> Result<Reread, String> {
            self.calls.send("run").unwrap();
            if self.closed {
                return Err("closed".into());
            }
            Ok(self.reread.pop_front().unwrap_or(Reread::Settled))
        }
    }

    /// A worker over a [`Recorder`], with the wakes its owner was given.
    fn recorder() -> (Worker<String>, mpsc::Receiver<&'static str>, WorkerWakeHandle<String>) {
        rereading(Vec::new())
    }

    /// A worker over a [`Recorder`] whose runs return `reread` in turn.
    fn rereading(reread: Vec<Reread>) -> (Worker<String>, mpsc::Receiver<&'static str>, WorkerWakeHandle<String>) {
        let (calls, observed) = mpsc::channel();
        let (handle, wakes) = mpsc::channel();
        let worker = Worker::spawn(
            move |wakes| {
                handle.send(wakes).unwrap();
                Ok(Box::new(Recorder { calls, closed: false, reread: reread.into() }) as Owner<String>)
            },
            4,
        )
        .unwrap();
        (worker, observed, wakes.recv_timeout(TIMEOUT).unwrap())
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
                    Box::new(Recorder { calls, closed: false, reread: VecDeque::new() }) as Owner<String>,
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
        let (mut worker, observed, _wakes) = recorder();
        assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "run");
        assert!(observed.recv_timeout(Duration::from_millis(200)).is_err());
        worker.close();
    }

    /// A memory pressure event wakes the owner like any arrival.
    #[test]
    fn a_memory_pressure_event_runs_the_owner() {
        let (mut worker, observed, wakes) = recorder();
        assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "run");
        (wakes.pressure())();
        assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "pressure");
        assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "run");
        worker.close();
    }

    /// While something stands the owner is woken to read again with no
    /// arrival, at each reading that stands, and once nothing stands no
    /// wake comes. The readings are scripted to take a nanosecond, so the
    /// waits are the search's, never the test's.
    #[test]
    fn a_standing_reading_is_read_again_until_it_settles() {
        let cost = Duration::from_nanos(1);
        let (mut worker, observed, _wakes) = rereading(vec![
            Reread::Stands { cost, changed: true, ceiling: None },
            Reread::Stands { cost, changed: false, ceiling: None },
            Reread::Settled,
        ]);
        assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "run");
        for _ in 0..2 {
            assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "reread");
            assert_eq!(observed.recv_timeout(TIMEOUT).unwrap(), "run");
        }
        assert!(observed.recv_timeout(Duration::from_millis(200)).is_err(), "nothing stands: no wake");
        worker.close();
    }

    /// The search's steps: a standing reading waits what it took; one that
    /// changes nothing doubles the wait, never past the shortest interval
    /// others were seen to move across; one that changes anything sets it
    /// back to what it took; an unread run keeps it; a settled one ends it.
    #[test]
    fn the_rereading_interval_starts_at_the_reading_doubles_to_its_ceiling_and_resets() {
        let now = Instant::now();
        let ms = Duration::from_millis;
        let stands = |cost, changed| Reread::Stands { cost: ms(cost), changed, ceiling: None };
        let first = Rereading::default().after(stands(3, false), now);
        assert_eq!(first, Rereading::Active { interval: ms(3), due: Some(now + ms(3)) });
        let doubled = first.after(stands(5, false), now).after(stands(5, false), now);
        assert_eq!(doubled, Rereading::Active { interval: ms(12), due: Some(now + ms(12)) });
        assert_eq!(doubled.after(Reread::Unread, now + ms(1)), doubled);
        let reset = doubled.after(stands(4, true), now);
        assert_eq!(reset, Rereading::Active { interval: ms(4), due: Some(now + ms(4)) });
        assert_eq!(reset.after(Reread::Settled, now), Rereading::Settled);
        // Others were seen to move across 10 ms: the doubling stops there.
        let bounded = |interval: Rereading| interval.after(Reread::Stands { cost: ms(4), changed: false, ceiling: Some(ms(10)) }, now);
        let capped = bounded(bounded(reset));
        assert_eq!(capped, Rereading::Active { interval: ms(10), due: Some(now + ms(10)) });
        assert_eq!(bounded(capped), Rereading::Active { interval: ms(10), due: Some(now + ms(10)) });
    }

    #[test]
    fn rereading_floors_zero_cost_from_coarse_clocks_instead_of_panicking() {
        let now = Instant::now();
        let ns = Duration::from_nanos;
        let first = Rereading::default().after(
            Reread::Stands { cost: Duration::ZERO, changed: false, ceiling: None },
            now,
        );
        assert_eq!(first, Rereading::Active { interval: ns(1), due: Some(now + ns(1)) });
        let doubled = first.after(
            Reread::Stands { cost: Duration::ZERO, changed: false, ceiling: None },
            now,
        );
        assert_eq!(doubled, Rereading::Active { interval: ns(2), due: Some(now + ns(2)) });
        let reset = doubled.after(
            Reread::Stands { cost: Duration::ZERO, changed: true, ceiling: Some(Duration::ZERO) },
            now,
        );
        assert_eq!(reset, Rereading::Active { interval: ns(1), due: Some(now + ns(1)) });
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
            fn run(&mut self, _: u64) -> Result<Reread, String> {
                if self.closed {
                    return Err("closed".into());
                }
                if !self.submitted {
                    self.submitted = true;
                    self.flight.send(self.wakes.completion()).unwrap();
                }
                Ok(Reread::Settled)
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
            fn run(&mut self, _: u64) -> Result<Reread, String> {
                Err("device lost".into())
            }
        }
        let mut worker =
            Worker::spawn(|_| Ok(Box::new(Stopping) as Owner<String>), 1).unwrap();
        let client = worker.client();
        let deadline = Instant::now() + TIMEOUT;
        while client.stopped().is_none() {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
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
            fn run(&mut self, _: u64) -> Result<Reread, String> {
                if self.closed {
                    return Err("closed".into());
                }
                Ok(Reread::Settled)
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
        let (mut worker, observed, _wakes) = recorder();
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
        client.release_measurement_room();
        client.cancel(RequestId(1));
        let mut seen = Vec::new();
        while !seen.contains(&"cancel") {
            seen.push(observed.recv_timeout(TIMEOUT).unwrap());
        }
        let ordered: Vec<_> = seen.into_iter().filter(|event| *event != "run").collect();
        assert_eq!(ordered, ["command", "command", "command", "command", "release room", "cancel"]);
        drop(held);
        worker.close();
    }
}
