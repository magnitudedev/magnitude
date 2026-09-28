//! Thread-confined execution behind one closed, device-free protocol.

use crate::{
    protocol::{WorkerCommand, WorkerReply},
    publication::{PublicationWake, RequestError},
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

pub enum Drive {
    Idle,
    Progress,
    AwaitingCompletion,
}

/// Worker-local execution domain. Implementations and all values they own stay
/// on this thread; only the closed command/reply vocabulary crosses it.
pub trait Driven {
    fn install_wake_handle(&mut self, _wakes: WorkerWakeHandle) {}
    fn admission_ready(&self) -> bool { true }
    fn command(&mut self, _command: WorkerCommand, _now: u64) -> Result<WorkerReply, String> {
        Err("execution domain does not implement the worker protocol".into())
    }
    fn publication_wake(
        &mut self,
        _request: RequestId,
        _wake: PublicationWake,
        _now: u64,
    ) -> Result<(), String> {
        Err("execution domain does not implement publication wakes".into())
    }
    fn advance(&mut self, now: u64, wake: CompletionWake) -> Result<Drive, String>;
    /// Reconcile a completed flight and release its storage bindings before
    /// queued admissions can grow the store. Submission resumes via advance.
    fn settle_completion(&mut self, _now: u64) -> Result<(), String> {
        Ok(())
    }
    /// Runs on the worker thread even while a submitted flight is outstanding.
    /// The owner must not release storage held by that flight here.
    fn periodic(&mut self, _now: u64) -> Result<(), String> {
        Ok(())
    }
    fn failed(&mut self, error: &str);
    /// The classified failure that stopped the domain, if any.
    fn failure(&self) -> Option<&RequestError>;
    fn shutdown(&mut self) -> Result<bool, String>;
}

struct ReplyState {
    abandoned: bool,
    result: Option<Result<WorkerReply, String>>,
    waker: Option<Waker>,
}
struct Reply {
    state: Mutex<ReplyState>,
    changed: Condvar,
}
struct Envelope {
    command: WorkerCommand,
    reply: Arc<Reply>,
}
struct Inbox {
    controls: VecDeque<Envelope>,
    cleanup: VecDeque<WorkerCommand>,
    publications: VecDeque<(RequestId, PublicationWake)>,
    completion: Option<u64>,
    stop: bool,
    closed: Option<String>,
    terminated: bool,
    outstanding: usize,
}
struct Mailbox {
    state: Mutex<Inbox>,
    changed: Condvar,
    capacity: usize,
}

/// Reserved publication wake capability. It never consumes ordinary command
/// capacity and carries no caller-controlled numerical behavior.
#[derive(Clone)]
pub struct WorkerWakeHandle {
    mailbox: Arc<Mailbox>,
}

impl WorkerWakeHandle {
    pub fn publication(&self, request: RequestId, wake: PublicationWake) {
        let mut state = self.mailbox.state.lock().unwrap();
        if state.terminated {
            return;
        }
        state.publications.push_back((request, wake));
        self.mailbox.changed.notify_one();
    }
}

impl Mailbox {
    fn release(&self) {
        self.state.lock().unwrap().outstanding -= 1;
    }
    fn close(&self, error: &str) {
        let pending = {
            let mut state = self.state.lock().unwrap();
            state.closed.get_or_insert_with(|| error.into());
            state.stop = true;
            std::mem::take(&mut state.controls)
        };
        self.changed.notify_one();
        for envelope in pending {
            if deliver(&envelope.reply, Err(error.into())) {
                self.release();
            }
        }
    }
    fn cleanup(&self, command: WorkerCommand) {
        let mut state = self.state.lock().unwrap();
        if state.terminated {
            return;
        }
        state.cleanup.push_back(command);
        self.changed.notify_one();
    }
}

/// One response to one closed worker command.
pub struct Call {
    reply: Arc<Reply>,
    mailbox: Arc<Mailbox>,
    consumed: bool,
}
impl Call {
    pub fn wait(mut self) -> Result<WorkerReply, String> {
        let result = {
            let mut state = self.reply.state.lock().unwrap();
            while state.result.is_none() {
                state = self.reply.changed.wait(state).unwrap();
            }
            state.result.take().unwrap()
        };
        self.consumed = true;
        self.mailbox.release();
        result
    }
}
impl Future for Call {
    type Output = Result<WorkerReply, String>;
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
impl Drop for Call {
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
                self.mailbox.cleanup(WorkerCommand::Cancel { request });
            }
            self.mailbox.release();
        }
    }
}

fn deliver(reply: &Reply, result: Result<WorkerReply, String>) -> bool {
    let (abandoned, waker) = {
        let mut state = reply.state.lock().unwrap();
        let abandoned = state.abandoned;
        if !abandoned {
            state.result = Some(result);
        } else if let Ok(WorkerReply::Admitted { request, receiver }) = result {
            // Preserve the only reply that owns a worker resource so the
            // execution thread can immediately cancel it.
            state.result = Some(Ok(WorkerReply::Admitted { request, receiver }));
        }
        (abandoned, state.waker.take())
    };
    reply.changed.notify_one();
    if let Some(waker) = waker {
        waker.wake();
    }
    abandoned
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

pub struct Worker {
    mailbox: Arc<Mailbox>,
    thread: Option<JoinHandle<()>>,
}
impl Worker {
    pub fn spawn(
        factory: impl FnOnce() -> Result<Box<dyn Driven>, String> + Send + 'static,
        capacity: usize,
    ) -> Result<Self, String> {
        Self::spawn_ready(move || factory().map(|owner| (owner, ())), capacity)
            .map(|(worker, ())| worker)
    }

    /// Start a thread-confined numerical owner and return device-free readiness
    /// evidence only after construction has completed on that worker.
    pub fn spawn_ready<R: Send + 'static>(
        factory: impl FnOnce() -> Result<(Box<dyn Driven>, R), String> + Send + 'static,
        capacity: usize,
    ) -> Result<(Self, R), String> {
        let (worker, ready, ()) = Self::spawn_ready_with(factory, capacity, || Ok(()))
            .map_err(|error| error.to_string())?;
        Ok((worker, ready))
    }

    /// Construct host artifacts on the caller while the execution thread
    /// builds its domain, then wait for both sides before publishing readiness.
    pub fn spawn_ready_with<R: Send + 'static, H, E: Send + 'static>(
        factory: impl FnOnce() -> Result<(Box<dyn Driven>, R), E> + Send + 'static,
        capacity: usize,
        host: impl FnOnce() -> Result<H, E>,
    ) -> Result<(Self, R, H), SpawnError<E>> {
        if capacity == 0 {
            return Err(SpawnError::Capacity);
        }
        let mailbox = Arc::new(Mailbox {
            state: Mutex::new(Inbox {
                controls: VecDeque::new(),
                cleanup: VecDeque::new(),
                publications: VecDeque::new(),
                completion: None,
                stop: false,
                closed: None,
                terminated: false,
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
                let made = catch_unwind(AssertUnwindSafe(factory))
                    .map_err(|_| SpawnError::Panicked)
                    .and_then(|made| made.map_err(SpawnError::Factory));
                match made {
                    Ok((mut owner, readiness)) => {
                        owner.install_wake_handle(WorkerWakeHandle {
                            mailbox: queue.clone(),
                        });
                        let _ = ready.send(Ok(readiness));
                        let result = catch_unwind(AssertUnwindSafe(|| run(owner.as_mut(), &queue)));
                        if result.is_err() {
                            owner.failed("execution owner panicked");
                        }
                        queue.state.lock().unwrap().terminated = true;
                        queue.close(&owner.failure().map_or_else(
                            || "execution worker stopped".to_owned(),
                            ToString::to_string,
                        ));
                    }
                    Err(error) => {
                        queue.state.lock().unwrap().terminated = true;
                        queue.close("execution construction failed");
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
    pub fn client(&self) -> Client {
        Client {
            mailbox: self.mailbox.clone(),
        }
    }
    pub fn close(&mut self) {
        if let Some(thread) = self.thread.take() {
            match self.client().dispatch(WorkerCommand::Close) {
                Ok(call) => {
                    let _ = call.wait();
                }
                Err(_) => self.mailbox.close("execution worker stopped"),
            }
            let _ = thread.join();
        }
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.close();
    }
}

/// Cloneable, concrete command capability.
#[derive(Clone)]
pub struct Client {
    mailbox: Arc<Mailbox>,
}
impl Client {
    pub fn is_closed(&self) -> bool {
        self.mailbox.state.lock().unwrap().closed.is_some()
    }
    /// Why the execution owner stopped accepting commands, once it has.
    pub fn closed_reason(&self) -> Option<String> {
        self.mailbox.state.lock().unwrap().closed.clone()
    }
    pub fn dispatch(&self, command: WorkerCommand) -> Result<Call, String> {
        let reply = Arc::new(Reply {
            state: Mutex::new(ReplyState {
                abandoned: false,
                result: None,
                waker: None,
            }),
            changed: Condvar::new(),
        });
        let mut state = self.mailbox.state.lock().unwrap();
        if let Some(error) = &state.closed {
            return Err(error.clone());
        }
        if state.outstanding >= self.mailbox.capacity {
            return Err("execution control queue is full".into());
        }
        state.outstanding += 1;
        state.controls.push_back(Envelope {
            command,
            reply: reply.clone(),
        });
        self.mailbox.changed.notify_one();
        Ok(Call {
            reply,
            mailbox: self.mailbox.clone(),
            consumed: false,
        })
    }
    /// Reserved lifecycle path; it cannot carry arbitrary caller behavior.
    pub fn cancel(&self, request: magnitude_executor::RequestId) {
        self.mailbox.cleanup(WorkerCommand::Cancel { request });
    }
}

fn run(owner: &mut dyn Driven, mailbox: &Arc<Mailbox>) {
    enum Event {
        Completion(u64),
        Publication(RequestId, PublicationWake),
        Stop,
        Cleanup(WorkerCommand),
        Control(Envelope),
        Tick,
    }
    let start = Instant::now();
    let now = || start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    let mut waiting = None;
    let mut admissions_to_drain = 0;
    let mut serial = 0u64;
    let mut drive = true;
    let mut stopping = false;
    let observation_interval = Duration::from_millis(100);
    let mut next_observation = Instant::now() + observation_interval;
    loop {
        if !stopping {
            if let Some(error) = owner.failure().map(ToString::to_string) {
                mailbox.close(&error);
                stopping = true;
            }
        }
        if stopping {
            match owner.shutdown() {
                Ok(true) => break,
                Ok(false) => {}
                Err(error) => {
                    owner.failed(&error);
                    break;
                }
            }
        }
        if !stopping && Instant::now() >= next_observation {
            if let Err(error) = owner.periodic(now()) {
                owner.failed(&error);
                mailbox.close(&error);
                stopping = true;
                continue;
            }
            next_observation = Instant::now() + observation_interval;
        }
        let event = {
            let mut state = mailbox.state.lock().unwrap();
            loop {
                if let Some(id) = state.completion.take() {
                    break Some(Event::Completion(id));
                }
                if let Some((request, wake)) = state.publications.pop_front() {
                    break Some(Event::Publication(request, wake));
                }
                if let Some(command) = state.cleanup.pop_front() {
                    break Some(Event::Cleanup(command));
                }
                if state.stop {
                    state.stop = false;
                    break Some(Event::Stop);
                }
                // An advance may hold a state transaction before it submits
                // a flight. Let normal drive reach its completion wait before
                // opening another request; the transaction can then release
                // its slab binding before queued admissions run. Lifecycle
                // and status commands remain available throughout.
                let control = if waiting.is_some()
                    || !owner.admission_ready()
                    || (drive && admissions_to_drain == 0)
                {
                    state.controls.iter().position(|envelope| {
                        !matches!(envelope.command, WorkerCommand::Admit(_))
                    })
                } else {
                    (!state.controls.is_empty()).then_some(0)
                };
                if let Some(envelope) = control.and_then(|index| state.controls.remove(index)) {
                    break Some(Event::Control(envelope));
                }
                if drive && waiting.is_none() {
                    break None;
                }
                let (next, elapsed) = mailbox
                    .changed
                    .wait_timeout(
                        state,
                        next_observation.saturating_duration_since(Instant::now()),
                    )
                    .unwrap();
                state = next;
                if elapsed.timed_out() {
                    break Some(Event::Tick);
                }
            }
        };
        match event {
            Some(Event::Stop) => {
                stopping = true;
                continue;
            }
            Some(Event::Tick) => {
                // A blind memory observation needs a fresh claim attempt even
                // when no command or completion arrives to move the epoch.
                drive = waiting.is_none();
            }
            Some(Event::Completion(id)) => {
                if waiting != Some(id) {
                    owner.failed("completion identity differs from outstanding work");
                    mailbox.close("completion identity differs from outstanding work");
                    stopping = true;
                    continue;
                }
                waiting = None;
                if let Err(error) = owner.settle_completion(now()) {
                    owner.failed(&error);
                    mailbox.close(&error);
                    stopping = true;
                    continue;
                }
                admissions_to_drain = mailbox
                    .state
                    .lock()
                    .unwrap()
                    .controls
                    .iter()
                    .filter(|envelope| matches!(envelope.command, WorkerCommand::Admit(_)))
                    .count();
                drive = true;
            }
            Some(Event::Publication(request, wake)) => {
                if let Err(error) = owner.publication_wake(request, wake, now()) {
                    owner.failed(&error);
                    mailbox.close(&error);
                    stopping = true;
                }
                drive = true;
            }
            Some(Event::Cleanup(command)) => {
                if let Err(error) = owner.command(command, now()) {
                    owner.failed(&error);
                    mailbox.close(&error);
                    stopping = true;
                }
                drive = true;
            }
            Some(Event::Control(envelope)) => {
                if matches!(envelope.command, WorkerCommand::Admit(_)) {
                    admissions_to_drain = admissions_to_drain.saturating_sub(1);
                }
                let close_requested = matches!(envelope.command, WorkerCommand::Close);
                let result = match envelope.command {
                    WorkerCommand::Close => Ok(WorkerReply::Acknowledged),
                    command => {
                        match catch_unwind(AssertUnwindSafe(|| owner.command(command, now()))) {
                            Ok(result) => result,
                            Err(_) => {
                                let error = "execution command panicked".to_string();
                                owner.failed(&error);
                                mailbox.close(&error);
                                stopping = true;
                                Err(error)
                            }
                        }
                    }
                };
                let abandoned = deliver(&envelope.reply, result);
                if abandoned {
                    let admitted = {
                        let mut state = envelope.reply.state.lock().unwrap();
                        match state.result.take() {
                            Some(Ok(WorkerReply::Admitted { request, .. })) => Some(request),
                            _ => None,
                        }
                    };
                    if let Some(request) = admitted {
                        let _ = owner.command(WorkerCommand::Cancel { request }, now());
                    }
                    mailbox.release();
                }
                if close_requested {
                    mailbox.close("execution worker stopped");
                    stopping = true;
                }
                drive = true;
            }
            None => {}
        }
        if waiting.is_some() {
            continue;
        }
        if drive && admissions_to_drain == 0 {
            serial = match serial.checked_add(1) {
                Some(id) => id,
                None => {
                    owner.failed("completion identity exhausted");
                    break;
                }
            };
            let id = serial;
            let queue = mailbox.clone();
            let wake = CompletionWake::new(move || {
                let mut state = queue.state.lock().unwrap();
                state.completion = Some(id);
                queue.changed.notify_one();
            });
            match owner.advance(now(), wake) {
                Ok(Drive::Idle) => drive = false,
                Ok(Drive::Progress) => drive = true,
                Ok(Drive::AwaitingCompletion) => {
                    waiting = Some(id);
                    drive = false;
                }
                Err(error) => {
                    owner.failed(&error);
                    mailbox.close(&error);
                    stopping = true;
                    drive = false;
                }
            }
        }
    }
}

#[cfg(test)]
mod overlap_tests {
    use super::*;
    use magnitude_family_contracts::TokenPlan;
    use magnitude_generation::{
        EndOfGeneration, GenerationSeed, InputLayout, MethodChoice, Options, Sampling, Shaping,
        TokenId,
    };
    use std::collections::BTreeSet;
    use std::sync::mpsc;
    use std::time::Duration;

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

    struct Idle;

    impl Driven for Idle {
        fn advance(&mut self, _: u64, _: CompletionWake) -> Result<Drive, String> {
            Ok(Drive::Idle)
        }
        fn failed(&mut self, _: &str) {}
        fn failure(&self) -> Option<&RequestError> {
            None
        }
        fn shutdown(&mut self) -> Result<bool, String> {
            Ok(true)
        }
    }

    #[test]
    fn host_construction_runs_before_worker_readiness() {
        let (started, observe_start) = mpsc::sync_channel(1);
        let (finish, allow_finish) = mpsc::sync_channel(1);
        let (mut worker, ready, host) = Worker::spawn_ready_with::<_, _, String>(
            move || {
                started.send(()).unwrap();
                allow_finish
                    .recv_timeout(Duration::from_secs(2))
                    .map_err(|error| error.to_string())?;
                Ok((Box::new(Idle) as Box<dyn Driven>, 7))
            },
            1,
            move || {
                observe_start
                    .recv_timeout(Duration::from_secs(2))
                    .map_err(|error| error.to_string())?;
                finish.send(()).unwrap();
                Ok(9)
            },
        )
        .unwrap();
        assert_eq!((ready, host), (7, 9));
        worker.close();
    }

    #[test]
    fn idle_owner_gets_periodic_retry_without_a_command() {
        struct Retry {
            attempts: usize,
            retried: mpsc::SyncSender<()>,
        }
        impl Driven for Retry {
            fn advance(&mut self, _: u64, _: CompletionWake) -> Result<Drive, String> {
                self.attempts += 1;
                if self.attempts == 2 {
                    self.retried.send(()).unwrap();
                }
                Ok(Drive::Idle)
            }
            fn failed(&mut self, _: &str) {}
            fn failure(&self) -> Option<&RequestError> {
                None
            }
            fn shutdown(&mut self) -> Result<bool, String> {
                Ok(true)
            }
        }
        let (retried, observed) = mpsc::sync_channel(1);
        let mut worker = Worker::spawn(
            move || {
                Ok(Box::new(Retry {
                    attempts: 0,
                    retried,
                }))
            },
            1,
        )
        .unwrap();
        observed.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.close();
    }

    #[test]
    fn observes_periodically_while_a_submission_is_outstanding() {
        struct InFlight {
            observed: mpsc::Sender<()>,
        }
        impl Driven for InFlight {
            fn advance(&mut self, _: u64, _: CompletionWake) -> Result<Drive, String> {
                Ok(Drive::AwaitingCompletion)
            }
            fn periodic(&mut self, _: u64) -> Result<(), String> {
                self.observed.send(()).unwrap();
                Ok(())
            }
            fn failed(&mut self, _: &str) {}
            fn failure(&self) -> Option<&RequestError> {
                None
            }
            fn shutdown(&mut self) -> Result<bool, String> {
                Ok(true)
            }
        }
        let (observed, receiver) = mpsc::channel();
        let mut worker = Worker::spawn(move || Ok(Box::new(InFlight { observed })), 1).unwrap();
        receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.close();
    }

    #[test]
    fn admission_waits_for_a_submitted_flight_while_status_remains_available() {
        struct InFlight {
            wake: mpsc::Sender<CompletionWake>,
            admitted: mpsc::Sender<()>,
            submitted: bool,
            settled: bool,
            admissions: usize,
        }
        impl Driven for InFlight {
            fn command(&mut self, command: WorkerCommand, _: u64) -> Result<WorkerReply, String> {
                match command {
                    WorkerCommand::Admit(_) => {
                        assert!(self.settled);
                        self.admissions += 1;
                        self.admitted.send(()).unwrap();
                        Ok(WorkerReply::Acknowledged)
                    }
                    WorkerCommand::Status { .. } => Ok(WorkerReply::Status(None)),
                    _ => Ok(WorkerReply::Acknowledged),
                }
            }
            fn advance(&mut self, _: u64, wake: CompletionWake) -> Result<Drive, String> {
                if self.submitted {
                    assert_eq!(self.admissions, 2);
                    Ok(Drive::Idle)
                } else {
                    self.submitted = true;
                    self.wake.send(wake).unwrap();
                    Ok(Drive::AwaitingCompletion)
                }
            }
            fn settle_completion(&mut self, _: u64) -> Result<(), String> {
                self.settled = true;
                Ok(())
            }
            fn failed(&mut self, _: &str) {}
            fn failure(&self) -> Option<&RequestError> { None }
            fn shutdown(&mut self) -> Result<bool, String> { Ok(true) }
        }
        let (wake_sender, wake_receiver) = mpsc::channel();
        let (admitted_sender, admitted_receiver) = mpsc::channel();
        let mut worker = Worker::spawn(
            move || Ok(Box::new(InFlight {
                wake: wake_sender,
                admitted: admitted_sender,
                submitted: false,
                settled: false,
                admissions: 0,
            })),
            3,
        ).unwrap();
        let wake = wake_receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        let client = worker.client();
        let admission = client.dispatch(admit_request()).unwrap();
        let second_admission = client.dispatch(admit_request()).unwrap();
        let status = client.dispatch(WorkerCommand::Status { request: RequestId(1) }).unwrap();
        assert!(matches!(status.wait().unwrap(), WorkerReply::Status(None)));
        assert!(admitted_receiver.try_recv().is_err());
        wake.complete();
        assert!(matches!(admission.wait().unwrap(), WorkerReply::Acknowledged));
        assert!(matches!(second_admission.wait().unwrap(), WorkerReply::Acknowledged));
        admitted_receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        admitted_receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.close();
    }

    #[test]
    fn admission_waits_while_advance_prepares_a_submission() {
        struct Preparing {
            entered: mpsc::Sender<()>,
            resume: mpsc::Receiver<()>,
            wake: mpsc::Sender<CompletionWake>,
            admitted: mpsc::Sender<()>,
            stage: usize,
            admissions: usize,
            settled: bool,
        }
        impl Driven for Preparing {
            fn command(&mut self, command: WorkerCommand, _: u64) -> Result<WorkerReply, String> {
                if matches!(command, WorkerCommand::Admit(_)) {
                    if self.admissions > 0 {
                        assert!(self.settled, "peer admitted during an active state transaction");
                    }
                    self.admissions += 1;
                    self.admitted.send(()).unwrap();
                }
                Ok(WorkerReply::Acknowledged)
            }
            fn advance(&mut self, _: u64, wake: CompletionWake) -> Result<Drive, String> {
                match self.stage {
                    0 => { self.stage = 1; Ok(Drive::Idle) }
                    1 => {
                        self.stage = 2;
                        self.entered.send(()).unwrap();
                        self.resume.recv().unwrap();
                        Ok(Drive::Progress)
                    }
                    2 => {
                        self.stage = 3;
                        self.wake.send(wake).unwrap();
                        Ok(Drive::AwaitingCompletion)
                    }
                    _ => Ok(Drive::Idle),
                }
            }
            fn settle_completion(&mut self, _: u64) -> Result<(), String> {
                self.settled = true;
                Ok(())
            }
            fn failed(&mut self, _: &str) {}
            fn failure(&self) -> Option<&RequestError> { None }
            fn shutdown(&mut self) -> Result<bool, String> { Ok(true) }
        }
        let (entered, entering) = mpsc::channel();
        let (resume, resumed) = mpsc::channel();
        let (wake, waking) = mpsc::channel();
        let (admitted, admissions) = mpsc::channel();
        let mut worker = Worker::spawn(
            move || Ok(Box::new(Preparing {
                entered, resume: resumed, wake, admitted,
                stage: 0, admissions: 0, settled: false,
            })),
            2,
        ).unwrap();
        let client = worker.client();
        assert!(matches!(client.dispatch(admit_request()).unwrap().wait().unwrap(), WorkerReply::Acknowledged));
        entering.recv_timeout(Duration::from_secs(2)).unwrap();
        let peer = client.dispatch(admit_request()).unwrap();
        resume.send(()).unwrap();
        let completion = waking.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(admissions.recv_timeout(Duration::from_secs(2)).is_ok());
        assert!(admissions.try_recv().is_err());
        completion.complete();
        assert!(matches!(peer.wait().unwrap(), WorkerReply::Acknowledged));
        admissions.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.close();
    }
}
