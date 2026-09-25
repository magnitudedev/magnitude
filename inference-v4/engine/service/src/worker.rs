//! Thread-confined execution behind one closed, device-free protocol.

use crate::{
    protocol::{WorkerCommand, WorkerReply},
    publication::PublicationWake,
};
pub use magnitude_model_executor::CompletionWake;
use magnitude_model_executor::RequestId;
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
    /// Runs on the worker thread even while a submitted flight is outstanding.
    /// The owner must not release storage held by that flight here.
    fn periodic(&mut self, _now: u64) -> Result<(), String> {
        Ok(())
    }
    fn failed(&mut self, error: &str);
    fn failure(&self) -> Option<&str>;
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
        let (worker, ready, ()) = Self::spawn_ready_with(factory, capacity, || Ok(()))?;
        Ok((worker, ready))
    }

    /// Construct host artifacts on the caller while the execution thread
    /// builds its domain, then wait for both sides before publishing readiness.
    pub fn spawn_ready_with<R: Send + 'static, H>(
        factory: impl FnOnce() -> Result<(Box<dyn Driven>, R), String> + Send + 'static,
        capacity: usize,
        host: impl FnOnce() -> Result<H, String>,
    ) -> Result<(Self, R, H), String> {
        if capacity == 0 {
            return Err("execution control capacity must be positive".into());
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
                    .unwrap_or_else(|_| Err("execution factory panicked".into()));
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
                        queue.close(owner.failure().unwrap_or("execution worker stopped"));
                    }
                    Err(error) => {
                        queue.state.lock().unwrap().terminated = true;
                        queue.close(&error);
                        let _ = ready.send(Err(error));
                    }
                }
            })
            .map_err(|error| error.to_string())?;
        let worker = Self {
            mailbox,
            thread: Some(thread),
        };
        let host = host()?;
        let readiness = receive.recv().map_err(|error| error.to_string())??;
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
    pub fn cancel(&self, request: magnitude_model_executor::RequestId) {
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
    let mut serial = 0u64;
    let mut drive = true;
    let mut stopping = false;
    let observation_interval = Duration::from_millis(100);
    let mut next_observation = Instant::now() + observation_interval;
    loop {
        if !stopping {
            if let Some(error) = owner.failure().map(str::to_owned) {
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
                if let Some(envelope) = state.controls.pop_front() {
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
        if drive {
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
    use std::sync::mpsc;
    use std::time::Duration;

    struct Idle;

    impl Driven for Idle {
        fn advance(&mut self, _: u64, _: CompletionWake) -> Result<Drive, String> {
            Ok(Drive::Idle)
        }
        fn failed(&mut self, _: &str) {}
        fn failure(&self) -> Option<&str> {
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
        let (mut worker, ready, host) = Worker::spawn_ready_with(
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
            fn failure(&self) -> Option<&str> {
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
            fn failure(&self) -> Option<&str> {
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
}
