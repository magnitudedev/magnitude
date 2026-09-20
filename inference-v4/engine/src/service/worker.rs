//! Thread-confined live execution with bounded control and reserved lifecycle wakes.
use std::{
    collections::VecDeque,
    future::Future,
    panic::{catch_unwind, AssertUnwindSafe},
    pin::Pin,
    sync::{Arc, Condvar, Mutex},
    task::{Context, Poll, Waker},
    thread::{self, JoinHandle},
    time::Instant,
};

pub enum Drive {
    Idle,
    Progress,
    AwaitingCompletion,
}
/// Construct implementations on the worker; they need not be Send.
pub trait Driven {
    fn advance(&mut self, now: u64, wake: CompletionWake) -> Result<Drive, String>;
    fn failed(&mut self, error: &str);
    fn failure(&self) -> Option<&str>;
    /// Cancel future progress and discard caller-abandoned output. Return true
    /// only after outstanding work has reconciled and owned requests are closed.
    fn shutdown(&mut self) -> Result<bool, String>;
}
/// Exactly one wake for an outstanding physical completion, independent of the
/// bounded control queue. Native completion owners may move this across threads.
pub struct CompletionWake(Option<Box<dyn FnOnce() + Send>>);
impl CompletionWake {
    pub fn complete(mut self) {
        self.0.take().expect("one completion wake")();
    }
}
trait Job<T>: Send {
    fn run(self: Box<Self>, owner: &mut T, now: u64) -> Result<(), String>;
    fn reject(self: Box<Self>, error: &str);
}
struct Inbox<T> {
    controls: VecDeque<Box<dyn Job<T>>>,
    cleanup: VecDeque<Box<dyn Job<T>>>,
    completion: Option<u64>,
    stop: bool,
    closed: Option<String>,
    terminated: bool,
    outstanding: usize,
}
struct Mailbox<T> {
    state: Mutex<Inbox<T>>,
    changed: Condvar,
    capacity: usize,
}
impl<T: 'static> Mailbox<T> {
    fn release(&self) {
        self.state.lock().unwrap().outstanding -= 1;
    }
    fn close(&self, error: &str) {
        let jobs = {
            let mut state = self.state.lock().unwrap();
            state.closed.get_or_insert_with(|| error.into());
            state.stop = true;
            std::mem::take(&mut state.controls)
        };
        self.changed.notify_one();
        for job in jobs {
            job.reject(error);
        }
    }
    fn cleanup(&self, job: Box<dyn Job<T>>) {
        let mut state = self.state.lock().unwrap();
        if state.terminated {
            drop(state);
            // The owner has already disposed its complete resource domain.
            job.reject("execution worker stopped");
            return;
        }
        // Even during shutdown, cleanup stays owner-local until final disposal.
        state.cleanup.push_back(job);
        self.changed.notify_one();
    }
}
type Cleanup<T, R> = Box<dyn FnOnce(&mut T, R) -> Result<(), String> + Send>;
struct Lifecycle<T>(Box<dyn FnOnce(&mut T) -> Result<(), String> + Send>);
impl<T: 'static> Job<T> for Lifecycle<T> {
    fn run(self: Box<Self>, owner: &mut T, _: u64) -> Result<(), String> {
        catch_unwind(AssertUnwindSafe(|| (self.0)(owner)))
            .map_err(|_| "execution lifecycle cleanup panicked".to_string())?
    }
    fn reject(self: Box<Self>, _: &str) {}
}
struct ReplyState<T, R> {
    abandoned: bool,
    result: Option<Result<R, String>>,
    cleanup: Option<Cleanup<T, R>>,
    waker: Option<Waker>,
}
struct Reply<T, R> {
    state: Mutex<ReplyState<T, R>>,
    changed: Condvar,
}
/// A control result. Dropping it before execution skips the call. Dropping it
/// during or after execution runs its result cleanup on the owner thread.
pub struct Call<T: 'static, R: Send + 'static> {
    reply: Arc<Reply<T, R>>,
    mailbox: Arc<Mailbox<T>>,
    consumed: bool,
}
impl<T: 'static, R: Send + 'static> Call<T, R> {
    pub fn wait(mut self) -> Result<R, String> {
        let result = {
            let mut state = self.reply.state.lock().unwrap();
            while state.result.is_none() {
                state = self.reply.changed.wait(state).unwrap();
            }
            state.cleanup.take();
            state.result.take().unwrap()
        };
        self.consumed = true;
        self.mailbox.release();
        result
    }
}
impl<T: 'static, R: Send + 'static> Future for Call<T, R> {
    type Output = Result<R, String>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        assert!(!this.consumed, "control result polled after consumption");
        let result = {
            let mut state = this.reply.state.lock().unwrap();
            match state.result.take() {
                Some(result) => {
                    state.cleanup.take();
                    Some(result)
                }
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
impl<T: 'static, R: Send + 'static> Drop for Call<T, R> {
    fn drop(&mut self) {
        if self.consumed {
            return;
        }
        let completed = {
            let mut state = self.reply.state.lock().unwrap();
            state.abandoned = true;
            state.waker.take();
            state
                .result
                .take()
                .map(|result| (result, state.cleanup.take().unwrap()))
        };
        if let Some((result, cleanup)) = completed {
            match result {
                Ok(value) => self.mailbox.cleanup(Box::new(CleanupJob {
                    value,
                    cleanup,
                    mailbox: self.mailbox.clone(),
                })),
                Err(_) => self.mailbox.release(),
            }
        }
    }
}
struct CleanupJob<T, R> {
    value: R,
    cleanup: Cleanup<T, R>,
    mailbox: Arc<Mailbox<T>>,
}
impl<T: 'static, R: Send + 'static> Job<T> for CleanupJob<T, R> {
    fn run(self: Box<Self>, owner: &mut T, _: u64) -> Result<(), String> {
        let Self {
            value,
            cleanup,
            mailbox,
        } = *self;
        let result = catch_unwind(AssertUnwindSafe(|| cleanup(owner, value)));
        mailbox.release();
        result.map_err(|_| "execution result cleanup panicked".to_string())?
    }
    fn reject(self: Box<Self>, _: &str) {
        self.mailbox.release();
    }
}
struct Control<T, R> {
    action: Box<dyn FnOnce(&mut T, u64) -> Result<R, String> + Send>,
    reply: Arc<Reply<T, R>>,
    mailbox: Arc<Mailbox<T>>,
}
impl<T: 'static, R: Send + 'static> Control<T, R> {
    fn deliver(self, owner: Option<&mut T>, result: Result<R, String>) -> Result<(), String> {
        let (abandoned, waker) = {
            let mut state = self.reply.state.lock().unwrap();
            let abandoned = if state.abandoned {
                Some((result, state.cleanup.take().unwrap()))
            } else {
                state.result = Some(result);
                None
            };
            (abandoned, state.waker.take())
        };
        self.reply.changed.notify_one();
        if let Some(waker) = waker {
            waker.wake();
        }
        if let Some((result, cleanup)) = abandoned {
            let cleaned = catch_unwind(AssertUnwindSafe(|| {
                if let (Some(owner), Ok(value)) = (owner, result) {
                    cleanup(owner, value)
                } else {
                    Ok(())
                }
            }));
            self.mailbox.release();
            cleaned.map_err(|_| "execution result cleanup panicked".to_string())??;
        }
        Ok(())
    }
}
impl<T: 'static, R: Send + 'static> Job<T> for Control<T, R> {
    fn run(mut self: Box<Self>, owner: &mut T, now: u64) -> Result<(), String> {
        if self.reply.state.lock().unwrap().abandoned {
            self.mailbox.release();
            return Ok(());
        }
        let action = std::mem::replace(&mut self.action, Box::new(|_, _| unreachable!()));
        let result = catch_unwind(AssertUnwindSafe(|| action(owner, now)));
        match result {
            Ok(result) => self.deliver(Some(owner), result),
            Err(_) => {
                let error = "execution control call panicked".to_string();
                self.deliver(None, Err(error.clone()))?;
                Err(error)
            }
        }
    }
    fn reject(self: Box<Self>, error: &str) {
        let _ = self.deliver(None, Err(error.into()));
    }
}

pub struct Worker<T: Driven + 'static> {
    mailbox: Arc<Mailbox<T>>,
    thread: Option<JoinHandle<()>>,
}
impl<T: Driven + 'static> Worker<T> {
    pub fn spawn(
        factory: impl FnOnce() -> Result<T, String> + Send + 'static,
        capacity: usize,
    ) -> Result<Self, String> {
        if capacity == 0 {
            return Err("execution control capacity must be positive".into());
        }
        let mailbox = Arc::new(Mailbox {
            state: Mutex::new(Inbox {
                controls: VecDeque::new(),
                cleanup: VecDeque::new(),
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
                    Ok(mut owner) => {
                        let _ = ready.send(Ok(()));
                        let result = catch_unwind(AssertUnwindSafe(|| run(&mut owner, &queue)));
                        if result.is_err() {
                            owner.failed("execution owner panicked");
                        }
                        queue.close(owner.failure().unwrap_or("execution worker stopped"));
                        let cleanup = {
                            let mut state = queue.state.lock().unwrap();
                            state.terminated = true;
                            std::mem::take(&mut state.cleanup)
                        };
                        for job in cleanup {
                            if let Err(error) = job.run(&mut owner, 0) {
                                owner.failed(&error);
                            }
                        }
                    }
                    Err(error) => {
                        queue.close(&error);
                        queue.state.lock().unwrap().terminated = true;
                        let _ = ready.send(Err(error));
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        match receive.recv().map_err(|e| e.to_string())? {
            Ok(()) => Ok(Self {
                mailbox,
                thread: Some(thread),
            }),
            Err(error) => {
                let _ = thread.join();
                Err(error)
            }
        }
    }
    pub fn client(&self) -> Client<T> {
        Client {
            mailbox: self.mailbox.clone(),
        }
    }
    pub fn close(&mut self) {
        self.mailbox.close("execution worker stopped");
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
/// Cloneable admission/control capability; dropping a client does not join the
/// execution thread. The host composition root retains the Worker lifecycle.
pub struct Client<T: 'static> {
    mailbox: Arc<Mailbox<T>>,
}
impl<T: 'static> Clone for Client<T> {
    fn clone(&self) -> Self {
        Self {
            mailbox: self.mailbox.clone(),
        }
    }
}
impl<T: 'static> Client<T> {
    pub(crate) fn is_closed(&self) -> bool {
        self.mailbox.state.lock().unwrap().closed.is_some()
    }
    /// Reserved for release of an existing bounded service resource. Ordinary
    /// control work must use `call`; request handles enqueue this at most once.
    pub(crate) fn release(
        &self,
        action: impl FnOnce(&mut T) -> Result<(), String> + Send + 'static,
    ) {
        self.mailbox.cleanup(Box::new(Lifecycle(Box::new(action))));
    }
    pub fn call<R: Send + 'static>(
        &self,
        action: impl FnOnce(&mut T, u64) -> Result<R, String> + Send + 'static,
    ) -> Result<Call<T, R>, String> {
        self.call_with_cleanup(action, |_, _| Ok(()))
    }
    pub fn call_with_cleanup<R: Send + 'static>(
        &self,
        action: impl FnOnce(&mut T, u64) -> Result<R, String> + Send + 'static,
        cleanup: impl FnOnce(&mut T, R) -> Result<(), String> + Send + 'static,
    ) -> Result<Call<T, R>, String> {
        let reply = Arc::new(Reply {
            state: Mutex::new(ReplyState {
                abandoned: false,
                result: None,
                cleanup: Some(Box::new(cleanup)),
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
        state.controls.push_back(Box::new(Control {
            action: Box::new(action),
            reply: reply.clone(),
            mailbox: self.mailbox.clone(),
        }));
        self.mailbox.changed.notify_one();
        Ok(Call {
            reply,
            mailbox: self.mailbox.clone(),
            consumed: false,
        })
    }
}
impl<T: Driven + 'static> Drop for Worker<T> {
    fn drop(&mut self) {
        self.close();
    }
}

fn run<T: Driven + 'static>(owner: &mut T, mailbox: &Arc<Mailbox<T>>) {
    enum Event<T> {
        Completion(u64),
        Stop,
        Job(Box<dyn Job<T>>),
    }
    let start = Instant::now();
    let now = || start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    let mut waiting = None;
    let mut serial = 0u64;
    let mut drive = true;
    let mut stopping = false;
    loop {
        if !stopping {
            if let Some(error) = owner.failure().map(str::to_string) {
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
                    mailbox.close(&error);
                    break;
                }
            }
        }
        let event = {
            let mut state = mailbox.state.lock().unwrap();
            loop {
                if let Some(id) = state.completion.take() {
                    break Some(Event::Completion(id));
                }
                if state.stop {
                    state.stop = false;
                    break Some(Event::Stop);
                }
                if let Some(job) = state
                    .cleanup
                    .pop_front()
                    .or_else(|| state.controls.pop_front())
                {
                    break Some(Event::Job(job));
                }
                if drive && waiting.is_none() {
                    break None;
                }
                state = mailbox.changed.wait(state).unwrap();
            }
        };
        match event {
            Some(Event::Stop) => {
                stopping = true;
                continue;
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
            Some(Event::Job(job)) => {
                if let Err(error) = job.run(owner, now()) {
                    owner.failed(&error);
                    mailbox.close(&error);
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
            let wake = CompletionWake(Some(Box::new(move || {
                let mut state = queue.state.lock().unwrap();
                state.completion = Some(id);
                queue.changed.notify_one();
            })));
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
