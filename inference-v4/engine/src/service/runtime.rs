//! Prepared-input admission and event-driven publication over the execution owner.
use super::{
    owner::{Executor, InputExecutor, Owner, Status},
    policy::{Limits, RequestId},
    worker::{self, CompletionWake, Drive, Driven, Worker},
};
use crate::{
    chat::PreparedInput,
    generation::{constraints::Vocabulary, FinishReason, Options, OutputToken, Usage},
};
use std::{
    collections::{BTreeMap, VecDeque},
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll, Waker},
};

struct Closed {
    usage: Option<Usage>,
    tokens: VecDeque<OutputToken>,
    finish: FinishReason,
    error: Option<String>,
}
#[derive(Default)]
struct Notice {
    revision: u64,
    closed: Option<Closed>,
    waker: Option<Waker>,
}
#[derive(Default)]
struct Signal(Mutex<Notice>);
impl Signal {
    fn revision(&self) -> u64 {
        self.0.lock().unwrap().revision
    }
    fn is_closed(&self) -> bool {
        self.0.lock().unwrap().closed.is_some()
    }
    fn take_closed(&self, count: usize) -> Option<Publication> {
        let mut notice = self.0.lock().unwrap();
        let closed = notice.closed.as_mut()?;
        let tokens = closed
            .tokens
            .drain(..count.min(closed.tokens.len()))
            .collect();
        Some(Publication {
            tokens,
            finish: closed.tokens.is_empty().then_some(closed.finish),
            error: closed.error.clone(),
            usage: closed.usage,
        })
    }
    fn notify(&self) {
        let waker = {
            let mut notice = self.0.lock().unwrap();
            notice.revision = notice.revision.wrapping_add(1);
            notice.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
    fn close(
        &self,
        tokens: Vec<OutputToken>,
        finish: FinishReason,
        error: Option<String>,
        usage: Option<Usage>,
    ) {
        let waker = {
            let mut notice = self.0.lock().unwrap();
            notice.closed.get_or_insert(Closed {
                usage,
                tokens: tokens.into(),
                finish,
                error,
            });
            notice.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}
struct Changed<'a> {
    signal: &'a Signal,
    revision: u64,
}
impl Future for Changed<'_> {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut notice = self.signal.0.lock().unwrap();
        if notice.closed.is_some() || notice.revision != self.revision {
            return Poll::Ready(());
        }
        notice.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}
impl Drop for Changed<'_> {
    fn drop(&mut self) {
        self.signal.0.lock().unwrap().waker.take();
    }
}

pub struct Runtime<E: Executor> {
    owner: Owner<E>,
    vocabulary: Vocabulary,
    signals: BTreeMap<RequestId, Arc<Signal>>,
}
struct Admitted {
    id: RequestId,
    signal: Arc<Signal>,
}
#[derive(Debug)]
pub struct Publication {
    pub usage: Option<Usage>,
    pub tokens: Vec<OutputToken>,
    /// Terminal only after all retained output has been collected.
    pub finish: Option<FinishReason>,
    pub error: Option<String>,
}
impl<E: Executor> Runtime<E> {
    pub fn new(executor: E, vocabulary: Vocabulary, limits: Limits) -> Result<Self, String> {
        Ok(Self {
            owner: Owner::new(executor, limits)?,
            vocabulary,
            signals: BTreeMap::new(),
        })
    }
    fn admit_input<I>(
        &mut self,
        input: &PreparedInput,
        source: I,
        options: Options,
        now: u64,
    ) -> Result<Admitted, String>
    where
        E: InputExecutor<I>,
    {
        let layout = self.owner.input_layout(&source, &input.tokens)?;
        let generation = self
            .vocabulary
            .prepare_generation_with_layout(input, options, layout)?;
        let id = self.owner.admit_input(generation, source, now)?;
        Ok(self.register(id))
    }
    fn register(&mut self, id: RequestId) -> Admitted {
        let signal = Arc::new(Signal::default());
        self.signals.insert(id, signal.clone());
        Admitted { id, signal }
    }
    fn admit(
        &mut self,
        input: &PreparedInput,
        options: Options,
        now: u64,
    ) -> Result<Admitted, String> {
        let generation = self.vocabulary.prepare_generation(input, options)?;
        let id = self.owner.admit(generation, now)?;
        Ok(self.register(id))
    }
    fn fork_checkpoint(
        &mut self,
        checkpoint: super::owner::CheckpointId,
        now: u64,
    ) -> Result<Admitted, String> {
        let id = self.owner.fork_checkpoint(checkpoint, now)?;
        let signal = Arc::new(Signal::default());
        self.signals.insert(id, signal.clone());
        Ok(Admitted { id, signal })
    }
    fn release(&mut self, id: RequestId) -> Result<(), String> {
        let Some(signal) = self.signals.remove(&id) else {
            return self.owner.release(id);
        };
        let usage = self.owner.usage(id).ok();
        let finish = match self.owner.status(id) {
            Ok(Status::Terminal(reason)) => reason,
            _ => FinishReason::Cancelled,
        };
        let error = self.owner.error(id).map(str::to_string);
        let result = self.owner.release(id);
        match &result {
            Ok(()) => signal.close(Vec::new(), finish, error, usage),
            Err(error) => {
                signal.close(Vec::new(), FinishReason::Failed, Some(error.clone()), usage)
            }
        }
        result
    }
    fn receive(&mut self, id: RequestId, count: usize) -> Result<Publication, String> {
        let tokens = self.owner.take(id, count)?;
        let finish = match self.owner.status(id)? {
            Status::Terminal(reason) if self.owner.output_len(id)? == 0 => Some(reason),
            _ => None,
        };
        Ok(Publication {
            tokens,
            finish,
            error: self.owner.error(id).map(str::to_string),
            usage: Some(self.owner.usage(id)?),
        })
    }
    fn close_publication(&mut self, fallback: FinishReason, error: Option<String>) {
        for (&id, signal) in &self.signals {
            if signal.is_closed() {
                continue;
            }
            let finish = match self.owner.status(id) {
                Ok(Status::Terminal(reason)) => reason,
                _ => fallback,
            };
            let error = if finish == FinishReason::Failed {
                self.owner
                    .error(id)
                    .map(str::to_string)
                    .or_else(|| error.clone())
            } else {
                None
            };
            match self.owner.take(id, usize::MAX) {
                Ok(tokens) => signal.close(tokens, finish, error, self.owner.usage(id).ok()),
                Err(error) => signal.close(
                    Vec::new(),
                    FinishReason::Failed,
                    Some(error),
                    self.owner.usage(id).ok(),
                ),
            }
        }
    }
    fn publish(&self) {
        for (&id, signal) in &self.signals {
            if self.owner.output_len(id).unwrap_or(0) > 0
                || matches!(self.owner.status(id), Ok(Status::Terminal(_)))
            {
                signal.notify();
            }
        }
    }
}
impl<E: Executor> Driven for Runtime<E> {
    fn advance(&mut self, now: u64, wake: CompletionWake) -> Result<Drive, String> {
        let result = self.owner.advance(now, wake);
        self.publish();
        result
    }
    fn failed(&mut self, error: &str) {
        self.owner.failed(error);
        self.close_publication(FinishReason::Failed, Some(error.into()));
    }
    fn failure(&self) -> Option<&str> {
        self.owner.failure()
    }
    fn shutdown(&mut self) -> Result<bool, String> {
        let error = self.owner.failure().map(str::to_string);
        let reason = if error.is_some() {
            FinishReason::Failed
        } else {
            FinishReason::Cancelled
        };
        self.close_publication(reason, error);
        self.owner.shutdown()
    }
}

/// Host composition root: the factory loads the vocabulary/executor on its
/// owning thread. Cloned clients never own or join that thread.
pub struct Service<E: Executor + 'static>(Worker<Runtime<E>>);
impl<E: Executor + 'static> Service<E> {
    pub fn spawn(
        factory: impl FnOnce() -> Result<Runtime<E>, String> + Send + 'static,
        control_capacity: usize,
    ) -> Result<Self, String> {
        Ok(Self(Worker::spawn(factory, control_capacity)?))
    }
    pub fn client(&self) -> Client<E> {
        Client(self.0.client())
    }
    pub fn close(&mut self) {
        self.0.close();
    }
}
pub struct Client<E: Executor + 'static>(worker::Client<Runtime<E>>);
impl<E: Executor + 'static> Clone for Client<E> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<E: Executor + 'static> Client<E> {
    pub async fn check(&self) -> Result<(), String> {
        self.0
            .call(|runtime, _| match runtime.owner.fatal_error() {
                Some(error) => Err(error.to_string()),
                None => Ok(()),
            })?
            .await
    }
    /// Only immutable, Send host preparation crosses this boundary. Encoder,
    /// feature, matcher, and sequence owners are created on the service worker.
    pub async fn admit_input<I: Send + 'static>(
        &self,
        input: PreparedInput,
        source: I,
        options: Options,
    ) -> Result<Request<E>, String>
    where
        E: InputExecutor<I>,
    {
        let admitted = self
            .0
            .call_with_cleanup(
                move |runtime, now| runtime.admit_input(&input, source, options, now),
                |runtime, admitted| runtime.release(admitted.id),
            )?
            .await?;
        Ok(Request {
            client: self.0.clone(),
            admitted,
            released: Arc::new(AtomicBool::new(false)),
        })
    }
    pub async fn admit(
        &self,
        input: PreparedInput,
        options: Options,
    ) -> Result<Request<E>, String> {
        let admitted = self
            .0
            .call_with_cleanup(
                move |runtime, now| runtime.admit(&input, options, now),
                |runtime, admitted| runtime.release(admitted.id),
            )?
            .await?;
        Ok(Request {
            client: self.0.clone(),
            admitted,
            released: Arc::new(AtomicBool::new(false)),
        })
    }
}
/// Unique publication receiver. Dropping it releases the request through a
/// reserved lifecycle path, including when the control queue is saturated.
pub struct Request<E: Executor + 'static> {
    client: worker::Client<Runtime<E>>,
    admitted: Admitted,
    released: Arc<AtomicBool>,
}
impl<E: Executor + 'static> Request<E> {
    pub fn id(&self) -> RequestId {
        self.admitted.id
    }
    /// Capture only reconciled resident state. A pending numerical advance is
    /// rejected rather than implicitly waiting or changing publication credit.
    pub async fn checkpoint(&self) -> Result<Checkpoint<E>, String> {
        if self.released.load(Ordering::Acquire) {
            return Err("request released".into());
        }
        let request = self.admitted.id;
        let id = self
            .client
            .call_with_cleanup(
                move |runtime, _| runtime.owner.checkpoint(request),
                |runtime, id| runtime.owner.release_checkpoint(id),
            )?
            .await?;
        Ok(Checkpoint {
            client: self.client.clone(),
            id,
        })
    }
    /// Stop through reserved lifecycle delivery and await the owner's final
    /// accepted-token snapshot. Pending native work remains owned but cannot
    /// accept additional output after the acknowledgement.
    pub async fn stop(&mut self) -> Result<Publication, String> {
        if !self.released.swap(true, Ordering::AcqRel) {
            let id = self.admitted.id;
            self.client.release(move |runtime| runtime.release(id));
        }
        loop {
            let revision = self.admitted.signal.revision();
            if let Some(mut publication) = self.admitted.signal.take_closed(usize::MAX) {
                publication.tokens.clear();
                return Ok(publication);
            }
            Changed {
                signal: &self.admitted.signal,
                revision,
            }
            .await;
        }
    }
    /// Cancelling this future cancels the request; accepted output then has an
    /// explicit discard owner rather than becoming an unclaimed control result.
    pub async fn receive(&mut self, count: usize) -> Result<Publication, String> {
        if count == 0 {
            return Err("publication count must be positive".into());
        }
        if self.released.load(Ordering::Acquire) {
            return Err("request released".into());
        }
        let id = self.admitted.id;
        let mut guard = ReceiveGuard {
            client: self.client.clone(),
            id,
            armed: true,
            released: self.released.clone(),
        };
        loop {
            let revision = self.admitted.signal.revision();
            if let Some(publication) = self.admitted.signal.take_closed(count) {
                guard.armed = false;
                return Ok(publication);
            }
            let result = match self
                .client
                .call(move |runtime, _| runtime.receive(id, count))
            {
                Ok(call) => call.await,
                Err(error) => Err(error),
            };
            match result {
                Ok(publication)
                    if !publication.tokens.is_empty() || publication.finish.is_some() =>
                {
                    guard.armed = false;
                    return Ok(publication);
                }
                Ok(_) => {}
                // Closing rejects queued controls before the owner can publish
                // retained output. Await that publication instead of losing it.
                Err(_) if self.client.is_closed() => {}
                Err(error) => return Err(error),
            }
            Changed {
                signal: &self.admitted.signal,
                revision,
            }
            .await;
        }
    }
}
/// Opaque owner-scoped snapshot. Live matcher and numerical objects remain on
/// the worker. Dropping the handle schedules reserved snapshot release.
pub struct Checkpoint<E: Executor + 'static> {
    client: worker::Client<Runtime<E>>,
    id: super::owner::CheckpointId,
}
impl<E: Executor + 'static> Checkpoint<E> {
    pub async fn fork(&self) -> Result<Request<E>, String> {
        let id = self.id;
        let admitted = self
            .client
            .call_with_cleanup(
                move |runtime, now| runtime.fork_checkpoint(id, now),
                |runtime, admitted| runtime.release(admitted.id),
            )?
            .await?;
        Ok(Request {
            client: self.client.clone(),
            admitted,
            released: Arc::new(AtomicBool::new(false)),
        })
    }
}
impl<E: Executor + 'static> Drop for Checkpoint<E> {
    fn drop(&mut self) {
        let id = self.id;
        self.client
            .release(move |runtime| runtime.owner.release_checkpoint(id));
    }
}
struct ReceiveGuard<E: Executor + 'static> {
    client: worker::Client<Runtime<E>>,
    id: RequestId,
    armed: bool,
    released: Arc<AtomicBool>,
}
impl<E: Executor + 'static> Drop for ReceiveGuard<E> {
    fn drop(&mut self) {
        if self.armed && !self.released.swap(true, Ordering::AcqRel) {
            let id = self.id;
            self.client.release(move |runtime| runtime.release(id));
        }
    }
}
impl<E: Executor + 'static> Drop for Request<E> {
    fn drop(&mut self) {
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        let id = self.admitted.id;
        self.client.release(move |runtime| runtime.release(id));
    }
}
