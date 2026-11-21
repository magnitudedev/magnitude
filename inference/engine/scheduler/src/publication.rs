//! One bounded, ordered outcome stream for a request.
//!
//! The data ring and terminal slot have separate capacity. The receiver drains
//! accepted data before observing the terminal event. Credit and cancellation
//! use a reserved wake supplied by the worker mailbox, so neither depends on
//! admission-command capacity.

use magnitude_generation::{DetailedUsage, FinishReason, OutputToken};
use magnitude_executor::{DeviceError, InvariantError, ResourceKind, WeightReadError};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll, Waker},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CapacityResource {
    Execution(ResourceKind),
    LiveSequence,
    SubmittedSuccessor,
    BranchCheckpoint,
    RetainedPrefix,
    RetainedMethodFeature,
    RetainedMediaFeature,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceCapacityError {
    pub resource: CapacityResource,
    pub required: u64,
    pub available: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelUnloadCause {
    MemoryPressure,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestError {
    Input(String),
    State(magnitude_state::Error),
    Capacity(ServiceCapacityError),
    /// The request's state needs `required` bytes of memory where nothing
    /// else the load holds can be freed for it: alone, it exceeds what the
    /// loaded model can ever give it, `available`.
    ContextExceedsMemory { required: u64, available: u64 },
    Device(DeviceError),
    Invariant(InvariantError),
    /// The request's weights (table rows or experts) could not be read from
    /// the model's artifact; other requests continue.
    WeightRead(WeightReadError),
    WorkerClosed,
    ModelUnloaded { cause: ModelUnloadCause },
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(reason) => write!(formatter, "invalid input: {reason}"),
            Self::State(error) => write!(formatter, "state failure: {error}"),
            Self::Capacity(capacity) => write!(
                formatter,
                "{:?} capacity exhausted: {} required, {} available",
                capacity.resource, capacity.required, capacity.available
            ),
            Self::ContextExceedsMemory { required, available } => write!(
                formatter,
                "the request's context needs {required} bytes of memory, past the {available} the loaded model can give it"
            ),
            Self::Device(error) => error.fmt(formatter),
            Self::Invariant(error) => error.fmt(formatter),
            Self::WeightRead(error) => error.fmt(formatter),
            Self::WorkerClosed => formatter.write_str("the execution worker closed"),
            Self::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            } => formatter.write_str("model unloaded due to memory pressure"),
        }
    }
}

impl std::error::Error for RequestError {}

/// A completed request's wall-clock timing, as llama.cpp's server reports a
/// slot's: `prompt_ns` from its first service to its first accepted output
/// token, `predicted_ns` from there to its last.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequestTimings {
    pub prompt_ns: u64,
    pub predicted_ns: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Publication {
    Output(Vec<OutputToken>),
    Completed {
        finish: FinishReason,
        usage: DetailedUsage,
        method: String,
        timings: RequestTimings,
    },
    Failed(RequestError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicationWakeKind {
    OutputCredit,
    Cancelled,
}

/// An opaque reserved wake. The host passes it through an independent,
/// nonblocking worker wake path; only the matching worker sender consumes it.
pub struct PublicationWake {
    kind: PublicationWakeKind,
    queue_id: u64,
    credit_epoch: u64,
}

impl PublicationWake {
    pub fn kind(&self) -> PublicationWakeKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublishError {
    Full,
    ExceedsCapacity,
    ReceiverClosed,
}

enum ChannelState {
    Open { receiver_waker: Option<Waker> },
    SenderClosed(Publication),
    ReceiverClosed,
}

enum RetirementState {
    Active(Option<Waker>),
    Retired,
}

struct State {
    data: VecDeque<Vec<OutputToken>>,
    capacity: usize,
    reserved: usize,
    reservation_waiting: bool,
    credit: Option<u64>,
    next_credit_epoch: u64,
    channel: ChannelState,
    retirement: RetirementState,
    /// The request became resident: it began, before anything it publishes.
    began: bool,
}

impl State {
    fn has_capacity(&self) -> bool {
        matches!(self.channel, ChannelState::Open { .. })
            && self.data.len().saturating_add(self.reserved) < self.capacity
    }

    fn try_issue_credit(&mut self) -> Option<u64> {
        if matches!(self.channel, ChannelState::Open { .. }) && self.credit.is_none() {
            self.reservation_waiting = false;
            let epoch = self
                .next_credit_epoch
                .checked_add(1)
                .expect("publication credit epoch exhausted");
            self.next_credit_epoch = epoch;
            self.credit = Some(epoch);
            Some(epoch)
        } else {
            None
        }
    }
}

struct Shared {
    id: u64,
    state: Mutex<State>,
    wake_worker: Box<dyn Fn(PublicationWake) + Send + Sync>,
}

/// Constructs the request's sole host-visible outcome stream.
pub struct PublicationQueue;

static NEXT_QUEUE_ID: AtomicU64 = AtomicU64::new(1);

impl PublicationQueue {
    /// `wake_worker` must return promptly and use a reserved worker wake path
    /// that remains available when the ordinary command mailbox is full.
    pub fn bounded(
        capacity: usize,
        wake_worker: impl Fn(PublicationWake) + Send + Sync + 'static,
    ) -> Result<(PublicationSender, PublicationReceiver), &'static str> {
        if capacity == 0 {
            return Err("publication data capacity must be positive");
        }
        let id = NEXT_QUEUE_ID.fetch_add(1, Ordering::Relaxed);
        assert_ne!(id, u64::MAX, "publication queue identity exhausted");
        let shared = Arc::new(Shared {
            id,
            state: Mutex::new(State {
                data: VecDeque::with_capacity(capacity),
                capacity,
                reserved: 0,
                reservation_waiting: false,
                credit: None,
                next_credit_epoch: 0,
                channel: ChannelState::Open {
                    receiver_waker: None,
                },
                retirement: RetirementState::Active(None),
                began: false,
            }),
            wake_worker: Box::new(wake_worker),
        });
        Ok((
            PublicationSender {
                shared: Arc::clone(&shared),
            },
            PublicationReceiver {
                shared,
                terminal_seen: false,
            },
        ))
    }
}

pub struct PublicationSender {
    shared: Arc<Shared>,
}

/// One non-cloneable claim on a request's bounded output ring. Dropping an
/// unused permit returns the slot; consuming it makes publication infallible
/// with respect to queue capacity.
pub struct PublicationPermit {
    shared: Option<Arc<Shared>>,
}

impl Drop for PublicationPermit {
    fn drop(&mut self) {
        let Some(shared) = self.shared.take() else {
            return;
        };
        let wake = {
            let mut state = shared.state.lock().unwrap();
            state.reserved = state
                .reserved
                .checked_sub(1)
                .expect("publication permit underflow");
            if state.reservation_waiting {
                state.try_issue_credit()
            } else {
                None
            }
        };
        if let Some(credit_epoch) = wake {
            (shared.wake_worker)(PublicationWake {
                kind: PublicationWakeKind::OutputCredit,
                queue_id: shared.id,
                credit_epoch,
            });
        }
    }
}

impl PublicationSender {
    /// The request became resident: it began, and the receiver is woken to
    /// read it ([`PublicationReceiver::began`]). Each later residency, after
    /// a preemption, changes nothing.
    pub fn began(&self) {
        let receiver_waker = {
            let mut state = self.shared.state.lock().unwrap();
            if std::mem::replace(&mut state.began, true) {
                None
            } else if let ChannelState::Open { receiver_waker } = &mut state.channel {
                receiver_waker.take()
            } else {
                None
            }
        };
        if let Some(waker) = receiver_waker {
            waker.wake();
        }
    }

    pub fn capacity(&self) -> usize {
        self.shared.state.lock().unwrap().capacity
    }

    pub fn reserve_outputs(&self, count: usize) -> Result<Vec<PublicationPermit>, PublishError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut state = self.shared.state.lock().unwrap();
        if !matches!(state.channel, ChannelState::Open { .. }) {
            return Err(PublishError::ReceiverClosed);
        }
        if count > state.capacity {
            return Err(PublishError::ExceedsCapacity);
        }
        let available = state
            .capacity
            .saturating_sub(state.data.len().saturating_add(state.reserved));
        if available < count {
            state.reservation_waiting = true;
            return Err(PublishError::Full);
        }
        state.reservation_waiting = false;
        state.reserved += count;
        Ok((0..count)
            .map(|_| PublicationPermit {
                shared: Some(Arc::clone(&self.shared)),
            })
            .collect())
    }

    pub fn output_with_permit(
        &mut self,
        tokens: Vec<OutputToken>,
        mut permit: PublicationPermit,
    ) -> Result<(), (PublishError, Vec<OutputToken>)> {
        assert!(
            Arc::ptr_eq(
                &self.shared,
                permit.shared.as_ref().expect("publication permit active"),
            ),
            "publication permit belongs to another queue"
        );
        let receiver_waker = {
            let mut state = self.shared.state.lock().unwrap();
            let state = &mut *state;
            let ChannelState::Open { receiver_waker } = &mut state.channel else {
                return Err((PublishError::ReceiverClosed, tokens));
            };
            state.reserved = state
                .reserved
                .checked_sub(1)
                .expect("publication permit underflow");
            permit.shared = None;
            assert!(
                state.data.len() < state.capacity,
                "reserved publication slot is unavailable"
            );
            state.data.push_back(tokens);
            receiver_waker.take()
        };
        if let Some(waker) = receiver_waker {
            waker.wake();
        }
        Ok(())
    }

    /// A full ring leaves the event with the caller for a later attempt.
    pub fn try_output(
        &mut self,
        tokens: Vec<OutputToken>,
    ) -> Result<(), (PublishError, Vec<OutputToken>)> {
        let receiver_waker = {
            let mut state = self.shared.state.lock().unwrap();
            let state = &mut *state;
            let ChannelState::Open { receiver_waker } = &mut state.channel else {
                return Err((PublishError::ReceiverClosed, tokens));
            };
            if state.data.len().saturating_add(state.reserved) == state.capacity {
                return Err((PublishError::Full, tokens));
            }
            state.data.push_back(tokens);
            receiver_waker.take()
        };
        if let Some(waker) = receiver_waker {
            waker.wake();
        }
        Ok(())
    }

    pub fn complete(
        self,
        finish: FinishReason,
        usage: DetailedUsage,
        method: String,
        timings: RequestTimings,
    ) {
        self.record_terminal(Publication::Completed {
            finish,
            usage,
            method,
            timings,
        });
    }

    pub fn fail(self, error: RequestError) {
        self.record_terminal(Publication::Failed(error));
    }

    fn record_terminal(&self, publication: Publication) {
        let receiver_waker = {
            let mut state = self.shared.state.lock().unwrap();
            match &mut state.channel {
                ChannelState::Open { receiver_waker } => {
                    let waker = receiver_waker.take();
                    state.channel = ChannelState::SenderClosed(publication);
                    waker
                }
                ChannelState::SenderClosed(_) | ChannelState::ReceiverClosed => None,
            }
        };
        if let Some(waker) = receiver_waker {
            waker.wake();
        }
    }

    /// Consume only this queue's returned credit wake. The result says whether
    /// the request may publish now, rather than merely whether the wake was
    /// current: another publication may have refilled the ring in the meantime.
    /// Clearing the flag while full lets the next full-to-non-full drain send
    /// a fresh wake. An older wake cannot clear that later credit.
    pub fn process_output_credit(&mut self, wake: PublicationWake) -> bool {
        if wake.kind != PublicationWakeKind::OutputCredit || self.shared.id != wake.queue_id {
            return false;
        }
        let mut state = self.shared.state.lock().unwrap();
        if state.credit == Some(wake.credit_epoch) {
            state.credit = None;
            state.has_capacity()
        } else {
            false
        }
    }

    /// Check a reserved cancellation wake against the owning queue. A wake
    /// from another request cannot cancel this sender.
    pub fn process_cancelled(&self, wake: PublicationWake) -> bool {
        wake.kind == PublicationWakeKind::Cancelled
            && self.shared.id == wake.queue_id
            && !self.receiver_open()
    }

    pub fn receiver_open(&self) -> bool {
        !matches!(
            self.shared.state.lock().unwrap().channel,
            ChannelState::ReceiverClosed
        )
    }

    /// The sole producer may schedule more output only while a receiver and
    /// data slot are available. Receiver drains can only increase this credit.
    pub fn has_capacity(&self) -> bool {
        self.shared.state.lock().unwrap().has_capacity()
    }
}

impl Drop for PublicationSender {
    fn drop(&mut self) {
        self.record_terminal(Publication::Failed(RequestError::WorkerClosed));
    }
}

pub struct PublicationReceiver {
    shared: Arc<Shared>,
    terminal_seen: bool,
}

/// Physical retirement is independent of a request publishing its terminal
/// outcome or losing its receiver. The scheduler signals it after closing
/// the domain request, and the session may wait after dropping the receiver.
pub struct Retirement(Arc<Shared>);

impl Retirement {
    pub(crate) fn retire(self) {
        let wake = {
            let mut state = self.0.state.lock().unwrap();
            match std::mem::replace(&mut state.retirement, RetirementState::Retired) {
                RetirementState::Active(waker) => waker,
                RetirementState::Retired => panic!("request retired twice"),
            }
        };
        if let Some(waker) = wake {
            waker.wake();
        }
    }

    pub fn poll(&self, waker: &Waker) -> Poll<()> {
        let mut state = self.0.state.lock().unwrap();
        match &mut state.retirement {
            RetirementState::Retired => Poll::Ready(()),
            RetirementState::Active(slot) => {
                *slot = Some(waker.clone());
                Poll::Pending
            }
        }
    }
}

impl PublicationReceiver {
    pub fn retirement(&self) -> Retirement {
        Retirement(Arc::clone(&self.shared))
    }

    /// Whether the request has begun ([`PublicationSender::began`]): set
    /// before anything it publishes, so a publication read while this is
    /// false came of a request that never became resident. A receiver
    /// [`Self::poll_next`] left pending is woken when it begins.
    pub fn began(&self) -> bool {
        self.shared.state.lock().unwrap().began
    }

    pub fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<Option<Publication>> {
        if self.terminal_seen {
            return Poll::Ready(None);
        }
        let (publication, wake_credit) = {
            let mut state = self.shared.state.lock().unwrap();
            let was_full = state.data.len().saturating_add(state.reserved) == state.capacity;
            if let Some(tokens) = state.data.pop_front() {
                let wake_credit = if was_full || state.reservation_waiting {
                    state.try_issue_credit()
                } else {
                    None
                };
                (Some(Publication::Output(tokens)), wake_credit)
            } else {
                match std::mem::replace(&mut state.channel, ChannelState::ReceiverClosed) {
                    ChannelState::Open { .. } => {
                        state.channel = ChannelState::Open {
                            receiver_waker: Some(cx.waker().clone()),
                        };
                        return Poll::Pending;
                    }
                    ChannelState::SenderClosed(terminal) => {
                        self.terminal_seen = true;
                        (Some(terminal), None)
                    }
                    ChannelState::ReceiverClosed => {
                        self.terminal_seen = true;
                        (None, None)
                    }
                }
            }
        };
        if let Some(credit_epoch) = wake_credit {
            (self.shared.wake_worker)(PublicationWake {
                kind: PublicationWakeKind::OutputCredit,
                queue_id: self.shared.id,
                credit_epoch,
            });
        }
        Poll::Ready(publication)
    }
}

impl Drop for PublicationReceiver {
    fn drop(&mut self) {
        let wake_cancel = {
            let mut state = self.shared.state.lock().unwrap();
            match std::mem::replace(&mut state.channel, ChannelState::ReceiverClosed) {
                ChannelState::ReceiverClosed => false,
                ChannelState::Open { .. } | ChannelState::SenderClosed(_) => {
                    state.data.clear();
                    true
                }
            }
        };
        if wake_cancel {
            (self.shared.wake_worker)(PublicationWake {
                kind: PublicationWakeKind::Cancelled,
                queue_id: self.shared.id,
                credit_epoch: 0,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_generation::TokenId;

    fn token(index: usize) -> OutputToken {
        OutputToken {
            index,
            token: TokenId(index as u32),
        }
    }

    fn next(receiver: &mut PublicationReceiver) -> Option<Publication> {
        match receiver.poll_next(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(event) => event,
            Poll::Pending => panic!("expected a ready publication"),
        }
    }

    #[test]
    fn retirement_survives_the_terminal_and_the_receiver() {
        let (sender, mut receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
        let owner = receiver.retirement();
        let session = receiver.retirement();
        sender.fail(RequestError::WorkerClosed);
        assert!(matches!(next(&mut receiver), Some(Publication::Failed(_))));
        drop(receiver);
        assert!(session.poll(Waker::noop()).is_pending());
        owner.retire();
        assert!(session.poll(Waker::noop()).is_ready());
    }

    #[test]
    fn full_data_ring_keeps_output_before_the_terminal_and_wakes_once() {
        let wakes = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&wakes);
        let (mut sender, mut receiver) = PublicationQueue::bounded(1, move |wake| {
            observed.lock().unwrap().push(wake);
        })
        .unwrap();

        sender.try_output(vec![token(0)]).unwrap();
        assert_eq!(
            sender.try_output(vec![token(1)]),
            Err((PublishError::Full, vec![token(1)]))
        );
        assert_eq!(
            next(&mut receiver),
            Some(Publication::Output(vec![token(0)]))
        );
        let wake = wakes.lock().unwrap().pop().unwrap();
        assert_eq!(wake.kind(), PublicationWakeKind::OutputCredit);
        assert!(sender.process_output_credit(wake));
        assert!(wakes.lock().unwrap().is_empty());

        sender.try_output(vec![token(1)]).unwrap();
        sender.complete(
            FinishReason::Stop,
            DetailedUsage {
                prompt_tokens: 1,
                completion_tokens: 2,
                cached_tokens: 0,
                draft_n: 0,
                draft_n_accepted: 0,
            },
            "plain".into(),
            RequestTimings::default(),
        );
        assert_eq!(
            next(&mut receiver),
            Some(Publication::Output(vec![token(1)]))
        );
        assert!(matches!(
            next(&mut receiver),
            Some(Publication::Completed { .. })
        ));
        assert_eq!(next(&mut receiver), None);
        assert!(wakes.lock().unwrap().is_empty());
    }

    /// A receiver pending on a request that has not begun is woken when it
    /// begins, and reads it begun; one whose request fails before it began
    /// reads the failure with the request never begun.
    #[test]
    fn a_request_reads_begun_once_resident_and_not_when_refused_before() {
        struct Woken(std::sync::atomic::AtomicBool);
        impl std::task::Wake for Woken {
            fn wake(self: Arc<Self>) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let (sender, mut receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
        let woken = Arc::new(Woken(Default::default()));
        let waker = Waker::from(Arc::clone(&woken));
        assert!(receiver.poll_next(&mut Context::from_waker(&waker)).is_pending());
        assert!(!receiver.began());
        sender.began();
        assert!(woken.0.load(Ordering::Relaxed), "beginning wakes the receiver");
        assert!(receiver.began());
        sender.began();
        assert!(receiver.began(), "a later residency changes nothing");

        let (sender, mut receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
        sender.fail(RequestError::WorkerClosed);
        assert_eq!(next(&mut receiver), Some(Publication::Failed(RequestError::WorkerClosed)));
        assert!(!receiver.began(), "refused before it began");
    }

    #[test]
    fn refilled_ring_does_not_turn_old_credit_into_false_capacity() {
        let wakes = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&wakes);
        let (mut sender, mut receiver) = PublicationQueue::bounded(1, move |wake| {
            observed.lock().unwrap().push(wake);
        })
        .unwrap();

        sender.try_output(vec![token(0)]).unwrap();
        assert!(matches!(next(&mut receiver), Some(Publication::Output(_))));
        let old_credit = wakes.lock().unwrap().pop().unwrap();
        sender.try_output(vec![token(1)]).unwrap();
        assert!(!sender.process_output_credit(old_credit));
        assert!(matches!(next(&mut receiver), Some(Publication::Output(_))));
        let fresh_credit = wakes.lock().unwrap().pop().unwrap();
        assert!(sender.process_output_credit(fresh_credit));
    }

    #[test]
    fn reserved_slot_cannot_be_stolen_and_unused_permit_restores_capacity() {
        let (mut sender, _receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
        let permit = sender.reserve_outputs(1).unwrap().pop().unwrap();
        assert!(!sender.has_capacity());
        assert_eq!(
            sender.try_output(vec![token(0)]),
            Err((PublishError::Full, vec![token(0)]))
        );
        drop(permit);
        assert!(sender.has_capacity());
        sender.try_output(vec![token(1)]).unwrap();
    }

    #[test]
    fn repeated_drains_coalesce_until_the_worker_consumes_credit() {
        let wakes = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&wakes);
        let (mut sender, mut receiver) = PublicationQueue::bounded(2, move |wake| {
            observed.lock().unwrap().push(wake);
        })
        .unwrap();

        sender.try_output(vec![token(0)]).unwrap();
        sender.try_output(vec![token(1)]).unwrap();
        assert!(matches!(next(&mut receiver), Some(Publication::Output(_))));
        assert!(matches!(next(&mut receiver), Some(Publication::Output(_))));
        assert_eq!(wakes.lock().unwrap().len(), 1);
        let credit = wakes.lock().unwrap().pop().unwrap();
        sender.try_output(vec![token(2)]).unwrap();
        sender.try_output(vec![token(3)]).unwrap();
        assert!(matches!(next(&mut receiver), Some(Publication::Output(_))));
        assert!(wakes.lock().unwrap().is_empty());
        assert!(sender.process_output_credit(credit));
        sender.try_output(vec![token(4)]).unwrap();
        assert!(matches!(next(&mut receiver), Some(Publication::Output(_))));
        assert_eq!(wakes.lock().unwrap().len(), 1);
    }

    #[test]
    fn sender_drop_records_one_failure_after_accepted_output() {
        let (mut sender, mut receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
        sender.try_output(vec![token(0)]).unwrap();
        drop(sender);
        assert_eq!(
            next(&mut receiver),
            Some(Publication::Output(vec![token(0)]))
        );
        assert_eq!(
            next(&mut receiver),
            Some(Publication::Failed(RequestError::WorkerClosed))
        );
        assert_eq!(next(&mut receiver), None);
    }

    #[test]
    fn receiver_drop_reserves_a_cancel_wake() {
        let wakes = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&wakes);
        let (mut sender, receiver) = PublicationQueue::bounded(1, move |wake| {
            observed.lock().unwrap().push(wake);
        })
        .unwrap();
        drop(receiver);
        assert_eq!(
            sender.try_output(vec![token(0)]),
            Err((PublishError::ReceiverClosed, vec![token(0)]))
        );
        let wake = wakes.lock().unwrap().pop().unwrap();
        assert_eq!(wake.kind(), PublicationWakeKind::Cancelled);
        assert!(sender.process_cancelled(wake));
    }

    #[test]
    fn dropping_a_full_receiver_cancels_without_waiting_for_output_credit() {
        let wakes = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&wakes);
        let (mut sender, receiver) = PublicationQueue::bounded(1, move |wake| {
            observed.lock().unwrap().push(wake);
        })
        .unwrap();
        sender.try_output(vec![token(0)]).unwrap();
        assert!(!sender.has_capacity());

        drop(receiver);
        assert!(!sender.receiver_open());
        assert_eq!(
            sender.try_output(vec![token(1)]),
            Err((PublishError::ReceiverClosed, vec![token(1)]))
        );
        let wakes = wakes.lock().unwrap();
        assert_eq!(wakes.len(), 1);
        assert_eq!(wakes[0].kind(), PublicationWakeKind::Cancelled);
    }

    #[test]
    fn cancellation_wake_cannot_close_another_request() {
        let wakes = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&wakes);
        let (other_sender, other_receiver) = PublicationQueue::bounded(1, move |wake| {
            observed.lock().unwrap().push(wake);
        })
        .unwrap();
        let (sender, _receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
        drop(other_receiver);
        let wake = wakes.lock().unwrap().pop().unwrap();
        assert!(!sender.process_cancelled(wake));
        assert!(!other_sender.receiver_open());
    }

    #[test]
    fn reserve_outputs_above_capacity_fails_without_setting_wait() {
        let wakes = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&wakes);
        let (sender, _receiver) = PublicationQueue::bounded(2, move |wake| {
            observed.lock().unwrap().push(wake);
        })
        .unwrap();
        assert_eq!(sender.capacity(), 2);
        assert!(matches!(
            sender.reserve_outputs(3),
            Err(PublishError::ExceedsCapacity)
        ));
        let permit = sender.reserve_outputs(1).unwrap().pop().unwrap();
        drop(permit);
        assert!(
            wakes.lock().unwrap().is_empty(),
            "rejected over-capacity reservation must not arm a credit wake"
        );
    }

    #[test]
    fn endpoints_are_sendable_host_worker_boundary_values() {
        fn assert_send<T: Send>() {}
        assert_send::<PublicationSender>();
        assert_send::<PublicationReceiver>();
        assert_send::<PublicationWake>();
    }
}
