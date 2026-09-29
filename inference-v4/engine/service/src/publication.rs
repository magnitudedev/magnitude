//! One bounded, ordered outcome stream for a request.
//!
//! The data ring and terminal slot have separate capacity. The receiver drains
//! accepted data before observing the terminal event. Credit and cancellation
//! use a reserved wake supplied by the worker mailbox, so neither depends on
//! admission-command capacity.

use magnitude_generation::{DetailedUsage, FinishReason, OutputToken};
use magnitude_model_executor::{DeviceError, InvariantError, ResourceKind};
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
    State(magnitude_model_state::Error),
    Capacity(ServiceCapacityError),
    Device(DeviceError),
    Invariant(InvariantError),
    WorkerClosed,
    ModelUnloaded { cause: ModelUnloadCause },
}

/// Elapsed native work measured per request from submission through completion.
/// A request in a packed batch receives that batch's wall duration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhysicalTimings {
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
        timings: PhysicalTimings,
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
    ReceiverClosed,
}

struct State {
    data: VecDeque<Vec<OutputToken>>,
    terminal: Option<Publication>,
    capacity: usize,
    reserved: usize,
    reservation_waiting: bool,
    credit: Option<u64>,
    next_credit_epoch: u64,
    receiver_open: bool,
    sender_open: bool,
    receiver_waker: Option<Waker>,
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
                terminal: None,
                capacity,
                reserved: 0,
                reservation_waiting: false,
                credit: None,
                next_credit_epoch: 0,
                receiver_open: true,
                sender_open: true,
                receiver_waker: None,
            }),
            wake_worker: Box::new(wake_worker),
        });
        Ok((
            PublicationSender {
                shared: Arc::clone(&shared),
                terminal_recorded: false,
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
    terminal_recorded: bool,
}

/// One non-cloneable claim on a request's bounded output ring. Dropping an
/// unused permit returns the slot; consuming it makes publication infallible
/// with respect to queue capacity.
pub struct PublicationPermit {
    shared: Arc<Shared>,
    active: bool,
}

impl Drop for PublicationPermit {
    fn drop(&mut self) {
        if self.active {
            let wake = {
                let mut state = self.shared.state.lock().unwrap();
                state.reserved = state
                    .reserved
                    .checked_sub(1)
                    .expect("publication permit underflow");
                if state.reservation_waiting
                    && state.sender_open
                    && state.receiver_open
                    && state.credit.is_none()
                {
                    state.reservation_waiting = false;
                    let epoch = state
                        .next_credit_epoch
                        .checked_add(1)
                        .expect("publication credit epoch exhausted");
                    state.next_credit_epoch = epoch;
                    state.credit = Some(epoch);
                    Some(epoch)
                } else {
                    None
                }
            };
            if let Some(credit_epoch) = wake {
                (self.shared.wake_worker)(PublicationWake {
                    kind: PublicationWakeKind::OutputCredit,
                    queue_id: self.shared.id,
                    credit_epoch,
                });
            }
        }
    }
}

impl PublicationSender {
    pub fn reserve_outputs(&self, count: usize) -> Result<Vec<PublicationPermit>, PublishError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut state = self.shared.state.lock().unwrap();
        if !state.receiver_open || state.terminal.is_some() {
            return Err(PublishError::ReceiverClosed);
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
                shared: Arc::clone(&self.shared),
                active: true,
            })
            .collect())
    }

    pub fn output_with_permit(
        &mut self,
        tokens: Vec<OutputToken>,
        mut permit: PublicationPermit,
    ) -> Result<(), (PublishError, Vec<OutputToken>)> {
        assert!(
            Arc::ptr_eq(&self.shared, &permit.shared),
            "publication permit belongs to another queue"
        );
        let receiver_waker = {
            let mut state = self.shared.state.lock().unwrap();
            if !state.receiver_open {
                return Err((PublishError::ReceiverClosed, tokens));
            }
            assert!(
                state.terminal.is_none(),
                "output after terminal publication"
            );
            state.reserved = state
                .reserved
                .checked_sub(1)
                .expect("publication permit underflow");
            permit.active = false;
            assert!(
                state.data.len() < state.capacity,
                "reserved publication slot is unavailable"
            );
            state.data.push_back(tokens);
            state.receiver_waker.take()
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
            assert!(
                state.terminal.is_none(),
                "output after terminal publication"
            );
            if !state.receiver_open {
                return Err((PublishError::ReceiverClosed, tokens));
            }
            if state.data.len().saturating_add(state.reserved) == state.capacity {
                return Err((PublishError::Full, tokens));
            }
            state.data.push_back(tokens);
            state.receiver_waker.take()
        };
        if let Some(waker) = receiver_waker {
            waker.wake();
        }
        Ok(())
    }

    pub fn complete(
        mut self,
        finish: FinishReason,
        usage: DetailedUsage,
        method: String,
        timings: PhysicalTimings,
    ) {
        self.record_terminal(Publication::Completed {
            finish,
            usage,
            method,
            timings,
        });
    }

    pub fn fail(mut self, error: RequestError) {
        self.record_terminal(Publication::Failed(error));
    }

    fn record_terminal(&mut self, publication: Publication) {
        let receiver_waker = {
            let mut state = self.shared.state.lock().unwrap();
            assert!(state.terminal.is_none(), "duplicate terminal publication");
            if state.receiver_open {
                state.terminal = Some(publication);
            }
            state.sender_open = false;
            state.receiver_waker.take()
        };
        self.terminal_recorded = true;
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
            state.receiver_open
                && state.terminal.is_none()
                && state.data.len().saturating_add(state.reserved) < state.capacity
        } else {
            false
        }
    }

    /// Check a reserved cancellation wake against the owning queue. A wake
    /// from another request cannot cancel this sender.
    pub fn process_cancelled(&self, wake: PublicationWake) -> bool {
        wake.kind == PublicationWakeKind::Cancelled
            && self.shared.id == wake.queue_id
            && !self.shared.state.lock().unwrap().receiver_open
    }

    pub fn receiver_open(&self) -> bool {
        self.shared.state.lock().unwrap().receiver_open
    }

    /// The sole producer may schedule more output only while a receiver and
    /// data slot are available. Receiver drains can only increase this credit.
    pub fn has_capacity(&self) -> bool {
        let state = self.shared.state.lock().unwrap();
        state.receiver_open
            && state.terminal.is_none()
            && state.data.len().saturating_add(state.reserved) < state.capacity
    }
}

impl Drop for PublicationSender {
    fn drop(&mut self) {
        if self.terminal_recorded {
            return;
        }
        self.record_terminal(Publication::Failed(RequestError::WorkerClosed));
    }
}

pub struct PublicationReceiver {
    shared: Arc<Shared>,
    terminal_seen: bool,
}

impl PublicationReceiver {
    pub fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<Option<Publication>> {
        if self.terminal_seen {
            return Poll::Ready(None);
        }
        let (publication, wake_credit) = {
            let mut state = self.shared.state.lock().unwrap();
            let was_full = state.data.len().saturating_add(state.reserved) == state.capacity;
            if let Some(tokens) = state.data.pop_front() {
                if (was_full || state.reservation_waiting)
                    && state.sender_open
                    && state.credit.is_none()
                {
                    state.reservation_waiting = false;
                    state.next_credit_epoch = state
                        .next_credit_epoch
                        .checked_add(1)
                        .expect("publication credit epoch exhausted");
                    let epoch = state.next_credit_epoch;
                    state.credit = Some(epoch);
                    (Some(Publication::Output(tokens)), Some(epoch))
                } else {
                    (Some(Publication::Output(tokens)), None)
                }
            } else if let Some(terminal) = state.terminal.take() {
                self.terminal_seen = true;
                state.receiver_open = false;
                (Some(terminal), None)
            } else if !state.sender_open {
                self.terminal_seen = true;
                state.receiver_open = false;
                (None, None)
            } else {
                state.receiver_waker = Some(cx.waker().clone());
                return Poll::Pending;
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
            if !state.receiver_open {
                false
            } else {
                state.receiver_open = false;
                state.data.clear();
                state.terminal.take();
                true
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
            PhysicalTimings::default(),
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
    fn endpoints_are_sendable_host_worker_boundary_values() {
        fn assert_send<T: Send>() {}
        assert_send::<PublicationSender>();
        assert_send::<PublicationReceiver>();
        assert_send::<PublicationWake>();
    }
}
