use magnitude_generation::{OutputToken, TokenId};
use magnitude_model_executor::{DomainError, RequestId};
use magnitude_service::{
    owner::{AdmissionError, Status},
    protocol::{CapacityStatus, WorkerCommand, WorkerReply},
    publication::{Publication, PublicationQueue, PublicationWakeKind},
    worker::{CompletionWake, Drive, Driven, Worker, WorkerWakeHandle},
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(3);

#[test]
fn blind_admission_crosses_the_worker_as_a_typed_refusal() {
    struct Blind;
    impl Driven for Blind {
        fn command(&mut self, _: WorkerCommand, _: u64) -> Result<WorkerReply, String> {
            Ok(WorkerReply::AdmissionRefused(AdmissionError::from(
                DomainError::Blind("host status unavailable".into()),
            )))
        }
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
    let mut worker = Worker::spawn(|| Ok(Box::new(Blind)), 1).unwrap();
    let reply = worker
        .client()
        .dispatch(WorkerCommand::Check)
        .unwrap()
        .wait()
        .unwrap();
    assert!(matches!(
        reply,
        WorkerReply::AdmissionRefused(AdmissionError::MemoryObservationUnavailable(message))
            if message == "host status unavailable"
    ));
    worker.close();
}

#[test]
fn reclaim_is_a_typed_admission_refusal() {
    assert_eq!(
        AdmissionError::from(DomainError::Reclaim),
        AdmissionError::MemoryReclaim
    );
}

#[derive(Default)]
struct Shared {
    wake: Mutex<Option<CompletionWake>>,
    publication_handle: Mutex<Option<WorkerWakeHandle>>,
    publication_credit_seen: AtomicBool,
    publication_cancel_seen: AtomicBool,
    reconciled: AtomicBool,
}

struct Owner {
    shared: Arc<Shared>,
    submitted: bool,
    failure: Option<String>,
    entered: Option<mpsc::Sender<()>>,
    release: Option<mpsc::Receiver<()>>,
}

impl Driven for Owner {
    fn install_wake_handle(&mut self, wakes: WorkerWakeHandle) {
        *self.shared.publication_handle.lock().unwrap() = Some(wakes);
    }
    fn publication_wake(
        &mut self,
        _: RequestId,
        wake: magnitude_service::publication::PublicationWake,
        _: u64,
    ) -> Result<(), String> {
        match wake.kind() {
            PublicationWakeKind::OutputCredit => self
                .shared
                .publication_credit_seen
                .store(true, Ordering::Release),
            PublicationWakeKind::Cancelled => self
                .shared
                .publication_cancel_seen
                .store(true, Ordering::Release),
        };
        Ok(())
    }
    fn command(&mut self, command: WorkerCommand, _: u64) -> Result<WorkerReply, String> {
        match command {
            WorkerCommand::Status {
                request: RequestId(1),
            } => {
                self.entered.take().unwrap().send(()).unwrap();
                self.release.take().unwrap().recv().unwrap();
                Ok(WorkerReply::Acknowledged)
            }
            WorkerCommand::Status { .. } => Ok(WorkerReply::Status(
                if self.shared.reconciled.load(Ordering::Acquire)
                    && self.shared.publication_credit_seen.load(Ordering::Acquire)
                    && self.shared.publication_cancel_seen.load(Ordering::Acquire)
                {
                    Status::Runnable
                } else {
                    Status::AwaitingCompletion
                },
            )),
            _ => Ok(WorkerReply::Acknowledged),
        }
    }
    fn advance(&mut self, _: u64, wake: CompletionWake) -> Result<Drive, String> {
        if !self.submitted {
            self.submitted = true;
            *self.shared.wake.lock().unwrap() = Some(wake);
            return Ok(Drive::AwaitingCompletion);
        }
        self.shared.reconciled.store(true, Ordering::Release);
        Ok(Drive::Idle)
    }

    fn failed(&mut self, error: &str) {
        self.failure = Some(error.into());
    }

    fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    fn shutdown(&mut self) -> Result<bool, String> {
        Ok(self.shared.reconciled.load(Ordering::Acquire))
    }
}

#[test]
fn reserved_completion_and_publication_bypass_a_saturated_control_queue() {
    let shared = Arc::new(Shared::default());
    let owner_shared = Arc::clone(&shared);
    let (entered, entered_rx) = mpsc::channel();
    let (release, release_rx) = mpsc::channel();
    let mut worker = Worker::spawn(
        move || {
            Ok(Box::new(Owner {
                shared: owner_shared,
                submitted: false,
                failure: None,
                entered: Some(entered),
                release: Some(release_rx),
            }) as Box<dyn Driven>)
        },
        2,
    )
    .unwrap();

    while shared.wake.lock().unwrap().is_none()
        || shared.publication_handle.lock().unwrap().is_none()
    {
        std::thread::yield_now();
    }
    let blocked = worker
        .client()
        .dispatch(WorkerCommand::Status {
            request: RequestId(1),
        })
        .unwrap();
    entered_rx.recv_timeout(TIMEOUT).unwrap();
    let queued = worker
        .client()
        .dispatch(WorkerCommand::Status {
            request: RequestId(2),
        })
        .unwrap();
    assert!(worker
        .client()
        .dispatch(WorkerCommand::Status {
            request: RequestId(3)
        })
        .is_err());

    let (published, publication_rx) = mpsc::channel();
    let (mut sender, mut receiver) = PublicationQueue::bounded(1, move |wake| {
        published.send(wake).unwrap();
    })
    .unwrap();
    sender
        .try_output(vec![OutputToken {
            index: 0,
            token: TokenId(1),
        }])
        .unwrap();
    assert!(matches!(
        receiver.poll_next(&mut Context::from_waker(Waker::noop())),
        Poll::Ready(Some(Publication::Output(_)))
    ));
    let credit = publication_rx.recv_timeout(TIMEOUT).unwrap();
    assert_eq!(credit.kind(), PublicationWakeKind::OutputCredit);
    drop(receiver);
    let cancellation = publication_rx.recv_timeout(TIMEOUT).unwrap();
    assert_eq!(cancellation.kind(), PublicationWakeKind::Cancelled);
    shared
        .publication_handle
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .publication(RequestId(1), credit);
    shared
        .publication_handle
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .publication(RequestId(1), cancellation);
    shared.wake.lock().unwrap().take().unwrap().complete();
    drop(sender);
    release.send(()).unwrap();
    blocked.wait().unwrap();
    assert!(matches!(
        queued.wait().unwrap(),
        WorkerReply::Status(Status::Runnable)
    ));
    worker.close();
}

struct AdmissionOwner {
    cancelled: Arc<AtomicBool>,
    failure: Option<String>,
}

impl Driven for AdmissionOwner {
    fn command(&mut self, command: WorkerCommand, _: u64) -> Result<WorkerReply, String> {
        match command {
            WorkerCommand::Check => {
                let (_, receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
                Ok(WorkerReply::Admitted {
                    request: RequestId(9),
                    receiver,
                })
            }
            WorkerCommand::Cancel {
                request: RequestId(9),
            } => {
                self.cancelled.store(true, Ordering::Release);
                Ok(WorkerReply::Acknowledged)
            }
            WorkerCommand::Capacity => Ok(WorkerReply::Capacity(CapacityStatus {
                active: 0,
                limit: 4,
            })),
            _ => Err("unexpected test command".into()),
        }
    }

    fn advance(&mut self, _: u64, _: CompletionWake) -> Result<Drive, String> {
        Ok(Drive::Idle)
    }

    fn failed(&mut self, error: &str) {
        self.failure = Some(error.into());
    }

    fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    fn shutdown(&mut self) -> Result<bool, String> {
        Ok(true)
    }
}

#[test]
fn abandoned_admission_reply_is_cancelled_on_the_worker() {
    let cancelled = Arc::new(AtomicBool::new(false));
    let owner_cancelled = cancelled.clone();
    let mut worker = Worker::spawn(
        move || {
            Ok(Box::new(AdmissionOwner {
                cancelled: owner_cancelled,
                failure: None,
            }) as Box<dyn Driven>)
        },
        1,
    )
    .unwrap();

    drop(worker.client().dispatch(WorkerCommand::Check).unwrap());
    let deadline = std::time::Instant::now() + TIMEOUT;
    while !cancelled.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(cancelled.load(Ordering::Acquire));
    assert!(matches!(
        worker
            .client()
            .dispatch(WorkerCommand::Capacity)
            .unwrap()
            .wait()
            .unwrap(),
        WorkerReply::Capacity(CapacityStatus {
            active: 0,
            limit: 4
        })
    ));
    worker.close();
}
