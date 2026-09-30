use magnitude_executor::{DomainError, RequestId};
use magnitude_generation::{OutputToken, TokenId};
use magnitude_scheduler::{
    owner::AdmissionError,
    protocol::{WorkerCommand, WorkerReply},
    publication::{Publication, PublicationQueue, PublicationWakeKind},
    worker::{DispatchError, Driven, Event, Owner, Panicked, Worker, WorkerWakeHandle},
};
use std::sync::mpsc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(3);
const NEVER: u64 = u64::MAX;

/// Why a test owner stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Stopped(&'static str);

impl From<Panicked> for Stopped {
    fn from(_: Panicked) -> Self {
        Self("panicked")
    }
}

/// Answers every command with `reply` and stops once closed.
struct Replying {
    reply: fn() -> WorkerReply,
    events: mpsc::Sender<Event>,
    closed: bool,
}

impl Driven for Replying {
    type Stopped = Stopped;
    fn command(&mut self, _: WorkerCommand, _: u64) -> Result<WorkerReply, String> {
        Ok((self.reply)())
    }
    fn event(&mut self, event: Event, _: u64) {
        if matches!(event, Event::Close) {
            self.closed = true;
        }
        self.events.send(event).unwrap();
    }
    fn run(&mut self, _: u64) -> Result<u64, Stopped> {
        if self.closed {
            return Err(Stopped("closed"));
        }
        Ok(NEVER)
    }
}

fn replying(
    reply: fn() -> WorkerReply,
    capacity: usize,
) -> (Worker<Stopped>, WorkerWakeHandle<Stopped>, mpsc::Receiver<Event>) {
    let (events, observed) = mpsc::channel();
    let (handle, wakes) = mpsc::channel();
    let worker = Worker::spawn(
        move |wake_handle| {
            handle.send(wake_handle).unwrap();
            Ok(Box::new(Replying {
                reply,
                events,
                closed: false,
            }) as Owner<Stopped>)
        },
        capacity,
    )
    .unwrap();
    (worker, wakes.recv_timeout(TIMEOUT).unwrap(), observed)
}

#[test]
fn blind_admission_crosses_the_worker_as_a_typed_refusal() {
    let (mut worker, _, _events) = replying(
        || {
            WorkerReply::AdmissionRefused(AdmissionError::from(DomainError::Blind(
                "host status unavailable".into(),
            )))
        },
        1,
    );
    let reply = worker
        .client()
        .dispatch(WorkerCommand::Observe)
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

/// Completion and publication wakes are reserved: they reach the owner while
/// every command slot holds an unconsumed call.
#[test]
fn reserved_wakes_bypass_a_saturated_control_queue() {
    let (mut worker, wakes, observed) = replying(|| WorkerReply::Status(None), 1);
    let client = worker.client();
    let held = client.dispatch(WorkerCommand::Observe).unwrap();
    assert!(matches!(
        client.dispatch(WorkerCommand::Observe),
        Err(DispatchError::Full)
    ));

    let (published, publication) = mpsc::channel();
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
    let credit = publication.recv_timeout(TIMEOUT).unwrap();
    assert_eq!(credit.kind(), PublicationWakeKind::OutputCredit);
    wakes.publication(RequestId(1), credit);
    wakes.completion().complete();

    let mut seen = (false, false);
    while seen != (true, true) {
        match observed.recv_timeout(TIMEOUT).unwrap() {
            Event::Publication(RequestId(1), wake) => {
                assert_eq!(wake.kind(), PublicationWakeKind::OutputCredit);
                seen.0 = true;
            }
            Event::Completion => seen.1 = true,
            _ => {}
        }
    }
    held.wait().unwrap();
    drop(sender);
    worker.close();
}

#[test]
fn abandoned_admission_reply_is_cancelled_on_the_worker() {
    let (mut worker, _, observed) = replying(
        || {
            let (_, receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
            WorkerReply::Admitted {
                request: RequestId(9),
                receiver,
            }
        },
        1,
    );
    drop(worker.client().dispatch(WorkerCommand::Observe).unwrap());
    loop {
        if let Event::Cancel(request) = observed.recv_timeout(TIMEOUT).unwrap() {
            assert_eq!(request, RequestId(9));
            break;
        }
    }
    worker.close();
}

#[test]
fn a_closed_worker_refuses_commands_with_its_cause() {
    let (mut worker, _, _events) = replying(|| WorkerReply::Status(None), 1);
    let client = worker.client();
    worker.close();
    assert_eq!(client.stopped(), Some(Stopped("closed")));
    assert!(matches!(
        client.dispatch(WorkerCommand::Observe),
        Err(DispatchError::Stopped(Stopped("closed")))
    ));
}
