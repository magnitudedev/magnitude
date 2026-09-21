use magnitude_engine::{
    generation::{Generation, Options, Sampling},
    inputs::{InputLayout, TokenId},
    models::sequence::Advance,
    service::{
        owner::{Completion, Executor, Owner, PrepareError, Status, Submitted, Work},
        policy::{Limits, RequestId},
        worker::{CompletionWake, Worker},
    },
};
use std::{
    collections::BTreeSet,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    time::Duration,
};
const TIMEOUT: Duration = Duration::from_secs(3);
#[derive(Default)]
struct Shared {
    input_sources: Mutex<std::collections::BTreeMap<RequestId, HostInput>>,
    input_open_threads: Mutex<Vec<std::thread::ThreadId>>,
    committed: AtomicUsize,
    positions: Mutex<std::collections::BTreeMap<RequestId, usize>>,
    snapshots: AtomicUsize,
    closed: Mutex<Vec<RequestId>>,
}
struct HostInput {
    tokens: Vec<TokenId>,
    layout: InputLayout,
    reject_open: bool,
    dropped_on: Arc<Mutex<Option<std::thread::ThreadId>>>,
}
impl Drop for HostInput {
    fn drop(&mut self) {
        *self.dropped_on.lock().unwrap() = Some(std::thread::current().id());
    }
}
impl magnitude_engine::service::owner::InputExecutor<HostInput> for Model {
    fn input_layout(&self, source: &HostInput, tokens: &[TokenId]) -> Result<InputLayout, String> {
        if source.tokens != tokens {
            return Err("source prompt differs".into());
        }
        Ok(source.layout.clone())
    }
    fn open_input(
        &mut self,
        request: RequestId,
        source: HostInput,
        tokens: &[TokenId],
        layout: &InputLayout,
    ) -> Result<(), String> {
        if source.reject_open || source.tokens != tokens || source.layout != *layout {
            return Err("source admission rejected".into());
        }
        self.open(request)?;
        self.shared
            .input_open_threads
            .lock()
            .unwrap()
            .push(std::thread::current().id());
        self.shared
            .input_sources
            .lock()
            .unwrap()
            .insert(request, source);
        Ok(())
    }
}
struct Gate {
    done: AtomicBool,
    fail: AtomicBool,
    wake: Mutex<Option<CompletionWake>>,
}
impl Gate {
    fn complete(&self) {
        let wake = {
            let mut wake = self.wake.lock().unwrap();
            self.done.store(true, Ordering::Release);
            wake.take()
        };
        if let Some(wake) = wake {
            wake.complete();
        }
    }
}
struct Batch(Arc<Gate>);
impl Completion for Batch {
    fn is_complete(&self) -> bool {
        self.0.done.load(Ordering::Acquire)
    }
    fn result(&mut self) -> Result<(), String> {
        assert!(self.is_complete());
        if self.0.fail.load(Ordering::Acquire) {
            Err("device failed".into())
        } else {
            Ok(())
        }
    }
    fn notify(&mut self, wake: CompletionWake) {
        let mut pending = self.0.wake.lock().unwrap();
        if self.is_complete() {
            drop(pending);
            wake.complete();
        } else {
            assert!(pending.is_none());
            *pending = Some(wake);
        }
    }
}
struct Row {
    gate: Arc<Gate>,
    shared: Arc<Shared>,
    sample: bool,
    request: RequestId,
    position: usize,
}
impl Advance for Row {
    fn is_complete(&self) -> bool {
        self.gate.done.load(Ordering::Acquire)
    }
    fn selected(&mut self) -> Result<Option<TokenId>, String> {
        assert!(self.is_complete());
        Ok(self.sample.then_some(TokenId(7)))
    }
    fn commit(&mut self) -> Result<(), String> {
        assert!(self.is_complete());
        self.shared.committed.fetch_add(1, Ordering::Relaxed);
        self.shared
            .positions
            .lock()
            .unwrap()
            .insert(self.request, self.position);
        Ok(())
    }
}
struct Model {
    shared: Arc<Shared>,
    submissions: Sender<Arc<Gate>>,
    _local: Rc<()>,
}
struct Snapshot {
    position: usize,
    shared: Arc<Shared>,
}
impl Drop for Snapshot {
    fn drop(&mut self) {
        self.shared.snapshots.fetch_sub(1, Ordering::Relaxed);
    }
}
impl Executor for Model {
    type Checkpoint = Snapshot;
    fn checkpoint(
        &self,
        request: RequestId,
    ) -> Result<magnitude_engine::service::owner::NumericalCheckpoint<Snapshot>, String> {
        let position = *self
            .shared
            .positions
            .lock()
            .unwrap()
            .get(&request)
            .ok_or("unknown numerical request")?;
        self.shared.snapshots.fetch_add(1, Ordering::Relaxed);
        Ok(magnitude_engine::service::owner::NumericalCheckpoint {
            position,
            state: Snapshot {
                position,
                shared: self.shared.clone(),
            },
        })
    }
    fn open_checkpoint(&mut self, request: RequestId, snapshot: &Snapshot) -> Result<(), String> {
        if !Arc::ptr_eq(&self.shared, &snapshot.shared) {
            return Err("foreign snapshot".into());
        }
        self.shared
            .positions
            .lock()
            .unwrap()
            .insert(request, snapshot.position);
        Ok(())
    }
    fn open(&mut self, request: RequestId) -> Result<(), String> {
        self.shared.positions.lock().unwrap().insert(request, 0);
        Ok(())
    }
    fn prepare(&mut self, work: &[Work]) -> Result<Submitted, PrepareError> {
        let gate = Arc::new(Gate {
            done: AtomicBool::new(false),
            fail: AtomicBool::new(false),
            wake: Mutex::new(None),
        });
        let rows = work
            .iter()
            .map(|w| {
                Box::new(Row {
                    gate: gate.clone(),
                    shared: self.shared.clone(),
                    sample: w.proposal.needs_sample(),
                    request: w.request,
                    position: w.proposal.position() + w.proposal.tokens().len(),
                }) as Box<dyn Advance>
            })
            .collect();
        self.submissions.send(gate.clone()).unwrap();
        Ok(Submitted {
            completion: Box::new(Batch(gate)),
            rows,
        })
    }
    fn preparation_identity(&self, _: &[Work]) -> Result<String, String> {
        Ok("one".into())
    }
    fn reclaim_idle(&mut self) -> Result<u64, String> {
        Ok(0)
    }
    fn reclaimable(&self, _: &[RequestId]) -> Result<u64, String> {
        Ok(0)
    }
    fn evict(&mut self, _: &[RequestId]) -> Result<u64, String> {
        Ok(0)
    }
    fn close(&mut self, id: RequestId) -> Result<(), String> {
        self.shared.input_sources.lock().unwrap().remove(&id);
        self.shared.positions.lock().unwrap().remove(&id);
        self.shared.closed.lock().unwrap().push(id);
        Ok(())
    }
}
fn worker(capacity: usize) -> (Worker<Owner<Model>>, Receiver<Arc<Gate>>, Arc<Shared>) {
    let shared = Arc::new(Shared::default());
    let state = shared.clone();
    let (tx, rx) = mpsc::channel();
    let worker = Worker::spawn(
        move || {
            Owner::new(
                Model {
                    shared: state,
                    submissions: tx,
                    _local: Rc::new(()),
                },
                Limits {
                    max_requests: 8,
                    max_batch: 2,
                    prefill_tokens: 8,
                    decode_tokens: 1,
                    decode_share: 0.5,
                    locality_seconds: 0.0,
                },
            )
        },
        capacity,
    )
    .unwrap();
    (worker, rx, shared)
}
fn generation() -> Generation {
    Generation::new(
        vec![TokenId(1), TokenId(2)],
        InputLayout::new(2, vec![]).unwrap(),
        Options {
            max_tokens: 8,
            output_capacity: 1,
            context_limit: 16,
            vocabulary: 16,
            stop_tokens: BTreeSet::from([TokenId(15)]),
            sampling: Sampling::Greedy,
            seed: 0,
            forced_quantum: 0,
        },
        None,
    )
    .unwrap()
}
#[test]
fn reserved_completion_precedes_saturated_control_queue_and_retirement_waits() {
    let (mut worker, submissions, shared) = worker(3);
    let id = worker
        .client()
        .call(|o, now| o.admit(generation(), now))
        .unwrap()
        .wait()
        .unwrap();
    let gate = submissions.recv_timeout(TIMEOUT).unwrap();
    let (entered, rx) = mpsc::channel();
    let (release, blocked) = mpsc::channel();
    let first = worker
        .client()
        .call(move |_, _| {
            entered.send(()).unwrap();
            blocked.recv().unwrap();
            Ok(())
        })
        .unwrap();
    rx.recv_timeout(TIMEOUT).unwrap();
    let second = worker.client().call(move |o, _| o.status(id)).unwrap();
    let third = worker.client().call(move |o, _| o.status(id)).unwrap();
    assert!(worker.client().call(|_, _| Ok(())).is_err());
    gate.complete();
    release.send(()).unwrap();
    first.wait().unwrap();
    assert_eq!(second.wait().unwrap(), Status::OutputBlocked);
    assert_eq!(third.wait().unwrap(), Status::OutputBlocked);
    assert_eq!(shared.committed.load(Ordering::Relaxed), 1);
    assert!(submissions.try_recv().is_err());
    worker.close();
    assert_eq!(*shared.closed.lock().unwrap(), vec![id]);
}
#[test]
fn shutdown_keeps_outstanding_work_until_reserved_completion() {
    let (mut worker, submissions, shared) = worker(2);
    let id = worker
        .client()
        .call(|o, now| o.admit(generation(), now))
        .unwrap()
        .wait()
        .unwrap();
    let gate = submissions.recv_timeout(TIMEOUT).unwrap();
    let (closed, rx) = mpsc::channel();
    let closer = std::thread::spawn(move || {
        worker.close();
        closed.send(()).unwrap();
    });
    assert!(rx.recv_timeout(Duration::from_millis(30)).is_err());
    assert!(shared.closed.lock().unwrap().is_empty());
    gate.complete();
    rx.recv_timeout(TIMEOUT).unwrap();
    closer.join().unwrap();
    assert_eq!(shared.committed.load(Ordering::Relaxed), 0);
    assert_eq!(*shared.closed.lock().unwrap(), vec![id]);
}
#[test]
fn abandoned_admission_is_cleaned_on_owner_before_numerical_submission() {
    let (mut worker, submissions, shared) = worker(2);
    let (entered, rx) = mpsc::channel();
    let (release, blocked) = mpsc::channel();
    let (cleaned, done) = mpsc::channel();
    let call = worker
        .client()
        .call_with_cleanup(
            move |o, now| {
                entered.send(()).unwrap();
                blocked.recv().unwrap();
                o.admit(generation(), now)
            },
            move |o, id| {
                o.cancel(id, true).unwrap();
                o.retire(id).unwrap();
                cleaned.send(id).unwrap();
                Ok(())
            },
        )
        .unwrap();
    rx.recv_timeout(TIMEOUT).unwrap();
    drop(call);
    release.send(()).unwrap();
    let id = done.recv_timeout(TIMEOUT).unwrap();
    assert!(submissions.try_recv().is_err());
    assert_eq!(*shared.closed.lock().unwrap(), vec![id]);
    worker.close();
}
#[test]
fn abandoned_delivered_result_has_reserved_cleanup_even_when_capacity_is_full() {
    let (mut worker, _, _) = worker(2);
    let (cleaned, done) = mpsc::channel();
    let result = worker
        .client()
        .call_with_cleanup(
            |_, _| Ok(42),
            move |_, value| {
                cleaned.send(value).unwrap();
                Ok(())
            },
        )
        .unwrap();
    // The second call proves the first result is already delivered. Keep another
    // ready result so both outstanding permits are held when cleanup is queued.
    worker.client().call(|_, _| Ok(())).unwrap().wait().unwrap();
    let held = worker.client().call(|_, _| Ok(())).unwrap();
    assert!(worker.client().call(|_, _| Ok(())).is_err());
    drop(result);
    assert_eq!(done.recv_timeout(TIMEOUT).unwrap(), 42);
    held.wait().unwrap();
    worker.close();
}
#[test]
fn panicking_call_fails_its_result_and_queued_callers_without_hanging() {
    let (mut worker, submissions, shared) = worker(2);
    let id = worker
        .client()
        .call(|o, now| o.admit(generation(), now))
        .unwrap()
        .wait()
        .unwrap();
    let gate = submissions.recv_timeout(TIMEOUT).unwrap();
    let (entered, rx) = mpsc::channel();
    let (release, blocked) = mpsc::channel();
    let panic = worker
        .client()
        .call::<()>(move |_, _| {
            entered.send(()).unwrap();
            blocked.recv().unwrap();
            panic!("fixture panic")
        })
        .unwrap();
    rx.recv_timeout(TIMEOUT).unwrap();
    let queued = worker.client().call(|_, _| Ok(42)).unwrap();
    release.send(()).unwrap();
    assert!(panic.wait().unwrap_err().contains("panicked"));
    assert!(queued.wait().unwrap_err().contains("panicked"));
    assert!(worker.client().call(|_, _| Ok(())).is_err());
    assert!(shared.closed.lock().unwrap().is_empty());
    gate.complete();
    worker.close();
    assert_eq!(*shared.closed.lock().unwrap(), vec![id]);
    assert_eq!(shared.committed.load(Ordering::Relaxed), 0);
}
#[test]
fn control_result_wakes_an_async_caller() {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, Wake, Waker},
    };
    struct Notify(Sender<()>);
    impl Wake for Notify {
        fn wake(self: Arc<Self>) {
            self.0.send(()).unwrap();
        }
    }
    let (mut worker, _, _) = worker(1);
    let (release, blocked) = mpsc::channel();
    let (tx, rx) = mpsc::channel();
    let mut call = worker
        .client()
        .call(move |_, _| {
            blocked.recv().unwrap();
            Ok(42)
        })
        .unwrap();
    let waker = Waker::from(Arc::new(Notify(tx)));
    let mut context = Context::from_waker(&waker);
    assert!(matches!(
        Pin::new(&mut call).poll(&mut context),
        Poll::Pending
    ));
    release.send(()).unwrap();
    rx.recv_timeout(TIMEOUT).unwrap();
    assert!(matches!(
        Pin::new(&mut call).poll(&mut context),
        Poll::Ready(Ok(42))
    ));
    worker.close();
}

#[test]
fn abandoned_admission_after_submission_defers_release_until_completion() {
    let (mut worker, submissions, shared) = worker(2);
    let (cleaned, done) = mpsc::channel();
    let admission = worker
        .client()
        .call_with_cleanup(
            |o, now| o.admit(generation(), now),
            move |o, id| {
                o.release(id)?;
                cleaned.send(id).unwrap();
                Ok(())
            },
        )
        .unwrap();
    let gate = submissions.recv_timeout(TIMEOUT).unwrap();
    drop(admission);
    let id = done.recv_timeout(TIMEOUT).unwrap();
    assert!(shared.closed.lock().unwrap().is_empty());
    assert_eq!(
        worker
            .client()
            .call(move |o, _| o.status(id))
            .unwrap()
            .wait()
            .unwrap(),
        Status::AwaitingCompletion
    );
    gate.complete();
    // Registration and completion may race; a worker call after the callback
    // observes reconciled retirement because completion has priority.
    worker
        .client()
        .call(move |o, _| {
            assert!(o.status(id).is_err());
            o.release(id)
        })
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(*shared.closed.lock().unwrap(), vec![id]);
    assert_eq!(shared.committed.load(Ordering::Relaxed), 0);
    worker.close();
}

#[test]
fn dropping_a_queued_call_skips_it_and_late_results_can_outlive_shutdown() {
    let (mut worker, _, _) = worker(3);
    let (entered, rx) = mpsc::channel();
    let (release, blocked) = mpsc::channel();
    let held = worker
        .client()
        .call(move |_, _| {
            entered.send(()).unwrap();
            blocked.recv().unwrap();
            Ok(())
        })
        .unwrap();
    rx.recv_timeout(TIMEOUT).unwrap();
    let skipped = worker
        .client()
        .call(|_, _| -> Result<(), String> { panic!("cancelled call ran") })
        .unwrap();
    drop(skipped);
    release.send(()).unwrap();
    held.wait().unwrap();
    let result = worker.client().call(|_, _| Ok(42)).unwrap();
    worker.client().call(|_, _| Ok(())).unwrap().wait().unwrap();
    worker.close();
    assert_eq!(result.wait().unwrap(), 42);
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Wake, Waker};
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}
fn runtime_service() -> (
    magnitude_engine::service::runtime::Service<Model>,
    magnitude_engine::chat::PreparedInput,
    Options,
    Receiver<Arc<Gate>>,
    Arc<Shared>,
    Arc<magnitude_engine::inputs::ByteBpeTokenizer>,
) {
    use magnitude_engine::{
        generation::constraints::{CacheLimits, Vocabulary},
        inputs::{BpeConfig, ByteBpeTokenizer, PieceKind},
        service::runtime::{Runtime, Service},
    };
    let included: BTreeSet<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let missing: Vec<_> = (0..=255).filter(|b| !included.contains(b)).collect();
    let mut pieces: Vec<_> = (0..=255)
        .map(|b| {
            char::from_u32(if included.contains(&b) {
                b as u32
            } else {
                256 + missing.iter().position(|&x| x == b).unwrap() as u32
            })
            .unwrap()
            .to_string()
        })
        .collect();
    pieces.push("<eos>".into());
    let mut kinds = vec![PieceKind::Normal; 256];
    kinds.push(PieceKind::Control);
    let tokenizer = Arc::new(
        ByteBpeTokenizer::new(BpeConfig {
            artifact_identity: "runtime-fixture".into(),
            pieces,
            kinds,
            merges: vec![],
            pattern: r".+|\s".into(),
            normalize_nfc: false,
            stop_tokens: BTreeSet::from([TokenId(256)]),
        })
        .unwrap(),
    );
    let input = magnitude_engine::chat::PreparedInput {
        artifact_identity: tokenizer.artifact_identity().into(),
        tokenizer_identity: tokenizer.identity().into(),
        tokens: vec![TokenId(1), TokenId(2)],
        constraint: None,
    };
    let options = Options {
        max_tokens: 2,
        output_capacity: 1,
        context_limit: 16,
        vocabulary: 257,
        stop_tokens: tokenizer.stop_tokens().clone(),
        sampling: Sampling::Greedy,
        seed: 0,
        forced_quantum: 0,
    };
    let shared = Arc::new(Shared::default());
    let state = shared.clone();
    let (tx, rx) = mpsc::channel();
    let owner_tokenizer = tokenizer.clone();
    let service = Service::spawn(
        move || {
            Runtime::new(
                Model {
                    shared: state,
                    submissions: tx,
                    _local: Rc::new(()),
                },
                Vocabulary::new(
                    owner_tokenizer,
                    257,
                    CacheLimits {
                        entries: 2,
                        bytes: 1024 * 1024,
                    },
                )?,
                Limits {
                    max_requests: 8,
                    max_batch: 2,
                    prefill_tokens: 8,
                    decode_tokens: 1,
                    decode_share: 0.5,
                    locality_seconds: 0.0,
                },
            )
        },
        8,
    )
    .unwrap();
    (service, input, options, rx, shared, tokenizer)
}
#[test]
fn prepared_admission_and_notified_publication_preserve_terminal_output() {
    use magnitude_engine::generation::FinishReason;
    let (mut service, input, mut options, submissions, shared, _) = runtime_service();
    options.output_capacity = 3;
    options.max_tokens = 3;
    let client = service.client();
    let mut bad = input.clone();
    bad.tokenizer_identity = "mismatch".into();
    assert!(block_on(client.admit(bad, options.clone())).is_err());
    assert!(submissions.try_recv().is_err());
    let mut request = block_on(client.admit(input, options)).unwrap();
    let id = request.id();
    let first = submissions.recv_timeout(TIMEOUT).unwrap();
    // Receiving starts before completion and sleeps until the reserved native wake.
    let completion = std::thread::spawn(move || first.complete());
    let publication = block_on(request.receive(1)).unwrap();
    completion.join().unwrap();
    assert_eq!(publication.tokens[0].index, 0);
    assert_eq!(publication.finish, None);
    let second = submissions.recv_timeout(TIMEOUT).unwrap();
    second.complete();
    let third = submissions.recv_timeout(TIMEOUT).unwrap();
    third.complete();
    let publication = block_on(request.receive(1)).unwrap();
    assert_eq!(publication.tokens[0].index, 1);
    assert_eq!(publication.finish, None);
    let publication = block_on(request.receive(1)).unwrap();
    assert_eq!(publication.tokens[0].index, 2);
    assert_eq!(publication.finish, Some(FinishReason::Length));
    drop(request);
    service.close();
    assert_eq!(*shared.closed.lock().unwrap(), vec![id]);
}
#[test]
fn cancelling_a_publication_future_releases_once_after_completion() {
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };
    struct Noop;
    impl Wake for Noop {
        fn wake(self: Arc<Self>) {}
    }
    let (mut service, input, options, submissions, shared, _) = runtime_service();
    let mut request = block_on(service.client().admit(input, options)).unwrap();
    let id = request.id();
    let gate = submissions.recv_timeout(TIMEOUT).unwrap();
    let mut receive = Box::pin(request.receive(1));
    let waker = Waker::from(Arc::new(Noop));
    let mut cx = Context::from_waker(&waker);
    assert!(matches!(receive.as_mut().poll(&mut cx), Poll::Pending));
    drop(receive);
    assert!(block_on(request.receive(1))
        .unwrap_err()
        .contains("released"));
    gate.complete();
    drop(request);
    service.close();
    assert_eq!(*shared.closed.lock().unwrap(), vec![id]);
    // Completion may win the owner event race; any accepted token is then
    // explicitly discarded by release, and no subsequent submission is allowed.
    assert!(shared.committed.load(Ordering::Relaxed) <= 1);
    assert!(submissions.try_recv().is_err());
}
#[test]
fn service_shutdown_wakes_a_waiting_publication_receiver() {
    let (mut service, input, options, submissions, shared, _) = runtime_service();
    let mut request = block_on(service.client().admit(input, options)).unwrap();
    let id = request.id();
    let gate = submissions.recv_timeout(TIMEOUT).unwrap();
    let closer = std::thread::spawn(move || service.close());
    assert_eq!(
        block_on(request.receive(1)).unwrap().finish,
        Some(magnitude_engine::generation::FinishReason::Cancelled)
    );
    gate.complete();
    closer.join().unwrap();
    drop(request);
    assert_eq!(*shared.closed.lock().unwrap(), vec![id]);
}

#[test]
fn fatal_owner_failure_retains_accepted_output_after_thread_teardown() {
    use magnitude_engine::generation::FinishReason;
    let (mut service, input, options, submissions, shared, _) = runtime_service();
    let client = service.client();
    let mut first = block_on(client.admit(input.clone(), options.clone())).unwrap();
    let gate = submissions.recv_timeout(TIMEOUT).unwrap();
    gate.complete();
    let second = block_on(client.admit(input, options)).unwrap();
    let gate = submissions.recv_timeout(TIMEOUT).unwrap();
    gate.fail.store(true, Ordering::Release);
    gate.complete();
    // Closing joins the thread; the first request's accepted token remains
    // available independently of live device objects and worker control calls.
    service.close();
    let publication = block_on(first.receive(1)).unwrap();
    assert_eq!(publication.tokens.len(), 1);
    assert_eq!(publication.tokens[0].index, 0);
    assert_eq!(publication.finish, Some(FinishReason::Failed));
    assert_eq!(publication.error.as_deref(), Some("device failed"));
    assert_eq!(shared.committed.load(Ordering::Relaxed), 1);
    drop(first);
    drop(second);
}

#[test]
fn native_chat_session_composes_scheduled_tokens_and_releases_on_string_stop() {
    use magnitude_engine::chat::{
        ChatRequest, Event, PreparedChat, Session, TemplateBundle, TemplateSelection,
        TemplateVariant, TerminalCause,
    };
    let (mut service, _, options, submissions, shared, tokenizer) = runtime_service();
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: "{{ messages[0].content }}".into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    let request = ChatRequest::new(
        vec![serde_json::json!({"role":"user","content":"hello"})],
        0,
    );
    let prepared =
        PreparedChat::prepare(&bundle, &tokenizer, &request, &TemplateSelection::default())
            .unwrap();
    let native = std::thread::spawn(move || {
        while let Ok(gate) = submissions.recv() {
            gate.complete();
        }
    });
    let mut session = block_on(Session::open(
        &service.client(),
        &prepared,
        &tokenizer,
        options,
        vec!["\u{7}".into()],
        4096,
    ))
    .unwrap();
    let publication = block_on(session.next()).unwrap().unwrap();
    assert_eq!(
        publication.events,
        vec![Event::Finish {
            cause: TerminalCause::UserStop
        }]
    );
    assert!(publication.error.is_none());
    assert!(block_on(session.next()).unwrap().is_none());
    drop(session);
    service.close();
    native.join().unwrap();
    assert_eq!(shared.closed.lock().unwrap().len(), 1);
}

#[test]
fn failed_chat_session_drains_multiple_publications_before_reporting_error() {
    use magnitude_engine::chat::{
        ChatRequest, Event, PreparedChat, Session, TemplateBundle, TemplateSelection,
        TemplateVariant, TerminalCause,
    };
    let (mut service, _, mut options, submissions, _, tokenizer) = runtime_service();
    options.max_tokens = 100;
    options.output_capacity = 64;
    options.context_limit = 256;
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: "{{ messages[0].content }}".into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    let prepared = PreparedChat::prepare(
        &bundle,
        &tokenizer,
        &ChatRequest::new(
            vec![serde_json::json!({"role":"user","content":"hello"})],
            0,
        ),
        &TemplateSelection::default(),
    )
    .unwrap();
    let mut session = block_on(Session::open(
        &service.client(),
        &prepared,
        &tokenizer,
        options,
        vec![],
        4096,
    ))
    .unwrap();
    for _ in 0..33 {
        submissions.recv_timeout(TIMEOUT).unwrap().complete();
    }
    let failed = submissions.recv_timeout(TIMEOUT).unwrap();
    failed.fail.store(true, Ordering::Release);
    failed.complete();
    service.close();
    let first = block_on(session.next()).unwrap().unwrap();
    assert!(first.error.is_none());
    assert!(!first
        .events
        .iter()
        .any(|event| matches!(event, Event::Finish { .. })));
    let last = block_on(session.next()).unwrap().unwrap();
    assert_eq!(last.error.as_deref(), Some("device failed"));
    assert_eq!(last.usage.unwrap().completion_tokens, 33);
    assert!(last.events.iter().any(|event| matches!(
        event,
        Event::Finish {
            cause: TerminalCause::Failed
        }
    )));
    assert!(block_on(session.next()).unwrap().is_none());
}

#[test]
fn stop_acknowledges_stable_usage_while_native_work_is_still_outstanding() {
    use magnitude_engine::generation::{FinishReason, Usage};
    let (mut service, input, mut options, submissions, shared, _) = runtime_service();
    options.output_capacity = 2;
    options.max_tokens = 3;
    let mut request = block_on(service.client().admit(input, options)).unwrap();
    submissions.recv_timeout(TIMEOUT).unwrap().complete();
    let pending = submissions.recv_timeout(TIMEOUT).unwrap();
    let stopped = block_on(request.stop()).unwrap();
    assert_eq!(
        stopped.usage,
        Some(Usage {
            prompt_tokens: 2,
            completion_tokens: 1
        })
    );
    assert_eq!(stopped.finish, Some(FinishReason::Cancelled));
    assert!(stopped.tokens.is_empty());
    assert_eq!(shared.committed.load(Ordering::Relaxed), 1);
    assert!(shared.closed.lock().unwrap().is_empty());
    pending.complete();
    drop(request);
    service.close();
    assert_eq!(shared.committed.load(Ordering::Relaxed), 1);
    assert_eq!(shared.closed.lock().unwrap().len(), 1);
}

#[test]
fn session_sse_uses_acknowledged_counts_including_suppressed_stop_tokens() {
    use magnitude_engine::chat::{
        ChatRequest, PreparedChat, Session, SseResponse, TemplateBundle, TemplateSelection,
        TemplateVariant,
    };
    let (mut service, _, options, submissions, shared, tokenizer) = runtime_service();
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: "{{ messages[0].content }}".into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    let prepared = PreparedChat::prepare(
        &bundle,
        &tokenizer,
        &ChatRequest::new(
            vec![serde_json::json!({"role":"user","content":"hello"})],
            0,
        ),
        &TemplateSelection::default(),
    )
    .unwrap();
    let native = std::thread::spawn(move || {
        while let Ok(gate) = submissions.recv() {
            gate.complete();
        }
    });
    let mut session = block_on(Session::open(
        &service.client(),
        &prepared,
        &tokenizer,
        options,
        vec!["\u{7}".into()],
        4096,
    ))
    .unwrap();
    let mut response =
        SseResponse::new("chatcmpl-runtime".into(), "fixture".into(), 0, true, 8192).unwrap();
    let frames = block_on(session.next_sse(&mut response)).unwrap().unwrap();
    assert_eq!(frames.last().unwrap(), b"data: [DONE]\n\n");
    let values = frames[..frames.len() - 1]
        .iter()
        .map(|frame| {
            serde_json::from_str::<serde_json::Value>(
                std::str::from_utf8(frame)
                    .unwrap()
                    .strip_prefix("data: ")
                    .unwrap()
                    .trim(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let usage = values.iter().find_map(|value| value.get("usage")).unwrap();
    assert_eq!(usage["prompt_tokens"], prepared.input().tokens.len());
    assert!(values
        .iter()
        .any(|value| value["choices"][0]["finish_reason"] == "stop"));
    assert!(!values
        .iter()
        .any(|value| value["choices"][0]["delta"].get("content").is_some()));
    assert!(block_on(session.next_sse(&mut response)).unwrap().is_none());
    drop(session);
    service.close();
    native.join().unwrap();
    assert_eq!(
        usage["completion_tokens"],
        shared.committed.load(Ordering::Relaxed)
    );
    assert!(usage["completion_tokens"].as_u64().unwrap() >= 1);
}

#[test]
fn nonstream_session_collects_the_same_parser_and_generation_usage() {
    use magnitude_engine::chat::{
        ChatRequest, CompleteResponse, PreparedChat, Session, TemplateBundle, TemplateSelection,
        TemplateVariant,
    };
    let (mut service, _, options, submissions, shared, tokenizer) = runtime_service();
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: "{{ messages[0].content }}".into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    let prepared = PreparedChat::prepare(
        &bundle,
        &tokenizer,
        &ChatRequest::new(
            vec![serde_json::json!({"role":"user","content":"hello"})],
            0,
        ),
        &TemplateSelection::default(),
    )
    .unwrap();
    let native = std::thread::spawn(move || {
        while let Ok(gate) = submissions.recv() {
            gate.complete();
        }
    });
    let mut session = block_on(Session::open(
        &service.client(),
        &prepared,
        &tokenizer,
        options,
        vec![],
        4096,
    ))
    .unwrap();
    let response =
        CompleteResponse::new("chatcmpl-runtime".into(), "fixture".into(), 0, 8192).unwrap();
    let bytes = block_on(session.complete(response)).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["object"], "chat.completion");
    assert_eq!(value["choices"][0]["finish_reason"], "length");
    assert_eq!(value["choices"][0]["message"]["content"], "\u{7}\u{7}");
    assert_eq!(
        value["usage"]["prompt_tokens"],
        prepared.input().tokens.len()
    );
    assert_eq!(value["usage"]["completion_tokens"], 2);
    drop(session);
    service.close();
    native.join().unwrap();
    assert_eq!(shared.committed.load(Ordering::Relaxed), 2);
    assert_eq!(shared.closed.lock().unwrap().len(), 1);
}

fn http_server(
    service: magnitude_engine::service::runtime::Service<Model>,
    tokenizer: Arc<magnitude_engine::inputs::ByteBpeTokenizer>,
) -> magnitude_engine::serving::Server<Model> {
    use magnitude_engine::{
        chat::{TemplateBundle, TemplateVariant},
        serving::{Config, Server},
    };
    let templates = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: "{{ messages[0].content }}".into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    Server::new(
        service,
        tokenizer,
        templates,
        Config {
            model: "fixture".into(),
            context_tokens: 4096,
            vocabulary: 257,
            output_capacity: 1,
            forced_quantum: 0,
            template_variant: None,
            template_override: None,
            max_body_bytes: 4096,
            max_response_bytes: 16384,
            max_connections: 8,
            request_timeout: Duration::from_secs(5),
        },
    )
    .unwrap()
}
async fn http_request(
    address: std::net::SocketAddr,
    method: &str,
    path: &str,
    body: &[u8],
) -> (u16, Vec<u8>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
    socket.write_all(format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n\r\n", body.len()).as_bytes()).await.unwrap();
    socket.write_all(body).await.unwrap();
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), socket.read_to_end(&mut response))
        .await
        .unwrap()
        .unwrap();
    let split = response
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .unwrap();
    let header = std::str::from_utf8(&response[..split]).unwrap();
    let status = header.split_whitespace().nth(1).unwrap().parse().unwrap();
    let mut data = &response[split + 4..];
    if header
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        let mut decoded = Vec::new();
        loop {
            let end = data.windows(2).position(|bytes| bytes == b"\r\n").unwrap();
            let size = usize::from_str_radix(
                std::str::from_utf8(&data[..end])
                    .unwrap()
                    .split(';')
                    .next()
                    .unwrap(),
                16,
            )
            .unwrap();
            data = &data[end + 2..];
            if size == 0 {
                break;
            }
            decoded.extend_from_slice(&data[..size]);
            assert_eq!(&data[size..size + 2], b"\r\n");
            data = &data[size + 2..];
        }
        (status, decoded)
    } else {
        (status, data.to_vec())
    }
}
#[tokio::test(flavor = "current_thread")]
async fn http_routes_validate_and_serve_streaming_and_complete_responses() {
    tokio::task::LocalSet::new().run_until(async {
        let (service, _, _, submissions, shared, tokenizer) = runtime_service();
        let server = http_server(service, tokenizer);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let serving = tokio::task::spawn_local(server.serve(listener, async { let _ = stopped.await; }));
        let native = std::thread::spawn(move || while let Ok(gate) = submissions.recv() { gate.complete(); });
        assert_eq!(http_request(address, "GET", "/health", b"").await.0, 200);
        let (status, models) = http_request(address, "GET", "/v1/models", b"").await;
        assert_eq!(status, 200);
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&models).unwrap()["data"][0]["id"], "fixture");
        assert_eq!(http_request(address, "GET", "/missing", b"").await.0, 404);
        assert_eq!(http_request(address, "GET", "/v1/chat/completions", b"").await.0, 405);
        assert_eq!(http_request(address, "POST", "/v1/chat/completions", b"{").await.0, 422);
        assert_eq!(http_request(address, "POST", "/v1/chat/completions", &vec![b' '; 4097]).await.0, 413);
        let mut body = serde_json::json!({"model":"other","messages":[{"role":"user","content":"hello"}],"max_tokens":2});
        assert_eq!(http_request(address, "POST", "/v1/chat/completions", &serde_json::to_vec(&body).unwrap()).await.0, 404);
        body["model"] = serde_json::json!("fixture"); body["temperature"] = serde_json::json!(0.5);
        assert_eq!(http_request(address, "POST", "/v1/chat/completions", &serde_json::to_vec(&body).unwrap()).await.0, 400);
        assert_eq!(shared.committed.load(Ordering::Relaxed), 0);
        body["temperature"] = serde_json::json!(0);
        let (status, complete) = http_request(address, "POST", "/v1/chat/completions", &serde_json::to_vec(&body).unwrap()).await;
        assert_eq!(status, 200);
        let complete: serde_json::Value = serde_json::from_slice(&complete).unwrap();
        assert_eq!(complete["choices"][0]["message"]["content"], "\u{7}\u{7}");
        assert_eq!(complete["usage"]["completion_tokens"], 2);
        body["stream"] = serde_json::json!(true); body["stream_options"] = serde_json::json!({"include_usage":true});
        let (status, streaming) = http_request(address, "POST", "/v1/chat/completions", &serde_json::to_vec(&body).unwrap()).await;
        assert_eq!(status, 200);
        let streaming = std::str::from_utf8(&streaming).unwrap();
        assert!(streaming.ends_with("data: [DONE]\n\n"));
        let chunks = streaming.split("\n\n").filter_map(|frame| frame.strip_prefix("data: ")).filter(|value| *value != "[DONE]")
            .map(|value| serde_json::from_str::<serde_json::Value>(value).unwrap()).collect::<Vec<_>>();
        assert_eq!(chunks.iter().find_map(|chunk| chunk.get("usage")).unwrap(), &complete["usage"]);
        stop.send(()).unwrap(); serving.await.unwrap().unwrap(); native.join().unwrap();
        assert_eq!(shared.closed.lock().unwrap().len(), 2);
    }).await;
}

#[tokio::test(flavor = "current_thread")]
async fn http_disconnect_retires_streaming_and_nonstream_requests_after_native_completion() {
    use tokio::io::AsyncWriteExt;
    tokio::task::LocalSet::new().run_until(async {
        for stream in [false, true] {
            let (service, _, _, submissions, shared, tokenizer) = runtime_service();
            let server = http_server(service, tokenizer);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let (stop, stopped) = tokio::sync::oneshot::channel();
            let serving = tokio::task::spawn_local(server.serve(listener, async { let _ = stopped.await; }));
            let body = serde_json::to_vec(&serde_json::json!({"model":"fixture","messages":[{"role":"user","content":"hello"}],"max_tokens":2,"stream":stream})).unwrap();
            let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
            socket.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n", body.len()).as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
            let pending = tokio::time::timeout(TIMEOUT, async {
                loop {
                    if let Ok(gate) = submissions.try_recv() { break gate; }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }).await.unwrap();
            drop(socket);
            // Let the connection observe EOF while native completion is withheld.
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(shared.closed.lock().unwrap().is_empty());
            pending.complete();
            tokio::time::timeout(TIMEOUT, async {
                while shared.closed.lock().unwrap().is_empty() {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }).await.unwrap();
            assert_eq!(shared.committed.load(Ordering::Relaxed), 0, "stream={stream}");
            assert_eq!(shared.closed.lock().unwrap().len(), 1);
            assert!(submissions.try_recv().is_err());
            stop.send(()).unwrap();
            serving.await.unwrap().unwrap();
        }
    }).await;
}

#[tokio::test(flavor = "current_thread")]
async fn http_shutdown_waits_for_native_completion_and_discards_pending_output() {
    use tokio::io::AsyncWriteExt;
    tokio::task::LocalSet::new().run_until(async {
        let (service, _, _, submissions, shared, tokenizer) = runtime_service();
        let server = http_server(service, tokenizer);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let finished = Arc::new(AtomicBool::new(false));
        let observed = finished.clone();
        let serving = tokio::task::spawn_local(async move {
            server.serve(listener, async { let _ = stopped.await; }).await.unwrap();
            finished.store(true, Ordering::Release);
        });
        let body = br#"{"model":"fixture","messages":[{"role":"user","content":"hello"}],"max_tokens":2,"stream":true}"#;
        let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
        socket.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n", body.len()).as_bytes()).await.unwrap();
        socket.write_all(body).await.unwrap();
        let pending = tokio::time::timeout(TIMEOUT, async {
            loop {
                if let Ok(gate) = submissions.try_recv() { break gate; }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }).await.unwrap();
        let state = shared.clone();
        let native = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            let closed_early = !state.closed.lock().unwrap().is_empty();
            let finished_early = observed.load(Ordering::Acquire);
            pending.complete();
            assert!(!closed_early);
            assert!(!finished_early);
        });
        stop.send(()).unwrap();
        serving.await.unwrap();
        native.join().unwrap();
        assert_eq!(shared.committed.load(Ordering::Relaxed), 0);
        assert_eq!(shared.closed.lock().unwrap().len(), 1);
        drop(socket);
    }).await;
}

#[test]
fn service_checkpoint_forks_retained_output_and_numerical_continuations_on_owner() {
    let (mut service, input, mut options, submissions, shared, _) = runtime_service();
    options.max_tokens = 3;
    options.output_capacity = 1;
    let original = block_on(service.client().admit(input, options)).unwrap();
    let pending = submissions.recv_timeout(TIMEOUT).unwrap();
    assert!(block_on(original.checkpoint()).is_err());
    assert_eq!(shared.snapshots.load(Ordering::Relaxed), 0);
    pending.complete();
    // Checkpoint after reconciliation without draining publication credit.
    let snapshot = {
        let start = std::time::Instant::now();
        loop {
            if let Ok(snapshot) = block_on(original.checkpoint()) {
                break snapshot;
            }
            assert!(start.elapsed() < TIMEOUT);
            std::thread::yield_now();
        }
    };
    assert_eq!(shared.snapshots.load(Ordering::Relaxed), 1);
    let mut fork = block_on(snapshot.fork()).unwrap();
    assert_ne!(original.id(), fork.id());
    assert_eq!(
        shared.positions.lock().unwrap().get(&original.id()),
        Some(&2)
    );
    assert_eq!(shared.positions.lock().unwrap().get(&fork.id()), Some(&2));
    assert!(submissions.try_recv().is_err()); // Both retain the same full output queue.
    drop(original);
    let first = block_on(fork.receive(1)).unwrap();
    assert_eq!(first.tokens.len(), 1);
    let advancing = submissions.recv_timeout(TIMEOUT).unwrap();
    assert!(block_on(fork.checkpoint()).is_err());
    advancing.complete();
    let second = block_on(fork.receive(1)).unwrap();
    assert_eq!(second.tokens.len(), 1);
    assert_eq!(second.usage.unwrap().completion_tokens, 2);
    let mut sibling = block_on(snapshot.fork()).unwrap();
    let retained = block_on(sibling.receive(1)).unwrap();
    assert_eq!(retained.tokens, first.tokens);
    assert_eq!(retained.usage.unwrap().completion_tokens, 1);
    drop(snapshot);
    // Complete outstanding work while shutdown drains all reserved releases.
    drop(fork);
    drop(sibling);
    let native = std::thread::spawn(move || {
        while let Ok(gate) = submissions.recv() {
            gate.complete();
        }
    });
    service.close();
    native.join().unwrap();
    assert_eq!(shared.snapshots.load(Ordering::Relaxed), 0);
    assert_eq!(shared.closed.lock().unwrap().len(), 3);
}

#[test]
fn service_checkpoints_are_bounded_releasable_and_invalid_after_shutdown() {
    let (mut service, input, mut options, submissions, shared, _) = runtime_service();
    options.max_tokens = 0;
    let original = block_on(service.client().admit(input, options)).unwrap();
    let mut snapshots = Vec::new();
    for _ in 0..8 {
        snapshots.push(block_on(original.checkpoint()).unwrap());
    }
    assert!(block_on(original.checkpoint())
        .err()
        .unwrap()
        .contains("checkpoint limit"));
    assert_eq!(shared.snapshots.load(Ordering::Relaxed), 8);
    drop(snapshots.pop());
    // Reserved cleanup precedes the next control.
    snapshots.push(block_on(original.checkpoint()).unwrap());
    let snapshot = snapshots.pop().unwrap();
    drop(original);
    let mut fork = block_on(snapshot.fork()).unwrap();
    let terminal = block_on(fork.receive(1)).unwrap();
    assert!(terminal.finish.is_some());
    assert_eq!(terminal.usage.unwrap().completion_tokens, 0);
    assert!(submissions.try_recv().is_err());
    service.close();
    assert_eq!(shared.snapshots.load(Ordering::Relaxed), 0);
    assert!(block_on(snapshot.fork()).is_err());
    drop(fork);
    drop(snapshots);
    drop(snapshot);
}

#[test]
fn typed_source_admission_preserves_layout_and_releases_only_on_owner_after_completion() {
    use magnitude_engine::inputs::{BoundaryRule, InputSpan};
    let (mut service, input, options, submissions, shared, _) = runtime_service();
    let caller = std::thread::current().id();
    let dropped_on = Arc::new(Mutex::new(None));
    let source = HostInput {
        tokens: input.tokens.clone(),
        layout: InputLayout::new(
            input.tokens.len(),
            vec![InputSpan {
                start: 0,
                end: 1,
                identity: "condition".into(),
                boundaries: BoundaryRule::Causal,
                language_history: false,
            }],
        )
        .unwrap(),
        reject_open: false,
        dropped_on: dropped_on.clone(),
    };
    let request = block_on(service.client().admit_input(input, source, options)).unwrap();
    let id = request.id();
    let gate = submissions.recv_timeout(TIMEOUT).unwrap();
    let owner_thread = shared.input_open_threads.lock().unwrap()[0];
    assert_ne!(owner_thread, caller);
    assert!(shared.input_sources.lock().unwrap().contains_key(&id));
    drop(request);
    block_on(service.client().check()).unwrap();
    assert!(dropped_on.lock().unwrap().is_none());
    gate.complete();
    service.close();
    assert_eq!(*dropped_on.lock().unwrap(), Some(owner_thread));
    assert!(shared.input_sources.lock().unwrap().is_empty());
    assert_eq!(*shared.closed.lock().unwrap(), vec![id]);
}
#[test]
fn typed_source_admission_rejects_bad_vocabulary_layout_and_open_without_resources() {
    let (mut service, input, options, submissions, shared, _) = runtime_service();
    for variant in 0..4 {
        let mut prepared = input.clone();
        if variant == 0 {
            prepared.artifact_identity = "foreign artifact".into();
        }
        let dropped_on = Arc::new(Mutex::new(None));
        let source = HostInput {
            tokens: if variant == 1 {
                vec![TokenId(7)]
            } else {
                input.tokens.clone()
            },
            layout: InputLayout::new(if variant == 2 { 1 } else { input.tokens.len() }, vec![])
                .unwrap(),
            reject_open: variant == 3,
            dropped_on: dropped_on.clone(),
        };
        assert!(block_on(
            service
                .client()
                .admit_input(prepared, source, options.clone())
        )
        .is_err());
        assert!(shared.positions.lock().unwrap().is_empty());
        assert!(shared.input_sources.lock().unwrap().is_empty());
        assert!(shared.input_open_threads.lock().unwrap().is_empty());
        assert!(submissions.try_recv().is_err());
        assert!(dropped_on.lock().unwrap().is_some());
    }
    service.close();
}
