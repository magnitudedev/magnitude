//! A device wait answered by host work alone: a call parks on its answer
//! word (CUDA, Metal: its kernel spins; Vulkan: its run holds at the park
//! boundary), the host writes the data it waits for into host-shared memory
//! and then the answer word (and on Vulkan signals the park's timeline, a
//! host semaphore signal), and the call and the call after it read that
//! memory in place. The answer makes no other driver call, so it lands
//! while other threads are stuck in the driver behind the waiting call: one
//! preparing and first launching a graph (sealing, allocating with a host
//! copy, its first submission), one queueing runs behind the call until its
//! submissions block. The call must end within a bound of its answer, with
//! the bytes the host wrote.

use seismic::{BackendName, Element, NativeGraphFamily, NativeSpecialization, Tensor, Timeline, TimelineNode};
use seismic_native_tests::{parked_scale, scale_rows};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// Kernels per graph a busy thread prepares and launches.
const NODES: usize = 16;
/// The most runs the queueing thread submits behind the waiting call.
const QUEUED: u64 = 1 << 14;

/// A `NODES`-kernel graph of `scale_rows` over one row, sealed on the
/// calling thread, and a launch of it.
fn busy_graph(device: &seismic::Device) -> impl FnMut() -> seismic::NativeGraphCompletion {
    let kernel = scale_rows::native_for_device(device, &NativeSpecialization::new().with_param("ROWS", 1)).unwrap();
    let mut graph = device.native_graph();
    let input = graph.port(Element::f32(), &[1, 4]).unwrap();
    let mut value = graph.enqueue(&kernel, scale_rows::WorkflowArgs { x: input.tensor().into(), factor: 1.0 }).unwrap().value;
    for _ in 1..NODES {
        value = graph.enqueue(&kernel, scale_rows::WorkflowArgs { x: (&value).into(), factor: 1.0 }).unwrap().value;
    }
    graph.export(&value).unwrap();
    let plan = graph.seal().unwrap();
    let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
    let mut slot = family.new_slot(&[family.upload_bytes()]).unwrap();
    let mut output_slot = Some(family.new_output_slot().unwrap());
    let x = Tensor::from_host(device, Element::f32(), &[1, 4], &f32_bytes(&[1.0; 4])).unwrap();
    move || {
        let outputs = output_slot.take().unwrap().activate(&plan).unwrap();
        let mut bindings = plan.bindings();
        bindings.set(&input, &x).unwrap();
        let (outputs, completion) = slot.activate(&plan).unwrap().attach(bindings, outputs).unwrap().submit().unwrap();
        output_slot = Some(outputs.recycle().unwrap());
        completion
    }
}

/// On its own thread of `scope`: launch a graph prepared before any park
/// (`busy_graph`; a graph slot is created only between flights) once
/// (`queue` 1), or queue `queue` launches of it; each submit counted once
/// it returned. The thread ends once its runs completed, and the scope ends
/// once it has, so no run outlives the test that queued it.
fn busy<'scope>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    mut launch: impl FnMut() -> seismic::NativeGraphCompletion + Send + 'scope,
    queue: u64,
    submitted: &'scope AtomicU64,
) {
    scope.spawn(move || {
        let completions = (0..queue)
            .map(|_| {
                let completion = launch();
                submitted.fetch_add(1, Ordering::Release);
                completion
            })
            .collect::<Vec<_>>();
        for completion in completions {
            completion.wait().unwrap();
        }
    });
}

/// The waiting call's answer: its word, then the park's host signal. Given
/// when the test gives it, or when it is dropped unanswered, so a failing
/// check ends the waiting call and the runs queued behind it drain rather
/// than leave the busy threads' join waiting on a call that never ends.
struct Answer<'a> {
    word: &'a Tensor,
    parks: &'a Timeline,
    parked: Option<u64>,
    answered: bool,
}

impl Answer<'_> {
    fn give(&mut self) -> Result<(), String> {
        self.answered = true;
        self.word.store_release(0, 1).map_err(|error| error.to_string())?;
        self.parked.map_or(Ok(()), |value| self.parks.answer(value, seismic::ParkAnswer { by: seismic::AnsweredBy::Arrival, park: "test", layer: 0 }).map_err(|error| error.to_string()))
    }
}

impl Drop for Answer<'_> {
    fn drop(&mut self) {
        if !self.answered {
            if let Err(error) = self.give() {
                eprintln!("the unanswered park could not be answered: {error}");
            }
        }
    }
}

#[test]
fn a_wait_answered_by_host_words_alone_ends_while_other_threads_are_stuck_in_the_driver() {
    let n = 1024u64;
    for device in seismic_native_tests::devices(&[BackendName::Metal, BackendName::Cuda, BackendName::Vulkan]) {
        let backend = device.backend();
        let kernel = parked_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
        let after_kernel = scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1)).unwrap();
        let mut graph = device.native_graph();
        let answer = graph.port(Element::i32(), &[1]).unwrap();
        let skip = graph.port(Element::i32(), &[1]).unwrap();
        let input = graph.port(Element::f32(), &[n]).unwrap();
        let park = graph.timeline(TimelineNode::Park);
        // The waiting call reads `input` in place once answered, and the
        // call after it reads what it wrote.
        let waited = graph
            .enqueue(
                &kernel,
                parked_scale::WorkflowArgs { answer: answer.tensor().into(), skip: skip.tensor().into(), x: input.tensor().into() },
            )
            .unwrap()
            .value;
        let value = graph
            .enqueue(&after_kernel, scale_rows::WorkflowArgs { x: (&waited.reshape(&[1, n])).into(), factor: 2.0 })
            .unwrap()
            .value;
        let after = graph.timeline(TimelineNode::Signal);
        graph.export(&value).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        let mut slot = family.new_slot(&[family.upload_bytes()]).unwrap();

        // The data the call waits for, host-shared as a host tier is, and
        // its words; all written before the busy threads start.
        let data = Tensor::shared(&device, Element::f32(), &[n]).unwrap();
        let word = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        let skipped = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        // SAFETY: no device work uses them yet.
        unsafe {
            data.write_shared(0, &vec![0; 4 * n as usize]).unwrap();
            word.write_shared(0, &[0; 4]).unwrap();
            skipped.write_shared(0, &[0; 4]).unwrap();
        }
        let (parks, done) = (Timeline::new(&device).unwrap(), Timeline::new(&device).unwrap());
        let start = Instant::now();
        let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
        let mut bindings = plan.bindings();
        bindings.set(&answer, &word).unwrap();
        bindings.set(&skip, &skipped).unwrap();
        bindings.set(&input, &data).unwrap();
        bindings.set_park(park, &parks).unwrap();
        bindings.set_timeline(after, &done, 1).unwrap();
        // The busy threads' graphs, prepared before the park: a graph slot
        // is created only between flights.
        let (first, queueing) = (busy_graph(&device), busy_graph(&device));
        let (outputs, completion) = slot.activate(&plan).unwrap().attach(bindings, outputs).unwrap().submit().unwrap();
        let parked = outputs.park(park);
        let answering = Answer { word: &word, parks: &parks, parked, answered: false };
        match parked {
            Some(value) => {
                let deadline = Instant::now() + Duration::from_secs(5);
                while !parks.parked(value).unwrap() {
                    assert!(Instant::now() < deadline, "{backend:?}: the run never reached its park");
                    std::thread::yield_now();
                }
            }
            None => std::thread::sleep(Duration::from_millis(100)),
        }
        assert!(!done.reached(1).unwrap(), "{backend:?}: the call waits for its answer");

        // Other threads in the driver behind the waiting call: one first
        // launching a graph, one queueing runs until its submits
        // block or it has queued them all. Each ends once its runs have, and
        // the scope once both have.
        let (prepared, queued) = (AtomicU64::new(0), AtomicU64::new(0));
        std::thread::scope(|scope| {
            // Dropped, and so given, before the scope joins the busy threads.
            let mut answering = answering;
            busy(scope, first, 1, &prepared);
            busy(scope, queueing, QUEUED, &queued);
            let (mut seen, mut progressed) = (0, Instant::now());
            while queued.load(Ordering::Acquire) < QUEUED && progressed.elapsed() < Duration::from_secs(1) {
                let now = queued.load(Ordering::Acquire);
                if now != seen {
                    (seen, progressed) = (now, Instant::now());
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            eprintln!(
                "{backend:?}: at the answer, the prepared graph launched {} time(s), {} runs queued behind the call, at {:?}",
                prepared.load(Ordering::Acquire),
                queued.load(Ordering::Acquire),
                start.elapsed()
            );

            // The answer: host writes alone, then the park's host signal.
            let values = (0..n).map(|value| value as f32 + 1.0).collect::<Vec<_>>();
            // SAFETY: the waiting call reads the data only once answered.
            unsafe { data.write_shared(0, &f32_bytes(&values)).unwrap() };
            answering.give().unwrap();
            let answered = Instant::now();
            if !done.wait(1, Duration::from_secs(10)).unwrap() {
                // A call that never ends after its answer leaves the runs
                // queued behind it in the driver: no unwinding ends them,
                // and the scope would wait for them forever.
                eprintln!(
                    "{backend:?}: the call did not end within 10 s of its answer (parked {parked:?}, park reached {:?}), \
                     {} runs queued behind it",
                    parked.map(|value| parks.parked(value)),
                    queued.load(Ordering::Acquire)
                );
                std::process::abort();
            }
            eprintln!("{backend:?}: the call ended {:?} after its answer", answered.elapsed());
            completion.wait().unwrap();
            let read = outputs
                .exported(&value)
                .unwrap()
                .read_to_host()
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(read, values.iter().map(|value| value * 2.0).collect::<Vec<_>>(), "{backend:?}: the calls read the host's bytes");
        });
    }
}

/// A wait for a batch on the device's transfer queue returns as the batch
/// lands, beside a call parked on its answer with a signal queued behind it:
/// what the wait sleeps on must not wait for that call. A wake held behind
/// the call runs the wait to its full timeout though the batch landed long
/// before.
#[test]
fn a_wait_for_a_transfer_returns_as_it_lands_beside_a_parked_call() {
    const RECORDS: u64 = 20;
    const RECORD: u64 = 1_900_544;
    const TIMEOUT: Duration = Duration::from_secs(5);
    let n = 1024u64;
    for device in seismic_native_tests::devices(&[BackendName::Cuda, BackendName::Vulkan]) {
        let backend = device.backend();
        let kernel = parked_scale::native_for_device(&device, &NativeSpecialization::new()).unwrap();
        let mut graph = device.native_graph();
        let answer = graph.port(Element::i32(), &[1]).unwrap();
        let skip = graph.port(Element::i32(), &[1]).unwrap();
        let input = graph.port(Element::f32(), &[n]).unwrap();
        let park = graph.timeline(TimelineNode::Park);
        let waited = graph
            .enqueue(&kernel, parked_scale::WorkflowArgs { answer: answer.tensor().into(), skip: skip.tensor().into(), x: input.tensor().into() })
            .unwrap()
            .value;
        let after = graph.timeline(TimelineNode::Signal);
        graph.export(&waited).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        let mut slot = family.new_slot(&[family.upload_bytes()]).unwrap();

        let data = Tensor::shared(&device, Element::f32(), &[n]).unwrap();
        let word = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        let skipped = Tensor::shared(&device, Element::i32(), &[1]).unwrap();
        // SAFETY: no device work uses them yet.
        unsafe {
            data.write_shared(0, &vec![0; 4 * n as usize]).unwrap();
            word.write_shared(0, &[0; 4]).unwrap();
            skipped.write_shared(0, &[0; 4]).unwrap();
        }
        let shape = [RECORDS, RECORD];
        let (mut tier, _resizer) = Tensor::reserved_in(seismic::ReservedContract::Shared, &device, Element::bool(), &shape, RECORDS, None).unwrap();
        tier.write_from_host(&vec![7u8; (RECORDS * RECORD) as usize]).unwrap();
        let into = Tensor::zeros(&device, Element::bool(), &shape).unwrap();
        let transfer = seismic::Transfer::new(&device).unwrap();
        let runs = (0..RECORDS)
            .map(|record| {
                seismic::TransferItem::Run(seismic::TransferRun {
                    source: &tier,
                    source_offset: record * RECORD,
                    destination: &into,
                    destination_offset: record * RECORD,
                    bytes: RECORD,
                })
            })
            .collect::<Vec<_>>();

        let (parks, done) = (Timeline::new(&device).unwrap(), Timeline::new(&device).unwrap());
        let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
        let mut bindings = plan.bindings();
        bindings.set(&answer, &word).unwrap();
        bindings.set(&skip, &skipped).unwrap();
        bindings.set(&input, &data).unwrap();
        bindings.set_park(park, &parks).unwrap();
        bindings.set_timeline(after, &done, 1).unwrap();
        let (outputs, completion) = slot.activate(&plan).unwrap().attach(bindings, outputs).unwrap().submit().unwrap();
        let parked = outputs.park(park);
        let mut answering = Answer { word: &word, parks: &parks, parked, answered: false };
        match parked {
            Some(value) => {
                let deadline = Instant::now() + Duration::from_secs(5);
                while !parks.parked(value).unwrap() {
                    assert!(Instant::now() < deadline, "{backend:?}: the run never reached its park");
                    std::thread::yield_now();
                }
            }
            None => std::thread::sleep(Duration::from_millis(100)),
        }

        let began = Instant::now();
        let value = transfer.copy(&runs).unwrap();
        let landed = transfer.wait(value, TIMEOUT).unwrap();
        let waited = began.elapsed();
        let still_parked = !done.reached(1).unwrap();
        eprintln!("{backend:?}: a {} MB batch waited for in {waited:?}, the call still parked: {still_parked}", RECORDS * RECORD / 1_000_000);

        answering.give().unwrap();
        assert!(done.wait(1, Duration::from_secs(10)).unwrap(), "{backend:?}: the call ends on its answer");
        completion.wait().unwrap();

        assert!(landed, "{backend:?}: the batch never landed");
        assert!(still_parked, "{backend:?}: the call ended before its answer, so the wait was not beside it");
        assert!(waited < TIMEOUT / 2, "{backend:?}: the wait took {waited:?} for a batch of milliseconds");
    }
}
