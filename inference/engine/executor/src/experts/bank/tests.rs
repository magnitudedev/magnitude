use crate::domain::BankHolding;
use crate::CompletionWake;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, AtomicU64};
use crate::experts::watcher::Join;
use super::served::{Step, grown, shrunk};
use super::resize::{Cut, Done, grow_by_slots};

/// A landed shrink holds the thread at most a step at its latest
/// measured rate: 10 GB/s over a 35 ms step is 350 MB; unmeasured, the
/// least, which measures it; with no step to hold it to, the whole
/// deficit.
#[test]
fn a_landed_shrink_releases_what_one_step_may_hold() {
    assert_eq!(super::landed_bytes(Some(0.035), Some(1e10)), Some(350_000_000));
    assert_eq!(super::landed_bytes(Some(0.035), None), Some(0));
    assert_eq!(super::landed_bytes(None, Some(1e10)), None);
}

use super::*;

fn step(holding: BankHolding, bytes: u64, saves: Option<f64>) -> Step {
    Step { holding, kind: crate::planning::BankTier::Bank, from: 0, to: 1, bytes, saves }
}

/// A shrink takes the step that saves least per byte, an unpriced one
/// first and the experts on a tie; a growth, once every holding's
/// resizes are measured, the priced step that saves most per byte, none
/// when nothing is priced; before, the smallest step of a holding not
/// yet measured, priced or not; the host tier's step before any.
#[test]
fn the_owner_resizes_the_holding_by_what_a_byte_saves() {
    let taken = |steps| shrunk(steps, &|_| 0).map(|step: Step| step.holding);
    let bounded = |steps: Vec<Step>| steps.into_iter().map(|step| (step, 1)).collect::<Vec<_>>();
    let added = |steps| grown(bounded(steps), &|_| true).map(|(step, _)| step.holding);
    // A slot of 100 bytes saving 1 s; a table record of 10 bytes 0.5 s.
    let priced = || vec![step(BankHolding::Experts, 100, Some(1.)), step(BankHolding::Table { table: 0 }, 10, Some(0.5))];
    assert_eq!(taken(priced()), Some(BankHolding::Experts));
    assert_eq!(added(priced()), Some(BankHolding::Table { table: 0 }));
    let cheap = vec![step(BankHolding::Experts, 100, Some(1.)), step(BankHolding::Table { table: 0 }, 10, Some(0.05))];
    assert_eq!(taken(cheap), Some(BankHolding::Table { table: 0 }));
    let unmeasured = || vec![step(BankHolding::Experts, 100, None), step(BankHolding::Table { table: 0 }, 10, None)];
    assert_eq!(taken(unmeasured()), Some(BankHolding::Experts));
    assert_eq!(added(unmeasured()), None);
    let table_unmeasured = vec![step(BankHolding::Experts, 100, Some(1.)), step(BankHolding::Table { table: 0 }, 10, None)];
    assert_eq!(taken(table_unmeasured), Some(BankHolding::Table { table: 0 }));
    assert_eq!(taken(Vec::new()), None);
    // Holdings no resize has measured: the smallest of their steps, over
    // any priced one; with none left, the priced rule.
    let first = |steps, unmeasured: &[BankHolding]| {
        grown(bounded(steps), &|holding| !unmeasured.contains(&holding)).map(|(step, _)| step.holding)
    };
    let host = |saves| step(BankHolding::Host, 50, saves);
    let mixed = || vec![step(BankHolding::Experts, 100, Some(1.)), host(None), step(BankHolding::Table { table: 0 }, 10, Some(0.5))];
    assert_eq!(first(mixed(), &[BankHolding::Host]), Some(BankHolding::Host));
    assert_eq!(first(mixed(), &[BankHolding::Host, BankHolding::Experts]), Some(BankHolding::Host));
    assert_eq!(first(unmeasured(), &[BankHolding::Experts, BankHolding::Table { table: 0 }]), Some(BankHolding::Table { table: 0 }));
    assert_eq!(first(mixed(), &[]), Some(BankHolding::Host), "the tier's step, measured or not, priced by nothing");
    let device = || vec![step(BankHolding::Experts, 100, Some(1.)), step(BankHolding::Table { table: 0 }, 10, Some(0.5))];
    assert_eq!(first(device(), &[]), Some(BankHolding::Table { table: 0 }));
}

/// A shrink toward a store's deficit takes only that store's holdings:
/// host RAM short, the host tier's step over a device slot that saves
/// less, and once the tier is at its floor nothing, as no device slot
/// gives host RAM back; with no store short, any.
#[test]
fn a_shrink_takes_only_the_short_stores_holdings() {
    use crate::planning::BankTier::{Bank, Host};
    let host = || Step { holding: BankHolding::Host, kind: Host, from: 0, to: 1, bytes: 100, saves: Some(1.) };
    let slot = || step(BankHolding::Experts, 100, Some(0.1));
    let host_short = |kind| match kind {
        Host => 50,
        _ => 0,
    };
    let taken = |steps, deficit: &dyn Fn(crate::planning::BankTier) -> u64| shrunk(steps, deficit).map(|step: Step| step.holding);
    assert_eq!(taken(vec![slot(), host()], &host_short), Some(BankHolding::Host));
    assert_eq!(taken(vec![slot()], &host_short), None, "the host tier at its floor");
    assert_eq!(taken(vec![slot(), host()], &|_| 0), Some(BankHolding::Experts));
    let device_short = |kind| match kind {
        Bank => 50,
        _ => 0,
    };
    assert_eq!(taken(vec![slot(), host()], &device_short), Some(BankHolding::Experts));
}

/// A resize step that panics is reported to its holding, its wake
/// completed, and the resize thread runs the next step: a panic never
/// leaves a resize running forever nor the bank's thread gone.
#[test]
fn a_resize_step_that_panics_is_reported_and_the_thread_runs_on() {
    let resize_thread = ResizeThread::start().unwrap();
    let waited = |done: &Done<u32>| loop {
        if let Some((outcome, _)) = done.lock().unwrap().take() {
            return outcome;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    let woken = Arc::new(AtomicBool::new(false));
    let wake = { let woken = woken.clone(); CompletionWake::new(move || woken.store(true, Ordering::Release)) };
    let panicked = resize_thread.run(wake, || -> u32 { panic!("a step's fault") });
    assert!(waited(&panicked).is_err_and(|why| why.contains("a step's fault")));
    // The outcome lands before the wake completes, so the owner it wakes
    // finds it.
    let deadline = Instant::now() + Duration::from_secs(10);
    while !woken.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "the holding's owner is woken to learn it");
        std::thread::yield_now();
    }
    let next = resize_thread.run(CompletionWake::new(|| {}), || 7);
    assert_eq!(waited(&next), Ok(7));
}

/// A growth joins slot by slot, each only once every tensor commits it,
/// and a stop ends it at the next slot: two record tensors of a slot a
/// row, grown from one slot toward six on the CPU backend, stopped as
/// slot 3 joins, joined 2 and 3 and stopped there.
#[test]
fn a_growth_joins_each_slot_as_every_tensor_backs_it() {
    let device = seismic::Devices::discover(&[]).unwrap().open_backend(seismic::BackendName::Cpu).unwrap();
    let row = device
        .info()
        .reserving(seismic::ReservedContract::DeviceOnly)
        .expect("the CPU backend says how it reserves")
        .expect("the CPU backend reserves device storage")
        .granule;
    let (tensors, resizers): (Vec<_>, Vec<_>) = (0..2)
        .map(|_| Tensor::reserved(&device, seismic::Element::bool(), &[8, row], 1).unwrap())
        .unzip();
    let (slots, stop, rises) = (AtomicU64::new(1), AtomicBool::new(false), Mutex::new(()));
    let posted = Mutex::new(Vec::new());
    let grown = grow_by_slots(
        resizers,
        (1, 6),
        |at| vec![at; 2],
        |at| Ok(Join::Bank { records: vec![at as usize] }),
        |join| {
            let Join::Bank { records } = join else { panic!("a bank's join") };
            let at = records[0];
            assert!(tensors.iter().all(|tensor| tensor.committed_rows() >= at as u64), "slot {at} joined before it is backed");
            posted.lock().unwrap().push(at);
            if at == 3 {
                stop.store(true, Ordering::Release);
            }
        },
        (&slots, &rises),
        &stop,
        || Some(()),
    );
    assert_eq!(*posted.lock().unwrap(), [2, 3]);
    assert_eq!(slots.load(Ordering::Acquire), 3);
    assert!(matches!(grown.cut, Cut::None) && grown.joined.is_some() && grown.resizers.len() == 2);
    assert!(tensors.iter().all(|tensor| tensor.committed_rows() == 3));
    // Stopped before its first slot, it joins none: its end says so.
    let again = grow_by_slots(grown.resizers, (3, 6), |at| vec![at; 2], |_| unreachable!("no slot joins"), |_| {}, (&slots, &rises), &stop, || Some(()));
    assert!(again.joined.is_none() && matches!(again.cut, Cut::None));
    // A slot the load's traffic does not let begin ends the growth there.
    let held = grow_by_slots(again.resizers, (3, 6), |at| vec![at; 2], |_| unreachable!("no slot joins"), |_| {}, (&slots, &rises), &AtomicBool::new(false), || None::<()>);
    assert!(held.joined.is_none() && matches!(held.cut, Cut::None));
    assert!(tensors.iter().all(|tensor| tensor.committed_rows() == 3));
}
