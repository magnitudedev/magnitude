use crate::CompletionWake;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, AtomicU64};
use crate::experts::watcher::Join;
use super::resize::{Cut, grow_by_slots};
use crate::storage_thread::{Done, StorageThread};

use super::*;

/// A resize step that panics is reported to its holding, its wake
/// completed, and the resize thread runs the next step: a panic never
/// leaves a resize running forever nor the bank's thread gone.
#[test]
fn a_resize_step_that_panics_is_reported_and_the_thread_runs_on() {
    let resize_thread = StorageThread::start("bank-test").unwrap();
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

/// A growth joins slot by slot, each only once every tensor commits it and
/// its memory is recorded in its ledger within the slot's change (a record
/// outside one panics in this scope), and a stop ends it at the next slot:
/// two record tensors of a slot a row, grown from one slot toward six on
/// the CPU backend, stopped as slot 3 joins, joined 2 and 3 and stopped
/// there.
#[test]
fn a_growth_joins_each_slot_as_every_tensor_backs_it() {
    let device = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
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
    let (posted, backed) = (Mutex::new(Vec::new()), Mutex::new(Vec::new()));
    let scope = crate::memory::ledger::Ledgers::strict();
    let _owner = scope.changes().own();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, std::sync::Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() }));
    ledger.read(ledger.sampled(Instant::now()), Ok(crate::memory::ledger::Available::new(1 << 40, 1 << 40, None).expect("a reading")));
    let changes = scope.changes().clone();
    let claim = Mutex::new(ledger.backing(6 * 2 * row).expect("the slots fit"));
    let record = |at: u64, changing: &crate::memory::ledger::Changing| {
        claim.lock().unwrap().backed(at * 2 * row, changing);
        backed.lock().unwrap().push(at);
    };
    let grown = grow_by_slots(
        resizers,
        (1, 6),
        |at| vec![at; 2],
        (&changes, record),
        |at| {
            assert_eq!(backed.lock().unwrap().last(), Some(&at), "slot {at} joined before its memory was recorded");
            Ok(Join::Bank { records: vec![at as usize] })
        },
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
    assert_eq!(ledger.snapshot().allocated(), 3 * 2 * row, "every slot joined is recorded");
    // Stopped before its first slot, it joins none: its end says so.
    let none = |_: u64, _: &crate::memory::ledger::Changing| {};
    let again = grow_by_slots(grown.resizers, (3, 6), |at| vec![at; 2], (&changes, none), |_| unreachable!("no slot joins"), |_| {}, (&slots, &rises), &stop, || Some(()));
    assert!(again.joined.is_none() && matches!(again.cut, Cut::None));
    // A slot the load's traffic does not let begin ends the growth there.
    let held = grow_by_slots(again.resizers, (3, 6), |at| vec![at; 2], (&changes, none), |_| unreachable!("no slot joins"), |_| {}, (&slots, &rises), &AtomicBool::new(false), || None::<()>);
    assert!(held.joined.is_none() && matches!(held.cut, Cut::None));
    assert!(tensors.iter().all(|tensor| tensor.committed_rows() == 3));
    changes.change(|_| drop(claim));
}
