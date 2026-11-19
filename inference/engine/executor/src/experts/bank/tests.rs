use crate::CompletionWake;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, AtomicU64};
use crate::experts::watcher::Join;
use super::resize::{Cut, Joining, RecordsAt, Stores, grow_by_slots};
use crate::storage_thread::StorageThread;

use super::*;

/// A growth joins slot by slot, each only once every group holds its
/// records, its pool's blocks charged to the store's pool entry within the
/// slot's change (a charge outside one panics in this scope), and a stop
/// ends it at the next slot: two groups of a record a slot in one pool of
/// two-granule blocks on the CPU backend, grown from one slot toward six,
/// stopped as slot 3 joins, joined 2 and 3 and stopped there; a record
/// freed is claimed again in its hole. A block is charged what the backend
/// allocates for it (whole host pages at least), as the first slot's block
/// was, beside what the pool and its stores charged as they opened.
#[test]
fn a_growth_joins_each_slot_as_every_group_holds_it() {
    let device = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
    let regions = [seismic::SlabRegion { element: seismic::Element::bool(), row_shape: vec![64] }];
    let layout = seismic::SlabLayout::for_regions(1, 8, &regions).unwrap();
    let granule = layout.slab_bytes.next_multiple_of(layout.slab_alignment);
    let scope = crate::memory::ledger::Ledgers::strict();
    let _owner = scope.changes().own();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, std::sync::Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() }));
    ledger.read(ledger.sampled(Instant::now()), crate::memory::supply::Reading::device(1 << 40, 1 << 40, crate::domain::fixtures::NO_LAG));
    let changes = scope.changes().clone();
    // A slot is a block: a granule of each group.
    let entries = ledger.pool_block(crate::memory::ledger::PoolClass::Records, std::num::NonZeroU64::new(2 * granule).expect("a slot's block"));
    let charges = seismic::Charges::tally(&device);
    let pool = seismic::Pool::open(&entries.charges(&device, seismic::Contract::Shared), seismic::Contract::Shared, granule, 2 * granule).unwrap();
    let groups = (0..2).map(|_| (seismic::SlabTensor::in_pool(&charges, &pool, 1, 8, &regions).unwrap(), 0, vec![64])).collect();
    let stores = Stores::new(pool, groups);
    let opened = ledger.estimate().allocated();
    let records: RecordsAt = Arc::new(|_, at| at);
    // The first slot's records, claimed as the bank opened; the growth's
    // claim holds the granules its slots take.
    changes.change(|_| (0..2).for_each(|group| assert_eq!(stores.claim_to(group, 1).unwrap(), [0])));
    let block = ledger.estimate().allocated() - opened;
    assert!(block >= 2 * granule, "a block is charged its granules at least: {block} bytes");
    let blocks = |slots: u64| opened + slots * block;
    let claim = ledger.change(|keeping| ledger.backing(&keeping.claims(), crate::memory::ledger::PoolClass::Records, 6 * 2 * granule)).expect("the slots fit");
    let (slots, stop, rises) = (AtomicU64::new(1), AtomicBool::new(false), Mutex::new(()));
    let posted = Mutex::new(Vec::new());
    let grown = grow_by_slots(
        &stores,
        (1, 6),
        &records,
        Joining::Bank,
        &changes,
        |join| {
            let Join::Bank { records, views } = join else { panic!("a bank's join") };
            assert_eq!(views.len(), records.len(), "views delivered for every group");
            let at = records[0][0] as u64 + 1;
            assert_eq!(records, vec![vec![at as usize - 1]; 2], "each group claims the slot's record in its lowest hole");
            assert!((0..2).all(|group| stores.present(group) >= at), "slot {at} joined before every group holds it");
            assert_eq!(ledger.estimate().allocated(), blocks(at), "slot {at} joined before its block was charged");
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
    assert!(matches!(grown.cut, Cut::None) && grown.joined.is_some());
    assert_eq!(stores.charged(), 3 * 2 * granule, "a granule a record held");
    // Stopped before its first slot, it joins none: its end says so.
    let again = grow_by_slots(&stores, (3, 6), &records, Joining::Bank, &changes, |_| unreachable!("no slot joins"), (&slots, &rises), &stop, || Some(()));
    assert!(again.joined.is_none() && matches!(again.cut, Cut::None));
    // A slot the load's traffic does not let begin ends the growth there.
    let held = grow_by_slots(&stores, (3, 6), &records, Joining::Bank, &changes, |_| unreachable!("no slot joins"), (&slots, &rises), &AtomicBool::new(false), || None::<()>);
    assert!(held.joined.is_none() && matches!(held.cut, Cut::None));
    assert!((0..2).all(|group| stores.present(group) == 3));
    // A record freed leaves a hole, which the next claim fills from the
    // granule free in its block.
    stores.free(0, &[1]).unwrap();
    assert_eq!(stores.present(0), 2);
    changes.change(|_| assert_eq!(stores.claim_to(0, 3).unwrap(), [1]));
    assert_eq!(ledger.estimate().allocated(), blocks(3), "the hole refilled in place");
    changes.change(|_| drop((claim, stores)));
}

/// A records pool's absent granule, which every free slot names and no
/// slab holds, is committed memory of its own entry, never the pool's
/// spare: opened on the ledger, the pool allocates it and holds no spare,
/// and the spare a claim then leaves is what a growth claims without
/// allocating. Its spare counting as the bank's room
/// (`Estimate::import_room`) sized a Metal bank one granule past its budget
/// when the absent granule was charged to the blocks' entry.
#[test]
fn a_pools_absent_granule_is_committed_never_spare() {
    use crate::memory::ledger::CacheKind;
    let device = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
    let regions = [seismic::SlabRegion { element: seismic::Element::bool(), row_shape: vec![64] }];
    let layout = seismic::SlabLayout::for_regions(1, 2, &regions).unwrap();
    let granule = layout.slab_bytes.next_multiple_of(layout.slab_alignment);
    let scope = crate::memory::ledger::Ledgers::strict();
    let _owner = scope.changes().own();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, std::sync::Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() }));
    ledger.read(ledger.sampled(Instant::now()), crate::memory::supply::Reading::device(1 << 40, 1 << 40, crate::domain::fixtures::NO_LAG));
    let changes = scope.changes().clone();
    let entries = ledger.pool_block(crate::memory::ledger::PoolClass::Records, std::num::NonZeroU64::new(2 * granule).expect("a block"));
    let charges = seismic::Charges::tally(&device);
    let pool = seismic::Pool::open(&entries.charges(&device, seismic::Contract::Shared), seismic::Contract::Shared, granule, 2 * granule).unwrap();
    let groups = vec![(seismic::SlabTensor::in_pool(&charges, &pool, 1, 2, &regions).unwrap(), 0, vec![64])];
    let stores = Stores::new(pool, groups);
    let opened = ledger.estimate();
    assert!(opened.allocated() >= granule, "the pool allocated its absent granule as it opened");
    assert_eq!((opened.cache(), opened.committed()), (0, opened.allocated()), "committed, none of it spare");
    // A growth's first record takes a block: its other granule and the
    // block's rounding are spare.
    let claim = ledger.change(|keeping| ledger.backing(&keeping.claims(), crate::memory::ledger::PoolClass::Records, granule)).expect("room");
    changes.change(|_| assert_eq!(stores.claim_to(0, 1).unwrap(), [0]));
    let one = ledger.estimate();
    let block = one.allocated() - opened.allocated();
    assert!(block >= 2 * granule, "a block is charged its granules at least: {block} bytes");
    assert_eq!(one.cache_of(CacheKind::Spare), block - granule, "spare is the block's free granule and its rounding");
    assert_eq!(one.committed(), opened.committed() + granule, "the absent granule beside the claim's record");
    // The next record is the spare granule, claimed in place.
    changes.change(|_| assert_eq!(stores.claim_to(0, 2).unwrap(), [1]));
    assert_eq!(ledger.estimate().allocated(), one.allocated(), "the spare granule is one a growth claims");
    changes.change(|_| drop((claim, stores)));
}

/// A long growth never starves the load's settled samples, nor they it,
/// nor does either panic: the bank grows from one slot to ninety on its
/// resize thread, each slot's claims one change begun only while no settled
/// sample waits ([`Changes::back`]), while the sampling thread, the scope's
/// owner with its own change open as a load's is (`OpenChange`), takes
/// settled samples one after another until the growth ends. A settled
/// sample waits for the steps under way when it began (and the entry changes
/// beside it, which this growth makes none of), because the owner's open
/// change does not hold steps off while its thread waits in the sample
/// (`Sampled::settled`).
///
/// What it guards: that no sample waits on the growth's length, in the
/// product's configuration. The 60-second guard is a bound, not a timing:
/// ninety slots finish far inside it only if samples yield to steps; a
/// regression shows as the guard's panic, not as a measured delay. Every sample returns
/// within the test's time guard, some while slots were joining, and the
/// growth ends whole: all ninety slots joined, their granules charged.
///
/// It guards the failure a real GLM run met as its expert bank grew from 88
/// to 89 slots: the sampler, which then waited with a deadline, found no
/// sample standing within a reading interval though no change was under
/// way, and panicked ("no sample stood within a reading interval with no
/// change under way"), its slot-by-slot growth having left it no quiet
/// moment.
///
/// The samples the owner takes also move as a streamed load's page cache
/// does: between samples clean file cache rises and free falls by the same
/// bytes, as expert reads fill the cache. What makes a sample stand is only
/// our own changes (`Sampled::attempt` compares the scope's change counter
/// before and after the read, never an OS figure), so OS page-cache churn
/// can neither starve nor stall one. And the cache is supply, never
/// others': each reading, folded into the ledger as the load folds it, leaves
/// the supply it derives (free above target plus clean cache) unchanged
/// while their sum is, and others' bytes never rise, the bank's own charges
/// only ever lowering what the store uses beyond its supply and the
/// process's.
#[test]
fn a_long_growth_beside_settled_samples_never_starves_or_panics_them() {
    const SLOTS: u64 = 90;
    let device = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
    let regions = [seismic::SlabRegion { element: seismic::Element::bool(), row_shape: vec![64] }];
    let layout = seismic::SlabLayout::for_regions(1, SLOTS, &regions).unwrap();
    let granule = layout.slab_bytes.next_multiple_of(layout.slab_alignment);
    let scope = crate::memory::ledger::Ledgers::strict();
    // The sampling thread owns the scope with a change open, as a load does
    // (`OpenChange::begin` in the load path), and takes its settled samples
    // inside it: the one configuration the product reaches.
    let _load = crate::memory::ledger::OpenChange::begin(scope.changes());
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, std::sync::Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() }));
    // Host RAM of 1 TiB whose free and clean cache always sum to 1 GiB.
    const TOTAL: u64 = 1 << 40;
    const SUPPLY: u64 = 1 << 30;
    let host = |clean: u64| seismic::HostSample {
        supply: seismic::Supply { free_beyond_target: i64::try_from(SUPPLY - clean).expect("bytes fit i64"), clean_cache: clean, cold_reclaimable: 0 },
        ..crate::memory::ledger::tests::host_sample(TOTAL, SUPPLY)
    };
    let first = host(0);
    let start = ledger.read(ledger.sampled(Instant::now()), crate::memory::supply::Reading::host(&first));
    let (others, supply) = (start.others(), start.supply());
    assert_eq!(supply, SUPPLY as i64);
    let changes = scope.changes().clone();
    let entries = ledger.pool_block(crate::memory::ledger::PoolClass::Records, std::num::NonZeroU64::new(2 * granule).expect("a slot's block"));
    let charges = seismic::Charges::tally(&device);
    let pool = seismic::Pool::open(&entries.charges(&device, seismic::Contract::Shared), seismic::Contract::Shared, granule, 2 * granule).unwrap();
    // A store of a slot for each of the ninety.
    let groups = (0..2).map(|_| (seismic::SlabTensor::in_pool(&charges, &pool, 1, SLOTS, &regions).unwrap(), 0, vec![64])).collect();
    let stores = Arc::new(Stores::new(pool, groups));
    let opened = ledger.estimate().allocated();
    let records: RecordsAt = Arc::new(|_, at| at);
    changes.change(|_| (0..2).for_each(|group| assert_eq!(stores.claim_to(group, 1).unwrap(), [0])));
    // What the backend charges a block (whole host pages at least).
    let block = ledger.estimate().allocated() - opened;
    let claim = ledger.change(|keeping| ledger.backing(&keeping.claims(), crate::memory::ledger::PoolClass::Records, SLOTS * 2 * granule)).expect("the slots fit");
    let (slots, stop, rises, joined) = (Arc::new(AtomicU64::new(1)), Arc::new(AtomicBool::new(false)), Arc::new(Mutex::new(())), Arc::new(AtomicU64::new(0)));
    // The growth runs on the resize thread as the bank's does.
    let resize_thread = StorageThread::start("bank-test-growth").unwrap();
    let done = {
        let (stores, changes, records, slots, stop, rises, joined) =
            (stores.clone(), changes.clone(), records.clone(), slots.clone(), stop.clone(), rises.clone(), joined.clone());
        resize_thread.run(CompletionWake::new(|| {}), move || {
            grow_by_slots(
                &stores,
                (1, SLOTS),
                &records,
                Joining::Bank,
                &changes,
                |join| {
                    let Join::Bank { .. } = join else { panic!("a bank's join") };
                    joined.fetch_add(1, Ordering::AcqRel);
                },
                (&slots, &rises),
                &stop,
                || Some(()),
            )
        })
    };
    // The owner samples, settled, until the growth ends: each sample is a
    // load's (`Sample::settled` beside the scope's changes).
    let deadline = Instant::now() + Duration::from_secs(60);
    let (mut samples, mut seen) = (0u64, std::collections::BTreeSet::new());
    while done.lock().unwrap().is_none() {
        assert!(Instant::now() < deadline, "a settled sample or the growth starved: {samples} samples, {} slots joined", joined.load(Ordering::Acquire));
        // The cache fills a megabyte a sample and wraps at half the supply.
        let sample = crate::memory::ledger::Sample::scripted(&changes, host((samples * (1 << 20)) % (SUPPLY / 2)));
        let read = ledger.read(sample.at(), crate::memory::supply::Reading::host(sample.host()));
        assert_eq!(read.supply(), supply, "cache movement is supply: the reading derives the same supply");
        assert!(read.others() <= others, "cache movement is never others': {} beside {others}", read.others());
        samples += 1;
        seen.insert(joined.load(Ordering::Acquire));
        if samples >= 1 {
            std::thread::sleep(Duration::from_micros(50));
        }
    }
    let (grown, _) = done.lock().unwrap().take().expect("the growth ended");
    assert!(samples >= 3 && seen.len() >= 3, "samples were taken while slots were joining: {samples} samples saw {seen:?}");
    assert!(matches!(grown.cut, Cut::None) && grown.joined.is_some());
    assert_eq!(joined.load(Ordering::Acquire), SLOTS - 1, "every slot after the first joined");
    assert_eq!(slots.load(Ordering::Acquire), SLOTS);
    assert!((0..2).all(|group| stores.present(group) == SLOTS));
    assert_eq!(stores.charged(), SLOTS * 2 * granule);
    assert_eq!(ledger.estimate().allocated(), opened + SLOTS * block, "a block a slot");
    changes.change(|_| drop((claim, stores)));
}

/// `Growing::ended` acquires `done` before reading the reached slot count or
/// evaluating the store's post-growth charge, so an advance racing the resize
/// thread's completion can never pair a pre-completion slot count (`from == to`)
/// with a completed `Grown` outcome or read `charged` while still `None`.
#[test]
fn growing_ended_reads_reached_slots_and_charge_after_done_is_taken() {
    use super::resize::{Cut, Growing, Grown, Spent};
    use crate::{BankHolding, BankResized};
    let done: crate::storage_thread::Done<Grown> = Arc::new(Mutex::new(None));
    let growing = Growing {
        from: 39,
        charged: 39 * 1024,
        decided: Instant::now(),
        stop: Arc::new(AtomicBool::new(false)),
        done: done.clone(),
    };
    assert!(
        growing
            .ended(BankHolding::Experts, || panic!("charged must not run before done is ready"))
            .is_none()
    );
    let now = Instant::now();
    *done.lock().unwrap() = Some((
        Grown {
            at: 40,
            joined: Some((now, now)),
            cut: Cut::None,
            spent: Spent::default(),
        },
        now,
    ));
    let (resized, _) = growing
        .ended(BankHolding::Experts, || 40 * 1024)
        .expect("done is ready")
        .expect("growth succeeded");
    assert!(
        matches!(resized, BankResized::Grown { holding: BankHolding::Experts, from: 39, to: 40, grown: 1024, refused: None, .. }),
        "{resized:?}"
    );
}
