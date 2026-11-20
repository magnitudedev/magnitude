use super::*;
use crate::entry::{Charge, Refused, Tally, Wiring};
use seismic_topology::StoreIdentity;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

/// Blocks at made-up addresses, a block apart, counting allocations, each
/// holding `slack` bytes beyond those asked (rounding, driver objects) and
/// charging its entry for all of them, as a backend does.
struct Counted {
    allocations: Arc<AtomicU64>,
    slack: u64,
}

struct Plain {
    address: u64,
    charge: Charge,
}

/// A flight that completes when the test says ([`Flight::complete`]),
/// waking whoever waits on it.
struct Flight {
    done: Mutex<bool>,
    ended: Condvar,
}

impl Flight {
    fn complete(&self) {
        *self.done.lock().unwrap() = true;
        self.ended.notify_all();
    }
}

impl Fence for Flight {
    fn completed(&self) -> bool {
        *self.done.lock().unwrap()
    }

    fn wait(&self) {
        let mut done = self.done.lock().unwrap();
        while !*done {
            done = self.ended.wait(done).unwrap();
        }
    }
}

impl Blocks for Counted {
    type Block = Plain;
    type Fence = Flight;
    type Error = Refused;

    fn allocate(&self, bytes: u64, entry: &Arc<dyn Entry>, _fill: SlabFill) -> Result<Plain, Refused> {
        let charge = Charge::take(&entry.change(), entry, bytes + self.slack, Wiring::Pageable)?;
        let made = self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(Plain { address: (made + 1) * bytes * 2, charge })
    }

    fn held(&self, block: &Plain) -> u64 {
        block.charge.bytes()
    }

    fn address(&self, block: &Plain) -> u64 {
        block.address
    }
}

/// Who holds a granule in these tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Holder {
    Bank(u32),
    Kv(u32),
}

const GRANULE: u64 = 1 << 20;

fn pool_of(granules_a_block: u64, slack: u64) -> (GranulePool<Counted, Holder>, Arc<AtomicU64>) {
    let allocations = Arc::new(AtomicU64::new(0));
    let blocks = Counted { allocations: allocations.clone(), slack };
    (GranulePool::new(blocks, GRANULE, granules_a_block * GRANULE), allocations)
}

fn pool(granules_a_block: u64) -> (GranulePool<Counted, Holder>, Arc<AtomicU64>, Tally) {
    let tally = Tally::unbounded(StoreIdentity::HostRam);
    let (pool, allocations) = pool_of(granules_a_block, 0);
    (pool, allocations, tally)
}

/// Allocate one zeroed block outside the pool and install it (mirrors
/// `PoolShared::grow_once`).
fn grow(pool: &mut GranulePool<Counted, Holder>, tally: &Tally) -> Result<BlockId, Refused> {
    let entry: Arc<dyn Entry> = Arc::new(tally.clone());
    let block = pool.backend().allocate(pool.block_bytes(), &entry, SlabFill::Zeroed)?;
    Ok(pool.install(block, SlabFill::Zeroed))
}

/// The bytes `pool` returns now, its returned blocks dropped.
fn returned(pool: &mut GranulePool<Counted, Holder>) -> u64 {
    pool.return_empty().into_iter().map(|(_, held)| held).sum()
}

fn flight() -> Arc<Flight> {
    Arc::new(Flight { done: Mutex::new(false), ended: Condvar::new() })
}

/// A KV claim takes a granule the bank released, in process: no block is
/// allocated or returned, and the store's entry holds the same bytes.
#[test]
fn a_kv_claim_takes_a_released_bank_granule_in_process() {
    let (mut pool, allocations, tally) = pool(4);
    grow(&mut pool, &tally).unwrap();
    let bank = (0..4).map(|slot| pool.take(Holder::Bank(slot)).unwrap()).collect::<Vec<_>>();
    assert!(pool.take(Holder::Kv(0)).is_none(), "every granule is the bank's");
    let read = flight();
    pool.release(bank[2].granule, Some(read.clone()));
    read.complete();
    let kv = pool.take(Holder::Kv(0)).expect("the released granule is claimable");
    assert_eq!((kv.granule, kv.address), (bank[2].granule, bank[2].address));
    assert!(!kv.zero, "a reused granule holds the bank's bytes");
    assert_eq!(pool.holder(kv.granule), Some(Holder::Kv(0)));
    assert_eq!(allocations.load(Ordering::Relaxed), 1, "no allocation");
    assert_eq!(tally.held(), 4 * GRANULE, "no OS return");
}

/// Releasing granules between kept ones moves nothing: the kept keep their
/// addresses, and the holes take the next claims.
#[test]
fn release_with_holes_needs_no_moves() {
    let (mut pool, allocations, tally) = pool(8);
    grow(&mut pool, &tally).unwrap();
    let bank = (0..8).map(|slot| pool.take(Holder::Bank(slot)).unwrap()).collect::<Vec<_>>();
    for hole in [1, 4, 6] {
        pool.release(bank[hole].granule, None);
    }
    for kept in [0, 2, 3, 5, 7] {
        assert_eq!(pool.holder(bank[kept].granule), Some(Holder::Bank(kept as u32)));
        assert_eq!(pool.address(bank[kept].granule), bank[kept].address, "a kept granule never moves");
    }
    assert_eq!(pool.free_granules(), 3);
    let refilled = (0..3).map(|slot| pool.take(Holder::Kv(slot)).unwrap().granule).collect::<BTreeSet<_>>();
    assert_eq!(refilled, [1, 4, 6].map(|hole| bank[hole].granule).into());
    assert_eq!(allocations.load(Ordering::Relaxed), 1);
}

/// A released granule is claimable only once the last flight that read it
/// completes, and its block is not empty until then.
#[test]
fn a_granule_frees_only_after_its_flight() {
    let (mut pool, _, tally) = pool(2);
    grow(&mut pool, &tally).unwrap();
    let first = pool.take(Holder::Bank(0)).unwrap();
    let second = pool.take(Holder::Bank(1)).unwrap();
    let read = flight();
    pool.release(first.granule, Some(read.clone()));
    pool.release(second.granule, Some(read.clone()));
    assert_eq!(pool.holder(first.granule), None);
    assert!(pool.take(Holder::Kv(0)).is_none(), "a fenced granule takes no claim");
    assert_eq!(returned(&mut pool), 0, "a fenced granule keeps its block");
    assert_eq!(pool.pending().count(), 2);
    read.complete();
    assert_eq!(returned(&mut pool), 2 * GRANULE);
    assert_eq!((pool.held_bytes(), tally.held()), (0, 0));
}

/// A pool's blocks are the least allocation the backend makes, in whole
/// granules, a granule where it bounds no count and rounds nothing, and the
/// least common multiple of a granule and its granularity where it rounds.
#[test]
fn block_sizing_rounds_the_least_allocation_to_granules() {
    let granule = 2u64 << 20;
    let at = |least, granularity| block_bytes(BlockSizing { granule, allocations: NativeAllocations { least, granularity } });
    let exact = |granules: u64| BlockSize { bytes: granules * granule, held: granules * granule, absent: granule };
    assert_eq!(at(21_474_837, 1).unwrap(), exact(11), "the fewest granules holding the least allocation");
    assert_eq!(at(1, 1).unwrap(), exact(1));
    assert_eq!(at(0, 1).unwrap(), exact(1));
    assert_eq!(block_bytes(BlockSizing { granule: 0, allocations: NativeAllocations::EXACT }), Err(SizingError::NoGranule));
    // A Qwen expert record under CUDA's 2 MiB granularity: 32 records fill
    // 27 granularity units exactly.
    let record = 1_769_472u64;
    let cuda = NativeAllocations { least: 0, granularity: 2 << 20 };
    let block = block_bytes(BlockSizing { granule: record, allocations: cuda }).unwrap();
    assert_eq!(block, BlockSize { bytes: 32 * record, held: 27 * (2 << 20), absent: 2 << 20 });
    assert_eq!(block.charge(record, 33), Some(2 * block.held), "granules cost whole blocks");
    // Whole granules fill whole units only far past the span: the block
    // wasting least per granule within it, its rounding held.
    let odd = (2 << 20) + 256;
    let block = block_bytes(BlockSizing { granule: odd, allocations: cuda }).unwrap();
    assert!(block.bytes <= MAX_BLOCK_UNITS * (2 << 20) && block.held % (2 << 20) == 0 && block.held >= block.bytes);
    assert_eq!(block.absent, 4 << 20);
    // A multi-tens-of-MiB state granule under CUDA's 2 MiB granularity stays
    // bounded by the 64-unit (128 MiB) span rather than scaling the span up
    // by the granule itself.
    let qwen_state = 65_625_472u64;
    let block = block_bytes(BlockSizing { granule: qwen_state, allocations: cuda }).unwrap();
    assert_eq!(block, BlockSize { bytes: 2 * qwen_state, held: 63 * (2 << 20), absent: 32 * (2 << 20) });
}

/// Emptying a block for outside demand copies its live granules into free
/// ones elsewhere; it returns once their sources' last reads complete.
#[test]
fn a_block_returns_only_once_its_live_granules_are_copied_out() {
    let (mut pool, _, tally) = pool(2);
    let older = grow(&mut pool, &tally).unwrap();
    let newer = grow(&mut pool, &tally).unwrap();
    let kept = pool.take(Holder::Kv(0)).unwrap();
    let moved = (0..2).map(|slot| pool.take(Holder::Bank(slot)).unwrap()).last().unwrap();
    assert_eq!(moved.granule.block, newer);
    pool.release(kept.granule, None);
    assert_eq!(pool.emptiest(), Some(newer), "one live granule, and one free elsewhere");
    let vacating = pool.vacate(newer).expect("a granule is free elsewhere");
    assert_eq!(vacating.moves().len(), 1, "only live granules are copied");
    let copy = vacating.moves()[0];
    assert_eq!((copy.holder, copy.from, copy.to), (Holder::Bank(1), moved.granule, kept.granule));
    assert_eq!(copy.to.block, older);
    // The holder copied and repointed: it releases the source behind the
    // flight that read it.
    let read = flight();
    pool.release(copy.from, Some(read.clone()));
    assert_eq!(pool.vacated(vacating), 2 * GRANULE, "every live granule moved out: the block is returning");
    assert_eq!(returned(&mut pool), 0, "the copy's source is read until its flight completes");
    read.complete();
    assert_eq!(returned(&mut pool), 0, "the block returns, its bytes reported by its vacate");
    assert_eq!((pool.held_bytes(), tally.held()), (2 * GRANULE, 2 * GRANULE));
    assert!(pool.take(Holder::Kv(1)).is_none(), "a returned block's granules leave with it");
}

/// A block vacated whole is returning: it pays what is owed at once, takes
/// no claim and is never vacated again, and returns as the flight that
/// copies out of it completes, that flight awaited though nothing more is
/// owed. It pays once.
#[test]
fn a_vacated_block_pays_what_is_owed_and_returns_behind_its_copy() {
    let (mut pool, _, tally) = pool(2);
    grow(&mut pool, &tally).unwrap();
    let newer = grow(&mut pool, &tally).unwrap();
    let freed_older = pool.take(Holder::Kv(0)).unwrap();
    pool.take(Holder::Kv(1)).unwrap();
    let freed_newer = pool.take(Holder::Bank(0)).unwrap();
    pool.take(Holder::Bank(1)).unwrap();
    // A shrink frees one granule of each block: neither empties.
    pool.release(freed_older.granule, None);
    pool.release(freed_newer.granule, None);
    pool.owe(2 * GRANULE);
    assert_eq!(returned(&mut pool), 0);
    assert_eq!(pool.emptiest(), Some(newer));
    let vacating = pool.vacate(newer).expect("a granule is free elsewhere");
    assert_eq!(vacating.moves().len(), 1);
    let copy = flight();
    pool.release(vacating.moves()[0].from, Some(copy.clone()));
    assert_eq!(pool.vacated(vacating), 2 * GRANULE);
    assert_eq!(pool.owed(), 0, "the returning block paid the debt");
    assert_eq!(pool.awaiting(), 0, "its fenced source is the returning block's, which paid");
    assert_eq!(pool.awaited().map(|fences| fences.len()), Some(1), "its return awaits the copy though nothing is owed");
    assert_eq!(returned(&mut pool), 0, "the copy reads its source until it completes");
    assert!(pool.emptiest().is_none(), "a returning block is never vacated again");
    assert!(pool.take(Holder::Kv(2)).is_none(), "nor takes a claim");
    copy.complete();
    assert_eq!(returned(&mut pool), 0, "it returns, its bytes reported once, by its vacate");
    assert_eq!((pool.owed(), pool.held_bytes(), tally.held()), (0, 2 * GRANULE, 2 * GRANULE));
}

/// A block whose copy out failed is taken back into service: its free
/// granules take claims again, and what its holders still hold stays.
#[test]
fn an_unvacated_block_takes_claims_again() {
    let (mut pool, _, tally) = pool(2);
    let older = grow(&mut pool, &tally).unwrap();
    let newer = grow(&mut pool, &tally).unwrap();
    let kept = pool.take(Holder::Kv(0)).unwrap();
    let stays = pool.take(Holder::Kv(1)).unwrap();
    let live = pool.take(Holder::Bank(0)).unwrap();
    assert_eq!((stays.granule.block, live.granule.block), (older, newer));
    pool.release(kept.granule, None);
    let vacating = pool.vacate(newer).expect("a granule is free elsewhere");
    assert!(pool.take(Holder::Kv(2)).is_none(), "a vacating block takes no claims");
    // The copy failed: its destination goes back free, the block back to
    // service, its live granule where it was.
    pool.release(vacating.moves()[0].to, None);
    pool.unvacate(vacating);
    assert_eq!(pool.holder(live.granule), Some(Holder::Bank(0)));
    assert_eq!(pool.free_granules(), 2);
    let claimed = (0..2).map(|slot| pool.take(Holder::Kv(slot + 3)).unwrap().granule.block).collect::<BTreeSet<_>>();
    assert_eq!(claimed, [older, newer].into(), "both blocks take claims again");
    assert_eq!(returned(&mut pool), 0);
}

/// The entry is charged the blocks' exact bytes, holes and driver slack
/// included, which is what the OS sees: never the granules held.
#[test]
fn the_entry_is_charged_exact_block_bytes() {
    let tally = Tally::unbounded(StoreIdentity::HostRam);
    let (mut pool, _) = pool_of(4, 4096);
    grow(&mut pool, &tally).unwrap();
    grow(&mut pool, &tally).unwrap();
    let held = (0..5).map(|slot| pool.take(Holder::Bank(slot)).unwrap()).collect::<Vec<_>>();
    pool.release(held[1].granule, None);
    assert_eq!(tally.held(), 2 * (4 * GRANULE + 4096));
    assert_eq!(tally.held(), pool.held_bytes());
    drop(pool);
    assert_eq!(tally.held(), 0, "a dropped pool returns every block");
}

/// A block its entry refuses is never allocated, and nothing changes.
#[test]
fn a_refused_block_is_never_allocated() {
    let tally = Tally::limited(StoreIdentity::HostRam, 6 * GRANULE);
    let (mut pool, allocations) = pool_of(4, 0);
    grow(&mut pool, &tally).unwrap();
    match grow(&mut pool, &tally) {
        Err(refused) => assert_eq!(refused, Refused { required: 4 * GRANULE, available: 2 * GRANULE }),
        other => panic!("expected a refusal, got {:?}", other.map(drop)),
    }
    assert_eq!(allocations.load(Ordering::Relaxed), 1);
    assert_eq!(tally.held(), 4 * GRANULE);
}

/// One block counts once in its entry: the backend charges it as it
/// allocates, the pool charges nothing more, and the charge leaves with the
/// block.
#[test]
fn one_block_counts_once_in_the_entry() {
    let tally = Tally::unbounded(StoreIdentity::HostRam);
    let (mut pool, _) = pool_of(4, 512);
    grow(&mut pool, &tally).unwrap();
    assert_eq!(tally.held(), 4 * GRANULE + 512);
    assert_eq!(pool.held_bytes(), tally.held());
    assert_eq!(returned(&mut pool), 4 * GRANULE + 512);
    assert_eq!(tally.held(), 0);
}

/// Granules freed for outside demand while their flight is in flight go
/// back to the OS as that flight ends, from its completion: the pool owes
/// their bytes, waits on the flight on a thread of its own, and returns
/// them then; with nothing owed, nothing waits.
#[test]
fn owed_granules_return_as_their_flight_ends() {
    let (pool, _, tally) = pool(2);
    let pool = Arc::new(Mutex::new(pool));
    let granules = {
        let mut pool = pool.lock().unwrap();
        grow(&mut pool, &tally).unwrap();
        (0..2).map(|slot| pool.take(Holder::Bank(slot)).unwrap().granule).collect::<Vec<_>>()
    };
    let read = flight();
    let awaited = {
        let mut pool = pool.lock().unwrap();
        for &granule in &granules {
            pool.release(granule, Some(read.clone()));
        }
        assert!(pool.awaited().is_none(), "nothing owed, nothing waits");
        pool.owe(2 * GRANULE);
        assert_eq!(returned(&mut pool), 0, "in flight, the block stays");
        assert_eq!(pool.awaiting(), 2 * GRANULE);
        pool.awaited().expect("owed bytes await the flight")
    };
    let returning = pool.clone();
    let settled = after(awaited, move || drop(returning.lock().unwrap().return_empty()));
    assert_eq!(tally.held(), 2 * GRANULE, "nothing returns before the flight ends");
    read.complete();
    settled.join().unwrap();
    let pool = pool.lock().unwrap();
    assert_eq!((pool.owed(), pool.held_bytes(), tally.held()), (0, 0, 0), "the owed bytes are back");
}

/// A claim that takes a granule a shrink freed back into use pays the debt
/// by its bytes: with blocks of many granules, the bytes owed are never
/// more than those still free or fenced, so no later return vacates a live
/// block for a debt the reuse already paid.
#[test]
fn a_claim_reusing_freed_granules_pays_what_is_owed() {
    let (mut pool, _, tally) = pool(4);
    let first = grow(&mut pool, &tally).unwrap();
    let second = grow(&mut pool, &tally).unwrap();
    let held = (0..8).map(|slot| pool.take(Holder::Bank(slot)).unwrap()).collect::<Vec<_>>();
    // A shrink frees two granules of each block, one behind a flight.
    let read = flight();
    for (index, claimed) in held.iter().enumerate().filter(|(index, _)| index % 4 < 2) {
        pool.release(claimed.granule, (index == 0).then(|| read.clone()));
    }
    pool.owe(4 * GRANULE);
    assert_eq!(pool.owed(), 4 * GRANULE, "four granules free or fenced cover the debt");
    // A KV claim reuses one: three granules could return now, so three are owed.
    let kv = pool.take(Holder::Kv(0)).unwrap();
    assert!([first, second].contains(&kv.granule.block));
    assert_eq!(pool.owed(), 3 * GRANULE);
    // Two more claims leave one free and one fenced.
    pool.take(Holder::Kv(1)).unwrap();
    assert_eq!(pool.owed(), 2 * GRANULE, "the fenced granule still counts toward the debt");
    pool.take(Holder::Kv(2)).unwrap();
    assert_eq!(pool.owed(), GRANULE);
    assert!(pool.take(Holder::Kv(3)).is_none(), "the last is fenced");
    read.complete();
    pool.take(Holder::Kv(3)).unwrap();
    assert_eq!(pool.owed(), 0, "every freed byte serves the process again: nothing is owed");
    assert!(pool.emptiest().is_none(), "and nothing is vacated for it");
}

/// Bytes owed for a shrink whose granules a claim reused before the return
/// was asked are owed only as far as granules are still free or fenced; a
/// copy into a vacating block's place is a move, not a reuse, and pays
/// nothing.
#[test]
fn a_debt_is_never_more_than_the_unheld_granules() {
    let (mut pool, _, tally) = pool(2);
    grow(&mut pool, &tally).unwrap();
    let newer = grow(&mut pool, &tally).unwrap();
    let held = (0..4).map(|slot| pool.take(Holder::Bank(slot)).unwrap()).collect::<Vec<_>>();
    pool.release(held[0].granule, None);
    pool.release(held[1].granule, None);
    pool.take(Holder::Kv(0)).unwrap();
    pool.owe(2 * GRANULE);
    assert_eq!(pool.owed(), GRANULE, "one of the two freed granules was reused");
    pool.release(held[2].granule, None);
    pool.owe(GRANULE);
    assert_eq!(pool.owed(), 2 * GRANULE);
    let vacating = pool.vacate(newer).expect("one live granule, one free elsewhere");
    assert_eq!(pool.owed(), 2 * GRANULE, "a move pays nothing");
    pool.release(vacating.moves()[0].from, None);
    assert_eq!(pool.vacated(vacating), 2 * GRANULE);
    assert_eq!(pool.owed(), 0, "the vacated block pays the debt");
    assert_eq!(returned(&mut pool), 0, "and returns, its bytes reported by its vacate");
}

/// Concurrent threads claiming from an empty pool via [`GrowGate`] allocate
/// only one block when that block has enough granules for both callers, and
/// the second caller deterministically parks on `GrowTurn::Waited`.
#[test]
fn single_flight_concurrent_growth_allocates_one_block() {
    use std::sync::atomic::AtomicBool;
    use std::sync::{Barrier, Mutex};
    let (pool, allocations, tally) = pool(2);
    let pool = Arc::new(Mutex::new(pool));
    let gate = Arc::new(GrowGate::new());
    let blocks = Arc::new(Counted { allocations: allocations.clone(), slack: 0 });
    let entry: Arc<dyn Entry> = Arc::new(tally);
    let t1_growing = Arc::new(Barrier::new(2));
    let t2_parked = Arc::new(Barrier::new(2));
    let t2_waited = Arc::new(AtomicBool::new(false));

    let t1 = {
        let pool = pool.clone();
        let gate = gate.clone();
        let blocks = blocks.clone();
        let entry = entry.clone();
        let t1_growing = t1_growing.clone();
        let t2_parked = t2_parked.clone();
        seismic_threads::spawn("pool-grow-t1", move || loop {
            if let Some(claimed) = pool.lock().unwrap().take(Holder::Bank(0)) {
                return claimed;
            }
            match gate.enter_or_wait() {
                GrowTurn::Waited => panic!("t1 must be the grower"),
                GrowTurn::Grow(_guard) => {
                    t1_growing.wait();
                    // Wait until t2 is deterministically parked inside GrowGate.
                    t2_parked.wait();
                    let block = blocks.allocate(2 * GRANULE, &entry, SlabFill::Unwritten).unwrap();
                    pool.lock().unwrap().install(block, SlabFill::Unwritten);
                }
            }
        })
        .unwrap()
    };

    t1_growing.wait();
    let t2 = {
        let pool = pool.clone();
        let gate = gate.clone();
        let t2_parked = t2_parked.clone();
        let t2_waited = t2_waited.clone();
        seismic_threads::spawn("pool-grow-t2", move || loop {
            if let Some(claimed) = pool.lock().unwrap().take(Holder::Bank(1)) {
                return claimed;
            }
            match gate.enter_or_wait_with(|| {
                t2_parked.wait();
            }) {
                GrowTurn::Waited => {
                    t2_waited.store(true, Ordering::Relaxed);
                    continue;
                }
                GrowTurn::Grow(_guard) => panic!("t2 must wait on t1's growth, not allocate a second block"),
            }
        })
        .unwrap()
    };

    let c1 = t1.join().unwrap();
    let c2 = t2.join().unwrap();
    assert!(t2_waited.load(Ordering::Relaxed), "t2 deterministically observed GrowTurn::Waited");
    assert_ne!(c1.granule, c2.granule, "each caller claims a distinct granule");
    assert_eq!(allocations.load(Ordering::Relaxed), 1, "single-flight growth allocates one block, not two");
}

/// A grower that returns an error or panics while holding [`GrowGuard`] clears
/// `growing` and wakes waiting threads cleanly without hanging.
#[test]
fn single_flight_growth_wakes_waiters_on_error_and_panic() {
    use std::sync::Barrier;
    let gate = Arc::new(GrowGate::new());

    // Error path: grower drops guard on Err after waiter is deterministically parked.
    let guard = match gate.enter_or_wait() {
        GrowTurn::Grow(g) => g,
        GrowTurn::Waited => panic!("first caller must grow"),
    };
    let parked = Arc::new(Barrier::new(2));
    let waiter = {
        let gate = gate.clone();
        let parked = parked.clone();
        seismic_threads::spawn("pool-grow-wait-err", move || {
            matches!(
                gate.enter_or_wait_with(|| {
                    parked.wait();
                }),
                GrowTurn::Waited
            )
        })
        .unwrap()
    };
    parked.wait();
    drop(guard);
    assert!(waiter.join().unwrap(), "waiter wakes after grower drops guard on error");

    // Panic path: grower panics while holding guard after waiter is deterministically parked.
    let mut waiter2 = None;
    let parked2 = Arc::new(Barrier::new(2));
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = match gate.enter_or_wait() {
            GrowTurn::Grow(g) => g,
            GrowTurn::Waited => panic!("expected grow turn"),
        };
        let gate_for_waiter = gate.clone();
        let parked_for_waiter = parked2.clone();
        waiter2 = Some(
            seismic_threads::spawn("pool-grow-wait-panic", move || {
                matches!(
                    gate_for_waiter.enter_or_wait_with(|| {
                        parked_for_waiter.wait();
                    }),
                    GrowTurn::Waited
                )
            })
            .unwrap(),
        );
        parked2.wait();
        panic!("simulated allocator panic");
    }));
    assert!(panicked.is_err());
    assert!(waiter2.unwrap().join().unwrap(), "waiter wakes after grower panics");
    assert!(matches!(gate.enter_or_wait(), GrowTurn::Grow(_)), "gate is usable after panic");
}

/// A block that holds both held granules and fenced (recently released)
/// granules can be vacated into free granules elsewhere: only the held
/// granules move, and the block returns once both the pre-existing fenced
/// flight and the copy's flight complete.
#[test]
fn a_block_with_fenced_granules_vacates_and_returns_behind_all_flights() {
    let (mut pool, _, tally) = pool(4);
    let older = grow(&mut pool, &tally).unwrap();
    let newer = grow(&mut pool, &tally).unwrap();
    let held = (0..8).map(|slot| pool.take(Holder::Bank(slot)).unwrap()).collect::<Vec<_>>();
    // Free 2 granules in the older block (immediately free) and 2 in the newer
    // block behind an in-flight read.
    pool.release(held[0].granule, None);
    pool.release(held[1].granule, None);
    let read = flight();
    pool.release(held[4].granule, Some(read.clone()));
    pool.release(held[5].granule, Some(read.clone()));
    pool.owe(4 * GRANULE);

    assert_eq!(pool.emptiest(), Some(newer), "newer has 2 held granules and 2 free elsewhere");
    let vacating = pool.vacate(newer).expect("fenced granules in newer do not block vacation");
    assert_eq!(vacating.moves().len(), 2, "only held granules move; fenced ones stay fenced");
    assert!(vacating.moves().iter().all(|m| m.from.block == newer && m.to.block == older));
    let copy = flight();
    for moved in vacating.moves() {
        pool.release(moved.from, Some(copy.clone()));
    }
    assert_eq!(pool.vacated(vacating), 4 * GRANULE);
    assert_eq!(pool.owed(), 0);
    assert_eq!(pool.awaited().map(|fences| fences.len()), Some(4), "awaits both read and copy fences");

    read.complete();
    assert_eq!(returned(&mut pool), 0, "still waiting on copy flight");
    assert_eq!(tally.held(), 8 * GRANULE);

    copy.complete();
    assert_eq!(returned(&mut pool), 0, "block returns, its bytes already reported by vacated");
    assert_eq!((pool.held_bytes(), tally.held()), (4 * GRANULE, 4 * GRANULE));
}

/// `emptiest_excluding` skips blocks in the exclusion set so callers can
/// move past a temporarily pinned block and vacate another.
#[test]
fn emptiest_excluding_skips_blocked_blocks() {
    let (mut pool, _, tally) = pool(2);
    let first = grow(&mut pool, &tally).unwrap();
    let second = grow(&mut pool, &tally).unwrap();
    let third = grow(&mut pool, &tally).unwrap();
    let held = (0..6).map(|slot| pool.take(Holder::Bank(slot)).unwrap()).collect::<Vec<_>>();
    // Free one granule in each block: each has 1 held and 2 free elsewhere.
    for index in [0, 2, 4] {
        pool.release(held[index].granule, None);
    }
    assert_eq!(pool.emptiest(), Some(third), "newest tied block first");
    let skip: BTreeSet<_> = [third].into_iter().collect();
    assert_eq!(pool.emptiest_excluding(&skip), Some(second), "skips third and picks second");
    let skip: BTreeSet<_> = [third, second].into_iter().collect();
    assert_eq!(pool.emptiest_excluding(&skip), Some(first), "skips third and second and picks first");
}

/// `return_empty()` must never remove a block in `BlockState::Vacating`, even
/// when all of its granules become `Free` before `vacated()` or `unvacate()`
/// runs (e.g. when a concurrent release frees the last granule while the pool
/// mutex is dropped during `vacate_emptiest`).
#[test]
fn return_empty_does_not_steal_vacating_block() {
    // 1. Vacated path: all granules become Free while Vacating, return_empty
    //    leaves the block in place, vacated() succeeds, and then return_empty
    //    removes it as a Returning block.
    let (mut pool, _, tally) = pool(2);
    grow(&mut pool, &tally).unwrap();
    let newer = grow(&mut pool, &tally).unwrap();
    let older_slot = pool.take(Holder::Kv(0)).unwrap();
    pool.take(Holder::Kv(1)).unwrap();
    let newer_slot = pool.take(Holder::Bank(0)).unwrap();
    assert_eq!(newer_slot.granule.block, newer);
    pool.release(older_slot.granule, None);

    let vacating = pool.vacate(newer).expect("one free granule elsewhere");
    assert_eq!(vacating.block(), newer);
    assert_eq!(vacating.moves().len(), 1);
    // Source granule is released immediately (no fence), leaving every granule
    // in `newer` Free while `newer` is still in `BlockState::Vacating`.
    pool.release(vacating.moves()[0].from, None);
    assert!(pool.return_empty().is_empty(), "return_empty must not steal a Vacating block");
    assert_eq!(pool.held_bytes(), 4 * GRANULE);

    assert_eq!(pool.vacated(vacating), 2 * GRANULE, "vacated succeeds on the still-held block");
    assert_eq!(returned(&mut pool), 0, "Returning block now leaves with 0 bytes (already reported)");
    assert_eq!((pool.held_bytes(), tally.held()), (2 * GRANULE, 2 * GRANULE));

    // 2. Unvacate path: all granules become Free while Vacating, return_empty
    //    does not steal it, unvacate() succeeds, and then return_empty removes it.
    let third = grow(&mut pool, &tally).unwrap();
    let vacating = pool.vacate(third).expect("empty block vacates with 0 moves");
    assert!(vacating.moves().is_empty());
    assert!(pool.return_empty().is_empty(), "return_empty must not steal a 0-move Vacating block");
    pool.unvacate(vacating);
    assert_eq!(returned(&mut pool), 2 * GRANULE, "Serving empty block returns after unvacate");
    assert_eq!((pool.held_bytes(), tally.held()), (2 * GRANULE, 2 * GRANULE));
}

/// Concurrent `owe()` or `take()` while a `vacate()` is in flight must not
/// double-count moving granules in `unheld_bytes()` or erase owed OS returns.
#[test]
fn owe_and_take_during_vacate_do_not_erase_owed_bytes() {
    let (mut pool, _, tally) = pool(2);
    let b0 = grow(&mut pool, &tally).unwrap();
    let b1 = grow(&mut pool, &tally).unwrap();
    let b2 = grow(&mut pool, &tally).unwrap();
    // Claim 2 granules in b0, 2 in b1, 1 in b2, then release 1 in b0 and 1 in b1
    // so b0/b1/b2 each have 1 held + 1 free (3 held, 3 free total).
    let g0 = pool.take(Holder::Bank(0)).unwrap();
    pool.take(Holder::Bank(1)).unwrap();
    let g2 = pool.take(Holder::Bank(2)).unwrap();
    pool.take(Holder::Bank(3)).unwrap();
    let g4 = pool.take(Holder::Bank(4)).unwrap();
    assert_eq!((g0.granule.block, g2.granule.block, g4.granule.block), (b0, b1, b2));
    pool.release(g0.granule, None);
    pool.release(g2.granule, None);
    pool.owe(3 * GRANULE);
    assert_eq!(pool.owed(), 3 * GRANULE);

    // Begin vacating b2: its 1 live granule moves into b0's free slot, so
    // `holding` increases by 1 while `in_flight_moves` increases by 1.
    let vacating = pool.vacate(b2).expect("b2 has 1 live granule and 2 free elsewhere");
    assert_eq!(vacating.moves().len(), 1);

    // `owe(0)` runs `cap_owed()`: must NOT clamp owed from 3 down to 2.
    pool.owe(0);
    assert_eq!(pool.owed(), 3 * GRANULE, "in-flight move must not reduce unheld_bytes");

    // A concurrent `take()` claims the remaining free granule in b1: owed
    // should decrease by exactly 1 granule (to 2), not by 2 (to 1).
    let reused = pool.take(Holder::Kv(0)).expect("b1 still has 1 free granule");
    assert_eq!(reused.granule.block, b1);
    assert_eq!(pool.owed(), 2 * GRANULE, "only the genuinely claimed granule pays owed");

    // Complete the move and vacate b2: its 2*GRANULE pays the remaining debt.
    pool.release(vacating.moves()[0].from, None);
    assert_eq!(pool.vacated(vacating), 2 * GRANULE);
    assert_eq!(pool.owed(), 0);
    assert_eq!(returned(&mut pool), 0);
    assert_eq!((pool.held_bytes(), tally.held()), (4 * GRANULE, 4 * GRANULE));
}

/// `block_bytes` completes in `O(MAX_BLOCK_UNITS)` time even for a tiny
/// granule coprime to a huge granularity, and matches brute-force search on
/// small inputs.
#[test]
fn block_bytes_small_granule_coprime_to_large_granularity() {
    // 1. Exact multiple within span: granule = 3, granularity = 1 MiB.
    let exact = block_bytes(BlockSizing {
        granule: 3,
        allocations: NativeAllocations { least: 0, granularity: 1 << 20 },
    })
    .unwrap();
    assert_eq!(exact, BlockSize { bytes: 3 * (1 << 20), held: 3 * (1 << 20), absent: 1 << 20 });

    // 2. No exact multiple within span: granule = 257 (prime), granularity = 1 GiB.
    //    `most = 64 * 1 GiB / 257 ≈ 267_386_880`, which previously looped ~267M times.
    let huge = block_bytes(BlockSizing {
        granule: 257,
        allocations: NativeAllocations { least: 0, granularity: 1 << 30 },
    })
    .unwrap();
    assert_eq!(huge.held % (1 << 30), 0);
    assert!(huge.held >= huge.bytes);
    assert!(huge.bytes <= MAX_BLOCK_UNITS * (1 << 30));
    assert!(huge.held - huge.bytes < 257);

    // 3. Cross-check against brute force across varied regimes (coprime,
    //    least > granularity, granule > granularity, ties).
    let brute = |sizing: BlockSizing| -> BlockSize {
        let BlockSizing { granule, allocations: NativeAllocations { least, granularity } } = sizing;
        let fewest = least.max(granule).div_ceil(granule);
        let most = (least.max(granularity) * MAX_BLOCK_UNITS / granule).max(fewest);
        let mut best: Option<BlockSize> = None;
        for g in fewest..=most {
            let bytes = g * granule;
            let held = bytes.next_multiple_of(granularity);
            let waste = held - bytes;
            let better = best.is_none_or(|b| {
                u128::from(waste) * u128::from(b.bytes / granule) < u128::from(b.held - b.bytes) * u128::from(g)
            });
            if better {
                best = Some(BlockSize { bytes, held, absent: granule.next_multiple_of(granularity) });
            }
            if waste == 0 {
                break;
            }
        }
        best.unwrap()
    };
    for (granule, least, granularity) in [
        (7, 0, 100),
        (13, 50, 100),
        (13, 300, 100),
        (137, 0, 100),
        (137, 500, 100),
        (101, 0, 100),
        (199, 0, 100),
        (17, 200, 19),
    ] {
        let sizing = BlockSizing { granule, allocations: NativeAllocations { least, granularity } };
        assert_eq!(block_bytes(sizing).unwrap(), brute(sizing), "mismatch for {sizing:?}");
    }
}
