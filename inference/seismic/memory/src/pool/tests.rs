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

    fn allocate(&self, bytes: u64, entry: &Arc<dyn Entry>) -> Result<Plain, Refused> {
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

fn pool_of(granules_a_block: u64, slack: u64, tally: &Tally) -> (GranulePool<Counted, Holder>, Arc<AtomicU64>) {
    let allocations = Arc::new(AtomicU64::new(0));
    let blocks = Counted { allocations: allocations.clone(), slack };
    (GranulePool::new(blocks, GRANULE, granules_a_block * GRANULE, Arc::new(tally.clone())), allocations)
}

fn pool(granules_a_block: u64) -> (GranulePool<Counted, Holder>, Arc<AtomicU64>, Tally) {
    let tally = Tally::unbounded(StoreIdentity::HostRam);
    let (pool, allocations) = pool_of(granules_a_block, 0, &tally);
    (pool, allocations, tally)
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
    pool.grow().unwrap();
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
    let (mut pool, allocations, _tally) = pool(8);
    pool.grow().unwrap();
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
    pool.grow().unwrap();
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
    let exact = |granules: u64| BlockSize { bytes: granules * granule, held: granules * granule };
    assert_eq!(at(21_474_837, 1).unwrap(), exact(11), "the fewest granules holding the least allocation");
    assert_eq!(at(1, 1).unwrap(), exact(1));
    assert_eq!(at(0, 1).unwrap(), exact(1));
    assert_eq!(block_bytes(BlockSizing { granule: 0, allocations: NativeAllocations::EXACT }), Err(SizingError::NoGranule));
    // A Qwen expert record under CUDA's 2 MiB granularity: 32 records fill
    // 27 granularity units exactly.
    let record = 1_769_472u64;
    let cuda = NativeAllocations { least: 0, granularity: 2 << 20 };
    let block = block_bytes(BlockSizing { granule: record, allocations: cuda }).unwrap();
    assert_eq!(block, BlockSize { bytes: 32 * record, held: 27 * (2 << 20) });
    assert_eq!(block.charge(record, 33), Some(2 * block.held), "granules cost whole blocks");
    // Whole granules fill whole units only far past the span: the block
    // wasting least per granule within it, its rounding held.
    let odd = (2 << 20) + 256;
    let block = block_bytes(BlockSizing { granule: odd, allocations: cuda }).unwrap();
    assert!(block.bytes <= MAX_BLOCK_UNITS * (2 << 20) && block.held % (2 << 20) == 0 && block.held >= block.bytes);
    // A multi-tens-of-MiB state granule under CUDA's 2 MiB granularity stays
    // bounded by the 64-unit (128 MiB) span rather than scaling the span up
    // by the granule itself.
    let qwen_state = 65_625_472u64;
    let block = block_bytes(BlockSizing { granule: qwen_state, allocations: cuda }).unwrap();
    assert_eq!(block, BlockSize { bytes: 2 * qwen_state, held: 63 * (2 << 20) });
}

/// Emptying a block for outside demand copies its live granules into free
/// ones elsewhere; it returns once their sources' last reads complete.
#[test]
fn a_block_returns_only_once_its_live_granules_are_copied_out() {
    let (mut pool, _, tally) = pool(2);
    let older = pool.grow().unwrap();
    let newer = pool.grow().unwrap();
    let kept = pool.take(Holder::Kv(0)).unwrap();
    let moved = (0..2).map(|slot| pool.take(Holder::Bank(slot)).unwrap()).last().unwrap();
    assert_eq!(moved.granule.block, newer);
    pool.release(kept.granule, None);
    assert_eq!(pool.emptiest(), Some(newer), "one live granule, and one free elsewhere");
    let vacating = pool.vacate(newer).expect("a granule is free elsewhere");
    assert_eq!(vacating.moves.len(), 1, "only live granules are copied");
    let copy = vacating.moves[0];
    assert_eq!((copy.holder, copy.from, copy.to), (Holder::Bank(1), moved.granule, kept.granule));
    assert_eq!(copy.to.block, older);
    // The holder copied and repointed: it releases the source behind the
    // flight that read it.
    let read = flight();
    pool.release(copy.from, Some(read.clone()));
    assert_eq!(pool.vacated(newer), 2 * GRANULE, "every live granule moved out: the block is returning");
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
    pool.grow().unwrap();
    let newer = pool.grow().unwrap();
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
    assert_eq!(vacating.moves.len(), 1);
    let copy = flight();
    pool.release(vacating.moves[0].from, Some(copy.clone()));
    assert_eq!(pool.vacated(newer), 2 * GRANULE);
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
    let (mut pool, _, _tally) = pool(2);
    let older = pool.grow().unwrap();
    let newer = pool.grow().unwrap();
    let kept = pool.take(Holder::Kv(0)).unwrap();
    let stays = pool.take(Holder::Kv(1)).unwrap();
    let live = pool.take(Holder::Bank(0)).unwrap();
    assert_eq!((stays.granule.block, live.granule.block), (older, newer));
    pool.release(kept.granule, None);
    let vacating = pool.vacate(newer).expect("a granule is free elsewhere");
    assert!(pool.take(Holder::Kv(2)).is_none(), "a vacating block takes no claims");
    // The copy failed: its destination goes back free, the block back to
    // service, its live granule where it was.
    pool.release(vacating.moves[0].to, None);
    pool.unvacate(vacating.block);
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
    let (mut pool, _) = pool_of(4, 4096, &tally);
    pool.grow().unwrap();
    pool.grow().unwrap();
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
    let (mut pool, allocations) = pool_of(4, 0, &tally);
    pool.grow().unwrap();
    match pool.grow() {
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
    let (mut pool, _) = pool_of(4, 512, &tally);
    pool.grow().unwrap();
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
        pool.grow().unwrap();
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
    let (mut pool, _, _tally) = pool(4);
    let first = pool.grow().unwrap();
    let second = pool.grow().unwrap();
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
    let (mut pool, _, _tally) = pool(2);
    pool.grow().unwrap();
    let newer = pool.grow().unwrap();
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
    pool.release(vacating.moves[0].from, None);
    assert_eq!(pool.vacated(newer), 2 * GRANULE);
    assert_eq!(pool.owed(), 0, "the vacated block pays the debt");
    assert_eq!(returned(&mut pool), 0, "and returns, its bytes reported by its vacate");
}
