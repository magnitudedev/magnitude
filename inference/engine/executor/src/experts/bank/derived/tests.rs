//! The bank's target as the catalog's failures met it, on a ledger with its
//! readings handed in: no OS, no device.

use super::*;
use crate::memory::ledger::{Available, Blind, Claim, Ledger, Ledgers, LoadShape, Need};
use std::sync::Arc;
use std::time::{Duration, Instant};

const GB: u64 = 1_000_000_000;

/// Flash Next q4's slot on the laptop's 64 GB M4 Max: 148,111,360 bytes
/// over every layer (its 131→132 growth).
const SLOT: u64 = 148_111_360;

/// The laptop's unified memory.
const TOTAL: u64 = 64 << 30;

/// Flash Next's bank on the laptop: a floor, the 226 slots its counts were
/// derived up to as its saturation, and a block of one slot (the
/// reservation's step).
fn shape() -> BankShape {
    BankShape::new(SLOT, 8, 226, 226)
}

/// The laptop's bank's charge: a slot's bytes each.
fn charge(slots: u64) -> u64 {
    slots * SLOT
}

/// A ledger of host RAM in a scope of its own, and its readings' clock.
struct Store {
    _scope: Ledgers,
    ledger: Ledger,
    start: Instant,
    readings: u32,
}

impl Store {
    fn new() -> Self {
        let scope = Ledgers::new();
        let ledger = scope.ledger(seismic::StoreIdentity::HostRam, Arc::new(LoadShape { stores: Vec::new() }));
        Self { _scope: scope, ledger, start: Instant::now(), readings: 0 }
    }

    /// A reading with other programs holding `others`: the OS shows free
    /// what neither they nor the process's allocations hold.
    fn read(&mut self, others: u64) {
        self.readings += 1;
        let at = self.start + Duration::from_millis(100) * self.readings;
        let allocated = self.ledger.snapshot().allocated();
        let available = TOTAL.checked_sub(others + allocated).expect("others and the process fit the store");
        self.ledger.read(self.ledger.sampled(at), Ok(Available::new(TOTAL, available, None).expect("a reading")));
    }

    /// A committed claim of exactly `bytes`, allocated.
    fn commit(&self, bytes: u64) -> Claim {
        let mut claim = self.ledger.claim(Need::Measurement { bytes }).expect("the claim fits");
        self.ledger.change(|changing| claim.allocated(bytes, changing));
        claim
    }

    fn target(&self, in_process: u64) -> u64 {
        target_slots(bank_room(&self.ledger.snapshot(), in_process), &shape(), &charge)
    }
}

/// What other programs hold where the laptop's bank derives `slots`
/// beside `committed`, with half a slot to spare.
fn others_at(slots: u64, committed: u64) -> u64 {
    TOTAL - committed - (slots - shape().floor_slots()) * SLOT - SLOT / 2
}

/// Item 9: the laptop's sequence (Flash Next q4, 64 GB M4 Max, main
/// b4c36dba9). A bank of 132 slots, a brief rise of others' memory and its
/// fall a reading later, then a 103-token prompt's state claimed and its
/// decode: the bank's target is 132 again at the reading after the rise
/// falls, and stays 132 through the prompt and its decode. On main a
/// settle reserved 104 places for that prompt and released the bank to 54
/// slots, and nothing grew it back.
#[test]
fn the_laptops_bank_holds_132_slots_through_a_prompt_and_its_decode() {
    let mut store = Store::new();
    let committed = 10 * GB + shape().floor_slots() * SLOT;
    let base = others_at(132, committed);
    store.read(base);
    let _weights = store.commit(committed);
    let _bank = store.ledger.cached(CacheKind::Bank, (132 - shape().floor_slots()) * SLOT).expect("the bank's room");
    store.read(base);
    assert_eq!(store.target(0), 132, "the bank as it stood");
    // Others take a third of a slot for a reading (the laptop's brief
    // Reclaim), and give it back.
    store.read(base + SLOT / 3);
    assert!(store.target(0) < 132, "the rise is room the bank gives up");
    store.read(base);
    assert_eq!(store.target(0), 132, "the next reading derives the bank afresh");
    // The prompt's state: 103 rows, a few MB, claimed as the prompt runs.
    let chunk = 8 << 20;
    let _state = store.commit(chunk);
    store.read(base);
    assert_eq!(store.target(chunk), 132, "the prompt's state takes no slot");
    // Its decode: 108 tokens, readings a tenth of a second apart.
    for _ in 0..10 {
        store.read(base);
        assert_eq!(store.target(0), 132, "the decode runs on 132 slots");
    }
}

/// Items 4, 5, 12, 13, 15, 44: growth is never dropped, nor its room read
/// once. As another program gives its memory back over readings, each
/// reading's target follows it up, by every slot the freed room holds, to
/// the saturation.
#[test]
fn the_target_follows_freed_memory_up_at_every_reading() {
    let mut store = Store::new();
    let committed = 10 * GB + shape().floor_slots() * SLOT;
    store.read(TOTAL - committed - SLOT);
    let _weights = store.commit(committed);
    let mut before = shape().floor_slots();
    for slots in (20..=300).step_by(20) {
        let others = TOTAL.saturating_sub(committed + (slots - shape().floor_slots()) * SLOT + SLOT / 2);
        store.read(others);
        // A fall is no rise: the guarantee's outside term is zero.
        let target = store.target(0);
        assert_eq!(target, slots.min(shape().saturation_slots()), "{slots} slots of room");
        assert!(target >= before, "the target never stays behind freed memory");
        before = target;
    }
}

/// Items 3, 42: one formula on one reading. A committed claim lowers the
/// target by the slots its bytes cover, at the reading that pairs it with
/// the OS's figure, and its release raises it back.
#[test]
fn a_committed_claim_moves_the_target_by_its_bytes() {
    let mut store = Store::new();
    let committed = 10 * GB + shape().floor_slots() * SLOT;
    let base = others_at(150, committed);
    store.read(base);
    let _weights = store.commit(committed);
    store.read(base);
    assert_eq!(store.target(0), 150);
    let job = store.commit(20 * SLOT);
    store.read(base);
    assert_eq!(store.target(0), 130, "a measurement job's room");
    drop(job);
    store.read(base);
    assert_eq!(store.target(0), 150, "given back");
}

/// The speed guarantee is subtracted, nothing else: one prompt chunk's
/// mandatory growth lowers the target by the slots it covers.
#[test]
fn the_guarantee_is_the_only_margin() {
    let mut store = Store::new();
    let committed = 10 * GB + shape().floor_slots() * SLOT;
    let base = others_at(150, committed);
    store.read(base);
    let _weights = store.commit(committed);
    store.read(base);
    assert_eq!(store.target(0), 150);
    assert_eq!(store.target(3 * SLOT), 147);
}

/// A blind store, or committed memory filling the budget, holds the floor.
#[test]
fn a_blind_or_overcommitted_store_holds_the_floor() {
    let mut store = Store::new();
    store.read(TOTAL / 2);
    store.ledger.read(store.ledger.sampled(Instant::now()), Err(Blind("no reading".into())));
    assert_eq!(store.target(0), shape().floor_slots(), "blind");
    let mut store = Store::new();
    store.read(0);
    let _weights = store.commit(TOTAL / 2);
    store.read(TOTAL / 2);
    assert_eq!(store.target(0), shape().floor_slots(), "committed fills the budget");
}

/// A shrink whenever the target falls below the bank; a growth only once
/// it is a block above.
#[test]
fn the_bank_grows_by_blocks_and_shrinks_at_once() {
    let block = BankShape::new(4 * SLOT, 8, 226, 226);
    assert_eq!(bank_move(131, 132, &block, &charge), Some(BankMove::Shrink { to: 131 }));
    assert_eq!(bank_move(132, 132, &block, &charge), None);
    assert_eq!(bank_move(135, 132, &block, &charge), None, "short of a block");
    assert_eq!(bank_move(136, 132, &block, &charge), Some(BankMove::Grow { to: 136 }));
}

/// A charge in steps (slots crossing their storage's granule, two slots a
/// step here): the room holds a slot count only where its whole step fits,
/// as the plan sizes its bank and the target reads it, by one function.
#[test]
fn slots_fit_where_their_charge_step_fits() {
    let stepped = |slots: u64| Ok::<_, std::convert::Infallible>(slots.div_ceil(2) * 2 * SLOT);
    let within = |room| slots_within(room, (8, 226), &stepped).unwrap_or_else(|never| match never {});
    assert_eq!(within(0), 8);
    assert_eq!(within(SLOT), 8, "half a step");
    assert_eq!(within(2 * SLOT), 10);
    assert_eq!(within(u64::MAX / 2), 226, "every slot");
}

/// The bank never passes its plan's ceiling, whatever the miss curve says,
/// nor where its misses stop falling below it.
#[test]
fn the_bank_holds_at_most_its_ceiling_or_saturation() {
    assert_eq!(BankShape::new(SLOT, 8, 226, 300).saturation_slots(), 226, "the ceiling");
    assert_eq!(BankShape::new(SLOT, 8, 226, 140).saturation_slots(), 140, "the curve");
    assert_eq!(BankShape::new(SLOT, 8, 226, 4).saturation_slots(), 8, "never below the floor");
}

/// Today's floor of a bank whose layers fall in `groups` record forms: each
/// group its own least records (`ExpertBankPlan::least_records`), the
/// decode step's parked layer and the hint window in every group, the lent
/// scratch in the lender's.
fn per_group(k: u64, e: u64, lent: u64, forms: u64) -> u64 {
    let decode = (crate::operators::routed::fused_graph::DECODE_ROWS * k).min(e);
    let reserve = 2 * k;
    let lender = decode.max(lent) + reserve;
    lender + (forms - 1) * (decode + reserve)
}

/// The pooled floor is one step's working set for the whole bank: the
/// parked layer or the lent scratch once, beside each staging pool's hint
/// window, never a group's decode records again in every other group.
#[test]
fn the_pooled_floor_is_one_steps_working_set() {
    let decode = |k: u64, e: u64| (crate::operators::routed::fused_graph::DECODE_ROWS * k).min(e);
    for (k, e, lent, forms) in [(8, 256, 64, 1), (8, 256, 64, 3), (6, 384, 400, 4), (8, 32, 16, 2)] {
        let set = WorkingSet::new(k, e, lent, forms);
        let pooled = decode(k, e).max(lent) + 2 * k * forms;
        assert_eq!(set.floor_records(), pooled, "{set:?}");
        assert!(set.floor_records() <= per_group(k, e, lent, forms), "{set:?}: never more than each group's own");
        assert_eq!(per_group(k, e, lent, forms) - set.floor_records(), (forms - 1) * decode(k, e), "{set:?}: the decode records once, not per group");
    }
}

/// The prompt's scratch is what the ledger grants, never less than one
/// decode step's set: a scratch of a few records holds the floor at the
/// decode set, and the prompt converts its layer in more rounds.
#[test]
fn the_scratch_is_never_below_the_decode_set() {
    let set = WorkingSet::new(6, 384, 0, 1);
    assert_eq!(set.lent(), 48);
    assert_eq!(set.floor_records(), 48 + 12, "V4.1's decode set at K = 6");
    assert_eq!(WorkingSet::new(6, 384, 430, 1).floor_records(), 442, "today's whole-layer scratch");
}

/// One store's holdings share its room in their order: the first takes
/// what its slots fit, the next what is left, each at least its floor.
#[test]
fn a_stores_holdings_share_its_room_in_order() {
    let table = |records: u64| records * (SLOT / 4);
    let holdings: [(BankShape, &dyn Fn(u64) -> u64); 2] = [(shape(), &charge), (BankShape::new(SLOT / 4, 2, 40, 40), &table)];
    assert_eq!(shared_targets(0, &holdings), vec![8, 2], "floors");
    assert_eq!(shared_targets(10 * SLOT + SLOT / 2, &holdings), vec![18, 4], "the bank's ten slots, then the table's two records");
    assert_eq!(shared_targets(u64::MAX / 2, &holdings), vec![226, 40], "both saturated");
}

/// A load imports what its first reading keeps: on the laptop with others
/// holding 40 GB, an import whose weights, floor and later allocations
/// place 7 GB in the store holds the experts' target beside them and a
/// guarantee of half a GB; once the load holds them (the slots past its
/// floor as the bank's cache), the first reading derives that same target.
#[test]
fn a_load_imports_what_its_first_reading_keeps() {
    let mut store = Store::new();
    let others = 40 * GB;
    let in_process = GB / 2;
    let floor = shape().floor_slots();
    store.read(others);
    let beside = 7 * GB;
    let imported = import_target(&store.ledger.snapshot(), in_process, beside, &shape(), &charge);
    assert!(floor < imported && imported < shape().saturation_slots(), "{imported} slots");
    let _held = store.commit(beside);
    let _cache = store.ledger.cached(CacheKind::Bank, charge(imported) - charge(floor)).expect("the import's room holds its cache");
    store.read(others);
    assert_eq!(store.target(in_process), imported);
}
