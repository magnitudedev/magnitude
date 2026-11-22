//! The bank's sizes from a store's room: the slots a room holds, how a
//! store's holdings share it, and the one resize toward a target. The room
//! itself is the ledger's estimate (`Estimate::target`), tested with it.

use super::*;

/// Flash Next q4's slot on the laptop's 64 GB M4 Max: 148,111,360 bytes
/// over every layer (its 131→132 growth).
const SLOT: u64 = 148_111_360;

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

/// Cache claims of the process took is the ledger's fact, not the room's
/// movement: a bank owing a claim a third of a slot (blocks of one slot)
/// sheds a whole slot though its room moved by less than a block, sheds two
/// for a slot and a third, and grows by none while it owes; owing nothing,
/// the same room keeps its slots.
#[test]
fn a_bank_sheds_what_claims_took_in_whole_blocks() {
    let room = (132 - 8) * SLOT;
    assert_eq!(held_target(room - SLOT / 3, 132, SLOT / 3, &shape(), &charge), 131, "a third of a slot owed");
    assert_eq!(held_target(room - SLOT / 3, 132, 0, &shape(), &charge), 132, "the same room, nothing owed: a shortage's yield");
    assert_eq!(held_target(room, 132, SLOT + SLOT / 3, &shape(), &charge), 130, "a slot and a third");
    assert_eq!(held_target(room + 10 * SLOT, 132, 1, &shape(), &charge), 131, "no growth while it owes a byte");
    let block = BankShape::new(4 * SLOT, 8, 226, 226);
    assert_eq!(held_target(room, 132, 1, &block, &charge), 128, "a block of four slots for a byte");
    assert_eq!(held_target(0, 9, SLOT, &shape(), &charge), 8, "never below the floor");
}

/// The block is the one threshold both ways: a bank of 132 slots in blocks
/// of four grows once its room holds a block more, and shrinks once it holds
/// a block more than its room, to what the room fits.
#[test]
fn the_bank_moves_by_a_block_either_way() {
    let block = BankShape::new(4 * SLOT, 8, 226, 226);
    let room = |slots: u64| (slots - 8) * SLOT;
    let held = |room| held_target(room, 132, 0, &block, &charge);
    assert_eq!(held(room(132)), 132);
    assert_eq!(held(room(135) + SLOT / 2), 132, "short of a block above");
    assert_eq!(held(room(136)), 136, "a block above");
    assert_eq!(held(room(129) + 1), 132, "short of a block below");
    assert_eq!(held(room(128)), 128, "a block below: to what the room fits");
    assert_eq!(held(room(120) + SLOT / 3), 120);
    assert_eq!(bank_move(132, 132), None);
    assert_eq!(bank_move(128, 132), Some(BankMove::Shrink { to: 128 }));
    assert_eq!(bank_move(136, 132), Some(BankMove::Grow { to: 136 }));
}

/// The laptop's bank in blocks of one slot: others' jitter of a third of a
/// slot for a reading lowers the room by less than a block, so the bank
/// holds; its next reading holds too.
#[test]
fn a_room_moving_within_a_block_resizes_nothing() {
    let room = (132 - 8) * SLOT;
    for jitter in [SLOT / 3, SLOT - 1, 0] {
        assert_eq!(bank_move(held_target(room - jitter, 132, 0, &shape(), &charge), 132), None, "{jitter} bytes of others' rise");
    }
    assert_eq!(held_target(room - SLOT, 132, 0, &shape(), &charge), 131, "a whole block's");
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
    // A host tier serves only what the device lacks: none beside a device
    // bank of every expert, whatever host RAM holds.
    assert_eq!(BankShape::host(SLOT, 256, 256, true).saturation_slots(), 0, "a whole device bank");
    assert_eq!(BankShape::host(SLOT, 256, 256, false).saturation_slots(), 256, "a device short of experts");
}

/// A bank grows all the way to its structural ceiling whenever room allows
/// and shrinks cleanly when room tightens — never capped by a shallow
/// intermediate slot count.
#[test]
fn the_bank_grows_to_its_ceiling_and_shrinks_to_room() {
    let room = |slots: u64| (slots - 8) * SLOT;
    let full = BankShape::new(SLOT, 8, 226, 226);
    assert_eq!(held_target(room(200), 126, 0, &full, &charge), 200, "grows into newly available room");
    assert_eq!(held_target(room(300), 126, 0, &full, &charge), 226, "grows up to ceiling");
    assert_eq!(held_target(room(100), 200, 0, &full, &charge), 100, "shrinks to tighter room");
    assert_eq!(held_target(room(226), 100, 0, &full, &charge), 226, "regrows to ceiling after shrink");
}

/// One store's holdings, each at its floor, share its room in their order:
/// the first takes what its slots fit, the next what is left, each at least
/// its floor.
#[test]
fn a_stores_holdings_share_its_room_in_order() {
    let table = |records: u64| records * (SLOT / 4);
    let holdings: [(BankShape, u64, u64, &dyn Fn(u64) -> u64); 2] = [(shape(), 8, 0, &charge), (BankShape::new(SLOT / 4, 2, 40, 40), 2, 0, &table)];
    assert_eq!(shared_targets(0, &holdings), vec![8, 2], "floors");
    assert_eq!(shared_targets(10 * SLOT + SLOT / 2, &holdings), vec![18, 4], "the bank's ten slots, then the table's two records");
    assert_eq!(shared_targets(u64::MAX / 2, &holdings), vec![226, 40], "both saturated");
}

/// A load imports what its first reading keeps: on the laptop's 64 GB with
/// others holding 40 GB and the load's in-process figure of half a GB held,
/// an import whose weights, floor and later allocations place 7 GB in the
/// store imports the experts' target beside them ([`import_target`]); once
/// the load holds them (the slots past its floor as the bank's cache), the
/// first reading derives that same target, so the load resizes nothing.
/// The load is idle, so its in-process figure counts in neither: nothing
/// grows in process until a sequence is admitted.
#[test]
fn a_load_imports_what_its_first_reading_keeps() {
    use crate::memory::ledger::{CacheKind, Ledgers, LoadShape};
    use crate::memory::supply::Reading;
    use std::time::{Duration, Instant};
    const GB: u64 = 1_000_000_000;
    const TOTAL: u64 = 64 << 30;
    let now = seismic::Lag { release: Duration::ZERO, growth: Duration::ZERO };
    let scope = Ledgers::new();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, std::sync::Arc::new(LoadShape { stores: Vec::new() }));
    let _pool = ledger.pool_block(crate::memory::ledger::PoolClass::Records, std::num::NonZeroU64::new(SLOT).unwrap()).blocks;
    let (start, others, beside) = (Instant::now(), 40 * GB, 7 * GB);
    // The OS shows free what neither other programs nor the load's
    // allocations hold.
    let read = |reading: u32, allocated: u64| {
        let at = ledger.sampled(start + Duration::from_millis(100) * reading);
        ledger.read(at, Reading::device(TOTAL, TOTAL - others - allocated, now));
    };
    read(1, 0);
    let imported = import_target(&ledger.estimate(), beside, &shape(), &charge);
    assert!(shape().floor_slots() < imported && imported < shape().saturation_slots(), "{imported} slots");
    let held = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, crate::memory::ledger::CommittedKind::Load, beside)).expect("the import's claims fit");
    ledger.change(|_| { let entry = held.entry(); entry.take(&entry.change(), beside, seismic::memory::Wiring::Pageable) }).expect("the import's own bytes");
    let cache = charge(imported) - charge(shape().floor_slots());
    let _cache = ledger.cached(CacheKind::Bank, cache).expect("the import's room holds its cache");
    read(2, beside + cache);
    let room = ledger.estimate().target();
    assert_eq!(target_slots(room, &shape(), &charge), imported, "the first reading keeps the import");
    assert_eq!(bank_move(held_target(room, imported, 0, &shape(), &charge), imported), None, "and resizes nothing");
}

/// A transient rise spike (`guarantee > 0`) lowers `grow_room` by `2x` (`budget`
/// drop + `guarantee`) while `hold_room` drops only once (`budget` drop): a
/// resident bank never shrinks on the transient `2x` derivative kick and holds
/// through two-thirds-block app jitter that old symmetric `held_target` would
/// have shrank and bounced back 10 ms later.
#[test]
fn asymmetric_hold_room_stops_guarantee_derivative_kick_oscillation() {
    let base = (132 - 8) * SLOT;
    let jitter = SLOT * 2 / 3;
    let hold_room = base - jitter;
    let grow_room = base - 2 * jitter;
    assert_eq!(held_target(grow_room, 132, 0, &shape(), &charge), 130, "symmetric target over-shrinks by 2x");
    assert_eq!(held_target_in(hold_room, grow_room, 132, 0, &shape(), &charge), 132, "hold_room absorbs single-block jitter");
    assert_eq!(held_target_in(base - SLOT, base - 2 * SLOT, 132, 0, &shape(), &charge), 131, "actual 1-slot drop sheds 1 slot, not 2");
}

/// Reproduces `f92704` on a 64 GiB Linux discrete-GPU host: at import time
/// `HostRam` supply (~59.84 GB) fits 135 planned host slots (384 MB each),
/// but subsequent domain allocation adds ~4 GB of uncharged process RSS plus
/// host floor/tables, dropping live `HostRam` `import_room()` below the 135-slot
/// charge. Re-clamping against the fresh post-domain reading shrinks the
/// host-tier target cleanly into live `import_room()`, never over-committing
/// past `HostRam` budget, and clamps all the way to 0 when post-domain supply
/// is exhausted.
#[test]
fn host_tier_target_clamps_to_post_domain_host_ram_supply() {
    use crate::memory::ledger::{CacheKind, CommittedKind, Ledgers, LoadShape};
    use crate::memory::supply::Reading;
    use std::time::{Duration, Instant};

    const MB: u64 = 1_000_000;
    const GB: u64 = 1_000_000_000;
    const TOTAL: u64 = 67_300 * MB;
    const HOST_SLOT: u64 = 384 * MB;
    const HOST_FLOOR: u64 = 3 * GB;
    const PLANNED_HOST_SLOTS: u64 = 135;

    let now = seismic::Lag { release: Duration::ZERO, growth: Duration::ZERO };
    let scope = Ledgers::new();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, std::sync::Arc::new(LoadShape { stores: Vec::new() }));
    let _pool = ledger.pool_block(crate::memory::ledger::PoolClass::Records, std::num::NonZeroU64::new(HOST_SLOT).unwrap()).blocks;
    let start = Instant::now();
    let host_charge = |slots: u64| HOST_FLOOR + slots * HOST_SLOT;
    let host_shape = BankShape::new(HOST_SLOT, 0, PLANNED_HOST_SLOTS, PLANNED_HOST_SLOTS);

    // 1. Pre-domain import reading: 59.84 GB supply (7.46 GB other/OS).
    let at_import = ledger.sampled(start);
    ledger.read(at_import, Reading::device(TOTAL, 59_840 * MB, now));
    let imported_slots = import_target(&ledger.estimate(), HOST_FLOOR, &host_shape, &host_charge);
    assert_eq!(imported_slots, PLANNED_HOST_SLOTS, "pre-domain reading fits all 135 planned host slots");

    // Claim committed host floor and hold the planned host-tier backing as
    // `CacheKind::Bank` (matching `ExpertBank::open` -> `open_host`).
    let floor_claim = ledger
        .change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, HOST_FLOOR))
        .expect("host floor fits");
    ledger
        .change(|_| {
            let entry = floor_claim.entry();
            entry.take(&entry.change(), HOST_FLOOR, seismic::memory::Wiring::Pageable)
        })
        .expect("host floor allocates");
    let planned_backing = PLANNED_HOST_SLOTS * HOST_SLOT;
    let _backing_cache = ledger.cache(CacheKind::Bank, planned_backing, crate::memory::ledger::Reclaim::Unbacked).expect("backing cache fits at import");

    // 2. Post-domain reading: domain allocation added ~8 GB of uncharged host
    //    RSS (bound graphs, CUDA driver host state, heap) beyond the 3 GB
    //    committed floor, leaving 48.35 GB live supply.
    let at_domain = ledger.sampled(start + Duration::from_secs(5));
    ledger.read(at_domain, Reading::device(TOTAL, 48_350 * MB, now));
    let post_domain = ledger.estimate();
    let clamped = target_slots(post_domain.import_room(), &host_shape, &host_charge);
    assert!(
        clamped < PLANNED_HOST_SLOTS,
        "post-domain clamp must drop below planned {PLANNED_HOST_SLOTS} slots, got {clamped}"
    );
    assert!(
        host_charge(clamped) - HOST_FLOOR <= post_domain.import_room(),
        "clamped host slots ({clamped}) must fit live post-domain import_room ({})",
        post_domain.import_room()
    );
    assert_eq!(clamped, (post_domain.import_room() / HOST_SLOT).min(PLANNED_HOST_SLOTS));

    // 3. Exhausted post-domain supply clamps cleanly to 0 host slots (floor only).
    let at_exhausted = ledger.sampled(start + Duration::from_secs(10));
    ledger.read(at_exhausted, Reading::device(TOTAL, 100 * MB, now));
    let clamped_zero = target_slots(ledger.estimate().pinned_import_room(), &host_shape, &host_charge);
    assert_eq!(clamped_zero, 0, "exhausted post-domain host supply clamps to 0 host slots");
}

/// On a 64 GiB Linux host (`Reading::host`), `Estimate::pinned_room()`,
/// `Estimate::pinned_import_room()` (used by `clamped_host_slots` and
/// `import_host_slots`), `Estimate::pinned_target()` (used by `derived::derive`),
/// and `Estimate::live_pinned_headroom()` (used by per-slot `grow_by_slots_gated`)
/// all agree via the single owner and preserve the 2 GiB OS/container clean-cache
/// reserve without counting swap-backed `cold_reclaimable`.
#[test]
fn pinned_host_budget_clamping_and_per_slot_guard_agree_on_64g_linux_container() {
    use crate::memory::ledger::{CommittedKind, Ledgers, LoadShape, PoolClass};
    use crate::memory::supply::Reading;
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    const TOTAL: u64 = 64 * GIB;
    const RECLAIM_BAND: u64 = 64 * MIB; // Linux 64 GiB zone (high - low)
    const HOST_SLOT: u64 = 384 * MIB;
    const HOST_FLOOR: u64 = 3 * GIB;
    const RESERVE: u64 = RECLAIM_BAND + HOST_SLOT;

    let t0 = Instant::now();
    let sample = seismic::HostSample {
        at: t0,
        page_bytes: 4096,
        total: TOTAL,
        supply: seismic::Supply {
            free_beyond_target: (36 * GIB) as i64,
            clean_cache: 4 * GIB,
            cold_reclaimable: 12 * GIB,
        },
        reclaim_band: RECLAIM_BAND,
        counted: seismic::Counted { at: t0, lag: Duration::ZERO },
        cold: seismic::ColdPages { machine: 12 * GIB, own: None },
        own_reclaimed: None,
        wired: None,
        harm: Some(seismic::Harm { machine: 0, own: 0 }),
        pressure: Some(seismic::PressureLevel::Normal),
        lag: seismic::Lag { release: Duration::ZERO, growth: Duration::ZERO },
    };

    let scope = Ledgers::new();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, Arc::new(LoadShape { stores: Vec::new() }));
    let _pool = ledger.pool_block(PoolClass::Records, NonZeroU64::new(HOST_SLOT).unwrap());
    ledger.read(ledger.sampled(t0), Reading::host(&sample));

    // Claim and allocate the 3 GiB committed host floor.
    let floor_claim = ledger
        .change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, HOST_FLOOR))
        .expect("host floor fits");
    ledger
        .change(|_| {
            let entry = floor_claim.entry();
            entry.take(&entry.change(), HOST_FLOOR, seismic::memory::Wiring::Pageable)
        })
        .expect("host floor allocates");

    // Post-floor sample: OS free dropped by HOST_FLOOR (36 -> 33 GiB).
    let t1 = t0 + Duration::from_secs(1);
    let post_floor = seismic::HostSample {
        at: t1,
        counted: seismic::Counted { at: t1, lag: Duration::ZERO },
        supply: seismic::Supply {
            free_beyond_target: (33 * GIB) as i64,
            clean_cache: 4 * GIB,
            cold_reclaimable: 12 * GIB,
        },
        ..sample
    };
    ledger.read(ledger.sampled(t1), Reading::host(&post_floor));
    let est = ledger.estimate();

    // Expected pinned room above committed floor:
    // immediate_supply (33 + 4 = 37 GiB) - reserve (448 MiB).
    // Note: 12 GiB cold_reclaimable is excluded, and derived reserve is preserved.
    let expected_room = 37 * GIB - RESERVE;
    assert_eq!(est.pinned_reserve(), RESERVE);
    assert_eq!(est.pinned_room(), expected_room);
    assert_eq!(est.pinned_import_room(), expected_room);
    assert_eq!(est.live_pinned_headroom(&post_floor), expected_room);

    // Clamped host slots and per-slot guard agree on the exact slot count.
    let host_charge = |slots: u64| HOST_FLOOR + slots * HOST_SLOT;
    let host_shape = BankShape::new(HOST_SLOT, 0, 133, 133);
    let clamped = target_slots(est.pinned_import_room(), &host_shape, &host_charge);
    assert_eq!(clamped, expected_room / HOST_SLOT);
    assert!(clamped * HOST_SLOT <= est.live_pinned_headroom(&post_floor));
    assert!((clamped + 1) * HOST_SLOT > est.live_pinned_headroom(&post_floor));
}
