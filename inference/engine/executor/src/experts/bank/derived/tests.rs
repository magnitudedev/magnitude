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

/// The block is the one threshold both ways under external pressure: a bank of
/// 132 slots in blocks of four grows once its room holds a block more, and
/// shrinks once external budget/cap pressure leaves it a block short, while
/// holding steady when `owed == 0` and `external_over == 0`.
#[test]
fn the_bank_moves_by_a_block_either_way() {
    let block = BankShape::new(4 * SLOT, 8, 226, 226);
    let room = |slots: u64| (slots - 8) * SLOT;
    let held_grow = |r| held_target(r, 132, 0, &block, &charge);
    assert_eq!(held_grow(room(132)), 132);
    assert_eq!(held_grow(room(135) + SLOT / 2), 132, "short of a block above");
    assert_eq!(held_grow(room(136)), 136, "a block above");
    assert_eq!(held_grow(room(128)), 132, "without owed or external pressure, holds steady");
    let held_pressure = |r| held_target_with_pressure(r, r, 132, 0, 4 * SLOT, &block, &charge);
    assert_eq!(held_pressure(room(129) + 1), 132, "short of a block below");
    assert_eq!(held_pressure(room(128)), 128, "a block below under pressure: to what the room fits");
    assert_eq!(held_pressure(room(120) + SLOT / 3), 120);
    assert_eq!(bank_move(132, 132), None);
    assert_eq!(bank_move(128, 132), Some(BankMove::Shrink { to: 128 }));
    assert_eq!(bank_move(136, 132), Some(BankMove::Grow { to: 136 }));
}

/// The laptop's bank in blocks of one slot: others' jitter of a third of a
/// slot for a reading lowers the room by less than a block, so the bank
/// holds; under real external pressure of a full block, it sheds one slot.
#[test]
fn a_room_moving_within_a_block_resizes_nothing() {
    let room = (132 - 8) * SLOT;
    for jitter in [SLOT / 3, SLOT - 1, 0] {
        assert_eq!(bank_move(held_target_with_pressure(room - jitter, room - jitter, 132, 0, jitter, &shape(), &charge), 132), None, "{jitter} bytes of others' rise");
    }
    assert_eq!(held_target_with_pressure(room - SLOT, room - SLOT, 132, 0, SLOT, &shape(), &charge), 131, "a whole block's external pressure");
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
/// and shrinks cleanly under external pressure — never capped by a shallow
/// intermediate slot count.
#[test]
fn the_bank_grows_to_its_ceiling_and_shrinks_to_room() {
    let room = |slots: u64| (slots - 8) * SLOT;
    let full = BankShape::new(SLOT, 8, 226, 226);
    assert_eq!(held_target(room(200), 126, 0, &full, &charge), 200, "grows into newly available room");
    assert_eq!(held_target(room(300), 126, 0, &full, &charge), 226, "grows up to ceiling");
    assert_eq!(held_target_with_pressure(room(100), room(100), 200, 0, 100 * SLOT, &full, &charge), 100, "shrinks to tighter room under pressure");
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
/// through two-thirds-block app jitter; an actual 1-slot external drop sheds
/// 1 slot, not 2.
#[test]
fn asymmetric_hold_room_stops_guarantee_derivative_kick_oscillation() {
    let base = (132 - 8) * SLOT;
    let jitter = SLOT * 2 / 3;
    let hold_room = base - jitter;
    let grow_room = base - 2 * jitter;
    assert_eq!(held_target_with_pressure(grow_room, grow_room, 132, 0, 2 * jitter, &shape(), &charge), 130, "symmetric target over-shrinks by 2x");
    assert_eq!(held_target_with_pressure(hold_room, grow_room, 132, 0, jitter, &shape(), &charge), 132, "hold_room absorbs single-block jitter");
    assert_eq!(held_target_with_pressure(base - SLOT, base - 2 * SLOT, 132, 0, SLOT, &shape(), &charge), 131, "actual 1-slot drop sheds 1 slot, not 2");
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

/// Reproduces F-047 on `cliqa-r3-cuda` (30.97 GiB physical RAM, 0 swap,
/// 2.49 GiB CMA, `qwen3.8-flash-next:gguf:q2` with 337 planned host slots at
/// 67.33 MB/slot = 22.69 GB): once Linux discovery excludes CMA and reserves
/// the swapless host floor (`2.58 GiB`), both `clamped_host_slots` and per-slot
/// `live_pinned_headroom` bound the host tier strictly below 337 slots and stop
/// before non-CMA RAM is exhausted.
#[test]
fn tight_ram_swapless_cma_host_clamps_and_guards_host_tier_below_oom() {
    use crate::memory::ledger::{Ledgers, LoadShape, PoolClass};
    use crate::memory::supply::Reading;
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    const MIB: u64 = 1 << 20;
    const TOTAL: u64 = 33_258_901_504; // 30.97 GiB
    const RECLAIM_BAND: u64 = 30_547_968; // 29.13 MiB zone (high - low)
    const HOST_SLOT: u64 = 67_326_669; // 22,689,087,488 / 337
    const POOL_BLOCK: u64 = 64 * MIB;
    const PLANNED_SLOTS: u64 = 337;
    const LANDING_FLOOR: u64 = 197_132_288;
    const RESERVE: u64 = RECLAIM_BAND + POOL_BLOCK;

    let scope = Ledgers::new();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, Arc::new(LoadShape { stores: Vec::new() }));
    let _pool = ledger.pool_block(PoolClass::Records, NonZeroU64::new(POOL_BLOCK).unwrap());

    // Post-warmup on cliqa-r3-cuda AFTER excluding 2.49 GiB CMA and reserving
    // 2.58 GiB swapless host floor (instead of the old 24.96 GB raw supply):
    // immediate non-CMA supply above swapless floor is ~19.68 GB (~18.33 GiB).
    let t0 = Instant::now();
    let post_warmup = seismic::HostSample {
        at: t0,
        page_bytes: 4096,
        total: TOTAL,
        supply: seismic::Supply {
            free_beyond_target: -2_000_000_000, // non-CMA free below 2.58 GiB swapless floor
            clean_cache: 21_680_000_000,        // non-CMA clean file cache
            cold_reclaimable: 0,
        },
        reclaim_band: RECLAIM_BAND,
        counted: seismic::Counted { at: t0, lag: Duration::ZERO },
        cold: seismic::ColdPages { machine: 0, own: None },
        own_reclaimed: None,
        wired: None,
        harm: None,
        pressure: Some(seismic::PressureLevel::Normal),
        lag: seismic::Lag { release: Duration::ZERO, growth: Duration::ZERO },
    };
    ledger.read(ledger.sampled(t0), Reading::host(&post_warmup));
    let landing = ledger
        .change(|keeping| {
            ledger.claim_at(
                &keeping.claims(),
                None,
                crate::memory::ledger::CommittedKind::Load,
                LANDING_FLOOR,
            )
        })
        .expect("landing fits");
    ledger
        .change(|_| {
            let entry = landing.entry();
            entry.take(&entry.change(), LANDING_FLOOR, seismic::memory::Wiring::Pageable)
        })
        .expect("landing allocates");

    let est = ledger.estimate();
    let host_charge = |slots: u64| LANDING_FLOOR + slots * HOST_SLOT;
    let host_shape = BankShape::new(POOL_BLOCK, 0, PLANNED_SLOTS, PLANNED_SLOTS);
    let clamped = target_slots(est.pinned_import_room(), &host_shape, &host_charge);

    assert!(
        clamped < PLANNED_SLOTS,
        "30.97 GiB swapless CMA host must clamp below all 337 planned slots (got {clamped})"
    );
    assert!(
        clamped >= 250,
        "clamped host tier should still utilize available non-CMA RAM (~280-290 slots, got {clamped})"
    );
    assert!(
        clamped * HOST_SLOT + RESERVE <= (21_680_000_000 - 2_000_000_000),
        "clamped host tier must leave swapless floor + pinned_reserve intact"
    );

    // Once per-slot backing drains clean_cache down to the swapless floor,
    // live_pinned_headroom drops to 0 and halts growth before OOM.
    let near_floor = seismic::HostSample {
        at: t0 + Duration::from_secs(10),
        supply: seismic::Supply {
            free_beyond_target: -2_500_000_000,
            clean_cache: 2_500_000_000 + RESERVE - 1,
            cold_reclaimable: 0,
        },
        ..post_warmup
    };
    assert_eq!(est.live_pinned_headroom(&near_floor), 0);
}

/// Decisive paired safety test (zero magic deadbands):
/// (a) Jittering `OtherCache` (e.g. `178 <-> 746 MiB` on macOS UMA DeepSeek
///     `Table 0`, or `±350 MiB` on `319 MiB` GLM slots) with `owed == 0` and
///     `held <= budget` produces **zero** releases on both `Experts` and `Table`.
/// (b) With `owed == 0`, a falling GPU wired grant / budget below `held`
///     fires physical relief cleanly, and when an overlapping in-process claim
///     (`owed > 0`) arrives mid-shrink, each physical release is spent once
///     (never double-released).
#[test]
fn other_cache_jitter_never_shrinks_while_falling_grant_relieves_without_double_shed() {
    use crate::memory::ledger::{CacheKind, CommittedKind, DeviceGrant, DeviceLedgers, Ledgers, LoadShape, PoolClass, Sample};
    use crate::memory::supply::Reading;
    use std::num::NonZeroU64;
    use std::time::{Duration, Instant};

    const MIB: u64 = 1024 * 1024;
    const POOL_BLOCK: u64 = 16 * MIB;
    const DS_SLOT: u64 = 352 * MIB;
    const TABLE_BLOCK: u64 = 10 * MIB;
    const TOTAL: u64 = 48 * 1024 * MIB;

    let now = seismic::Lag { release: Duration::ZERO, growth: Duration::ZERO };
    let scope = Ledgers::new();
    let start = Instant::now();
    let init_sample = seismic::HostSample {
        at: start,
        page_bytes: 16_384,
        total: TOTAL,
        supply: seismic::Supply { free_beyond_target: (40 * 1024 * MIB) as i64, clean_cache: 0, cold_reclaimable: 0 },
        reclaim_band: 0,
        counted: seismic::Counted { at: start, lag: Duration::ZERO },
        cold: seismic::ColdPages { machine: 0, own: None },
        own_reclaimed: None,
        wired: None,
        harm: None,
        pressure: Some(seismic::PressureLevel::Normal),
        lag: now,
    };
    let (ledgers, _init) = DeviceLedgers::unified_in(&scope, std::sync::Arc::new(LoadShape { stores: Vec::new() }))
        .read_granted(&Sample::scripted(scope.changes(), init_sample), DeviceGrant::Host);
    let ledger = ledgers.device().clone();
    let _records_pool = ledger.pool_block(PoolClass::Records, NonZeroU64::new(POOL_BLOCK).unwrap()).blocks;
    let _state_pool = ledger.pool_block(PoolClass::State, NonZeroU64::new(POOL_BLOCK).unwrap()).blocks;

    let mut tick = 1u32;
    let mut read_host_supply = |free_beyond_target: i64| {
        tick += 1;
        let t = start + Duration::from_millis(100) * tick;
        let sample = seismic::HostSample {
            at: t,
            supply: seismic::Supply { free_beyond_target, clean_cache: 0, cold_reclaimable: 0 },
            counted: seismic::Counted { at: t, lag: Duration::ZERO },
            ..init_sample
        };
        ledger.read(ledger.sampled(t), Reading::host(&sample));
    };

    // Start with plenty of headroom: 40 GiB available.
    read_host_supply((40 * 1024 * MIB) as i64);
    let experts_at = 58u64;
    let table_at = 23u64;
    let experts_floor = 10u64;
    let table_floor = 1u64;
    let experts_above = (experts_at - experts_floor) * DS_SLOT;
    let table_above = (table_at - table_floor) * TABLE_BLOCK;
    let floor_bytes = experts_floor * DS_SLOT + table_floor * TABLE_BLOCK;

    let floor_claim = ledger
        .change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, floor_bytes))
        .expect("floor fits");
    ledger
        .change(|_| {
            let e = floor_claim.entry();
            e.take(&e.change(), floor_bytes, seismic::memory::Wiring::Pageable)
        })
        .expect("floor allocates");
    let _experts_cache = ledger.cached(CacheKind::Bank, experts_above).expect("experts fit");
    let _table_cache = ledger.cached(CacheKind::Table, table_above).expect("table fits");

    let make_targets = |owed_experts: u64, owed_table: u64| -> Vec<HoldingTarget<'static>> {
        vec![
            HoldingTarget {
                holding: BankHolding::Experts,
                tier: BankTier::Bank,
                at: experts_at,
                owed: owed_experts,
                shape: BankShape::new(POOL_BLOCK, experts_floor, 128, 128),
                charge: Box::new(|slots| slots * DS_SLOT),
            },
            HoldingTarget {
                holding: BankHolding::Table { table: 0 },
                tier: BankTier::Bank,
                at: table_at,
                owed: owed_table,
                shape: BankShape::new(TABLE_BLOCK, table_floor, 23, 23),
                charge: Box::new(|records| records * TABLE_BLOCK),
            },
        ]
    };

    // (a) Jitter OtherCache (`CacheKind::Prefix`) between 178 MiB and 746 MiB
    // while `held <= budget` (`supply == 0`, zero external pressure, zero
    // extra room to grow) and `owed == 0`: ZERO releases on Experts or Table 0.
    for round in 0..6u32 {
        let other_bytes = if round % 2 == 0 { 178 * MIB } else { 746 * MIB };
        // Two readings at 2 GiB settle `guarantee == 0` so `other` is admitted,
        // then `supply = 0` keeps `budget == in_use == held`.
        read_host_supply((2 * 1024 * MIB) as i64);
        read_host_supply((2 * 1024 * MIB) as i64);
        let other = ledger.cached(CacheKind::Prefix, other_bytes).expect("other cache fits");
        read_host_supply(0);
        let targets = make_targets(0, 0);
        let derived = derive(&ledgers, &targets);
        assert_eq!(derived[0].1, experts_at, "OtherCache jitter must not shrink Experts on round {round}");
        assert_eq!(derived[1].1, table_at, "OtherCache jitter must not shrink Table 0 on round {round}");
        drop(other);
    }

    // (b) Falling budget/grant below `held` with `owed == 0`: physical relief MUST fire!
    // Drop host supply by 2 GiB below `in_use` so `budget = held - 2 GiB < held`.
    let short_by = 2 * 1024 * MIB;
    read_host_supply(-(short_by as i64));
    assert!(
        ledger.estimate().held() > ledger.estimate().budget(),
        "test setup: held ({}) must exceed budget ({})",
        ledger.estimate().held(),
        ledger.estimate().budget()
    );
    let targets_pressure = make_targets(0, 0);
    let derived_pressure = derive(&ledgers, &targets_pressure);
    let shrunk_experts = derived_pressure[0].1;
    assert!(
        shrunk_experts < experts_at,
        "falling budget below held with owed == 0 must fire physical relief (got {shrunk_experts} vs {experts_at})"
    );

    // Now an overlapping in-process claim (`owed = 1 slot = 352 MiB`) arrives
    // mid-shrink while the 2 GiB budget deficit is still active: the physical
    // release is spent ONCE (`held.min(sheds)`), never subtracting 1 slot on
    // top of the already-deeper 2 GiB pressure shrink.
    let targets_overlap = make_targets(DS_SLOT, 0);
    let derived_overlap = derive(&ledgers, &targets_overlap);
    assert_eq!(
        derived_overlap[0].1, shrunk_experts,
        "overlapping claim smaller than active pressure shrink must not double-release"
    );
}
