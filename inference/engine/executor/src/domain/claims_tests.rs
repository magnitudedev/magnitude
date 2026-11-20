//! Committed claims: what a load takes, and what it never sets aside.

use super::super::fixtures::{bank_backends, banked_chain, NO_LAG};
use crate::memory::supply::Reading;

/// A load holds no prompt group's class: its groups hold one place until
/// one opens with chunks behind it (memory-failures items 7, 27, 50),
/// whatever pass its plan holds.
#[test]
fn a_load_holds_one_place_until_a_group_opens() {
    for backend in bank_backends() {
        let (domain, _bindings) = banked_chain(backend, 4, 3);
        assert_eq!(domain.group_places().get(), 1, "{backend:?}: no class is held as the load starts");
        assert!(committed_of(&domain, crate::memory::ledger::CommittedKind::GroupClass).is_empty(), "{backend:?}");
    }
}

/// The committed entries of `domain`'s device ledger of `kind`.
fn committed_of<F: crate::domain::ProgramFamily>(
    domain: &crate::domain::ExecutorDomain<F>,
    kind: crate::memory::ledger::CommittedKind,
) -> Vec<u64> {
    use crate::memory::ledger::Holding;
    domain
        .resources
        .ledgers()
        .device()
        .entries()
        .into_iter()
        .filter(|entry| entry.holding == Holding::Committed(kind))
        .map(|entry| entry.bytes)
        .collect()
}

/// A load commits its arena on the device's ledger as it starts, before
/// any prompt or bank growth (memory-failures items 22, 33, 50): one claim
/// holding what its graph scratch charged, so a bank sized from the ledger
/// afterwards can never take the room places run in.
#[test]
fn a_load_commits_its_arena_before_any_prompt() {
    for backend in bank_backends() {
        let (domain, _bindings) = banked_chain(backend, 4, 3);
        let arena = committed_of(&domain, crate::memory::ledger::CommittedKind::Arena);
        let (claim, allocated, _) = domain.arena_terms();
        assert!(
            matches!(arena.as_slice(), [bytes] if *bytes > 0 && *bytes == claim) && claim == allocated,
            "{backend:?}: the arena's claims {arena:?}, {allocated} bytes of them allocated"
        );
    }
}

/// The `position`th one-row prompt chunk of request 90.
fn chunk(position: usize) -> crate::Operation {
    crate::Operation::Forward {
        request: crate::RequestId(90),
        kind: crate::WorkKind::Prefill,
        tokens: vec![crate::TokenId(1)],
        position,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        front: Vec::new(),
        chain: None,
        committed: 1,
        prime: None,
    }
}

/// A group opening with `planned` chunks planned behind its first: the
/// places its class holds, and the class's committed claims on the
/// device's ledger.
fn opened<F: crate::domain::ProgramFamily>(domain: &mut crate::domain::ExecutorDomain<F>, planned: usize) -> (usize, Vec<u64>) {
    domain.plan_successors((1..=planned).map(chunk).collect());
    domain.open_group(&[chunk(0)]).unwrap();
    (domain.group_places().get(), committed_of(domain, crate::memory::ledger::CommittedKind::GroupClass))
}

use crate::memory::ledger::{CacheKind, Holding};

/// A group opening where room allows claims its class as it opens (design
/// §2: an optional committed claim, never reserved ahead): a place for
/// each of its chunks, whatever their rows, at most the three its pass
/// holds, committed on the ledger; and its close frees its places, its
/// claim turning into the idle slots' cache, which the next group's class
/// takes up again.
#[test]
fn a_group_claims_its_places_as_it_opens_and_frees_them_as_it_closes() {
    for backend in bank_backends() {
        let (mut domain, _bindings) = banked_chain(backend, 4, 3);
        for (planned, places) in [(0, 1), (1, 2), (2, 3), (5, 3)] {
            let (held, claims) = opened(&mut domain, planned);
            assert_eq!(held, places, "{backend:?}: {planned} chunks behind the first");
            assert_eq!(!claims.is_empty(), places > 1, "{backend:?}: {planned} chunks behind: {claims:?}");
            assert!(claims.iter().all(|bytes| *bytes > 0), "{backend:?}: {claims:?}");
            domain.close_group().unwrap();
            assert_eq!(domain.group_places().get(), 1, "{backend:?}: the close frees the group's places");
            assert!(committed_of(&domain, crate::memory::ledger::CommittedKind::GroupClass).is_empty(), "{backend:?}: and its claim");
            // What the pools keep idles as cache until they release it; the
            // group's buffers went back at the close.
            let buffers = crate::domain::ProgramFamily::group_demand(&domain.family, places).unwrap().checked_bytes().unwrap();
            let expected = match places > 1 {
                true => claims.iter().sum::<u64>() - buffers,
                false => 0,
            };
            assert_eq!(idle_cache(&domain), expected, "{backend:?}: the class's idle slots as cache");
        }
    }
}

/// The idle graph slots' cache on `domain`'s device ledger.
fn idle_cache<F: crate::domain::ProgramFamily>(domain: &crate::domain::ExecutorDomain<F>) -> u64 {
    domain
        .resources
        .ledgers()
        .device()
        .entries()
        .into_iter()
        .filter(|entry| entry.holding == Holding::Cache(CacheKind::GraphSlots))
        .map(|entry| entry.bytes)
        .sum()
}

/// A closed group's idle slots a claim takes cache of in process are
/// released as the load frees what the ledger took (`release_yielded`):
/// their release begins, and once it ends their cache is gone.
#[test]
fn idle_slots_a_claim_takes_are_released_with_what_yielded() {
    use crate::memory::ledger::CommittedKind;
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked_chain(backend, 4, 3);
        let (places, _) = opened(&mut domain, 2);
        assert_eq!(places, 3, "{backend:?}");
        domain.close_group().unwrap();
        assert!(idle_cache(&domain) > 0, "{backend:?}: the closed class idles as cache");
        assert!(!domain.idle_class.taken(), "{backend:?}");
        // A claim takes every byte that yields, the idle slots' cache among
        // it (it yields before the bank's).
        let ledger = domain.resources.ledgers().device().clone();
        let (_all, _) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Scratch, std::num::NonZeroU64::new(1).unwrap(), u64::MAX));
        assert!(domain.idle_class.taken(), "{backend:?}: the claim took the idle slots' cache");
        assert_eq!(
            domain.release_yielded(&mut bindings, crate::CompletionWake::new(|| {})).unwrap(),
            crate::domain::Yielded::Pending,
            "{backend:?}: the slots' release begins"
        );
        while domain.advance_slot_release().is_none() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(idle_cache(&domain), 0, "{backend:?}: the released slots' cache is gone");
        assert!(!domain.idle_class.taken(), "{backend:?}");
    }
}

/// A claim that took the cache of a closed group's idle slots, all of
/// which a release let go meanwhile: where the backend still holds memory
/// of the cache's entry (the CPU's idle page blocks, which return as their
/// holder goes), the next release of what yielded lets the cache go as a
/// change of its own, under the process's rule, which a record outside one
/// breaks; where the slots' free returned every byte they held (each Metal
/// buffer freed whole), the cache holds nothing and nothing is taken.
#[test]
fn a_taken_idle_cache_with_no_slot_left_goes_in_a_change() {
    use crate::memory::ledger::CommittedKind;
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked_chain(backend, 4, 3);
        let (places, _) = opened(&mut domain, 2);
        assert_eq!(places, 3, "{backend:?}");
        domain.close_group().unwrap();
        let ledger = domain.resources.ledgers().device().clone();
        let (all, _) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Scratch, std::num::NonZeroU64::new(1).unwrap(), u64::MAX));
        assert!(domain.idle_class.taken(), "{backend:?}: the claim took the idle slots' cache");
        // The slots are released apart from their cache, which stays taken.
        assert!(domain.resources.release_idle_slots(Vec::new(), crate::CompletionWake::new(|| {})), "{backend:?}");
        while domain.advance_slot_release().is_none() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let lent = domain.idle_class.taken();
        assert!(lent || idle_cache(&domain) == 0, "{backend:?}: no slot idles, and its cache lends nothing and holds nothing");
        let changes = domain.resources.ledgers().changes().clone();
        changes.strict();
        let expected = match lent {
            true => crate::domain::Yielded::Released { released: 0, returned: 0 },
            false => crate::domain::Yielded::Nothing,
        };
        assert_eq!(
            domain.release_yielded(&mut bindings, crate::CompletionWake::new(|| {})).unwrap(),
            expected,
            "{backend:?}: nothing left to release, a cache still lending is let go at once"
        );
        assert_eq!(idle_cache(&domain), 0, "{backend:?}: the cache is let go");
        changes.change(|_| drop(all));
    }
}

/// A group opening while a closed group's idle slots are being released
/// claims its whole class: the released slots are not its, their cache
/// goes with their release, allocated until it ends, and none of it is
/// counted twice.
#[test]
fn a_group_opening_during_a_slot_release_claims_its_whole_class() {
    use crate::memory::ledger::CommittedKind;
    let graph_slots = |domain: &crate::domain::ExecutorDomain<_>| -> u64 {
        domain.resources.ledgers().device().entries().into_iter().filter(|entry| entry.holding == Holding::Cache(CacheKind::GraphSlots)).map(|entry| entry.releasing).sum()
    };
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked_chain(backend, 4, 3);
        let (places, first) = opened(&mut domain, 2);
        assert_eq!(places, 3, "{backend:?}");
        domain.close_group().unwrap();
        let idle = idle_cache(&domain);
        assert!(idle > 0, "{backend:?}");
        let ledger = domain.resources.ledgers().device().clone();
        let (all, _) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Scratch, std::num::NonZeroU64::new(1).unwrap(), u64::MAX));
        assert_eq!(
            domain.release_yielded(&mut bindings, crate::CompletionWake::new(|| {})).unwrap(),
            crate::domain::Yielded::Pending,
            "{backend:?}"
        );
        assert_eq!(graph_slots(&domain), idle, "{backend:?}: the taken slots stay allocated while they release");
        drop(all);
        let (places, claims) = opened(&mut domain, 2);
        assert_eq!(places, 3, "{backend:?}");
        assert_eq!(claims, first, "{backend:?}: the class claims every slot it grows again");
        while domain.advance_slot_release().is_none() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(graph_slots(&domain), 0, "{backend:?}: freed with the release");
    }
}

/// A group whose class takes a closed group's idle slots up again and
/// grows past them, where free room is short of that growth: its claim
/// yields cache, and the idle slots, worth nothing, would yield first. They
/// are the class's own: committed before its growth is claimed, the claim
/// never takes them (it panicked as the class turned them committed, "idle
/// slots' cache taken in process as their class opens"). The group runs
/// with what yields beside them, and the slots stay whole, idle again where
/// it gets nothing.
#[test]
fn a_class_opening_short_of_room_never_takes_its_own_idle_slots() {
    use crate::memory::ledger::PoolClass;
    for backend in bank_backends() {
        let (mut domain, _bindings) = banked_chain(backend, 4, 3);
        let (places, _) = opened(&mut domain, 1);
        assert_eq!(places, 2, "{backend:?}");
        domain.close_group().unwrap();
        let idle = idle_cache(&domain);
        assert!(idle > 0, "{backend:?}: the closed class idles as cache");
        // A class of three places grows a place's slots past the idle two.
        let device = domain.domain.device().clone();
        let demand = super::class_demand(&domain.execution, &mut domain.resources, &domain.family, &device, 3).unwrap();
        let (grows, _) = domain.resources.ledgers().placed(&device, demand);
        assert!(grows > 0, "{backend:?}: a third place grows slots");
        // Another holding takes every byte of free room, none of the cache:
        // the most the ledger backs from free room alone.
        let ledger = domain.resources.ledgers().device().clone();
        let backs = |bytes| ledger.change(|keeping| ledger.backing(&keeping.claims(), PoolClass::Records, bytes)).is_ok();
        let (mut low, mut high) = (0u64, 1u64 << 50);
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if backs(mid) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        let room = ledger.change(|keeping| ledger.backing(&keeping.claims(), PoolClass::Records, low)).unwrap();
        // And the spare granules, which yield before the idle slots: the
        // growth's claim then yields the idle slots, worth nothing, before
        // the bank.
        let spare = ledger.estimate().cache_of(CacheKind::Spare);
        let (granules, _) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), crate::memory::ledger::CommittedKind::Scratch, std::num::NonZeroU64::MIN, spare));
        assert_eq!(idle_cache(&domain), idle, "{backend:?}: the room and spare granules alone were taken");
        let (places, claims) = opened(&mut domain, 2);
        assert!(!domain.idle_class.taken(), "{backend:?}: the class's claim took none of its own slots");
        match places {
            1 => assert_eq!(idle_cache(&domain), idle, "{backend:?}: unopened, the slots idle whole"),
            _ => assert!(idle_cache(&domain) == 0 && !claims.is_empty(), "{backend:?}: opened, the class holds its slots: {claims:?}"),
        }
        drop((room, granules));
    }
}

/// A group opening where nothing yields runs with one place: its class is
/// optional, never preempting a sequence and never refused.
#[test]
fn a_group_opens_with_one_place_where_nothing_yields() {
    use crate::memory::ledger::CommittedKind;
    for backend in bank_backends() {
        let (mut domain, _bindings) = banked_chain(backend, 4, 3);
        let ledger = domain.resources.ledgers().device().clone();
        // Another holding takes every byte that yields at once.
        let (_all, _) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Scratch, std::num::NonZeroU64::new(1).unwrap(), u64::MAX));
        let (held, claims) = opened(&mut domain, 2);
        assert_eq!(held, 1, "{backend:?}: no room, one place");
        assert!(claims.is_empty(), "{backend:?}: {claims:?}");
    }
}

/// A drafted sequence's state claim reaches its position, the rows the
/// group advances it and its drafter's draft depth, the most speculative
/// rows its head adds; an undrafted one's, its position and rows.
#[test]
fn a_drafted_sequences_claim_covers_its_draft_depth() {
    use crate::PlannedMethod;
    let mtp = PlannedMethod::Mtp { greedy_proposals: 3, sampled_proposals: 2, widths: crate::DraftWidths::Adaptive };
    assert_eq!(mtp.draft_rows(), 3);
    assert_eq!(super::claimed_through(40, 5, mtp), 40 + 5 + 3);
    assert_eq!(super::claimed_through(40, 5, PlannedMethod::Plain), 40 + 5);
}

/// A drafted domain's provision claims each sequence's state through its
/// position, the rows the group advances it and the draft depth, sized
/// over the history domains of both its stores: an MTP head's own beside
/// the target's.
#[test]
fn a_drafted_domains_provision_claims_through_its_draft_depth_in_both_stores() {
    use super::super::fixtures::{fixture_over, limits, tiny_definition, tiny_manifest};
    use crate::memory::ledger::LoadShape;
    let definition = crate::planning::fixtures::with_mtp_head(tiny_definition());
    let manifest = tiny_manifest(&definition);
    let method = crate::PlannedMethod::Mtp { greedy_proposals: 2, sampled_proposals: 1, widths: crate::DraftWidths::Adaptive };
    let (mut domain, mut bindings) =
        fixture_over(definition, manifest, limits(2, 1, false), method, None, false, crate::platform::DeviceRequest::Automatic);
    let request = crate::RequestId(7);
    domain.install_input(request, magnitude_family_contracts::PreparedModelInput::continuation_only()).unwrap();
    assert!(domain.open_served(&mut bindings, request, None).unwrap().is_empty());
    let rows = 2;
    let prefill = crate::Operation::Forward {
        request,
        kind: crate::WorkKind::Prefill,
        tokens: vec![crate::TokenId(1); rows],
        position: 0,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        front: Vec::new(),
        chain: None,
        committed: rows,
        prime: None,
    };
    domain.provision(&mut bindings, &[prefill], None).unwrap();
    let through = (rows + method.draft_rows()) as u64;
    let state = &domain.state_claims[&request];
    assert_eq!(state.through(), through, "the claim reaches the rows and the draft depth");
    let shape = LoadShape::of(&domain.execution);
    let resources = domain.execution.resources();
    let head = resources.head_state().expect("an MTP head keeps its own history");
    let domains = shape.stores.iter().map(|store| store.domains.len()).collect::<Vec<_>>();
    assert_eq!(domains, [resources.target_state().history.len(), head.history.len()], "both stores' domains");
    let context = resources.target_state().context_rows as u64;
    assert_eq!(state.rows(), context + method.draft_rows() as u64, "the served open reserved the whole context, draft depth and all");
    assert_eq!(state.bytes(), shape.state_bytes(state.rows()), "the reservation holds both stores' pages through its rows");
}

/// A reservation draws on what the request slot holds first and gives
/// back exactly what it drew as it closes; one opened beside it draws what
/// is left, and the rest from free room.
#[test]
fn a_reservation_draws_the_slot_first_and_gives_back_what_it_drew() {
    use super::super::fixtures::{fixture_over, limits, tiny_definition, tiny_manifest};
    use crate::memory::ledger::CommittedKind;
    let (mut domain, mut bindings) = fixture_over(
        tiny_definition(),
        tiny_manifest(&tiny_definition()),
        limits(2, 1, false),
        crate::PlannedMethod::Plain,
        None,
        false,
        crate::platform::DeviceRequest::Automatic,
    );
    let (first, second) = (crate::RequestId(1), crate::RequestId(2));
    let [slot] = committed_of(&domain, CommittedKind::RequestSlot)[..] else { panic!("one request slot") };
    for request in [first, second] {
        domain.install_input(request, magnitude_family_contracts::PreparedModelInput::continuation_only()).unwrap();
        assert!(domain.open_served(&mut bindings, request, None).unwrap().is_empty());
    }
    let reserved = domain.state_claims[&first].bytes();
    assert_eq!(domain.state_claims[&second].bytes(), reserved, "both reserved their whole requests");
    let drawn = slot.min(reserved);
    let left = slot - drawn;
    assert_eq!(committed_of(&domain, CommittedKind::RequestSlot), [left - left.min(reserved)], "each drew what the slot held, the first first");
    domain.close(second).unwrap();
    assert_eq!(committed_of(&domain, CommittedKind::RequestSlot), [left], "the second gave back what it drew");
    domain.close(first).unwrap();
    assert_eq!(committed_of(&domain, CommittedKind::RequestSlot), [slot], "the first gave back the rest");
}

/// An open refused for room holds nothing and yields nothing: the retained
/// prefixes' cache is the same after it, and it opens once room frees.
#[test]
fn an_open_refused_short_keeps_the_prefix_cache() {
    use super::super::fixtures::{fixture_over, limits, tiny_definition, tiny_manifest};
    use crate::memory::ledger::{CacheKind, CommittedKind};
    let (mut domain, mut bindings) = fixture_over(
        tiny_definition(),
        tiny_manifest(&tiny_definition()),
        limits(2, 1, false),
        crate::PlannedMethod::Plain,
        None,
        false,
        crate::platform::DeviceRequest::Automatic,
    );
    let (holder, refused) = (crate::RequestId(1), crate::RequestId(2));
    for request in [holder, refused] {
        domain.install_input(request, magnitude_family_contracts::PreparedModelInput::continuation_only()).unwrap();
    }
    assert!(domain.open_served(&mut bindings, holder, None).unwrap().is_empty());
    let ledger = domain.resources.ledgers().device().clone();
    let _prefix = ledger.cache(CacheKind::Prefix, 4096, crate::memory::ledger::Reclaim::Deferred).expect("the prefix fits");
    let (room, _) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Scratch, std::num::NonZeroU64::new(1).unwrap(), u64::MAX));
    assert!(
        matches!(domain.open_served(&mut bindings, refused, None), Err(crate::domain::DomainError::Capacity(_))),
        "nothing is left for a second request"
    );
    assert!(!domain.state_claims.contains_key(&refused), "a refused open holds nothing");
    assert_eq!(ledger.estimate().cache_of(CacheKind::Prefix), 4096, "a refused open yields no prefix");
    drop(room);
    assert!(domain.open_served(&mut bindings, refused, None).unwrap().is_empty(), "it opens once room frees");
}

/// A component's import, made under its claim and marked allocated, is the
/// process's memory at the next reading, never others': a 1 GB head import
/// with the bank at its floor and free room under 1 GB raises no guarantee
/// and unloads nothing.
#[test]
fn an_imports_bytes_are_never_others_memory() {
    use crate::memory::{ClaimDemand, ledger::CommittedKind};
    const GB: u64 = 1 << 30;
    for backend in bank_backends() {
        // The bank's one committed row is its floor: it holds no cache.
        let (domain, _bindings) = banked_chain(backend, 4, 1);
        let ledger = domain.resources.ledgers().device();
        let start = std::time::Instant::now();
        let others = GB;
        let total = ledger.estimate().committed() + ledger.estimate().cache() + GB + GB / 2 + others;
        let read = |tick: u32| {
            let process = ledger.estimate().allocated();
            let at = start + std::time::Duration::from_millis(100) * tick;
            ledger.read(ledger.sampled(at), Reading::device(total, total - others - process, NO_LAG));
        };
        read(1);
        read(2);
        assert_eq!(ledger.estimate().cache(), ledger.estimate().cache_of(CacheKind::Spare), "{backend:?}: the bank is at its floor");
        let guarantee = ledger.estimate().guarantee();
        let changes = domain.resources.ledgers().changes().clone();
        let import = changes
            .change(|keeping| domain.claim_growth(&keeping.claims(), CommittedKind::Component, ClaimDemand { device: GB, ..Default::default() }))
            .expect("the import fits");
        import.made();
        read(3);
        let after = ledger.estimate();
        assert_eq!(after.others(), others, "{backend:?}: the import read as others' memory");
        assert!(after.free() < GB, "{backend:?}: free room is under the import");
        assert_eq!(after.guarantee(), guarantee, "{backend:?}: the import raised the guarantee");
        assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none_or(|shortage| !shortage.response.unload), "{backend:?}: the import unloads the model");
        drop(import);
    }
}

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// A store's ledger, read at 64 GiB total and `available`, `at` ms after
/// `epoch`, one instant fixed per test: the readings' spacing is the
/// scripted one, whatever the host's load does between them.
fn read_at(ledger: &crate::memory::ledger::Ledger, epoch: std::time::Instant, available: u64, at: u64) {
    ledger.read(ledger.sampled(epoch + std::time::Duration::from_millis(at)), Reading::device(64 * GIB, available, NO_LAG));
}

/// A load's ledger with 1 GiB free beside the bank at its floor, a
/// request of more than 800 MiB reserved from its slot, and the stores'
/// slabs as the load started: the slot and the reservation, the floor's
/// claim with the records pool's entry that allocates its granules and
/// the state pool's entry, and the slabs.
fn grown_rows() -> (
    crate::memory::ledger::Ledgers,
    crate::memory::ledger::Ledger,
    (crate::memory::ledger::Claim, std::sync::Arc<dyn seismic::memory::Entry>, std::sync::Arc<dyn seismic::memory::Entry>),
    (crate::memory::ledger::Claim, crate::memory::ledger::Reservation),
    super::StateSlabs,
    std::time::Instant,
) {
    use crate::memory::ledger::{LoadShape, Need};
    let component = magnitude_state::ComponentDescriptor::new(
        magnitude_state::LayerRef::Target(0),
        magnitude_state::CodecSpec::dense(seismic::DType::F32, 1024, 1024),
        1,
    )
    .expect("a component");
    let domains = vec![magnitude_state::HistoryDomainLayout::Token { components: vec![component] }];
    let shape = std::sync::Arc::new(LoadShape { stores: vec![crate::memory::ledger::StoreShape::paged(domains, 1 << 20, 16)] });
    // Its changes are stamped at the epoch, before every scripted reading
    // after the first: on the process's clock, a change made more than a
    // reading's offset after the epoch (a loaded host) postdates that
    // reading, which then counts the change's bytes as not yet read.
    let epoch = std::time::Instant::now();
    let scope = crate::memory::ledger::Ledgers::scripted(move || epoch);
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, shape);
    read_at(&ledger, epoch, 2 * GIB, 0);
    let records = ledger.pool_block(crate::memory::ledger::PoolClass::Records, std::num::NonZeroU64::new(GIB).expect("a block")).blocks;
    let pool = ledger.pool_block(crate::memory::ledger::PoolClass::State, std::num::NonZeroU64::new(GIB).expect("a block")).blocks;
    let floor = ledger.change(|keeping| ledger.claim(&keeping.claims(), Need::BankFloor { bytes: GIB })).expect("the floor fits");
    ledger.change(|_| { let entry = &records; entry.take(&entry.change(), GIB, seismic::memory::Wiring::Pageable) }).expect("the pool backs the floor");
    read_at(&ledger, epoch, GIB, 1);
    assert_eq!(ledger.estimate().free(), GIB, "1 GiB free beside the bank at its floor");
    let sequence = crate::RequestId(1);
    let rows = (1..).map(|pages: u64| pages * 1024).find(|rows| ledger.bytes(&Need::Request { sequence, rows: *rows }) >= 800 * MIB).expect("a history past 800 MiB");
    let slot = ledger.change(|keeping| ledger.claim(&keeping.claims(), Need::RequestSlot { bytes: ledger.bytes(&Need::Request { sequence, rows }) })).expect("the slot fits");
    let reservation = ledger.change(|keeping| ledger.reserve(&keeping.claims(), &slot, sequence, rows)).expect("the request draws its slot");
    (scope, ledger, (floor, records, pool), (slot, reservation), super::StateSlabs::default(), epoch)
}

/// A state growth's slabs are the request's reservation: with 1 GiB free
/// and the bank at its floor, an 800 MiB growth past its rows is held
/// whole, its rows by their reservation and only the slab's rounding past
/// them anew, as spare, never the rows' bytes twice.
#[test]
fn a_state_growth_is_not_admitted_beside_its_own_rows() {
    use crate::memory::ledger::CacheKind;
    let (_scope, ledger, (_floor, _records, pool), (slot, rows), mut slabs, _epoch) = grown_rows();
    let committed = slot.bytes() + rows.bytes();
    ledger.change(|_| slabs.settle(&ledger, 0, 0, 0, committed));
    let growth = rows.bytes() + 16 * MIB;
    ledger.change(|keeping| slabs.reserve(&keeping.claims(), &ledger, growth)).expect("the growth past the rows fits free room");
    ledger.change(|_| { let entry = &pool; entry.take(&entry.change(), growth, seismic::memory::Wiring::Pageable) }).expect("the pool backs the growth");
    ledger.change(|_| slabs.settle(&ledger, growth, rows.bytes(), rows.bytes(), committed));
    assert_eq!(ledger.estimate().cache_of(CacheKind::Spare), 16 * MIB, "only the slab's excess is held anew");
    // A second growth past rows already backed is all spare, and refused
    // past free room.
    assert!(ledger.change(|keeping| slabs.reserve(&keeping.claims(), &ledger, GIB)).is_err());
}

/// After a reading, a request's state bytes are the process's own, never
/// other programs' memory: allocating them moves others nothing.
#[test]
fn a_sequences_state_is_never_read_as_others_memory() {
    let (_scope, ledger, (_floor, _records, pool), (slot, rows), mut slabs, epoch) = grown_rows();
    let others = ledger.estimate().others();
    let committed = slot.bytes() + rows.bytes();
    ledger.change(|_| slabs.settle(&ledger, 0, 0, 0, committed));
    let growth = rows.bytes() + 16 * MIB;
    ledger.change(|keeping| slabs.reserve(&keeping.claims(), &ledger, growth)).expect("the growth fits");
    ledger.change(|_| { let entry = &pool; entry.take(&entry.change(), growth, seismic::memory::Wiring::Pageable) }).expect("the pool backs the growth");
    ledger.change(|_| slabs.settle(&ledger, growth, rows.bytes(), rows.bytes(), committed));
    read_at(&ledger, epoch, GIB - growth, 2);
    assert_eq!(ledger.estimate().others(), others, "the state's bytes read as the process's own");
    // Closed, its rows' slab bytes stay the process's: the slot's room.
    drop(rows);
    ledger.change(|_| slabs.settle(&ledger, growth, 0, 0, slot.bytes()));
    read_at(&ledger, epoch, GIB - growth, 3);
    assert_eq!(ledger.estimate().others(), others, "a closed request's slabs read as the process's own");
}

/// When `claim_group_class` hits a capacity refusal after partially allocating
/// slots, fitting each claim and idling the non-zero claims into `IdleClass`
/// converts them to `CacheKind::GraphSlots` instead of stranding them as
/// `CommittedKind::GroupClass`.
#[test]
fn partial_group_class_refusal_idles_allocated_claims_as_graph_slots() {
    use crate::memory::ledger::{CacheKind, CommittedKind};
    let (_scope, ledger, _pools, _res, _slabs, _epoch) = grown_rows();
    let before_allocated = ledger.estimate().allocated();
    let partial = ledger
        .change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::GroupClass, 128 * MIB))
        .expect("room for group class");
    let unallocated = ledger
        .change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::GroupClass, 64 * MIB))
        .expect("room for second store claim");
    // Simulate partial allocation of 32 MiB on the first claim and 0 on the second.
    ledger
        .change(|_| partial.entry().take(&partial.entry().change(), 32 * MIB, seismic::memory::Wiring::Pageable))
        .expect("partial allocation admitted");
    let mut idle_class = super::IdleClass::default();
    ledger.change(|_| {
        let mut claims = vec![partial, unallocated];
        claims.iter_mut().for_each(crate::memory::ledger::Claim::fit_allocated);
        let allocated: Vec<_> = claims.into_iter().filter(|claim| claim.bytes() > 0).collect();
        assert_eq!(allocated.len(), 1, "zero-byte unallocated claim drops without idling");
        idle_class.idle(allocated);
    });
    let est = ledger.estimate();
    assert_eq!(est.cache_of(CacheKind::GraphSlots), 32 * MIB, "partially allocated slots idle as GraphSlots cache");
    assert_eq!(est.allocated(), before_allocated + 32 * MIB);
    let released = idle_class.release();
    assert_eq!(released.len(), 1);
    ledger.change(|_| {
        let cached = &released[0];
        cached.entry().freed(&cached.entry().change(), 32 * MIB, seismic::memory::Wiring::Pageable);
    });
    drop(released);
    assert_eq!(ledger.estimate().cache_of(CacheKind::GraphSlots), 0);
}
