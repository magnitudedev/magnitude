//! Committed claims: what a load takes, and what it never sets aside.

use super::super::fixtures::{bank_backends, banked_chain};

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
/// holding at least its graph scratch, so a bank sized from the ledger
/// afterwards can never take the room places run in.
#[test]
fn a_load_commits_its_arena_before_any_prompt() {
    for backend in bank_backends() {
        let (domain, _bindings) = banked_chain(backend, 4, 3);
        let arena = committed_of(&domain, crate::memory::ledger::CommittedKind::Arena);
        let scratch = domain.execution.resources().bytes().scratch;
        assert!(
            matches!(arena.as_slice(), [bytes] if *bytes >= scratch),
            "{backend:?}: the arena's claims {arena:?} beside {scratch} bytes of graph scratch"
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
/// released at the next follow of the cache's targets, as a bank's slots
/// are once a claim takes theirs: their release begins, and once it ends
/// their cache is gone.
#[test]
fn idle_slots_a_claim_takes_are_released_at_the_next_follow() {
    use crate::memory::ledger::CommittedKind;
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked_chain(backend, 4, 3);
        let (places, _) = opened(&mut domain, 2);
        assert_eq!(places, 3, "{backend:?}");
        domain.close_group().unwrap();
        assert!(idle_cache(&domain) > 0, "{backend:?}: the closed class idles as cache");
        assert!(!domain.idle_slots_taken(), "{backend:?}");
        // A claim takes every byte that yields, the idle slots' cache among
        // it (it yields before the bank's).
        let ledger = domain.resources.ledgers().device().clone();
        let (_all, _) = ledger.claim_units(CommittedKind::Measurement, std::num::NonZeroU64::new(1).unwrap(), u64::MAX);
        assert!(domain.idle_slots_taken(), "{backend:?}: the claim took the idle slots' cache");
        assert_eq!(
            domain.follow_cache_targets(&mut bindings, crate::CompletionWake::new(|| {})).unwrap(),
            crate::domain::Followed::ReleasingSlots,
            "{backend:?}: the slots' release begins"
        );
        while domain.advance_slot_release().unwrap().is_none() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(idle_cache(&domain), 0, "{backend:?}: the released slots' cache is gone");
        assert!(!domain.idle_slots_taken(), "{backend:?}");
    }
}

/// A claim that took the cache of a closed group's idle slots, all of
/// which a release let go meanwhile: the next follow lets the cache go as a
/// change of its own, under the process's rule, which a record outside one
/// breaks.
#[test]
fn a_taken_idle_cache_with_no_slot_left_goes_in_a_change() {
    use crate::memory::ledger::CommittedKind;
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked_chain(backend, 4, 3);
        let (places, _) = opened(&mut domain, 2);
        assert_eq!(places, 3, "{backend:?}");
        domain.close_group().unwrap();
        let ledger = domain.resources.ledgers().device().clone();
        let (all, _) = ledger.claim_units(CommittedKind::Measurement, std::num::NonZeroU64::new(1).unwrap(), u64::MAX);
        assert!(domain.idle_slots_taken(), "{backend:?}: the claim took the idle slots' cache");
        // The slots are released apart from their cache, which stays taken.
        assert!(domain.resources.release_idle_slots(Vec::new(), crate::CompletionWake::new(|| {})), "{backend:?}");
        while domain.advance_slot_release().unwrap().is_none() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(domain.idle_slots_taken(), "{backend:?}: no slot idles, its cache still taken");
        let changes = domain.resources.ledgers().changes().clone();
        changes.strict();
        assert_eq!(
            domain.follow_cache_targets(&mut bindings, crate::CompletionWake::new(|| {})).unwrap(),
            crate::domain::Followed::Holds,
            "{backend:?}: nothing left to release"
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
        let (all, _) = ledger.claim_units(CommittedKind::Measurement, std::num::NonZeroU64::new(1).unwrap(), u64::MAX);
        assert_eq!(
            domain.follow_cache_targets(&mut bindings, crate::CompletionWake::new(|| {})).unwrap(),
            crate::domain::Followed::ReleasingSlots,
            "{backend:?}"
        );
        assert_eq!(graph_slots(&domain), idle, "{backend:?}: the taken slots stay allocated while they release");
        drop(all);
        let (places, claims) = opened(&mut domain, 2);
        assert_eq!(places, 3, "{backend:?}");
        assert_eq!(claims, first, "{backend:?}: the class claims every slot it grows again");
        while domain.advance_slot_release().unwrap().is_none() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(graph_slots(&domain), 0, "{backend:?}: freed with the release");
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
        let (_all, _) = ledger.claim_units(CommittedKind::Measurement, std::num::NonZeroU64::new(1).unwrap(), u64::MAX);
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
    assert!(domain.open_state(&mut bindings, request, None).unwrap().is_empty());
    let rows = 2;
    let prefill = crate::Operation::Forward {
        request,
        kind: crate::WorkKind::Prefill,
        tokens: vec![crate::TokenId(1); rows],
        position: 0,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        committed: rows,
        prime: None,
    };
    domain.provision(&mut bindings, &[prefill], None).unwrap();
    let through = (rows + method.draft_rows()) as u64;
    let state = &domain.state_claims[&request];
    assert_eq!(state.through, through, "the claim reaches the rows and the draft depth");
    let shape = LoadShape::of(&domain.execution);
    let resources = domain.execution.resources();
    let head = resources.head_state().expect("an MTP head keeps its own history");
    let domains = shape.stores.iter().map(|store| store.domains.len()).collect::<Vec<_>>();
    assert_eq!(domains, [resources.target_state().history.len(), head.history.len()], "both stores' domains");
    assert_eq!(state.claim.bytes(), shape.state_bytes(through), "the claim holds both stores' pages through the draft depth");
}

/// A sequence that closes while its growth waits on a preemption lets the
/// preemption go with it: it holds nothing past the close, and the
/// sequence it named, no longer reserved to it, frees its state to the
/// store as it closes in turn.
#[test]
fn a_closed_sequences_preemption_is_let_go() {
    use super::super::fixtures::{fixture_over, limits, tiny_definition, tiny_manifest};
    use crate::memory::ledger::{CommittedKind, Need};
    let definition = tiny_definition();
    let manifest = tiny_manifest(&definition);
    let (mut domain, mut bindings) = fixture_over(
        definition,
        manifest,
        limits(2, 1, false),
        crate::PlannedMethod::Plain,
        None,
        false,
        crate::platform::DeviceRequest::Automatic,
    );
    let (first, later) = (crate::RequestId(1), crate::RequestId(2));
    for request in [first, later] {
        domain.install_input(request, magnitude_family_contracts::PreparedModelInput::continuation_only()).unwrap();
        assert!(domain.open_state(&mut bindings, request, None).unwrap().is_empty());
    }
    let rows = 4;
    domain.claim_state_rows(later, rows).unwrap();
    // Everything else is taken: the first sequence's growth of as many rows
    // is covered by the later one's state alone.
    let ledger = domain.resources.ledgers().device().clone();
    let snapshot = ledger.snapshot();
    let _room = ledger.claim(Need::Measurement { bytes: snapshot.free() + snapshot.cache() }).expect("the room fits");
    assert!(matches!(
        domain.claim_state_rows(first, rows),
        Err(crate::domain::DomainError::Preempt(named)) if named == [later]
    ));
    domain.close(first).unwrap();
    assert!(committed_of(&domain, CommittedKind::State(first)).is_empty(), "the preemption is let go with its sequence");
    let (committed, held) = (ledger.snapshot().committed(), domain.state_claims[&later].claim.bytes());
    domain.close(later).unwrap();
    assert_eq!(ledger.snapshot().committed(), committed - held, "the named sequence's state frees to the store");
}

/// A component's import, made under its claim and marked allocated, is the
/// process's memory at the next reading, never others': a 1 GB head import
/// with the bank at its floor and free room under 1 GB raises no guarantee
/// and unloads nothing.
#[test]
fn an_imports_bytes_are_never_others_memory() {
    use crate::memory::{ClaimDemand, ledger::{Available, CommittedKind}};
    const GB: u64 = 1 << 30;
    for backend in bank_backends() {
        // The bank's one committed row is its floor: it holds no cache.
        let (domain, _bindings) = banked_chain(backend, 4, 1);
        let ledger = domain.resources.ledgers().device();
        let start = std::time::Instant::now();
        let others = GB;
        let total = ledger.snapshot().committed() + ledger.snapshot().cache() + GB + GB / 2 + others;
        let read = |tick: u32| {
            let process = ledger.snapshot().allocated();
            let at = start + std::time::Duration::from_millis(100) * tick;
            ledger.read(ledger.sampled(at), Ok(Available::new(total, total - others - process, None).expect("a reading")));
        };
        read(1);
        read(2);
        assert_eq!(ledger.snapshot().cache(), ledger.snapshot().cache_of(CacheKind::Spare), "{backend:?}: the bank is at its floor");
        let mut import = domain
            .claim_growth(CommittedKind::Component, ClaimDemand { device: GB, ..Default::default() })
            .expect("the import fits");
        import.made();
        read(3);
        let after = ledger.snapshot();
        assert_eq!(after.others(), Some(others), "{backend:?}: the import read as others' memory");
        assert!(after.free() < GB, "{backend:?}: free room is under the import");
        assert_eq!(after.guarantee(0), 0, "{backend:?}: the import raised the guarantee");
        assert!(!after.unload(0), "{backend:?}: the import unloads the model");
        drop(import);
    }
}

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// A store's ledger, read at 64 GiB total and `available`, `at` ms in.
fn read_at(ledger: &crate::memory::ledger::Ledger, available: u64, at: u64) {
    let at = std::time::Instant::now() + std::time::Duration::from_millis(at);
    ledger.read(ledger.sampled(at), Ok(crate::memory::ledger::Available::new(64 * GIB, available, None).expect("a reading")));
}

/// A load's ledger with 1 GiB free beside the bank at its floor, a
/// sequence's rows claimed past 800 MiB, and the stores' slabs as the load
/// started: the rows' claim, the floor's, and the slabs.
fn grown_rows() -> (
    crate::memory::ledger::Ledgers,
    crate::memory::ledger::Ledger,
    crate::memory::ledger::Claim,
    crate::memory::ledger::Claim,
    super::StateSlabs,
) {
    use crate::memory::ledger::{LoadShape, Need};
    let component = magnitude_state::ComponentDescriptor::new(
        magnitude_state::LayerRef::Target(0),
        magnitude_state::CodecSpec::dense(seismic::DType::F32, 1024, 1024),
        1,
    )
    .expect("a component");
    let domains = vec![magnitude_state::HistoryDomainLayout::Token { components: vec![component] }];
    let shape = std::sync::Arc::new(LoadShape { stores: vec![crate::memory::ledger::StoreShape { domains, context: 1 << 20, max_advance: 16 }] });
    let scope = crate::memory::ledger::Ledgers::new();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, shape);
    read_at(&ledger, 2 * GIB, 0);
    let mut floor = ledger.claim(Need::BankFloor { bytes: GIB }).expect("the floor fits");
    ledger.change(|changing| floor.allocated(GIB, changing));
    read_at(&ledger, GIB, 1);
    assert_eq!(ledger.snapshot().free(), GIB, "1 GiB free beside the bank at its floor");
    let sequence = crate::RequestId(1);
    let to = (1..).map(|pages: u64| pages * 1024).find(|rows| ledger.bytes(&Need::StateRows { sequence, from: 0, to: *rows }) >= 800 * MIB).expect("a history past 800 MiB");
    let rows = ledger.claim(Need::StateRows { sequence, from: 0, to }).expect("the rows fit");
    (scope, ledger, floor, rows, super::StateSlabs::new(64 * MIB))
}

/// A state growth's slabs are the rows its sequence claimed: with 1 GiB
/// free and the bank at its floor, an 800 MiB growth past rows claimed is
/// held whole, its rows allocated on their claim and only the slab's
/// rounding past them held anew, as spare, never the rows' bytes twice.
#[test]
fn a_state_growth_is_not_admitted_beside_its_own_rows() {
    use crate::memory::ledger::CacheKind;
    let (_scope, ledger, _floor, mut rows, mut slabs) = grown_rows();
    let growth = rows.bytes() + 16 * MIB;
    slabs.reserve(&ledger, growth, std::iter::once(&rows)).expect("the growth past the rows fits free room");
    ledger.change(|changing| slabs.settle(&ledger, 64 * MIB + growth, std::iter::once(&mut rows), changing));
    assert_eq!(rows.allocated_bytes(), rows.bytes(), "the slabs back the claimed rows");
    assert_eq!(ledger.snapshot().cache_of(CacheKind::Spare), 16 * MIB, "only the slab's excess is held anew");
    // A second growth past rows already backed is all spare, and refused
    // past free room.
    assert!(slabs.reserve(&ledger, GIB, std::iter::once(&rows)).is_err());
}

/// After a reading, a sequence's state bytes are the process's own, never
/// other programs' memory: allocating them moves others nothing.
#[test]
fn a_sequences_state_is_never_read_as_others_memory() {
    let (_scope, ledger, _floor, mut rows, mut slabs) = grown_rows();
    let others = ledger.snapshot().others();
    let growth = rows.bytes() + 16 * MIB;
    slabs.reserve(&ledger, growth, std::iter::once(&rows)).expect("the growth fits");
    ledger.change(|changing| slabs.settle(&ledger, 64 * MIB + growth, std::iter::once(&mut rows), changing));
    read_at(&ledger, GIB - growth, 2);
    assert_eq!(ledger.snapshot().others(), others, "the state's bytes read as the process's own");
    // Evicted, its rows' slab bytes stay the process's, as spare.
    drop(rows);
    ledger.change(|changing| slabs.settle(&ledger, 64 * MIB + growth, std::iter::empty(), changing));
    read_at(&ledger, GIB - growth, 3);
    assert_eq!(ledger.snapshot().others(), others, "an evicted sequence's slabs read as the process's own");
}
