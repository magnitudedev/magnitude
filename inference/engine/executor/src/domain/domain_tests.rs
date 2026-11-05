//! Controlled completion tests for the production execution domain.

use super::fixtures::*;
use super::*;
use crate::{
    ResourceLimits,
    completion::CompletionWake,
    platform,
    programs::{CompletedHeadWork, CompletedVisionWork},
};
use magnitude_state::ShrinkPolicy;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn forward(request: RequestId, position: usize) -> Operation {
    Operation::Forward {
        request,
        kind: WorkKind::Replay,
        tokens: vec![crate::TokenId(1)],
        position,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        committed: 1,
        prime: None,
    }
}

/// Make `request` resident on fresh state with no prompt rows.
fn open<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    bindings: &mut StateBindings,
    request: RequestId,
) {
    domain
        .install_input(request, PreparedModelInput::continuation_only())
        .unwrap();
    assert!(
        domain
            .open_state(bindings, request, None)
            .unwrap()
            .is_empty()
    );
}

/// Reserve and submit one target group; the flight takes the bindings.
fn submit_reserved_target<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    mut bindings: StateBindings,
    operations: Vec<Operation>,
) -> TargetFlight<F> {
    let resources = domain
        .reserve(&mut bindings, &operations)
        .unwrap()
        .into_resources();
    let ReservedResources::Target(reservation) = resources else {
        panic!("target operation produced another reservation lane")
    };
    domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("target submission failed: {}", failure.error()))
}

/// Finish a target flight with one outcome.
fn finish_one<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    flight: TargetFlight<F>,
) -> (PendingOperationOutcome, StateBindings) {
    let (mut pending, bindings) = domain.finish_target(flight).unwrap();
    (pending.pop().unwrap(), bindings)
}

fn prefill(request: RequestId, position: usize) -> Operation {
    Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: vec![crate::TokenId(1)],
        position,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        committed: 1,
        prime: None,
    }
}

/// Submit and reconcile a prefill chunk at 0 whose planned next chunk is
/// queued behind it: the returned bindings hold that lookahead.
fn queue_lookahead(
    domain: &mut ExecutorDomain<TestFamily>,
    bindings: StateBindings,
    request: RequestId,
) -> StateBindings {
    domain.plan_successors(vec![prefill(request, 1)]);
    let flight = submit_reserved_target(domain, bindings, vec![prefill(request, 0)]);
    let (pending, bindings) = finish_one(domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    assert_eq!(domain.queue.len(), 1, "the next chunk is queued");
    bindings
}

#[test]
fn ready_target_can_abort_then_reconcile() {
    let (mut domain, mut bindings) = fixture(None);
    let request = RequestId(1);
    open(&mut domain, &mut bindings, request);
    let mut flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    assert!(flight.completion().is_complete());
    assert!(domain.resume_state(request).is_err());
    let (pending, bindings) = finish_one(&mut domain, flight);
    domain.abort(pending).unwrap();
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 0);
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    let (pending, _) = finish_one(&mut domain, flight);
    assert!(
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
            .is_ok()
    );
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 1);
}

/// A launch is charged the host work since the flight before it finished
/// when that flight carried its request: a round's host time is part of its
/// cost. Time the owner spent idle, or a flight of other requests, is
/// charged to no launch. Where the service starts is an order, so each
/// charge is bounded by instants the test takes around it, whatever the
/// machine's load: one charged from the turn before it spans the gap the
/// test slept there, and one charged from its submission fits within its
/// own step, which excludes that gap.
#[test]
fn a_launch_is_charged_its_requests_host_turn_but_not_idle_time() {
    let (mut domain, mut bindings) = fixture(None);
    let gap = std::time::Duration::from_millis(20);
    let (first, second) = (RequestId(70), RequestId(71));
    open(&mut domain, &mut bindings, first);
    open(&mut domain, &mut bindings, second);
    // A step's charge, and the time from just before its submission to
    // just after its finish.
    let step = |domain: &mut ExecutorDomain<TestFamily>,
                bindings: StateBindings,
                request: RequestId,
                position: usize| {
        let before = std::time::Instant::now();
        let flight = submit_reserved_target(domain, bindings, vec![forward(request, position)]);
        let (pending, bindings) = finish_one(domain, flight);
        let within = before.elapsed();
        let duration = pending.physical_duration();
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
            .unwrap();
        (duration, within, bindings)
    };
    let (_, _, bindings) = step(&mut domain, bindings, first, 0);
    std::thread::sleep(gap);
    let (charged, _, bindings) = step(&mut domain, bindings, first, 1);
    assert!(charged >= gap, "the host turn of its own request: {charged:?}");
    std::thread::sleep(gap);
    let (other, within, bindings) = step(&mut domain, bindings, second, 0);
    assert!(other <= within, "another request's turn: {other:?}, its step {within:?}");
    std::thread::sleep(gap);
    domain.host_idle();
    let (idle, within, _) = step(&mut domain, bindings, second, 1);
    assert!(idle <= within, "after the owner idled: {idle:?}, its step {within:?}");
}

/// A request that stops continuing its queued step discards it, and waiting
/// for that step is part of its next launch: the switch away from plain
/// decoding costs the round that makes it.
#[test]
fn a_launch_that_discards_its_requests_lookahead_is_charged_the_wait() {
    let (mut domain, mut bindings) = fixture_with(None, true, true);
    let gap = std::time::Duration::from_millis(60);
    let request = RequestId(72);
    open(&mut domain, &mut bindings, request);
    let bindings = queue_lookahead(&mut domain, bindings, request);
    std::thread::sleep(gap);
    // A replay row is not the queued prefill step: submitting it discards
    // that step first.
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 1)]);
    let (pending, _) = finish_one(&mut domain, flight);
    assert!(domain.queue.is_empty(), "the queued step was discarded");
    assert!(
        pending.physical_duration() >= gap,
        "the wait and the host turn before it: {:?}",
        pending.physical_duration()
    );
}

/// A queued step's service starts where its predecessor's ends, once the
/// predecessor's results are read: pipelined flights charge the time they
/// overlap once, so a chain's durations add up to no more than its span.
#[test]
fn a_queued_step_is_charged_from_where_its_predecessor_ends() {
    let (mut domain, mut bindings) = fixture_with(None, true, true);
    let request = RequestId(73);
    open(&mut domain, &mut bindings, request);
    domain.plan_successors(vec![prefill(request, 1)]);
    let flight = submit_reserved_target(&mut domain, bindings, vec![prefill(request, 0)]);
    let began = flight.work.runnable;
    let (first, bindings) = finish_one(&mut domain, flight);
    let first_duration = first.physical_duration();
    assert_eq!(domain.queue.len(), 1, "the next chunk is queued");
    domain
        .reconcile(first, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    let flight = submit_reserved_target(&mut domain, bindings, vec![prefill(request, 1)]);
    assert_eq!(
        flight.work.runnable,
        began + first_duration,
        "the claimed step starts where its predecessor ends"
    );
    let (second, _) = finish_one(&mut domain, flight);
    let span = began.elapsed();
    assert!(
        first_duration + second.physical_duration() <= span,
        "{first_duration:?} + {:?} exceeds the chain's span {span:?}",
        second.physical_duration()
    );
}

/// Growth changes bindings: every entry that can grow resolves a queued
/// lookahead first, then commits the slab.
#[test]
fn growth_orphans_a_queued_lookahead_then_grows() {
    let (mut domain, mut bindings) = fixture_with(None, true, true);
    let request = RequestId(60);
    open(&mut domain, &mut bindings, request);
    let mut bindings = queue_lookahead(&mut domain, bindings, request);
    let head = domain.head_store.clone().unwrap();
    let history = head.sole_history_domain().unwrap();
    let committed = head.committed_rows(history);
    let demand = [RowDemand {
        domain: history,
        rows: head.free_rows(history) + 1,
    }];
    domain.provision_open(&mut bindings).unwrap();
    assert!(
        domain.queue.is_empty(),
        "the growing entry orphaned the lookahead"
    );
    domain
        .grant_state_growth(&mut bindings, true, &demand, &[], 0)
        .unwrap();
    assert!(head.committed_rows(history) > committed);
    // The orphaned step changed no accepted state.
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 1);
}

/// A queued lookahead reserves the rows after its request's history, so that
/// history's last page reads as blocked by another. Drafter work for the same
/// request, which does not claim the lookahead, must resolve it before any
/// layout question: it provisions without relocating, and the lookahead is
/// gone. (Regression: relocation planned against the lookahead's rows, then
/// orphaned it and failed with "only a blocked last page is relocated".)
#[test]
fn drafter_work_resolves_its_requests_lookahead_before_provisioning() {
    let (mut domain, mut bindings) = fixture_with(None, true, true);
    let request = RequestId(61);
    open(&mut domain, &mut bindings, request);
    let mut bindings = queue_lookahead(&mut domain, bindings, request);
    assert!(
        !domain.target[&request].blocked_tails().is_empty(),
        "the lookahead's rows read as another history's"
    );
    let draft = Operation::Head {
        request,
        phase: crate::HeadPhase::Generation,
        tokens: vec![crate::TokenId(1)],
        conditioning: crate::FeatureRows::new(vec![0u8; 4].into(), 1).unwrap(),
        position: 1,
        proposals: Vec::new(),
        form: crate::DraftForm::Chained,
    };
    // The fixture has no drafter weights, so provisioning ends at loading the
    // head component: after the lookahead and every relocation are resolved.
    match domain.provision(&mut bindings, &[draft], None) {
        Ok(()) => {}
        Err(DomainError::Input(detail)) if detail == "head component is disabled" => {}
        Err(error) => panic!("provisioning drafter work failed: {error}"),
    }
    assert!(
        domain.queue.is_empty(),
        "drafter work resolved the lookahead"
    );
    assert!(domain.target[&request].blocked_tails().is_empty());
}

/// A domain launching one-row prompt chunks in chains of up to three.
fn chain_fixture() -> Fixture {
    fixture_of(
        tiny_definition(),
        limits(2, 3, false),
        None,
        false,
        platform::DeviceRequest::Automatic,
    )
}

/// Reserve and submit a group that must claim the front queued step.
fn submit_claimed(
    domain: &mut ExecutorDomain<TestFamily>,
    mut bindings: StateBindings,
    operations: Vec<Operation>,
) -> TargetFlight<TestFamily> {
    let reservation = domain.reserve(&mut bindings, &operations).unwrap();
    let ReservedResources::Target(reservation @ TargetGraphReservation::Claim(_)) =
        reservation.into_resources()
    else {
        panic!("the group does not claim the queued step")
    };
    domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("claim failed: {}", failure.error()))
}

/// A group of no operations opens no chain and provisions nothing, whatever
/// chain capacity the load has.
#[test]
fn an_empty_group_provisions_nothing_on_a_chaining_load() {
    let (mut domain, mut bindings) = chain_fixture();
    domain.provision(&mut bindings, &[], None).unwrap();
    assert!(domain.queue.is_empty());
}

/// A lone prompt chunk opens a chain of the planned chunks continuing it,
/// within the capacity; its holder claims each queued chunk in turn. The
/// chain holds its holder's state until its last chunk publishes it whole.
#[test]
fn a_prompt_chunk_opens_a_chain_its_holder_claims() {
    let (mut domain, mut bindings) = chain_fixture();
    let request = RequestId(80);
    open(&mut domain, &mut bindings, request);
    // Four chunks are planned: the chain takes the capacity's three.
    domain.plan_successors((1..4).map(|position| prefill(request, position)).collect());
    let flight = submit_reserved_target(&mut domain, bindings, vec![prefill(request, 0)]);
    assert_eq!(domain.queue.len(), 2, "the chain queues two chunks");
    let (pending, mut bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    for position in 1..3 {
        assert_eq!(domain.chain_holder(), Some(request));
        assert!(domain.resume_state(request).is_err(), "the chain holds the state");
        let flight = submit_claimed(&mut domain, bindings, vec![prefill(request, position)]);
        let (pending, returned) = finish_one(&mut domain, flight);
        bindings = returned;
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
            .unwrap();
    }
    assert!(domain.queue.is_empty());
    assert_eq!(domain.chain_holder(), None);
    let state = domain.resume_state(request).unwrap().target;
    let token = magnitude_state::HistoryDomainId(0);
    assert_eq!((state.position(), state.history_ranges(token).len()), (3, 1));
    // The group's class is kept while memory is Normal, and released to the
    // floor of one place at Reclaim's surplus rung.
    assert_eq!(domain.group_places().get(), 3);
    domain.shrink_state(&mut bindings, ShrinkPolicy::Idle).unwrap();
    assert_eq!(domain.group_places().get(), 3);
    domain.shrink_state(&mut bindings, ShrinkPolicy::Reclaim).unwrap();
    assert_eq!(domain.group_places().get(), 1);
}

/// A bank short of its ceiling keeps the room it can grow into: a prompt's
/// group grows its places only beyond it, so with the device's gigabyte of
/// room all the bank's to grow into the group stays at its floor of one
/// place, and with a bank at its ceiling the same group takes its chunks'
/// places, whether the ceiling is every expert or a weight cap's short of
/// them.
#[test]
fn places_grow_only_beyond_what_the_bank_can_grow_into() {
    for backend in bank_backends() {
        let places = |shortfall: u64, whole: bool| {
            let (mut domain, mut bindings) = banked_chain(backend, 4, 3);
            let bank = domain.family.bank.as_mut().expect("the fixture holds a bank");
            (bank.shortfall, bank.whole) = (shortfall, whole);
            // A reading of the device's store with a gigabyte of room.
            let store = domain.domain.device().device_store();
            let room = 1 << 30;
            domain.pace.borrow_mut().domains.insert(store, DomainPace {
                role: DomainRole::Allocation,
                headroom: room,
                planning: 0,
                emergency: 0,
                ceiling: room,
                charged: 0,
                band: MemoryBand::Normal,
                unseen: 0,
                available: 0,
                base: (std::time::Instant::now(), 0),
                rate: 0.0,
                fastest: 0.0,
                fluctuation: Fluctuation::default(),
            });
            let request = RequestId(80);
            open(&mut domain, &mut bindings, request);
            domain.plan_successors((1..4).map(|position| prefill(request, position)).collect());
            let flight = submit_reserved_target(&mut domain, bindings, vec![prefill(request, 0)]);
            let places = domain.group_places().get();
            let _ = finish_one(&mut domain, flight);
            places
        };
        assert_eq!(places(0, true), 3, "{backend:?}: a whole bank leaves the group its chunks' places");
        assert_eq!(places(0, false), 3, "{backend:?}: a bank at a weight cap's ceiling leaves the group its chunks' places");
        assert_eq!(places(u64::MAX / 4, false), 1, "{backend:?}: a bank that can grow into the room keeps it");
    }
}

/// A prompt chunk's readout exports at most its final row, so the places a
/// group's class holds beyond the floor's are charged what that exports,
/// whatever the load's widest export; the launch pool's floor holds the
/// widest.
#[test]
fn a_groups_places_hold_what_a_prompt_chunk_exports() {
    let load = |exported| {
        let limits = ResourceLimits {
            exported_logits_rows: exported,
            ..limits(2, 3, false)
        };
        let (mut domain, _) =
            fixture_of(tiny_definition(), limits, None, false, platform::DeviceRequest::Automatic);
        let floor = domain.resources.target_readout_graph(usize::MAX).committed_bytes().unwrap();
        let places = domain.resources.class_bytes(limits.class_slots(3));
        (floor, places)
    };
    let ((narrow_floor, narrow_places), (wide_floor, wide_places)) = (load(1), load(2));
    assert!(wide_floor > narrow_floor, "the floor's readout holds the widest export");
    assert_eq!(wide_places, narrow_places);
}

/// A prompt chunk plans the rest of its group behind it, or without chains
/// the one chunk a pipelined successor runs.
#[test]
fn a_prompt_chunk_plans_the_rest_of_its_group() {
    for (fixture, planned) in [(chain_fixture(), 2), (fixture_with(None, true, false), 1)] {
        let (mut domain, _) = fixture;
        let mut counted = None;
        domain
            .plan_prompt(None, true, |chunks| {
                counted = Some(chunks);
                Ok::<_, ()>(Vec::new())
            })
            .unwrap();
        assert_eq!(counted, Some(planned));
    }
}

/// A pipelined successor, which other work orphans, is planned only for a
/// request the device serves alone; a chain, never orphaned, regardless.
#[test]
fn only_a_chain_is_planned_beside_other_work() {
    for (fixture, planned) in [(chain_fixture(), 2), (fixture_with(None, true, false), 0)] {
        let (mut domain, _) = fixture;
        let mut counted = None;
        domain
            .plan_prompt(None, false, |chunks| {
                counted = Some(chunks);
                Ok::<_, ()>(Vec::new())
            })
            .unwrap();
        assert_eq!(counted, Some(planned));
    }
}

/// While decode waits, a load that chains plans the rest of a group of the
/// size rule's chunks: a rule of one plans none behind the chunk, so decode
/// waits behind that one chunk alone.
#[test]
fn a_waiting_group_of_one_plans_no_chunk_behind() {
    let (mut domain, _) = chain_fixture();
    for (waiting, planned) in [(1, 0), (2, 1), (3, 2)] {
        domain.family.waiting = waiting;
        let mut counted = None;
        domain
            .plan_prompt(Some(0.5), true, |chunks| {
                counted = Some(chunks);
                Ok::<_, ()>(Vec::new())
            })
            .unwrap();
        assert_eq!(counted, Some(planned), "a rule of {waiting}");
    }
}

/// A chain under way is never orphaned: other work is refused while it is.
/// A holder that ended releases it, its queued launches waited for and
/// dropped and its state evicted with it; one that closes drops it, its
/// launches resolved with the queue.
#[test]
fn a_chain_is_released_by_its_holder_never_orphaned() {
    let (mut domain, mut bindings) = chain_fixture();
    let request = RequestId(81);
    open(&mut domain, &mut bindings, request);
    let held = |domain: &mut ExecutorDomain<TestFamily>, bindings: StateBindings| {
        domain.plan_successors(vec![prefill(request, 1), prefill(request, 2)]);
        let flight = submit_reserved_target(domain, bindings, vec![prefill(request, 0)]);
        let (pending, bindings) = finish_one(domain, flight);
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
            .unwrap();
        assert_eq!((domain.queue.len(), domain.chain_holder()), (2, Some(request)));
        bindings
    };
    let mut bindings = held(&mut domain, bindings);
    // A replay row is not the queued prompt chunk.
    assert!(matches!(
        domain.provision(&mut bindings, &[forward(request, 1)], None),
        Err(DomainError::Invariant(_))
    ));
    domain.release_chain().unwrap();
    assert!(domain.queue.is_empty());
    assert_eq!(domain.chain_holder(), None);
    assert!(domain.resume_state(request).is_err(), "the holder was evicted");

    domain.open_state(&mut bindings, request, None).unwrap();
    let mut bindings = held(&mut domain, bindings);
    domain.close(request).unwrap();
    assert_eq!(domain.chain_holder(), None);
    domain.shrink_state(&mut bindings, ShrinkPolicy::Idle).unwrap();
    assert!(domain.queue.is_empty());
}

/// A rejected chain returns every launch: the domain restores the first's
/// accepted state, and the launches behind it return their rows as they
/// drop.
#[test]
fn a_rejected_chain_returns_every_launch() {
    let (mut domain, mut bindings) = chain_fixture();
    let request = RequestId(82);
    open(&mut domain, &mut bindings, request);
    domain.plan_successors(vec![prefill(request, 1), prefill(request, 2)]);
    let operations = vec![prefill(request, 0)];
    let ReservedResources::Target(reservation) = domain
        .reserve(&mut bindings, &operations)
        .unwrap()
        .into_resources()
    else {
        panic!("a forward reserves the target lane")
    };
    REJECTED.with(|rejected| rejected.set(Some(0)));
    let failure = domain.submit_target(bindings, &operations, reservation);
    assert!(matches!(failure, Err(SubmitFailure::Failed(_))));
    assert_eq!(REJECTED.with(|rejected| rejected.take()), Some(3));
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 0);
}

/// A history's next rows in its partly used last page need no free page: a
/// store whose every page is referenced still reserves them.
#[test]
fn in_place_rows_reserve_without_a_free_page() {
    let (mut domain, mut bindings) = fixture(None);
    let request = RequestId(10);
    open(&mut domain, &mut bindings, request);
    let accept =
        |domain: &mut ExecutorDomain<TestFamily>, bindings: StateBindings, position| {
            let flight = submit_reserved_target(domain, bindings, vec![forward(request, position)]);
            let (pending, bindings) = finish_one(domain, flight);
            domain
                .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
                .unwrap();
            bindings
        };
    let mut bindings = accept(&mut domain, bindings, 0);
    // Hold every other page of the backed slabs with one-row histories.
    let store = domain.target_store.clone();
    let mut held = Vec::new();
    while store
        .history_domains()
        .any(|history| store.free_rows(history) != 0)
    {
        let advance = OwnedStateAdvance::begin(store.create().unwrap(), 1)
            .ok()
            .unwrap();
        let OwnedAdvanceResolution::Committed(filler) = advance.commit_all().ok().unwrap() else {
            panic!("a one-row advance commits");
        };
        held.push(filler);
    }
    for position in 1..4 {
        bindings = accept(&mut domain, bindings, position);
    }
    assert!(
        store
            .history_domains()
            .all(|history| store.free_rows(history) == 0)
    );
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 4);
}

/// Continuing a checkpoint relocates a tail another request extended. The
/// reservation must describe the new layout rather than demand a second page.
#[test]
fn reservation_recomputes_row_demand_after_shared_tail_relocation() {
    let (mut domain, mut bindings) = fixture(None);
    let source = RequestId(71);
    open(&mut domain, &mut bindings, source);
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(source, 0)]);
    let (pending, mut bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    let checkpoint = domain.resume_state(source).unwrap();
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(source, 1)]);
    let (pending, returned) = finish_one(&mut domain, flight);
    bindings = returned;
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    let copy = RequestId(72);
    domain
        .install_input(copy, PreparedModelInput::continuation_only())
        .unwrap();
    domain
        .open_state(&mut bindings, copy, Some(&checkpoint))
        .unwrap();
    assert!(!domain.target[&copy].blocked_tails().is_empty());
    let operations = [forward(copy, 1)];
    let before = domain.requirements(&operations).unwrap();
    assert!(before.state_demand.iter().any(|demand| demand.rows > 0));
    let reservation = domain.reserve(&mut bindings, &operations).unwrap();
    assert!(
        reservation
            .requirements
            .state_demand
            .iter()
            .all(|demand| demand.rows == 0)
    );
}

/// Two requests continuing one checkpoint share its partial last page, whose
/// free rows either could grow into. One launch advancing both relocates one
/// tail first and grows the other in place. (Regression #237: neither tail
/// read as blocked, the first advance took the rows, and the second failed
/// with "target state capacity changed after availability check".)
#[test]
fn one_launch_advances_two_continuations_of_a_shared_tail() {
    let (mut domain, mut bindings) = fixture(None);
    let source = RequestId(81);
    open(&mut domain, &mut bindings, source);
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(source, 0)]);
    let (pending, mut bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    let checkpoint = domain.resume_state(source).unwrap();
    let copies = [RequestId(82), RequestId(83)];
    for copy in copies {
        domain
            .install_input(copy, PreparedModelInput::continuation_only())
            .unwrap();
        domain
            .open_state(&mut bindings, copy, Some(&checkpoint))
            .unwrap();
    }
    for copy in copies {
        assert!(!domain.target[&copy].blocked_tails().is_empty());
    }
    let flight = submit_reserved_target(
        &mut domain,
        bindings,
        copies.iter().map(|&copy| forward(copy, 1)).collect(),
    );
    let (pending, _bindings) = domain.finish_target(flight).unwrap();
    assert_eq!(pending.len(), 2);
    for outcome in pending {
        domain
            .reconcile(outcome, PhysicalDecision { accepted_rows: 1 })
            .unwrap();
    }
    for copy in copies {
        assert_eq!(domain.resume_state(copy).unwrap().target.position(), 2);
    }
}

/// A cancelled completed request aborts and the bindings return; a device
/// failure is returned as an error that consumes them.
#[test]
fn pending_target_request_cancellation_aborts_then_device_failure_consumes_the_bindings() {
    let control = PendingControl::default();
    let (mut domain, mut bindings) = fixture(Some(control.clone()));
    let request = RequestId(2);
    open(&mut domain, &mut bindings, request);
    let mut flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    assert!(!flight.completion().is_complete());
    assert!(domain.resume_state(request).is_err());
    let woke = Arc::new(AtomicBool::new(false));
    let observed = woke.clone();
    flight.completion().notify(CompletionWake::new(move || {
        observed.store(true, Ordering::SeqCst)
    }));
    assert!(!woke.load(Ordering::SeqCst));
    control.resolve(Ok(()));
    assert!(woke.load(Ordering::SeqCst));
    let (pending, _bindings) = finish_one(&mut domain, flight);
    domain.abort(pending).unwrap();
    assert_eq!(domain.resume_state(request).unwrap().target.position(), 0);
    let failed = PendingControl::default();
    let (mut failed_domain, mut bindings) = fixture(Some(failed.clone()));
    let failed_request = RequestId(3);
    open(&mut failed_domain, &mut bindings, failed_request);
    let flight = submit_reserved_target(
        &mut failed_domain,
        bindings,
        vec![forward(failed_request, 0)],
    );
    failed.resolve(Err(crate::DeviceError::Execution(
        "injected target failure".into(),
    )));
    assert!(matches!(
        failed_domain.finish_target(flight),
        Err(DomainError::Device(_))
    ));
    assert!(failed_domain.resume_state(failed_request).is_err());
}

#[test]
fn pending_target_state_is_classified_as_in_flight() {
    let control = PendingControl::default();
    let (mut domain, mut bindings) = fixture(Some(control.clone()));
    let request = RequestId(9);
    open(&mut domain, &mut bindings, request);
    let before = domain.reconcile_memory_charge(&[]).unwrap();
    assert_eq!(before.unattributed, 0, "{before:?}");
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    let pending = domain.reconcile_memory_charge(&[]).unwrap();
    assert!(pending.target_state.in_flight > before.target_state.in_flight);
    assert_eq!(pending.unattributed, 0);
    control.resolve(Ok(()));
    for outcome in domain.finish_target(flight).unwrap().0 {
        domain.abort(outcome).unwrap();
    }
}

#[test]
fn fresh_memory_observation_reconciles_released_state_backing() {
    let (mut domain, mut bindings) = fixture(None);
    let static_bytes = domain.static_holding_bytes().unwrap();
    domain.static_holding = Some(
        domain
            .memory
            .borrow_mut()
            .heap_mut()
            .insert(static_bytes, HoldingClass::Model)
            .unwrap(),
    );
    domain.refresh_memory().unwrap();
    let before = domain.memory().standing().unwrap();
    assert_eq!(before.unattributed_bytes, 0);
    let released = domain
        .shrink_state(&mut bindings, ShrinkPolicy::Reclaim)
        .unwrap();
    assert!(released > 0);
    domain.refresh_memory().unwrap();
    let after = domain.memory().standing().unwrap();
    assert!(before.classified_bytes - after.classified_bytes >= released);
    assert_eq!(after.unattributed_bytes, 0);
}

/// The growth rule's inputs: the load's latest miss record (with its
/// layers' conversion), and the resident step it is priced on: 15 ms at
/// every depth, its all-hit steps' time, so no `σ`.
fn price_growth(domain: &mut ExecutorDomain<TestFamily>, bindings: &mut StateBindings, record: crate::assessment::MissRecord) {
    domain.family.bank.as_mut().unwrap().record = Some(record);
    domain.measure_experts(crate::ExpertMeasures {
        lines: crate::assessment::Priced::Measured(std::sync::Arc::from([])),
        store: None,
        resident: crate::assessment::Priced::Measured(Box::new(|_, _| Ok(0.015))),
        pass: crate::assessment::Priced::Pending("unpriced".into()),
        moves: crate::assessment::Priced::Pending("unmeasured".into()),
        tier_copy: crate::assessment::Priced::Measured(None),
        convert: crate::assessment::Priced::Measured(None),
    }, bindings).unwrap();
}

/// The storage kinds whose bank growth step pays (`expert_bank_growths`),
/// in turn.
fn paying(domain: &ExecutorDomain<TestFamily>) -> Vec<crate::planning::BankTier> {
    let offers = domain.expert_bank_growths().unwrap();
    offers.into_iter().filter(|offer| offer.pays).map(|offer| offer.growth.kind).collect()
}

/// Whether the bank's device step pays.
fn device_pays(domain: &ExecutorDomain<TestFamily>) -> bool {
    paying(domain).contains(&crate::planning::BankTier::Bank)
}

/// A growth step's target is the largest commitment whose bytes are within
/// its store's room, up to where the record's misses stop falling: one step
/// reaches the lesser of the two, never past either, and a holding at
/// either has none. Bytes by reservation granules, as a record tensor's
/// charge rises, find the same target an exhaustive search finds.
#[test]
fn a_growth_step_reaches_the_room_or_the_saturation_and_never_past_either() {
    let target = |saturation, room, bytes: &dyn Fn(u64) -> u64| {
        crate::planning::growth_target(2, saturation, room, |to| Ok::<_, ()>(bytes(to))).unwrap()
    };
    // Ten bytes a slot beyond 2 slots.
    let slots = |to: u64| (to - 2) * 10;
    assert_eq!(target(40, 1_000, &slots), Some(40), "the saturation binds");
    assert_eq!(target(40, 109, &slots), Some(12), "the room binds");
    assert_eq!(target(40, 110, &slots), Some(13), "a step takes all of its room");
    assert_eq!(target(2, 1_000, &slots), None, "saturated");
    assert_eq!(target(40, 9, &slots), None, "not one slot fits");
    let granules = |to: u64| ((to - 2) * 10).div_ceil(32) * 32;
    for bytes in [&slots as &dyn Fn(u64) -> u64, &granules] {
        for saturation in 0..48 {
            for room in 0..400 {
                let largest = (3..=saturation).filter(|&to| bytes(to) <= room).max();
                assert_eq!(target(saturation, room, bytes), largest, "saturation {saturation}, room {room}");
            }
        }
    }
}

/// A growth step is sized within what the heap's claim of it admits
/// (`DeviceHeap::reserved_room`), which admits exactly its room: that many
/// bytes and not one more. With a claim the heap holds leaving room for two
/// of the three slots the record's misses would use, the step takes the two
/// and lands; sized past the heap's room, as by a room the claim does not
/// test, it was refused at every decision.
#[test]
fn a_growth_step_is_sized_within_what_the_heap_admits() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 1);
        let row = bank_row(&domain);
        // Misses stop falling past the bank's 4 rows.
        price_growth(&mut domain, &mut bindings, misses_at(4, 100, 4));
        let device = crate::planning::BankTier::Bank;
        let available = match domain.memory.borrow_mut().claim(u64::MAX / 2, 0, HoldingClass::Live) {
            Err(ClaimRefusal::Deficit { available, .. }) => available,
            refused => panic!("the heap granted every byte: {refused:?}"),
        };
        let held = domain.memory.borrow_mut().claim_reserved(device, available - 5 * row / 2, 0, HoldingClass::Live).unwrap();
        let room = domain.memory.borrow_mut().reserved_room(device, 0).unwrap();
        assert!(
            matches!(
                domain.memory.borrow_mut().claim_reserved(device, room + 1, 0, HoldingClass::Live),
                Err(ClaimRefusal::Deficit { .. })
            ),
            "one byte past the room is refused"
        );
        let claim = domain.memory.borrow_mut().claim_reserved(device, room, 0, HoldingClass::Live).unwrap();
        domain.memory.borrow_mut().release(claim);
        assert!(device_pays(&domain));
        assert_eq!(domain.grow_expert_bank(device, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank), "within the room");
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { from: 1, to: 3, .. }), "two slots of the room");
        domain.memory.borrow_mut().release(held);
    }
}

/// Device memory and host memory grow each their own step in turn: where
/// both pay, the store after the last to grow comes first, so a host tier
/// fills while the device bank still grows. The tier's step holds what the
/// bank's slots miss up to where the record's misses stop falling.
#[test]
fn both_stores_grow_in_turn_where_both_pay() {
    use crate::planning::BankTier::{Bank as Device, Host};
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 1);
        domain.family.bank.as_mut().unwrap().host_ceiling = 4;
        // Misses stop falling at 5 slots: the bank's 4 rows and a host slot.
        price_growth(&mut domain, &mut bindings, misses_at(4, 100, 4));
        assert_eq!(paying(&domain), [Device, Host]);
        assert_eq!(domain.grow_expert_bank(Host, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Host));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { holding: crate::BankHolding::Host, to: 4, .. }));
        assert_eq!(paying(&domain), [Device], "the device's turn; the tier holds the rest");
        assert_eq!(domain.grow_expert_bank(Device, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { holding: crate::BankHolding::Experts, to: 4, .. }));
        assert!(domain.expert_bank_growths().unwrap().is_empty(), "both saturated");
    }
}

/// The host tier is the inclusive copy of every expert: its step to every
/// expert pays wherever its store has room, before any miss record or
/// resident step prices the device's, and while a prompt's places want
/// device memory, which holds the device's step alone.
#[test]
fn the_host_tier_grows_to_every_expert_unpriced_and_beside_a_places_want() {
    use crate::planning::BankTier::{Bank as Device, Host};
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 1);
        domain.family.bank.as_mut().unwrap().host_ceiling = 6;
        assert_eq!(paying(&domain), [Host], "no record, no resident step: the tier alone");
        domain.places_want = Some(super::state::PlacesWant { request: RequestId(80), places: 3, slots: 2 });
        price_growth(&mut domain, &mut bindings, misses_at(4, 100, 4));
        assert_eq!(paying(&domain), [Host], "a want holds the device's step, not the tier's");
        assert_eq!(domain.grow_expert_bank(Host, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Host));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { holding: crate::BankHolding::Host, to: 6, .. }));
        domain.places_want = None;
        assert_eq!(paying(&domain), [Device], "the tier holds every expert");
    }
}

/// Each store's room is its own: a device store with no room left leaves
/// a host store's whole room to a host step, where the one least room of
/// every store held both back.
#[test]
fn a_full_device_leaves_the_host_stores_room() {
    const GB: u64 = 1_000_000_000;
    let [device, host] = [(); 2].map(|()| seismic::Devices::discover(&[]).unwrap().topology().host_ram().id());
    let domain = |role, ceiling| DomainPace {
        role,
        headroom: ceiling,
        planning: 0,
        emergency: 0,
        ceiling,
        charged: 0,
        band: MemoryBand::Normal,
        unseen: 0,
        available: 0,
        base: (Instant::now(), 0),
        rate: 0.0,
        fastest: 0.0,
        fluctuation: Fluctuation::default(),
    };
    let mut pace = MemoryPace::new();
    pace.domains.insert(device, domain(DomainRole::Allocation, 0));
    pace.domains.insert(host, domain(DomainRole::Staging, 20 * GB));
    assert_eq!(pace.growth_room(device), Some((0, 0)));
    assert_eq!(pace.growth_room(host), Some((20 * GB, 0)));
}

/// A wake no one waits on.
fn unwatched() -> CompletionWake {
    CompletionWake::new(|| {})
}

/// The flight boundaries of a running resize, until it finishes.
fn finish(domain: &mut ExecutorDomain<TestFamily>) -> crate::BankResized {
    loop {
        if let Some((finished, _)) = domain.advance_expert_bank(unwatched()).unwrap() {
            return finished;
        }
    }
}

/// A head's own weights are what the residency cache it inherits holds
/// beyond the target's imports: a streamed target's bank, charged to the
/// target but never an import of the cache, does not count against them
/// (counted, a head beside a streamed target refused the load).
#[test]
fn a_heads_own_weights_leave_out_the_targets_bank() {
    for backend in bank_backends() {
        let (domain, _) = banked(backend, 4, 2);
        let charged = domain.family.target_weight_bytes().unwrap();
        let imports = domain.family.target_cached_bytes().unwrap();
        assert!(charged > 0, "the bank is charged to the target");
        assert_eq!(imports, 0, "the bank is no import of the cache");
        // A cache of the target's imports and a head of half what the bank
        // charges, which counting the bank would take for inherited.
        let head = charged / 2;
        assert_eq!(super::beyond_inherited(imports + head, imports), Ok(head));
        assert!(super::beyond_inherited(imports + head, charged + imports).is_err());
    }
}

/// A holding no resize has measured takes its first growth step, to its
/// sized target: the most slots within the store's room up to where the
/// record's misses stop falling, never the bank's every row past them. The
/// step measures what a resize holds the submitting thread. After that a
/// step grows only where its decode time saved over the record's decode
/// exceeds what the holding's last resize held the thread; one that does
/// not pay grows nothing, and a saturated holding has no step.
#[test]
fn a_holding_grows_once_unmeasured_then_only_where_its_saving_exceeds_its_resize() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 1);
        // 100 steps of 4 misses a step at 2 slots, none at 3: the misses stop
        // falling at 3 slots, of the bank's 4 rows.
        price_growth(&mut domain, &mut bindings, misses_at(2, 100, 4));
        assert_eq!(domain.resize_held.last.get(&crate::BankHolding::Experts), None);
        assert!(device_pays(&domain), "an unmeasured holding takes its first step");
        assert_eq!(domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { from: 1, to: 3, .. }), "the sized target");
        assert!(domain.resize_held.last.get(&crate::BankHolding::Experts).is_some_and(|&held| held < 0.4));
        assert!(!device_pays(&domain), "saturated: a fourth slot saves nothing");
        // Misses at 3 slots, none at 4: the fourth slot saves about 0.4 s, which
        // outweighs the resize; a resize held longer than that is not paid.
        price_growth(&mut domain, &mut bindings, misses_at(3, 100, 4));
        assert!(device_pays(&domain), "0.4 s saved outweighs the resize");
        domain.resize_held.last.insert(crate::BankHolding::Experts, 1.);
        assert!(!device_pays(&domain), "a step that does not pay grows nothing");
    }
}

/// A load priced before the executor's costs were stored grows nothing;
/// priced again once they are, on the same live domain, its bank grows by
/// the rule as a load priced at its start would.
#[test]
fn a_live_load_priced_again_on_newly_stored_costs_grows() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 2);
        domain.family.bank.as_mut().unwrap().record = Some(misses_at(3, 100, 4));
        let unpriced = crate::ExpertMeasures {
            lines: crate::assessment::Priced::Measured(std::sync::Arc::from([])),
            store: None,
            resident: crate::assessment::Priced::Pending("the executor's costs are not measured yet".into()),
            pass: crate::assessment::Priced::Pending("unpriced".into()),
            moves: crate::assessment::Priced::Pending("unmeasured".into()),
            tier_copy: crate::assessment::Priced::Measured(None),
            convert: crate::assessment::Priced::Measured(None),
        };
        domain.measure_experts(unpriced, &mut bindings).unwrap();
        assert!(!device_pays(&domain), "an unpriced load grows nothing");
        price_growth(&mut domain, &mut bindings, misses_at(3, 100, 4));
        assert!(device_pays(&domain));
        assert_eq!(domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { to: 4, .. }));
    }
}

/// The measurement job's room is made by the bank's shrink: a claim
/// beyond what opportunistic growth may take is short by the rest, which a
/// shrink of that many bytes frees, unless the rest is more than the bank
/// holds above its floor, which no shrink frees; while the claim holds, it is held out
/// of what opportunistic growth may take, so the step the record says pays
/// finds no room, and once released the bank grows again.
#[test]
fn a_measurement_claim_shrinks_the_bank_and_holds_its_growth() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        price_growth(&mut domain, &mut bindings, misses_at(3, 100, 4));
        // A probe in the Reclaim band reads memory as any does.
        match domain.probe_memory(std::time::Duration::from_millis(100)) {
            Ok(()) | Err(DomainError::Reclaim(_)) => {}
            Err(error) => panic!("{error}"),
        }
        // What opportunistic growth may take now, read through a claim no
        // device holds.
        let beyond = u64::MAX / 4;
        domain.claim_measurement_room(beyond);
        let Some(crate::MeasurementRoom::Beyond { short, releasable }) = domain.measurement_room().unwrap() else {
            panic!("a claim beyond the device is beyond the bank's floor")
        };
        // Three of its four slots are above its floor of one: three quarters of
        // its records' charge.
        let residency = domain.expert_residency().unwrap().unwrap();
        let charged = residency.stores.iter().map(|(_, bytes)| bytes).sum::<u64>();
        assert!(charged >= 4 * row);
        assert_eq!(releasable, charged * 3 / 4, "the bank holds its slots above its floor");
        let room = beyond - short;
        domain.claim_measurement_room(room);
        assert_eq!(domain.measurement_room().unwrap(), Some(crate::MeasurementRoom::Made));
        domain.claim_measurement_room(room + releasable + 1);
        assert_eq!(
            domain.measurement_room().unwrap(),
            Some(crate::MeasurementRoom::Beyond { short: releasable + 1, releasable })
        );
        domain.claim_measurement_room(room + row);
        assert_eq!(domain.measurement_room().unwrap(), Some(crate::MeasurementRoom::Short(row)));
        assert_eq!(domain.shrink_expert_bank_by(&mut bindings, row, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Shrunk { from: 4, to: 3, released, .. } if released == row));
        // A fourth slot would save the decode about 0.4 s, but the claim holds
        // the room the shrink made: no step fits the room left.
        assert!(!device_pays(&domain), "the claimed room is not the bank's");
        assert_eq!(domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(), BankStep::AtLimit);
        domain.release_measurement_room();
        assert_eq!(domain.measurement_room().unwrap(), None);
        assert_eq!(domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank), "released, the bank grows again");
    }
}

/// A measurement's claim ends the bank's running growth at its next slot,
/// none joined, and no growth is offered while it stands: held, the
/// growth's slots would wait for the claim's release, and the claim is
/// decided only once no resize runs. Released, the bank grows again.
#[test]
fn a_measurement_claim_stops_the_banks_growth() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 2);
        price_growth(&mut domain, &mut bindings, misses_at(3, 100, 4));
        assert_eq!(
            domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(),
            BankStep::Begun(crate::planning::BankTier::Bank)
        );
        domain.claim_measurement_room(0);
        assert!(matches!(finish(&mut domain), crate::BankResized::Stopped { at: 2, .. }), "stopped before its first slot");
        assert!(domain.expert_bank_growths().unwrap().is_empty(), "nothing grows beside the claim");
        domain.release_measurement_room();
        assert!(device_pays(&domain), "released, the step pays again");
        assert_eq!(
            domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(),
            BankStep::Begun(crate::planning::BankTier::Bank)
        );
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { from: 2, to: 4, .. }));
    }
}

/// A prompt's places want holds the bank's growth while its prompt has
/// chunks in hand, and is retired once the request's group holds none (it
/// decodes): another request's group leaves it, its own decode ends it.
#[test]
fn a_places_want_ends_with_its_prompt() {
    // Three slots of four: a fourth saves the decode its misses.
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 3);
        price_growth(&mut domain, &mut bindings, misses_at(3, 100, 4));
        assert!(device_pays(&domain));
        let (prompt, other) = (RequestId(80), RequestId(81));
        domain.places_want = Some(super::state::PlacesWant { request: prompt, places: 3, slots: 2 });
        assert!(domain.expert_bank_growths().unwrap().is_empty(), "a want holds the bank's growth");
        domain.grant_group(&[forward(other, 0)]).unwrap();
        assert!(domain.places_want.is_some(), "another request's group leaves the want");
        domain.grant_group(&[forward(prompt, 4)]).unwrap();
        assert!(domain.places_want.is_none(), "the prompt's own group past its chunks ends it");
        assert!(device_pays(&domain));
    }
}

/// A launch's duration prices the draft width only where no expert bank
/// resize ran across it: one with none is timed as before; one a growth
/// began across, or submitted while that growth runs, is not, so a resize
/// changes no drafting policy.
#[test]
fn a_launch_a_bank_resize_ran_across_is_not_timed() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 2);
        price_growth(&mut domain, &mut bindings, misses_at(3, 100, 4));
        let request = RequestId(90);
        open(&mut domain, &mut bindings, request);
        let step = |domain: &mut ExecutorDomain<TestFamily>, bindings, position, during: &dyn Fn(&mut ExecutorDomain<TestFamily>)| {
            let flight = submit_reserved_target(domain, bindings, vec![forward(request, position)]);
            during(domain);
            let (pending, bindings) = finish_one(domain, flight);
            let timed = pending.launch_duration().is_some();
            domain.reconcile(pending, PhysicalDecision { accepted_rows: 1 }).unwrap();
            (timed, bindings)
        };
        let quiet = |_: &mut ExecutorDomain<TestFamily>| {};
        // The first launch records its graphs.
        let (_, bindings) = step(&mut domain, bindings, 0, &quiet);
        let (timed, bindings) = step(&mut domain, bindings, 1, &quiet);
        assert!(timed, "a launch nothing ran across is timed");
        let grow = |domain: &mut ExecutorDomain<TestFamily>| {
            assert_eq!(domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        };
        let (timed, bindings) = step(&mut domain, bindings, 2, &grow);
        assert!(!timed, "a growth began across it");
        let (timed, bindings) = step(&mut domain, bindings, 3, &quiet);
        assert!(!timed, "submitted while the growth runs");
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { to: 4, .. }));
        let (timed, _) = step(&mut domain, bindings, 4, &quiet);
        assert!(timed, "timed again once the growth finished");
    }
}

/// A growth decided while a step runs waits for no gap: its slots join at a
/// boundary steps still run at, and it finishes there.
#[test]
fn a_growth_decided_mid_flight_joins_beside_the_running_steps() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 2);
        price_growth(&mut domain, &mut bindings, misses_at(3, 100, 4));
        domain.family.bank.as_ref().unwrap().running_steps(1);
        assert_eq!(domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(domain.advance_expert_bank(unwatched()).unwrap().is_none(), "joined while a step runs");
        assert_eq!(domain.expert_bank_slots(), Some(4));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { to: 4, .. }));
    }
}

/// A shrink whose queue drained with a step still running is deferred, not
/// refused: nothing begins, and the next boundary in a gap begins it.
#[test]
fn a_shrink_with_a_step_still_running_is_deferred_to_the_gap() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        domain.family.bank.as_ref().unwrap().running_steps(1);
        assert_eq!(domain.shrink_expert_bank_by(&mut bindings, row, unwatched()).unwrap(), BankStep::Deferred);
        assert_eq!(domain.expert_bank_slots(), Some(4));
        domain.family.bank.as_ref().unwrap().running_steps(0);
        assert_eq!(domain.shrink_expert_bank_by(&mut bindings, row, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Shrunk { to: 3, .. }));
    }
}

/// Under pipelined decode a growth breaks no lookahead: it waits for no gap,
/// so a successor queues behind each flight while it runs, as without one.
#[test]
fn a_growth_keeps_the_pipelined_lookahead() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked_with(backend, 4, 2, true);
        let request = RequestId(80);
        open(&mut domain, &mut bindings, request);
        domain.plan_successors(vec![prefill(request, 1)]);
        let flight = submit_reserved_target(&mut domain, bindings, vec![prefill(request, 0)]);
        let (pending, mut bindings) = finish_one(&mut domain, flight);
        domain.reconcile(pending, PhysicalDecision { accepted_rows: 1 }).unwrap();
        assert_eq!(domain.queue.len(), 1, "with no growth, the next chunk queues");
        // A growth begun while that chunk runs.
        price_growth(&mut domain, &mut bindings, misses_at(3, 100, 4));
        domain.family.bank.as_ref().unwrap().running_steps(1);
        assert_eq!(domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        domain.plan_successors(vec![prefill(request, 2)]);
        let flight = submit_reserved_target(&mut domain, bindings, vec![prefill(request, 1)]);
        let (pending, _) = finish_one(&mut domain, flight);
        domain.reconcile(pending, PhysicalDecision { accepted_rows: 1 }).unwrap();
        assert_eq!(domain.queue.len(), 1, "a successor queues beside a running growth");
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { to: 4, .. }));
        assert_eq!(domain.resume_state(request).unwrap().target.position(), 2, "the request ran on unaffected");
    }
}

/// A growth that pays stops at the planning line: with less than its
/// step's bytes left above the line, the heap admits no step and the bank
/// keeps its slots; with more, it grows. The host's memory is held
/// still (`StillHost`), so the line is where the heap's reading puts it.
#[test]
fn a_paying_growth_stops_at_the_line() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 2, 2);
        price_growth(&mut domain, &mut bindings, misses_at(1, 100, 4));
        assert_eq!(domain.shrink_expert_bank(&mut bindings, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        finish(&mut domain);
        assert!(device_pays(&domain));
        let resident = &**domain.experts.as_ref().unwrap().resident.measured().unwrap();
        let crate::Growth { kind, bytes: step, .. } =
            domain.family.expert_bank().unwrap().growth(Some(resident), &|_| true, crate::planning::BankTier::Bank, u64::MAX).unwrap().unwrap();
        // What the heap grants now, by whichever constraint binds: its refusal
        // of more than any domain holds says.
        let available = match domain.memory.borrow_mut().claim(u64::MAX / 2, 0, HoldingClass::Live) {
            Err(ClaimRefusal::Deficit { available, .. }) => available,
            refused => panic!("the heap granted every byte: {refused:?}"),
        };
        // Claimed as a reservation's, the bytes take the store's room itself,
        // which `available` reads: the free space of the device's blocks serves
        // none of them.
        let leaving = |domain: &ExecutorDomain<TestFamily>, left: u64| {
            domain.memory.borrow_mut().claim_reserved(kind, available - left, 0, HoldingClass::Live).unwrap()
        };
        let held = leaving(&domain, step / 2);
        assert!(matches!(
            domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()),
            Ok(BankStep::AtLimit) | Err(DomainError::Reclaim(_))
        ));
        assert_eq!(domain.expert_bank_slots(), Some(1));
        domain.memory.borrow_mut().release(held);
        let held = leaving(&domain, 2 * step);
        assert_eq!(domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { to: 2, .. }));
        domain.memory.borrow_mut().release(held);
    }
}

/// A holding updated while a bank shrink releases: the release falls off
/// the owner's thread, once the device work before it completes, and an
/// observation a claim takes meanwhile reads the lower charge before any
/// advance reports it. The update still classifies within the charge (the
/// static holding read after it), so no request fails because the bank
/// resized; every byte stays classified.
#[test]
fn a_holding_update_beside_a_shrinks_release_classifies_within_the_charge() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        let optional = seismic::Tensor::zeros(domain.domain.device(), seismic::Element::bool(), &[row]).unwrap();
        domain.family.optional = Some(optional);
        domain.sync_optional_holding().unwrap();
        assert_eq!(domain.shrink_expert_bank(&mut bindings, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        // The first advance installs; the release is the bank's own thread's.
        assert!(domain.advance_expert_bank(unwatched()).unwrap().is_none());
        domain.family.bank.as_ref().unwrap().release_now();
        // A claim's observation, which no holding follows.
        domain.memory.borrow_mut().refresh().unwrap();
        domain.sync_optional_holding().unwrap();
        assert_eq!(domain.memory().standing().unwrap().unattributed_bytes, 0);
        assert!(matches!(finish(&mut domain), crate::BankResized::Shrunk { from: 4, to: 3, released, .. } if released == row));
    }
}

/// The ladder's bank rung reaches the family's bank: each step begins at
/// one boundary, installs at the next and is credited exactly what Seismic
/// stopped charging once released, the static holding following; a step
/// its storage refuses breaks an invariant (a shrink needs no memory), and at
/// the floor a step begins none. A growth step backs what it claimed, the claim then returned.
#[test]
fn an_expert_bank_shrink_credits_what_seismic_releases() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        // A region pinning every row refuses a shrink beneath it.
        let tensor = domain.family.bank.as_ref().unwrap().tensor.clone();
        let pin = tensor.region(seismic::Element::bool(), &[4, row], 0).unwrap();
        assert_eq!(domain.shrink_expert_bank(&mut bindings, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        let refused = loop {
            match domain.advance_expert_bank(unwatched()) {
                Ok(None) => {}
                Ok(Some((finished, _))) => panic!("{backend:?}: a pinned shrink finished: {finished:?}"),
                Err(error) => break error,
            }
        };
        assert!(matches!(refused, DomainError::Invariant(_)), "{backend:?}: {refused}");
        drop((pin, tensor, domain, bindings));
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let before = domain.memory().standing().unwrap();
        for slots in [3, 2, 1] {
            assert_eq!(domain.shrink_expert_bank(&mut bindings, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
            assert!(domain.advance_expert_bank(unwatched()).unwrap().is_none(), "installed, not yet released");
            assert_eq!(
                domain.shrink_expert_bank(&mut bindings, unwatched()).unwrap(),
                BankStep::Running,
                "one resize at a time"
            );
            assert!(matches!(
                domain.advance_expert_bank(unwatched()).unwrap(),
                Some((crate::BankResized::Shrunk { from, to: at, released, .. }, _)) if at == slots && from == slots + 1 && released == row
            ));
            let residency = domain.expert_residency().unwrap().unwrap();
            assert_eq!((residency.slots, residency.disk_bytes), (slots, 4 - slots));
        }
        assert_eq!(domain.shrink_expert_bank(&mut bindings, unwatched()).unwrap(), BankStep::AtLimit);
        domain.refresh_memory().unwrap();
        let after = domain.memory().standing().unwrap();
        assert_eq!(before.classified_bytes - after.classified_bytes, 3 * row);
        assert_eq!(after.unattributed_bytes, 0);
        // Without a miss record there are no grounds to grow: the rule grows
        // nothing.
        assert!(!device_pays(&domain));
        // A growth step, priced on the measured resident step, backs exactly
        // what it claimed, the holding following: one step to every row, where
        // the record's misses never stop falling short of them.
        price_growth(&mut domain, &mut bindings, misses_at(4, 100, 4));
        assert_eq!(domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { from: 1, to: 4, grown, .. } if grown == 3 * row));
        assert_eq!(domain.grow_expert_bank(crate::planning::BankTier::Bank, unwatched()).unwrap(), BankStep::AtLimit);
        domain.refresh_memory().unwrap();
        let regrown = domain.memory().standing().unwrap();
        assert_eq!(regrown.classified_bytes, before.classified_bytes);
        assert_eq!(regrown.unattributed_bytes, 0);
    }
}

const PACE_GB: u64 = 1_000_000_000;

/// One reading of `store` with `headroom` bytes against a 10 GB planning
/// line, the process charging `charged`.
fn read_pace(pace: &mut MemoryPace, store: seismic::StoreId, headroom: u64, charged: u64, interval: Duration) {
    let topology = seismic::Devices::discover(&[]).unwrap().topology();
    let thresholds = crate::platform::DomainThresholds { planning_bytes: 10 * PACE_GB, emergency_bytes: 2 * PACE_GB };
    let reading = DomainReading {
        role: DomainRole::Allocation,
        domain: store,
        headroom_bytes: headroom,
        identity: topology.store_identity(topology.host_ram().id()).unwrap(),
        constraint: crate::platform::MemoryConstraint::HostRam,
        thresholds,
        ceiling_bytes: thresholds.ceiling(headroom),
        distress: None,
        held: None,
        displacement: None,
        band: if headroom <= thresholds.planning_bytes { MemoryBand::Reclaim } else { MemoryBand::Normal },
    };
    pace.observe(&[reading], &[(store, charged)], true, interval);
}

/// The host shows each release `lag` readings late, whatever `lag`: the bank
/// releases what the pressure needs rounded up to one rung and waits for the
/// rest, and pressure that comes after the release shows is answered in turn.
#[test]
fn a_release_the_host_shows_late_is_not_released_again() {
    const RUNG: u64 = 2 * PACE_GB;
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    // No span reaches the interval: the rate plays no part.
    let interval = Duration::from_secs(3600);
    for lag in 0..8 {
        let mut pace = MemoryPace::new();
        let (mut charged, mut releases) = (20 * PACE_GB, Vec::<(usize, u64)>::new());
        for at in 0..40 {
            let taken = if at < 20 { 3 } else { 6 } * PACE_GB;
            let shown: u64 = releases.iter().filter(|(when, _)| when + lag <= at).map(|(_, bytes)| bytes).sum();
            read_pace(&mut pace, store, (10 * PACE_GB + shown).saturating_sub(taken), charged, interval);
            if let Shortfall::Short(bytes) = pace.shortfall(store, charged) {
                let released = bytes.div_ceil(RUNG) * RUNG;
                charged -= released;
                releases.push((at, released));
            }
            let total: u64 = releases.iter().map(|(_, bytes)| bytes).sum();
            assert!(total >= taken && total <= taken.div_ceil(RUNG) * RUNG, "lag {lag}, reading {at}: {total} released for {taken}");
        }
    }
}

/// A release in small rungs the host shows at once is never credited, so each
/// next rung sees the whole of what is still short; one still unseen is, until
/// it shows.
#[test]
fn only_what_the_readings_have_yet_to_show_is_credited() {
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let interval = Duration::from_secs(3600);
    let mut pace = MemoryPace::new();
    read_pace(&mut pace, store, 7 * PACE_GB, 20 * PACE_GB, interval);
    read_pace(&mut pace, store, 8 * PACE_GB, 19 * PACE_GB, interval);
    assert_eq!(pace.shortfall(store, 19 * PACE_GB), Shortfall::Short(2 * PACE_GB), "shown at once");
    read_pace(&mut pace, store, 8 * PACE_GB, 18 * PACE_GB, interval);
    assert_eq!(pace.shortfall(store, 18 * PACE_GB), Shortfall::Short(PACE_GB), "a GB unseen");
    read_pace(&mut pace, store, 9 * PACE_GB, 18 * PACE_GB, interval);
    assert_eq!(pace.shortfall(store, 18 * PACE_GB), Shortfall::Short(PACE_GB), "shown now");
    // The host takes a GB with none unseen: all of it is short.
    read_pace(&mut pace, store, 8 * PACE_GB, 18 * PACE_GB, interval);
    assert_eq!(pace.shortfall(store, 18 * PACE_GB), Shortfall::Short(2 * PACE_GB));
}

/// A release after the reading a decision uses is credited too; a reading
/// above the line ends what is owed, so pressure after it stands alone; and a
/// lapsed credit leaves the reading alone.
#[test]
fn the_credit_counts_releases_after_the_reading_until_memory_is_above_the_line() {
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let interval = Duration::from_secs(3600);
    let mut pace = MemoryPace::new();
    read_pace(&mut pace, store, 7 * PACE_GB, 20 * PACE_GB, interval);
    assert_eq!(pace.shortfall(store, 20 * PACE_GB), Shortfall::Short(3 * PACE_GB));
    assert_eq!(pace.shortfall(store, 19 * PACE_GB), Shortfall::Short(2 * PACE_GB), "released after the reading");
    assert_eq!(pace.shortfall(store, 17 * PACE_GB), Shortfall::Covered);
    read_pace(&mut pace, store, 7 * PACE_GB, 17 * PACE_GB, interval);
    assert_eq!(pace.shortfall(store, 17 * PACE_GB), Shortfall::Covered, "unseen at this reading");
    read_pace(&mut pace, store, 11 * PACE_GB, 17 * PACE_GB, interval);
    assert_eq!(pace.shortfall(store, 17 * PACE_GB), Shortfall::Above);
    read_pace(&mut pace, store, 9 * PACE_GB, 17 * PACE_GB, interval);
    assert_eq!(pace.shortfall(store, 17 * PACE_GB), Shortfall::Short(PACE_GB), "the old release is not credited again");
    read_pace(&mut pace, store, 9 * PACE_GB, 16 * PACE_GB, interval);
    assert_eq!(pace.shortfall(store, 16 * PACE_GB), Shortfall::Covered);
    pace.lapse_credit();
    assert_eq!(pace.shortfall(store, 16 * PACE_GB), Shortfall::Short(PACE_GB));
}

/// The process's own release, unseen, is no taking by the host, and the
/// taking measured before it is not projected past it: the host that took
/// memory may have stopped. The next span measures it again.
#[test]
fn a_release_ends_the_measured_taking_until_a_span_after_it() {
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let mut pace = MemoryPace::new();
    for charged in [20, 17, 17] {
        read_pace(&mut pace, store, 7 * PACE_GB, charged * PACE_GB, Duration::ZERO);
        assert_eq!(pace.domains[&store].fastest, 0.0, "at a charge of {charged} GB");
    }
    read_pace(&mut pace, store, 6 * PACE_GB, 17 * PACE_GB, Duration::ZERO);
    assert!(pace.domains[&store].rate > 0.0, "the host took a GB");
    read_pace(&mut pace, store, 6 * PACE_GB, 15 * PACE_GB, Duration::ZERO);
    assert_eq!(pace.domains[&store].rate, 0.0, "released across this span");
}

/// Each domain's deficit is its own: a shrink of what is charged in a
/// domain 3 GB below its planning line covers 3 GB, one of what is charged
/// in a domain above its line nothing, and one in a domain no probe
/// observed nothing.
#[test]
fn each_domains_deficit_is_the_shrink_of_what_it_charges() {
    const GB: u64 = 1_000_000_000;
    // Three distinct stores: host RAM of three discoveries, each a revision
    // of its own.
    let [short, ample, unobserved] =
        [(); 3].map(|()| seismic::Devices::discover(&[]).unwrap().topology().host_ram().id());
    assert!(short != ample && ample != unobserved && short != unobserved);
    let domain = |headroom| DomainPace {
        role: DomainRole::Allocation,
        headroom,
        planning: 10 * GB,
        emergency: 2 * GB,
        ceiling: 0,
        charged: 0,
        band: MemoryBand::Normal,
        unseen: 0,
        available: 0,
        base: (Instant::now(), 0),
        rate: 0.0,
        fastest: 0.0,
        fluctuation: Fluctuation::default(),
    };
    let mut pace = MemoryPace::new();
    pace.domains.insert(short, domain(7 * GB));
    pace.domains.insert(ample, domain(12 * GB));
    assert_eq!(pace.shortfall(short, 0), Shortfall::Short(3 * GB));
    assert_eq!(pace.shortfall(ample, 0), Shortfall::Above);
    assert_eq!(pace.shortfall(unobserved, 0), Shortfall::Above);
    // A host in distress above the line: nothing is short, so nothing is
    // released for it a slot at a time.
    pace.domains.get_mut(&ample).unwrap().band = MemoryBand::Reclaim;
    assert_eq!(pace.shortfall(ample, 0), Shortfall::Covered);
}

/// A shrink covers its deficit now and what the host takes at its rate
/// until the next shrink takes effect: 3 GB/s carried 0.25 s is 750 MB.
#[test]
fn a_shrink_covers_what_is_taken_until_the_next_takes_effect() {
    let horizon = std::time::Duration::from_millis(250);
    assert_eq!(projected(40, 3e9, horizon), 750_000_040);
    assert_eq!(projected(40, 0.0, horizon), 40, "a host taking nothing: the deficit now");
}

/// The time to the next shrink's effect: the reading interval, and twice
/// the last finished release's latency (this shrink's, then the next one's
/// after the reading that decides it); the interval alone before any.
#[test]
fn the_shrink_horizon_runs_from_a_decision_to_the_next_effect() {
    let start = Instant::now();
    let at = |ms| start + std::time::Duration::from_millis(ms);
    let times = |decided, released| crate::domain::ResizeTimes {
        decided: at(decided),
        queued: Default::default(),
        planned: Default::default(),
        trips: Default::default(),
        moves: Default::default(),
        prepared: at(released),
        installed: at(released),
        released: at(released),
    };
    let ms = std::time::Duration::from_millis;
    let mut pace = MemoryPace::new();
    pace.reading_interval = ms(100);
    assert_eq!(pace.horizon(), ms(100));
    pace.released(&times(0, 120));
    assert_eq!(pace.horizon(), ms(340));
    pace.released(&times(130, 280));
    assert_eq!(pace.horizon(), ms(400));
}

/// The wait bound is what the host may take from the planning line to the
/// emergency reserve at the fastest rate it was measured taking: 8 GB at
/// 1 GB/s is 8 s. None in a domain seen taking nothing, nor in one that
/// grows nothing; none of a pace before any probe.
#[test]
fn the_wait_bound_is_the_reserve_between_the_lines_at_the_fastest_taking() {
    const GB: u64 = 1_000_000_000;
    let domain = |role, fastest| DomainPace {
        role,
        headroom: 0,
        planning: 10 * GB,
        emergency: 2 * GB,
        ceiling: 0,
        charged: 0,
        band: MemoryBand::Normal,
        unseen: 0,
        available: 0,
        base: (Instant::now(), 0),
        rate: 0.0,
        fastest,
        fluctuation: Fluctuation::default(),
    };
    assert_eq!(domain(DomainRole::Allocation, 1e9).wait_bound(), Some(8.0));
    assert_eq!(domain(DomainRole::Allocation, 0.0).wait_bound(), None);
    assert_eq!(domain(DomainRole::Staging, 1e9).wait_bound(), None);
    assert_eq!(MemoryPace::new().wait_bound(), None);
}

/// The fluctuation is the largest fall between consecutive readings while
/// the process's demand is unchanged: a rise, a pair across a change of
/// the charge and a reading while a resize runs measure nothing.
#[test]
fn the_fluctuation_is_the_largest_steady_fall_between_readings() {
    let mut fluctuation = Fluctuation::default();
    let read = |fluctuation: &mut Fluctuation, readings: &[(u64, u64, bool)]| {
        readings
            .iter()
            .map(|&(available, charged, settled)| {
                fluctuation.read(available, charged, settled);
                fluctuation.largest
            })
            .collect::<Vec<_>>()
    };
    // Falls of 2 and 1, a rise of 20: the largest fall stays.
    assert_eq!(read(&mut fluctuation, &[(30, 5, true), (28, 5, true), (48, 5, true), (47, 5, true)]), [0, 2, 2, 2]);
    // A fall of 9 across a change of the charge, then falls of 7 and 6 to
    // readings while a resize ran: none is steady.
    assert_eq!(read(&mut fluctuation, &[(38, 6, true), (31, 6, false), (25, 6, false)]), [2, 2, 2]);
    // Settled again, a fall of 3 counts.
    assert_eq!(read(&mut fluctuation, &[(25, 6, true), (22, 6, true)]), [2, 3]);
}

/// A failed head or vision flight returns its device error, and its
/// bindings go with it.
#[test]
fn pending_head_and_vision_device_failures_consume_the_bindings() {
    let failure = |lane: &'static str| {
        crate::DeviceError::Execution(format!("failed pending {lane} submission"))
    };

    let (mut head_domain, bindings) = fixture(None);
    let head_control = PendingControl::default();
    let mut head = HeadFlight {
        requests: Vec::new(),
        steps: 0,
        submission: PendingFailureSubmission::<CompletedHeadWork>::new(head_control.clone()).into(),
        started: Instant::now(),
        epoch: None,
        launch_trace: None,
        bindings,
    };
    assert!(!head.completion().is_complete());
    head_control.resolve(Err(failure("head")));
    assert!(matches!(
        head_domain.finish_head(head),
        Err(DomainError::Device(_))
    ));

    let (mut vision_domain, bindings) = fixture(None);
    let vision_control = PendingControl::default();
    let mut vision = VisionFlight {
        request: RequestId(41),
        image: ImageRef::new(vision_domain.resource_identity().clone(), 1).unwrap(),
        submission: PendingFailureSubmission::<CompletedVisionWork>::new(vision_control.clone()),
        started: Instant::now(),
        epoch: None,
        bindings,
    };
    assert!(!vision.completion().is_complete());
    vision_control.resolve(Err(failure("vision")));
    assert!(matches!(
        vision_domain.finish_vision(vision),
        Err(DomainError::Device(_))
    ));
}

#[test]
fn completed_head_and_vision_request_cancellation_restores_or_drops() {
    let (mut domain, mut bindings) = fixture(None);
    let request = RequestId(50);
    open(&mut domain, &mut bindings, request);

    // A head advance is independent from the accepted target lane. The target
    // state remaining resident must not be mistaken for a conflicting owner.
    let head_source = domain.target.get(&request).unwrap().checkpoint().fork();
    let head_advance = OwnedStateAdvance::begin(head_source, 1).ok().unwrap();
    domain
        .abort(PendingOperationOutcome {
            request,
            outcome: Outcome::Head {
                proposals: Vec::new(),
            },
            advance: Some(Advance::Step(head_advance)),
            primed: None,
            rows: 1,
            committed_rows: 0,
            kind: WorkKind::Decode,
            physical_duration: Duration::ZERO,
            timed: true,
            image: None,
        })
        .unwrap();
    assert_eq!(domain.target.get(&request).unwrap().position(), 0);
    assert_eq!(domain.head.remove(&request).unwrap().position(), 0);

    // Vision output is stateless until reconciliation installs it. Cancelling
    // the request simply drops the retained feature claim.
    let image = ImageRef::new(domain.resource_identity().clone(), 1).unwrap();
    let vision_features = FeatureRef::logical(
        domain.resource_identity().clone(),
        1,
        domain.definition.decoder.hidden as usize,
    )
    .unwrap();
    domain
        .abort(PendingOperationOutcome {
            request,
            outcome: Outcome::Encode {
                features: vision_features,
            },
            advance: None,
            primed: None,
            rows: 0,
            committed_rows: 0,
            kind: WorkKind::Prefill,
            physical_duration: Duration::ZERO,
            timed: true,
            image: Some(image),
        })
        .unwrap();
    assert_eq!(domain.target.get(&request).unwrap().position(), 0);
}

/// The tiny model with a minimal vision description: 2 x 2 patches of two
/// frames (24-wide patch rows), 2 x 2 cells, a 4 x 4 position table. Only
/// input preparation reads it; the domain encodes nothing here.
fn vision_definition() -> ModelDefinition {
    use magnitude_family_contracts::{
        ActivationDType, CellReduction, PositionSampling, VisionDescription, VisionMerger,
        VisionPositions, VisionPreprocessing, VisionResampling, VisionResize, VisionStem,
        WeightDescriptor,
    };
    let stored = |name: &str, shape: &[u64]| WeightDescriptor::stored(name, shape);
    let mut definition = tiny_definition();
    definition.vision = Some(VisionDescription {
        activation_dtype: ActivationDType::BF16,
        hidden: 128,
        output_hidden: definition.decoder.hidden,
        preprocessing: VisionPreprocessing {
            resize: VisionResize::PixelBounds {
                min_pixels: 16,
                max_pixels: 64,
            },
            resampling: VisionResampling::Bicubic,
            mean: [0.5; 3],
            std: [0.5; 3],
            channels: 3,
            patch: 2,
            merge: 2,
        },
        stem: VisionStem::Patch {
            frames: vec![
                stored("patch.0", &[128, 3, 2, 2]),
                stored("patch.1", &[128, 3, 2, 2]),
            ],
            bias: None,
            positions: Some(VisionPositions {
                table: stored("position", &[16, 128]),
                sampling: PositionSampling::AlignedCorners { side: 4 },
            }),
            norm: None,
            registers: None,
        },
        window: None,
        blocks: Vec::new(),
        merger: VisionMerger {
            norm: None,
            reduction: CellReduction::Concatenate,
            standardize: None,
            projection_norm: None,
            stages: Vec::new(),
            output_norm: None,
        },
    });
    definition
}

/// A prompt of `count` `TEXT` rows with images placed at the given rows,
/// each named by identity. Every image is one 2 x 2-patch cell, so one
/// media row.
fn vision_input(placements: &[(usize, &str)], count: usize) -> PreparedModelInput {
    use magnitude_artifacts::{
        BoundaryRule, InputLayout, InputSpan,
        media::{DType, PreparedTensor},
    };
    use magnitude_family_contracts::{
        PreparedVisionInput, TokenPlan, VisionPositionBlend, VisionSpatialControls,
    };
    let definition = vision_definition();
    let image = || {
        let pixels = PreparedTensor::new(
            "pixel_values".into(),
            DType::F32,
            vec![4, 24],
            vec![0; 4 * 24 * 4],
        )
        .unwrap();
        let spatial = VisionSpatialControls::new(
            (0..4).collect(),
            vec![[0, 0]; 4],
            Some(VisionPositionBlend {
                indices: std::array::from_fn(|_| vec![0; 4]),
                coefficients: std::array::from_fn(|_| vec![0.25; 4]),
            }),
            Vec::new(),
        )
        .unwrap();
        PreparedVisionInput::new(&definition, [1, 2, 2], pixels, spatial).unwrap()
    };
    let mut tokens = vec![TEXT; count];
    let spans = placements
        .iter()
        .map(|&(start, identity)| {
            tokens[start] = MEDIA;
            InputSpan {
                start,
                end: start + 1,
                identity: identity.into(),
                boundaries: BoundaryRule::Causal,
                language_history: false,
            }
        })
        .collect();
    PreparedModelInput::new(
        &definition,
        TokenPlan::new(tokens, InputLayout::new(count, spans).unwrap()).unwrap(),
        (0..count as i32).map(|position| [position; 3]).collect(),
        count,
        placements
            .iter()
            .map(|&(_, identity)| (identity.to_owned(), image()))
            .collect(),
    )
    .unwrap()
}

/// An image encodes its declared cells within its device's bound, across
/// launches; one whose rows attend each other in the decoder fits one launch.
#[test]
fn an_image_encodes_its_declared_cells_and_one_launch_when_bidirectional() {
    use magnitude_family_contracts::{MediaRowAttention, Operator, VisionResize};
    let any = crate::MAX_IMAGE_CELLS;
    assert_eq!(crate::image_cell_limit(&tiny_definition(), 2, any).unwrap(), 0);
    let mut definition = vision_definition();
    assert_eq!(crate::image_cell_limit(&definition, 2, any).unwrap(), 4);
    let vision = definition.vision.as_mut().unwrap();
    vision.preprocessing.resize = VisionResize::PixelBounds {
        min_pixels: 16,
        max_pixels: 1 << 30,
    };
    assert_eq!(
        crate::image_cell_limit(&definition, 512, any).unwrap(),
        crate::MAX_IMAGE_CELLS
    );
    for block in &mut definition.decoder.blocks {
        for sublayer in &mut block.sublayers {
            if let Operator::Attention(attention) = &mut sublayer.op {
                attention.media_rows = MediaRowAttention::Bidirectional;
            }
        }
    }
    assert_eq!(crate::image_cell_limit(&definition, 512, any).unwrap(), 512);
}

/// A CPU encodes an image in one launch it completes during its submission:
/// its images are bounded to [`crate::CPU_IMAGE_CELLS`], a GPU's to
/// [`crate::MAX_IMAGE_CELLS`].
#[test]
fn a_cpu_encodes_fewer_image_cells_than_a_gpu() {
    use magnitude_family_contracts::VisionResize;
    use seismic::BackendName;
    assert!(crate::CPU_IMAGE_CELLS < crate::MAX_IMAGE_CELLS);
    let mut definition = vision_definition();
    definition.vision.as_mut().unwrap().preprocessing.resize = VisionResize::PixelBounds {
        min_pixels: 16,
        max_pixels: 1 << 30,
    };
    for (backend, cells) in [
        (BackendName::Cpu, crate::CPU_IMAGE_CELLS),
        (BackendName::Metal, crate::MAX_IMAGE_CELLS),
        (BackendName::Cuda, crate::MAX_IMAGE_CELLS),
        (BackendName::Vulkan, crate::MAX_IMAGE_CELLS),
    ] {
        assert_eq!(crate::max_image_cells(backend), cells, "{backend:?}");
        assert_eq!(
            crate::image_cell_limit(&definition, 512, crate::max_image_cells(backend)).unwrap(),
            cells,
            "{backend:?}"
        );
    }
}

const TEXT: crate::TokenId = crate::TokenId(5);
const MEDIA: crate::TokenId = crate::TokenId(9);

/// The image each encode names, in order.
fn encoded_images(operations: &[Operation]) -> Vec<ImageRef> {
    operations
        .iter()
        .map(|operation| match operation {
            Operation::Encode { image, .. } => image.clone(),
            _ => panic!("opening a request yields only encodes"),
        })
        .collect()
}

/// Install `image`'s encoder output (one feature row per cell) as a
/// reconciled encode, and return it.
fn reconcile_encode<F: ProgramFamily>(
    domain: &mut ExecutorDomain<F>,
    request: RequestId,
    image: ImageRef,
) -> FeatureRef {
    let tensor = Tensor::zeros(
        domain.domain.device(),
        seismic::Element::f32(),
        &[
            (image.patches() / 4) as u64,
            domain.definition.decoder.hidden,
        ],
    )
    .unwrap();
    let features = domain.domain.publish_features(tensor).unwrap();
    domain
        .reconcile(
            PendingOperationOutcome {
                request,
                outcome: Outcome::Encode {
                    features: features.clone(),
                },
                advance: None,
                primed: None,
                rows: 0,
                committed_rows: 0,
                kind: WorkKind::Prefill,
                physical_duration: Duration::ZERO,
                timed: true,
                image: Some(image),
            },
            PhysicalDecision { accepted_rows: 0 },
        )
        .unwrap();
    features
}

/// Where each conditioning slice of the prompt rows from `position` reads
/// from: its destination row, source features, and source row range.
fn placements<F: ProgramFamily>(
    domain: &ExecutorDomain<F>,
    request: RequestId,
    position: usize,
) -> Vec<(usize, FeatureRef, usize, usize)> {
    let tokens = domain.input[&request].input.tokens()[position..].to_vec();
    let (_, slices) = domain.input_rows(request, position, &tokens).unwrap();
    slices
        .into_iter()
        .map(|slice| {
            (
                slice.destination,
                slice.source.features,
                slice.source.start,
                slice.source.count,
            )
        })
        .collect()
}

/// One image placed twice in one prompt is one image: it is admitted,
/// encoded once, and both placements read its features. A distinct image
/// between them keeps its own encode.
#[test]
fn a_repeated_image_is_encoded_once_and_placed_at_every_occurrence() {
    let (mut domain, mut bindings) = fixture(None);
    let request = RequestId(60);
    domain
        .install_input(
            request,
            vision_input(&[(1, "repeated"), (3, "other"), (5, "repeated")], 7),
        )
        .unwrap();
    assert_eq!(domain.input[&request].images.len(), 2);

    let images = encoded_images(&domain.open_state(&mut bindings, request, None).unwrap());
    let [first, second] = images.as_slice() else {
        panic!("two distinct images form two encodes, not three")
    };
    assert_ne!(first, second);
    let features =
        [first, second].map(|image| reconcile_encode(&mut domain, request, image.clone()));
    let repeated_features = if domain.input[&request].images["repeated"].image == *first {
        &features[0]
    } else {
        &features[1]
    };
    let placed = placements(&domain, request, 0);
    assert_eq!(placed.len(), 3);
    assert_eq!(
        placed
            .iter()
            .map(|(row, _, start, count)| (*row, *start, *count))
            .collect::<Vec<_>>(),
        vec![(1, 0, 1), (3, 0, 1), (5, 0, 1)]
    );
    assert!(placed[0].1 == *repeated_features && placed[2].1 == *repeated_features);
    assert!(placed[1].1 != *repeated_features);
}

/// A later turn re-sends the conversation, so an image from an earlier turn
/// placed again is a repeat within the new prompt. Resuming past the first
/// placement encodes the image once more for the remaining placement only.
#[test]
fn an_image_repeated_across_turns_resumes_and_encodes_its_remaining_placement() {
    let (mut domain, mut bindings) = fixture(None);

    // Turn one: prefill the whole prompt, then take its resume state.
    let first = RequestId(61);
    domain
        .install_input(first, vision_input(&[(1, "shared")], 3))
        .unwrap();
    let [encode] = encoded_images(&domain.open_state(&mut bindings, first, None).unwrap())
        .try_into()
        .unwrap_or_else(|_| panic!("one image forms one encode"));
    reconcile_encode(&mut domain, first, encode);
    let prefill_rows = |position: usize, tokens: Vec<crate::TokenId>| Operation::Forward {
        request: first,
        kind: WorkKind::Prefill,
        committed: tokens.len(),
        tokens,
        position,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        prime: None,
    };
    for (position, tokens) in [(0, vec![TEXT, MEDIA]), (2, vec![TEXT])] {
        let rows = tokens.len();
        let flight =
            submit_reserved_target(&mut domain, bindings, vec![prefill_rows(position, tokens)]);
        let (pending, returned) = finish_one(&mut domain, flight);
        domain
            .reconcile(
                pending,
                PhysicalDecision {
                    accepted_rows: rows,
                },
            )
            .unwrap();
        bindings = returned;
    }
    let resume = domain.resume_state(first).unwrap();
    assert_eq!(resume.position(), 3);

    // Turn two: the same image again after the resumed prefix.
    let second = RequestId(62);
    domain
        .install_input(second, vision_input(&[(1, "shared"), (4, "shared")], 6))
        .unwrap();
    let [encode] = encoded_images(
        &domain
            .open_state(&mut bindings, second, Some(&resume))
            .unwrap(),
    )
    .try_into()
    .unwrap_or_else(|_| panic!("the unresumed placement forms one encode"));
    let features = reconcile_encode(&mut domain, second, encode);
    let placed = placements(&domain, second, 3);
    let [(row, source, start, count)] = placed.as_slice() else {
        panic!("one placement past the resumed position")
    };
    assert_eq!((*row, *start, *count), (1, 0, 1));
    assert!(*source == features);
}

/// Publishing target output must not synchronously wait for head completion.
/// Its second flight keeps the completion wake and preserves device failures.
#[test]
fn deferred_priming_keeps_pending_completion_and_reports_device_failure() {
    let (domain, bindings) = fixture(None);
    let control = PendingControl::default();
    let mut flight = PrimingCompletionFlight {
        work: PrimingFlight {
            request: RequestId(88),
            submission: PendingFailureSubmission::new(control.clone()).into(),
            continuation: None,
        },
        target_end: 1,
        bindings,
        started: Instant::now(),
    };
    assert!(!flight.completion().is_complete());
    let woke = Arc::new(AtomicBool::new(false));
    let observed = woke.clone();
    flight.completion().notify(CompletionWake::new(move || {
        observed.store(true, Ordering::SeqCst);
    }));
    assert!(!woke.load(Ordering::SeqCst));
    control.resolve(Err(crate::DeviceError::Execution(
        "deferred entry failed".into(),
    )));
    assert!(woke.load(Ordering::SeqCst));
    assert!(flight.completion().is_complete());
    let mut domain = domain;
    assert!(matches!(
        domain.finish_priming(flight),
        Err(DomainError::Device(_))
    ));
}

/// Form real owned state advances and a checked head launch, with controlled
/// completion instead of numerical kernels. This fixture borrows the plain
/// model's readout pool for the head's leases; no graph consumes those buffers.
/// The target has two final prompt rows and the head has their one known
/// successor. The scheduler's final-prompt detection is outside this fixture.
fn target_with_pending_priming(
    request: RequestId,
) -> (
    ExecutorDomain<TestFamily>,
    TargetFlight<TestFamily>,
    PendingControl,
) {
    let (mut domain, mut bindings) = fixture_on(
        None,
        false,
        true,
        platform::DeviceRequest::Backend(seismic::BackendName::Cpu),
    );
    open(&mut domain, &mut bindings, request);
    let mut operation = prefill(request, 0);
    if let Operation::Forward {
        tokens, committed, ..
    } = &mut operation
    {
        tokens.push(TokenId(2));
        *committed = 2;
    }
    // Target submission precedes its head submission, just as in production.
    // This controlled target is already ready and has returned its workspace.
    let mut flight = submit_reserved_target(&mut domain, bindings, vec![operation]);
    let advance = TentativeAdvance::Accepted(
        OwnedStateAdvance::begin(domain.head.remove(&request).unwrap(), 1)
            .ok()
            .unwrap(),
    );
    let slot = domain
        .head_slot(request, &[TokenId(2)], 0, &[], &advance, 0)
        .unwrap();
    let batch = crate::batching::ValidatedHeadBatch::from_slots(
        &[slot],
        domain.definition.decoder.vocabulary as usize,
        domain.head_class_limits().unwrap(),
    )
    .unwrap();
    let row_bytes = domain.head_conditioning_bytes();
    let conditioning = FeatureRows::new(vec![0u8; row_bytes].into(), 1).unwrap();
    // The launch pool: only it holds any launch's projected rows.
    let pool = domain.resources.target_readout_graph(usize::MAX);
    let inputs = crate::HeadLaunchInputs::new(
        batch,
        vec![advance],
        crate::HeadConditioning::Rows(vec![conditioning]),
        true,
        pool.acquire_workspace().unwrap(),
        pool.acquire_output().unwrap(),
    );
    let launch = ValidatedHeadLaunch::new(
        inputs,
        domain.head_store.as_ref().unwrap(),
        domain.resource_identity(),
        row_bytes,
    )
    .unwrap_or_else(|(_, error)| panic!("controlled head launch: {error}"));
    let (core, workspace, output) = launch.into_submission_parts();
    let control = PendingControl::default();
    let head = ControlledHeadSubmission::Work(PendingSubmission {
        completion: PendingCompletion(control.clone()),
        core,
        workspace: (workspace, output),
        output: crate::programs::HeadOutput {
            selections: None,
            landed_reads: Vec::new(),
        },
    });
    assert!(flight.work.priming.is_none());
    flight.work.priming = Some(PrimingFlight {
        request,
        submission: head,
        continuation: None,
    });
    (domain, flight, control)
}

fn publish_before_priming(
    domain: &mut ExecutorDomain<TestFamily>,
    flight: TargetFlight<TestFamily>,
) -> (PendingOperationOutcome, PrimingCompletionFlight<TestFamily>) {
    let (outcomes, continuation) = domain.finish_target_for_publication(flight, true).unwrap();
    let [pending] = outcomes
        .try_into()
        .unwrap_or_else(|_| panic!("one target outcome"));
    let TargetContinuation::Priming(mut priming) = continuation else {
        panic!("the pending head must retain the binding right");
    };
    assert!(!priming.completion().is_complete());
    (pending, priming)
}

#[test]
fn deferred_priming_commits_after_target_publication_and_releases_leases() {
    let request = RequestId(93);
    let (mut domain, flight, control) = target_with_pending_priming(request);
    let held_outputs = domain.resources.target_readout_graph(usize::MAX).available_output();
    let (pending, mut priming) = publish_before_priming(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 2 })
        .unwrap();
    assert_eq!(domain.target[&request].position(), 2);
    assert!(!domain.head.contains_key(&request));
    assert!(
        domain.resume_state(request).is_err(),
        "no half-resolved checkpoint"
    );
    assert!(
        domain.release_state(&[request]).is_err(),
        "no eviction before head completion"
    );
    assert_eq!(
        domain.resources.target_readout_graph(usize::MAX).available_output(),
        held_outputs
    );
    let woke = Arc::new(AtomicBool::new(false));
    let observed = woke.clone();
    priming.completion().notify(CompletionWake::new(move || {
        observed.store(true, Ordering::SeqCst);
    }));
    control.resolve(Ok(()));
    assert!(woke.load(Ordering::SeqCst));
    let (finished, _, mut bindings) = domain.finish_priming(priming).unwrap();
    assert_eq!(finished, request);
    let checkpoint = domain.resume_state(request).unwrap();
    assert_eq!(checkpoint.target.position(), 2);
    assert_eq!(checkpoint.head.as_ref().unwrap().position(), 1);
    assert_eq!(
        domain.resources.target_readout_graph(usize::MAX).available_output(),
        held_outputs + 1
    );
    // The returned binding right is usable only after the second flight ends.
    domain.provision_open(&mut bindings).unwrap();
    domain.close(request).unwrap();
}

#[test]
fn deferred_priming_aborts_when_its_target_chunk_is_rejected() {
    let request = RequestId(94);
    let (mut domain, flight, control) = target_with_pending_priming(request);
    let (pending, priming) = publish_before_priming(&mut domain, flight);
    domain.abort(pending).unwrap();
    assert_eq!(domain.target[&request].position(), 0);
    assert!(domain.resume_state(request).is_err());
    control.resolve(Ok(()));
    let (_, _, bindings) = domain.finish_priming(priming).unwrap();
    let checkpoint = domain.resume_state(request).unwrap();
    assert_eq!(checkpoint.target.position(), 0);
    assert_eq!(checkpoint.head.as_ref().unwrap().position(), 0);
    // Aborting releases the tentative rows and leaves the target reusable.
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    let (pending, _) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    domain.close(request).unwrap();
}

/// Cancellation at the executor boundary must wait to close resident state.
/// The scheduler's cancel control itself is not exercised by this domain test.
#[test]
fn deferred_priming_prevents_cancelled_request_retirement_until_completion() {
    let request = RequestId(95);
    let (mut domain, flight, control) = target_with_pending_priming(request);
    let (pending, priming) = publish_before_priming(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 2 })
        .unwrap();
    assert_eq!(
        domain.close(request).unwrap_err(),
        "request has unresolved work"
    );
    assert!(domain.input.contains_key(&request));
    assert_eq!(domain.target[&request].position(), 2);
    control.resolve(Ok(()));
    let (_, _, mut bindings) = domain.finish_priming(priming).unwrap();
    domain.close(request).unwrap();
    assert!(!domain.input.contains_key(&request));
    assert!(!domain.target.contains_key(&request));
    assert!(!domain.head.contains_key(&request));
    // Retirement leaves both stores usable by a different request.
    let next = RequestId(96);
    open(&mut domain, &mut bindings, next);
    let fresh = domain.resume_state(next).unwrap();
    assert_eq!(fresh.target.position(), 0);
    assert_eq!(fresh.head.as_ref().unwrap().position(), 0);
    domain.close(next).unwrap();
}

#[test]
fn boundary_features_commit_fork_and_abort_with_numerical_state() {
    let (mut domain, mut bindings) = fixture(None);
    let request = RequestId(91);
    open(&mut domain, &mut bindings, request);
    let first = domain.allocate_boundary().unwrap();
    let attach = |pending: &mut PendingOperationOutcome, boundary: &FeatureRef| {
        let aggregate =
            FeatureRef::logical(boundary.domain().clone(), 1, boundary.allocation().width())
                .unwrap()
                .with_boundary(boundary.clone())
                .unwrap();
        pending.outcome = Outcome::Forward {
            rows: vec![RowResult {
                features: Some(aggregate),
                ..RowResult::default()
            }],
        };
    };
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    let (mut pending, mut bindings) = finish_one(&mut domain, flight);
    attach(&mut pending, &first);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    let checkpoint = domain.resume_state(request).unwrap();
    assert!(
        checkpoint
            .boundary
            .as_ref()
            .unwrap()
            .same_allocation(&first)
    );
    let fork = RequestId(92);
    domain
        .install_input(fork, PreparedModelInput::continuation_only())
        .unwrap();
    domain
        .open_state(&mut bindings, fork, Some(&checkpoint))
        .unwrap();
    domain.release_state(&[request]).unwrap();
    assert!(domain.input[&request].boundary.is_none());
    assert!(
        domain.input[&fork]
            .boundary
            .as_ref()
            .unwrap()
            .same_allocation(&first)
    );
    let second = domain.allocate_boundary().unwrap();
    for invalid in [false, true] {
        let flight = submit_reserved_target(&mut domain, bindings, vec![forward(fork, 1)]);
        let (mut pending, returned) = finish_one(&mut domain, flight);
        bindings = returned;
        attach(&mut pending, &second);
        if invalid {
            assert!(
                domain
                    .reconcile(pending, PhysicalDecision { accepted_rows: 0 })
                    .is_err()
            );
        } else {
            domain.abort(pending).unwrap();
        }
        let restored = domain.resume_state(fork).unwrap();
        assert_eq!(restored.position(), 1);
        assert!(restored.boundary.as_ref().unwrap().same_allocation(&first));
    }
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(fork, 1)]);
    let (mut pending, _) = finish_one(&mut domain, flight);
    attach(&mut pending, &second);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    assert!(
        domain
            .resume_state(fork)
            .unwrap()
            .boundary
            .as_ref()
            .unwrap()
            .same_allocation(&second)
    );
    assert!(
        checkpoint
            .boundary
            .as_ref()
            .unwrap()
            .same_allocation(&first)
    );
}

/// A forward of `values` at `position`, every row committed.
fn feed(request: RequestId, position: usize, values: &[u32]) -> Operation {
    Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: values.iter().map(|value| crate::TokenId(*value)).collect(),
        position,
        conditioning: None,
        demand: crate::batching::Demand::NONE,
        select: Vec::new(),
        committed: values.len(),
        prime: None,
    }
}

fn token_values(tokens: &[crate::TokenId]) -> Vec<u32> {
    tokens.iter().map(|token| token.0).collect()
}

/// A request's tokens follow its sequence as its state does: an aborted
/// step's tokens give way to the forward fed again at its position, and a
/// sequence resumed from another's state holds that state's tokens and feeds
/// on from them, leaving its source's unchanged.
#[test]
fn token_history_follows_rollback_and_fork() {
    let (mut domain, mut bindings) = fixture(None);
    let source = RequestId(81);
    open(&mut domain, &mut bindings, source);
    // The fixture launches at most two rows.
    let flight = submit_reserved_target(&mut domain, bindings, vec![feed(source, 0, &[10, 11])]);
    let (pending, returned) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 2 })
        .unwrap();
    let flight = submit_reserved_target(&mut domain, returned, vec![feed(source, 2, &[12])]);
    let (pending, returned) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    // A step past the accepted rows is fed, then aborted.
    let flight = submit_reserved_target(&mut domain, returned, vec![feed(source, 3, &[20, 21])]);
    let (pending, returned) = finish_one(&mut domain, flight);
    domain.abort(pending).unwrap();
    assert_eq!(domain.resume_state(source).unwrap().position(), 3);
    assert_eq!(
        token_values(domain.preceding_tokens(source, 3, 2).unwrap()),
        [11, 12]
    );
    let flight = submit_reserved_target(&mut domain, returned, vec![feed(source, 3, &[40])]);
    let (pending, mut bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    assert_eq!(
        token_values(domain.preceding_tokens(source, 4, 2).unwrap()),
        [12, 40]
    );
    // Nothing is known past the fed tokens.
    assert!(domain.preceding_tokens(source, 5, 2).is_err());

    let checkpoint = domain.resume_state(source).unwrap();
    let copy = RequestId(82);
    domain
        .install_input(copy, PreparedModelInput::continuation_only())
        .unwrap();
    domain
        .open_state(&mut bindings, copy, Some(&checkpoint))
        .unwrap();
    assert_eq!(
        token_values(domain.preceding_tokens(copy, 4, 8).unwrap()),
        [10, 11, 12, 40]
    );
    let flight = submit_reserved_target(&mut domain, bindings, vec![feed(copy, 4, &[50])]);
    let (pending, _) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    assert_eq!(
        token_values(domain.preceding_tokens(copy, 5, 2).unwrap()),
        [40, 50]
    );
    assert!(domain.preceding_tokens(source, 5, 1).is_err());
}

#[test]
fn a_forward_replaces_the_tokens_at_and_after_its_position() {
    use super::input::TokenHistory;
    let tokens = |values: &[u32]| values.iter().map(|value| crate::TokenId(*value)).collect::<Vec<_>>();
    let mut history = TokenHistory::default();
    history.feed(0, &tokens(&[1, 2, 3, 4])).unwrap();
    // A verification's rejected rows are fed again from the first rejected.
    history.feed(2, &tokens(&[9])).unwrap();
    assert_eq!(token_values(history.before(3).unwrap()), [1, 2, 9]);
    assert!(history.before(4).is_err());
    // No forward follows tokens never fed.
    assert!(history.feed(5, &tokens(&[7])).is_err());
    let resumed = TokenHistory::starting(history.before(2).unwrap());
    assert_eq!(token_values(resumed.before(2).unwrap()), [1, 2]);
}

/// The host's rows for a step (`NGramHash::rows` over the request's token
/// history) equal the rows the device's `ngram_rows` names from the step's
/// tokens and the bank's token window: through a prefill chunk, a
/// verification of which only some rows are accepted, and the decode that
/// continues from the tape version.
#[test]
fn host_rows_from_the_token_history_equal_the_device_rows() {
    use magnitude_family_contracts::NGramHash;
    use magnitude_kernels::ngram_rows;
    use seismic::{BackendName, Element, NativeSpecialization, SlabRegion, SlabTensor, Tensor};
    let hash = NGramHash {
        context: 2,
        heads_per_order: 8,
        multipliers: vec![(i64::MAX as u64 / 248_320) | 1, 1_000_003, (1 << 41) + 17],
        sizes: (0..16).map(|head| 20_000_003 + 2 * head).collect(),
        offsets: (0..16).map(|head| head * 20_000_040).collect(),
        reset: 7,
    };
    let (context, heads) = (2u64, 16usize);
    let tape = 3u64;
    let (banks, slab_banks) = (4u64, 2u64);
    let catalog = seismic::Devices::discover(&[]).unwrap();
    let devices = [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan, BackendName::Cpu]
        .into_iter()
        .filter(|&backend| crate::test_support::available(&catalog, backend))
        .map(|backend| {
            catalog
                .open_backend(backend)
                .unwrap_or_else(|error| panic!("an available {backend:?} executor opens: {error}"))
        })
        .collect::<Vec<_>>();
    let ints = |device: &seismic::Device, shape: &[u64], values: &[i32]| {
        Tensor::from_host(
            device,
            Element::i32(),
            shape,
            &values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let words = |device: &seismic::Device, values: &[u64]| {
        Tensor::from_host(
            device,
            Element::u64(),
            &[values.len() as u64],
            &values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let tokens = |values: &[u32]| values.iter().map(|value| crate::TokenId(*value)).collect::<Vec<_>>();
    // (position, tokens, the version read (bank, tape rows), stop, the bank
    // published, rows accepted).
    let steps: [(usize, Vec<u32>, (i32, i32), i32, i32, usize); 3] = [
        (0, vec![3, 7, 7, 12, 40, 41, 7, 9, 13], (0, 0), 9, 1, 9),
        (9, vec![22, 7, 31, 5], (1, 0), 1, 2, 2),
        (11, vec![17], (2, 1), 1, 3, 1),
    ];
    for device in devices {
        let backend = device.backend().as_str();
        let mut slabs = SlabTensor::new(
            &device,
            slab_banks,
            banks,
            vec![SlabRegion {
                element: Element::u32(),
                row_shape: vec![context + tape, 1],
            }],
        )
        .unwrap();
        for slab in 0..banks.div_ceil(slab_banks) {
            assert_eq!(slabs.add_slab().unwrap(), slab as usize);
            slabs
                .region_rows(0, slab * slab_banks, slab_banks)
                .unwrap()
                .write_from_host(&vec![0u8; (slab_banks * (context + tape) * 4) as usize])
                .unwrap();
        }
        let mut window = slabs.logical_region(0).unwrap();
        let implementation = seismic::generated::native_implementation::<ngram_rows::Entry>(&device)
            .unwrap()
            .unwrap();
        let statics = match backend {
            "cpu" => NativeSpecialization::new(),
            _ => NativeSpecialization::new()
                .with_static("C", context)
                .with_static("P", 8),
        };
        let kernel = ngram_rows::native_for_device(
            &device,
            &implementation.default_specialization(&statics).unwrap(),
        )
        .unwrap();
        let mut history = super::input::TokenHistory::default();
        for (position, fed, (source, taped), stop, target, accepted) in &steps {
            let fed = tokens(fed);
            let preceding = history
                .before(*position)
                .unwrap()
                .iter()
                .rev()
                .take(context as usize)
                .rev()
                .map(|token| u64::from(token.0))
                .collect::<Vec<_>>();
            let host = hash.rows(&preceding, &fed.iter().map(|token| u64::from(token.0)).collect::<Vec<_>>());
            let rows = fed.len();
            let result = kernel
                .call(ngram_rows::Args {
                    tokens: &ints(
                        &device,
                        &[rows as u64, 2],
                        &fed.iter().flat_map(|token| [token.0 as i32, 0]).collect::<Vec<_>>(),
                    ),
                    multipliers: &words(&device, &hash.multipliers),
                    sizes: &words(&device, &hash.sizes),
                    offsets: &words(&device, &hash.offsets),
                    segments: &ints(&device, &[2, 2], &[0, rows as i32, rows as i32, rows as i32]),
                    stop: &ints(&device, &[1], &[*stop]),
                    previous_bank: &ints(&device, &[1], &[*source]),
                    previous_tape: &ints(&device, &[1], &[*taped]),
                    following_bank: &ints(&device, &[1], &[*target]),
                    window: &mut window,
                    reset: hash.reset as i32,
                    slab_banks: slab_banks as u32,
                })
                .unwrap()
                .value;
            let device_rows = result
                .read_to_host()
                .unwrap()
                .chunks_exact(8)
                .map(|pair| u64::from(u32::from_le_bytes(pair[..4].try_into().unwrap())))
                .collect::<Vec<_>>();
            assert_eq!(device_rows.len(), rows * heads);
            assert_eq!(device_rows, host, "{backend}: step at {position}");
            // The step's tokens are fed; the next forward follows the
            // accepted ones.
            history.feed(*position, &fed).unwrap();
            history.before(position + accepted).unwrap();
        }
    }
}

/// A stored table of 4096 IQ4_NL rows 160 wide whose step 1 named row 1 for
/// batch slot 0 and row 3500 for slot 1, read after the artifact was cut
/// short under the open table: slot 1's row could not be read. The table
/// file's path, to remove, and the step's read.
fn failed_table_read(name: &str) -> (std::path::PathBuf, crate::programs::StoredRead) {
    use crate::stored_table::{StepRows, StoredTable};
    const ROWS: u64 = 4096;
    const ROW_BYTES: u64 = 90;
    let path = std::env::temp_dir().join(format!(
        "magnitude-domain-table-{name}-{}.bin",
        std::process::id()
    ));
    std::fs::write(&path, vec![0x11u8; (ROWS * ROW_BYTES) as usize]).unwrap();
    let device = seismic::Devices::discover(&[])
        .unwrap()
        .open_backend(seismic::BackendName::Cpu)
        .unwrap();
    let table = Arc::new(
        StoredTable::new(
            &device,
            seismic::Element::named("gguf_iq4_nl").unwrap(),
            seismic::Element::named("iq4g32s@rows16").unwrap(),
            ROWS,
            160,
            crate::stored_table::StoredSlots::Whole { records: 1, record_rows: 64 },
            crate::planning::DirectoryForm::Hashed,
            magnitude_artifacts::ArtifactReader::open(
                &path,
                magnitude_artifacts::CachePolicy::Uncached,
            )
            .unwrap(),
            0,
            None,
        )
        .unwrap(),
    );
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(2048 * ROW_BYTES)
        .unwrap();
    let step = table
        .queue(StepRows::Known(vec![vec![1], vec![3500]]))
        .unwrap();
    assert!(step.gate.wait(step.value, std::time::Duration::from_secs(10)).unwrap());
    // The device read the step's rows.
    step.read.signal(step.value).unwrap();
    (path, crate::programs::StoredRead::new(table, step.value))
}

/// Spec §7.3: a step whose stored-table rows could not be read for one of
/// its two sequences fails that sequence alone, with its typed error. Its
/// outputs are discarded and its accepted state returns, as an orphan's;
/// it is poisoned, so its next step fails too though that step reads
/// nothing that failed, until it closes. The other sequence commits both
/// steps.
#[test]
fn a_table_read_failure_fails_only_the_sequence_that_named_the_rows() {
    let (mut domain, mut bindings) = fixture(None);
    let (kept, failed) = (RequestId(91), RequestId(92));
    open(&mut domain, &mut bindings, kept);
    open(&mut domain, &mut bindings, failed);
    let (path, read) = failed_table_read("poison");
    STORED_READS.with(|reads| reads.borrow_mut().push(read));
    let mut bindings = bindings;
    for (step, position) in [(0, 0usize), (1, 1)] {
        let flight = submit_reserved_target(
            &mut domain,
            bindings,
            vec![feed(kept, position, &[10 + step]), feed(failed, 0, &[20])],
        );
        let (pending, returned) = domain.finish_target(flight).unwrap();
        bindings = returned;
        assert_eq!(pending.len(), 2);
        for pending in pending {
            if pending.request() == kept {
                assert!(matches!(pending.outcome(), Outcome::Forward { .. }));
                domain
                    .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
                    .unwrap();
            } else {
                let Outcome::WeightReadFailed(error) = pending.outcome() else {
                    panic!("step {step}: the failed sequence's outcome is not its failure");
                };
                assert!(error.detail.contains("3500"), "{error}");
                assert!(matches!(
                    domain.reconcile(pending, PhysicalDecision { accepted_rows: 1 }),
                    Err(DomainError::WeightRead(_))
                ));
            }
        }
        assert_eq!(domain.resume_state(kept).unwrap().position(), position + 1);
        assert_eq!(domain.resume_state(failed).unwrap().position(), 0);
    }
    // Closing the sequence clears its poison.
    domain.close(failed).unwrap();
    open(&mut domain, &mut bindings, failed);
    let flight = submit_reserved_target(&mut domain, bindings, vec![feed(failed, 0, &[20])]);
    let (pending, _) = finish_one(&mut domain, flight);
    assert!(matches!(pending.outcome(), Outcome::Forward { .. }));
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    std::fs::remove_file(path).unwrap();
}

/// A step whose expert reads failed fails the sequences of that step
/// alone, with the one read-failure error, and the next request succeeds:
/// its sequences commit nothing of the failed step, and a sequence opened
/// after it runs.
#[test]
fn an_expert_read_failure_fails_only_its_steps_sequences() {
    let (mut domain, mut bindings) = fixture(None);
    let (first, second, later) = (RequestId(81), RequestId(82), RequestId(83));
    open(&mut domain, &mut bindings, first);
    open(&mut domain, &mut bindings, second);
    EXPERT_STEPS.with(|steps| {
        *steps.borrow_mut() = Some(crate::experts::ExpertStep::failed(
            "expert record at 4096 of 1572864 bytes in model file 0: failed to fill whole buffer",
        ))
    });
    let flight = submit_reserved_target(
        &mut domain,
        bindings,
        vec![feed(first, 0, &[10]), feed(second, 0, &[20])],
    );
    let (pending, mut bindings) = domain.finish_target(flight).unwrap();
    assert_eq!(pending.len(), 2);
    for pending in pending {
        let Outcome::WeightReadFailed(error) = pending.outcome() else {
            panic!("a sequence of the failed step did not fail");
        };
        assert!(error.detail.contains("failed to fill"), "{error}");
        assert!(matches!(
            domain.reconcile(pending, PhysicalDecision { accepted_rows: 1 }),
            Err(DomainError::WeightRead(_))
        ));
    }
    assert_eq!(domain.resume_state(first).unwrap().position(), 0);
    open(&mut domain, &mut bindings, later);
    let flight = submit_reserved_target(&mut domain, bindings, vec![feed(later, 0, &[30])]);
    let (pending, _) = finish_one(&mut domain, flight);
    assert!(matches!(pending.outcome(), Outcome::Forward { .. }));
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
}

/// The planning fixture with its attention selecting one block of four rows
/// past the dense limit, by an indexer of two 64-wide heads, over a 64-token
/// context.
fn selected_definition() -> ModelDefinition {
    use magnitude_family_contracts::{
        HistoryReads, Indexer, Operator, RmsNorm, WeightDescriptor,
    };
    let mut definition = tiny_definition();
    definition.decoder.context_limit = 64;
    let Operator::Attention(attention) = &mut definition.decoder.blocks[0].sublayers[0].op else {
        panic!("the fixture's mixer is attention");
    };
    let hidden = definition.decoder.hidden;
    let norm = |name: &str| RmsNorm {
        weight: WeightDescriptor::stored(name, vec![64]),
        epsilon: 1e-6,
    };
    attention.reads = HistoryReads::Selected {
        block: 4,
        blocks: 1,
        indexer: Indexer {
            heads: 2,
            width: 64,
            query: WeightDescriptor::stored("indexer_q", vec![128, hidden]),
            key: WeightDescriptor::stored("indexer_k", vec![64, hidden]),
            query_norm: norm("indexer_q_norm"),
            key_norm: norm("indexer_k_norm"),
        },
    };
    definition
}

/// A selected read's launches carry its indexer's histories as the state
/// holds them: every row reads the raw keys of the three positions before
/// it and the blocks complete before it, writes its raw key and the block
/// it completes; a commit of an accepted prefix keeps the blocks that prefix
/// completes, so a rollback across a block boundary continues as a straight
/// run does. The graphs of every class (the indexer, the Segments decode,
/// the WorkList prefill) prepare and seal on the device.
#[test]
fn selected_histories_follow_blocks_across_rollback() {
    use magnitude_state::HistoryDomainId;
    const RAW: HistoryDomainId = HistoryDomainId(1);
    const BLOCKS: HistoryDomainId = HistoryDomainId(2);
    let (mut domain, mut bindings) = fixture_of(
        selected_definition(),
        limits(16, 1, false),
        None,
        false,
        platform::DeviceRequest::Automatic,
    );
    let blocks = |domain: &ExecutorDomain<TestFamily>, request| {
        domain
            .resume_state(request)
            .unwrap()
            .target
            .history_ranges(BLOCKS)
            .iter()
            .map(|(_, rows)| rows)
            .sum::<usize>()
    };
    let source = RequestId(91);
    open(&mut domain, &mut bindings, source);
    // Ten prompt rows complete blocks 0 and 1.
    let tokens = (0..10).collect::<Vec<u32>>();
    let flight = submit_reserved_target(&mut domain, bindings, vec![feed(source, 0, &tokens)]);
    let (pending, bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 10 })
        .unwrap();
    assert_eq!(blocks(&domain, source), 2);
    // Six rows from 10 complete blocks 2 and 3, then roll back whole: the
    // history keeps blocks 0 and 1.
    let flight = submit_reserved_target(
        &mut domain,
        bindings,
        vec![feed(source, 10, &[20, 21, 22, 23, 24, 25])],
    );
    let (pending, bindings) = finish_one(&mut domain, flight);
    domain.abort(pending).unwrap();
    assert_eq!(domain.resume_state(source).unwrap().position(), 10);
    assert_eq!(blocks(&domain, source), 2);
    // Two other rows from 10 (positions 10, 11) complete block 2 alone.
    let flight = submit_reserved_target(&mut domain, bindings, vec![feed(source, 10, &[40, 41])]);
    let (pending, bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 2 })
        .unwrap();
    assert_eq!(domain.resume_state(source).unwrap().position(), 12);
    assert_eq!(blocks(&domain, source), 3);
    // The raw keys of positions 9..12 remain for the next block.
    let checkpoint = domain.resume_state(source).unwrap();
    assert_eq!(
        checkpoint
            .target
            .history_ranges(RAW)
            .iter()
            .map(|(_, rows)| rows)
            .sum::<usize>(),
        3
    );
    // Single decode rows continue past the dense limit, completing block 3.
    let mut bindings = bindings;
    for (position, token) in (12..16).zip(30..) {
        let flight = submit_reserved_target(&mut domain, bindings, vec![feed(source, position, &[token])]);
        let (pending, returned) = finish_one(&mut domain, flight);
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
            .unwrap();
        bindings = returned;
    }
    assert_eq!(blocks(&domain, source), 4);
}
