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

/// The owner's own work between flights (memory observation, and what it
/// releases or grows) serves no request: a launch is charged its request's
/// host turn without it. The owner's span is long against the request's
/// own host work, so a charge that kept it would exceed both together.
#[test]
fn a_launch_is_not_charged_the_owners_work_in_its_requests_host_turn() {
    let (mut domain, mut bindings) = fixture(None);
    let request = RequestId(72);
    open(&mut domain, &mut bindings, request);
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    let (pending, bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    let own = std::time::Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let own = own.elapsed();
    let owner = std::time::Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let owner = owner.elapsed();
    domain.owner_worked(owner);
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 1)]);
    let (pending, _) = finish_one(&mut domain, flight);
    let charged = pending.physical_duration();
    assert!(charged >= own, "the request's own host work: {charged:?}");
    assert!(charged < own + owner, "the owner's work is no launch's: {charged:?}, owner {owner:?}");
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

/// [`chained`] on the device `Automatic` prefers, priced ([`measure_load`])
/// where `priced`.
fn chain_load(priced: bool) -> Fixture {
    let (mut domain, mut bindings) = chained(platform::DeviceRequest::Automatic);
    if priced {
        measure_load(&mut domain, &mut bindings);
    }
    (domain, bindings)
}

/// [`chain_load`], priced.
fn chain_fixture() -> Fixture {
    chain_load(true)
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
    super::fixtures::shrink_settled(&mut domain, &mut bindings, ShrinkPolicy::Idle);
    assert_eq!(domain.group_places().get(), 3);
    super::fixtures::shrink_settled(&mut domain, &mut bindings, ShrinkPolicy::Reclaim);
    assert_eq!(domain.group_places().get(), 1);
}

/// A prompt group's places are granted by their price like any holding:
/// with the device's gigabyte of room, a priced load's group takes its
/// chunks' places, each saving a group's submission at the prior's
/// requests a second, banked or not; an unpriced load's places are worth
/// nothing known, and its group takes none, the room left free.
#[test]
fn a_group_takes_its_places_by_price() {
    for backend in bank_backends() {
        let banked = |priced: bool| {
            let (mut domain, mut bindings) = banked_chain(backend, 4, 3);
            with_room(&mut domain, 1 << 30);
            if priced {
                measure_load(&mut domain, &mut bindings);
            }
            (domain, bindings)
        };
        assert_eq!(group_places_opened(banked(true)), 3, "{backend:?}: the group holds its chunks' places");
        assert_eq!(group_places_opened(banked(false)), 1, "{backend:?}: an unpriced group takes no place");
    }
    assert_eq!(group_places_opened(chain_load(true)), 3, "a load without a bank prices its places too");
    assert_eq!(group_places_opened(chain_load(false)), 1, "an unpriced group takes no place");
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
            .plan_prompt(None, true, true, |chunks| {
                counted = Some(chunks);
                Ok::<_, ()>(Vec::new())
            })
            .unwrap();
        assert_eq!(counted, Some(planned));
    }
}

/// A pipelined successor, which other work orphans, is planned only for a
/// request the device serves alone; a chain behind its round's last group,
/// whose holder the scheduler selects first, regardless.
#[test]
fn only_a_chain_is_planned_beside_other_work() {
    for (fixture, planned) in [(chain_fixture(), 2), (fixture_with(None, true, false), 0)] {
        let (mut domain, _) = fixture;
        let mut counted = None;
        domain
            .plan_prompt(None, false, true, |chunks| {
                counted = Some(chunks);
                Ok::<_, ()>(Vec::new())
            })
            .unwrap();
        assert_eq!(counted, Some(planned));
    }
}

/// A group with another of its round submitted after it opens no chain, which
/// that group would orphan: it plans no chunk behind it.
#[test]
fn no_chain_opens_behind_a_group_its_round_follows() {
    let (mut domain, _) = chain_fixture();
    let mut counted = None;
    domain
        .plan_prompt(None, false, false, |chunks| {
            counted = Some(chunks);
            Ok::<_, ()>(Vec::new())
        })
        .unwrap();
    assert_eq!(counted, Some(0));
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
            .plan_prompt(Some(0.5), true, true, |chunks| {
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
    super::fixtures::shrink_settled(&mut domain, &mut bindings, ShrinkPolicy::Idle);
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

/// A reservation that grows nothing reads the heap's charge once: no
/// provisioning or import ran to change it after the entry's reading.
#[test]
fn a_reservation_that_grows_nothing_reads_memory_once() {
    let (mut domain, mut bindings) = fixture(None);
    let request = RequestId(61);
    open(&mut domain, &mut bindings, request);
    let flight = submit_reserved_target(&mut domain, bindings, vec![forward(request, 0)]);
    let (pending, mut bindings) = finish_one(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    let committed = domain.committed_state_bytes();
    let refreshes = domain.memory_refreshes();
    let reservation = domain.reserve(&mut bindings, &[forward(request, 1)]).unwrap();
    assert!(matches!(
        reservation.into_resources(),
        ReservedResources::Target(TargetGraphReservation::Launch(_))
    ));
    assert_eq!(domain.committed_state_bytes(), committed);
    assert_eq!(domain.memory_refreshes() - refreshes, 1);
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

/// What Seismic charges the domain's device on every store: the load's
/// holding, which nothing else counts.
fn charged<F: ProgramFamily>(domain: &ExecutorDomain<F>) -> u64 {
    domain.domain.device().ledgers().own.iter().map(|(_, ledger)| ledger.charged).sum()
}

/// Released state backing leaves the device's charge, which the heap's next
/// observation reads.
#[test]
fn fresh_memory_observation_reads_released_state_backing() {
    let (mut domain, mut bindings) = fixture(None);
    domain.refresh_memory().unwrap();
    let before = charged(&domain);
    let released = super::fixtures::shrink_settled(&mut domain, &mut bindings, ShrinkPolicy::Reclaim);
    assert!(released > 0);
    domain.refresh_memory().unwrap();
    assert!(before - charged(&domain) >= released);
}

/// A tier copy whose grid the device refuses is the caller's error and
/// changes nothing: the bank keeps the copy costs it took before, grid and
/// price, and the domain the measures it priced by.
#[test]
fn a_refused_tier_copy_changes_neither_its_grid_nor_its_price() {
    let backend = bank_backends()[0];
    let (mut domain, mut bindings) = banked(backend, 4, 4);
    let measures = |blocks| crate::LoadMeasures {
        costs: Ok(load_costs()),
        experts: Some(crate::ExpertMeasures {
            lines: crate::assessment::Priced::Measured(std::sync::Arc::from([])),
            read: crate::assessment::Priced::Pending("unmeasured".into()),
            pass: crate::assessment::Priced::Pending("unpriced".into()),
            moves: crate::assessment::Priced::Pending("unmeasured".into()),
            tier_copy: crate::assessment::Priced::Measured(Some(crate::assessment::TierCopyCosts {
                blocks,
                whole_seconds: 0.001,
                stall_share: 0.1,
            })),
            convert: crate::assessment::Priced::Measured(None),
        }),
    };
    domain.measure_load(measures(8), &mut bindings).unwrap();
    domain.family.bank.as_mut().unwrap().refuse_tier_copy = true;
    let refused = domain.measure_load(measures(16), &mut bindings).unwrap_err();
    assert!(matches!(refused, DomainError::Invariant(_)), "{refused}");
    let bank = domain.family.bank.as_ref().unwrap();
    assert_eq!(bank.tier_copy.lock().unwrap().as_ref().map(|costs| costs.blocks), Some(8));
    let kept = domain.expert_measures().unwrap().tier_copy.measured().cloned().flatten().map(|costs| costs.blocks);
    assert_eq!(kept, Some(8));
}

/// The storage kinds the heap's settle grows (`settle_memory`) at a fresh
/// probe, as the owner settles, the most valuable per byte first, with
/// nothing of an owner's; none before the bank is priced. Nothing carries
/// the decision out, so the next settles afresh (`settle_again`).
fn paying(domain: &mut ExecutorDomain<TestFamily>) -> Vec<crate::planning::BankTier> {
    let kinds = settled_kinds(domain);
    domain.settle_again();
    kinds
}

/// [`paying`], its decision standing: a settle on the same inputs settles
/// nothing.
fn settled_kinds(domain: &mut ExecutorDomain<TestFamily>) -> Vec<crate::planning::BankTier> {
    domain.probe_memory().unwrap();
    let settled = domain.settle_memory(&crate::OwnerHoldings::default()).unwrap();
    let mut kinds = Vec::new();
    for grant in settled.grants {
        if !kinds.contains(&grant.kind) {
            kinds.push(grant.kind);
        }
    }
    kinds
}

/// Whether the bank's device step pays.
fn device_pays(domain: &mut ExecutorDomain<TestFamily>) -> bool {
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
/// (`DeviceHeap::reserved_rooms`), which admits exactly its room: that many
/// bytes and not one more. With a claim the heap holds leaving room for two
/// of the three slots the record's misses would use, the step takes the two
/// and lands; sized past the heap's room, as by a room the claim does not
/// test, it was refused at every decision.
#[test]
fn a_growth_step_is_sized_within_what_the_heap_admits() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 1);
        let row = bank_row(&domain);
        // Of each step's 8 reads, 7 hit a bank of 3 slots and the last one of
        // 4: the second and third slots save 7 reads a step together, more a
        // slot than the fourth's one, so the two move apart from it.
        price_growth(&mut domain, &mut bindings, misses_by(100, &[(2, 7), (3, 1)]));
        let device = crate::planning::BankTier::Bank;
        // What a growth may take: the free headroom's room, which a growth's
        // claim reads.
        let available = domain.memory.borrow_mut().reserved_rooms(device, 0, &[crate::platform::FitMode::Live], 0).unwrap()[0];
        let held = domain.memory.borrow_mut().claim_reserved(device, available - 5 * row / 2, 0, HoldingClass::Live, crate::platform::FitMode::Live, 0).unwrap();
        let room = domain.memory.borrow_mut().reserved_rooms(device, 0, &[crate::platform::FitMode::Live], 0).unwrap()[0];
        assert!(
            matches!(
                domain.memory.borrow_mut().claim_reserved(device, room + 1, 0, HoldingClass::Live, crate::platform::FitMode::Live, 0),
                Err(ClaimRefusal::Deficit { .. })
            ),
            "one byte past the room is refused"
        );
        let claim = domain.memory.borrow_mut().claim_reserved(device, room, 0, HoldingClass::Live, crate::platform::FitMode::Live, 0).unwrap();
        domain.memory.borrow_mut().release(claim);
        assert!(device_pays(&mut domain));
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank), "within the room");
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { from: 1, to: 3, .. }), "two slots of the room");
        domain.memory.borrow_mut().release(held);
    }
}

/// Device memory and host memory grow by what each byte saves: the bank's
/// slots, which save the decode step, before the host tier's records,
/// which the fake prices at next to nothing ([`HOST_SLOT_SAVES`]); each in
/// one resize to its settled size.
#[test]
fn both_stores_grow_by_what_a_byte_saves() {
    use crate::planning::BankTier::{Bank as Device, Host};
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 1);
        domain.family.bank.as_mut().unwrap().host_ceiling = 4;
        // Each step's reads hit a bank of the 4 rows and no fewer: every
        // slot to 4 saves, and the misses stop falling there.
        price_growth(&mut domain, &mut bindings, misses_at(3, 100));
        assert_eq!(paying(&mut domain), [Device, Host]);
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(Device));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { holding: crate::BankHolding::Experts, from: 1, to: 4, .. }));
        assert_eq!(paying(&mut domain), [Host], "the bank saturated; the tier holds the rest");
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(Host));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { holding: crate::BankHolding::Host, to: 4, .. }));
        assert!(paying(&mut domain).is_empty(), "both saturated");
    }
}

/// Before the load's costs price its bank nothing grows: the host tier's
/// records, as the bank's slots, save nothing known.
#[test]
fn nothing_grows_before_the_bank_is_priced() {
    for backend in bank_backends() {
        let (mut domain, _bindings) = banked(backend, 4, 1);
        domain.family.bank.as_mut().unwrap().host_ceiling = 6;
        assert!(paying(&mut domain).is_empty(), "no costs: nothing priced");
    }
}

/// Before the load counted a decode step its bank is priced on the generic
/// model alone (`DecodePrice::prior`): it grows at once, in one resize to
/// every slot the prior's misses justify.
#[test]
fn the_bank_grows_on_the_prior_before_a_decode_is_counted() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 1);
        measure_load(&mut domain, &mut bindings);
        assert!(domain.family.expert_bank().unwrap().live().is_none(), "no decode counted");
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { from: 1, to: 4, .. }), "one growth to the settled size");
    }
}

/// The class room a plan reserves is the places' from the plan on
/// (`class_reserve`): a partial bank that grows before the first group
/// takes its places grows only into the room beside it, never by timing,
/// and the group then takes every place from the room kept for it.
#[test]
fn a_bank_grown_before_the_first_group_leaves_its_places_room() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked_chain(backend, 4, 1);
        // Each slot saves fewer reads than the one before it, and some reads
        // miss every size (a step always pays the read's latency): each
        // growth by a row is worth its bytes alone, so room alone stops it.
        price_growth(&mut domain, &mut bindings, misses_by(100, &[(1, 8), (2, 4), (3, 2), (4, 1)]));
        domain.probe_memory().unwrap();
        let row = bank_row(&domain);
        let room = |domain: &ExecutorDomain<TestFamily>| {
            domain.growth_room(crate::planning::BankTier::Bank, &[super::borrowing::Room::Committed]).unwrap()[0]
        };
        assert!(room(&domain) > 4 * row, "{backend:?}: the device has room for the bank");
        // The plan's class keeps all but a row and a half of the room: the
        // least reserve that leaves the growth no more (the room is the
        // device's less the reserve, within the bank's own reservation).
        let leaves = |domain: &mut ExecutorDomain<TestFamily>, reserve: u64| {
            domain.class_reserve = reserve;
            room(domain)
        };
        let (mut low, mut high) = (0u64, u64::MAX / 2);
        while low < high {
            let mid = low + (high - low) / 2;
            match leaves(&mut domain, mid) <= row + row / 2 {
                true => high = mid,
                false => low = mid + 1,
            }
        }
        let reserve = low;
        let left = leaves(&mut domain, reserve);
        assert!(left > row && left <= row + row / 2, "{backend:?}: the reserve leaves a row and a half ({left} of {row})");
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(
            matches!(finish(&mut domain), crate::BankResized::Grown { from: 1, to: 2, .. }),
            "{backend:?}: the bank grows by the row beside the class's room"
        );
        assert_eq!(domain.class_reserve, reserve, "{backend:?}: no growth takes the class's room");
        assert_eq!(group_places_opened((domain, bindings)), 3, "{backend:?}: the group takes its places from the room kept for them");
    }
}

/// Regrowth after an ask is one growth to the settled size: the ask takes
/// the bank down in one shrink to what covers it, and the next settle
/// grows it back in one resize, not a slot a reading.
#[test]
fn regrowth_is_one_growth_to_the_settled_size() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        // Each step's reads hit a bank of the 4 rows and no fewer.
        price_growth(&mut domain, &mut bindings, misses_at(3, 100));
        let row = bank_row(&domain);
        assert_eq!(domain.shrink_expert_bank_by(&mut bindings, 3 * row, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Shrunk { from: 4, to: 1, .. }), "one shrink covers the ask");
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { from: 1, to: 4, .. }), "one growth back");
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
        thresholds: crate::platform::DomainThresholds { planning_bytes: 0, emergency_bytes: 0 },
        charged: 0,
        band: MemoryBand::Normal,
        unseen: 0,
        witness: None,
        untaken: Default::default(),
        rate: 0.0,
        fluctuation: Fluctuation::default(),
        host: false,
        grant_left: None,
        borrowable: None,
        takes: None,
        first: std::time::Instant::now(),
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

/// A settle runs only where its inputs moved (`SettleStands`): a reading
/// that moved no holding, no request and no room past a step decides
/// nothing anew, so it settles nothing however often readings come. Run at
/// every reading, a settle took a resident Qwen3.6 q4 4k decode on an RTX
/// 3090 from about 10 ms a step to 108 ms: each step outlasted the reading
/// interval, so every step read memory and settled. A request's arrival and
/// new measures settle again, as does a decision the owner could not carry
/// out (`settle_again`).
#[test]
fn a_settle_runs_only_where_its_inputs_moved() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 1);
        price_growth(&mut domain, &mut bindings, misses_at(2, 100));
        let settles = domain.settles.get();
        assert!(settled_kinds(&mut domain).contains(&crate::planning::BankTier::Bank), "{backend:?}: the first settle grants the bank's step");
        assert_eq!(domain.settles.get(), settles + 1);
        for _ in 0..3 {
            assert!(settled_kinds(&mut domain).is_empty(), "{backend:?}: an unmoved reading settles nothing");
        }
        assert_eq!(domain.settles.get(), settles + 1, "{backend:?}: nor runs the settle");
        domain.settle_again();
        assert!(settled_kinds(&mut domain).contains(&crate::planning::BankTier::Bank), "{backend:?}: a decision not carried out is settled again");
        domain.served_request(false);
        assert!(settled_kinds(&mut domain).contains(&crate::planning::BankTier::Bank), "{backend:?}: a request's arrival settles again");
        measure_load(&mut domain, &mut bindings);
        assert!(settled_kinds(&mut domain).contains(&crate::planning::BankTier::Bank), "{backend:?}: new measures settle again");
        assert_eq!(domain.settles.get(), settles + 4);
    }
}

/// A holding no resize has measured takes its first growth step, to its
/// sized target: the most slots within the store's room up to where the
/// record's misses stop falling, never the bank's every row past them. The
/// step measures what a resize holds the submitting thread. After that a
/// step grows only where what it saves over the settle's horizon covers
/// what the holding's last resize held the thread; one that does not pay
/// grows nothing, and a saturated holding has no step.
#[test]
fn a_holding_grows_once_unmeasured_then_only_where_its_saving_exceeds_its_resize() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 1);
        // 100 steps of 4 misses a step at 2 slots, none at 3: the misses stop
        // falling at 3 slots, of the bank's 4 rows.
        price_growth(&mut domain, &mut bindings, misses_at(2, 100));
        assert_eq!(domain.resize_held.last.get(&crate::BankHolding::Experts), None);
        assert!(device_pays(&mut domain), "an unmeasured holding takes its first step");
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { from: 1, to: 3, .. }), "the sized target");
        assert!(domain.resize_held.last.get(&crate::BankHolding::Experts).is_some_and(|&held| held < 0.4));
        assert!(!device_pays(&mut domain), "saturated: a fourth slot saves nothing");
        // Misses at 3 slots, none at 4: the fourth slot saves the step's
        // reads, seconds over the horizon, which outweigh the resize. A saving
        // is less than the step it is taken from and the decode runs one step
        // at a time, so it saves under a second a second: a resize held the
        // horizon's 60 s is not paid.
        price_growth(&mut domain, &mut bindings, misses_at(3, 100));
        assert!(device_pays(&mut domain), "the step's reads saved outweigh the resize");
        domain.resize_held.last.insert(crate::BankHolding::Experts, 60.);
        assert!(!device_pays(&mut domain), "a step that does not pay grows nothing");
    }
}

/// A load priced on its executor's costs before the job stored the read
/// line its decode is priced along grows nothing; priced again once it is,
/// on the same live domain, its bank grows by the rule as a load priced at
/// its start would.
#[test]
fn a_live_load_priced_again_on_newly_stored_figures_grows() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 2);
        let decode = misses_at(3, 100);
        count_decode(&mut domain, decode);
        // Priced on the executor's costs, before the job stored the model
        // store's read line its decode is priced along.
        let unpriced = crate::LoadMeasures {
            costs: Ok(load_costs()),
            experts: Some(crate::ExpertMeasures {
                lines: crate::assessment::Priced::Measured(std::sync::Arc::from([])),
                read: crate::assessment::Priced::Pending("the store's read line is not measured yet".into()),
                pass: crate::assessment::Priced::Pending("unpriced".into()),
                moves: crate::assessment::Priced::Pending("unmeasured".into()),
                tier_copy: crate::assessment::Priced::Measured(None),
                convert: crate::assessment::Priced::Measured(None),
            }),
        };
        domain.measure_load(unpriced, &mut bindings).unwrap();
        assert!(!device_pays(&mut domain), "an unpriced load grows nothing");
        price_growth(&mut domain, &mut bindings, misses_at(3, 100));
        assert!(device_pays(&mut domain));
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
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
        price_growth(&mut domain, &mut bindings, misses_at(3, 100));
        // A probe in the Reclaim band reads memory as any does.
        match domain.probe_memory() {
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
        assert!(!device_pays(&mut domain), "the claimed room is not the bank's");
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::AtLimit);
        domain.release_measurement_room();
        assert_eq!(domain.measurement_room().unwrap(), None);
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank), "released, the bank grows again");
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
        price_growth(&mut domain, &mut bindings, misses_at(3, 100));
        assert_eq!(
            domain.grow_settled(unwatched()).unwrap(),
            BankStep::Begun(crate::planning::BankTier::Bank)
        );
        domain.claim_measurement_room(0);
        assert!(matches!(finish(&mut domain), crate::BankResized::Stopped { at: 2, .. }), "stopped before its first slot");
        assert!(paying(&mut domain).is_empty(), "nothing grows beside the claim");
        domain.release_measurement_room();
        assert!(device_pays(&mut domain), "released, the step pays again");
        assert_eq!(
            domain.grow_settled(unwatched()).unwrap(),
            BankStep::Begun(crate::planning::BankTier::Bank)
        );
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { from: 2, to: 4, .. }));
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
        price_growth(&mut domain, &mut bindings, misses_at(3, 100));
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
            assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
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

/// A launch's duration says whether the expert bank's fill ran beside it:
/// one with the fill running at its submission and finish is timed as
/// filling, one a fill began or finished across is not timed, and one with
/// no fill beside it is settled.
#[test]
fn a_launch_says_whether_the_banks_fill_ran_beside_it() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 2);
        let request = RequestId(91);
        open(&mut domain, &mut bindings, request);
        let step = |domain: &mut ExecutorDomain<TestFamily>, bindings, position, during: &dyn Fn(&ExecutorDomain<TestFamily>)| {
            let flight = submit_reserved_target(domain, bindings, vec![forward(request, position)]);
            during(domain);
            let (pending, bindings) = finish_one(domain, flight);
            let duration = pending.launch_duration();
            domain.reconcile(pending, PhysicalDecision { accepted_rows: 1 }).unwrap();
            (duration.map(|duration| duration.filling), bindings)
        };
        let quiet = |_: &ExecutorDomain<TestFamily>| {};
        let fill = |filling: bool| move |domain: &ExecutorDomain<TestFamily>| domain.family.bank.as_ref().unwrap().fill(filling);
        // The first launch records its graphs.
        let (_, bindings) = step(&mut domain, bindings, 0, &quiet);
        let (timed, bindings) = step(&mut domain, bindings, 1, &quiet);
        assert_eq!(timed, Some(false), "{backend:?}: no fill beside it");
        let (timed, bindings) = step(&mut domain, bindings, 2, &fill(true));
        assert_eq!(timed, None, "{backend:?}: a fill began across it");
        let (timed, bindings) = step(&mut domain, bindings, 3, &quiet);
        assert_eq!(timed, Some(true), "{backend:?}: the fill ran beside it");
        let (timed, bindings) = step(&mut domain, bindings, 4, &fill(false));
        assert_eq!(timed, None, "{backend:?}: the fill finished across it");
        let (timed, _) = step(&mut domain, bindings, 5, &quiet);
        assert_eq!(timed, Some(false), "{backend:?}: settled again");
    }
}

/// A growth decided while a step runs waits for no gap: its slots join at a
/// boundary steps still run at, and it finishes there.
#[test]
fn a_growth_decided_mid_flight_joins_beside_the_running_steps() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 2);
        price_growth(&mut domain, &mut bindings, misses_at(3, 100));
        domain.family.bank.as_ref().unwrap().running_steps(1);
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
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

/// A shrink decided while a prompt chain is under way (a settle's, at a
/// reading between its chunks) is deferred, never orphaning the chunks its
/// holder claims; once the chain ends it begins.
#[test]
fn a_shrink_during_a_chain_is_deferred_to_its_end() {
    for backend in bank_backends() {
        // Priced, with room beside the bank, the prompt's group takes its
        // chunks' places and chains them (`a_group_takes_its_places_by_price`).
        let (mut domain, mut bindings) = banked_chain(backend, 4, 4);
        with_room(&mut domain, 1 << 30);
        measure_load(&mut domain, &mut bindings);
        let request = RequestId(80);
        open(&mut domain, &mut bindings, request);
        domain.plan_successors((1..4).map(|position| prefill(request, position)).collect());
        let flight = submit_reserved_target(&mut domain, bindings, vec![prefill(request, 0)]);
        let (pending, mut bindings) = finish_one(&mut domain, flight);
        domain.reconcile(pending, PhysicalDecision { accepted_rows: 1 }).unwrap();
        assert_eq!(domain.chain_holder(), Some(request), "{backend:?}");
        let shrink = |domain: &mut ExecutorDomain<TestFamily>, bindings: &mut StateBindings| {
            domain.shrink_expert_bank_to(bindings, crate::BankHolding::Experts, 3, unwatched()).unwrap()
        };
        assert_eq!(shrink(&mut domain, &mut bindings), BankStep::Deferred, "{backend:?}");
        assert_eq!(domain.chain_holder(), Some(request), "{backend:?}: the chain keeps its chunks");
        assert_eq!(domain.expert_bank_slots(), Some(4));
        domain.release_chain().unwrap();
        assert_eq!(shrink(&mut domain, &mut bindings), BankStep::Begun(crate::planning::BankTier::Bank), "{backend:?}");
        assert!(matches!(finish(&mut domain), crate::BankResized::Shrunk { to: 3, .. }));
    }
}

/// A one-row decode at `position` that selects its next token on the device.
fn decode(request: RequestId, position: usize) -> Operation {
    Operation::Forward {
        request,
        kind: WorkKind::Decode,
        tokens: vec![crate::TokenId(1)],
        position,
        conditioning: None,
        demand: crate::batching::Demand::SELECT,
        select: vec![crate::SelectSpec {
            sampling: crate::Sampling::Greedy,
            seed: 1,
            position,
            domain: 0,
            mask: None,
            shaping: crate::Shaping::default(),
            history: None,
        }],
        committed: 1,
        prime: None,
    }
}

/// Pipelined decode queues each slot's next row behind a step only while
/// every slot has one: a step filling a sequence's last context row ends
/// that request, so nothing is queued behind it, whatever room its
/// neighbours have. Queueing it anyway asked the state for a successor past
/// the context, which faulted a run of four sequences decoding to the
/// context its longest one was sized to.
#[test]
fn no_decode_is_queued_past_a_sequences_context() {
    let (domain, _) = fixture_with(None, true, false);
    let context = domain.execution.resources().target_state().context_rows;
    let (near, far) = (RequestId(1), RequestId(2));
    match domain.behind(&[decode(near, context - 2), decode(far, context - 4)]) {
        queue::Behind::Continuation { operations } => {
            let positions = operations
                .iter()
                .map(|operation| match operation {
                    Operation::Forward { position, .. } => *position,
                    _ => unreachable!("a continuation is a forward"),
                })
                .collect::<Vec<_>>();
            assert_eq!(positions, [context - 1, context - 3]);
        }
        _ => panic!("a step with room in every slot queues its continuation"),
    }
    assert!(matches!(
        domain.behind(&[decode(near, context - 1), decode(far, context - 3)]),
        queue::Behind::Nothing
    ));
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
        price_growth(&mut domain, &mut bindings, misses_at(3, 100));
        domain.family.bank.as_ref().unwrap().running_steps(1);
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
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
        price_growth(&mut domain, &mut bindings, misses_at(1, 100));
        assert_eq!(domain.shrink_expert_bank(&mut bindings, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        finish(&mut domain);
        assert!(device_pays(&mut domain));
        let price = domain.decode_price().unwrap();
        let kind = crate::planning::BankTier::Bank;
        let bank = domain.family.expert_bank().unwrap();
        let step = bank.curves(price.as_ref(), crate::Reaches::every(crate::Reach { held: 0, offered: 1 })).unwrap()[0].offered[0].bytes;
        // A curve priced only as far as a reach goes is the whole curve's
        // first steps, the step crossing the reach included.
        let whole = bank.curves(price.as_ref(), crate::Reaches::every(crate::Reach { held: u64::MAX, offered: u64::MAX })).unwrap();
        let near = bank.curves(price.as_ref(), crate::Reaches::every(crate::Reach { held: 2 * step - 1, offered: step + 1 })).unwrap();
        assert_eq!((near[0].held.len(), near[0].offered.len()), (2.min(whole[0].held.len()), 2.min(whole[0].offered.len())));
        assert_eq!(near[0].offered[..], whole[0].offered[..near[0].offered.len()]);
        assert_eq!(near[0].held[..], whole[0].held[..near[0].held.len()]);
        // What the heap grants a growth now, by whichever constraint binds:
        // the free headroom's room, which a growth's claim reads.
        let available = domain.memory.borrow_mut().reserved_rooms(kind, 0, &[crate::platform::FitMode::Live], 0).unwrap()[0];
        // Claimed as a reservation's, the bytes take the store's room itself,
        // which `available` reads: the free space of the device's blocks serves
        // none of them.
        let leaving = |domain: &ExecutorDomain<TestFamily>, left: u64| {
            domain.memory.borrow_mut().claim_reserved(kind, available - left, 0, HoldingClass::Live, crate::platform::FitMode::Live, 0).unwrap()
        };
        let held = leaving(&domain, step / 2);
        assert!(matches!(
            domain.grow_settled(unwatched()),
            Ok(BankStep::AtLimit) | Err(DomainError::Reclaim(_))
        ));
        assert_eq!(domain.expert_bank_slots(), Some(1));
        domain.memory.borrow_mut().release(held);
        let held = leaving(&domain, 2 * step);
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { to: 2, .. }));
        domain.memory.borrow_mut().release(held);
    }
}

/// A holding updated while a bank shrink releases: the release falls off
/// the owner's thread, once the device work before it completes, and an
/// observation a claim takes meanwhile reads the lower charge before any
/// advance reports it. The update still classifies within the charge (the
/// holding read after it), so no request fails because the bank
/// resized; every byte stays classified.
#[test]
fn a_holding_update_beside_a_shrinks_release_classifies_within_the_charge() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        let optional = seismic::Tensor::zeros(domain.domain.device(), seismic::Element::bool(), &[row]).unwrap();
        domain.family.optional = Some(optional);
        domain.refresh_memory().unwrap();
        assert_eq!(domain.shrink_expert_bank(&mut bindings, unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        // The first advance installs; the release is the bank's own thread's.
        assert!(domain.advance_expert_bank(unwatched()).unwrap().is_none());
        domain.family.bank.as_ref().unwrap().release_now();
        // A claim's observation, which no holding follows.
        domain.memory.borrow_mut().refresh(0).unwrap();
        domain.refresh_memory().unwrap();
        assert!(matches!(finish(&mut domain), crate::BankResized::Shrunk { from: 4, to: 3, released, .. } if released == row));
    }
}

/// The ladder's bank rung reaches the family's bank: each step begins at
/// one boundary, installs at the next and is credited exactly what Seismic
/// stopped charging once released; a step
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
        domain.refresh_memory().unwrap();
        let before = charged(&domain);
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
        assert_eq!(before - charged(&domain), 3 * row);
        // Without a priced decode there are no grounds to grow: the rule
        // grows nothing.
        assert!(!device_pays(&mut domain));
        // A growth step, priced on the measured resident step, backs exactly
        // what it claimed, the holding following: one step to every row, where
        // the record's misses stop falling.
        price_growth(&mut domain, &mut bindings, misses_at(3, 100));
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::Begun(crate::planning::BankTier::Bank));
        assert!(matches!(finish(&mut domain), crate::BankResized::Grown { from: 1, to: 4, grown, .. } if grown == 3 * row));
        assert_eq!(domain.grow_settled(unwatched()).unwrap(), BankStep::AtLimit);
        domain.refresh_memory().unwrap();
        assert_eq!(charged(&domain), before);
    }
}

const PACE_GB: u64 = 1_000_000_000;

/// One reading of `store` with `headroom` bytes against a 10 GB planning
/// line, the process charging `charged` and its releases' witness reading
/// `witness`, with nothing releasable, at the last reading's instant (now
/// for the first), so no span reaches the reading interval and no rate
/// plays a part.
fn read_pace(pace: &mut MemoryPace, store: seismic::StoreId, headroom: u64, charged: u64, witness: u64) {
    let at = pace.read.unwrap_or_else(Instant::now);
    read_pace_at(pace, store, headroom, charged, Some(witness), at, 0);
}

/// [`read_pace`] read `at`, its witness reading `witness` (none where none
/// is read), the load able to release `releasable` bytes of host RAM.
#[allow(clippy::too_many_arguments)]
fn read_pace_at(
    pace: &mut MemoryPace,
    store: seismic::StoreId,
    headroom: u64,
    charged: u64,
    witness: Option<u64>,
    at: Instant,
    releasable: u64,
) {
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
        borrowable_bytes: None,
        displacing: false,
        band: if headroom <= thresholds.planning_bytes { MemoryBand::Reclaim } else { MemoryBand::Normal },
    };
    let releasable = crate::platform::Releasable { host: releasable, device: 0 };
    pace.observe(&[reading], &[(store, charged)], witness, true, releasable, pace_release(), at);
}

/// The release rate the pace's readings are taken beside: 10 GB/s.
fn pace_release() -> crate::platform::ReleaseRate {
    crate::platform::ReleaseRate::new(10.0 * PACE_GB as f64)
}

/// The host shows each release `lag` readings late, whatever `lag`, its
/// witness falling as it does: the bank
/// releases what the pressure needs rounded up to one rung and waits for the
/// rest, and pressure that comes after the release shows is answered in turn.
#[test]
fn a_release_the_host_shows_late_is_not_released_again() {
    const RUNG: u64 = 2 * PACE_GB;
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    for lag in 0..8 {
        let mut pace = MemoryPace::new();
        let (mut charged, mut releases) = (20 * PACE_GB, Vec::<(usize, u64)>::new());
        for at in 0..40 {
            let taken = if at < 20 { 3 } else { 6 } * PACE_GB;
            let shown: u64 = releases.iter().filter(|(when, _)| when + lag <= at).map(|(_, bytes)| bytes).sum();
            let unshown = releases.iter().map(|(_, bytes)| bytes).sum::<u64>() - shown;
            read_pace(&mut pace, store, (10 * PACE_GB + shown).saturating_sub(taken), charged, charged + unshown);
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
/// next rung sees the whole of what is still short; one the witness has yet
/// to drop is, until it does.
#[test]
fn only_what_the_witness_has_yet_to_drop_is_credited() {
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let mut pace = MemoryPace::new();
    read_pace(&mut pace, store, 7 * PACE_GB, 20 * PACE_GB, 20 * PACE_GB);
    read_pace(&mut pace, store, 8 * PACE_GB, 19 * PACE_GB, 19 * PACE_GB);
    assert_eq!(pace.shortfall(store, 19 * PACE_GB), Shortfall::Short(2 * PACE_GB), "shown at once");
    read_pace(&mut pace, store, 8 * PACE_GB, 18 * PACE_GB, 19 * PACE_GB);
    assert_eq!(pace.shortfall(store, 18 * PACE_GB), Shortfall::Short(PACE_GB), "a GB unseen");
    read_pace(&mut pace, store, 9 * PACE_GB, 18 * PACE_GB, 18 * PACE_GB);
    assert_eq!(pace.shortfall(store, 18 * PACE_GB), Shortfall::Short(PACE_GB), "shown now");
    // The host takes a GB with none unseen: all of it is short.
    read_pace(&mut pace, store, 8 * PACE_GB, 18 * PACE_GB, 18 * PACE_GB);
    assert_eq!(pace.shortfall(store, 18 * PACE_GB), Shortfall::Short(2 * PACE_GB));
}

/// The process backing its own charge is no other program's take: a
/// reservation charged before its pages are wired (an import, a bank's
/// first fill) lowers the headroom by what its witness rises, and the pace
/// measures no motion from it, so no margin follows; the same fall with the
/// witness still is another program's, and is measured.
#[test]
fn backing_the_process_s_own_charge_measures_no_take() {
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let start = Instant::now();
    let mut pace = MemoryPace::new();
    let read = |pace: &mut MemoryPace, reading: u32, headroom: u64, witness: u64| {
        read_pace_at(pace, store, headroom, 29 * PACE_GB, Some(witness), start + READING_INTERVAL * reading, 0);
    };
    // 22 GB charged at once, then wired over the next readings, as a
    // 28 GB-capped load's import on a 48 GB Mac reads.
    read(&mut pace, 0, 40 * PACE_GB, 8 * PACE_GB);
    read(&mut pace, 1, 29 * PACE_GB, 19 * PACE_GB);
    read(&mut pace, 2, 18 * PACE_GB, 30 * PACE_GB);
    assert!(pace.motions().host.is_none(), "the process's own backing is measured as no take");
    // Another program takes 4 GB: the headroom falls, the witness holds.
    read(&mut pace, 3, 14 * PACE_GB, 30 * PACE_GB);
    let motion = pace.motions().host.expect("another program's take is measured");
    assert!(motion.take_rate() > 0.0);
}

/// A take the pace reads is the one sustained over the span a margin holds
/// it across: 100 ms readings, 20 GB releasable at 10 GB/s, so a 2.1 s
/// window. Another program's 22 GB fall within one reading span (220 GB/s
/// across it) reads as at most its share of the window, and once the
/// readings stay flat for the window it reads no take at all, so no margin
/// follows it.
#[test]
fn a_transient_fall_sets_no_take_past_its_window() {
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let start = Instant::now();
    let mut pace = MemoryPace::new();
    let read = |pace: &mut MemoryPace, reading: u64, headroom: u64| {
        read_pace_at(pace, store, headroom, 20 * PACE_GB, Some(20 * PACE_GB), start + READING_INTERVAL * reading as u32, 20 * PACE_GB);
    };
    for reading in 0..30 {
        read(&mut pace, reading, 30 * PACE_GB);
    }
    assert!(pace.motions().host.is_none(), "flat readings take nothing");
    read(&mut pace, 30, 8 * PACE_GB);
    let rate = pace.domains[&store].rate;
    assert!(rate > 0.0 && rate <= 22.0 * PACE_GB as f64 / 2.1 * (1.0 + 1e-9), "{rate} B/s: a share of the window, not the span's 220 GB/s");
    for reading in 31..51 {
        read(&mut pace, reading, 8 * PACE_GB);
        assert!(pace.domains[&store].rate > 0.0, "reading {reading}: the fall is within the window");
    }
    read(&mut pace, 51, 8 * PACE_GB);
    assert_eq!(pace.domains[&store].rate, 0.0, "the window holds no fall once flat for it");
    assert!(pace.motions().host.is_none(), "a take that stopped sets no margin");
}

/// A take that sustains reads as itself: another program taking 2 GB/s in
/// 100 ms readings reads 2 GB/s once the 2.1 s window holds it, and the
/// margin's motion is that rate.
#[test]
fn a_sustained_take_reads_at_its_rate() {
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let start = Instant::now();
    let mut pace = MemoryPace::new();
    for reading in 0..=100u64 {
        let headroom = 40 * PACE_GB - reading * PACE_GB / 5;
        read_pace_at(&mut pace, store, headroom, 20 * PACE_GB, Some(20 * PACE_GB), start + READING_INTERVAL * reading as u32, 20 * PACE_GB);
        let rate = pace.domains[&store].rate;
        match reading {
            0 => assert_eq!(rate, 0.0),
            1..=20 => assert!(rate > 0.0 && rate < 2.0 * PACE_GB as f64, "reading {reading}: {rate} B/s, the window not yet full"),
            _ => assert!((rate / (2.0 * PACE_GB as f64) - 1.0).abs() < 1e-9, "reading {reading}: {rate} B/s"),
        }
    }
    let motion = pace.motions().host.expect("a sustained take is measured");
    assert!((motion.take_rate() / (2.0 * PACE_GB as f64) - 1.0).abs() < 1e-9);
}

/// A release after the reading a decision uses is credited too; a reading
/// above the line ends what is owed, so pressure after it stands alone.
#[test]
fn the_credit_counts_releases_after_the_reading_until_memory_is_above_the_line() {
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let mut pace = MemoryPace::new();
    read_pace(&mut pace, store, 7 * PACE_GB, 20 * PACE_GB, 20 * PACE_GB);
    assert_eq!(pace.shortfall(store, 20 * PACE_GB), Shortfall::Short(3 * PACE_GB));
    assert_eq!(pace.shortfall(store, 19 * PACE_GB), Shortfall::Short(2 * PACE_GB), "released after the reading");
    assert_eq!(pace.shortfall(store, 17 * PACE_GB), Shortfall::Covered);
    read_pace(&mut pace, store, 7 * PACE_GB, 17 * PACE_GB, 20 * PACE_GB);
    assert_eq!(pace.shortfall(store, 17 * PACE_GB), Shortfall::Covered, "unseen at this reading");
    read_pace(&mut pace, store, 11 * PACE_GB, 17 * PACE_GB, 17 * PACE_GB);
    assert_eq!(pace.shortfall(store, 17 * PACE_GB), Shortfall::Above);
    read_pace(&mut pace, store, 9 * PACE_GB, 17 * PACE_GB, 17 * PACE_GB);
    assert_eq!(pace.shortfall(store, 17 * PACE_GB), Shortfall::Short(PACE_GB), "the old release is not credited again");
    read_pace(&mut pace, store, 9 * PACE_GB, 16 * PACE_GB, 17 * PACE_GB);
    assert_eq!(pace.shortfall(store, 16 * PACE_GB), Shortfall::Covered);
}

/// A release the witness never drops is awaited, never released again,
/// however many readings it stays unseen; what the host takes past it is
/// short, so pressure escalates past a release the host never shows.
#[test]
fn a_release_the_host_never_shows_yields_to_more_taken() {
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let mut pace = MemoryPace::new();
    read_pace(&mut pace, store, 7 * PACE_GB, 20 * PACE_GB, 20 * PACE_GB);
    assert_eq!(pace.shortfall(store, 20 * PACE_GB), Shortfall::Short(3 * PACE_GB));
    // 3 GB released; the witness never falls by it.
    for reading in 0..30 {
        read_pace(&mut pace, store, 7 * PACE_GB, 17 * PACE_GB, 20 * PACE_GB);
        assert_eq!(pace.shortfall(store, 17 * PACE_GB), Shortfall::Covered, "reading {reading} after the release");
    }
    // The host takes a GB more.
    read_pace(&mut pace, store, 6 * PACE_GB, 17 * PACE_GB, 20 * PACE_GB);
    assert_eq!(pace.shortfall(store, 17 * PACE_GB), Shortfall::Short(PACE_GB));
}

/// The process's own release is no taking by the host and ends none: its
/// witness falls by what the headroom rises, so a steady 1 GB/s take by
/// another program reads at its rate on every reading, across a 2 GB
/// release as before it, and a release with nothing taken reads none.
/// Without a witness the take is measured only across spans of an
/// unchanged charge: a release shown in its own reading or two readings
/// late invents no take (nothing reads above the steady 1 GB/s, no ask
/// counted faster), and the steady take reads at its rate again within a
/// window of the release showing.
#[test]
fn a_release_keeps_the_measured_taking() {
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let start = Instant::now();
    // 0.1 GB taken a reading; 2 GB released at reading 10, shown `lag`
    // readings later, its witness falling as it shows (or none read).
    let run = |witnessed: bool, lag: u64, taking: bool| {
        let mut pace = MemoryPace::new();
        let mut rates = Vec::new();
        for reading in 0..20u64 {
            let released = if reading >= 10 { 2 * PACE_GB } else { 0 };
            let shown = if reading >= 10 + lag { released } else { 0 };
            let taken = if taking { reading * PACE_GB / 10 } else { 0 };
            let headroom = 20 * PACE_GB - taken + shown;
            let charged = 20 * PACE_GB - released;
            let witness = witnessed.then_some(20 * PACE_GB - shown);
            read_pace_at(&mut pace, store, headroom, charged, witness, start + READING_INTERVAL * reading as u32, 0);
            rates.push(pace.domains[&store].rate);
        }
        (rates, pace.domains[&store].takes)
    };
    let (rates, _) = run(true, 0, true);
    assert_eq!(rates[0], 0.0);
    for (reading, rate) in rates.iter().enumerate().skip(1) {
        assert!((rate / PACE_GB as f64 - 1.0).abs() < 1e-6, "reading {reading}: {rate} B/s");
    }
    let (rates, takes) = run(true, 0, false);
    assert!(rates.iter().all(|rate| *rate == 0.0) && takes.is_none(), "a release with nothing taken reads none");
    for lag in [0, 2] {
        let (rates, takes) = run(false, lag, true);
        let steady = PACE_GB as f64 * (1.0 + 1e-6);
        assert!(rates.iter().all(|rate| *rate <= steady), "no witness, shown {lag} readings late: {rates:?}");
        assert!(takes.is_none_or(|takes| takes.mean() <= steady), "no witness, shown {lag} readings late: {takes:?}");
        for (reading, rate) in rates.iter().enumerate().skip(10 + lag as usize + 1) {
            assert!((rate / PACE_GB as f64 - 1.0).abs() < 1e-6, "no witness, shown {lag} late, reading {reading}: {rate} B/s");
        }
    }
}

/// A release that lands in a span in which the host takes as much is seen by
/// its witness's fall, not credited: the readings after it, flat below the
/// line, are short by the whole deficit, not covered.
/// (On a GB10 under a press, a 0.97 GB release landed as the press fell; the
/// load then read itself 0.65 GB below the line for 0.9 s and released
/// nothing.)
#[test]
fn a_release_the_host_takes_as_it_lands_is_not_credited() {
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let mut pace = MemoryPace::new();
    read_pace(&mut pace, store, 7 * PACE_GB, 20 * PACE_GB, 20 * PACE_GB);
    assert_eq!(pace.shortfall(store, 20 * PACE_GB), Shortfall::Short(3 * PACE_GB));
    // 3 GB released, and the host takes 3 GB in the same span.
    for reading in 0..3 {
        read_pace(&mut pace, store, 7 * PACE_GB, 17 * PACE_GB, 17 * PACE_GB);
        assert_eq!(pace.shortfall(store, 17 * PACE_GB), Shortfall::Short(3 * PACE_GB), "reading {reading} after the release");
    }
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
        thresholds: crate::platform::DomainThresholds { planning_bytes: 10 * GB, emergency_bytes: 2 * GB },
        charged: 0,
        band: MemoryBand::Normal,
        unseen: 0,
        witness: None,
        untaken: Default::default(),
        rate: 0.0,
        fluctuation: Fluctuation::default(),
        host: false,
        grant_left: None,
        borrowable: None,
        takes: None,
        first: std::time::Instant::now(),
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
    const GB: u64 = 1_000_000_000;
    let ms = std::time::Duration::from_millis;
    let mut pace = MemoryPace::new();
    assert_eq!(pace.horizon(), ms(100));
    pace.released(GB, &times(0, 120));
    assert_eq!(pace.horizon(), ms(340));
    pace.released(GB, &times(130, 280));
    assert_eq!(pace.horizon(), ms(400));
    // The release rate kept is the slowest: a gigabyte over 150 ms.
    assert_eq!(pace.motions().release, Some(crate::platform::ReleaseRate::new(GB as f64 / 0.15)));
    pace.released(GB, &times(300, 400));
    assert_eq!(pace.motions().release, Some(crate::platform::ReleaseRate::new(GB as f64 / 0.15)));
}

/// A shrink's wait for the steps queued on the device is latency, never
/// release time: an M4 Pro's 18.9 MB table release waited 164.3 ms of its
/// 165.4 ms for a gap. Read over the whole span, its 114 MB/s rate made the
/// device deficit of a Flash Next bank 1.82 TB on a 48 GB machine, and the
/// bank shrank from 185 slots to its floor; at a 10 GB/s take, a 20 GB
/// bank's margin would be 1.7 TB.
#[test]
fn a_release_rate_excludes_its_wait_for_queued_steps() {
    let start = Instant::now();
    let us = std::time::Duration::from_micros;
    let released = start + us(165_410);
    let times = crate::domain::ResizeTimes {
        decided: start,
        queued: us(164_272),
        planned: Default::default(),
        trips: Default::default(),
        moves: Default::default(),
        prepared: released,
        installed: released,
        released,
    };
    const TABLE: u64 = 18_874_368;
    let mut pace = MemoryPace::new();
    pace.released(TABLE, &times);
    assert_eq!(pace.horizon(), std::time::Duration::from_millis(100) + 2 * us(165_410), "the wait is latency");
    let rate = pace.motions().release.expect("a release was measured");
    assert_eq!(rate, crate::platform::ReleaseRate::new(TABLE as f64 / us(1_138).as_secs_f64()));
    let bank = 20_000_000_000;
    let margin = crate::platform::StoreMotion::new(0, 10e9).margin(
        READING_INTERVAL,
        times.latency(),
        Some(crate::platform::Releasing { rate, bytes: bank, store_bytes: 48_000_000_000 }),
    );
    assert!(margin < bank, "a margin of {margin} bytes for a {bank}-byte bank");
}

/// The wait bound is what the host may take from the planning line to the
/// emergency reserve at the rate it sustains taking now: 8 GB at 1 GB/s is
/// 8 s. None in a domain taking nothing now, nor in one that grows nothing;
/// none of a pace before any probe.
#[test]
fn the_wait_bound_is_the_reserve_between_the_lines_at_the_sustained_taking() {
    const GB: u64 = 1_000_000_000;
    let domain = |role, rate| DomainPace {
        role,
        headroom: 0,
        thresholds: crate::platform::DomainThresholds { planning_bytes: 10 * GB, emergency_bytes: 2 * GB },
        charged: 0,
        band: MemoryBand::Normal,
        unseen: 0,
        witness: None,
        untaken: Default::default(),
        rate,
        fluctuation: Fluctuation::default(),
        host: false,
        grant_left: None,
        borrowable: None,
        takes: None,
        first: std::time::Instant::now(),
    };
    assert_eq!(domain(DomainRole::Allocation, 1e9).wait_bound(), Some(8.0));
    assert_eq!(domain(DomainRole::Allocation, 0.0).wait_bound(), None);
    assert_eq!(domain(DomainRole::Staging, 1e9).wait_bound(), None);
    assert_eq!(MemoryPace::new().wait_bound(), None);
}

/// What a serving load measured of the rest of the host's motion, which
/// its reserves then hold their margin at: nothing on a quiet host (the
/// reserves keep the machine's page-fill rate), and once the host took
/// memory, the steady fluctuation and the sustained take, host RAM's and a
/// device's own memory's each their own; and the last release's latency
/// once one finished.
#[test]
fn a_serving_load_measures_the_hosts_motion_and_its_release_latency() {
    const GB: u64 = 1_000_000_000;
    let [host, device] = [(); 2].map(|()| seismic::Devices::discover(&[]).unwrap().topology().host_ram().id());
    let domain = |host: bool, rate: f64, fluctuation: u64| DomainPace {
        role: DomainRole::Allocation,
        headroom: 0,
        thresholds: crate::platform::DomainThresholds { planning_bytes: 0, emergency_bytes: 0 },
        charged: 0,
        band: MemoryBand::Normal,
        unseen: 0,
        witness: None,
        untaken: Default::default(),
        rate,
        fluctuation: Fluctuation { settled: None, largest: fluctuation },
        host,
        grant_left: None,
        borrowable: None,
        takes: None,
        first: Instant::now(),
    };
    let none = PaceMotions { host: None, device: None, latency: None, release: None };
    let mut pace = MemoryPace::new();
    assert_eq!(pace.motions(), none, "nothing measured before a probe");
    pace.domains.insert(host, domain(true, 0.0, GB / 10));
    pace.domains.insert(device, domain(false, 0.0, 0));
    assert_eq!(pace.motions(), none, "a quiet host took nothing");
    pace.domains.insert(host, domain(true, 5.0 * GB as f64, GB / 10));
    pace.latency = Duration::from_millis(200);
    assert_eq!(
        pace.motions(),
        PaceMotions {
            host: Some(crate::platform::StoreMotion::new(GB / 10, 5.0 * GB as f64)),
            latency: Some(Duration::from_millis(200)),
            ..none
        }
    );
    // The reserves hold the margin over a reading interval and that latency.
    let reserves = crate::platform::MemoryReserves::new(crate::platform::StoreMotion::new(0, GB as f64), crate::platform::ReleaseRate::new(1e9))
        .measured(pace.motions().host, pace.motions().device)
        .with_latency(Duration::from_millis(200));
    let m4_pro = seismic::fixtures::recorded::m4_pro();
    assert_eq!(reserves.host_margin(m4_pro.host_ram()), GB / 10 + 3 * GB / 2);
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
                confidence: None,
            },
            advance: Some(Advance::Step(head_advance)),
            primed: None,
            rows: 1,
            committed_rows: 0,
            kind: WorkKind::Decode,
            physical_duration: Duration::ZERO,
            timed: Timed::Settled,
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
            timed: Timed::Settled,
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
                timed: Timed::Settled,
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
    let control = attach_pending_priming(&mut domain, &mut flight, request, TokenId(2));
    (domain, flight, control)
}

/// Attach to `flight`, `request`'s target chunk, a checked head launch
/// entering `token` after the request's drafter state, with controlled
/// completion: the chunk's drafter entry, still running.
fn attach_pending_priming(
    domain: &mut ExecutorDomain<TestFamily>,
    flight: &mut TargetFlight<TestFamily>,
    request: RequestId,
    token: TokenId,
) -> PendingControl {
    let advance = TentativeAdvance::Accepted(
        OwnedStateAdvance::begin(domain.head.remove(&request).unwrap(), 1)
            .ok()
            .unwrap(),
    );
    let slot = domain
        .head_slot(request, &[token], 0, &[], &advance, 0)
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
    control
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

/// A chunk opening a chain whose drafter entry finishes after the chunk
/// publishes: the entry commits with the chunk, whose state the chain under
/// way holds, so the drafter follows the chunks queued behind it and the
/// holder claims each of them. An entry aborted for want of an accepted
/// target state would leave the drafter trailing the chain, and the
/// holder's next primed chunk, no longer the queued one, would orphan it.
#[test]
fn a_deferred_entry_commits_with_the_chain_its_chunk_opens() {
    let request = RequestId(95);
    let (mut domain, mut bindings) = fixture_of(
        tiny_definition(),
        limits(2, 3, false),
        None,
        true,
        platform::DeviceRequest::Backend(seismic::BackendName::Cpu),
    );
    // A load chains once it is priced (`chain_load`).
    with_room(&mut domain, 1 << 30);
    measure_load(&mut domain, &mut bindings);
    open(&mut domain, &mut bindings, request);
    domain.plan_successors(vec![prefill(request, 1), prefill(request, 2)]);
    let mut flight = submit_reserved_target(&mut domain, bindings, vec![prefill(request, 0)]);
    assert_eq!(domain.queue.len(), 2, "the chain queues two chunks");
    let control = attach_pending_priming(&mut domain, &mut flight, request, TokenId(1));
    let (pending, priming) = publish_before_priming(&mut domain, flight);
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
        .unwrap();
    assert_eq!(domain.chain_holder(), Some(request));
    control.resolve(Ok(()));
    let (_, _, mut bindings) = domain.finish_priming(priming).unwrap();
    assert_eq!(domain.head[&request].position(), 1, "the entry committed with its chunk");
    for position in 1..3 {
        let flight = submit_claimed(&mut domain, bindings, vec![prefill(request, position)]);
        let (pending, returned) = finish_one(&mut domain, flight);
        bindings = returned;
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
            .unwrap();
    }
    assert_eq!(domain.chain_holder(), None);
    let checkpoint = domain.resume_state(request).unwrap();
    assert_eq!(checkpoint.target.position(), 3);
    assert_eq!(checkpoint.head.as_ref().unwrap().position(), 1);
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
            crate::test_support::open_backend(&catalog, backend)
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
    let device = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu)
        .unwrap();
    let table = Arc::new(
        StoredTable::new(
            &device,
            &magnitude_family_contracts::WeightDescriptor::stored("ngram", [ROWS, 160]),
            seismic::Element::named("gguf_iq4_nl").unwrap(),
            seismic::Element::named("iq4g32s@rows16").unwrap(),
            ROWS,
            160,
            crate::stored_table::StoredSlots::Whole { records: 1, record_rows: 64 },
            crate::planning::DirectoryForm::Hashed,
            vec![crate::stored_table::RowRange {
                source: seismic_io::Reader::open(&path, seismic_io::CachePolicy::Uncached).unwrap(),
                first: 0,
                bytes: ROW_BYTES as usize,
            }],
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
    assert!(step.gate.wait_signalled(step.value, std::time::Duration::from_secs(10)).unwrap());
    // The device read the step's rows.
    step.read.reach(step.value).unwrap();
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

/// The supply a settle reads from the latest probe: free room above the
/// planning line, the shortfall below it owed, borrowed room up to the
/// line borrowing is held within, at no price while no other program was
/// seen asking and at `takes / (elapsed × t)` once one was, `t` the mean
/// rate of its asks, and the
/// grant's hard limit where one bounds the store.
#[test]
fn the_supply_reads_the_latest_probe() {
    const GB: u64 = 1_000_000_000;
    let store = seismic::Devices::discover(&[]).unwrap().topology().host_ram().id();
    let now = Instant::now();
    let pace_of = |headroom: u64, takes: u32| DomainPace {
        role: DomainRole::Allocation,
        headroom,
        thresholds: crate::platform::DomainThresholds { planning_bytes: 2 * GB, emergency_bytes: GB },
        charged: 0,
        band: if headroom > 2 * GB { MemoryBand::Normal } else { MemoryBand::Reclaim },
        unseen: 0,
        witness: None,
        untaken: Default::default(),
        rate: 0.0,
        fluctuation: Fluctuation::default(),
        host: true,
        grant_left: Some(30 * GB),
        borrowable: Some(20 * GB),
        takes: std::num::NonZeroU32::new(takes).map(|count| Takes { count, rate_sum: f64::from(takes) * 4.0 * GB as f64 }),
        first: now - Duration::from_secs(10),
    };
    let mut pace = MemoryPace::new();
    pace.read = Some(now);
    pace.domains.insert(store, pace_of(5 * GB, 0));
    let supply = pace.supply(store, 0, Duration::from_secs(60)).unwrap();
    assert_eq!((supply.free, supply.owed, supply.borrowable, supply.ceiling), (3 * GB, 0, 15 * GB, 30 * GB));
    assert_eq!(supply.borrow_price, 0.0, "no ask seen: borrowing harms no one");
    assert_eq!(supply.horizon, Duration::from_secs(60));
    // Two asks in 10 s, at 4 GB a second.
    pace.domains.insert(store, pace_of(GB + GB / 2, 2));
    let short = pace.supply(store, 0, Duration::from_secs(60)).unwrap();
    assert_eq!((short.free, short.owed), (0, GB / 2));
    assert!((short.borrow_price - 2.0 / (10.0 * 4.0 * GB as f64)).abs() < 1e-24);
}
