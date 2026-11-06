use super::*;

/// A shrink plan its owner drops uncommitted (its moves never queued)
/// holds the records it named no more for any holder of residency's lock,
/// whichever thread and before residency's own turn: the request service
/// taking a prompt group's first request then lends the lender's bottom
/// records, which the plan had held as its move targets, with no panic.
#[test]
fn a_dropped_shrink_plan_is_undone_for_every_holder_of_residency() {
    let harness = lending_bank("dropped-plan", EXPERTS, 4, 1, STAGING, WHOLE, TIER, Placer::Tier, true, 2, false);
    let lent = |record: u32| harness.watcher.bank_lent(0, record);
    harness.promoted(0, &[0, 1, 2, 3]);
    let shared = harness.watcher.shared();
    let pending = || residency::lock(&harness.watcher.state).core.directory.shrink_pending();
    let plan = shared.plan_shrink(Candidates::one(2, vec![2]), None).unwrap();
    assert!(pending(), "the plan holds the records it names");
    drop(plan);
    assert!(!pending(), "dropped, it holds none for the next holder of the lock");
    let step = harness.prompt_step();
    harness.scratch(&step, 0, &[5]);
    eventually("the group was never lent the records", || lent(0) && lent(1));
    drop(step.complete());
}

/// A shrink runs only between flights. It keeps the most recently read
/// experts: its plan evicts the rest at once (their slot map entries
/// cleared), while the kept ones above the new commitment stay mapped where
/// their bytes are until the plan, its moves carried, is committed; then
/// their map entries follow, from the first step not yet begun, and the
/// records it carries are the telemetry's. Its owner finds what each layer
/// holds published as the plan is answered. A dropped plan leaves every kept
/// expert where it was. A regrowth joins beside a running step and adds free records, which the
/// next seeding takes.
#[test]
fn a_resize_between_flights_keeps_the_most_recent_experts() {
    let harness = prompted("resize", EXPERTS, 4);
    // Seeds take the free records in order, the higher expert first: 4 and
    // 3 records 0 and 1, then 2 and 1 records 2 and 3; a decode step then
    // reads 1 and 2 in the bank.
    assert_eq!(harness.seeded(0, &[3, 4]), [(4, 0), (3, 1)]);
    assert_eq!(harness.seeded(0, &[1, 2]), [(2, 2), (1, 3)]);
    let shared = harness.watcher.shared();
    let steps = shared.costs().hints.layer_steps;
    let read = harness.request(&[1, 2]);
    assert!(!harness.layer.1.requested(read.sequence(), 0), "the bank serves both");
    let sequence = read.sequence();
    read.complete().unwrap();
    // Its reads rank once residency has recorded its completion and the
    // ranking's stack is fed them.
    eventually("the read was never recorded", || shared.costs().hints.layer_steps > steps);
    eventually("the read was never ranked", || shared.telemetry.fed_all());
    assert!(shared.in_gap());
    let running = harness.step();
    assert!(!shared.in_gap(), "no gap while a step runs");
    assert!(shared.plan_shrink(Candidates::one(2, vec![2]), None).is_err(), "no resize while a step runs");
    drop(running);
    assert!(shared.in_gap(), "the step's end is a gap");
    // Experts 2 and 1 move into records 1 and 0, which 3 and 4 leave; the
    // plan evicts 3 and 4 and moves nothing yet.
    let plan = shared.plan_shrink(Candidates::one(2, vec![2]), None).unwrap();
    assert_eq!(plan.moves().collect::<Vec<_>>(), [vec![(2, 1), (3, 0)]]);
    let map = &harness.layer.0;
    let mapped = || (1..=4).map(|expert| map.held(expert).map_or(-1, |record| record as i32)).collect::<Vec<_>>();
    assert_eq!(mapped(), [3, 2, -1, -1], "evicted at once, the kept where their bytes are");
    assert_eq!(shared.holdings().0, [2], "the plan's evictions are published as it is answered");
    assert_eq!(shared.costs().records, [4]);
    drop(plan);
    assert_eq!(mapped(), [3, 2, -1, -1], "a dropped plan moves nothing");
    // Planned again, the kept move into the records the evictions freed, in
    // record order, the most valuable first: 2 into record 0, 1 into 1.
    let plan = shared.plan_shrink(Candidates::one(2, vec![2]), None).unwrap();
    shared.commit_shrink(plan).unwrap();
    assert_eq!(mapped(), [1, 0, -1, -1]);
    let next = sequence + 2;
    assert_eq!((map.resolve(1, next - 1), map.resolve(1, next)), (None, Some(1)), "moved for the steps not yet begun");
    assert_eq!(shared.costs().records, [2]);
    assert_eq!(shared.held(), [vec![1, 2]]);
    // A regrowth adds free records, which the next seeds take; it joins
    // beside a running step, at the watcher's next pass.
    let running = harness.step();
    shared.join(super::Join::Bank { records: vec![4] });
    shared.joins.wait_taken();
    drop(running);
    assert_eq!(shared.costs().records, [4]);
    assert_eq!(harness.seeded(0, &[5, 6]), [(6, 2), (5, 3)]);
}

/// The placed parts' words a group's places hold are handed over between
/// flights alone (`Shared::place`): from then on the watcher reads each
/// place's scratch flags from the new words and the telemetry reads each
/// group's reads from them.
#[test]
fn a_groups_places_are_served_from_the_words_handed_over_between_flights() {
    let mut harness = layered_over("place", EXPERTS, 12, 2, STAGING, WHOLE);
    let shared = Arc::clone(harness.watcher.shared());
    let device = cpu();
    let words = |count: u64| cpu_words(&device, count, 0);
    let flags = [words(EXPERTS as u64), words(EXPERTS as u64)];
    let reads = harness
        .layers
        .iter()
        .map(|_| words(crate::FlightDepth::PIPELINED.steps(PLACES) * EXPERTS as u64))
        .collect::<Vec<_>>();
    let placed = || PlacedWords {
        flags: flags.to_vec(),
        layers: reads.iter().map(|reads| PlacedLayerWords { group_reads: reads.clone(), stride: EXPERTS }).collect(),
    };
    let running = harness.step();
    assert!(shared.place(placed()).is_err(), "no hand-off while a step runs");
    drop(running);
    shared.place(placed()).unwrap();
    // The harness plays the device through the handed-over words.
    let [first, second] = flags;
    (harness.load.0, harness.second) = (first, second);
    harness.reads = reads;
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    assert_eq!(harness.converted(&steps[0], 0, &[0, 1]).concat().len(), 2);
    assert_eq!(harness.converted(&steps[1], 0, &[1, 2]).concat(), [2], "each place served");
    for step in steps {
        step.complete().unwrap();
    }
    let steps = begin(&shared, &[StepRoute::Grouped; 2], false).unwrap();
    let record = steps[0].group_reads().unwrap();
    harness.reads[1].store((record * 2 + 1) * EXPERTS + 6, 1);
    let last = steps[1].sequence();
    for step in steps {
        step.complete().unwrap();
    }
    let read = shared.telemetry.last_read();
    assert_eq!(read(1, 6), Some(last), "the group's reads, read from the new words");
}

/// A host tier's shrink keeps the records a prompt group the device is past
/// still holds claimed in place: a group whose step ended while its
/// conveyor's reads were still queued (behind other demand reads) holds its
/// lender's records until those reads end, and the shrink, asked between
/// flights meanwhile, answers without waiting for them.
#[test]
fn a_host_shrink_answers_beside_a_past_prompt_group_still_clearing() {
    let _alone = alone();
    let harness = hosted_prompted("shrink-clearing", 12, 1, 16);
    let ahead = Ahead::queue("shrink-clearing-ahead", 384);
    let step = harness.prompt_step();
    harness.publish(&step, 0, &[4, 5]);
    std::thread::sleep(Duration::from_millis(50));
    drop(step);
    let claimed = || {
        let state = residency::lock(&harness.watcher.state);
        let directory = &state.core.directory;
        (0..directory.host(0).records()).filter(|&record| directory.lent_claimed(0, crate::experts::directory::HostRecord(record))).collect::<Vec<_>>()
    };
    let held = claimed();
    assert!(!held.is_empty(), "the group holds its lender's records claimed");
    let shared = harness.watcher.shared().clone();
    let shrink = std::thread::spawn(move || shared.shrink_tier(Candidates::one(12, vec![12])));
    eventually("the shrink never answered while the group cleared", || shrink.is_finished());
    shrink.join().unwrap().expect("the shrink is served");
    assert_eq!(claimed(), held, "the claimed records stayed in place");
    ahead.finish();
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
}

/// A host tier's shrink asked with a landing in flight answers at once,
/// cut at the deepest of its candidates that keeps the landing's record: a
/// demand read queued behind others has taken the lowest ready record, just
/// above the three the earlier misses landed in, so a shrink toward one
/// record cuts just above it, and the read ends into that record once the
/// reads ahead do.
#[test]
fn a_host_shrink_asked_with_a_landing_in_flight_answers_at_once() {
    let _alone = alone();
    let harness = layered("shrink-landing", EXPERTS, 4, 1, 4);
    for expert in 0..3 {
        harness.request(&[expert]).complete().unwrap();
    }
    harness.tier_ready();
    // The landing ends past its request, so it leaves flight the moment it
    // ends, kept (`HostRecords::answering`): the reads ahead must outlast the
    // shrink and its checks. Sixteen uncached 16 MiB reads end within tens
    // of milliseconds on a fast disk.
    let ahead = Ahead::queue("shrink-landing-ahead", 384);
    let served = || harness.watcher.shared().costs().layers[0].unanticipated;
    let before = served();
    let step = harness.step();
    assert!(harness.ask_at(&step, 0, &[5]));
    eventually("the request was never served", || served() > before);
    harness.layer.1.answer(step.sequence(), 0, true);
    drop(step);
    let floor = residency::lock(&harness.watcher.state).core.directory.host_floor(0);
    assert!((2..7).contains(&floor), "the landing's record, {}, lies below the tier's top", floor - 1);
    let shared = harness.watcher.shared().clone();
    let candidates = Candidates { shallowest: 6, deepest: 1, records: Box::new(|slots| vec![slots]) };
    let shrink = std::thread::spawn(move || shared.shrink_tier(candidates));
    eventually("the shrink never answered beside the landing", || shrink.is_finished());
    let cut = shrink.join().unwrap().expect("the shrink is served").expect("a cut keeps the landing");
    assert_eq!(cut.slots, floor as usize, "the deepest cut above the landing");
    assert_eq!(residency::lock(&harness.watcher.state).core.directory.host(0).records(), floor);
    ahead.finish();
    eventually("the landing never ended in the record kept for it", || {
        residency::lock(&harness.watcher.state).core.directory.host_floor(0) < floor
    });
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
}

/// A host tier's shrink asked right after a growth, with a landing in
/// flight, answers at once at the size it grew from: a landing takes the
/// lowest ready record, below the grown ones a shrink cuts first, so no
/// growth leaves a landing where its shrink cuts. Beside seven records,
/// three kept and the reserve's four ready, grown by one, a demand read
/// queued behind others takes record 3, and a shrink back to seven cuts
/// there at once.
#[test]
fn a_host_shrink_right_after_a_growth_answers_beside_its_landings() {
    let _alone = alone();
    let harness = layered("shrink-grown", EXPERTS, 4, 1, 4);
    for expert in 0..3 {
        harness.request(&[expert]).complete().unwrap();
    }
    harness.tier_ready();
    let shared = harness.watcher.shared().clone();
    let step = harness.step();
    shared.join(super::Join::Landed { records: vec![8], views: vec![harness.host.views(8)] });
    shared.joins.wait_taken();
    drop(step);
    // Enough reads ahead to outlast the shrink, as in
    // `a_host_shrink_asked_with_a_landing_in_flight_answers_at_once`.
    let ahead = Ahead::queue("shrink-grown-ahead", 384);
    let floor = || residency::lock(&harness.watcher.state).core.directory.host_floor(0);
    let served = || harness.watcher.shared().costs().layers[0].unanticipated;
    let before = served();
    let step = harness.step();
    assert!(harness.ask_at(&step, 0, &[5]));
    eventually("the demand never took a record", || floor() != 0);
    eventually("the request was never served", || served() > before);
    harness.layer.1.answer(step.sequence(), 0, true);
    drop(step);
    assert_eq!(floor(), 4, "the landing took the lowest ready record, below the growth");
    let candidates = Candidates { shallowest: 7, deepest: 7, records: Box::new(|slots| vec![slots]) };
    let shrink = std::thread::spawn(move || shared.shrink_tier(candidates));
    eventually("the shrink never answered beside the landing", || shrink.is_finished());
    let cut = shrink.join().unwrap().expect("the shrink is served").expect("the cut answers, not deferred");
    assert_eq!(cut.slots, 7, "back to the size it grew from");
    assert_eq!(residency::lock(&harness.watcher.state).core.directory.host(0).records(), 7);
    ahead.finish();
    eventually("the landing never ended in the record kept for it", || harness.in_tier(0, 5) == Some(3));
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
}

/// A host tier's shrink on an idle load answers at once beside a demand
/// landing whose step has ended: the step's wait gave up on the read
/// queued behind others and the step ended, so the device is past the
/// request, and no step follows; the read then lands, kept rather than
/// its request's outstanding answer, so nothing is in flight and the
/// shrink cuts at its deepest candidate.
#[test]
fn a_host_shrink_on_an_idle_load_answers_beside_a_landing_its_step_passed() {
    let _alone = alone();
    let harness = layered("shrink-idle", EXPERTS, 4, 1, 4);
    for expert in 0..3 {
        harness.request(&[expert]).complete().unwrap();
    }
    harness.tier_ready();
    let ahead = Ahead::queue("shrink-idle-ahead", 16);
    let floor = || residency::lock(&harness.watcher.state).core.directory.host_floor(0);
    let served = || harness.watcher.shared().costs().layers[0].unanticipated;
    let before = served();
    let step = harness.step();
    assert!(harness.ask_at(&step, 0, &[5]));
    eventually("the demand never took a record", || floor() != 0);
    eventually("the request was never served", || served() > before);
    harness.layer.1.answer(step.sequence(), 0, true);
    drop(step);
    ahead.finish();
    eventually("the landing never ended in its record", || harness.in_tier(0, 5) == Some(3));
    eventually("the landing past its request was left in flight", || floor() == 0);
    let shared = harness.watcher.shared().clone();
    let candidates = Candidates { shallowest: 6, deepest: 1, records: Box::new(|slots| vec![slots]) };
    let cut = shared.shrink_tier(candidates).expect("the shrink is served").expect("the cut answers, not deferred");
    assert_eq!(cut.slots, 1, "the deepest candidate");
    assert_eq!(residency::lock(&harness.watcher.state).core.directory.host(0).records(), 1);
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
}

/// A filled tier whose growth is decided fills again from that decision,
/// through every slot that joins and the fill of each, until the growth
/// ends: its slots joining one by one never leave it filled between them.
/// Two layers' sixteen experts beside fifteen tier records, eight the
/// reserve, fill seven; grown to sixteen, the tier fills every expert.
#[test]
fn a_filled_tier_fills_again_while_it_grows() {
    let harness = layered("tier-grows", EXPERTS, 12, 2, 4 + 4);
    let shared = harness.watcher.shared();
    let tiered = || (0..2).map(|layer| (0..EXPERTS).filter(|&expert| harness.in_tier(layer, expert).is_some()).count()).sum::<usize>();
    eventually("the tier never filled", || shared.tier_fill() == crate::TierFill::Filled && tiered() == 7);
    shared.join(super::Join::Growing);
    eventually("the decided growth never made the tier fill", || shared.tier_fill() == crate::TierFill::Filling);
    shared.join(super::Join::Landed { records: vec![16], views: vec![harness.host.views(16)] });
    eventually("the joined record was never filled", || tiered() == 2 * EXPERTS);
    assert_eq!(shared.tier_fill(), crate::TierFill::Filling, "filled between its slots, the growth still runs");
    shared.join(super::Join::Grown);
    shared.wait_filled().expect("the grown tier fills");
    assert_eq!(shared.tier_fill(), crate::TierFill::Filled);
}

/// A host tier grows by its records and the views that address them,
/// joined beside a running step, and its new record is the next a landing
/// takes: beside a tier of its reserve and one record more, every one lent
/// (so the grown record is the lowest ready past them), which expert 0
/// keeps (none unmapped, so none graced), expert 1 from the file lands into
/// the grown record, its bytes there once landed.
#[test]
fn a_host_tier_grows_by_its_records_alone() {
    let harness = lending("host-grow", EXPERTS, 4, 1, STAGING, Conveying::of(u64::from(STAGING) + 1), 0, Placer::Tier, false);
    harness.request(&[0]).complete().unwrap();
    let shared = harness.watcher.shared();
    let step = harness.step();
    shared.join(super::Join::Landed { records: vec![STAGING + 2], views: vec![harness.host.views(STAGING + 2)] });
    shared.joins.wait_taken();
    drop(step);
    let step = harness.request(&[1]);
    let grown = STAGING as i32 + 1;
    assert_eq!(harness.answer(0, step.sequence(), 0), (grown, Some(Answer::Answered)), "the grown record");
    // Residency maps the landing as it learns it, which may follow the answer.
    eventually("expert 1 never mapped to the grown record", || harness.in_tier(0, 1) == Some(grown));
    assert!(harness.in_tier(0, 0).is_some(), "expert 0 kept");
    step.complete().unwrap();
}
