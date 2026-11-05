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
/// records it carries are the telemetry's. A dropped plan leaves every kept
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
    let steps = harness.group(&[StepRoute::Prompt; 2]);
    assert_eq!(harness.converted(&steps[0], 0, &[0, 1]).concat().len(), 2);
    assert_eq!(harness.converted(&steps[1], 0, &[1, 2]).concat(), [2], "each place served");
    for step in steps {
        step.complete().unwrap();
    }
    let steps = begin(&shared, &[StepRoute::Prompt; 2], false).unwrap();
    let record = steps[0].group_reads().unwrap();
    harness.reads[1].store((record * 2 + 1) * EXPERTS + 6, 1);
    let last = steps[1].sequence();
    for step in steps {
        step.complete().unwrap();
    }
    let read = shared.telemetry.last_read();
    assert_eq!(read(1, 6), Some(last), "the group's reads, read from the new words");
}

/// A host tier shrinks only once every prompt group the device is past has
/// cleared, returning the records it claimed: a group whose step ended while
/// its conveyor's reads were still queued (behind other demand reads) holds
/// its lender's records until those reads end, and the shrink, asked
/// between flights meanwhile, waits for them rather than finding a claimed
/// record.
#[test]
fn a_host_shrink_waits_for_a_past_prompt_group_to_clear() {
    let _alone = alone();
    let harness = hosted_prompted("shrink-clearing", 12, 1, 16);
    let ahead = Ahead::queue("shrink-clearing-ahead", 384);
    let step = harness.prompt_step();
    harness.publish(&step, 0, &[4, 5]);
    std::thread::sleep(Duration::from_millis(50));
    drop(step);
    let shared = harness.watcher.shared().clone();
    let views = harness.host.views(12);
    let shrink = std::thread::spawn(move || shared.shrink_host(vec![12], vec![views]));
    ahead.finish();
    shrink.join().unwrap().expect("the shrink ends once the group cleared");
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
/// takes: beside a tier of its reserve and one record more, which expert 0
/// keeps (none unmapped, so none graced), expert 1 from the file lands into
/// the grown record, its bytes there once landed.
#[test]
fn a_host_tier_grows_by_its_records_alone() {
    let harness = hosted("host-grow", 4, STAGING + 1);
    harness.request(&[0]).complete().unwrap();
    let shared = harness.watcher.shared();
    let step = harness.step();
    shared.join(super::Join::Landed { records: vec![STAGING + 2], views: vec![harness.host.views(STAGING + 2)] });
    shared.joins.wait_taken();
    drop(step);
    let step = harness.request(&[1]);
    let grown = STAGING as i32 + 1;
    assert_eq!(harness.answer(0, step.sequence(), 0), (grown, Some(Answer::Answered)), "the grown record");
    assert_eq!(harness.in_tier(0, 1), Some(grown));
    assert!(harness.in_tier(0, 0).is_some(), "expert 0 kept");
    step.complete().unwrap();
}
