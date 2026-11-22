use super::*;
use super::super::resize::Evicting;

/// The records of group 0's host tier a shrink may not evict now: in
/// flight, held, or lent (`Directory::host_evictions`).
fn unevictable(harness: &Harness) -> Vec<u32> {
    let state = residency::lock(&harness.watcher.state);
    let directory = &state.core.directory;
    let evictable = directory.host_evictions(0, 0);
    directory.host(0).backed().filter(|record| !evictable.contains(record)).collect()
}

/// A shrink keeps the most recently read experts where their bytes are and
/// evicts the rest, moving nothing: each evicted expert's record is a hole
/// from the first step not yet begun, while the steps before still read it
/// there. Its owner finds what each layer holds published as the shrink is
/// answered. A regrowth backs the holes again beside a running step, and
/// the next seeding takes them.
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
    drop(harness.step());
    // Experts 4 and 3, least recently read, leave records 0 and 1; 2 and 1
    // stay in records 2 and 3.
    let evicted = shared.shrink(Evicting::Bank, Candidates::one(2, vec![2]), None).unwrap().expect("the shrink evicts beside no step");
    let mut records = evicted.records[0].clone();
    records.sort();
    assert_eq!((evicted.slots, records), (2, vec![0, 1]));
    let map = &harness.layer.0;
    let mapped = || (1..=4).map(|expert| map.held(expert).map_or(-1, |record| record as i32)).collect::<Vec<_>>();
    assert_eq!(mapped(), [3, 2, -1, -1], "evicted, the kept where their bytes are");
    let next = sequence + 2;
    assert_eq!(
        (map.resolve(4, next - 1), map.resolve(4, next)),
        (Some(0), None),
        "evicted from the first step not yet begun, read where it was by the step before"
    );
    assert_eq!(shared.holdings().0, [2], "the evictions are published as the shrink is answered");
    assert_eq!(shared.costs().records, [2]);
    assert_eq!(shared.held(), [vec![1, 2]]);
    // A regrowth backs the holes, which the next seeds take; it joins
    // beside a running step, at the watcher's next pass.
    let running = harness.step();
    shared.join(super::Join::Bank { records: vec![vec![0, 1]] });
    shared.joins.wait_taken();
    drop(running);
    assert_eq!(shared.costs().records, [4]);
    assert_eq!(harness.seeded(0, &[5, 6]), [(6, 0), (5, 1)]);
}

/// A device bank's shrink lands beside a running chain, at residency's
/// next turn, not at the chain's end, and moves nothing: the bank's records
/// allow holes, so the records it evicts are exactly those of the experts
/// it evicts, which leave the map for the steps not yet begun while the
/// running step still reads them, and every kept expert stays where its
/// bytes are. Four experts promoted into
/// a device bank of four records beside its host tier, a decode step
/// running, the bank shrunk to two.
#[test]
fn a_device_banks_shrink_lands_beside_a_running_chain() {
    let harness = promoting("running-device-shrink", 4, TIER);
    harness.promoted(0, &[0, 1, 2, 3]);
    let shared = harness.watcher.shared();
    let map = &harness.layer.0;
    let before = (0..4).map(|expert| map.held(expert)).collect::<Vec<_>>();
    let running = harness.step();
    let evicted = shared
        .shrink(Evicting::Bank, Candidates::one(2, vec![2]), None)
        .unwrap()
        .expect("a device bank shrinks beside a running chain");
    let kept = (0..4).filter(|&expert| map.held(expert).is_some()).collect::<Vec<_>>();
    assert_eq!(kept.len(), 2, "the bank keeps two experts");
    for &expert in &kept {
        assert_eq!(map.held(expert), before[expert], "a kept expert stays where its bytes are");
    }
    let gone = (0..4).filter(|expert| !kept.contains(expert)).collect::<Vec<_>>();
    let mut records = evicted.records[0].clone();
    records.sort();
    let mut held = gone.iter().map(|&expert| before[expert].expect("every expert was promoted") as u32).collect::<Vec<_>>();
    held.sort();
    assert_eq!(records, held, "the evicted records are those of the evicted experts");
    for &expert in &gone {
        assert_eq!(map.resolve(expert, running.sequence()), before[expert], "the running step still reads an evicted expert");
        assert_eq!(map.held(expert), None, "no step not yet begun reads an evicted expert");
    }
    assert_eq!(shared.costs().records, [2]);
    drop(harness.route_at(&running, 0, &[]));
    running.complete().unwrap();
}

/// A device bank's shrink lands beside a real running chain, the chain its
/// running step begins: the experts none of the chain's requests routed at
/// their layer leave the map from the chain's first step, so its steps still
/// to decide their hits resolve them a miss, and the bank keeps the rest
/// where their bytes are.
#[test]
fn a_device_banks_shrink_lands_beside_its_running_chain() {
    let harness = promoting("running-chain-shrink", 4, TIER);
    harness.promoted(0, &[0, 1, 2, 3]);
    let shared = harness.watcher.shared();
    let map = &harness.layer.0;
    let before = (0..4).map(|expert| map.held(expert)).collect::<Vec<_>>();
    let running = harness.prompt_step();
    let chain = running.scratch_group().expect("the running chain is scratch-served");
    let evicted = shared
        .shrink(Evicting::Bank, Candidates::one(2, vec![2]), Some(chain))
        .unwrap()
        .expect("a device bank shrinks beside its running chain");
    let kept = (0..4).filter(|&expert| map.held(expert).is_some()).collect::<Vec<_>>();
    assert_eq!(kept.len(), 2, "the bank keeps two experts");
    for &expert in &kept {
        assert_eq!(map.held(expert), before[expert], "a kept expert stays where its bytes are");
    }
    let gone = (0..4).filter(|expert| !kept.contains(expert)).collect::<Vec<_>>();
    assert_eq!(evicted.records[0].len(), gone.len(), "the evicted records are those of the evicted experts");
    for &expert in &gone {
        assert_eq!(map.resolve(expert, running.sequence()), None, "an expert the chain never routed leaves from its first step");
        assert_eq!(map.held(expert), None, "no step not yet begun reads an evicted expert");
    }
    assert_eq!(shared.costs().records, [2]);
    harness.scratch(&running, 0, &[]);
    running.complete().unwrap();
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
    (harness.flags, harness.second) = (first, second);
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
    for step in steps {
        step.complete().unwrap();
    }
    assert!(shared.telemetry.group_union(1), "the group's reads, read from the new words");
    let state = residency::lock(&harness.watcher.state);
    let reuse = state.core.directory.reuse_now();
    assert!(
        reuse.rank(1, 6, crate::experts::horizon::Tier::COMPUTE, crate::experts::horizon::Reader::Move, None)
            > reuse.rank(1, 0, crate::experts::horizon::Tier::COMPUTE, crate::experts::horizon::Reader::Move, None),
        "expert 6 ranked above unrouted experts from the handed-over group_reads words"
    );
}

/// A host tier's shrink keeps the records a prompt group the device is past
/// still holds claimed in place: a group whose step ended while its
/// conveyor's reads were still in flight (held, `Conveyor::hold_reads`)
/// holds its lender's records until those reads end, and the shrink, asked
/// between flights meanwhile, answers without waiting for them. Once the
/// reads end, the group clears and its claims return.
#[test]
fn a_host_shrink_answers_beside_a_past_prompt_group_still_clearing() {
    let harness = hosted_prompted("shrink-clearing", 12, 1, 16);
    let hold = |held: bool| residency::lock(&harness.watcher.state).scratch.conveyor.hold_reads(held);
    hold(true);
    let step = harness.prompt_step();
    harness.publish(&step, 0, &[4, 5]);
    eventually("the group's reads were never submitted", || {
        residency::lock(&harness.watcher.state).scratch.conveyor.reads_in_flight() > 0
    });
    drop(step);
    let claimed = || {
        let state = residency::lock(&harness.watcher.state);
        let directory = &state.core.directory;
        directory.host(0).backed().filter(|&record| directory.lent_claimed(0, crate::experts::directory::HostRecord(record))).collect::<Vec<_>>()
    };
    let held = claimed();
    assert!(!held.is_empty(), "the group holds its lender's records claimed");
    let shared = harness.watcher.shared().clone();
    let shrink = std::thread::spawn(move || shared.shrink(Evicting::Landed, Candidates::one(12, vec![12]), None));
    eventually("the shrink never answered while the group cleared", || shrink.is_finished());
    shrink.join().unwrap().expect("the shrink is served");
    assert_eq!(claimed(), held, "the claimed records stayed in place");
    hold(false);
    harness.watcher.shared().events().notify();
    eventually("the group never cleared once its reads ended", || claimed().is_empty());
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
}

/// A host tier's shrink asked with a landing in flight answers at once,
/// evicting around the landing's record: a demand read queued behind others
/// has taken a ready record of the seven, so a shrink toward one record
/// evicts the six others, and the read ends into the record kept for it
/// once the reads ahead do.
#[test]
fn a_host_shrink_asked_with_a_landing_in_flight_answers_at_once() {
    let _alone = alone();
    let harness = layered("shrink-landing", EXPERTS, 4, 1, 4);
    for expert in 0..3 {
        harness.request(&[expert]).complete().unwrap();
    }
    harness.tier_ready();
    // The landing ends past its request, so it leaves flight the moment it
    // ends, kept (`HostRecords::answering`): it must stay queued behind the
    // line until the shrink is served, so the line is held, its workers
    // released only by `finish` below (`Ahead::held`).
    let ahead = Ahead::held("shrink-landing-ahead");
    let served = || harness.watcher.shared().costs().layers[0].unanticipated;
    let before = served();
    let step = harness.step();
    assert!(harness.ask_at(&step, 0, &[5]));
    eventually("the demand never took a record", || !unevictable(&harness).is_empty());
    eventually("the request was never served", || served() > before);
    harness.layer.1.answer(step.sequence(), 0, true);
    drop(step);
    let landing = match unevictable(&harness)[..] {
        [record] => record,
        ref records => panic!("the landing's record alone is in flight, not {records:?}"),
    };
    let shared = harness.watcher.shared().clone();
    let candidates = Candidates { shallowest: 6, deepest: 1, records: Box::new(|slots| vec![slots]) };
    let shrink = std::thread::spawn(move || shared.shrink(Evicting::Landed, candidates, None));
    eventually("the shrink never answered beside the landing", || shrink.is_finished());
    let evicted = shrink.join().unwrap().expect("the shrink is served").expect("an eviction keeps the landing");
    assert_eq!(evicted.slots, 1, "the deepest candidate, around the landing");
    assert!(!evicted.records[0].contains(&landing), "the landing's record is kept");
    assert_eq!(residency::lock(&harness.watcher.state).core.directory.host(0).records(), 1);
    ahead.finish();
    // At 1 record (< reserve 4), `refill_tier` immediately unmaps the landed
    // record back into the ready reserve once residency settles its landing:
    // its bytes are whole in `landing` and `landing` leaves flight.
    eventually("the landing never ended in the record kept for it", || {
        residency::lock(&harness.watcher.state).core.directory.host_evictions(0, 0) == [landing]
            && (0..3).all(|kind| harness.tiered(kind, landing as i32) == (100 * kind + 5) as u8)
    });
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
}

/// A host tier's shrink asked right after a growth, with a landing in
/// flight, answers at once at the size it grew from, evicting around the
/// landing. Beside seven records, three kept and the reserve's four ready,
/// grown by one, a demand read queued behind others takes the lowest ready
/// record, 3, and a shrink back to seven evicts another at once.
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
    shared.join(super::Join::Landed { records: vec![vec![7]], views: vec![harness.host.views(8)] });
    shared.joins.wait_taken();
    drop(step);
    // The landing held in flight until the shrink is served, as in
    // `a_host_shrink_asked_with_a_landing_in_flight_answers_at_once`.
    let ahead = Ahead::held("shrink-grown-ahead");
    let served = || harness.watcher.shared().costs().layers[0].unanticipated;
    let before = served();
    let step = harness.step();
    assert!(harness.ask_at(&step, 0, &[5]));
    eventually("the demand never took a record", || !unevictable(&harness).is_empty());
    eventually("the request was never served", || served() > before);
    harness.layer.1.answer(step.sequence(), 0, true);
    drop(step);
    assert_eq!(unevictable(&harness), [3], "the landing took the lowest ready record");
    let candidates = Candidates { shallowest: 7, deepest: 7, records: Box::new(|slots| vec![slots]) };
    let shrink = std::thread::spawn(move || shared.shrink(Evicting::Landed, candidates, None));
    eventually("the shrink never answered beside the landing", || shrink.is_finished());
    let evicted = shrink.join().unwrap().expect("the shrink is served").expect("the eviction answers, not deferred");
    assert_eq!(evicted.slots, 7, "back to the size it grew from");
    assert!(!evicted.records[0].contains(&3), "the landing's record is kept");
    assert_eq!(residency::lock(&harness.watcher.state).core.directory.host(0).records(), 7);
    ahead.finish();
    eventually("the landing never ended in the record kept for it", || harness.in_tier(0, 5) == Some(3));
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
}

/// A growth that makes the host tier cover every expert joins at once and
/// starts the cover's reads, which queue behind every step's: the growth
/// never waits on them, and the tier is covered once they end.
#[test]
fn a_growth_covering_the_tier_joins_at_once_and_starts_its_reads() {
    let _alone = alone();
    let harness = lending("cover-grown", EXPERTS, 2, 1, 3 * STAGING, DECODE, EXPERTS as u32 - 1, Placer::Tier, false);
    let shared = harness.watcher.shared().clone();
    std::thread::sleep(Duration::from_millis(50));
    assert!((0..EXPERTS).all(|expert| harness.in_tier(0, expert).is_none()), "a tier short of every expert reads none ahead of use");
    let ahead = Ahead::queue("cover-grown-ahead");
    shared.join(super::Join::Landed { records: vec![vec![EXPERTS as u32 - 1]], views: vec![harness.host.views(EXPERTS as u32)] });
    shared.joins.wait_taken();
    assert_eq!(residency::lock(&harness.watcher.state).core.directory.host(0).records(), EXPERTS as u32, "the growth joined");
    eventually("the covered tier never read ahead", || residency::lock(&harness.watcher.state).prefetch.outstanding());
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
    ahead.finish();
    harness.tiering(0, &(0..EXPERTS).collect::<Vec<_>>());
}

/// A host tier's shrink on an idle load answers at once beside a demand
/// landing whose step has ended: the step's wait gave up on the read
/// queued behind others and the step ended, so the device is past the
/// request, and no step follows; the read then lands, kept rather than
/// its request's outstanding answer, so nothing is in flight and the
/// shrink evicts down to its deepest candidate.
#[test]
fn a_host_shrink_on_an_idle_load_answers_beside_a_landing_its_step_passed() {
    let _alone = alone();
    // 7 tier records with reserve 3: 3 incumbents plus the landing leave 3 ready
    // records (== reserve), so `refill_tier` does not evict the landed record.
    let harness = lending("shrink-idle", EXPERTS, 4, 1, 3, DECODE, 7, Placer::Tier, false);
    for expert in 0..3 {
        harness.request(&[expert]).complete().unwrap();
    }
    harness.tier_ready();
    // The demand read, as urgent as the line's, is taken only once `finish`
    // below ends the hold, after the step is given up.
    let ahead = Ahead::held("shrink-idle-ahead");
    let served = || harness.watcher.shared().costs().layers[0].unanticipated;
    let before = served();
    let step = harness.step();
    assert!(harness.ask_at(&step, 0, &[5]));
    eventually("the demand never took a record", || !unevictable(&harness).is_empty());
    eventually("the request was never served", || served() > before);
    harness.layer.1.answer(step.sequence(), 0, true);
    drop(step);
    ahead.finish();
    eventually("the landing never ended in its record", || harness.in_tier(0, 5) == Some(3));
    eventually("the landing past its request was left in flight", || unevictable(&harness).is_empty());
    let shared = harness.watcher.shared().clone();
    let candidates = Candidates { shallowest: 6, deepest: 1, records: Box::new(|slots| vec![slots]) };
    let evicted = shared.shrink(Evicting::Landed, candidates, None).expect("the shrink is served").expect("the eviction answers, not deferred");
    assert_eq!(evicted.slots, 1, "the deepest candidate");
    assert_eq!(residency::lock(&harness.watcher.state).core.directory.host(0).records(), 1);
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
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
    shared.join(super::Join::Landed { records: vec![vec![STAGING + 1]], views: vec![harness.host.views(STAGING + 2)] });
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

/// An eviction from the one store beside a running prompt group is
/// answered at once and moves nothing: each expert the evicted records kept
/// is unmapped from the group's first step, but one a request of the group
/// already routed at a layer whose hits it decided, which its later passes
/// there still resolve as that request's hit did, until the first step not
/// yet begun. The answer names that layer, whose passes the store's
/// storage is released after. The group's later requests read the evicted
/// experts as cold ones, from its conveyor, and nothing lands in an evicted
/// record. Asked once a later group began, whose steps decided their hits
/// by the map before it, it evicts nothing. Two layers of eight experts in
/// a store of fifteen records, its bottom eight lent, decode steps'
/// misses keeping seven in records 8 to 14, each landing into the lowest
/// ready past the lent ones; the shrink to the lent eight evicts every
/// kept one.
#[test]
fn an_eviction_beside_a_running_group_drops_all_it_has_not_decided() {
    const CUT: usize = 8;
    let harness = lending("running-cut", EXPERTS, 8, 2, 8, WHOLE, 7, Placer::Bank, false);
    let shared = harness.watcher.shared().clone();
    let layers = &harness.layers;
    let kept = || {
        (0..2)
            .flat_map(|layer| (0..EXPERTS).filter_map(move |expert| Some((layer, expert, layers[layer].0.held(expert)?))))
            .collect::<Vec<_>>()
    };
    // Layer 0's 0 and 1 and layer 1's 0 and 1 into records 8 to 11, then
    // layer 0's 2 and layer 1's 2 into 12 and 13, then layer 0's 3 into 14.
    let misses: [[&[usize]; 2]; 3] = [[&[0, 1], &[0, 1]], [&[2], &[2]], [&[3], &[]]];
    for routes in misses {
        let step = harness.step();
        for (layer, experts) in routes.into_iter().enumerate() {
            harness.request_at(&step, layer, experts);
        }
        step.complete().unwrap();
    }
    eventually("the misses never landed", || kept().len() == 7);
    let past = kept();
    let &(_, routed, at) = past.iter().find(|&&(layer, ..)| layer == 0).expect("the misses keep a layer 0 expert");
    // A step before the group, which reads the store as the misses left it.
    let before = begin(&shared, &[StepRoute::Decode], false).unwrap().pop().unwrap();
    let earlier = before.sequence();
    drop(before);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    let chain = steps[0].scratch_group().expect("the group is scratch-served");
    let (first, last) = (chain.first(), chain.last());
    // Step 0 routes `routed` at layer 0, a hit its pass reads in place.
    assert!(harness.scratch(&steps[0], 0, &[routed]).concat().is_empty(), "a hit is read in place");
    let candidates = Candidates { shallowest: 15, deepest: CUT, records: Box::new(|slots| vec![slots]) };
    let evicted = shared.shrink(Evicting::Landed, candidates, Some(chain)).unwrap().expect("the eviction keeps every record in flight");
    assert_eq!((evicted.slots, evicted.decided), (CUT, Some(0)), "the deepest candidate, its hits decided at layer 0");
    let mut records = evicted.records[0].clone();
    records.sort();
    assert_eq!(records, (CUT as u32..15).collect::<Vec<_>>(), "every kept record is evicted");
    assert_eq!(residency::lock(&harness.watcher.state).core.directory.host(0).records(), CUT as u32);
    let map = |layer: usize| &layers[layer].0;
    assert_eq!(map(0).resolve(routed, last), Some(at), "the group's later passes read its decided hit");
    assert_eq!(map(0).held(routed), None, "no step begun after the group reads it");
    for &(layer, expert, record) in past.iter().filter(|&&(layer, expert, _)| (layer, expert) != (0, routed)) {
        assert_eq!(map(layer).resolve(expert, earlier), Some(record), "a step before the group still reads it");
        assert_eq!(map(layer).resolve(expert, first), None, "no step of the group reads it");
    }
    // Step 1 reads a dropped layer 0 expert as a cold one, and both steps
    // every dropped one of layer 1.
    let dropped = |of: usize| past.iter().filter(|&&(layer, expert, _)| layer == of && expert != routed).map(|&(_, expert, _)| expert).collect::<Vec<_>>();
    let cold = |experts: &[usize]| experts.iter().map(|&expert| expert as i32).collect::<Vec<_>>();
    let (zero, one) = (dropped(0), dropped(1));
    assert_eq!(Harness::experts(&harness.scratch(&steps[1], 0, &[&[routed][..], &zero].concat()).concat()), cold(&zero));
    assert_eq!(Harness::experts(&harness.scratch(&steps[0], 1, &one).concat()), cold(&one));
    harness.scratch(&steps[1], 1, &[]);
    for step in steps {
        step.complete().unwrap();
    }
    assert!(kept().iter().all(|&(.., record)| !records.contains(&(record as u32))), "nothing lands in an evicted record");
    // A later group begun beside the chain: nothing is evicted.
    let chain = harness.group(&[StepRoute::Grouped; 2]);
    let later = begin(&shared, &[StepRoute::Decode], false).unwrap();
    let candidates = Candidates { shallowest: CUT, deepest: CUT, records: Box::new(|slots| vec![slots]) };
    let evicted = shared.shrink(Evicting::Landed, candidates, chain[0].scratch_group()).unwrap();
    assert!(evicted.is_none(), "a group began after the chain");
    assert_eq!(residency::lock(&harness.watcher.state).core.directory.host(0).records(), CUT as u32);
    drop((later, chain));
    assert_eq!(shared.life.ended(), None, "the watcher serves on");
}

/// A bank imported whole into the one store whose shrink evicts only the
/// tier's reserve keeps every expert at its home, but the store no longer
/// holds every record its commits span: no layer's whole word stays set, so
/// no step after the shrink binds the records directly in a resident form.
/// Two layers of eight experts at their homes in records 0 to 15, the
/// conveyor lent the bottom four, and a reserve of eight above them; the
/// shrink to twenty records evicts four of the reserve and keeps every
/// expert.
#[test]
fn a_cut_of_a_whole_banks_reserve_alone_ends_its_resident_form() {
    const RESERVE: u32 = 8;
    let conveying = Conveying::of(4);
    let tier = 2 * EXPERTS as u32 + RESERVE;
    let harness = lending_bank("whole-reserve-cut", EXPERTS, 2 * EXPERTS as u64, 2, STAGING, conveying, tier, Placer::Bank, false, 0, true);
    let whole = |layer: usize| harness.layers[layer].5.load(constants::POLICY_WHOLE as usize);
    assert!((0..2).all(|layer| whole(layer) == 1), "a bank imported whole is resident");
    let shared = harness.watcher.shared().clone();
    let candidates = Candidates { shallowest: 20, deepest: 20, records: Box::new(|slots| vec![slots]) };
    let evicted = shared.shrink(Evicting::Landed, candidates, None).expect("the shrink is served").expect("the eviction answers, not deferred");
    assert_eq!(evicted.slots, 20);
    assert!(evicted.records[0].iter().all(|&record| record >= 2 * EXPERTS as u32), "only the reserve is evicted");
    {
        let state = residency::lock(&harness.watcher.state);
        let directory = &state.core.directory;
        assert!(directory.group_whole(0) && directory.group_at_home(0), "the shrink kept every expert at its home");
    }
    eventually("a layer stayed resident beside evicted records", || (0..2).all(|layer| whole(layer) == 0));
    assert_eq!(shared.life.ended(), None, "the watcher serves on");
}
/// Admission is decided once at the claim (design §3.7, §6): a landing
/// whose step has ended before its read arrives still lands into the
/// record it claimed and stays kept there without being tested again,
/// even when its expert ranks below the tier's incumbents.
#[test]
fn a_landing_keeps_its_claimed_record() {
    let _alone = alone();
    // 7 tier records with reserve 3: 3 incumbents plus the landing leave 3 ready
    // records (== reserve), so `refill_tier` does not evict the landed record.
    let harness = lending("landing-keeps", EXPERTS, 4, 1, 3, DECODE, 7, Placer::Tier, false);
    for _ in 0..5 {
        for expert in 0..3 {
            harness.request(&[expert]).complete().unwrap();
        }
    }
    harness.tier_ready();
    let ahead = Ahead::held("landing-keeps-ahead");
    let served = || harness.watcher.shared().costs().layers[0].unanticipated;
    let before = served();
    let step = harness.step();
    assert!(harness.ask_at(&step, 0, &[5]));
    eventually("the demand never took a record", || !unevictable(&harness).is_empty());
    eventually("the request was never served", || served() > before);
    // The step gives up and ends while the read is still held; once it
    // arrives, it lands into its claimed record (3) without re-testing.
    harness.layer.1.answer(step.sequence(), 0, true);
    drop(step);
    ahead.finish();
    eventually("the landing never kept its claimed record", || harness.in_tier(0, 5) == Some(3));
    eventually("the landing past its request was left in flight", || unevictable(&harness).is_empty());
    assert!((0..3).all(|expert| harness.in_tier(0, expert).is_some()), "incumbents stay kept");
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
}

/// Attempting to acquire a `Pool` or `SlabTensor` allocation lock (or call
/// `Pool::grow`) while holding `residency` panics immediately, preventing
/// lock-order deadlocks against a device parked on the host answer path.
#[test]
fn entering_pool_or_slab_lock_while_holding_residency_panics() {
    let harness = prompted("no-blocking-alloc-guard", EXPERTS, 4);
    let device = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
    let charges = seismic::Charges::tally(&device);
    let regions = [seismic::SlabRegion { element: seismic::Element::i32(), row_shape: vec![16] }];
    let slabs = crate::experts::bank::pooled_slabs(&charges, seismic::Contract::Shared, 1, 4, &regions, 2).unwrap();
    let pool = slabs.pool().clone();

    fn assert_no_blocking_alloc_panic(label: &str, f: impl FnOnce() + std::panic::UnwindSafe) {
        let err = std::panic::catch_unwind(f).expect_err(&format!("{label} must panic under residency"));
        let msg = err
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| err.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(msg.contains("no-blocking-allocation scope"), "{label} panicked with unexpected message: {msg}");
    }

    {
        let _held = residency::lock(&harness.watcher.state);
        assert_no_blocking_alloc_panic("Pool::free_granules", std::panic::AssertUnwindSafe(|| {
            let _ = pool.free_granules();
        }));
        assert_no_blocking_alloc_panic("Pool::grow", std::panic::AssertUnwindSafe(|| {
            let _ = pool.grow();
        }));
        assert_no_blocking_alloc_panic("SlabTensor::backed", std::panic::AssertUnwindSafe(|| {
            let _ = slabs.backed();
        }));
    }

    // Outside residency, pool and slab operations succeed normally.
    assert!(pool.grow().is_ok());
    assert_eq!(slabs.add_slab(seismic::SlabFill::Unwritten).unwrap(), 0);
}

/// `Promotions` (including into bank records rejoined via `Join::Bank`) and
/// `Requests` (decode and scratch) never enter driver or pool allocation locks
/// while holding `residency` or on the host answer path.
#[test]
fn promotions_and_requests_never_enter_blocking_allocation() {
    let harness = promoting("no-blocking-alloc-ok", 4, TIER);
    harness.promoted(0, &[0, 1, 2, 3]);
    let shared = harness.watcher.shared();
    let evicted = shared
        .shrink(Evicting::Bank, Candidates::one(2, vec![2]), None)
        .unwrap()
        .expect("shrink evicts two records");
    let rejoined: Vec<usize> = evicted.records[0].iter().map(|&r| r as usize).collect();
    shared.join(super::Join::Bank { records: vec![rejoined] });
    shared.joins.wait_taken();
    // Promote new experts into the rejoined records (resolved on demand from
    // `Group::bank`) and serve a decode request.
    harness.promoted(0, &[4, 5]);
    harness.request(&[4, 5]).complete().unwrap();
    assert_eq!(shared.life.ended(), None, "serving never tripped the no-blocking-allocation guard");
}
