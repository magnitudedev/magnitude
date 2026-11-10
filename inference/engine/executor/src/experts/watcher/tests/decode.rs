use super::*;

/// A request whose misses all landed after its route resolved them (a
/// read it never saw) is answered with their records and has no lead:
/// each layer's lead is summed over the requests that saw a read. Its kept
/// misses read nothing, so they count apart from the read misses.
#[test]
fn a_request_of_misses_landed_already_has_no_lead() {
    let harness = harness("landed-lead", EXPERTS, 4);
    harness.request(&[3]).complete().unwrap();
    let record = harness.in_tier(0, 3).expect("the first request landed expert 3");
    // The next step's route resolved expert 3 absent before its landing
    // was mapped.
    let step = harness.step();
    harness.route_sourced(&step, 0, &[3], &[-1]);
    harness.publish_request(&step, 0, &[3]);
    eventually("the request was never answered", || harness.verdict(0, step.sequence()) == Some(Answer::Answered));
    assert_eq!(harness.answer(0, step.sequence(), 0).0, record);
    step.complete().unwrap();
    let layer = harness.watcher.shared().costs().layers[0];
    assert_eq!((layer.led, layer.lead), (1, 0), "only the read on demand has a lead");
    assert_eq!(layer.misses, 1, "the kept miss reads nothing");
}

/// A decode request's absent experts land into tier records, each read
/// from the file, converted into its record and mapped there before its
/// request's slot is answered: once it holds its step, each choice's record
/// is the one the tier map names, the expert's bytes whole. A repeated
/// expert answers both its choices with one record. A later step's route
/// finds a landed expert in the tier map and serves it in place, requesting
/// nothing for it; the misses served there count as the tier's.
#[test]
fn a_decode_request_lands_its_absent_experts_into_the_tier_and_answers_each_choice() {
    let harness = harness("decode", EXPERTS, 4);
    let step = harness.step();
    assert!(harness.ask_at(&step, 0, &[3, 5, 3]));
    let landed = |choice: usize, expert: usize| {
        eventually("a choice was never answered", || harness.verdict(0, step.sequence()) == Some(Answer::Answered));
        let (record, _) = harness.answer(0, step.sequence(), choice);
        assert_eq!(harness.in_tier(0, expert), Some(record), "the tier map names the record before its answer");
        record
    };
    let (first, other, repeat) = (landed(0, 3), landed(1, 5), landed(2, 3));
    assert_eq!(first, repeat, "a repeated expert answers both choices with one record");
    assert_ne!(first, other);
    assert_eq!(harness.verdict(0, step.sequence()), Some(Answer::Answered));
    let first_step = step.sequence();
    step.complete().unwrap();
    let costs = harness.watcher.shared().costs();
    assert_eq!(costs.completed.map(|(sequence, step)| (sequence, step.misses)), Some((first_step, 2)));
    let next = harness.step();
    harness.request_at(&next, 0, &[3, 6]);
    let file = &harness.layer.1;
    assert_eq!(file.load(next.sequence(), 0, choice_word(0)), -1 - 3, "the tier serves expert 3");
    assert_eq!(harness.verdict(0, next.sequence()), Some(Answer::Answered));
    next.complete().unwrap();
    let costs = harness.watcher.shared().costs();
    let layer = costs.layers[0];
    assert_eq!((layer.misses, layer.hosted), (4, 1));
    assert_eq!(layer.missed_bytes, 4 * 3 * RECORD);
    assert!(layer.read > Duration::ZERO, "each read is counted as it lands");
    let (_, step) = costs.completed.unwrap();
    assert_eq!((step.misses, step.missed_bytes), (2, 2 * 3 * RECORD));
    assert_eq!((step.layers[0].misses, step.layers[0].hosted), (1, 1));
}

/// On unified memory's one store, a file miss is read into staging and its
/// entry appended to the layer's landing list, which the device's landing
/// launch converts into a store record, answered as itself (a bank record,
/// not -2 - t); once the device is past the launch, having stored the
/// entry's DONE, the host maps the record for every step not yet begun,
/// and the next step's route finds it held in the store.
#[test]
fn a_device_landed_miss_lands_into_the_store_and_is_mapped_past_its_launch() {
    let harness = lending("one-store", EXPERTS, u64::from(TIER), 1, STAGING, DECODE, TIER, Placer::Bank, false);
    let step = harness.request(&[1]);
    eventually("the choice was never answered", || harness.verdict(0, step.sequence()) == Some(Answer::Answered));
    let record = harness.layers[0].1.load(step.sequence(), 0, record_word(0));
    assert!(record >= 0, "answered as the store record {record}");
    let list = harness.landing[0].as_ref().expect("the one store's landing list");
    assert_eq!(list.listed(step.sequence()).iter().map(|&(_, to)| to).collect::<Vec<_>>(), [record]);
    assert_eq!([0, 1, 2].map(|kind| harness.tiered(kind, record)), [1, 101, 201]);
    assert_eq!(harness.layers[0].0.resolve(1, step.sequence()), None, "the step resolved it absent");
    step.complete().unwrap();
    let map = &harness.layers[0].0;
    eventually("the landing never mapped its record", || map.held(1) == Some(record as usize));
    let next = harness.step();
    assert_eq!(harness.route_at(&next, 0, &[1]), vec![-1 - 1], "held in the store");
    next.complete().unwrap();
}

/// Where the bank places, a parked step's landing is mapped, once the
/// device is past its launch, from the first step that had not resolved its
/// layer when the read took its record: a step begun behind it (lookahead)
/// that has yet to route the layer hits it, by the map the host decides its
/// hits by as the device's route does, so the two agree; the parked step
/// itself, which resolved it absent, does not.
#[test]
fn a_miss_landed_for_a_parked_step_is_a_hit_for_the_step_begun_behind_it() {
    let harness = lending("lookahead-hit", EXPERTS, u64::from(TIER), 1, STAGING, DECODE, TIER, Placer::Bank, false);
    let parked = harness.step();
    let behind = harness.step();
    harness.request_at(&parked, 0, &[1]);
    eventually("the choice was never answered", || harness.verdict(0, parked.sequence()) == Some(Answer::Answered));
    let record = harness.layers[0].1.load(parked.sequence(), 0, record_word(0));
    assert!(record >= 0, "answered as the store record {record}");
    // Mapped once the device is past the parked step's landing launch.
    let sequence = parked.sequence();
    parked.complete().unwrap();
    let map = &harness.layers[0].0;
    eventually("the landing never mapped its record", || map.resolve(1, behind.sequence()) == Some(record as usize));
    assert_eq!(map.resolve(1, sequence), None, "the parked step resolved it absent");
    assert_eq!(harness.route_at(&behind, 0, &[1]), vec![-1 - 1], "its route finds it held");
    behind.complete().unwrap();
}

/// Where the bank places, each miss's read appends its entry to the layer's
/// landing list as it lands, in landing order, before the slot is answered
/// at its last absent expert: every entry converts a staging record into the
/// record its choices are answered with, and the device's DONE of each
/// lets the host map it once past the launch. A repeated expert lands once.
#[test]
fn the_device_lands_each_miss_as_its_read_lands() {
    let harness = lending("device-landing", EXPERTS, u64::from(TIER), 2, STAGING, DECODE, TIER, Placer::Bank, false);
    let step = harness.step();
    harness.request_at(&step, 0, &[1, 2, 1]);
    let records = (0..3).map(|choice| harness.layers[0].1.load(step.sequence(), 0, record_word(choice))).collect::<Vec<_>>();
    assert_eq!(records[0], records[2], "a repeated expert answers both choices with one record");
    let list = harness.landing[0].as_ref().expect("the one store's landing list");
    let listed = list.listed(step.sequence());
    let mut landed = listed.iter().map(|&(_, to)| to).collect::<Vec<_>>();
    landed.sort_unstable();
    let mut answered = vec![records[0], records[1]];
    answered.sort_unstable();
    assert_eq!(landed, answered, "an entry for each distinct miss, into its answered record");
    for (entry, &(from, _)) in listed.iter().enumerate() {
        assert!((0..STAGING as i32).contains(&from), "entry {entry} converts staging record {from}");
        assert!(list.done(step.sequence(), entry), "entry {entry} placed");
    }
    for (expert, record) in [(1, records[0]), (2, records[1])] {
        assert_eq!([0, 1, 2].map(|kind| harness.tiered(kind, record)), [expert, 100 + expert, 200 + expert]);
    }
    let (map, sequence) = (&harness.layers[0].0, step.sequence());
    harness.route_at(&step, 1, &[]);
    step.complete().unwrap();
    eventually("the landings never mapped their records", || {
        map.resolve(1, sequence + 1) == Some(records[0] as usize) && map.resolve(2, sequence + 1) == Some(records[1] as usize)
    });
    assert_eq!(harness.watcher.shared().device_landed(), 2);
}

/// A request of more misses than staging holds lands them all on the
/// device: each entry's DONE frees its staging record for the next miss's
/// read while the launch converts, so the slot is answered whole without
/// waiting for the launch to end.
#[test]
fn a_request_past_its_staging_lands_whole_as_done_frees_staging() {
    let harness = lending("short-staging", EXPERTS, u64::from(TIER), 2, 2, DECODE, TIER, Placer::Bank, false);
    let step = harness.step();
    harness.request_at(&step, 0, &[1, 2, 3, 4]);
    let list = harness.landing[0].as_ref().expect("the one store's landing list");
    let listed = list.listed(step.sequence());
    assert_eq!(listed.len(), 4, "every miss lands on the device");
    for (choice, expert) in (0..4).zip(1..) {
        let record = harness.layers[0].1.load(step.sequence(), 0, record_word(choice));
        assert!(record >= 0, "choice {choice} landed");
        assert!(listed.iter().any(|&(_, to)| to == record), "choice {choice}'s record is an entry's");
        assert_eq!([0, 1, 2].map(|kind| harness.tiered(kind, record)), [expert, 100 + expert, 200 + expert]);
    }
    harness.route_at(&step, 1, &[]);
    step.complete().unwrap();
}

/// A read that fails appends nothing: its choices are answered -1 and the
/// launch converts only the entries of the reads that landed.
#[test]
fn a_failed_read_appends_nothing_and_answers_no_record() {
    // The file holds 6 experts of each tensor: expert 7's reads of the
    // last tensor lie past its end.
    let harness = lending("device-failure", 6, u64::from(TIER), 1, STAGING, DECODE, TIER, Placer::Bank, false);
    let step = harness.request(&[1, 7]);
    let sequence = step.sequence();
    assert_eq!(harness.verdict(0, sequence), Some(Answer::Answered));
    let landed = harness.layers[0].1.load(sequence, 0, record_word(0));
    assert_eq!(harness.layers[0].1.load(sequence, 0, record_word(1)), -1, "the read that failed answers no record");
    let list = harness.landing[0].as_ref().expect("the one store's landing list");
    assert_eq!(list.listed(sequence).iter().map(|&(_, to)| to).collect::<Vec<_>>(), [landed], "only the landed read's entry");
    assert!(matches!(step.complete(), Err(StepFailure::Read { .. })));
    assert_eq!(harness.layers[0].0.held(7), None, "nothing is mapped for it");
}

/// An expert the tier has unmapped for its reserve lives only in the file:
/// a request for it reads it on demand, whatever staging record last held
/// it, its bytes the file's.
#[test]
fn a_request_reads_an_expert_its_tier_unmapped() {
    // The store starts empty: each of the first two requests' experts
    // misses and is read on demand, and expert 0, once unmapped, is read on
    // demand again. Reads ahead stand behind the line, taken only once a
    // request hastens them, so every miss is read on demand, whatever the
    // threads' timing.
    let _alone = alone();
    let ahead = Ahead::queue("kept-ahead");
    // The tier's records are short of its reserve: it keeps nothing it has
    // landed.
    let harness = lending("kept", EXPERTS, 4, 1, 8, DECODE, 7, Placer::Tier, false);
    for experts in [[0, 1, 2, 3], [4, 5, 6, 7]] {
        harness.request(&experts).complete().unwrap();
    }
    eventually("the reserve never unmapped expert 0", || harness.in_tier(0, 0).is_none());
    // Its record graced, a request takes it back unread: the file holds it
    // alone once the grace retired.
    eventually("expert 0's grace never retired", || residency::lock(&harness.watcher.state).core.directory.host(0).graced() == 0);
    let step = harness.request(&[0]);
    let (record, verdict) = harness.answer(0, step.sequence(), 0);
    assert_eq!(verdict, Some(Answer::Answered));
    assert_eq!([0, 1, 2].map(|kind| harness.tiered(kind, record)), [0, 100, 200]);
    step.complete().unwrap();
    let layer = harness.watcher.shared().costs().layers[0];
    assert_eq!((layer.misses, layer.unanticipated), (9, 9));
    ahead.finish();
}

/// A tier record unmapped for the reserve is graced through its expert's
/// layer of the latest step begun once its unmapping was visible, which may
/// have resolved it before. A step parked at a layer's request makes that
/// exact: steps run in order, so only what its own route resolved there is
/// still read. Here a step begun before the unmapping, whose route
/// resolved none of them, lands its miss into one at once, while the step
/// before is still running; no row fails.
#[test]
fn a_parked_step_lands_into_records_its_route_did_not_resolve() {
    let harness = hosted("grace", 4, STAGING);
    let map = &harness.host.maps[0];
    let (first, graced) = (harness.step(), harness.step());
    harness.request_at(&first, 0, &[0, 1, 2, 3]);
    eventually("the reserve never unmapped the records", || (0..4).all(|expert| map.load(expert) == -1));
    harness.request_at(&graced, 0, &[4]);
    let (record, verdict) = harness.answer(0, graced.sequence(), 0);
    assert_eq!(verdict, Some(Answer::Answered), "answered while the step before runs");
    assert!(record >= 0, "landed into a record its route did not resolve");
    assert_eq!([0, 1, 2].map(|kind| harness.tiered(kind, record)), [4, 104, 204]);
    first.complete().unwrap();
    graced.complete().unwrap();
}

/// A routing record whose count is no count of its choices is the device's
/// broken invariant at a park: what it names shields records from every
/// claim, so a shortened list would let a landing take a record the step
/// still reads.
#[test]
fn a_parked_routing_record_of_a_broken_count_panics() {
    let harness = hosted("parked-count", 4, STAGING);
    let step = harness.step();
    let sequence = step.sequence();
    let count = (sequence % WORDS.flight) as usize * WORDS.routing_words() as usize + constants::RING_COUNT as usize;
    harness.route_sourced(&step, 0, &[], &[]);
    for broken in [-1, WORDS.choices as i32 + 1] {
        harness.layers[0].4.store(count, broken);
        let held = residency::lock(&harness.watcher.state);
        let parked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| super::super::decode::parked_reads(&held.core, harness.watcher.shared(), 0, sequence)));
        drop(held);
        assert!(parked.is_err(), "a count of {broken}");
    }
    harness.layers[0].4.store(count, 0);
    step.complete().unwrap();
}

/// A park read after its step ended reads no routing record: the step's
/// slot is free, and a later step of it may be rewriting the record, as the
/// device writes its choices before its sequence word. Here the step of the
/// slot after it has written one choice's source of its own when residency
/// reads the stale park: the record read is the parked step's whole, or
/// none, never a mix of both steps'.
#[test]
fn a_park_read_after_its_step_ended_reads_no_later_steps_record() {
    let harness = hosted("lapped", 4, STAGING);
    let step = harness.step();
    let sequence = step.sequence();
    harness.route_sourced(&step, 0, &[0, 1], &[-2, -3]);
    drop(step);
    for _ in 1..WORDS.flight {
        drop(harness.step());
    }
    let lapping = harness.step();
    assert_eq!(lapping.sequence(), sequence + WORDS.flight, "the step of the slot after");
    let base = (sequence % WORDS.flight) as usize * WORDS.routing_words() as usize;
    harness.layers[0].4.store(base + WORDS.routing_source(0) as usize, -2 - 3);
    let held = residency::lock(&harness.watcher.state);
    let parked = super::super::decode::parked_reads(&held.core, harness.watcher.shared(), 0, sequence);
    drop(held);
    assert!(parked.as_deref().is_none_or(|read| read == [(0, 0), (0, 1)]), "a torn record: {parked:?}");
    drop(lapping);
}

/// Where the host places (the one store), a parked step's route resolves
/// held experts to store records themselves (sources >= 0): the records the
/// reserve unmapped after that resolution are still read by the step, so
/// its miss lands into the one it did not resolve.
#[test]
fn a_parked_step_never_lands_over_a_store_record_its_route_resolved() {
    let harness = lending("store-grace", EXPERTS, 4, 1, STAGING, DECODE, 4, Placer::Bank, false);
    let map = &harness.layers[0].0;
    let (first, graced) = (harness.step(), harness.step());
    harness.request_at(&first, 0, &[0, 1, 2, 3]);
    let file = &harness.layers[0].1;
    let answer = |step: &ExpertStep, choice: u64| file.load(step.sequence(), 0, record_word(choice));
    let records = (0..4).map(|choice| answer(&first, choice)).collect::<Vec<_>>();
    assert!(records.iter().all(|&record| record >= 0), "landed into store records {records:?}");
    eventually("the reserve never unmapped the records", || (0..4).all(|expert| map.held(expert).is_none()));
    // The step begun before the unmapping resolved experts 1 to 3 as held.
    let choices = [-1 - 1, -1 - 2, -1 - 3, 4];
    harness.route_sourced(&graced, 0, &choices, &[records[1], records[2], records[3], -1]);
    harness.publish_request(&graced, 0, &choices);
    eventually("the miss was never answered", || harness.verdict(0, graced.sequence()) == Some(Answer::Answered));
    assert_eq!(answer(&graced, 3), records[0], "the record its route did not resolve");
    first.complete().unwrap();
    graced.complete().unwrap();
}

/// Where the bank places, a parked step's device landings are mapped only
/// once the device is past their launch, at the step's next park, from the
/// first step that had not resolved their layer; the refill unmaps them
/// into the reserve in that same advance, and the park restamps them from
/// that step, so they are ready however many steps have begun behind it,
/// and a hint, which takes only ready records beyond one request's, is
/// read. A step's misses take the ready records; a second step is begun
/// while they land.
#[test]
fn a_park_readies_the_reserve_for_the_hints_behind_it() {
    // Eight store records, all the reserve (one per staging record), for
    // two layers' sixteen experts.
    let harness = lending("park-readies", EXPERTS, 8, 2, 8, DECODE, 8, Placer::Bank, false);
    let map = &harness.layers[0].0;
    let (parked, behind) = (harness.step(), harness.step());
    harness.request_at(&parked, 0, &[4, 5, 6, 7]);
    assert!((4..8).all(|expert| map.resolve(expert, behind.sequence()).is_none()), "mapped before the device is past the launch");
    harness.request_at(&parked, 1, &[7]);
    assert!((4..8).all(|expert| map.held(expert).is_none()), "the refill unmapped the landed experts");
    assert!((4..8).all(|expert| map.resolve(expert, behind.sequence()).is_none()), "restamped at the park");
    harness.hint(&behind, 0, &[5]);
    eventually("the hint was never read", || harness.watcher.shared().costs().hints.reads == 1);
    parked.complete().unwrap();
    for layer in 0..2 {
        harness.route_at(&behind, layer, &[]);
    }
    behind.complete().unwrap();
}

/// A record answered for a parked request is read by the request's
/// entries after the park: its other miss never takes it, under either
/// placer, though the reserve unmaps it once it lands. With one record,
/// the second miss waits.
#[test]
fn a_parked_requests_answered_record_is_never_taken_by_its_other_miss() {
    for placer in [Placer::Tier, Placer::Bank] {
        let harness = lending("answered", EXPERTS, 1, 1, STAGING, DECODE, 1, placer, false);
        let file = &harness.layers[0].1;
        let step = harness.step();
        // A choice's record word, -1 until its landing writes it.
        let record = |choice: u64| file.load(step.sequence(), 0, record_word(choice));
        // As serving sets them, before the request: no zero word of a slot
        // never served reads as a record.
        for choice in 0..2 {
            file.store(step.sequence(), 0, record_word(choice), -1);
        }
        assert!(harness.ask_at(&step, 0, &[5, 6]));
        eventually("neither miss was answered", || (0..2).any(|choice| record(choice) != -1));
        std::thread::sleep(Duration::from_millis(50));
        let answered = (0..2).map(record).filter(|&record| record != -1).collect::<Vec<_>>();
        assert_eq!(answered.len(), 1, "{placer:?}: both misses answered with {answered:?}, one record between them");
        assert_eq!(harness.verdict(0, step.sequence()), None, "{placer:?}: the slot waits for its other miss");
        step.complete().unwrap();
    }
}
