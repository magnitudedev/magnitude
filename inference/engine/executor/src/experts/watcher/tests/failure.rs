use super::*;

/// A read that fails with an I/O error maps nothing and answers its
/// choices with no record (-1), while the expert that landed is answered
/// with its record; the step fails only the rows that named the failed
/// expert, no request fails whole, and the next step's request is served.
#[test]
fn a_failed_read_answers_no_record_and_fails_only_the_rows_that_named_it() {
    // The file holds 6 experts of each tensor: expert 7's reads of the
    // last tensor lie past its end.
    let harness = harness("failure", 6, 4);
    let failed = harness.request(&[1, 7]);
    let sequence = failed.sequence();
    let (landed, _) = harness.answer(0, sequence, 0);
    assert_eq!(harness.in_tier(0, 1), Some(landed), "the read that landed is answered with its record");
    assert_eq!(harness.answer(0, sequence, 1), (-1, Some(Answer::Answered)), "the read that failed answers no record");
    assert_eq!(harness.in_tier(0, 7), None, "nothing is mapped for it");
    match failed.complete() {
        Err(StepFailure::Read { rows: FailedRows::Only(rows), .. }) => assert_eq!(rows, [1].into()),
        other => panic!("the step failed {other:?}"),
    }
    let next = harness.request(&[2]);
    assert_eq!(harness.verdict(0, next.sequence()), Some(Answer::Answered));
    next.complete().unwrap();
}

/// A panic while serving fails the request it served and every later
/// one, and every step completes with it: the load fails instead of
/// hanging. (A request of more choices than the decode form's, which the
/// route never publishes, makes one.)
#[test]
fn a_watcher_panic_fails_every_request_and_step() {
    let harness = harness("panic", EXPERTS, 4);
    let file = &harness.layer.1;
    let first = harness.step();
    let choices = harness.route_at(&first, 0, &[0, 1, 2, 3]);
    for (choice, &word) in choices.iter().enumerate() {
        file.store(first.sequence(), 0, choice_word(choice as u64), word);
    }
    // More choices than the decode form's.
    file.store(first.sequence(), 0, constants::FILE_COUNT, WORDS.choices as i32 + 1);
    file.request(first.sequence(), 0);
    assert_eq!(until_answered(file, first.sequence(), 0), Answer::Failed);
    assert!(matches!(first.complete(), Err(StepFailure::Watcher(_))));
    let next = harness.request(&[4]);
    assert_eq!(harness.verdict(0, next.sequence()), Some(Answer::Failed));
    assert!(matches!(next.complete(), Err(StepFailure::Watcher(_))));
}

/// A beat of the load's pulse that comes later than any wait's patience
/// (its host starved) lets a live host's spinning wait end before its
/// answer, which its kernel records (SLOT_EXPIRED): the step of a wait that
/// ended so is interrupted (to be re-run), a prompt step's seeds are never
/// mapped (its seed copy may have read what its wait never had answered),
/// and the load goes on: the next step completes, and the next prompt step
/// seeds the same experts afresh. A shorter stall, which no wait outlasts,
/// interrupts nothing, nor does a late beat no wait of the step outlasted.
#[test]
fn a_step_whose_wait_outlasted_a_still_pulse_is_interrupted_and_the_load_goes_on() {
    let harness = prompted("late pulse", EXPERTS, 4);
    let shared = harness.watcher.shared();
    let pulse = shared.steps.pulse();
    // A prompt step seeds experts 3 and 4 at layer 0, its seed commits
    // played, then the host starves its pulse past the device's patience.
    let step = harness.prompt_step();
    for index in 0..harness.layers.len() {
        harness.scratch(&step, index, if index == 0 { &[3, 4] } else { &[] });
        harness.commit_seeds(&step);
    }
    let before = pulse.late_beats();
    pulse.beat_after(Duration::from_millis(4 * constants::PULSE_SPINS as u64 * 70 / 1_000_000));
    assert_eq!(pulse.late_beats(), before + 1, "the late beat was never counted");
    // Its wait at layer 0 outlasted the still pulse, as its kernel records.
    harness.expire(&step, 0);
    let sequence = step.sequence();
    // A lone step: its group's only one.
    let starved = crate::experts::directory::Point::whole(sequence, 0, sequence);
    match step.complete() {
        Err(StepFailure::Interrupted(reason)) => assert!(reason.contains("pulse"), "{reason}"),
        outcome => panic!("a step whose wait expired completed {:?}", outcome.map(|_| ())),
    }
    // The load goes on: a step begun after the beat completes, and the
    // starved step's seeds were never mapped, so a prompt step seeds the
    // same experts afresh.
    assert_eq!(shared.steps.ending(starved), crate::experts::directory::Ending::Tainted, "the starved step is tainted");
    let counted = |steps: u64| format!("steps interrupted by a wait ending on a still pulse and re-run: {steps} (");
    assert!(shared.steps.interruptions().starts_with(&counted(1)), "{}", shared.steps.interruptions());
    assert!(harness.request(&[1]).complete().is_ok(), "the next step completes");
    eventually("residency never passed the starved step", || shared.steps.ending(starved) != crate::experts::directory::Ending::Tainted);
    let mut seeded = harness.seeded(0, &[3, 4]).into_iter().map(|(expert, _)| expert).collect::<Vec<_>>();
    seeded.sort();
    assert_eq!(seeded, [3, 4], "the starved step's seeds were mapped");
    // A stall of half the least patience: no wait could have ended, and
    // nothing is interrupted.
    let step = harness.request(&[2]);
    let before = pulse.late_beats();
    pulse.beat_after(Duration::from_millis(constants::PULSE_SPINS as u64 * 70 / 2_000_000));
    assert_eq!(pulse.late_beats(), before, "a short stall counted as late");
    assert!(step.complete().is_ok(), "a step spanning a short stall is interrupted");
    // A late beat no wait of the step outlasted interrupts nothing either.
    let step = harness.request(&[2]);
    pulse.beat_after(Duration::from_millis(4 * constants::PULSE_SPINS as u64 * 70 / 1_000_000));
    assert!(step.complete().is_ok(), "a step no wait of which expired is interrupted");
    assert!(shared.steps.interruptions().starts_with(&counted(1)), "{}", shared.steps.interruptions());
}

/// A step whose read fails converts nothing for the group's later steps:
/// each of its experts a later step routes is that step's to convert, and
/// one none routes is not seeded.
#[test]
fn a_failed_steps_experts_are_the_next_routing_steps() {
    // The file holds 6 experts of each tensor: expert 7's reads fail.
    let harness = prompted("forfeit", 6, 4);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    harness.publish(&steps[0], 0, &[1, 7]);
    harness.publish(&steps[1], 0, &[0, 1]);
    assert_eq!(harness.every_round(&steps[0], 0), Answer::Failed);
    let lists = harness.rounds(&steps[1], 0);
    let mut converted = lists.concat().into_iter().map(|(_, expert)| expert).collect::<Vec<_>>();
    converted.sort_unstable();
    assert_eq!(converted, [0, 1]);
    let seeds = listed(&harness.load.2);
    assert_eq!(harness.load.2.load(constants::LIST_SEQUENCE as usize), low_word(steps[1].sequence()));
    assert!(!seeds.is_empty() && seeds.iter().all(|&(expert, _)| expert != 7), "expert 7 is not seeded");
    let mut steps = steps.into_iter();
    assert!(matches!(steps.next().unwrap().complete(), Err(StepFailure::Read { .. })));
    steps.next().unwrap().complete().unwrap();
}

/// A failed read stays its step's tier's: the step fails at its round at
/// once, and a later step that routes the expert takes it over and reads it
/// again, failing promptly where that read fails too, never left waiting
/// for an expert nothing reads.
#[test]
fn a_failed_read_taken_over_is_read_again_by_the_later_step() {
    // The file holds 6 experts of each tensor: expert 7's reads fail.
    for (name, conveying) in [
        ("retake", WHOLE),
        ("retake-round", Conveying::of(ROUND_RECORDS)),
    ] {
        let harness = lending(name, 6, 4, 1, STAGING, conveying, TIER, Placer::Tier, false);
        let steps = harness.group(&[StepRoute::Grouped; 2]);
        harness.publish(&steps[0], 0, &[1, 7]);
        harness.publish(&steps[1], 0, &[1, 7]);
        for step in &steps {
            assert_eq!(harness.every_round(step, 0), Answer::Failed, "{name}: step {} fails", step.sequence());
        }
        for step in steps {
            assert!(matches!(step.complete(), Err(StepFailure::Read { .. })), "{name}");
        }
    }
}

/// A step that failed runs on unparked, so the device can pass a layer
/// while the step's request there waits for its round: here the group's
/// last step publishes at layer 2, past layer 1, before the failed step's
/// round at layer 1 arrives. The tier of layer 1 goes with the layer, its
/// request with it, and the watcher serves on, the next layer's rounds
/// first (the real run's panic, `a served request's tier`).
#[test]
fn a_failed_steps_request_the_device_passed_is_not_served() {
    // The file holds 6 experts of each tensor: expert 7's reads fail.
    let harness = layered_over("passed", 6, 12, 3, STAGING, WHOLE);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    harness.publish(&steps[0], 0, &[7]);
    harness.publish(&steps[1], 0, &[0]);
    assert_eq!(harness.every_round(&steps[0], 0), Answer::Failed);
    harness.rounds(&steps[1], 0);
    // The failed step's front at layer 1, which the watcher takes and
    // waits at for its round; then, layer-major, every step's front at
    // layer 1 and at layer 2, the last step's past layer 1.
    harness.publish(&steps[0], 1, &[1]);
    std::thread::sleep(Duration::from_millis(50));
    harness.publish(&steps[1], 1, &[2]);
    harness.publish(&steps[0], 2, &[4]);
    harness.publish(&steps[1], 2, &[3]);
    std::thread::sleep(Duration::from_millis(50));
    harness.layers[1].2.request(steps[0].sequence(), round(0));
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
    // Layer 2: the failed step's rounds, then the last step's.
    harness.every_round(&steps[0], 2);
    assert_eq!(harness.from_conveyor(&harness.rounds(&steps[1], 2).concat()), [3]);
    let mut steps = steps.into_iter();
    assert!(matches!(steps.next().unwrap().complete(), Err(StepFailure::Read { .. })));
    steps.next().unwrap().complete().unwrap();
    assert_eq!(harness.watcher.shared().life.ended(), None, "the watcher serves on");
}

/// No read lands in a record another read takes: a step that fails while
/// its read still runs lets its staging record go, which frees only once
/// that read ends; the next request (here of the one record staging
/// holds) waits for it, then reads its own expert into the record under a
/// new read, and lands it: its bytes are its own, in staging and in its
/// tier record.
#[test]
fn a_failed_steps_record_frees_only_once_its_read_ends() {
    let _alone = alone();
    let harness = layered("stale-read", EXPERTS, 4, 1, 1);
    // Reads queued ahead of the request's, as in
    // `a_read_slower_than_the_patience_is_a_slow_step`.
    let ahead = Ahead::queue("stale-ahead", 16);
    let failed = harness.step();
    assert!(harness.ask_at(&failed, 0, &[3]));
    eventually("the request was never served", || harness.watcher.shared().costs().layers[0].unanticipated == 1);
    // The step is given up before its read lands.
    harness.layer.1.answer(failed.sequence(), 0, true);
    drop(failed);
    let next = harness.request(&[4]);
    let (record, verdict) = harness.answer(0, next.sequence(), 0);
    assert_eq!(verdict, Some(Answer::Answered));
    assert_eq!(harness.in_tier(0, 4), Some(record), "expert 4's bytes in its tier record");
    assert_eq!((0..3).map(|kind| harness.staged(kind, 0)).collect::<Vec<_>>(), [4, 104, 204], "expert 4's bytes");
    next.complete().unwrap();
    ahead.finish();
}

/// A hint's read that fails fails nothing and maps nothing: its step
/// completes, and a later demand for the expert reads it again, failing
/// that step's rows that named it alone.
#[test]
fn a_failed_hint_fails_nothing() {
    // The file holds 6 experts of each tensor: expert 7's reads lie past
    // its end.
    let harness = layered("hint-failure", 6, 12, 2, 4 + 4);
    // The tier's fill lands its records beyond the reserve first, every
    // layer's lowest experts, which no step below names.
    harness.tiering(0, &[0, 1, 2, 3]);
    harness.tiering(1, &[0, 1, 2]);
    harness.demand_step(5);
    let step = harness.step();
    harness.hint(&step, 0, &[7]);
    harness.request_at(&step, 0, &[4]);
    // The hint's read ends (fails) before the step moves on.
    eventually("the hint was never read", || harness.watcher.shared().costs().hints.reads > 0);
    std::thread::sleep(Duration::from_millis(20));
    harness.request_at(&step, 1, &[3]);
    step.complete().expect("a failed hint fails nothing");
    let next = harness.step();
    // Expert 4 is held in the tier at layer 0.
    assert!(!harness.ask_at(&next, 0, &[4]));
    harness.request_at(&next, 1, &[7]);
    assert_eq!(harness.answer(1, next.sequence(), 0), (-1, Some(Answer::Answered)), "no request failed whole");
    assert!(matches!(next.complete(), Err(StepFailure::Read { rows: FailedRows::Only(rows), .. }) if rows == [0].into()));
}
