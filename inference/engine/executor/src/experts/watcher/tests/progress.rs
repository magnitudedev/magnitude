use super::*;

/// Completing a step takes nothing serving holds (it returns while a later
/// step runs); a step dropped without completing ends too, so the watcher
/// idles once no step runs.
#[test]
fn steps_end_without_waiting_for_serving() {
    let harness = harness("ending", EXPERTS, 4);
    let first = harness.request(&[0]);
    let second = harness.step();
    first.complete().unwrap();
    drop(second);
    assert_eq!(harness.watcher.shared().steps.in_flight(), 0);
}

/// A layer keeps a routing record for each step in flight, the one of a
/// step's sequence modulo them, and no group begins beyond them: a step's
/// record holds until it ends, whatever steps began after it. Here two
/// groups of the harness's two places are in flight (four records); a
/// fifth step waits for one of them to end.
#[test]
fn no_step_begins_beyond_the_routing_records_in_flight() {
    let harness = harness("flight", EXPERTS, 4);
    let shared = harness.watcher.shared();
    assert_eq!(WORDS.flight, 4);
    let first = begin(shared, &[StepRoute::Decode; 2], false).unwrap();
    let second = begin(shared, &[StepRoute::Decode; 2], false).unwrap();
    assert!(begin(shared, &[StepRoute::Decode], false).is_err(), "a fifth step in flight");
    // Each step writes its own record: none of the four shares one.
    let records = first.iter().chain(&second).map(|step| step.sequence() % WORDS.flight);
    assert_eq!(records.collect::<std::collections::BTreeSet<_>>().len(), 4);
    drop(first);
    let third = begin(shared, &[StepRoute::Decode; 2], false).unwrap();
    drop((second, third));
}

/// The frontier and the first step not yet begun, read together, are at
/// most the flight apart while steps begin and end around them: a release's
/// wait, sampled from them, never exceeds the flight (the reserve's sizing
/// asserts it).
#[test]
fn the_frontier_and_the_next_step_are_read_together() {
    let harness = harness("frontier-next", EXPERTS, 4);
    let shared = harness.watcher.shared().clone();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let churn = {
        let (shared, stop) = (shared.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                if let Ok(steps) = begin(&shared, &[StepRoute::Decode; 2], false) {
                    drop(steps);
                }
            }
        })
    };
    for _ in 0..20_000 {
        shared.steps.unbegun_beside_frontier(|frontier, at| {
            let next = at.next();
            assert!(next - frontier <= WORDS.flight, "{frontier}..{next} beyond the {} in flight", WORDS.flight);
        });
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    churn.join().unwrap();
}

/// The last flight's fills are published held through the gap after it:
/// the watcher retires the completed steps where it sees no step running,
/// not at the next flight's first pass.
#[test]
fn a_gap_publishes_what_the_last_flight_filled() {
    let harness = prompted("gap-held", EXPERTS, 4);
    let step = harness.prompt_step();
    harness.scratch(&step, 0, &[3, 5]);
    assert_eq!(harness.commit_seeds(&step).len(), 2);
    step.complete().unwrap();
    let shared = harness.watcher.shared();
    eventually("the gap never published the fills", || shared.holdings().0[0] == 2);
}

/// A landing that finds no tier record free waits, a shortage of records
/// being backpressure, never a failure: four landings take every record of
/// a tier of four, and the next step's miss lands once they have landed
/// and the reserve frees one its route did not resolve.
#[test]
fn a_landing_with_no_free_record_waits_and_never_fails() {
    let harness = hosted("waits", 4, STAGING);
    let step = harness.step();
    // Every record takes a landing at once: the reserve holds none ready
    // until they land and the reserve unmaps them.
    harness.request_at(&step, 0, &[0, 1, 2, 3]);
    let next = harness.step();
    harness.request_at(&next, 0, &[5]);
    let (record, verdict) = harness.answer(0, next.sequence(), 0);
    assert_eq!(verdict, Some(Answer::Answered), "no request failed");
    assert!(record >= 0, "the waiting landing landed");
    step.complete().unwrap();
    next.complete().unwrap();
}

/// A burst of file misses across the layers of one step lands every one:
/// the records the reserve unmaps from a layer's experts are graced
/// through that layer of the step alone, so a later layer's landings take
/// them as soon as its request proves the device past the earlier layers.
/// Four layers each miss four experts into a tier of four records, sixteen
/// landings in one step, far past one request's choices and the hints'
/// records.
#[test]
fn a_burst_of_file_misses_across_one_steps_layers_lands_every_one() {
    const LAYERS: usize = 4;
    let harness = hosted_layers("burst", 2, LAYERS, STAGING);
    let step = harness.step();
    for layer in 0..LAYERS {
        harness.request_at(&step, layer, &[0, 1, 2, 3]);
        for choice in 0..4 {
            let (record, verdict) = harness.answer(layer, step.sequence(), choice);
            assert_eq!(verdict, Some(Answer::Answered), "layer {layer} choice {choice} answered");
            assert!(record >= 0, "layer {layer} choice {choice} landed into a record");
        }
    }
    step.complete().unwrap();
}

/// The request service's rest as its thread waits it, a wait on the load's
/// events no event ends, at the thread's default timer slack and once its
/// rests are precise (`rest_precisely`): the median of 2,000 rests of
/// `IDLE` each, on the host's clock. The default slack (50 µs) lengthens a
/// rest beyond it and the precise rest is shorter.
#[cfg(target_os = "linux")]
#[test]
fn the_request_services_rests_last_what_they_ask() {
    let rests = || {
        let events = crate::experts::events::Events::default();
        let mut slept = (0..2000)
            .map(|_| {
                let started = Instant::now();
                events.wait_at_most(events.seen(), super::requests::IDLE);
                started.elapsed()
            })
            .collect::<Vec<_>>();
        slept.sort();
        slept[slept.len() / 2]
    };
    let (default, precise) = std::thread::spawn(move || {
        let default = rests();
        super::requests::rest_precisely().unwrap();
        (default, rests())
    })
    .join()
    .unwrap();
    assert!(precise < default, "a precise rest is shorter than one at the default slack");
}

/// A bank at its minimum (`ExpertBankPlan::least_records`) makes progress
/// under either placer, every shrink's end: decode steps each run with a
/// lookahead successor begun behind it, so the reserve's unmappings are
/// graced past every park, each layer missing a whole request of experts no
/// record holds; and prompt groups of both places, each begun behind a
/// decode step and with one begun behind it. Every request is answered and no
/// step fails. The harness's `R` is its request's choices and its `K` one
/// (`Extents::selected`), so its staging is `R + 2K` records. Across a link
/// the bank holds a layer's experts, a prompt group's scratch records
/// (promoting into them), and the tier its own floor, the landing reserve
/// beside the conveyor's lent records (`host_records` at no host slot); on
/// unified memory the one store holds the more of `R` and a layer's experts
/// beside the landing reserve and the lent records (`prompt_records`).
#[test]
fn a_bank_at_its_minimum_makes_progress() {
    const LAYERS: usize = 4;
    let request = WORDS.choices as u32;
    let staging = request + 2;
    for placer in [Placer::Tier, Placer::Bank] {
        // On unified memory the one store is the harness's tier records; its
        // device-placed records go unused, one as a reservation commits.
        let (records, tier) = match placer {
            Placer::Tier => (EXPERTS as u64, staging),
            Placer::Bank => (1, request.max(EXPERTS as u32) + staging),
        };
        assert!(tier < (EXPERTS * LAYERS) as u32, "the minimum lacks experts, so misses land");
        let conveying = Conveying { conveyed: 4, lent: 4, bytes: RECORD };
        let harness = lending("minimum", EXPERTS, records, LAYERS, staging, conveying, tier, placer, placer == Placer::Tier);
        // Each round's request misses half of each layer's experts, the
        // other half the round before.
        let decode = |step: &ExpertStep, round: usize| {
            for layer in 0..LAYERS {
                let experts = (0..request as usize).map(|choice| (round * request as usize + choice) % EXPERTS).collect::<Vec<_>>();
                harness.request_at(step, layer, &experts);
                assert_ne!(harness.verdict(layer, step.sequence()), Some(Answer::Failed), "{placer:?} round {round} layer {layer}");
            }
        };
        let prompt = |group: &[ExpertStep]| {
            for layer in 0..LAYERS {
                for (place, step) in group.iter().enumerate() {
                    harness.publish(step, layer, &[place, place + 3, 7]);
                }
                for step in group {
                    harness.park(step, layer);
                }
            }
        };
        let mut ahead = harness.step();
        for round in 0..12 {
            let step = std::mem::replace(&mut ahead, harness.step());
            decode(&step, round);
            step.complete().unwrap();
            // Every other round, a prompt group begun behind the decode step
            // ahead, and once that ends, a decode step begun behind the
            // group: two groups in flight at every point (`FlightDepth::groups`).
            if round % 2 == 1 {
                let group = harness.group(&[StepRoute::Prompt; 2]);
                decode(&ahead, round);
                ahead.complete().unwrap();
                ahead = harness.step();
                prompt(&group);
                for step in group {
                    step.complete().unwrap();
                }
            }
        }
        decode(&ahead, 12);
        ahead.complete().unwrap();
    }
}
