use super::*;
use crate::assessment::Notice;

/// A scratch request lists none of the bank's hits, which its back reads in
/// place, and converts the rest a round at a time from the conveyor records
/// they were read into; every unused round's list is empty for the step.
#[test]
fn a_scratch_request_reads_hits_in_place_and_converts_the_rest_round_by_round() {
    let harness = prompted("scratch", EXPERTS, 4);
    harness.seeded(0, &[2]);
    let step = harness.prompt_step();
    let listed = harness.scratch(&step, 0, &[0, 1, 2, 4, 5, 6, 7]);
    // Six cold experts: rounds 0 to 2, two each; round 3 neither parks nor
    // converts.
    assert_eq!(listed.iter().map(Vec::len).collect::<Vec<_>>(), [2, 2, 2]);
    assert_eq!(harness.from_conveyor(&listed.concat()), [0, 1, 4, 5, 6, 7]);
    step.complete().unwrap();
}

/// Each layer's rounds append to landing lists of their own, each entry
/// as its expert is held whole: a step's rounds at the next layer, which
/// it reaches before the host appends for it there, find none of this
/// layer's entries under the step's head, and every entry names the
/// conveyor record holding its expert whole.
#[test]
fn each_layers_rounds_list_their_own_entries() {
    let harness = lending("round-lists", EXPERTS, 4, 2, STAGING, Conveying::of(EXPERTS as u64), TIER, Placer::Tier, false);
    let step = harness.prompt_step();
    let first = harness.scratch(&step, 0, &[0, 1, 2]);
    assert_eq!(harness.from_conveyor(&first.concat()), [0, 1, 2]);
    assert_eq!(first.iter().map(Vec::len).collect::<Vec<_>>(), [2, 1], "rounds of two records");
    assert_eq!(harness.round_listed(&step, 1, 0), [], "layer 1's list holds none of layer 0's entries");
    let second = harness.scratch(&step, 1, &[3, 4]);
    assert_eq!(harness.from_conveyor(&second.concat()), [3, 4]);
    step.complete().unwrap();
}

/// A request's cold experts are read into conveyor records as it is
/// published: every round lists conveyor records holding its experts
/// whole, staging is never written, and a conveyor of a round's records
/// cycles as rounds convert until every expert is.
#[test]
fn a_conveyors_records_cycle_as_rounds_convert() {
    for (name, conveyed) in [("whole", EXPERTS as u64), ("cycle", ROUND_RECORDS)] {
        let harness = lending(name, EXPERTS, 4, 1, STAGING, Conveying::of(conveyed), TIER, Placer::Tier, false);
        let step = harness.prompt_step();
        let cold = [0, 1, 2, 3, 4, 5, 6];
        harness.publish(&step, 0, &cold);
        let scratch = &harness.layer.2;
        let (mut converted, mut rounds, mut index) = (Vec::new(), 1, 0);
        while index < rounds {
            scratch.request(step.sequence(), round(index));
            assert_eq!(until_answered(scratch, step.sequence(), round(index)), Answer::Answered);
            if index == 0 {
                rounds = (scratch.load(step.sequence(), round(0), constants::SCRATCH_STAGED) as u64)
                    .div_ceil(ROUND_RECORDS);
            }
            // Until the device arrives at the next round, this round's
            // records hold what its landing launch converts.
            converted.extend(harness.from_conveyor(&harness.round_listed(&step, 0, index)));
            index += 1;
        }
        assert_eq!(rounds, (cold.len() as u64).div_ceil(ROUND_RECORDS));
        converted.sort_unstable();
        assert_eq!(converted, cold.map(|expert| expert as i32));
        for record in 0..STAGING as i32 {
            assert_eq!((0..3).map(|kind| harness.staged(kind, record)).collect::<Vec<_>>(), [0, 0, 0], "staging untouched");
        }
        step.complete().unwrap();
        let layer = harness.watcher.shared().costs().group.unwrap().layers[0].clone();
        assert_eq!(layer.conveyed, cold.len() as u64);
    }
}

/// A scratch round's answer is host work into host memory alone: a cold
/// expert the host tier keeps whole is answered by naming its tier record
/// in the tier list, which the device's tier copy reads after round 0, so
/// round 0 is answered while the transfer queue can issue nothing (the
/// scratch slots' host access held, which any transfer into them waits
/// on), and a decode request is served meanwhile. No transfer is posted.
#[test]
fn a_rounds_answer_needs_no_transfer() {
    let harness = lending("answer-alone", EXPERTS, 2, 1, STAGING, WHOLE, TIER - 1, Placer::Tier, false);
    harness.request(&[5]).complete().unwrap();
    harness.tiering(0, &[5]);
    let record = harness.host.maps[0].load(5);
    assert!(record >= 0, "the tier keeps 5");
    let (held, holding) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    let writer = {
        let mut slots = harness.scratch[0].clone();
        // Its host access held from the upload's start until released: what
        // keeps a transfer into the slots from being issued, as a device's
        // queue full behind a parked step keeps one waiting in the driver.
        std::thread::spawn(move || {
            slots
                .write_chunks(|_| {
                    held.send(()).unwrap();
                    released.recv().unwrap();
                    Ok(())
                })
                .unwrap()
        })
    };
    holding.recv().unwrap();
    let decode = harness.step();
    let prompt = harness.prompt_step();
    let scratch = &harness.layer.2;
    harness.publish(&prompt, 0, &[5]);
    scratch.request(prompt.sequence(), round(0));
    eventually("round 0 answered while no transfer can be issued", || scratch.answer_of(prompt.sequence(), round(0)) == Some(Answer::Answered));
    let absent = (0..EXPERTS).find(|&expert| expert != 5 && harness.in_tier(0, expert).is_none()).unwrap();
    assert!(harness.ask_at(&decode, 0, &[absent]));
    eventually("the decode request served meanwhile", || harness.verdict(0, decode.sequence()).is_some());
    let tiered = listed(&harness.tiered[0]);
    release.send(()).unwrap();
    writer.join().unwrap();
    assert_eq!(harness.verdict(0, decode.sequence()), Some(Answer::Answered));
    assert_eq!(harness.tiered[0].load(constants::LIST_SEQUENCE as usize), low_word(prompt.sequence()));
    assert_eq!(tiered, [(record, 5)], "the tier list names 5 at its tier record");
    harness.park(&prompt, 0);
    decode.complete().unwrap();
    prompt.complete().unwrap();
}

/// Beside a host tier, a prompt group's cold expert the tier keeps whole
/// is copied from its tier record into its scratch slot by the device's
/// tier copy (which the harness plays at round 0), in the resident form:
/// it is never read nor listed for a convert, while the rest are read and
/// converted as ever. A request whose every cold expert the tier keeps
/// converts nothing. Both count as the group's experts from the tier.
#[test]
fn a_prompt_group_copies_tier_experts_into_the_scratch_tier() {
    let harness = lending("pull", EXPERTS, 2, 1, STAGING, WHOLE, TIER, Placer::Tier, false);
    // Expert 5 misses a decode step and lands in the tier, which holds it
    // alone (the host never places into the bank).
    harness.request(&[5]).complete().unwrap();
    harness.tiering(0, &[5]);
    let map = &harness.host.maps[0];
    let slot = |kind: usize, expert: u64| {
        let mut bytes = [0u8; RECORD as usize];
        unsafe { harness.scratch[kind].read_shared(expert * RECORD, &mut bytes) }.unwrap();
        bytes
    };
    let step = harness.prompt_step();
    assert_eq!(harness.layer.0.held(5), None, "the bank lacks 5");
    let listed = harness.scratch(&step, 0, &[3, 4, 5]);
    assert_eq!(listed.iter().map(Vec::len).collect::<Vec<_>>(), [2], "one round, of the experts read");
    assert_eq!(harness.from_conveyor(&listed.concat()), [3, 4]);
    assert_eq!(harness.holding(5), None, "the tier's expert is never read");
    for kind in 0..3 {
        assert_eq!(slot(kind, 5), [(100 * kind + 5) as u8; RECORD as usize], "expert 5 copied whole");
    }
    step.complete().unwrap();
    let layer = harness.watcher.shared().costs().group.unwrap().layers[0].clone();
    assert_eq!((layer.conveyed, layer.pulled, layer.tier_bytes), (2, 1, 3 * RECORD));
    // A request of the tier's expert alone converts nothing.
    let step = harness.prompt_step();
    eventually("expert 5 left the tier map", || map.load(5) >= 0);
    unsafe { harness.scratch[1].write_shared(5 * RECORD, &[0; RECORD as usize]) }.unwrap();
    let listed = harness.scratch(&step, 0, &[5]);
    assert!(listed.is_empty(), "no round converts: {listed:?}");
    assert_eq!(slot(1, 5), [105; RECORD as usize], "copied again for this group");
    step.complete().unwrap();
}

/// A prompt group's steps all begin before its requests are taken: an
/// expert banked meanwhile (a promotion's) is mapped from the first step not
/// yet begun, which the group's steps resolve absent, so the group's request
/// fills its scratch slot (a round's convert, or a pull from the tier) for
/// the pass that reads it there, never counting it a hit the pass does not
/// read.
#[test]
fn an_expert_banked_after_its_prompt_group_began_fills_its_scratch_slot() {
    let harness = promoting("begun", 2, TIER);
    let map = &harness.layer.0;
    let slot = |kind: usize, expert: u64| {
        let mut bytes = [0u8; RECORD as usize];
        unsafe { harness.scratch[kind].read_shared(expert * RECORD, &mut bytes) }.unwrap();
        bytes
    };
    let landing = harness.request(&[5]);
    let prompt = harness.prompt_step();
    landing.complete().unwrap();
    eventually("5 was never promoted", || map.held(5).is_some());
    assert_eq!(map.resolve(5, prompt.sequence()), None, "the group's step resolves 5 absent");
    let converted = harness.converted(&prompt, 0, &[5]).concat();
    let pulled = (0..3).all(|kind| slot(kind, 5) == [(100 * kind + 5) as u8; RECORD as usize]);
    assert!(converted == [5] || pulled, "expert 5 neither converted nor pulled into its slot: {converted:?}");
    prompt.complete().unwrap();
}

/// A group's later step at a layer reads only what the scratch tier
/// lacks: each expert once per group and layer. Another layer's request
/// writes the tier, and the next group starts it empty. Each step's
/// compute is measured, the last layer's last to its completion.
#[test]
fn a_groups_steps_read_each_expert_once_per_layer() {
    let harness = layered_over("union", EXPERTS, 12, 2, STAGING, WHOLE);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    let listed = |flagged: &[usize], step: &ExpertStep, layer| {
        let mut experts = harness.converted(step, layer, flagged).concat();
        experts.sort_unstable();
        experts
    };
    assert_eq!(listed(&[0, 1], &steps[0], 0), [0, 1]);
    assert_eq!(listed(&[1, 2], &steps[1], 0), [2]);
    assert_eq!(listed(&[0, 3], &steps[0], 1), [0, 3]);
    assert_eq!(listed(&[0, 3, 4], &steps[1], 1), [4]);
    let first = steps[0].sequence();
    for step in steps {
        step.complete().unwrap();
    }
    let group = harness.watcher.shared().costs().group.unwrap();
    assert_eq!((group.first, group.steps), (first, 2));
    let took = group
        .layers
        .iter()
        .map(|layer| (layer.union, layer.conveyed, layer.cold_bytes))
        .collect::<Vec<_>>();
    assert_eq!(took, [(3, 3, 3 * 3 * RECORD), (3, 3, 3 * 3 * RECORD)]);
    let measured = group.layers.iter().map(|layer| layer.computed).collect::<Vec<_>>();
    assert_eq!(measured, [2, 2]);
    // The next group fills its scratch tier anew: each expert converted
    // again, or copied from the host tier its group's read landed it in.
    let next = harness.prompt_step();
    let mut filled = listed(&[0, 3], &next, 1);
    filled.extend(super::tests::listed(&harness.tiered[1]).into_iter().map(|(_, expert)| expert));
    filled.sort_unstable();
    assert_eq!(filled, [0, 3], "a new group, {first}'s tier forgotten");
    next.complete().unwrap();
}

/// Experts are read into the conveyor in the order the device converts
/// them, by the step that reads each: with a conveyor of one round's
/// records, the later step's experts (lower numbered) take none while the
/// first step's lack theirs, so the first step's round finds its experts
/// and every round is answered.
#[test]
fn a_conveyor_reads_experts_in_the_order_the_device_converts_them() {
    let harness = lending("order", EXPERTS, 4, 1, STAGING, Conveying::of(ROUND_RECORDS), TIER, Placer::Tier, false);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    harness.publish(&steps[0], 0, &[4, 5]);
    harness.publish(&steps[1], 0, &[0, 1]);
    std::thread::sleep(Duration::from_millis(50));
    // Each round is checked whole as it is answered: the device's arrival
    // at the next frees its records for the other step's reads.
    assert_eq!(Harness::experts(&harness.rounds_checked(&steps[0], 0).concat()), [4, 5]);
    assert_eq!(Harness::experts(&harness.rounds_checked(&steps[1], 0).concat()), [0, 1]);
    for step in steps {
        step.complete().unwrap();
    }
}

/// A group's steps request a layer before any round of it there, each from
/// its own chain place (every step's front there precedes its first back): no
/// request is served before its own first round arrives, each is served as it
/// does, in step order (the order the device reaches them), and every request
/// of the group had been published when the first was.
#[test]
fn a_groups_requests_of_a_layer_are_served_in_order_once_their_rounds_arrive() {
    let harness = prompted("fronts", EXPERTS, 4);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    let scratch = &harness.layer.2;
    for expert in 0..EXPERTS {
        harness.flags.store(expert, i32::from(expert < 2));
        harness.second.store(expert, i32::from((1..3).contains(&expert)));
    }
    for step in &steps {
        scratch.request(step.sequence(), constants::SCRATCH_ASK);
    }
    // Requested, but no round has arrived: neither is served.
    std::thread::sleep(Duration::from_millis(50));
    for step in &steps {
        assert_eq!(scratch.answer_of(step.sequence(), round(0)), None);
    }
    for step in &steps {
        harness.park(step, 0);
    }
    for step in steps {
        step.complete().unwrap();
    }
    let layer = harness.watcher.shared().costs().group.unwrap().layers[0].clone();
    assert_eq!((layer.union, layer.conveyed), (3, 3));
}

/// A request's experts the bank lacks are read into the conveyor as it is
/// published, before any of its rounds arrives, and nothing of the next
/// layer is; its rounds convert them from there (past the staging records)
/// and list none of its hits, read in place.
#[test]
fn a_request_reads_its_cold_experts_as_it_is_published() {
    let harness = layered_over("conveyor", EXPERTS, 12, 2, STAGING, WHOLE);
    harness.seeded(0, &[5]);
    let step = harness.prompt_step();
    harness.publish(&step, 0, &[1, 5, 6]);
    harness.conveying(1);
    harness.conveying(6);
    // Expert 5's bytes may lie in the conveyor still from the group that
    // seeded it; every other expert's never did.
    let mut unread = (0..EXPERTS as i32).filter(|expert| ![1, 5, 6].contains(expert));
    assert!(unread.all(|expert| harness.holding(expert).is_none()), "nothing else read");
    assert_eq!(harness.from_conveyor(&harness.rounds(&step, 0).concat()), [1, 6]);
    harness.scratch(&step, 1, &[0]);
    step.complete().unwrap();
    let layers = harness.watcher.shared().costs().group.unwrap().layers;
    assert_eq!((layers[0].conveyed, layers[1].conveyed), (2, 1), "the hit is read in place");
}

/// A scratch request awaiting its round 0 holds back no decode request:
/// the request service carries each request on its own slot, never waiting
/// inside one, so a decode step's request at another layer is answered
/// while the prompt step's round 0 has not arrived, and the prompt step is
/// served whole once it does.
#[test]
fn a_scratch_request_awaiting_its_arrival_holds_back_no_decode_request() {
    let harness = layered_over("independent", EXPERTS, 12, 2, STAGING, WHOLE);
    let decode = harness.step();
    let prompt = harness.prompt_step();
    harness.route_at(&decode, 0, &[]);
    harness.publish(&prompt, 0, &[1, 6]);
    // Taken: its cold experts are read into the conveyor; its round 0 has
    // not arrived.
    harness.conveying(1);
    harness.conveying(6);
    harness.request_at(&decode, 1, &[2]);
    assert_eq!(harness.verdict(1, decode.sequence()), Some(Answer::Answered));
    assert_eq!(harness.layer.2.answer_of(prompt.sequence(), round(0)), None, "the round awaits its arrival");
    decode.complete().unwrap();
    assert_eq!(harness.from_conveyor(&harness.rounds(&prompt, 0).concat()), [1, 6]);
    harness.scratch(&prompt, 1, &[0]);
    prompt.complete().unwrap();
}

/// A group runs as a wavefront: a step's request at the next layer is
/// taken, and its experts read, while the layer is still served, into the
/// records the layer's experts leave free (the layers share their records'
/// form). Its rounds convert them from the conveyor, a whole layer's or
/// five records.
#[test]
fn a_wavefront_reads_the_next_layers_experts_as_their_records_free() {
    for (name, conveying) in [("wavefront", WHOLE), ("wavefront-five", Conveying::of(5))] {
        // A record past TIER: the 5 the group lands and the reserve of 4.
        let harness = lending(name, EXPERTS, 12, 2, STAGING, conveying, TIER + 1, Placer::Tier, false);
        let steps = harness.group(&[StepRoute::Grouped; 2]);
        harness.publish(&steps[0], 0, &[0, 1]);
        harness.publish(&steps[1], 0, &[1, 3]);
        assert_eq!(harness.from_conveyor(&harness.rounds(&steps[0], 0).concat()), [0, 1]);
        // Step 0's front at layer 1, between the layer's backs: its experts
        // 3 (layer 1's) and 4 are read beside layer 0's 3, which step 1
        // converts.
        harness.publish(&steps[0], 1, &[3, 4]);
        assert_eq!(harness.from_conveyor(&harness.rounds(&steps[1], 0).concat()), [3]);
        harness.publish(&steps[1], 1, &[4]);
        assert_eq!(harness.from_conveyor(&harness.rounds(&steps[0], 1).concat()), [3, 4]);
        harness.rounds(&steps[1], 1);
        for step in steps {
            step.complete().unwrap();
        }
        // A decode step after them runs residency's refill as its requests
        // arrive (`Residency::advance`): whether each expert landed from the
        // conveyor (when free conveyor records exceeded the measured knee) or
        // reads on demand, the tier's reserve of 4 refills beside the lent
        // records and keeps all 5 without unmapping any for lent records.
        let next = harness.step();
        for (layer, experts) in [(0, &[0, 1, 3][..]), (1, &[3, 4][..])] {
            if harness.ask_at(&next, layer, experts) {
                assert_eq!(until_answered(&harness.layers[layer].1, next.sequence(), 0), Answer::Answered);
            }
        }
        next.complete().unwrap();
        for (layer, expert) in [(0, 0), (0, 1), (0, 3), (1, 3), (1, 4)] {
            eventually("decode landing never kept expert in tier", || harness.in_tier(layer, expert).is_some());
        }
    }
}

/// A round's records free as the device arrives at the round after, the
/// last round's too: the step a layer serves last leaves its records to the
/// next layer's reads before the device reaches that layer's rounds. Step
/// 0's front at layer 1 proves the device past step 0 at layer 0, not
/// past step 1 there; a conveyor of one round's records reads layer 1's
/// experts 4 and 5 only once step 1's records free.
#[test]
fn a_last_rounds_records_free_as_the_device_arrives_after_it() {
    let harness = lending("last-round", EXPERTS, 12, 2, STAGING, Conveying::of(ROUND_RECORDS), TIER, Placer::Tier, false);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    harness.publish(&steps[0], 0, &[0, 1]);
    harness.publish(&steps[1], 0, &[2, 3]);
    assert_eq!(harness.from_conveyor(&harness.rounds(&steps[0], 0).concat()), [0, 1]);
    assert_eq!(harness.from_conveyor(&harness.rounds(&steps[1], 0).concat()), [2, 3]);
    harness.publish(&steps[0], 1, &[4, 5]);
    harness.conveying(4);
    harness.conveying(5);
    assert_eq!(harness.from_conveyor(&harness.rounds(&steps[0], 1).concat()), [4, 5]);
    harness.scratch(&steps[1], 1, &[4, 5]);
    for step in steps {
        step.complete().unwrap();
    }
}

/// A round's records free at whichever proof of its converts comes first,
/// once: the device's arrival at the round after, or its progress past the
/// step (its next step's round 0). In either order, until one comes, no
/// read takes a record a convert may still read; then the next step's
/// experts are read into them, and the later proof frees nothing again. A
/// conveyor of one round's records: step 1's experts 2 and 3 wait on step
/// 0's records.
#[test]
fn a_rounds_records_free_once_whichever_proof_of_its_converts_comes_first() {
    for arrival_first in [true, false] {
        let name = if arrival_first { "arrival-first" } else { "progress-first" };
        let harness = lending(name, EXPERTS, 12, 1, STAGING, Conveying::of(ROUND_RECORDS), TIER, Placer::Tier, false);
        let steps = harness.group(&[StepRoute::Grouped; 2]);
        harness.publish(&steps[0], 0, &[0, 1]);
        harness.publish(&steps[1], 0, &[2, 3]);
        assert_eq!(harness.from_conveyor(&harness.rounds_arriving(&steps[0], 0, false).concat()), [0, 1]);
        // The device has neither arrived after step 0's round nor passed
        // the step: its records still hold what the convert reads.
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert_eq!((harness.holding(2), harness.holding(3)), (None, None), "{name}: a record freed before its convert was proven done");
        assert_eq!((harness.holding(0).is_some(), harness.holding(1).is_some()), (true, true), "{name}: step 0's records overwritten");
        if arrival_first {
            harness.arrive_after(&steps[0], 0, 1);
            harness.conveying(2);
            harness.conveying(3);
            assert_eq!(harness.from_conveyor(&harness.rounds(&steps[1], 0).concat()), [2, 3]);
        } else {
            assert_eq!(harness.from_conveyor(&harness.rounds(&steps[1], 0).concat()), [2, 3]);
            harness.arrive_after(&steps[0], 0, 1);
        }
        for step in steps {
            step.complete().unwrap();
        }
    }
}

/// A group's last step lists its seeds at every layer: the most recently
/// routed experts (then the higher expert, as placement ranks a tie),
/// into free records; a step before it lists none, so its seed list
/// names another step.
#[test]
fn a_groups_last_step_seeds_the_most_recent_experts() {
    let harness = prompted("seeds", EXPERTS, 4);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    let seeds = &harness.seeds[0];
    harness.scratch(&steps[0], 0, &[6, 7]);
    assert_ne!(seeds.load(constants::LIST_SEQUENCE as usize), low_word(steps[0].sequence()));
    harness.scratch(&steps[1], 0, &[0, 1, 2, 3, 4, 5, 7]);
    assert_eq!(seeds.load(constants::LIST_SEQUENCE as usize), low_word(steps[1].sequence()));
    let listed = listed(seeds);
    // The last step's experts, the highest first; expert 6, routed only
    // by the step before, ranks below them all.
    assert_eq!(listed, [(7, 0), (5, 1), (4, 2), (3, 3)]);
    for step in steps {
        step.complete().unwrap();
    }
}

/// A seed retires only once its step has ended whole: the device past a
/// prompt step's layer 0 (its request at layer 1 arrives) while the step
/// still runs maps nothing, since a late beat of the host's pulse during
/// layer 0's rounds may yet taint the step; once it ends tainted, its seeds
/// are never mapped.
#[test]
fn a_running_steps_seeds_wait_for_its_end_and_a_tainted_ones_are_never_mapped() {
    let harness = hosted_prompted("seed-taint", 4, 2, TIER);
    let pulse = harness.watcher.shared().steps.pulse();
    let step = harness.prompt_step();
    harness.scratch(&step, 0, &[3, 4]);
    assert_eq!(harness.commit_seeds(&step, 0).len(), 2, "layer 0 seeds 3 and 4");
    let before = pulse.late_beats();
    pulse.beat_after(Duration::from_millis(4 * constants::PULSE_SPINS as u64 * 70 / 1_000_000));
    assert_eq!(pulse.late_beats(), before + 1, "the late beat was never counted");
    // The device passes layer 0: its request at layer 1 arrives and is
    // served, the step still running.
    harness.scratch(&step, 1, &[]);
    let map = &harness.layers[0].0;
    assert!(map.held(3).is_none() && map.held(4).is_none(), "a running step's seeds were mapped");
    // A wait of the step outlasted the still pulse, as its kernel records.
    harness.expire(&step, 1);
    assert!(matches!(step.complete(), Err(StepFailure::Interrupted(_))));
    let next = harness.step();
    harness.request_at(&next, 0, &[1]);
    drop(harness.route_at(&next, 1, &[]));
    next.complete().unwrap();
    assert!(map.held(3).is_none() && map.held(4).is_none(), "a tainted step's seeds were mapped");
}

/// A group's last step whose wait at a layer outlasts the still pulse
/// before the host takes its request there seeds nothing there: the seed
/// copy after the layer's last round reads the layer's own seed list, which
/// holds only earlier steps', never the list the step's request at the
/// layer before wrote. That list names records of the layer before's bank
/// group, which in the next layer's group may hold live experts.
#[test]
fn a_last_step_expiring_before_its_request_is_taken_seeds_nothing_there() {
    let harness = hosted_prompted("seed-expired", 4, 2, TIER);
    let pulse = harness.watcher.shared().steps.pulse();
    let step = harness.prompt_step();
    harness.scratch(&step, 0, &[3, 4]);
    assert_eq!(harness.commit_seeds(&step, 0).len(), 2, "layer 0 seeds 3 and 4");
    // The device reaches layer 1, the host's pulse still, and its wait
    // there ends on it before the host takes the request.
    pulse.beat_after(Duration::from_millis(4 * constants::PULSE_SPINS as u64 * 70 / 1_000_000));
    harness.expire(&step, 1);
    assert_eq!(harness.seed_pairs(&step, 1), [], "layer 1's seed copy copies layer 0's seeds");
    assert!(matches!(step.complete(), Err(StepFailure::Interrupted(_))));
}

/// Beside a host tier a prompt group's scratch lies in its lender's bottom
/// bank records, which no promotion takes: the group's lease takes them at
/// once; no seed takes one (the bank otherwise holds only its pool, so the
/// group seeds nothing); once the device is past the group they return
/// free, so a shrink after the group evicts past them, decode's capacity is
/// the records past them, and the next group is leased them again.
#[test]
fn a_discrete_groups_scratch_leases_its_bank_records_and_returns_them_free() {
    let harness = lending_bank("bank-lent", EXPERTS, 4, 1, STAGING, WHOLE, TIER, Placer::Tier, true, 2, false);
    let map = &harness.layer.0;
    let lent = |record: u32| harness.watcher.bank_lent(0, record);
    // Promotions take the free records past the lent ones in order: 2 and
    // 3 records 2 and 3, the pool of two.
    harness.promoted(0, &[2, 3]);
    assert_eq!((0..4).map(|expert| map.held(expert)).collect::<Vec<_>>(), [None, None, Some(2), Some(3)]);
    let step = harness.prompt_step();
    harness.scratch(&step, 0, &[5]);
    assert!(lent(0) && lent(1) && !lent(2) && !lent(3), "the group holds the lent records alone");
    assert_eq!(harness.commit_seeds(&step, 0), [], "no seed takes a lent record or the pool's");
    // A step the load interrupts fails here; its seeds are none anyway.
    drop(step.complete());
    eventually("the lent records never returned free", || !lent(0) && !lent(1));
    let shared = harness.watcher.shared();
    eventually("the group's reads were never ranked", || shared.telemetry.fed_all());
    let evicted = shared.shrink(super::super::resize::Evicting::Bank, Candidates::one(3, vec![3]), None).unwrap().expect("the shrink evicts beside no step");
    assert!(evicted.records[0].iter().all(|&record| record >= 2), "a shrink never evicts a lent record");
    assert_eq!(shared.costs().records, [1], "decode's capacity: the records past the lent ones");
    let step = harness.prompt_step();
    harness.scratch(&step, 0, &[6]);
    assert!(lent(0) && lent(1), "the next group is leased the records again");
    harness.commit_seeds(&step, 0);
    drop(step.complete());
}

/// A request taken while the layer before is served counts a hit what the
/// bank then holds; that layer's seeding evicting it (it lies outside its
/// layer's share) makes it a read, so its back reads no record the seeds
/// overwrite. Through a conveyor of five records, the demoted read and the
/// next layer's wait for the layer's records, which free as the device
/// passes it. The group that seeded the hit landed its read into the host
/// tier too, which keeps it (its room holds every expert), so the demoted
/// hit is copied from it whole into its scratch slot, never converted.
#[test]
fn a_hit_the_layer_befores_seeding_evicts_is_read() {
    for (name, conveying) in [("demote", WHOLE), ("demote-five", Conveying::of(5))] {
        // One group of 4 records for both layers, a share of two each,
        // beside a host tier of room for every expert, so its reserve never
        // unmaps one the group read.
        let harness = lending(name, EXPERTS, 4, 2, STAGING, conveying, 2 * EXPERTS as u32 + STAGING, Placer::Tier, true);
        harness.seeded(1, &[5]);
        // Layer 1's decode reads of 6 and 7 and layer 0's of 6 fill the
        // group's 4 records beside `(1, 5)`; two decode steps routing
        // `(0, [6, 7])` and `(1, [6, 7])` give layer 0 measured decode uses
        // so `(1, 5)` falls out of the top 4 by rank while all 4 records
        // stay full (no free record may take the seed, so seeding must
        // evict `(1, 5)`).
        harness.promoted(1, &[6, 7]);
        harness.promoted(0, &[6]);
        // Give `(0, 3)` one decode use so it strictly outranks `(1, 5)` (0
        // decode uses) without entering the 4-record `in_top` pool (held by
        // `(0, 6), (0, 7), (1, 6), (1, 7)` with 2+ uses each), so seeding
        // evicts `(1, 5)` while promotions never touch `(0, 3)`.
        let once = harness.step();
        harness.route_at(&once, 0, &[3]);
        harness.route_at(&once, 1, &[]);
        once.complete().unwrap();
        for _ in 0..2 {
            let decode = harness.step();
            harness.route_at(&decode, 0, &[6, 7]);
            harness.route_at(&decode, 1, &[6, 7]);
            decode.complete().unwrap();
        }
        let telemetry = &harness.watcher.shared().telemetry;
        eventually("the decode reads were never ranked", || telemetry.fed_all() && !telemetry.in_top()(1, 5));
        assert!(harness.layers[1].0.held(5).is_some(), "(1, 5) stays held in the full bank before the prompt group");
        // The refill past the step, which decides what the group finds in
        // the tier.
        harness.tier_ready();
        let steps = harness.group(&[StepRoute::Grouped; 2]);
        harness.publish(&steps[0], 0, &[0, 1]);
        harness.publish(&steps[1], 0, &[0, 1, 2, 3]);
        harness.rounds(&steps[0], 0);
        // Layer 1's expert 5 is held when step 0's request there is taken
        // (expert 4 read); step 1 routes layer 0's seeds after it.
        harness.publish(&steps[0], 1, &[4, 5]);
        harness.conveying(4);
        // Step 1, the group's last at layer 0, seeds into the full bank,
        // evicting layer 1's 5 (the one kept record outside the top 4).
        harness.rounds(&steps[1], 0);
        assert_eq!(listed(&harness.seeds[0]).len(), 1);
        let first = steps[0].sequence();
        assert_eq!(harness.layers[1].0.held(5), None, "the evicted hit is unmapped");
        assert_eq!(harness.layers[1].0.resolve(5, first), None, "for every step of the group");
        harness.publish(&steps[1], 1, &[3]);
        // Expert 5's scratch slot, empty before step 0's tier there opens.
        let slot = |kind: usize| {
            let mut bytes = [0u8; RECORD as usize];
            unsafe { harness.scratch[kind].read_shared(5 * RECORD, &mut bytes) }.unwrap();
            bytes
        };
        for kind in 0..3 {
            unsafe { harness.scratch[kind].write_shared(5 * RECORD, &[0; RECORD as usize]) }.unwrap();
        }
        // The demoted hit 5 is either copied whole from the host tier (if
        // already landed there) or read and converted from the conveyor.
        let mut converted = harness.from_conveyor(&harness.rounds(&steps[0], 1).concat());
        converted.sort_unstable();
        let copied = (0..3).all(|kind| slot(kind) == [(100 * kind + 5) as u8; RECORD as usize]);
        assert!(
            converted == [4, 5] || (converted == [4] && copied),
            "demoted hit 5 neither converted from conveyor nor copied from tier: converted={converted:?} copied={copied}"
        );
        // Step 1's 3, cold (6 is banked), is read ahead as its request was
        // taken, into the conveyor and on into the tier. Its round converts
        // it from the conveyor, or where that landing ended and the conveyor
        // let it go before the device arrived, its tier copy reads it: the
        // one race here is which, and either fills its slot once.
        let converted = harness.from_conveyor(&harness.rounds(&steps[1], 1).concat());
        let tiered = listed(&harness.tiered[1]).iter().any(|&(_, expert)| expert == 3);
        assert!(converted == [3] || (converted.is_empty() && tiered), "converted {converted:?}, tier copied {tiered}");
        for step in steps {
            step.complete().unwrap();
        }
    }
}

/// The size rule: the fewest steps whose compute covers every layer's
/// stage, within the capacity; to capacity until every layer's compute is
/// measured and its lines priced.
#[test]
fn a_group_is_the_fewest_steps_covering_each_layers_stage() {
    let line = magnitude_cost_model::ReadLine::new(0.001, 1e-9).unwrap();
    let lines = vec![magnitude_cost_model::ExpertLines::reference(1_000_000); 2];
    let ms = Duration::from_millis;
    let mut facts = GroupFacts {
        layers: vec![
            LayerFacts {
                compute: Some(ms(10)),
                held: 56,
                alone: 0,
                read: None,
                pulled: None,
                union: None,
                bytes: 1_000_000,
                resident: None,
                experts: 256,
            },
            LayerFacts {
                compute: None,
                held: 0,
                alone: 0,
                read: None,
                pulled: None,
                union: None,
                bytes: 1_000_000,
                resident: None,
                experts: 256,
            },
        ],
        windows: 1,
    };
    assert_eq!(facts.group_size(Some(&lines), Some(line), Notice::Written, 16), 16, "layer 1 is unmeasured");
    facts.layers[1].compute = Some(ms(40));
    assert_eq!(facts.group_size(None, Some(line), Notice::Written, 16), 16, "the lines are unpriced");
    // Layer 0: 1 ms + 200 MB at 1 GB/s, then 200 moves of 1 MB records,
    // 44.56 µs each, 209.912 ms over 10 ms: 21 steps; layer 1: 268.407 ms over
    // 40 ms: 7.
    assert_eq!(facts.group_size(Some(&lines), Some(line), Notice::Written, 32), 21);
    assert_eq!(facts.group_size(Some(&lines), Some(line), Notice::Written, 16), 16);
    facts.layers[0].held = 256;
    assert_eq!(facts.group_size(Some(&lines), Some(line), Notice::Written, 16), 7);
}

/// A layer's file experts are every expert it lacks until a group has
/// ended there, then what that group read there. Where prompt groups pull
/// from a host tier, the experts it holds alone are pulled instead until a
/// group has ended there, then what that group pulled.
#[test]
fn a_layers_read_is_what_the_last_group_to_end_there_read() {
    let mut facts = GroupFacts {
        layers: vec![
            LayerFacts {
                compute: Some(Duration::from_millis(10)),
                held: 56,
                alone: 0,
                read: None,
                pulled: None,
                union: None,
                bytes: 1_000_000,
                resident: None,
                experts: 256,
            },
            LayerFacts {
                compute: Some(Duration::from_millis(10)),
                held: 0,
                alone: 0,
                read: None,
                pulled: None,
                union: None,
                bytes: 1_000_000,
                resident: None,
                experts: 256,
            },
        ],
        windows: 1,
    };
    let moved = |facts: &GroupFacts| {
        let terms = facts.terms().unwrap();
        terms.iter().map(|layer| (layer.filed, layer.tiered)).collect::<Vec<_>>()
    };
    assert_eq!(moved(&facts), [(200, 0), (256, 0)]);
    facts.layers[0].read = Some(3);
    assert_eq!(moved(&facts), [(3, 0), (256, 0)]);
    facts.layers[0].alone = 10;
    facts.layers[1].alone = 6;
    assert_eq!(moved(&facts), [(3, 0), (256, 0)], "without pulls the tier's experts are read");
    for layer in &mut facts.layers {
        layer.resident = Some(2_000_000);
    }
    assert_eq!(moved(&facts), [(3, 10), (250, 6)]);
    facts.layers[1].pulled = Some(4);
    assert_eq!(moved(&facts)[1], (250, 4));
}

/// Unpriced, nothing weighs a merge set: a layer with nothing cold after
/// it merges whole, however long the run goes unpriced, and only a layer
/// before cold experts splits off its first back. The last layer is
/// followed by nothing.
#[test]
fn an_unpriced_group_splits_only_before_cold_experts() {
    let mut facts = GroupFacts {
        layers: vec![
            LayerFacts {
                compute: None,
                held: 256,
                alone: 0,
                read: None,
                pulled: None,
                union: None,
                bytes: 1_000_000,
                resident: Some(2_000_000),
                experts: 256,
            };
            4
        ],
        windows: 1,
    };
    assert_eq!(facts.merge_sets(None, None, None, Notice::Written, 6), [6, 6, 6, 6], "a group with nothing cold");
    facts.layers[2].pulled = Some(52);
    assert_eq!(facts.merge_sets(None, None, None, Notice::Written, 6), [6, 5, 6, 6], "the pulls before layer 2");
    facts.windows = 2;
    assert_eq!(facts.merge_sets(None, None, None, Notice::Written, 6), [6, 6, 6, 6], "a windowed scratch tier merges every layer whole");
    facts.windows = 1;
    facts.layers[2].pulled = Some(0);
    facts.layers[3].held = 200;
    assert_eq!(facts.merge_sets(None, None, None, Notice::Written, 6), [6, 6, 5, 6], "the reads before layer 3");
    assert_eq!(facts.merge_sets(None, None, None, Notice::Written, 1), [1; 4]);
}

/// Policy whole is per-layer: a layer holding every expert at its home sets
/// its own `POLICY_WHOLE` and `POLICY_RESOLVED` immediately even while a
/// sibling layer in the same group is still partial; once the sibling also
/// holds every expert at its home, both layers have `POLICY_WHOLE == 1`.
#[test]
fn a_layer_is_whole_independent_of_partial_siblings_in_its_group() {
    // The tier keeps every expert either layer lands, beyond its reserve.
    let harness = lending("whole-layer", EXPERTS, 2 * EXPERTS as u64, 2, STAGING, DECODE, STAGING + 2 * EXPERTS as u32, Placer::Tier, true);
    let shared = harness.watcher.shared();
    let whole = |layer: usize| harness.layers[layer].5.load(constants::POLICY_WHOLE as usize);
    let resolved = |layer: usize| harness.layers[layer].5.load(constants::POLICY_RESOLVED as usize);
    harness.promoted(0, &(0..EXPERTS).collect::<Vec<_>>());
    harness.promoted(1, &(0..6).collect::<Vec<_>>());
    assert_eq!(shared.holdings().0, [EXPERTS, 6]);
    // Layer 0 holds all of its experts at home; layer 1 is still partial.
    until(&harness.layers[0].5, constants::POLICY_WHOLE as usize, 1);
    assert_eq!((whole(0), whole(1)), (1, 0), "a full layer at home is whole while its sibling is partial");
    assert_eq!(resolved(0), 1, "a whole layer is also resolved");
    harness.promoted(1, &[6, 7]);
    until(&harness.layers[1].5, constants::POLICY_WHOLE as usize, 1);
    assert_eq!((whole(0), whole(1)), (1, 1));
}

/// Reads run ahead of the device's rounds: every cold expert of a layer,
/// more than staging beyond its reserve holds, and the next layer's,
/// published while the layer is served, are read into conveyor records
/// before any round there arrives; each layer's rounds convert them from
/// there, and staging is never written.
#[test]
fn a_conveyor_reads_the_next_layer_ahead_of_its_rounds() {
    conveys_the_next_layer_ahead(&lending("ahead", EXPERTS, 12, 2, STAGING, WHOLE, TIER, Placer::Tier, false));
}

/// [`a_conveyor_reads_the_next_layer_ahead_of_its_rounds`] on the one
/// store, the conveyor's records lent from the lender's host records,
/// which a prompt group claims as its first request is taken.
#[test]
fn the_one_stores_conveyor_reads_the_next_layer_ahead_of_its_rounds() {
    conveys_the_next_layer_ahead(&lending("ahead-store", EXPERTS, 2 * EXPERTS as u64, 2, STAGING, WHOLE, 2 * EXPERTS as u32, Placer::Bank, false));
}

/// On the one store with a conveyor of two rounds (as a load there lays
/// it), lent from the lender's host records: a group of two steps, step
/// 0's cold experts at layer 0 one round; step 0's front at layer 1,
/// published while step 1 is still served at layer 0, has its cold
/// experts read into the conveyor's other round before step 1's round
/// there arrives, though layer 0's round still holds its records.
#[test]
fn the_one_stores_two_round_conveyor_reads_the_next_layer_ahead() {
    let conveying = Conveying::of(2 * ROUND_RECORDS);
    let harness = lending("ahead-rounds", EXPERTS, 2 * EXPERTS as u64, 2, STAGING, conveying, 2 * EXPERTS as u32, Placer::Bank, false);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    harness.publish(&steps[0], 0, &[0, 1]);
    harness.publish(&steps[1], 0, &[0]);
    assert_eq!(harness.from_conveyor(&harness.rounds(&steps[0], 0).concat()), [0, 1]);
    harness.publish(&steps[0], 1, &[6, 7]);
    harness.conveying(6);
    harness.conveying(7);
    assert!(harness.rounds(&steps[1], 0).is_empty(), "step 1 converts nothing at layer 0");
    harness.publish(&steps[1], 1, &[7]);
    assert_eq!(harness.from_conveyor(&harness.rounds(&steps[0], 1).concat()), [6, 7]);
    harness.rounds(&steps[1], 1);
    for step in steps {
        step.complete().unwrap();
    }
    // The group's timeline shows it: layer 1's reads were submitted before
    // the layer before was last served whole.
    let layers = harness.watcher.shared().costs().group.unwrap().layers;
    let ahead = layers[1].timeline.submitted.unwrap();
    let before = layers[0].timeline.steps[1].served.unwrap();
    assert!(ahead.first <= before, "layer 1's reads at {ahead:?}, layer 0's last step served at {before:?}");
    // Each step's own events, at every layer: its request taken, its
    // round 0 arrived and served whole after it.
    for layer in &layers {
        assert_eq!(layer.timeline.steps.len(), 2);
        for step in &layer.timeline.steps {
            let (requested, arrived, served) = (step.requested.unwrap(), step.arrived.unwrap(), step.served.unwrap());
            assert!(requested <= served && arrived <= served, "{step:?}");
        }
    }
}

fn conveys_the_next_layer_ahead(harness: &Harness) {
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    let cold = [0, 1, 2, 3, 4, 5];
    harness.publish(&steps[0], 0, &cold);
    harness.publish(&steps[1], 0, &[0]);
    for expert in cold {
        harness.conveying(expert as i32);
    }
    assert_eq!(harness.from_conveyor(&harness.rounds(&steps[0], 0).concat()), cold.map(|expert| expert as i32));
    // Step 0's front at layer 1, taken while layer 0 is served: its cold
    // experts arrive in conveyor records the layer's leave free, before
    // step 1's round at layer 0.
    harness.publish(&steps[0], 1, &[6, 7]);
    harness.conveying(6);
    harness.conveying(7);
    assert!(harness.rounds(&steps[1], 0).is_empty(), "step 1 converts nothing at layer 0");
    harness.publish(&steps[1], 1, &[7]);
    assert_eq!(harness.from_conveyor(&harness.rounds(&steps[0], 1).concat()), [6, 7]);
    harness.rounds(&steps[1], 1);
    for step in steps {
        step.complete().unwrap();
    }
    for record in 0..STAGING as i32 {
        assert_eq!(harness.staged(0, record), 0, "staging untouched");
    }
    let layers = harness.watcher.shared().costs().group.unwrap().layers;
    assert_eq!(layers.iter().map(|layer| layer.conveyed).collect::<Vec<_>>(), [6, 2]);
}

/// A group claims its lender's records as its first request is taken: each
/// expert a lent record kept leaves the tier (its tier map entry cleared),
/// and once the group is past the records return empty and take landings
/// again.
#[test]
fn a_group_claims_its_lent_records_and_returns_them_empty() {
    let conveying = Conveying::of(4);
    // The tier is the four lent records alone. Its reserve (a record per
    // staging record) is two, so two landings leave it whole: a reserve of
    // every record would unmap each landing as residency learns it.
    let harness = lending("lender", EXPERTS, 12, 1, 2, conveying, 0, Placer::Tier, false);
    harness.request(&[0, 1]).complete().unwrap();
    // Residency maps a landing into the tier as it learns it, which may
    // follow the answer.
    harness.tiering(0, &[0, 1]);
    let step = harness.prompt_step();
    harness.publish(&step, 0, &[2]);
    eventually("the lent records were never claimed", || harness.in_tier(0, 0).is_none() && harness.in_tier(0, 1).is_none());
    harness.conveying(2);
    harness.rounds(&step, 0);
    harness.commit_seeds(&step, 0);
    step.complete().unwrap();
    // The group returns its lent records once the watcher learns the device
    // is past it and its conveyor quiet, after its step completes. The tier
    // is the lent records alone, so a request taken before then finds no
    // record to land in and reads from the file.
    eventually("the group never returned its lent records", || {
        let state = residency::lock(&harness.watcher.state);
        (0..conveying.lent as u32).all(|record| !state.core.directory.lent_claimed(0, crate::experts::directory::HostRecord(record)))
    });
    harness.request(&[3]).complete().unwrap();
    // The returned records take a landing, mapped as residency learns it.
    harness.tiering(0, &[3]);
}

/// A bank imported whole into the one store keeps every expert where a
/// pass reads it in place, its lender's lent records among them: none can
/// miss, so every layer's whole word is set from the bank's open, and a
/// prompt group is resident, not scratch-served (`ExpertBank::resident`).
/// It requests nothing, so it claims no lent record and evicts nothing:
/// the store still keeps every expert, none read from storage again.
#[test]
fn a_whole_banks_prompt_group_reads_nothing_and_claims_no_lent_record() {
    let conveying = Conveying::of(4);
    let harness = lending_bank("whole-store", EXPERTS, 2 * EXPERTS as u64, 2, STAGING, conveying, 2 * EXPERTS as u32, Placer::Bank, false, 0, true);
    let whole = (0..2).all(|layer| harness.layers[layer].5.load(constants::POLICY_WHOLE as usize) == 1);
    assert!(whole, "a bank imported whole into the one store is whole");
    let steps = begin(harness.watcher.shared(), &[StepRoute::Grouped; 2], !whole).unwrap();
    for step in steps {
        step.complete().unwrap();
    }
    let state = residency::lock(&harness.watcher.state);
    let directory = &state.core.directory;
    assert!((0..4).all(|record| !directory.lent_claimed(0, crate::experts::directory::HostRecord(record))), "a lent record was claimed");
    assert!(directory.tier_complete() && directory.group_whole(0), "the store gave up an expert");
}

/// On the one store a hint's landing is mapped past the steps already
/// begun, so a running group's step misses an expert the store keeps by
/// then. No tier copy binds there: an expert beyond the conveyor's free
/// records when the rounds are fixed stays the rounds' own, read from the
/// file, and the tier list, whose bound there admits no record, stays empty.
#[test]
fn the_one_stores_rounds_read_a_cold_expert_the_store_keeps() {
    let conveying = Conveying::of(2);
    // Two layers, so a decode step's ring at layer 0 hints layer 1.
    let harness = lending("store-kept", EXPERTS, 2 * EXPERTS as u64, 2, 2 * STAGING, conveying, 2 * EXPERTS as u32, Placer::Bank, false);
    let step = harness.prompt_step();
    // A decode step behind the group hints layer 1's kept experts.
    let decode = harness.step();
    let kept = [5, 6, 7];
    harness.hint(&decode, 0, &[5, 6, 7]);
    let map = &harness.layers[1].0;
    eventually("the hints never landed the experts", || kept.iter().all(|&expert| map.held(expert).is_some()));
    assert!(kept.iter().all(|&expert| map.resolve(expert, step.sequence()).is_none()), "landed past the begun step");
    harness.scratch(&step, 0, &[]);
    let rounds = harness.scratch(&step, 1, &kept);
    let mut converted = rounds.concat().iter().map(|&(_, expert)| expert as usize).collect::<Vec<_>>();
    converted.sort_unstable();
    assert_eq!(converted, kept, "the rounds convert every one");
    assert_eq!(harness.from_conveyor(rounds.last().unwrap()).len(), 1, "the last, read into the conveyor once a record freed");
    assert_eq!(listed(&harness.tiered[1]), [], "no tier copy on the one store");
    step.complete().unwrap();
    for layer in 0..2 {
        harness.route_at(&decode, layer, &[]);
    }
    decode.complete().unwrap();
}

/// The one store holds only what its steps use: a load opens it empty, no
/// record committed before a step routes an expert, and a decode step that
/// routes nothing leaves it so.
#[test]
fn the_one_store_opens_empty() {
    let harness = lending("store-empty", EXPERTS, EXPERTS as u64, 2, STAGING, Conveying::of(2), TIER, Placer::Bank, false);
    let held = |layer: usize| (0..EXPERTS).filter(|&expert| harness.layers[layer].0.held(expert).is_some()).collect::<Vec<_>>();
    assert_eq!((held(0), held(1)), (vec![], vec![]), "the store opened holding experts");
    let step = harness.step();
    for layer in 0..2 {
        harness.request_at(&step, layer, &[]);
    }
    step.complete().unwrap();
    harness.tier_ready();
    assert_eq!((held(0), held(1)), (vec![], vec![]), "the store took experts no step routed");
}

/// On the one store a prompt group's last step seeds what it routed, read
/// into a conveyor record or straight into a scratch slot alike: its seed
/// list names every routed expert, the device's seed copy writes each from
/// its slot past the lent records, and once the device commits them and the
/// group ends the store keeps them, mapped past the group, which reads them
/// from its slots. Nothing lands from the conveyor. The group counts its
/// seeds.
#[test]
fn a_prompt_seeds_the_one_store_with_every_expert_it_routed() {
    let conveying = Conveying::of(ROUND_RECORDS).lending_scratch();
    let harness = lending("seed-store", EXPERTS, 12, 1, STAGING, conveying, TIER, Placer::Bank, false);
    let map = &harness.layers[0].0;
    let routed = [0, 2, 4, 6];
    let step = harness.prompt_step();
    harness.publish(&step, 0, &routed);
    let rounds = harness.rounds_checked(&step, 0).concat();
    assert_eq!(Harness::experts(&rounds), routed.map(|expert| expert as i32));
    assert!(!Harness::from_slots(&rounds).is_empty(), "no expert was read into a slot: {rounds:?}");
    let seeds = harness.commit_seeds(&step, 0);
    let mut seeded = seeds.iter().map(|&(expert, _)| expert).collect::<Vec<_>>();
    seeded.sort_unstable();
    assert_eq!(seeded, routed.map(|expert| expert as i32), "every routed expert seeds");
    assert!(seeds.iter().all(|&(_, record)| record as usize >= conveying.lent), "a seed took a lent record: {seeds:?}");
    assert!(routed.iter().all(|&expert| map.resolve(expert, step.sequence()).is_none()), "seeded within the running group");
    step.complete().unwrap();
    eventually("the seeds were never mapped", || {
        seeds.iter().all(|&(expert, record)| map.held(expert as usize) == Some(record as usize))
    });
    let layers = harness.watcher.shared().costs().group.unwrap().layers;
    assert_eq!(layers[0].seeded, routed.len() as u64);
}

/// A resident prompt group requests nothing; its passes' grouping writes
/// its reads into its group reads record (`BankPart::GroupReads`). Once
/// every step of it has ended the stack holds them, by layer at the
/// group's last step, and a decode step after it still feeds the stack.
#[test]
fn a_resident_prompt_groups_reads_reach_the_stack() {
    use crate::experts::horizon::{Reader, Tier};

    let harness = layered("group-reads", EXPERTS, 12, 2, STAGING);
    let steps = begin(harness.watcher.shared(), &[StepRoute::Grouped; 2], false).unwrap();
    let record = steps[0].group_reads().unwrap();
    assert_eq!(steps[1].group_reads(), Some(record), "one record a group");
    // Play the passes' counts: layer 0's single set at place 0, then its
    // merged set at place 1; layer 1's one set at place 0.
    let count = |layer: usize, place: usize, expert: usize, choices: i32| {
        harness.reads[layer].store((record * 2 + place) * EXPERTS + expert, choices);
    };
    count(0, 0, 1, 3);
    count(0, 1, 4, 2);
    count(1, 0, 6, 1);
    let shared = harness.watcher.shared();
    assert!(!shared.telemetry.group_union(0) && !shared.telemetry.group_union(1), "fed only once the group ends");
    for step in steps {
        step.complete().unwrap();
    }
    assert!(shared.telemetry.group_union(0) && shared.telemetry.group_union(1), "prompt group unions reached the stack");
    {
        let state = residency::lock(&harness.watcher.state);
        let reuse = state.core.directory.reuse_now();
        assert!(reuse.has_read(0) && reuse.has_read(1), "both layers logged the prompt union");
        let rank = |layer: usize, expert: usize| reuse.rank(layer, expert, Tier::COMPUTE, Reader::Move, None);
        assert!(rank(0, 1) > rank(0, 4) && rank(0, 4) > rank(0, 6), "layer 0 ranked both places' counts above unrouted experts");
        assert!(rank(1, 6) > rank(1, 1), "layer 1 ranked only what it grouped");
    }
    // Prompt evidence never enters decode last_read.
    let read = shared.telemetry.last_read();
    assert_eq!([read(0, 1), read(0, 4), read(1, 6)], [None; 3]);
    drop(read);
    // A decode step after the group feeds its routing records.
    let decode = harness.step();
    for (layer, experts) in [(0, [2]), (1, [5])] {
        harness.request_at(&decode, layer, &experts);
    }
    let sequence = decode.sequence();
    decode.complete().unwrap();
    let read = shared.telemetry.last_read();
    assert_eq!([read(0, 2), read(1, 5), read(0, 1)], [Some(sequence), Some(sequence), None]);
}

/// A generation group in the grouped form (a verify wider than the decode
/// form) is no prompt group: no prompt runs while it does, its reads rank
/// the bank as a decode's do, and no layer takes its union as a prompt
/// group's read, which prices a prompt's places.
#[test]
fn a_generation_group_in_the_grouped_form_prices_no_prompt() {
    let harness = layered("generation-group", EXPERTS, 12, 2, STAGING);
    let shared = harness.watcher.shared();
    let steps = begin_as(shared, &[(StepRoute::Grouped, LaunchWork::Generation)], false).unwrap();
    assert!(!shared.steps.prompt_running(), "a verify is no prompt");
    let record = steps[0].group_reads().expect("the grouped form records its reads");
    harness.reads[0].store(record * 2 * EXPERTS + 3, 2);
    let sequence = steps[0].sequence();
    for step in steps {
        step.complete().unwrap();
    }
    let read = shared.telemetry.last_read();
    assert_eq!(read(0, 3), Some(sequence), "its reads rank the bank");
    drop(read);
    assert!(!shared.telemetry.group_union(0), "no prompt's union");
    // A prompt group's reads are its union.
    let steps = begin(shared, &[StepRoute::Grouped], false).unwrap();
    assert!(shared.steps.prompt_running());
    let record = steps[0].group_reads().unwrap();
    harness.reads[0].store(record * 2 * EXPERTS + 5, 1);
    for step in steps {
        step.complete().unwrap();
    }
    assert!(shared.telemetry.group_union(0));
}

/// A prompt group routing every expert of every layer, each layer's share
/// chosen three times as often as the rest, leaves each layer at its ranked
/// share as decode starts: the experts the ranking's share of it names once
/// the group's reads are logged (`Reuse::in_top`), the union's choice counts
/// its prior. On the one
/// store the group takes only its lent records, so the first decode step
/// after it finds each layer's decode share held wherever it lay outside
/// them. Beside a host tier that keeps every expert, no seed takes
/// a record of a layer's share (the last layer's seeds never take the bank
/// from the layers before it) and promotions bring each layer's share in:
/// the bank holds exactly each layer's share.
#[test]
fn a_prompt_group_leaves_every_layer_its_ranked_share() {
    for placer in [Placer::Bank, Placer::Tier] {
        ranked_share(placer);
    }
}

/// [`a_prompt_group_leaves_every_layer_its_ranked_share`] on `placer`: four
/// layers whose decode reads rank two experts each, a bank of two records a
/// layer.
fn ranked_share(placer: Placer) {
    const LAYERS: usize = 4;
    let share = |layer: usize| [2 * layer, 2 * layer + 1];
    let tier = match placer {
        Placer::Bank => EXPERTS as u32 + 8,
        Placer::Tier => (EXPERTS * LAYERS) as u32 + 8,
    };
    let harness = lending("ranked-share", EXPERTS, 2 * LAYERS as u64, LAYERS, 8, WHOLE, tier, placer, placer == Placer::Tier);
    for _ in 0..4 {
        let step = harness.step();
        for layer in 0..LAYERS {
            harness.request_at(&step, layer, &share(layer));
        }
        step.complete().unwrap();
    }
    let telemetry = &harness.watcher.shared().telemetry;
    let map = |layer: usize| &harness.layers[layer].0;
    let held = |layer: usize| (0..EXPERTS).filter(|&expert| map(layer).held(expert).is_some()).collect::<Vec<_>>();
    let tops = || {
        let top = telemetry.in_top();
        (0..LAYERS).map(|layer| (0..EXPERTS).filter(|&expert| top(layer, expert)).collect::<Vec<_>>()).collect::<Vec<_>>()
    };
    // The lent records beneath the group's records: the one store's
    // conveyor's (the fixture's scratch lies apart).
    let lent = match placer {
        Placer::Bank => WHOLE.lent,
        Placer::Tier => 0,
    };
    for layer in 0..LAYERS {
        eventually(&format!("layer {layer}'s share was never held"), || share(layer).iter().all(|&expert| map(layer).held(expert).is_some()));
    }
    assert_eq!(tops(), (0..LAYERS).map(|layer| share(layer).to_vec()).collect::<Vec<_>>(), "each layer's share is its decode reads");
    // Each layer's share where it lies outside the lent records.
    let shared = (0..LAYERS).map(|layer| share(layer).into_iter().filter(|&expert| map(layer).held(expert).unwrap() >= lent).collect::<Vec<_>>()).collect::<Vec<_>>();
    assert!(shared.iter().any(|share| !share.is_empty()), "some share lies outside the lent records");
    let step = harness.prompt_step();
    let record = step.group_reads().unwrap();
    let every = (0..EXPERTS).collect::<Vec<_>>();
    for layer in 0..LAYERS {
        harness.scratch(&step, layer, &every);
        harness.commit_seeds(&step, layer);
        for expert in 0..EXPERTS {
            let choices = if share(layer).contains(&expert) { 3 } else { 1 };
            harness.reads[layer].store(record * 2 * EXPERTS + expert, choices);
        }
    }
    step.complete().unwrap();
    match placer {
        Placer::Bank => {
            // The first decode step after the group asks each layer's share
            // where it lay outside the lent records: the store serves every
            // one, none requested.
            let next = harness.step();
            for (layer, share) in shared.iter().enumerate() {
                assert!(!harness.ask_at(&next, layer, share), "layer {layer} lost its share {share:?}: {:?}", held(layer));
            }
            next.complete().unwrap();
        }
        Placer::Tier => eventually("the bank never held exactly every layer's share", || (0..LAYERS).map(held).collect::<Vec<_>>() == tops()),
    }
}

/// A prompt group routing every expert of every layer into an empty bank
/// beside a host tier leaves each layer at its share, wherever it falls in
/// the pass: the first layer's seeds take no more free records than its
/// share, so the last layer's seeds find theirs. With no promotion to
/// rebalance after it, the bank holds each layer's share as decode begins.
#[test]
fn a_prompt_group_grows_no_layer_past_its_share() {
    const LAYERS: usize = 4;
    let harness = lending("share-cap", EXPERTS, 2 * LAYERS as u64, LAYERS, 8, WHOLE, (EXPERTS * LAYERS) as u32 + 8, Placer::Tier, false);
    let step = harness.prompt_step();
    let record = step.group_reads().unwrap();
    let every = (0..EXPERTS).collect::<Vec<_>>();
    let mut seeded = Vec::new();
    for layer in 0..LAYERS {
        harness.scratch(&step, layer, &every);
        seeded.push(harness.commit_seeds(&step, layer).len());
        for expert in 0..EXPERTS {
            harness.reads[layer].store(record * 2 * EXPERTS + expert, 1);
        }
    }
    step.complete().unwrap();
    assert_eq!(seeded, vec![2; LAYERS], "each layer seeds its share");
    let held = |layer: usize| (0..EXPERTS).filter(|&expert| harness.layers[layer].0.held(expert).is_some()).count();
    eventually("a layer never held its seeds", || (0..LAYERS).all(|layer| held(layer) == 2));
}

/// A windowed scratch tier (`window_slots < EXPERTS`, `conveyor_records == ROUND_RECORDS`)
/// streams all 7 cold experts of an 8-expert layer (plus 1 bank hit) across 4
/// windows of 2 scratch slots and 2 conveyor records without deadlock — even
/// when step 0 routes higher-window cold experts `[4, 5, 6, 7]` before step 1
/// publishes window-0/1 cold experts `[0, 2, 3]` — writes each window's exact
/// pass routing table (`0..window_slots`, `-2` hits at window 0, `-1` outside),
/// seeds across windows, and resets `windows[0]` to `-2` once past the layer.
#[test]
fn a_windowed_scratch_tier_streams_more_cold_experts_than_slots_without_deadlock() {
    let conveying = Conveying::of(ROUND_RECORDS);
    let harness = lending_windowed(
        "windowed-conveyor",
        EXPERTS,
        4,
        1,
        STAGING,
        conveying,
        0,
        Placer::Tier,
        false,
        0,
        false,
        ROUND_RECORDS as usize,
    );
    let window_row = |w: usize| (0..EXPERTS).map(|e| harness.windows[0][w].load(e)).collect::<Vec<_>>();
    assert_eq!(window_row(0), [-2; EXPERTS]);
    for w in 1..4 {
        assert_eq!(window_row(w), [-1; EXPERTS]);
    }

    // Seed expert 1 into the bank via a one-step windowed prompt group.
    let seed_step = harness.prompt_step();
    harness.publish(&seed_step, 0, &[1]);
    let mut initial_seeds = Vec::new();
    for w in 0..harness.window_count {
        let converted = harness.rounds_window(&seed_step, 0, w);
        assert_eq!(converted, if w == 0 { vec![1] } else { vec![] });
        for (slot, record) in harness.commit_seeds(&seed_step, 0) {
            initial_seeds.push((w as i32 * harness.window_slots as i32 + slot, record));
        }
    }
    assert_eq!(initial_seeds.len(), 1);
    assert_eq!(initial_seeds[0].0, 1);
    seed_step.complete().unwrap();
    harness.retired();
    eventually("expert 1 was never mapped", || harness.layer.0.held(1).is_some());
    assert_eq!(window_row(0), [-2; EXPERTS], "windows[0] reset for resident passes");

    // Two-step group: step 0 publishes window 2..4 cold experts [4, 5, 6, 7] + hit 1
    // before step 1 publishes window 0..2 cold experts [0, 2, 3] + shared 6.
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    harness.publish(&steps[0], 0, &[1, 4, 5, 6, 7]);
    // Wait until step 0's window-2 cold experts hold both conveyor records, so
    // step 1's window-0 expert must preempt by `(window, step, round)` order.
    harness.conveying(4);
    harness.conveying(5);
    harness.publish(&steps[1], 0, &[0, 2, 3, 6]);

    let mut all_converted = Vec::new();
    let mut multi_seeds = Vec::new();
    let expected_rows = [
        [0, -2, -1, -1, -1, -1, -1, -1],
        [-1, -1, 0, 1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, 1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, 1],
    ];
    for w in 0..harness.window_count {
        for (place, step) in steps.iter().enumerate() {
            let converted = harness.rounds_window(step, 0, w);
            all_converted.extend(converted);
            if place + 1 == steps.len() {
                for (slot, record) in harness.commit_seeds(step, 0) {
                    multi_seeds.push((w as i32 * harness.window_slots as i32 + slot, record));
                }
            }
        }
        assert_eq!(window_row(w as usize), expected_rows[w as usize], "window {w} pass map row");
    }
    all_converted.sort_unstable();
    // Some cold experts may have landed into the host tier's 0 extra records? `tier = 0`,
    // `lent = 2` (claimed by the conveyor), so 0 records are free in the host tier and
    // every one of the 7 cold experts converts through the 2 conveyor records!
    assert_eq!(all_converted, [0, 2, 3, 4, 5, 6, 7]);
    // Bank has 4 records (1 held expert 1 + 3 free records, with expert 6 routed twice).
    assert_eq!(multi_seeds.len(), 3, "three remaining bank records seeded across windows: {multi_seeds:?}");
    for step in steps {
        step.complete().unwrap();
    }
    harness.retired();
    let group = harness.watcher.shared().costs().group.unwrap();
    let layer = &group.layers[0];
    assert_eq!((layer.union, layer.hits, layer.conveyed, layer.seeded, layer.computed), (8, 1, 7, 3, 2));
    eventually("windowed seeds were never mapped", || {
        multi_seeds.iter().all(|&(expert, record)| harness.layer.0.held(expert as usize) == Some(record as usize))
    });
    assert_eq!(window_row(0), [-2; EXPERTS], "windows[0] restored after windowed group");
    for w in 1..4 {
        assert_eq!(window_row(w), [-1; EXPERTS], "windows[{w}] cleared after windowed group");
    }
}

/// `Scratch::land_conveyed` is gated on free conveyor records strictly
/// exceeding `knee_records = source.knee().div_ceil(expert_bytes)`: when
/// conveyor records are scarce (equal to or below the knee), cold experts
/// convert through the conveyor without spawning background CPU repacks into
/// the host tier; when free conveyor records strictly exceed `knee_records`
/// and the tier covers every expert, cold experts land into the host tier in
/// the background.
#[test]
fn land_conveyed_is_gated_on_measured_knee_records() {
    // Scarce conveyor (2 records): free records can never strictly exceed
    // `knee_records` (>= 1 on any volume), so no host-tier landing is spawned.
    let scarce = lending("knee-scarce", EXPERTS, 2, 1, STAGING, Conveying::of(ROUND_RECORDS), TIER, Placer::Tier, false);
    let step = scarce.prompt_step();
    let mut converted = scarce.converted(&step, 0, &[0, 1]).concat();
    converted.sort_unstable();
    assert_eq!(converted, [0, 1]);
    step.complete().unwrap();
    scarce.retired();
    assert_eq!(scarce.host.maps[0].load(0), -1, "scarce conveyor must not spawn background tier landing for expert 0");
    assert_eq!(scarce.host.maps[0].load(1), -1, "scarce conveyor must not spawn background tier landing for expert 1");
}

/// With `windows > 1` and multiple pass sets (`pass_launches = 1` over a
/// 2-step group -> sets `[0..1, 1..2]`), `Scratch::publish` ingests step 1's
/// request into `sets[1]` immediately and pipelines its window-0 NVMe reads
/// into free conveyor records while `open_set` is still 0 (before step 0's
/// rounds even start). Shared experts between `sets[0]` and `sets[1]` wait
/// for `sets[0]` to release them, then re-read cleanly for `sets[1]`, and
/// `windows` tables clear when `open_set` advances to `1`.
#[test]
fn windowed_pass_sets_pipeline_next_set_reads_while_open_set_runs() {
    let conveying = Conveying::of(4);
    let harness = lending_windowed_passes(
        "windowed-pass-sets",
        EXPERTS,
        4,
        1,
        STAGING,
        conveying,
        0,
        Placer::Tier,
        false,
        0,
        false,
        ROUND_RECORDS as usize,
        1,
    );
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    // Publish set 0 (place 0) with [0, 4] and set 1 (place 1) with [0, 1]
    // before step 0 runs any round.
    harness.publish(&steps[0], 0, &[0, 4]);
    harness.publish(&steps[1], 0, &[0, 1]);
    // Set 1's window-0 expert 1 must pipeline into a free conveyor record
    // immediately while open_set is still 0!
    harness.conveying(1);

    let window_row = |w: usize| (0..EXPERTS).map(|e| harness.windows[0][w].load(e)).collect::<Vec<_>>();

    // Drain set 0 across all 4 windows.
    let mut set0_converted = Vec::new();
    for w in 0..harness.window_count {
        set0_converted.extend(harness.rounds_window(&steps[0], 0, w));
    }
    set0_converted.sort_unstable();
    assert_eq!(set0_converted, [0, 4]);
    assert_eq!(window_row(2), [-1, -1, -1, -1, 0, -1, -1, -1], "set 0 populated window 2");

    // Drain set 1 across all 4 windows: window 0 clears set 0's window tables
    // and converts both shared expert 0 and pre-pipelined expert 1.
    let mut set1_converted = Vec::new();
    for w in 0..harness.window_count {
        set1_converted.extend(harness.rounds_window(&steps[1], 0, w));
        if w == 0 {
            assert_eq!(window_row(2), [-1; EXPERTS], "clear_windows ran when open_set advanced to 1");
            assert_eq!(window_row(0), [0, 1, -1, -1, -1, -1, -1, -1], "set 1 populated window 0");
        }
    }
    set1_converted.sort_unstable();
    assert_eq!(set1_converted, [0, 1]);

    for step in steps {
        step.complete().unwrap();
    }
    harness.retired();
}

/// Once `fix_rounds` commits a cold expert to a conveyor round
/// (`ColdStage::Scheduled`), the request's tier copy list is already published
/// and only the conveyor can deliver it: even if the expert subsequently lands
/// in (or begins landing into) the host tier (`tier_serves == true`),
/// `Scratch::waits` must return false so `convey()` still reads it and
/// `Scratch::awaited` must not report it as awaiting a host-tier landing.
#[test]
fn scheduled_cold_experts_are_conveyed_even_after_landing_in_host_tier() {
    let harness = lending("sched-after-tier", EXPERTS, 2, 1, STAGING, Conveying::of(ROUND_RECORDS), TIER, Placer::Tier, false);
    let step = harness.prompt_step();
    let seq = step.sequence();
    harness.publish(&step, 0, &[0, 1, 2, 3]);
    // Request round 0 so `fix_rounds` schedules all 4 cold experts across 2
    // rounds of 2; round 1's two experts remain `Scheduled(1)` + `Deferred`.
    let scratch = &harness.layers[0].2;
    scratch.request(seq, round(0));
    assert_eq!(until_answered(scratch, seq, round(0)), Answer::Answered);
    let round0 = Harness::experts(&harness.round_listed(&step, 0, 0));
    assert_eq!(round0.len(), 2);
    let deferred: Vec<usize> = (0..4).filter(|e| !round0.contains(&(*e as i32))).collect();
    assert_eq!(deferred.len(), 2, "two experts must remain scheduled for round 1: round0={round0:?}");
    let (e_kept, e_landing) = (deferred[0], deferred[1]);

    // Put `e_kept` whole into the host tier and start an in-flight host-tier
    // landing for `e_landing`, so `tier_serves` becomes true for both.
    let landing_record = {
        let mut state = residency::lock(&harness.watcher.state);
        let crate::experts::host_records::Claim::Taken(kept_grant) = state.core.directory.claim_landing(
            0,
            e_kept,
            0,
            crate::experts::host_records::Taking::Out,
            crate::experts::host_records::Refuses,
        ) else {
            panic!("host tier must have a ready record for e_kept={e_kept}");
        };
        let kept_rec = crate::experts::directory::HostRecord(kept_grant.record());
        state.core.directory.landed(0, kept_rec, e_kept, None);

        let crate::experts::host_records::Claim::Taken(landing_grant) = state.core.directory.claim_landing(
            0,
            e_landing,
            0,
            crate::experts::host_records::Taking::Out,
            crate::experts::host_records::Refuses,
        ) else {
            panic!("host tier must have a ready record for e_landing={e_landing}");
        };
        let landing_rec = crate::experts::directory::HostRecord(landing_grant.record());

        assert!(state.core.directory.tier_serves(0, e_kept, seq));
        assert!(state.core.directory.tier_serves(0, e_landing, seq));
        assert!(state.core.directory.lands(0, e_landing));
        assert!(
            state.scratch.awaited(&state.core).next().is_none(),
            "scheduled cold experts must not be reported as awaited"
        );
        landing_rec
    };

    // Request round 1: frees round 0's conveyor records and must convey both
    // `e_kept` and `e_landing` instead of skipping them via `waits()`.
    scratch.request(seq, round(1));
    assert_eq!(until_answered(scratch, seq, round(1)), Answer::Answered);
    let round1 = Harness::experts(&harness.round_listed(&step, 0, 1));
    let mut expected = vec![e_kept as i32, e_landing as i32];
    expected.sort_unstable();
    assert_eq!(round1, expected, "round 1 must convey both scheduled experts despite host-tier presence");

    // Arrive past round 1 to release its conveyor records, clean up the
    // synthetic in-flight landing, and retire the step.
    scratch.request(seq, round(2));
    assert_eq!(until_answered(scratch, seq, round(2)), Answer::Answered);
    {
        let mut state = residency::lock(&harness.watcher.state);
        state.core.directory.unlanded(0, landing_record);
    }
    step.complete().unwrap();
    harness.retired();
}

/// When an expert is already in the host tier (`tier_serves == true`) before
/// `fix_rounds`, but another cold expert borrows its scratch slot
/// (`borrower(expert % lw).is_some()`), `fix_rounds` cannot tier-copy it and
/// instead schedules it in a conveyor round while still `ColdRead::Deferred`.
/// `convey()` must still read that scheduled expert rather than skipping it
/// via `waits()` and deadlocking the round.
#[test]
fn host_tier_expert_whose_slot_was_borrowed_before_fix_rounds_still_conveys() {
    let conveying = Conveying::of(ROUND_RECORDS).lending_scratch();
    let harness = lending("sched-borrow-tier", EXPERTS, 2, 1, STAGING, conveying, TIER, Placer::Tier, false);
    // Keep [1, 3, 5, 6, 7] in the host tier (and absent from the bank) so they
    // have `tier_serves == true`, while [0, 2, 4] must be read from the file.
    // With all `EXPERTS` routed, no slot is unrouted, so 0 borrows slot 1, 2
    // borrows slot 3, and 4 borrows slot 5 — forcing host-tier experts 1, 3, 5
    // into scheduled conveyor rounds while still `ColdRead::Deferred`.
    {
        let mut state = residency::lock(&harness.watcher.state);
        for expert in [1, 3, 5, 6, 7] {
            let crate::experts::host_records::Claim::Taken(grant) = state.core.directory.claim_landing(
                0,
                expert,
                0,
                crate::experts::host_records::Taking::Out,
                crate::experts::host_records::Refuses,
            ) else {
                panic!("host tier must have a ready record for expert {expert}");
            };
            let record = crate::experts::directory::HostRecord(grant.record());
            state.core.directory.landed(0, record, expert, None);
        }
    }

    let step = harness.prompt_step();
    let every: Vec<usize> = (0..EXPERTS).collect();
    harness.publish(&step, 0, &every);
    // Wait until 0, 2, and 4 have been read into lent scratch slots before
    // `fix_rounds` runs.
    eventually("experts 0, 2, 4 never borrowed scratch slots", || {
        let state = residency::lock(&harness.watcher.state);
        state.scratch.conveyor.whole(0, 0)
            && state.scratch.conveyor.whole(0, 2)
            && state.scratch.conveyor.whole(0, 4)
    });

    let rounds = harness.rounds_checked(&step, 0);
    let converted = Harness::experts(&rounds.concat());
    let lent = Harness::from_slots(&rounds.concat());
    assert!(!lent.is_empty(), "experts 0, 2, 4 must borrow scratch slots: rounds={rounds:?}");
    for &(expert, slot) in &lent {
        assert!(
            converted.contains(&slot),
            "host-tier expert {slot} whose slot was borrowed by {expert} must be conveyed by a round: rounds={rounds:?}"
        );
    }
    step.complete().unwrap();
    harness.retired();
}

/// When a borrower and its slot's owner end up in the same round,
/// `reclaim_slots` yields the `Held` borrower back to `ColdRead::Deferred`.
/// `Scratch::round` must immediately re-run `convey()` so the yielded borrower
/// is re-read into a conveyor record instead of stranding the round with no
/// in-flight reads to wake the watcher.
#[test]
fn reclaim_slots_yielded_borrower_in_same_round_re_conveys_immediately() {
    let conveying = Conveying::of(ROUND_RECORDS).lending_scratch();
    let harness = lending("reclaim-reconvey", EXPERTS, 2, 1, STAGING, conveying, TIER, Placer::Tier, false);
    // Keep 1..EXPERTS in the host tier so only expert 0 reads before
    // `fix_rounds`, borrowing slot 1. `fix_rounds` then puts [0, 1] into a
    // single round 0, forcing `reclaim_slots` to yield 0 and re-convey both.
    {
        let mut state = residency::lock(&harness.watcher.state);
        for expert in 1..EXPERTS {
            let crate::experts::host_records::Claim::Taken(grant) = state.core.directory.claim_landing(
                0,
                expert,
                0,
                crate::experts::host_records::Taking::Out,
                crate::experts::host_records::Refuses,
            ) else {
                panic!("host tier must have a ready record for expert {expert}");
            };
            let record = crate::experts::directory::HostRecord(grant.record());
            state.core.directory.landed(0, record, expert, None);
        }
    }

    let step = harness.prompt_step();
    let every: Vec<usize> = (0..EXPERTS).collect();
    harness.publish(&step, 0, &every);
    eventually("expert 0 never borrowed slot 1", || {
        let state = residency::lock(&harness.watcher.state);
        state.scratch.conveyor.whole(0, 0) && state.scratch.conveyor.borrower(1) == Some((0, 0))
    });

    let rounds = harness.rounds_checked(&step, 0);
    let converted = Harness::experts(&rounds.concat());
    assert_eq!(converted, [0, 1], "both reclaimed borrower 0 and host-tier slot owner 1 must convey: rounds={rounds:?}");
    step.complete().unwrap();
    harness.retired();
}
