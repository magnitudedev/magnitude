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
        let sequence = low_word(step.sequence());
        let scratch = &harness.layer.2;
        let (_, lists, _) = &harness.load;
        let (mut converted, mut rounds, mut index) = (Vec::new(), 1, 0);
        while index < rounds {
            scratch.request(step.sequence(), round(index));
            assert_eq!(until_answered(scratch, step.sequence(), round(index)), Answer::Answered);
            if index == 0 {
                rounds = (scratch.load(step.sequence(), constants::SCRATCH_ASK, constants::SCRATCH_STAGED) as u64)
                    .div_ceil(ROUND_RECORDS);
            }
            // Until the device arrives at the next round, this round's
            // records hold what its convert reads.
            let list = &lists[index as usize];
            assert_eq!(list.load(constants::LIST_SEQUENCE as usize), sequence);
            converted.extend(harness.from_conveyor(&listed(list)));
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

/// A reader that holds its tensor's host access from its first read until
/// released: what keeps a transfer into the tensor from being issued, as a
/// device's queue full behind a parked step keeps one waiting in the driver.
struct Holding {
    held: Option<std::sync::mpsc::Sender<()>>,
    release: std::sync::mpsc::Receiver<()>,
}

impl std::io::Read for Holding {
    fn read(&mut self, into: &mut [u8]) -> std::io::Result<usize> {
        if let Some(held) = self.held.take() {
            held.send(()).unwrap();
            self.release.recv().unwrap();
        }
        into.fill(0);
        Ok(into.len())
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
        std::thread::spawn(move || slots.write_from_reader(&mut Holding { held: Some(held), release: released }).unwrap())
    };
    holding.recv().unwrap();
    let decode = harness.step();
    let prompt = harness.prompt_step();
    let scratch = &harness.layer.2;
    harness.publish(&prompt, 0, &[5]);
    scratch.request(prompt.sequence(), round(0));
    // Within a bound, so a service stuck behind a transfer is reported, not
    // hung on.
    let within = |check: &dyn Fn() -> bool| {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !check() && Instant::now() < deadline {
            std::thread::yield_now();
        }
        check()
    };
    let answered = within(&|| scratch.answer_of(prompt.sequence(), round(0)) == Some(Answer::Answered));
    let absent = (0..EXPERTS).find(|&expert| expert != 5 && harness.in_tier(0, expert).is_none()).unwrap();
    assert!(harness.ask_at(&decode, 0, &[absent]));
    let served = within(&|| harness.verdict(0, decode.sequence()).is_some());
    let tiered = listed(&harness.tiered);
    release.send(()).unwrap();
    writer.join().unwrap();
    assert!(answered, "round 0 is answered while no transfer can be issued");
    assert!(served, "the decode request is served meanwhile");
    assert_eq!(harness.verdict(0, decode.sequence()), Some(Answer::Answered));
    assert_eq!(harness.tiered.load(constants::LIST_SEQUENCE as usize), low_word(prompt.sequence()));
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
    filled.extend(super::tests::listed(&harness.tiered).into_iter().map(|(_, expert)| expert));
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
    assert_eq!(harness.from_conveyor(&harness.rounds(&steps[0], 0).concat()), [4, 5]);
    assert_eq!(harness.from_conveyor(&harness.rounds(&steps[1], 0).concat()), [0, 1]);
    for step in steps {
        step.complete().unwrap();
    }
}

/// A group's steps request a layer before any round of it there, each from
/// its own chain place (every step's front there precedes its first back): each
/// request is served in step order once its own first round arrives, and every
/// request of the group had been published when the first was.
#[test]
fn a_groups_requests_of_a_layer_are_served_in_order_once_their_rounds_arrive() {
    let harness = prompted("fronts", EXPERTS, 4);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    let scratch = &harness.layer.2;
    for expert in 0..EXPERTS {
        harness.load.0.store(expert, i32::from(expert < 2));
        harness.second.store(expert, i32::from((1..3).contains(&expert)));
    }
    for step in &steps {
        scratch.request(step.sequence(), constants::SCRATCH_ASK);
    }
    // The later step's first round arrives first: neither is served until the
    // earlier step's does.
    scratch.request(steps[1].sequence(), round(0));
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
        let harness = lending(name, EXPERTS, 12, 2, STAGING, conveying, TIER, Placer::Tier, false);
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
        // Each read lands into the host tier too, its bytes the file's (the
        // staging beyond it too short for any fill): no later miss reads it
        // again. The tier's reserve of 4 counts the lent records the group
        // holds, which it returns ready, so the 5 stay: none is unmapped
        // for records a group only borrows.
        for (layer, expert) in [(0, 0), (0, 1), (0, 3), (1, 3), (1, 4)] {
            eventually("a prompt's read never landed into the tier", || harness.in_tier(layer, expert).is_some());
        }
        // A decode step after them runs residency's refill as its requests
        // arrive (`Residency::advance`): the tier serves all five, none
        // unmapped for the reserve.
        let next = harness.step();
        assert!(!harness.ask_at(&next, 0, &[0, 1, 3]), "the tier serves layer 0's");
        assert!(!harness.ask_at(&next, 1, &[3, 4]), "the tier serves layer 1's");
        next.complete().unwrap();
        for (layer, expert) in [(0, 0), (0, 1), (0, 3), (1, 3), (1, 4)] {
            assert!(harness.in_tier(layer, expert).is_some(), "the reserve unmapped {expert} of layer {layer} for the lent records");
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
    let seeds = &harness.load.2;
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

/// Once the device is past a group, the experts its conveyor holds whole
/// that the store does not hold are copied into staging beyond the demand
/// reserve, and none of them is read again: with the file emptied once the
/// group's reads ended, a later request of all four is served each from where
/// it now is, landed into a record the group returned or still staged, with
/// the file's bytes. Four layers of a store holding a layer's scratch records
/// beside its landing reserve (`ExpertBankPlan::prompt_records`), its fill
/// done beyond the reserve with each layer's lowest experts, before a group
/// routes layer 0's highest four. Staging is sized past what the fill can
/// hold when the group ends (what the records returned admit), so the copy
/// always finds its room. The seed commit is not played, so the store keeps
/// none.
#[test]
fn a_groups_conveyed_experts_stay_staged_past_it() {
    const LAYERS: usize = 4;
    let harness = lending("conveyed", EXPERTS, 7, LAYERS, 8, WHOLE, EXPERTS as u32 + 8, Placer::Bank, false);
    let kept = || harness.layers.iter().map(|layer| (0..EXPERTS).filter(|&expert| layer.0.held(expert).is_some()).count()).sum::<usize>();
    eventually("the fill never filled beyond the reserve", || kept() == 16);
    // The experts the lent records hold, which the group unmaps.
    let lent = (0..LAYERS)
        .map(|layer| (0..EXPERTS).filter(|&expert| harness.layers[layer].0.held(expert).is_some_and(|record| record < WHOLE.lent)).count())
        .sum::<usize>();
    let step = harness.prompt_step();
    let converted = harness.scratch(&step, 0, &[4, 5, 6, 7]).concat();
    assert_eq!(harness.from_conveyor(&converted), [4, 5, 6, 7]);
    for layer in 1..LAYERS {
        harness.scratch(&step, layer, &[]);
    }
    // The group holds the conveyor's and the scratch's lent records, which
    // the reserve counts, so nothing else is unmapped and no fill lands:
    // none holds a staging record as the device passes the group.
    eventually("the store never settled under the group", || kept() == 16 - lent);
    std::fs::OpenOptions::new().write(true).open(&harness.file).unwrap().set_len(0).unwrap();
    step.complete().unwrap();
    let request = harness.request(&[4, 5, 6, 7]);
    for layer in 1..LAYERS {
        harness.route_at(&request, layer, &[]);
    }
    request.complete().unwrap();
    let map = &harness.layers[0].0;
    eventually("the group's experts never landed", || (4..8).all(|expert| map.held(expert).is_some()));
    for expert in 4..8 {
        let record = map.held(expert).unwrap() as i32;
        assert_eq!([0, 1, 2].map(|kind| harness.tiered(kind, record)), [expert as u8, 100 + expert as u8, 200 + expert as u8]);
    }
}

/// A group's conveyed experts are staged the most reused first while
/// staging has room beyond the demand reserve: four layers of a store, six
/// staging records and a reserve of four, so room for two (one where a fill
/// read holds a record as the device passes the group), its fill done with
/// layer 0's experts 0 to 4 among them, before a group routes layer 0's 4
/// to 7. 4 the store holds (staging may keep a copy of it from the fill);
/// 5, 6 and 7, read alike, rank by expert, so 5 stays staged, 6 only beside
/// it, and 7 lies past the room. The group's four are its own demand reads;
/// a request of 5 and 7 then takes 5 back staged and reads 7 on demand.
#[test]
fn a_groups_staging_keeps_the_most_reused_and_one_past_its_room_reads_on_demand() {
    const LAYERS: usize = 4;
    let harness = lending("staged-room", EXPERTS, 7, LAYERS, 6, WHOLE, EXPERTS as u32 + 8, Placer::Bank, false);
    let map = &harness.layers[0].0;
    let kept = || harness.layers.iter().map(|layer| (0..EXPERTS).filter(|&expert| layer.0.held(expert).is_some()).count()).sum::<usize>();
    eventually("the fill never filled beyond the reserve", || kept() == 18 && (0..5).all(|expert| map.held(expert).is_some()));
    let step = harness.prompt_step();
    harness.scratch(&step, 0, &[4, 5, 6, 7]);
    for layer in 1..LAYERS {
        harness.scratch(&step, layer, &[]);
    }
    step.complete().unwrap();
    eventually("the group's experts were never staged", || harness.watcher.staged().contains(&(0, 5)));
    let staged = harness.watcher.staged();
    assert!(!staged.contains(&(0, 7)), "7 lies past the room: {staged:?}");
    let costs = || {
        let layer = harness.watcher.shared().costs().layers[0];
        (layer.staged, layer.unanticipated)
    };
    assert_eq!(costs(), (0, 4), "the group's four are its demand reads");
    let request = harness.step();
    harness.request_at(&request, 0, &[5, 7]);
    for layer in 1..LAYERS {
        harness.route_at(&request, layer, &[]);
    }
    request.complete().unwrap();
    assert_eq!(costs(), (1, 5), "5 staged, 7 read");
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
    assert_eq!(harness.commit_seeds(&step).len(), 2, "layer 0 seeds 3 and 4");
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

/// Beside a host tier a prompt group's scratch lies in its lender's bottom
/// bank records: the group claims them, each expert they held leaving the
/// bank and unmapped from the group's first step; no seed takes one (the
/// bank otherwise holds only shares, so the group seeds nothing); once the
/// device is past the group they return free, so a shrink after the group
/// keeps them and the next group claims them again.
#[test]
fn a_discrete_groups_scratch_claims_its_bank_records_and_returns_them_free() {
    let harness = lending_bank("bank-lent", EXPERTS, 4, 1, STAGING, WHOLE, TIER, Placer::Tier, true, 2, false);
    let map = &harness.layer.0;
    let lent = |record: u32| harness.watcher.bank_lent(0, record);
    // Promotions take the free records in order: 0 to 3 records 0 to 3,
    // every one in the layer's share of four.
    harness.promoted(0, &[0, 1, 2, 3]);
    assert_eq!((0..4).map(|expert| map.held(expert)).collect::<Vec<_>>(), [Some(0), Some(1), Some(2), Some(3)]);
    let step = harness.prompt_step();
    let first = step.sequence();
    harness.scratch(&step, 0, &[5]);
    // The bank was settled, every record held: the claims took them at once.
    assert!(lent(0) && lent(1) && !lent(2) && !lent(3), "the group holds the lent records alone");
    for expert in [0, 1] {
        assert_eq!((map.held(expert), map.resolve(expert, first)), (None, None), "{expert} left the bank from the group's first step");
    }
    assert_eq!(harness.commit_seeds(&step), [], "no seed takes a lent record or a share's");
    // A step the load interrupts fails here; its seeds are none anyway.
    drop(step.complete());
    eventually("the lent records never returned free", || !lent(0) && !lent(1));
    let shared = harness.watcher.shared();
    eventually("the group's reads were never ranked", || shared.telemetry.fed_all());
    let plan = shared.plan_shrink(Candidates::one(3, vec![3]), None).unwrap();
    shared.commit_shrink(plan).unwrap();
    assert_eq!(shared.costs().records, [3]);
    let step = harness.prompt_step();
    harness.scratch(&step, 0, &[6]);
    // The group began once every promotion's copy into one had completed;
    // its lend drops one yet to copy.
    eventually("the next group was never lent the records again", || lent(0) && lent(1));
    harness.commit_seeds(&step);
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
        // Layer 1's decode reads of 6 and 7 take its share, promoted beside
        // 5, which lies outside it: one record is left free.
        harness.promoted(1, &[6, 7]);
        let telemetry = &harness.watcher.shared().telemetry;
        eventually("the decode reads were never ranked", || telemetry.fed_all() && !telemetry.in_top()(1, 5));
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
        // Step 1, the group's last at layer 0, seeds two of experts 0 to 3,
        // its share: into the free record, and the one outside every share,
        // evicting layer 1's 5.
        harness.rounds(&steps[1], 0);
        assert_eq!(listed(&harness.load.2).len(), 2);
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
        // The tier keeps 5, which its seeding group read: the demoted hit
        // is copied from it whole, and only 4 is converted.
        assert_eq!(harness.from_conveyor(&harness.rounds(&steps[0], 1).concat()), [4]);
        assert!((0..3).all(|kind| slot(kind) == [(100 * kind + 5) as u8; RECORD as usize]), "the demoted hit is copied from the tier");
        // Step 1's 3, cold (6 is banked), is read ahead as its request was
        // taken, into the conveyor and on into the tier. Its round converts
        // it from the conveyor, or where that landing ended and the conveyor
        // let it go before the device arrived, its tier copy reads it: the
        // one race here is which, and either fills its slot once.
        let converted = harness.from_conveyor(&harness.rounds(&steps[1], 1).concat());
        let tiered = listed(&harness.tiered).iter().any(|&(_, expert)| expert == 3);
        assert!(converted == [3] || (converted.is_empty() && tiered), "converted {converted:?}, tier copied {tiered}");
        for step in steps {
            step.complete().unwrap();
        }
    }
}

/// Layers sharing any pool share a conveyor form: a Q4_K_M model's
/// layers share gate and up pools while their down pools alternate.
#[test]
fn layers_sharing_a_pool_share_conveyor_records() {
    assert_eq!(conveyor_forms(4, &[vec![0, 1, 2], vec![0, 1, 3], vec![0, 1, 2]]), [0, 0, 0]);
    assert_eq!(conveyor_forms(6, &[vec![0, 1, 2], vec![3, 4, 5], vec![0, 1, 2]]), [0, 1, 0]);
    // Joined through a third layer.
    assert_eq!(conveyor_forms(5, &[vec![0, 1, 2], vec![3, 1, 4], vec![3, 3, 4]]), [0, 0, 0]);
    // Up-only records: two routed tensors each.
    assert_eq!(conveyor_forms(3, &[vec![0, 1], vec![2, 1]]), [0, 0]);
}

/// The size rule: the fewest steps whose compute covers every layer's
/// stage, within the capacity; to capacity until every layer's compute is
/// measured and its lines priced.
#[test]
fn a_group_is_the_fewest_steps_covering_each_layers_stage() {
    let line = crate::assessment::ReadLine::new(0.001, 1e-9).unwrap();
    let lines = vec![crate::assessment::ExpertLines::reference(1_000_000); 2];
    let ms = Duration::from_millis;
    let mut facts = GroupFacts {
        compute: vec![Some(ms(10)), None],
        held: vec![56, 0],
        alone: vec![0; 2],
        read: vec![None; 2],
        pulled: vec![None; 2],
        unions: vec![None; 2],
        bytes: vec![1_000_000; 2],
        resident: None,
        experts: vec![256; 2],
    };
    assert_eq!(facts.group_size(Some(&lines), Some(line), Notice::Written, 16), 16, "layer 1 is unmeasured");
    facts.compute[1] = Some(ms(40));
    assert_eq!(facts.group_size(None, Some(line), Notice::Written, 16), 16, "the lines are unpriced");
    // Layer 0: 1 ms + 200 MB at 1 GB/s, then 200 moves of 1 MB records,
    // 44.56 µs each, 209.912 ms over 10 ms: 21 steps; layer 1: 268.407 ms over
    // 40 ms: 7.
    assert_eq!(facts.group_size(Some(&lines), Some(line), Notice::Written, 32), 21);
    assert_eq!(facts.group_size(Some(&lines), Some(line), Notice::Written, 16), 16);
    facts.held[0] = 256;
    assert_eq!(facts.group_size(Some(&lines), Some(line), Notice::Written, 16), 7);
}

/// A layer's file experts are every expert it lacks until a group has
/// ended there, then what that group read there. Where prompt groups pull
/// from a host tier, the experts it holds alone are pulled instead until a
/// group has ended there, then what that group pulled.
#[test]
fn a_layers_read_is_what_the_last_group_to_end_there_read() {
    let mut facts = GroupFacts {
        compute: vec![Some(Duration::from_millis(10)); 2],
        held: vec![56, 0],
        alone: vec![0; 2],
        read: vec![None; 2],
        pulled: vec![None; 2],
        unions: vec![None; 2],
        bytes: vec![1_000_000; 2],
        resident: None,
        experts: vec![256; 2],
    };
    let moved = |facts: &GroupFacts| {
        let terms = facts.terms().unwrap();
        terms.iter().map(|layer| (layer.filed, layer.tiered)).collect::<Vec<_>>()
    };
    assert_eq!(moved(&facts), [(200, 0), (256, 0)]);
    facts.read[0] = Some(3);
    assert_eq!(moved(&facts), [(3, 0), (256, 0)]);
    facts.alone = vec![10, 6];
    assert_eq!(moved(&facts), [(3, 0), (256, 0)], "without pulls the tier's experts are read");
    facts.resident = Some(vec![2_000_000; 2]);
    assert_eq!(moved(&facts), [(3, 10), (250, 6)]);
    facts.pulled[1] = Some(4);
    assert_eq!(moved(&facts)[1], (250, 4));
}

/// Unpriced, nothing weighs a merge set: a layer with nothing cold after
/// it merges whole, however long the run goes unpriced, and only a layer
/// before cold experts splits off its first back. The last layer is
/// followed by nothing.
#[test]
fn an_unpriced_group_splits_only_before_cold_experts() {
    let mut facts = GroupFacts {
        compute: vec![None; 4],
        held: vec![256; 4],
        alone: vec![0; 4],
        read: vec![None; 4],
        pulled: vec![None; 4],
        unions: vec![None; 4],
        bytes: vec![1_000_000; 4],
        resident: Some(vec![2_000_000; 4]),
        experts: vec![256; 4],
    };
    assert_eq!(facts.merge_sets(None, None, None, Notice::Written, 6), [6, 6, 6, 6], "a group with nothing cold");
    facts.pulled[2] = Some(52);
    assert_eq!(facts.merge_sets(None, None, None, Notice::Written, 6), [6, 5, 6, 6], "the pulls before layer 2");
    facts.pulled[2] = Some(0);
    facts.held[3] = 200;
    assert_eq!(facts.merge_sets(None, None, None, Notice::Written, 6), [6, 6, 5, 6], "the reads before layer 3");
    assert_eq!(facts.merge_sets(None, None, None, Notice::Written, 1), [1; 4]);
}

/// A group is whole only while every layer of it holds every expert: a
/// layer holding every expert beside a partial one keeps its whole word
/// clear; once both hold every expert, both words are set. The words are
/// published between flights.
#[test]
fn a_group_is_whole_only_while_every_layer_holds_every_expert() {
    // The tier keeps every expert either layer lands, beyond its reserve.
    let harness = lending("whole-group", EXPERTS, 2 * EXPERTS as u64, 2, STAGING, DECODE, STAGING + 2 * EXPERTS as u32, Placer::Tier, true);
    let shared = harness.watcher.shared();
    let whole = |layer: usize| harness.layers[layer].5.load(constants::POLICY_WHOLE as usize);
    harness.promoted(0, &(0..EXPERTS).collect::<Vec<_>>());
    harness.promoted(1, &(0..6).collect::<Vec<_>>());
    assert_eq!(shared.holdings().0, [EXPERTS, 6]);
    // A gap publishes the words after it retired the fills.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!((whole(0), whole(1)), (0, 0), "a full layer beside a partial one is not whole");
    harness.promoted(1, &[6, 7]);
    until(&harness.layers[0].5, 0, 1);
    until(&harness.layers[1].5, 0, 1);
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
    let conveying = Conveying { conveyed: 2 * ROUND_RECORDS, lent: 2 * ROUND_RECORDS as usize, bytes: RECORD };
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
    let conveying = Conveying { conveyed: 4, lent: 4, bytes: RECORD };
    let harness = lending("lender", EXPERTS, 12, 1, STAGING, conveying, 0, Placer::Tier, false);
    harness.request(&[0, 1]).complete().unwrap();
    // Residency maps a landing into the tier as it learns it, which may
    // follow the answer.
    harness.tiering(0, &[0, 1]);
    let step = harness.prompt_step();
    harness.publish(&step, 0, &[2]);
    eventually("the lent records were never claimed", || harness.in_tier(0, 0).is_none() && harness.in_tier(0, 1).is_none());
    harness.conveying(2);
    harness.rounds(&step, 0);
    harness.commit_seeds(&step);
    step.complete().unwrap();
    harness.request(&[3]).complete().unwrap();
    assert!(harness.in_tier(0, 3).is_some(), "the returned records take a landing");
}

/// A bank imported whole into the one store keeps every expert where a
/// pass reads it in place, its lender's lent records among them: none can
/// miss, so every layer's whole word is set from the bank's open, and a
/// prompt group is resident, not scratch-served (`ExpertBank::resident`).
/// It requests nothing, so it claims no lent record and evicts nothing:
/// the store still keeps every expert, none read from storage again.
#[test]
fn a_whole_banks_prompt_group_reads_nothing_and_claims_no_lent_record() {
    let conveying = Conveying { conveyed: 4, lent: 4, bytes: RECORD };
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

/// On the one store a fill's landing is mapped past the steps already begun,
/// so a running group's step misses an expert the store keeps by then. No
/// tier copy binds there: an expert beyond the conveyor's free records when
/// the rounds are fixed stays the rounds' own, read from the file, and the
/// tier list, whose bound there admits no record, stays empty.
#[test]
fn the_one_stores_rounds_read_a_cold_expert_the_store_keeps() {
    let _alone = alone();
    let ahead = Ahead::queue("store-kept-ahead", 512);
    let conveying = Conveying::of(2);
    let harness = lending("store-kept", EXPERTS, EXPERTS as u64, 1, 2 * STAGING, conveying, EXPERTS as u32, Placer::Bank, false);
    let step = harness.prompt_step();
    let map = &harness.layers[0].0;
    ahead.finish();
    let kept = [5, 6, 7];
    eventually("the fill never landed the experts", || kept.iter().all(|&expert| map.held(expert).is_some()));
    assert!(kept.iter().all(|&expert| map.resolve(expert, step.sequence()).is_none()), "landed past the begun step");
    let rounds = harness.scratch(&step, 0, &kept);
    let mut converted = rounds.concat().iter().map(|&(_, expert)| expert as usize).collect::<Vec<_>>();
    converted.sort_unstable();
    assert_eq!(converted, kept, "the rounds convert every one");
    assert_eq!(harness.from_conveyor(rounds.last().unwrap()).len(), 1, "the last, read into the conveyor once a record freed");
    assert_eq!(listed(&harness.tiered), [], "no tier copy on the one store");
    step.complete().unwrap();
}

/// A resident prompt group requests nothing; its passes' grouping writes
/// its reads into its group reads record (`BankPart::GroupReads`). Once
/// every step of it has ended the stack holds them, by layer at the
/// group's last step, and a decode step after it still feeds the stack.
#[test]
fn a_resident_prompt_groups_reads_reach_the_stack() {
    let harness = layered("group-reads", EXPERTS, 12, 2, STAGING);
    let steps = begin(harness.watcher.shared(), &[StepRoute::Grouped; 2], false).unwrap();
    let record = steps[0].group_reads().unwrap();
    assert_eq!(steps[1].group_reads(), Some(record), "one record a group");
    let last = steps[1].sequence();
    // Play the passes' counts: layer 0's single set at place 0, then its
    // merged set at place 1; layer 1's one set at place 0.
    let count = |layer: usize, place: usize, expert: usize, choices: i32| {
        harness.reads[layer].store((record * 2 + place) * EXPERTS + expert, choices);
    };
    count(0, 0, 1, 3);
    count(0, 1, 4, 2);
    count(1, 0, 6, 1);
    let read = harness.watcher.shared().telemetry.last_read();
    assert_eq!(read(0, 1), None, "fed only once the group ends");
    drop(read);
    for step in steps {
        step.complete().unwrap();
    }
    let read = harness.watcher.shared().telemetry.last_read();
    assert_eq!([read(0, 1), read(0, 4), read(1, 6)], [Some(last); 3]);
    assert_eq!([read(0, 6), read(1, 1)], [None, None], "only what each layer grouped");
    drop(read);
    // A decode step after the group feeds its routing records.
    let decode = harness.step();
    for (layer, experts) in [(0, [2]), (1, [5])] {
        harness.request_at(&decode, layer, &experts);
    }
    let sequence = decode.sequence();
    decode.complete().unwrap();
    let read = harness.watcher.shared().telemetry.last_read();
    assert_eq!([read(0, 2), read(1, 5), read(0, 1)], [Some(sequence), Some(sequence), Some(last)]);
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

/// A prompt group routing every expert of every layer leaves each layer at
/// its ranked share as decode starts: the experts the ranking's share of it
/// names once the group's reads are logged (`Reuse::in_top`). On the one
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
        harness.commit_seeds(&step);
        for expert in 0..EXPERTS {
            harness.reads[layer].store(record * 2 * EXPERTS + expert, 1);
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
        seeded.push(harness.commit_seeds(&step).len());
        for expert in 0..EXPERTS {
            harness.reads[layer].store(record * 2 * EXPERTS + expert, 1);
        }
    }
    step.complete().unwrap();
    assert_eq!(seeded, vec![2; LAYERS], "each layer seeds its share");
    let held = |layer: usize| (0..EXPERTS).filter(|&expert| harness.layers[layer].0.held(expert).is_some()).count();
    eventually("a layer never held its seeds", || (0..LAYERS).all(|layer| held(layer) == 2));
}
