use super::*;

/// Residency's jobs wait on no request: while a prompt step's round 0 has
/// not arrived, a decode step's miss lands into the tier, and once that
/// step completed the ranking's promotion of it is carried (its copy's
/// completion an event) and mapped; the prompt step is then served whole.
#[test]
fn a_promotion_is_carried_while_a_scratch_request_awaits_its_arrival() {
    let harness = lending("promote-beside", EXPERTS, 2, 1, STAGING, WHOLE, TIER, Placer::Tier, true);
    let map = &harness.layer.0;
    let decode = harness.step();
    let prompt = harness.prompt_step();
    harness.publish(&prompt, 0, &[1, 6]);
    harness.conveying(1);
    harness.conveying(6);
    harness.request_at(&decode, 0, &[5]);
    decode.complete().unwrap();
    eventually("5 was never promoted", || map.held(5).is_some());
    assert_eq!(harness.layer.2.answer_of(prompt.sequence(), round(0)), None, "the round awaits its arrival");
    assert_eq!(harness.from_conveyor(&harness.rounds(&prompt, 0).concat()), [1, 6]);
    prompt.complete().unwrap();
}

/// A group's hit stays a hit for the group's steps when a promotion evicts
/// it meanwhile: the promotion unmaps it only from the first step not yet
/// begun, so the group's pass reads it where the bank held it and nothing
/// fills its scratch slot. The hit weighs its landing read and the group's
/// (one read, at the latest step logged), the want two reads, the later at
/// that step and the earlier after the hit's landing: the want outweighs
/// the hit at every half-life. The group's last step therefore never seeds
/// it, though the bank no longer holds it: its seed copy would copy the
/// slot, which holds no byte of it, and map that record as the expert.
#[test]
fn a_hit_a_promotion_evicts_during_its_group_is_never_seeded() {
    let harness = promoting("seed-hit", 2, TIER);
    let map = &harness.layer.0;
    harness.promoted(0, &[5, 6]);
    let five = map.held(5).expect("5 is banked");
    // Two decode steps reading 7, begun before the group: their reads are
    // ranked once they end, which every step before them has.
    let sevens = [harness.request(&[7]), harness.request(&[7])];
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    // Step 0 routes 5, a hit: read where the bank holds it.
    harness.publish(&steps[0], 0, &[5]);
    harness.rounds(&steps[0], 0);
    // 7 is promoted over 5, which it outweighs, while the group's steps may
    // still read 5's record.
    for seven in sevens {
        seven.complete().unwrap();
    }
    eventually("5 was never evicted", || map.held(5).is_none());
    assert_eq!(map.resolve(5, steps[1].sequence()), Some(five), "the group's steps still hold 5");
    // Step 1, the group's last, routes nothing new and seeds: 5, which the
    // bank no longer holds, is the group's only candidate.
    harness.publish(&steps[1], 0, &[]);
    harness.rounds(&steps[1], 0);
    let seeds = harness.commit_seeds(&steps[1], 0);
    assert!(seeds.iter().all(|&(expert, _)| expert != 5), "5, a hit no round filled, is seeded: {seeds:?}");
    for step in steps {
        step.complete().unwrap();
    }
}

/// A prompt group reserves its lent bank records as its steps begin, while
/// promotions with victims in them are planned: whether a promotion copied
/// before the group began (the group's claim then evicts it) or was still
/// draining (it then never copies, its victim unmapped from a step before
/// the group's first), no hit lies in a lent record, every routed expert's
/// scratch slot holds the file's bytes, and no promoted expert lies in a
/// lent record while the group holds them. Experts 0 and 1 hold the lent
/// records 0 and 1, 2 and 3 records 2 and 3; decode reads of 2, 3, 6 and 7
/// rank 6 and 7 into the layer's share of four, so their promotions take
/// 0's and 1's records, and the group begins at once, routing 0 and 5.
/// Repeated, the group begins on either side of the promotions' copies.
#[test]
fn a_group_beginning_beside_promotions_into_its_lent_records_reads_no_hit_there() {
    for round in 0..8 {
        let harness = lending_bank(&format!("lent-promotion-{round}"), EXPERTS, 4, 1, STAGING, WHOLE, TIER, Placer::Tier, true, 2, false);
        let map = &harness.layer.0;
        let lent = |record: u32| harness.watcher.bank_lent(0, record);
        harness.promoted(0, &[0, 1]);
        harness.promoted(0, &[2, 3]);
        assert_eq!((0..4).map(|expert| map.held(expert)).collect::<Vec<_>>(), [Some(0), Some(1), Some(2), Some(3)]);
        for _ in 0..3 {
            let step = harness.step();
            harness.request_at(&step, 0, &[2, 3, 6, 7]);
            step.complete().unwrap();
        }
        let step = harness.prompt_step();
        let first = step.sequence();
        let converted = harness.scratch(&step, 0, &[0, 5]).concat();
        assert!(lent(0) && lent(1));
        // Each converted expert's conveyor record holds the file's bytes.
        let converted = harness.from_conveyor(&converted);
        assert!((0..EXPERTS).all(|expert| map.resolve(expert, first).is_none_or(|record| record >= 2)), "a hit lies in a lent record");
        assert!((0..EXPERTS).all(|expert| map.held(expert).is_none_or(|record| record >= 2)), "a promoted expert lies in a lent record");
        // Each routed expert is read whole: converted, a hit outside the
        // lent records, or copied from the tier into its slot.
        for expert in [0usize, 5] {
            if converted.contains(&(expert as i32)) || map.resolve(expert, first).is_some() {
                continue;
            }
            for kind in 0..3 {
                let mut bytes = [0u8; RECORD as usize];
                unsafe { harness.scratch[kind].read_shared(expert as u64 * RECORD, &mut bytes) }.unwrap();
                assert_eq!(bytes, [(100 * kind + expert) as u8; RECORD as usize], "expert {expert}'s slot of tensor {kind}");
            }
        }
        harness.commit_seeds(&step, 0);
        drop(step.complete());
    }
}

/// After a prompt group's union read of every expert, decode's reads
/// decide what the bank keeps: once decode has read its working set over
/// enough steps, the kept experts are the ranking's most reused, whatever
/// their indices, and no record (lent or not) keeps an expert the ranking
/// ranks below one the bank lacks while no prompt group holds it. On the
/// one store, seven records, the bottom two lent, of a layer of eight
/// experts; beside a host tier keeping every expert, a bank of four
/// records the promotions fill.
#[test]
fn after_a_prefill_decode_keeps_the_rankings_top_share() {
    let conveying = Conveying::of(2);
    decode_keeps_the_rankings_top_share(&lending("freeze", EXPERTS, EXPERTS as u64, 1, STAGING, conveying, 5, Placer::Bank, false));
    decode_keeps_the_rankings_top_share(&lending("freeze-tier", EXPERTS, 4, 1, STAGING, WHOLE, TIER, Placer::Tier, true));
}

/// A prompt group reads every expert of `harness`'s one layer, expert 3
/// its most chosen, then 24 decode steps read 5, 6 and 7: the ranking puts
/// 3 first of the experts only the prompt read, and what the layer's map
/// holds comes to be decode's working set and the ranking's most reused of
/// the rest, none held ranked below one lacked.
fn decode_keeps_the_rankings_top_share(harness: &Harness) {
    let step = harness.prompt_step();
    // The group's pass counts its choices (`BankPart::GroupReads`), at its
    // one place.
    let record = step.group_reads().expect("a prompt step's group reads record");
    for expert in 0..EXPERTS {
        harness.reads[0].store(record * 2 * EXPERTS + expert, if expert == 3 { 9 } else { 1 });
    }
    drop(harness.scratch(&step, 0, &(0..EXPERTS).collect::<Vec<_>>()));
    harness.commit_seeds(&step, 0);
    step.complete().unwrap();
    for _ in 0..24 {
        harness.request(&[5, 6, 7]).complete().unwrap();
    }
    let kept = || (0..EXPERTS).filter(|&expert| harness.layer.0.held(expert).is_some()).collect::<Vec<_>>();
    let depths = || {
        let state = residency::lock(&harness.watcher.state);
        let ranking = state.core.directory.ranking();
        let reuse = crate::experts::reuse::Reuse::of(&ranking);
        (0..EXPERTS).map(|expert| reuse.depth(0, expert)).collect::<Vec<_>>()
    };
    let ranked = || {
        let (kept, depths) = (kept(), depths());
        let lacked = (0..EXPERTS).filter(|expert| !kept.contains(expert));
        let deepest = kept.iter().map(|&expert| depths[expert]).max().unwrap_or(0);
        let prompt_only = [0, 1, 2, 4].iter().all(|&expert| depths[3] < depths[expert]);
        prompt_only && [5, 6, 7].iter().all(|expert| kept.contains(expert)) && lacked.into_iter().all(|expert| depths[expert] >= deepest)
    };
    eventually("the kept are the ranking's most reused", ranked);
}

/// The host promotes what the ranking keeps, off every step's path: a step
/// that read an expert from the tier completes, and the expert is copied
/// into a free bank record and mapped from the first step not yet begun;
/// the next step's route finds it in the bank. Once the bank is full, an
/// expert read after the others earns a copy over the least weighed of
/// them, and the release takes the least ranked: one read least and
/// longest ago, the least at every half-life, so it is both whichever
/// measure ranks. That read is by a step that ends while a later one runs:
/// the evicted entry is unmapped from the first step not yet begun, while a step begun before
/// still reads it and its record keeps its bytes until that step has
/// completed; then the copy takes the record and its expert is mapped.
/// Then a reserve pays, as the reserve's model prices it: 6 and 7 were
/// each read again from the top of the stack, and the second record never
/// hit, so holding it free costs no hit and saves the drain a promotion
/// would wait; once the decode steps reading 7 have sized it, the bank
/// releases 6 into the reserve and holds 7 alone.
#[test]
fn a_promotion_reuses_its_record_only_once_every_step_that_could_read_it_completed() {
    let harness = promoting("promote", 2, TIER);
    let shared = harness.watcher.shared();
    let map = &harness.layer.0;
    let step = |sequence: &ExpertStep| sequence.sequence();
    // 5, then 6, read from the tier, each promoted into a free record.
    let landing = harness.request(&[5]);
    let landed = step(&landing);
    landing.complete().unwrap();
    eventually("5 was never promoted", || map.held(5).is_some());
    let five = map.held(5).unwrap();
    assert_eq!(map.resolve(5, landed), None, "mapped from the first step not yet begun");
    assert_eq!([0, 1, 2].map(|kind| harness.banked(kind, five)), [5, 105, 205]);
    harness.request(&[6]).complete().unwrap();
    eventually("6 was never promoted", || map.held(6).is_some());
    let next = harness.step();
    assert_eq!(harness.route_at(&next, 0, &[6]), [-1 - 6], "the bank holds 6");
    next.complete().unwrap();
    // 7, read after both, outweighs 5 over any stay (one read each, 5's
    // older; the copy is priced at nothing), and 5, read once before 6's
    // two reads, is the least ranked at every half-life: 7 evicts it, while
    // a step begun before the eviction still runs and reads 5's record:
    // begun before 7's read ends, so before the ranking learns it.
    let seven = harness.step();
    let running = harness.step();
    harness.request_at(&seven, 0, &[7]);
    seven.complete().unwrap();
    eventually("5 was never unmapped", || map.held(5).is_none());
    assert_eq!(map.resolve(5, step(&running)), Some(five), "the running step still holds 5");
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!([0, 1, 2].map(|kind| harness.banked(kind, five)), [5, 105, 205], "kept while the step runs");
    assert_eq!(map.held(7), None);
    let after = step(&running) + 1;
    harness.route_at(&running, 0, &[]);
    running.complete().unwrap();
    eventually("7 was never promoted", || map.held(7) == Some(five));
    assert_eq!([0, 1, 2].map(|kind| harness.banked(kind, five)), [7, 107, 207]);
    assert_eq!((map.resolve(5, after), map.resolve(7, after)), (None, Some(five)));
    // Each decode step reading 7 sizes the reserve again; past the change's
    // hysteresis it holds the one record the model prices as paying,
    // whichever turns the steps' ends fell in.
    for _ in 0..4 {
        let next = harness.step();
        harness.route_at(&next, 0, &[7]);
        next.complete().unwrap();
    }
    let reserved = || residency::lock(&harness.watcher.state).core.directory.reserved(0);
    eventually("the reserve never took the record its model prices as paying", || reserved() == 1 && shared.holdings().0[0] == 1);
    assert_eq!((map.held(6), map.held(7)), (None, Some(five)), "6 released into the reserve, 7 kept");
}

/// The promotions the ranking asks together copy together: every one
/// drained by then in one batch of the transfer queue. At one record a
/// batch the bank took the ranking's share at one queue round trip a
/// record, and on an RTX 3090 (48 of 256 experts) that held decode at 246.7
/// tier misses a step against the ranking's 44: the ranking asked ~40
/// promotions a step, and its replay misses what the run did only with
/// the copies capped at 15-20 a step.
#[test]
fn the_promotions_asked_together_copy_in_one_batch() {
    let harness = promoting("one-batch", 4, TIER);
    let map = &harness.layer.0;
    let experts = [2, 3, 5, 6];
    harness.request(&experts).complete().unwrap();
    eventually("every expert read was never promoted", || experts.iter().all(|&expert| map.held(expert).is_some()));
    assert_eq!(harness.copies.posted(), 1, "one batch: {}", harness.copies.describe());
    for expert in experts {
        let record = map.held(expert).unwrap();
        assert_eq!([0, 1, 2].map(|kind| harness.banked(kind, record)), [0, 100, 200].map(|base| (base + expert) as u8));
    }
}
