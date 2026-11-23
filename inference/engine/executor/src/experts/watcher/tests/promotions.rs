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
/// fills its scratch slot. Decode reads 5 and 6 at every step after they
/// land, then 6 and 7: 7, read as 5 and 6 were at their first reads, which
/// every later step read again, is expected at nearly every step of the
/// turnover, and outweighs 5, whose one expected use is the group's read. The group's last step therefore never seeds it,
/// though the bank no longer holds it: its seed copy would copy the slot,
/// which holds no byte of it, and map that record as the expert.
#[test]
fn a_hit_a_promotion_evicts_during_its_group_is_never_seeded() {
    let harness = promoting("seed-hit", 2, TIER);
    let map = &harness.layer.0;
    harness.promoted(0, &[5, 6]);
    let five = map.held(5).expect("5 is banked");
    for _ in 0..8 {
        harness.request(&[5, 6]).complete().unwrap();
    }
    // Two decode steps reading 6 and 7, begun before the group: their
    // reads are ranked once they end, which every step before them has.
    let sevens = [harness.request(&[6, 7]), harness.request(&[6, 7])];
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

/// No promotion takes a lent bank record, the scratch's: decode's reads
/// promote into the records past them alone, however the ranking moves
/// while a prompt group begins, so the group's lease takes the lent records
/// at once, no hit lies in one, every routed expert's scratch slot holds
/// the file's bytes, and no promoted expert lies in a lent record. Experts
/// 2 and 3 hold records 2 and 3, the pool of two; decode reads of 6 and 7
/// rank them above 2 and 3, so their promotions take those records as the
/// group begins, routing 0 and 5. Repeated, the group begins on either side
/// of the promotions' copies.
#[test]
fn promotions_beside_a_group_beginning_take_no_lent_record() {
    for round in 0..8 {
        let harness = lending_bank(&format!("lent-promotion-{round}"), EXPERTS, 4, 1, STAGING, WHOLE, TIER, Placer::Tier, true, 2, false);
        let map = &harness.layer.0;
        let lent = |record: u32| harness.watcher.bank_lent(0, record);
        harness.promoted(0, &[2, 3]);
        assert_eq!((0..4).map(|expert| map.held(expert)).collect::<Vec<_>>(), [None, None, Some(2), Some(3)]);
        for _ in 0..3 {
            let step = harness.step();
            harness.request_at(&step, 0, &[6, 7]);
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
/// every expert only the prompt read below decode's, and what the layer's
/// map holds comes to be decode's working set and the ranking's most
/// reused of the rest, none held ranked below one lacked.
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
        let prompt_only = [0, 1, 2, 3, 4].iter().all(|&expert| [5, 6, 7].iter().all(|&decoded| depths[decoded] < depths[expert]));
        prompt_only && [5, 6, 7].iter().all(|expert| kept.contains(expert)) && lacked.into_iter().all(|expert| depths[expert] >= deepest)
    };
    eventually("the kept are the ranking's most reused", ranked);
}

/// The host promotes what the ranking keeps, off every step's path: a step
/// that read an expert from the tier completes, and the expert is copied
/// into a free bank record and mapped from the first step not yet begun;
/// the next steps' routes find them in the bank. Once the bank is full, an
/// expert read after the others earns a copy over the least weighed of
/// them, and the release takes the least ranked: the one no read since
/// expects, so it is both. That read is by a step that ends while a later
/// one runs:
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
    // 5, then 6, read from the tier, each promoted into a free record, then
    // 6 read from the bank past `DEPTHS` so the turnover horizon spans into
    // `history` (ranking 6 well above 5) and the ranking learns an expert
    // just read repeats at the next step, while the second record never hits.
    let landing = harness.request(&[5]);
    let landed = step(&landing);
    landing.complete().unwrap();
    eventually("5 was never promoted", || map.held(5).is_some());
    let five = map.held(5).unwrap();
    assert_eq!(map.resolve(5, landed), None, "mapped from the first step not yet begun");
    assert_eq!([0, 1, 2].map(|kind| harness.banked(kind, five)), [5, 105, 205]);
    harness.request(&[6]).complete().unwrap();
    eventually("6 was never promoted", || map.held(6).is_some());
    for _ in 0..12 {
        let next = harness.step();
        assert_eq!(harness.route_at(&next, 0, &[6]), [-1 - 6], "the bank holds 6");
        next.complete().unwrap();
    }
    // 7, read alone as 6 was before its repeats, is expected again from the
    // top of the stack while 5, unread since, is not (the copy is priced at
    // nothing): 7 evicts 5, while a step begun before the eviction still
    // runs and reads 5's record: begun before 7's read ends, so before the
    // ranking learns it.
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

/// The promotions the ranking asks together copy one batch each, in due
/// order, as many in flight as the transfer queue's measured depth keeps
/// the link full (`TierCopyCosts::depth`, one beside the latency a record's
/// copy hides): no record waits a round trip for the one before it. The
/// harness prices no latency, so each posts as the one before it lands.
#[test]
fn the_promotions_asked_together_copy_each_as_its_own_batch() {
    let harness = promoting("one-batch", 4, TIER);
    let map = &harness.layer.0;
    let experts = [2, 3, 5, 6];
    harness.request(&experts).complete().unwrap();
    eventually("every expert read was never promoted", || experts.iter().all(|&expert| map.held(expert).is_some()));
    assert_eq!(harness.copies.posted(), experts.len() as u64, "each its own batch: {}", harness.copies.describe());
    for expert in experts {
        let record = map.held(expert).unwrap();
        assert_eq!([0, 1, 2].map(|kind| harness.banked(kind, record)), [0, 100, 200].map(|base| (base + expert) as u8));
    }
}

/// Mandatory 5-way integrated invariant test for Zero-Recopy Demand-Landing
/// Promotion (all 6 `astra@magnitop` invariants on CPU backend):
/// 1. Flight 1 holds victim `V` in slot 1; release at step 2 transitions slot 1
///    to `Record::Releasing` (stamped unmap) with zero reuse before retirement.
/// 2. After Flight 1 ends and `drain()` frees slot 1, Flight 2 lands `E` into
///    slot 1 (`Record::DemandLanding`, unpublished in `StampedMap`) with 1 H2D.
/// 3. Flight 3 joins slot 1 (`demand_joins == 1`, `transfers_issued == 1`).
/// 4. Flight 2 cancels its `ConsumerLease` while Flight 3 still pins slot 1,
///    and a concurrent `shrink` runs: shrink evicts ONLY unpinned slot 0
///    (never pinned slot 1) and bytes for Flight 3 remain exact.
/// 5. On completion, slot 1 is O(1)-promoted `Record::DemandLanding -> Record::Held((0, E))`
///    with `map.held(E) == Some(1)`, `transfers_issued == 1` (zero second H2D),
///    exact bytes, and empty `deferred_syncs` after drain.
#[test]
fn zero_recopy_demand_landing_five_way_invariant_test() {
    use crate::experts::directory::{BankRecord, DemandSlotClaim, Directory, HostRecord, HostTier, Placement, Point, RoutedRead};
    use crate::experts::host_records::{Claim, Refuses, Taking};
    use crate::experts::map::StampedMap;
    use crate::experts::reuse::{Evidence, Reuse};
    use crate::experts::transport::{DestSlot, ExpertTransport};
    use crate::experts::Words;
    use magnitude_kernels::constants;
    use seismic::{BackendName, Charges, Element, Tensor, Transfer, TransferItem, TransferRun};

    let device = crate::test_support::open_backend(
        &seismic::Devices::discover(&[]).expect("discover devices"),
        BackendName::Cpu,
    )
    .expect("CPU device");
    let charges = Charges::tally(&device);
    let transfer = Transfer::new(&device).expect("CPU transfer");

    const EXPERTS: usize = 8;
    const U: usize = 0; // Unpinned expert in slot 0
    const V: usize = 1; // Victim expert in slot 1 (held by Flight 1)
    const E: usize = 2; // New demand-landed expert (lands into slot 1)

    let bank_map_tensor = Tensor::shared(&charges, Element::i32(), &[EXPERTS as u64, constants::MAP_WORDS]).unwrap();
    let bank_map = StampedMap::new(Words::new(bank_map_tensor, EXPERTS as u64 * constants::MAP_WORDS, 0).unwrap());
    let map = bank_map.view().clone();

    let tier_map_tensor = Tensor::shared(&charges, Element::i32(), &[EXPERTS as u64]).unwrap();
    let tier_map = crate::experts::map::LandingMap::Tier(Words::new(tier_map_tensor, EXPERTS as u64, 0).unwrap());

    let mut directory = Directory::new(
        &[2], // 2 unified VRAM slots: slot 0 and slot 1
        &[0], // 1 layer in group 0
        &[EXPERTS],
        HostTier {
            records: vec![4],
            lent: vec![0],
            reserve: 0,
            placement: Placement::Tier {
                bank_maps: vec![bank_map],
                tier_maps: vec![tier_map],
                lent: (0, 0),
                stage_slots: vec![0],
            },
        },
    );
    directory.price_copies(0, Some(0.0));

    // Land U (tier_record 0), V (tier_record 1), E (tier_record 2) into host tier.
    let mut tier_records = [0u32; 3];
    for (idx, expert) in [U, V, E].into_iter().enumerate() {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, Refuses) else {
            panic!("host tier must have room for expert {expert}");
        };
        let tr = grant.record();
        tier_records[idx] = tr;
        directory.landed(0, HostRecord(tr), expert, None);
    }
    let tier_e = tier_records[2] as usize;

    // Physical host & bank byte buffers (64 bytes per expert) to verify exact bytes.
    let mut host_e = Tensor::shared(&charges, Element::bool(), &[64]).unwrap();
    host_e.write_from_host(&[0xEEu8; 64]).unwrap();
    let mut bank_slot0 = Tensor::shared(&charges, Element::bool(), &[64]).unwrap();
    bank_slot0.write_from_host(&[0xAAu8; 64]).unwrap();
    let mut bank_slot1 = Tensor::shared(&charges, Element::bool(), &[64]).unwrap();
    bank_slot1.write_from_host(&[0xBBu8; 64]).unwrap();

    let steps = Steps::new(
        crate::planning::FlightDepth::PIPELINED,
        1,
        crate::experts::exchange::Pulse::of(BackendName::Cpu),
    );
    let transport = ExpertTransport::new_registry_only();

    // Seed initial permanent VRAM: slot 0 holds U, slot 1 holds V.
    let ranking = directory.ranking();
    {
        let time = directory.time();
        time.lock().register(0, crate::experts::horizon::Advance::Known(crate::experts::horizon::Tokens::new(1.0)));
    }
    Reuse::of(&ranking).read(&RoutedRead { sequence: 0, layer: 0, choices: 2, experts: vec![U, V] }, None, Evidence::Decode);
    let asked = steps.unbegun(|at| directory.promotions(at, 2));
    assert_eq!(asked.promotions.len(), 2);
    let mut completed = Vec::new();
    for (group, record, (layer, expert)) in directory.queued() {
        let tier = directory.hold_tier(layer, expert).expect("tier kept");
        directory.copy(group, record, tier, 1);
        completed.push((group, record));
    }
    directory.promoted(completed, Point::whole(0, 0, 0), &steps);
    let slot_v = directory.held(0, V).expect("V held in VRAM");
    let slot_u = directory.held(0, U).expect("U held in VRAM");
    assert_ne!(slot_u, slot_v);

    // 1. Flight 1 (step 1) begins and reads V from slot_v.
    let flight1 = steps.begin_group(1, None);
    assert_eq!(map.resolve(V, flight1), Some(slot_v));

    // Release V at step 2 while Flight 1 is still running -> Record::Releasing.
    let released = steps.unbegun(|at| {
        directory.replenish_demand_free(0, 1, at, |slot| {
            // Protect U so V is deterministically chosen as the victim.
            slot == slot_v && transport.can_reuse_slot(DestSlot { group: 0, slot })
        })
    });
    assert_eq!(released, 1, "V must be released into Record::Releasing");
    assert_eq!(map.resolve(V, flight1), Some(slot_v), "Flight 1 must still see V");
    assert_eq!(map.resolve(V, flight1 + 1), None, "steps >= 2 must see V unmapped");

    // Zero reuse before Flight 1 retires!
    assert!(!directory.drain(Point::whole(flight1, 0, flight1), &steps), "slot_v must NOT drain while Flight 1 runs");
    assert_eq!(
        directory.claim_demand_landing(0, tier_e as u32, false, |slot| transport.can_reuse_slot(DestSlot { group: 0, slot })),
        DemandSlotClaim::Overflow { staged: Some((0, E)) },
        "no free slot available while Flight 1 holds Releasing slot_v"
    );

    // Flight 1 cancels/completes -> drain() frees slot_v!
    steps.end_played(flight1);
    assert!(directory.drain(Point::whole(flight1, 0, flight1), &steps), "slot_v must drain after Flight 1 ends");

    // 2. Flight 2 (step 2) claims freed slot_v for E and issues ONE H2D copy.
    let flight2 = steps.begin_group(1, None);
    let claim = directory.claim_demand_landing(0, tier_e as u32, false, |slot| {
        transport.can_reuse_slot(DestSlot { group: 0, slot })
    });
    assert_eq!(claim, DemandSlotClaim::Allocated { staged: Some((0, E)), slot: slot_v });
    let dest = DestSlot { group: 0, slot: slot_v };
    let (id, lease2) = transport.begin_demand_transfer(0, 0, E, tier_e, dest);
    let (_, sync) = transfer
        .copy_synced(&[TransferItem::Run(TransferRun {
            source: &host_e,
            source_offset: 0,
            destination: if slot_v == 1 { &bank_slot1 } else { &bank_slot0 },
            destination_offset: 0,
            bytes: 64,
        })])
        .expect("copy_synced E");
    transport.attach_sync(id, sync);
    assert_eq!(transport.transfers_issued(), 1, "exactly 1 H2D transfer issued");
    assert_eq!(map.held(E), None, "E must NOT be published in StampedMap while in flight");

    // 3. Flight 3 (step 3) joins slot_v (`demand_joins == 1`, zero second H2D!).
    let flight3 = steps.begin_group(1, None);
    let joined = transport
        .join_by_tier_record(0, 0, tier_e)
        .expect("Flight 3 must join in-flight demand transfer of E");
    assert_eq!(joined.dest, dest);
    assert_eq!(transport.demand_joins(), 1, "expected 1 demand join");
    assert_eq!(transport.transfers_issued(), 1, "joining must NOT issue a second H2D");

    // 4. Flight 2 cancels its ConsumerLease while Flight 3 still pins slot_v,
    //    and a concurrent shrink runs.
    lease2.release();
    steps.end_played(flight2);
    assert!(!transport.can_reuse_slot(dest), "Flight 3 still pins slot_v");

    let evictions = directory.evictions_filtered_with(
        0,
        1,
        |_, _| None,
        false,
        |slot| transport.can_reuse_slot(DestSlot { group: 0, slot }),
    );
    assert_eq!(evictions, vec![slot_u], "shrink must evict ONLY unpinned slot_u, NEVER pinned slot_v");
    let got_bytes = (if slot_v == 1 { &bank_slot1 } else { &bank_slot0 })
        .read_to_host()
        .expect("read bank slot");
    assert_eq!(got_bytes, vec![0xEEu8; 64], "Flight 3 must observe exact original bytes of E");

    // 5. Flight 3 completes -> O(1) zero-recopy promotion `DemandLanding -> Held((0, E))`!
    transport.complete_dma(id);
    joined.lease.release();
    transport.drain_deferred_syncs();
    assert_eq!(transport.deferred_syncs_len(), 0, "deferred_syncs must be empty after drain");

    // Feed E's decode read so Reuse ranks E positively.
    {
        let time = directory.time();
        time.lock().register(flight3, crate::experts::horizon::Advance::Known(crate::experts::horizon::Tokens::new(1.0)));
    }
    Reuse::of(&ranking).read(&RoutedRead { sequence: flight3, layer: 0, choices: 1, experts: vec![E] }, None, Evidence::Decode);

    let promoted = directory.finish_demand_landing(
        0,
        slot_v,
        Some((0, E)),
        Point::whole(flight3, 0, flight3),
        &steps,
        transport.can_reuse_slot(dest),
    );
    steps.end_played(flight3);
    assert!(promoted, "E must be O(1)-promoted in place with copy_hits = 0.0");
    assert_eq!(transport.transfers_issued(), 1, "O(1) promotion must issue ZERO second H2D");
    assert_eq!(directory.held(0, E), Some(slot_v), "slot_v must now be Record::Held((0, E))");
    assert_eq!(map.held(E), Some(slot_v), "StampedMap must publish E at slot_v");
    let got_after = (if slot_v == 1 { &bank_slot1 } else { &bank_slot0 })
        .read_to_host()
        .expect("read bank slot after promotion");
    assert_eq!(got_after, vec![0xEEu8; 64], "promoted slot must retain exact bytes");
    let _ = BankRecord(slot_v as u32);
}
