use super::*;

/// A refill after the last step ended graces its victims through a step the
/// device is past, which no later retirement of the frontier reaches until
/// the next step: they are ready at once.
#[test]
fn a_refill_past_the_last_step_leaves_no_grace() {
    let harness = layered("refill-past", EXPERTS, 12, 3, 4 + 4);
    harness.demand_step(4);
    harness.tier_ready();
    let reserve = harness.staging_records as usize;
    let mut state = residency::lock(&harness.watcher.state);
    let mut grants = Vec::new();
    for expert in 0..EXPERTS {
        if let crate::experts::host_records::Claim::Taken(grant) = state.core.directory.claim_landing(0, expert, 0) {
            grants.push(grant);
        }
        if state.core.directory.refill_short(reserve) {
            break;
        }
    }
    assert!(state.core.directory.refill_short(reserve), "the claims left the reserve short");
    state.refill_tier();
    assert!(!state.core.directory.refill_short(reserve), "the refill made the reserve");
    assert_eq!(state.core.directory.host(0).graced(), 0, "graced through a step the device is past");
}

/// A host tier with a record for every expert fills to every expert from
/// the bank's open on, before any step, each record's bytes its expert's
/// from the file: a step's route then names nothing absent, every choice
/// the bank lacks served from the tier, and the tier keeps them all, with
/// no reserve unmapped. No expert can then be absent: the next gap sets the
/// layer's resolved word, its group not whole, so its tier wait and the
/// wait's park are left out of every step.
#[test]
fn a_tier_of_every_expert_fills_before_any_step() {
    // Staging records beyond the demand reserve and a layer's hints, which
    // a fill reads into.
    let harness = lending("filled", EXPERTS, 2, 1, 3 * STAGING, DECODE, EXPERTS as u32, Placer::Tier, false);
    harness.tiering(0, &(0..EXPERTS).collect::<Vec<_>>());
    let step = harness.step();
    let sources = harness.route_at(&step, 0, &[1, 5, 7]);
    assert!(sources.iter().all(|&source| source < -1), "every choice served from the tier: {sources:?}");
    step.complete().unwrap();
    assert!((0..EXPERTS).all(|expert| harness.in_tier(0, expert).is_some()), "the tier keeps every expert");
    let policy = &harness.layers[0].5;
    until(policy, constants::POLICY_RESOLVED as usize, 1);
    assert_eq!(policy.load(constants::POLICY_WHOLE as usize), 0, "a bank of two records is not whole");
}

/// A host tier whose records keep every expert only with its lent records
/// leaves its layer unresolved: a prompt group's claim of a lent record
/// unmaps its expert mid-flight, so the decode step chained behind the group
/// finds it absent and requests the file, which a resolved layer's watcher
/// takes for an invariant violation and ends serving on ("step N requested
/// the file at layer L, whose experts cannot be absent").
#[test]
fn a_decode_step_behind_a_prompt_group_claiming_the_lent_records_is_served() {
    // Eight experts: six tier records beyond the conveyor's two lent ones.
    let harness = lending("lent-complete", EXPERTS, 2, 1, 3 * STAGING, Conveying::of(2), EXPERTS as u32 - 2, Placer::Tier, false);
    harness.tiering(0, &(0..EXPERTS).collect::<Vec<_>>());
    let shared = harness.watcher.shared();
    eventually("the tier never filled", || shared.tier_fill() == crate::TierFill::Filled);
    // The turn that filled it publishes the policy words next.
    std::thread::sleep(Duration::from_millis(50));
    let lent = (0..EXPERTS).filter(|&expert| harness.in_tier(0, expert).is_some_and(|record| record < 2)).collect::<Vec<_>>();
    assert_eq!(lent.len(), 2, "the fill kept two experts in the lent records");
    let steps = begin_as(&shared, &[(StepRoute::Grouped, LaunchWork::Prompt), (StepRoute::Decode, LaunchWork::Generation)], true).unwrap();
    harness.scratch(&steps[0], 0, &lent);
    eventually("the group never claimed the lent records", || lent.iter().all(|&expert| harness.in_tier(0, expert).is_none()));
    harness.request_at(&steps[1], 0, &lent);
    for step in steps {
        step.complete().unwrap();
    }
}

/// A host tier smaller than every expert fills from the bank's open on by
/// the one rule that fills a tier of every expert: its records beyond the
/// reserve its landings keep (a record for each staging record), with the
/// experts it lands first (none used yet: each layer's alike, by index),
/// and no more.
#[test]
fn a_partial_tier_fills_its_records_beyond_the_reserve() {
    // Two layers' 16 experts in one group, its tier of 15 records 3 beyond
    // the reserve of 12.
    let harness = lending("partial", EXPERTS, 2, 2, 3 * STAGING, DECODE, 15, Placer::Tier, false);
    harness.tiering(0, &[0, 1]);
    harness.tiering(1, &[0]);
    std::thread::sleep(Duration::from_millis(50));
    let kept = (0..2).flat_map(|layer| (0..EXPERTS).map(move |expert| (layer, expert)));
    assert_eq!(kept.filter(|&(layer, expert)| harness.in_tier(layer, expert).is_some()).count(), 3, "the reserve is the landings'");
    let resolved = |layer: usize| harness.layers[layer].5.load(constants::POLICY_RESOLVED as usize);
    assert_eq!((resolved(0), resolved(1)), (0, 0), "a tier short of every expert leaves its layers' experts absentable");
}

/// A host tier smaller than every expert fills while its fill's reads run,
/// queued behind others, and is filled once its records beyond the reserve
/// each keep an expert, though it lacks the rest; a miss's landing then
/// displaces what it keeps at the same records, and it stays filled.
#[test]
fn a_tier_is_filled_once_its_records_beyond_the_reserve_keep_experts() {
    let _alone = alone();
    let ahead = Ahead::queue("tier-fill-ahead", 64);
    // Two layers' sixteen experts beside fifteen tier records, eight of them
    // the reserve.
    let harness = layered("tier-fill", EXPERTS, 12, 2, 4 + 4);
    let shared = harness.watcher.shared();
    let staged = || {
        let state = residency::lock(&harness.watcher.state);
        state.prefetch.staging.free() < state.prefetch.staging.records()
    };
    eventually("the fill never read", staged);
    assert_eq!(shared.tier_fill(), crate::TierFill::Filling, "its reads wait");
    ahead.finish();
    eventually("the tier never filled", || shared.tier_fill() == crate::TierFill::Filled);
    let tiered = || (0..2).map(|layer| (0..EXPERTS).filter(|&expert| harness.in_tier(layer, expert).is_some()).count()).sum::<usize>();
    assert_eq!(tiered(), 15 - 8, "its records beyond the reserve, not every expert");
    harness.demand_step(7);
    harness.tiering(1, &[7]);
    assert_eq!(shared.tier_fill(), crate::TierFill::Filled, "a miss's landing leaves it filled");
}

/// The host keeps a reserve of ready tier records: once landings leave
/// fewer ready than the staging records, it unmaps its kept records, first
/// in taking order (held alone, the least recently read, then the lower
/// expert), until the ready and the unmapped make the reserve, and no
/// more. The watcher learns a landing, and keeps the reserve, as it serves
/// a running step, so it takes only from the landings that have settled:
/// the two that leave the reserve short land after two settled ones. A
/// later step's route finds the kept ones in the tier map and the tier
/// serves them, requesting nothing.
#[test]
fn the_tier_keeps_a_reserve_of_ready_records() {
    let harness = hosted("tier-reserve", 4, STAGING + 2);
    let map = &harness.host.maps[0];
    harness.request(&[0, 1]).complete().unwrap();
    eventually("the first landings never settled", || harness.in_tier(0, 0).is_some() && harness.in_tier(0, 1).is_some());
    harness.request(&[2, 3]).complete().unwrap();
    let step = harness.step();
    eventually("the reserve never unmapped a record", || map.load(0) == -1 && map.load(1) == -1);
    std::thread::sleep(Duration::from_millis(20));
    assert!([2, 3].iter().all(|&expert| harness.in_tier(0, expert).is_some()), "the reserve unmaps no more");
    assert!(!harness.ask_at(&step, 0, &[2, 3]), "the tier serves both");
    step.complete().unwrap();
    assert_eq!(harness.watcher.shared().costs().layers[0].hosted, 2);
}

/// Where the host places into the one store, every landing of an expert
/// (a demand's, a hint's, the fill's, one revived from staging) claims it
/// through the store's records (`HostRecords::claim`), so no two land it
/// at once and a demand that finds one in flight waits for it: a store
/// short of every expert, whose reserve unmaps what the fill and the misses
/// land while staging keeps their bytes, under a run of steps each missing
/// experts drawn across the layer, answers every miss and fails no step.
/// The interleavings are the threads'; the claim's rule itself is
/// `host_records`' `a_claim_takes_a_record_only_for_a_lacked_expert`.
#[test]
fn every_landing_of_an_expert_claims_it_once_in_the_one_store() {
    let harness = lending("claims", EXPERTS, 6, 1, STAGING, DECODE, 6, Placer::Bank, false);
    // A fixed generator, so a failure replays.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..400 {
        let mut experts = Vec::new();
        while experts.len() < 3 {
            let expert = (next() % EXPERTS as u64) as usize;
            if !experts.contains(&expert) {
                experts.push(expert);
            }
        }
        let step = harness.request(&experts);
        if let Some(verdict) = harness.verdict(0, step.sequence()) {
            assert_eq!(verdict, Answer::Answered, "a miss failed");
        }
        step.complete().unwrap();
    }
}

/// A request waiting while the keeper's turn runs takes residency between
/// two of its jobs (`Residency::tend`), and finds it as at a turn's end.
/// Under a traffic hold beside the idle load, the turn's refill takes
/// victims the tier kept alone (graced through steps the device is past,
/// so ready at once), and its promotions find the hold in force: the
/// request, taking the lock after them, reads holdings already published
/// without the victims. It begins a step while it holds the
/// lock, which ends the hold's effect: the turn's fill, after it, reads
/// the hold as it stands, so it reads ahead and leaves the traffic not
/// quiet, as the running step asks.
#[test]
fn a_request_between_a_turns_jobs_finds_residency_as_at_a_turns_end() {
    let harness = layered("turn-handoff", EXPERTS, 12, 3, 4 + 4);
    harness.demand_step(4);
    harness.tier_ready();
    let shared = harness.watcher.shared().clone();
    let state = harness.watcher.state.clone();
    let _hold = shared.hold_traffic();
    shared.wait_traffic_quiet().unwrap();
    let reserve = harness.staging_records as usize;
    // Claims that leave the reserve short, so the turn's refill graces
    // victims; kept until the turn ends.
    let grants = std::cell::RefCell::new(Vec::new());
    let alone = |state: &residency::Residency| (0..state.core.layers.len()).map(|layer| state.core.directory.alone(layer)).collect::<Vec<_>>();
    let claimed = std::cell::RefCell::new(Vec::new());
    let short = |state: &mut residency::Residency, _: &mut residency::Turn, _: Option<&super::super::promotions::Copier>, _: Option<&crate::experts::directory::MissCurve>| {
        for expert in 0..EXPERTS {
            if let crate::experts::host_records::Claim::Taken(grant) = state.core.directory.claim_landing(0, expert, 0) {
                grants.borrow_mut().push(grant);
            }
            if state.core.directory.refill_short(reserve) {
                break;
            }
        }
        assert!(state.core.directory.refill_short(reserve), "the claims left the reserve short");
        state.publish_held();
        *claimed.borrow_mut() = alone(state);
        true
    };
    // The request: waiting once the refill's and the promotions' jobs have
    // run, it takes the lock as the latter gives way, reads the holdings
    // and begins a step, before the fill.
    let request = std::cell::RefCell::new(None);
    let wait = |held: &mut residency::Residency, _: &mut residency::Turn, _: Option<&super::super::promotions::Copier>, _: Option<&crate::experts::directory::MissCurve>| {
        assert_ne!(alone(held), *claimed.borrow(), "the refill took no victim the tier kept alone");
        let (requested, shared) = (state.clone(), shared.clone());
        let records = harness.layers.iter().map(|layer| layer.4.clone()).collect::<Vec<_>>();
        // Its own count, not any holder's: the load's request service is
        // counted too while it looks, and a turn that gave way to it alone
        // would run the fill before this request begins its step.
        let counted = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let waits = counted.clone();
        let thread = std::thread::spawn(move || {
            let held = residency::lock_counted(&requested, || waits.store(true, std::sync::atomic::Ordering::Release));
            let directory = &held.core.directory;
            let layers = 0..held.core.layers.len();
            let found = (layers.clone().map(|layer| directory.held_count(layer)).collect::<Vec<_>>(), layers.map(|layer| directory.alone(layer)).collect::<Vec<_>>());
            let step = begin(&shared, &[StepRoute::Decode], false).unwrap().pop().unwrap();
            // Its route chooses nothing at every layer, published before
            // the lock is released, as a route publishes before any read.
            let base = (step.sequence() % WORDS.flight) as usize * WORDS.routing_words() as usize;
            for record in &records {
                record.store(base + constants::RING_COUNT as usize, 0);
                record.store(base + constants::RING_SEQUENCE as usize, super::super::steps::low_word(step.sequence()));
            }
            (shared.telemetry.holdings(), found, step)
        });
        eventually("the request never waited for residency", || counted.load(std::sync::atomic::Ordering::Acquire));
        *request.borrow_mut() = Some(thread);
        true
    };
    let quiet = std::cell::Cell::new(true);
    let after = |held: &mut residency::Residency, _: &mut residency::Turn, _: Option<&super::super::promotions::Copier>, _: Option<&crate::experts::directory::MissCurve>| {
        quiet.set(held.shared.traffic_quiet.load(std::sync::atomic::Ordering::Acquire));
        true
    };
    let turn = residency::Residency::TURN;
    residency::Residency::turn(&state, &[&short, turn[0], turn[1], &wait, turn[2], turn[3], &after], None, None);
    let (published, found, step) = request.into_inner().expect("the request ran").join().unwrap();
    assert_eq!(published, found, "the request read holdings the refill left unpublished");
    assert!(!quiet.get(), "the fill called the traffic quiet while a step ran");
    drop(grants);
    step.complete().unwrap();
}
