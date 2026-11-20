use super::*;

/// A front row's skips beyond the read-ahead window are held, never
/// dropped: with room beyond the demand reserve for one read ahead, a ring
/// of three skipped experts reads the first at once and each of the others
/// as the read before it frees its records, in the ring's order, so every
/// one lands for the next step's exact row.
#[test]
fn a_full_front_window_drops_nothing_and_reads_the_most_urgent_first() {
    let harness = layered("front-window", EXPERTS, 12, 2, WORDS.choices as u32 + 1);
    harness.demand_step(4);
    harness.staging_free();
    harness.tier_ready();
    let skipped = [5, 6, 7];
    assert!(skipped.iter().all(|&expert| harness.in_tier(0, expert).is_none()), "the tier holds a skipped expert before its skip");
    let step = harness.two_row_step();
    harness.front_at(&step, 0, &skipped.map(|expert| expert as i32));
    // Each lands before the one ranked after it is read.
    eventually("a held skip was never read", || {
        let landed = skipped.map(|expert| harness.in_tier(0, expert).is_some());
        assert!(landed.windows(2).all(|pair| pair[0] || !pair[1]), "a skip landed before one ranked ahead of it: {landed:?}");
        landed.iter().all(|&landed| landed)
    });
    drop(step);
}

/// A front row's read whose layer the next step has routed without
/// requesting it is cancelled as that step's front ring there shows the
/// pass, before the step ends: behind a line of demand reads, which no read
/// ahead passes, it ends only so, and holds the drive no longer, nor its
/// records.
#[test]
fn a_front_read_whose_layer_has_passed_is_cancelled_at_the_pass() {
    let _alone = alone();
    let harness = layered("front-passed", EXPERTS, 12, 2, 4 + 4);
    harness.demand_step(4);
    harness.staging_free();
    harness.tier_ready();
    let ahead = Ahead::queue("front-passed-ahead");
    let fronts = || {
        let inbound = residency::lock(&harness.watcher.state).prefetch.describe();
        inbound.split("front ").nth(1).and_then(|rest| rest.split(',').next()).map(|count| count.parse::<usize>().unwrap())
    };
    let step = harness.two_row_step();
    harness.front_at(&step, 0, &[5]);
    eventually("the skip was never read ahead", || fronts() == Some(1));
    harness.route_at(&step, 0, &[]);
    harness.route_at(&step, 1, &[]);
    step.complete().unwrap();
    // The next step routes layer 0 without requesting the skip, and its
    // front ring there shows it; it has not ended.
    let next = harness.two_row_step();
    harness.route_at(&next, 0, &[]);
    harness.front_at(&next, 0, &[]);
    eventually("the passed skip's read was never cancelled", || fronts() == Some(0));
    ahead.finish();
    harness.route_at(&next, 1, &[]);
    next.complete().unwrap();
}

/// A front row's read lands ranked as read at the step that skipped it,
/// the newest the tier has seen, though its front row ran without it: the
/// reserve's refill, which unmaps the kept records no step holds least
/// recently read first, takes every older record before it, so it stays
/// for the next step's exact row. Here the next step's misses make the
/// refill unmap records kept before the skip landed, and it keeps the skip.
#[test]
fn a_landed_front_read_outlasts_every_older_kept_record() {
    let harness = layered("front-ranked", EXPERTS, 12, 2, 4 + 4);
    // Older kept records: a step's misses before the skip, most of layer
    // 0's among them, so the next step's misses there are few.
    let seeding = harness.step();
    harness.request_at(&seeding, 0, &[0, 1, 2, 3]);
    harness.request_at(&seeding, 1, &[0, 1, 2]);
    seeding.complete().unwrap();
    harness.demand_step(4);
    harness.staging_free();
    harness.tier_ready();
    let older = (0..2)
        .flat_map(|layer| (0..EXPERTS).map(move |expert| (layer, expert)))
        .filter(|&(layer, expert)| harness.in_tier(layer, expert).is_some())
        .collect::<Vec<_>>();
    let step = harness.two_row_step();
    harness.front_at(&step, 0, &[5]);
    harness.tiering(0, &[5]);
    harness.route_at(&step, 0, &[]);
    harness.route_at(&step, 1, &[]);
    step.complete().unwrap();
    let lacking = (0..EXPERTS).filter(|&expert| harness.in_tier(0, expert).is_none()).take(WORDS.choices as usize).collect::<Vec<_>>();
    assert!(!lacking.is_empty() && !lacking.contains(&5), "{lacking:?}");
    let next = harness.step();
    harness.request_at(&next, 0, &lacking);
    harness.route_at(&next, 1, &[]);
    next.complete().unwrap();
    harness.tier_ready();
    let unmapped = older.iter().filter(|&&(layer, expert)| harness.in_tier(layer, expert).is_none()).count();
    assert!(unmapped > 0, "the refill unmapped nothing older: {older:?}");
    assert!(harness.in_tier(0, 5).is_some(), "the skip was unmapped before {} older records still kept", older.len() - unmapped);
}

/// A front row's names resolve against the exact row of the step they were
/// for once that step completes (`Ledger::resolve`), never before: a name it
/// did not choose counts against the ahead row's precision for its rank,
/// at which the row's next names are published.
#[test]
fn a_front_rows_names_calibrate_its_row_once_their_step_completes() {
    use crate::experts::interest::{Combination, Look, Row, Source};
    let harness = layered("front-calibrated", EXPERTS, 12, 2, 4 + 4);
    harness.demand_step(4);
    harness.staging_free();
    harness.tier_ready();
    let key = Combination { nearest: Look::of(Source::Ahead(Row { index: 0, front: Some(crate::GuessKind::Chained) })), rank: 0, looks: 1 };
    let precision = || residency::lock(&harness.watcher.state).prefetch.rate(0, key);
    let step = harness.two_row_step();
    let before = residency::lock(&harness.watcher.state).prefetch.front_read(0);
    harness.front_at(&step, 0, &[5]);
    eventually("the front ring was never read", || residency::lock(&harness.watcher.state).prefetch.front_read(0) != before);
    harness.route_at(&step, 0, &[]);
    harness.route_at(&step, 1, &[]);
    step.complete().unwrap();
    assert_eq!(precision(), 1., "no outcome before the step it named completes: its bound");
    // The next step's exact row chooses another expert there.
    // It is the request's next pass, which binds the names to its step.
    let next = harness.begun().decoding([row(1000, None)].into_iter(), 0);
    harness.route_at(&next, 0, &[3]);
    harness.route_at(&next, 1, &[]);
    next.complete().unwrap();
    eventually("the step's outcomes were never learned", || precision() == 0.);
}

/// The front inbound reads of a harness's prefetch (`Prefetch::describe`).
fn fronts(harness: &Harness) -> Option<usize> {
    let inbound = residency::lock(&harness.watcher.state).prefetch.describe();
    inbound.split("front ").nth(1).and_then(|rest| rest.split(',').next()).map(|count| count.parse::<usize>().unwrap())
}

/// A verdict that a pass's guess did not stand (`Shared::decided`, which
/// the domain tells as it reconciles the pass, `ElasticBank::decided`)
/// drops the names of its request's rows past the wrong guess: the read of
/// one that no worker has begun is cancelled, and ends holding no drive nor
/// records, while the exact step it was named for has not even run.
#[test]
fn a_wrong_guess_cancels_the_unbegun_read_of_its_names() {
    use crate::operation::{RequestId, Rows, TokenId};
    use crate::{AheadVerdict, Guess, GuessKind};

    let _alone = alone();
    let harness = layered("front-wrong", EXPERTS, 12, 2, 4 + 4);
    harness.demand_step(4);
    harness.staging_free();
    harness.tier_ready();
    let ahead = Ahead::queue("front-wrong-ahead");
    let step = harness.two_row_step();
    let sequence = step.sequence();
    harness.front_at(&step, 0, &[5]);
    eventually("the skip was never read ahead", || fronts(&harness) == Some(1));
    harness.route_at(&step, 0, &[]);
    harness.route_at(&step, 1, &[]);
    step.complete().unwrap();
    // Its pass's verdict: its front row's guess, 7, was not the truth, 8.
    let pass = Rows::front(vec![Guess { token: TokenId(7), kind: GuessKind::Chained }], 0);
    harness.watcher.shared().decided(sequence, RequestId(1), &AheadVerdict::new(&pass, &[TokenId(8), TokenId(9)]));
    eventually("the wrong guess's read was never cancelled", || fronts(&harness) == Some(0));
    ahead.finish();
    assert_eq!(harness.in_tier(0, 5), None, "the cancelled read landed nothing");
}

/// Evicting a request's state or closing it tells the expert bank
/// (`ExecutorDomain::release_state` / `ExecutorDomain::close` ->
/// `ElasticBank::closed`), dropping its unbound ahead names and cancelling
/// their unbegun reads.
#[test]
fn evicting_or_closing_a_request_cancels_the_unbegun_read_of_its_names() {
    use crate::domain::fixtures::{bank_backends, banked, fake_bank_mut};
    use crate::operation::RequestId;
    use magnitude_family_contracts::PreparedModelInput;

    let _alone = alone();
    for evict in [true, false] {
        let label = if evict { "front-evict" } else { "front-close" };
        let harness = layered(label, EXPERTS, 12, 2, 4 + 4);
        harness.demand_step(4);
        harness.staging_free();
        harness.tier_ready();
        let (mut domain, mut bindings) = banked(bank_backends()[0], 4, 4);
        let shared = harness.watcher.shared().clone();
        fake_bank_mut(&mut domain).on_closed = Some(Box::new(move |request| {
            shared.closed(request);
        }));
        let request = RequestId(1);
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(domain.unchained(&mut bindings).expect("no chain is under way"), request, None, 0).unwrap();

        let ahead = Ahead::queue(&format!("{label}-ahead"));
        let step = harness.two_row_step();
        harness.front_at(&step, 0, &[5]);
        eventually("the skip was never read ahead", || fronts(&harness) == Some(1));
        if evict {
            domain.release_state(&[request]).unwrap();
        } else {
            domain.close(request).unwrap();
        }
        eventually("the request's ahead read was never cancelled", || fronts(&harness) == Some(0));
        ahead.finish();
        harness.route_at(&step, 0, &[]);
        harness.route_at(&step, 1, &[]);
        step.complete().unwrap();
        assert_eq!(harness.in_tier(0, 5), None, "{label}: the cancelled read landed nothing");
    }
}

/// On a two-row pass that never parks on a demand miss, reading a later
/// layer's front ring proves the device is past earlier layers' passes:
/// kept records unmapped into `Host::Graced` at layer 0 retire to ready as
/// layer 1's front ring is read, so front-row skips beyond the initial
/// ready reserve read ahead and land during the pass rather than waiting
/// for the step to end.
#[test]
fn a_non_parking_pass_retires_graced_tier_records_from_its_rings() {
    use crate::experts::horizon::{Reader, Tier};

    let harness = lending("front-ring-retire", EXPERTS, 4, 2, WORDS.choices as u32 + 1, DECODE, 11, Placer::Tier, false);
    let seeding = harness.step();
    harness.request_at(&seeding, 0, &[0, 1, 2, 3]);
    harness.request_at(&seeding, 1, &[0, 1]);
    seeding.complete().unwrap();
    harness.staging_free();
    harness.tier_ready();
    eventually("decode history never ranked", || harness.watcher.shared().telemetry.fed_all());
    harness.tier_ready();

    // With 11 tier records and reserve 5 (demand 4 + 1 ahead), 6 unprotected
    // incumbents stay kept (four at layer 0 ranked below the skips' certain
    // uses, and `(1, 0..2)` read every step) and 5 records are ready.
    // Skipping `[4, 5, 6, 7]` at layer 0 lands all 4 and graces layer 0's
    // four incumbents, leaving only 1 ready record; reading layer 1's front
    // ring of `[2, 3]` advances residency past layer 0, retiring those 4
    // graced records to ready while `step` still runs so `(1, 3)` claims a
    // record and lands mid-step.
    let step = harness.two_row_step();
    {
        let state = residency::lock(&harness.watcher.state);
        let directory = &state.core.directory;
        let host = directory.host(0);
        assert_eq!((host.records(), host.reserve(), host.ready()), (11, 5, 5), "record counts before the pass");
        assert!(directory.in_tier(1, 0) && directory.in_tier(1, 1), "layer 1's incumbents kept");
        assert!((4..8).all(|expert| !directory.in_tier(0, expert)), "layer 0's skips lacked");
        assert!((2..4).all(|expert| !directory.in_tier(1, expert)), "layer 1's skips lacked");
        let victims: Vec<usize> = (0..EXPERTS).filter(|&expert| directory.in_tier(0, expert)).collect();
        assert_eq!(victims.len(), 4, "four layer-0 incumbents kept before the pass: {victims:?}");
        let reuse = directory.reuse_now();
        for &victim in &victims {
            let victim_rank = reuse.rank(0, victim, Tier::Host, Reader::Move, None);
            for skip in 4..8 {
                let skip_rank = reuse.rank_of(0, skip, reuse.expected_with(0, skip, Tier::Host, step.sequence() + 1, 1.0));
                assert!(victim_rank < skip_rank, "incumbent (0, {victim}) must rank below skip (0, {skip})");
            }
        }
    }
    harness.front_at(&step, 0, &[4, 5, 6, 7]);
    eventually("layer 0's fourth skip never landed", || harness.in_tier(0, 7).is_some());
    harness.front_at(&step, 1, &[2, 3]);
    eventually("graced tier records never retired from ring progress during the pass", || {
        harness.in_tier(1, 3).is_some()
    });
    harness.route_at(&step, 0, &[]);
    harness.route_at(&step, 1, &[]);
    step.complete().unwrap();
}
