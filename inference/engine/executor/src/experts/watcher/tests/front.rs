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

/// A front row's read lands protected for the next step's exact row at its
/// layer (`is_future_read`): the reserve's refill unmaps older kept records
/// before it while its target `(step, layer)` is still ahead, so it stays for
/// the next step's exact row. Here `step` skips `(1, 5)` at layer 1, and
/// `next`'s misses at layer 0 trigger a refill that unmaps older kept records
/// while keeping `(1, 5)` ready for `next`'s layer 1.
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
    harness.front_at(&step, 1, &[5]);
    harness.tiering(1, &[5]);
    harness.route_at(&step, 0, &[]);
    harness.route_at(&step, 1, &[]);
    step.complete().unwrap();
    let lacking = (0..EXPERTS).filter(|&expert| harness.in_tier(0, expert).is_none()).take(WORDS.choices as usize).collect::<Vec<_>>();
    assert!(!lacking.is_empty(), "{lacking:?}");
    let next = harness.step();
    harness.request_at(&next, 0, &lacking);
    harness.tier_ready();
    let unmapped = older.iter().filter(|&&(layer, expert)| harness.in_tier(layer, expert).is_none()).count();
    assert!(unmapped > 0, "the refill unmapped nothing older: {older:?}");
    assert!(harness.in_tier(1, 5).is_some(), "the skip was unmapped before {} older records still kept", older.len() - unmapped);
    harness.route_at(&next, 1, &[5]);
    next.complete().unwrap();
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
    let precision = || residency::lock(&harness.watcher.state).ledger.rate(0, key);
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
            let victim_rank = reuse.rank(0, victim, Tier::HOST, Reader::Move, None);
            for skip in 4..8 {
                let skip_rank = reuse.rank_of(0, skip, reuse.expected_with(0, skip, Tier::HOST, step.sequence() + 1, 1.0));
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

/// Goal A Deletion Gate: with zero layer-hint machinery compiled, a
/// `FrontRows`-only name published to `NameLedger`
/// (`When::Pass { request, pass: P + 1 }`) launches a `Prefetch::read_ahead`
/// during `Told::DraftWindow`, and `Prefetch::want` at `P + 1` adopts that
/// exact same in-flight read once.
#[test]
fn front_rows_read_ahead_launches_in_draft_window_and_want_adopts_once() {
    use crate::experts::interest::{Lineage, Name, Row, Source, Told, When};
    use crate::experts::watcher::At;
    use crate::{GuessKind, RequestId};
    use std::num::NonZeroU8;

    let _alone = alone();
    let harness = layered("front-draft-window-adopt", EXPERTS, 12, 2, 4 + 4);
    harness.demand_step(4);
    harness.staging_free();
    harness.tier_ready();

    // Hold the drive queue so the read ahead launched during `DraftWindow`
    // stays in flight until `P + 1` arrives and calls `Prefetch::want`.
    let ahead = Ahead::queue("front-draft-window-queue");

    // Pass P completes on request 1 (pass 1).
    let pass_p = harness.two_row_step();
    let pass_p_seq = pass_p.sequence();
    harness.route_at(&pass_p, 0, &[]);
    harness.route_at(&pass_p, 1, &[]);
    pass_p.complete().unwrap();

    // Publish a FrontRows-only name for `When::Pass { request: 1, pass: 2 }`
    // and open `Told::DraftWindow`.
    {
        let mut state = residency::lock(&harness.watcher.state);
        let name = Name {
            source: Source::Ahead(Row { index: 0, front: Some(GuessKind::Head) }),
            when: When::Pass { request: RequestId(1), pass: 2, estimate: pass_p_seq + 1 },
            rank: 0,
            from: At::whole(pass_p_seq, 0),
            candidate: true,
            lineage: Some(Lineage { request: RequestId(1), row: 1, guessed: NonZeroU8::MIN }),
        };
        state.ledger.publish(0, 5, name, Instant::now());
    }
    harness.watcher.shared().tell(Told::DraftWindow { estimated_seconds: 1.0 });

    // `Prefetch::read_ahead` launches during `DraftWindow`.
    eventually("front-rows read ahead never launched in DraftWindow", || {
        harness.watcher.shared().costs().front.reads == 1
    });

    // Pass P + 1 begins and requests expert 5 at layer 0: `Prefetch::want`
    // adopts that exact same in-flight read once without issuing a second read.
    let pass_p1 = harness.two_row_step();
    let before_unanticipated = harness.watcher.shared().costs().layers[0].unanticipated;
    harness.request_at(&pass_p1, 0, &[5]);
    let (record, verdict) = harness.answer(0, pass_p1.sequence(), 0);
    assert_eq!(verdict, Some(Answer::Answered));
    assert_eq!(harness.in_tier(0, 5), Some(record));
    harness.route_at(&pass_p1, 1, &[]);
    pass_p1.complete().unwrap();

    let costs = harness.watcher.shared().costs();
    assert_eq!(costs.front.reads, 1, "only the single DraftWindow read ahead was issued");
    assert_eq!(costs.layers[0].reading, 1, "Prefetch::want adopted the in-flight read once");
    assert_eq!(costs.layers[0].unanticipated, before_unanticipated, "adopted read is not an unanticipated demand read");
    ahead.finish();
}

/// Design A mixed `(1 + K + F)` decode step: `STEP_FRONT` marks ONLY the `F`
/// tail row(s), so `1 + K` exact+acceptable rows stay on the demand path while
/// `F`-only misses are excluded from immediate demand routing (`exact`).
#[test]
fn mixed_verify_and_front_step_withholds_f_only_rows_from_demand() {
    use crate::experts::watcher::steps::AheadOf;

    let harness = layered("mixed-kf-front", EXPERTS, 12, 2, 4 + 4);
    harness.demand_step(4);
    harness.staging_free();
    harness.tier_ready();

    // Build a mixed 3-row decode step: row 0 = exact, row 1 = acceptable (K=1), row 2 = front (F=1).
    let rows = [
        row(1000, None),
        row(1001, Some(AheadOf::new(1, 1, 0))),
        row(1002, Some(AheadOf::new(1, 1, 1))),
    ];
    let step = harness.begun().decoding(rows.into_iter(), 0);
    // Bit 0 (exact) and bit 1 (acceptable K=1) are NOT front; bit 2 (front F=1) IS front!
    assert_eq!(step.front(), 1 << 2, "only the F tail row sets STEP_FRONT");
    harness.route_at(&step, 0, &[]);
    harness.route_at(&step, 1, &[]);
    step.complete().unwrap();
}
