use super::*;

/// A step's time runs from the later of its begin and the last step's end:
/// a gap between steps (a client's, a tool call's) is no step's time, so it
/// never reads as idle volume.
#[test]
fn a_gap_between_steps_is_no_steps_time() {
    let harness = layered("gap", EXPERTS, 12, 2, 4 + 4);
    harness.demand_step(4);
    std::thread::sleep(Duration::from_millis(300));
    let began = Instant::now();
    harness.demand_step(5);
    let (seconds, step) = (harness.watcher.shared().costs().completed.expect("a step completed").1.seconds, began.elapsed());
    assert!(seconds <= step.as_secs_f64(), "the step's time ({seconds} s) took in the gap: it ran {step:?}");
}

/// A counted decode is live at its launches' rows and context: a decode of
/// four-row launches (a verify's) counts its steps at four rows, each its
/// rows' mean context, and its requests as single rows' (every step here
/// misses, so a row's share is `1 − (1 − 1)^¼ = 1`); a step not counted
/// (a warm-up's) is not.
#[test]
fn a_counted_decode_is_live_at_its_launches_rows_and_context() {
    let harness = harness("live", EXPERTS, 2);
    let warm = harness.begun().decoding([row(10, None); 4].into_iter(), 0);
    harness.request_at(&warm, 0, &[0, 1]);
    warm.complete().unwrap();
    assert_eq!(harness.watcher.shared().live(), None, "a warm-up's step is not counted");
    let steps = 6u64;
    for position in 0..steps {
        // Two experts a launch, cycling through every expert: each misses.
        let first = (2 * (position as usize + 1)) % EXPERTS;
        let step = harness.begun().decoding([row(1000 + position, None); 4].into_iter(), 0).recorded(Recording);
        harness.request_at(&step, 0, &[first, first + 1]);
        step.complete().unwrap();
    }
    let shared = harness.watcher.shared();
    eventually("the decode reads were never ranked", || shared.telemetry.fed_all());
    let live = shared.live().expect("a counted decode");
    assert_eq!((live.steps, live.rows), (steps, 4.));
    // Each launch's context is its rows' positions plus one.
    assert_eq!(live.depth, 1001 + (steps as u32 - 1) / 2);
    assert_eq!(live.layers.len(), 1);
}

/// A counted decode with front rows counts its exact rows alone
/// (`RoutingRecord::exact`): a launch of an exact row and a front row (the
/// next position's, as its pass places it) reads its exact row's choice
/// alone and counts one row at its exact row's context.
#[test]
fn a_counted_decode_counts_its_exact_rows_alone() {
    let harness = harness("exact", EXPERTS, 2);
    let routed = || {
        let costs = harness.watcher.shared().costs();
        let (_, step) = costs.completed.expect("a step completed");
        step.experts[0].routed.as_ref().expect("a decode step's routing").experts.clone()
    };
    let steps = 4u64;
    for position in 0..steps {
        let exact = (2 * position as usize) % EXPERTS;
        let paired = harness.begun().decoding([row(1000 + position, None), row(1001 + position, front_row())].into_iter(), 0).recorded(Recording);
        harness.request_at(&paired, 0, &[exact, exact + 1]);
        paired.complete().unwrap();
        assert_eq!(routed(), [exact], "the exact row's choice alone");
    }
    let shared = harness.watcher.shared();
    eventually("the decode reads were never ranked", || shared.telemetry.fed_all());
    let live = shared.live().expect("a counted decode");
    assert_eq!((live.steps, live.rows), (steps, 1.));
    assert_eq!(live.depth, 1001 + (steps as u32 - 1) / 2);
}

/// With two prompt groups in flight, the watcher counts the second's first
/// request before the owner completes the first's last step: that step's
/// acknowledgement went with its group, so its completion counts nothing
/// and never finds a group gone; the second group's own acknowledgement
/// still counts its compute.
#[test]
fn a_groups_acknowledgement_goes_with_its_group() {
    use super::steps::{CompletedRoute, CompletedStep};
    use super::telemetry::GroupLayer;
    use crate::experts::directory::Point;
    let harness = prompted("acknowledged", EXPERTS, 4);
    let telemetry = &harness.watcher.shared().telemetry;
    // Group A of steps 1 and 2, its last request served whole.
    let last = Point::whole(1, 0, 2);
    telemetry.group(last, 2, GroupLayer::default());
    telemetry.served_whole(Some((last, Instant::now(), true)));
    // Group B of steps 3 and 4 counted, then A's last step completes.
    let next = Point::whole(3, 0, 3);
    telemetry.group(next, 4, GroupLayer::default());
    telemetry.complete_step(&harness.watcher.shared().steps, 2, CompletedStep { work: LaunchWork::Prompt, begun: Instant::now(), route: CompletedRoute::Grouped }, &|_| (0, 0, Duration::ZERO), None).expect("no reads, no line");
    telemetry.served_whole(Some((next, Instant::now(), true)));
    telemetry.arrived(3, Instant::now());
    let group = harness.watcher.shared().costs().group.expect("group B is counted");
    assert_eq!((group.first, group.layers[0].computed), (3, 1));
}

/// `Telemetry::costs.ahead` prunes entries for steps `<= sequence` on
/// `complete_step` and ignores late `read_ahead` records for already-completed
/// steps so dead entries never accumulate over engine lifetime.
#[test]
fn telemetry_ahead_prunes_past_completed_steps_and_ignores_late_reads() {
    use super::steps::{CompletedRoute, CompletedStep};
    use crate::experts::interest::Source;
    let harness = prompted("ahead-prune", EXPERTS, 4);
    let telemetry = &harness.watcher.shared().telemetry;
    let lead = Source::Lead { distance: 1, row: None };
    for step in [1u64, 2, 3, 5] {
        telemetry.read_ahead(lead, step, (0, 0), 0, 1024);
    }
    assert_eq!(telemetry.ahead_len(), 4);
    // Completing step 4 prunes skipped/uncompleted earlier steps 1..=3.
    telemetry
        .complete_step(
            &harness.watcher.shared().steps,
            4,
            CompletedStep { work: LaunchWork::Prompt, begun: Instant::now(), route: CompletedRoute::Grouped },
            &|_| (0, 0, Duration::ZERO),
            None,
        )
        .expect("complete step 4");
    assert_eq!(telemetry.ahead_len(), 1, "steps <= 4 pruned, future step 5 kept");
    // Late read_ahead for already-completed step 2 must not leak back in.
    telemetry.read_ahead(lead, 2, (0, 0), 0, 1024);
    assert_eq!(telemetry.ahead_len(), 1, "late read_ahead for completed step ignored");
    telemetry
        .complete_step(
            &harness.watcher.shared().steps,
            5,
            CompletedStep { work: LaunchWork::Prompt, begun: Instant::now(), route: CompletedRoute::Grouped },
            &|_| (0, 0, Duration::ZERO),
            None,
        )
        .expect("complete step 5");
    assert_eq!(telemetry.ahead_len(), 0, "all completed steps pruned");
}

/// Queued `ExecutedStep`s read `(epoch, now)` under `Telemetry::shadows` when
/// drained rather than capturing a detached `(epoch, now)` at step completion:
/// advancing `bank_epoch` via resizes between step completion and
/// `Telemetry::shadow()` never regresses `Shadows::deltas`'s window epoch.
#[test]
fn shadow_executed_never_regresses_epoch_across_resizes() {
    use crate::experts::horizon::Tier;
    let harness = harness("shadow-epoch", EXPERTS, 2);
    let shared = harness.watcher.shared();
    // Enable shadow tracking (as regime pricing does).
    let _ = shared.telemetry.measured();

    // Run enough decode steps for `Shadows` to advance past its behind-buffer
    // and populate `U_shadow(0)`, while interspersing bank resizes that bump
    // `bank_epoch` rapidly (the exact sequence that previously left
    // `ExecutedStep` carrying a stale epoch behind `Shadows::deltas`).
    for position in 0..16u64 {
        let first = (2 * position as usize) % EXPERTS;
        let step = harness.begun().decoding([row(position, None)].into_iter(), 0);
        harness.request_at(&step, 0, &[first, first + 1]);
        step.complete().unwrap();
        // Bump `bank_epoch` multiple times right after step completion,
        // before/alongside `expert-residency`'s `telemetry.shadow()` turn.
        {
            let mut times = shared.time().lock();
            times.resized(0, Tier::Compute, 2);
            times.resized(0, Tier::Compute, 2);
        }
        shared.telemetry.feed();
        shared.telemetry.shadow();
    }
    eventually("the decode reads were never ranked", || shared.telemetry.fed_all());
    assert!(shared.time().lock().bank_epoch() >= 13, "bank_epoch advanced across resizes");
    assert!(shared.life.ended().is_none(), "expert-residency must not panic on epoch ordering");
    assert!(shared.telemetry.measured().is_some(), "regime pricing reads measured shadow gaps");
}
