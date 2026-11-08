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
    let warm = harness.step().decoding([10; 4].into_iter());
    harness.request_at(&warm, 0, &[0, 1]);
    warm.complete().unwrap();
    assert_eq!(harness.watcher.shared().live(), None, "a warm-up's step is not counted");
    let steps = 6u64;
    for position in 0..steps {
        // Two experts a launch, cycling through every expert: each misses.
        let first = (2 * (position as usize + 1)) % EXPERTS;
        let step = harness.step().decoding([1000 + position; 4].into_iter()).recorded(Recording);
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
    let last = Point { group: 1, layer: 0, sequence: 2 };
    telemetry.group(last, 2, GroupLayer::default());
    telemetry.served_whole(Some((last, Instant::now())));
    // Group B of steps 3 and 4 counted, then A's last step completes.
    let next = Point { group: 3, layer: 0, sequence: 3 };
    telemetry.group(next, 4, GroupLayer::default());
    telemetry.complete_step(2, CompletedStep { work: LaunchWork::Prompt, begun: Instant::now(), route: CompletedRoute::Grouped }, &|_| (0, 0, Duration::ZERO), None).expect("no reads, no line");
    telemetry.served_whole(Some((next, Instant::now())));
    telemetry.arrived(3, Instant::now());
    let group = harness.watcher.shared().costs().group.expect("group B is counted");
    assert_eq!((group.first, group.layers[0].computed), (3, 1));
}
