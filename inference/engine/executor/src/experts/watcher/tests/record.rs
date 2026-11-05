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

/// The lines of the harness's one layer, the reference cell's
/// (`ExpertLines::reference`); an expert's stored record is three tensors'
/// [`RECORD`] bytes.
fn tier_lines() -> std::sync::Arc<[crate::assessment::ExpertLines]> {
    vec![crate::assessment::ExpertLines::reference(3 * RECORD)].into()
}

/// A decode of four-row launches alone (a verify's), whose every launch
/// misses, measures `σ` at four rows and becomes due once its misses reach
/// `1 / ε²`: no one-row launch, nor any precision of `σ`, is asked for.
#[test]
fn a_decode_of_multi_row_launches_is_recorded_at_their_rows() {
    use crate::assessment::MISS_RATE_ERROR;
    let harness = harness("launch-rows", EXPERTS, 2);
    let lines = tier_lines();
    let misses = MISS_RATE_ERROR.powi(-2) as usize;
    let mut record = None;
    for position in 0..misses as u64 {
        // Two experts a launch, cycling through every expert: each misses.
        let first = (2 * position as usize) % EXPERTS;
        let step = harness
            .step()
            .decoding(4, 1000 + position)
            .recorded(Recording { lines: Some(lines.clone()) });
        harness.request_at(&step, 0, &[first, first + 1]);
        if let Some(due) = step.complete().unwrap() {
            record = Some(due);
            break;
        }
    }
    let record = record.expect("a record is due");
    let launches = record.residuals.rows.map(|at| at.launches);
    assert!(launches[3] > 0 && launches.iter().sum::<u64>() == launches[3], "{launches:?}");
    assert!(record.compute(&|rows, _| Ok::<_, ()>(0.01 * rows as f64)).unwrap().is_ok());
}

/// A replay of a decode from what the run recorded of it, as the harness
/// keeps a cell (each step's layers and the launch it sampled for σ, the
/// served file's reads, through their serialized
/// form), assembles through the one record assembly (`RecordFeed`) the
/// record the live telemetry made due. The reads' curve and returns are the
/// ranking's stack's, which the replay feeds from the same routing.
#[test]
fn a_replay_of_the_recorded_steps_assembles_the_live_record() {
    use crate::assessment::{LaunchSample, LayerStep, RecordFeed, MISS_RATE_ERROR};
    let harness = harness("replayed", EXPERTS, 2);
    let lines = tier_lines();
    // As a cell records each completed step's expert step.
    let mut recorded = Vec::new();
    let mut live = None;
    for position in 0..MISS_RATE_ERROR.powi(-2) as u64 {
        let first = (2 * position as usize) % EXPERTS;
        // Launches of one to four rows, so σ is sampled at each.
        let step = harness
            .step()
            .decoding(1 + position as usize % 4, 1000 + position)
            .recorded(Recording { lines: Some(lines.clone()) });
        harness.request_at(&step, 0, &[first, first + 1]);
        let due = step.complete().unwrap();
        let (_, cost) = harness.watcher.shared().costs().completed.expect("a completed step");
        recorded.push(serde_json::json!({ "layers": cost.layers, "launch": cost.launch }));
        if due.is_some() {
            live = due;
            break;
        }
    }
    let live = live.expect("a record is due");
    let costs = harness.watcher.shared().costs();
    let bank = serde_json::json!({ "reads": costs.reads });
    let mut feed = RecordFeed::default();
    // The lines as the record took them, with the landing line observed.
    let lines = live.layers.iter().map(|layer| layer.lines).collect::<std::sync::Arc<[_]>>();
    for step in &recorded {
        let layers = serde_json::from_value::<Vec<LayerStep>>(step["layers"].clone()).unwrap();
        let sample = serde_json::from_value::<Option<LaunchSample>>(step["launch"].clone()).unwrap();
        feed.decode(&layers, sample, Some(lines.clone()));
    }
    let curve = crate::experts::MissCurve {
        steps: live.steps,
        layers: live.layers.iter().map(|layer| layer.curve.clone()).collect(),
    };
    let reads = serde_json::from_value(bank["reads"].clone()).unwrap();
    let replayed = feed.record(&curve, reads).expect("the replay sampled σ");
    let rows = live.residuals.rows.map(|at| at.launches);
    assert!(rows[..4].iter().all(|&launches| launches > 0), "σ sampled at one to four rows: {rows:?}");
    assert_eq!(replayed, *live);
}

/// With two prompt groups in flight, the watcher counts the second's first
/// request before the owner completes the first's last step: that step's
/// acknowledgement went with its group, so its completion counts nothing
/// and never finds a group gone; the second group's own acknowledgement
/// still counts its compute.
#[test]
fn a_groups_acknowledgement_goes_with_its_group() {
    use super::steps::CompletedStep;
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
    telemetry.complete_step(2, CompletedStep::Prompt, &|_| (0, 0, Duration::ZERO), None).expect("no reads, no line");
    telemetry.served_whole(Some((next, Instant::now())));
    telemetry.arrived(3, Instant::now());
    let group = harness.watcher.shared().costs().group.expect("group B is counted");
    assert_eq!((group.first, group.layers[0].computed), (3, 1));
}
