use super::*;

/// A hinted expert of the next layer is read beyond the demand reserve and
/// lands into the tier ahead of its request: its route finds it there and
/// requests nothing, the tier serving it; a hint of an expert the tier
/// holds reads nothing. The step records each layer's file and tier
/// misses. The tier's fill lands its records
/// beyond the reserve first, every layer's lowest experts, which no step
/// below names.
#[test]
fn a_hinted_expert_lands_into_the_tier_ahead_of_its_request() {
    let harness = layered("hinted", EXPERTS, 12, 3, 4 + 4);
    for (layer, experts) in [(0, &[0, 1, 2][..]), (1, &[0, 1, 2]), (2, &[0, 1])] {
        harness.tiering(layer, experts);
    }
    harness.demand_step(4);
    harness.tier_ready();
    let step = harness.step();
    harness.hint(&step, 0, &[5, 6]);
    harness.request_at(&step, 0, &[4]);
    harness.tiering(1, &[5, 6]);
    harness.hint(&step, 1, &[7]);
    assert!(!harness.ask_at(&step, 1, &[5, 6]), "the tier serves both");
    harness.tiering(2, &[7]);
    harness.request_at(&step, 2, &[7, 2]);
    step.complete().unwrap();
    let costs = harness.watcher.shared().costs();
    assert_eq!(costs.hints.reads, 3);
    let [first, second, third] = [0, 1, 2].map(|layer| costs.layers[layer]);
    assert_eq!((first.unanticipated, first.lead), (1, 0));
    // The first step's one request at the second layer.
    assert_eq!(second.hosted, 2);
    assert_eq!((third.hosted, third.unanticipated, third.lead), (1, 1, 0));
    let timeline = &costs.completed.as_ref().unwrap().1.layers;
    let steps = timeline.iter().map(|layer| (layer.misses, layer.hosted)).collect::<Vec<_>>();
    assert_eq!(steps, [(1, 0), (0, 2), (1, 1)]);
    // Expert 5 is now held by the tier: hinting it again reads nothing.
    let next = harness.step();
    harness.hint(&next, 0, &[5]);
    harness.request_at(&next, 0, &[3]);
    drop(next);
    assert_eq!(harness.watcher.shared().costs().hints.reads, 3);
}

/// A request that names a hinted expert whose read still runs (queued
/// behind other demand reads) promotes that read, which lands into its
/// tier record and answers the request's choice: nothing is read on
/// demand, and the request's lead is the layer the hint ran ahead.
#[test]
fn a_hinted_experts_read_still_running_answers_its_request() {
    let _alone = alone();
    let harness = layered("hint-reading", EXPERTS, 12, 2, 4 + 4);
    harness.demand_step(4);
    harness.staging_free();
    harness.tier_ready();
    let ahead = Ahead::queue("hint-ahead", 512);
    let step = harness.step();
    harness.hint(&step, 0, &[5]);
    eventually("the hint was never read", || harness.watcher.shared().costs().hints.reads == 1);
    harness.route_at(&step, 0, &[]);
    harness.request_at(&step, 1, &[5]);
    let (record, verdict) = harness.answer(1, step.sequence(), 0);
    assert_eq!(verdict, Some(Answer::Answered));
    assert_eq!(harness.in_tier(1, 5), Some(record));
    step.complete().unwrap();
    let costs = harness.watcher.shared().costs();
    let layer = costs.layers[1];
    // The first step's request read on demand.
    assert_eq!((layer.reading, layer.unanticipated, layer.lead), (1, 1, 1));
    // Its hint named the miss, and had not landed before the route.
    assert_eq!((costs.hints.named, costs.hints.early), (1, 0));
    ahead.finish();
}

/// A ring's second row hints the layer two ahead: its read runs from the
/// first layer's route, a request there two layers on is answered by it, and
/// the step's trace keeps the candidate at its lead.
#[test]
fn a_second_leads_hint_answers_the_request_two_layers_on() {
    let _alone = alone();
    let harness = layered("hint-lead-two", EXPERTS, 12, 3, 4 + 4);
    harness.demand_step(4);
    harness.staging_free();
    harness.tier_ready();
    let ahead = Ahead::queue("hint-two-ahead", 512);
    let step = harness.step();
    harness.hint_at(&step, 0, 1, &[5]);
    eventually("the second lead's hint was never read", || harness.watcher.shared().costs().hints.reads == 1);
    harness.route_at(&step, 0, &[]);
    harness.route_at(&step, 1, &[]);
    harness.request_at(&step, 2, &[5]);
    let (record, verdict) = harness.answer(2, step.sequence(), 0);
    assert_eq!(verdict, Some(Answer::Answered));
    assert_eq!(harness.in_tier(2, 5), Some(record));
    step.complete().unwrap();
    let costs = harness.watcher.shared().costs();
    assert_eq!(costs.layers[2].reading, 1, "the hint's read, not a demand one");
    ahead.finish();
}

/// A hint whose target layer the device reaches without naming it (its
/// request there names another expert) is withdrawn while its read waits
/// behind others: it lands nothing, the request's own expert is answered,
/// and a later request for the hinted expert reads it anew.
#[test]
fn a_passed_hint_reads_nothing() {
    let _alone = alone();
    let harness = layered("hint-passed", EXPERTS, 12, 2, 4 + 4);
    // The tier's fill lands its records beyond the reserve first, every
    // layer's lowest experts: its reads must not wait behind the reads
    // queued ahead below, holding the staging records the hint needs.
    harness.tiering(0, &[0, 1, 2, 3]);
    harness.tiering(1, &[0, 1, 2]);
    harness.demand_step(4);
    harness.tier_ready();
    let ahead = Ahead::queue("hint-passed-ahead", 512);
    let step = harness.step();
    harness.hint(&step, 0, &[5]);
    eventually("the hint was never issued", || harness.watcher.shared().costs().hints.reads == 1);
    harness.route_at(&step, 0, &[]);
    harness.request_at(&step, 1, &[6]);
    assert_eq!(harness.verdict(1, step.sequence()), Some(Answer::Answered));
    step.complete().unwrap();
    ahead.finish();
    assert_eq!(harness.in_tier(1, 5), None, "the passed hint landed nothing");
    let next = harness.step();
    harness.route_at(&next, 0, &[]);
    harness.request_at(&next, 1, &[5]);
    assert_eq!(harness.verdict(1, next.sequence()), Some(Answer::Answered));
    next.complete().unwrap();
    let layer = harness.watcher.shared().costs().layers[1];
    assert_eq!(layer.reading, 0, "no request found the passed hint reading");
}

/// Let the tier's fill land all it will, and run a step past it. The
/// experts of layer 1 still absent from the bank and the tier, ascending.
fn settled(harness: &Harness) -> Vec<usize> {
    let held = || (0..harness.layers.len()).map(|layer| (0..EXPERTS).filter(|&expert| harness.in_tier(layer, expert).is_some()).count()).sum::<usize>();
    let mut last = usize::MAX;
    while held() != last {
        last = held();
        harness.staging_free();
        std::thread::sleep(Duration::from_millis(50));
    }
    let step = harness.step();
    for layer in 0..harness.layers.len() {
        harness.route_at(&step, layer, &[]);
    }
    step.complete().unwrap();
    harness.tier_ready();
    (0..EXPERTS).filter(|&expert| harness.in_tier(1, expert).is_none() && harness.layers[1].0.held(expert).is_none()).collect()
}

/// A hint withdrawn before any of it was read lands nothing.
#[test]
fn a_hint_withdrawn_unread_lands_nothing() {
    let _alone = alone();
    let harness = layered("hint-withdrawn", EXPERTS, 12, 2, 4 + 4);
    harness.demand_step(4);
    let absent = settled(&harness);
    let ahead = Ahead::queue("hint-withdrawn-ahead", 512);
    let step = harness.step();
    harness.hint(&step, 0, &[absent[0] as i32]);
    eventually("the hint was never issued", || harness.watcher.shared().costs().hints.reads == 1);
    harness.route_at(&step, 0, &[]);
    harness.request_at(&step, 1, &[absent[1]]);
    assert_eq!(harness.verdict(1, step.sequence()), Some(Answer::Answered));
    step.complete().unwrap();
    ahead.finish();
    assert_eq!(harness.in_tier(1, absent[0]), None, "the withdrawn hint read nothing");
}

/// A hint of a layer the device routes past with no request there reads
/// nothing more, on the next layer's ring alone: no request was served to
/// withdraw it, and its step has not ended.
#[test]
fn a_hint_of_a_layer_routed_without_a_request_reads_nothing() {
    let _alone = alone();
    let harness = layered("hint-routed", EXPERTS, 12, 4, 4 + 4);
    for layer in 0..4 {
        harness.tiering(layer, &[0, 1]);
    }
    harness.demand_step(4);
    harness.tier_ready();
    let ahead = Ahead::queue("hint-routed-ahead", 512);
    let step = harness.step();
    let reads = harness.watcher.shared().costs().hints.reads;
    harness.hint(&step, 0, &[5]);
    eventually("the hint was never issued", || harness.watcher.shared().costs().hints.reads == reads + 1);
    harness.route_at(&step, 0, &[]);
    harness.route_at(&step, 1, &[]);
    harness.route_at(&step, 2, &[]);
    // Layer 2's ring is read after it withdraws layer 1's hint.
    harness.hint(&step, 2, &[6]);
    eventually("the next ring was never read", || harness.watcher.shared().costs().hints.reads == reads + 2);
    ahead.finish();
    harness.route_at(&step, 3, &[]);
    step.complete().unwrap();
    assert_eq!(harness.in_tier(1, 5), None, "the hint routed past landed nothing");
}

/// Hints are read in the ring's order, best ranked first: where the records
/// beyond the demand reserve hold one (staging of a request's choices and
/// one more), the ring's first is read and the ring's second reads nothing;
/// the ring is read before the step's requests, which then find the first
/// read ahead.
#[test]
fn hints_are_read_in_the_rings_order() {
    let harness = layered("hint-order", EXPERTS, 12, 2, WORDS.choices as u32 + 1);
    harness.demand_step(4);
    // The tier's fill has landed its records beyond the reserve.
    let tiered = || (0..2).map(|layer| (0..EXPERTS).filter(|&expert| harness.in_tier(layer, expert).is_some()).count()).sum::<usize>();
    eventually("the tier never filled its records beyond the reserve", || tiered() == WORDS.choices as usize + 1);
    harness.staging_free();
    harness.tier_ready();
    let step = harness.step();
    harness.hint(&step, 0, &[6, 5]);
    eventually("the ring was never read", || harness.watcher.shared().costs().hints.reads == 1);
    harness.request_at(&step, 0, &[4]);
    harness.request_at(&step, 1, &[6]);
    step.complete().unwrap();
    let costs = harness.watcher.shared().costs();
    assert_eq!((costs.hints.candidates, costs.hints.reads), (2, 1));
    // Only the first step's one: expert 6 was read ahead.
    assert_eq!(costs.layers[1].unanticipated, 1);
    assert_eq!(harness.in_tier(1, 5), None, "the ring's second read nothing");
}

/// Hints take only records beyond the demand reserve: with none beyond
/// it, a hint reads nothing and every request still finds its records.
#[test]
fn hints_never_take_the_demand_reserve() {
    let harness = layered("reserve", EXPERTS, 12, 2, 4);
    harness.demand_step(0);
    let step = harness.step();
    harness.hint(&step, 0, &[4, 5, 6, 7]);
    harness.request_at(&step, 0, &[0, 1, 2, 3]);
    harness.request_at(&step, 1, &[4, 5, 6, 7]);
    step.complete().unwrap();
    let costs = harness.watcher.shared().costs();
    assert_eq!(costs.hints.reads, 0);
    // And the first step's one.
    assert_eq!(costs.layers[1].unanticipated, 4 + 1);
}

/// A lead counts the layer runs between in the device's order: a chain runs
/// layer-major, so its later step's hint at the layer before precedes its
/// earlier step's request.
#[test]
fn a_lead_counts_layer_runs_in_the_devices_order() {
    use super::core::At;
    use crate::experts::directory::Point;
    let at = |group, steps, layer, sequence| At::new(Point { group, layer, sequence }, steps, 3);
    // Lone steps: one layer of step 1 to layer 2 of step 2 is 4 runs.
    assert_eq!(at(1, 1, 1, 1).runs_until(at(2, 1, 2, 2)), 4);
    // A chain of steps 3 and 4: step 4 at layer 0, then step 3 at layer 1.
    assert_eq!(at(3, 2, 0, 4).runs_until(at(3, 2, 1, 3)), 1);
    assert_eq!(at(3, 2, 0, 3).runs_until(at(3, 2, 1, 3)), 2);
    // Past the chain: step 5 follows both steps' every layer.
    assert_eq!(at(3, 2, 2, 4).runs_until(at(5, 1, 0, 5)), 1);
}
