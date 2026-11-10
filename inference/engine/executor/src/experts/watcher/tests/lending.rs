use super::*;

/// Where the host addresses the scratch, the slots no pass reads before
/// an expert's round carry it, at no conveyor record: the served layer's
/// experts take the slots of experts it never converts, and the next
/// layer's, read while it is served (a wavefront: step 0's front at layer
/// 1 before step 1's rounds at layer 0), never one the served layer
/// converts into, which its pass reads. Each round converts its experts
/// from the slots their bytes were read into. A conveyor of one round's
/// records.
#[test]
fn the_next_layers_experts_borrow_no_slot_the_served_layer_converts() {
    let conveying = Conveying::of(ROUND_RECORDS).lending_scratch();
    let harness = lending("lend-next", EXPERTS, 12, 2, STAGING, conveying, TIER, Placer::Bank, false);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    harness.publish(&steps[0], 0, &[0, 1]);
    harness.publish(&steps[1], 0, &[2]);
    // Slot 2 converts only at step 1, after step 0's experts.
    let first = harness.rounds_checked(&steps[0], 0).concat();
    // Each expert's round follows its read's landing, in no set order.
    let mut lent = Harness::from_slots(&first);
    lent.sort_unstable();
    assert_eq!(lent, [(0, 2), (1, 3)]);
    harness.publish(&steps[0], 1, &[4, 5]);
    assert_eq!(Harness::experts(&harness.rounds_checked(&steps[1], 0).concat()), [2]);
    harness.publish(&steps[1], 1, &[]);
    let next = harness.rounds_checked(&steps[0], 1).concat();
    assert_eq!(Harness::experts(&next), [4, 5]);
    // Slots 0 to 2 hold what layer 0's passes read while layer 1's experts
    // are read.
    let lent = Harness::from_slots(&next);
    assert!(!lent.is_empty() && lent.iter().all(|&(_, slot)| slot > 2), "{next:?}");
    harness.rounds(&steps[1], 1);
    for step in steps {
        step.complete().unwrap();
    }
}

/// A slot an earlier step of the group converts into is never lent to a
/// later step's expert: the earlier step's pass reads it. A slot a later
/// step converts into is lent to an earlier step's expert, whose round
/// converts it out first. A conveyor of one round's records.
#[test]
fn a_later_steps_experts_borrow_no_slot_an_earlier_step_converts() {
    let conveying = Conveying::of(ROUND_RECORDS).lending_scratch();
    let harness = lending("lend-steps", EXPERTS, 12, 1, STAGING, conveying, TIER, Placer::Bank, false);
    let steps = harness.group(&[StepRoute::Grouped; 2]);
    harness.publish(&steps[0], 0, &[0, 1, 2]);
    harness.publish(&steps[1], 0, &[3, 4, 5]);
    let first = harness.rounds_checked(&steps[0], 0).concat();
    // Each expert's round follows its read's landing, in no set order.
    let mut lent = Harness::from_slots(&first);
    lent.sort_unstable();
    assert_eq!(lent, [(0, 3), (1, 4), (2, 5)]);
    let second = harness.rounds_checked(&steps[1], 0).concat();
    assert_eq!(Harness::experts(&second), [3, 4, 5]);
    // Step 1 converts each of 3 to 5 into its own slot once step 0's
    // experts are converted out; none borrows one of slots 0 to 2.
    assert!(Harness::from_slots(&second).iter().all(|&(_, slot)| slot > 2), "{second:?}");
    for step in steps {
        step.complete().unwrap();
    }
}

/// Where no slot is free outright, an expert borrows the slot of one its
/// own step converts later, which is not yet read: its rounds put that
/// slot's expert in a round after the borrower's. All eight experts cold
/// at one step, a conveyor of one round's records.
#[test]
fn a_borrowed_slots_expert_converts_in_a_later_round() {
    let conveying = Conveying::of(ROUND_RECORDS).lending_scratch();
    let harness = lending("lend-composed", EXPERTS, 12, 1, STAGING, conveying, TIER, Placer::Bank, false);
    let step = harness.prompt_step();
    let cold = (0..EXPERTS).collect::<Vec<_>>();
    harness.publish(&step, 0, &cold);
    let rounds = harness.rounds_checked(&step, 0);
    assert_eq!(Harness::experts(&rounds.concat()), (0..EXPERTS as i32).collect::<Vec<_>>());
    let lent = Harness::from_slots(&rounds.concat());
    assert!(!lent.is_empty(), "no slot was lent: {rounds:?}");
    let round_of = |expert: i32| rounds.iter().position(|round| round.iter().any(|&(_, listed)| listed == expert)).unwrap();
    for (expert, slot) in lent {
        assert!(round_of(slot) > round_of(expert), "expert {slot} converts into the slot expert {expert} is converted from: {rounds:?}");
    }
    step.complete().unwrap();
}
