use crate::experts::watcher::low_word;
use super::stack::{RoutedRead, Stack};

use super::*;

/// Steps none of which begins: what a test's directory maps from.
fn steps() -> Steps {
    Steps::new(crate::planning::FlightDepth::SERIAL, 1, crate::experts::exchange::Pulse::new())
}

/// Steps 1 to 64 begun, each a group of one run whole, none ended: what a
/// test's device runs.
fn played() -> Steps {
    let steps = steps();
    steps.begin_through(64);
    steps
}

/// The point past every layer of group `group`'s steps, run whole.
fn past(group: Step) -> Point {
    Point::whole(group, usize::MAX, Step::MAX)
}

impl Directory {
    /// [`Self::retire`] beside steps 1 to 64 begun ([`played`]).
    fn retired(&mut self, progress: Point, commit: impl Fn(usize, BankRecord) -> i32, ending: impl Fn(Point) -> Ending) {
        self.retire(progress, commit, ending, &played())
    }

    /// What the ranking asks now, a prompt group holding the lent
    /// records reserved where `reserved`.
    fn promote(&mut self, reserved: bool) -> Wants {
        let lent = LentRecords::new();
        if reserved {
            assert!(lent.reserve());
        }
        steps().unbegun(|at| self.promotions(at, &mut lent.lock()))
    }

    /// Carry `promotion` through its copy to its completion: its expert.
    fn complete(&mut self, promotion: &Promotion) -> (usize, usize) {
        let lent = LentRecords::new();
        let tier = self.hold_tier(promotion.layer, promotion.expert).expect("a promoted expert is kept in the tier");
        self.copy(promotion.group, promotion.record, tier, 0, &mut lent.lock());
        self.promoted(vec![(promotion.group, promotion.record)], &steps(), &lent);
        (promotion.layer, promotion.expert)
    }

    /// [`Self::evict`] and [`Self::commit_shrink`] between flights.
    fn evicted(&mut self, group: usize, plan: &Shrink, last_read: impl Fn(usize, usize) -> Option<Step>) {
        steps().unbegun(|at| self.evict(group, plan, last_read, at));
    }

    fn committed(&mut self, group: usize, plan: &Shrink) {
        steps().unbegun(|at| self.commit_shrink(group, plan, at));
    }

    /// A claim of `layer`'s lacked `expert`: the record taken, none
    /// where short.
    fn land(&mut self, layer: usize, expert: usize, keep: usize) -> Option<HostRecord> {
        match self.claim_landing(layer, expert, keep, Taking::Out) {
            Claim::Taken(grant) => Some(HostRecord(grant.record())),
            Claim::Short => None,
            Claim::Kept(_) | Claim::Landing => panic!("({layer}, {expert}) is not lacked"),
        }
    }
}

/// Log step `sequence`'s read of `experts` of `layer` into `directory`'s
/// ranking, as the stack feeds it once the step completed.
fn log(directory: &Directory, sequence: Step, layer: usize, experts: &[usize]) {
    let read = RoutedRead { sequence, layer, experts: experts.to_vec(), choices: experts.len() as u64 };
    Reuse::of(&directory.ranking).read(&read, &vec![1; experts.len()], crate::experts::reuse::Evidence::Decode);
}

/// The directory names every streamed layer's map words where the plan
/// places its records and the device resolves its choices: at the
/// plan's home, so a placement's or a landing's word means one (layer,
/// expert) to both.
#[test]
fn the_directory_names_map_words_at_the_plans_homes() {
    use crate::planning::fixtures::{qwen4exp_fixture_definition_of, streamed_experts_manifest};
    let definition = qwen4exp_fixture_definition_of(3, 32);
    let manifest = streamed_experts_manifest(&definition);
    let selection = crate::ComponentSelection { head: false, vision: false };
    let backend = seismic::BackendName::Metal;
    let resident = crate::ModelLoadPlan::derive(&manifest, &definition, selection, crate::resident_layout(crate::ExecutionPath::Native, backend), crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS).unwrap();
    let reservation = seismic::Reservation { granule: 2 << 20, host_addressed: true, chunks: None };
    let substrate = crate::planning::BankSubstrate { backend, reservation, host: crate::planning::HostTier::Unified };
    let floor = resident.bank_floor(&definition, 1, crate::FlightDepth::PIPELINED, substrate).unwrap();
    let load = resident.with_expert_bank(&definition, floor, 1, crate::FlightDepth::PIPELINED, substrate).unwrap();
    let plan = load.expert_bank().unwrap();
    let records = (0..plan.groups().len()).map(|group| plan.records(group) as usize).collect::<Vec<_>>();
    let groups = plan.layers().iter().map(|layer| layer.group).collect::<Vec<_>>();
    let experts = plan.layers().iter().map(|layer| layer.experts as usize).collect::<Vec<_>>();
    let directory = Directory::new(&records, &groups, &experts, HostTier::replay(vec![0; records.len()]));
    assert!(plan.layers().len() > 1);
    for layer in 0..plan.layers().len() {
        assert_eq!(directory.home(layer) as u64, plan.home(layer), "layer {layer}");
    }
}

/// Layer `layer` of step `sequence`, a group of one.
fn at(sequence: Step, layer: usize) -> Point {
    Point::whole(sequence, layer, sequence)
}

/// A group seeds the most recent experts of a prompt, as the ranking
/// does before its reads tell otherwise (recency), into records outside
/// every layer's share: each seed whose scratch slot the group filled,
/// most recent first, takes the group's least recently read record whose
/// expert lies outside its own layer's share while it outranks it, by
/// the step, then (layer, expert); a share's records and the merge set's
/// pinned experts are never taken.
#[test]
fn seeding_keeps_the_most_recent_experts() {
    // Record 0 holds layer 0's expert 0, records 1 to 3 layer 1's experts
    // 0 to 2, last read at steps 2, 3, 9 and 7: layer 1's share of two
    // holds its experts 1 and 2, so its expert 0 lies outside it.
    let held = || {
        let mut directory = Directory::new(&[4], &[0, 0], &[8, 8], HostTier::replay(vec![0; 1]));
        directory.place(at(1, 0), 0, &[], &[0], |_| 0).unwrap();
        directory.place(at(1, 1), 1, &[], &[0, 1, 2], |_| 0).unwrap();
        directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
        for (sequence, layer, expert) in [(2, 0, 0), (3, 1, 0), (7, 1, 2), (9, 1, 1)] {
            log(&directory, sequence, layer, &[expert]);
        }
        let reuse = Reuse::of(&directory.ranking);
        assert_eq!((0..3).map(|expert| reuse.in_top(1, expert)).collect::<Vec<_>>(), [false, true, true]);
        assert!(reuse.in_top(0, 0));
        drop(reuse);
        directory
    };
    let mut directory = held();
    let (seeds, evicted) = directory.seed(at(2, 0), 0, &[(2, 5), (3, 4)], &[], |_| true, 0);
    assert_eq!(seeds, [(2, BankRecord(1))], "the most recent seed takes the one record outside every share");
    assert_eq!(evicted, [(1, 0)]);
    // A seed read at the same step as that record ranks below it when its
    // (layer, expert) does.
    let mut directory = held();
    let (seeds, evicted) = directory.seed(at(2, 0), 0, &[(5, 3)], &[], |_| true, 0);
    assert!(seeds.is_empty() && evicted.is_empty());
    // A pinned expert's record is never taken.
    let mut directory = held();
    let (seeds, evicted) = directory.seed(at(2, 0), 1, &[(3, 5)], &[0], |_| true, 0);
    assert!(seeds.is_empty() && evicted.is_empty());
    // A slot the group did not fill seeds nothing.
    let mut directory = held();
    let (seeds, _) = directory.seed(at(2, 0), 0, &[(2, 5), (3, 4)], &[], |expert| expert != 2, 0);
    assert_eq!(seeds.iter().map(|&(expert, _)| expert).collect::<Vec<_>>(), [3]);
}

/// Beside a host tier a prompt group is lent its scratch's lent bank
/// records as its conveyor starts serving it: a free one, or a held one
/// whose expert leaves the bank, every one the group's until it ends;
/// no seed takes a lent record, and the group's end returns every one
/// free and forgets its pending reads.
#[test]
fn a_prompt_group_is_lent_its_bank_records_and_returns_them_free() {
    let mut directory = Directory::new(&[4], &[0, 0], &[8, 8], tiered(&[0], 2, 8, 2));
    directory.place(at(1, 0), 0, &[], &[0], |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    assert_eq!(directory.held(0, 0), Some(0));
    // Records 0 and 1 are lent: 0 holds layer 0's 0, which leaves.
    let lent = LentRecords::new();
    assert!(lent.reserve());
    directory.lend(2, &lent.lock());
    assert_eq!(directory.held(0, 0), None);
    assert!((0..2).all(|record| directory.bank.records(0)[record] == Record::Lent { from: 2 }));
    directory.read_pending(1, 3, 2, 1);
    // Seeds take the records the group does not hold.
    let (seeds, evicted) = directory.seed(at(2, 1), 1, &[(5, 2), (6, 2), (7, 2)], &[], |_| true, 0);
    assert_eq!((seeds.iter().map(|&(_, record)| record).collect::<Vec<_>>(), evicted), (vec![BankRecord(2), BankRecord(3)], vec![]));
    directory.end_prompt_group(0);
    assert!((0..2).all(|record| directory.bank.records(0)[record] == Record::Free));
    assert!(directory.prompt_reads.is_empty(), "its reads rank nothing apart");
    // Free again: a placement takes them.
    directory.retired(past(2), |_, _| 2, |_| Ending::Ended);
    assert_eq!(directory.place(at(3, 0), 0, &[], &[1, 2], |_| 0).unwrap().records, [BankRecord(0), BankRecord(1)]);
}

/// While a prompt group holds the lent records reserved, no promotion
/// takes a free one and no release one held by an unwanted expert: the
/// wanted experts take the other records, or wait.
#[test]
fn no_promotion_is_planned_into_a_reserved_lent_record() {
    let landed = || {
        let mut directory = Directory::new(&[4], &[0], &[8], tiered(&[8], 1, 8, 2));
        for expert in 0..6 {
            let record = directory.land(0, expert, 0).unwrap();
            directory.landed(0, record, expert, Some(1));
        }
        log(&directory, 2, 0, &[0, 1, 2, 3, 4, 5]);
        directory
    };
    let records = |wants: &Wants| wants.promotions.iter().map(|promotion| promotion.record).collect::<Vec<_>>();
    let reserved = landed().promote(true);
    assert_eq!((records(&reserved), reserved.releases), (vec![BankRecord(2), BankRecord(3)], vec![]));
    assert_eq!(records(&landed().promote(false)), [BankRecord(0), BankRecord(1), BankRecord(2), BankRecord(3)]);
    // The bank holds four experts, every record; then the two in the
    // lent records leave the layer's share, the experts 4 and 5 read
    // in their place.
    let held = || {
        let mut directory = Directory::new(&[4], &[0], &[8], tiered(&[8], 1, 8, 2));
        for expert in 0..6 {
            let record = directory.land(0, expert, 0).unwrap();
            directory.landed(0, record, expert, Some(1));
        }
        log(&directory, 2, 0, &[0, 1, 2, 3]);
        for promotion in directory.promote(false).promotions {
            directory.complete(&promotion);
        }
        let holder = |record: usize| match directory.bank.records(0)[record] {
            Record::Held((_, expert)) => expert,
            other => panic!("record {record} holds {other:?}"),
        };
        let (lent, kept) = ([holder(0), holder(1)], [holder(2), holder(3)]);
        assert!(!lent.contains(&4) && !lent.contains(&5) && !kept.contains(&4) && !kept.contains(&5));
        for sequence in [3, 4] {
            log(&directory, sequence, 0, &[kept[0], kept[1], 4, 5]);
        }
        directory
    };
    assert_eq!(held().promote(true), Wants::default(), "a reserved record released from an unwanted expert");
    let unreserved = held().promote(false);
    let mut released = unreserved.releases.iter().map(|release| release.record).collect::<Vec<_>>();
    released.sort_unstable();
    assert_eq!((unreserved.promotions, released), (vec![], vec![BankRecord(0), BankRecord(1)]), "the unwanted experts' records released");
}

/// A group of steps 2 and 3 runs layer-major: step 3's request at layer
/// 0 retires step 2's fill at layer 0, not one placed at layer 1; step
/// 2's request at layer 1 retires step 3's at layer 0.
#[test]
fn a_group_retires_in_layer_major_order() {
    let mut directory = directory();
    let point = |layer, sequence| Point::whole(2, layer, sequence);
    directory.place(point(0, 2), 0, &[], &[1], |_| 0).unwrap();
    directory.place(point(1, 2), 1, &[], &[1], |_| 0).unwrap();
    directory.retired(point(0, 3), |_, _| 2, |_| Ending::Ended);
    assert_eq!((directory.held(0, 1), directory.held(1, 1)), (Some(0), None));
    directory.place(point(0, 3), 0, &[1], &[2], |_| 0).unwrap();
    directory.retired(point(1, 2), |_, record| [2, 2, 3][record.index()], |_| Ending::Ended);
    assert_eq!(directory.held(0, 2), Some(2));
    assert_eq!(directory.held(1, 1), None, "layer 1 of step 2 is not passed");
    directory.retired(past(2), |_, record| [2, 2, 3][record.index()], |_| Ending::Ended);
    assert_eq!(directory.held(1, 1), Some(1));
}

/// Records read in the same step are evicted by what they hold, the
/// lowest (layer, expert) first, wherever they lie: two directories that
/// placed the same experts in opposite records evict the same one.
#[test]
fn a_tie_is_broken_by_what_records_hold() {
    for order in [[5, 3], [3, 5]] {
        let mut directory = Directory::new(&[2], &[0], &[8], HostTier::replay(vec![0; 1]));
        for (step, expert) in order.into_iter().enumerate() {
            let step = step as Step + 1;
            directory.place(at(step, 0), 0, &[], &[expert], |_| 0).unwrap();
            directory.retired(past(step), |_, _| low_word(step), |_| Ending::Ended);
        }
        let placed = directory.place(at(3, 0), 0, &[], &[7], |_| 1).unwrap();
        assert_eq!(placed.evicted, [(0, 3, 1)]);
    }
}

/// One layer of two records reading 0, 1, 2, 0, 1 from empty: least
/// recently read misses every step; the farthest next read keeps 0 for
/// its return. Starting from 0 and 1 held, their first reads hit.
#[test]
fn a_replay_counts_each_rankings_misses() {
    let reads = [0, 1, 2, 0, 1]
        .into_iter()
        .enumerate()
        .map(|(step, expert)| RoutedRead {
            sequence: step as Step + 1,
            layer: 0,
            experts: vec![expert],
            choices: 1,
        })
        .collect::<Vec<_>>();
    let misses = |held: &[Vec<usize>], recency| -> Vec<u64> {
        replay(&[2], &[0], 4, held, &reads, recency).unwrap().misses
    };
    assert_eq!(replay(&[2], &[0], 4, &[vec![]], &reads, Recency::Reuse).unwrap().held, [vec![0, 1]]);
    assert_eq!(misses(&[vec![]], Recency::Reuse), [1, 1, 1, 1, 1]);
    assert_eq!(misses(&[vec![]], Recency::NextRead), [1, 1, 1, 0, 1]);
    assert_eq!(misses(&[vec![0, 1]], Recency::Reuse), [0, 0, 1, 1, 1]);
}

/// A shrink keeps the most recently read experts, wherever they lie:
/// the ones above the commitment move into records below it that the
/// least recent ones leave free, and a regrowth adds free records.
#[test]
fn a_shrink_keeps_the_most_recent_experts_below_the_commitment() {
    let mut directory = directory();
    directory.place(at(1, 0), 0, &[], &[0, 1], |_| 0).unwrap();
    directory.place(at(1, 1), 1, &[], &[0, 1], |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    // Records 0, 1 hold layer 0's experts 0, 1; records 2, 3 layer 1's.
    let reads = [[Some(5), Some(2)], [Some(7), Some(3)]];
    let plan = directory.plan_shrink(0, 2, |layer, expert| reads[layer][expert]);
    assert_eq!(plan.evicted, [(1, 1), (0, 1)], "the least recent two leave");
    assert_eq!(plan.moved, [(1, 0, 2, 1)], "layer 1's expert 0 moves to the record left free");
    assert_eq!(
        (directory.held(1, 0), directory.records()[0]),
        (Some(2), 4),
        "a plan changes nothing"
    );
    directory.evicted(0, &plan, |layer, expert| reads[layer][expert]);
    assert_eq!(
        (directory.held(0, 1), directory.held(1, 0), directory.records()[0]),
        (None, Some(2), 4),
        "the evictions leave, the moved expert stays where its bytes are"
    );
    directory.committed(0, &plan);
    assert_eq!((directory.held(0, 0), directory.held(1, 0)), (Some(0), Some(1)));
    assert_eq!((directory.held(0, 1), directory.held(1, 1)), (None, None));
    assert_eq!(directory.records()[0], 2);
    directory.grow(0, 4);
    let placed = directory.place(at(2, 0), 0, &[0], &[4, 5], |_| 9).unwrap();
    assert_eq!(placed.records, [BankRecord(2), BankRecord(3)], "the regrown records are free");
    assert!(placed.evicted.is_empty());
}

/// While the free records cover a placement, it takes them in record
/// order without ranking any held record; one they fall short of takes
/// every free record, then the least recent held ones.
#[test]
fn free_records_are_placed_without_ranking_the_held() {
    let mut directory = directory();
    directory.place(at(1, 0), 0, &[], &[0, 1], |_| unreachable!()).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    let placed = directory.place(at(2, 0), 0, &[], &[2], |_| unreachable!()).unwrap();
    assert_eq!(placed.records, [BankRecord(2)]);
    let stamps = [4, 3];
    let placed = directory.place(at(2, 1), 1, &[], &[0, 1], |record| stamps[record]).unwrap();
    assert_eq!(placed.records, [BankRecord(3), BankRecord(1)], "the last free record, then the least recent");
    assert_eq!(placed.evicted, [(0, 1, 3)]);
}

/// `count` words of a CPU tensor of `shape`, each `fill`, as a load's bank
/// words are.
fn cpu_words(shape: &[u64], fill: i32) -> crate::experts::Words {
    let cpu = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
    let tensor = seismic::Tensor::shared(&cpu, seismic::Element::i32(), shape).unwrap();
    crate::experts::Words::new(tensor, shape.iter().product(), fill).unwrap()
}

/// A host tier of `records` records for each group beside a device that
/// places from it, as a load opens one: each of `layers` layers of
/// `experts` experts has its bank map and its tier map, and group 0's
/// lender lends its bottom `lent` bank records.
fn tiered(records: &[u32], layers: usize, experts: u64, lent: usize) -> HostTier {
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    HostTier {
        records: records.to_vec(),
        lent: vec![0; records.len()],
        placement: Placement::Tier {
            bank_maps: (0..layers).map(|_| StampedMap::new(cpu_words(&[experts, map_words], 0))).collect(),
            tier_maps: (0..layers).map(|_| LandingMap::Tier(cpu_words(&[experts], -1))).collect(),
            lent: (0, lent),
        },
    }
}

/// The directory owns each expert's locations, bank and host tier, the
/// tier inclusive: every file miss lands into a ready tier record, which
/// holds it alone from then; a promotion of it into the bank keeps its
/// copy, banked; the bank evicting it costs nothing (its copy is held
/// alone at once, nothing written); a failed landing's record is ready
/// again; a shrink drops what it evicts to the tier.
#[test]
fn the_directory_keeps_the_host_tier_inclusive_of_the_bank() {
    let mut directory = Directory::new(&[2], &[0], &[8], HostTier::replay(vec![3]));
    let alone = |directory: &Directory| directory.host(0).alone()[0];
    // Experts 3 and 4 land from the file into ready records, read at
    // step 1; a third landing's read fails.
    let (three, four) = (directory.land(0, 3, 0).unwrap(), directory.land(0, 4, 0).unwrap());
    let failed = directory.land(0, 5, 0).unwrap();
    assert!(directory.lands(0, 5));
    assert_eq!(directory.land(0, 6, 0), None, "every record taken");
    directory.landed(0, three, 3, Some(1));
    directory.landed(0, four, 4, Some(1));
    directory.unlanded(0, failed);
    assert!(!directory.lands(0, 5));
    assert!(directory.in_tier(0, 3) && directory.in_tier(0, 4) && alone(&directory) == 2);
    assert_eq!(directory.land(0, 5, 1), None, "the one ready record is the demand reserve");
    // Step 2 reads 3 from the tier; the ranking promotes it into a free
    // record: its tier copy kept, banked.
    log(&directory, 2, 0, &[3]);
    let promoted = directory.promote(false);
    assert_eq!(promoted.promotions, [Promotion { group: 0, record: BankRecord(0), layer: 0, expert: 3 }], "4 is never read");
    assert_eq!(directory.complete(&promoted.promotions[0]), (0, 3));
    assert_eq!(directory.held(0, 3), Some(0));
    assert!(directory.in_tier(0, 3) && !directory.host(0).holds_alone((0, 3)) && alone(&directory) == 1);
    // A shrink evicts 3: its banked copy is held alone.
    let last_read = |_: usize, expert: usize| Some(expert as Step);
    let plan = directory.plan_shrink(0, 0, last_read);
    assert_eq!(plan.evicted, [(0, 3)]);
    directory.evicted(0, &plan, last_read);
    assert!(directory.in_tier(0, 3) && alone(&directory) == 2);
}

/// A landing into the host tier of an expert the bank already holds (here
/// beside a bank imported whole) keeps a banked copy: the tier
/// holds alone no more experts than the bank lacks, so the experts only
/// the file holds never underflow.
#[test]
fn a_landing_beside_a_whole_bank_is_banked() {
    let mut directory = Directory::new(&[2], &[0], &[2], HostTier::replay(vec![2]));
    directory.hold_homes();
    for expert in 0..2 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, None);
        assert!(directory.in_tier(0, expert) && !directory.host(0).holds_alone((0, expert)));
    }
    let lacked = 2 - directory.held_count(0);
    assert_eq!(lacked, 0);
    assert!(directory.alone(0) <= lacked, "the tier holds alone only what the bank lacks");
}

/// Each group keeps its reserve of tier records: the refill unmaps kept
/// records first in taking order, publishing the unmapping before it
/// reads the latest step begun, and a record it unmapped is ready only
/// once the device is past that step, so no step that could have
/// resolved it reads what lands in it.
#[test]
fn the_reserve_is_graced_through_its_layer_of_the_latest_step_begun() {
    let mut directory = Directory::new(&[2], &[0], &[8], HostTier::replay(vec![2]));
    for (expert, read) in [(1, 3), (2, 2)] {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, Some(read));
    }
    let victims = directory.refill_victims(1, false);
    directory.refill(victims, false, &crate::experts::watcher::Unbegun::at(6));
    assert!(!directory.in_tier(0, 2) && directory.in_tier(0, 1), "the expert read least recently unmapped first");
    assert_eq!(directory.land(0, 3, 0), None, "graced through layer 0 of step 5");
    directory.retired(at(5, 0), |_, _| 0, |_| Ending::Ended);
    assert_eq!(directory.land(0, 3, 0), None, "step 5's layer 0 still runs");
    directory.retired(at(6, 0), |_, _| 0, |_| Ending::Ended);
    assert!(directory.land(0, 3, 0).is_some(), "the device past layer 0 of step 5");
}

/// A wanted expert enters only once its reads pay for its copy: priced
/// past what one read weighs, it waits, and it is asked again when its
/// price falls.
#[test]
fn a_promotion_waits_until_its_reads_earn_its_copy() {
    let mut directory = Directory::new(&[3], &[0, 0], &[8, 8], HostTier::replay(vec![16]));
    for (layer, expert) in (0..2).flat_map(|layer| (0..8).map(move |expert| (layer, expert))) {
        let record = directory.land(layer, expert, 0).unwrap();
        directory.landed(layer, record, expert, None);
    }
    (0..2).for_each(|layer| directory.price_copies(layer, Some(2.)));
    log(&directory, 1, 0, &[4]);
    assert_eq!(directory.promote(false), Wants::default(), "one read earns no copy priced at two");
    assert!(!directory.settle_changes(), "nothing read or priced since, nothing asked");
    log(&directory, 2, 0, &[4]);
    assert!(directory.settle_changes(), "a read since the want was withheld asks again");
    // Where a hit saves nothing, no read earns the copy.
    directory.price_copies(0, None);
    log(&directory, 3, 0, &[4]);
    assert_eq!(directory.promote(false), Wants::default(), "a layer whose hit saves nothing promotes nothing");
    log(&directory, 4, 0, &[4]);
    assert!(!directory.settle_changes(), "no read earns such a want, so none withholds it and a read asks nothing again");
    directory.price_copies(0, Some(0.));
    assert_eq!(directory.promote(false).promotions.len(), 1, "the price fell, and the want is asked again");
}

/// A want is weighed against its victim by one measure, the stay's,
/// however the live candidate changes while promotions are asked.
/// Layer 0's victim, expert 1, was read at each of steps 1 to 40, and its
/// want, expert 2, at each of 100 to 163: with no stay measured, at the
/// longest half-life (256), the want weighs about 59 reads, the victim
/// about 27, so it earns its copy (at a price of one read) whichever
/// half-life ranks. Each layer's share is the one record its top expert
/// takes at a half-life of one and of 256, so only the half-life that
/// ranks changes, switched back and forth throughout. Weighed by the live
/// half-life, scores taken at one and weighed at another never earn: the
/// victim's at 256 lies over a thousand steps past the latest, which a
/// half-life of one weighs as unbounded.
#[test]
fn a_want_and_its_victim_weigh_at_one_half_life_while_the_live_one_switches() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    for round in 0..200 {
        let mut directory = Directory::new(&[2], &[0, 0], &[8, 8], HostTier::replay(vec![16]));
        for (layer, expert) in (0..2).flat_map(|layer| (0..8).map(move |expert| (layer, expert))) {
            let record = directory.land(layer, expert, 0).unwrap();
            directory.landed(layer, record, expert, None);
        }
        directory.place(at(1, 0), 0, &[], &[1], |_| 0).unwrap();
        directory.place(at(1, 1), 1, &[], &[0], |_| 0).unwrap();
        directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
        (0..2).for_each(|layer| directory.price_copies(layer, Some(1.)));
        for step in 1..=163 {
            match step {
                1..=40 => log(&directory, step, 0, &[1]),
                100.. => log(&directory, step, 0, &[2]),
                _ => {}
            }
            log(&directory, step, 1, &[0]);
        }
        let ranking = directory.ranking();
        let (done, switches) = (AtomicBool::new(false), AtomicU64::new(0));
        let asked = std::thread::scope(|scope| {
            scope.spawn(|| {
                while !done.load(Relaxed) {
                    let switch = switches.fetch_add(1, Relaxed);
                    Reuse::of(&ranking).go_live([1., 256.][switch as usize % 2]);
                }
            });
            while switches.load(Relaxed) < 2 {
                std::hint::spin_loop();
            }
            let asked = directory.promote(false);
            done.store(true, Relaxed);
            asked
        });
        assert_eq!(
            asked.releases,
            [Release { group: 0, record: BankRecord(0), layer: 0, expert: 1 }],
            "round {round}: the want earned its copy, and its victim is released"
        );
    }
}

/// The ranking promotes, across a group's layers, the experts its share
/// of each layer holds (each layer's top of its stack) that the bank
/// lacks and the tier keeps, into free records, which no step reads;
/// each want left without one releases the record of an expert outside
/// its own layer's share, least reused first, which a promotion takes
/// once released. A promotion's or a release's record is neither held
/// nor free until its copy completes or every step that may
/// read it has ended; what nothing changed asks nothing; a promotion
/// that ended without its expert leaves it wanted again.
#[test]
fn promotions_bring_each_layers_share_into_the_bank() {
    // Layers 0 and 1 share three records, a share of one each; their
    // tier keeps everything.
    let mut directory = Directory::new(&[3], &[0, 0], &[8, 8], HostTier::replay(vec![16]));
    for (layer, expert) in (0..2).flat_map(|layer| (0..8).map(move |expert| (layer, expert))) {
        let record = directory.land(layer, expert, 0).unwrap();
        directory.landed(layer, record, expert, None);
    }
    directory.place(at(1, 0), 0, &[], &[0, 1], |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    // Records 0 and 1 hold layer 0's 0 and 1, never read; layer 0's
    // share is 4, read at 1, and layer 1's is 5, read at 2.
    log(&directory, 1, 0, &[4]);
    log(&directory, 2, 1, &[5]);
    assert_eq!(
        directory.promote(false),
        Wants {
            promotions: vec![Promotion { group: 0, record: BankRecord(2), layer: 1, expert: 5 }],
            releases: vec![Release { group: 0, record: BankRecord(0), layer: 0, expert: 0 }],
        },
        "the free record taken by the want read latest, then a release for the want left without one"
    );
    assert_eq!(directory.held(0, 0), None, "the released expert left the bank");
    assert!(directory.in_tier(0, 0), "its tier copy kept");
    assert_eq!(directory.promote(false), Wants::default(), "nothing changed, nothing asked");
    assert!(directory.place(at(7, 0), 0, &[], &[6, 7], |_| 0).is_none(), "promotions' and releases' records are taken");
    assert_eq!(directory.complete(&Promotion { group: 0, record: BankRecord(2), layer: 1, expert: 5 }), (1, 5));
    assert_eq!(directory.held(1, 5), Some(2));
    directory.drain(Point::START, &steps());
    assert_eq!(
        directory.promote(false).promotions,
        [Promotion { group: 0, record: BankRecord(0), layer: 0, expert: 4 }],
        "the released record taken"
    );
    directory.unpromoted(0, BankRecord(0), &LentRecords::new());
    assert_eq!(directory.held(0, 4), None, "a promotion that copied nothing holds nothing");
    assert_eq!(directory.place(at(7, 0), 0, &[], &[6], |_| 0).unwrap().records, [BankRecord(0)], "and frees its record");
    // 4 is wanted again, and releases the one record outside a share (1).
    assert_eq!(
        directory.promote(false),
        Wants { promotions: vec![], releases: vec![Release { group: 0, record: BankRecord(1), layer: 0, expert: 1 }] }
    );
}

/// The stack counts every decode step it reads, counted into the live
/// decode or not, by depth: what a reserve is sized by, at the whole share,
/// whether or not the load records.
#[test]
fn the_stack_counts_every_decode_reads_misses_at_its_share() {
    let mut stack = Stack::alone(1, 8, 2);
    for (sequence, experts) in [(1, vec![0, 1]), (2, vec![0, 2]), (3, vec![3])] {
        stack.read(&RoutedRead { sequence, layer: 0, experts, choices: 2 }, None);
    }
    // At a share of 2: step 1 misses both, step 2 misses 2 (0 at
    // depth 1), step 3 misses 3; at a share of 1, step 2 misses 0 too.
    assert_eq!(stack.decoded().steps, 3);
    assert_eq!((stack.decoded().layers[0].at(1).1, stack.decoded().layers[0].at(2).1), (5, 4));
    assert_eq!(stack.live(), None, "none counted into the live decode");
}

/// A group's reserve keeps records free beyond its wants, its shares
/// the records less it: a promotion takes a free record at once, never
/// waiting for a release to drain, and the record its least reused
/// expert outside the share holds is released in its place.
#[test]
fn a_reserve_lets_a_promotion_copy_at_once() {
    let mut directory = Directory::new(&[3], &[0], &[8], HostTier::replay(vec![16]));
    for expert in 0..8 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, None);
    }
    directory.reserve(0, 1);
    assert_eq!(directory.shares(), [2], "three records, one of them reserved");
    log(&directory, 1, 0, &[0]);
    log(&directory, 2, 0, &[1]);
    let asked = directory.promote(false);
    assert_eq!(asked.promotions.len(), 2);
    assert!(asked.releases.is_empty(), "the third record is the reserve");
    for promotion in asked.promotions {
        directory.complete(&promotion);
    }
    // 2 enters the share and 0 leaves it: 2 takes the reserve's free
    // record at once, and 0's record is released to refill it.
    log(&directory, 3, 0, &[2]);
    let free = bank_record((0..3).find(|&record| directory.held(0, 0) != Some(record) && directory.held(0, 1) != Some(record)).unwrap());
    let zero = bank_record(directory.held(0, 0).unwrap());
    assert_eq!(
        directory.promote(false),
        Wants {
            promotions: vec![Promotion { group: 0, record: free, layer: 0, expert: 2 }],
            releases: vec![Release { group: 0, record: zero, layer: 0, expert: 0 }],
        }
    );
    directory.drain(Point::START, &steps());
    assert_eq!(directory.promote(false), Wants::default(), "the reserve is whole again");
}

/// The directory ranks its victims by reuse as its reads measure it,
/// recency the tie-break. Before any read it is recency: of two records
/// read at one step, the lesser (layer, expert) is taken first. Reads
/// that reward frequency (an expert read every third step, a fresh one
/// each step, which recency evicts it for) move it off recency: the
/// expert read once is taken before the one read every third step,
/// though read at one step, and a shrink keeps the more reused.
#[test]
fn victims_rank_by_reuse_with_recency_the_tie_break() {
    // Expert 0 in record 0, 59 in 1, both last read at 59.
    let held = |steps: Step| {
        let mut directory = Directory::new(&[2], &[0], &[128], HostTier::replay(vec![0; 1]));
        directory.place(at(1, 0), 0, &[], &[0, 59], |_| 0).unwrap();
        directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
        for step in 1..=steps {
            log(&directory, step, 0, &super::super::reuse::tests::every_third(step));
        }
        directory
    };
    let evicted = |mut directory: Directory| directory.place(at(60, 0), 0, &[], &[100], |_| 59).unwrap().evicted;
    assert_eq!(evicted(held(0)), [(0, 0, 59)], "recency");
    let measured = held(58);
    assert!(measured.half_life() > 0., "the reads measured a half-life");
    assert_eq!(evicted(held(58)), [(0, 59, 59)], "reuse");
    let plan = measured.plan_shrink(0, 1, |_, _| Some(59));
    assert_eq!((plan.evicted, plan.moved), (vec![(0, 59)], vec![]), "the shrink keeps the more reused");
}

/// Where the host tier is the bank's own records (unified memory's one
/// store, the bank's directory groups placing none), the ranking prices
/// the tier's records: reads that reward frequency move it off recency,
/// as at the bank's records beside a tier. Priced at the bank's none,
/// every candidate would miss every read and it would rank by recency
/// forever. A shrink of the tier shrinks its shares.
#[test]
fn the_one_stores_ranking_prices_its_records() {
    let mut directory =
        Directory::new(&[0], &[0], &[128], HostTier {
            records: vec![2],
            lent: vec![0],
            placement: Placement::Bank {
                landing_maps: vec![LandingMap::Bank {
                    map: std::sync::Arc::new(StampedMap::new(cpu_words(&[128, magnitude_kernels::constants::MAP_WORDS], 0))),
                    steps: std::sync::Arc::new(steps()),
                }],
            },
        });
    for step in 1..=60 {
        log(&directory, step, 0, &super::super::reuse::tests::every_third(step));
    }
    assert!(directory.half_life() > 0., "the reads measured a half-life at the store's records");
    let shared = |directory: &Directory| (0..128).filter(|&expert| directory.reuse_now().in_top(0, expert)).count();
    assert_eq!(shared(&directory), 2, "a share of the store's two records");
    directory.shrink_host(0, 1);
    assert_eq!(shared(&directory), 1, "shrunk with the store");
}

/// The one store holds each layer to its share of the records, as the
/// ranking measures each layer's: a layer holding past its share gives its
/// own least ranked expert back first, never another layer's less ranked
/// one, which a pool of the group's layers by rank would take.
#[test]
fn the_one_store_takes_back_a_layer_past_its_share_first() {
    let mut directory = one_store(4, 2, 16);
    log(&directory, 1, 1, &[0]);
    log(&directory, 2, 0, &[0]);
    log(&directory, 3, 0, &[1]);
    log(&directory, 4, 0, &[2]);
    for (layer, expert) in [(1, 0), (0, 0), (0, 1), (0, 2)] {
        let record = directory.land(layer, expert, 0).expect("a ready record");
        directory.landed(layer, record, expert, None);
    }
    let victims = directory.refill_victims(1, false)[0].iter().map(|&(_, staged)| staged).collect::<Vec<_>>();
    assert_eq!(victims, [(0, 0)], "layer 0 holds three of its share of two: its least ranked, not layer 1's");
}

/// A grown bank beside a full host tier: the tier holds more experts
/// alone than the bank's commitment leaves (its new record is free), so
/// what only the file holds is each layer's experts neither the bank nor
/// the tier holds, not those beyond the bank's slots less the tier's
/// (which underflowed).
#[test]
fn a_grown_bank_beside_a_full_tier_leaves_the_rest_to_the_file() {
    let mut directory = Directory::new(&[2], &[0], &[4], HostTier::replay(vec![4]));
    for expert in 0..4 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, Some(1));
    }
    // The device promotes 1 and 3 into the bank; 0 and 2 stay alone.
    directory.place(at(2, 0), 0, &[], &[1, 3], |_| 0).unwrap();
    directory.retired(past(2), |_, _| 2, |_| Ending::Ended);
    directory.grow(0, 3);
    let (experts, slots) = (4, directory.records()[0]);
    let (held, alone) = (directory.held_count(0), directory.alone(0));
    assert_eq!((slots, held, alone), (3, 2, 2));
    assert!(alone > experts - slots, "the old count underflows here");
    assert_eq!(experts - held - alone, 0, "the file holds no expert alone");
}

/// Two layers of one group of four records.
fn directory() -> Directory {
    Directory::new(&[4], &[0, 0], &[8, 8], HostTier::replay(vec![0; 1]))
}

/// A fill holds its expert only once the device committed it for its
/// step, observed when the device is past its layer; until then its
/// record is neither held nor free.
#[test]
fn a_fill_holds_its_expert_on_the_devices_commit() {
    let mut directory = directory();
    let placed = directory.place(at(1, 0), 0, &[], &[3, 5], |_| 0).unwrap();
    assert_eq!(placed.records, [BankRecord(0), BankRecord(1)]);
    assert_eq!(directory.held(0, 3), None);
    directory.retired(at(1, 0), |_, _| 1, |_| Ending::Ended);
    assert_eq!(directory.held(0, 3), None, "the device is not past the layer");
    // Record 0's fill committed, record 1's (a failed read) did not.
    directory.retired(at(1, 1), |_, record| [1, 0][record.index()], |_| Ending::Ended);
    assert_eq!(directory.held(0, 3), Some(0));
    assert_eq!(directory.held(0, 5), None);
    let next = directory.place(at(2, 0), 0, &[3], &[5, 6, 7], |_| 0).unwrap();
    assert_eq!(next.records, [BankRecord(1), BankRecord(2), BankRecord(3)], "the failed fill's record is free again");
    assert!(next.evicted.is_empty());
}

/// Misses take free records first, then the least recently read held
/// records of any layer of the group, never the request's own held
/// experts; an eviction is reported for its map entry.
#[test]
fn misses_evict_the_least_recently_read_records_but_the_requests_own() {
    let mut directory = directory();
    directory.place(at(1, 0), 0, &[], &[0, 1], |_| 0).unwrap();
    directory.place(at(1, 1), 1, &[], &[0, 1], |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    // Layer 0's experts read at steps 5 and 2, layer 1's at 3, 4.
    let reads = [5, 2, 3, 4];
    let placed = directory
        .place(at(6, 0), 0, &[1], &[2, 3], |record| reads[record])
        .unwrap();
    assert_eq!(placed.records, [BankRecord(2), BankRecord(3)]);
    assert_eq!(placed.evicted, [(1, 0, 3), (1, 1, 4)]);
    assert_eq!(directory.held(1, 0), None);
    assert_eq!(directory.held(0, 1), Some(1));
    // Records 2 and 3 are filling: two records remain for three misses.
    assert_eq!(
        directory.place(at(6, 1), 1, &[], &[4, 5, 6], |record| reads[record]),
        None
    );
}

/// A one-store directory of one group of `records` records over `layers`
/// layers of `experts` experts each, every record ready, nothing read.
fn one_store(records: u32, layers: usize, experts: u64) -> Directory {
    let map = || LandingMap::Bank {
        map: std::sync::Arc::new(StampedMap::new(cpu_words(&[experts, magnitude_kernels::constants::MAP_WORDS], 0))),
        steps: std::sync::Arc::new(steps()),
    };
    Directory::new(&[0], &vec![0; layers], &vec![experts as usize; layers], HostTier {
        records: vec![records],
        lent: vec![0],
        placement: Placement::Bank { landing_maps: (0..layers).map(|_| map()).collect() },
    })
}

/// A one-store directory as [`one_store`] on `steps`, none begun yet, and
/// the views its steps resolve through.
fn one_store_on(steps: &std::sync::Arc<Steps>, records: u32, layers: usize, experts: u64) -> (Directory, Vec<crate::experts::map::MapView>) {
    let maps = (0..layers)
        .map(|_| std::sync::Arc::new(StampedMap::new(cpu_words(&[experts, magnitude_kernels::constants::MAP_WORDS], 0))))
        .collect::<Vec<_>>();
    let views = maps.iter().map(|map| map.view().clone()).collect();
    let directory = Directory::new(&[0], &vec![0; layers], &vec![experts as usize; layers], HostTier {
        records: vec![records],
        lent: vec![0],
        placement: Placement::Bank { landing_maps: maps.into_iter().map(|map| LandingMap::Bank { map, steps: steps.clone() }).collect() },
    });
    (directory, views)
}

/// Land `experts` of layer 0 into `directory` before any step begins:
/// mapped from step 1, so every step resolves them. Each one's record.
fn landed_before(directory: &mut Directory, experts: &[usize]) -> Vec<u32> {
    experts
        .iter()
        .map(|&expert| {
            let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out) else { panic!("a ready record for {expert}") };
            grant.map();
            directory.landed(0, HostRecord(grant.record()), expert, None);
            grant.record()
        })
        .collect()
}

/// A prompt group's seeds into a full one store take its never-routed
/// experts first: of four records holding experts 0 to 3, every one
/// landed and none ready, 1 and 3 read by decode steps, a group's last
/// step seeds its routed 4 and 5 in place of 0 and 2, which no step ever
/// routed. Seeding never waits for a ready record. Once the device commits
/// the seeds and the group ends whole, the store keeps 4 and 5, mapped
/// from the first step not yet begun; 1 and 3 stay.
#[test]
fn a_prompt_seeds_a_full_store_in_place_of_its_never_routed_experts() {
    let steps = std::sync::Arc::new(steps());
    let (mut directory, views) = one_store_on(&steps, 4, 1, 16);
    landed_before(&mut directory, &[0, 1, 2, 3]);
    log(&directory, 1, 0, &[1]);
    log(&directory, 2, 0, &[3]);
    steps.begin_through(1000);
    let point = at(1000, 0);
    let (seeds, mut evicted) = directory.seed(point, 0, &[(4, 1000), (5, 1000)], &[4, 5], |_| true, 0);
    evicted.sort_unstable();
    assert_eq!(evicted, [(0, 0), (0, 2)], "the never-routed experts are the first victims");
    assert_eq!(seeds.len(), 2, "both seed into the full store");
    for expert in [0, 2] {
        assert_eq!(views[0].resolve(expert, 1000), None, "{expert} is unmapped from the seeding group on");
    }
    let records = seeds.iter().map(|&(expert, record)| (expert, record.index())).collect::<Vec<_>>();
    directory.retire(past(1000), |_, record| match records.iter().any(|&(_, of)| of == record.index()) {
        true => low_word(1000),
        false => 0,
    }, |_| Ending::Ended, &steps);
    for (expert, record) in records {
        assert_eq!(directory.located(0, expert), Located::Kept(record as u32), "the store keeps seed {expert} after the prompt");
        assert_eq!(views[0].resolve(expert, 1001), Some(record), "seed {expert} is mapped from the first step not begun");
    }
    for expert in [1, 3] {
        assert!(matches!(directory.located(0, expert), Located::Kept(_)), "decode's {expert} stays");
    }
}

/// The post-prompt ramp, from a routing trace: a prompt routes each layer
/// with a skewed use of its experts, and decode keeps routing by the same
/// use. The empty one store a prompt group's seeds leave once its step
/// commits them misses decode's first steps at a fraction of a store
/// filled in index order (as a covering tier lands, or a fill that knew no
/// read), the same size.
#[test]
fn the_store_a_prompt_seeds_misses_decode_far_less_than_an_index_ordered_one() {
    const EXPERTS: usize = 64;
    const RECORDS: usize = 16;
    const LAYERS: usize = 2;
    // A deterministic trace: expert e of a layer is used in proportion to
    // 1 / (1 + rank(e)), its rank a fixed permutation unrelated to e.
    let rank = |layer: usize, expert: usize| (expert * 37 + layer * 11) % EXPERTS;
    let weight = |layer: usize, expert: usize| 1. / (1. + rank(layer, expert) as f64);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut draw = |layer: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let total: f64 = (0..EXPERTS).map(|expert| weight(layer, expert)).sum();
        let mut at = (state >> 11) as f64 / (1u64 << 53) as f64 * total;
        (0..EXPERTS).find(|&expert| { at -= weight(layer, expert); at <= 0. }).unwrap_or(EXPERTS - 1)
    };
    let prompt = (0..LAYERS).map(|layer| (0..2048).map(|_| draw(layer)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let decode = (0..LAYERS).map(|layer| (0..400).map(|_| draw(layer)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let steps = std::sync::Arc::new(steps());
    let (mut directory, _) = one_store_on(&steps, (RECORDS * LAYERS) as u32, LAYERS, EXPERTS as u64);
    steps.begin_through(1000);
    // The group's one step routes the prompt; each layer seeds what it
    // routed, its use pending in the ranking until the group ends.
    for (layer, routes) in prompt.iter().enumerate() {
        let mut routed = vec![0u32; EXPERTS];
        routes.iter().for_each(|&expert| routed[expert] += 1);
        let used = (0..EXPERTS).filter(|&expert| routed[expert] > 0).collect::<Vec<_>>();
        for &expert in &used {
            directory.read_pending(layer, expert, 1000, routed[expert]);
        }
        let routed = used.iter().map(|&expert| (expert, 1000)).collect::<Vec<(usize, Step)>>();
        let (seeds, evicted) = directory.seed(at(1000, layer), layer, &routed, &[], |_| true, 0);
        assert_eq!((seeds.len(), evicted.len()), (RECORDS, 0), "layer {layer} seeds its share of the empty store");
    }
    directory.retire(past(1000), |_, _| low_word(1000), |_| Ending::Ended, &steps);
    let kept = |layer: usize, expert: usize| matches!(directory.located(layer, expert), Located::Kept(_));
    let stored = (0..LAYERS).flat_map(|layer| (0..EXPERTS).map(move |expert| (layer, expert)));
    assert_eq!(stored.filter(|&(layer, expert)| kept(layer, expert)).count(), RECORDS * LAYERS, "every seed committed");
    let misses = |held: &dyn Fn(usize, usize) -> bool| -> usize {
        let missed = |(layer, routes): (usize, &Vec<usize>)| {
            routes.iter().filter(|&&expert| !held(layer, expert)).count()
        };
        decode.iter().enumerate().map(missed).sum()
    };
    let by_use = misses(&kept);
    let by_index = misses(&|_, expert| expert < RECORDS);
    assert!(by_use * 2 < by_index, "the prompt's store misses {by_use}, the index-ordered one {by_index}");
}

/// A seed whose step ends tainted, or that the device never committed,
/// leaves its record ready and its expert lacked; what it evicted stays
/// evicted.
#[test]
fn an_uncommitted_seed_leaves_its_record_ready() {
    let steps = std::sync::Arc::new(steps());
    let (mut directory, views) = one_store_on(&steps, 2, 1, 16);
    landed_before(&mut directory, &[0, 1]);
    steps.begin_through(1000);
    let (seeds, evicted) = directory.seed(at(1000, 0), 0, &[(4, 1000)], &[4], |_| true, 0);
    assert_eq!((seeds.len(), evicted.len()), (1, 1));
    directory.retire(past(1000), |_, _| low_word(1000), |_| Ending::Tainted, &steps);
    assert_eq!(directory.located(0, 4), Located::Lacked);
    assert_eq!(views[0].resolve(4, 1001), None);
    assert_eq!(directory.host(0).ready(), 1, "the seed's record is ready again");
}

/// A running prompt group's planned hits keep their records: its step 500
/// plans hits on the store's experts 0, 1 and 3; a seed of the group whose
/// first step is 1001 (begun after 500) of 2 evicts 1, the store's least
/// ranked, unmapped from 1001 on. Step 500's planned hits still resolve to
/// their records: the invariant a prompt group's rounds rest on
/// (`Scratch::fix_rounds`), where a hit with neither a record nor a
/// scratch slot would read a slot no round filled.
#[test]
fn a_running_groups_planned_hits_keep_their_records_through_a_later_seed() {
    let steps = std::sync::Arc::new(steps());
    let (mut directory, views) = one_store_on(&steps, 3, 1, 16);
    landed_before(&mut directory, &[0, 1, 3]);
    steps.begin_through(1001);
    let planned = [0, 1, 3].map(|expert| (expert, views[0].resolve(expert, 500).expect("a hit the group's step plans")));
    directory.read_pending(0, 0, 500, 3);
    directory.read_pending(0, 3, 500, 2);
    let (seeds, evicted) = directory.seed(at(1001, 0), 0, &[(2, 1001)], &[2], |_| true, 0);
    assert_eq!((seeds.len(), evicted), (1, vec![(0, 1)]), "2 seeds in place of 1");
    assert_eq!(views[0].resolve(1, 1001), None, "1 is unmapped from the seeding group on");
    for (expert, record) in planned {
        assert_eq!(views[0].resolve(expert, 500), Some(record), "step 500's planned hit on expert {expert} lost its record");
    }
}
