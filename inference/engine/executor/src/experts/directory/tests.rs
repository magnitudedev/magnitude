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

/// The prompt group `point` is a point of, its steps through `point`'s.
fn prompt_of(point: Point) -> crate::experts::watcher::ScratchGroup {
    crate::experts::watcher::ScratchGroup::at(point.group, point.sequence - point.group + 1)
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

    /// What the ranking asks now.
    fn promote(&mut self) -> Wants {
        steps().unbegun(|at| self.promotions(at, usize::MAX))
    }

    /// Carry `promotion` through its copy to its completion: its expert.
    fn complete(&mut self, promotion: &Promotion) -> (usize, usize) {
        let tier = self.hold_tier(promotion.layer, promotion.expert).expect("a promoted expert is kept in the tier");
        self.copy(promotion.group, promotion.record, tier, 0);
        self.promoted(vec![(promotion.group, promotion.record)], Point::START, &steps());
        (promotion.layer, promotion.expert)
    }

    /// [`Self::evict`] group `group`'s `records`, each held expert
    /// unmapped from the first step not yet begun, none written back.
    fn evicted(&mut self, group: usize, records: &[usize], last_read: impl Fn(usize, usize) -> Option<Step>) {
        let mut all = vec![Vec::new(); self.records().len()];
        all[group] = records.to_vec();
        assert!(steps().unbegun(|at| self.evict(&all, last_read, |_| at.into(), at)), "a shrink with nothing written back evicts at once");
    }

    /// A claim of `layer`'s lacked `expert`: the record taken, none
    /// where short.
    fn land(&mut self, layer: usize, expert: usize, keep: usize) -> Option<HostRecord> {
        match self.claim_landing(layer, expert, keep, Taking::Out, crate::experts::host_records::Refuses) {
            Claim::Taken(grant) => Some(HostRecord(grant.record())),
            Claim::Short => None,
            Claim::Kept(_) | Claim::Landing => panic!("({layer}, {expert}) is not lacked"),
        }
    }
}

/// Log step `sequence`'s read of `experts` of `layer` into `directory`'s
/// ranking, as the stack feeds it once the step completed.
fn log(directory: &Directory, sequence: Step, layer: usize, experts: &[usize]) {
    {
        let time = directory.time();
        let mut times = time.lock();
        if times.clock.latest() < Some(sequence) {
            times.register(sequence, crate::experts::horizon::Advance::Known(crate::experts::horizon::Tokens::new(1.0)));
        }
    }
    let read = RoutedRead { sequence, layer, experts: experts.to_vec(), choices: experts.len() as u64 };
    Reuse::of(&directory.ranking).read(&read, None, crate::experts::reuse::Evidence::Decode);
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
    let host = crate::planning::HostTier::Unified(crate::planning::Parking::InKernels);
    let substrate = crate::planning::bank_substrate(backend, seismic::NativeAllocations::EXACT, Some(host), &definition).unwrap();
    let floor = resident.bank_floor(&definition, 1, crate::FlightDepth::PIPELINED, substrate).unwrap();
    let load = resident.with_expert_bank(&definition, floor, None, 1, crate::FlightDepth::PIPELINED, substrate).unwrap();
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

/// A group seeds a prompt's experts into records outside the group's pool:
/// each seed whose scratch slot the group filled, most ranked first, takes
/// the group's least ranked record whose expert lies outside the pool while
/// it outranks it, by expected uses, then the step that last read it, then
/// (layer, expert); the pool's records and the merge set's pinned experts
/// are never taken.
#[test]
fn seeding_keeps_the_most_recent_experts() {
    // Record 0 holds layer 0's expert 0, records 1 to 3 layer 1's experts
    // 0 to 2. Steps 1 to 70 read layer 0's 0 and 7 (in no record) and
    // layer 1's 1 and 2, step 1 also layer 0's 1: the pool of four holds
    // the four read at every step, and layer 1's 0, never read, lies
    // outside the pool so a routed prompt seed displaces it (most recent
    // seed first).
    let held = || {
        let mut directory = Directory::new(&[4], &[0, 0], &[8, 8], HostTier::replay(vec![0; 1]));
        directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &[0], |_| 0).unwrap();
        directory.place(at(1, 1), prompt_of(at(1, 1)), 1, &[], &[0, 1, 2], |_| 0).unwrap();
        directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
        for step in 1..=70 {
            let zero: &[usize] = if step == 1 { &[0, 1, 7] } else { &[0, 7] };
            log(&directory, step, 0, zero);
            log(&directory, step, 1, &[1, 2]);
        }
        let reuse = Reuse::of(&directory.ranking);
        assert_eq!((0..3).map(|expert| reuse.in_top(1, expert)).collect::<Vec<_>>(), [false, true, true]);
        assert!(reuse.in_top(0, 0));
        drop(reuse);
        directory
    };
    let mut directory = held();
    let (seeds, evicted) = directory.seed(at(2, 0), prompt_of(at(2, 0)), 0, &[(2, 1, 5), (3, 1, 4)], &[], |_| true);
    assert_eq!(seeds, [(2, BankRecord(1))], "the most recent seed takes the one record outside the pool");
    assert_eq!(evicted, [(1, 0)]);
    // A seed ranked below the record outside the pool takes nothing: give
    // layer 1's 0 enough decode uses to outrank an unread prompt seed while
    // staying outside the 4-record pool.
    let mut directory = held();
    for step in 71..=100 {
        log(&directory, step, 0, &[0, 7]);
        log(&directory, step, 1, &[0, 1, 2]);
    }
    assert!(!directory.reuse_now().in_top(1, 0), "layer 1's 0 stays outside the pool");
    let (seeds, evicted) = directory.seed(at(2, 0), prompt_of(at(2, 0)), 0, &[(5, 1, 1)], &[], |_| true);
    assert!(seeds.is_empty() && evicted.is_empty());
    // A pinned expert's record is never taken.
    let mut directory = held();
    let (seeds, evicted) = directory.seed(at(2, 0), prompt_of(at(2, 0)), 1, &[(3, 1, 5)], &[0], |_| true);
    assert!(seeds.is_empty() && evicted.is_empty());
    // A slot the group did not fill seeds nothing.
    let mut directory = held();
    let (seeds, _) = directory.seed(at(2, 0), prompt_of(at(2, 0)), 0, &[(2, 1, 5), (3, 1, 4)], &[], |expert| expert != 2);
    assert_eq!(seeds.iter().map(|&(expert, _)| expert).collect::<Vec<_>>(), [3]);
}

/// Beside a host tier a prompt group is leased its scratch's lent bank
/// records as its conveyor starts serving it: a free one, or a held one
/// whose expert leaves the bank, every one the group's until it ends;
/// no seed takes a lent record, and the group's end returns every one
/// free and forgets its pending reads.
#[test]
fn a_prompt_group_is_lent_its_bank_records_and_returns_them_free() {
    let mut directory = Directory::new(&[4], &[0, 0], &[8, 8], tiered(&[0], 2, 8, 2));
    directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &[0], |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    assert_eq!(directory.held(0, 0), Some(0));
    // Records 0 and 1 are lent: 0 holds layer 0's 0, which leaves.
    assert!(directory.lease(0, crate::experts::watcher::ScratchGroup::at(2, 1)), "every lent record is free or held, so leased at once");
    assert_eq!(directory.held(0, 0), None);
    assert!((0..2).all(|record| directory.bank.records(0)[record] == Record::Lent { from: 2 }));
    directory.read_pending(1, 3, 2);
    // Seeds take the records the group does not hold.
    let (seeds, evicted) = directory.seed(at(2, 1), prompt_of(at(2, 1)), 1, &[(5, 1, 2), (6, 1, 2), (7, 1, 2)], &[], |_| true);
    assert_eq!((seeds.iter().map(|&(_, record)| record).collect::<Vec<_>>(), evicted), (vec![BankRecord(2), BankRecord(3)], vec![]));
    directory.end_prompt_group(0);
    assert!((0..2).all(|record| directory.bank.records(0)[record] == Record::Free));
    assert!(directory.prompt_reads.is_empty(), "its reads rank nothing apart");
    // Free again: a placement takes them.
    directory.retired(past(2), |_, _| 2, |_| Ending::Ended);
    assert_eq!(directory.place(at(3, 0), prompt_of(at(3, 0)), 0, &[], &[1, 2], |_| 0).unwrap().records, [BankRecord(0), BankRecord(1)]);
}

/// No promotion takes a lent bank record, the scratch's, and no release
/// one held (a home): the wanted experts take the other records, and the
/// ranking's pool is the records less the lent ones, so it wants no more
/// than they hold.
#[test]
fn no_promotion_is_planned_into_a_lent_record() {
    let mut directory = Directory::new(&[4], &[0], &[8], tiered(&[8], 1, 8, 2));
    directory.price_copies(0, Some(0.));
    assert_eq!(directory.reuse_now().records(0), 2, "the pool is the records less the lent ones");
    for expert in 0..6 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, Some(1));
    }
    log(&directory, 2, 0, &[0, 1, 2, 3, 4, 5]);
    let records = |wants: &Wants| wants.promotions.iter().map(|promotion| promotion.record).collect::<Vec<_>>();
    let asked = directory.promote();
    assert_eq!((records(&asked), asked.releases), (vec![BankRecord(2), BankRecord(3)], vec![]));
}

/// A group of steps 2 and 3 runs layer-major: step 3's request at layer
/// 0 retires step 2's fill at layer 0, not one placed at layer 1; step
/// 2's request at layer 1 retires step 3's at layer 0.
#[test]
fn a_group_retires_in_layer_major_order() {
    let mut directory = directory();
    let point = |layer, sequence| Point::whole(2, layer, sequence);
    directory.place(point(0, 2), prompt_of(point(0, 2)), 0, &[], &[1], |_| 0).unwrap();
    directory.place(point(1, 2), prompt_of(point(1, 2)), 1, &[], &[1], |_| 0).unwrap();
    directory.retired(point(0, 3), |_, _| 2, |_| Ending::Ended);
    assert_eq!((directory.held(0, 1), directory.held(1, 1)), (Some(0), None));
    directory.place(point(0, 3), prompt_of(point(0, 3)), 0, &[1], &[2], |_| 0).unwrap();
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
            directory.place(at(step, 0), prompt_of(at(step, 0)), 0, &[], &[expert], |_| 0).unwrap();
            directory.retired(past(step), |_, _| low_word(step), |_| Ending::Ended);
        }
        let placed = directory.place(at(3, 0), prompt_of(at(3, 0)), 0, &[], &[7], |_| 1).unwrap();
        assert_eq!(placed.evicted, [(0, 3, 1)]);
    }
}

/// One layer of two records reading 0, 1, 2, 0, 1 from empty and from
/// `[0, 1]` held: `NextRead` (Bélády's bound) keeps 0 at read 3 and hits
/// read 4; `Reuse`'s route order `(key, uses, last)` already separates
/// position 2 (expert 0, Q₁ = 5/12) from position 1 (expert 1, Q₁ = 1/6)
/// at read 3, matching `NextRead`'s exact miss vector; at the final read
/// `Reuse` keeps `[0, 1]` (0 has higher expected uses than 2) whereas
/// `NextRead`, seeing no future reads, breaks the tie by expert index.
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
    let run = |held: &[Vec<usize>], recency| replay(&[2], &[0], 4, held, &reads, recency).unwrap();
    assert_eq!(run(&[vec![]], Recency::NextRead).misses, [1, 1, 1, 0, 1]);
    assert_eq!(run(&[vec![0, 1]], Recency::NextRead).misses, [0, 0, 1, 0, 1]);
    let empty = run(&[vec![]], Recency::Reuse);
    assert_eq!(empty.misses, [1, 1, 1, 0, 1]);
    assert_eq!(empty.held, [vec![0, 1]]);
    let held = run(&[vec![0, 1]], Recency::Reuse);
    assert_eq!(held.misses, [0, 0, 1, 0, 1]);
    assert_eq!(held.held, [vec![0, 1]]);
}

/// A cyclic trace of period 3 over 2 records (20 periods = 60 reads),
/// from an empty start and from `[0, 1]` held (design §5 #8, §6): LRU
/// misses every read after the warm start, while `Reuse` measures the
/// period-3 cycle and breaks equal-N ties by `S` (sooner use), matching
/// `NextRead` (Bélády's bound: half the reads) from period 3 on and
/// ending with the exact same holdings `[1, 2]`.
#[test]
fn a_cyclic_trace_is_placed_as_belady_once_measured() {
    const PERIODS: usize = 20;
    let reads: Vec<RoutedRead> = (0..PERIODS * 3)
        .map(|step| RoutedRead {
            sequence: step as Step + 1,
            layer: 0,
            experts: vec![step % 3],
            choices: 1,
        })
        .collect();
    let expected = |prefix: [u64; 2]| -> Vec<u64> {
        let mut misses = prefix.to_vec();
        for position in 2..reads.len() {
            misses.push(u64::from(position % 2 == 0));
        }
        misses
    };
    for (label, start, prefix) in [
        ("empty", vec![vec![]], [1, 1]),
        ("held", vec![vec![0, 1]], [0, 0]),
    ] {
        let belady = replay(&[2], &[0], 3, &start, &reads, Recency::NextRead).unwrap();
        let reuse = replay(&[2], &[0], 3, &start, &reads, Recency::Reuse).unwrap();
        let want = expected(prefix);
        assert_eq!(belady.misses, want, "{label}: Bélády miss vector");
        assert_eq!(reuse.misses[..9], want[..9], "{label}: first three periods");
        assert_eq!(reuse.misses[6..], belady.misses[6..], "{label}: matches NextRead from period 3 on");
        assert_eq!(reuse.held, [vec![1, 2]], "{label}: final holdings");
        assert_eq!(belady.held, [vec![1, 2]], "{label}: Bélády final holdings");
    }
}

/// A shrink keeps the most recently read experts where their bytes are,
/// wherever they lie: it evicts the least recent ones, leaving holes, and
/// moves nothing; a regrowth backs the holes again, free.
#[test]
fn a_shrink_keeps_the_most_recent_experts_where_they_are() {
    let mut directory = directory();
    directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &[0, 1], |_| 0).unwrap();
    directory.place(at(1, 1), prompt_of(at(1, 1)), 1, &[], &[0, 1], |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    // Records 0, 1 hold layer 0's experts 0, 1; records 2, 3 layer 1's.
    let reads = [[Some(5), Some(2)], [Some(7), Some(3)]];
    let mut evicted = directory.evictions(0, 2, |layer, expert| reads[layer][expert]);
    evicted.sort_unstable();
    assert_eq!(evicted, [1, 3], "the least recent two leave: layer 0's expert 1, layer 1's expert 1");
    assert_eq!((directory.held(0, 1), directory.records()[0]), (Some(1), 4), "choosing changes nothing");
    directory.evicted(0, &evicted, |layer, expert| reads[layer][expert]);
    assert_eq!((directory.held(0, 0), directory.held(1, 0)), (Some(0), Some(2)), "the kept stay where their bytes are");
    assert_eq!((directory.held(0, 1), directory.held(1, 1)), (None, None));
    assert_eq!(directory.records()[0], 2);
    assert_eq!(directory.evictions(0, 2, |layer, expert| reads[layer][expert]), Vec::<usize>::new(), "at its target it evicts nothing");
    directory.grow(0, vec![1, 3]);
    assert_eq!(directory.records()[0], 4);
    let placed = directory.place(at(2, 0), prompt_of(at(2, 0)), 0, &[0], &[4, 5], |_| 9).unwrap();
    assert_eq!(placed.records, [BankRecord(1), BankRecord(3)], "the regrown records are free");
    assert!(placed.evicted.is_empty());
}

/// While the free records cover a placement, it takes them in record
/// order without ranking any held record; one they fall short of takes
/// every free record, then the least recent held ones.
#[test]
fn free_records_are_placed_without_ranking_the_held() {
    let mut directory = directory();
    directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &[0, 1], |_| unreachable!()).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    let placed = directory.place(at(2, 0), prompt_of(at(2, 0)), 0, &[], &[2], |_| unreachable!()).unwrap();
    assert_eq!(placed.records, [BankRecord(2)]);
    let stamps = [4, 3];
    let placed = directory.place(at(2, 1), prompt_of(at(2, 1)), 1, &[], &[0, 1], |record| stamps[record]).unwrap();
    assert_eq!(placed.records, [BankRecord(3), BankRecord(1)], "the last free record, then the least recent");
    assert_eq!(placed.evicted, [(0, 1, 3)]);
}

/// `count` words of a CPU tensor of `shape`, each `fill`, as a load's bank
/// words are.
fn cpu_words(shape: &[u64], fill: i32) -> crate::experts::Words {
    let cpu = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
    crate::experts::Words::on_cpu(&cpu, shape.iter().product(), fill)
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
        reserve: 0,
        placement: Placement::Tier {
            bank_maps: (0..layers).map(|_| StampedMap::new(cpu_words(&[experts, map_words], 0))).collect(),
            tier_maps: (0..layers).map(|_| LandingMap::Tier(cpu_words(&[experts], -1))).collect(),
            lent: (0, lent),
        },
    }
}

/// The directory owns each expert's locations, bank and host tier, the
/// tiers disjoint: every file miss lands into a ready tier record, which
/// holds it alone from then; a promotion of it into the bank leaves its
/// copy banked only until the next ask gives it up; a failed landing's
/// record is ready again; a shrink drops what it evicts to the file.
#[test]
fn the_directory_keeps_the_host_tier_disjoint_from_the_bank() {
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
    // record: its tier copy banked until the next ask gives it up.
    directory.price_copies(0, Some(0.));
    log(&directory, 2, 0, &[3]);
    let promoted = directory.promote();
    assert_eq!(promoted.promotions, [Promotion { group: 0, record: BankRecord(0), layer: 0, expert: 3 }], "4 is never read");
    assert_eq!(directory.complete(&promoted.promotions[0]), (0, 3));
    assert_eq!(directory.held(0, 3), Some(0));
    assert!(directory.in_tier(0, 3) && !directory.host(0).holds_alone((0, 3)) && alone(&directory) == 1);
    assert!(directory.settle_changes(), "the banked copy waits to be given up");
    directory.promote();
    assert!(!directory.in_tier(0, 3) && directory.in_tier(0, 4) && alone(&directory) == 1, "each expert in one place");
    // A shrink evicts 3: only the file holds it.
    let last_read = |_: usize, expert: usize| Some(expert as Step);
    let evicted = directory.evictions(0, 0, last_read);
    assert_eq!(evicted, [1, 0], "the free record first, then 3's");
    directory.evicted(0, &evicted, last_read);
    assert!(!directory.in_tier(0, 3) && alone(&directory) == 1);
    assert_eq!((directory.held(0, 3), directory.records()[0]), (None, 0));
}

/// A landing into the host tier of an expert the bank already holds (here
/// beside a bank imported whole) keeps a banked copy: the tier holds alone
/// no more experts than the bank lacks, so the experts only the file holds
/// never underflow. A tier with a record for every expert keeps its banked
/// copies past the next ask: no other expert could take their records.
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
    directory.promote();
    assert!((0..2).all(|expert| directory.in_tier(0, expert)), "a covering tier gives up nothing");
}

/// Each group keeps its reserve of tier records: the refill unmaps kept
/// records first in taking order, publishing the unmapping before it
/// reads the latest step begun, and a record it unmapped is ready only
/// once the device is past that step, so no step that could have
/// resolved it reads what lands in it.
#[test]
fn the_reserve_is_graced_through_its_layer_of_the_latest_step_begun() {
    let mut directory = Directory::new(&[2], &[0], &[8], HostTier::replay(vec![2]).reserving(1));
    for (expert, read) in [(1, 3), (2, 2)] {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, Some(read));
    }
    let victims = directory.refill_victims(true);
    directory.refill(victims, &crate::experts::watcher::Unbegun::lending_at(6));
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
    // Priced at four reads, past the turnover set by three moves of the
    // three-record pool (τ = 3·T / 4 ≤ 3 over steps 1..4).
    {
        let time = directory.time();
        let mut times = time.lock();
        for _ in 0..3 {
            times.moved(0, crate::experts::horizon::Tier::Compute);
        }
    }
    // Unpriced by default (`None`): no promotion runs before `price_copies`.
    log(&directory, 1, 0, &[4]);
    assert_eq!(directory.promote(), Wants::default(), "unpriced promotions are never treated as free");
    (0..2).for_each(|layer| directory.price_copies(layer, Some(4.)));
    assert_eq!(directory.promote(), Wants::default(), "no read earns a copy priced past the turnover");
    assert!(!directory.settle_changes(), "nothing read or priced since, nothing asked");
    log(&directory, 2, 0, &[4]);
    assert!(directory.settle_changes(), "a read since the want was withheld asks again");
    // Where a hit saves nothing, no read earns the copy.
    directory.price_copies(0, None);
    log(&directory, 3, 0, &[4]);
    assert_eq!(directory.promote(), Wants::default(), "a layer whose hit saves nothing promotes nothing");
    log(&directory, 4, 0, &[4]);
    assert!(!directory.settle_changes(), "no read earns such a want, so none withholds it and a read asks nothing again");
    directory.price_copies(0, Some(0.));
    assert_eq!(directory.promote().promotions.len(), 1, "the price fell, and the want is asked again");
}

/// A want is weighed against its victim by one measure, expected uses
/// within the group's turnover. Layer 0's victim, expert 1, was read at
/// each of steps 1 to 40, and its want, expert 2, at each of 100 to 163;
/// layer 1's 0 at every step. The group's pool of two records holds layer
/// 1's 0 and layer 0's 2, read at every step since, while expert 1, unread
/// for over a horizon, is expected as rarely as the experts never chosen:
/// the want outweighs its victim by more than its price of one read, so
/// it earns its copy, and its victim is released for it.
#[test]
fn a_want_earns_its_copy_and_its_victim_is_released() {
    let mut directory = Directory::new(&[2], &[0, 0], &[8, 8], HostTier::replay(vec![16]));
    for (layer, expert) in (0..2).flat_map(|layer| (0..8).map(move |expert| (layer, expert))) {
        let record = directory.land(layer, expert, 0).unwrap();
        directory.landed(layer, record, expert, None);
    }
    directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &[1], |_| 0).unwrap();
    directory.place(at(1, 1), prompt_of(at(1, 1)), 1, &[], &[0], |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    (0..2).for_each(|layer| directory.price_copies(layer, Some(1.)));
    for step in 1..=163 {
        match step {
            1..=40 => log(&directory, step, 0, &[1]),
            100.. => log(&directory, step, 0, &[2]),
            _ => {}
        }
        log(&directory, step, 1, &[0]);
        if step % 20 == 0 {
            directory.time().lock().moved(0, crate::experts::horizon::Tier::Compute);
        }
    }
    let asked = directory.promote();
    assert_eq!(asked.releases, [Release { group: 0, record: BankRecord(0), layer: 0, expert: 1 }], "the want earned its copy, and its victim is released");
}

/// The ranking promotes, across a group's layers, the experts its pool
/// holds that the bank lacks and the tier keeps, into free records, which
/// no step reads; each want left without one releases the record of an
/// expert outside the pool, least ranked first, which a promotion takes
/// once released. A promotion's or a release's record is neither held
/// nor free until its copy completes or every step that may
/// read it has ended; what nothing changed asks nothing; a promotion
/// that ended without its expert leaves it wanted again.
#[test]
fn promotions_bring_the_groups_pool_into_the_bank() {
    // Layers 0 and 1 pool three records; their tier keeps every expert.
    let mut directory = Directory::new(&[3], &[0, 0], &[8, 8], HostTier::replay(vec![16]));
    (0..2).for_each(|layer| directory.price_copies(layer, Some(0.)));
    for (layer, expert) in (0..2).flat_map(|layer| (0..8).map(move |expert| (layer, expert))) {
        let record = directory.land(layer, expert, 0).unwrap();
        directory.landed(layer, record, expert, None);
    }
    directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &[0, 1], |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    // Records 0 and 1 hold layer 0's 0 and 1, never read. Steps 1 to 40
    // read layer 0's 4 and layer 1's 5: the pool holds both, expected
    // alike, and nothing of the experts never read.
    for step in 1..=40 {
        log(&directory, step, 0, &[4]);
        log(&directory, step, 1, &[5]);
    }
    assert!((0..2).all(|layer| (0..8).filter(|&expert| directory.reuse_now().in_top(layer, expert)).count() == 1));
    assert_eq!(
        directory.promote(),
        Wants {
            promotions: vec![Promotion { group: 0, record: BankRecord(2), layer: 0, expert: 4 }],
            releases: vec![Release { group: 0, record: BankRecord(0), layer: 0, expert: 0 }],
        },
        "the free record taken by the first of the wants, then a release for the want left without one"
    );
    assert_eq!(directory.held(0, 0), None, "the released expert left the bank");
    assert!(directory.in_tier(0, 0), "its tier copy kept");
    assert_eq!(directory.promote(), Wants::default(), "nothing changed, nothing asked");
    assert!(directory.place(at(7, 0), prompt_of(at(7, 0)), 0, &[], &[6, 7], |_| 0).is_none(), "promotions' and releases' records are taken");
    assert_eq!(directory.complete(&Promotion { group: 0, record: BankRecord(2), layer: 0, expert: 4 }), (0, 4));
    assert_eq!(directory.held(0, 4), Some(2));
    directory.drain(Point::START, &steps());
    assert_eq!(
        directory.promote().promotions,
        [Promotion { group: 0, record: BankRecord(0), layer: 1, expert: 5 }],
        "the released record taken"
    );
    directory.unpromoted(0, BankRecord(0));
    assert_eq!(directory.held(1, 5), None, "a promotion that copied nothing holds nothing");
    assert_eq!(directory.place(at(7, 0), prompt_of(at(7, 0)), 0, &[], &[6], |_| 0).unwrap().records, [BankRecord(0)], "and frees its record");
    // 5 is wanted again, and releases the one record outside the pool (1).
    assert_eq!(
        directory.promote(),
        Wants { promotions: vec![], releases: vec![Release { group: 0, record: BankRecord(1), layer: 0, expert: 1 }] }
    );
}

/// The stack counts every decode step it reads, counted into the live
/// decode or not, by depth: what a reserve is sized by, at the whole pool,
/// whether or not the load records.
#[test]
fn the_stack_counts_every_decode_reads_misses_at_its_depth() {
    let mut stack = Stack::alone(1, 8, 2);
    for (sequence, experts) in [(1, vec![0, 1]), (2, vec![0, 2]), (3, vec![3])] {
        stack.read(&RoutedRead { sequence, layer: 0, experts, choices: 2 }, None, None);
    }
    // At 2 slots: step 1 misses both, step 2 misses 2 (0 at depth 1),
    // step 3 misses 3; at 1 slot, step 2 misses 0 too.
    assert_eq!(stack.decoded().steps, 3);
    assert_eq!((stack.decoded().layers[0].at(1).1, stack.decoded().layers[0].at(2).1), (5, 4));
    assert_eq!(stack.live(), None, "none counted into the live decode");
}

/// A group's reserve keeps records free beyond its wants, its pool the
/// records less it: a promotion takes a free record at once, never
/// waiting for a release to drain, and the record its least reused
/// expert outside the pool holds is released in its place.
#[test]
fn a_reserve_lets_a_promotion_copy_at_once() {
    let mut directory = Directory::new(&[3], &[0], &[8], HostTier::replay(vec![16]));
    directory.price_copies(0, Some(0.));
    for expert in 0..8 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, None);
    }
    directory.reserve(0, 1);
    assert_eq!(directory.reuse_now().records(0), 2, "three records, one of them reserved");
    for step in 1..=8 {
        log(&directory, step, 0, &[0, 1]);
    }
    let asked = directory.promote();
    assert_eq!(asked.promotions.len(), 2);
    assert!(asked.releases.is_empty(), "the third record is the reserve");
    for promotion in asked.promotions {
        directory.complete(&promotion);
    }
    // Step 9 reads 1 and 2: 2, read as 0 and 1 were at their first
    // reads, which every later step read again, enters the pool and 0,
    // unread, leaves it. 2 takes the reserve's free record at once, and
    // 0's record is released to refill it.
    log(&directory, 9, 0, &[1, 2]);
    let free = bank_record((0..3).find(|&record| directory.held(0, 0) != Some(record) && directory.held(0, 1) != Some(record)).unwrap());
    let zero = bank_record(directory.held(0, 0).unwrap());
    assert_eq!(
        directory.promote(),
        Wants {
            promotions: vec![Promotion { group: 0, record: free, layer: 0, expert: 2 }],
            releases: vec![Release { group: 0, record: zero, layer: 0, expert: 0 }],
        }
    );
    directory.drain(Point::START, &steps());
    assert_eq!(directory.promote(), Wants::default(), "the reserve is whole again");
}

/// `Directory::promotions` caps both promotions from free records and
/// victim releases to the copy engine's remaining `budget`, leaving
/// remaining held experts mapped and serving hits until copy-engine
/// headroom opens up, while keeping `settle_changes()` true so capped
/// wants are revisited once budget is available.
#[test]
fn promotions_and_releases_are_capped_by_copy_engine_budget() {
    // Part 1: 4 free records, 4 earning wants, budget = 2 -> only 2 promoted.
    let mut directory = Directory::new(&[4], &[0], &[8], HostTier::replay(vec![8]));
    directory.price_copies(0, Some(0.));
    for expert in 0..8 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, None);
    }
    for step in 1..=16 {
        log(&directory, step, 0, &[0, 1, 2, 3]);
    }
    let first = steps().unbegun(|at| directory.promotions(at, 2));
    assert_eq!(first.promotions.len(), 2, "at most budget promotions issued");
    assert!(first.releases.is_empty());
    assert!(directory.settle_changes(), "capped wants keep stirred true for the next budget window");
    // Budget 0 promotes nothing and still keeps stirred true.
    assert_eq!(steps().unbegun(|at| directory.promotions(at, 0)), Wants::default());
    assert!(directory.settle_changes());
    // Next budget window of 2 promotes the remaining 2 wants.
    let second = steps().unbegun(|at| directory.promotions(at, 2));
    assert_eq!(second.promotions.len(), 2, "remaining wants promoted once budget opens");
    assert!(!directory.settle_changes(), "no remaining wants once all 4 are promoted");
    for promotion in first.promotions.into_iter().chain(second.promotions) {
        directory.complete(&promotion);
    }
    assert!((0..4).all(|expert| directory.held(0, expert).is_some()));

    // Part 2: All 4 records are held by experts 0..4; now experts 4..8
    // become the top-4 wants. With budget = 2, only 2 victims are released
    // and the other 2 held experts stay mapped in VRAM.
    for step in 17..=80 {
        log(&directory, step, 0, &[4, 5, 6, 7]);
    }
    let third = steps().unbegun(|at| directory.promotions(at, 2));
    assert!(third.promotions.is_empty(), "no free records yet");
    assert_eq!(third.releases.len(), 2, "only budget victims released, not all 4");
    let still_held = (0..4).filter(|&expert| directory.held(0, expert).is_some()).count();
    assert_eq!(still_held, 2, "remaining 2 held experts stay mapped and serving hits");
    // While those 2 releases are still covering, another call with budget = 2
    // releases 0 additional victims.
    let fourth = steps().unbegun(|at| directory.promotions(at, 2));
    assert!(fourth.releases.is_empty(), "covering releases satisfy want_releases; no extra eviction");
    assert_eq!((0..4).filter(|&expert| directory.held(0, expert).is_some()).count(), 2);
}

/// The directory ranks its victims by their expected uses, recency the
/// tie-break. Before any read nothing is expected of either: of two
/// records read at one step, the lesser (layer, expert) is taken first.
/// Once reads measure reuse (an expert read at every step beside a fresh
/// one each step), the expert read again is expected again, the one the
/// log never read only as often as a fresh expert recurs, so it is taken
/// first, though both are read at one step, and a shrink keeps the more
/// reused.
#[test]
fn victims_rank_by_reuse_with_recency_the_tie_break() {
    // Expert 0 in record 0, 59 in 1, both last read at 59.
    let held = |steps: Step| {
        let mut directory = Directory::new(&[2], &[0], &[128], HostTier::replay(vec![0; 1]));
        directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &[0, 59], |_| 0).unwrap();
        directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
        for step in 1..=steps {
            log(&directory, step, 0, &[0, step as usize]);
        }
        directory
    };
    let evicted = |mut directory: Directory| directory.place(at(60, 0), prompt_of(at(60, 0)), 0, &[], &[100], |_| 59).unwrap().evicted;
    assert_eq!(evicted(held(0)), [(0, 0, 59)], "recency");
    let measured = held(58);
    assert_eq!(evicted(held(58)), [(0, 59, 59)], "reuse");
    assert_eq!(measured.evictions(0, 1, |_, _| Some(59)), [1], "the shrink keeps the more reused: 59's record leaves");
}

/// Where the host tier is the bank's own records (unified memory's one
/// store, the bank's directory groups placing none), the ranking prices
/// the tier's records: reads that reward frequency move it off recency,
/// as at the bank's records beside a tier. Priced at the bank's none,
/// every candidate would miss every read and it would rank by recency
/// forever. A shrink of the tier shrinks its pool.
#[test]
fn the_one_stores_ranking_prices_its_records() {
    let mut directory =
        Directory::new(&[0], &[0], &[128], HostTier {
            records: vec![2],
            lent: vec![0],
            reserve: 0,
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
    let shared = |directory: &Directory| (0..128).filter(|&expert| directory.reuse_now().in_top(0, expert)).count();
    assert_eq!(shared(&directory), 2, "a pool of the store's two records");
    let evicted = directory.host_evictions(0, 1);
    assert_eq!(evicted, [0], "one ready record");
    directory.evict_host(0, &evicted, |_| None);
    assert_eq!(shared(&directory), 1, "shrunk with the store");
}

/// The one store takes back the group's least ranked expert, whatever its
/// layer: layer 0 holding three of the four records (its records less the
/// reserve, three, its pool) gives none back while layer 1's lone expert,
/// read once before every read of them, ranks least.
#[test]
fn the_one_store_takes_back_the_groups_least_ranked_expert() {
    let mut directory = one_store_reserving(4, 2, 16, 1);
    log(&directory, 1, 1, &[0]);
    for step in 1..=8 {
        log(&directory, step, 0, &[0, 1, 2]);
    }
    for (layer, expert) in [(1, 0), (0, 0), (0, 1), (0, 2)] {
        let record = directory.land(layer, expert, 0).expect("a ready record");
        directory.landed(layer, record, expert, None);
    }
    let victims = directory.refill_victims(true)[0].iter().map(|&(_, staged)| staged).collect::<Vec<_>>();
    assert_eq!(victims, [(1, 0)], "the group's least ranked, not layer 0's least");
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
    directory.place(at(2, 0), prompt_of(at(2, 0)), 0, &[], &[1, 3], |_| 0).unwrap();
    directory.retired(past(2), |_, _| 2, |_| Ending::Ended);
    directory.grow(0, vec![2]);
    let (experts, slots) = (4, directory.records()[0]);
    let (held, alone) = (directory.held_count(0), directory.alone(0));
    assert_eq!((slots, held, alone), (3, 2, 2));
    assert!(alone > experts - slots, "the old count underflows here");
    assert_eq!(experts - held - alone, 0, "the file holds no expert alone");
}

/// A write-back pays only where the drive binds the step and the link,
/// carrying the copy over the tier's stay, does not take its place.
#[test]
fn a_write_back_pays_where_the_drive_binds() {
    let terms = |compute, drive, link| magnitude_cost_model::Terms { compute, drive, link, cpu: 0., memory: 0. };
    let drive_binds = WriteBack { terms: terms(1., 4., 1.), drive: 1., link: 1. };
    assert!(drive_binds.pays(8., 4.), "two drive reads a step saved for a quarter of a copy");
    assert!(!drive_binds.pays(0., 4.), "an expert never used again saves nothing");
    let link_binds = WriteBack { terms: terms(1., 4., 4.), ..drive_binds };
    assert!(!link_binds.pays(8., 4.), "the link binds: the copy only adds to it");
    let compute_binds = WriteBack { terms: terms(5., 4., 1.), ..drive_binds };
    assert!(!compute_binds.pays(8., 4.), "the drive hides under compute: nothing saved");
    let dear = WriteBack { link: 16., ..drive_binds };
    assert!(!dear.pays(8., 4.), "a copy dearer than the drive it saves");
}

/// A bank of two records beside a full tier of 12, short of the 16
/// experts, the bank holding layer 0's 1 and layer 1's 0 (the tier gave up
/// its copies), write-backs priced where `priced`: layer 0's 1 read at
/// steps 1 to 40 and every fourth step from 100, so a use of it is still
/// expected; layer 0's 2 at every step from 100, as layer 1's 0 at every
/// step, to 163.
fn written_back_victim(priced: bool) -> Directory {
    let mut directory = Directory::new(&[2], &[0, 0], &[8, 8], HostTier::replay(vec![12]));
    for (layer, expert) in (0..2).flat_map(|layer| (0..[8, 4][layer]).map(move |expert| (layer, expert))) {
        let record = directory.land(layer, expert, 0).unwrap();
        directory.landed(layer, record, expert, Some(1));
    }
    directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &[1], |_| 0).unwrap();
    directory.place(at(1, 1), prompt_of(at(1, 1)), 1, &[], &[0], |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    // The tier gives up its copies, ready once the device is past them,
    // two ready records a write-back takes without displacing a kept
    // expert.
    directory.promote();
    assert!(!directory.in_tier(0, 1) && !directory.in_tier(1, 0));
    directory.retire_graces(past(1), &played());
    assert_eq!(directory.located(0, 1), Located::Lacked);
    (0..2).for_each(|layer| directory.price_copies(layer, Some(1.)));
    let terms = magnitude_cost_model::Terms { compute: 0., drive: 1., link: 0., cpu: 0., memory: 0. };
    let price = priced.then_some(WriteBack { terms, drive: 1., link: 0. });
    (0..2).for_each(|layer| directory.price_write_backs(layer, price));
    for step in 1..=163 {
        match step {
            1..=40 => log(&directory, step, 0, &[1]),
            100.. if step % 4 == 0 => log(&directory, step, 0, &[1, 2]),
            100.. => log(&directory, step, 0, &[2]),
            _ => {}
        }
        log(&directory, step, 1, &[0]);
        if step % 20 == 0 {
            directory.time().lock().moved(0, crate::experts::horizon::Tier::Compute);
        }
    }
    assert!(
        directory
            .reuse_now()
            .expected(0, 1, crate::experts::horizon::Tier::Host, crate::experts::horizon::Reader::Move, None)
            .uses()
            > 0.,
        "a use of the victim is expected"
    );
    directory
}

/// A promotion's victim, released beside a host tier that lacks it, is
/// written back into a ready tier record where the write-back pays: the
/// release keeps its bytes while the copy runs, and the copy done, the
/// tier keeps the expert alone. Unpriced, it drops to the file.
#[test]
fn a_released_victim_is_written_back_where_it_pays() {
    for priced in [false, true] {
        let mut directory = written_back_victim(priced);
        let asked = played().unbegun(|at| directory.promotions(at, usize::MAX));
        assert_eq!(asked.releases, [Release { group: 0, record: BankRecord(0), layer: 0, expert: 1 }]);
        if !priced {
            assert!(directory.backs_queued().is_empty() && directory.located(0, 1) == Located::Lacked, "unpriced, only the file holds it");
            continue;
        }
        let [(0, record, (0, 1), tier)] = directory.backs_queued()[..] else {
            panic!("the released victim's write-back is queued: {:?}", directory.backs_queued());
        };
        assert_eq!(record, BankRecord(0));
        assert_eq!(directory.located(0, 1), Located::Lacked, "a write-back stays lacked so no parked step waits on the transfer queue");
        directory.back_copy(0, record, 7);
        assert_eq!(directory.backing(), [(0, record, 7)]);
        assert!(!directory.drain(past(64), &played()), "the release holds its bytes until the copy ends");
        directory.written_back(0, record, true);
        assert!(directory.in_tier(0, 1) && directory.host(0).holds_alone((0, 1)));
        assert_eq!(directory.located(0, 1), Located::Kept(tier.0));
        assert!(directory.drain(past(64), &played()), "the copy ended, the release drains");
    }
}

/// A decode demand for an expert whose write-back is queued or in flight on
/// the transfer queue never waits on the transfer queue: the expert stays
/// lacked and unclaimed in `InboundTable`, so the demand posts a file read
/// immediately into its own record, and when the superseded write-back ends
/// later it returns its reserved tier record ready without panicking.
#[test]
fn a_demand_for_an_expert_with_a_pending_write_back_reads_from_file_and_supersedes_it() {
    let mut directory = written_back_victim(true);
    let steps = steps();
    steps.begin_through(164);
    let asked = steps.unbegun(|at| directory.promotions(at, usize::MAX));
    assert_eq!(asked.releases, [Release { group: 0, record: BankRecord(0), layer: 0, expert: 1 }]);
    let [(0, record, (0, 1), back_tier)] = directory.backs_queued()[..] else {
        panic!("expected queued write-back: {:?}", directory.backs_queued());
    };
    assert_eq!(directory.located(0, 1), Located::Lacked, "pending write-back leaves expert lacked");
    assert!(directory.inbound().claim((0, 1)), "decode look claims the expert for a file read immediately");
    let park = steps.park(at(164, 0), 2);
    let crate::experts::host_records::Claim::Taken(demand_grant) =
        directory.claim_demand((0, 1), crate::experts::host_records::Demand::Posted, &[], &park)
    else {
        panic!("demand claim must take a tier record without waiting on the write-back");
    };
    let demand_tier = HostRecord(demand_grant.record());
    assert_ne!(demand_tier, back_tier, "demand never overwrites the write-back's reserved record");
    directory.landed(0, demand_tier, 1, Some(164));
    assert_eq!(directory.located(0, 1), Located::Kept(demand_tier.0));
    // Superseded write-back finishes later: returns `back_tier` to ready and drains cleanly.
    directory.back_copy(0, record, 7);
    directory.written_back(0, record, true);
    assert_eq!(directory.located(0, 1), Located::Kept(demand_tier.0), "demand landing stays kept");
    assert!(directory.drain(past(164), &steps), "superseded write-back's release drains cleanly");
}

/// Every exit of a banked expert is its one exit's: a seed never takes the
/// record of one whose write-back pays (it would overwrite its bytes at
/// once), and a shrink evicting it releases it into its write-back and
/// waits a boundary, evicting nothing, then evicts its record once the
/// copy ended and it drained. Unpriced, the shrink evicts at once.
#[test]
fn every_exit_of_a_banked_expert_is_priced() {
    let shrink = |directory: &mut Directory| played().unbegun(|at| directory.evict(&[vec![0]], |_, _| None, |_| at.into(), at));
    for priced in [false, true] {
        let mut directory = written_back_victim(priced);
        let seeded = directory.place(at(164, 0), prompt_of(at(164, 0)), 0, &[], &[5], |_| 0);
        assert_eq!(seeded.is_none(), priced, "a seed takes a record only where its victim's write-back does not pay");
        let mut directory = written_back_victim(priced);
        if !priced {
            assert!(shrink(&mut directory), "unpriced, the shrink evicts at once");
            continue;
        }
        assert!(!shrink(&mut directory), "a victim written back holds its record: the shrink waits");
        let [(0, record, (0, 1), _)] = directory.backs_queued()[..] else {
            panic!("the shrink's victim's write-back is queued: {:?}", directory.backs_queued());
        };
        assert_eq!(directory.held(0, 1), None, "it left the bank");
        assert!(!shrink(&mut directory), "and waits while the write-back runs");
        directory.back_copy(0, record, 7);
        directory.written_back(0, record, true);
        assert!(directory.in_tier(0, 1));
        assert!(shrink(&mut directory), "its released record is evicted once the copy ended");
        assert_eq!(directory.records(), [1]);
    }
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
    let placed = directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &[3, 5], |_| 0).unwrap();
    assert_eq!(placed.records, [BankRecord(0), BankRecord(1)]);
    assert_eq!(directory.held(0, 3), None);
    directory.retired(at(1, 0), |_, _| 1, |_| Ending::Ended);
    assert_eq!(directory.held(0, 3), None, "the device is not past the layer");
    // Record 0's fill committed, record 1's (a failed read) did not.
    directory.retired(at(1, 1), |_, record| [1, 0][record.index()], |_| Ending::Ended);
    assert_eq!(directory.held(0, 3), Some(0));
    assert_eq!(directory.held(0, 5), None);
    let next = directory.place(at(2, 0), prompt_of(at(2, 0)), 0, &[3], &[5, 6, 7], |_| 0).unwrap();
    assert_eq!(next.records, [BankRecord(1), BankRecord(2), BankRecord(3)], "the failed fill's record is free again");
    assert!(next.evicted.is_empty());
}

/// Misses take free records first, then the least recently read held
/// records of any layer of the group, never the request's own held
/// experts; an eviction is reported for its map entry.
#[test]
fn misses_evict_the_least_recently_read_records_but_the_requests_own() {
    let mut directory = directory();
    directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &[0, 1], |_| 0).unwrap();
    directory.place(at(1, 1), prompt_of(at(1, 1)), 1, &[], &[0, 1], |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    // Layer 0's experts read at steps 5 and 2, layer 1's at 3, 4.
    let reads = [5, 2, 3, 4];
    let placed = directory
        .place(at(6, 0), prompt_of(at(6, 0)), 0, &[1], &[2, 3], |record| reads[record])
        .unwrap();
    assert_eq!(placed.records, [BankRecord(2), BankRecord(3)]);
    assert_eq!(placed.evicted, [(1, 0, 3), (1, 1, 4)]);
    assert_eq!(directory.held(1, 0), None);
    assert_eq!(directory.held(0, 1), Some(1));
    // Records 2 and 3 are filling: two records remain for three misses.
    assert_eq!(
        directory.place(at(6, 1), prompt_of(at(6, 1)), 1, &[], &[4, 5, 6], |record| reads[record]),
        None
    );
}

/// A one-store directory of one group of `records` records over `layers`
/// layers of `experts` experts each, every record ready, nothing read,
/// keeping a reserve of `reserve` records.
fn one_store_reserving(records: u32, layers: usize, experts: u64, reserve: usize) -> Directory {
    let map = || LandingMap::Bank {
        map: std::sync::Arc::new(StampedMap::new(cpu_words(&[experts, magnitude_kernels::constants::MAP_WORDS], 0))),
        steps: std::sync::Arc::new(steps()),
    };
    Directory::new(&[0], &vec![0; layers], &vec![experts as usize; layers], HostTier {
        records: vec![records],
        lent: vec![0],
        reserve,
        placement: Placement::Bank { landing_maps: (0..layers).map(|_| map()).collect() },
    })
}

/// A one-store directory as [`one_store_reserving`] on `steps`, none begun yet, and
/// the views its steps resolve through.
fn one_store_on(steps: &std::sync::Arc<Steps>, records: u32, layers: usize, experts: u64) -> (Directory, Vec<crate::experts::map::MapView>) {
    lending_store_on(steps, records, (0, 0), layers, experts)
}

/// [`one_store_on`], its bottom records lent to prompt groups and a
/// reserve kept, `(lent, reserve)`.
fn lending_store_on(steps: &std::sync::Arc<Steps>, records: u32, (lent, reserve): (u32, usize), layers: usize, experts: u64) -> (Directory, Vec<crate::experts::map::MapView>) {
    let maps = (0..layers)
        .map(|_| std::sync::Arc::new(StampedMap::new(cpu_words(&[experts, magnitude_kernels::constants::MAP_WORDS], 0))))
        .collect::<Vec<_>>();
    let views = maps.iter().map(|map| map.view().clone()).collect();
    let directory = Directory::new(&[0], &vec![0; layers], &vec![experts as usize; layers], HostTier {
        records: vec![records],
        lent: vec![lent],
        reserve,
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
            let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, crate::experts::host_records::Refuses) else { panic!("a ready record for {expert}") };
            grant.map();
            directory.landed(0, HostRecord(grant.record()), expert, None);
            grant.record()
        })
        .collect()
}

/// A prompt group's seeds into a full one store take its never-routed
/// experts first: of four records holding experts 0 to 3, every one
/// landed and none ready, 1 and 3 read by every decode step, a group's
/// last step seeds its routed 4 and 5 in place of 0 and 2, which no step
/// ever routed. Seeding never waits for a ready record. Once the device
/// commits the seeds and the group ends whole, the store keeps 4 and 5,
/// mapped from the first step not yet begun; 1 and 3 stay.
#[test]
fn a_prompt_seeds_a_full_store_in_place_of_its_never_routed_experts() {
    let steps = std::sync::Arc::new(self::steps());
    let (mut directory, views) = one_store_on(&steps, 4, 1, 16);
    landed_before(&mut directory, &[0, 1, 2, 3]);
    for step in 1..=8 {
        log(&directory, step, 0, &[1, 3]);
    }
    steps.begin_through(1000);
    let point = at(1000, 0);
    let (seeds, mut evicted) = directory.seed(point, prompt_of(point), 0, &[(4, 1, 1000), (5, 1, 1000)], &[4, 5], |_| true);
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
    let steps = std::sync::Arc::new(self::steps());
    let (mut directory, _) = one_store_on(&steps, (RECORDS * LAYERS) as u32, LAYERS, EXPERTS as u64);
    steps.begin_through(1000);
    // The group's one step routes the prompt; each layer seeds what it
    // routed, its use pending in the ranking until the group ends.
    for (layer, routes) in prompt.iter().enumerate() {
        let mut routed = vec![0u32; EXPERTS];
        routes.iter().for_each(|&expert| routed[expert] += 1);
        let used = (0..EXPERTS).filter(|&expert| routed[expert] > 0).collect::<Vec<_>>();
        for &expert in &used {
            directory.read_pending(layer, expert, 1000);
        }
        let routed = used.iter().map(|&expert| (expert, routed[expert], 1000)).collect::<Vec<(usize, u32, Step)>>();
        let (seeds, _evicted) = directory.seed(at(1000, layer), prompt_of(at(1000, layer)), layer, &routed, &[], |_| true);
        assert!(!seeds.is_empty(), "layer {layer} seeds into the store");
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
    let steps = std::sync::Arc::new(self::steps());
    let (mut directory, views) = one_store_on(&steps, 2, 1, 16);
    landed_before(&mut directory, &[0, 1]);
    steps.begin_through(1000);
    let (seeds, evicted) = directory.seed(at(1000, 0), prompt_of(at(1000, 0)), 0, &[(4, 1, 1000)], &[4], |_| true);
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
    let steps = std::sync::Arc::new(self::steps());
    let (mut directory, views) = one_store_on(&steps, 3, 1, 16);
    landed_before(&mut directory, &[0, 1, 3]);
    steps.begin_through(1001);
    let planned = [0, 1, 3].map(|expert| (expert, views[0].resolve(expert, 500).expect("a hit the group's step plans")));
    directory.read_pending(0, 0, 500);
    directory.read_pending(0, 3, 500);
    let (seeds, evicted) = directory.seed(at(1001, 0), prompt_of(at(1001, 0)), 0, &[(2, 1, 1001)], &[2], |_| true);
    assert_eq!((seeds.len(), evicted), (1, vec![(0, 1)]), "2 seeds in place of 1");
    assert_eq!(views[0].resolve(1, 1001), None, "1 is unmapped from the seeding group on");
    for (expert, record) in planned {
        assert_eq!(views[0].resolve(expert, 500), Some(record), "step 500's planned hit on expert {expert} lost its record");
    }
}

/// A one store's shape: 40 records over 4 layers of 64 experts, lending
/// its bottom 16 to prompt groups and keeping a reserve of 4.
const LENDING: (u32, (u32, usize), usize, u64) = (40, (16, 4), 4, 64);

/// [`LENDING`]'s store on fresh steps, every step to 1,000 begun.
fn lending_store() -> Directory {
    let steps = std::sync::Arc::new(self::steps());
    let (records, lend, layers, experts) = LENDING;
    let (directory, _) = lending_store_on(&steps, records, lend, layers, experts);
    steps.begin_through(1000);
    directory
}

/// The one store's pool is decode's capacity: its records less the
/// reserve, less the lease a prompt group holds claimed. The lease shrinks
/// the pool by the records it took; once the group returns it every lent
/// record is ready, none claimed, and the pool is the whole capacity
/// again, so decode keeps experts in every record the store backs but the
/// reserve's.
#[test]
fn a_lease_shrinks_the_pool_only_while_a_prompt_group_holds_it() {
    let (records, (lent, reserve), _, _) = LENDING;
    let (records, lent) = (records as usize, lent as usize);
    let mut directory = lending_store();
    let pool = |directory: &Directory| directory.reuse_now().records(0);
    assert_eq!(pool(&directory), records - reserve, "no lend held: the records less the reserve");
    assert!(directory.lease(0, prompt_of(at(1000, 0))), "every lent record is ready, so leased at once");
    let host = directory.host(0);
    assert_eq!((host.claimed(), host.capacity()), (lent, records - reserve - lent), "the lease held");
    assert_eq!(pool(&directory), records - lent - reserve, "present - claimed - reserve");
    directory.end_prompt_group(0);
    let host = directory.host(0);
    assert!((0..lent as u32).all(|record| !host.is_claimed(record)), "no lent record stays claimed");
    assert_eq!((host.ready(), host.claimed(), host.capacity()), (records, 0, records - reserve), "every lent record returned ready");
    assert_eq!(pool(&directory), records - reserve, "the pool is the whole capacity again");
}

/// While a prompt group holds its lend, its seeds take only the records
/// outside the lend and the reserve (capacity is pooled across layers,
/// design §3.7): no seed ever takes a lent record, and the reserve stays
/// ready.
#[test]
fn seeds_fill_no_layer_past_the_capacity_a_lend_leaves() {
    let (records, (lent, reserve), layers, experts) = LENDING;
    let mut directory = lending_store();
    assert!(directory.lease(0, prompt_of(at(1000, 0))), "every lent record is ready, so leased at once");
    let available = records as usize - lent as usize - reserve;
    let routed = (0..experts as usize).map(|expert| (expert, 1, 1000)).collect::<Vec<(usize, u32, Step)>>();
    let mut seeded = 0usize;
    for layer in 0..layers {
        let (seeds, evicted) = directory.seed(at(1000, layer), prompt_of(at(1000, layer)), layer, &routed, &[], |_| true);
        assert!(seeds.iter().all(|&(_, BankRecord(record))| record >= lent), "no seed takes a lent record");
        seeded = seeded + seeds.len() - evicted.len();
        assert!(seeded <= available, "seeded {seeded} past available {available}");
    }
    assert_eq!(seeded, available, "the pooled capacity outside the lend and reserve is filled");
    assert_eq!(directory.host(0).ready(), reserve, "the reserve stays ready");
}

/// A read-ahead claim tests against its own `keep`, not the whole reserve
/// (design §3.7, §6, #1): with nothing kept in the host tier and all three
/// records ready (equal to the reserve), a read-ahead with `keep = 2` finds
/// a ready record past its keep and admits, whereas `keep = 3` (no ready
/// record past the keep and nothing kept to displace) is short.
#[test]
fn a_read_ahead_into_a_ready_record_past_the_keep_admits() {
    use crate::experts::horizon::{Advance, Tokens};

    let mut directory = Directory::new(&[1], &[0], &[8], HostTier::replay(vec![3]).reserving(3));
    log(&directory, 1, 0, &[0]);
    directory.time().lock().register(2, Advance::Known(Tokens::new(1.0)));
    assert_eq!((directory.host(0).ready(), directory.host(0).reserve()), (3, 3));
    let admitted = directory.claim_ahead(0, 5, 2, Taking::Ahead, 2, 0.5);
    assert!(matches!(admitted, Claim::Taken(_)), "a ready record past the claim's keep admits");
    let short = directory.claim_ahead(0, 6, 3, Taking::Ahead, 2, 1.0);
    assert!(matches!(short, Claim::Short), "at the whole reserve with nothing kept, short admits none");
}

/// Read-ahead admission (`Reuse::expected_with` and `Directory::claim_landing`,
/// design §3.7, §6, finding 1):
/// 1. `expected_with` replaces lead `g_s`'s term by `p`, never mixes: when
///    the calibrated lead-1 probability is `q = 0.5`, `expected_with(..., 0.5)`
///    equals `expected(...)` (lead term `0.5`, never `0.75`).
/// 2. With no ready record past `keep`, `claim_landing` displaces a kept
///    record at the host tier's move `τ` iff the candidate's rank exceeds
///    the victim's.
#[test]
fn a_read_ahead_displaces_only_a_lower_rank() {
    use crate::experts::horizon::{Advance, Reader, Tier, Tokens};

    // Part 1: p = q replaces lead 1's term cleanly (never mixes p and q).
    let directory = Directory::new(&[1], &[0], &[2], HostTier::replay(vec![2]).reserving(1));
    // Steps 1..=3: 0, 0, 1 -> at step 3, route[1] has seen=2, used=1; with
    // Bayesian shrinkage `(used + prior) / (seen + 1)` and prior=1, q = 2/3.
    log(&directory, 1, 0, &[0]);
    log(&directory, 2, 0, &[0]);
    log(&directory, 3, 0, &[1]);
    directory.time().lock().register(4, Advance::Known(Tokens::new(1.0)));
    {
        let reuse = directory.reuse_now();
        let q = 2.0 / 3.0;
        let base = reuse.expected(0, 1, Tier::Host, Reader::Move, None).uses();
        let with_q = reuse.expected_with(0, 1, Tier::Host, 4, q).uses();
        let with_one = reuse.expected_with(0, 1, Tier::Host, 4, 1.0).uses();
        let with_zero = reuse.expected_with(0, 1, Tier::Host, 4, 0.0).uses();
        assert!((with_q - base).abs() < 1e-9, "p = q must keep lead term {q}: base={base}, with_q={with_q}");
        assert!((with_one - base - (1.0 - q)).abs() < 1e-9, "p = 1.0 adds {} over q={q}: base={base}, with_one={with_one}", 1.0 - q);
        assert!((base - with_zero - q).abs() < 1e-9, "p = 0.0 subtracts {q} from q={q}: base={base}, with_zero={with_zero}");
    }

    // Part 2: victim test at Tier::Host, Reader::Move with keep = reserve = 1.
    let mut hot_tier = Directory::new(&[1], &[0], &[8], HostTier::replay(vec![2]).reserving(1));
    for step in 1..=10 {
        log(&hot_tier, step, 0, &[0]);
    }
    let rec0 = hot_tier.land(0, 0, 0).unwrap();
    hot_tier.landed(0, rec0, 0, Some(10));
    hot_tier.time().lock().register(11, Advance::Known(Tokens::new(1.0)));
    assert!(
        matches!(hot_tier.claim_ahead(0, 2, 1, Taking::Ahead, 11, 0.5), Claim::Short),
        "candidate with p = 0.5 must not displace hot victim (0, 0)"
    );

    let mut cold_tier = Directory::new(&[1], &[0], &[8], HostTier::replay(vec![2]).reserving(1));
    log(&cold_tier, 1, 0, &[1]);
    for step in 2..=10 {
        log(&cold_tier, step, 0, &[0]);
    }
    let rec1 = cold_tier.land(0, 1, 0).unwrap();
    cold_tier.landed(0, rec1, 1, Some(1));
    cold_tier.time().lock().register(11, Advance::Known(Tokens::new(1.0)));
    assert!(
        matches!(cold_tier.claim_ahead(0, 2, 1, Taking::Ahead, 11, 1.0), Claim::Taken(_)),
        "candidate with p = 1.0 must displace cold victim (0, 1)"
    );
}

/// Seeds take free records first, then displace lower-ranked kept records
/// of any layer in the group (design §3.7 Seeds, §6): capacity is pooled
/// across layers, with no per-layer seed room.
#[test]
fn seeds_take_free_records_then_lower_ranks_of_any_layer() {
    let steps = std::sync::Arc::new(self::steps());
    let (mut directory, _) = one_store_on(&steps, 2, 2, 8);
    steps.begin_through(1000);

    // Free records first: layer 0 takes one free record, layer 1 takes the
    // second free record without evicting layer 0.
    let (seeds0, evicted0) = directory.seed(at(1000, 0), prompt_of(at(1000, 0)), 0, &[(0, 10, 1000)], &[], |_| true);
    assert_eq!((seeds0.len(), evicted0.len()), (1, 0), "layer 0 takes a free record");
    let (seeds1, evicted1) = directory.seed(at(1000, 1), prompt_of(at(1000, 1)), 1, &[(0, 1, 1000)], &[], |_| true);
    assert_eq!((seeds1.len(), evicted1.len()), (1, 0), "layer 1 takes the remaining free record");

    // Full store: layer 0 holds hot (0, 0) and cold (0, 1). A later layer's
    // seed (1, 5) with higher prompt rank displaces layer 0's cold (0, 1)
    // and leaves hot (0, 0) intact.
    let steps = std::sync::Arc::new(self::steps());
    let (mut directory, _) = one_store_on(&steps, 2, 2, 8);
    landed_before(&mut directory, &[0, 1]);
    for step in 1..=8 {
        log(&directory, step, 0, &[0]);
    }
    steps.begin_through(1000);
    let (seeds, evicted) = directory.seed(at(1000, 1), prompt_of(at(1000, 1)), 1, &[(5, 10, 1000)], &[], |_| true);
    assert_eq!(seeds.iter().map(|&(expert, _)| expert).collect::<Vec<_>>(), [5], "layer 1 seeds expert 5");
    assert_eq!(evicted, [(0, 1)], "layer 1's seed displaces layer 0's lower-ranked expert");
}

/// Prompt seeds rank by their prompt counts (design §3.7 Seeds, §6): given
/// equal recency, the experts with the largest counts in the prompt union
/// take the store's records rather than lower-index experts, and a higher
/// count displaces a lower-count incumbent.
#[test]
fn seeds_rank_by_the_prompts_counts() {
    let steps = std::sync::Arc::new(self::steps());
    let (mut directory, _) = one_store_on(&steps, 2, 1, 8);
    steps.begin_through(1000);
    let routed = [(0, 1, 1000), (1, 1, 1000), (5, 50, 1000), (7, 100, 1000)];
    let (seeds, evicted) = directory.seed(at(1000, 0), prompt_of(at(1000, 0)), 0, &routed, &[], |_| true);
    assert!(evicted.is_empty());
    let chosen: Vec<usize> = seeds.iter().map(|&(expert, _)| expert).collect();
    assert_eq!(chosen, [7, 5], "highest prompt counts win over lower indices");
    directory.retire(past(1000), |_, _| low_word(1000), |_| Ending::Ended, &steps);

    steps.begin_through(1001);
    let (seeds, evicted) = directory.seed(at(1001, 0), prompt_of(at(1001, 0)), 0, &[(2, 200, 1001)], &[], |_| true);
    assert_eq!(seeds.iter().map(|&(expert, _)| expert).collect::<Vec<_>>(), [2]);
    assert_eq!(evicted, [(0, 5)], "higher-count seed displaces the lower-ranked incumbent");
}

/// The host tier ranks at its own turnover `τ₁`, independent of tier 0's
/// `τ₀` (design §3.2, §4, §6): on an expert pair whose expected uses cross
/// between `τ₀` (short) and `τ₁` (long), tier 0 keeps expert 1 (certain at
/// lead 1 via front row 1, rare past it) over expert 0 (absent at lead 1,
/// frequent past it), while the host tier keeps expert 0 and names expert 1
/// as its victim.
#[test]
fn the_host_tier_ranks_at_its_own_turnover() {
    use crate::experts::horizon::{Advance, Reader, Tier, Tokens};
    use crate::experts::reuse::{Evidence, Rows};

    let mut directory = Directory::new(&[2], &[0], &[128], HostTier::replay(vec![100]).reserving(99));
    for expert in [0, 1] {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, Some(1));
    }
    // Expert 0 is read on every step 1..=20 (high long-horizon rate p_∞ ≈ 1);
    // expert 1 is unread (p_∞ ≈ 0). Two tier-0 moves per step keep τ₀ ≤ 1,
    // while tier 1 has only its two initial landings (τ₁ ≫ 100).
    for step in 1..=20 {
        log(&directory, step, 0, &[0]);
        let time = directory.time();
        let mut times = time.lock();
        times.moved(0, Tier::Compute);
        times.moved(0, Tier::Compute);
    }
    // Step 21: exact row (depth 0) chooses expert 0; front row (depth 1)
    // chooses expert 1 (σ₁ = 1 unjudged -> p₁(1) = 1, p₁(0) = 0).
    {
        let time = directory.time();
        let mut times = time.lock();
        times.register(21, Advance::Known(Tokens::new(1.0)));
        times.moved(0, Tier::Compute);
        times.moved(0, Tier::Compute);
    }
    let mut keys = vec![0.0f32; 256];
    keys[0] = 1.0;
    keys[128 + 1] = 1.0;
    let rows = std::sync::Arc::new(Rows::of(1, &[0, 1], &keys));
    Reuse::of(&directory.ranking).read(
        &RoutedRead { sequence: 21, layer: 0, experts: vec![0], choices: 1 },
        Some(rows),
        Evidence::Decode,
    );

    {
        let reuse = directory.reuse_now();
        let compute_0 = reuse.rank(0, 0, Tier::Compute, Reader::Move, None);
        let compute_1 = reuse.rank(0, 1, Tier::Compute, Reader::Move, None);
        let host_0 = reuse.rank(0, 0, Tier::Host, Reader::Move, None);
        let host_1 = reuse.rank(0, 1, Tier::Host, Reader::Move, None);
        assert!(compute_1 > compute_0, "at short τ₀, lead-1 expert 1 outranks expert 0");
        assert!(host_0 > host_1, "at long τ₁, frequent expert 0 outranks expert 1");
    }
    let victims: Vec<(usize, usize)> = directory.refill_victims(true)[0].iter().map(|&(_, staged)| staged).collect();
    assert_eq!(victims, [(0, 1)], "host tier refill evicts expert 1 at τ₁");
}

/// The ranking's memory charge (`Reuse::bytes`) is constant in run length
/// and turnover (design §1 bar 2, §6): identical after 10 steps and after
/// 10,000 steps across thousands of epoch rolls.
#[test]
fn the_rankings_size_is_constant_in_run_length() {
    use crate::experts::horizon::Tier;

    let directory = Directory::new(&[4], &[0, 0], &[16, 16], HostTier::replay(vec![8]));
    let run = |steps: std::ops::RangeInclusive<Step>| {
        for step in steps {
            let e = (step as usize) % 16;
            let mut a = [e, (e + 1) % 16];
            a.sort_unstable();
            let mut b = [(e + 3) % 16, (e + 5) % 16];
            b.sort_unstable();
            log(&directory, step, 0, &a);
            log(&directory, step, 1, &b);
            let time = directory.time();
            let mut times = time.lock();
            times.moved(0, Tier::Compute);
            times.moved(0, Tier::Host);
        }
    };
    run(1..=10);
    let after_10 = directory.reuse_now().bytes();
    assert!(after_10 > 0);
    run(11..=10_000);
    let after_10k = directory.reuse_now().bytes();
    assert_eq!(after_10, after_10k, "ranking memory must not grow with run length");
}

/// Repeating the exact same prompt at a later step (`step_2 > step_1`) with
/// tied `Expected` (`ΔE == 0`) seeds zero kept records: `seeding` requires
/// strictly higher expected future uses to evict a kept record, never
/// thrashing across layers on the `last` recency tie-breaker.
#[test]
fn warm_prompt_repeat_with_tied_expected_seeds_zero_kept_records() {
    use crate::experts::reuse::{Evidence, Rows};

    let steps = std::sync::Arc::new(self::steps());
    let (mut directory, _) = one_store_on(&steps, 4, 2, 8);
    steps.begin_through(1000);
    let counts = [(0, 10u32), (1, 10), (2, 10), (3, 10)];
    let routed = |step: Step| counts.map(|(expert, count)| (expert, count, step));

    // Cold prompt at step 1000: fills all 4 free records across the 2 layers.
    for layer in 0..2 {
        let (seeds, evicted) = directory.seed(at(1000, layer), prompt_of(at(1000, layer)), layer, &routed(1000), &[], |_| true);
        assert_eq!((seeds.len(), evicted.len()), (2, 0), "layer {layer} fills its free records");
    }
    directory.retire(past(1000), |_, _| low_word(1000), |_| Ending::Ended, &steps);
    {
        let time = directory.time();
        let mut times = time.lock();
        times.register(1000, crate::experts::horizon::Advance::Known(crate::experts::horizon::Tokens::new(1.0)));
    }
    for layer in 0..2 {
        let read = RoutedRead { sequence: 1000, layer, experts: vec![0, 1, 2, 3], choices: 2 };
        let union = std::sync::Arc::new(Rows::union(8, 2, &counts));
        Reuse::of(&directory.ranking).read(&read, Some(union), Evidence::Prompt);
    }

    // Warm repeat of the exact same prompt at step 1001 > 1000: every absent
    // expert has the same `Expected` as the kept records (`ΔE == 0`), only a
    // newer `last` step, so neither layer may evict or re-seed any record.
    steps.begin_through(1001);
    for layer in 0..2 {
        let (seeds, evicted) = directory.seed(at(1001, layer), prompt_of(at(1001, layer)), layer, &routed(1001), &[], |_| true);
        assert!(seeds.is_empty() && evicted.is_empty(), "layer {layer} thrashed on tied Expected: seeds={seeds:?} evicted={evicted:?}");
    }
}
