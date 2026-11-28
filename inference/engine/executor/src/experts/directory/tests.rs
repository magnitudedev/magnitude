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
    let substrate = crate::planning::bank_substrate(
        backend,
        seismic::NativeAllocations::EXACT,
        Some(host),
        backend.fills_beside_compute(),
        &definition,
    )
    .unwrap();
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
    assert!(directory.lease(0, crate::experts::watcher::ScratchGroup::at(2, 1), false, None), "every lent record is free or held, so leased at once");
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

/// During decode (`!lending`), `0..lent` are reclaimed for permanent decode
/// residency: the ranking's pool is all 4 records, and promotions fill
/// non-lent records (`2, 3`) first before overflowing into `0..lent` (`0, 1`).
/// While a prompt group is lending (`Unbegun::lending_at` / `lease`), no
/// promotion targets `0..lent` and the pool shrinks to the records past `lent`.
#[test]
fn decode_reclaims_lent_records_and_lending_excludes_them() {
    let mut directory = Directory::new(&[4], &[0], &[8], tiered(&[8], 1, 8, 2));
    directory.price_copies(0, Some(0.));
    assert_eq!(directory.reuse_now().records(0), 4, "during decode, the pool includes 0..lent");
    for expert in 0..6 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, Some(1));
    }
    log(&directory, 2, 0, &[0, 1, 2, 3, 4, 5]);
    let records = |wants: &Wants| wants.promotions.iter().map(|promotion| promotion.record).collect::<Vec<_>>();
    // While a prompt group has begun (`lending == true`), promotions only take records past `lent`.
    let lending_asked = directory.promotions(&crate::experts::watcher::Unbegun::lending_at(3), usize::MAX);
    assert_eq!((records(&lending_asked), lending_asked.releases), (vec![BankRecord(2), BankRecord(3)], vec![]));
    // Cancel those queued promotions so we can verify decode fills `2, 3` first then `0, 1`.
    directory.unpromoted(0, BankRecord(2));
    directory.unpromoted(0, BankRecord(3));
    let decode_asked = directory.promote();
    assert_eq!(
        (records(&decode_asked), decode_asked.releases),
        (vec![BankRecord(2), BankRecord(3), BankRecord(0), BankRecord(1)], vec![]),
        "decode fills non-lent records first, then reclaims 0..lent"
    );
    // Leasing cancels queued promotions in 0..lent and shrinks the pool to 2.
    assert!(directory.lease(0, ScratchGroup::at(4, 1), false, None));
    assert_eq!(directory.reuse_now().records(0), 2, "while leased, the pool excludes 0..lent");
    // Ending the prompt group restores the full decode pool of 4.
    directory.end_prompt_group(0);
    assert_eq!(directory.reuse_now().records(0), 4, "after unlend, 0..lent return to the decode pool");
}

/// Promoting an expert into the lent scratch slice `0..lent` retains its
/// host-tier copy (`Kept::Alone`, not surrendered), whereas promoting into a
/// Both non-lent (`>= lent`) and lent (`0..lent`) VRAM promotions mark their
/// Host-RAM copies `Kept::Banked` during decode so `surrender_banked` reclaims
/// all duplicate Host slots (`VRAM ∩ Host = ∅`). When a prompt group leases
/// `0..lent`, hot lent experts are written back cleanly via `exit_now`
/// (`backs_queued`), and once D2H completes and drains, the lease completes
/// with zero experts dropped to NVMe.
#[test]
fn lent_vram_promotion_retains_host_copy_so_prefill_lease_has_zero_writebacks_or_drops() {
    let steps = steps();
    steps.begin_through(20);
    let mut directory = Directory::new(&[4], &[0], &[8], tiered(&[6], 1, 8, 2));
    directory.price_copies(0, Some(0.));
    directory.price_write_backs(
        0,
        Some(WriteBack {
            terms: magnitude_cost_model::Terms { compute: 0., drive: 1., link: 0., cpu: 0., memory: 0. },
            drive: 1.,
            link: 0.,
        }),
    );
    for expert in 0..4 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, Some(1));
    }
    log(&directory, 2, 0, &[0, 1, 2, 3]);
    let asked = directory.promote();
    assert_eq!(asked.promotions.len(), 4);
    let mut non_lent_experts = Vec::new();
    let mut lent_experts = Vec::new();
    let mut completed = Vec::new();
    for p in &asked.promotions {
        let tier = directory.hold_tier(p.layer, p.expert).expect("in host tier");
        directory.copy(p.group, p.record, tier, p.record.0 as u64);
        completed.push((p.group, p.record));
        if p.record.index() < 2 {
            lent_experts.push(p.expert);
        } else {
            non_lent_experts.push(p.expert);
        }
    }
    directory.promoted(completed, Point::whole(2, 0, 2), &steps);
    // Run a promotions turn to trigger `host.surrender(at)` and retire graces.
    directory.promotions(&Unbegun::at(3), 0);
    directory.retire_graces(Point::whole(3, 1, 3), &steps);
    for &e in &non_lent_experts {
        assert!(directory.tier_record(0, e).is_none(), "non-lent VRAM expert {e} must be surrendered from host tier");
    }
    for &e in &lent_experts {
        assert!(directory.tier_record(0, e).is_none(), "0..lent VRAM expert {e} must be surrendered from host tier during decode");
    }
    // Prefill lease queues D2H write-backs for hot `0..lent` experts rather than dropping them to NVMe.
    assert!(!directory.lease(0, ScratchGroup::at(4, 1), false, None), "lease waits for 0..lent WriteBacks");
    let backs = directory.backs_queued();
    assert_eq!(backs.len(), lent_experts.len(), "hot 0..lent experts must queue D2H WriteBacks on lease");
    for (i, (group, bank_rec, _, _)) in backs.into_iter().enumerate() {
        directory.back_copy(group, bank_rec, i as u64 + 1);
        directory.written_back(group, bank_rec, true);
    }
    directory.drain(Point::whole(4, 1, 4), &steps);
    assert!(directory.lease(0, ScratchGroup::at(4, 1), false, None), "lease completes once 0..lent WriteBacks drain");
    for &e in &lent_experts {
        assert!(directory.held(0, e).is_none(), "lent expert {e} vacated VRAM");
        assert!(directory.tier_record(0, e).is_some(), "lent expert {e} written back to Host RAM (zero NVMe drop)");
    }
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
    directory.grow(0, vec![1, 3], 0);
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
            stage_slots: vec![0; records.len()],
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

/// A wanted expert that must displace a held record enters only once its reads
/// pay for its copy: priced past what one read weighs, it waits, and it is
/// asked again when its price falls.
#[test]
fn a_promotion_waits_until_its_reads_earn_its_copy() {
    let mut directory = Directory::new(&[3], &[0, 0], &[8, 8], HostTier::replay(vec![16]));
    for (layer, expert) in (0..2).flat_map(|layer| (0..8).map(move |expert| (layer, expert))) {
        let record = directory.land(layer, expert, 0).unwrap();
        directory.landed(layer, record, expert, None);
    }
    // Fill all three bank records with cold experts so promotions must displace
    // a resident record (`displaced > 0` path in `earning`).
    directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &[0, 1, 2], |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    // Priced at four reads, past the turnover set by three moves of the
    // three-record pool (τ = 3·T / 4 ≤ 3 over steps 1..4).
    {
        let time = directory.time();
        let mut times = time.lock();
        for _ in 0..3 {
            times.moved(0, crate::experts::horizon::Tier::COMPUTE);
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
    assert_eq!(directory.promote().releases.len(), 1, "the price fell, and the want releases a victim");
}

/// Cold-start promotions into completely free bank slots (`displaced == 0`)
/// fill available VRAM slots even when `τ_0 = K * T / (moves + 1)` has
/// collapsed below `copy_hits`, while once slots are full the displacement
/// check still gates swaps.
#[test]
fn cold_start_free_slot_admission_fills_empty_vram_despite_tau_collapse() {
    let mut directory = Directory::new(&[4], &[0], &[16], HostTier::replay(vec![16]));
    for expert in 0..16 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, None);
    }
    // Collapse τ_0 on step 1: K = 4, T = 1, moves = 39 -> τ_0 = 4 * 1 / 40 = 0.1 steps.
    {
        let time = directory.time();
        let mut times = time.lock();
        for _ in 0..39 {
            times.moved(0, crate::experts::horizon::Tier::COMPUTE);
        }
    }
    // High copy_hits (4.0) would starve every promotion if checked against τ_0 = 0.1.
    directory.price_copies(0, Some(4.0));
    log(&directory, 1, 0, &[0, 1, 2, 3]);
    let asked = directory.promote();
    assert_eq!(
        asked.promotions.len(),
        4,
        "all 4 free VRAM slots must be admitted despite collapsed τ_0: {asked:?}"
    );
    assert!(asked.releases.is_empty(), "free slots require no releases");
    // Complete all 4 promotions so the bank is now completely full (0 free slots).
    for promo in &asked.promotions {
        directory.complete(promo);
    }
    // Now a new expert (4) wants a slot, but bank is full (`displaced > 0`):
    // with copy_hits = 4.0 and collapsed τ_0, displacement must still be withheld!
    log(&directory, 2, 0, &[4]);
    let full_asked = directory.promote();
    assert_eq!(
        full_asked,
        Wants::default(),
        "once bank is full, unearned displacement must remain withheld"
    );
}

/// Request 2 domain switch regression: Request 1 fills all 30 VRAM slots with
/// hot experts `0..30` and completes 61 decode steps (`has_decode() == true`).
/// Request 2 then runs a prefill routing hot experts `100..130`.
/// 1. `Directory::seed` must replace Request 1's stale `0..30` with Request 2's
///    `100..130` in VRAM (not lock them out via stale `in_top` or ignore prefill counts).
/// 2. On Request 2 Decode Step 1, `Directory::promotions()` must NOT evict
///    `100..130` back to Request 1's stale `0..30`.
#[test]
fn request_2_prefill_seeding_replaces_stale_request_1_experts_and_survives_decode_step_1() {
    use crate::experts::reuse::{Evidence, Rows};

    let mut directory = Directory::new(&[30], &[0], &[256], HostTier::replay(vec![256]));
    for expert in 0..256 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, None);
    }
    directory.price_copies(0, Some(1.0));

    // Request 1: place all 30 hot experts `0..30` into VRAM and run 61 decode steps
    // (6 experts per step cycling through `0..30`, so all 30 are hot and `in_top`).
    let req1_hot: Vec<usize> = (0..30).collect();
    directory.place(at(1, 0), prompt_of(at(1, 0)), 0, &[], &req1_hot, |_| 0).unwrap();
    directory.retired(past(1), |_, _| 1, |_| Ending::Ended);
    for step in 2..=62u64 {
        let base = ((step as usize - 2) * 6) % 30;
        let mut chosen: Vec<usize> = (0..6).map(|i| (base + i) % 30).collect();
        chosen.sort_unstable();
        log(&directory, step, 0, &chosen);
    }
    assert!(
        (0..30).all(|e| directory.held(0, e).is_some() && directory.reuse_now().in_top(0, e)),
        "Request 1's 0..30 must fill VRAM and be in_top before Request 2"
    );

    // Request 2 prefill at step 100: routes hot experts `100..130` heavily (count = 200 each).
    let req2_routed: Vec<(usize, u32, Step)> = (100..130).map(|e| (e, 200, 100)).collect();
    let (seeds, evicted) = directory.seed(at(100, 0), prompt_of(at(100, 0)), 0, &req2_routed, &[], |_| true);
    assert_eq!(seeds.len(), 30, "all 30 Request 2 hot experts must be seeded into VRAM");
    assert_eq!(evicted.len(), 30, "all 30 stale Request 1 experts must be evicted");
    directory.retired(past(100), |_, _| 100, |_| Ending::Ended);
    for e in 100..130 {
        assert!(directory.held(0, e).is_some(), "Request 2 expert {e} must be resident in VRAM");
    }
    for e in 0..30 {
        assert!(directory.held(0, e).is_none(), "stale Request 1 expert {e} must be evicted from VRAM");
    }

    // Request 2 prompt group completes and logs `Evidence::Prompt` into `Reuse`.
    let req2_counts: Vec<(usize, u32)> = (100..130).map(|e| (e, 200)).collect();
    let req2_read = RoutedRead { sequence: 100, layer: 0, experts: (100..130).collect(), choices: 6 };
    let req2_union = std::sync::Arc::new(Rows::union(256, 6, &req2_counts));
    Reuse::of(&directory.ranking).read(&req2_read, Some(req2_union), Evidence::Prompt);

    // Request 2 prefill Chunk 2 (step 101): must NOT thrash Chunk 1's `in_top`
    // experts `100..130` out of VRAM!
    let req2_chunk2: Vec<(usize, u32, Step)> = (150..180).map(|e| (e, 200, 101)).collect();
    let (chunk2_seeds, chunk2_evicted) = directory.seed(at(101, 0), prompt_of(at(101, 0)), 0, &req2_chunk2, &[], |_| true);
    assert!(
        chunk2_seeds.is_empty() && chunk2_evicted.is_empty(),
        "Request 2 prefill Chunk 2 must NOT thrash Chunk 1's in_top VRAM experts: seeds={chunk2_seeds:?} evicted={chunk2_evicted:?}"
    );

    // Request 2 Decode Step 1 (step 102): routes experts from `100..130`.
    log(&directory, 102, 0, &[100, 101, 102, 103, 104, 105]);
    let wants = directory.promote();
    assert!(
        wants.releases.iter().all(|r| !(100..130).contains(&r.expert)),
        "Decode Step 1 must NOT evict Request 2's 100..130 back to Request 1's 0..30: {wants:?}"
    );
    assert!(
        wants.promotions.iter().all(|p| !(0..30).contains(&p.expert)),
        "Decode Step 1 must NOT promote Request 1's stale 0..30 over Request 2's 100..130: {wants:?}"
    );
    assert!(
        (100..130).all(|e| directory.held(0, e).is_some()),
        "all 30 Request 2 experts must remain resident in VRAM after Decode Step 1"
    );
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
            directory.time().lock().moved(0, crate::experts::horizon::Tier::COMPUTE);
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
            surrendered: vec![],
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
        Wants { promotions: vec![], releases: vec![Release { group: 0, record: BankRecord(1), layer: 0, expert: 1 }], surrendered: vec![] }
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
            surrendered: vec![],
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
    directory.grow(0, vec![2], 0);
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
            directory.time().lock().moved(0, crate::experts::horizon::Tier::COMPUTE);
        }
    }
    assert!(
        directory
            .reuse_now()
            .expected(0, 1, crate::experts::horizon::Tier::HOST, crate::experts::horizon::Reader::Move, None)
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
    assert!(directory.lease(0, prompt_of(at(1000, 0)), false, None), "every lent record is ready, so leased at once");
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
    assert!(directory.lease(0, prompt_of(at(1000, 0)), false, None), "every lent record is ready, so leased at once");
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
    let admitted = directory.claim_ahead(0, 5, 2, Taking::Ahead, 2, 0.5, 0.0);
    assert!(matches!(admitted, Claim::Taken(_)), "a ready record past the claim's keep admits");
    let short = directory.claim_ahead(0, 6, 3, Taking::Ahead, 2, 1.0, 0.0);
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
        let base = reuse.expected(0, 1, Tier::HOST, Reader::Move, None).uses();
        let with_q = reuse.expected_with(0, 1, Tier::HOST, 4, q).uses();
        let with_one = reuse.expected_with(0, 1, Tier::HOST, 4, 1.0).uses();
        let with_zero = reuse.expected_with(0, 1, Tier::HOST, 4, 0.0).uses();
        assert!((with_q - base).abs() < 1e-9, "p = q must keep lead term {q}: base={base}, with_q={with_q}");
        assert!((with_one - base - (1.0 - q)).abs() < 1e-9, "p = 1.0 adds {} over q={q}: base={base}, with_one={with_one}", 1.0 - q);
        assert!((base - with_zero - q).abs() < 1e-9, "p = 0.0 subtracts {q} from q={q}: base={base}, with_zero={with_zero}");
    }

    // Part 2: victim test at Tier::HOST, Reader::Move with keep = reserve = 1.
    let mut hot_tier = Directory::new(&[1], &[0], &[8], HostTier::replay(vec![2]).reserving(1));
    for step in 1..=10 {
        log(&hot_tier, step, 0, &[0]);
    }
    let rec0 = hot_tier.land(0, 0, 0).unwrap();
    hot_tier.landed(0, rec0, 0, Some(10));
    hot_tier.time().lock().register(11, Advance::Known(Tokens::new(1.0)));
    assert!(
        matches!(hot_tier.claim_ahead(0, 2, 1, Taking::Ahead, 11, 0.5, 0.0), Claim::Short),
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
        matches!(cold_tier.claim_ahead(0, 2, 1, Taking::Ahead, 11, 1.0, 0.0), Claim::Taken(_)),
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
        times.moved(0, Tier::COMPUTE);
        times.moved(0, Tier::COMPUTE);
    }
    // Step 21: exact row (depth 0) chooses expert 0; front row (depth 1)
    // chooses expert 1 (σ₁ = 1 unjudged -> p₁(1) = 1, p₁(0) = 0).
    {
        let time = directory.time();
        let mut times = time.lock();
        times.register(21, Advance::Known(Tokens::new(1.0)));
        times.moved(0, Tier::COMPUTE);
        times.moved(0, Tier::COMPUTE);
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
        let compute_0 = reuse.rank(0, 0, Tier::COMPUTE, Reader::Move, None);
        let compute_1 = reuse.rank(0, 1, Tier::COMPUTE, Reader::Move, None);
        let host_0 = reuse.rank(0, 0, Tier::HOST, Reader::Move, None);
        let host_1 = reuse.rank(0, 1, Tier::HOST, Reader::Move, None);
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
            times.moved(0, Tier::COMPUTE);
            times.moved(0, Tier::HOST);
        }
    };
    let warmup = (crate::experts::DEPTHS + 2) as u64;
    run(1..=warmup);
    let after_warmup = directory.reuse_now().bytes();
    assert!(after_warmup > 0);
    run(warmup + 1..=10_000);
    let after_10k = directory.reuse_now().bytes();
    assert_eq!(after_warmup, after_10k, "ranking memory must not grow with run length");
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

/// On a discrete `Placer::Tier` bank with `lent = 2` (`0..2`) and a dedicated
/// even/odd staging ring `stage_slots = 4` (`2..6`), `Directory::seeded_from`
/// remains `lent` (matching `ExpertBankPlan::seeded_from` and the GPU
/// `routed_seed_commit` base offset), and `Directory::seed` never hands out
/// any record in `0..lent` or `2..6` (by construction via `Record::StageIdle`),
/// so every seeded record satisfies `record.index() >= seeded_from` and
/// `record.index() - seeded_from` equals its true offset past `lent`.
#[test]
fn seed_skips_lent_and_stage_ring_and_keeps_seeded_from_at_lent() {
    let lent = 2usize;
    let stage_slots = 4usize;
    let total_records = 10usize; // 0..2 lent, 2..6 staging ring, 6..10 permanent bank
    let experts = 8u64;
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let host = HostTier {
        records: vec![8],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier {
            bank_maps: vec![StampedMap::new(cpu_words(&[experts, map_words], 0))],
            tier_maps: vec![LandingMap::Tier(cpu_words(&[experts], -1))],
            lent: (0, lent),
            stage_slots: vec![stage_slots],
        },
    };
    let mut directory = Directory::new(&[total_records], &[0], &[experts as usize], host);
    let steps = steps();
    steps.begin_through(10);

    // `seeded_from` MUST stay equal to `lent` (matching GPU `routed_seed_commit`).
    // During decode (`!leasing`), `0..lent` are reclaimed for permanent decode residency.
    assert_eq!(directory.seeded_from(0), lent);
    assert_eq!(directory.capacity(0), total_records - stage_slots);
    assert_eq!(directory.stage_ring_half(0), stage_slots / 2);

    // Lease `0..lent` for prompt group starting at step 1, then seed 4 experts.
    let group = ScratchGroup::at(1, 1);
    assert!(directory.lease(0, group, false, None), "lent records 0..2 lease cleanly");
    assert_eq!(directory.capacity(0), total_records - lent - stage_slots, "while leased, 0..lent leave decode capacity");

    let routed = [(0usize, 10u32, 1u64), (1, 10, 1), (2, 10, 1), (3, 10, 1)];
    let (seeds, evicted) = directory.seed(at(1, 0), group, 0, &routed, &[], |_| true);
    assert_eq!(seeds.len(), 4, "all 4 permanent records 6..10 are seeded");
    assert!(evicted.is_empty());
    let mut indices: Vec<usize> = seeds.iter().map(|&(_, rec)| rec.index()).collect();
    indices.sort_unstable();
    assert_eq!(indices, vec![6, 7, 8, 9], "seed must never touch 0..lent or staging ring 2..6");
    for &idx in &indices {
        assert!(!directory.is_stage_slot(0, idx));
        assert!(idx >= directory.seeded_from(0));
        // GPU `routed_seed_commit` offset relative to `ExpertBankPlan::seeded_from` (`lent`).
        assert_eq!(idx - directory.seeded_from(0), idx - lent);
    }

    // Staging ring `2..6` remains completely untouched and ready for even/odd DMA staging.
    assert_eq!(directory.free_stage_slot(0, 2), Some(2));
    assert_eq!(directory.free_stage_slot(0, 3), Some(4));
}

#[test]
fn claim_carryover_prices_eviction_on_full_two_tier_bank() {
    use crate::experts::host_records::{Claim, Taking};

    // 2-tier unified bank (`Placement::Bank`): 4 total records (2 kept + 2 reserve).
    let experts = 8usize;
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let steps = steps();
    let host = HostTier {
        records: vec![4],
        lent: vec![0],
        reserve: 2,
        placement: Placement::Bank {
            landing_maps: vec![LandingMap::Bank {
                map: std::sync::Arc::new(StampedMap::new(cpu_words(&[experts as u64, map_words], 0))),
                steps: std::sync::Arc::new(steps),
            }],
        },
    };
    let mut directory = Directory::new(&[0], &[0], &[experts], host);
    // Land experts 0 and 1 so the 2 non-reserve slots are full of resident experts.
    for expert in 0..2 {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
            panic!("expected free slot for resident expert {expert}");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(1));
    }

    // 1. Non-positive net_benefit is ALWAYS rejected (even if reserve slots exist).
    assert!(matches!(directory.claim_carryover(0, 2, 1, Taking::Ahead, 0.0, true), Claim::Short));
    assert!(matches!(directory.claim_carryover(0, 2, 1, Taking::Ahead, -0.25, true), Claim::Short));

    // 2. On a full bank (`keep = 2`, zero free slots), `strict_rank = true`
    // rejects tied candidates (`cand_uses == victim_uses`) to prevent tie-churn.
    assert!(
        matches!(directory.claim_carryover(0, 2, 2, Taking::Ahead, 0.5, true), Claim::Short),
        "strict_rank carryover must reject tied eviction on full 2-tier bank"
    );

    // 3. When a free slot is available (`keep = 1`), positive net_benefit admits without evicting.
    let claim = directory.claim_carryover(0, 2, 1, Taking::Ahead, 0.5, true);
    assert!(matches!(claim, Claim::Taken(_)), "profitable carryover must admit into free slot");
}

#[test]
fn alone_excludes_lent_slot_dual_resident_experts() {
    use crate::experts::host_records::{Claim, Taking};

    let experts = 4usize;
    let lent = 2usize;
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let host = HostTier {
        records: vec![experts as u32],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier {
            bank_maps: vec![StampedMap::new(cpu_words(&[experts as u64, map_words], 0))],
            tier_maps: vec![LandingMap::Tier(cpu_words(&[experts as u64], -1))],
            lent: (0, lent),
            stage_slots: vec![0],
        },
    };
    let mut directory = Directory::new(&[lent], &[0], &[experts], host);
    let steps = steps();
    steps.begin_through(2);
    for expert in 0..experts {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
            panic!("expected host slot for expert {expert}");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(1));
    }
    let DemandSlotClaim::Allocated { staged, slot } = directory.claim_demand_landing(0, 0, false, |_| true) else {
        panic!("expected free lent slot for demand landing");
    };
    assert!(directory.bank_lends(0, slot));
    assert!(directory.finish_demand_landing(0, slot, staged, at(1, 0), &steps, true));
    assert_eq!(directory.held_count(0), 1);
    assert!(
        !directory.host(0).holds_alone((0, 0)),
        "0..lent decode expert must be marked Kept::Banked so Host RAM does not duplicate it"
    );
    assert_eq!(directory.host(0).alone()[0], experts - 1);
    assert_eq!(directory.alone(0), experts - 1);
    assert!(directory.held_count(0) + directory.alone(0) <= experts);
}

#[test]
fn promotions_and_stage_slots_skip_transport_pinned_slots() {
    use crate::experts::host_records::{Claim, Taking};
    use crate::experts::transport::{DestSlot, ExpertKey, ExpertTransport};

    let experts = 6usize;
    let lent = 1usize;
    let stage_slots = 4usize; // 2 even + 2 odd slots at 1..5
    let bank_slots = 7usize;  // 0 (lent), 1..5 (stage ring), 5..7 (permanent)
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let host = HostTier {
        records: vec![experts as u32],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier {
            bank_maps: vec![StampedMap::new(cpu_words(&[experts as u64, map_words], 0))],
            tier_maps: vec![LandingMap::Tier(cpu_words(&[experts as u64], -1))],
            lent: (0, lent),
            stage_slots: vec![stage_slots],
        },
    };
    let mut directory = Directory::new(&[bank_slots], &[0], &[experts], host);
    let steps = steps();
    steps.begin_through(2);
    for expert in 0..experts {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
            panic!("expected host slot for expert {expert}");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(1));
    }

    // 1. Register an active transport hint on stage slot 1 (first even slot) and
    //    permanent bank slot 5. Neither allocator may hand those pinned slots out.
    let transport = ExpertTransport::new_registry_only();
    let pinned_stage = DestSlot { group: 0, slot: 1 };
    let pinned_perm = DestSlot { group: 0, slot: 5 };
    let stage_id = transport.register_hint(ExpertKey { group: 0, layer: 0, expert: 0 }, 0, pinned_stage);
    let perm_id = transport.register_hint(ExpertKey { group: 0, layer: 0, expert: 1 }, 1, pinned_perm);

    assert_eq!(
        directory.free_stage_slot_with(0, 2, |slot| transport.can_reuse_slot(DestSlot { group: 0, slot })),
        Some(2),
        "free_stage_slot_with must skip transport-pinned slot 1"
    );

    // Feed decode activity so promotions are eligible.
    let ranking = directory.ranking();
    {
        let time = directory.time();
        time.lock().register(2, crate::experts::horizon::Advance::Known(crate::experts::horizon::Tokens::new(1.0)));
    }
    Reuse::of(&ranking).read(
        &RoutedRead { sequence: 2, layer: 0, choices: 1, experts: vec![2] },
        None,
        crate::experts::reuse::Evidence::Decode,
    );
    directory.price_copies(0, Some(0.0));

    let asked = steps.unbegun_beside_frontier(|_, at| {
        transport.with_slot_reuse(|can_reuse| {
            directory.promotions_with(at, 1, |group, slot| can_reuse(DestSlot { group, slot }))
        })
    });
    assert_eq!(asked.promotions.len(), 1);
    assert_ne!(
        asked.promotions[0].record.index(),
        pinned_perm.slot,
        "promotions_with must skip transport-pinned permanent slot 5"
    );

    // 2. Once DMA completes and hint ownership releases, the slots become reusable.
    transport.complete_dma(stage_id);
    assert!(transport.try_release_hint_slot(pinned_stage));
    assert_eq!(
        directory.free_stage_slot_with(0, 2, |slot| transport.can_reuse_slot(DestSlot { group: 0, slot })),
        Some(1)
    );

    transport.complete_dma(perm_id);
    assert!(transport.try_release_hint_slot(pinned_perm));
    assert!(transport.can_reuse_slot(pinned_perm));
}

/// During prefill on a non-covering 3-tier bank (`HostTier` < E), seeding an
/// expert into a non-lent VRAM record immediately marks its host-tier copy
/// `Kept::Banked` so subsequent `claim_conveyed` / `surrender_banked` reclaim
/// those duplicate host slots for non-VRAM experts (`VRAM ∩ HostTier == ∅`),
/// raising total in-memory coverage from `H` to `V + H`. A tainted seed
/// (`!committed`) restores `Kept::Banked -> Kept::Alone` if still present.
#[test]
fn prefill_seed_marks_host_tier_banked_so_conveyed_reclaims_duplicates_for_disjoint_coverage() {
    use crate::experts::host_records::Claim;

    let steps = steps();
    steps.begin_through(10);
    // 1 group, 1 layer of 8 experts; VRAM = 2 non-lent slots, HostTier = 4 slots (non-covering).
    let mut directory = Directory::new(&[2], &[0], &[8], HostTier::replay(vec![4]));
    directory.price_write_backs(
        0,
        Some(WriteBack {
            terms: magnitude_cost_model::Terms { compute: 0., drive: 4., link: 0., cpu: 0., memory: 0. },
            drive: 1.,
            link: 0.,
        }),
    );
    // 1. Land experts 0..4 into HostTier during prefill.
    for expert in 0..4 {
        let Claim::Taken(grant) = directory.claim_conveyed(0, expert, Some(1)) else {
            panic!("free host slot for expert {expert}");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(1));
    }
    assert_eq!(directory.alone(0), 4);

    // 2. Seed top 2 experts (0, 1) into VRAM at step 1 (still Record::Placing before retire).
    let routed = vec![(0, 10, 1), (1, 9, 1), (2, 1, 1), (3, 1, 1)];
    let (seeds, evicted) = directory.seed(at(1, 0), ScratchGroup::at(1, 1), 0, &routed, &[], |_| true);
    assert_eq!(seeds.len(), 2);
    assert!(evicted.is_empty());
    // Both seeded experts are immediately marked Kept::Banked (not Kept::Alone) even before retire!
    assert_eq!(directory.alone(0), 2, "seeded experts must not remain Kept::Alone in HostTier");
    // `claim_conveyed` rejects experts already placing into non-lent VRAM.
    assert!(matches!(directory.claim_conveyed(0, 0, Some(1)), Claim::Short));

    // 3. Surrender banked copies -> frees 2 HostTier slots for cold experts 4, 5!
    let surrendered = directory.surrender_banked(&Unbegun::at(2));
    assert_eq!(surrendered.len(), 2);
    directory.retire_graces(at(2, 1), &steps);
    for expert in 4..6 {
        let Claim::Taken(grant) = directory.claim_conveyed(0, expert, Some(1)) else {
            panic!("reclaimed host slot must accept cold expert {expert}");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(1));
    }

    // 4. Retire seeds as committed -> VRAM holds {0, 1}, HostTier holds {2, 3, 4, 5}.
    directory.retire(
        at(2, 1),
        |_, _| crate::experts::watcher::low_word(1),
        |_| Ending::Ended,
        &steps,
    );
    assert_eq!(directory.held_count(0), 2);
    assert_eq!(directory.alone(0), 4);
    for expert in 0..8 {
        assert!(
            !(directory.held(0, expert).is_some() && directory.in_tier(0, expert)),
            "VRAM ∩ HostTier must be empty for expert {expert}"
        );
    }
    let in_memory: usize = (0..8)
        .filter(|&e| directory.held(0, e).is_some() || directory.in_tier(0, e))
        .count();
    assert_eq!(in_memory, 6, "disjoint tiers must achieve V + H = 6/8 in-memory coverage");

    // 5. Tainted seed restores Kept::Banked -> Kept::Alone if not yet surrendered.
    let mut tainted_dir = Directory::new(&[1], &[0], &[4], HostTier::replay(vec![2]));
    let Claim::Taken(g) = tainted_dir.claim_conveyed(0, 0, Some(1)) else { panic!("host slot") };
    tainted_dir.landed(0, HostRecord(g.record()), 0, Some(1));
    let (seeds, _) = tainted_dir.seed(at(1, 0), ScratchGroup::at(1, 1), 0, &[(0, 5, 1)], &[], |_| true);
    assert_eq!(seeds.len(), 1);
    assert_eq!(tainted_dir.alone(0), 0, "marked Kept::Banked while placing");
    tainted_dir.retire(at(2, 1), |_, _| 0, |_| Ending::Tainted, &steps);
    assert_eq!(tainted_dir.held_count(0), 0, "tainted seed did not commit");
    assert_eq!(tainted_dir.alone(0), 1, "tainted seed restores HostTier copy to Kept::Alone");
}

/// `replenish_demand_free` must never leak `Host::WritingBack` records when
/// `WriteBack` pays: it releases the victim into `Record::Releasing` with its
/// write-back queued, so `written_back` + `drain` move the victim cleanly to
/// `HostTier` and free the VRAM slot with `VRAM ∩ HostTier == ∅`.
#[test]
fn replenish_demand_free_writes_back_victims_without_leaking_host_records() {
    let steps = steps();
    steps.begin_through(10);
    // VRAM = 2 slots, HostTier = 3 slots (non-covering for 6 experts).
    let mut directory = Directory::new(&[2], &[0], &[6], HostTier::replay(vec![3]));
    directory.price_copies(0, Some(0.));
    directory.price_write_backs(
        0,
        Some(WriteBack {
            terms: magnitude_cost_model::Terms { compute: 0., drive: 4., link: 0., cpu: 0., memory: 0. },
            drive: 1.,
            link: 0.,
        }),
    );
    // Land experts 0, 1 into HostTier and promote both into VRAM.
    for expert in 0..2 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, Some(1));
    }
    log(&directory, 2, 0, &[0, 1]);
    let asked = directory.promote();
    assert_eq!(asked.promotions.len(), 2);
    for p in &asked.promotions {
        directory.complete(p);
    }
    // Surrender both from HostTier so VRAM = {0, 1}, HostTier = {} (3 ready).
    directory.surrender_banked(&Unbegun::at(3));
    directory.retire_graces(at(3, 1), &steps);
    assert_eq!(directory.held_count(0), 2);
    assert_eq!(directory.alone(0), 0);
    assert_eq!(directory.host(0).ready(), 3);

    // Replenish 1 free VRAM slot while WriteBack pays -> must queue WriteBack, NOT leak!
    let released = directory.replenish_demand_free(0, 1, &Unbegun::at(4), |_| true);
    assert_eq!(released, 1);
    let backs = directory.backs_queued();
    assert_eq!(backs.len(), 1, "evicted hot victim must have its WriteBack queued");
    let (group, bank_rec, (_, victim), _tier_rec) = backs[0];
    directory.back_copy(group, bank_rec, 1);
    directory.written_back(group, bank_rec, true);
    directory.drain(at(4, 1), &steps);

    assert_eq!(directory.held_count(0), 1);
    assert_eq!(directory.alone(0), 1);
    assert!(directory.in_tier(0, victim), "victim must now reside in HostTier");
    assert!(directory.held(0, victim).is_none(), "victim must no longer be in VRAM");
    assert_eq!(directory.host(0).ready(), 2, "zero leaked Host::WritingBack records");
}

/// On a non-covering `Placer::Tier` host tier, `claim_ahead` and
/// `claim_carryover` must skip experts already held in non-lent VRAM so
/// speculative NVMe prefetches never duplicate VRAM experts in `HostTier`.
#[test]
fn claim_ahead_and_carryover_skip_non_lent_vram_experts_on_non_covering_tier() {
    use crate::experts::host_records::{Claim, Taking};

    let steps = steps();
    steps.begin_through(10);
    let mut directory = Directory::new(&[2], &[0], &[6], HostTier::replay(vec![3]));
    directory.price_copies(0, Some(0.));
    directory.price_write_backs(
        0,
        Some(WriteBack {
            terms: magnitude_cost_model::Terms { compute: 0., drive: 4., link: 0., cpu: 0., memory: 0. },
            drive: 1.,
            link: 0.,
        }),
    );
    let rec = directory.land(0, 0, 0).unwrap();
    directory.landed(0, rec, 0, Some(1));
    log(&directory, 2, 0, &[0]);
    let asked = directory.promote();
    assert_eq!(asked.promotions.len(), 1);
    directory.complete(&asked.promotions[0]);
    directory.surrender_banked(&Unbegun::at(3));
    directory.retire_graces(at(3, 1), &steps);
    assert!(directory.held(0, 0).is_some() && !directory.in_tier(0, 0));

    assert!(matches!(directory.claim_ahead(0, 0, 0, Taking::Ahead, 4, 1.0, 0.0), Claim::Short));
    assert!(matches!(directory.claim_carryover(0, 0, 0, Taking::Ahead, 1.0, false), Claim::Short));
    // Cold expert 1 (not in VRAM) is still admitted normally.
    assert!(matches!(directory.claim_ahead(0, 1, 0, Taking::Ahead, 4, 1.0, 0.0), Claim::Taken(_)));
}

/// `Directory::host_has_free_slot` is true iff the group's host tier has an
/// empty ready record beyond its reserve (so `Scratch::land_conveyed` can fill
/// unused host RAM without evicting any kept expert), becomes false once all
/// non-reserve slots are occupied, and turns true again after surrendering a
/// banked duplicate.
#[test]
fn host_has_free_slot_tracks_non_evicting_ready_headroom() {
    use crate::experts::host_records::Claim;

    let steps = steps();
    steps.begin_through(10);
    // HostTier = 3 total slots, reserve = 1 -> 2 non-evicting free slots.
    let mut directory = Directory::new(&[1], &[0], &[6], HostTier::replay(vec![3]).reserving(1));
    directory.price_write_backs(
        0,
        Some(WriteBack {
            terms: magnitude_cost_model::Terms { compute: 0., drive: 4., link: 0., cpu: 0., memory: 0. },
            drive: 1.,
            link: 0.,
        }),
    );
    assert!(directory.host_has_free_slot(0), "2 non-reserve slots are ready");

    let Claim::Taken(g0) = directory.claim_conveyed(0, 0, Some(1)) else { panic!("slot 0") };
    directory.landed(0, HostRecord(g0.record()), 0, Some(1));
    assert!(directory.host_has_free_slot(0), "1 non-reserve slot remains ready");

    let Claim::Taken(g1) = directory.claim_conveyed(0, 1, Some(1)) else { panic!("slot 1") };
    directory.landed(0, HostRecord(g1.record()), 1, Some(1));
    assert!(!directory.host_has_free_slot(0), "only the 1 reserve slot remains; next landing would evict");

    // Seed expert 0 into VRAM and surrender its duplicate HostTier slot -> free slot returns!
    let (seeds, _) = directory.seed(at(1, 0), ScratchGroup::at(1, 1), 0, &[(0, 5, 1)], &[], |_| true);
    assert_eq!(seeds.len(), 1);
    assert_eq!(directory.surrender_banked(&Unbegun::at(2)).len(), 1);
    directory.retire_graces(at(2, 1), &steps);
    assert!(directory.host_has_free_slot(0), "surrendered duplicate restores non-evicting free slot");
}

#[test]
fn decode_transient_budget_eliminates_dma_overflow_for_m6_k8_by_construction() {
    use crate::experts::host_records::{Claim, Taking};

    // Group with 4 layers, 256 experts/layer, 300 bank records (280 permanent + 20 stage).
    // Verify burst M = 6, K = 8 -> W_max = 48 distinct misses in one layer.
    let layers = 4usize;
    let experts = 256usize;
    let present = 300usize;
    let stage_slots = 20usize;
    let m_max = 6usize;
    let k = 8usize;
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let bank_maps = (0..layers)
        .map(|_| StampedMap::new(cpu_words(&[experts as u64, map_words], 0)))
        .collect();
    let tier_maps = (0..layers)
        .map(|_| LandingMap::Tier(cpu_words(&[experts as u64], -1)))
        .collect();
    let host = HostTier {
        records: vec![(layers * experts) as u32],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier {
            bank_maps,
            tier_maps,
            lent: (0, 0),
            stage_slots: vec![stage_slots],
        },
    };
    let mut directory = Directory::new(&[present], &vec![0; layers], &vec![experts; layers], host);
    // Non-speculative M=1 decode reserves only K=8 slots (never 48!), preserving
    // resident VRAM for Held experts.
    directory.configure_decode_budget(1, k, 3);
    assert_eq!(directory.decode_budget(0).demand_floor, k);
    assert_eq!(directory.reserved(0), k);

    // Speculative M=6 verify reserves expected unique misses `ceil(48 * p_miss)`
    // (`34` free slots), and `StageIdle` borrow absorbs any outlier burst up to
    // `M * K = 48` with zero `dma_overflow`.
    directory.configure_decode_budget(m_max, k, 3);

    let p_miss = 1.0 - (present as f64 / (layers * experts) as f64);
    let expected_floor = ((m_max * k) as f64 * p_miss).ceil() as usize;
    let budget = directory.decode_budget(0);
    assert_eq!(budget.demand_floor, expected_floor, "demand_floor tracks expected unique misses ceil(W_max * p_miss)");
    assert_eq!(budget.promote_headroom, k);
    assert_eq!(directory.reserved(0), budget.demand_floor);
    assert_eq!(directory.demand_free_floor(0), budget.demand_floor);
    assert_eq!(directory.demand_replenish_target(0), budget.demand_floor + budget.promote_headroom);

    // Land all experts into the host tier and allow zero-cost promotions.
    let mut tier_records = Vec::with_capacity(experts);
    for layer in 0..layers {
        (0..experts).for_each(|expert| {
            let Claim::Taken(grant) = directory.claim_landing(layer, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
                panic!("host tier has room for every expert");
            };
            directory.landed(layer, HostRecord(grant.record()), expert, Some(1));
            if layer == 0 {
                tier_records.push(grant.record());
            }
        });
        directory.price_copies(layer, Some(0.0));
    }

    // Feed decode reads so Reuse wants to fill permanent VRAM via promotions.
    let steps = steps();
    let step1 = steps.begin_group(1, None);
    let hot: Vec<usize> = (0..100).collect();
    for layer in 0..layers {
        log(&directory, step1, layer, &hot);
    }
    assert!(directory.settle_changes());
    // Run promotions up to full budget: `Directory::promotions` MUST leave
    // `demand_free_floor == K` free slots untouched for on-demand decode DMA.
    let wants = steps.unbegun(|at| directory.promotions(at, present));
    assert!(!wants.promotions.is_empty());
    for p in &wants.promotions {
        let tier = directory.hold_tier(p.layer, p.expert).expect("promoted expert is in host tier");
        directory.copy(p.group, p.record, tier, 0);
    }
    let completed: Vec<(usize, BankRecord)> = wants.promotions.iter().map(|p| (p.group, p.record)).collect();
    directory.promoted(completed, at(step1, 0), &steps);
    assert!(
        directory.bank_free_count(0) >= budget.demand_floor,
        "promotions must preserve >= {} free slots, left {}",
        budget.demand_floor,
        directory.bank_free_count(0)
    );

    // Claim M*K = 48 unheld distinct misses in layer 0 in a single verify burst:
    // Free + StageIdle borrow + `reclaim_served_demand_landings` give zero overflow!
    let mut claimed = 0usize;
    let mut overflows = Vec::new();
    for (idx, &tier_rec) in tier_records.iter().skip(100).take(m_max * k).enumerate() {
        match directory.claim_demand_landing(0, tier_rec, false, |_| true) {
            DemandSlotClaim::Allocated { .. } | DemandSlotClaim::AlreadyHeld { .. } => claimed += 1,
            DemandSlotClaim::Overflow { staged } => overflows.push((idx, staged)),
        }
    }
    if !overflows.is_empty() {
        let progress = at(step1, 0);
        let reclaimed = steps.unbegun_with_served(progress, directory.layers(), |at_unbegun, park, is_served| {
            directory.reclaim_served_demand_landings(
                0,
                &overflows,
                at_unbegun,
                park,
                progress,
                is_served,
                |_| true,
            )
        });
        claimed += reclaimed.len();
    }
    assert_eq!(claimed, m_max * k, "M=6, K=8 must have dma_overflow == 0 with served-victim reclaim");
}

#[test]
fn tiny_vram_borrows_stage_idle_without_promotion_and_returns_to_stage_idle() {
    use crate::experts::host_records::{Claim, Taking};
    use crate::experts::reuse::Evidence;

    // Tiny-VRAM group: 4 total records = 2 permanent + 2 stage_rings slots.
    let experts = 8usize;
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let host = HostTier {
        records: vec![experts as u32],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier {
            bank_maps: vec![StampedMap::new(cpu_words(&[experts as u64, map_words], 0))],
            tier_maps: vec![LandingMap::Tier(cpu_words(&[experts as u64], -1))],
            lent: (0, 0),
            stage_slots: vec![2],
        },
    };
    let mut directory = Directory::new(&[4], &[0], &[experts], host);
    let steps = steps();
    steps.begin_through(3);
    let mut tier_recs = Vec::new();
    for expert in 0..experts {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
            panic!("host tier has room");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(1));
        tier_recs.push(grant.record());
    }

    // Fill both permanent `Free` slots with demand landings (slots 2 and 3).
    let DemandSlotClaim::Allocated { slot: s0, .. } = directory.claim_demand_landing(0, tier_recs[0], false, |_| true) else {
        panic!("expected free slot 0");
    };
    let DemandSlotClaim::Allocated { slot: s1, .. } = directory.claim_demand_landing(0, tier_recs[1], false, |_| true) else {
        panic!("expected free slot 1");
    };
    assert!(s0 >= 2 && s1 >= 2, "initial claims must come from Free permanent slots");
    assert_eq!(directory.bank_free_count(0), 0);

    // Next claim with `Free` empty borrows an unpinned `StageIdle` slot (0 or 1)
    // instead of returning `Overflow`!
    let DemandSlotClaim::Allocated { staged, slot: borrowed } =
        directory.claim_demand_landing(0, tier_recs[2], false, |_| true)
    else {
        panic!("expected tiny-VRAM borrow from StageIdle instead of Overflow");
    };
    assert!(borrowed < 2, "borrowed slot {borrowed} must come from stage_rings 0..2");

    // Even if `Reuse` ranks the expert strongly, `finish_demand_landing` MUST
    // NEVER promote a borrowed staging slot to `Record::Held`, and MUST return
    // it to `Record::StageIdle` so `free_stage_slot` sees it again.
    let ranking = directory.ranking();
    Reuse::of(&ranking).read(
        &RoutedRead { sequence: 1, layer: 0, choices: 4, experts: vec![2] },
        None,
        Evidence::Decode,
    );
    let promoted = directory.finish_demand_landing(0, borrowed, staged, at(1, 0), &steps, true);
    assert!(!promoted, "borrowed StageIdle slot must NEVER be promoted to Record::Held");
    assert_eq!(directory.held(0, 2), None, "expert 2 must not be held in permanent VRAM");
    assert_eq!(
        directory.free_stage_slot(0, borrowed as u64),
        Some(borrowed),
        "borrowed slot must be returned to Record::StageIdle"
    );
}

/// Fix 1: When `WriteBack::pays()` is false (Link/HostMemory binds) on a
/// non-covering `VRAM ⊥ HostTier` bank, promoting an expert to VRAM keeps its
/// RAM copy as lowest-priority `Kept::Banked` (NOT eagerly surrendered). When
/// VRAM later evicts that expert (`replenish_demand_free` / `left_bank`), it
/// transitions `Kept::Banked -> Kept::Alone` in `HostTier` at zero D2H cost
/// (zero NVMe re-read!) unless `HostTier` actually overwrote that `Kept::Banked`
/// slot for a genuinely cold `Kept::Alone` miss.
#[test]
fn non_paying_write_back_keeps_banked_ram_copy_so_vram_eviction_flips_to_alone_at_zero_cost() {
    let steps = steps();
    steps.begin_through(10);
    // 1 group, 1 layer of 8 experts; VRAM = 2 slots, HostTier = 3 slots (non-covering).
    let mut directory = Directory::new(&[2], &[0], &[8], HostTier::replay(vec![3]));
    directory.price_copies(0, Some(0.));
    // Link/HostMemory binds -> WriteBack::pays() is always false!
    let link_bound = WriteBack {
        terms: magnitude_cost_model::Terms { compute: 0.050, drive: 0.025, link: 0.060, cpu: 0.010, memory: 0.060 },
        drive: 0.005,
        link: 0.002,
    };
    assert!(!link_bound.can_pay());
    assert!(!link_bound.pays(10.0, 1.0));
    directory.price_write_backs(0, Some(link_bound));

    // Land experts 0, 1 into HostTier and promote both into VRAM.
    for expert in 0..2 {
        let record = directory.land(0, expert, 0).unwrap();
        directory.landed(0, record, expert, Some(1));
    }
    log(&directory, 2, 0, &[0, 1]);
    let asked = directory.promote();
    assert_eq!(asked.promotions.len(), 2);
    for p in &asked.promotions {
        directory.complete(p);
    }

    // `surrender_banked` and `promotions` must NOT eagerly unmap `Kept::Banked`
    // while `WriteBack::can_pay()` is false!
    assert!(!directory.has_surrenders());
    assert!(directory.surrender_banked(&Unbegun::at(3)).is_empty());
    assert!(directory.in_tier(0, 0) && directory.in_tier(0, 1), "banked RAM copies stay resident");
    assert_eq!(directory.alone(0), 0, "banked copies do not count as Kept::Alone");

    // Evict one expert from VRAM via `replenish_demand_free`: zero D2H write-back,
    // and the evicted expert flips `Kept::Banked -> Kept::Alone` in `HostTier`!
    let released = directory.replenish_demand_free(0, 1, &Unbegun::at(4), |_| true);
    assert_eq!(released, 1);
    assert!(directory.backs_queued().is_empty(), "zero D2H write-back when Kept::Banked copy is still in RAM");
    directory.drain(at(4, 1), &steps);
    assert_eq!(directory.held_count(0), 1);
    let evicted_expert = if directory.held(0, 0).is_none() { 0 } else { 1 };
    let kept_in_vram = 1 - evicted_expert;
    assert!(
        directory.in_tier(0, evicted_expert),
        "evicted VRAM expert {evicted_expert} must remain resident in HostTier (zero NVMe re-read!)"
    );
    assert_eq!(directory.alone(0), 1, "evicted VRAM expert flipped Kept::Banked -> Kept::Alone");

    // Meanwhile, the remaining `Kept::Banked` copy (`kept_in_vram`) is still the
    // lowest-priority eviction victim (`Kept::Banked < Kept::Alone`): landing cold
    // experts 2, 3 into the 3-slot HostTier takes the free slot first, then evicts
    // `Kept::Banked` (`kept_in_vram`) before ever touching `Kept::Alone` (`evicted_expert`)!
    let r2 = directory.land(0, 2, 0).unwrap();
    directory.landed(0, r2, 2, Some(5));
    assert!(directory.in_tier(0, kept_in_vram), "free slot used first");
    // HostTier is now full: {evicted_expert (Alone), 2 (Alone), kept_in_vram (Banked)}.
    // Refilling 1 slot for a cold miss must pick `kept_in_vram` (Kept::Banked) first!
    let mut dir_refill = Directory::new(&[1], &[0], &[6], HostTier::replay(vec![2]).reserving(1));
    dir_refill.price_copies(0, Some(0.));
    dir_refill.price_write_backs(0, Some(link_bound));
    let a0 = dir_refill.land(0, 0, 0).unwrap();
    dir_refill.landed(0, a0, 0, Some(1));
    log(&dir_refill, 2, 0, &[0]);
    let p = dir_refill.promote();
    dir_refill.complete(&p.promotions[0]);
    assert!(dir_refill.in_tier(0, 0), "0 stays Kept::Banked in HostTier");
    // Land cold expert 1 into the remaining ready slot -> reserve (1) is now short,
    // so refill_victims must evict `Kept::Banked` (0) instead of `Kept::Alone` (1)!
    let a1 = dir_refill.land(0, 1, 0).unwrap();
    dir_refill.landed(0, a1, 1, Some(3));
    let victims = dir_refill.refill_victims(false);
    assert_eq!(victims[0].len(), 1);
    assert_eq!(victims[0][0].1, (0, 0), "Kept::Banked (0) must be evicted before Kept::Alone (1)");
}

/// Fix 3: `Directory::refill_short` and `refill_victims` bound the reserve by
/// file-only lacked experts (`lacked - vram_lacked`) rather than raw
/// `HostRecords::lacked()`, so a disjoint `VRAM + HostTier` bank covering all
/// experts never evicts from `HostTier` to hold an unnecessary reserve.
#[test]
fn refill_short_bounds_by_file_only_lacked_experts_excluding_vram() {
    let steps = steps();
    steps.begin_through(10);
    // 1 group, 1 layer of 5 experts; VRAM = 2 slots, HostTier = 3 slots, reserve = 2.
    // Together VRAM (2) + HostTier (3) = 5 cover all 5 experts!
    let mut directory = Directory::new(&[2], &[0], &[5], HostTier::replay(vec![3]).reserving(2));
    directory.price_copies(0, Some(0.));
    directory.price_write_backs(
        0,
        Some(WriteBack {
            terms: magnitude_cost_model::Terms { compute: 0., drive: 4., link: 0., cpu: 0., memory: 0. },
            drive: 1.,
            link: 0.,
        }),
    );
    // Promote experts 0, 1 into VRAM and surrender their HostTier copies so VRAM = {0, 1}.
    for expert in 0..2 {
        let rec = directory.land(0, expert, 0).unwrap();
        directory.landed(0, rec, expert, Some(1));
    }
    log(&directory, 2, 0, &[0, 1]);
    let asked = directory.promote();
    for p in &asked.promotions {
        directory.complete(p);
    }
    directory.surrender_banked(&Unbegun::at(3));
    directory.retire_graces(at(3, 1), &steps);
    assert_eq!(directory.held_count(0), 2);
    assert_eq!(directory.host(0).ready(), 3);

    // Land remaining experts 2, 3, 4 into HostTier -> all 5 experts are now in memory!
    for expert in 2..5 {
        let rec = directory.land(0, expert, 0).unwrap();
        directory.landed(0, rec, expert, Some(4));
    }
    // Raw `host(0).lacked()` is 2 (experts 0 and 1 in VRAM), but file-only lacked is 0!
    assert_eq!(directory.host(0).lacked(), 2);
    assert!(
        !directory.refill_short(),
        "when VRAM + HostTier cover all experts, refill_short must be false (0 file-only lacked)"
    );
    assert!(directory.refill_victims(false)[0].is_empty());
}

/// Dynamic `set_decode_rows` scales `DecodeTransientBudget::demand_floor` to
/// actual active `M` (`1` -> `4` -> `1`) using `ceil(min(E, M * K) * p_miss)`
/// rather than compile-time `VERIFY_ROWS = 16`, preserving permanent VRAM on
/// 8-group DeepSeek-like banks (`~176` total reserve at M=4 instead of `701`).
#[test]
fn set_decode_rows_scales_dynamically_for_8_group_deepseek() {
    let group_layers = [20usize, 4, 4, 4, 2, 2, 2, 2];
    let present: Vec<usize> = group_layers.iter().map(|&n| n * 45).collect();
    let mut layer_groups = Vec::new();
    let mut layer_experts = Vec::new();
    for (g, &n) in group_layers.iter().enumerate() {
        for _ in 0..n {
            layer_groups.push(g);
            layer_experts.push(384usize);
        }
    }
    let host = HostTier::replay(vec![200; 8]);
    let mut directory = Directory::new(&present, &layer_groups, &layer_experts, host);
    let k = 6usize;

    // Initial M=1: each group reserves K=6 (48 total across 8 groups).
    directory.configure_decode_budget(1, k, 3);
    let reserve_m1: usize = (0..8).map(|g| directory.reserved(g)).sum();
    assert_eq!(reserve_m1, 8 * k);

    // First M=4 verify step: each group scales to ceil(24 * (1 - 45/384)) = 22 (176 total, NOT 701!).
    directory.set_decode_rows(4);
    let reserve_m4: usize = (0..8).map(|g| directory.reserved(g)).sum();
    assert_eq!(reserve_m4, 8 * 22);
    assert!(reserve_m4 < 200, "M=4 8-group reserve must stay ~176 (not 701!), got {reserve_m4}");

    // Returning to M=1 plain decode shrinks reserve back to 8 * K = 48 instead
    // of permanently locking to a wide verify spike.
    directory.set_decode_rows(1);
    let reserve_after_m1: usize = (0..8).map(|g| directory.reserved(g)).sum();
    assert_eq!(reserve_after_m1, 8 * k, "live set_decode_rows shrinks back to M=1 floor");

    // 8 target groups + 1 non-splitting DSpark draft group (16 slots/layer = 48):
    // draft group 8 keeps `demand_floor == 0` (`vram_reserve == 0`) so total
    // M=1 reserve is `8 * K = 48` (NOT `9 * K = 54`) and stays `0` at `M=4`.
    let mut with_draft_present: Vec<usize> = present.clone();
    with_draft_present.push(48);
    let mut with_draft_groups = layer_groups.clone();
    let mut with_draft_experts = layer_experts.clone();
    for _ in 0..3 {
        with_draft_groups.push(8);
        with_draft_experts.push(64);
    }
    let mut with_draft = Directory::new(&with_draft_present, &with_draft_groups, &with_draft_experts, HostTier::replay(vec![200; 9]));
    with_draft.configure_decode_budget(1, k, 3);
    with_draft.set_group_splitting(8, false);
    assert_eq!((0..9).map(|g| with_draft.reserved(g)).sum::<usize>(), 8 * k, "M=1 reserve is 8*K=48, NOT 54");
    with_draft.set_decode_rows(4);
    assert!(with_draft.reserved(0) > 0, "splitting target group retains M=4 demand_floor");
    assert_eq!(with_draft.decode_budget(8).demand_floor, 0, "non-splitting draft group 8 must have demand_floor == 0");
    assert_eq!(with_draft.reserved(8), 0, "non-splitting 16-slot/layer draft group 8 must keep vram_reserve == 0");
}

/// Real DeepSeek-V4.1 8-group geometry on RTX 5090-128g (`E=384, K=6, 8 groups,
/// vram_usable=1258, host_slots=317/layer, B_dma_alone=43.57 GB/s,
/// B_mem=47.80 GB/s, M=4`) asserting all 5 adversarial-audit invariants:
/// 1. `total_vram_reserve < 200` slots across all 8 groups (`< 16%` of 1,258, NOT 701).
/// 2. `effective_cpu_for` returns `None` on Gen5 2-ch DDR5 (0 CPU-site experts
///    even at `M=4` with 20 misses/layer) and `Some(...)` on Gen4 3090.
/// 3. `HostTier` per-group reserve is `ceil(71/8) = 9` (NOT `71 × 8 = 568`).
/// 4. When `WriteBack::pays() == false`, promoting to VRAM and evicting from
///    VRAM leaves the expert resident in `HostTier` (`Kept::Banked -> Kept::Alone`, 0 NVMe drop).
/// 5. Speculative `Drive` prefetch defers when `host_memory_binds() && demand_active()`,
///    and `Parked(0) < Streamed(1) < Expected(2)` strictly prioritizes decode demand.
#[test]
fn deepseek_v41_rtx5090_8_group_all_five_invariants() {
    use crate::experts::bank::group_tier_reserve;
    use crate::experts::transport::ExpertTransport;
    use magnitude_cost_model::{
        CpuLines, ExpertLines, GpuLines, HostLines, Moves, Nanoseconds, Pass, PassLine,
        PathAdmission, ReadLine, RowCost, RowLine, Site, Terms,
    };
    use seismic_io::Want;

    // 40 layers across 8 mixed-quant groups, totaling 1,258 usable VRAM slots
    // (~31.45 slots/layer; group counts sum to 1,258) and 317 HostTier slots/layer.
    let group_layers = [20usize, 4, 4, 4, 2, 2, 2, 2];
    let present = [628usize, 126, 126, 126, 63, 63, 63, 63];
    assert_eq!(present.iter().sum::<usize>(), 1_258);
    let host_records: Vec<u32> = group_layers.iter().map(|&n| (n * 317) as u32).collect();
    let mut layer_groups = Vec::new();
    let mut layer_experts = Vec::new();
    for (g, &n) in group_layers.iter().enumerate() {
        for _ in 0..n {
            layer_groups.push(g);
            layer_experts.push(384usize);
        }
    }
    let k = 6usize;

    // Invariant 3: HostTier per-group reserve is ceil(71/8) = 9 (72 total, NOT 568!).
    let host_reserve_per_group = group_tier_reserve(71, 8, k);
    assert_eq!(host_reserve_per_group, 9);
    assert_eq!(host_reserve_per_group * 8, 72);

    let host = HostTier::replay(host_records).reserving(host_reserve_per_group);
    let mut directory = Directory::new(&present, &layer_groups, &layer_experts, host);

    // Invariant 1: Dynamic M=4 decode reserve across all 8 groups < 200 (< 16% of 1,258, NOT 701!).
    directory.configure_decode_budget(1, k, 3);
    directory.set_decode_rows(4);
    let total_vram_reserve: usize = (0..8).map(|g| directory.reserved(g)).sum();
    assert!(
        total_vram_reserve < 200 && (total_vram_reserve as f64 / 1_258.0) < 0.16,
        "expected total_vram_reserve < 200 (<16% of 1,258), got {total_vram_reserve}"
    );

    // Invariant 2: Gen5 2-ch DDR5 (`B_dma_alone = 43.57 GB/s`, `B_mem = 47.80 GB/s`,
    // `resident = 12 µs`, `B_joint = 43.0 GB/s` <= `B_dma_alone`) suppresses CPU-site
    // (`effective_cpu_for == None`, 0 CPU-site experts even at M=4 with 20 misses/layer),
    // while Gen4 3090 (`B_dma_alone = 19.31 GB/s`, `B_joint = 36.90 GB/s`) keeps CPU-site enabled.
    let record_bytes = 8_000_000u64;
    let ns = |v: u32| Nanoseconds::try_from(v).unwrap();
    let ns_of_gbs = |gbs: f64| Nanoseconds::of(record_bytes as f64 / (gbs * 1e9)).unwrap();
    let resident_ns = 12_000u32;
    let make_lines = |b_dma_gbs: f64, b_mem_gbs: f64, b_joint_gbs: f64| ExpertLines {
        host: HostLines {
            cpu: Some(CpuLines {
                expert: RowLine { one: ns(180_000), row: RowCost::Measured(ns(2_000)) },
                handshake: ns(4_000),
            }),
            memory: Some(ns_of_gbs(b_mem_gbs)),
            joint_memory: Some(ns_of_gbs(b_joint_gbs)),
            landing: None,
            record_bytes,
        },
        gpu: GpuLines {
            link: PassLine { weights: ns(ns_of_gbs(b_dma_gbs).get() as u32 + resident_ns), row: ns(100) },
            resident: PassLine { weights: ns(resident_ns), row: ns(100) },
            moves: Moves::NONE,
            baseline_dma: None,
            contended_link: None,
        },
    };
    let gen5 = make_lines(43.57, 47.80, 43.0);
    assert_eq!(gen5.effective_cpu(), None, "Gen5 2-ch DDR5 must suppress CPU-site");
    let hits = [magnitude_cost_model::Hits { rows: 1.0, experts: 4.0 }];
    let misses = vec![magnitude_cost_model::Miss { rows: 1.0, ready_seconds: 0.0 }; 20];
    let served_gen5 = gen5.serve(Pass::Decode, &hits, &misses);
    assert_eq!(
        served_gen5.sites.iter().filter(|&&s| s == Site::Cpu).count(),
        0,
        "0 CPU-site experts even at M=4 with 20 misses/layer"
    );
    assert_eq!(served_gen5.sites.iter().filter(|&&s| s == Site::Link).count(), 20);

    let gen4 = make_lines(19.31, 47.80, 36.90);
    assert!(gen4.effective_cpu().is_some(), "Gen4 3090 must keep CPU-site enabled");

    // Invariant 4: When `WriteBack::pays() == false` (Link/HostMemory binds),
    // promoting an expert to VRAM and evicting it from VRAM leaves it resident
    // in `HostTier` (`Kept::Banked -> Kept::Alone`, 0 NVMe drop, 0 D2H write-back).
    let steps = steps();
    steps.begin_through(10);
    let link_bound = WriteBack {
        terms: Terms { compute: 0.050, drive: 0.025, link: 0.060, cpu: 0.010, memory: 0.060 },
        drive: 0.005,
        link: 0.002,
    };
    assert!(!link_bound.can_pay());
    for layer in 0..layer_groups.len() {
        directory.price_copies(layer, Some(0.0));
        directory.price_write_backs(layer, Some(link_bound));
    }
    let rec = directory.land(0, 0, 0).expect("free host slot");
    directory.landed(0, rec, 0, Some(1));
    log(&directory, 2, 0, &[0]);
    let asked = directory.promote();
    let p0 = asked.promotions.iter().find(|p| p.layer == 0 && p.expert == 0).expect("expert 0 promoted");
    directory.complete(p0);
    assert!(directory.surrender_banked(&Unbegun::at(3)).is_empty(), "must not eagerly surrender Kept::Banked");
    assert!(directory.in_tier(0, 0) && directory.held(0, 0).is_some());
    // Evict expert 0 from VRAM via shrink/eviction -> flips Kept::Banked -> Kept::Alone in HostTier!
    directory.evicted(0, &[p0.record.index()], |_, _| Some(2));
    assert!(directory.held(0, 0).is_none());
    assert!(directory.in_tier(0, 0), "evicted VRAM expert must stay resident in HostTier (0 NVMe drop)");
    assert!(directory.host(0).holds_alone((0, 0)), "Kept::Banked flipped to Kept::Alone at zero D2H cost");

    // Invariant 5: `host_memory_binds()` and `transport.demand_active()` track
    // shared HostMemory binding and live demand flight, and strict deadline
    // ordering `Parked(0) < Streamed(1) < Expected(2) < None(3)`.
    let drive_line = ReadLine::new(0.14e-3, 1.0 / 4.6e9).unwrap();
    let drive_mem_bound = PathAdmission::drive(
        Terms { compute: 0.050, drive: 0.025, link: 0.035, cpu: 0.010, memory: 0.060 },
        Some(drive_line),
        Some(80e-6),
        None,
    );
    assert!(drive_mem_bound.host_memory_binds());
    let transport = ExpertTransport::new_registry_only();
    assert!(!(drive_mem_bound.host_memory_binds() && transport.demand_active()));
    {
        let _guard = transport.enter_demand_flight();
        assert!(drive_mem_bound.host_memory_binds() && transport.demand_active());
    }
    let streamed = Want::streamed();
    let parked = Want::now();
    let expected = Want::expected(1, 1.0);
    let undated = Want::undated(1.0);
    assert!(parked < streamed && streamed < expected && expected < undated);
}

/// Under async `tier_copy_dma` (`SeedMode::FreeOnly`), `Directory::seed_with`
/// MUST never evict live `Record::Held` slots belonging to another layer in
/// the same bank group — even when the bank is completely full (zero `Free`
/// slots) and the prior layer's slots are stale from an earlier request.
#[test]
fn seed_with_free_only_never_evicts_held_slots_even_when_bank_is_full() {
    use crate::experts::directory::SeedMode;
    use crate::experts::host_records::{Claim, Taking};

    let experts = 8usize;
    let lent = 1usize;
    let total_records = 5usize; // 0 = lent, 1..5 = 4 permanent slots shared by layers 0 and 1
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let host = HostTier {
        records: vec![16],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier {
            bank_maps: vec![
                StampedMap::new(cpu_words(&[experts as u64, map_words], 0)),
                StampedMap::new(cpu_words(&[experts as u64, map_words], 0)),
            ],
            tier_maps: vec![
                LandingMap::Tier(cpu_words(&[experts as u64], -1)),
                LandingMap::Tier(cpu_words(&[experts as u64], -1)),
            ],
            lent: (0, lent),
            stage_slots: vec![0],
        },
    };
    let mut directory = Directory::new(&[total_records], &[0, 0], &[experts, experts], host);
    let steps = steps();
    steps.begin_through(20);

    // Fill all 4 permanent slots (1..5): 2 with Layer 0's `Record::Held`
    // experts and 2 with Layer 1's (4.div_ceil(2) = 2 per layer -> zero `Free`
    // slots left in the bank).
    for layer in 0..2usize {
        for expert in 0..2usize {
            let Claim::Taken(grant) = directory.claim_landing(layer, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
                panic!("expected host slot for ({layer}, {expert})");
            };
            directory.landed(layer, HostRecord(grant.record()), expert, Some(1));
        }
    }
    let group1 = ScratchGroup::at(1, 1);
    assert!(directory.lease(0, group1, false, None));
    let routed_l0 = [(0usize, 10u32, 1u64), (1, 10, 1)];
    let (seeds_l0, _) = directory.seed_with(
        at(1, 0),
        group1,
        0,
        &routed_l0,
        &[],
        |_| true,
        SeedMode::FreeOnly,
        |_, _| true,
    );
    assert_eq!(seeds_l0.len(), 2, "layer 0 fills its 2 permanent free slots");
    let routed_l1_init = [(0usize, 10u32, 1u64), (1, 10, 1)];
    let (seeds_l1_init, _) = directory.seed_with(
        at(1, 1),
        group1,
        1,
        &routed_l1_init,
        &[],
        |_| true,
        SeedMode::FreeOnly,
        |_, _| true,
    );
    assert_eq!(seeds_l1_init.len(), 2, "layer 1 fills the remaining 2 permanent free slots (bank now 100% full)");
    // Commit all 4 seeds so they become `Record::Held` (0 Free slots remain).
    directory.retire(
        at(5, 0),
        |_, _| 1,
        |_| Ending::Ended,
        &steps,
    );
    directory.end_prompt_group(0);
    let l0_slots: Vec<(usize, usize)> = (0..2)
        .map(|e| (e, directory.held(0, e).expect("layer 0 expert held")))
        .collect();

    // New prompt group (Request 2) at step 10: bank is completely FULL (0 Free
    // slots). Layer 1 requests `SeedMode::FreeOnly` with high-count cold experts.
    let group2 = ScratchGroup::at(10, 1);
    assert!(directory.lease(0, group2, false, None));
    let routed_l1 = [(2usize, 100u32, 10u64), (3, 100, 10), (4, 100, 10), (5, 100, 10)];
    let (seeds_l1, evicted_l1) = directory.seed_with(
        at(10, 1),
        group2,
        1,
        &routed_l1,
        &[],
        |_| true,
        SeedMode::FreeOnly,
        |_, _| true,
    );
    assert!(seeds_l1.is_empty(), "FreeOnly must skip seeding on full bank (0 Free slots)");
    assert!(evicted_l1.is_empty(), "FreeOnly must NEVER evict Layer 0's Record::Held slots");
    for &(expert, expected_slot) in &l0_slots {
        assert_eq!(
            directory.held(0, expert),
            Some(expected_slot),
            "Layer 0 held expert {expert} at slot {expected_slot} must remain intact"
        );
    }
}

/// Even under `SeedMode::AllowEvict`, a `Record::Held` slot promoted by
/// `finish_demand_landings_batch` while still pinned by an active transport
/// consumer lease (`can_reuse == false`) MUST NEVER be selected as a seed
/// victim or overwritten.
#[test]
fn seed_with_allow_evict_never_evicts_transport_pinned_held_slot() {
    use crate::experts::directory::SeedMode;
    use crate::experts::host_records::{Claim, Taking};
    use crate::experts::transport::{DestSlot, ExpertTransport};

    let experts = 6usize;
    let lent = 1usize;
    let total_records = 2usize; // 0 = lent, 1 = sole permanent slot
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let host = HostTier {
        records: vec![8],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier {
            bank_maps: vec![StampedMap::new(cpu_words(&[experts as u64, map_words], 0))],
            tier_maps: vec![LandingMap::Tier(cpu_words(&[experts as u64], -1))],
            lent: (0, lent),
            stage_slots: vec![0],
        },
    };
    let mut directory = Directory::new(&[total_records], &[0], &[experts], host);
    let steps = steps();
    steps.begin_through(20);

    for expert in 0..experts {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
            panic!("expected host slot for expert {expert}");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(1));
    }

    // Claim permanent slot 1 via DemandLanding and pin it in `ExpertTransport`
    // with an active consumer lease (`can_reuse_slot == false`).
    let transport = ExpertTransport::new_registry_only();
    let DemandSlotClaim::Allocated { staged, slot } =
        directory.claim_demand_landing(0, 0, true, |_| true)
    else {
        panic!("expected permanent slot 1 for demand landing");
    };
    assert_eq!(slot, 1);
    let dest = DestSlot { group: 0, slot };
    let (tid, lease) = transport.begin_demand_transfer(0, 0, 0, 0, dest);
    transport.complete_dma(tid);
    assert!(!transport.can_reuse_slot(dest), "consumer lease still pins slot 1");

    // `finish_demand_landings_batch` promotes slot 1 to `Record::Held` while
    // `can_reuse == false` (the consumer still holds `lease`).
    let promoted = directory.finish_demand_landings_batch(&[(0, slot, staged, false)], at(1, 0), &steps);
    assert!(promoted, "slot 1 promoted to Record::Held while consumer lease active");
    assert_eq!(directory.held(0, 0), Some(slot));

    // Next prefill attempts `SeedMode::AllowEvict` with a much higher-ranked
    // expert (expert 5, count 1000). Because `can_reuse_slot(dest)` is false,
    // slot 1 MUST NOT be evicted!
    let group = ScratchGroup::at(10, 1);
    assert!(directory.lease(0, group, false, None));
    let (seeds, evicted) = transport.with_slot_reuse(|can_reuse| {
        directory.seed_with(
            at(10, 0),
            group,
            0,
            &[(5, 1000, 10)],
            &[],
            |_| true,
            SeedMode::AllowEvict,
            |g, s| can_reuse(DestSlot { group: g, slot: s }),
        )
    });
    assert!(seeds.is_empty(), "pinned Record::Held slot must NOT be taken as seed victim");
    assert!(evicted.is_empty());
    assert_eq!(directory.held(0, 0), Some(slot));

    // Once the consumer drops its lease (`can_reuse_slot == true`),
    // `SeedMode::AllowEvict` may safely select slot 1 as a victim.
    drop(lease);
    assert!(transport.can_reuse_slot(dest));
    let (seeds_after, evicted_after) = transport.with_slot_reuse(|can_reuse| {
        directory.seed_with(
            at(10, 0),
            group,
            0,
            &[(5, 1000, 10)],
            &[],
            |_| true,
            SeedMode::AllowEvict,
            |g, s| can_reuse(DestSlot { group: g, slot: s }),
        )
    });
    assert_eq!(seeds_after.len(), 1);
    assert_eq!(seeds_after[0].0, 5);
    assert_eq!(seeds_after[0].1.index(), slot);
    assert_eq!(evicted_after, vec![(0, 0)]);
}

/// Fix 4: An already-served demand miss at `(S, layer 0)` must NOT receive
/// `+1.0` `Unlogged` immunity when a later layer `(S, layer 1)` needs a demand
/// victim — the 1-shot cold demand miss is evicted while the genuinely hot
/// expert survives. Conversely, a strictly future unconsumed read-ahead at
/// `(S, layer 2)` (`layer > parked_layer`) or `(S + 1, ...)` retains protection.
#[test]
fn already_served_demand_miss_does_not_get_unlogged_immunity_against_hot_expert() {
    use crate::experts::horizon::{Advance, Tokens};
    use crate::experts::host_records::{Demand, Taking};
    use crate::experts::reuse::{Evidence, Reuse};

    // 3 layers sharing 1 pooled group of 2 records (unified memory host tier).
    let steps = std::sync::Arc::new(steps());
    steps.begin_through(41);
    let (mut directory, _) = one_store_on(&steps, 2, 3, 16);
    let time = directory.time().clone();
    let ranking = directory.ranking();

    // Warm up Reuse for steps 1..=40: (layer 0, expert 0) is hot every step;
    // (layer 0, expert 7) and (layer 2, expert 9) are never read.
    for step in 1..=40u64 {
        time.lock().register(step, Advance::Known(Tokens::new(1.0)));
        let mut reuse = Reuse::of(&ranking);
        reuse.read(&RoutedRead { sequence: step, layer: 0, experts: vec![0], choices: 1 }, None, Evidence::Decode);
        reuse.read(&RoutedRead { sequence: step, layer: 1, experts: vec![1 + (step as usize % 10)], choices: 1 }, None, Evidence::Decode);
        reuse.read(&RoutedRead { sequence: step, layer: 2, experts: vec![1 + (step as usize % 10)], choices: 1 }, None, Evidence::Decode);
        if step % 2 == 0 {
            time.lock().moved(0, crate::experts::horizon::Tier::COMPUTE);
        }
    }

    // Step S = 41 (not yet logged into Reuse):
    // - Record 0 holds hot expert (0, 0) from prior steps (read = None).
    // - Record 1 holds 1-shot cold demand miss (0, 7) that just landed at (S=41, layer=0).
    let step = 41u64;
    let Claim::Taken(g0) = directory.claim_landing(0, 0, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
        panic!("free slot for hot (0, 0)");
    };
    directory.landed(0, HostRecord(g0.record()), 0, None);
    let Claim::Taken(g1) = directory.claim_landing(0, 7, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
        panic!("free slot for cold demand miss (0, 7)");
    };
    directory.landed(0, HostRecord(g1.record()), 7, Some(step));

    // Layer 1 in step S=41 parks and claims a demand victim.
    let park_l1 = steps.park(at(step, 1), 3);
    let Claim::Taken(victim) = directory.claim_demand((1, 5), Demand::Lacked, &[], &park_l1) else {
        panic!("expected demand claim to evict a victim");
    };
    assert_eq!(
        victim.record(),
        g1.record(),
        "already-served cold demand miss (0, 7) at (S, layer 0) must be evicted, not hot expert (0, 0)"
    );
    directory.landed(1, HostRecord(victim.record()), 5, Some(step));

    // Now test the converse: a FUTURE unconsumed read-ahead for (S=41, layer=2)
    // while parked at (S=41, layer=1) -> layer 2 > parked_layer 1 retains +1.0 protection.
    let steps2 = std::sync::Arc::new(super::Steps::new(
        crate::planning::FlightDepth::SERIAL,
        1,
        crate::experts::exchange::Pulse::new(),
    ));
    steps2.begin_through(41);
    let (mut dir2, _) = one_store_on(&steps2, 2, 3, 16);
    let time2 = dir2.time().clone();
    let ranking2 = dir2.ranking();
    for s in 1..=40u64 {
        time2.lock().register(s, Advance::Known(Tokens::new(1.0)));
        let mut reuse = Reuse::of(&ranking2);
        // Expert (0, 0) chosen every 4th step (~25%) so N < 1.0 within short τ.
        let e0 = if s % 4 == 0 { 0 } else { 1 + (s as usize % 12) };
        reuse.read(&RoutedRead { sequence: s, layer: 0, experts: vec![e0], choices: 1 }, None, Evidence::Decode);
        reuse.read(&RoutedRead { sequence: s, layer: 1, experts: vec![1 + (s as usize % 12)], choices: 1 }, None, Evidence::Decode);
        reuse.read(&RoutedRead { sequence: s, layer: 2, experts: vec![1 + (s as usize % 12)], choices: 1 }, None, Evidence::Decode);
        time2.lock().moved(0, crate::experts::horizon::Tier::COMPUTE);
    }
    let Claim::Taken(h0) = dir2.claim_landing(0, 0, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
        panic!("free slot for (0, 0)");
    };
    dir2.landed(0, HostRecord(h0.record()), 0, None);
    let Claim::Taken(h1) = dir2.claim_landing(2, 9, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
        panic!("free slot for future read-ahead (2, 9)");
    };
    dir2.landed(2, HostRecord(h1.record()), 9, Some(step));
    let park2 = steps2.park(at(step, 1), 3);
    let Claim::Taken(evicted2) = dir2.claim_demand((1, 5), Demand::Lacked, &[], &park2) else {
        panic!("expected demand claim");
    };
    assert_eq!(
        evicted2.record(),
        h0.record(),
        "future unconsumed read-ahead (2, 9) at layer 2 > parked layer 1 must keep +1.0 protection"
    );
}

/// Synthetic mixed-layer pooled replay (stationary Zipf + short-burst layers)
/// comparing online `Recency::Reuse` against clairvoyant Belady `Recency::NextRead`.
#[test]
fn pooled_mixed_layer_replay_tracks_belady_closely() {
    // 4 layers in 1 pooled bank of 12 records, 64 experts per layer, 300 steps:
    // - Layers 0 & 1: stationary Zipf (4 hot experts each hit ~60% of the time + cold tail)
    // - Layers 2 & 3: rapidly shifting 1-shot/short-burst traffic
    let mut reads = Vec::with_capacity(4 * 300);
    for step in 1..=300u64 {
        let s = step as usize;
        for layer in 0..2 {
            let hot = (s + layer) % 4;
            let e = if s % 5 != 0 { hot } else { 4 + ((s * 7 + layer * 13) % 60) };
            reads.push(RoutedRead { sequence: step, layer, experts: vec![e], choices: 1 });
        }
        for layer in 2..4 {
            let burst = ((step - 1) / 3) as usize;
            let e = (burst * 3 + layer * 17) % 64;
            reads.push(RoutedRead { sequence: step, layer, experts: vec![e], choices: 1 });
        }
    }
    let held = vec![Vec::new(); 4];
    let online = replay(&[12], &[0, 0, 0, 0], 64, &held, &reads, Recency::Reuse).expect("online replay");
    let belady = replay(&[12], &[0, 0, 0, 0], 64, &held, &reads, Recency::NextRead).expect("belady replay");
    let online_misses: u64 = online.misses.iter().sum();
    let belady_misses: u64 = belady.misses.iter().sum();
    let ratio = online_misses as f64 / belady_misses.max(1) as f64;
    eprintln!("mixed-layer pooled replay: online={online_misses}, belady={belady_misses}, ratio={ratio:.3}x");
    assert!(
        ratio < 1.35,
        "online ranker ({online_misses}) should be within 1.35x of Belady ({belady_misses}), got {ratio:.3}x"
    );
}

/// `reclaim_served_demand_landings` reclaims ONLY already-served `Record::Held`
/// victims (`steps.served(from, until, v_layer, progress) == true`), never
/// unserved layers (`v_layer >= current_layer`), never pinned slots
/// (`!can_reuse(slot)`), and preserves `HostTier` copies when write-back does
/// not pay (`Kept::Banked -> Kept::Alone`).
#[test]
fn reclaim_served_demand_landings_evicts_only_served_unpinned_victims() {
    use crate::experts::host_records::{Claim, Taking};

    let layers = 2usize;
    let experts = 16usize;
    let present = 6usize; // 6 total VRAM slots, 0 free once filled
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let bank_maps = (0..layers)
        .map(|_| StampedMap::new(cpu_words(&[experts as u64, map_words], 0)))
        .collect::<Vec<_>>();
    let tier_maps = (0..layers)
        .map(|_| crate::experts::map::LandingMap::Tier(cpu_words(&[experts as u64], -1)))
        .collect::<Vec<_>>();
    let host = HostTier {
        records: vec![(layers * experts) as u32],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier { bank_maps, tier_maps, lent: (0, 0), stage_slots: vec![0] },
    };
    let mut directory = Directory::new(&[present], &[0, 0], &[experts, experts], host);
    for layer in 0..layers {
        directory.price_copies(layer, Some(0.0));
        // Zero write-back (discrete GPU where HostTier retains copy / NVMe holds it).
        directory.price_write_backs(layer, None);
        for expert in 0..experts {
            let Claim::Taken(grant) = directory.claim_landing(layer, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
                panic!("host tier has room");
            };
            directory.landed(layer, HostRecord(grant.record()), expert, Some(1));
        }
    }

    let steps = steps();
    let step0 = steps.begin_group(1, None);
    // Log reads so layer 0 experts 0..3 and layer 1 experts 0..3 are promoted into all 6 VRAM slots.
    log(&directory, step0, 0, &[0, 1, 2]);
    log(&directory, step0, 1, &[0, 1, 2]);
    assert!(directory.settle_changes());
    let wants = steps.unbegun(|at| directory.promotions(at, present));
    assert_eq!(wants.promotions.len(), present, "all 6 VRAM slots should be promoted");
    for p in &wants.promotions {
        let tier = directory.hold_tier(p.layer, p.expert).expect("promoted expert is in host tier");
        directory.copy(p.group, p.record, tier, 0);
    }
    let completed: Vec<(usize, BankRecord)> = wants.promotions.iter().map(|p| (p.group, p.record)).collect();
    directory.promoted(completed, at(step0, 0), &steps);
    steps.end_played(step0);
    assert_eq!(directory.bank_free_count(0), 0, "zero free slots remain");

    // Begin step1 and advance to layer 1 (`progress = at(step1, 1)`):
    // Layer 0 is already served; Layer 1 is NOT yet served.
    let step1 = steps.begin_group(1, None);
    let progress = at(step1, 1);

    let layer0_slots: Vec<usize> = (0..3).filter_map(|e| directory.held(0, e)).collect();
    let layer1_slots: Vec<usize> = (0..3).filter_map(|e| directory.held(1, e)).collect();
    assert_eq!(layer0_slots.len(), 3);
    assert_eq!(layer1_slots.len(), 3);

    // Pin one layer-0 slot via `can_reuse` so only 2 of the 3 layer-0 slots are eligible.
    let pinned_slot = layer0_slots[0];
    let overflows = vec![(0, Some((1, 10))), (1, Some((1, 11))), (2, Some((1, 12)))];
    let reclaimed = steps.unbegun_with_served(progress, directory.layers(), |at, park, is_served| {
        directory.reclaim_served_demand_landings(0, &overflows, at, park, progress, is_served, |slot| slot != pinned_slot)
    });

    assert_eq!(
        reclaimed.len(),
        2,
        "only the 2 unpinned served layer-0 slots may be reclaimed (never pinned slot or unserved layer 1)"
    );
    for &(_, _, slot) in &reclaimed {
        assert_ne!(slot, pinned_slot, "pinned slot must never be reclaimed");
        assert!(layer0_slots.contains(&slot), "reclaimed slot {slot} must come from served layer 0, not unserved layer 1");
        assert!(!layer1_slots.contains(&slot), "unserved layer 1 slot {slot} must never be reclaimed");
    }
    // Unserved layer 1 experts remain 100% held.
    for e in 0..3 {
        assert!(directory.held(1, e).is_some(), "layer 1 expert {e} must remain held");
    }
    // Pinned layer 0 expert remains held; the 2 reclaimed slots are now DemandLanding (not Free, not Held).
    assert_eq!(directory.bank_free_count(0), 0);
}

/// At layer 0 of a multi-layer group (where 0 earlier layers exist),
/// `reclaim_served_demand_landings` reclaims held slots from later unsubmitted
/// layers (`v_layer > 0`) by unmapping them at `park.unresolved(v_layer)`
/// (`<= step1`), guaranteeing `step1` sees a miss when it reaches `v_layer`
/// while NEVER reclaiming from the current layer (`v_layer == 0`, whose
/// `DecodeHits` is already in flight).
#[test]
fn reclaim_served_demand_landings_reclaims_later_layers_at_layer_zero_and_unmaps_before_step() {
    use crate::experts::host_records::{Claim, Taking};

    let layers = 3usize;
    let experts = 16usize;
    let present = 9usize; // 3 held per layer, 0 free once filled
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let bank_maps = (0..layers)
        .map(|_| StampedMap::new(cpu_words(&[experts as u64, map_words], 0)))
        .collect::<Vec<_>>();
    let tier_maps = (0..layers)
        .map(|_| crate::experts::map::LandingMap::Tier(cpu_words(&[experts as u64], -1)))
        .collect::<Vec<_>>();
    let host = HostTier {
        records: vec![(layers * experts) as u32],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier { bank_maps, tier_maps, lent: (0, 0), stage_slots: vec![0] },
    };
    let mut directory = Directory::new(&[present], &[0, 0, 0], &[experts, experts, experts], host);
    for layer in 0..layers {
        directory.price_copies(layer, Some(0.0));
        directory.price_write_backs(layer, None);
        for expert in 0..experts {
            let Claim::Taken(grant) = directory.claim_landing(layer, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
                panic!("host tier has room");
            };
            directory.landed(layer, HostRecord(grant.record()), expert, Some(1));
        }
    }

    let steps = steps();
    let step0 = steps.begin_group(1, None);
    for layer in 0..layers {
        log(&directory, step0, layer, &[0, 1, 2]);
    }
    assert!(directory.settle_changes());
    let wants = steps.unbegun(|at| directory.promotions(at, present));
    assert_eq!(wants.promotions.len(), present);
    for p in &wants.promotions {
        let tier = directory.hold_tier(p.layer, p.expert).expect("promoted expert is in host tier");
        directory.copy(p.group, p.record, tier, 0);
    }
    let completed: Vec<(usize, BankRecord)> = wants.promotions.iter().map(|p| (p.group, p.record)).collect();
    directory.promoted(completed, at(step0, 0), &steps);
    steps.end_played(step0);
    assert_eq!(directory.bank_free_count(0), 0);

    // Begin step1 at layer 0 (`progress = at(step1, 0)`):
    // Layer 0 is the CURRENT layer (must NEVER be reclaimed);
    // Layers 1 and 2 have not yet been submitted and MUST be reclaimable with
    // `MAP_UNTIL == step1` so `step1` misses when it reaches them.
    let step1 = steps.begin_group(1, None);
    let progress = at(step1, 0);
    let layer0_slots: Vec<usize> = (0..3).filter_map(|e| directory.held(0, e)).collect();
    assert_eq!(layer0_slots.len(), 3);

    let overflows: Vec<(usize, Option<(usize, usize)>)> = (0..6).map(|i| (i, Some((0, 10 + i)))).collect();
    let reclaimed = steps.unbegun_with_served(progress, directory.layers(), |at, park, is_served| {
        directory.reclaim_served_demand_landings(0, &overflows, at, park, progress, is_served, |_| true)
    });

    assert_eq!(
        reclaimed.len(),
        6,
        "all 6 slots from later layers (1 & 2) must be reclaimed at layer 0"
    );
    for &(_, _, slot) in &reclaimed {
        assert!(!layer0_slots.contains(&slot), "current layer 0 slot {slot} must never be reclaimed");
    }
    // Current layer 0 experts remain 100% held and resolvable for step1.
    for e in 0..3 {
        assert!(directory.held(0, e).is_some(), "layer 0 expert {e} must remain held");
        assert!(directory.resolve(0, e, step1).is_some(), "layer 0 expert {e} must remain mapped for step1");
    }
    // Reclaimed later-layer experts must be unmapped for step1 (`resolve == None`).
    for layer in 1..layers {
        for e in 0..3 {
            assert!(directory.held(layer, e).is_none(), "later layer {layer} expert {e} was reclaimed");
            assert!(
                directory.resolve(layer, e, step1).is_none(),
                "reclaimed later layer {layer} expert {e} must be unmapped before step1 so step1 never reads overwritten slot"
            );
        }
    }
}

/// On a 3-tier bank where held VRAM experts have surrendered their host copies
/// On a 3-tier (`!tier_complete`) bank where `Record::Held` experts have no
/// Host-RAM copy (`Located::Lacked`), `reclaim_served_demand_landings` MUST NOT
/// evict them with `back: None` (which would drop them straight to NVMe): excess
/// misses overflow cleanly to SM zero-copy instead, while `Record::Held` experts
/// that still have a Host-RAM copy (`Located::Kept(_)`) ARE reclaimed losslessly.
#[test]
fn reclaim_served_demand_landings_never_drops_vram_only_victims_on_three_tier_bank() {
    use crate::experts::host_records::{Claim, Taking};

    let layers = 2usize;
    let experts = 16usize;
    let present = 4usize; // 2 held per layer, 0 free once filled
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let bank_maps = (0..layers)
        .map(|_| StampedMap::new(cpu_words(&[experts as u64, map_words], 0)))
        .collect::<Vec<_>>();
    let tier_maps = (0..layers)
        .map(|_| crate::experts::map::LandingMap::Tier(cpu_words(&[experts as u64], -1)))
        .collect::<Vec<_>>();
    let host = HostTier {
        records: vec![8], // 3-tier bank: 4 VRAM, 8 host, 32 total experts
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier { bank_maps, tier_maps, lent: (0, 0), stage_slots: vec![0] },
    };
    let mut directory = Directory::new(&[present], &[0, 0], &[experts, experts], host);
    assert!(!directory.tier_complete(), "3-tier bank must be !tier_complete");
    for layer in 0..layers {
        directory.price_copies(layer, Some(1.0));
        for expert in 0..4 {
            let Claim::Taken(grant) = directory.claim_landing(layer, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
                panic!("host tier has room");
            };
            directory.landed(layer, HostRecord(grant.record()), expert, Some(1));
        }
    }

    let steps = steps();
    let step0 = steps.begin_group(1, None);
    log(&directory, step0, 0, &[0, 1]);
    log(&directory, step0, 1, &[0, 1]);
    assert!(directory.settle_changes());
    let wants = steps.unbegun(|at| directory.promotions(at, present));
    assert_eq!(wants.promotions.len(), present);
    for p in &wants.promotions {
        let tier = directory.hold_tier(p.layer, p.expert).expect("promoted expert is in host tier");
        directory.copy(p.group, p.record, tier, 0);
    }
    let completed: Vec<(usize, BankRecord)> = wants.promotions.iter().map(|p| (p.group, p.record)).collect();
    directory.promoted(completed, at(step0, 0), &steps);
    steps.end_played(step0);

    // Price write-backs so `group_write_back_pays` holds, then surrender host
    // copies of promoted experts so they become `Located::Lacked` (VRAM-only).
    let terms = magnitude_cost_model::Terms { compute: 0.0, drive: 1.0, link: 0.0, cpu: 0.0, memory: 0.0 };
    let price = Some(WriteBack { terms, drive: 1.0, link: 0.0 });
    for layer in 0..layers {
        directory.price_write_backs(layer, price);
    }
    let surrendered = steps.unbegun(|at| directory.surrender_banked(at));
    assert!(!surrendered.is_empty(), "promoted experts must surrender host copies");
    directory.retire_graces(past(step0), &steps);
    assert_eq!(directory.located(0, 0), Located::Lacked);
    assert_eq!(directory.located(0, 1), Located::Lacked);
    for s in 1..=20 {
        log(&directory, step0 + s, 0, &[0, 1]);
        log(&directory, step0 + s, 1, &[0, 1]);
    }
    directory.time().lock().moved(0, crate::experts::horizon::Tier::COMPUTE);

    // Begin step1 at layer 1: layer 0's 2 held experts are served, but they are
    // `Located::Lacked` (VRAM-only), so `reclaim_served_demand_landings` MUST NOT
    // evict them with `back: None` — excess misses overflow cleanly to zero-copy.
    let step1 = steps.begin_group(1, None);
    let progress = at(step1, 1);
    let overflows = vec![(0, Some((1, 10))), (1, Some((1, 11)))];
    let reclaimed = steps.unbegun_with_served(progress, directory.layers(), |at, park, is_served| {
        directory.reclaim_served_demand_landings(0, &overflows, at, park, progress, is_served, |_| true)
    });
    assert!(
        reclaimed.is_empty(),
        "must never drop Located::Lacked VRAM-only experts to NVMe for transient demand landings"
    );
    assert!(directory.held(0, 0).is_some() && directory.held(0, 1).is_some(), "VRAM-only experts stay held");
}

/// On a 3-tier (`!tier_complete`) bank:
/// 1. `finish_demand_landings_batch` does not promote cold `!in_top` misses and
///    never evicts `Located::Lacked` (VRAM-only) resident experts during swap.
/// 2. `replenish_demand_free` never drops a `Located::Lacked` resident expert
///    with `back: None` (to NVMe) when write-back cannot pay / is unpriced.
#[test]
fn three_tier_demand_finish_and_replenish_never_drop_resident_experts_to_nvme() {
    use crate::experts::host_records::{Claim, Taking};

    let layers = 2usize;
    let experts = 16usize;
    let present = 6usize; // 4 held + 2 free
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let bank_maps = (0..layers)
        .map(|_| StampedMap::new(cpu_words(&[experts as u64, map_words], 0)))
        .collect::<Vec<_>>();
    let tier_maps = (0..layers)
        .map(|_| crate::experts::map::LandingMap::Tier(cpu_words(&[experts as u64], -1)))
        .collect::<Vec<_>>();
    let host = HostTier {
        records: vec![8], // 3-tier bank: 6 VRAM, 8 host, 32 total experts (!tier_complete)
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier { bank_maps, tier_maps, lent: (0, 0), stage_slots: vec![0] },
    };
    let mut directory = Directory::new(&[present], &[0, 0], &[experts, experts], host);
    assert!(!directory.tier_complete());
    directory.configure_decode_budget(1, 2, 1);

    let mut layer0_tier_recs = Vec::new();
    for layer in 0..layers {
        directory.price_copies(layer, Some(0.0));
        for expert in 0..4 {
            let Claim::Taken(grant) = directory.claim_landing(layer, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
                panic!("host tier has room");
            };
            directory.landed(layer, HostRecord(grant.record()), expert, Some(1));
            if layer == 0 {
                layer0_tier_recs.push(grant.record());
            }
        }
    }

    // Promote 4 hot experts into VRAM and surrender their Host-RAM copies so
    // all 4 are VRAM-only (`Located::Lacked`).
    let steps = steps();
    let step0 = steps.begin_group(1, None);
    for s in 0..4 {
        let step = if s == 0 { step0 } else { steps.begin_group(1, None) };
        log(&directory, step, 0, &[0, 1]);
        log(&directory, step, 1, &[0, 1]);
        if s > 0 {
            steps.end_played(step);
        }
    }
    assert!(directory.settle_changes());
    let wants = steps.unbegun(|at| directory.promotions(at, 4));
    for p in &wants.promotions {
        let tier = directory.hold_tier(p.layer, p.expert).expect("in host tier");
        directory.copy(p.group, p.record, tier, 0);
    }
    let completed: Vec<(usize, BankRecord)> = wants.promotions.iter().map(|p| (p.group, p.record)).collect();
    directory.promoted(completed, at(step0, 0), &steps);
    steps.end_played(step0);

    let surrendered = steps.unbegun(|at| directory.surrender_banked(at));
    assert!(!surrendered.is_empty());
    directory.retire_graces(past(step0), &steps);
    assert_eq!(directory.located(0, 0), Located::Lacked);
    assert_eq!(directory.located(0, 1), Located::Lacked);

    // Claim a free slot for a cold host-tier expert (0, 2) that is NOT `in_top`.
    let DemandSlotClaim::Allocated { staged, slot } =
        directory.claim_demand_landing(0, layer0_tier_recs[2], false, |_| true)
    else {
        panic!("expected free slot for cold miss");
    };
    assert_eq!(staged, Some((0, 2)));
    let step1 = steps.begin_group(1, None);
    let promoted = directory.finish_demand_landings_batch(&[(0, slot, staged, true)], at(step1, 0), &steps);
    assert!(!promoted, "cold !in_top miss must NOT promote on !tier_complete 3-tier bank");
    assert!(directory.held(0, 2).is_none());
    assert!(matches!(directory.located(0, 2), Located::Kept(_)), "Host-RAM copy of (0, 2) stays intact");
    assert!(directory.held(0, 0).is_some() && directory.held(0, 1).is_some(), "VRAM-only experts untouched");

    // `replenish_demand_free` with write-backs unpriced (`back: None`): MUST NOT
    // evict `Located::Lacked` VRAM-only experts straight to NVMe!
    let released = steps.unbegun(|at| directory.replenish_demand_free(0, 5, at, |_| true));
    assert_eq!(
        released, 0,
        "replenish_demand_free must not drop Located::Lacked experts with back: None to NVMe"
    );
    assert!(directory.held(0, 0).is_some() && directory.held(0, 1).is_some());
    steps.end_played(step1);
}

/// `replenish_demand_free` counts in-flight non-borrowed `DemandLanding` slots
/// as part of the transient supply so claiming `Free -> DemandLanding` for a
/// layer never over-evicts hot `Record::Held` experts into `Record::Releasing`.
#[test]
fn replenish_demand_free_counts_in_flight_demand_landings_without_over_evicting() {
    use crate::experts::host_records::{Claim, Taking};

    let layers = 2usize;
    let experts = 32usize;
    let present = 32usize;
    let k = 6usize;
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let bank_maps = (0..layers)
        .map(|_| StampedMap::new(cpu_words(&[experts as u64, map_words], 0)))
        .collect::<Vec<_>>();
    let tier_maps = (0..layers)
        .map(|_| crate::experts::map::LandingMap::Tier(cpu_words(&[experts as u64], -1)))
        .collect::<Vec<_>>();
    let host = HostTier {
        records: vec![(layers * experts) as u32],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier { bank_maps, tier_maps, lent: (0, 0), stage_slots: vec![0] },
    };
    let mut directory = Directory::new(&[present], &[0, 0], &[experts, experts], host);
    // M=1, K=6 -> demand_floor=6, promote_headroom=6, demand_replenish_target=12.
    directory.configure_decode_budget(1, k, 1);
    assert_eq!(directory.demand_free_floor(0), 6);
    assert_eq!(directory.demand_replenish_target(0), 12);

    let mut tier_recs = Vec::new();
    for layer in 0..layers {
        directory.price_copies(layer, Some(0.0));
        directory.price_write_backs(layer, None);
        for expert in 0..experts {
            let Claim::Taken(grant) = directory.claim_landing(layer, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
                panic!("host tier has room");
            };
            directory.landed(layer, HostRecord(grant.record()), expert, Some(1));
            if layer == 0 {
                tier_recs.push(grant.record());
            }
        }
    }

    let steps = steps();
    let step1 = steps.begin_group(1, None);
    let hot: Vec<usize> = (0..20).collect();
    log(&directory, step1, 0, &hot);
    assert!(directory.settle_changes());
    // Promotions leave `demand_replenish_target = 12` free slots untouched (20 promoted -> Held).
    let wants = steps.unbegun(|at| directory.promotions(at, present));
    assert_eq!(wants.promotions.len(), 20);
    for p in &wants.promotions {
        let tier = directory.hold_tier(p.layer, p.expert).expect("promoted expert is in host tier");
        directory.copy(p.group, p.record, tier, 0);
    }
    let completed: Vec<(usize, BankRecord)> = wants.promotions.iter().map(|p| (p.group, p.record)).collect();
    directory.promoted(completed, at(step1, 0), &steps);
    assert_eq!(directory.bank_free_count(0), 12);

    // Claim 10 of the 12 Free slots as in-flight DemandLanding (2 Free + 10 DemandLanding = 12).
    for &tr in tier_recs.iter().skip(20).take(10) {
        let DemandSlotClaim::Allocated { .. } = directory.claim_demand_landing(0, tr, false, |_| true) else {
            panic!("expected Allocated from Free pool");
        };
    }
    assert_eq!(directory.bank_free_count(0), 2);

    // Replenishing to `want_free = 10` (clamped up to `demand_replenish_target = 12`)
    // MUST evict 0 Held experts because 2 Free + 10 DemandLanding == 12!
    let released_zero = steps.unbegun(|at| directory.replenish_demand_free(0, 10, at, |_| true));
    assert_eq!(released_zero, 0, "in-flight DemandLanding slots must count toward supply (zero over-eviction)");

    // Replenishing to `want_free = 15` MUST evict ONLY the 3 missing slots (15 - 12 = 3), not 13!
    let released_three = steps.unbegun(|at| directory.replenish_demand_free(0, 15, at, |_| true));
    assert_eq!(released_three, 3, "only the 3-slot shortfall above Free+DemandLanding should be evicted");
}

/// Live `set_decode_rows` + empirical `observe_demand_misses` sizes `demand_floor`
/// to the current step's single-layer miss count so `M=4` verify gets enough
/// landing slots for `0` overflow while never locking to `M_max = 16` or
/// uniform-random `p_miss` over-reservation.
#[test]
fn decode_transient_budget_preserves_permanent_held_on_64_slot_bank() {
    // 1-layer 64-slot GLM-5.3 bank at M=4 with empirical p_miss=0.35:
    // demand_floor = ceil(32 * 0.35) = 12 -> 52/64 permanent Held (81.3%) and 0 overflow for 12 misses!
    let glm_64 = DecodeTransientBudget::compute(4, 8, 288, 0.35, 64, 64, 1, 1, 1);
    assert_eq!(glm_64.demand_floor, 12);
    assert!(64 - glm_64.demand_floor >= 52, "GLM-5.3 64-slot bank must keep >= 52 Held slots");

    // 1-layer 66-slot M5 Pro DS4 bank at M=4 with empirical p_miss=0.35:
    // demand_floor = 12 -> 54/66 permanent Held (81.8%)!
    let ds4_66 = DecodeTransientBudget::compute(4, 8, 384, 0.35, 66, 66, 1, 1, 1);
    assert_eq!(ds4_66.demand_floor, 12);
    assert!(66 - ds4_66.demand_floor >= 54, "DS4 66-slot bank must keep >= 54 Held slots");

    // 3-group GLM-5.3 RTX 5090 (2,701 VRAM slots, K=8):
    // 1. At M=1 with empirical p_miss ≈ 0.35: demand_floor = 8/group (24 total).
    // 2. At M=4 with empirical p_miss ≈ 0.45 (14 misses / 32 picks): demand_floor = 14-15/group
    //    (42-45 total, enough for 0 overflow on M=4 verify bursts, NOT 262!).
    // 3. After a temporary M=16 spike, returning to M=4 immediately drops reserve back to ~45
    //    instead of staying permanently locked to M=16!
    let glm_groups = [14usize, 14, 14];
    let glm_present = [901usize, 900, 900];
    let mut glm_layer_groups = Vec::new();
    let mut glm_layer_experts = Vec::new();
    for (g, &n) in glm_groups.iter().enumerate() {
        for _ in 0..n {
            glm_layer_groups.push(g);
            glm_layer_experts.push(288usize);
        }
    }
    let mut glm_dir = Directory::new(&glm_present, &glm_layer_groups, &glm_layer_experts, HostTier::replay(vec![4032; 3]));
    glm_dir.configure_decode_budget(1, 8, 1);
    assert_eq!((0..3).map(|g| glm_dir.reserved(g)).sum::<usize>(), 24, "M=1 reserve is 3*K=24");

    // Observe empirical ~43.75% miss rate (14 misses out of 32 picks at M=4) across all 3 groups.
    glm_dir.set_decode_rows(4);
    for g in 0..3 {
        glm_dir.observe_demand_misses(g, 14, 4);
    }
    let glm_reserve_m4: usize = (0..3).map(|g| glm_dir.reserved(g)).sum();
    assert_eq!(glm_reserve_m4, 42, "M=4 empirical 14/32 misses -> 14/group (42 total, 0 overflow, not 262!)");
    assert!(glm_dir.demand_free_floor(0) >= 14, "M=4 floor must cover 14 simultaneous single-layer misses");

    // Occasional M=16 spike raises floor transiently, then M=4 immediately recovers it!
    glm_dir.set_decode_rows(16);
    let glm_reserve_m16: usize = (0..3).map(|g| glm_dir.reserved(g)).sum();
    assert!(glm_reserve_m16 > glm_reserve_m4);
    glm_dir.set_decode_rows(4);
    let glm_reserve_recovered: usize = (0..3).map(|g| glm_dir.reserved(g)).sum();
    assert_eq!(glm_reserve_recovered, glm_reserve_m4, "M=4 after M=16 spike must immediately recover to 42 (never lock at M=16!)");

    // 8-group DS4 RTX 5090 (1,836 VRAM slots, K=8, M=4, empirical 13/32 misses):
    // total reserve == 8 * 13 = 104 (5.7% of 1,836, NOT 284!).
    let ds4_groups = [20usize, 4, 4, 4, 2, 2, 2, 2];
    let ds4_present = [918usize, 184, 184, 184, 92, 92, 92, 90];
    let mut ds4_layer_groups = Vec::new();
    let mut ds4_layer_experts = Vec::new();
    for (g, &n) in ds4_groups.iter().enumerate() {
        for _ in 0..n {
            ds4_layer_groups.push(g);
            ds4_layer_experts.push(384usize);
        }
    }
    let mut ds4_dir = Directory::new(&ds4_present, &ds4_layer_groups, &ds4_layer_experts, HostTier::replay(vec![200; 8]));
    ds4_dir.configure_decode_budget(1, 8, 1);
    ds4_dir.set_decode_rows(4);
    for g in 0..8 {
        ds4_dir.observe_demand_misses(g, 13, 4);
    }
    let ds4_reserve: usize = (0..8).map(|g| ds4_dir.reserved(g)).sum();
    assert_eq!(ds4_reserve, 104, "8-group DS4 at M=4 empirical 13/32 misses -> 104 total (not 284!)");
}

/// `finish_demand_landings_batch` promotes a high-rank demand miss by swapping
/// out a lower-rank `Record::Held` victim even when `free_now == free_floor`
/// (preserving `free_now >= free_floor` and `Held` count!) and when
/// `free_now < free_floor` after a wide verify burst.
#[test]
fn finish_demand_landings_batch_swaps_hot_miss_with_cold_held_at_and_below_free_floor() {
    use crate::experts::host_records::{Claim, Taking};

    let layers = 2usize;
    let experts = 64usize;
    let present = 24usize;
    let k = 4usize;
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let bank_maps = (0..layers)
        .map(|_| crate::experts::map::StampedMap::new(cpu_words(&[experts as u64, map_words], 0)))
        .collect::<Vec<_>>();
    let tier_maps = (0..layers)
        .map(|_| crate::experts::map::LandingMap::Tier(cpu_words(&[experts as u64], -1)))
        .collect::<Vec<_>>();
    let host = HostTier {
        records: vec![(layers * experts) as u32],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier { bank_maps, tier_maps, lent: (0, 0), stage_slots: vec![0] },
    };
    let mut directory = Directory::new(&[present], &[0, 0], &[experts, experts], host);

    let mut tier_recs = Vec::new();
    for layer in 0..layers {
        directory.price_copies(layer, Some(0.0));
        directory.price_write_backs(layer, None);
        for expert in 0..experts {
            let Claim::Taken(grant) = directory.claim_landing(layer, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
                panic!("host tier has room");
            };
            directory.landed(layer, HostRecord(grant.record()), expert, Some(1));
            if layer == 0 {
                tier_recs.push(grant.record());
            }
        }
    }

    // Fill 19 Held slots (experts 0..19 on layer 0) so 5 Free slots remain (4 = free_floor + 1 to claim).
    let steps = steps();
    let step1 = steps.begin_group(1, None);
    let cold: Vec<usize> = (0..19).collect();
    log(&directory, step1, 0, &cold);
    assert!(directory.settle_changes());
    let wants = steps.unbegun(|at| directory.promotions(at, present));
    assert_eq!(wants.promotions.len(), 19);
    for p in &wants.promotions {
        let tier = directory.hold_tier(p.layer, p.expert).expect("promoted expert is in host tier");
        directory.copy(p.group, p.record, tier, 0);
    }
    let completed: Vec<(usize, BankRecord)> = wants.promotions.iter().map(|p| (p.group, p.record)).collect();
    directory.promoted(completed, at(step1, 0), &steps);
    assert_eq!(directory.held_count(0), 19);
    assert_eq!(directory.bank_free_count(0), 5);
    // Enable decode demand free-floor (`k = 4`) now that 19 Held + 5 Free slots are seeded.
    directory.configure_decode_budget(1, k, 1);
    assert_eq!(directory.demand_free_floor(0), k);

    // Claim 1 Free slot for hot expert 30 -> now `free_now == 4 == free_floor`!
    let DemandSlotClaim::Allocated { staged, slot } = directory.claim_demand_landing(0, tier_recs[30], false, |_| true) else {
        panic!("expected Allocated from Free pool");
    };
    assert_eq!(staged, Some((0, 30)));
    assert_eq!(directory.bank_free_count(0), k, "free_now == free_floor");

    // Make expert 30 strictly hotter than the cold Held experts (logged across multiple steps).
    for _ in 0..4 {
        let s = steps.begin_group(1, None);
        log(&directory, s, 0, &[30]);
    }
    assert!(directory.settle_changes());

    // Finish at `free_now == free_floor`: MUST promote expert 30 by evicting a cold Held victim,
    // preserving `free_now == free_floor` and `held_count == 19`!
    let promoted = directory.finish_demand_landings_batch(&[(0, slot, staged, true)], at(step1, 0), &steps);
    assert!(promoted, "hot demand miss must promote via victim swap at free_now == free_floor");
    assert!(directory.held(0, 30).is_some(), "expert 30 must now be permanently Held");
    assert_eq!(directory.held_count(0), 19, "Held count must stay 19 (+1 promoted, -1 evicted)");
    assert!(
        directory.bank_free_count(0) >= directory.demand_free_floor(0),
        "free_now ({}) must stay >= free_floor ({})",
        directory.bank_free_count(0),
        directory.demand_free_floor(0)
    );

    // Now drain Free BELOW free_floor (`free_now == 2 < 4`) as in a wide verify burst!
    let DemandSlotClaim::Allocated { staged: staged_hot2, slot: slot_hot2 } =
        directory.claim_demand_landing(0, tier_recs[31], false, |_| true)
    else {
        panic!("expected Allocated");
    };
    let DemandSlotClaim::Allocated { staged: staged_cold2, slot: slot_cold2 } =
        directory.claim_demand_landing(0, tier_recs[50], false, |_| true)
    else {
        panic!("expected Allocated");
    };
    assert_eq!(directory.bank_free_count(0), 2, "free_now < free_floor");

    // Keep the 19 held experts strictly warmer than unlogged expert 50,
    // and make expert 31 hotter than all of them (logged 6x).
    let refresh = steps.begin_group(1, None);
    log(&directory, refresh, 0, &cold);
    for _ in 0..6 {
        let s = steps.begin_group(1, None);
        log(&directory, s, 0, &[31]);
    }
    assert!(directory.settle_changes());

    let promoted2 = directory.finish_demand_landings_batch(
        &[(0, slot_hot2, staged_hot2, true), (0, slot_cold2, staged_cold2, true)],
        at(step1, 0),
        &steps,
    );
    assert!(promoted2, "hot miss 31 must still swap in even when free_now < free_floor");
    assert!(directory.held(0, 31).is_some(), "expert 31 must be permanently Held");
    assert!(directory.held(0, 50).is_none(), "unranked cold miss 50 must NOT evict a hotter Held expert");
    assert_eq!(directory.held_count(0), 19, "Held count remains 19");
    assert_eq!(directory.bank_free_count(0), 3, "unpromoted cold miss 50 returned its slot to Free (2 -> 3)");
}

/// A shared physical stack visited at `visit 0` and `visit 2` (`last_visit = 2`)
/// must:
/// 1. Map a demand landing at `visit 0` of step `q` from step `q`
///    (`park.unresolved(last_visit = 2) == q`) so `visit 2` of the same step
///    resolves it as a hit without a redundant miss.
/// 2. Keep a graced record unfreed while step `q` is between `visit 0` and
///    `visit 2` (`Point::whole(q, 1, q)`), and free it only once the device
///    passes `last_visit = 2` (`Point::whole(q, 3, q)`).
#[test]
fn shared_physical_stack_preserves_unmap_safety_until_last_visit() {
    let steps = std::sync::Arc::new(self::steps());
    let map = |experts: u64| {
        let m = StampedMap::new(cpu_words(&[experts, magnitude_kernels::constants::MAP_WORDS], 0));
        let v = m.view().clone();
        (m, v)
    };
    let (map0, view0) = map(8);
    let (map1, _view1) = map(8);
    let host = HostTier {
        records: vec![2],
        lent: vec![0],
        reserve: 2,
        placement: Placement::Bank {
            landing_maps: vec![
                LandingMap::Bank { map: std::sync::Arc::new(map0), steps: steps.clone() },
                LandingMap::Bank { map: std::sync::Arc::new(map1), steps: steps.clone() },
            ],
        },
    };
    // 2 physical stacks (0, 1) unrolled to 4 visits: visits (0, 2) -> stack 0, visits (1, 3) -> stack 1.
    let mut directory = Directory::new_with_stacks(&[0], &[0, 0], &[8, 8], &[2, 3], &[1, 1], host);
    steps.begin_through(1);

    // Demand miss at visit 0 of step 1 (4 visits total).
    let park = steps.park(Point::whole(1, 0, 1), 4);
    let Claim::Taken(grant) = directory.claim_demand((0, 5), Demand::Lacked, &[], &park) else {
        panic!("ready record expected for demand claim");
    };
    let rec = grant.record();
    grant.map();
    directory.landed(0, HostRecord(rec), 5, Some(1));

    // Same step 1 at visit 2 must resolve expert 5 as a hit immediately!
    assert_eq!(view0.resolve(5, 1), Some(rec as usize), "visit 2 of step 1 hits demand landing from visit 0");

    // Unmap expert 5 for step 2+ (grace holds [1, 2)).
    let victims = directory.refill_victims(false);
    steps.unbegun(|at| directory.refill(victims, at));
    assert_eq!(directory.host(0).graced(), 1);

    // Step 1 at visit 1 (past visit 0, BEFORE visit 2): record MUST NOT be freed yet!
    directory.retire(Point::whole(1, 1, 1), |_, _| 0, |_| Ending::Running, &steps);
    assert_eq!(directory.host(0).graced(), 1, "record stays graced while step 1 has not passed last_visit = 2");

    // Step 1 at visit 3 (past last_visit = 2): record is now safe to free!
    directory.retire(Point::whole(1, 3, 1), |_, _| 0, |_| Ending::Running, &steps);
    assert_eq!(directory.host(0).graced(), 0, "record frees once step 1 passes last_visit = 2");
}

/// A shared physical stack visited at `visit 0` and `visit 2` (`last_visit = 2`)
/// on a `Placer::Tier` bank must:
/// 1. Publish a promotion (`Directory::promoted`) completing between `visit 0`
///    and `visit 2` (`progress = Point::whole(1, 1, 1)`) from step 1 so
///    `visit 2` of step 1 resolves it as a hit immediately.
/// 2. Refuse to recycle (`Directory::reclaim_served_demand_landings`) a held
///    record of stack 0 while step 1 has only passed `visit 0` and has NOT yet
///    passed `last_visit = 2` (delayed device completion), and recycle it once
///    progress passes `last_visit = 2`.
#[test]
fn shared_physical_stack_promotion_and_recycle_respect_last_visit() {
    let steps = std::sync::Arc::new(self::steps());
    let map = |experts: u64| {
        let m = StampedMap::new(cpu_words(&[experts, magnitude_kernels::constants::MAP_WORDS], 0));
        let v = m.view().clone();
        (m, v)
    };
    let (bank_map0, bank_view0) = map(8);
    let (bank_map1, _bank_view1) = map(8);
    let tier_words = || cpu_words(&[8, 1], -1);
    let host = HostTier {
        records: vec![4],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier {
            bank_maps: vec![bank_map0, bank_map1],
            tier_maps: vec![
                LandingMap::Tier(tier_words()),
                LandingMap::Tier(tier_words()),
            ],
            lent: (0, 0),
            stage_slots: vec![0],
        },
    };
    // 2 physical stacks (0, 1) unrolled to 4 visits: visits (0, 2) -> stack 0, visits (1, 3) -> stack 1.
    let mut directory = Directory::new_with_stacks(&[2], &[0, 0], &[8, 8], &[2, 3], &[1, 1], host);
    directory.price_copies(0, Some(0.0));
    directory.price_copies(1, Some(0.0));

    // Seed host tier with stack 0 expert 5 and log a decode read so Reuse wants it.
    let Claim::Taken(grant) = directory.claim_landing(0, 5, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
        panic!("tier claim expected");
    };
    let tier_rec = HostRecord(grant.record());
    grant.map();
    directory.landed(0, tier_rec, 5, Some(1));
    log(&directory, 1, 0, &[5]);

    steps.begin_through(1);
    let wants = steps.unbegun(|at| directory.promotions(at, 4));
    assert_eq!(wants.promotions.len(), 1, "expert 5 promoted into bank");
    let promo = wants.promotions[0];
    let held_tier = directory.hold_tier(0, 5).expect("tier holds expert 5");
    directory.copy(promo.group, promo.record, held_tier, 1);

    // 1. Promotion completes between visit 0 and visit 2 of step 1 (`progress = (1, visit=1)`):
    //    must map from step 1 (`unresolved.from(last_visit = 2) == 1`) so visit 2 hits!
    //    Note: host RAM still holds expert 5 (`Kept::Banked`), so it is eligible
    //    for zero-writeback reclaim once step 1 passes `last_visit = 2`.
    let done = directory.promoted(vec![(promo.group, promo.record)], Point::whole(1, 1, 1), &steps);
    assert_eq!(done, vec![((0, 5), promo.record)]);
    assert_eq!(
        bank_view0.resolve(5, 1),
        Some(promo.record.index()),
        "promotion between visit 0 and visit 2 must publish from step 1 for visit 2"
    );

    // 2. Delayed device completion: step 2 begins while step 1 has passed
    //    `visit 0` (`progress = (step 1, visit 1)`) and has NOT yet passed
    //    `last_visit = 2`. `unmap_until = park.unresolved(last_visit = 2) = 2`
    //    leaves step 1 in `[from=1, unmap_until=2)` still unresolved at visit 2,
    //    so reclaim MUST NOT recycle stack 0's held slot!
    steps.begin_through(2);
    let overflows = vec![(0usize, Some((1usize, 0usize)))];
    let reclaimed_early = steps.unbegun_with_served(Point::whole(2, 0, 2), 4, |at, park, is_served| {
        directory.reclaim_served_demand_landings(0, &overflows, at, park, Point::whole(1, 1, 1), is_served, |_| true)
    });
    assert!(
        reclaimed_early.is_empty(),
        "must NOT reclaim stack 0 while step 1 has not passed last_visit = 2"
    );

    // 3. Once device passes `last_visit = 2` (`progress = (step 1, visit 3)`),
    //    reclaim safely recycles stack 0's held slot!
    let reclaimed_after = steps.unbegun_with_served(Point::whole(2, 0, 2), 4, |at, park, is_served| {
        directory.reclaim_served_demand_landings(0, &overflows, at, park, Point::whole(1, 3, 1), is_served, |_| true)
    });
    assert_eq!(
        reclaimed_after.len(),
        1,
        "must reclaim stack 0 once step 1 passes last_visit = 2"
    );
}

/// `store_rank`, `claim_ahead`, and `claim_carryover` must compare candidates
/// and victims in identical footprint-normalized units (charging bytes once),
/// even when record footprints are large byte counts (e.g., 1 MiB / 2 MiB).
#[test]
fn claim_ahead_and_carryover_compare_footprint_normalized_units() {
    // 2 host records with reserve = 1: 1 Kept victim + 1 Ready reserve record
    // so `Taking::Ahead` evaluates victim admission (`outranking` / `claim_carryover`).
    let host = HostTier::replay(vec![2]).reserving(1);
    let footprints = [1_000_000u64, 2_000_000u64];
    let mut directory = Directory::new_with_stacks(&[1], &[0, 0], &[8, 8], &[0, 1], &footprints, host);
    // Warm up history so both stacks have finite positive expected uses.
    for step in 1..=20 {
        log(&directory, step, 0, &[0]);
        log(&directory, step, 1, &[0]);
        directory.time().lock().moved(0, Tier::COMPUTE);
        directory.time().lock().moved(0, Tier::HOST);
    }
    // Land cold expert (stack 1, expert 7: never read, 2 MB) into the host tier
    // (1 Kept, 1 Ready == reserve).
    let Claim::Taken(g) = directory.claim_landing(1, 7, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
        panic!("initial claim expected");
    };
    directory.landed(1, HostRecord(g.record()), 7, Some(1));

    // Hot candidate (stack 0, expert 0, 1 MB) MUST displace cold victim (stack 1, expert 7, 2 MB)
    // via `claim_ahead` with `keep = 1` (equal to `ready = 1`, forcing victim admission).
    // Previously broken because `store_rank` was unnormalized while `rank_of` divided by 1_000_000.
    let Claim::Taken(g2) = directory.claim_ahead(0, 0, 1, Taking::Ahead, 21, 0.9, 0.0) else {
        panic!("hot 1 MB candidate must displace cold 2 MB victim in claim_ahead");
    };
    directory.unlanded(0, HostRecord(g2.record()));

    // Now replace the kept slot with hot (stack 0, expert 0, 1 MB): a never-read 2 MB
    // carryover candidate with tiny net_benefit = 0.01 MUST NOT displace it
    // (`keep = 1 == ready`, forcing victim admission; previously broken because
    // raw net_benefit wasn't divided by footprint).
    let mut directory2 = Directory::new_with_stacks(&[1], &[0, 0], &[8, 8], &[0, 1], &footprints, HostTier::replay(vec![2]).reserving(1));
    for step in 1..=20 {
        log(&directory2, step, 0, &[0]);
        log(&directory2, step, 1, &[0]);
        directory2.time().lock().moved(0, Tier::COMPUTE);
        directory2.time().lock().moved(0, Tier::HOST);
    }
    let Claim::Taken(g3) = directory2.claim_landing(0, 0, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
        panic!("initial claim expected");
    };
    directory2.landed(0, HostRecord(g3.record()), 0, Some(20));
    assert!(
        matches!(directory2.claim_carryover(1, 6, 1, Taking::Ahead, 0.01, false), Claim::Short),
        "cold 2 MB carryover with tiny net_benefit must not displace hot 1 MB resident"
    );
}

#[test]
fn claim_carryover_never_evicts_hotter_resident_for_colder_candidate() {
    use crate::experts::host_records::{Claim, Taking};

    let experts = 8usize;
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let steps = std::sync::Arc::new(steps());
    let host = HostTier {
        records: vec![4],
        lent: vec![0],
        reserve: 2,
        placement: Placement::Bank {
            landing_maps: vec![LandingMap::Bank {
                map: std::sync::Arc::new(StampedMap::new(cpu_words(&[experts as u64, map_words], 0))),
                steps: std::sync::Arc::clone(&steps),
            }],
        },
    };
    let mut directory = Directory::new(&[0], &[0], &[experts], host);
    // Land experts 0 and 1 into the 2 non-reserve slots.
    for expert in 0..2 {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
            panic!("expected free slot for resident expert {expert}");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(1));
    }
    // Feed decode history making residents 0 and 1 strictly hotter than cold candidate 2.
    for _ in 0..8 {
        let s = steps.begin_group(1, None);
        log(&directory, s, 0, &[0, 1]);
    }
    assert!(directory.settle_changes());

    // Cold candidate 2 has lower expected uses than hotter residents 0 & 1:
    // with `keep = 2` (matching the 2 reserve slots so zero free slots remain),
    // strict_rank rejects it, and small net_benefit (0.15) that does not
    // overcome the victim's expected-uses margin also rejects it.
    assert!(
        matches!(directory.claim_carryover(0, 2, 2, Taking::Ahead, 0.15, true), Claim::Short),
        "strict_rank carryover must never evict a strictly hotter resident"
    );
    assert!(
        matches!(directory.claim_carryover(0, 2, 2, Taking::Ahead, 0.15, false), Claim::Short),
        "small net_benefit must not evict a much hotter resident"
    );

    // When `!strict_rank` (free drive slack hides the read) and the avoided
    // exposed stall (`net_benefit`) exceeds the victim's turnover margin,
    // `claim_carryover` admits on a full UMA bank (while `strict_rank = true`
    // still rejects it to protect drive-bound hosts).
    let tier = directory.host_tier();
    let victim_uses = directory
        .reuse_now()
        .expected(0, 0, tier, crate::experts::horizon::Reader::Move, None)
        .uses();
    let stall_benefit = victim_uses + 0.5;
    assert!(
        matches!(directory.claim_carryover(0, 2, 2, Taking::Ahead, stall_benefit, true), Claim::Short),
        "drive-bound strict_rank must still reject colder carryover"
    );
    assert!(
        matches!(directory.claim_carryover(0, 2, 2, Taking::Ahead, stall_benefit, false), Claim::Taken(_)),
        "non-drive-bound free-slack carryover must credit avoided exposed stall on full UMA bank"
    );
}

#[test]
fn claim_ahead_stall_credit_admits_on_full_bank_while_preserving_token_protection() {
    use crate::experts::host_records::{Claim, Taking};
    use crate::experts::horizon::{Advance, Reader, Tokens};

    let experts = 8usize;
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let steps = std::sync::Arc::new(steps());
    let host = HostTier {
        records: vec![3],
        lent: vec![0],
        reserve: 1,
        placement: Placement::Bank {
            landing_maps: vec![LandingMap::Bank {
                map: std::sync::Arc::new(StampedMap::new(cpu_words(&[experts as u64, map_words], 0))),
                steps: std::sync::Arc::clone(&steps),
            }],
        },
    };
    let mut directory = Directory::new(&[0], &[0], &[experts], host);
    for expert in 0..2 {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
            panic!("expected free slot for resident expert {expert}");
        };
        directory.landed(0, HostRecord(grant.record()), expert, None);
    }
    // Resident 0 is warmer (2 hits) and token-protected; resident 1 is colder (1 hit).
    let s1 = steps.begin_group(1, None);
    log(&directory, s1, 0, &[0, 1]);
    let s2 = steps.begin_group(1, None);
    log(&directory, s2, 0, &[0]);
    let next_step = s2 + 1;
    directory.time().lock().register(next_step, Advance::Known(Tokens::new(1.0)));

    // Protect resident 0 as needed in the upcoming burst (`token_reads`, p = 0.90).
    directory.set_token_reads([((0, 0), next_step, 0.90)]);

    let tier = directory.host_tier();
    let victim_uses = directory.reuse_now().expected(0, 1, tier, Reader::Move, None).uses();
    let cand_uses = directory.reuse_now().expected_with(0, 2, tier, next_step, 0.5).uses();
    let credit = (victim_uses - cand_uses).max(0.0) + 0.25;

    // With stall_credit = 0.0 (drive-bound), candidate 2 (p = 0.5) cannot beat
    // warmer residents 0 (protected) or 1 (keep = 1 matches the 1 ready reserve).
    assert!(matches!(directory.claim_ahead(0, 2, 1, Taking::Ahead, next_step, 0.5, 0.0), Claim::Short));

    // With stall_credit > 0.0 (free drive slack), candidate 2 is admitted and
    // leaves token-protected resident 0 intact.
    let Claim::Taken(_) = directory.claim_ahead(0, 2, 1, Taking::Ahead, next_step, 0.5, credit) else {
        panic!("free-slack claim_ahead with stall_credit must admit on full bank");
    };
    assert!(directory.in_tier(0, 0), "token_reads-protected resident 0 must remain resident");
}

#[test]
fn token_reads_protect_high_confidence_predicted_residents_from_demand_eviction() {
    use crate::experts::host_records::{Claim, Demand, Located, Taking};

    let experts = 8usize;
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let steps = std::sync::Arc::new(steps());
    let host = HostTier {
        records: vec![2],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Bank {
            landing_maps: vec![LandingMap::Bank {
                map: std::sync::Arc::new(StampedMap::new(cpu_words(&[experts as u64, map_words], 0))),
                steps: std::sync::Arc::clone(&steps),
            }],
        },
    };
    let mut directory = Directory::new(&[0], &[0], &[experts], host);
    // Land experts 0 and 1 with identical history so expert 0 would normally be
    // evicted first (tie-broken by record/index order).
    for expert in 0..2 {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
            panic!("expected free slot for resident expert {expert}");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(1));
    }
    let s1 = steps.begin_group(1, None);
    log(&directory, s1, 0, &[0, 1]);
    assert!(directory.settle_changes());

    // Protect expert 0 with a high-confidence `Source::Token` prediction for
    // upcoming step `s1 + 1`.
    let next_step = s1 + 1;
    directory.set_token_reads([((0, 0), next_step, 0.85)]);

    // Prepare and claim a demand landing for expert 2 at step `s1` (before
    // `next_step` runs): expert 1 must be evicted instead of protected expert 0.
    directory.prepare_demand(0, 1, s1);
    let park = steps.park(at(s1, 0), 1);
    let claim = directory.claim_demand((0, 2), Demand::Lacked, &[], &park);
    assert!(matches!(claim, Claim::Taken(_)), "demand claim must succeed by evicting unprotected victim");
    assert!(
        matches!(directory.located(0, 0), Located::Kept(_)),
        "high-confidence Source::Token predicted expert 0 must remain protected from eviction"
    );
    assert!(
        !matches!(directory.located(0, 1), Located::Kept(_)),
        "unprotected resident expert 1 must be chosen as the demand eviction victim"
    );
}

/// On a disjoint 3-tier (`!tier_complete`) bank with `Host` 100% full (`0` ready
/// slots) and `WriteBack::pays == true`, `finish_demand_landings_batch_with_reuse`
/// swaps a hot `E_miss` (`DemandLanding -> Held`) with a cold VRAM-only victim
/// `E_victim` (`Located::Lacked`) by immediately repurposing `E_miss`'s vacated
/// Host slot `A` into `Host::WritingBack { staged: E_victim }` — maintaining
/// `VRAM ∩ Host = ∅` without dropping `E_victim` to NVMe.
#[test]
fn finish_demand_landings_batch_swaps_disjoint_vram_only_victim_into_vacated_host_slot() {
    use crate::experts::host_records::{Claim, Taking};

    let layers = 2usize;
    let experts = 16usize;
    let present = 8usize; // 4 held + 4 free (demand_floor = 2, promote_headroom = 2, replenish_target = 4)
    let host_slots = 4u32; // strictly non-covering: 4 host slots < 32 total experts
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let bank_maps = (0..layers)
        .map(|_| StampedMap::new(cpu_words(&[experts as u64, map_words], 0)))
        .collect::<Vec<_>>();
    let tier_maps = (0..layers)
        .map(|_| LandingMap::Tier(cpu_words(&[experts as u64], -1)))
        .collect::<Vec<_>>();
    let host = HostTier {
        records: vec![host_slots],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier { bank_maps, tier_maps, lent: (0, 0), stage_slots: vec![0] },
    };
    let mut directory = Directory::new(&[present], &[0, 0], &[experts, experts], host);
    assert!(!directory.tier_complete());
    directory.configure_decode_budget(1, 2, 1);
    assert_eq!(directory.demand_free_floor(0), 2);

    let wb_price = Some(WriteBack {
        terms: magnitude_cost_model::Terms { compute: 0.0, drive: 4.0, link: 0.0, cpu: 0.0, memory: 0.0 },
        drive: 1.0,
        link: 0.0,
    });
    for layer in 0..layers {
        directory.price_copies(layer, Some(0.0));
        directory.price_write_backs(layer, wb_price);
    }

    // 1. Land experts 0..4 (layer 0) into Host and promote all 4 into VRAM.
    let steps = steps();
    let step0 = steps.begin_group(1, None);
    for expert in 0..4 {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
            panic!("host tier has room");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(step0));
    }
    log(&directory, step0, 0, &[0, 1, 2, 3]);
    assert!(directory.settle_changes());
    let wants = steps.unbegun(|at| directory.promotions(at, 4));
    assert_eq!(wants.promotions.len(), 4);
    for p in &wants.promotions {
        let tier = directory.hold_tier(p.layer, p.expert).expect("in host tier");
        directory.copy(p.group, p.record, tier, 0);
    }
    let completed: Vec<(usize, BankRecord)> = wants.promotions.iter().map(|p| (p.group, p.record)).collect();
    directory.promoted(completed, at(step0, 0), &steps);
    steps.end_played(step0);

    // 2. Surrender Host copies of 0..4 so VRAM = {0,1,2,3} (all Located::Lacked).
    let surrendered = steps.unbegun(|at| directory.surrender_banked(at));
    assert_eq!(surrendered.len(), 4);
    directory.retire_graces(past(step0), &steps);
    for expert in 0..4 {
        assert_eq!(directory.located(0, expert), Located::Lacked);
    }
    assert_eq!(directory.host(0).ready(), 4);

    // 3. Fill ALL 4 Host slots with distinct experts 4..8 (`0` ready slots left!).
    let mut slot_of_miss = None;
    for expert in 4..8 {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
            panic!("host tier has room for unique expert {expert}");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(step0));
        if expert == 4 {
            slot_of_miss = Some(grant.record());
        }
    }
    let slot_a = slot_of_miss.unwrap();
    assert_eq!(directory.host(0).ready(), 0, "Host tier is 100% full (0 ready slots)");
    // Ensure host tier horizon tau exists (`moved(0, HOST)` happened in `landed`).
    for s in 1..=6 {
        let step = steps.begin_group(1, None);
        // Make expert 4 strictly hotter than 0..4, while keeping 0..4 warm enough that WriteBack::pays.
        log(&directory, step, 0, &[1, 2, 3, 4]);
        steps.end_played(step);
    }
    assert!(directory.settle_changes());

    // 4. Claim a Free VRAM slot `B` for `E_miss = (0, 4)` (held_total + replenish_target == 4 + 4 == cap).
    let DemandSlotClaim::Allocated { staged, slot: slot_b } =
        directory.claim_demand_landing(0, slot_a, false, |_| true)
    else {
        panic!("expected Free VRAM slot for E_miss");
    };
    assert_eq!(staged, Some((0, 4)));
    assert_eq!(directory.bank_free_count(0), 3);

    // 5. Finish demand landing for `(0, 4)` at step_cur past layer 0 (`at(step_cur, 1)`):
    //    MUST swap cold VRAM-only victim `(0, 0)` into `slot_a` via `Host::WritingBack`!
    let step_cur = steps.begin_group(1, None);
    let mut reclaimed_host_slots = Vec::new();
    let promoted = directory.finish_demand_landings_batch_with_reuse(
        &[(0, slot_b, staged, true)],
        at(step_cur, 1),
        &steps,
        true,
        &mut reclaimed_host_slots,
        |_, _| true,
    );
    assert!(promoted, "hot miss (0, 4) must promote via WriteBack swap of VRAM-only victim");
    assert_eq!(directory.held(0, 4), Some(slot_b), "E_miss must be Held in slot B");
    assert_eq!(directory.located(0, 4), Located::Lacked, "E_miss's Host slot A must be vacated (VRAM ∩ Host = ∅)");
    assert_eq!(reclaimed_host_slots, vec![(0, slot_a as usize)], "reclaimed Host slot A reported for generation bump");

    let backs = directory.backs_queued();
    assert_eq!(backs.len(), 1, "evicted VRAM-only victim must have WriteBack queued");
    let (wb_group, wb_bank_rec, wb_victim, wb_host_rec) = backs[0];
    assert_eq!(wb_group, 0);
    assert_eq!(wb_victim, (0, 0), "coldest VRAM-only expert (0, 0) must be chosen as victim");
    assert_eq!(wb_host_rec, HostRecord(slot_a), "WriteBack destination must be E_miss's vacated Host slot A");
    assert!(directory.held(0, 0).is_none(), "victim is no longer Held in VRAM");
    assert_eq!(directory.located(0, 0), Located::Lacked, "victim stays Lacked until D2H completes");

    // 6. Complete D2H WriteBack and drain: victim lands in Host slot A, VRAM slot C returns to Free!
    directory.back_copy(wb_group, wb_bank_rec, 1);
    directory.written_back(wb_group, wb_bank_rec, true);
    steps.end_played(step_cur);
    assert!(directory.drain(past(step_cur), &steps), "slot C must drain to Free after WriteBack completes");
    assert_eq!(directory.located(0, 0), Located::Kept(slot_a), "victim must now reside in Host slot A");
    assert_eq!(directory.held_count(0), 4, "VRAM Held count preserved at 4");
    assert_eq!(directory.alone(0), 4, "Host holds 4 unique experts (0, 5, 6, 7)");
    // Verify strict disjointness: no expert is both Held in VRAM and Kept in Host.
    for expert in 0..experts {
        assert!(
            !(directory.held(0, expert).is_some() && matches!(directory.located(0, expert), Located::Kept(_))),
            "strict VRAM ∩ Host = ∅ violated for expert {expert}"
        );
    }
}

/// `0..lent` experts are marked `Kept::Banked` during decode (so Host RAM does
/// not duplicate them once surrendered) and are cleanly written back to Host on
/// `lease_bank_records` rather than dropped to NVMe.
#[test]
fn lent_slot_experts_marked_banked_during_decode_and_written_back_on_lease() {
    use crate::experts::host_records::{Claim, Taking};

    let experts = 8usize;
    let lent = 2usize;
    let present = 4usize; // 0..2 lent, 2..4 non-lent
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let host = HostTier {
        records: vec![4], // non-covering: 4 host slots < 8 experts
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier {
            bank_maps: vec![StampedMap::new(cpu_words(&[experts as u64, map_words], 0))],
            tier_maps: vec![LandingMap::Tier(cpu_words(&[experts as u64], -1))],
            lent: (0, lent),
            stage_slots: vec![0],
        },
    };
    let mut directory = Directory::new(&[present], &[0], &[experts], host);
    directory.price_copies(0, Some(0.0));
    directory.price_write_backs(
        0,
        Some(WriteBack {
            terms: magnitude_cost_model::Terms { compute: 0.0, drive: 4.0, link: 0.0, cpu: 0.0, memory: 0.0 },
            drive: 1.0,
            link: 0.0,
        }),
    );

    let steps = steps();
    let step0 = steps.begin_group(1, None);
    let Claim::Taken(grant) = directory.claim_landing(0, 0, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
        panic!("host tier has room");
    };
    let host_rec0 = grant.record();
    directory.landed(0, HostRecord(host_rec0), 0, Some(step0));

    // Demand-land expert 0 into lent VRAM slot (`0..lent`) and promote it.
    let DemandSlotClaim::Allocated { staged, slot } = directory.claim_demand_landing(0, host_rec0, false, |_| true) else {
        panic!("expected lent VRAM slot");
    };
    assert!(directory.bank_lends(0, slot), "slot must be inside 0..lent");
    log(&directory, step0, 0, &[0]);
    assert!(directory.finish_demand_landing(0, slot, staged, at(step0, 1), &steps, true));
    steps.end_played(step0);

    // Must be marked `Kept::Banked` (NOT `Kept::Alone`) and surrendered to free its Host slot!
    assert!(!directory.host(0).holds_alone((0, 0)), "0..lent expert must be Kept::Banked during decode");
    let surrendered = steps.unbegun(|at| directory.surrender_banked(at));
    assert_eq!(surrendered, vec![(0, host_rec0 as usize)], "0..lent expert's Host copy must be surrendered");
    directory.retire_graces(past(step0), &steps);
    assert_eq!(directory.located(0, 0), Located::Lacked, "Host slot reclaimed for unique experts");

    // Warm up expert 0 so WriteBack pays when prefill leases `0..lent`.
    for _ in 0..3 {
        let s = steps.begin_group(1, None);
        log(&directory, s, 0, &[0]);
        steps.end_played(s);
    }

    // Prefill entry: `lease_bank_records` MUST write back hot `0..lent` expert 0 via `exit_now`!
    let scratch = ScratchGroup::at(10, 1);
    directory.lease_bank_records(0, scratch, &[true; 2]);
    assert!(!directory.leased(0), "lease must wait while lent slot's WriteBack is in flight");
    let backs = directory.backs_queued();
    assert_eq!(backs.len(), 1, "lease_bank_records must queue WriteBack for hot 0..lent expert");
    let (wb_group, wb_rec, wb_staged, _wb_tier) = backs[0];
    assert_eq!(wb_staged, (0, 0));
    assert_eq!(wb_rec.index(), slot);

    // Complete WriteBack + drain, then re-call `lease_bank_records` -> slot transitions to `Record::Lent`!
    directory.back_copy(wb_group, wb_rec, 1);
    directory.written_back(wb_group, wb_rec, true);
    assert!(directory.drain(past(10), &steps));
    directory.lease_bank_records(0, scratch, &[true; 2]);
    assert!(directory.leased(0), "all 0..lent slots must now be leased");
    assert!(directory.lent(0, slot), "slot must be Record::Lent");
    assert!(matches!(directory.located(0, 0), Located::Kept(_)), "hot 0..lent expert preserved in Host RAM");
}

/// Targeted 3-record lifetime regression test (`A` = old Host slot of `E_miss`,
/// `B` = completed demand VRAM slot of `E_miss`, `C` = lent VRAM slot of `E_victim`):
/// 1. While a zero-copy reader is still active on `A` (`!is_served`) and Host has
///    no `Ready` slot, `finish_demand_landings_batch_with_reuse` MUST NOT overwrite
///    `A` or drop `E_victim` (`A`'s simulated bytes `0xAA` remain untouched).
/// 2. Once the zero-copy reader passes `A`'s layer, swap-on-promote succeeds:
///    `B` becomes `Held(E_miss)`, `C` becomes `Releasing { back: Some(A) }`, and
///    `A` becomes `Host::WritingBack { staged: E_victim }`.
/// 3. While D2H `C -> A` is delayed in flight, a prompt lease (`lease_bank_records`)
///    arriving on `0..lent` (covering `C`) MUST skip `C` — `C` stays `Releasing`
///    and `E_victim` stays unmapped in Host until delayed D2H completion copies
///    `0xCC` into `A` and calls `written_back` + `drain`.
#[test]
fn swap_on_promote_respects_active_zero_copy_reader_and_prompt_lease_on_releasing_victim() {
    use crate::experts::host_records::{Claim, Taking};

    let layers = 2usize;
    let experts = 16usize;
    let lent = 2usize; // slots 0..2 are lent (victim C will sit in 0..lent!)
    let present = 6usize; // 2 held (0..2) + 4 free (demand_floor = 2, replenish_target = 4)
    let host_slots = 2u32; // strictly full once 2 unique Host experts land
    let map_words = magnitude_kernels::constants::MAP_WORDS;
    let bank_maps = (0..layers)
        .map(|_| StampedMap::new(cpu_words(&[experts as u64, map_words], 0)))
        .collect::<Vec<_>>();
    let tier_maps = (0..layers)
        .map(|_| LandingMap::Tier(cpu_words(&[experts as u64], -1)))
        .collect::<Vec<_>>();
    let host = HostTier {
        records: vec![host_slots],
        lent: vec![0],
        reserve: 0,
        placement: Placement::Tier { bank_maps, tier_maps, lent: (0, lent), stage_slots: vec![0] },
    };
    let mut directory = Directory::new(&[present], &[0, 0], &[experts, experts], host);
    directory.configure_decode_budget(1, 2, 1);

    let wb_price = Some(WriteBack {
        terms: magnitude_cost_model::Terms { compute: 0.0, drive: 4.0, link: 0.0, cpu: 0.0, memory: 0.0 },
        drive: 1.0,
        link: 0.0,
    });
    for layer in 0..layers {
        directory.price_copies(layer, Some(0.0));
        directory.price_write_backs(layer, wb_price);
    }

    // Simulated byte storage for Host slots and VRAM slots to assert distinct patterns.
    let mut host_bytes = vec![0x00u8; host_slots as usize];
    let mut vram_bytes = vec![0x00u8; present];

    // Place cold victim E_victim = (0, 0) into lent VRAM slot C (`0..lent`) and (0, 1) into slot 1,
    // then surrender both Host copies so VRAM = {(0,0), (0,1)} are both Located::Lacked.
    let steps = steps();
    let step0 = steps.begin_group(1, None);
    let mut initial_claims = Vec::new();
    for expert in 0..2 {
        let Claim::Taken(grant) = directory.claim_landing(0, expert, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
            panic!("host room");
        };
        directory.landed(0, HostRecord(grant.record()), expert, Some(step0));
        let DemandSlotClaim::Allocated { staged, slot } =
            directory.claim_demand_landing(0, grant.record(), false, |_| true)
        else {
            panic!("lent VRAM room");
        };
        assert!(directory.bank_lends(0, slot));
        vram_bytes[slot] = if expert == 0 { 0xCC } else { 0xDD };
        initial_claims.push((0usize, slot, staged, true));
    }
    log(&directory, step0, 0, &[0, 1]);
    assert!(directory.finish_demand_landings_batch(&initial_claims, at(step0, 1), &steps));
    steps.end_played(step0);
    let slot_c = directory.held(0, 0).expect("victim C held in 0..lent");
    assert!(directory.bank_lends(0, slot_c));
    assert_eq!(vram_bytes[slot_c], 0xCC);

    steps.unbegun(|at| directory.surrender_banked(at));
    directory.retire_graces(past(step0), &steps);
    assert_eq!(directory.located(0, 0), Located::Lacked);
    assert_eq!(directory.located(0, 1), Located::Lacked);

    // Fill BOTH Host slots (`0` ready): slot A holds E_miss = (0, 4) with byte pattern `0xAA`.
    let Claim::Taken(grant_a) = directory.claim_landing(0, 4, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
        panic!("host room for E_miss");
    };
    let slot_a = grant_a.record();
    host_bytes[slot_a as usize] = 0xAA;
    directory.landed(0, HostRecord(slot_a), 4, Some(step0));
    let Claim::Taken(grant_other) = directory.claim_landing(0, 5, 0, Taking::Out, (|_, _| (), |_, _| true)) else {
        panic!("host room for (0, 5)");
    };
    host_bytes[grant_other.record() as usize] = 0x55;
    directory.landed(0, HostRecord(grant_other.record()), 5, Some(step0));
    assert_eq!(directory.host(0).ready(), 0);

    // Make E_miss = (0, 4) hot and (0, 1) hotter than E_victim = (0, 0).
    for _ in 0..5 {
        let s = steps.begin_group(1, None);
        log(&directory, s, 0, &[1, 4]);
        steps.end_played(s);
    }
    assert!(directory.settle_changes());

    // 1. Active zero-copy reader step is STILL at layer 0 (`progress = at(reader_step, 0)`):
    //    `claim_demand_landing` takes slot B, and `finish_demand_landings_batch_with_reuse`
    //    MUST NOT immediately reuse Host slot A while the zero-copy reader is outstanding!
    let reader_step = steps.begin_group(1, None);
    let DemandSlotClaim::Allocated { staged, slot: slot_b } =
        directory.claim_demand_landing(0, slot_a, false, |_| true)
    else {
        panic!("expected Free VRAM slot B");
    };
    vram_bytes[slot_b] = host_bytes[slot_a as usize]; // H2D copy `0xAA`
    let mut reclaimed = Vec::new();
    let promoted_while_active = directory.finish_demand_landings_batch_with_reuse(
        &[(0, slot_b, staged, true)],
        at(reader_step, 0), // reader_step still at layer 0 -> `!is_served`!
        &steps,
        true,
        &mut reclaimed,
        |_, _| true,
    );
    assert!(!promoted_while_active, "must NOT evict VRAM-only victim or overwrite A while zero-copy reader is active on A");
    assert!(reclaimed.is_empty(), "Host slot A must NOT be reclaimed while reader is active");
    assert_eq!(directory.located(0, 4), Located::Kept(slot_a), "E_miss stays mapped in A for active zero-copy reader");
    assert_eq!(host_bytes[slot_a as usize], 0xAA, "A's bytes remain intact for active zero-copy reader");
    assert_eq!(directory.held(0, 0), Some(slot_c), "VRAM-only victim C untouched");
    assert!(directory.backs_queued().is_empty());

    // 2. Reader finishes layer 0 (`progress = at(reader_step, 1)`): re-claim B and finish ->
    //    swap-on-promote succeeds cleanly!
    let DemandSlotClaim::Allocated { staged, slot: slot_b2 } =
        directory.claim_demand_landing(0, slot_a, false, |_| true)
    else {
        panic!("expected Free VRAM slot B2");
    };
    vram_bytes[slot_b2] = host_bytes[slot_a as usize];
    let promoted_after_reader = directory.finish_demand_landings_batch_with_reuse(
        &[(0, slot_b2, staged, true)],
        at(reader_step, 1), // reader_step is strictly past layer 0!
        &steps,
        true,
        &mut reclaimed,
        |_, _| true,
    );
    assert!(promoted_after_reader, "swap-on-promote must succeed once zero-copy reader passed layer 0");
    assert_eq!(reclaimed, vec![(0, slot_a as usize)]);
    assert_eq!(directory.held(0, 4), Some(slot_b2));
    assert_eq!(vram_bytes[slot_b2], 0xAA);
    let backs = directory.backs_queued();
    assert_eq!(backs.len(), 1);
    let (wb_group, wb_rec, wb_staged, wb_tier) = backs[0];
    assert_eq!((wb_group, wb_rec.index(), wb_staged, wb_tier), (0, slot_c, (0, 0), HostRecord(slot_a)));

    // 3. Prompt lease arrives on `0..lent` (covering C!) while D2H WriteBack is queued/delayed:
    //    `lease_bank_records` MUST skip Releasing slot C and MUST NOT overwrite C!
    let scratch = ScratchGroup::at(reader_step + 1, 1);
    directory.lease_bank_records(0, scratch, &[true; 2]);
    assert!(!directory.lent(0, slot_c), "prompt lease MUST NOT take slot C while WriteBack is queued/in-flight");
    assert_eq!(directory.located(0, 0), Located::Lacked, "E_victim must stay unmapped in Host until D2H completes");
    assert_eq!(vram_bytes[slot_c], 0xCC, "C's bytes remain intact until D2H completes");

    // 4. Delayed D2H WriteBack now executes (`C -> A`), completes, and drains:
    directory.back_copy(wb_group, wb_rec, 42);
    host_bytes[wb_tier.0 as usize] = vram_bytes[slot_c]; // D2H copies `0xCC` into A
    directory.written_back(wb_group, wb_rec, true);
    steps.end_played(reader_step);
    assert!(directory.drain(past(reader_step), &steps));

    // Now prompt lease claims C cleanly, while E_victim is intact in Host slot A!
    directory.lease_bank_records(0, scratch, &[true; 2]);
    assert!(directory.lent(0, slot_c), "slot C cleanly leased after delayed WriteBack completed and drained");
    assert_eq!(directory.located(0, 0), Located::Kept(slot_a), "E_victim published in Host slot A after D2H");
    assert_eq!(host_bytes[slot_a as usize], 0xCC, "Host slot A holds E_victim's exact byte pattern");
}
