use super::*;
use super::records::record_bytes;
use crate::error::{PlanError, UnstreamedExperts};
use magnitude_kernels::constants::VERIFY_ROWS;
use crate::planning::ModelLoadPlan;
use crate::planning::weights::{Holding, representation_bytes};
use magnitude_family_contracts::{ImportTransform, ModelDefinition, Operator};

impl ModelLoadPlan {
    /// The plan at its bank's least slots, of one chain place.
    pub(crate) fn with_expert_bank_at_floor(self, definition: &ModelDefinition, substrate: BankSubstrate) -> Result<Self, PlanError> {
        let floor = self.bank_floor(definition, 1, FlightDepth::PIPELINED, substrate)?;
        self.with_expert_bank(definition, floor, None, 1, FlightDepth::PIPELINED, substrate)
    }
}

/// A CUDA device's substrate beside `host`, without KV slabs.
pub(crate) fn cuda(host: HostTier) -> BankSubstrate {
    BankSubstrate { backend: seismic::BackendName::Cuda, allocations: seismic::NativeAllocations::EXACT, host }
}

/// The fixture a partial bank streams through on the device
/// (`ngram_steps_tests::a_partial_banks_shared_graphs_read_each_layers_own_experts`):
/// at 128 experts a layer its banks' weights rise with their slots, a
/// granule of the pool a record, on unified memory's one store and beside
/// a linked host tier, so a cap below the whole bank's admits a partial
/// one.
#[test]
fn a_fixture_of_more_experts_admits_a_partial_bank() {
    use crate::planning::fixtures::{qwen4exp_fixture_definition_of, streamed_experts_manifest};
    for (backend, host) in [(seismic::BackendName::Metal, HostTier::Unified(Parking::InKernels)), (seismic::BackendName::Cuda, HostTier::Linked)] {
        let definition = qwen4exp_fixture_definition_of(3, 128);
        let manifest = streamed_experts_manifest(&definition);
        let selection = crate::ComponentSelection { head: false, vision: false };
        let resident = crate::ModelLoadPlan::derive(&manifest, &definition, selection, crate::resident_layout(crate::ExecutionPath::Native, backend), crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS)
            .unwrap();
        let substrate = BankSubstrate { backend, allocations: seismic::NativeAllocations::EXACT, host };
        let floor = resident.bank_floor(&definition, 1, FlightDepth::PIPELINED, substrate).unwrap();
        let [floor, half, whole] = [floor, 64, 128].map(|slots| {
            resident.clone().with_expert_bank(&definition, slots, None, 1, FlightDepth::PIPELINED, substrate).unwrap().weight_bytes().unwrap()
        });
        assert!(floor < half && half < whole, "{backend:?}: {floor} {half} {whole}");
    }
}

/// A shrink crosses a charge's plateaus to the shallowest holding that
/// releases what is owed, never past the floor, and none where the
/// floor releases nothing.
#[test]
fn physical_shrink_targets_cross_plateaus_and_preserve_the_floor() {
    let charge = |slots: u64| Ok::<_, ()>(slots.max(2).div_ceil(4) * 4096);
    assert_eq!(releasing(12, 2, 0, charge).unwrap(), Some(8));
    assert_eq!(releasing(12, 2, 5000, charge).unwrap(), Some(4));
    assert_eq!(releasing(12, 2, u64::MAX, charge).unwrap(), Some(4));
    assert_eq!(releasing(4, 2, 1, charge).unwrap(), None);
    // The host record minimum keeps its allocation even at zero slots.
    assert_eq!(releasing(1, 0, 1, |slots| Ok::<_, ()>(slots.max(1))).unwrap(), None);
    assert_eq!(releasing(9, 2, 4, |slots| Ok::<_, ()>(slots)).unwrap(), Some(5));
    assert_eq!(releasing(12, 2, 1, |slots| if slots == 7 { Err("charge refused") } else { Ok(slots) }), Err("charge refused"));
}

use crate::{ArtifactComponent, ArtifactComponentKind};
use magnitude_artifacts::ArtifactIdentity;
use magnitude_family_contracts::WeightDescriptor;
use seismic::Layout;

fn tensor(source: &str, resident: &str) -> RecordTensor {
    RecordTensor {
        source: Element::named(source).unwrap(),
        resident: Element::stored(resident, Layout::Rows16).unwrap(),
        matrix: [512, 2048],
    }
}

/// A stand-in for a layer's resident plans, which only tuning reads.
fn resident() -> WeightPlan {
    WeightPlan {
        role: WeightRole {
            scope: WeightScope::Target,
            kind: WeightKind::ExpertDown,
        },
        component: ArtifactComponent {
            kind: ArtifactComponentKind::Target,
            identity: ArtifactIdentity([7; 32]),
        },
        source: Element::f16(),
        upload: Element::f16(),
        resident: Element::f16(),
        shape: vec![1],
        descriptor: WeightDescriptor::stored("experts", [1]),
        source_bytes: 2,
        resident_bytes: 2,
        scale: None,
        bound: 0,
        holding: Holding::Whole,
    }
}

/// A list of no items is refused, so the greatest of a list made exists:
/// a bank's widest layer and floor are its layers', never a default.
#[test]
fn a_non_empty_list_has_a_greatest_item() {
    assert_eq!(NonEmpty::<u64>::new(Vec::new()), None);
    let mut list = NonEmpty::one(3u64);
    list.push(7);
    list.push(5);
    assert_eq!((list.max_of(|&item| item), list.len()), (7, 3));
    // A group's widest member prices its records, not the bank's widest.
    let plan = bank_of(&[2, 1], 4, |layer| [64, 256, 128][layer], RecordForm::Gated);
    assert_eq!(plan.widest().get(), 256);
    assert_eq!((plan.widest_member(0), plan.widest_member(1)), (256, 128));
}

fn bank(groups: &[usize], slots: u64) -> ExpertBankPlan {
    bank_of(groups, slots, |_| 256, RecordForm::Gated)
}

/// A bank of groups of `groups` layers each, layer `l` of `experts(l)`
/// experts, records of `form`.
fn bank_of(groups: &[usize], slots: u64, experts: impl Fn(usize) -> u64, form: RecordForm) -> ExpertBankPlan {
    let mut layers = Vec::new();
    let groups: Vec<BankGroup> = groups
        .iter()
        .enumerate()
        .map(|(index, &members)| {
            let down = [("gguf_q4_k", "q4k"), ("gguf_q6_k", "q6k")][index % 2];
            let record = form
                .each(|kind| {
                    Ok::<_, ()>(match kind {
                        RoutedTensor::Down => tensor(down.0, down.1),
                        _ => tensor("gguf_q4_k", "q4k"),
                    })
                })
                .unwrap();
            let first = layers.len();
            for _ in 0..members {
                layers.push(BankLayer {
                    scope: WeightScope::Target,
                    group: index,
                    experts: experts(layers.len()),
                    resident: form.each(|_| Ok::<_, ()>(resident())).unwrap(),
                    hinted: Vec::new(),
                    shared_down: None,
                });
            }
            BankGroup {
                record,
                layers: NonEmpty::new((first..first + members).collect()).unwrap(),
            }
        })
        .collect();
    let layers = BankLayers::new(layers).unwrap();
    let structured = conveyor::Structured::new(8, HostTier::Linked, groups, layers, None, VERIFY_ROWS).unwrap();
    let allocations = seismic::NativeAllocations::EXACT;
    let blocks = size_pool_blocks(structured.granules(), allocations).unwrap();
    ExpertBankPlan {
        structured,
        backend: seismic::BackendName::Cuda,
        allocations,
        blocks,
        slots,
        ceiling: slots,
        resident: false,
        host_slots: 0,
        host_cap: None,
        places: 1,
        pass: crate::PassLaunches::FLOOR,
        flights: FlightDepth::PIPELINED,
        tables: Vec::new(),
        sites: BankSites::Unadmitted,
        paths: BankPaths::Priced { line: LINE, records: PATHS },
        recording: false,
    }
}

/// The path records the fixtures' banks are priced at (`PathRecords::of`,
/// as a settle prices them): 16 records in flight, and as many free beside
/// the device's reads.
const PATHS: magnitude_cost_model::PathRecords = magnitude_cost_model::PathRecords { ahead: 16, reserve: 16 };

/// The fixtures' stored read line.
const LINE: magnitude_cost_model::ReadLine = match magnitude_cost_model::ReadLine::new(100e-6, 1. / 2e9) {
    Ok(line) => line,
    Err(_) => panic!("a fixture's line"),
};

/// `plan` with its structure changed by `change`, and that structure's
/// lending.
fn restructured(
    plan: ExpertBankPlan,
    change: impl FnOnce(&mut HostTier, &mut Vec<BankGroup>),
) -> ExpertBankPlan {
    let decode_rows = plan.decode_rows();
    let (selected, mut host, mut groups, layers) = plan.structured.into_parts();
    change(&mut host, &mut groups);
    let structured = conveyor::Structured::new(selected, host, groups, layers, None, decode_rows).unwrap();
    let blocks = size_pool_blocks(structured.granules(), plan.allocations).unwrap();
    ExpertBankPlan { structured, blocks, ..plan }
}

/// `plan` with its scratch bounded to `window_slots` experts of a layer,
/// and its structure's lending derived again; or why it cannot be.
fn windowed(plan: &ExpertBankPlan, window_slots: u64) -> Result<ExpertBankPlan, String> {
    let decode_rows = plan.decode_rows();
    let (selected, host, groups, layers) = plan.structured.clone().into_parts();
    let structured = conveyor::Structured::new(selected, host, groups, layers, Some(window_slots), decode_rows)?;
    Ok(ExpertBankPlan { structured, ..plan.clone() })
}

/// `plan` beside `host`.
fn beside(plan: ExpertBankPlan, host: HostTier) -> ExpertBankPlan {
    restructured(plan, |tier, _| *tier = host)
}

/// The bank's minimum is its derivation (`least_records`), on groups of
/// 37 layers and 3 of 256 experts, `K` 8 (Qwen3.6-35B-A3B's shape):
/// across a link no step takes a bank record, so each group holds one
/// record and the floor is no slot; on
/// unified memory each group holds one decode launch's distinct experts
/// (`VERIFY_ROWS · K`, at most a layer's experts) beside the tier's
/// reserve, its priced path's, and the floor is the most slots
/// within them. A group commits its least records at every slots up to
/// the floor, its record regions so committed, and no other group more
/// for it.
#[test]
fn the_minimum_is_what_progress_holds_at_once() {
    let linked = beside(bank(&[37, 3], 0), HostTier::Linked);
    let unified = beside(bank(&[37, 3], 1), HostTier::Unified(Parking::InKernels));
    let request = (VERIFY_ROWS * 8).min(256);
    // The tier's reserve: its priced path's (a demand reclaims at its park).
    let reserve = PATHS.reserve;
    assert_eq!((request, reserve, linked.tier_reserve()), (64, 16, 16));
    for group in 0..2 {
        for plan in [&linked, &unified] {
            let need = plan.decode_records(group).max(plan.prompt_records(group)).max(1);
            assert_eq!(plan.least_records(group), need);
        }
    }
    // Only the lender holds a prompt group's records: its lent scratch,
    // one widest layer of every form, beside a host tier; on the one
    // store the conveyor and the reserve beside it. The other group
    // holds what a decode step does.
    // Beside a host tier the scratch is one widest layer of the widest
    // form, 256 Q6_K down records, in the lender's 589,824-byte Q4_K
    // records; on the one store it lies past the conveyor's two rounds.
    let (q4, q6) = (tensor("gguf_q4_k", "q4k").record_bytes().unwrap(), tensor("gguf_q6_k", "q6k").record_bytes().unwrap());
    let conveyor = unified.conveyor().conveyor.bytes;
    let (scratch, lent) = ((256 * q6).div_ceil(q4), (conveyor.down + 256 * q6).div_ceil(q4));
    assert_eq!((scratch, lent), (374, 438));
    assert_eq!((linked.scratch().lender.records, unified.scratch().lender.records), (scratch, lent));
    assert_eq!((linked.least_records(0), linked.least_records(1)), (scratch, 1));
    assert_eq!((unified.least_records(0), unified.least_records(1)), (lent + reserve, request + reserve));
    // Linked, the small group's one record holds no slot of its three
    // layers; unified, the lender's least holds 438 / 37 = 11 slots.
    assert_eq!((linked.needed_slots(), unified.needed_slots()), (0, 11));
    // A load holds what a step needs to make progress, and no more is
    // required of it: fewer slots cost the decode reads, which the settle
    // weighs.
    assert_eq!((linked.floor(), unified.floor()), (0, 11));
    for plan in [&linked, &unified] {
        for slots in 0..=plan.needed_slots() {
            assert_eq!(plan.records_at(0, slots), plan.least_records(0));
            assert_eq!(plan.records_at(1, slots), plan.least_records(1));
        }
        let layout = plan.layout().unwrap();
        for (group, records) in layout.groups.iter().enumerate() {
            for (kind, records) in records.records.iter() {
                assert_eq!(records, &BankAllocation::Records { tier: BankTier::Bank, group, kind }, "a group's records are its store's regions");
            }
        }
    }
    // Past the floor the large group commits its slots, the small one
    // its least records until its slots exceed them.
    assert_eq!((unified.records_at(0, 12), unified.records_at(1, 12)), (37 * 12 + reserve, request + reserve));
    assert_eq!(unified.records_at(1, 22), 3 * 22 + reserve);
    // Beside a host tier the lender holds its lent scratch until its
    // slots exceed it, the other group its slots.
    assert_eq!((linked.records_at(0, 10), linked.records_at(1, 10)), (scratch, 30));
    assert_eq!((linked.records_at(0, 11), linked.records_at(1, 1)), (37 * 11, 3));
}

/// All staging pools carried by the same tensor share one conveyor region
/// starting at offset 0 (mirroring `ScratchLayout`), so each carrier's bytes
/// equal `records * max_pool_record_bytes` rather than the sum across pools.
#[test]
fn the_lent_conveyors_share_one_region_in_their_carriers() {
    let plan = restructured(bank(&[37, 3], 64), |host, groups| {
        *host = HostTier::Unified(Parking::InKernels);
        // Mixed-quant down tensors across groups: Q5_K and Q6_K beside Q4_K.
        groups[0].record.down = tensor("gguf_q5_k", "q4k");
    });
    let pools = plan.staging_pools().0;
    let form = plan.form();
    let records = plan.conveyor_records();
    let conveyor = plan.structure().conveyor_layout(&pools, records).unwrap();
    let mut max_end = [0u64; 3];
    for (pool, &offset) in pools.iter().zip(&conveyor.offsets) {
        assert_eq!(offset, 0, "every pool shares its carrier from offset 0");
        let end = &mut max_end[form.carrier(pool.kind) as usize];
        *end = (*end).max(records * pool.record_bytes().unwrap());
    }
    for (kind, &bytes) in conveyor.bytes.iter() {
        assert_eq!(bytes, max_end[kind as usize], "{kind:?}'s conveyor reaches its largest pool's records");
    }
}

/// Multi-quant (`P >= 3` staging pools on one carrier) shares one slot array
/// (`total_bytes == conveyor_records * max_pool_record_bytes`, not `P * ...`)
/// and `HostTier::Linked` uses `2 * round_records` (not `widest`).
#[test]
fn multi_quant_conveyor_shares_one_slot_array_and_linked_uses_two_rounds() {
    let base = bank(&[37, 3], 64);
    let plan = restructured(base, |host, _| {
        *host = HostTier::Linked;
    });
    assert_eq!(plan.conveyor_records(), (2 * plan.round_records()).min(plan.window_slots()));
    assert!(plan.conveyor_records() < plan.widest().get(), "linked uses 2 * round_records, not widest");
    let q4 = tensor("gguf_q4_k", "q4k");
    let q5 = tensor("gguf_q5_k", "q4k");
    let q6 = tensor("gguf_q6_k", "q6k");
    let pools = vec![
        StagingPool::of(RoutedTensor::Gate, q4),
        StagingPool::of(RoutedTensor::Up, q4),
        StagingPool::of(RoutedTensor::Down, q4),
        StagingPool::of(RoutedTensor::Down, q5),
        StagingPool::of(RoutedTensor::Down, q6),
    ];
    let records = plan.conveyor_records();
    let layout = plan.structure().conveyor_layout(&pools, records).unwrap();
    assert_eq!(layout.offsets, vec![0; pools.len()]);
    let max_down = q4.stored_bytes().unwrap().max(q5.stored_bytes().unwrap()).max(q6.stored_bytes().unwrap());
    let sum_down = q4.stored_bytes().unwrap() + q5.stored_bytes().unwrap() + q6.stored_bytes().unwrap();
    assert_eq!(layout.bytes.gate, Some(records * max_down));
    assert!(layout.bytes.gate.unwrap() < records * sum_down, "3 down pools share one slot array, not end-to-end");
}

/// Each bank group gets its own record-slab granule (deduplicated across
/// identical groups, so uniform models still open 1 pool), so a mixed-quant
/// bank charges `sum(group_granule * group_records)` instead of
/// `max_granule * total_records` for both VRAM and Host RAM, and a linked host
/// tier sizes its host pools at 4 KiB host page granularity rather than VRAM's
/// 2 MiB granularity.
#[test]
fn per_group_granules_charge_each_groups_own_slab_on_vram_and_host() {
    let plan = bank(&[3, 1], 64);
    let slabs = (0..plan.groups().len())
        .map(|group| {
            let regions = plan.regions(group);
            assert!(regions.iter().map(|(kind, _)| *kind).eq(plan.groups()[group].record.iter().map(|(kind, _)| kind)));
            let regions = regions.into_iter().map(|(_, region)| region).collect::<Vec<_>>();
            PoolSlab::of(&seismic::SlabLayout::for_regions(1, plan.records(group), &regions).unwrap())
        })
        .collect::<Vec<_>>();
    for (group, slab) in slabs.iter().enumerate() {
        let record = plan.groups()[group].record.iter().map(|(_, tensor)| tensor.record_bytes().unwrap()).sum::<u64>();
        assert!(slab.bytes >= record, "group {group}'s slab holds its record");
        assert_eq!(plan.group_granule(group), slab.granule().unwrap(), "group {group} gets its own exact slab granule");
    }
    let small_granule = plan.group_granule(0);
    let large_granule = plan.group_granule(1);
    assert!(small_granule < large_granule, "q4k group granule ({small_granule}) must be smaller than q6k group granule ({large_granule})");
    assert_eq!(plan.granules(), &[small_granule, large_granule], "distinct granules deduplicated in group order");
    assert_eq!(plan.group_pools(), &[0, 1]);
    assert_eq!(plan.table_granule(), large_granule, "banked stored tables use the largest granule pool");
    assert!(plan.tiered());
    assert!(!beside(plan.clone(), HostTier::Unified(Parking::InKernels)).tiered());

    // VRAM charge at slots = 128 (past group 0's lent scratch records, so BOTH
    // groups hold pooled records): each group is charged its own granule (under
    // EXACT allocations), strictly less than charging all groups at max_granule.
    let slots = 128;
    assert!(plan.records_at(0, slots) > plan.lent(0), "group 0 must hold pooled VRAM records at {slots} slots");
    assert!(plan.records_at(1, slots) > plan.lent(1), "group 1 must hold pooled VRAM records at {slots} slots");
    let vram_expected: u64 = (0..plan.groups().len())
        .map(|group| (plan.records_at(group, slots) - plan.lent(group)) * plan.group_granule(group))
        .sum();
    let vram_max_granule: u64 = (0..plan.groups().len())
        .map(|group| plan.records_at(group, slots) - plan.lent(group))
        .sum::<u64>()
        * large_granule;
    assert_eq!(plan.resident_charge(slots).unwrap(), vram_expected);
    assert!(vram_expected < vram_max_granule, "per-group VRAM charge ({vram_expected}) must save over max-granule ({vram_max_granule})");

    // Host RAM charge: each group charged its own granule in the device's
    // native allocation blocks (matching Linux 2 MiB large-page registration
    // granularity), strictly less than charging all groups at max_granule.
    let blocks = plan.blocks();
    let host_expected: u64 = (0..plan.groups().len())
        .map(|group| {
            let pool = plan.group_pool(group);
            let pooled = plan.host_records(group, 64) - plan.host_lent(group);
            blocks[pool].charge(plan.group_granule(group), pooled).unwrap()
        })
        .sum();
    let host_max_granule: u64 = (0..plan.groups().len())
        .map(|group| plan.host_records(group, 64) - plan.host_lent(group))
        .sum::<u64>()
        * large_granule;
    assert_eq!(plan.host_charge(64).unwrap(), host_expected);
    assert!(host_expected < host_max_granule, "per-group Host charge ({host_expected}) must save over max-granule ({host_max_granule})");

    // Uniform 1-group bank opens only 1 distinct granule pool (zero overhead).
    let uniform = bank(&[4], 64);
    assert_eq!(uniform.granules().len(), 1);
    assert_eq!(uniform.blocks().len(), 1);

    // Under CUDA's 2 MiB granularity, a 1.4 MiB granule packs multiple records
    // per block so linked host tiers batch `cuMemHostRegister` into 2 MiB-aligned
    // large-page blocks rather than 1-record allocations.
    let cuda_allocs = seismic::NativeAllocations { least: 0, granularity: 2 << 20 };
    let cuda_blocks = size_pool_blocks(&[1_468_006u64.next_multiple_of(4096)], cuda_allocs).unwrap();
    assert!(
        cuda_blocks[0].granules(1_468_006u64.next_multiple_of(4096)) > 1,
        "2 MiB registration granularity must batch multiple ~1.4 MiB records per block"
    );
}

/// A host tier grows in one step from `from` toward every expert, its
/// ceiling, within its room ([`growth_target`] on its charge, `BankCounts::host_growth`); a bank
/// without a tier grows none, nor a whole one or one out of room.
#[test]
fn a_host_tier_grows_toward_every_expert_within_its_room() {
    let without = beside(bank(&[3, 1], 26), HostTier::Unified(Parking::InKernels));
    let plan = beside(without.clone(), HostTier::Linked);
    let all = u64::MAX;
    let counts = plan.counts().unwrap();
    assert_eq!(counts.host_growth(0, 26, all), Some(256), "every expert");
    assert_eq!(counts.host_growth(256, 26, all), None, "whole");
    assert_eq!(without.counts().unwrap().host_growth(0, 26, all), None, "no tier");
    let room = plan.host_charge(40).unwrap() - plan.host_charge(0).unwrap();
    assert_eq!(counts.host_growth(0, 26, room), Some(40), "within its room");
    // Up to 26 slots the tier holds its floor, the landing reserve.
    assert_eq!(counts.host_growth(0, 10, 0), Some(26), "slots within the reserve cost nothing");
    assert_eq!(counts.host_growth(26, 10, 0), None, "no room for one past it");
    // Its first slots, up to the bank's, hold the bank's copies: a growth
    // reaches past them, or none fits.
    assert_eq!(counts.host_growth(0, 26, 0), None, "no room past the bank's 26 slots");
    assert_eq!(counts.host_growth(0, 50, room), None, "a room of 40 slots holds none past the bank's 50");
    let past = plan.host_charge(60).unwrap() - plan.host_charge(0).unwrap();
    assert_eq!(counts.host_growth(0, 50, past), Some(60), "one step past the bank's 50");
}

/// A bank's counts are the plan's derivations at every count it may hold:
/// each group's records and their charge at every slot count up to the
/// ceiling, the host tier's at every host slot count, the records the host
/// lands into, and the floor.
#[test]
fn a_banks_counts_are_its_plans_derivations_at_every_count() {
    let unified = beside(bank(&[37, 3], 1), HostTier::Unified(Parking::InKernels));
    let linked = beside(bank(&[37, 3], 0), HostTier::Linked);
    for plan in [unified, linked] {
        // Growth reaches every expert.
        let plan = ExpertBankPlan { ceiling: plan.widest().get(), ..plan };
        let counts = plan.counts().unwrap();
        assert_eq!(counts.floor(), plan.floor());
        for slots in 0..=plan.ceiling {
            assert_eq!(counts.resident_charge(slots), plan.resident_charge(slots).unwrap(), "{slots} slots");
            for group in 0..plan.groups().len() {
                assert_eq!(counts.records_at(group, slots), plan.records_at(group, slots), "group {group} at {slots} slots");
                if plan.placer() == Placer::Bank {
                    assert_eq!(counts.landed_records(group, slots), plan.records_at(group, slots), "the host lands in the one store's slots");
                }
            }
        }
        let Some(ceiling) = plan.host_ceiling() else {
            assert_eq!(counts.host_charge(0), 0, "no host tier is charged");
            continue;
        };
        for host in 0..=ceiling {
            assert_eq!(counts.host_charge(host), plan.host_charge(host).unwrap(), "{host} host slots");
            for group in 0..plan.groups().len() {
                assert_eq!(counts.host_records(group, host), plan.host_records(group, host), "group {group} at {host} host slots");
                assert_eq!(counts.landed_records(group, host), plan.host_records(group, host), "a host tier lands at its host slots");
            }
        }
    }
}

/// A tiered record prices the misses its host tier holds as served by
/// the cost function at once, not after a storage read: beside a bank of
/// one slot, a tier of the decode's saturation in host slots holds every
/// miss, and the step costs what serving them from it does, the CPU and
/// the link bounded together by the host memory they read
/// ([`magnitude_cost_model::StepInputs`]).
#[test]
fn a_tiered_records_host_held_misses_are_priced_as_served() {
    use magnitude_cost_model::{DecodePrice, ExpertLines, FormSeconds, HeldSlots, StreamedLayers};
    let lines = ExpertLines::reference(1_000_000);
    let layers = StreamedLayers::new(vec![4], 4, std::sync::Arc::from([lines]));
    let decode = crate::domain::fixtures::misses_at(3, 100);
    let price = DecodePrice::new(layers, decode, crate::domain::fixtures::READ, FormSeconds { resident: 0.015, streamed: 0.015 });
    let host = price.saturation().expect("the decode found its reads") as usize;
    assert_eq!(host, 4, "the misses stop falling at the decode's 4 slots");
    let held = HeldSlots { slots: 1, host };
    let step = |held| price.step(held);
    let (bank_alone, tiered) = (step(HeldSlots::bank(1)), step(held));
    assert!(tiered.total_seconds < bank_alone.total_seconds, "{tiered:?} against {bank_alone:?}");
    // Four cold tier misses of one row a step and no hit, of 1 MB
    // records: three on the CPU after its handshake (4 + 3 · 20.7 µs)
    // and one in place over the link (44.6 µs) end at 66.2 µs, but every
    // one leaves host memory at its record's 19.7 µs, so the four take
    // 78.8 µs of it.
    let cpu = lines.host.cpu.unwrap();
    let served = (cpu.handshake.get() + 3. * cpu.expert.nanoseconds(1.)).max(lines.gpu.link.nanoseconds(1.)) * 1e-9;
    let memory = 4. * lines.host.memory.unwrap().get() * 1e-9;
    assert!(served < memory, "host memory bounds this step: {served} against {memory}");
    assert!((tiered.exposed_seconds - memory).abs() < 1e-12, "{tiered:?}");
}

/// A host tier is each group's record tensors again in their resident
/// form, regions of a slab store in the tier's own pool for every expert
/// of the group's layers (what the host lands file misses into and the
/// device reads in place and promotes from) and committed by host slots,
/// a record of every layer each and at least the landing reserve; its
/// charge is a granule of its pool for each record so committed past the
/// conveyor's lent ones, counted host storage. Staging is the file form's
/// one shared space with or without it, and a bank without a tier binds
/// none.
#[test]
fn a_host_tier_holds_each_groups_records_in_their_resident_form() {
    let without = beside(bank(&[3, 1], 26), HostTier::Unified(Parking::InKernels));
    let plan = beside(without.clone(), HostTier::Linked);
    assert_eq!((plan.host_ceiling(), without.host_ceiling()), (Some(256), None));
    let lent = plan.conveyor().carrier.lender().records;
    assert_eq!((plan.host_extent(0), plan.host_extent(1), without.host_extent(0)), (3 * 256 + lent, 256, 0));
    // Never fewer than the landing reserve.
    assert_eq!(plan.landing_records(), 80);
    assert_eq!((plan.host_records(0, 0), plan.host_records(0, 30), plan.host_records(1, 30)), (80 + lent, 90 + lent, 80));
    assert_eq!(plan.host_records(0, 300), 3 * 256 + lent, "a layer holds at most its experts");
    assert_eq!(without.host_records(0, 2), 0);
    let layout = plan.layout().unwrap();
    let tiers = layout.host.as_ref().expect("a tier's record tensors");
    assert_eq!(tiers.len(), 2);
    for (group, tier) in tiers.iter().enumerate() {
        for (kind, records) in tier.iter() {
            assert_eq!(records, &BankAllocation::Records { tier: BankTier::Host, group, kind });
        }
    }
    assert!(without.layout().unwrap().host.is_none());
    assert_eq!(layout.staging, without.layout().unwrap().staging, "staging is the tier's alike");
    // Charged each group's granule in its host pool's blocks past the lent
    // ones, which their lent allocation holds.
    let blocks = plan.blocks();
    let charge_host = |g0: u64, g1: u64| {
        blocks[0].charge(plan.group_granule(0), g0).unwrap()
            + blocks[1].charge(plan.group_granule(1), g1).unwrap()
    };
    assert_eq!(plan.host_lent(0), lent);
    assert_eq!(plan.host_charge(30).unwrap(), charge_host(90, 80));
    assert_eq!(plan.host_charge(0).unwrap(), charge_host(80, 80));
    assert_eq!(without.host_charge(2).unwrap(), 0);
    // A host tier adds its records, host memory, and the words the
    // device serves from it by: each layer's tier map; and each layer's
    // CPU site table, rows and output, four bytes each.
    let site = |layer: usize| {
        let binding = plan.binding(layer).unwrap();
        let (table, hidden) = (binding.words.site(), binding.record.down.matrix[0]);
        table.words() + table.slots() * 2 * VERIFY_ROWS * hidden
    };
    let words = plan.layers().iter().map(|layer| layer.experts).sum::<u64>() + (0..plan.layers().len()).map(site).sum::<u64>();
    // The conveyor and the scratch lie in their lenders' lent records,
    // allocations of their own beside the pools.
    let (held, bare) = (plan.storage_demand().unwrap(), without.storage_demand().unwrap());
    let lent = |plan: &ExpertBankPlan| {
        let layout = plan.layout().unwrap();
        ClaimDemand::checked_sum(layout.lent.iter().flat_map(|lent| lent.iter().map(|(_, lent)| lent.storage_demand(plan.host()).unwrap()))).unwrap()
    };
    // Beside the commits, each placer's own records' span.
    let commits = |plan: &ExpertBankPlan| (0..plan.groups().len()).map(|group| 4 * plan.records(group)).sum::<u64>();
    // Every landing launch declares its list, whoever places.
    assert_eq!(
        held.shared - lent(&plan).shared - commits(&plan),
        bare.shared - lent(&without).shared - commits(&without) + 4 * words
    );
    assert_eq!(
        (held.host - lent(&plan).host) - (bare.host - lent(&without).host),
        plan.host_charge(0).unwrap() + plan.absent_demand().host
    );
    // Admission charges a site only for the layers it admitted one for,
    // and only where a tier lets the plan hold one; a refusal is kept.
    let sites = (0..plan.layers().len())
        .map(|layer| if layer == 0 { LayerSite::Served } else { LayerSite::Refused("refused".into()) })
        .collect::<Vec<_>>();
    let admitted = plan.clone().with_sites(sites.clone()).unwrap();
    assert_eq!(plan.bytes().unwrap() - admitted.bytes().unwrap(), 4 * (1..plan.layers().len()).map(site).sum::<u64>());
    assert!(admitted.layout().unwrap().layers.iter().map(|layer| layer.sited.is_some()).eq((0..plan.layers().len()).map(|layer| layer == 0)));
    assert!(admitted.part_shape(0, BankPart::SiteOutput).is_ok());
    assert!(admitted.part_shape(1, BankPart::SiteOutput).is_err());
    assert_eq!(admitted.refused_sites(), (1..plan.layers().len()).map(|layer| (layer, "refused".to_owned())).collect::<Vec<_>>());
    assert!(without.clone().with_sites(sites).is_err(), "no site without a tier");
    assert!(plan.with_sites(vec![LayerSite::Served]).is_err(), "an admission names every layer");
}

/// The demand reserve holds the widest decode request's distinct misses
/// (`decode_set(K, widest) = (VERIFY_ROWS · K).min(widest)`) on every host
/// tier (unified in-kernel, unified at-boundaries, and linked): 64 of
/// `VERIFY_ROWS · 8` on layers of 256 experts. Every layer has its own
/// landing list and each round's, of a request's choices and a round's
/// records.
#[test]
fn the_demand_reserve_holds_the_widest_decode_set() {
    let expected = (VERIFY_ROWS * 8).min(256);
    let parked = |parking| beside(bank(&[37, 3], 1), HostTier::Unified(parking));
    let (kernels, boundaries) = (parked(Parking::InKernels), parked(Parking::AtBoundaries));
    assert_eq!((kernels.demand_records(), boundaries.demand_records()), (expected, expected));
    assert_eq!(beside(bank(&[37, 3], 0), HostTier::Linked).demand_records(), expected);
    let words = |(_, extents): (seismic::Element, Vec<u64>)| extents.iter().product::<u64>();
    for layer in [0, 39] {
        assert_eq!(words(kernels.part_shape(layer, BankPart::LandingList).unwrap()), magnitude_kernels::constants::LANDING_ENTRY(kernels.choices()));
        for round in 0..kernels.rounds() {
            let list = kernels.part_shape(layer, BankPart::RoundList { round: round as u8 }).unwrap();
            assert_eq!(words(list), magnitude_kernels::constants::LANDING_ENTRY(kernels.round_records()));
        }
    }
}

/// `demand_records()` equals `K` when `widest == 1` (so `K == 1`) and
/// `decode_set(K, widest)` for `widest > 1` on both
/// `HostTier::Unified(Parking::InKernels)` and `HostTier::Linked`, while
/// `staging_records()` leaves `hint_records()` free beyond the reserve.
#[test]
fn demand_records_sizes_for_single_and_multi_row_decode() {
    let with_shape = |selected: u64, widest: u64, host: HostTier| {
        let mut plan = bank_of(&[2], 1, |_| widest, RecordForm::Gated);
        let decode_rows = plan.decode_rows();
        let (_, _, groups, layers) = plan.structured.into_parts();
        plan.structured = conveyor::Structured::new(selected, host, groups, layers, None, decode_rows).unwrap();
        plan
    };
    for host in [HostTier::Unified(Parking::InKernels), HostTier::Linked] {
        let single = with_shape(1, 1, host);
        assert_eq!(single.demand_records(), 1, "{host:?} widest == 1");
        assert_eq!(single.staging_records() - single.demand_records(), single.hint_records(), "{host:?} widest == 1");

        let capped = with_shape(8, 24, host);
        assert_eq!(capped.demand_records(), (VERIFY_ROWS * 8).min(24), "{host:?} widest == 24");
        assert_eq!(capped.staging_records() - capped.demand_records(), capped.hint_records(), "{host:?} widest == 24");

        let wide = with_shape(8, 256, host);
        assert_eq!(wide.demand_records(), (VERIFY_ROWS * 8).min(256), "{host:?} widest == 256");
        assert_eq!(wide.staging_records() - wide.demand_records(), wide.hint_records(), "{host:?} widest == 256");
    }
}

/// On unified memory the bank is the one store: no tier records or
/// maps, so no promotion, and its landings land into bank records, so a
/// group commits its landing reserve; across a link the tier is its
/// own.
#[test]
fn a_unified_bank_is_one_store_the_host_places_into() {
    let without = beside(bank(&[3, 1], 26), HostTier::Unified(Parking::InKernels));
    let linked = beside(without.clone(), HostTier::Linked);
    let unified = beside(without.clone(), HostTier::Unified(Parking::InKernels));
    assert_eq!((linked.placer(), unified.placer()), (Placer::Tier, Placer::Bank));
    assert_eq!((linked.tiered(), unified.tiered()), (true, false));
    assert_eq!((unified.host_ceiling(), unified.host_extent(0), unified.host_records(0, 30)), (None, 0, 0));
    assert_eq!(unified.holding_storage(crate::BankHolding::Host), None);
    let layout = unified.layout().unwrap();
    assert!(layout.host.is_none(), "no tier records");
    assert!(layout.layers.iter().all(|layer| layer.tier.is_none()), "no tier map");
    assert!(unified.part_shape(0, BankPart::TierMap).is_err());
    assert_eq!(unified.binding(0).unwrap().placer, Placer::Bank);
    // The host's landing reserve lies in the bank's own records, and the
    // group holds a decode request beside it; across a link no step
    // takes a bank record.
    assert_eq!(unified.records_at(1, 26), unified.least_records(1));
    assert_eq!(linked.records_at(1, 26), 26, "a group that lends nothing holds its slots alone");
}

/// The scratch lender holds one widest layer of scratch for every group,
/// whatever its record form, and every group commits, at every slot
/// count, what a decode step and a prompt group hold live at once
/// (`least_records`): the lender its lent scratch (and on the one store
/// the conveyor and the reserve beside it), any other group only what a
/// decode step holds. At the Qwen3.6-35B q4 geometry (groups of 37, 2 and
/// 1 layers of 256 experts) from the floor to every expert, the one
/// store's and beside a host tier, and at a two-group fixture; each
/// layer's scratch view fits the region of its tensor.
#[test]
fn the_scratch_lender_holds_one_widest_layer_across_forms() {
    for (groups, host) in [&[37, 2, 1][..], &[3, 1]]
        .into_iter()
        .flat_map(|groups| [HostTier::Unified(Parking::InKernels), HostTier::Linked].map(|host| (groups, host)))
    {
        {
            let plan = beside(bank(groups, 26), host);
            let scratch = plan.scratch();
            assert_eq!(scratch.lender.group, 0, "the group of the most experts lends");
            // Each tensor's region is its largest form's experts from
            // where the region starts (past the conveyor on the one
            // store), end to end.
            let start = match host {
                HostTier::Unified(_) => plan.conveyor().conveyor.bytes,
                HostTier::Linked => plan.form().each(|_| Ok::<_, String>(0)).unwrap(),
            };
            let (forms, experts) = plan.structure().scratch_forms();
            for (kind, &bytes) in scratch.layout.bytes.iter() {
                let from = *start.get(kind).unwrap();
                let largest = forms
                    .iter()
                    .zip(&experts)
                    .filter(|(form, _)| form.kind == kind)
                    .map(|(form, &rows)| {
                        let record = record_bytes(form.resident, form.matrix).unwrap();
                        let (_, alignment) = seismic::Tensor::canonical_layout(form.resident, &[rows, form.matrix[0], form.matrix[1]]).unwrap();
                        from.next_multiple_of(alignment.max(1)) + rows * record
                    })
                    .max()
                    .unwrap();
                assert_eq!(bytes, largest, "{groups:?} {host:?}: {kind:?}'s region is its largest form's");
            }
            let forms = plan.layer_scratch_forms();
            for (layer, of) in forms.iter().enumerate() {
                let binding = plan.binding(layer).unwrap();
                for (kind, &form) in of.iter() {
                    let tensor = binding.record.get(kind).unwrap();
                    let view = record_bytes(tensor.resident, tensor.matrix).unwrap() * 256;
                    assert!(scratch.layout.offsets[form] + view <= *scratch.layout.bytes.get(kind).unwrap(), "layer {layer}'s view fits its tensor's region");
                }
            }
            for slots in plan.floor()..=256 {
                for group in 0..groups.len() {
                    let committed = plan.records_at(group, slots);
                    assert!(committed >= plan.prompt_records(group) && committed >= plan.decode_records(group), "{groups:?} {host:?} group {group} at {slots} slots");
                    assert!(committed <= plan.records(group));
                }
            }
            // The one-layer group at 145 slots (16 GB) holds its slots and
            // the reserve beside them: no scratch, which the lender lends.
            if groups.len() == 3 && host == HostTier::Unified(Parking::InKernels) {
                assert_eq!(plan.records_at(2, 145), 145 + plan.tier_reserve());
            }
        }
    }
}

/// Past every group's decode request, a slot commits `slot_bytes`, a
/// record of every layer in its group's form, and nothing else the bank
/// holds; the pool is charged a granule for each record committed.
#[test]
fn a_slot_commits_a_record_of_every_layer() {
    // Past the slots at which a group commits only its least records.
    let small = bank(&[3, 2], 129);
    let large = bank(&[3, 2], 130);
    let records = |plan: &ExpertBankPlan| {
        plan.layers()
            .iter()
            .map(|layer| {
                plan.groups()[layer.group]
                    .record
                    .iter()
                    .map(|(_, tensor)| plan.slots * tensor.record_bytes().unwrap())
                    .sum::<u64>()
            })
            .sum::<u64>()
    };
    assert_eq!(records(&large) - records(&small), small.slot_bytes().unwrap());
    // A record of each of the five layers, its own group's granule each.
    let pooled = large.resident_charge(130).unwrap() - small.resident_charge(129).unwrap();
    let expected_pooled: u64 = small.layers().iter().map(|layer| small.group_granule(layer.group)).sum();
    assert_eq!(pooled, expected_pooled);
    assert_eq!(
        large.charged_beyond_records().unwrap() - small.charged_beyond_records().unwrap(),
        pooled - small.slot_bytes().unwrap(),
        "beyond its records the bank holds only the granules' room past them"
    );
}

/// The bank's lent allocations are each one allocation of its lender's
/// lent records of a routed tensor: the largest is the most lent records
/// of any of the lender's tiers in its largest record, which the
/// device's single allocation bounds (`UnstreamedExperts::Allocation`).
#[test]
fn the_largest_lent_allocation_is_the_lenders_largest_record_tensor() {
    for host in [HostTier::Unified(Parking::InKernels), HostTier::Linked] {
        let plan = beside(bank(&[37, 3], 64), host);
        let lender = plan.scratch().lender.group;
        let record = plan.groups()[lender].record.iter().map(|(_, tensor)| tensor.record_bytes().unwrap()).max().unwrap();
        let lent = plan.lent(lender).max(plan.host_lent(lender));
        assert_eq!(plan.largest_lent().unwrap(), lent * record, "{host:?}");
        let layout = plan.layout().unwrap();
        assert_eq!(layout.lent.len(), match host {
            HostTier::Unified(_) => 1,
            HostTier::Linked => 2,
        });
    }
}

/// A bank holds a stored table's slots in records of the most rows a
/// granule of its pool holds, as many as the fixed slots hold, from the
/// floor of the most rows a step names, and no more than its rows fill
/// whole; unbanked, the table is planned fixed again. Beside a link,
/// where the host does not write the bank's pool, it stays fixed, as does
/// a table whose row exceeds a granule.
#[test]
fn a_bank_holds_its_stored_table_in_granule_records() {
    use crate::planning::fixtures::{qwen4exp_fixture_definition, streamed_experts_manifest};
    // The load of a whole table (its fixed slots are its rows) of `rows`
    // rows, resident and banked.
    let load = |rows: u64| {
        let mut definition = qwen4exp_fixture_definition();
        let (heads, width) = match &mut definition.decoder.blocks[0].sublayers[2].op {
            Operator::NGramInput(input) => {
                input.table.shape[0] = rows;
                (input.hash.heads().unwrap(), input.width)
            }
            _ => unreachable!("the fixture's first block ends in its n-gram input"),
        };
        let manifest = streamed_experts_manifest(&definition);
        let selection = crate::ComponentSelection { head: false, vision: false };
        let resident = ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows32, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS).unwrap();
        // CUDA's 2 MiB allocation granularity: the pool's blocks hold
        // several granules.
        let substrate = BankSubstrate {
            allocations: seismic::NativeAllocations { least: 0, granularity: 2 << 20 },
            ..cuda(HostTier::Unified(Parking::InKernels))
        };
        let banked = resident.clone().with_expert_bank_at_floor(&definition, substrate).unwrap();
        (definition, heads, width, resident, banked)
    };
    // Rows no number of records fills exactly, a record short of the
    // probe's: the table's records leave spare granules in their last block.
    let probe = load((1 << 20) + 1).4;
    let probe = probe.expert_bank().unwrap();
    let table_rows = (1 << 20) + 1 - probe.tables[0].record_rows.get();
    let (definition, heads, width, resident, banked) = load(table_rows);
    let fixed = resident.target().iter().find(|weight| weight.stored_in_slots()).unwrap().clone();
    assert_eq!(fixed.shape, [1, table_rows, width]);
    let bank = banked.expert_bank().unwrap();
    let [table] = &bank.tables[..] else {
        panic!("the bank holds the stored table");
    };
    let granule = bank.table_granule();
    let row = representation_bytes(fixed.resident, &[1, width]).unwrap();
    let (records, record_rows, record_bytes) = (table.records.get(), table.record_rows.get(), table.record_bytes.get());
    assert_eq!(record_rows, granule / row);
    assert_eq!(record_bytes, record_rows * row);
    assert!(record_bytes <= granule && granule - record_bytes < row);
    assert_eq!(records, table_rows / record_rows);
    assert!(records * record_rows <= table_rows);
    let step_rows = heads * magnitude_batching::MAX_CLASS_ROWS as u64;
    assert_eq!(table.floor, step_rows.div_ceil(record_rows));
    let held = banked.target().iter().find(|weight| weight.role == fixed.role).unwrap();
    assert_eq!(held.shape, [records, record_rows, width]);
    // The table's records are its own, a granule each: what its slots are
    // charged as they hold them (`StoredTable::charged_bytes`). They leave
    // spare granules in the last block, the bank's pool's, which the bank
    // is charged beyond its records' (`ExpertBankPlan::storage_beyond_records`).
    assert_eq!(held.resident_bytes, BankTable::held_bytes(granule, records).unwrap());
    assert_ne!(records % bank.table_block().granules(granule), 0, "the table's records leave spare granules in their last block");
    let blocks = BankTable::charge(bank.table_block(), granule, records).unwrap();
    assert!(blocks > held.resident_bytes);
    assert_eq!(bank.tables_charge().unwrap(), blocks);
    let bare = ExpertBankPlan { tables: Vec::new(), ..bank.clone() };
    assert_eq!(
        bank.charged_beyond_records().unwrap() - bare.charged_beyond_records().unwrap(),
        blocks - held.resident_bytes,
        "the pool's spare granules are the bank's"
    );
    assert_eq!(held.record_table_entries().unwrap(), records);
    let ported = banked.clone().with_record_ports(seismic::BackendName::Cuda);
    assert_eq!(ported.weight_device_allocations(held).unwrap(), Vec::<u64>::new(), "banked stored table's slot table is host-shared, not device-only");
    let directory = held.storage_bytes().unwrap() - held.resident_bytes;
    let slot_table = seismic::record_table_bytes(records).unwrap();
    assert_eq!(
        ported.weight_storage_bytes(held).unwrap(),
        held.resident_bytes + directory + slot_table,
        "planned storage bytes include the banked stored table's host-shared slot address table"
    );
    assert_eq!(banked.clone().unbanked().target(), resident.target());
    let linked = resident.clone().with_expert_bank_at_floor(&definition, cuda(HostTier::Linked)).unwrap();
    assert!(linked.expert_bank().unwrap().tables.is_empty());
    assert_eq!(linked.target().iter().find(|weight| weight.role == fixed.role), Some(&fixed));
    // A granule short of a row holds none: the table stays fixed.
    assert_eq!(BankTable::of(&fixed, 1, row - 1).unwrap(), None);
}

/// A bank's layers are at least one, each of experts, so the most any
/// has is positive: none, or a layer of none, is no bank's.
#[test]
fn a_banks_layers_have_experts() {
    let layers = bank(&[3, 1], 26).layers().to_vec();
    assert_eq!(BankLayers::new(layers.clone()).unwrap().widest().get(), 256);
    assert_eq!(BankLayers::new(Vec::new()), None);
    let mut empty = layers;
    empty[1].experts = 0;
    assert_eq!(BankLayers::new(empty), None);
}

/// Every routed layer's expert stacks become its group's records: the
/// shape every member addresses, charged its own slots, imported from
/// nothing; the weights grow by a granule a record a slot commits, and the program
/// plan names each streamed block's bank.
#[test]
fn a_bank_plans_every_routed_layer_into_its_groups_records() {
    use crate::planning::fixtures::{
        qwen4exp_fixture_definition, stored_table_manifest, streamed_experts_manifest,
    };
    let definition = qwen4exp_fixture_definition();
    let manifest = streamed_experts_manifest(&definition);
    let selection = crate::ComponentSelection {
        head: false,
        vision: false,
    };
    // Dense experts bank as packed ones do: the fill converts each
    // element as `import_dense` imports it.
    let dense = ModelLoadPlan::derive(
        &stored_table_manifest(&definition),
        &definition,
        selection,
        Layout::Rows32,
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    assert!(dense.with_expert_bank_at_floor(&definition, cuda(HostTier::Linked)).is_ok());
    let resident =
        ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows32, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS).unwrap();
    let banked = |slots| resident.clone().with_expert_bank(&definition, slots, None, 1, FlightDepth::PIPELINED, cuda(HostTier::Linked)).unwrap();
    let floor = resident.bank_floor(&definition, 1, FlightDepth::PIPELINED, cuda(HostTier::Linked)).unwrap();
    let (three, four) = (banked(floor), banked(floor + 1));
    let bank = three.expert_bank().unwrap();
    assert_eq!((bank.widest().get(), bank.selected(), bank.groups().len()), (32, 2, 1));
    assert_eq!(bank.layers().len(), 2);
    // Where the device places, a prompt group's lent scratch is all a
    // step takes of the bank: the floor holds a layer's experts; each
    // slot past it a record of each layer.
    assert_eq!(floor, 16);
    assert_eq!(bank.floor(), floor);
    assert_eq!(banked(floor).expert_bank().unwrap().committed_records(0), 32);
    assert_eq!(banked(floor + 1).expert_bank().unwrap().committed_records(0), 34);
    for weight in three.target() {
        let Some(group) = weight.banked() else {
            continue;
        };
        let tensor = *bank.groups()[group].record.get(RoutedTensor::of(weight.role.kind).unwrap()).unwrap();
        assert_eq!(weight.shape, [64, tensor.matrix[0], tensor.matrix[1]]);
        assert_eq!(weight.resident_bytes, floor * tensor.record_bytes().unwrap());
        assert_eq!(weight.source_bytes, 0);
    }
    assert_eq!(three.target().iter().filter(|weight| weight.banked().is_some()).count(), 6);
    // A slot past the floor commits a record of each layer, a granule of
    // its group's pool each.
    assert_eq!(four.weight_bytes().unwrap() - three.weight_bytes().unwrap(), 2 * bank.group_granule(0));
    assert!(bank.group_granule(0) >= bank.slot_bytes().unwrap() / 2);
    let programs = three
        .program_plan(&definition, magnitude_state::KvCodec::AffineK8V4)
        .unwrap();
    assert!(
        programs
            .target()
            .blocks()
            .iter()
            .enumerate()
            .all(|(layer, block)| block.bank().map(|(_, bank)| bank) == Some(bank.binding(layer).unwrap()))
    );
    // A bank goes only beside routed experts: never a lone mixer's.
    let first = programs.target().blocks()[0];
    assert_eq!(first.bank().map(|(banked, _)| banked.slot()), first.feed_forward());
    let lone = crate::TargetBlockProgramSlot::new(first.mixer(), None);
    assert!(lone.with_bank(bank.binding(0).unwrap()).is_err());
    // From its floor to every expert: a bank of every expert is whole,
    // its layers' homes its records end to end. On unified memory a
    // bank below its floor is refused.
    assert!(resident.clone().with_expert_bank(&definition, 33, None, 1, FlightDepth::PIPELINED, cuda(HostTier::Linked)).is_err());
    let unified = |slots| resident.clone().with_expert_bank(&definition, slots, None, 1, FlightDepth::PIPELINED, cuda(HostTier::Unified(Parking::InKernels)));
    let floor = unified(32).unwrap().expert_bank().unwrap().floor();
    assert!(floor > 0 && unified(floor).is_ok() && unified(floor - 1).is_err(), "unified floor {floor}");
    let whole = resident.clone().with_expert_bank(&definition, 32, None, 1, FlightDepth::PIPELINED, cuda(HostTier::Linked)).unwrap();
    let bank = whole.expert_bank().unwrap();
    assert!(bank.whole() && !three.expert_bank().unwrap().whole());
    assert_eq!((bank.home(0), bank.home(1)), (0, 32));
    assert_eq!(bank.committed_records(0), bank.records(0));
    // Without its bank the plan is the resident one.
    assert_eq!(whole.unbanked(), resident);
}

/// Each layer records what the route before it reads of it, alike in
/// every layer; a layer whose router is held in another element than
/// the first's is refused at planning, before any graph reads it as the
/// route's own.
#[test]
fn a_route_predicts_only_layers_of_its_own_elements() {
    use crate::planning::fixtures::{
        qwen4exp_fixture_definition, relaid_manifest, streamed_experts_manifest,
    };
    use magnitude_artifacts::gguf::Encoding;
    let definition = qwen4exp_fixture_definition();
    let selection = crate::ComponentSelection {
        head: false,
        vision: false,
    };
    let banked = |manifest| {
        ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows32, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS)
            .unwrap()
            .with_expert_bank_at_floor(&definition, cuda(HostTier::Linked))
    };
    let manifest = streamed_experts_manifest(&definition);
    let plan = banked(manifest.clone()).unwrap();
    let bank = plan.expert_bank().unwrap();
    let routers = bank
        .layers()
        .iter()
        .map(|layer| {
            let router = layer.hinted.iter().find(|hint| hint.kind == WeightKind::Router);
            router.and_then(|router| router.element).unwrap()
        })
        .collect::<Vec<_>>();
    assert!(routers.len() > 1 && routers.iter().all(|router| *router == routers[0]));
    // The second streamed layer's router stored as Q8_0 blocks.
    let WeightScope::TargetSublayer(second) = bank.layers()[1].scope else {
        panic!("a streamed layer is a target sublayer");
    };
    let router = format!("blk.{}.ffn_gate_inp", second.block);
    let relaid = relaid_manifest(manifest, &[router], Encoding::Q8_0, 34);
    assert!(matches!(
        banked(relaid),
        Err(PlanError::Unstreamed(UnstreamedExperts::SelectionOrder(index))) if index == second
    ));
}

/// A banked load's target graphs build on Metal at every class, beside a
/// linked tier and as the one store: the streamed decode and
/// scratch-served forms check against the entries' contracts and
/// natives (a convert's conveyor and scratch, views of one store's
/// records, alias no binding).
#[test]
fn banked_target_graphs_build_on_metal() {
    use crate::planning::fixtures::{qwen4exp_fixture_definition, streamed_experts_manifest};
    use crate::{
        ExecutionPath, PlannedMethod, ResourceCapacity, ResourceLimits, ResourcePlanner,
        resident_layout,
    };
    let backend = seismic::BackendName::Metal;
    let definition = qwen4exp_fixture_definition();
    let manifest = streamed_experts_manifest(&definition);
    let selection = crate::ComponentSelection {
        head: false,
        vision: false,
    };
    let codec = magnitude_state::KvCodec::Dense;
    let limits = ResourceLimits {
        max_launch_rows: 64,
        max_launch_slots: 8,
        max_selected_rows: 64,
        max_drafting_slots: 8,
        exported_logits_rows: 0,
        max_images_per_request: 1,
        max_image_cells: 0,
        lookahead: false,
        chain_capacity: std::num::NonZeroUsize::MIN,
        conversations: std::num::NonZeroUsize::MIN,
    };
    for host in [HostTier::Linked, HostTier::Unified(Parking::InKernels)] {
    let resident = ModelLoadPlan::derive(
        &manifest,
        &definition,
        selection,
        resident_layout(ExecutionPath::Native, backend),
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    let substrate = crate::planning::bank_substrate(backend, seismic::NativeAllocations::EXACT, Some(host), &definition).unwrap();
    let load = resident.with_expert_bank_at_floor(&definition, substrate).unwrap();
    let plan = load.program_plan(&definition, codec).unwrap();
    let state = ResourcePlanner::state_plan(
        &definition,
        &load,
        PlannedMethod::Plain,
        codec,
        limits,
        ResourceCapacity {
            domain_bytes: 16 * 1024 * 1024 * 1024,
            tensor_operations: crate::TensorOperations::Absent,
            allocations: seismic::NativeAllocations::EXACT,
        },
    )
    .unwrap();
    let upload_regions = crate::programs::native_target_graph::checked_target_family(
        backend,
        &load,
        &definition.decoder,
        &state,
        plan.target(),
        limits,
        crate::PassLaunches::whole_chain(limits),
    )
    .unwrap_or_else(|error| panic!("{error}"))
    .upload_regions;
    // A step's runs that hold an upload region: its entry, the launch's
    // controls, which every block binds (the mixers' and the n-gram
    // input's tables), and block 0's front, which writes its sequence.
    assert_eq!(upload_regions.len(), 3);
    }
}

/// A recorded discrete Vulkan device (the RTX 3090's) reads files into its
/// host tier, linked across its link: the plan's carrier is the tier's,
/// so a conveyor in memory the host does not read into has no
/// representation.
#[test]
fn a_discrete_vulkan_conveyor_lies_in_its_host_tier() {
    use crate::planning::fixtures::{qwen4exp_fixture_definition, streamed_experts_manifest};
    let topology = seismic::fixtures::recorded::rtx3090();
    let executor = topology.executors().iter().find(|executor| executor.backend() == seismic::BackendName::Vulkan).unwrap();
    let host = crate::platform::host_tier(&topology, executor).unwrap();
    assert_eq!(host, HostTier::Linked, "a discrete device's tier is linked");
    let definition = qwen4exp_fixture_definition();
    let manifest = streamed_experts_manifest(&definition);
    let selection = crate::ComponentSelection { head: false, vision: false };
    let resident = ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows32, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS).unwrap();
    let load = resident.with_expert_bank_at_floor(&definition, BankSubstrate { backend: seismic::BackendName::Vulkan, allocations: seismic::NativeAllocations::EXACT, host }).unwrap();
    let bank = load.expert_bank().unwrap();
    assert!(matches!(bank.conveyor().carrier, ConveyorCarrier::HostTier(_)));
    assert!(bank.layout().unwrap().host.is_some());
}

/// The conveyor lies in the bottom records of the group of the most
/// experts, two rounds' records of each staging pool end to end, in the
/// store the host reads files into: on unified memory the bank's own, its
/// lent records the scratch's too; beside a discrete device the host
/// tier's, so the bank's least records count its lent scratch alone. The
/// lent records are each lender's allocation of their own, which the
/// store's bottom records view.
#[test]
fn the_conveyor_lies_in_the_store_the_host_reads_into() {
    let discrete = bank(&[37, 3], 64);
    let unified = beside(discrete.clone(), HostTier::Unified(Parking::InKernels));
    // Gate and up have one stored form, down two: Q4_K then Q6_K.
    let q4 = tensor("gguf_q4_k", "q4k").stored_bytes().unwrap();
    let q6 = tensor("gguf_q6_k", "q6k").stored_bytes().unwrap();
    // Both linked and unified use two rounds' records bounded by window_slots.
    let expected = (2 * unified.round_records()).min(unified.window_slots());
    assert_eq!((discrete.conveyor_records(), unified.conveyor_records()), (expected, expected));
    for plan in [&discrete, &unified] {
        let records = plan.conveyor_records();
        let conveyor = plan.conveyor();
        assert_eq!(conveyor.carrier.lender().group, 0, "the group of the most experts lends");
        // Each tensor's conveyor in the next tensor's records: gate's in
        // up's, up's in down's, down's in gate's, pools sharing offset 0.
        assert_eq!(
            conveyor.conveyor.bytes,
            RecordTensors { gate: Some(records * q4.max(q6)), up: records * q4, down: records * q4 }
        );
        assert_eq!(plan.binding(0).unwrap().conveyor, records);
    }
    let ConveyorCarrier::HostTier(lender) = discrete.conveyor().carrier else {
        panic!("beside a discrete device the host tier carries the conveyor");
    };
    assert!(lender.records > 0);
    assert_eq!(discrete.host_extent(0), 37 * 256 + lender.records);
    assert_eq!(discrete.host_extent(1), 3 * 256);
    assert_eq!(discrete.prompt_records(0), discrete.scratch().lender.records, "a discrete bank's scratch alone");
    assert_eq!(discrete.prompt_records(1), 0, "a group that lends nothing");
    let ConveyorCarrier::Bank(lender) = unified.conveyor().carrier else {
        panic!("on unified memory the one store carries the conveyor");
    };
    assert_eq!(lender, unified.scratch().lender, "one lending, the conveyor's and the scratch's");
    assert_eq!(unified.prompt_records(0), lender.records + unified.tier_reserve());
    assert_eq!(unified.prompt_records(1), 0, "a group that lends nothing");
    assert!(unified.least_records(0) >= unified.prompt_records(0));
    assert!(unified.records_at(0, unified.floor()) >= unified.prompt_records(0));
    // Each lent allocation: the lender's lent records of each of its
    // routed tensors, in its tier.
    for (plan, lent) in [(&unified, vec![(BankTier::Bank, unified.lent(0))]), (&discrete, vec![(BankTier::Bank, discrete.lent(0)), (BankTier::Host, discrete.host_lent(0))])] {
        let layout = plan.layout().unwrap();
        assert_eq!(layout.lent.len(), lent.len());
        for (allocations, (tier, records)) in layout.lent.iter().zip(lent) {
            for ((kind, allocation), (_, tensor)) in allocations.iter().zip(plan.groups()[0].record.iter()) {
                assert_eq!(
                    allocation,
                    &BankAllocation::Lent { tier, group: 0, kind, element: tensor.resident, extents: vec![records, tensor.matrix[0], tensor.matrix[1]] }
                );
            }
        }
    }
}

/// The bank's bytes are its layout's, each allocation as charged, and its
/// pools' granules of the records it commits: every word part of every
/// layer, group and the load at its graph ports' shape, the staging
/// pools and the lent records.
#[test]
fn a_bank_charges_every_allocation_of_its_layout() {
    let plan = bank(&[3, 1], 64);
    let layout = plan.layout().unwrap();
    // Group 0's three layers, then group 1's one: a layer's words and
    // a group's map and commits as its parts are shaped.
    for (index, layer) in layout.layers.iter().enumerate() {
        let words = |part| {
            let (element, extents) = plan.part_shape(index, part).unwrap();
            BankAllocation::Words { element, extents }
        };
        assert_eq!(layer.rings, words(BankPart::Rings));
        assert_eq!(layer.rounds.len() as u64, plan.rounds(), "a round list of every round, the layer's own");
    }
    assert_eq!(layout.staging.len(), plan.staging_pools().0.len());
    // The floor's one chain place of each placed part, at its stride.
    let floor = plan.placed(std::num::NonZeroU64::MIN).unwrap();
    assert_eq!(floor.step, [(Element::i32(), vec![plan.place_stride(0, BankPart::Step).unwrap()])]);
    let floor = floor
        .iter()
        .map(|(element, extents)| BankAllocation::Words { element: *element, extents: extents.clone() })
        .collect::<Vec<_>>();
    let total = layout
        .iter()
        .chain(&floor)
        .map(|allocation| allocation.charged_bytes(plan.host()).unwrap())
        .sum::<u64>()
        + plan.resident_charge(plan.slots).unwrap()
        + plan.host_charge(0).unwrap()
        + plan.absent_demand().bytes();
    assert_eq!(plan.bytes().unwrap(), total);
    let records = (0..plan.layers().len()).map(|index| plan.banked_bytes(index).unwrap().into_iter().sum::<u64>());
    assert_eq!(plan.charged_beyond_records().unwrap(), total - records.sum::<u64>());
}

/// A group's placed words are each placed part's shape once a chain
/// place, its group reads once a step in flight, and are charged as
/// they grow; no group holds more places than its plan's.
#[test]
fn a_banks_placed_words_grow_with_its_places() {
    let plan = ExpertBankPlan { places: 3, ..bank(&[3, 1], 64) };
    let places = std::num::NonZeroU64::new(3).unwrap();
    let (one, all) = (plan.placed(std::num::NonZeroU64::MIN).unwrap(), plan.placed(places).unwrap());
    let rows = |(_, extents): &(Element, Vec<u64>)| extents[0];
    // Each place's step words apart, a place's own.
    assert_eq!(all.step, [&one.step[..]; 3].concat());
    assert_eq!(rows(&all.flags), 3 * rows(&one.flags));
    for (one, all) in one.layers.iter().zip(&all.layers) {
        assert_eq!(
            rows(&all.group_reads) * plan.flights.steps(1),
            rows(&one.group_reads) * plan.flights.steps(3)
        );
    }
    assert!(plan.placed_bytes(places).unwrap() > plan.placed_bytes(std::num::NonZeroU64::MIN).unwrap());
    assert!(plan.placed(places.checked_add(1).unwrap()).is_err(), "beyond the plan's places");
}

/// Layers routing over different expert counts (Nemotron 3.5 Bolt's) and
/// up-only records take one rule: each layer's homes, slots and words
/// are its own experts', and what every layer shares in turn (the
/// conveyor, the scratch flags, the seed list, the rounds) is the
/// widest layer's; the floor holds each group's least records, and the
/// bank is whole once the slots reach the widest.
#[test]
fn layers_of_different_expert_counts_size_their_records_by_their_own() {
    let experts = [240, 496, 304];
    let plan = bank_of(&[2, 1], 300, |layer| experts[layer], RecordForm::Up);
    assert_eq!(plan.widest().get(), 496);
    assert_eq!((plan.records(0), plan.records(1)), (240 + 496, 304));
    assert_eq!((plan.home(0), plan.home(1), plan.home(2)), (0, 240, 0));
    // The narrowest layer holds every expert at 300 slots, the others
    // their slots: a group that lends nothing holds no scratch.
    assert_eq!((plan.committed_records(0), plan.committed_records(1)), (240 + 300, 300));
    assert!(!plan.whole() && bank_of(&[2, 1], 496, |layer| experts[layer], RecordForm::Up).whole());
    let shape = |layer, part| plan.part_shape(layer, part).unwrap().1;
    assert_eq!(shape(0, BankPart::SlotMap), [240, constants::MAP_WORDS]);
    assert_eq!(shape(0, BankPart::Map), [240 + 496, constants::MAP_WORDS]);
    assert_eq!(shape(1, BankPart::GroupReads), [496]);
    assert_eq!(shape(0, BankPart::ScratchFlags), [496]);
    assert_eq!(shape(2, BankPart::Scratch { kind: RoutedTensor::Up })[0], 304);
    assert_eq!(shape(0, BankPart::Conveyor { kind: RoutedTensor::Down })[0], plan.conveyor_records());
    assert_eq!(plan.rounds(), 496_u64.div_ceil(plan.round_records()));
    assert_eq!(plan.binding(0).unwrap().next_experts, 496);
    assert_eq!(plan.binding(2).unwrap().next_experts, 304, "the last layer hints nothing");
    // An up-only record's group holds no gate records.
    let layout = plan.layout().unwrap();
    assert!(layout.groups.iter().all(|group| group.records.gate.is_none()));
}

/// A banked stack is read from its stored experts' bytes as they lie:
/// whole, or a range of each expert's rows (Gemma 4's gate‖up halves),
/// whose first row it reads from; a second-level scale leaves the bytes
/// as stored, and any other host preparation is refused.
#[test]
fn a_banked_stack_is_its_stored_experts_or_a_row_range_of_each() {
    use magnitude_family_contracts::{RowRange, WeightDescriptor};
    let with = |transforms| WeightDescriptor {
        transforms,
        ..WeightDescriptor::stored("blk.0.ffn_gate_up_exps.weight", [128, 704, 2816])
    };
    assert_eq!(stored_rows(&with(vec![])), Some(0));
    assert_eq!(stored_rows(&with(vec![ImportTransform::Rows(RowRange { start: 704, rows: 704 })])), Some(704));
    assert_eq!(stored_rows(&with(vec![ImportTransform::PermuteRows { order: vec![1, 0] }])), None);
    let ranges = vec![ImportTransform::Rows(RowRange { start: 0, rows: 704 }); 2];
    assert_eq!(stored_rows(&with(ranges)), None);
}

/// Every byte the bank demands has its owner: the owners' charges sum
/// to its demand, with a host tier or without.
#[test]
fn every_byte_the_bank_demands_has_its_owner() {
    let without = beside(bank(&[3, 1], 26), HostTier::Unified(Parking::InKernels));
    let with = beside(without.clone(), HostTier::Linked);
    for plan in [without, with] {
        let charges = plan.charges().unwrap();
        let total = charges.iter().try_fold(ClaimDemand::default(), |total, &(_, demand)| total.checked_plus(demand)).unwrap();
        assert_eq!(total, plan.storage_demand().unwrap());
        assert!(charges.iter().any(|&(owner, _)| owner == "staging"), "{charges:?}");
    }
}

/// Rounds deliver every expert, half the staging at a time; a step's
/// scratch slot holds its request's lane, then a lane each round.
#[test]
fn scratch_rounds_deliver_every_expert() {
    let plan = bank(&[4], 64);
    assert_eq!(plan.staging_records(), (VERIFY_ROWS * 8).min(256) + PATHS.ahead);
    assert_eq!(plan.round_records(), 32);
    assert_eq!(plan.rounds(), 8);
    let table = plan.binding(0).unwrap().words.scratch();
    assert_eq!(table.lanes(), constants::ROUND_LANE(8));
    assert_eq!(table.width(), constants::SCRATCH_SLOT);
    assert_eq!(table.lane(1, constants::ROUND_LANE(7)), (table.lanes() + constants::ROUND_LANE(7)) * table.width());
}

/// Every placed part's instances begin at whole u64s at any count of
/// places, so each place's words the host views (`experts::Words`) are
/// 8-byte aligned, whatever the part's own words (here odd expert
/// counts, so a group reads' and the flags' are odd).
#[test]
fn every_placed_parts_stride_is_whole_u64s() {
    let experts = [241, 497, 305];
    let plan = bank_of(&[2, 1], 300, |layer| experts[layer], RecordForm::Up);
    for places in 1..=plan.places {
        let places = std::num::NonZeroU64::new(places).unwrap();
        let placed = plan.placed(places).unwrap();
        for (layer, part) in (0..plan.layers().len()).flat_map(|layer| {
            [BankPart::Step, BankPart::ScratchFlags, BankPart::GroupReads].map(|part| (layer, part))
        }) {
            let stride = plan.place_stride(layer, part).unwrap();
            assert_eq!(stride % 2, 0, "{part:?} of layer {layer} at {places} places");
            assert!(stride >= plan.part_shape(layer, part).unwrap().1[0]);
        }
        let rows = |(_, extents): &(Element, Vec<u64>)| extents[0];
        assert_eq!(rows(&placed.flags) % 2, 0);
        assert!(placed.layers.iter().all(|layer| rows(&layer.group_reads) % 2 == 0));
    }
    assert!(plan.place_stride(0, BankPart::FileTable).is_err(), "a part bound whole has no stride");
}

/// Item 63: a linked tier's landing rows are a committed claim at load, a
/// request's misses and the hints' records (`landing_records`), each a
/// record of every group as resident; unified memory lands in the bank's
/// own records, which its floor holds.
#[test]
fn a_linked_tiers_landing_rows_are_its_loads_claim() {
    let linked = beside(bank(&[37, 3], 0), HostTier::Linked);
    let unified = beside(bank(&[37, 3], 1), HostTier::Unified(Parking::InKernels));
    let row: u64 = linked
        .groups()
        .iter()
        .map(|group| group.record.iter().map(|(_, tensor)| tensor.record_bytes().unwrap()).sum::<u64>())
        .sum();
    assert_eq!(
        linked.landing().unwrap(),
        Some(crate::memory::ledger::Need::LandingRows { tier: crate::planning::BankTier::Host, rows: linked.landing_records(), slot_bytes: row })
    );
    assert_eq!(unified.landing().unwrap(), None);
}

/// Windowed scratch (`W < widest`) bounds `ScratchPlan`, `BankPart::Scratch`,
/// `prompt_records`, `least_records`, `floor`, `Window`, and `WindowTable`,
/// while `W == widest` leaves `P == 1` and every plan byte-identical.
#[test]
fn windowed_scratch_bounds_lending_and_layout() {
    let full = bank(&[4], 64);
    let widest = full.widest().get();
    let round = full.round_records();
    assert_eq!(full.window_slots(), widest);
    assert_eq!(full.windows(), 1);
    assert_eq!(full.window_rounds(), widest.div_ceil(round));
    assert_eq!(full.rounds(), full.window_rounds());
    assert_eq!(full.layer_window(0), widest);
    assert_eq!(full.expert_list_words(), constants::LIST_ENTRY(widest));
    assert_eq!(windowed(&full, widest).unwrap(), full);

    assert!(windowed(&full, round - 1).is_err());
    assert!(windowed(&full, widest + 1).is_err());

    let w = round;
    let win = windowed(&full, w).unwrap();
    let p = widest.div_ceil(w);
    assert_eq!(win.window_slots(), w);
    assert_eq!(win.windows(), p);
    assert_eq!(win.window_rounds(), 1);
    assert_eq!(win.rounds(), p);
    assert_eq!(win.layer_window(0), w);
    assert_eq!(win.expert_list_words(), constants::LIST_ENTRY(w));
    assert!(win.prompt_records(0) < full.prompt_records(0));
    assert!(win.least_records(0) < full.least_records(0));
    assert!(win.lent(0) < full.lent(0));

    let shape = |part| win.part_shape(0, part).unwrap().1;
    assert_eq!(shape(BankPart::Scratch { kind: RoutedTensor::Up })[0], w);
    assert_eq!(shape(BankPart::Window), [1]);
    assert_eq!(shape(BankPart::WindowTable), [widest]);
    assert!(BankPart::Window.windowed() && BankPart::WindowTable.windowed() && BankPart::RoundList { round: 0 }.windowed());
    assert!(!BankPart::Step.windowed());

    let layout = win.layout().unwrap();
    assert_eq!(layout.load.window, vec![BankAllocation::Device { element: Element::i32(), extents: vec![1] }; p as usize]);
    assert_eq!(layout.layers[0].windows, vec![BankAllocation::Words { element: Element::i32(), extents: vec![widest] }; p as usize]);
    assert_eq!(layout.layers[0].rounds.len(), win.rounds() as usize);
}

/// A bank's staging and its tier's reserve are its priced path's records
/// (`ExpertBankPlan::paths`): staging the demand reserve and the moves in
/// flight, a step's landings its choices and those moves, the tier's
/// reserve the path's free records, which the one store holds beside every
/// expert. Unpriced, a bank reads nothing ahead: staging holds demand
/// alone and the tier keeps nothing free.
#[test]
fn a_banks_staging_and_reserve_are_its_priced_paths() {
    let priced = |plan: ExpertBankPlan, paths| ExpertBankPlan { paths, ..plan };
    let paths = BankPaths::Priced { line: LINE, records: magnitude_cost_model::PathRecords { ahead: 5, reserve: 23 } };
    for host in [HostTier::Linked, HostTier::Unified(Parking::InKernels)] {
        let plan = priced(beside(bank(&[4], 64), host), paths);
        assert_eq!(plan.staging_records(), plan.demand_records() + 5, "{host:?}");
        assert_eq!(plan.landing_records(), plan.choices() + 5, "{host:?}");
        assert_eq!(plan.tier_reserve(), 23, "{host:?}");
        let unpriced = priced(plan.clone(), BankPaths::Unpriced);
        assert_eq!((unpriced.staging_records(), unpriced.tier_reserve()), (plan.demand_records(), 0), "{host:?}");
    }
    let unified = priced(beside(bank(&[4], 64), HostTier::Unified(Parking::InKernels)), paths);
    assert_eq!(unified.records(0), (4 * 256 + 23).max(unified.least_records(0)), "the one store holds the reserve beside every expert");
}

/// Two streamed fused layers that share an expert record group, expert count,
/// and site status, but differ in `SharedDown` quant, produce distinct
/// `routed_output` `Kind`s and therefore must get distinct `RoutedKind`s so
/// `Census::new` does not panic.
#[test]
fn layers_with_different_shared_down_quants_get_distinct_routed_kinds_in_census() {
    use magnitude_cost_model::{Census, Kind, LaunchCounts, LayerCall, RoutedCall};
    use seismic::measure::pricing::{Bytes, Format, Work};

    let mut plan = bank(&[2], 64);
    let mut layers: Vec<BankLayer> = plan.layers().to_vec();
    let q4 = Element::stored("q4k", Layout::Rows16).unwrap();
    let q6 = Element::stored("q6k", Layout::Rows16).unwrap();
    layers[0].shared_down = Some((q4, 1024, vec![64, 64]));
    layers[1].shared_down = Some((q6, 1536, vec![64, 64]));
    plan.structured = conveyor::Structured::new(
        plan.choices(),
        HostTier::Linked,
        plan.groups().to_vec(),
        BankLayers::new(layers).unwrap(),
        None,
        plan.decode_rows(),
    )
    .unwrap();

    let kinds = plan.routed_kinds();
    assert_ne!(kinds[0], kinds[1], "distinct SharedDown quants must yield distinct RoutedKinds");

    let decode_rows = 1u64;
    let call = |format: &'static str, bytes: u64, layer: usize| {
        (
            Kind {
                work: Work {
                    weights: vec![Bytes { format: Format(format), bytes }],
                    ..Work::default()
                },
                launches: LaunchCounts::new(vec![Ok(1); decode_rows as usize]),
            },
            Some(LayerCall::Routed(RoutedCall { kind: kinds[layer], layer })),
        )
    };
    let census = Census::new([call("q4k", 1024, 0), call("q6k", 1536, 1)], 0, decode_rows);
    assert_eq!(census.kinds().len(), 2);
}

/// On unified memory (`HostTier::Unified`, where lent allocations live in
/// `Contract::Shared` while pooled records live in `device`), growing a
/// mixed-quant bank from `floor` to any `slots > floor` — including while the
/// lender group is still below `lent(lender)` — must keep `shared` constant
/// and grow `device` monotonically by `resident_charge(slots) - resident_charge(floor)`,
/// so `full.checked_minus(floor_demand)` never underflows.
#[test]
fn unified_mixed_quant_target_weight_demand_never_underflows_floor() {
    let mut plan = beside(bank(&[3, 1], 64), HostTier::Unified(Parking::InKernels));
    let floor = plan.floor();
    let widest = plan.widest().get();
    plan.slots = floor;
    plan.ceiling = widest;
    let mut target = Vec::new();
    for (index, layer) in plan.layers().iter().enumerate() {
        let banked = plan.banked_bytes(index).unwrap();
        for (kind, mut weight) in layer.resident.clone().into_tensors() {
            weight.role = WeightRole {
                scope: WeightScope::TargetSublayer(magnitude_family_contracts::SublayerIndex { block: index as u32, sublayer: 0 }),
                kind: kind.kind(),
            };
            weight.descriptor.name = format!("layer{index}.{kind:?}");
            weight.shape[0] = plan.records(layer.group);
            weight.source_bytes = 0;
            weight.resident_bytes = *banked.get(kind).unwrap();
            weight.holding = Holding::Banked { group: layer.group };
            target.push(weight);
        }
    }
    let load = ModelLoadPlan::for_bank_test(target, plan);

    let floor_plan = load.clone().with_load_slots(floor, 0).unwrap();
    let floor_demand = floor_plan.target_weight_demand().unwrap();
    let floor_bank = floor_plan.expert_bank().unwrap();
    let floor_resident = floor_bank.resident_charge(floor).unwrap();

    // Exercise the exact regime where `slots > floor` while the lender group
    // (group 0) still commits only its lent records.
    let mut exercised_lent_regime = false;
    for slots in floor..=widest {
        let sized = load.clone().with_load_slots(slots, 0).unwrap();
        let full = sized.target_weight_demand().unwrap();
        let delta = full
            .checked_minus(floor_demand)
            .unwrap_or_else(|| panic!("bank growth demand underflowed floor demand at slots={slots} (floor={floor})"));
        let resident = sized.expert_bank().unwrap().resident_charge(slots).unwrap();
        assert_eq!(delta.shared, 0, "shared/lent demand must stay constant at slots={slots}");
        assert_eq!(delta.device, resident - floor_resident, "device delta must equal pooled record growth at slots={slots}");
        assert_eq!(delta.host, 0);
        if slots > floor && floor_bank.records_at(0, slots) == floor_bank.least_records(0) {
            exercised_lent_regime = true;
        }
    }
    assert!(exercised_lent_regime, "test must exercise slots > floor while lender group is still at its lent floor");
}
