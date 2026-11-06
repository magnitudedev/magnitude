use super::*;
use super::records::record_bytes;
use super::table::gcd;
use crate::error::{PlanError, UnstreamedExperts};
use crate::operators::routed::fused_graph::DECODE_ROWS;
use crate::planning::ModelLoadPlan;
use crate::planning::weights::{Holding, representation_bytes};
use magnitude_family_contracts::{ImportTransform, ModelDefinition, Operator};

impl ModelLoadPlan {
    /// The plan at its bank's least slots, of one chain place.
    pub(crate) fn with_expert_bank_at_floor(self, definition: &ModelDefinition, reservation: seismic::Reservation, host: HostTier) -> Result<Self, PlanError> {
        let floor = self.bank_floor(definition, 1, FlightDepth::PIPELINED, reservation, host)?;
        self.with_expert_bank(definition, floor, 1, FlightDepth::PIPELINED, reservation, host)
    }
}

/// The fixture a partial bank streams through on the device
/// (`ngram_steps_tests::a_partial_banks_shared_graphs_read_each_layers_own_experts`):
/// at 128 experts a layer its banks' weights rise with their slots, on
/// unified memory's one store and beside a linked host tier, so a cap
/// below the whole bank's admits a partial one. At 32 experts a layer
/// every bank weighs what the whole one does (a reservation's granules
/// hold every slot), so no cap admits one.
#[test]
fn a_fixture_of_more_experts_admits_a_partial_bank() {
    use crate::planning::fixtures::{HIDDEN, mix_site_hidden, qwen4exp_fixture_definition_of, streamed_experts_manifest};
    // 2 MiB granules at the planning fixture's width, scaled with the
    // fixtures' hidden width as their expert records are, so a granule
    // holds the same share of the bank at any width.
    let granule = (2 << 20) * mix_site_hidden() / HIDDEN;
    let granules = |host_addressed| seismic::Reservation { granule, host_addressed, chunks: None };
    for (backend, reservation, host) in [
        (seismic::BackendName::Metal, granules(true), HostTier::Unified),
        (seismic::BackendName::Cuda, granules(false), HostTier::Linked(granules(true))),
    ] {
        let weights = |experts: u64| {
            let definition = qwen4exp_fixture_definition_of(3, experts);
            let manifest = streamed_experts_manifest(&definition);
            let selection = crate::ComponentSelection { head: false, vision: false };
            let resident = crate::ModelLoadPlan::derive(&manifest, &definition, selection, crate::resident_layout(crate::ExecutionPath::Native, backend), crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS)
                .unwrap();
            let floor = resident.bank_floor(&definition, 1, FlightDepth::PIPELINED, reservation, host).unwrap();
            [floor, experts / 2, experts].map(|slots| {
                resident.clone().with_expert_bank(&definition, slots, 1, FlightDepth::PIPELINED, reservation, host).unwrap().weight_bytes().unwrap()
            })
        };
        let [floor, half, whole] = weights(128);
        assert!(floor < half && half < whole, "{backend:?}: {floor} {half} {whole}");
        let [floor, _, whole] = weights(32);
        assert_eq!(floor, whole, "{backend:?}");
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

/// Metal's reservation granule.
const GRANULE: u64 = 2 << 20;

/// Metal's reservations: host ranges backed in its granule.
const RESERVATION: seismic::Reservation = seismic::Reservation {
    granule: GRANULE,
    host_addressed: true,
    chunks: None,
};

/// Host reservations backed byte by byte: the weights follow the slots
/// exactly.
const BYTES: seismic::Reservation = seismic::Reservation {
    granule: 1,
    host_addressed: true,
    chunks: None,
};

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
                });
            }
            BankGroup {
                record,
                layers: NonEmpty::new((first..first + members).collect()).unwrap(),
            }
        })
        .collect();
    let layers = BankLayers::new(layers).unwrap();
    let (reservation, host) = (RESERVATION, HostTier::Linked(RESERVATION));
    ExpertBankPlan {
        structured: conveyor::Structured::new(8, reservation, host, groups, layers).unwrap(),
        slots,
        ceiling: slots,
        host_slots: 0,
        places: 1,
        flights: FlightDepth::PIPELINED,
        tables: Vec::new(),
        sites: BankSites::Unadmitted,
    }
}

/// `plan` with its structure changed by `change`, and that structure's
/// lending.
fn restructured(
    plan: ExpertBankPlan,
    change: impl FnOnce(&mut seismic::Reservation, &mut HostTier, &mut Vec<BankGroup>),
) -> ExpertBankPlan {
    let (selected, mut reservation, mut host, mut groups, layers) = plan.structured.into_parts();
    change(&mut reservation, &mut host, &mut groups);
    let structured = conveyor::Structured::new(selected, reservation, host, groups, layers).unwrap();
    ExpertBankPlan { structured, ..plan }
}

/// `plan` beside `host`.
fn beside(plan: ExpertBankPlan, host: HostTier) -> ExpertBankPlan {
    restructured(plan, |_, tier, _| *tier = host)
}

/// The bank's minimum is its derivation (`least_records`), on groups of
/// 37 layers and 3 of 256 experts, `K` 8 (Qwen3.6-35B-A3B's shape):
/// across a link no step takes a bank record, so each group holds the
/// one record a reservation commits and the floor is no slot; on
/// unified memory each group holds one decode launch's distinct experts
/// (`DECODE_ROWS · K`, at most a layer's experts) beside the landing
/// reserve (`DECODE_ROWS · K + 2K`), and the floor is the most slots
/// within them. A group commits its least records at every slots up to
/// the floor, its record tensors so committed, and no other group more
/// for it.
#[test]
fn the_minimum_is_what_progress_holds_at_once() {
    let linked = beside(bank(&[37, 3], 0), HostTier::Linked(BYTES));
    let unified = beside(bank(&[37, 3], 1), HostTier::Unified);
    let request = (DECODE_ROWS * 8).min(256);
    // The tier's reserve: the reads ahead's, a record per hint staging
    // record (a demand reclaims at its park).
    let reserve = 2 * 8;
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
    // A tenth of 256 experts stands above both: a load holds at least
    // that, whatever a step needs to make progress.
    assert_eq!((linked.floor(), unified.floor()), (26, 26));
    for plan in [&linked, &unified] {
        for slots in 0..=plan.needed_slots() {
            assert_eq!(plan.records_at(0, slots), plan.least_records(0));
            assert_eq!(plan.records_at(1, slots), plan.least_records(1));
        }
        let layout = plan.layout().unwrap();
        for (group, records) in layout.groups.iter().enumerate() {
            for (_, records) in records.records.iter() {
                let BankAllocation::Reserved { committed, .. } = records else {
                    panic!("a group's records are reserved");
                };
                assert_eq!(*committed, plan.records_at(group, plan.slots), "group {group} commits its records at its slots");
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

/// In chunked records the lent conveyors pack their records whole into
/// their carriers' chunks (`RecordForm::carrier`), the packing running
/// on from one pool to the next in a carrier: no record crosses a
/// chunk, no gap before a record, within a pool or between pools,
/// reaches a record, and each carrier's bytes end where its last pool's
/// last record does.
#[test]
fn a_chunked_lender_packs_conveyor_records_into_its_chunks() {
    let plan = restructured(bank(&[37, 3], 64), |reservation, host, groups| {
        *reservation = seismic::Reservation {
            chunks: Some(seismic::ReservationChunks { unit: 16 << 10, table: seismic::Contract::DeviceOnly, whole: false }),
            ..RESERVATION
        };
        *host = HostTier::Unified;
        // The lender stores down in Q5_K and holds it in Q4_K, so its Q5_K
        // conveyor ends inside a chunk and the next pool packs on from there.
        groups[0].record.down = tensor("gguf_q5_k", "q4k");
    });
    let pools = plan.staging_pools().0;
    let form = plan.form();
    let carriers = plan.groups()[0].record.map(|kind, tensor| plan.reservation().of_records(plan.reserved_records(0, kind, tensor, crate::planning::BankTier::Bank).unwrap()));
    let conveyor = plan.structure().conveyor_layout(&pools, &carriers, plan.widest().get()).unwrap();
    let mut end = [0u64; 3];
    for (pool, &offset) in pools.iter().zip(&conveyor.offsets) {
        let carrier = *carriers.get(form.carrier(pool.kind)).unwrap();
        let (chunk, record) = (carrier.granule, pool.record_bytes().unwrap());
        let packing = carrier.packing(record).unwrap();
        let end = &mut end[form.carrier(pool.kind) as usize];
        assert_eq!(packing.at(offset, 0), offset);
        for expert in 0..conveyor.records {
            let start = packing.at(offset, expert);
            assert_eq!(start / chunk, (start + record - 1) / chunk, "record {expert} crosses a chunk");
            assert!(start - *end < record, "record {expert} wastes a record before it");
            *end = start + record;
        }
        assert_eq!(packing.end(offset, conveyor.records), *end);
    }
    for (kind, &bytes) in conveyor.bytes.iter() {
        assert_eq!(bytes, end[kind as usize], "{kind:?}'s conveyors end at their last record");
    }
    // The lender's chunks are four of its 589,824-byte Q4_K records, and
    // Q5_K records, 3 to a chunk, waste 196,608 bytes of each.
    let record = tensor("gguf_q4_k", "q4k").record_bytes().unwrap();
    let q5 = tensor("gguf_q5_k", "q5k").record_bytes().unwrap();
    assert_eq!(carriers.get(RoutedTensor::Gate).unwrap().granule, 4 * record);
    assert_eq!((4 * record / q5, 4 * record % q5), (3, 196_608));
}

/// A host tier grows in one step from `from` toward every expert, its
/// ceiling, within its room ([`growth_target`] on its charge, `BankCounts::host_growth`); a bank
/// without a tier grows none, nor a whole one or one out of room.
#[test]
fn a_host_tier_grows_toward_every_expert_within_its_room() {
    let without = beside(bank(&[3, 1], 26), HostTier::Unified);
    let plan = beside(without.clone(), HostTier::Linked(BYTES));
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
    let unified = beside(bank(&[37, 3], 1), HostTier::Unified);
    let linked = beside(bank(&[37, 3], 0), HostTier::Linked(BYTES));
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
        for host in 0..=ceiling.get() {
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
/// ([`crate::assessment::StepInputs`]).
#[test]
fn a_tiered_records_host_held_misses_are_priced_as_served() {
    use crate::assessment::{DecodePrice, ExpertLines, HeldSlots, StreamedLayers};
    let decode = crate::domain::fixtures::misses_at(3, 100);
    let host = decode.saturation as usize;
    assert_eq!(host, 4, "the misses stop falling at the decode's 4 slots");
    let held = HeldSlots { slots: 1, host };
    let lines = ExpertLines::reference(1_000_000);
    let layers = StreamedLayers::new(vec![4], 4, std::sync::Arc::from([lines]));
    let price = DecodePrice::new(layers, decode.live, crate::domain::fixtures::READ, 0.015);
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
/// form, reserved in host storage for every expert of the
/// group's layers (what the host lands file misses into and the device
/// reads in place and promotes from) and committed by host slots, a
/// record of every layer each and at least the landing reserve; its charge is the resident bytes so committed,
/// counted host-shared. Staging is the file form's one shared space
/// with or without it, and a bank without a tier binds none.
#[test]
fn a_host_tier_holds_each_groups_records_in_their_resident_form() {
    let without = beside(bank(&[3, 1], 26), HostTier::Unified);
    let plan = beside(without.clone(), HostTier::Linked(BYTES));
    assert_eq!((plan.host_ceiling(), without.host_ceiling()), (NonZeroU64::new(256), None));
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
        for ((_, records), (kind, tensor)) in tier.iter().zip(plan.groups()[group].record.iter()) {
            assert_eq!(
                records,
                &BankAllocation::Reserved {
                    kind: crate::planning::BankTier::Host,
                    contract: seismic::ReservedContract::Shared,
                    element: tensor.resident,
                    records: plan.host_extent(group),
                    matrix: tensor.matrix,
                    committed: plan.host_records(group, 0),
                    carries: plan.reserved_records(group, kind, *tensor, crate::planning::BankTier::Host).unwrap().carries,
                }
            );
        }
    }
    assert!(without.layout().unwrap().host.is_none());
    assert_eq!(layout.staging, without.layout().unwrap().staging, "staging is the tier's alike");
    // Charged the resident bytes of the committed records, byte by byte.
    let resident = |group: usize| plan.groups()[group].record.iter().map(|(_, tensor)| tensor.record_bytes().unwrap()).sum::<u64>();
    assert_eq!(plan.host_charge(30).unwrap(), (90 + lent) * resident(0) + 80 * resident(1));
    assert_eq!(plan.host_charge(0).unwrap(), (80 + lent) * resident(0) + 80 * resident(1));
    assert_eq!(without.host_charge(2).unwrap(), 0);
    // A host tier adds its records, host memory, and the words the
    // device serves from it by: each layer's tier map; and each layer's
    // CPU site table, rows and output, four bytes each.
    let site = |layer: usize| {
        let binding = plan.binding(layer).unwrap();
        let (table, hidden) = (binding.words.site(), binding.record.down.matrix[0]);
        table.words() + table.slots() * 2 * constants::DECODE_ROWS * hidden
    };
    let words = plan.layers().iter().map(|layer| layer.experts).sum::<u64>() + (0..plan.layers().len()).map(site).sum::<u64>();
    // No conveyor or scratch is an allocation of its own: they are
    // records of the tier or the bank.
    let (held, bare) = (plan.storage_demand().unwrap(), without.storage_demand().unwrap());
    // Beside the move lists and commits, each placer's own records' span.
    let moves = |plan: &ExpertBankPlan| {
        (0..plan.groups().len()).map(|group| 4 * (list_extent(plan.move_records(group)) + plan.records(group))).sum::<u64>()
    };
    // The barrier's convert grids, only where the host places: every
    // tier wait declares its list.
    let landing = (0..without.layers().len()).map(|layer| without.part_shape(layer, BankPart::LandingGrids).unwrap().1.iter().product::<u64>()).sum::<u64>();
    assert_eq!(held.shared - moves(&plan), bare.shared - moves(&without) + 4 * words - 4 * landing);
    assert_eq!(held.host - bare.host, plan.host_charge(0).unwrap());
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

/// On unified memory the bank is the one store: no tier records or
/// maps, so no promotion, and its landings land into bank records, so a
/// group commits its landing reserve; across a link the tier is its
/// own.
#[test]
fn a_unified_bank_is_one_store_the_host_places_into() {
    let without = beside(bank(&[3, 1], 26), HostTier::Unified);
    let linked = beside(without.clone(), HostTier::Linked(BYTES));
    let unified = beside(without.clone(), HostTier::Unified);
    assert_eq!((linked.placer(), unified.placer()), (Placer::Tier, Placer::Bank));
    assert_eq!((linked.tier_reservation(), unified.tier_reservation()), (Some(BYTES), None));
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
    let chunked = seismic::Reservation { granule: 1 << 16, host_addressed: true, chunks: Some(seismic::ReservationChunks { unit: 1 << 22, table: seismic::Contract::DeviceOnly, whole: false }) };
    for (groups, host, reservation) in [&[37, 2, 1][..], &[3, 1]]
        .into_iter()
        .flat_map(|groups| [HostTier::Unified, HostTier::Linked(BYTES)].map(|host| (groups, host)))
        .flat_map(|(groups, host)| [RESERVATION, chunked].map(|reservation| (groups, host, reservation)))
    {
        {
            let plan = restructured(bank(groups, 26), |reserved, tier, _| (*reserved, *tier) = (reservation, host));
            let scratch = plan.scratch();
            assert_eq!(scratch.lender.group, 0, "the group of the most experts lends");
            // Each tensor's region is its largest form's experts from
            // where the region starts (past the conveyor on the one
            // store), packed whole into chunks where storage is chunked.
            let start = match host {
                HostTier::Unified => plan.conveyor().conveyor.bytes,
                HostTier::Linked(_) => plan.form().each(|_| Ok::<_, String>(0)).unwrap(),
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
                        let at = from.next_multiple_of(alignment.max(1));
                        // Whole records to a chunk on chunked storage
                        // (seismic's packing, `Reservation::packing`).
                        let tensor = plan.groups()[0].record.get(kind).unwrap();
                        let carrier = reservation.of_records(plan.reserved_records(0, kind, *tensor, crate::planning::BankTier::Bank).unwrap());
                        let packing = carrier.packing(record).unwrap();
                        let end = packing.end(packing.at(at, 0), rows);
                        if reservation.chunks.is_none() {
                            assert_eq!(end, at + rows * record, "unchunked storage packs records end to end");
                        }
                        end
                    })
                    .max()
                    .unwrap();
                assert_eq!(bytes, largest, "{groups:?} {host:?} {reservation:?}: {kind:?}'s region is its largest form's");
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
            if groups.len() == 3 && host == HostTier::Unified {
                assert_eq!(plan.records_at(2, 145), 145 + plan.tier_reserve());
            }
        }
    }
}

/// Past every group's decode request, a slot commits `slot_bytes`, a
/// record of every layer in its group's form, and nothing else the bank
/// holds; each record tensor is charged its committed bytes in whole
/// granules.
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
    // The granules backing each record tensor's committed records
    // beyond their bytes.
    let rounding = |plan: &ExpertBankPlan| {
        let layout = plan.layout().unwrap();
        layout
            .groups
            .iter()
            .flat_map(|group| group.records.iter().map(|(_, records)| records))
            .map(|records| {
                let BankAllocation::Reserved {
                    element,
                    matrix,
                    committed,
                    ..
                } = records
                else {
                    panic!("a group's records are reserved");
                };
                let record = record_bytes(*element, *matrix).unwrap();
                records.charged_bytes(&RESERVATION).unwrap() - committed * record
            })
            .sum::<u64>()
    };
    let beside = |plan: &ExpertBankPlan| plan.charged_beyond_records().unwrap() - rounding(plan);
    assert_eq!(beside(&large), beside(&small));
    for plan in [&small, &large] {
        let rounding = rounding(plan);
        assert_eq!((records(plan) + rounding) % GRANULE, 0);
        assert!(rounding < 6 * GRANULE);
    }
}

/// The most slots keep every record tensor's backed granules within the
/// single-allocation limit, and one more does not.
#[test]
fn the_allocation_limit_bounds_each_record_tensors_commitment() {
    let limit = 3_000_000_000;
    let most = bank(&[37, 3], 64).most_slots(limit).unwrap();
    let largest = |slots| {
        let plan = bank(&[37, 3], slots);
        plan.record_tensors()
            .map(|(group, _, tensor)| {
                (plan.committed_records(group) * tensor.record_bytes().unwrap())
                    .next_multiple_of(GRANULE)
            })
            .max()
            .unwrap()
    };
    assert!(largest(most) <= limit && largest(most + 1) > limit);
}

/// A range its backend binds as one buffer once every granule is backed
/// (Metal) is one native allocation then: just within the limit, a bank's
/// slots reach every expert, so it can be whole and run resident; a byte
/// short, its slots stop short of backing every granule of its largest
/// range, so it is never whole and stays streamed.
#[test]
fn a_range_bound_whole_past_the_allocation_limit_keeps_the_bank_streamed() {
    let plan = restructured(bank(&[37, 3], 64), |reservation, _, _| {
        *reservation = seismic::Reservation {
            chunks: Some(seismic::ReservationChunks { unit: 16 << 10, table: seismic::Contract::DeviceOnly, whole: true }),
            ..RESERVATION
        };
    });
    // The largest range one buffer binds: a group's records as reserved.
    let largest = plan
        .record_tensors()
        .map(|(group, kind, tensor)| {
            let held = plan.reserved_records(group, kind, tensor, crate::planning::BankTier::Bank).unwrap();
            plan.reservation().of_records(held).backing(plan.records(group) * held.record.get()).unwrap()
        })
        .max()
        .unwrap();
    let widest = plan.widest().get();
    assert_eq!(plan.most_slots(largest).unwrap(), widest, "within the limit every expert has a slot");
    let short = plan.most_slots(largest - 1).unwrap();
    assert!(short < widest, "past the limit the bank is never whole: {short} of {widest} slots");
    // Bound in chunks alone, the same range is never one allocation.
    let chunked = restructured(bank(&[37, 3], 64), |reservation, _, _| {
        *reservation = seismic::Reservation {
            chunks: Some(seismic::ReservationChunks { unit: 16 << 10, table: seismic::Contract::DeviceOnly, whole: false }),
            ..RESERVATION
        };
    });
    assert_eq!(chunked.most_slots(largest - 1).unwrap(), widest);
}

/// A bank holds a stored table's slots in records of the least rows that
/// fill whole granules, as many as the fixed slots hold, from the floor
/// of the most rows a step names, and no more than its rows fill whole;
/// unbanked, the table is planned fixed again. A device whose host does
/// not address reserved bytes keeps it fixed.
#[test]
fn a_bank_holds_its_stored_table_in_granule_records() {
    use crate::planning::fixtures::{qwen4exp_fixture_definition, streamed_experts_manifest};
    // A whole table (its fixed slots are its rows) of rows no number of
    // records fills exactly.
    const TABLE_ROWS: u64 = (1 << 20) + 1;
    let mut definition = qwen4exp_fixture_definition();
    let (heads, width) = match &mut definition.decoder.blocks[0].sublayers[2].op {
        Operator::NGramInput(input) => {
            input.table.shape[0] = TABLE_ROWS;
            (input.hash.heads().unwrap(), input.width)
        }
        _ => unreachable!("the fixture's first block ends in its n-gram input"),
    };
    let manifest = streamed_experts_manifest(&definition);
    let selection = crate::ComponentSelection { head: false, vision: false };
    let resident = ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows32, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS).unwrap();
    let fixed = resident.target().iter().find(|weight| weight.stored_in_slots()).unwrap().clone();
    assert_eq!(fixed.shape, [1, TABLE_ROWS, width]);
    let reservation = seismic::Reservation { granule: 1 << 16, host_addressed: true, chunks: Some(seismic::ReservationChunks { unit: 1 << 14, table: seismic::Contract::DeviceOnly, whole: false }) };
    let banked = resident.clone().with_expert_bank_at_floor(&definition, reservation, HostTier::Linked(reservation)).unwrap();
    let [table] = &banked.expert_bank().unwrap().tables[..] else {
        panic!("the bank holds the stored table");
    };
    let row = representation_bytes(fixed.resident, &[1, width]).unwrap();
    let (records, record_rows, record_bytes) = (table.records.get(), table.record_rows.get(), table.record_bytes.get());
    assert_eq!(record_bytes % reservation.granule, 0);
    assert_eq!(record_bytes, record_rows * row);
    assert_eq!(record_bytes / reservation.granule, row / gcd(row, reservation.granule));
    assert_eq!(records, TABLE_ROWS / record_rows);
    assert!(records * record_rows <= TABLE_ROWS);
    let step_rows = heads * magnitude_batching::MAX_CLASS_ROWS as u64;
    assert_eq!(table.floor, step_rows.div_ceil(record_rows));
    let held = banked.target().iter().find(|weight| weight.role == fixed.role).unwrap();
    assert_eq!(held.shape, [records, record_rows, width]);
    assert_eq!(held.resident_bytes, records * record_bytes);
    assert_eq!(held.record_table_entries().unwrap(), records);
    assert_eq!(banked.clone().unbanked().target(), resident.target());
    let unaddressed = seismic::Reservation { host_addressed: false, ..reservation };
    let fixed_table = resident.clone().with_expert_bank_at_floor(&definition, unaddressed, HostTier::Linked(unaddressed)).unwrap();
    assert!(fixed_table.expert_bank().unwrap().tables.is_empty());
    assert_eq!(fixed_table.target().iter().find(|weight| weight.role == fixed.role), Some(&fixed));
    // A reservation that backs no granule holds no records: refused.
    let ungranular = seismic::Reservation { granule: 0, ..reservation };
    assert!(BankTable::of(&fixed, 1, ungranular).is_err());
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
/// nothing; the weights grow by `slot_bytes` a slot, and the program
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
    assert!(dense.with_expert_bank_at_floor(&definition, RESERVATION, HostTier::Linked(RESERVATION)).is_ok());
    let resident =
        ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows32, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS).unwrap();
    // Byte granules: the weights follow the slots exactly.
    let banked = |slots| resident.clone().with_expert_bank(&definition, slots, 1, FlightDepth::PIPELINED, BYTES, HostTier::Linked(BYTES)).unwrap();
    let floor = resident.bank_floor(&definition, 1, FlightDepth::PIPELINED, BYTES, HostTier::Linked(BYTES)).unwrap();
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
    assert_eq!(
        four.weight_bytes().unwrap() - three.weight_bytes().unwrap(),
        bank.slot_bytes().unwrap()
    );
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
    assert!(resident.clone().with_expert_bank(&definition, 33, 1, FlightDepth::PIPELINED, BYTES, HostTier::Linked(BYTES)).is_err());
    let unified = |slots| resident.clone().with_expert_bank(&definition, slots, 1, FlightDepth::PIPELINED, BYTES, HostTier::Unified);
    let floor = unified(32).unwrap().expert_bank().unwrap().floor();
    assert!(floor > 0 && unified(floor).is_ok() && unified(floor - 1).is_err(), "unified floor {floor}");
    let whole = resident.clone().with_expert_bank(&definition, 32, 1, FlightDepth::PIPELINED, BYTES, HostTier::Linked(BYTES)).unwrap();
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
            .with_expert_bank_at_floor(&definition, BYTES, HostTier::Linked(BYTES))
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
    for host in [HostTier::Linked(RESERVATION), HostTier::Unified] {
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        selection,
        resident_layout(ExecutionPath::Native, backend),
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap()
    .with_expert_bank_at_floor(&definition, RESERVATION, host)
    .unwrap();
    let codec = magnitude_state::KvCodec::Dense;
    let plan = load.program_plan(&definition, codec).unwrap();
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
    let state = ResourcePlanner::state_plan(
        &definition,
        &load,
        PlannedMethod::Plain,
        codec,
        limits,
        ResourceCapacity {
            domain_bytes: 16 * 1024 * 1024 * 1024,
            tensor_operations: crate::TensorOperations::Absent,
        },
    )
    .unwrap();
    let (_, _, upload_regions, _) = crate::programs::native_target_graph::checked_target_family(
        backend,
        &load,
        &definition.decoder,
        &state,
        plan.target(),
        limits,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    // A step's runs that hold an upload region: its entry, the launch's
    // controls, which every block binds (the mixers' and the n-gram
    // input's tables), and block 0's front, which writes its sequence.
    assert_eq!(upload_regions.len(), 3);
    }
}

/// A recorded discrete Vulkan device (the RTX 3090's reservation facts,
/// `seismic_vulkan::reservation_form`) reads files into its host tier,
/// whatever its device reservation says: the plan's carrier is the
/// tier's, in a reservation the host addresses, so a conveyor in memory
/// the host does not read into has no representation.
#[test]
fn a_discrete_vulkan_conveyor_lies_in_its_host_tier() {
    use crate::planning::fixtures::{qwen4exp_fixture_definition, streamed_experts_manifest};
    let topology = seismic::fixtures::recorded::rtx3090();
    let executor = topology.executors().iter().find(|executor| executor.backend() == seismic::BackendName::Vulkan).unwrap();
    let reservation = executor.reserving(seismic::ReservedContract::DeviceOnly).unwrap().unwrap();
    let host = crate::platform::host_tier(&topology, executor).unwrap().unwrap();
    let HostTier::Linked(tier) = host else {
        panic!("a discrete device's tier is linked");
    };
    assert!(tier.host_addressed);
    let definition = qwen4exp_fixture_definition();
    let manifest = streamed_experts_manifest(&definition);
    let selection = crate::ComponentSelection { head: false, vision: false };
    let resident = ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows32, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS).unwrap();
    let load = resident.with_expert_bank_at_floor(&definition, reservation, host).unwrap();
    let bank = load.expert_bank().unwrap();
    assert!(matches!(bank.conveyor().carrier, ConveyorCarrier::HostTier(_)));
    assert!(bank.layout().unwrap().host.is_some());
}

/// The conveyor lies in the bottom records of the group of the most
/// experts, two rounds' records of each staging pool laid as its
/// lender's records pack them, in the store the host reads files into:
/// on unified memory the bank's own, its lent records the scratch's too;
/// beside a discrete device the host tier's, so the bank's least records
/// count its lent scratch alone. No allocation of its own exists.
#[test]
fn the_conveyor_lies_in_the_store_the_host_reads_into() {
    let discrete = bank(&[37, 3], 64);
    let unified = beside(discrete.clone(), HostTier::Unified);
    // Gate and up have one stored form, down two: Q4_K then Q6_K.
    let q4 = tensor("gguf_q4_k", "q4k").stored_bytes().unwrap();
    let q6 = tensor("gguf_q6_k", "q6k").stored_bytes().unwrap();
    // Beside a host tier a whole layer; on the one store two rounds.
    assert_eq!((discrete.conveyor_records(), unified.conveyor_records()), (256, 2 * unified.round_records()));
    for plan in [&discrete, &unified] {
        let records = plan.conveyor_records();
        let conveyor = plan.conveyor();
        assert_eq!(conveyor.carrier.lender().group, 0, "the group of the most experts lends");
        // Each tensor's conveyor in the next tensor's records: gate's in
        // up's, up's in down's, down's in gate's.
        assert_eq!(
            conveyor.conveyor.bytes,
            RecordTensors { gate: Some(records * (q4 + q6)), up: records * q4, down: records * q4 }
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
}

/// The bank's bytes are its layout's, each allocation as charged: every
/// word part of every layer, group and the load at its graph ports'
/// shape, and the staging pools.
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
    }
    assert_eq!(layout.load.lists.len() as u64, plan.rounds());
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
        .map(|allocation| allocation.charged_bytes(&RESERVATION).unwrap())
        .sum::<u64>();
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
    let without = beside(bank(&[3, 1], 26), HostTier::Unified);
    let with = beside(without.clone(), HostTier::Linked(BYTES));
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
    assert_eq!(plan.staging_records(), 8 + 2 * 8);
    assert_eq!(plan.round_records(), 32);
    assert_eq!(plan.rounds(), 8);
    let table = plan.binding(0).unwrap().words.scratch();
    assert_eq!(table.lanes(), constants::SCRATCH_ROUND + 8);
    assert_eq!(table.width(), constants::SCRATCH_SLOT);
    assert_eq!(table.lane(1, constants::SCRATCH_ROUND + 7), (table.lanes() + constants::SCRATCH_ROUND + 7) * table.width());
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
