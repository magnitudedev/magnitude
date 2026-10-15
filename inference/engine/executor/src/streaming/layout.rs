//! The device tensors of a streamed load (spec 4.3): the step counter, every
//! routed layer's directory (slot map, stamps, request region, hint rings,
//! convert lists) and the staging pools, as the graphs' bank ports declare
//! them and as the load allocates them.

use crate::ModelLoadPlan;
use crate::planning::{BANK_KINDS, ExpertBankPlan};
use crate::programs::native_target_graph::BankPart;
use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};
use crate::operators::routed::fused_graph::DECODE_ROWS;
use seismic::{Device, Element, Tensor};

/// Prediction distances (spec 4.7): a routed layer hints the layers one and
/// two ahead.
pub(crate) const DISTANCES: [u8; 2] = [1, 2];

/// Words of a layer's request region: the request, acknowledgement,
/// failure and count words, then every choice of the decode form's rows
/// (`routed_route_streamed`).
pub(crate) fn control_words(selected: u64) -> u64 {
    4 + DECODE_ROWS * selected
}

/// Words of a hint ring: the sequence and count, then every choice of the
/// decode form's rows (`routed_route_streamed`).
pub(crate) fn ring_words(selected: u64) -> u64 {
    2 + DECODE_ROWS * selected
}

/// Staging records per pool (spec 4.3): the prefetch window, two layers of
/// one row's choices, beside every demand read of the widest bank-served
/// launch (the decode form's rows).
pub(crate) fn staging_records(selected: u64) -> u64 {
    prefetch_records(selected) + DECODE_ROWS * selected
}

/// Staging records prefetches may hold (spec 4.3, 4.7): one row's predicted
/// experts for each distance ahead.
pub(crate) fn prefetch_records(selected: u64) -> u64 {
    DISTANCES.len() as u64 * selected
}

/// Rounds a scratch-served layer's graph holds (spec 4.9): each stages half
/// the pool, so one half fills while the device converts the other.
pub(crate) fn scratch_rounds(experts: u64, selected: u64) -> u64 {
    experts.div_ceil((staging_records(selected) / 2).max(1))
}

/// Words of a load's scratch request region (`routed_scratch_request`):
/// the request, layer, failure and a reserved word, then each round's
/// acknowledgement and each round's arrival (`routed_scratch_wait`).
pub(crate) fn scratch_control_words(rounds: u64) -> u64 {
    4 + 2 * rounds
}

fn bank(load: &ModelLoadPlan) -> Result<&ExpertBankPlan, String> {
    load.expert_bank()
        .ok_or_else(|| "a bank port of a load that does not stream".to_owned())
}

fn layer_tensor(
    bank: &ExpertBankPlan,
    scope: WeightScope,
    kind: u8,
) -> Result<crate::planning::BankTensor, String> {
    let layer = bank
        .layers
        .iter()
        .find(|layer| layer.scope == scope)
        .ok_or_else(|| format!("{scope:?} is not a streamed routed layer"))?;
    layer
        .tensors
        .get(usize::from(kind))
        .copied()
        .ok_or_else(|| format!("routed tensor kind {kind} of {BANK_KINDS:?}"))
}

/// The element and extents of `part` of the routed layer of `scope`.
pub(crate) fn bank_port(
    load: &ModelLoadPlan,
    scope: WeightScope,
    part: BankPart,
) -> Result<(Element, Vec<u64>), String> {
    let bank = bank(load)?;
    let words = |count: u64| Ok((Element::i32(), vec![count]));
    match part {
        BankPart::Step | BankPart::Heartbeat => words(1),
        BankPart::SlotMap => words(bank.experts),
        BankPart::Stamps => words(bank.records(scope)?),
        BankPart::Commits => words(bank.records(scope)?),
        BankPart::Control => words(bank.control_words()),
        BankPart::Placement => Ok((Element::i32(), vec![5, bank.placement_choices()])),
        BankPart::Ring { .. } => words(bank.ring_words()),
        BankPart::Staging { kind } => {
            let pool = &bank.pools[layer_tensor(bank, scope, kind)?.pool];
            Ok((
                pool.source,
                vec![bank.staging_records, pool.matrix[0], pool.matrix[1]],
            ))
        }
        BankPart::Scratch { kind } => {
            let scratch = &bank.scratch[layer_tensor(bank, scope, kind)?.scratch];
            Ok((
                scratch.element,
                vec![bank.experts, scratch.matrix[0], scratch.matrix[1]],
            ))
        }
        BankPart::ScratchControl => words(bank.scratch_control_words()),
        BankPart::ScratchFlags => words(bank.experts),
        BankPart::ScratchList { .. } => words(bank.list_words()),
        BankPart::CopyList { .. } => words(bank.copy_list_words()),
    }
}

/// What a streamed block's graph depends on beyond its own weights: whether
/// it advances the step (the first streamed layer), and each layer it hints
/// (spec 4.7) with its distance in blocks and the planned norm and router it
/// predicts through. Blocks that agree share a graph.
pub(crate) fn block_key(load: &ModelLoadPlan, block: usize) -> Result<Option<String>, String> {
    let Some(bank) = load.expert_bank() else {
        return Ok(None);
    };
    let block_of = |scope: WeightScope| match scope {
        WeightScope::TargetSublayer(index) => Ok(index.block as usize),
        _ => Err(format!("streamed layer {scope:?} outside the target")),
    };
    let mut position = None;
    for (index, layer) in bank.layers.iter().enumerate() {
        if block_of(layer.scope)? == block {
            position = Some(index);
        }
    }
    let Some(position) = position else {
        return Ok(None);
    };
    let mut ahead = Vec::new();
    for distance in DISTANCES {
        let Some(layer) = bank.layers.get(position + usize::from(distance)) else {
            break;
        };
        let planned = |kind| {
            load.weights()
                .find(|plan| plan.role == WeightRole { scope: layer.scope, kind })
                .map(|plan| (plan.resident, plan.shape.clone()))
                .ok_or_else(|| format!("{:?} of {:?} is not planned", kind, layer.scope))
        };
        ahead.push((
            block_of(layer.scope)? - block,
            planned(WeightKind::InputNorm)?,
            planned(WeightKind::Router)?,
            planned(WeightKind::SharedRouter)?,
        ));
    }
    Ok(Some(format!("{:?}", (position == 0, ahead))))
}

/// One routed layer's directory tensors.
pub(crate) struct LayerTensors {
    pub scope: WeightScope,
    pub slot_map: Tensor,
    /// The layer's bank group's stamps and commits, shared by its layers.
    pub stamps: Tensor,
    pub commits: Tensor,
    pub control: Tensor,
    /// [5, P]: per choice, the slot, the staging records and the
    /// acknowledgement of an absent expert (`routed_bank_fill`).
    pub placement: Tensor,
    /// This layer's hints for the layers one and two ahead.
    pub rings: [Tensor; 2],
    /// The layer's scratch request region.
    pub scratch_control: Tensor,
    pub pools: [usize; 3],
    pub scratch: [usize; 3],
}

/// A streamed load's device tensors, standalone allocations the host
/// addresses while steps run (`Tensor::host_window`).
pub(crate) struct BankTensors {
    pub step: Tensor,
    pub heartbeat: Tensor,
    pub layers: Vec<LayerTensors>,
    pub pools: Vec<Tensor>,
    /// The scratch fill's: one bank per resident form, the request region,
    /// each round's convert lists and the copy lists (spec 4.9).
    pub scratch: Vec<Tensor>,
    pub scratch_flags: Tensor,
    pub scratch_lists: Vec<[Tensor; 3]>,
    pub copy_lists: [Tensor; 3],
}

impl BankTensors {
    /// Allocate the directory of a streamed load over its shared bank plan.
    /// All fallible allocations remain private until this complete value
    /// returns; the construction peak therefore equals its persistent device
    /// charge.
    pub(crate) fn allocate(device: &Device, bank: &ExpertBankPlan,
        shared: &super::shared_bank::SharedBankPlan) -> Result<Self, String> {
        if shared != &super::shared_bank::SharedBankPlan::admitted(bank)? {
            return Err("shared directory plan differs from bank metadata".into());
        }
        let words = |count: u64| {
            Tensor::zeros(device, Element::i32(), &[count])
                .map_err(|error| format!("bank directory allocation failed: {error}"))
        };
        let group_words = shared.groups.iter()
            .map(|group| Ok((words(group.records)?, words(group.records)?)))
            .collect::<Result<Vec<_>, String>>()?;
        let layers = bank
            .layers
            .iter()
            .enumerate()
            .map(|(position, layer)| {
                let (stamps, commits) = group_words[shared.group_of_layer[position]].clone();
                Ok(LayerTensors {
                    scope: layer.scope,
                    slot_map: words(bank.experts)?,
                    stamps,
                    commits,
                    control: words(bank.control_words())?,
                    placement: Tensor::zeros(device, Element::i32(), &[5, bank.placement_choices()])
                        .map_err(|error| format!("bank directory allocation failed: {error}"))?,
                    rings: [words(bank.ring_words())?, words(bank.ring_words())?],
                    scratch_control: words(bank.scratch_control_words())?,
                    pools: layer.tensors.map(|tensor| tensor.pool),
                    scratch: layer.tensors.map(|tensor| tensor.scratch),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let pools = bank
            .pools
            .iter()
            .map(|pool| {
                Tensor::zeros(
                    device,
                    pool.source,
                    &[bank.staging_records, pool.matrix[0], pool.matrix[1]],
                )
                .map_err(|error| format!("staging allocation failed: {error}"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        // One arena per routed tensor, sized for its largest form; each
        // form is a view of it at offset 0 (one layer fills it at a time).
        let arenas = (0..3)
            .map(|kind| {
                Tensor::zeros(device, Element::bool(), &[bank.scratch_arena_bytes(kind)?])
                    .map_err(|error| format!("scratch arena allocation failed: {error}"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let scratch = bank
            .scratch
            .iter()
            .map(|scratch| {
                arenas[scratch.kind]
                    .region(
                        scratch.element,
                        &[bank.experts, scratch.matrix[0], scratch.matrix[1]],
                        0,
                    )
                    .map_err(|error| format!("scratch bank view failed: {error}"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let lists = || -> Result<[Tensor; 3], String> {
            Ok([
                words(bank.list_words())?,
                words(bank.list_words())?,
                words(bank.list_words())?,
            ])
        };
        let scratch_lists = (0..bank.rounds())
            .map(|_| lists())
            .collect::<Result<Vec<_>, String>>()?;
        let copy_words = bank.copy_list_words();
        Ok(Self {
            step: words(1)?,
            heartbeat: words(1)?,
            layers,
            pools,
            scratch,
            scratch_flags: words(bank.experts)?,
            scratch_lists,
            copy_lists: [
                words(copy_words)?,
                words(copy_words)?,
                words(copy_words)?,
            ],
        })
    }

    /// The tensor `part` of the routed layer of `scope` binds.
    pub(crate) fn part(&self, scope: WeightScope, part: BankPart) -> Result<&Tensor, String> {
        let layer = self
            .layers
            .iter()
            .find(|layer| layer.scope == scope)
            .ok_or_else(|| format!("{scope:?} is not a streamed routed layer"))?;
        let index = |value: u8, bound: usize| {
            let value = usize::from(value);
            (value < bound)
                .then_some(value)
                .ok_or_else(|| format!("bank part {part:?} out of range"))
        };
        Ok(match part {
            BankPart::Step => &self.step,
            BankPart::Heartbeat => &self.heartbeat,
            BankPart::SlotMap => &layer.slot_map,
            BankPart::Stamps => &layer.stamps,
            BankPart::Commits => &layer.commits,
            BankPart::Control => &layer.control,
            BankPart::Placement => &layer.placement,
            BankPart::Ring { distance } => {
                &layer.rings[index(distance.wrapping_sub(1), 2)?]
            }
            BankPart::Staging { kind } => &self.pools[layer.pools[index(kind, 3)?]],
            BankPart::Scratch { kind } => &self.scratch[layer.scratch[index(kind, 3)?]],
            BankPart::ScratchControl => &layer.scratch_control,
            BankPart::ScratchFlags => &self.scratch_flags,
            BankPart::ScratchList { round, kind } => {
                &self.scratch_lists[index(round, self.scratch_lists.len())?][index(kind, 3)?]
            }
            BankPart::CopyList { kind } => &self.copy_lists[index(kind, 3)?],
        })
    }
}
