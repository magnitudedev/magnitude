//! The expert bank's size in one store, derived from the store's ledger.
//!
//! The bank is cache: it holds whatever room no committed claim, no other
//! cache and the speed guarantee hold, between its floor
//! and the slot count past which the measured miss curve saves nothing.
//! [`bank_room`] and [`shared_targets`] are that arithmetic on one
//! [`Snapshot`], recomputed at every reading; nothing of it is stored, so a reading after memory frees
//! derives the larger bank whatever came before. [`bank_move`] turns a
//! target and the slots held into the one resize a store may run.

use crate::domain::BankHolding;
use crate::memory::ledger::{CacheKind, DeviceLedgers, Snapshot};
use crate::planning::BankTier;

/// Facts of one store's bank for one load: its floor and ceiling from the
/// plan, the granule pool's block and the miss curve's saturation. None of
/// them is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankShape {
    /// The granule pool's block: the least growth worth a resize.
    block_bytes: u64,
    /// The slots one decode step holds at once, held as committed memory
    /// (`CommittedKind::BankFloor`), never cache.
    floor_slots: u64,
    /// The most slots the bank may hold: the plan's ceiling, or fewer where
    /// the measured miss curve stops falling sooner, as a slot past it
    /// saves nothing and so takes no room.
    saturation_slots: u64,
}

impl BankShape {
    /// The shape of a bank of `floor` to `ceiling` slots (the plan's, its
    /// structure's most), growing by `block_bytes`, whose misses stop
    /// falling at `curve` slots: it holds at most the lesser of the two.
    ///
    /// # Panics
    /// Where the floor lies past the ceiling: a plan that is no bank.
    pub fn new(block_bytes: u64, floor: u64, ceiling: u64, curve: u64) -> Self {
        assert!(floor <= ceiling, "invariant violated: a bank floor of {floor} slots past its ceiling of {ceiling}");
        Self { block_bytes, floor_slots: floor, saturation_slots: ceiling.min(curve).max(floor) }
    }

    pub fn floor_slots(&self) -> u64 {
        self.floor_slots
    }

    pub fn saturation_slots(&self) -> u64 {
        self.saturation_slots
    }
}

/// One step's working set of a bank, from its header's counts and the
/// prompt's scratch the ledger grants: what a pooled bank's floor holds,
/// once for the whole bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkingSet {
    /// `K`: the experts a row selects.
    selected: u64,
    /// A decode step's parked layer: `min(DECODE_ROWS · K, E)`, `E` the
    /// widest layer's experts. The device runs in layer order, so one park
    /// holds records at once, whichever layer.
    decode: u64,
    /// The prompt's scratch records a prompt group converts its rounds
    /// through: never fewer than the decode set.
    lent: u64,
    /// The staging pools: one per routed tensor's stored form, each with
    /// its own hint window.
    staging_forms: u64,
}

impl WorkingSet {
    /// The working set of a bank whose rows select `selected` of at most
    /// `widest` experts a layer, in `staging_forms` stored forms, its
    /// prompt's scratch `lent` records, raised to the decode set where
    /// fewer: a prompt converts a layer in as many rounds as its scratch
    /// takes, and the floor holds at least what a decode step needs.
    pub fn new(selected: u64, widest: u64, lent: u64, staging_forms: u64) -> Self {
        let decode = (crate::operators::routed::fused_graph::DECODE_ROWS * selected).min(widest);
        Self { selected, decode, lent: lent.max(decode), staging_forms }
    }

    /// The prompt's scratch records, at least the decode set.
    pub fn lent(&self) -> u64 {
        self.lent
    }

    /// The records the bank holds at its floor: the prompt's scratch, which
    /// holds the decode step's parked layer too, beside every staging
    /// pool's read-ahead landings (its hint window, `2K`: a read ahead holds
    /// a staging record from its read to its landing and waits for one past
    /// them, so a pool's window bounds its landings in flight however many
    /// leads its routes predict).
    pub fn floor_records(&self) -> u64 {
        self.lent + 2 * self.selected * self.staging_forms
    }
}

/// The most slots in `[from, to]` whose charge beyond `from`'s fits `room`,
/// `charge` the bank's charge at a slot count (its resident charge, which
/// rises in steps where a slot crosses its storage's granule, and never
/// falls): what a load's plan sizes its bank by beside what it commits,
/// and what the bank's target is at every reading.
///
/// # Panics
/// Where `from` lies past `to`. A charge that fails fails the search.
pub fn slots_within<E>(room: u64, (from, to): (u64, u64), charge: &dyn Fn(u64) -> Result<u64, E>) -> Result<u64, E> {
    assert!(from <= to, "invariant violated: slots from {from} past {to}");
    let admitted = charge(from)?.saturating_add(room);
    // The charge never falls with more slots: the last count within it.
    let (mut fits, mut over) = (from, to + 1);
    while over - fits > 1 {
        let middle = fits + (over - fits) / 2;
        match charge(middle)? <= admitted {
            true => fits = middle,
            false => over = middle,
        }
    }
    Ok(fits)
}

/// The room the bank's cache may hold in the store of `snapshot`: the
/// budget less committed memory (the bank's floor among it), every other
/// cache and the speed guarantee (`in_process`, one prompt group's
/// mandatory growth, and others' rise between the last two readings). The
/// bank's own cache and the pool's spare granules are the room it derives;
/// every other cache stays where it is. Blind, or committed memory filling
/// the budget, leaves none.
pub fn bank_room(snapshot: &Snapshot, in_process: u64) -> u64 {
    let others = snapshot.cache() - snapshot.cache_of(CacheKind::Bank) - snapshot.cache_of(CacheKind::Spare);
    let taken = snapshot
        .committed()
        .saturating_add(others)
        .saturating_add(snapshot.guarantee(in_process));
    snapshot.budget().saturating_sub(taken)
}

/// The slots a holding of `shape` holds within `room` bytes above its
/// floor, as `charge` holds slots ([`slots_within`]), within
/// `[floor, saturation]`: its floor where the room holds no step.
pub fn target_slots(room: u64, shape: &BankShape, charge: &dyn Fn(u64) -> u64) -> u64 {
    slots_within(room, (shape.floor_slots, shape.saturation_slots), &|slots| Ok::<_, std::convert::Infallible>(charge(slots)))
        .unwrap_or_else(|never| match never {})
}

/// The state the largest batch a load of `execution` admits grows by at
/// once, its arena's `places` prompt-group places each a launch of rows: the
/// pages the ledger sizes state rows by (`LoadShape::state_bytes`), and
/// the slab the stores add in each stored domain to hold them, which
/// they grow by whole: the in-process term of the speed guarantee
/// (`memory::ledger::Snapshot::guarantee`) every derived target and a load's
/// import leave, so a growth the bank has left room for is never refused.
pub fn in_process_growth(execution: &crate::ExecutionPlan, places: std::num::NonZeroUsize) -> u64 {
    let launch = execution.policy().limits().max_launch_rows as u64;
    let rows = launch * places.get() as u64;
    let resources = execution.resources();
    let slabs = std::iter::once(resources.target_state())
        .chain(resources.head_state())
        .map(|state| state.history_slab_bytes().expect("invariant violated: a load's history domains lay out their slabs"))
        .sum::<u64>();
    crate::memory::ledger::LoadShape::of(execution).state_bytes(rows) + slabs
}

/// The experts' slots a load imports its bank `plan` at, `floor` the
/// import's weights with the bank at its floor: [`import_target`] in the
/// bank's store, of what those weights place there beside every claim the
/// store's ledger holds (the allocations after the import claimed before
/// it), held at most at the plan's slots.
pub fn import_slots(
    ledgers: &DeviceLedgers,
    device: &seismic::Device,
    plan: &crate::planning::ExpertBankPlan,
    floor: crate::memory::ClaimDemand,
    in_process: u64,
) -> u64 {
    let shape = BankShape::new(plan.reservation().granule, plan.floor(), plan.ceiling, plan.slots);
    let charge = |slots| plan.resident_charge(slots).expect("invariant violated: a bank plan charges every size within its ceiling");
    let beside = ledgers.placed_in(device, floor, BankTier::Bank);
    import_target(&ledgers.of(BankTier::Bank).snapshot(), in_process, beside, &shape, &charge)
}

/// The experts' target a load's import holds, `beside` the bytes the
/// import (its bank at its floor) places in the bank's store: the target as a reading of the store derives it once
/// they are held ([`bank_room`] beside the guarantee of `in_process`, held
/// by [`target_slots`]), so a load imports what its first reading keeps.
pub fn import_target(snapshot: &Snapshot, in_process: u64, beside: u64, shape: &BankShape, charge: &dyn Fn(u64) -> u64) -> u64 {
    target_slots(bank_room(snapshot, in_process).saturating_sub(beside), shape, charge)
}

/// The targets of one store's bank holdings sharing its `room`
/// ([`bank_room`]), in the order given: each holds what its slots fit of
/// the room those before it leave ([`target_slots`]).
pub fn shared_targets(room: u64, holdings: &[(BankShape, &dyn Fn(u64) -> u64)]) -> Vec<u64> {
    let mut left = room;
    holdings
        .iter()
        .map(|(shape, charge)| {
            let target = target_slots(left, shape, *charge);
            left -= charge(target) - charge(shape.floor_slots);
            target
        })
        .collect()
}

/// A bank holding as its derived target reads it: the tier whose store it
/// lies in, its size now, its shape and its charge at any size
/// (`ElasticBank::targets`).
pub struct HoldingTarget<'a> {
    pub holding: BankHolding,
    pub tier: BankTier,
    pub at: u64,
    pub shape: BankShape,
    pub charge: Box<dyn Fn(u64) -> u64 + 'a>,
}

/// Each of `targets` with its derived target, read from `ledgers` now: in
/// each store, the room its snapshot leaves the bank's cache beside the
/// guarantee's `in_process` growth ([`bank_room`]), shared by the store's
/// holdings in `targets`' order ([`shared_targets`]). The one derivation
/// that following the targets and reporting them read.
pub fn derive<'t>(ledgers: &DeviceLedgers, targets: &'t [HoldingTarget<'t>], in_process: u64) -> Vec<(&'t HoldingTarget<'t>, u64)> {
    let mut stores: Vec<seismic::StoreIdentity> = Vec::new();
    for target in targets {
        let store = ledgers.of(target.tier).store();
        if !stores.contains(&store) {
            stores.push(store);
        }
    }
    stores
        .into_iter()
        .flat_map(|store| {
            let members: Vec<&'t HoldingTarget<'t>> = targets.iter().filter(|target| ledgers.of(target.tier).store() == store).collect();
            let room = bank_room(&ledgers.of(members[0].tier).snapshot(), in_process);
            let shapes: Vec<(BankShape, &dyn Fn(u64) -> u64)> = members.iter().map(|target| (target.shape, &*target.charge)).collect();
            let slots = shared_targets(room, &shapes);
            members.into_iter().zip(slots).collect::<Vec<_>>()
        })
        .collect()
}

/// The one resize toward `derived` targets ([`derive`]): a shrink before
/// any growth; none where each holding holds its target.
pub fn next_move(derived: &[(&HoldingTarget<'_>, u64)]) -> Option<(BankHolding, BankMove)> {
    let moves: Vec<(BankHolding, BankMove)> = derived
        .iter()
        .filter_map(|(target, slots)| bank_move(*slots, target.at, &target.shape, &*target.charge).map(|resize| (target.holding, resize)))
        .collect();
    let shrink = moves.iter().copied().find(|(_, resize)| matches!(resize, BankMove::Shrink { .. }));
    shrink.or_else(|| moves.first().copied())
}

/// A bank holding's derived target in its store, beside what it holds now,
/// in its own units (slots, or a table's records).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedTarget {
    pub holding: crate::BankHolding,
    pub store: seismic::StoreIdentity,
    pub held: u64,
    pub target: u64,
}

/// The one resize a holding runs toward its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankMove {
    Grow { to: u64 },
    Shrink { to: u64 },
}

/// The resize toward `target` from `held` slots: a shrink whenever the
/// target is below what the holding holds; a growth only once the target's
/// charge (`charge`, as [`target_slots`] reads it) is a block or more above
/// what it holds, as a growth is made of blocks; otherwise none.
pub fn bank_move(target: u64, held: u64, shape: &BankShape, charge: &dyn Fn(u64) -> u64) -> Option<BankMove> {
    if target < held {
        return Some(BankMove::Shrink { to: target });
    }
    (charge(target).saturating_sub(charge(held)) >= shape.block_bytes.max(1)).then_some(BankMove::Grow { to: target })
}

#[cfg(test)]
mod tests;
