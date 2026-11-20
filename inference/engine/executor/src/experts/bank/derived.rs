//! The expert bank's size in one store, derived from the store's ledger.
//!
//! The bank is cache: it holds whatever room no committed claim, no other
//! cache and the speed guarantee hold, between its floor
//! and the slot count past which the measured miss curve saves nothing.
//! The ledger's estimate gives that room (`Estimate::target`), and
//! [`shared_targets`] divides it among a store's holdings, recomputed at
//! every reading; nothing of it is stored, so a reading after memory frees
//! derives the larger bank whatever came before. The block is the one
//! threshold, both ways ([`held_target`]): the bank grows once the room
//! holds a block more than it holds, and shrinks once it holds a block more
//! than the room, so a move of the room within a block (others' jitter)
//! resizes nothing. [`bank_move`] turns a target and the slots held into
//! the one resize a store may run.

use crate::domain::BankHolding;
use crate::memory::ledger::{DeviceLedgers, Estimate};
use crate::planning::BankTier;

/// Facts of one store's bank for one load: its floor and ceiling from the
/// plan, the granule pool's block and the miss curve's saturation. None of
/// them is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankShape {
    /// The granule pool's block: the least resize either way.
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

    /// The shape of a holding of `floor` to `ceiling` slots holding `at`,
    /// growing by `block_bytes`, priced by a miss curve that found its
    /// deepest reuse `curve` slots deep (none before it found one, which
    /// bounds nothing): [`Self::new`], the curve's saturation never below
    /// `at`. The curve counts every step since the load, so its deepest
    /// reuse only deepens; a slot past it is one no step has shown useful
    /// yet, not one shown useless, so it bounds a growth and never shrinks
    /// what the holding holds (Gemma 4 26B on an M4 Pro: 78 slots deep
    /// after a dozen steps, 123 seconds later).
    pub fn curved(block_bytes: u64, floor: u64, ceiling: u64, curve: Option<u64>, at: u64) -> Self {
        Self::new(block_bytes, floor, ceiling, curve.map_or(ceiling, |curve| curve.max(at)))
    }

    /// The granule pool's block: the least resize either way, so a target
    /// moves what the holding holds only once it lies a block from it
    /// ([`held_target`]).
    pub fn block_bytes(&self) -> u64 {
        self.block_bytes
    }

    /// The shape of a host tier of at most `ceiling` host slots, growing by
    /// `block_bytes`, bounded by `bound`, beside a device bank that is
    /// `whole` ([`crate::planning::ExpertBankPlan::whole_at`]). A host tier
    /// serves only the experts the device lacks: while the device holds
    /// every expert it holds none, however much host RAM is free.
    pub fn host(block_bytes: u64, ceiling: u64, bound: u64, whole: bool) -> Self {
        let bound = match whole {
            true => 0,
            false => bound,
        };
        Self::new(block_bytes, 0, ceiling, bound)
    }

    pub fn floor_slots(&self) -> u64 {
        self.floor_slots
    }

    pub fn saturation_slots(&self) -> u64 {
        self.saturation_slots
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

/// The slots a holding of `shape` holds within `room` bytes above its
/// floor, as `charge` holds slots ([`slots_within`]), within
/// `[floor, saturation]`: its floor where the room holds no step.
pub fn target_slots(room: u64, shape: &BankShape, charge: &dyn Fn(u64) -> u64) -> u64 {
    slots_within(room, (shape.floor_slots, shape.saturation_slots), &|slots| Ok::<_, std::convert::Infallible>(charge(slots)))
        .unwrap_or_else(|never| match never {})
}

/// The experts' slots a load imports its bank `plan` at, `floor` the
/// import's weights with the bank at its floor: [`import_target`] in the
/// bank's store at the load's estimate `at_load`
/// (`memory::ledger::DeviceReading::room_after`), of what those weights
/// place there beside every claim the store's ledger holds (the
/// allocations after the import claimed before it, the load's request slot
/// among them), held at most at the plan's slots.
pub fn import_slots(
    at_load: &crate::memory::ledger::LoadEstimate,
    ledgers: &DeviceLedgers,
    device: &seismic::Device,
    plan: &crate::planning::ExpertBankPlan,
    floor: crate::memory::ClaimDemand,
) -> u64 {
    let shape = BankShape::new(plan.granule(), plan.floor(), plan.ceiling, plan.slots);
    let charge = |slots| plan.resident_charge(slots).expect("invariant violated: a bank plan charges every size within its ceiling");
    let beside = ledgers.placed_in(device, floor, BankTier::Bank);
    import_target(at_load.of(BankTier::Bank), beside, &shape, &charge)
}

/// The experts' target a load's import holds, `beside` the bytes the
/// import (its bank at its floor) places in the bank's store: the target as a reading of the store derives it once
/// they are held (the estimate's bank room, `Estimate::import_room`, held by
/// [`target_slots`]), so a load imports what its first reading keeps.
pub fn import_target(estimate: &Estimate, beside: u64, shape: &BankShape, charge: &dyn Fn(u64) -> u64) -> u64 {
    target_slots(estimate.import_room().saturating_sub(beside), shape, charge)
}

/// The slots a holding of `shape` holding `at` slots keeps within `room`
/// bytes above its floor, owing claims `owed` bytes, as `charge` holds
/// slots: the block is the one threshold both ways for the room's own
/// movement. It grows to what the room fits ([`target_slots`])
/// once that is a block or more above what it holds; it shrinks to what
/// the room fits once what it holds lies a block or more past the room, or
/// past its saturation; otherwise it keeps what it holds.
///
/// `owed` is cache claims of the process took that its slots still hold
/// (`BankCache::owed`): a fact of the ledger, which those claims allocate
/// only once the slots free it, never the room's movement. So whatever the
/// threshold says, a holding owing claims sheds at least what it owes, in
/// whole blocks rounded up, and grows by none. Cache a shortage took to
/// fit the budget as others moved owes nothing: the threshold judges it as
/// it judges the room.
pub fn held_target(room: u64, at: u64, owed: u64, shape: &BankShape, charge: &dyn Fn(u64) -> u64) -> u64 {
    let fits = target_slots(room, shape, charge);
    let block = shape.block_bytes.max(1);
    let above_floor = charge(at).saturating_sub(charge(shape.floor_slots));
    let keeps = match fits.cmp(&at) {
        std::cmp::Ordering::Equal => true,
        std::cmp::Ordering::Greater => charge(fits) - charge(at) < block,
        std::cmp::Ordering::Less => at <= shape.saturation_slots && above_floor.saturating_sub(room) < block,
    };
    let held = match keeps {
        true => at,
        false => fits,
    };
    if owed == 0 {
        return held;
    }
    let left = above_floor.saturating_sub(owed.div_ceil(block).saturating_mul(block));
    let sheds = slots_within(left, (shape.floor_slots, at), &|slots| Ok::<_, std::convert::Infallible>(charge(slots)))
        .unwrap_or_else(|never| match never {});
    held.min(sheds)
}

/// The targets of one store's bank holdings sharing its `room`
/// (`Estimate::target`), each with the slots it holds now and what it owes
/// claims, in the order given: each keeps what [`held_target`] keeps of the
/// room those before it leave.
pub fn shared_targets(room: u64, holdings: &[(BankShape, u64, u64, &dyn Fn(u64) -> u64)]) -> Vec<u64> {
    let mut left = room;
    holdings
        .iter()
        .map(|(shape, at, owed, charge)| {
            let target = held_target(left, *at, *owed, shape, *charge);
            left = left.saturating_sub(charge(target) - charge(shape.floor_slots));
            target
        })
        .collect()
}

/// A bank holding as its derived target reads it: the tier whose store it
/// lies in, its size now, what it owes claims that took its cache
/// (`BankCache::owed`), its shape and its charge at any size
/// (`ElasticBank::targets`).
pub struct HoldingTarget<'a> {
    pub holding: BankHolding,
    pub tier: BankTier,
    pub at: u64,
    pub owed: u64,
    pub shape: BankShape,
    pub charge: Box<dyn Fn(u64) -> u64 + 'a>,
}

/// Each of `targets` with its derived target, read from `ledgers` now: in
/// each store, the room its estimate leaves the bank's cache
/// (`Estimate::target`), shared by the store's holdings in `targets'`
/// order ([`shared_targets`]). The one derivation that following the
/// targets and reporting them read.
pub fn derive<'t>(ledgers: &DeviceLedgers, targets: &'t [HoldingTarget<'t>]) -> Vec<(&'t HoldingTarget<'t>, u64)> {
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
            let room = ledgers.of(members[0].tier).estimate().target();
            let shapes: Vec<(BankShape, u64, u64, &dyn Fn(u64) -> u64)> =
                members.iter().map(|target| (target.shape, target.at, target.owed, &*target.charge)).collect();
            let slots = shared_targets(room, &shapes);
            members.into_iter().zip(slots).collect::<Vec<_>>()
        })
        .collect()
}

/// The one resize toward `derived` targets ([`derive`]): a shrink before
/// any growth; none where each holding holds its target.
pub fn next_move(derived: &[(&HoldingTarget<'_>, u64)]) -> Option<(BankHolding, BankMove)> {
    let moves: Vec<(BankHolding, BankMove)> =
        derived.iter().filter_map(|(target, slots)| bank_move(*slots, target.at).map(|resize| (target.holding, resize))).collect();
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
    /// Its pool's block ([`BankShape::block_bytes`]): the least it shrinks by.
    pub block: u64,
}

/// The one resize a holding runs toward its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankMove {
    Grow { to: u64 },
    Shrink { to: u64 },
}

/// The resize toward `target` from `held` slots, the target a block or
/// more away wherever it differs ([`held_target`]).
pub fn bank_move(target: u64, held: u64) -> Option<BankMove> {
    match target.cmp(&held) {
        std::cmp::Ordering::Less => Some(BankMove::Shrink { to: target }),
        std::cmp::Ordering::Equal => None,
        std::cmp::Ordering::Greater => Some(BankMove::Grow { to: target }),
    }
}

#[cfg(test)]
mod tests;
