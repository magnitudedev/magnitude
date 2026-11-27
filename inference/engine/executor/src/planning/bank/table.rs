use crate::error::PlanError;
use crate::planning::weights::{Holding, WeightPlan, representation_bytes};
use magnitude_family_contracts::{ImportTransform, ModelDefinition, Operator, WeightRole, WeightScope};
use std::num::NonZeroU64;

/// A stored table (`TableFill::Stored`, an n-gram table) whose slots the
/// bank holds: `[records, record_rows, width]`, each record one granule of
/// the bank's pool holding the most whole rows it fits, committed from
/// `floor` records up to every one (its plan's ceiling: the fewest records
/// holding the slots a fixed table holds, at most the whole records its
/// rows fill).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankTable {
    pub role: WeightRole,
    /// Additional weight roles that share this exact physical stored table
    /// (deduplicated when backing, subrange, layout, and holding all match).
    pub shared_roles: Vec<WeightRole>,
    /// Its ceiling: more than its floor.
    pub records: NonZeroU64,
    pub record_rows: NonZeroU64,
    /// A record's resident bytes: at most a granule.
    pub record_bytes: NonZeroU64,
    /// The least records it holds: those of the most rows one step names.
    pub floor: u64,
    /// One row's bytes as stored: what a miss reads.
    pub stored_row_bytes: u64,
    /// The table planned fixed, as the load holds it unbanked.
    pub fixed: WeightPlan,
}

impl BankTable {
    /// Every weight role backed by this table: its primary role and any
    /// deduplicated sibling roles.
    pub fn roles(&self) -> impl Iterator<Item = WeightRole> + '_ {
        std::iter::once(self.role).chain(self.shared_roles.iter().copied())
    }

    /// Whether `role` is backed by this table.
    pub fn has_role(&self, role: WeightRole) -> bool {
        self.role == role || self.shared_roles.contains(&role)
    }

    /// Whether `other` addresses the exact same physical stored table:
    /// identical backing (`storage_identity`), subrange/descriptor, layout,
    /// and holding/mutability.
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        self.fixed.storage_identity() == other.fixed.storage_identity()
            && self.fixed.descriptor == other.fixed.descriptor
            && self.fixed.source == other.fixed.source
            && self.fixed.upload == other.fixed.upload
            && self.fixed.shape == other.fixed.shape
            && self.fixed.scale == other.fixed.scale
            && self.fixed.bound == other.fixed.bound
            && self.fixed.holding == other.fixed.holding
            && self.records == other.records
            && self.record_rows == other.record_rows
            && self.record_bytes == other.record_bytes
            && self.stored_row_bytes == other.stored_row_bytes
    }

    /// Merge a deduplicated sibling table sharing this physical stored table,
    /// keeping the maximum floor and recording its role(s).
    pub(super) fn merge_shared(&mut self, other: &Self) {
        debug_assert!(self.shares_storage_with(other));
        self.floor = self.floor.max(other.floor);
        for role in other.roles() {
            if !self.has_role(role) {
                self.shared_roles.push(role);
            }
        }
    }

    /// The table of `fixed`, the stored table of an n-gram input whose
    /// steps name at most `step_rows` rows, held in records of `granule`
    /// bytes, the bank's pool's: none where a row exceeds a granule or its
    /// floor fills its ceiling.
    pub fn of(fixed: &WeightPlan, step_rows: u64, granule: u64) -> Result<Option<Self>, String> {
        let width = fixed.table_width();
        let row = representation_bytes(fixed.resident, &[1, width])?;
        if row == 0 {
            return Err("a stored table's row has no bytes".to_owned());
        }
        // The most whole rows a granule holds.
        let Some(record_rows) = NonZeroU64::new(granule / row) else {
            return Ok(None);
        };
        // No more slots than the table has rows: a record past them could
        // never hold one (`StoredTable::new` refuses it).
        let records = fixed
            .table_slots()
            .div_ceil(record_rows.get())
            .min(fixed.stored_shape()[0] / record_rows);
        let floor = step_rows.div_ceil(record_rows.get());
        // A table whose floor fills its ceiling is not banked: one banked
        // holds more records than its floor, so at least one.
        let Some(records) = NonZeroU64::new(records).filter(|records| records.get() > floor) else {
            return Ok(None);
        };
        Ok(Some(Self {
            role: fixed.role,
            shared_roles: Vec::new(),
            records,
            record_rows,
            record_bytes: NonZeroU64::new(row * record_rows.get()).expect("whole rows of a granule have bytes"),
            floor,
            stored_row_bytes: representation_bytes(fixed.upload, &[1, width])?,
            fixed: fixed.clone(),
        }))
    }

    /// What `committed` of its records are charged in a pool of `granule`
    /// bytes in `block`s: a granule each, in the whole blocks holding them
    /// ([`seismic::BlockSize::charge`]). The pool's charge: what its blocks
    /// cost, the granules no record holds with them.
    pub fn charge(block: seismic::BlockSize, granule: u64, committed: u64) -> Result<u64, String> {
        block.charge(granule, committed).ok_or_else(|| "banked table charge overflows".to_owned())
    }

    /// What `held` of its records hold in a pool of `granule` bytes: a
    /// granule each. The table's own charge, its plan's and its slots'
    /// (`StoredTable::charged_bytes`); the rest of the whole blocks holding
    /// them ([`Self::charge`]) are the pool's spare granules, the bank's
    /// (`ExpertBankPlan::storage_beyond_records`).
    pub fn held_bytes(granule: u64, held: u64) -> Result<u64, String> {
        granule.checked_mul(held).ok_or_else(|| "banked table records' bytes overflow".to_owned())
    }

    /// The table's plan as the bank holds it, every record held in a pool
    /// of `granule` bytes ([`Self::held_bytes`]), beside the table of them
    /// every stored table's slots hold (`WeightPlan::record_table_entries`).
    pub(super) fn plan(&self, granule: u64) -> Result<WeightPlan, String> {
        let mut plan = self.fixed.clone();
        plan.shape = vec![self.records.get(), self.record_rows.get(), self.fixed.table_width()];
        plan.resident_bytes = Self::held_bytes(granule, self.records.get())?;
        let slots = plan.table_slots();
        if let Holding::Streamed(streamed) = &mut plan.holding {
            streamed.directory = super::super::DirectoryForm::of(streamed.table_rows, slots);
        }
        Ok(plan)
    }
}

/// The table rows one position of the n-gram input whose stored table is
/// `role` names (`NGramHash::heads`).
pub(super) fn stored_table_heads(definition: &ModelDefinition, role: WeightRole) -> Result<u64, PlanError> {
    definition
        .decoder
        .sublayers()
        .find_map(|(index, sublayer)| match &sublayer.op {
            Operator::NGramInput(input) if role.scope == WeightScope::TargetSublayer(index) => Some(input),
            _ => None,
        })
        .ok_or(PlanError::Topology("a stored table without its n-gram input"))?
        .hash
        .heads()
        .map_err(|error| PlanError::InvalidDefinition(error.to_string()))
}

/// The first row of each stored expert's matrix an expert stack takes: 0
/// for a stack stored whole, the range's first for a range of each expert's
/// rows (`ImportTransform::Rows`, Gemma's / Inkling 2's gate‖up halves);
/// `None` for a stack whose host-prepared bytes are not one range of each
/// stored expert. A second-level scale leaves the bytes as stored, as does
/// a grid of block scales (`ScaleBlocks`: the source's own second part)
/// whenever any row range aligns with the grid's row tile (`tile[0]`).
pub(super) fn stored_rows(descriptor: &magnitude_family_contracts::WeightDescriptor) -> Option<u64> {
    let mut start = 0;
    let mut rows = 0;
    let mut ranges = 0;
    let mut grid_row_tile: Option<u64> = None;
    for transform in &descriptor.transforms {
        match transform {
            ImportTransform::ScaleByTensor { .. } => {}
            ImportTransform::ScaleBlocks { tile, .. } => grid_row_tile = Some(tile[0]),
            ImportTransform::Rows(range) => {
                start = range.start;
                rows = range.rows;
                ranges += 1;
            }
            _ => return None,
        }
    }
    if ranges > 1 {
        return None;
    }
    if ranges == 1 {
        if let Some(tile_rows) = grid_row_tile {
            if tile_rows == 0 || start % tile_rows != 0 || rows % tile_rows != 0 {
                return None;
            }
        }
    }
    Some(start)
}
