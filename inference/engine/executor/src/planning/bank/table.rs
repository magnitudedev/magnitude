use super::records::reserved_record;
use crate::error::PlanError;
use crate::planning::weights::{Holding, WeightPlan, representation_bytes};
use magnitude_family_contracts::{ImportTransform, ModelDefinition, Operator, WeightRole, WeightScope};
use std::num::NonZeroU64;

/// A stored table (`TableFill::Stored`, an n-gram table) whose slots the
/// bank holds: `[records, record_rows, width]` in reserved storage, each
/// record the least whole number of rows filling whole granules (the unit
/// its storage commits, and Metal's chunk), committed from `floor` records
/// up to every one (its plan's ceiling: the fewest records holding the
/// slots a fixed table holds, at most the whole records its rows fill).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankTable {
    pub role: WeightRole,
    /// Its ceiling: more than its floor.
    pub records: NonZeroU64,
    pub record_rows: NonZeroU64,
    /// A record's resident bytes.
    pub record_bytes: NonZeroU64,
    /// The least records it holds: those of the most rows one step names.
    pub floor: u64,
    /// One row's bytes as stored: what a miss reads.
    pub stored_row_bytes: u64,
    /// The table planned fixed, as the load holds it unbanked.
    pub fixed: WeightPlan,
}

impl BankTable {
    /// The table of `fixed`, the stored table of an n-gram input whose
    /// steps name at most `step_rows` rows, held in records of `reservation`;
    /// none where its host-filled slots cannot be reserved (the host does not
    /// address reserved bytes) or its floor fills its ceiling.
    pub fn of(fixed: &WeightPlan, step_rows: u64, reservation: seismic::Reservation) -> Result<Option<Self>, String> {
        let width = fixed.table_width();
        let row = representation_bytes(fixed.resident, &[1, width])?;
        let granule = NonZeroU64::new(reservation.granule).ok_or("a reservation backs no granule")?;
        // The least rows filling whole granules: the granule over what it
        // shares with a row, at least one.
        let record_rows = NonZeroU64::new(granule.get() / gcd(granule.get(), row))
            .expect("a granule's quotient by one of its divisors is at least one");
        // No more slots than the table has rows: a record past them could
        // never hold one (`StoredTable::new` refuses it).
        let records = fixed
            .table_slots()
            .div_ceil(record_rows.get())
            .min(fixed.stored_shape()[0] / record_rows);
        let floor = step_rows.div_ceil(record_rows.get());
        // A table whose floor fills its ceiling is not banked: one banked
        // holds more records than its floor, so at least one.
        let Some(records) = NonZeroU64::new(records).filter(|records| reservation.host_addressed && records.get() > floor)
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            role: fixed.role,
            records,
            record_rows,
            record_bytes: reserved_record(
                row.checked_mul(record_rows.get()).ok_or("a banked table's record bytes overflow")?,
            )?,
            floor,
            stored_row_bytes: representation_bytes(fixed.upload, &[1, width])?,
            fixed: fixed.clone(),
        }))
    }

    /// What `committed` of its records are charged in `reservation`, their
    /// record table included (`Reservation::charge`).
    pub fn charge(&self, reservation: seismic::Reservation, committed: u64) -> Result<u64, String> {
        committed
            .checked_mul(self.record_bytes.get())
            .and_then(|bytes| reservation.charge(seismic::ReservedRecords::of(self.record_bytes), self.records.get(), bytes))
            .ok_or_else(|| "banked table charge overflows".to_owned())
    }

    /// The table's plan as the bank holds it, every record committed: its
    /// records' backing, beside the table of them every stored table's
    /// slots hold (`WeightPlan::record_table_entries`).
    pub(super) fn plan(&self, reservation: seismic::Reservation) -> Result<WeightPlan, String> {
        let mut plan = self.fixed.clone();
        plan.shape = vec![self.records.get(), self.record_rows.get(), self.fixed.table_width()];
        plan.resident_bytes = self
            .records
            .get()
            .checked_mul(self.record_bytes.get())
            .and_then(|bytes| reservation.of_records(seismic::ReservedRecords::of(self.record_bytes)).backing(bytes))
            .ok_or("a banked table's backing overflows")?;
        let slots = plan.table_slots();
        if let Holding::Streamed(streamed) = &mut plan.holding {
            streamed.directory = super::super::DirectoryForm::of(streamed.table_rows, slots);
        }
        Ok(plan)
    }
}

pub(super) fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
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
/// rows (`ImportTransform::Rows`, Gemma's gate‖up halves); `None` for a
/// stack whose host-prepared bytes are not one range of each stored expert.
/// A second-level scale leaves the bytes as stored.
pub(super) fn stored_rows(descriptor: &magnitude_family_contracts::WeightDescriptor) -> Option<u64> {
    let mut start = 0;
    let mut ranges = 0;
    for transform in &descriptor.transforms {
        match transform {
            ImportTransform::ScaleByTensor { .. } => {}
            ImportTransform::Rows(range) => {
                start = range.start;
                ranges += 1;
            }
            _ => return None,
        }
    }
    (ranges <= 1).then_some(start)
}
