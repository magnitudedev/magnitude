use super::{ExpertBankPlan, Placer, growth_target};

/// What a bank's plan derives of each count it may hold, derived once
/// ([`ExpertBankPlan::counts`]): each group's records and the record
/// reservations' charge at every slot count up to the ceiling, the host
/// tier's at every host slot count up to every expert, and the floor. The
/// plan derives each from its whole layout (its conveyor, scratch and
/// staging pools), so an opened bank, whose plan no longer changes, reads
/// them here: a resize decided on the step's thread derives nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankCounts {
    placer: Placer,
    floor: u64,
    /// Indexed by slots, `0..=ceiling`: each group's records
    /// ([`ExpertBankPlan::records_at`]) and their charge
    /// ([`ExpertBankPlan::resident_charge`]).
    slots: Vec<Held>,
    /// Indexed by host slots, `0..=` the host ceiling: each group's host
    /// tier records ([`ExpertBankPlan::host_records`]) and their charge
    /// ([`ExpertBankPlan::host_charge`]); none without a host tier.
    host: Vec<Held>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Held {
    records: Vec<u64>,
    charge: u64,
}

impl ExpertBankPlan {
    /// The bank's counts ([`BankCounts`]), derived from the plan as it is
    /// now. Why a charge does not fit `u64`.
    pub fn counts(&self) -> Result<BankCounts, String> {
        let groups = 0..self.groups.len();
        let slots = (0..=self.ceiling)
            .map(|slots| {
                Ok(Held {
                    records: groups.clone().map(|group| self.records_at(group, slots)).collect(),
                    charge: self.resident_charge(slots)?,
                })
            })
            .collect::<Result<_, String>>()?;
        let host = match self.host_ceiling() {
            Some(ceiling) => (0..=ceiling.get())
                .map(|host| {
                    Ok(Held {
                        records: groups.clone().map(|group| self.host_records(group, host)).collect(),
                        charge: self.host_charge(host)?,
                    })
                })
                .collect::<Result<_, String>>()?,
            None => Vec::new(),
        };
        Ok(BankCounts { placer: self.placer(), floor: self.floor(), slots, host })
    }
}

impl BankCounts {
    /// [`ExpertBankPlan::floor`].
    pub fn floor(&self) -> u64 {
        self.floor
    }

    /// [`ExpertBankPlan::records_at`].
    pub fn records_at(&self, group: usize, slots: u64) -> u64 {
        self.at_slots(slots).records[group]
    }

    /// [`ExpertBankPlan::resident_charge`].
    pub fn resident_charge(&self, slots: u64) -> u64 {
        self.at_slots(slots).charge
    }

    /// [`ExpertBankPlan::host_records`].
    pub fn host_records(&self, group: usize, host_slots: u64) -> u64 {
        self.at_host(host_slots).records[group]
    }

    /// [`ExpertBankPlan::host_charge`]: none without a host tier.
    pub fn host_charge(&self, host_slots: u64) -> u64 {
        match self.host.is_empty() {
            true => 0,
            false => self.at_host(host_slots).charge,
        }
    }

    /// The records the host lands into of group `group` at `slots` slots: a
    /// host tier's host slots where the device places, the one store's slots
    /// where the host does.
    pub fn landed_records(&self, group: usize, slots: u64) -> u64 {
        match self.placer {
            Placer::Tier => self.host_records(group, slots),
            Placer::Bank => self.records_at(group, slots),
        }
    }

    /// The host slots a host tier at `from` grows to in one step
    /// ([`growth_target`]) beside a bank of `slots` slots: toward every
    /// expert, its ceiling (it is inclusive of the bank: it holds the bank's
    /// experts too), past the bank's slots, its charge beyond `from`'s
    /// ([`Self::host_charge`]) within `room`; none without a tier, whole, or
    /// with no room for one past the bank. What a live load's growth step
    /// takes where the memory's budget grows; the tier's load-time size is
    /// the plan's ([`ExpertBankPlan::host_slots`]).
    pub fn host_growth(&self, from: u64, slots: u64, room: u64) -> Option<u64> {
        let bound = u64::try_from(self.host.len().checked_sub(1)?).expect("host slots fit u64");
        let held = self.host_charge(from);
        // An inclusive tier's first slots, up to the bank's, hold copies of
        // the bank's experts and save nothing: a growth reaches past them in
        // one step, or none fits.
        let least = from.max(slots);
        growth_target(least, bound, room, |to| Ok::<_, std::convert::Infallible>(self.host_charge(to) - held))
            .unwrap_or_else(|never| match never {})
    }

    fn at_slots(&self, slots: u64) -> &Held {
        usize::try_from(slots)
            .ok()
            .and_then(|index| self.slots.get(index))
            .unwrap_or_else(|| panic!("{slots} bank slots lie past the ceiling, {}", self.slots.len() - 1))
    }

    fn at_host(&self, host_slots: u64) -> &Held {
        usize::try_from(host_slots)
            .ok()
            .and_then(|index| self.host.get(index))
            .unwrap_or_else(|| panic!("{host_slots} host slots lie past the host tier's ceiling ({} counts)", self.host.len()))
    }
}
