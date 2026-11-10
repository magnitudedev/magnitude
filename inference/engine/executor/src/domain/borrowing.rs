//! The expert bank's borrowed room as the host's reading needs it: the
//! bytes the one store's bank holds beyond the hold line, and what judges
//! the host's displacement while it holds them.
//!
//! Whether a byte is borrowed is the memory heap's decision
//! ([`super::value`]): a step takes borrowed room only where its rate per
//! byte beats the borrow price, and what an ask or a claim of the load's
//! own is owed the settle releases lowest rate per byte first, which the
//! borrowed tail of the bank's slots is. What stays here is what the
//! host's reading of host RAM needs to read the load's hold on that room
//! (`platform::Serving`): the bytes held on it, which host RAM's headroom
//! then counts within what the OS can obtain, and the displacement baseline
//! that judges displacement now, from the bank's last growth on.
use crate::platform::{DisplacementBaseline, DomainReading, Recovery};
use std::time::Instant;

/// Which room a growth of the bank's slots takes: what a claim of it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Room {
    /// Within the hold line.
    Committed,
    /// Beyond it: where the host reports displacement, what the OS can
    /// obtain by displacing others.
    Borrowed,
}

/// The one store's bank's slots on borrowed room, and what judges the
/// host's displacement while the bank holds them ([`BankRoom::lent`],
/// [`BankRoom::baseline`]).
#[derive(Clone, Debug)]
pub struct BankRoom {
    /// Whether the host reports displacement: only there is room borrowed
    /// beyond allocatable memory.
    counted: bool,
    slot_bytes: u64,
    /// The bank's slots now, and those of them on borrowed room: its last
    /// grown, released first as the bank's least valuable.
    slots: u64,
    borrowed: u64,
    /// The borrowed slots a growth that runs takes, as it began.
    growing: Option<u64>,
    /// What displacement now is judged against, from the bank's last
    /// growth on.
    baseline: DisplacementBaseline,
    /// Whether the host has shown no displacement now and no distress for
    /// `RECOVERY_STABLE_TIME`, as of the latest reading.
    quiet: Recovery<Instant>,
    recovered: bool,
}

impl BankRoom {
    /// A bank of `slots` slots of `slot_bytes` each, none borrowed (the
    /// plan sized them within the hold line), serving from `now` on a host
    /// that reports displacement where `counted`.
    pub fn new(counted: bool, slot_bytes: u64, slots: u64, now: Instant) -> Self {
        Self {
            counted,
            slot_bytes,
            slots,
            borrowed: 0,
            growing: None,
            baseline: DisplacementBaseline::since(now),
            quiet: Recovery::new(),
            recovered: false,
        }
    }

    /// The bytes of the bank's borrowed slots.
    pub fn borrowed(&self) -> u64 {
        self.borrowed.saturating_mul(self.slot_bytes)
    }

    /// While the bank holds room borrowed beyond allocatable memory, or
    /// grows into it, the bytes it holds there, which host RAM's hold then
    /// counts within what the OS can obtain (`platform::Serving::lent`).
    /// None on a host that reports no displacement, or holding none.
    pub fn lent(&self) -> Option<u64> {
        let growing = self.growing.unwrap_or(0);
        (self.counted && self.borrowed + growing > 0).then(|| (self.borrowed + growing).saturating_mul(self.slot_bytes))
    }

    /// What displacement now is judged against: none while the bank grows.
    pub fn baseline(&self) -> Option<DisplacementBaseline> {
        self.growing.is_none().then_some(self.baseline)
    }

    /// Whether a growth may take borrowed room now: none runs, and where
    /// the host reports displacement it has shown none now and no distress
    /// for `RECOVERY_STABLE_TIME`, so no borrowed room is taken under a
    /// program's demand that still holds its memory.
    pub fn borrows(&self) -> bool {
        self.growing.is_none() && (!self.counted || self.recovered)
    }

    /// A growth of the slots began, `borrowed` of its bytes on borrowed
    /// room.
    pub fn grow(&mut self, borrowed: u64) {
        assert!(self.growing.is_none(), "invariant violated: a growth of the bank's slots began beside another");
        self.growing = Some(borrowed.div_ceil(self.slot_bytes));
    }

    /// A growth of the slots ended at `now`, holding `slots` slots: of the
    /// slots it joined, its last are those it borrowed. Displacement now is
    /// judged from here on. Whether it borrowed any.
    pub fn grown(&mut self, slots: u64, now: Instant) -> bool {
        let growing = self.growing.take().expect("invariant violated: a growth of the bank's slots ended that never began");
        assert!(slots >= self.slots, "invariant violated: a growth left the bank fewer slots");
        let joined = (slots - self.slots).min(growing);
        self.borrowed += joined;
        self.slots = slots;
        self.baseline = DisplacementBaseline::since(now);
        joined > 0
    }

    /// A shrink left the bank `slots` slots: its last, the borrowed, went
    /// first.
    pub fn shrunk(&mut self, slots: u64) {
        assert!(slots <= self.slots, "invariant violated: a shrink left the bank more slots");
        self.borrowed = self.borrowed.saturating_sub(self.slots - slots);
        self.slots = slots;
    }

    /// Fold in a reading of host RAM at `now`: its displacement windows into
    /// the baseline (none while the bank grows, whose backing displaces),
    /// and whether it showed displacement now or distress into the wait
    /// before borrowing.
    pub fn observe(&mut self, reading: &DomainReading, now: Instant) {
        if self.growing.is_some() {
            return;
        }
        if let Some(displacement) = &reading.displacement {
            self.baseline.fold(displacement);
        }
        let quiet = !reading.displacing && reading.distress.is_none();
        self.recovered = self.quiet.observe(quiet, now, |since, now| now.duration_since(since));
    }
}

#[cfg(test)]
mod tests {
    //! The split alone, over readings each test writes.
    use super::*;
    use crate::platform::{DomainRole, DomainThresholds, MemoryBand, MemoryConstraint, RECOVERY_STABLE_TIME};
    use std::time::Duration;

    const SLOT: u64 = 100;

    /// A host RAM reading, displacing now where `displacing`, in the band
    /// the hold on borrowed room then reads.
    fn reading(displacing: bool) -> DomainReading {
        let topology = seismic::Devices::discover(&[]).unwrap().topology();
        let store = topology.host_ram().id();
        let thresholds = DomainThresholds { planning_bytes: 1_000, emergency_bytes: 500 };
        DomainReading {
            role: DomainRole::Allocation,
            domain: store,
            headroom_bytes: 2_000,
            shown: crate::platform::Shown::headroom(2_000),
            identity: topology.store_identity(store).unwrap(),
            constraint: MemoryConstraint::HostRam,
            thresholds,
            ceiling_bytes: thresholds.ceiling(2_000),
            distress: None,
            held: None,
            displacement: None,
            borrowable_bytes: Some(4_000),
            displacing,
            band: if displacing { MemoryBand::Reclaim } else { MemoryBand::Normal },
        }
    }

    /// A bank of 10 committed slots that has stayed quiet long enough to
    /// borrow.
    fn quiet_bank(counted: bool) -> (BankRoom, Instant) {
        let start = Instant::now();
        let mut bank = BankRoom::new(counted, SLOT, 10, start);
        bank.observe(&reading(false), start);
        let now = start + RECOVERY_STABLE_TIME;
        bank.observe(&reading(false), now);
        assert!(bank.borrows(), "quiet for the stable time");
        (bank, now)
    }

    /// A growth's borrowed slots are its last joined; releases take them
    /// first, and the hold reads what the bank holds on borrowed room.
    #[test]
    fn releases_take_borrowed_slots_first() {
        let (mut bank, now) = quiet_bank(true);
        bank.grow(0);
        assert!(!bank.grown(12, now));
        assert_eq!(bank.lent(), None, "nothing borrowed");
        bank.grow(5 * SLOT);
        assert_eq!(bank.lent(), Some(5 * SLOT), "a borrowing growth is held as it begins");
        assert!(bank.grown(17, now));
        bank.shrunk(14);
        assert_eq!((bank.borrowed(), bank.lent()), (2 * SLOT, Some(2 * SLOT)), "three borrowed slots went first");
        bank.shrunk(9);
        assert_eq!((bank.borrowed(), bank.lent()), (0, None), "then the rest");
    }

    /// Displacement now stops borrowing until the host stayed quiet, and a
    /// growth that joined fewer slots than it borrowed for holds only those.
    #[test]
    fn displacement_now_holds_borrowing_and_a_growth_holds_what_it_joined() {
        let (mut bank, now) = quiet_bank(true);
        bank.grow(6 * SLOT);
        assert_eq!(bank.baseline(), None, "nothing is judged while the bank grows");
        bank.observe(&reading(true), now);
        assert!(!bank.borrows(), "no growth begins beside it");
        bank.grown(13, now);
        assert_eq!((bank.borrowed(), bank.lent()), (3 * SLOT, Some(3 * SLOT)));
        bank.observe(&reading(true), now + Duration::from_secs(1));
        assert!(!bank.borrows(), "no borrowing until it stayed quiet");
        bank.observe(&reading(false), now + Duration::from_secs(2));
        bank.observe(&reading(false), now + Duration::from_secs(2) + RECOVERY_STABLE_TIME);
        assert!(bank.borrows(), "quiet for the stable time again");
    }

    /// Where the host reports no displacement, allocatable memory counts
    /// all the bank holds: the hold reads nothing borrowed.
    #[test]
    fn an_uncounted_host_reads_no_borrowed_room() {
        let mut bank = BankRoom::new(false, SLOT, 10, Instant::now());
        assert!(bank.borrows(), "no displacement to wait out");
        bank.grow(3 * SLOT);
        assert_eq!(bank.lent(), None);
    }
}
