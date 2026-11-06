//! The expert bank's room beyond the hold line: the one store's slots split
//! into committed slots, held within what the host gives without displacing
//! anyone (allocatable memory less the planning reserve, less the class room
//! of the planned prompt places no group holds), and borrowed slots above
//! them.
//!
//! Borrowed slots come from two rooms. The places' room: the plan leaves a
//! prompt group's class of its planned places free beside the bank
//! (`AssessmentMemoryCharge::with_group_class`), and while no group holds
//! it the bank may grow into it. And, where the host reports displacement,
//! what the OS can obtain by compressing other programs' data
//! (`platform::FitMode::Borrow`), the room a load reading obtainable memory
//! would plan on: a bank that holds it decodes as fast as one that did,
//! and gives it back before anything the load committed.
//!
//! Releases take borrowed slots first by construction: the committed count
//! only ever falls to the slots the bank holds, so every slot released comes
//! off the borrowed ones while any is left. The load gives borrowed room
//! back on what [`BankRoom::owed`] says: a claim of the load's own that took
//! it back (a prompt group's places among them), and, where the host
//! reports displacement, all of it once the host shows displacement now
//! (`platform::Serving`). The hold line's level decides only once none is
//! left, as for any load.

use crate::platform::{DisplacementBaseline, DomainReading, MemoryBand, Recovery};
use std::time::Instant;

/// Which room a growth of the bank's slots takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Room {
    /// Within the hold line, beside the class room no group holds.
    Committed,
    /// Beyond it: the places' room, and where the host reports
    /// displacement what the OS can obtain by displacing others.
    Borrowed,
}

/// The one store's bank slots, committed and borrowed, and what the load's
/// hold on the borrowed ones reads ([`BankRoom::lent`], [`BankRoom::baseline`]).
#[derive(Clone, Debug)]
pub struct BankRoom {
    /// Whether the host reports displacement: only there is room borrowed
    /// beyond allocatable memory, and held on displacement now. Elsewhere
    /// the bank borrows the places' room alone, which allocatable memory
    /// counts, and the hold line's level holds it.
    counted: bool,
    slot_bytes: u64,
    /// The bank's slots now, and those of them within the hold line.
    slots: u64,
    committed: u64,
    /// The room a growth of the slots that runs takes.
    growing: Option<Room>,
    /// The borrowed bytes the load's claims may still take back; the rest
    /// of what the bank borrowed it owes ([`Self::owed`]). A growth into
    /// borrowed room lends its bytes as it begins.
    lent: u64,
    /// What displacement now is judged against, from the bank's last
    /// growth on.
    baseline: DisplacementBaseline,
    /// Whether the host has shown no displacement now and no distress for
    /// `RECOVERY_STABLE_TIME`, as of the latest reading.
    quiet: Recovery<Instant>,
    recovered: bool,
}

impl BankRoom {
    /// A bank of `slots` slots of `slot_bytes` each, every one committed
    /// (the plan sized them within the hold line), serving from `now` on a
    /// host that reports displacement where `counted`.
    pub fn new(counted: bool, slot_bytes: u64, slots: u64, now: Instant) -> Self {
        Self {
            counted,
            slot_bytes,
            slots,
            committed: slots,
            growing: None,
            lent: 0,
            baseline: DisplacementBaseline::since(now),
            quiet: Recovery::new(),
            recovered: false,
        }
    }

    /// The bytes of the bank's borrowed slots.
    pub fn borrowed(&self) -> u64 {
        (self.slots - self.committed).saturating_mul(self.slot_bytes)
    }

    /// The borrowed bytes the bank owes back: what the load's claims took
    /// back of them, or all of them once the host showed displacement now.
    pub fn owed(&self) -> u64 {
        self.borrowed().saturating_sub(self.lent)
    }

    /// While the bank holds borrowed room beyond allocatable memory, or
    /// grows into it: the bytes of it the load's claims may still take back
    /// (`platform::Serving::lent`), which host RAM's hold then reads in
    /// place of allocatable memory. None on a host that reports no
    /// displacement, where allocatable memory counts all the bank borrows.
    pub fn lent(&self) -> Option<u64> {
        let borrowing = self.borrowed() > 0 || self.growing == Some(Room::Borrowed);
        (self.counted && borrowing).then_some(self.lent)
    }

    /// The borrowed bytes the load's claims may still take back, on any
    /// host.
    pub fn lends(&self) -> u64 {
        self.lent
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

    /// A growth of the slots into `room` began, backing `bytes`.
    pub fn grow(&mut self, room: Room, bytes: u64) {
        assert!(self.growing.is_none(), "invariant violated: a growth of the bank's slots began beside another");
        self.growing = Some(room);
        if room == Room::Borrowed {
            self.lent = self.lent.saturating_add(bytes);
        }
    }

    /// A growth of the slots ended at `now`, holding `slots` slots: the room
    /// it took, the slots it joined to it (none where it stopped or was
    /// refused before its first). Displacement now is judged from here on.
    pub fn grown(&mut self, slots: u64, now: Instant) -> Room {
        let room = self.growing.take().expect("invariant violated: a growth of the bank's slots ended that never began");
        assert!(slots >= self.slots, "invariant violated: a growth left the bank fewer slots");
        if room == Room::Committed {
            self.committed += slots - self.slots;
        }
        self.slots = slots;
        self.lent = self.lent.min(self.borrowed());
        self.baseline = DisplacementBaseline::since(now);
        room
    }

    /// A shrink left the bank `slots` slots: its borrowed ones go first.
    pub fn shrunk(&mut self, slots: u64) {
        assert!(slots <= self.slots, "invariant violated: a shrink left the bank more slots");
        self.slots = slots;
        self.committed = self.committed.min(slots);
        self.lent = self.lent.min(self.borrowed());
    }

    /// A claim of the load's own took `bytes` of host RAM while the bank
    /// lends borrowed room, or wants them: the bank owes back as much of
    /// what it lends. What it took back.
    pub fn take_back(&mut self, bytes: u64) -> u64 {
        let taken = bytes.min(self.lent);
        self.lent -= taken;
        taken
    }

    /// Fold in a reading of host RAM at `now`: its displacement windows into
    /// the baseline (none while the bank grows, whose backing displaces),
    /// and whether it showed displacement now or distress into the wait
    /// before borrowing. A reading in the Reclaim band while the hold reads
    /// borrowed room (`platform::Serving::lent`: displacement now, distress,
    /// what the OS can obtain at the planning reserve, or nothing left lent)
    /// takes all of it back.
    pub fn observe(&mut self, reading: &DomainReading, now: Instant) {
        if self.growing.is_some() {
            return;
        }
        if let Some(displacement) = &reading.displacement {
            self.baseline.fold(displacement);
        }
        let quiet = !reading.displacing && reading.distress.is_none();
        self.recovered = self.quiet.observe(quiet, now, |since, now| now.duration_since(since));
        if reading.band == MemoryBand::Reclaim && self.lent().is_some() {
            self.lent = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    //! The split alone, over readings each test writes.
    use super::*;
    use crate::platform::{DomainRole, DomainThresholds, MemoryConstraint, RECOVERY_STABLE_TIME};
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

    /// Growths add to the room they took; releases take the borrowed slots
    /// first, the committed ones only once none is left.
    #[test]
    fn releases_take_borrowed_slots_before_committed_ones() {
        let (mut bank, now) = quiet_bank(true);
        bank.grow(Room::Committed, 2 * SLOT);
        assert_eq!(bank.grown(12, now), Room::Committed);
        bank.grow(Room::Borrowed, 5 * SLOT);
        assert_eq!(bank.lent(), Some(5 * SLOT), "a borrowing growth lends its bytes as it begins");
        assert_eq!(bank.grown(17, now), Room::Borrowed);
        assert_eq!((bank.borrowed(), bank.owed()), (5 * SLOT, 0));
        bank.shrunk(14);
        assert_eq!((bank.borrowed(), bank.lent()), (2 * SLOT, Some(2 * SLOT)), "three borrowed slots went first");
        bank.shrunk(9);
        assert_eq!((bank.borrowed(), bank.lent()), (0, None), "then the committed ones");
        bank.grow(Room::Committed, SLOT);
        bank.grown(10, now);
        assert_eq!(bank.borrowed(), 0, "a committed growth after it commits");
    }

    /// A claim of the load's takes back borrowed room up to what is lent,
    /// which the bank then owes; displacement now takes all of it back.
    #[test]
    fn claims_and_displacement_now_take_borrowed_room_back() {
        let (mut bank, now) = quiet_bank(true);
        bank.grow(Room::Borrowed, 6 * SLOT);
        bank.grown(16, now);
        assert_eq!(bank.take_back(2 * SLOT), 2 * SLOT);
        assert_eq!((bank.owed(), bank.lent()), (2 * SLOT, Some(4 * SLOT)));
        assert_eq!(bank.take_back(9 * SLOT), 4 * SLOT, "no more than is lent");
        let (mut bank, now) = quiet_bank(true);
        bank.grow(Room::Borrowed, 6 * SLOT);
        bank.grown(16, now);
        bank.observe(&reading(true), now + Duration::from_secs(1));
        assert_eq!((bank.owed(), bank.lent()), (6 * SLOT, Some(0)), "displacement now: all of it");
        assert!(!bank.borrows(), "and no borrowing until it stayed quiet");
        bank.observe(&reading(false), now + Duration::from_secs(2));
        assert!(!bank.borrows());
        bank.observe(&reading(false), now + Duration::from_secs(2) + RECOVERY_STABLE_TIME);
        assert!(bank.borrows(), "quiet for the stable time again");
    }

    /// No displacement is judged while the bank grows, and a growth that
    /// joined fewer slots than it lent for lends only what it borrowed.
    #[test]
    fn a_growth_is_not_displacement_now_and_lends_what_it_joined() {
        let (mut bank, now) = quiet_bank(true);
        bank.grow(Room::Borrowed, 6 * SLOT);
        assert_eq!(bank.baseline(), None, "nothing is judged while the bank grows");
        bank.observe(&reading(true), now);
        assert_eq!(bank.lent(), Some(6 * SLOT), "its own backing displaces");
        assert!(!bank.borrows(), "no growth begins beside it");
        bank.grown(13, now);
        assert_eq!((bank.borrowed(), bank.lent()), (3 * SLOT, Some(3 * SLOT)));
    }

    /// Where the host reports no displacement the bank borrows the places'
    /// room alone, which allocatable memory counts: no wait, and the hold
    /// line's level holds it.
    #[test]
    fn an_uncounted_host_borrows_the_places_room_on_the_level() {
        let mut bank = BankRoom::new(false, SLOT, 10, Instant::now());
        assert!(bank.borrows(), "no displacement to wait out");
        bank.grow(Room::Borrowed, 3 * SLOT);
        assert_eq!(bank.lent(), None, "the hold line reads allocatable memory");
        bank.grown(13, Instant::now());
        assert_eq!(bank.take_back(SLOT), SLOT, "a group's places still take it back");
        assert_eq!(bank.owed(), SLOT);
    }
}
