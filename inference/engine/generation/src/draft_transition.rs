//! Device-free geometry of target-conditioned drafter operations.
//!
//! The method owns state transitions: the boundary a drafter settles at, the
//! head transaction a proposal or maintenance submits, and the boundary each
//! reconciles to.

/// The most complete pairs an own-history drafter holds while its rounds
/// draft nothing. Entering pairs is the drafter's work alone, so it waits
/// for the round that drafts next (`catching_up`), and a round drafting
/// nothing stays an ordinary decode; past this many pairs a round enters
/// them anyway, so the pairs held stay bounded. A catch-up transaction is
/// one launch of at most this many rows.
pub const CATCH_UP_ROWS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftHistory {
    Own,
    Target,
}

/// A physical drafter transaction, including zero-proposal maintenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftHeadCall {
    pub entry_rows: usize,
    pub generated: usize,
    pub position: usize,
}

/// A settled method boundary, without features, tokens or device ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftRoundState {
    history: DraftHistory,
    entry_bound: usize,
    position: usize,
    pending_rows: usize,
    open_row: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftTargetTransition {
    /// State after target features are appended, before maintenance submits.
    pub appended: DraftRoundState,
    pub maintenance: Option<DraftHeadCall>,
    /// Boundary after any maintenance has successfully reconciled.
    pub successor: DraftRoundState,
}

impl DraftRoundState {
    pub fn new(
        history: DraftHistory,
        entry_bound: usize,
        position: usize,
        pending_rows: usize,
        open_row: bool,
    ) -> Result<Self, String> {
        if entry_bound == 0 {
            return Err("drafter entry bound must be positive".into());
        }
        if history == DraftHistory::Target && (pending_rows > 1 || open_row) {
            return Err("target-reading drafter has an invalid boundary".into());
        }
        let state = Self {
            history,
            entry_bound,
            position,
            pending_rows,
            open_row,
        };
        state.target_position()?;
        Ok(state)
    }

    pub fn position(self) -> usize {
        self.position
    }
    pub fn pending_rows(self) -> usize {
        self.pending_rows
    }

    pub fn target_position(self) -> Result<usize, String> {
        match self.history {
            DraftHistory::Target => Ok(self.position),
            DraftHistory::Own => self
                .position
                .checked_add(self.pending_rows)
                .and_then(|position| position.checked_add(usize::from(self.open_row)))
                .ok_or_else(|| "drafter target position exhausted".into()),
        }
    }

    /// Advance only rows that the physical head has successfully entered.
    pub fn entered(self, rows: usize) -> Result<Self, String> {
        let position = match self.history {
            DraftHistory::Own => self
                .position
                .checked_add(rows)
                .ok_or_else(|| "drafter position exhausted".to_owned())?,
            DraftHistory::Target => self.position,
        };
        Ok(Self { position, ..self })
    }

    /// Describe a proposal and its reconciled boundary. Zero proposals submit
    /// no head; their target may still cause maintenance afterward.
    pub fn proposing(self, generated: usize) -> Result<(Option<DraftHeadCall>, Self), String> {
        if generated == 0 {
            return Ok((None, self));
        }
        if generated >= self.entry_bound
            || self.pending_rows == 0
            || self.pending_rows > self.entry_bound
            || self.open_row
        {
            return Err("drafter proposal does not fit its settled boundary".into());
        }
        let call = DraftHeadCall {
            entry_rows: self.pending_rows,
            generated,
            position: self.position,
        };
        let successor = self.entered(self.pending_rows)?;
        Ok((
            Some(call),
            Self {
                pending_rows: 0,
                ..successor
            },
        ))
    }

    /// The pairs to enter before a round drafts: every pending pair but the
    /// anchor, when they are more than a drafting transaction admits.
    pub fn catching_up(self) -> (Option<DraftHeadCall>, Self) {
        if self.pending_rows <= self.entry_bound || self.open_row {
            return (None, self);
        }
        self.flushing(1)
    }

    /// The most pairs this drafter holds while it drafts nothing.
    fn held_bound(self) -> usize {
        self.entry_bound.max(CATCH_UP_ROWS)
    }

    /// Complete pairs only: an open feature has no successor token to enter.
    /// The returned state has removed the submitted rows but has not yet
    /// advanced their physical position.
    pub fn flushing(self, keep: usize) -> (Option<DraftHeadCall>, Self) {
        if self.history == DraftHistory::Target || self.pending_rows <= keep {
            return (None, self);
        }
        let call = DraftHeadCall {
            entry_rows: self.pending_rows - keep,
            generated: 0,
            position: self.position,
        };
        (
            Some(call),
            Self {
                pending_rows: keep,
                ..self
            },
        )
    }

    pub fn observed(self, accepted: usize) -> Result<DraftTargetTransition, String> {
        if accepted == 0 {
            return Err("drafter accepted prefix must be nonempty".into());
        }
        let appended = match self.history {
            DraftHistory::Target => Self {
                position: self
                    .position
                    .checked_add(accepted)
                    .ok_or_else(|| "drafter position exhausted".to_owned())?,
                pending_rows: 1,
                ..self
            },
            DraftHistory::Own => Self {
                pending_rows: self
                    .pending_rows
                    .checked_add(usize::from(self.open_row))
                    .and_then(|rows| rows.checked_add(accepted))
                    .ok_or_else(|| "drafter pending rows exhausted".to_owned())?,
                open_row: false,
                ..self
            },
        };
        appended.target_position()?;
        let (maintenance, remaining) = if appended.history == DraftHistory::Own
            && appended.pending_rows > appended.held_bound()
        {
            appended.flushing(1)
        } else {
            (None, appended)
        };
        let successor = match maintenance {
            Some(call) => remaining.entered(call.entry_rows)?,
            None => remaining,
        };
        Ok(DraftTargetTransition {
            appended,
            maintenance,
            successor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{DraftHeadCall, DraftHistory, DraftRoundState};

    fn state(
        history: DraftHistory,
        entry_bound: usize,
        position: usize,
        pending_rows: usize,
        open_row: bool,
    ) -> DraftRoundState {
        DraftRoundState::new(history, entry_bound, position, pending_rows, open_row).unwrap()
    }

    #[test]
    fn boundaries_are_checked_at_construction() {
        assert!(DraftRoundState::new(DraftHistory::Own, 0, 0, 0, false).is_err());
        // A head reading the target's history holds at most its anchor.
        assert!(DraftRoundState::new(DraftHistory::Target, 4, 0, 2, false).is_err());
        assert!(DraftRoundState::new(DraftHistory::Target, 4, 0, 0, true).is_err());
        assert!(DraftRoundState::new(DraftHistory::Own, 4, usize::MAX, 1, false).is_err());
        // The target stands past every row an own-history drafter holds.
        assert_eq!(
            state(DraftHistory::Own, 4, 5, 2, true).target_position(),
            Ok(8)
        );
        assert_eq!(
            state(DraftHistory::Target, 4, 5, 1, false).target_position(),
            Ok(5)
        );
    }

    #[test]
    fn a_proposal_enters_every_pending_row_and_drafts_the_whole_width() {
        let settled = state(DraftHistory::Own, 4, 1, 2, false);
        let (call, drafted) = settled.proposing(3).unwrap();
        assert_eq!(
            call,
            Some(DraftHeadCall {
                entry_rows: 2,
                generated: 3,
                position: 1,
            })
        );
        assert_eq!((drafted.position(), drafted.pending_rows()), (3, 0));
        // No proposal submits no head and leaves the boundary settled.
        assert_eq!(settled.proposing(0), Ok((None, settled)));
        // A draft needs an anchor, a closed boundary and a width inside the
        // entry bound.
        assert!(settled.proposing(4).is_err());
        assert!(state(DraftHistory::Own, 4, 1, 0, false)
            .proposing(1)
            .is_err());
        assert!(state(DraftHistory::Own, 4, 1, 1, true)
            .proposing(1)
            .is_err());
    }

    #[test]
    fn a_reading_head_drafts_at_the_target_position_and_enters_nothing() {
        let settled = state(DraftHistory::Target, 4, 4, 1, false);
        let (call, drafted) = settled.proposing(3).unwrap();
        assert_eq!(
            call,
            Some(DraftHeadCall {
                entry_rows: 1,
                generated: 3,
                position: 4,
            })
        );
        assert_eq!((drafted.position(), drafted.pending_rows()), (4, 0));
        // Accepted target rows move it; the last one's feature is its anchor.
        let observed = drafted.observed(3).unwrap();
        assert_eq!(observed.maintenance, None);
        assert_eq!(observed.appended, observed.successor);
        assert_eq!(
            (
                observed.successor.position(),
                observed.successor.pending_rows()
            ),
            (7, 1)
        );
        assert_eq!(observed.successor.target_position(), Ok(7));
        // It keeps no history to maintain.
        assert_eq!(
            observed.successor.flushing(0),
            (None, observed.successor)
        );
    }

    #[test]
    fn verification_appends_accepted_rows_behind_an_own_history_drafter() {
        let (_, drafted) = state(DraftHistory::Own, 4, 1, 1, false)
            .proposing(3)
            .unwrap();
        let observed = drafted.observed(3).unwrap();
        assert_eq!(observed.maintenance, None);
        assert_eq!(
            (
                observed.successor.position(),
                observed.successor.pending_rows()
            ),
            (2, 3)
        );
        assert_eq!(observed.successor.target_position(), Ok(5));
        assert!(drafted.observed(0).is_err());
        // An open feature pairs with the verified anchor, ahead of the
        // accepted rows' own pairs.
        let replayed = state(DraftHistory::Own, 4, 2, 0, true);
        assert_eq!(replayed.target_position(), Ok(3));
        let observed = replayed.observed(1).unwrap();
        assert_eq!(observed.successor, state(DraftHistory::Own, 4, 2, 2, false));
        assert_eq!(observed.successor.target_position(), Ok(4));
    }

    #[test]
    fn rows_past_the_catch_up_bound_are_entered_by_maintenance_keeping_the_anchor() {
        // Rounds that draft nothing accumulate one pair each, past what a
        // drafting transaction admits, without entering any.
        let mut settled = state(DraftHistory::Own, 3, 1, 1, false);
        for pending in 2..=super::CATCH_UP_ROWS {
            let observed = settled.observed(1).unwrap();
            assert_eq!(observed.maintenance, None);
            settled = observed.successor;
            assert_eq!(settled.pending_rows(), pending);
        }
        let observed = settled.observed(1).unwrap();
        assert_eq!(observed.appended.pending_rows(), super::CATCH_UP_ROWS + 1);
        assert_eq!(
            observed.maintenance,
            Some(DraftHeadCall {
                entry_rows: super::CATCH_UP_ROWS,
                generated: 0,
                position: 1,
            })
        );
        assert_eq!(
            (
                observed.successor.position(),
                observed.successor.pending_rows()
            ),
            (1 + super::CATCH_UP_ROWS, 1)
        );
        assert_eq!(
            observed.successor.target_position(),
            observed.appended.target_position()
        );
    }

    #[test]
    fn a_round_that_drafts_after_rounds_that_did_not_first_catches_up() {
        // Pairs a drafting transaction admits draft at once.
        let held = state(DraftHistory::Own, 4, 2, 4, false);
        assert_eq!(held.catching_up(), (None, held));
        // More are entered first, all but the anchor, which then drafts.
        let held = state(DraftHistory::Own, 4, 2, 9, false);
        let (call, caught) = held.catching_up();
        assert_eq!(
            call,
            Some(DraftHeadCall {
                entry_rows: 8,
                generated: 0,
                position: 2,
            })
        );
        let caught = caught.entered(8).unwrap();
        assert_eq!(caught.target_position(), held.target_position());
        let (call, _) = caught.proposing(3).unwrap();
        assert_eq!(call.map(|call| (call.entry_rows, call.position)), Some((1, 10)));
        // An open feature waits for its successor.
        let open = state(DraftHistory::Own, 4, 2, 9, true);
        assert_eq!(open.catching_up(), (None, open));
    }

    #[test]
    fn flushing_enters_complete_pairs_only_once_they_reconcile() {
        let settled = state(DraftHistory::Own, 4, 2, 3, false);
        let (call, flushed) = settled.flushing(1);
        assert_eq!(
            call,
            Some(DraftHeadCall {
                entry_rows: 2,
                generated: 0,
                position: 2,
            })
        );
        // The submitted rows leave the pending pairs before they are entered.
        assert_eq!((flushed.position(), flushed.pending_rows()), (2, 1));
        assert_eq!(flushed.entered(2).unwrap().position(), 4);
        assert_eq!(settled.flushing(3), (None, settled));
        assert!(state(DraftHistory::Own, 4, usize::MAX - 1, 1, false)
            .entered(2)
            .is_err());
    }
}
