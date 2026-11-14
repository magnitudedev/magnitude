//! The ledger's journal of its own changes (design §4.1): every change of
//! what the process allocated in a store, `(t, Δ)`, kept until every later
//! sample shows it.
//!
//! - **When a change shows.** A change made at `t` shows in a figure once
//!   the sample was taken after it (its place among the scope's changes,
//!   [`super::ledger::Sampled`]) and the figure shows the machine at or past
//!   `t + λ`, the store's lag for a change of its direction
//!   ([`seismic::Lag`]). The live figures show the machine at the sample's
//!   instant; the counters (cold, harm, wire count) at their own
//!   ([`seismic::Counted`]), so a change made after their snapshot never
//!   shows in them. Changes show in their order: one shows only once every
//!   earlier one of its figure does, so a release never shows ahead of the
//!   growth whose bytes it returns (a Linux cgroup shows a growth within
//!   its 2 s statistics flush and a release in the call), and what a sample
//!   shows of the process is its figure at some instant it held it.
//! - **What a sample shows of the process.** `allocated_visible(t)` is what
//!   the process allocated less every change the sample does not show yet,
//!   so `others = used − allocated_visible` never moves with the process's
//!   own action; and its wired bytes likewise, so others' wired bytes never
//!   move with it either.
//! - **Dropped once shown.** Samples are taken in order, so a change one
//!   sample shows every later one shows: it leaves the journal then.

use std::time::Instant;

use super::ledger::Sampled;

/// The process's own figures in a store that the OS shows: what it
/// allocated there, and of it what the OS wires while the process holds it
/// (`seismic::memory::Wiring::Wired`), which a wired ceiling's count shows
/// (design §4.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Own {
    pub(crate) allocated: u64,
    pub(crate) wired: u64,
}

/// One change of the process's figures in the store: each figure's signed
/// move, which shows after the lag of its own direction.
#[derive(Clone, Copy, Debug)]
struct Change {
    /// Its place among the scope's changes.
    number: u64,
    /// The instant it was made, on the owner's clock.
    at: Instant,
    allocated: i128,
    wired: i128,
}

/// A store's own changes no sample shows yet.
#[derive(Debug, Default)]
pub(crate) struct Journal {
    changes: Vec<Change>,
}

impl Journal {
    /// The process's figures in the store moved from `before` to `now` at
    /// `at`, as change `number`.
    pub(crate) fn record(&mut self, number: u64, at: Instant, before: Own, now: Own) {
        if before != now {
            let moved = |before: u64, now: u64| i128::from(now) - i128::from(before);
            self.changes.push(Change { number, at, allocated: moved(before.allocated, now.allocated), wired: moved(before.wired, now.wired) });
        }
    }

    /// Whether a figure of `sample` showing the machine at `seen`, from a
    /// store lagging by `lag`, shows a move of `moved` bytes made by
    /// `change`: none shows at once.
    fn shows(sample: Sampled, seen: Instant, lag: seismic::Lag, change: &Change, moved: i128) -> bool {
        let lag = if moved > 0 { lag.growth } else { lag.release };
        moved == 0 || (change.number < sample.after() && change.at + lag <= seen)
    }

    /// The changes of `figure` a figure of `sample` showing the machine at
    /// `seen` does not show: every one from the first it does not, as they
    /// show in their order.
    fn unshown(
        &self,
        sample: Sampled,
        seen: Instant,
        lag: seismic::Lag,
        figure: fn(&Change) -> i128,
    ) -> impl Iterator<Item = &Change> {
        self.changes.iter().skip_while(move |change| Self::shows(sample, seen, lag, change, figure(change)))
    }

    /// What the process held as a figure of `sample` showing the machine at
    /// `seen` shows it, `own` now: every move it does not show undone.
    ///
    /// # Panics
    /// Where that is not a count of bytes: the journal holds every change
    /// no sample shows, and those undone are every one after an instant, so
    /// undoing them gives what the process held then, never less than zero.
    pub(crate) fn visible(&self, own: Own, sample: Sampled, seen: Instant, lag: seismic::Lag) -> Own {
        let shown = |now: u64, figure: fn(&Change) -> i128| {
            let shown = self
                .unshown(sample, seen, lag, figure)
                .fold(i128::from(now), |shown, change| shown - figure(change));
            u64::try_from(shown).unwrap_or_else(|_| panic!("invariant violated: the process's own figure reads {shown} bytes at a sample"))
        };
        Own { allocated: shown(own.allocated, |change| change.allocated), wired: shown(own.wired, |change| change.wired) }
    }

    /// Drop the changes `sample`'s figures all show, from the first, the
    /// slowest of them showing the machine at `seen`: every later sample
    /// shows them too.
    pub(crate) fn settle(&mut self, sample: Sampled, seen: Instant, lag: seismic::Lag) {
        let shown = self
            .changes
            .iter()
            .take_while(|change| {
                Self::shows(sample, seen, lag, change, change.allocated) && Self::shows(sample, seen, lag, change, change.wired)
            })
            .count();
        self.changes.drain(..shown);
    }

    /// Whether a change of the process's is not yet shown by the latest
    /// sample.
    pub(crate) fn pending(&self) -> bool {
        !self.changes.is_empty()
    }
}
