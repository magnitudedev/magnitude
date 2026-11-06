//! What each volume did, counted since the process began and never
//! consumed: every reader of a count takes the difference of two looks
//! ([`Totals::since`]), so readers never take counts from each other.
//!
//! Per volume: the seconds it held any piece in flight (its busy time) and
//! its pieces in flight integrated over time (its depth); for due pieces
//! and for pieces read ahead (not due when a worker took them, [`Want`]):
//! the bytes and requests (pieces) that ended, read or failed, the seconds
//! a piece of the kind was in flight and its depth, and each ended
//! request's wait. A request's wait, from its submission to its last
//! piece's end, is its volume time (any piece of it in flight) and its time
//! waiting (none in flight, one queued), which is split by what the volume
//! held meanwhile: behind pieces read ahead (one in flight), or behind due
//! pieces otherwise (only due pieces in flight, or a worker not yet
//! woken). The busy time's rate is its bytes over its busy time
//! ([`Totals::rate`]), and the depth over the busy time its mean depth
//! while busy ([`Totals::depth_while_busy`]). The record is the same on
//! every backend: the scheduler keeps it, not a backend, in each volume's
//! book under its queue's lock, which it holds wherever a piece is taken
//! or ends, so counting takes no lock of its own; a look ([`totals`])
//! takes that lock once.

#[cfg(doc)]
use crate::Want;
use crate::Volume;
use std::time::Instant;

/// The counts of `volume` now, the time running counted to now: none
/// before the scheduler took a piece of it.
pub fn totals(volume: Volume) -> Totals {
    crate::scheduler::totals(volume, Instant::now())
}

/// One volume's counts, kept by the scheduler under its queue's lock as it
/// takes and ends the volume's pieces.
#[derive(Debug)]
pub(crate) struct Book {
    /// Until when the integrals are counted.
    at: Instant,
    /// The due pieces in flight, and those read ahead.
    flight: [usize; 2],
    /// The seconds a piece read ahead was in flight, which a waiting
    /// request's time behind them is the difference of ([`Wait`]).
    ahead: f64,
    /// When a piece last ended, none before one did.
    last_ended: Option<Instant>,
    totals: Totals,
}

/// A volume's counts, from the process's start or over a span
/// ([`Totals::since`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Totals {
    /// The seconds any piece was in flight.
    pub busy: f64,
    /// The pieces in flight, integrated over time (piece-seconds).
    pub depth: f64,
    /// Of the pieces due when a worker took them, and of those read ahead.
    pub due: Class,
    pub ahead: Class,
}

/// The counts of due pieces, or of pieces read ahead, on a volume
/// ([`Totals`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Class {
    /// The bytes and pieces that ended, read or failed.
    pub bytes: u64,
    pub requests: u64,
    /// The seconds a piece of the kind was in flight, and its pieces in
    /// flight integrated over time.
    pub busy: f64,
    pub depth: f64,
    /// The waits of the requests that ended, each counted by whether its
    /// last piece was due when taken.
    pub waits: Waits,
}

/// Requests' waits from submission to their last piece's end, split as
/// the module documentation says.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Waits {
    pub requests: u64,
    pub behind_ahead: f64,
    pub behind_due: f64,
    pub device: f64,
}

impl Totals {
    /// Both kinds, due first.
    fn kinds(&self) -> [&Class; 2] {
        [&self.due, &self.ahead]
    }

    fn kind_mut(&mut self, due: bool) -> &mut Class {
        if due { &mut self.due } else { &mut self.ahead }
    }

    /// Every piece's bytes.
    pub fn bytes(&self) -> u64 {
        self.kinds().iter().map(|kind| kind.bytes).sum()
    }

    /// Every piece.
    pub fn requests(&self) -> u64 {
        self.kinds().iter().map(|kind| kind.requests).sum()
    }

    /// The bytes a second while busy, none before it was.
    pub fn rate(&self) -> Option<f64> {
        (self.busy > 0.).then(|| self.bytes() as f64 / self.busy)
    }

    /// The mean pieces in flight while busy, none before it was.
    pub fn depth_while_busy(&self) -> Option<f64> {
        (self.busy > 0.).then(|| self.depth / self.busy)
    }

    /// What happened between `earlier`, a look at the same volume, and
    /// this one.
    pub fn since(&self, earlier: &Self) -> Self {
        let class = |now: &Class, then: &Class| Class {
            bytes: now.bytes - then.bytes,
            requests: now.requests - then.requests,
            busy: now.busy - then.busy,
            depth: now.depth - then.depth,
            waits: Waits {
                requests: now.waits.requests - then.waits.requests,
                behind_ahead: now.waits.behind_ahead - then.waits.behind_ahead,
                behind_due: now.waits.behind_due - then.waits.behind_due,
                device: now.waits.device - then.waits.device,
            },
        };
        Self {
            busy: self.busy - earlier.busy,
            depth: self.depth - earlier.depth,
            due: class(&self.due, &earlier.due),
            ahead: class(&self.ahead, &earlier.ahead),
        }
    }
}

impl Book {
    /// A volume's book, counted from `now`.
    pub(crate) fn new(now: Instant) -> Self {
        Self { at: now, flight: [0; 2], ahead: 0., last_ended: None, totals: Totals::default() }
    }

    /// Its counts at `now`, the time running counted to then.
    pub(crate) fn totals_at(&mut self, now: Instant) -> Totals {
        self.advance(now);
        self.totals
    }

    /// The pieces that ended: no integral, so counted without advancing.
    pub(crate) fn requests(&self) -> u64 {
        self.totals.requests()
    }

    /// When a piece last ended, none before one did.
    pub(crate) fn last_ended(&self) -> Option<Instant> {
        self.last_ended
    }

    /// A piece, due when taken or not, was taken at `now`.
    pub(crate) fn began(&mut self, due: bool, now: Instant) {
        self.advance(now);
        self.flight[usize::from(!due)] += 1;
    }

    /// A piece of `bytes`, due when taken or not, ended at `now`.
    pub(crate) fn ended(&mut self, due: bool, bytes: u64, now: Instant) {
        self.advance(now);
        // Instants taken on several threads may reach the queue's lock out of
        // order: the latest end stands.
        self.last_ended = self.last_ended.max(Some(now));
        let flight = &mut self.flight[usize::from(!due)];
        *flight = flight.checked_sub(1).expect("an ended piece was in flight");
        let counted = self.totals.kind_mut(due);
        counted.bytes += bytes;
        counted.requests += 1;
    }

    /// The seconds a piece read ahead was in flight, to `now`.
    fn ahead(&mut self, now: Instant) -> f64 {
        self.advance(now);
        self.ahead
    }

    fn waited(&mut self, due: bool, waits: Waits) {
        let counted = &mut self.totals.kind_mut(due).waits;
        counted.requests += 1;
        counted.behind_ahead += waits.behind_ahead;
        counted.behind_due += waits.behind_due;
        counted.device += waits.device;
    }

    /// Count the integrals to `now`. Instants taken on several threads may
    /// reach the queue's lock out of order: one before the last counted
    /// adds nothing.
    fn advance(&mut self, now: Instant) {
        let seconds = now.saturating_duration_since(self.at).as_secs_f64();
        self.at = self.at.max(now);
        let pieces = self.flight.iter().sum::<usize>();
        if seconds == 0. || pieces == 0 {
            return;
        }
        self.totals.busy += seconds;
        self.totals.depth += pieces as f64 * seconds;
        for (due, flight) in [(true, self.flight[0]), (false, self.flight[1])] {
            if flight > 0 {
                let counted = self.totals.kind_mut(due);
                counted.busy += seconds;
                counted.depth += flight as f64 * seconds;
            }
        }
        if self.flight[1] > 0 {
            self.ahead += seconds;
        }
    }
}

/// One request's wait as its pieces are taken and end ([`Waits`]), kept by
/// the scheduler under its queue's lock beside its volume's book and
/// counted in it when its last piece has ended.
#[derive(Debug)]
pub(crate) struct Wait {
    /// Since when it waits or is served, and the volume's seconds of pieces
    /// read ahead then.
    since: Instant,
    ahead: f64,
    queued: usize,
    in_flight: usize,
    /// Whether its last piece taken was due.
    due: bool,
    waits: Waits,
}

impl Wait {
    /// A request of `pieces` pieces, due or not, of the volume `book`
    /// counts, submitted at `now`.
    pub(crate) fn new(pieces: usize, due: bool, book: &mut Book, now: Instant) -> Self {
        Self { since: now, ahead: book.ahead(now), queued: pieces, in_flight: 0, due, waits: Waits::default() }
    }

    /// The time waiting since it last was served (or submitted), to `now`.
    fn waited(&mut self, book: &mut Book, now: Instant) {
        let ahead = book.ahead(now);
        let seconds = now.saturating_duration_since(self.since).as_secs_f64();
        let behind = (ahead - self.ahead).clamp(0., seconds);
        self.waits.behind_ahead += behind;
        self.waits.behind_due += seconds - behind;
        self.since = now;
    }

    /// One of its pieces, due or not, was taken at `now`.
    pub(crate) fn taken(&mut self, due: bool, book: &mut Book, now: Instant) {
        if self.in_flight == 0 {
            self.waited(book, now);
        }
        self.queued -= 1;
        self.in_flight += 1;
        self.due = due;
    }

    /// One of its pieces in flight ended at `now`: whether that ended the
    /// job, its wait now counted in `book`.
    pub(crate) fn ended(&mut self, book: &mut Book, now: Instant) -> bool {
        self.in_flight -= 1;
        if self.in_flight > 0 {
            return false;
        }
        self.waits.device += now.saturating_duration_since(self.since).as_secs_f64();
        self.since = now;
        self.ahead = book.ahead(now);
        self.counted(book)
    }

    /// Its queued pieces were cancelled at `now`: whether that ended the
    /// request (none was in flight), its wait now counted in `book`.
    pub(crate) fn cancelled(&mut self, book: &mut Book, now: Instant) -> bool {
        self.queued = 0;
        if self.in_flight > 0 {
            return false;
        }
        self.waited(book, now);
        self.counted(book)
    }

    /// Count it in `book` once no piece is queued or in flight: whether it
    /// was.
    fn counted(&self, book: &mut Book) -> bool {
        let ended = self.queued == 0;
        if ended {
            book.waited(self.due, self.waits);
        }
        ended
    }
}
