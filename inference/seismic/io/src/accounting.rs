//! What each volume did, counted since the process began and never
//! consumed: every reader of a count takes the difference of two looks
//! ([`Totals::since`]), so readers never take counts from each other.
//!
//! Per volume: the seconds it held any piece in flight (its busy time) and
//! its pieces in flight integrated over time (its depth); per class
//! ([`Priority`]): the bytes and requests (pieces) that ended, read or
//! failed, the seconds a piece of the class was in flight and the class's
//! depth, and each ended job's wait. A job's wait, from its submission to
//! its last piece's end, is its volume time (any piece of it in flight)
//! and its time waiting (none in flight, one queued), which is split by
//! what the volume held meanwhile: behind speculative pieces (a
//! [`Priority::Prefetch`] or [`Priority::Load`] piece in flight), or behind
//! demand otherwise (only demand pieces in flight, or a worker not yet
//! woken). The busy time's rate is its bytes over its busy time
//! ([`Totals::rate`]), and the depth over the busy time its mean depth
//! while busy ([`Totals::depth_while_busy`]). The record is the same on
//! every backend: the scheduler keeps it, not a backend, in each volume's
//! book under its queue's lock, which it holds wherever a piece is taken
//! or ends, so counting takes no lock of its own; a look ([`totals`])
//! takes that lock once.

use crate::{Priority, Volume};
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
    /// Each class's pieces in flight.
    flight: [usize; 3],
    /// The seconds a speculative piece was in flight, which a waiting job's
    /// time behind speculative pieces is the difference of ([`Wait`]).
    speculative: f64,
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
    /// By class, in [`Priority::ALL`]'s order.
    pub classes: [Class; 3],
}

/// A class's counts on a volume ([`Totals`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Class {
    /// The bytes and pieces that ended, read or failed.
    pub bytes: u64,
    pub requests: u64,
    /// The seconds a piece of the class was in flight, and its pieces in
    /// flight integrated over time.
    pub busy: f64,
    pub depth: f64,
    /// The waits of its jobs that ended, each counted at its class when its
    /// last piece was taken.
    pub waits: Waits,
}

/// Jobs' waits from submission to their last piece's end, split as the
/// module documentation says.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Waits {
    pub jobs: u64,
    pub behind_speculative: f64,
    pub behind_demand: f64,
    pub device: f64,
}

impl Totals {
    pub fn class(&self, priority: Priority) -> &Class {
        &self.classes[priority as usize]
    }

    /// Every class's bytes.
    pub fn bytes(&self) -> u64 {
        self.classes.iter().map(|class| class.bytes).sum()
    }

    /// Every class's requests.
    pub fn requests(&self) -> u64 {
        self.classes.iter().map(|class| class.requests).sum()
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
                jobs: now.waits.jobs - then.waits.jobs,
                behind_speculative: now.waits.behind_speculative - then.waits.behind_speculative,
                behind_demand: now.waits.behind_demand - then.waits.behind_demand,
                device: now.waits.device - then.waits.device,
            },
        };
        Self {
            busy: self.busy - earlier.busy,
            depth: self.depth - earlier.depth,
            classes: std::array::from_fn(|at| class(&self.classes[at], &earlier.classes[at])),
        }
    }
}

impl Book {
    /// A volume's book, counted from `now`.
    pub(crate) fn new(now: Instant) -> Self {
        Self { at: now, flight: [0; 3], speculative: 0., totals: Totals::default() }
    }

    /// Its counts at `now`, the time running counted to then.
    pub(crate) fn totals_at(&mut self, now: Instant) -> Totals {
        self.advance(now);
        self.totals
    }

    /// The pieces that ended, of every class: no integral, so counted
    /// without advancing.
    pub(crate) fn requests(&self) -> u64 {
        self.totals.requests()
    }

    /// A piece of `class` was taken at `now`.
    pub(crate) fn began(&mut self, class: Priority, now: Instant) {
        self.advance(now);
        self.flight[class as usize] += 1;
    }

    /// A piece of `class` and `bytes` ended at `now`.
    pub(crate) fn ended(&mut self, class: Priority, bytes: u64, now: Instant) {
        self.advance(now);
        let flight = &mut self.flight[class as usize];
        *flight = flight.checked_sub(1).expect("an ended piece was in flight");
        let counted = &mut self.totals.classes[class as usize];
        counted.bytes += bytes;
        counted.requests += 1;
    }

    /// The seconds a speculative piece was in flight, to `now`.
    fn speculative(&mut self, now: Instant) -> f64 {
        self.advance(now);
        self.speculative
    }

    fn waited(&mut self, class: Priority, waits: Waits) {
        let counted = &mut self.totals.classes[class as usize].waits;
        counted.jobs += 1;
        counted.behind_speculative += waits.behind_speculative;
        counted.behind_demand += waits.behind_demand;
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
        for (class, &flight) in Priority::ALL.iter().zip(&self.flight) {
            if flight > 0 {
                let counted = &mut self.totals.classes[*class as usize];
                counted.busy += seconds;
                counted.depth += flight as f64 * seconds;
            }
        }
        if Priority::ALL.iter().zip(&self.flight).any(|(class, flight)| class.speculative() && *flight > 0) {
            self.speculative += seconds;
        }
    }
}

/// One job's wait as its pieces are taken and end ([`Waits`]), kept by the
/// scheduler under its queue's lock beside its volume's book and counted
/// in it when its last piece has ended.
#[derive(Debug)]
pub(crate) struct Wait {
    /// Since when it waits or is served, and the volume's speculative
    /// seconds then.
    since: Instant,
    speculative: f64,
    queued: usize,
    in_flight: usize,
    class: Priority,
    waits: Waits,
}

impl Wait {
    /// A job of `pieces` pieces at `class` of the volume `book` counts,
    /// submitted at `now`.
    pub(crate) fn new(pieces: usize, class: Priority, book: &mut Book, now: Instant) -> Self {
        Self { since: now, speculative: book.speculative(now), queued: pieces, in_flight: 0, class, waits: Waits::default() }
    }

    /// The time waiting since it last was served (or submitted), to `now`.
    fn waited(&mut self, book: &mut Book, now: Instant) {
        let speculative = book.speculative(now);
        let seconds = now.saturating_duration_since(self.since).as_secs_f64();
        let behind = (speculative - self.speculative).clamp(0., seconds);
        self.waits.behind_speculative += behind;
        self.waits.behind_demand += seconds - behind;
        self.since = now;
    }

    /// One of its pieces was taken at `now`, at `class`.
    pub(crate) fn taken(&mut self, class: Priority, book: &mut Book, now: Instant) {
        if self.in_flight == 0 {
            self.waited(book, now);
        }
        self.queued -= 1;
        self.in_flight += 1;
        self.class = class;
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
        self.speculative = book.speculative(now);
        self.counted(book)
    }

    /// Its queued pieces were withdrawn at `now`: whether that ended the
    /// job (none was in flight), its wait now counted in `book`.
    pub(crate) fn withdrawn(&mut self, book: &mut Book, now: Instant) -> bool {
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
            book.waited(self.class, self.waits);
        }
        ended
    }
}
