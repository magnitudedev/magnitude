//! What each volume did, counted since the process began and never
//! consumed: every reader of a count takes the difference of two looks
//! ([`Totals::since`]), so readers never take counts from each other.
//!
//! Per volume: the seconds it held any piece in flight (its busy time), its
//! pieces in flight integrated over time (its depth), and the seconds it
//! held due pieces and pieces read ahead at once (its shared time: in
//! decode the time reads ahead held the volume beside a parked read, the
//! most they delayed it, the knee bounding them, "Order and depth"); for
//! due pieces
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
//!
//! The counts are the one owner of a volume's drive line ([`Totals::line`]):
//! what reading an expert costs a step that waits on it, `λ + b/D`, fitted
//! from the volume's own busy time and its due requests' waits. Every
//! consumer of what a read takes (admission of a read ahead, a step's
//! drive time, a prompt group's size) reads this line, and none fits its
//! own.

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
    /// The seconds a due piece and a piece read ahead were in flight at
    /// once.
    pub shared: f64,
    /// Bytes of pieces that landed straight into their destination with no
    /// worker bounce buffer or copy.
    pub direct_bytes: u64,
    /// Bytes of pieces that went through a worker bounce buffer and copy
    /// (or failed).
    pub bounced_bytes: u64,
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
    /// The waits of the requests that ran at least one piece and ended,
    /// counted as due if any piece was taken as due (with `device` covering
    /// only that class's pieces).
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

/// A volume's drive line ([`Totals::line`]): a read of `b` bytes a step
/// waits on takes `latency + b · byte_seconds`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    pub latency: f64,
    pub byte_seconds: f64,
}

impl Line {
    /// The seconds a read of `bytes` a step waits on takes.
    pub fn seconds(&self, bytes: u64) -> f64 {
        self.latency + bytes as f64 * self.byte_seconds
    }
}

impl std::iter::Sum for Totals {
    /// Several volumes' counts as one, field by field: the counts of the
    /// reads a load makes across the volumes its files lie on.
    fn sum<I: Iterator<Item = Self>>(totals: I) -> Self {
        let class = |left: Class, right: Class| Class {
            bytes: left.bytes + right.bytes,
            requests: left.requests + right.requests,
            busy: left.busy + right.busy,
            depth: left.depth + right.depth,
            waits: Waits {
                requests: left.waits.requests + right.waits.requests,
                behind_ahead: left.waits.behind_ahead + right.waits.behind_ahead,
                behind_due: left.waits.behind_due + right.waits.behind_due,
                device: left.waits.device + right.waits.device,
            },
        };
        totals.fold(Self::default(), |left, right| Self {
            busy: left.busy + right.busy,
            depth: left.depth + right.depth,
            shared: left.shared + right.shared,
            direct_bytes: left.direct_bytes + right.direct_bytes,
            bounced_bytes: left.bounced_bytes + right.bounced_bytes,
            due: class(left.due, right.due),
            ahead: class(left.ahead, right.ahead),
        })
    }
}

impl Totals {
    /// The volume's drive line: its bytes at its own rate while busy
    /// ([`Self::rate`]: no request's time gives it, as a request's holds
    /// the bytes ahead of it), and of its latency what its due requests'
    /// time in flight took beyond their bytes at that rate (a request's
    /// time queued is the drive's other work, not its own), at least none.
    /// None before a due request ended or the volume was busy.
    pub fn line(&self) -> Option<Line> {
        let byte_seconds = self.rate().filter(|&rate| rate > 0.)?.recip();
        let waits = &self.due.waits;
        (waits.requests > 0).then(|| Line {
            latency: ((waits.device - self.due.bytes as f64 * byte_seconds) / waits.requests as f64).max(0.),
            byte_seconds,
        })
    }

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

    /// Fraction of ended bytes that went through a worker bounce buffer
    /// (`bounced_bytes / (direct_bytes + bounced_bytes)`), none before any
    /// byte ended.
    pub fn bounce_rate(&self) -> Option<f64> {
        let total = self.direct_bytes + self.bounced_bytes;
        (total > 0).then(|| self.bounced_bytes as f64 / total as f64)
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
            shared: self.shared - earlier.shared,
            direct_bytes: self.direct_bytes - earlier.direct_bytes,
            bounced_bytes: self.bounced_bytes - earlier.bounced_bytes,
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

    /// A piece of `bytes`, due when taken or not, ended at `now`, having
    /// landed directly (`direct == true`) or through a bounce buffer.
    pub(crate) fn ended(&mut self, due: bool, bytes: u64, direct: bool, now: Instant) {
        self.advance(now);
        // Instants taken on several threads may reach the queue's lock out of
        // order: the latest end stands.
        self.last_ended = self.last_ended.max(Some(now));
        let flight = &mut self.flight[usize::from(!due)];
        *flight = flight.checked_sub(1).expect("an ended piece was in flight");
        if direct {
            self.totals.direct_bytes += bytes;
        } else {
            self.totals.bounced_bytes += bytes;
        }
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
        if self.flight[0] > 0 && self.flight[1] > 0 {
            self.totals.shared += seconds;
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
    /// Whether at least one piece of this request was ever taken.
    taken: bool,
    /// Whether the currently in-flight piece(s) were taken as due.
    current_due: bool,
    /// Whether any piece of this request was taken as due.
    ever_due: bool,
    /// Device time in flight split by piece class (`[due, ahead]`), so
    /// `due.waits.device` and `due.bytes` always cover the exact same pieces
    /// even when a request is repriced from ahead to due mid-flight.
    device: [f64; 2],
    waits: Waits,
}

impl Wait {
    /// A request of `pieces` pieces, due or not, of the volume `book`
    /// counts, submitted at `now`.
    pub(crate) fn new(pieces: usize, due: bool, book: &mut Book, now: Instant) -> Self {
        Self {
            since: now,
            ahead: book.ahead(now),
            queued: pieces,
            in_flight: 0,
            taken: false,
            current_due: due,
            ever_due: false,
            device: [0.; 2],
            waits: Waits::default(),
        }
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
            self.current_due = due;
        } else if self.current_due != due {
            // Repriced while a piece of the previous class is still in flight:
            // close out the previous class's elapsed device window before
            // attributing subsequent flight time to the new class.
            self.device[usize::from(!self.current_due)] += now.saturating_duration_since(self.since).as_secs_f64();
            self.since = now;
            self.current_due = due;
        }
        self.queued -= 1;
        self.in_flight += 1;
        self.taken = true;
        self.ever_due |= due;
    }

    /// One of its pieces in flight ended at `now`: whether that ended the
    /// job, its wait now counted in `book`.
    pub(crate) fn ended(&mut self, book: &mut Book, now: Instant) -> bool {
        self.in_flight -= 1;
        if self.in_flight > 0 {
            return false;
        }
        self.device[usize::from(!self.current_due)] += now.saturating_duration_since(self.since).as_secs_f64();
        self.since = now;
        self.ahead = book.ahead(now);
        self.counted(book)
    }

    /// Its queued pieces were cancelled at `now`: whether that ended the
    /// request (none was in flight), its wait now counted in `book` if any
    /// piece ran.
    pub(crate) fn cancelled(&mut self, book: &mut Book, now: Instant) -> bool {
        self.queued = 0;
        if self.in_flight > 0 {
            return false;
        }
        if self.taken {
            self.waited(book, now);
        }
        self.counted(book)
    }

    /// Count it in `book` once no piece is queued or in flight: whether it
    /// was. A request cancelled before any piece was taken never ran on the
    /// volume and does not dilute `waits.requests`.
    fn counted(&self, book: &mut Book) -> bool {
        let ended = self.queued == 0;
        if ended && self.taken {
            let device = self.device[usize::from(!self.ever_due)];
            book.waited(self.ever_due, Waits { device, ..self.waits });
        }
        ended
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The line's rate is the volume's busy rate, and its latency what the
    /// due requests' time in flight took beyond their bytes at it, their
    /// time queued apart: 4 MB in 1 ms busy (4 GB/s), two due requests of
    /// 1 MB in flight 0.9 ms between them beyond their bytes' 0.5 ms,
    /// 0.2 ms each. None before a due request ended.
    #[test]
    fn a_volumes_line_is_its_busy_rate_and_its_due_waits_beyond_it() {
        let waits = Waits { requests: 2, behind_ahead: 0.1e-3, behind_due: 0.2e-3, device: 0.9e-3 };
        let due = Class { bytes: 2_000_000, requests: 8, waits, ..Class::default() };
        let ahead = Class { bytes: 2_000_000, requests: 8, ..Class::default() };
        let totals = Totals { busy: 1e-3, due, ahead, ..Totals::default() };
        let line = totals.line().expect("a line");
        assert!((line.byte_seconds - 0.25e-9).abs() < 1e-18, "{line:?}");
        assert!((line.latency - 0.2e-3).abs() < 1e-12, "{line:?}");
        assert!((line.seconds(1_000_000) - 0.45e-3).abs() < 1e-12);
        assert_eq!(Totals { due: Class { waits: Waits::default(), ..due }, ..totals }.line(), None);
        assert_eq!([totals, totals].into_iter().sum::<Totals>().line(), Some(line), "two like volumes read as one");
    }

    /// A due request cancelled before any piece is taken must not increment
    /// `due.waits.requests` or dilute `Totals::line().latency` toward zero.
    #[test]
    fn zero_piece_cancelled_due_request_does_not_dilute_line() {
        let at = Instant::now();
        let ms = |ms: u64| at + Duration::from_millis(ms);
        let mut book = Book::new(at);

        // One ahead piece + one due piece in flight together for 1 ms:
        // 2 MB in 1 ms busy => rate 2 GB/s (byte_seconds = 0.5e-9),
        // due.bytes = 1 MB, due.waits.device = 1 ms => latency = 0.5 ms.
        book.began(false, ms(0));
        let mut due_wait = Wait::new(1, true, &mut book, ms(0));
        book.began(true, ms(0));
        due_wait.taken(true, &mut book, ms(0));
        book.ended(false, 1_000_000, true, ms(1));
        book.ended(true, 1_000_000, true, ms(1));
        assert!(due_wait.ended(&mut book, ms(1)));

        let before = book.totals_at(ms(1));
        let before_line = before.line().expect("a line");
        assert_eq!(before.due.waits.requests, 1);
        assert!((before_line.latency - 0.5e-3).abs() < 1e-12, "{before_line:?}");

        // Cancel 9 due requests before any piece is taken.
        for _ in 0..9 {
            let mut cancelled = Wait::new(2, true, &mut book, ms(1));
            assert!(cancelled.cancelled(&mut book, ms(2)));
        }

        let after = book.totals_at(ms(2));
        let after_line = after.line().expect("a line");
        assert_eq!(after.due.waits.requests, 1, "zero-piece cancelled requests must not be counted");
        assert!((after_line.latency - before_line.latency).abs() < 1e-12, "{after_line:?} vs {before_line:?}");
    }

    /// A request whose first piece runs as `Ahead` and is then repriced to
    /// `Due` for its second piece must not attribute the ahead piece's device
    /// time to `due.waits.device` (which would inflate `Totals::line().latency`).
    #[test]
    fn ahead_then_repriced_to_due_does_not_inflate_due_device_wait() {
        let at = Instant::now();
        let ms = |ms: u64| at + Duration::from_millis(ms);
        let mut book = Book::new(at);

        // Sequential repricing: piece 1 runs as Ahead (1 MB in 1 ms),
        // piece 2 runs as Due (1 MB in 1 ms).
        let mut wait = Wait::new(2, false, &mut book, ms(0));
        book.began(false, ms(0));
        wait.taken(false, &mut book, ms(0));
        book.ended(false, 1_000_000, false, ms(1));
        assert!(!wait.ended(&mut book, ms(1)));

        book.began(true, ms(1));
        wait.taken(true, &mut book, ms(1));
        book.ended(true, 1_000_000, true, ms(2));
        assert!(wait.ended(&mut book, ms(2)));

        let totals = book.totals_at(ms(2));
        assert_eq!(totals.ahead.bytes, 1_000_000);
        assert_eq!(totals.due.bytes, 1_000_000);
        assert_eq!(totals.due.waits.requests, 1);
        assert!(
            (totals.due.waits.device - 1e-3).abs() < 1e-12,
            "due.waits.device should count only the due piece (1 ms), got {:?}",
            totals.due.waits.device
        );
        let line = totals.line().expect("a line");
        assert!((line.byte_seconds - 1e-9).abs() < 1e-18, "{line:?}");
        assert!(line.latency.abs() < 1e-12, "ahead piece must not inflate latency: {line:?}");
    }
}
