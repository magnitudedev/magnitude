use super::job::{first_failures, JobState};
use super::{Ended, Priority, COALESCE_GAP};
use crate::accounting::{Book, Wait};
use crate::disk::{SourceFile, VolumeReads};
use crate::{Volume, ALIGNMENT};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

/// An aligned read `[start, end)` and the parts of ranges it serves.
#[derive(Debug)]
pub(crate) struct Piece {
    pub(crate) start: u64,
    pub(crate) end: u64,
    pub(crate) parts: Vec<Part>,
}

/// Bytes of one range a piece serves: `length` bytes at `from` in the
/// piece, landing at `sink`, of range `range` of the job.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Part {
    pub(crate) range: usize,
    pub(crate) from: usize,
    pub(crate) length: usize,
    pub(crate) sink: NonNull<u8>,
}

impl Piece {
    /// Each part as a piece of its own: the aligned span of its own bytes, within this piece's,
    /// so a read that fails for this piece is told apart by part.
    pub(crate) fn parts_alone(&self) -> impl Iterator<Item = Piece> + '_ {
        self.parts.iter().map(|part| {
            let offset = self.start + part.from as u64;
            let start = offset - offset % ALIGNMENT;
            let end = (offset + part.length as u64).div_ceil(ALIGNMENT) * ALIGNMENT;
            Piece {
                start,
                end: end.max(start + ALIGNMENT).min(self.end),
                parts: vec![Part { from: (offset - start) as usize, ..*part }],
            }
        })
    }
}

/// A range of a job as its pieces are cut: its destination's bytes at `sink`
/// ([`crate::Range`]), which its job holds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Span {
    pub(crate) offset: u64,
    pub(crate) length: usize,
    pub(crate) sink: NonNull<u8>,
}

// SAFETY: a part's sink is in a destination its job holds until the job's
// last piece ends (`Reader::submit`), written by the one worker that takes
// the part's piece.
unsafe impl Send for Part {}

/// The pieces serving `ranges`, each a range's index in its job and its
/// span, in ascending offset: the aligned reads coalesced as the module
/// documentation says, each split into pieces of at most `read` bytes (its
/// volume's read size). A read's end may pass the file's end by less than
/// a block.
pub(crate) fn pieces(ranges: &[(usize, Span)], size: u64, read: u64) -> Vec<Piece> {
    let aligned = |offset: u64, length: usize| {
        let start = offset - offset % ALIGNMENT;
        let end = (offset + length as u64).div_ceil(ALIGNMENT) * ALIGNMENT;
        (start, end.max(start + ALIGNMENT).min(size.div_ceil(ALIGNMENT) * ALIGNMENT))
    };
    // Each read: its span and the ranges it serves, by their place in
    // `ranges`.
    let mut reads: Vec<(u64, u64, Vec<usize>)> = Vec::new();
    for (at, (_, range)) in ranges.iter().enumerate() {
        let (start, end) = aligned(range.offset, range.length);
        match reads.last_mut() {
            Some(read) if start <= read.1 + COALESCE_GAP => {
                read.1 = read.1.max(end);
                read.2.push(at);
            }
            _ => reads.push((start, end, vec![at])),
        }
    }
    let mut pieces = Vec::new();
    for (start, end, served) in reads {
        let mut at = start;
        while at < end {
            let stop = (at + read).min(end);
            let parts = served
                .iter()
                .filter_map(|served| {
                    let (index, range) = ranges[*served];
                    let from = range.offset.max(at);
                    let to = (range.offset + range.length as u64).min(stop);
                    (from < to || (range.length == 0 && (at..stop).contains(&range.offset))).then(
                        || Part {
                            range: index,
                            from: (from - at) as usize,
                            length: to.saturating_sub(from) as usize,
                            // SAFETY: within the range's sink, `from - offset`
                            // bytes in.
                            sink: unsafe {
                                range.sink.add((from - range.offset) as usize)
                            },
                        },
                    )
                })
                .collect::<Vec<_>>();
            if !parts.is_empty() {
                pieces.push(Piece {
                    start: at,
                    end: stop,
                    parts,
                });
            }
            at = stop;
        }
    }
    pieces
}

/// A job's pieces not yet taken.
pub(crate) struct Queued {
    pub(crate) state: Arc<JobState>,
    pub(crate) file: Arc<SourceFile>,
    pub(crate) volume: VolumeReads,
    pub(crate) pieces: VecDeque<Piece>,
}

/// A piece a worker took, with its job (by its submission, and its state),
/// file and volume, and the class it was taken at.
pub(crate) struct Taken {
    pub(crate) id: u64,
    pub(crate) state: Arc<JobState>,
    pub(crate) file: Arc<SourceFile>,
    pub(crate) volume: VolumeReads,
    pub(crate) piece: Piece,
    pub(crate) class: Priority,
}

impl Taken {
    /// This piece, its read returned at `at`, to end ([`Queue::ended`]).
    pub(crate) fn returned(&self, at: Instant) -> Returned {
        Returned { id: self.id, volume: self.volume.id, class: self.class, bytes: self.piece.end - self.piece.start, at }
    }
}

/// A taken piece whose read returned at `at`: what ending it counts.
#[derive(Clone, Copy)]
pub(crate) struct Returned {
    id: u64,
    volume: Volume,
    class: Priority,
    bytes: u64,
    at: Instant,
}

/// The jobs with pieces not yet taken, most urgent first: by priority, then
/// by submission; each volume's reads in flight and its book; and each
/// job's wait. Every count of a piece taken or ended is kept here, under
/// the lock the queue is held by wherever one is, so none takes a lock of
/// its own.
#[derive(Default)]
pub(crate) struct Queue {
    jobs: BTreeMap<(Priority, u64), Queued>,
    /// Each queued job's priority, by its submission.
    priorities: HashMap<u64, Priority>,
    submitted: u64,
    /// Each volume the queue has held a job of.
    pub(crate) volumes: HashMap<Volume, VolumeFlight>,
    /// The wait of each job with a piece queued or in flight, by its
    /// submission.
    waits: HashMap<u64, Wait>,
}

/// A volume's reads in flight: the bytes workers hold, at most its window
/// (or one piece larger), and whether a piece waits for them to fall; and
/// its book ([`crate::accounting`]).
pub(crate) struct VolumeFlight {
    pub(crate) bytes: u64,
    pub(crate) held_back: bool,
    pub(crate) book: Book,
}

impl VolumeFlight {
    /// A volume holding nothing, counted from `now`.
    pub(crate) fn new(now: Instant) -> Self {
        Self { bytes: 0, held_back: false, book: Book::new(now) }
    }
}

/// The pieces of `job`, in order, its volume's window at `priority` has
/// room for beside `held` bytes in flight: at least one where nothing is (a
/// piece past the window is taken alone), none where the window is full.
fn room(held: u64, job: &Queued, priority: Priority) -> usize {
    let mut fits = 0;
    let mut bytes = held;
    for piece in &job.pieces {
        bytes += piece.end - piece.start;
        if bytes > job.volume.window_for(priority) && !(held == 0 && fits == 0) {
            break;
        }
        fits += 1;
    }
    fits
}

/// Change job `id`'s wait by `change`, removed once `change` says it has
/// been counted: a job with a piece queued or in flight is waited.
fn settle(waits: &mut HashMap<u64, Wait>, id: u64, change: impl FnOnce(&mut Wait) -> bool) {
    let Entry::Occupied(mut wait) = waits.entry(id) else {
        panic!("job {id} has a piece queued or in flight and no wait");
    };
    if change(wait.get_mut()) {
        wait.remove();
    }
}

impl Queue {
    /// Queue `job` at `priority`, ending it at once when it has no piece:
    /// its submission.
    pub(crate) fn push(&mut self, job: Queued, priority: Priority) -> (u64, usize) {
        self.submitted += 1;
        let id = self.submitted;
        if job.pieces.is_empty() {
            let failed = std::mem::take(&mut job.state.progress.lock().expect("job progress").failed);
            job.state.end(Ended::Read {
                failed: first_failures(failed),
            });
            return (id, 0);
        }
        let now = Instant::now();
        let volume = self.volumes.entry(job.volume.id).or_insert_with(|| VolumeFlight::new(now));
        self.waits.insert(id, Wait::new(job.pieces.len(), priority, &mut volume.book, now));
        let woken = room(volume.bytes, &job, priority);
        if crate::timeline::recording() {
            let bytes = job.pieces.iter().map(|piece| piece.end - piece.start).sum::<u64>();
            crate::timeline::note(crate::timeline::Kind::Submit, id, (priority as i64, job.pieces.len() as i64, bytes as i64));
        }
        self.priorities.insert(id, priority);
        self.jobs.insert((priority, id), job);
        (id, woken)
    }

    /// The pieces of `job`, in order, its volume's window at `priority` has
    /// room for beside the bytes in flight ([`room`]).
    pub(crate) fn room(&self, job: &Queued, priority: Priority) -> usize {
        room(self.volumes.get(&job.volume.id).map_or(0, |flight| flight.bytes), job, priority)
    }

    /// The most urgent piece whose volume holds it within its window (or
    /// holds nothing), now in flight. A volume's most urgent piece that
    /// would pass its window holds back every other of the volume's, so no
    /// less urgent piece takes the room it waits for.
    pub(crate) fn take(&mut self) -> Option<Taken> {
        let mut blocked = Vec::new();
        let mut chosen = None;
        for (key, job) in &self.jobs {
            let volume = &job.volume;
            if blocked.contains(&volume.id) {
                continue;
            }
            let piece = job.pieces.front().expect("a queued job has a piece");
            let held = self.volumes.get(&volume.id).map_or(0, |flight| flight.bytes);
            if held == 0 || held + (piece.end - piece.start) <= volume.window_for(key.0) {
                chosen = Some(*key);
                break;
            }
            blocked.push(volume.id);
        }
        for volume in blocked {
            self.volumes.get_mut(&volume).expect("a volume holding bytes").held_back = true;
        }
        let key = chosen?;
        let job = self.jobs.get_mut(&key).expect("the chosen job");
        let piece = job.pieces.pop_front().expect("a queued job has a piece");
        let now = Instant::now();
        let flight = self.volumes.get_mut(&job.volume.id).expect("a queued job's volume");
        crate::timeline::note_at(now, crate::timeline::Kind::Take, key.1, (piece.start as i64, (piece.end - piece.start) as i64, flight.bytes as i64));
        flight.bytes += piece.end - piece.start;
        flight.book.began(key.0, now);
        self.waits.get_mut(&key.1).expect("a queued job is waited").taken(key.0, &mut flight.book, now);
        job.state.started.get_or_init(|| now);
        let taken = Taken { id: key.1, state: job.state.clone(), file: job.file.clone(), volume: job.volume.clone(), piece, class: key.0 };
        if job.pieces.is_empty() {
            self.jobs.remove(&key);
            self.priorities.remove(&key.1);
        }
        Some(taken)
    }

    /// `returned`'s piece ended as its read returned, counted in its
    /// volume's book and its job's wait: whether a piece of the volume was
    /// held back, which may now be taken.
    pub(crate) fn ended(&mut self, returned: Returned) -> bool {
        let Returned { id, volume, class, bytes, at } = returned;
        let flight = self.volumes.get_mut(&volume).expect("an ended piece was in flight");
        flight.bytes -= bytes;
        flight.book.ended(class, bytes, at);
        settle(&mut self.waits, id, |wait| wait.ended(&mut flight.book, at));
        std::mem::take(&mut flight.held_back)
    }

    /// Read job `id` no less urgently than `priority`, its pieces joining
    /// the bytes in flight as that priority's do: the workers its wider
    /// window has room for, to wake.
    pub(crate) fn promote(&mut self, id: u64, priority: Priority) -> usize {
        let Some(current) = self.priorities.get_mut(&id) else {
            return 0;
        };
        if priority >= *current {
            return 0;
        }
        let job = self.jobs.remove(&(*current, id)).expect("a queued job");
        *current = priority;
        crate::timeline::note(crate::timeline::Kind::Promote, id, (priority as i64, job.pieces.len() as i64, 0));
        let woken = self.room(&job, priority);
        self.jobs.insert((priority, id), job);
        woken
    }

    /// Withdraw job `id`: its pieces no worker took, none where every one
    /// was.
    pub(crate) fn withdraw(&mut self, id: u64) -> Option<Queued> {
        let priority = self.priorities.remove(&id)?;
        let job = self.jobs.remove(&(priority, id))?;
        crate::timeline::note(crate::timeline::Kind::Withdraw, id, (job.pieces.len() as i64, 0, 0));
        let flight = self.volumes.get_mut(&job.volume.id).expect("a queued job's volume");
        settle(&mut self.waits, id, |wait| wait.withdrawn(&mut flight.book, Instant::now()));
        Some(job)
    }
}
