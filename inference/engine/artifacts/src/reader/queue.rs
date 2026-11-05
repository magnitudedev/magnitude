use super::file::SourceFile;
use super::job::{first_failures, JobState};
use super::volume::{Delivery, Volume, VolumeKey};
use super::{Ended, Priority, Range, ALIGNMENT, COALESCE_GAP};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

/// An aligned read `[start, end)` and the parts of ranges it serves.
#[derive(Debug)]
pub(super) struct Piece {
    pub(super) start: u64,
    pub(super) end: u64,
    pub(super) parts: Vec<Part>,
}

/// Bytes of one range a piece serves: `length` bytes at `from` in the
/// piece, landing at `sink`, of range `range` of the job.
#[derive(Clone, Copy, Debug)]
pub(super) struct Part {
    pub(super) range: usize,
    pub(super) from: usize,
    pub(super) length: usize,
    pub(super) sink: NonNull<u8>,
}

// SAFETY: a part's sink is memory its job's submitter handed the pool until
// the job ends (`ArtifactReader::submit`), written by the one worker that
// takes the part's piece.
unsafe impl Send for Part {}

/// The pieces serving `ranges` in ascending `order`: the aligned reads
/// coalesced as the module documentation says, each split into pieces of
/// at most `read` bytes (its volume's read size). A read's end may pass
/// the file's end by less than a block.
pub(super) fn pieces(ranges: &[Range], order: &[usize], size: u64, read: u64) -> Vec<Piece> {
    let aligned = |offset: u64, length: usize| {
        let start = offset - offset % ALIGNMENT;
        let end = (offset + length as u64).div_ceil(ALIGNMENT) * ALIGNMENT;
        (start, end.max(start + ALIGNMENT).min(size.div_ceil(ALIGNMENT) * ALIGNMENT))
    };
    // Each read: its span and the ranges it serves.
    let mut reads: Vec<(u64, u64, Vec<usize>)> = Vec::new();
    for index in order {
        let range = ranges[*index];
        let (start, end) = aligned(range.offset, range.length);
        match reads.last_mut() {
            Some(read) if start <= read.1 + COALESCE_GAP => {
                read.1 = read.1.max(end);
                read.2.push(*index);
            }
            _ => reads.push((start, end, vec![*index])),
        }
    }
    let mut pieces = Vec::new();
    for (start, end, served) in reads {
        let mut at = start;
        while at < end {
            let stop = (at + read).min(end);
            let parts = served
                .iter()
                .filter_map(|index| {
                    let range = ranges[*index];
                    let from = range.offset.max(at);
                    let to = (range.offset + range.length as u64).min(stop);
                    (from < to || (range.length == 0 && (at..stop).contains(&range.offset))).then(
                        || Part {
                            range: *index,
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
pub(super) struct Queued {
    pub(super) state: Arc<JobState>,
    pub(super) file: Arc<SourceFile>,
    pub(super) volume: Volume,
    pub(super) pieces: VecDeque<Piece>,
}

/// A piece a worker took, with its job, file and volume.
pub(super) type Taken = (Arc<JobState>, Arc<SourceFile>, Volume, Piece);

/// The jobs with pieces not yet taken, most urgent first: by priority, then
/// by submission; and each volume's reads in flight.
#[derive(Default)]
pub(super) struct Queue {
    jobs: BTreeMap<(Priority, u64), Queued>,
    /// Each queued job's priority, by its submission.
    priorities: HashMap<u64, Priority>,
    submitted: u64,
    /// Each volume's reads in flight, of a volume with any.
    pub(super) in_flight: HashMap<VolumeKey, VolumeFlight>,
}

/// A volume's reads in flight while it holds any: the bytes workers hold,
/// at most its window (or one piece larger), whether a piece waits for them
/// to fall, and the volume's delivery, busy from the first of them taken
/// until the last ends ([`Delivery`]).
pub(super) struct VolumeFlight {
    pub(super) bytes: u64,
    pub(super) held_back: bool,
    pub(super) delivery: Arc<Delivery>,
}

impl Queue {
    /// Queue `job` at `priority`, ending it at once when it has no piece:
    /// its submission.
    pub(super) fn push(&mut self, job: Queued, priority: Priority) -> (u64, usize) {
        self.submitted += 1;
        let id = self.submitted;
        if job.pieces.is_empty() {
            let failed = std::mem::take(&mut job.state.progress.lock().expect("job progress").failed);
            job.state.end(Ended::Read {
                failed: first_failures(failed),
            });
            return (id, 0);
        }
        let woken = self.room(&job, priority);
        self.priorities.insert(id, priority);
        self.jobs.insert((priority, id), job);
        (id, woken)
    }

    /// The pieces of `job`, in order, its volume's window at `priority` has room for
    /// beside the bytes in flight: at least one where nothing is (a piece
    /// past the window is taken alone), none where the window is full.
    pub(super) fn room(&self, job: &Queued, priority: Priority) -> usize {
        let held = self.in_flight.get(&job.volume.key).map_or(0, |flight| flight.bytes);
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

    /// The most urgent piece whose volume holds it within its window (or
    /// holds nothing), now in flight. A volume's most urgent piece that
    /// would pass its window holds back every other of the volume's, so no
    /// less urgent piece takes the room it waits for.
    pub(super) fn take(&mut self) -> Option<Taken> {
        let mut blocked = Vec::new();
        let mut chosen = None;
        for (key, job) in &self.jobs {
            let volume = &job.volume;
            if blocked.contains(&volume.key) {
                continue;
            }
            let piece = job.pieces.front().expect("a queued job has a piece");
            let held = self.in_flight.get(&volume.key).map_or(0, |flight| flight.bytes);
            if held == 0 || held + (piece.end - piece.start) <= volume.window_for(key.0) {
                chosen = Some(*key);
                break;
            }
            blocked.push(volume.key);
        }
        for volume in blocked {
            self.in_flight.get_mut(&volume).expect("a volume holding bytes").held_back = true;
        }
        let key = chosen?;
        let job = self.jobs.get_mut(&key).expect("the chosen job");
        let piece = job.pieces.pop_front().expect("a queued job has a piece");
        let now = Instant::now();
        let flight = self.in_flight.entry(job.volume.key).or_insert_with(|| {
            job.volume.delivery.began(now);
            VolumeFlight { bytes: 0, held_back: false, delivery: job.volume.delivery.clone() }
        });
        flight.bytes += piece.end - piece.start;
        job.state.started.get_or_init(|| now);
        let taken = (job.state.clone(), job.file.clone(), job.volume.clone(), piece);
        if job.pieces.is_empty() {
            self.jobs.remove(&key);
            self.priorities.remove(&key.1);
        }
        Some(taken)
    }

    /// A piece of `bytes` of `volume` ended at `now`, counted in the
    /// volume's delivery: whether a piece of the volume was held back, which
    /// may now be taken.
    pub(super) fn ended(&mut self, volume: VolumeKey, bytes: u64, now: Instant) -> bool {
        let flight = self.in_flight.get_mut(&volume).expect("an ended piece was in flight");
        flight.bytes -= bytes;
        let held_back = std::mem::take(&mut flight.held_back);
        let idle = flight.bytes == 0;
        flight.delivery.ended(bytes, idle, now);
        if idle {
            self.in_flight.remove(&volume);
        }
        held_back
    }

    /// Read job `id` no less urgently than `priority`, its pieces joining
    /// the bytes in flight as that priority's do: the workers its wider
    /// window has room for, to wake.
    pub(super) fn promote(&mut self, id: u64, priority: Priority) -> usize {
        let Some(current) = self.priorities.get_mut(&id) else {
            return 0;
        };
        if priority >= *current {
            return 0;
        }
        let job = self.jobs.remove(&(*current, id)).expect("a queued job");
        *current = priority;
        let woken = self.room(&job, priority);
        self.jobs.insert((priority, id), job);
        woken
    }

    /// Withdraw job `id`: its pieces no worker took, none where every one
    /// was.
    pub(super) fn withdraw(&mut self, id: u64) -> Option<Queued> {
        let priority = self.priorities.remove(&id)?;
        self.jobs.remove(&(priority, id))
    }
}
