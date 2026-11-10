use super::engine::Need;
use super::handle::{first_failures, RequestState};
use super::{Deadline, Ended, Standing, Want, COALESCE_GAP};
use crate::accounting::{Book, Wait};
use crate::disk::{Shape, SourceFile, VolumeReads};
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
/// piece, landing at `sink`, of range `range` of the request.
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

/// A range of a request as its pieces are cut: its destination's bytes at
/// `sink` ([`crate::Range`]), which its request holds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Span {
    pub(crate) offset: u64,
    pub(crate) length: usize,
    pub(crate) sink: NonNull<u8>,
}

// SAFETY: a part's sink is in a destination its request holds until the
// request's last piece ends (`Reader::request`), written by the one worker
// that takes the part's piece.
unsafe impl Send for Part {}

/// The pieces serving `ranges`, each a range's index in its request and its
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

/// A request's pieces not yet taken, and its claim on the pool's workers,
/// held while any of them is queued or in flight.
pub(crate) struct Queued {
    pub(crate) state: Arc<RequestState>,
    pub(crate) file: Arc<SourceFile>,
    pub(crate) volume: VolumeReads,
    pub(crate) need: Arc<Need>,
    pub(crate) pieces: VecDeque<Piece>,
}

/// A piece a worker took, with its request (by its submission, its state
/// and its claim on the pool), file and volume, and how its request stood
/// when it was taken.
pub(crate) struct Taken {
    pub(crate) id: u64,
    pub(crate) state: Arc<RequestState>,
    pub(crate) file: Arc<SourceFile>,
    pub(crate) volume: VolumeReads,
    pub(crate) need: Arc<Need>,
    pub(crate) piece: Piece,
    pub(crate) standing: Standing,
}

impl Taken {
    /// This piece, its read returned at `at`, to end ([`Queue::ended`]).
    pub(crate) fn returned(&self, at: Instant) -> Returned {
        Returned { id: self.id, volume: self.volume.id, standing: self.standing, bytes: self.piece.end - self.piece.start, at, need: self.need.clone() }
    }
}

/// A taken piece whose read returned at `at`: what ending it counts, its
/// claim's among it.
pub(crate) struct Returned {
    id: u64,
    volume: Volume,
    standing: Standing,
    bytes: u64,
    at: Instant,
    need: Arc<Need>,
}

/// Every volume's link, and each request's wait. Every count of a piece
/// taken or ended is kept here, under the lock the queue is held by
/// wherever one is, so none takes a lock of its own.
#[derive(Default)]
pub(crate) struct Queue {
    submitted: u64,
    /// The pieces taken and not yet ended, on every link.
    taken: usize,
    /// Each volume the queue has held a request of.
    pub(crate) links: HashMap<Volume, Link>,
    /// Each queued request's volume and want, by its submission.
    wants: HashMap<u64, (Volume, Want)>,
    /// The wait of each request with a piece queued or in flight, by its
    /// submission.
    waits: HashMap<u64, Wait>,
}

/// A volume's link, the one authority on what of it is in flight ("Order
/// and depth"): its requests with pieces not yet taken, most urgent first
/// (by [`Want`], then by submission), those a stream consumes apart from
/// the rest, so the due ones of each lead; the bytes in flight of pieces by
/// how they stood when taken; and its book ([`crate::accounting`]).
pub(crate) struct Link {
    requests: BTreeMap<(Want, u64), Queued>,
    streamed: BTreeMap<(Want, u64), Queued>,
    pub(crate) flight: InFlight,
    pub(crate) book: Book,
}

/// A link's bytes in flight, by how their pieces stood when taken.
#[derive(Clone, Copy, Default)]
pub(crate) struct InFlight {
    pub(crate) parked: u64,
    pub(crate) streamed: u64,
    pub(crate) ahead: u64,
}

impl InFlight {
    fn of(&mut self, standing: Standing) -> &mut u64 {
        match standing {
            Standing::Parked => &mut self.parked,
            Standing::Streamed => &mut self.streamed,
            Standing::Ahead => &mut self.ahead,
        }
    }

    /// Every byte in flight.
    pub(crate) fn total(&self) -> u64 {
        self.parked + self.streamed + self.ahead
    }
}

impl Link {
    /// A link holding nothing, counted from `now`.
    fn new(now: Instant) -> Self {
        Self { requests: BTreeMap::new(), streamed: BTreeMap::new(), flight: InFlight::default(), book: Book::new(now) }
    }

    fn map(&mut self, want: &Want) -> &mut BTreeMap<(Want, u64), Queued> {
        match want.deadline() {
            Deadline::Streamed(_) => &mut self.streamed,
            Deadline::Parked(_) | Deadline::Expected(_) | Deadline::None => &mut self.requests,
        }
    }

    /// The request whose next piece may start now, and how it stands:
    /// the more urgent of the two maps' due heads, where the due bytes in
    /// flight leave it room; else, where no parked piece is queued, the
    /// head read ahead, where the bytes in flight it is counted against
    /// leave it room ([`admits`]). A due head held back at the due depth
    /// holds back the due pieces behind it, a parked one everything read
    /// ahead too; a streamed one holds back nothing read ahead.
    fn next(&self) -> Option<((Want, u64), Standing)> {
        let head = self.requests.first_key_value().map(|(key, request)| (*key, request, key.0.standing()));
        let streamed = self.streamed.first_key_value().map(|(key, request)| (*key, request, key.0.standing()));
        let due = [head, streamed].into_iter().flatten().filter(|(_, _, standing)| standing.due()).min_by_key(|(key, ..)| *key);
        if let Some((key, request, standing)) = due {
            if self.admits(request, standing) {
                return Some((key, standing));
            }
        }
        match head {
            Some((key, request, Standing::Ahead)) => self.admits(request, Standing::Ahead).then_some((key, Standing::Ahead)),
            _ => None,
        }
    }

    fn admits(&self, request: &Queued, standing: Standing) -> bool {
        let piece = request.pieces.front().expect("a queued request has a piece");
        admits(self.flight, &request.volume.shape, piece.end - piece.start, standing)
    }

    /// Whether a piece of it may start now.
    fn ready(&self) -> bool {
        self.next().is_some()
    }

    /// How many pieces, in order, of the request [`Self::next`] names, then
    /// of the head read ahead where nothing parked is queued, may start
    /// now beside what is in flight: the workers to wake for them.
    fn room(&self) -> usize {
        let Some((key, standing)) = self.next() else {
            return 0;
        };
        let mut flight = self.flight;
        let mut fits = 0;
        let mut count = |request: &Queued, standing: Standing| {
            for piece in &request.pieces {
                let bytes = piece.end - piece.start;
                if !admits(flight, &request.volume.shape, bytes, standing) {
                    break;
                }
                *flight.of(standing) += bytes;
                fits += 1;
            }
        };
        let request = self.requests.get(&key).or_else(|| self.streamed.get(&key)).expect("the request named");
        count(request, standing);
        if standing == Standing::Streamed {
            if let Some((_, ahead)) = self.requests.first_key_value().filter(|(key, _)| key.0.standing() == Standing::Ahead) {
                count(ahead, Standing::Ahead);
            }
        }
        fits
    }
}

/// Whether a piece of `bytes`, standing `standing`, may start beside
/// `flight` by `shape`'s depths ("Order and depth"): a due piece while the
/// due bytes in flight, parked and streamed, leave it room under the due
/// depth; a piece read ahead (the caller's to know that nothing parked
/// waits) while the parked bytes and those read ahead leave it room under
/// the knee: a stream's bytes are its own, at the rate the stream takes.
/// A piece past its depth starts where nothing it is counted against is in
/// flight.
fn admits(flight: InFlight, shape: &Shape, bytes: u64, standing: Standing) -> bool {
    let (counted, depth) = match standing {
        Standing::Parked | Standing::Streamed => (flight.parked + flight.streamed, shape.due),
        Standing::Ahead => (flight.parked + flight.ahead, shape.knee),
    };
    counted == 0 || counted + bytes <= depth
}

/// Change request `id`'s wait by `change`, removed once `change` says it
/// has been counted: a request with a piece queued or in flight is waited.
fn settle(waits: &mut HashMap<u64, Wait>, id: u64, change: impl FnOnce(&mut Wait) -> bool) {
    let Entry::Occupied(mut wait) = waits.entry(id) else {
        panic!("request {id} has a piece queued or in flight and no wait");
    };
    if change(wait.get_mut()) {
        wait.remove();
    }
}

impl Queue {
    /// Queue `request` on its volume's link for `want`, ending it at once
    /// when it has no piece: its submission, and the workers to wake for
    /// the pieces its link may start now.
    pub(crate) fn push(&mut self, request: Queued, want: Want) -> (u64, usize) {
        self.submitted += 1;
        let id = self.submitted;
        if request.pieces.is_empty() {
            let failed = std::mem::take(&mut request.state.progress.lock().expect("request progress").failed);
            request.state.end(Ended::Read {
                failed: first_failures(failed),
            });
            return (id, 0);
        }
        let now = Instant::now();
        let volume = request.volume.id;
        let link = self.links.entry(volume).or_insert_with(|| Link::new(now));
        self.waits.insert(id, Wait::new(request.pieces.len(), want.due(), &mut link.book, now));
        if crate::timeline::recording() {
            let bytes = request.pieces.iter().map(|piece| piece.end - piece.start).sum::<u64>();
            crate::timeline::note(crate::timeline::Kind::Submit, id, (i64::from(!want.due()), request.pieces.len() as i64, bytes as i64));
        }
        self.wants.insert(id, (volume, want));
        link.map(&want).insert((want, id), request);
        (id, link.room())
    }

    /// The most urgent piece of every link's that its link may start now
    /// ([`admits`]), its request the most urgent of each link's
    /// most urgent: now in flight. A link's most urgent request whose next
    /// piece may not start holds back every other of the link's, so none
    /// takes the room it waits for, and none read ahead starts while one
    /// due waits.
    pub(crate) fn take(&mut self) -> Option<Taken> {
        let now = Instant::now();
        let (volume, key, standing) = self
            .links
            .iter()
            .filter_map(|(volume, link)| link.next().map(|(key, standing)| (*volume, key, standing)))
            .min_by_key(|(_, key, _)| *key)?;
        let link = self.links.get_mut(&volume).expect("a ready link");
        let flight = link.flight.total();
        let map = link.map(&key.0);
        let request = map.get_mut(&key).expect("the chosen request");
        let piece = request.pieces.pop_front().expect("a queued request has a piece");
        let bytes = piece.end - piece.start;
        crate::timeline::note_at(now, crate::timeline::Kind::Take, key.1, (piece.start as i64, bytes as i64, flight as i64));
        request.state.started.get_or_init(|| now);
        let taken = Taken {
            id: key.1,
            state: request.state.clone(),
            file: request.file.clone(),
            volume: request.volume.clone(),
            need: request.need.clone(),
            piece,
            standing,
        };
        self.taken += 1;
        if request.pieces.is_empty() {
            map.remove(&key);
            self.wants.remove(&key.1);
        }
        *link.flight.of(standing) += bytes;
        link.book.began(standing.due(), now);
        self.waits.get_mut(&key.1).expect("a queued request is waited").taken(standing.due(), &mut link.book, now);
        Some(taken)
    }

    /// The pieces taken and not yet ended, on every link.
    pub(crate) fn taken(&self) -> usize {
        self.taken
    }

    /// Whether any link's next piece may start now: what a worker that
    /// took one wakes another for, so a link's depth fills one piece after
    /// another however many its last change freed.
    pub(crate) fn ready(&self) -> bool {
        self.links.values().any(Link::ready)
    }

    /// `returned`'s piece ended as its read returned, counted in its link's
    /// book and its request's wait: whether the link holds a request
    /// queued, whose next piece the room it freed may admit.
    pub(crate) fn ended(&mut self, returned: Returned) -> bool {
        let Returned { id, volume, standing, bytes, at, need } = returned;
        need.ended(1);
        let link = self.links.get_mut(&volume).expect("an ended piece was in flight");
        self.taken -= 1;
        *link.flight.of(standing) -= bytes;
        link.book.ended(standing.due(), bytes, at);
        settle(&mut self.waits, id, |wait| wait.ended(&mut link.book, at));
        !(link.requests.is_empty() && link.streamed.is_empty())
    }

    /// Order request `id`'s pieces not yet taken by `want`: the workers to
    /// wake for the pieces its link may start now.
    pub(crate) fn reprice(&mut self, id: u64, want: Want) -> usize {
        let Some((volume, current)) = self.wants.get_mut(&id) else {
            return 0;
        };
        let link = self.links.get_mut(volume).expect("a queued request's link");
        let request = link.map(current).remove(&(*current, id)).expect("a queued request");
        *current = want;
        crate::timeline::note(crate::timeline::Kind::Reprice, id, (i64::from(!want.due()), request.pieces.len() as i64, 0));
        link.map(&want).insert((want, id), request);
        link.room()
    }

    /// Cancel request `id`: its pieces no worker took, none where every one
    /// was.
    pub(crate) fn cancel(&mut self, id: u64) -> Option<Queued> {
        let (volume, want) = self.wants.remove(&id)?;
        let link = self.links.get_mut(&volume).expect("a queued request's link");
        let request = link.map(&want).remove(&(want, id)).expect("a queued request");
        crate::timeline::note(crate::timeline::Kind::Cancel, id, (request.pieces.len() as i64, 0, 0));
        settle(&mut self.waits, id, |wait| wait.cancelled(&mut link.book, Instant::now()));
        request.need.ended(request.pieces.len());
        Some(request)
    }
}
