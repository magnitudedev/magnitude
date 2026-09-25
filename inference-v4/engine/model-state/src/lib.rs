//! Family-neutral accepted sequence state and transactional history ownership.
//! Owned transactions can cross a submission boundary. Their callers reconcile
//! only after physical completion has been observed.
mod advance;
mod codec;
mod layout;
pub mod placement;

pub use advance::{
    CodecConversionStep, OwnedAdvanceBindings, OwnedAdvanceResolution, OwnedCodecAdvance,
    OwnedCodecBindings, OwnedCompaction, OwnedCompactionBindings, OwnedCompactionPreparation,
    OwnedStateAdvance, OwnedSuccessorAdvance, TentativeAdvance,
};

pub use codec::{
    Codec, CodecIdentity, CodecSpec, ComponentDescriptor, KvCodec, LayerRef, LayoutError,
    PlaneDescriptor, PlaneName, VectorKind, AFFINE_GROUP,
};
pub use layout::ModelStateLayout;

use placement::{AllocationId, Generation, LogicalId, PhysicalSlot, Placement};
use seismic::{DType, Device, Element, Tensor, TensorStorageObserver};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

pub const MAX_VISIBLE_SEGMENTS: usize = 16;

/// Failures owned by the state store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Request(String),
    Tensor(seismic::TensorError),
    Layout(LayoutError),
    Capacity { required: u64, available_bytes: u64 },
    BanksExhausted { capacity: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(message) => f.write_str(message),
            Self::Tensor(error) => write!(f, "{error}"),
            Self::Layout(error) => write!(f, "{error}"),
            Self::Capacity {
                required,
                available_bytes,
            } => write!(
                f,
                "state capacity requires {required} bytes but {available_bytes} bytes are available"
            ),
            Self::BanksExhausted { capacity } => {
                write!(f, "all {capacity} recurrent banks have live claims")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::Request(message.to_owned())
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Request(message)
    }
}

impl From<seismic::TensorError> for Error {
    fn from(error: seismic::TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<LayoutError> for Error {
    fn from(error: LayoutError) -> Self {
        Self::Layout(error)
    }
}

fn element(dtype: DType) -> Element {
    match dtype {
        DType::F32 => Element::f32(),
        DType::F16 => Element::f16(),
        DType::BF16 => Element::bf16(),
        DType::I32 => Element::i32(),
        DType::U32 => Element::u32(),
        DType::Bool => Element::bool(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
}
impl ComponentSpec {
    /// Physical bytes in one component of a recurrent state bank.
    pub fn bytes(&self) -> Result<usize, String> {
        if self.shape.is_empty() || self.shape.contains(&0) || !self.dtype.is_float() {
            return Err("state components require nonempty floating tensors".into());
        }
        self.shape
            .iter()
            .try_fold(self.dtype.bytes() as usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| "state component allocation overflow".into())
    }
}

/// One allocated attention-history plane. `base_row` is stable and currently
/// always zero because all components share the arena-global row domain.
#[derive(Clone)]
pub struct PlaneBuffer {
    pub plane_index: usize,
    pub component_index: usize,
    pub layer: LayerRef,
    pub vector: VectorKind,
    pub name: PlaneName,
    pub row_bytes: usize,
    pub base_row: usize,
    pub buffer: Tensor,
}

/// History rows of one store: address-ordered, coalesced free holes, and the
/// referenced rows as address-ordered runs with a reference count.
///
/// Placement keeps every sequence's history in few segments however requests
/// interleave: a sequence grows in place into the hole that begins at its
/// history end, and a sequence that cannot grow in place (a fresh sequence, a
/// fork whose sibling took the rows, or a neighbour boundary) starts at the
/// middle of the largest hole, leaving the rows before it as growth room for
/// the history that ends there. Only a hole at row 0 has no such history and
/// is filled from its start.
///
/// A row is referenced once by every history ([`Claims`]) covering it, so a
/// prefix is shared by any number of sequences and checkpoints at row
/// granularity, and a row returns to the free holes only when its last
/// history drops it. Every history is registered, so [`Arena::relayout`] can
/// place all referenced rows anew and rewrite every history.
#[derive(Clone)]
struct Arena {
    free: Vec<(usize, usize)>,
    runs: BTreeMap<usize, Run>,
    referenced: usize,
    entries: BTreeMap<u64, Entry>,
    next_entry: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    count: usize,
    references: usize,
}

impl Arena {
    fn new(rows: usize) -> Self {
        Self {
            free: if rows == 0 { vec![] } else { vec![(0, rows)] },
            runs: BTreeMap::new(),
            referenced: 0,
            entries: BTreeMap::new(),
            next_entry: 0,
        }
    }

    fn register(&mut self, ranges: Vec<(usize, usize)>, live: bool) -> u64 {
        let id = self.next_entry;
        self.next_entry += 1;
        let mut joined = Vec::with_capacity(ranges.len());
        append_ranges(&mut joined, ranges);
        self.entries.insert(
            id,
            Entry {
                ranges: joined,
                live,
            },
        );
        id
    }

    fn available(&self) -> usize {
        self.free.iter().map(|(_, count)| count).sum()
    }

    /// Free rows directly after `end`: room for the history ending there to
    /// grow in place.
    fn room_at(&self, end: usize) -> usize {
        self.free
            .binary_search_by_key(&end, |(start, _)| *start)
            .map_or(0, |hole| self.free[hole].1)
    }

    /// Live histories ending at row `end`.
    fn live_ending_at(&self, end: usize) -> impl Iterator<Item = u64> + '_ {
        self.entries
            .iter()
            .filter(move |(_, entry)| entry.live && entry.end() == Some(end))
            .map(|(&id, _)| id)
    }

    /// Whether a relayout can give live history `id` room after its last
    /// row: no other live history also covers that row (a branch point is a
    /// prefix every branch but one continues in a new run, wherever the
    /// rows are placed).
    fn owns_its_end(&self, id: u64) -> bool {
        let Some(last) = self.entries[&id].end().map(|end| end - 1) else {
            return false;
        };
        !self.entries.iter().any(|(&other, entry)| {
            other != id
                && entry.live
                && entry
                    .ranges
                    .iter()
                    .any(|&(start, count)| start <= last && last < start + count)
        })
    }

    /// Place every referenced row anew within `rows` rows and rewrite every
    /// history. Live histories come first, those in one run before the rest
    /// (a relayout never splits a contiguous history to join another) and
    /// longer before shorter (a branch sharing a prefix follows it): each
    /// is laid out as one run where its rows are not already placed as a
    /// prefix of an earlier history, followed by `gaps[id]` free rows for it
    /// to grow into in place. Rows only frozen histories (checkpoints)
    /// reference are packed after. Reference counts are unchanged. Returns
    /// the row moves `(from, to, count)`.
    fn relayout(&mut self, rows: usize, gaps: &BTreeMap<u64, usize>) -> Vec<(usize, usize, usize)> {
        let mut order = self
            .entries
            .iter()
            .map(|(&id, entry)| {
                (
                    !entry.live,
                    entry.ranges.len() > 1,
                    std::cmp::Reverse(entry.rows()),
                    id,
                )
            })
            .collect::<Vec<_>>();
        order.sort_unstable();
        // Placed rows: old start -> (count, new start).
        let mut placed: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
        let mut next = 0;
        for (_, _, _, id) in order {
            for &(start, count) in &self.entries[&id].ranges {
                let (mut row, end) = (start, start + count);
                while row < end {
                    if let Some((&from, &(moved, _))) = placed.range(..=row).next_back() {
                        if from + moved > row {
                            row = (from + moved).min(end);
                            continue;
                        }
                    }
                    let stop = placed.range(row..end).next().map_or(end, |(&from, _)| from);
                    placed.insert(row, (stop - row, next));
                    next += stop - row;
                    row = stop;
                }
            }
            next += gaps.get(&id).copied().unwrap_or(0);
        }
        debug_assert!(next <= rows, "relayout exceeds its rows");
        let translate = |start: usize, count: usize| {
            let mut out = Vec::new();
            let (mut row, end) = (start, start + count);
            while row < end {
                let (&from, &(moved, to)) = placed
                    .range(..=row)
                    .next_back()
                    .expect("every referenced row is placed");
                let taken = (from + moved).min(end) - row;
                append_ranges(&mut out, [(to + row - from, taken)]);
                row += taken;
            }
            out
        };
        for entry in self.entries.values_mut() {
            let mut ranges = Vec::with_capacity(entry.ranges.len());
            for &(start, count) in &entry.ranges {
                append_ranges(&mut ranges, translate(start, count));
            }
            entry.ranges = ranges;
        }
        // Reference counts follow the histories covering each row.
        let mut events = self
            .entries
            .values()
            .flat_map(|entry| entry.ranges.iter())
            .flat_map(|&(start, count)| [(start, 1isize), (start + count, -1)])
            .collect::<Vec<_>>();
        events.sort_unstable();
        self.runs.clear();
        let (mut depth, mut from) = (0isize, 0);
        let mut last: Option<usize> = None;
        for (row, delta) in events {
            if depth > 0 && row > from {
                let references = depth as usize;
                let extended = last.and_then(|start| {
                    let run = self.runs.get_mut(&start)?;
                    (start + run.count == from && run.references == references)
                        .then(|| run.count += row - from)
                });
                if extended.is_none() {
                    self.runs.insert(
                        from,
                        Run {
                            count: row - from,
                            references,
                        },
                    );
                    last = Some(from);
                }
            }
            depth += delta;
            from = row;
        }
        debug_assert_eq!(
            self.runs.values().map(|run| run.count).sum::<usize>(),
            self.referenced
        );
        let mut cursor = 0;
        self.free.clear();
        for (&start, run) in &self.runs {
            if start > cursor {
                self.free.push((cursor, start - cursor));
            }
            cursor = start + run.count;
        }
        if rows > cursor {
            self.free.push((cursor, rows - cursor));
        }
        let mut moves: Vec<(usize, usize, usize)> = Vec::with_capacity(placed.len());
        for (from, (count, to)) in placed {
            match moves.last_mut() {
                Some((f, t, n)) if *f + *n == from && *t + *n == to => *n += count,
                _ => moves.push((from, to, count)),
            }
        }
        moves
    }

    /// Claim `count` rows, in logical order, for a sequence whose history ends
    /// at row `after`. The caller has checked `count <= self.available()`.
    fn claim(&mut self, after: Option<usize>, count: usize) -> Vec<(usize, usize)> {
        let mut claimed = Vec::new();
        let mut remaining = count;
        if let Some(end) = after {
            if let Some(hole) = self.free.iter().position(|(start, _)| *start == end) {
                let taken = self.free[hole].1.min(remaining);
                claimed.push(self.take(hole, 0, taken));
                remaining -= taken;
            }
        }
        while remaining > 0 {
            let hole = (0..self.free.len())
                .max_by_key(|&hole| (self.free[hole].1, std::cmp::Reverse(self.free[hole].0)))
                .expect("claimed rows never exceed the available rows");
            let (start, size) = self.free[hole];
            let taken = size.min(remaining);
            let offset = if start == 0 { 0 } else { (size - taken) / 2 };
            claimed.push(self.take(hole, offset, taken));
            remaining -= taken;
        }
        claimed
    }

    /// Commit rows `from..to` as free (growth), or give back the free tail
    /// `to..from` (release; every row there is unreferenced).
    fn resize(&mut self, from: usize, to: usize) {
        if to > from {
            match self.free.last_mut() {
                Some((start, count)) if *start + *count == from => *count += to - from,
                _ => self.free.push((from, to - from)),
            }
            return;
        }
        let (start, count) = self.free.pop().expect("a released tail is a free hole");
        debug_assert!(
            start <= to && start + count == from,
            "released rows are free"
        );
        if start < to {
            self.free.push((start, to - start));
        }
    }

    /// Claim the first hole of at least `count` rows from its start.
    fn claim_contiguous(&mut self, count: usize) -> Option<usize> {
        let hole = self.free.iter().position(|(_, size)| *size >= count)?;
        Some(self.take(hole, 0, count).0)
    }

    /// Remove `count` rows at `offset` within hole `hole`, keeping the free
    /// list address-ordered, and reference them once.
    fn take(&mut self, hole: usize, offset: usize, count: usize) -> (usize, usize) {
        let (start, size) = self.free[hole];
        let before = (offset > 0).then_some((start, offset));
        let after =
            (offset + count < size).then(|| (start + offset + count, size - offset - count));
        self.free
            .splice(hole..=hole, before.into_iter().chain(after));
        let start = start + offset;
        self.runs.insert(
            start,
            Run {
                count,
                references: 1,
            },
        );
        self.referenced += count;
        self.coalesce(start, start + count);
        (start, count)
    }

    /// Make `row` a run boundary if it lies strictly inside a run.
    fn boundary(&mut self, row: usize) {
        let Some((&start, &run)) = self.runs.range(..row).next_back() else {
            return;
        };
        if start + run.count > row {
            self.runs.insert(
                start,
                Run {
                    count: row - start,
                    references: run.references,
                },
            );
            self.runs.insert(
                row,
                Run {
                    count: start + run.count - row,
                    references: run.references,
                },
            );
        }
    }

    /// Add one reference to every row of a referenced range.
    fn retain(&mut self, start: usize, count: usize) {
        let end = start + count;
        self.boundary(start);
        self.boundary(end);
        let mut covered = 0;
        for (_, run) in self.runs.range_mut(start..end) {
            run.references += 1;
            covered += run.count;
        }
        debug_assert_eq!(covered, count, "a claim covers only referenced rows");
        self.coalesce(start, end);
    }

    /// Remove one reference from every row of a range; rows left without a
    /// reference return to the free holes.
    fn release(&mut self, start: usize, count: usize) {
        let end = start + count;
        self.boundary(start);
        self.boundary(end);
        let starts = self
            .runs
            .range(start..end)
            .map(|(&row, _)| row)
            .collect::<Vec<_>>();
        let mut freed = false;
        for row in starts {
            let run = self.runs.get_mut(&row).expect("run listed above");
            run.references -= 1;
            if run.references == 0 {
                let count = run.count;
                self.runs.remove(&row);
                self.referenced -= count;
                self.free.push((row, count));
                freed = true;
            }
        }
        if freed {
            self.free.sort_unstable();
            let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.free.len());
            for (start, count) in self.free.drain(..) {
                match merged.last_mut() {
                    Some((s, n)) if *s + *n == start => *n += count,
                    _ => merged.push((start, count)),
                }
            }
            self.free = merged;
        }
        self.coalesce(start, end);
    }

    /// Merge adjacent runs with equal reference counts around `[start, end)`
    /// so the run map stays proportional to sharing boundaries.
    fn coalesce(&mut self, start: usize, end: usize) {
        let first = self
            .runs
            .range(..start)
            .next_back()
            .map_or(start, |(&row, _)| row);
        let rows = self
            .runs
            .range(first..=end)
            .map(|(&row, _)| row)
            .collect::<Vec<_>>();
        let mut current: Option<usize> = None;
        for row in rows {
            let run = self.runs[&row];
            if let Some(previous) = current {
                let before = self.runs[&previous];
                if previous + before.count == row && before.references == run.references {
                    self.runs.remove(&row);
                    self.runs.get_mut(&previous).expect("previous run").count += run.count;
                    continue;
                }
            }
            current = Some(row);
        }
    }

    /// Rows of `ranges` (with multiplicity) that no claim outside them
    /// references: the rows released if exactly those claims were dropped.
    fn exclusive_rows(&self, ranges: &[(usize, usize)]) -> usize {
        let mut events = ranges
            .iter()
            .flat_map(|&(start, count)| [(start, 1isize), (start + count, -1)])
            .collect::<Vec<_>>();
        events.sort_unstable();
        let mut exclusive = 0;
        let mut depth = 0isize;
        let mut from = 0;
        for (row, delta) in events {
            if depth > 0 && row > from {
                exclusive += self.rows_with_references(from, row, depth as usize);
            }
            depth += delta;
            from = row;
        }
        exclusive
    }

    fn rows_with_references(&self, start: usize, end: usize, references: usize) -> usize {
        let first = self
            .runs
            .range(..=start)
            .next_back()
            .map_or(start, |(&row, _)| row);
        self.runs
            .range(first..end)
            .filter(|(_, run)| run.references == references)
            .map(|(&row, run)| (row + run.count).min(end).saturating_sub(row.max(start)))
            .sum()
    }
}

/// One registered history: address ranges in logical order (physically
/// adjacent ranges joined). `live` marks the history of a sequence, which can
/// still grow; checkpoints and tentative rows are frozen.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    ranges: Vec<(usize, usize)>,
    live: bool,
}

impl Entry {
    fn rows(&self) -> usize {
        self.ranges.iter().map(|(_, count)| count).sum()
    }
    fn end(&self) -> Option<usize> {
        self.ranges.last().map(|(start, count)| start + count)
    }
}

/// Append `ranges` to `into`, joining physically adjacent ranges.
fn append_ranges(into: &mut Vec<(usize, usize)>, ranges: impl IntoIterator<Item = (usize, usize)>) {
    for (start, count) in ranges {
        match into.last_mut() {
            Some((last, rows)) if *last + *rows == start => *rows += count,
            _ => into.push((start, count)),
        }
    }
}

/// A history's claims: one reference on each row of its ranges, registered in
/// the arena, so the arena knows every history and can relocate them all.
/// Cloning adds a reference to every row, dropping removes one; appending and
/// splitting move references without touching the arena's counts.
struct Claims {
    arena: Rc<RefCell<Arena>>,
    id: u64,
}

impl Claims {
    /// Take ownership of rows already referenced once for this history.
    fn new(arena: &Rc<RefCell<Arena>>, ranges: Vec<(usize, usize)>) -> Self {
        let id = arena.borrow_mut().register(ranges, false);
        Self {
            arena: arena.clone(),
            id,
        }
    }

    fn entry<T>(&self, read: impl FnOnce(&Entry) -> T) -> T {
        read(&self.arena.borrow().entries[&self.id])
    }

    fn ranges(&self) -> Vec<(usize, usize)> {
        self.entry(|entry| entry.ranges.clone())
    }

    fn rows(&self) -> usize {
        self.entry(Entry::rows)
    }

    fn is_empty(&self) -> bool {
        self.entry(|entry| entry.ranges.is_empty())
    }

    fn end(&self) -> Option<usize> {
        self.entry(Entry::end)
    }

    fn set_live(&self, live: bool) {
        self.arena
            .borrow_mut()
            .entries
            .get_mut(&self.id)
            .expect("registered history")
            .live = live;
    }

    /// Unregister without releasing: the caller takes the references.
    fn into_ranges(self) -> Vec<(usize, usize)> {
        let entry = self
            .arena
            .borrow_mut()
            .entries
            .remove(&self.id)
            .expect("registered history");
        entry.ranges
    }

    /// Move `next`'s rows (logically following this history's) onto its end.
    fn append(&mut self, next: Claims) {
        debug_assert!(Rc::ptr_eq(&self.arena, &next.arena));
        let ranges = next.into_ranges();
        let mut arena = self.arena.borrow_mut();
        let entry = arena.entries.get_mut(&self.id).expect("registered history");
        append_ranges(&mut entry.ranges, ranges);
    }

    /// Keep the first `keep` rows; return the remainder as a frozen history.
    fn split_off(&mut self, keep: usize) -> Claims {
        let tail = {
            let mut arena = self.arena.borrow_mut();
            let entry = arena.entries.get_mut(&self.id).expect("registered history");
            let (head, tail) = split_ranges(&entry.ranges, keep);
            entry.ranges = head;
            tail
        };
        Claims::new(&self.arena, tail)
    }

    /// Release the first `rows` rows.
    fn drop_front(&mut self, rows: usize) {
        let head = {
            let mut arena = self.arena.borrow_mut();
            let entry = arena.entries.get_mut(&self.id).expect("registered history");
            let (head, tail) = split_ranges(&entry.ranges, rows);
            entry.ranges = tail;
            head
        };
        drop(Claims::new(&self.arena, head));
    }
}

/// The first `keep` rows of `ranges` and the remainder.
fn split_ranges(
    ranges: &[(usize, usize)],
    keep: usize,
) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let mut remaining = keep;
    let (mut head, mut tail) = (Vec::new(), Vec::new());
    for &(start, count) in ranges {
        if remaining >= count {
            remaining -= count;
            head.push((start, count));
        } else if remaining == 0 {
            tail.push((start, count));
        } else {
            head.push((start, remaining));
            tail.push((start + remaining, count - remaining));
            remaining = 0;
        }
    }
    (head, tail)
}

impl Clone for Claims {
    /// A frozen history referencing the same rows.
    fn clone(&self) -> Self {
        let mut arena = self.arena.borrow_mut();
        let ranges = arena.entries[&self.id].ranges.clone();
        for &(start, count) in &ranges {
            arena.retain(start, count);
        }
        let id = arena.register(ranges, false);
        Self {
            arena: self.arena.clone(),
            id,
        }
    }
}

impl Drop for Claims {
    fn drop(&mut self) {
        let mut arena = self.arena.borrow_mut();
        if let Some(entry) = arena.entries.remove(&self.id) {
            for (start, count) in entry.ranges {
                arena.release(start, count);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankCapacity {
    pub active: usize,
    pub in_flight: usize,
    pub retained: usize,
}

impl BankCapacity {
    pub fn total(self) -> Result<usize, Error> {
        if self.active == 0 || self.in_flight == 0 {
            return Err(Error::Request(
                "bank pool requires positive active and in-flight capacity".into(),
            ));
        }
        self.active
            .checked_add(self.in_flight)
            .and_then(|value| value.checked_add(self.retained))
            .ok_or_else(|| Error::Request("bank pool capacity overflow".into()))
    }

    /// Writable banks plus one permanently pristine seed for new sequences.
    pub fn storage_total(self) -> Result<usize, Error> {
        self.total()?
            .checked_add(1)
            .ok_or_else(|| Error::Request("zero seed bank count overflow".into()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateAllocationTrace {
    pub context_capacity: usize,
    pub history_capacity: usize,
    pub history_row_bytes: u64,
    pub history_bytes: u64,
    pub bank_capacity: BankCapacity,
    pub recurrent_bank_bytes: u64,
    pub zero_seed_bytes: u64,
    pub recurrent_pool_bytes: u64,
}

/// The permanently pristine bank every new sequence starts from. It is never
/// on the free list, so no advance can name it as its successor.
pub const ZERO_SEED_BANK: usize = 0;

struct BankPoolInner {
    capacity: usize,
    placement: RefCell<Placement>,
    next_logical: Cell<u64>,
    next_allocation: Cell<u64>,
}

/// A stable logical claim on one row of every recurrent component. The
/// published placement alone decides which physical row holds it.
struct BankClaim {
    pool: Rc<BankPoolInner>,
    id: LogicalId,
}

impl Drop for BankClaim {
    fn drop(&mut self) {
        if self.id != LogicalId(0) {
            let published = self.pool.placement.borrow().clone();
            let mut resources = published.resources().clone();
            assert!(
                resources.remove(&self.id).is_some(),
                "bank claim has a placement"
            );
            *self.pool.placement.borrow_mut() = published
                .with_resources(resources)
                .expect("bank release preserves placement invariants");
        }
    }
}

#[derive(Clone)]
struct BankHandle(Rc<BankClaim>);

impl BankHandle {
    fn index(&self) -> usize {
        self.0
            .pool
            .placement
            .borrow()
            .resolve(self.0.id)
            .expect("live bank has a physical placement")
            .index
    }
}

/// Logical bank ownership and its published physical placement. The zero
/// seed is permanently claimed; writable rows are acquired lowest first.
struct BankPool {
    inner: Rc<BankPoolInner>,
}

impl BankPool {
    fn new(capacity: BankCapacity, committed: usize) -> Result<(Self, BankHandle), Error> {
        let placement = Placement::new(
            Generation(0),
            BTreeMap::from([(AllocationId(0), committed)]),
            BTreeMap::from([(
                LogicalId(0),
                PhysicalSlot {
                    allocation: AllocationId(0),
                    index: ZERO_SEED_BANK,
                },
            )]),
        )
        .map_err(|error| Error::Request(format!("invalid initial bank placement: {error:?}")))?;
        let inner = Rc::new(BankPoolInner {
            capacity: capacity.total()?,
            placement: RefCell::new(placement),
            next_logical: Cell::new(1),
            next_allocation: Cell::new(1),
        });
        let seed = BankHandle(Rc::new(BankClaim {
            pool: inner.clone(),
            id: LogicalId(0),
        }));
        Ok((Self { inner }, seed))
    }

    fn acquire(&self) -> Result<BankHandle, Error> {
        let published = self.inner.placement.borrow().clone();
        let allocation = published
            .resolve(LogicalId(0))
            .expect("seed placement")
            .allocation;
        let occupied = published
            .resources()
            .values()
            .map(|slot| slot.index)
            .collect::<BTreeSet<_>>();
        let index = (1..self.committed())
            .find(|index| !occupied.contains(index))
            .ok_or(Error::BanksExhausted {
                capacity: self.inner.capacity,
            })?;
        let id = LogicalId(self.inner.next_logical.get());
        let mut resources = published.resources().clone();
        resources.insert(id, PhysicalSlot { allocation, index });
        *self.inner.placement.borrow_mut() = published
            .with_resources(resources)
            .map_err(|error| Error::Request(format!("bank acquisition placement: {error:?}")))?;
        self.inner
            .next_logical
            .set(id.0.checked_add(1).expect("bank logical id exhausted"));
        Ok(BankHandle(Rc::new(BankClaim {
            pool: self.inner.clone(),
            id,
        })))
    }

    fn committed(&self) -> usize {
        let placement = self.inner.placement.borrow();
        let allocation = placement
            .resolve(LogicalId(0))
            .expect("seed placement")
            .allocation;
        placement
            .allocation_capacity(allocation)
            .expect("current bank allocation")
    }

    fn available(&self) -> usize {
        self.committed() - self.inner.placement.borrow().resources().len()
    }

    fn required(&self, committed: usize) -> usize {
        let placement = self.inner.placement.borrow();
        let allocation = placement
            .resolve(LogicalId(0))
            .expect("seed placement")
            .allocation;
        placement
            .highest_occupied_slot(allocation)
            .map_or(1, |bank| bank + 1)
            .min(committed)
    }

    fn resize(&self, from: usize, to: usize) {
        assert_eq!(self.committed(), from);
        let published = self.inner.placement.borrow().clone();
        let allocation = published
            .resolve(LogicalId(0))
            .expect("seed placement")
            .allocation;
        *self.inner.placement.borrow_mut() = published
            .with_capacity(allocation, to)
            .expect("bank resize preserves every live placement");
    }
}

/// History rows the backing commits at a time: it grows by at least this many
/// rows or half its size, and releases whole granules.
const HISTORY_GRANULE: usize = 256;
/// Growth keeps `referenced / HEADROOM_DIVISOR` rows free beside a launch's
/// demand: the growth room a relayout shares among live histories.
const HEADROOM_DIVISOR: usize = 2;
/// A history without room is relaid out at the same size only while at least
/// `referenced / RELAYOUT_SLACK_DIVISOR` free rows remain to share; a full
/// reservation below that continues the history in a new run instead.
const RELAYOUT_SLACK_DIVISOR: usize = 8;
/// Banks committed at creation beside the zero seed, and the least growth.
const BANK_GRANULE: usize = 2;

/// Recurrent banks a store with recurrent components commits at creation,
/// the zero seed included, out of `reserved`.
pub fn initial_banks(reserved: usize) -> usize {
    reserved.min(1 + BANK_GRANULE)
}

/// Grow `committed` to cover `required`, geometrically, within `reserved`.
fn grown(committed: usize, required: usize, granule: usize, reserved: usize) -> usize {
    required
        .max(committed + committed / 2)
        .max(committed + granule)
        .div_ceil(granule)
        .saturating_mul(granule)
        .min(reserved)
}

fn allocation_capacity(error: &Error) -> bool {
    matches!(
        error,
        Error::Tensor(seismic::TensorError::Execution(
            seismic::ExecutionError::AllocationCapacity { .. }
                | seismic::ExecutionError::AllocationFailed(_)
        ))
    )
}

/// Form replacements that need separate storage before changing any CUDA
/// reservation in place. If a mixed set cannot allocate its replacements,
/// every original plane remains physically intact. In-place growth retains
/// its own rollback link until this complete vector is published.
fn recommit_planes(planes: &[Tensor], leading: u64) -> Result<Vec<Tensor>, Error> {
    let mut replacements = vec![None; planes.len()];
    if planes
        .first()
        .is_some_and(|plane| leading < plane.committed_rows())
    {
        // Unmapping a CUDA tail destroys its contents. Prepare every other
        // plane first, then shrink at most one reservation in place as the
        // final fallible operation. Prefer the largest plane to minimize
        // temporary replacement charge.
        let final_shrink = planes
            .iter()
            .enumerate()
            .filter(|(_, plane)| plane.can_recommit_in_place())
            .max_by_key(|(_, plane)| plane.storage_bytes())
            .map(|(index, _)| index);
        for (index, plane) in planes.iter().enumerate() {
            if Some(index) != final_shrink {
                replacements[index] = Some(plane.recommitted_separately(leading)?);
            }
        }
        if let Some(index) = final_shrink {
            replacements[index] = Some(planes[index].recommitted(leading)?);
        }
    } else {
        for in_place in [false, true] {
            for (index, plane) in planes.iter().enumerate() {
                if plane.can_recommit_in_place() == in_place {
                    replacements[index] = Some(plane.recommitted(leading)?);
                }
            }
        }
    }
    Ok(replacements
        .into_iter()
        .map(|plane| plane.expect("every state plane has a replacement"))
        .collect())
}

fn leading_extents(leading: usize, rest: &[usize]) -> Result<Vec<u64>, Error> {
    std::iter::once(leading)
        .chain(rest.iter().copied())
        .map(|extent| {
            u64::try_from(extent).map_err(|_| Error::Request("state extent exceeds u64".into()))
        })
        .collect()
}

/// The physical backing: reserved tensors whose leading rows (history rows,
/// recurrent banks) are committed on demand. Graphs are sealed over the
/// reserved shapes; only committed rows are ever handed out.
struct Backing {
    history: Vec<Tensor>,
    rows: usize,
    recurrent: Rc<[Tensor]>,
    banks: usize,
}

/// Counts transactions that captured this store's tensors and write them
/// later (advances, compactions, conversions). The backing may be
/// recommitted only while none exists: a reallocating backend gives the store
/// new tensors, and a captured old one would receive writes nobody reads.
#[derive(Default)]
struct TransactionClaims {
    active: usize,
    histories: BTreeMap<u64, usize>,
    banks: BTreeMap<usize, usize>,
}

#[derive(Clone)]
struct Transactions(Rc<RefCell<TransactionClaims>>);

struct Transaction {
    claims: Rc<RefCell<TransactionClaims>>,
    histories: Vec<u64>,
    banks: Vec<usize>,
}

impl Transactions {
    fn begin(&self) -> Transaction {
        self.0.borrow_mut().active += 1;
        Transaction {
            claims: self.0.clone(),
            histories: Vec::new(),
            banks: Vec::new(),
        }
    }
    fn idle(&self) -> bool {
        self.0.borrow().active == 0
    }
}

impl Transaction {
    fn track(&mut self, claims: &Claims, bank: &BankHandle) {
        self.track_history(claims);
        self.track_bank(bank);
    }
    fn track_history(&mut self, claims: &Claims) {
        self.histories.push(claims.id);
        *self
            .claims
            .borrow_mut()
            .histories
            .entry(claims.id)
            .or_default() += 1;
    }
    fn track_bank(&mut self, bank: &BankHandle) {
        self.banks.push(bank.index());
        *self
            .claims
            .borrow_mut()
            .banks
            .entry(bank.index())
            .or_default() += 1;
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        let mut tracked = self.claims.borrow_mut();
        tracked.active -= 1;
        for id in &self.histories {
            let count = tracked
                .histories
                .get_mut(id)
                .expect("tracked history claim");
            *count -= 1;
            if *count == 0 {
                tracked.histories.remove(id);
            }
        }
        for index in &self.banks {
            let count = tracked.banks.get_mut(index).expect("tracked bank claim");
            *count -= 1;
            if *count == 0 {
                tracked.banks.remove(index);
            }
        }
    }
}
/// History rows are a shared arena; recurrent components are arenas of banks,
/// and each accepted version is one immutable bank. A checkpoint retains both
/// without copying tensor contents.
///
/// `history_capacity` rows and the bank capacity are reservations sealed into
/// graphs; the backing commits rows and banks as demand grows and releases
/// unreferenced tails ([`StateStore::provision`], [`StateStore::shrink`]).
pub struct StateStore {
    device: Rc<Device>,
    context_capacity: usize,
    history_capacity: usize,
    components: Vec<ComponentDescriptor>,
    total_history_row_bytes: u64,
    bank_capacity: BankCapacity,
    recurrent_bank_bytes: u64,
    component_specs: Vec<ComponentSpec>,
    backing: RefCell<Backing>,
    retired_storage: RefCell<Vec<TensorStorageObserver>>,
    arena: Rc<RefCell<Arena>>,
    banks: BankPool,
    zero_seed: BankHandle,
    owners: Cell<usize>,
    transactions: Transactions,
    relayouts: Cell<Relayouts>,
    recommits: Cell<usize>,
}

/// When [`StateStore::shrink`] releases backing: at idle only once the
/// backing is at least twice what the store needs (hysteresis), under memory
/// pressure everything beyond it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShrinkPolicy {
    Idle,
    Pressure,
}

/// History relayouts a store performed and the rows they copied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Relayouts {
    pub count: usize,
    pub rows: usize,
}

/// One store's demand for history rows: a sequence whose history ends at
/// `after` (none for a fresh one) appending `rows`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowDemand {
    pub after: Option<usize>,
    pub rows: usize,
}

/// Additional Seismic charge needed at the peak of elastic state growth.
/// A reallocating backend briefly holds old and new backing together, so
/// these amounts include that transient, not merely the final byte increase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateGrowthClaim {
    pub minimum_bytes: u64,
    pub preferred_bytes: u64,
}

/// A physical census of one state store. Shared rows and recurrent banks are
/// assigned to the strongest holder class, without charging aliases twice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateHoldingCensus {
    pub surplus: u64,
    pub retained: u64,
    pub live: u64,
    pub in_flight: u64,
    pub model_seed: u64,
}

/// Read-only shape of a pressure shrink, including the peak new charge needed
/// before old history or recurrent planes can be released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatePressureShrinkShape {
    pub committed_rows: usize,
    pub referenced_rows: usize,
    pub highest_referenced_row: usize,
    pub target_rows: usize,
    pub committed_banks: usize,
    pub claimed_banks: usize,
    pub target_banks: usize,
    pub active_transactions: usize,
    pub history_peak_bytes: u64,
    pub bank_peak_bytes: u64,
    pub bank_in_place_ready: bool,
}

fn history_shrink_target(referenced: usize, policy: ShrinkPolicy) -> usize {
    let headroom = match policy {
        ShrinkPolicy::Idle => 1,
        ShrinkPolicy::Pressure => HEADROOM_DIVISOR,
    };
    (referenced + referenced / headroom)
        .max(HISTORY_GRANULE)
        .div_ceil(HISTORY_GRANULE)
        * HISTORY_GRANULE
}

impl StateHoldingCensus {
    pub fn total(self) -> u64 {
        self.surplus + self.retained + self.live + self.in_flight + self.model_seed
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrowthChoice {
    Minimum,
    Preferred,
}

struct HistoryGrowthPlan {
    target: usize,
    relayout: bool,
    demanded: BTreeMap<u64, usize>,
    total: usize,
}
impl StateStore {
    pub fn pressure_shrink_shape(&self) -> Result<StatePressureShrinkShape, Error> {
        let arena = self.arena.borrow();
        let backing = self.backing.borrow();
        let highest_referenced_row = arena
            .runs
            .last_key_value()
            .map_or(0, |(start, run)| start + run.count);
        let target_rows = history_shrink_target(arena.referenced, ShrinkPolicy::Pressure);
        let committed_rows = backing.rows;
        let committed_banks = backing.banks;
        let claimed_banks = committed_banks - 1 - self.banks.available();
        let dense_target_banks = 1 + claimed_banks + self.owners.get() + BANK_GRANULE;
        let relocation_needed = dense_target_banks < self.banks.required(committed_banks)
            && dense_target_banks < committed_banks;
        let target_banks = dense_target_banks;
        let bank_in_place_ready = !relocation_needed
            && Rc::strong_count(&backing.recurrent) == 1
            && backing.recurrent.iter().all(Tensor::can_recommit_in_place);
        let replacement_peak = |planes: &[Tensor], target: usize| -> Result<u64, Error> {
            let in_place = planes
                .iter()
                .enumerate()
                .filter(|(_, plane)| plane.can_recommit_in_place())
                .max_by_key(|(_, plane)| plane.storage_bytes())
                .map(|(index, _)| index);
            planes
                .iter()
                .enumerate()
                .filter(|(index, _)| Some(*index) != in_place)
                .try_fold(0u64, |total, (_, plane)| {
                    let rows = plane.committed_rows();
                    let bytes = plane
                        .storage_bytes()
                        .checked_div(rows)
                        .and_then(|row| row.checked_mul(target as u64))
                        .ok_or_else(|| {
                            Error::Request("pressure replacement peak overflows".into())
                        })?;
                    total
                        .checked_add(bytes)
                        .ok_or_else(|| Error::Request("pressure replacement peak overflows".into()))
                })
        };
        let history_peak_bytes = if committed_rows <= target_rows {
            0
        } else if highest_referenced_row > target_rows {
            (target_rows as u64)
                .checked_mul(self.total_history_row_bytes)
                .ok_or_else(|| Error::Request("pressure relayout peak overflows".into()))?
        } else {
            replacement_peak(&backing.history, target_rows)?
        };
        let bank_peak_bytes = if relocation_needed {
            (target_banks as u64)
                .checked_mul(self.recurrent_bank_bytes)
                .ok_or_else(|| Error::Request("bank relocation peak overflows".into()))?
        } else if committed_banks > target_banks && !bank_in_place_ready {
            replacement_peak(&backing.recurrent, target_banks)?
        } else {
            0
        };
        Ok(StatePressureShrinkShape {
            committed_rows,
            referenced_rows: arena.referenced,
            highest_referenced_row,
            target_rows,
            committed_banks,
            claimed_banks,
            target_banks,
            active_transactions: self.transactions.0.borrow().active,
            history_peak_bytes,
            bank_peak_bytes,
            bank_in_place_ready,
        })
    }
    pub fn new(
        device: Rc<Device>,
        context_capacity: usize,
        history_capacity: usize,
        components: Vec<ComponentDescriptor>,
        component_specs: Vec<ComponentSpec>,
        bank_capacity: BankCapacity,
    ) -> Result<Rc<Self>, Error> {
        if context_capacity == 0 || context_capacity > history_capacity {
            return Err(Error::Request(
                "history capacity must fit a positive sequence context".into(),
            ));
        }
        let total_history_row_bytes = validate_history_layout(&components, history_capacity)?;
        for spec in &component_specs {
            spec.bytes().map_err(Error::Request)?;
        }
        let recurrent_bank_bytes = component_specs.iter().try_fold(0_u64, |total, spec| {
            total
                .checked_add(
                    u64::try_from(spec.bytes().map_err(Error::Request)?)
                        .map_err(|_| Error::Request("recurrent bank bytes exceed u64".into()))?,
                )
                .ok_or_else(|| Error::Request("recurrent bank byte count overflow".into()))
        })?;
        let reserved_banks = bank_capacity.storage_total()?;
        // Banks of a store without recurrent components have no storage.
        let committed_banks = if component_specs.is_empty() {
            reserved_banks
        } else {
            initial_banks(reserved_banks)
        };
        let (banks, zero_seed) = BankPool::new(bank_capacity, committed_banks)?;
        let recurrent = component_specs
            .iter()
            .map(|spec| {
                Tensor::reserved(
                    &device,
                    element(spec.dtype),
                    &leading_extents(reserved_banks, &spec.shape)?,
                    committed_banks as u64,
                )
                .map_err(Error::from)
            })
            .collect::<Result<Rc<[Tensor]>, _>>()?;
        Ok(Rc::new(Self {
            device,
            context_capacity,
            history_capacity,
            components,
            total_history_row_bytes,
            bank_capacity,
            recurrent_bank_bytes,
            component_specs,
            backing: RefCell::new(Backing {
                history: Vec::new(),
                rows: 0,
                recurrent,
                banks: committed_banks,
            }),
            retired_storage: RefCell::new(Vec::new()),
            arena: Rc::new(RefCell::new(Arena::new(0))),
            banks,
            zero_seed,
            owners: Cell::new(0),
            transactions: Transactions(Rc::new(RefCell::new(TransactionClaims::default()))),
            relayouts: Cell::new(Relayouts::default()),
            recommits: Cell::new(0),
        }))
    }
    pub fn component_specs(&self) -> &[ComponentSpec] {
        &self.component_specs
    }
    pub fn has_recurrent_components(&self) -> bool {
        !self.component_specs.is_empty()
    }
    /// One arena per recurrent component, `[banks, ..component shape]`. A
    /// bank index selects the same row of every arena. Kernels read a
    /// sequence's accepted bank and write only an advance's successor bank;
    /// they never write an accepted bank or [`ZERO_SEED_BANK`].
    pub fn recurrent_arenas(&self) -> Rc<[Tensor]> {
        self.backing.borrow().recurrent.clone()
    }
    /// Banks reserved in each recurrent arena, including the zero seed.
    pub fn recurrent_bank_count(&self) -> Result<usize, Error> {
        self.bank_capacity.storage_total()
    }
    /// History rows and recurrent banks physically committed now.
    pub fn committed(&self) -> (usize, usize) {
        let backing = self.backing.borrow();
        (backing.rows, backing.banks)
    }
    /// Physical bytes of the committed backing.
    pub fn committed_bytes(&self) -> u64 {
        let backing = self.backing.borrow();
        backing
            .history
            .iter()
            .chain(backing.recurrent.iter())
            .map(Tensor::storage_bytes)
            .sum()
    }

    /// Old backing still charged because a caller holds a tensor view after
    /// this store recommitted or relaid out. The weak observers neither pin
    /// the allocations nor invent a charge: each live byte count comes from
    /// Seismic's physical allocation.
    pub fn external_pinned_bytes(&self) -> Result<u64, Error> {
        let backing = self.backing.borrow();
        let current = backing
            .history
            .iter()
            .chain(backing.recurrent.iter())
            .map(|tensor| tensor.observe_storage().identity())
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        let mut retired = self.retired_storage.borrow_mut();
        retired.retain(|storage| storage.charged_bytes().is_some());
        retired
            .iter()
            .filter(|storage| !current.contains(&storage.identity()))
            .filter(|storage| seen.insert(storage.identity()))
            .filter_map(TensorStorageObserver::charged_bytes)
            .try_fold(0u64, |total, bytes| {
                total
                    .checked_add(bytes)
                    .ok_or_else(|| Error::Request("external state pin charge overflows".into()))
            })
    }

    fn record_retired_storage(&self, storage: Vec<TensorStorageObserver>) {
        let mut retired = self.retired_storage.borrow_mut();
        retired.extend(storage);
        retired.retain(|allocation| allocation.charged_bytes().is_some());
    }
    /// Classify committed state backing from its actual row and bank claims.
    /// Every registered holder must be supplied or owned by an outstanding
    /// transaction. An untracked omission is an error rather than surplus.
    pub fn holding_census(
        self: &Rc<Self>,
        live: &[Holder<'_>],
        retained: &[Holder<'_>],
        in_flight: &[Holder<'_>],
    ) -> Result<StateHoldingCensus, Error> {
        let holders = live
            .iter()
            .chain(retained)
            .chain(in_flight)
            .collect::<Vec<_>>();
        let arena = self.arena.borrow();
        let mut supplied = BTreeSet::new();
        let mut banks = BTreeSet::new();
        for holder in &holders {
            if !Rc::ptr_eq(self, holder.store()) {
                return Err(Error::Request(
                    "state census holder belongs to another store".into(),
                ));
            }
            if !supplied.insert(holder.claims().id) {
                return Err(Error::Request(
                    "state census holder appears more than once".into(),
                ));
            }
            banks.insert(holder.bank().index());
        }
        let tracked = self.transactions.0.borrow();
        if live
            .iter()
            .chain(retained)
            .any(|holder| tracked.histories.contains_key(&holder.claims().id))
        {
            return Err(Error::Request(
                "state census marks a submitted history as live or retained".into(),
            ));
        }
        if arena
            .entries
            .keys()
            .any(|id| !supplied.contains(id) && !tracked.histories.contains_key(id))
        {
            return Err(Error::Request(
                "state census omits a registered history claim".into(),
            ));
        }
        let (rows, committed_banks) = self.committed();
        let placement = self.banks.inner.placement.borrow();
        let occupied_banks = placement
            .resources()
            .values()
            .map(|slot| slot.index)
            .collect::<BTreeSet<_>>();
        if (1..committed_banks).any(|bank| {
            occupied_banks.contains(&bank)
                && !banks.contains(&bank)
                && !tracked.banks.contains_key(&bank)
        }) {
            return Err(Error::Request(
                "state census omits a recurrent bank claim".into(),
            ));
        }
        let occupied_rows = u64::try_from(arena.referenced)
            .map_err(|_| Error::Request("occupied row count exceeds u64".into()))?;
        let used_banks = u64::try_from(
            (1..committed_banks)
                .filter(|bank| occupied_banks.contains(bank))
                .count(),
        )
        .map_err(|_| Error::Request("occupied bank count exceeds u64".into()))?;
        let occupied = occupied_rows
            .checked_mul(self.total_history_row_bytes)
            .and_then(|history| {
                used_banks
                    .checked_mul(self.recurrent_bank_bytes)
                    .and_then(|bank| history.checked_add(bank))
            })
            .ok_or_else(|| Error::Request("state census byte count overflow".into()))?;
        let model_seed = self.recurrent_bank_bytes;
        let committed = self.committed_bytes();
        let surplus = committed
            .checked_sub(occupied)
            .and_then(|bytes| bytes.checked_sub(model_seed))
            .ok_or_else(|| Error::Request("state census exceeds Seismic charged backing".into()))?;
        drop(placement);
        drop(arena);
        drop(tracked);
        let retained_only = self.exclusive_bytes(retained)?;
        let non_flight = live.iter().chain(retained).copied().collect::<Vec<_>>();
        let in_flight_bytes = occupied
            .checked_sub(self.exclusive_bytes(&non_flight)?)
            .ok_or_else(|| Error::Request("state census flight exceeds occupied bytes".into()))?;
        let live_bytes = occupied
            .checked_sub(retained_only)
            .and_then(|bytes| bytes.checked_sub(in_flight_bytes))
            .ok_or_else(|| Error::Request("state census classes overlap".into()))?;
        let census = StateHoldingCensus {
            surplus,
            retained: retained_only,
            live: live_bytes,
            in_flight: in_flight_bytes,
            model_seed,
        };
        if census.total() != committed || rows < self.occupied_rows() {
            return Err(Error::Request(
                "state census does not reconcile to committed backing".into(),
            ));
        }
        Ok(census)
    }
    pub fn history_capacity(&self) -> usize {
        self.history_capacity
    }
    pub fn history_components(&self) -> &[ComponentDescriptor] {
        &self.components
    }
    pub fn total_history_row_bytes(&self) -> u64 {
        self.total_history_row_bytes
    }
    pub fn total_history_bytes(&self) -> u64 {
        self.total_history_row_bytes * self.history_capacity as u64
    }
    pub fn allocation_trace(&self) -> Result<StateAllocationTrace, Error> {
        let zero_seed_bytes = self.recurrent_bank_bytes;
        let recurrent_pool_bytes = self
            .recurrent_bank_bytes
            .checked_mul(
                u64::try_from(self.bank_capacity.storage_total()?)
                    .map_err(|_| Error::Request("bank capacity exceeds u64".into()))?,
            )
            .ok_or_else(|| Error::Request("recurrent pool byte count overflow".into()))?;
        Ok(StateAllocationTrace {
            context_capacity: self.context_capacity,
            history_capacity: self.history_capacity,
            history_row_bytes: self.total_history_row_bytes,
            history_bytes: self.total_history_bytes(),
            bank_capacity: self.bank_capacity,
            recurrent_bank_bytes: self.recurrent_bank_bytes,
            zero_seed_bytes,
            recurrent_pool_bytes,
        })
    }
    pub fn history_allocated(&self) -> bool {
        !self.backing.borrow().history.is_empty()
    }

    /// Commit backing for a launch before any of its transactions begin: the
    /// rows the demands need beyond the free committed rows, plus the rows a
    /// history ending at the committed frontier needs to keep growing in
    /// place, and `banks` free successor banks. Growth is geometric and
    /// bounded by the reservation. A refused allocation returns a capacity
    /// error without publishing tentative backing or changing accepted state.
    /// Nothing changes while a transaction holds this store's tensors.
    pub fn provision(&self, demands: &[RowDemand], banks: usize) -> Result<(), Error> {
        self.provision_with_growth(demands, banks, GrowthChoice::Preferred)
    }

    /// A read-only claim for the additional peak charge of a launch's state
    /// growth. The minimum commits only rows and banks the launch needs; the
    /// preferred amount includes geometric headroom and eligible relayout.
    pub fn growth_claim(
        &self,
        demands: &[RowDemand],
        banks: usize,
    ) -> Result<StateGrowthClaim, Error> {
        Ok(StateGrowthClaim {
            minimum_bytes: self.growth_bytes(demands, banks, GrowthChoice::Minimum)?,
            preferred_bytes: self.growth_bytes(demands, banks, GrowthChoice::Preferred)?,
        })
    }

    fn growth_bytes(
        &self,
        demands: &[RowDemand],
        banks: usize,
        choice: GrowthChoice,
    ) -> Result<u64, Error> {
        if !self.transactions.idle() {
            return Ok(0);
        }
        let history = self.history_growth_plan(demands, choice)?;
        let backing = self.backing.borrow();
        let bank_target = self.bank_growth_target(banks, choice)?;
        let bytes = |rows: usize, row_bytes: u64| {
            u64::try_from(rows)
                .ok()
                .and_then(|rows| rows.checked_mul(row_bytes))
                .ok_or_else(|| Error::Request("state growth byte count overflow".into()))
        };
        let history_net = bytes(
            history.target.saturating_sub(backing.rows),
            self.total_history_row_bytes,
        )?;
        // The full replacement remains the safe claim for a backing that may
        // have external views or require a relayout. An exclusive CUDA VMM
        // recommit transfers its old charge and uses only the delta, so this
        // claim can be conservative until exclusivity is part of the plan.
        let history_peak = if history.relayout || history.target > backing.rows {
            bytes(history.target, self.total_history_row_bytes)?
        } else {
            history_net
        };
        let bank_net = bytes(
            bank_target.saturating_sub(backing.banks),
            self.recurrent_bank_bytes,
        )?;
        let bank_peak = if bank_target > backing.banks {
            bytes(bank_target, self.recurrent_bank_bytes)?
        } else {
            bank_net
        };
        Ok(history_peak.max(
            history_net
                .checked_add(bank_peak)
                .ok_or_else(|| Error::Request("state growth peak overflow".into()))?,
        ))
    }

    pub fn provision_with_growth(
        &self,
        demands: &[RowDemand],
        banks: usize,
        choice: GrowthChoice,
    ) -> Result<(), Error> {
        if !self.transactions.idle() {
            return Ok(());
        }
        if !self.components.is_empty() {
            self.provision_history(demands, choice)?;
        }
        let shortage = banks.saturating_sub(self.banks.available());
        if shortage != 0 && self.has_recurrent_components() {
            let target = self.bank_growth_target(banks, choice)?;
            self.recommit_banks(target)?;
        }
        Ok(())
    }

    fn bank_growth_target(&self, banks: usize, choice: GrowthChoice) -> Result<usize, Error> {
        let committed = self.backing.borrow().banks;
        let shortage = banks.saturating_sub(self.banks.available());
        if shortage == 0 || !self.has_recurrent_components() {
            return Ok(committed);
        }
        let required = committed.saturating_add(shortage);
        let reserved = self.bank_capacity.storage_total()?;
        Ok(match choice {
            GrowthChoice::Minimum => required.min(reserved),
            GrowthChoice::Preferred => grown(committed, required, BANK_GRANULE, reserved),
        })
    }

    /// History backing for a launch's demands. Every sequence history must
    /// stay one run, so a history that cannot grow in place is never split
    /// while the backing can make room: the store relays out every history
    /// (each live one contiguous, followed by growth room) into a backing
    /// grown when the rows in use plus demand and headroom exceed it.
    /// Backends that reallocate on growth pay that copy anyway, so every
    /// growth there is a relayout; a backend that resizes in place grows
    /// in place and relays out only for a history without room.
    fn history_growth_plan(
        &self,
        demands: &[RowDemand],
        choice: GrowthChoice,
    ) -> Result<HistoryGrowthPlan, Error> {
        let total = demands.iter().map(|demand| demand.rows).sum::<usize>();
        if total == 0 || self.components.is_empty() {
            return Ok(HistoryGrowthPlan {
                target: self.backing.borrow().rows,
                relayout: false,
                demanded: BTreeMap::new(),
                total: 0,
            });
        }
        let (committed, in_place) = {
            let backing = self.backing.borrow();
            let in_place = backing.history.first().is_none_or(Tensor::resizes_in_place);
            (backing.rows, in_place)
        };
        let (referenced, growing, frontier) = {
            let arena = self.arena.borrow();
            // Histories that grow in this launch, and the rows each needs.
            let mut growing = BTreeMap::new();
            for demand in demands {
                if let Some(end) = demand.after {
                    for id in arena.live_ending_at(end) {
                        growing.insert(id, (end, demand.rows));
                    }
                }
            }
            let frontier = arena
                .free
                .last()
                .filter(|(start, count)| start + count == committed)
                .map_or(committed, |(start, _)| *start);
            (arena.referenced, growing, frontier)
        };
        let required = referenced.saturating_add(total);
        let target = match choice {
            GrowthChoice::Minimum => required.max(committed).min(self.history_capacity),
            GrowthChoice::Preferred
                if required.saturating_add(referenced / HEADROOM_DIVISOR) > committed =>
            {
                grown(
                    committed,
                    required.saturating_add(referenced / HEADROOM_DIVISOR),
                    HISTORY_GRANULE,
                    self.history_capacity,
                )
            }
            GrowthChoice::Preferred => committed,
        };
        // A history without room that a relayout can give room to. In-place
        // growth extends the free rows after the frontier history.
        let stranded = {
            let arena = self.arena.borrow();
            growing.iter().any(|(&id, &(end, rows))| {
                let extension = if in_place && end == frontier {
                    target - committed
                } else {
                    0
                };
                arena.room_at(end) + extension < rows && arena.owns_its_end(id)
            })
        };
        let slack = target.saturating_sub(required);
        let relayout = referenced != 0
            && ((target > committed && !in_place)
                || (stranded
                    && (choice == GrowthChoice::Minimum
                        || slack >= referenced / RELAYOUT_SLACK_DIVISOR)));
        Ok(HistoryGrowthPlan {
            target,
            relayout,
            demanded: growing
                .into_iter()
                .map(|(id, (_, rows))| (id, rows))
                .collect(),
            total,
        })
    }

    fn provision_history(&self, demands: &[RowDemand], choice: GrowthChoice) -> Result<(), Error> {
        let plan = self.history_growth_plan(demands, choice)?;
        if plan.total == 0 {
            return Ok(());
        }
        let committed = self.backing.borrow().rows;
        if plan.relayout {
            self.relayout_history(plan.target, &plan.demanded, plan.total)?;
            return Ok(());
        }
        if plan.target > committed {
            self.recommit_history(plan.target)?;
        }
        Ok(())
    }

    /// Relay out every history into `rows` committed rows with
    /// [`Arena::relayout`], copying the referenced rows into new planes. The
    /// free rows beyond `demand` become growth room: each demanding history
    /// gets its demand, and every live history a share of the rest in
    /// proportion to the rows it may still grow (up to the context).
    fn relayout_history(
        &self,
        rows: usize,
        demanded: &BTreeMap<u64, usize>,
        demand: usize,
    ) -> Result<(), Error> {
        let mut backing = self.backing.borrow_mut();
        let mut arena = self.arena.borrow().clone();
        let required = arena.referenced.saturating_add(demand);
        let Some(spare) = rows.checked_sub(required) else {
            return Err(Error::Capacity {
                required: (required as u64).saturating_mul(self.total_history_row_bytes),
                available_bytes: (rows as u64).saturating_mul(self.total_history_row_bytes),
            });
        };
        let rooms = arena
            .entries
            .iter()
            .filter(|(_, entry)| entry.live)
            .map(|(&id, entry)| (id, self.context_capacity.saturating_sub(entry.rows())))
            .collect::<Vec<_>>();
        let total_room = rooms.iter().map(|(_, room)| room).sum::<usize>().max(1);
        let gaps = rooms
            .into_iter()
            .map(|(id, room)| {
                let share = (spare as u128 * room as u128 / total_room as u128) as usize;
                (
                    id,
                    demanded.get(&id).copied().unwrap_or(0) + share.min(room),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let moves = arena
            .relayout(rows, &gaps)
            .into_iter()
            .map(|(from, to, count)| (from as u64, to as u64, count as u64))
            .collect::<Vec<_>>();
        let planes = backing
            .history
            .iter()
            .map(|plane| plane.relocated(rows as u64, &moves))
            .collect::<Result<Vec<_>, _>>();
        let planes = planes?;
        let copied = moves
            .iter()
            .map(|(_, _, count)| *count as usize)
            .sum::<usize>();
        let retired = backing
            .history
            .iter()
            .map(Tensor::observe_storage)
            .collect();
        *self.arena.borrow_mut() = arena;
        backing.history = planes;
        self.record_retired_storage(retired);
        backing.rows = rows;
        let stats = self.relayouts.get();
        self.relayouts.set(Relayouts {
            count: stats.count + 1,
            rows: stats.rows + copied,
        });
        Ok(())
    }

    /// History relayouts performed so far and the rows they copied.
    pub fn relayouts(&self) -> Relayouts {
        self.relayouts.get()
    }

    /// Release committed history rows and banks beyond what the store needs:
    /// the referenced rows plus growth headroom, and the claimed banks plus a
    /// successor for every owner plus a granule. Idle shrinking releases only
    /// once the backing is at least twice that, so serving never alternates
    /// between growing and shrinking; pressure releases everything beyond it.
    /// An unreferenced tail is released in place; live rows above the kept
    /// rows are relaid out downward first. Returns the physical bytes
    /// released. Nothing changes while a transaction exists.
    pub fn shrink(&self, policy: ShrinkPolicy) -> Result<u64, Error> {
        if !self.transactions.idle() {
            return Ok(0);
        }
        let release = |committed: usize, needed: usize| match policy {
            ShrinkPolicy::Idle => committed >= 2 * needed,
            ShrinkPolicy::Pressure => committed > needed,
        };
        // Idle keeps the room growth would give the rows in use again (a
        // relayout copies every live row, so it must not recur as requests
        // come and go); pressure keeps only the growth headroom.
        // An earlier backing may still be pinned by a caller that holds a
        // tensor view. Only the device ledger can say how much was actually
        // released; the change in the store's current backing is insufficient.
        let before = self.device.memory_usage().charged;
        let (rows, banks) = self.committed();
        let (top, referenced) = {
            let arena = self.arena.borrow();
            let top = arena
                .runs
                .last_key_value()
                .map_or(0, |(start, run)| start + run.count);
            (top, arena.referenced)
        };
        let needed = history_shrink_target(referenced, policy);
        if rows != 0 && release(rows, needed) {
            let result = if top <= needed {
                self.recommit_history(needed)
            } else {
                self.relayout_history(needed, &BTreeMap::new(), 0)
            };
            if let Err(error) = result {
                if !allocation_capacity(&error) {
                    return Err(error);
                }
            }
        }
        if self.has_recurrent_components() {
            let claimed = banks - 1 - self.banks.available();
            let dense_needed = 1 + claimed + self.owners.get() + BANK_GRANULE;
            if policy == ShrinkPolicy::Pressure
                && dense_needed < self.banks.required(banks)
                && dense_needed < banks
            {
                if let Err(error) = self.relocate_banks(dense_needed) {
                    if !allocation_capacity(&error) {
                        return Err(error);
                    }
                }
            }
            let needed = dense_needed.max(self.banks.required(self.committed().1));
            let banks = self.committed().1;
            if release(banks, needed) {
                if let Err(error) = self.recommit_banks(needed) {
                    if !allocation_capacity(&error) {
                        return Err(error);
                    }
                }
            }
        }
        Ok(before.saturating_sub(self.device.memory_usage().charged))
    }

    /// Recommit every history plane to `rows` leading rows. The arena gains
    /// the new rows as a hole, or loses a tail that no claim references.
    fn recommit_history(&self, rows: usize) -> Result<(), Error> {
        let mut backing = self.backing.borrow_mut();
        if rows == backing.rows {
            return Ok(());
        }
        let planes = if backing.history.is_empty() {
            self.components
                .iter()
                .flat_map(ComponentDescriptor::planes)
                .map(|plane| {
                    Tensor::reserved(
                        &self.device,
                        element(plane.dtype),
                        &leading_extents(self.history_capacity, &plane.row_extents)?,
                        rows as u64,
                    )
                    .map_err(Error::from)
                })
                .collect::<Result<Vec<_>, _>>()
        } else {
            recommit_planes(&backing.history, rows as u64)
        };
        let planes = planes?;
        let retired = backing
            .history
            .iter()
            .map(Tensor::observe_storage)
            .collect();
        self.arena.borrow_mut().resize(backing.rows, rows);
        backing.history = planes;
        self.record_retired_storage(retired);
        backing.rows = rows;
        self.recommits.set(self.recommits.get() + 1);
        Ok(())
    }

    /// Times the history rows or banks were recommitted (grown or shrunk
    /// without a relayout); each may copy the backing.
    pub fn recommits(&self) -> usize {
        self.recommits.get()
    }

    /// Prepare a complete dense destination while the old backing and
    /// placement remain published. Every plane is copied before either the
    /// physical backing or its placement generation changes.
    fn relocate_banks(&self, target: usize) -> Result<(), Error> {
        assert!(
            self.transactions.idle(),
            "bank relocation requires an idle store"
        );
        let mut backing = self.backing.borrow_mut();
        let published = self.banks.inner.placement.borrow().clone();
        let allocation = AllocationId(self.banks.inner.next_allocation.get());
        let plan = published
            .plan_dense_prefix(allocation, target)
            .map_err(|error| Error::Request(format!("bank relocation plan: {error:?}")))?;
        let moves = plan
            .copies()
            .iter()
            .map(|copy| (copy.from.index as u64, copy.to.index as u64, 1u64))
            .collect::<Vec<_>>();
        let destination = backing
            .recurrent
            .iter()
            .map(|plane| plane.relocated(target as u64, &moves).map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;
        let mut transaction = plan.begin();
        for index in 0..transaction.copies().len() {
            transaction
                .mark_copied(index)
                .map_err(|error| Error::Request(format!("bank relocation copy: {error:?}")))?;
        }
        let retired = backing
            .recurrent
            .iter()
            .map(Tensor::observe_storage)
            .collect();
        transaction
            .commit(&mut self.banks.inner.placement.borrow_mut())
            .map_err(|error| Error::Request(format!("bank relocation commit: {error:?}")))?;
        self.banks.inner.next_allocation.set(
            allocation
                .0
                .checked_add(1)
                .expect("bank allocation id exhausted"),
        );
        backing.recurrent = destination.into();
        backing.banks = target;
        self.record_retired_storage(retired);
        self.recommits.set(self.recommits.get() + 1);
        Ok(())
    }

    fn recommit_banks(&self, banks: usize) -> Result<(), Error> {
        let mut backing = self.backing.borrow_mut();
        if banks == backing.banks {
            return Ok(());
        }
        if banks < backing.banks {
            if let Some(planes) = Rc::get_mut(&mut backing.recurrent) {
                if planes.iter().all(Tensor::can_recommit_in_place) {
                    // The released bank tail is free. Publish each CUDA VMM
                    // shrink immediately, so no second plane needs a peak
                    // replacement allocation under pressure. A driver error
                    // after the first publication cannot be reported as a
                    // recoverable old backing.
                    let mut published = 0;
                    for plane in planes {
                        let replacement = match plane.recommitted(banks as u64) {
                            Ok(replacement) => replacement,
                            Err(error) if published == 0 => return Err(error.into()),
                            Err(error) => panic!("partial in-place bank shrink failed: {error}"),
                        };
                        let old = std::mem::replace(plane, replacement);
                        drop(old);
                        published += 1;
                    }
                    self.banks.resize(backing.banks, banks);
                    backing.banks = banks;
                    self.recommits.set(self.recommits.get() + 1);
                    return Ok(());
                }
            }
        }
        let arenas: Rc<[Tensor]> = recommit_planes(&backing.recurrent, banks as u64)?.into();
        let retired = backing
            .recurrent
            .iter()
            .map(Tensor::observe_storage)
            .collect();
        self.banks.resize(backing.banks, banks);
        backing.recurrent = arenas;
        self.record_retired_storage(retired);
        backing.banks = banks;
        self.recommits.set(self.recommits.get() + 1);
        Ok(())
    }

    pub fn history_planes(&self) -> Result<Vec<PlaneBuffer>, Error> {
        let backing = self.backing.borrow();
        let tensors = &backing.history;
        Ok(self
            .components
            .iter()
            .enumerate()
            .flat_map(|(component_index, component)| {
                component
                    .planes()
                    .iter()
                    .map(move |plane| (component_index, component.layer, plane))
            })
            .zip(tensors.iter())
            .enumerate()
            .map(
                |(plane_index, ((component_index, layer, plane), buffer))| PlaneBuffer {
                    plane_index,
                    component_index,
                    layer,
                    vector: plane.vector,
                    name: plane.name,
                    row_bytes: plane.row_bytes,
                    base_row: 0,
                    buffer: buffer.clone(),
                },
            )
            .collect())
    }
    /// History rows referenced by at least one claim; a shared row counts once.
    pub fn occupied_rows(&self) -> usize {
        self.arena.borrow().referenced
    }
    /// Bytes released if exactly the given holders were dropped: history rows
    /// and recurrent banks that no claim outside the set references. Shared
    /// prefixes are priced once, as a set; checkpoints, descendants and
    /// in-flight advances outside the set pin what they reference, and the
    /// zero seed is never released. Repeated holders count once.
    pub fn exclusive_bytes(self: &Rc<Self>, holders: &[Holder<'_>]) -> Result<u64, Error> {
        let mut distinct: Vec<&Holder<'_>> = Vec::with_capacity(holders.len());
        for holder in holders {
            if !Rc::ptr_eq(self, holder.store()) {
                return Err(Error::Request(
                    "exclusive accounting requires holders from this store".into(),
                ));
            }
            if !distinct.iter().any(|seen| seen.same(holder)) {
                distinct.push(holder);
            }
        }
        let ranges = distinct
            .iter()
            .flat_map(|holder| holder.claims().ranges())
            .collect::<Vec<_>>();
        let rows = self.arena.borrow().exclusive_rows(&ranges);
        let mut banks = 0u64;
        let mut counted: Vec<*const BankClaim> = Vec::new();
        for holder in &distinct {
            let bank = holder.bank();
            let claim = Rc::as_ptr(&bank.0);
            if bank.index() == ZERO_SEED_BANK || counted.contains(&claim) {
                continue;
            }
            counted.push(claim);
            let selected = distinct
                .iter()
                .filter(|other| Rc::ptr_eq(&other.bank().0, &bank.0))
                .count();
            if selected == Rc::strong_count(&bank.0) {
                banks += 1;
            }
        }
        (rows as u64)
            .checked_mul(self.total_history_row_bytes)
            .and_then(|history| {
                self.recurrent_bank_bytes
                    .checked_mul(banks)
                    .and_then(|recurrent| history.checked_add(recurrent))
            })
            .ok_or_else(|| Error::Request("exclusive state byte count overflow".into()))
    }
    pub fn idle(&self) -> bool {
        self.owners.get() == 0
    }

    pub fn available_banks(&self) -> usize {
        self.banks.available()
    }
    pub fn available_rows(&self) -> usize {
        if self.components.is_empty() {
            return usize::MAX;
        }
        self.arena.borrow().available()
    }
    /// Drop the store's arena allocations when no sequence/checkpoint owns them.
    /// External completion/buffer pins may still retain physical storage.
    pub fn release_idle(&self) -> Result<usize, Error> {
        if !self.idle() || !self.transactions.idle() {
            return Ok(0);
        }
        let planes = {
            let mut backing = self.backing.borrow_mut();
            backing.rows = 0;
            std::mem::take(&mut backing.history)
        };
        *self.arena.borrow_mut() = Arena::new(0);
        usize::try_from(Tensor::reclaimable_bytes(planes.iter())?)
            .map_err(|_| Error::Request("reclaimable history bytes exceed host range".into()))
    }
    pub fn create(self: &Rc<Self>) -> Result<SequenceState, Error> {
        let bank = self.zero_seed.clone();
        let claims = Claims::new(&self.arena, vec![]);
        claims.set_live(true);
        self.owners.set(self.owners.get() + 1);
        Ok(SequenceState {
            store: self.clone(),
            position: 0,
            expected_end: 0,
            history_start: 0,
            claims,
            bank,
            tape: 0,
        })
    }
    /// Reserve `count` rows, in logical order, to follow `history` (none for
    /// rows of a new history; see [`Arena`] for placement). Provisioning may
    /// relay the history out, so its end is read again before claiming.
    fn reserve(&self, history: Option<&Claims>, count: usize) -> Result<Claims, Error> {
        if self.components.is_empty() {
            return Ok(Claims::new(&self.arena, vec![]));
        }
        let after = history.and_then(Claims::end);
        let needs_preparation = {
            let arena = self.arena.borrow();
            arena.available() < count
                || (self.transactions.idle()
                    && history.is_some_and(|claim| {
                        after.is_some_and(|end| {
                            arena.room_at(end) < count && arena.owns_its_end(claim.id)
                        })
                    }))
        };
        // Exact row demand can require a same-size relayout even when free
        // rows suffice in total. Prepare that layout before reserving a
        // transaction; Seismic still enforces any physical peak charge.
        if needs_preparation {
            self.provision(&[RowDemand { after, rows: count }], 0)?;
        }
        let after = history.and_then(Claims::end);
        let mut arena = self.arena.borrow_mut();
        let available_rows = arena.available();
        if available_rows < count {
            let capacity = self.history_capacity as u64;
            let total_bytes = self.total_history_bytes();
            return Err(Error::Capacity {
                required: capacity_charge(count, total_bytes, capacity, true)?,
                available_bytes: capacity_charge(available_rows, total_bytes, capacity, false)?,
            });
        }
        let ranges = arena.claim(after, count);
        drop(arena);
        Ok(Claims::new(&self.arena, ranges))
    }

    /// A free successor bank, committing more banks first when none is free
    /// and no transaction holds this store's tensors.
    fn successor_bank(&self) -> Result<BankHandle, Error> {
        if self.banks.available() == 0 {
            self.provision(&[], 1)?;
        }
        self.banks.acquire()
    }

    fn begin_transaction(&self) -> Transaction {
        self.transactions.begin()
    }

    fn reserve_contiguous(&self, count: usize) -> Option<Claims> {
        if count == 0 || self.components.is_empty() {
            return None;
        }
        let start = self.arena.borrow_mut().claim_contiguous(count)?;
        Some(Claims::new(&self.arena, vec![(start, count)]))
    }
}

/// A claim holder priced by [`StateStore::exclusive_bytes`].
#[derive(Clone, Copy)]
pub enum Holder<'a> {
    State(&'a SequenceState),
    Checkpoint(&'a StateCheckpoint),
}

impl Holder<'_> {
    fn store(&self) -> &Rc<StateStore> {
        match self {
            Self::State(state) => &state.store,
            Self::Checkpoint(checkpoint) => &checkpoint.store,
        }
    }
    fn claims(&self) -> &Claims {
        match self {
            Self::State(state) => &state.claims,
            Self::Checkpoint(checkpoint) => &checkpoint.claims,
        }
    }
    fn bank(&self) -> &BankHandle {
        match self {
            Self::State(state) => &state.bank,
            Self::Checkpoint(checkpoint) => &checkpoint.bank,
        }
    }
    fn same(&self, other: &Holder<'_>) -> bool {
        match (self, other) {
            (Holder::State(left), Holder::State(right)) => std::ptr::eq(*left, *right),
            (Holder::Checkpoint(left), Holder::Checkpoint(right)) => std::ptr::eq(*left, *right),
            _ => false,
        }
    }
}

fn validate_history_layout(
    components: &[ComponentDescriptor],
    history_capacity: usize,
) -> Result<u64, LayoutError> {
    let mut layers = BTreeSet::new();
    let mut total_row_bytes = 0u64;
    for component in components {
        if !layers.insert(component.layer) {
            return Err(LayoutError::DuplicateLayer(component.layer));
        }
        for plane in component.planes() {
            let row_bytes = u64::try_from(plane.row_bytes)
                .map_err(|_| LayoutError::ArithmeticOverflow("plane row bytes"))?;
            row_bytes
                .checked_mul(history_capacity as u64)
                .ok_or(LayoutError::ArithmeticOverflow("history plane capacity"))?;
            total_row_bytes = total_row_bytes
                .checked_add(row_bytes)
                .ok_or(LayoutError::ArithmeticOverflow("history row bytes"))?;
        }
    }
    total_row_bytes
        .checked_mul(history_capacity as u64)
        .ok_or(LayoutError::ArithmeticOverflow("total history capacity"))?;
    Ok(total_row_bytes)
}

fn capacity_charge(
    rows: usize,
    total_bytes: u64,
    history_capacity: u64,
    round_up: bool,
) -> Result<u64, LayoutError> {
    let numerator = (rows as u64)
        .checked_mul(total_bytes)
        .ok_or(LayoutError::ArithmeticOverflow("state capacity charge"))?;
    Ok(if round_up {
        numerator.div_ceil(history_capacity)
    } else {
        numerator / history_capacity
    })
}
pub struct SequenceState {
    store: Rc<StateStore>,
    position: usize,
    expected_end: usize,
    history_start: usize,
    /// Claims on exactly the visible rows `[history_start, position)`, in
    /// logical order; a live history.
    claims: Claims,
    bank: BankHandle,
    /// Rows of `bank`'s tape that complete the accepted recurrent state: the
    /// bank holds the state `tape` rows before `position`, and the next
    /// advance replays those tape rows before its own (see
    /// [`OwnedStateAdvance::begin_speculative`]). 0 after a plain advance.
    tape: usize,
}
impl Drop for SequenceState {
    fn drop(&mut self) {
        self.store.owners.set(self.store.owners.get() - 1);
    }
}
impl SequenceState {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        Rc::ptr_eq(&self.store, store)
    }

    pub fn expected_end(&self) -> usize {
        self.expected_end
    }
    /// The accepted recurrent bank: the row of every recurrent arena that
    /// holds this sequence's state.
    pub fn bank_index(&self) -> usize {
        self.bank.index()
    }
    /// The tape rows of the accepted bank that complete this sequence's
    /// recurrent state; the next advance reads version (bank, tape).
    pub fn tape_rows(&self) -> usize {
        self.tape
    }
    pub fn anticipate(&mut self, position: usize) -> Result<(), String> {
        if position > self.store.context_capacity {
            return Err("anticipated position exceeds context capacity".into());
        }
        self.expected_end = self.expected_end.max(position);
        Ok(())
    }
    /// The arena row just past this sequence's last accepted row: where its
    /// next rows continue its history without a new segment.
    fn history_end(&self) -> Option<usize> {
        self.claims.end()
    }
    /// This sequence's demand for appending `rows`, for provisioning a
    /// launch's backing before its advances begin.
    pub fn demand(&self, rows: usize) -> RowDemand {
        RowDemand {
            after: self.history_end(),
            rows,
        }
    }
    pub fn history_ranges(&self) -> Vec<(usize, usize)> {
        self.claims.ranges()
    }
    /// Stop seeing rows before logical position `before`. Exactly the trimmed
    /// rows lose this sequence's reference.
    pub fn trim_history(&mut self, before: usize) -> Result<(), String> {
        if before < self.history_start || before > self.position {
            return Err("history trim must lie within accepted logical positions".into());
        }
        if !self.store.components.is_empty() {
            self.claims.drop_front(before - self.history_start);
        }
        self.history_start = before;
        Ok(())
    }

    /// At the launch segment limit: the next advance could start one more
    /// run, so the history is repacked before its next launch.
    pub fn compaction_needed(&self) -> bool {
        self.history_ranges().len() >= MAX_VISIBLE_SEGMENTS
    }

    pub fn checkpoint(&self) -> StateCheckpoint {
        self.store.owners.set(self.store.owners.get() + 1);
        StateCheckpoint {
            store: self.store.clone(),
            position: self.position,
            history_start: self.history_start,
            claims: self.claims.clone(),
            bank: self.bank.clone(),
            tape: self.tape,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneCopy {
    pub plane_index: usize,
    pub from: Vec<usize>,
    pub to: Vec<usize>,
}

pub struct StateCheckpoint {
    store: Rc<StateStore>,
    position: usize,
    history_start: usize,
    claims: Claims,
    bank: BankHandle,
    tape: usize,
}

/// An accepted state temporarily owned by a submitted continuation rather
/// than the domain's request map. Its history and bank remain in the
/// in-flight census until the continuation reconciles or is dropped.
pub struct InFlightState {
    state: SequenceState,
    _transaction: Transaction,
}

impl InFlightState {
    pub fn new(state: SequenceState) -> Self {
        let mut transaction = state.store.begin_transaction();
        transaction.track(&state.claims, &state.bank);
        Self {
            state,
            _transaction: transaction,
        }
    }

    pub fn into_state(self) -> SequenceState {
        self.state
    }
}
impl Drop for StateCheckpoint {
    fn drop(&mut self) {
        self.store.owners.set(self.store.owners.get() - 1);
    }
}
impl StateCheckpoint {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn bank_index(&self) -> usize {
        self.bank.index()
    }
    pub fn tape_rows(&self) -> usize {
        self.tape
    }
    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        Rc::ptr_eq(&self.store, store)
    }
    pub fn store(&self) -> &Rc<StateStore> {
        &self.store
    }
    pub fn history_ranges(&self) -> Vec<(usize, usize)> {
        self.claims.ranges()
    }
    pub fn fork(&self) -> SequenceState {
        let claims = self.claims.clone();
        claims.set_live(true);
        self.store.owners.set(self.store.owners.get() + 1);
        SequenceState {
            store: self.store.clone(),
            position: self.position,
            expected_end: self.position,
            history_start: self.history_start,
            claims,
            bank: self.bank.clone(),
            tape: self.tape,
        }
    }
}
/// Publish `count` rows whose recurrent version is (`following`, `tape`).
fn install_commit(
    state: &mut SequenceState,
    claims: Claims,
    following: &mut BankHandle,
    tape: usize,
    count: usize,
) {
    // Appending moves references: what a checkpoint or fork sharing this
    // history's rows sees never changes.
    state.claims.append(claims);
    std::mem::swap(&mut state.bank, following);
    state.tape = tape;
    state.position += count;
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, DeviceCatalog};

    fn cpu_device() -> Option<Rc<Device>> {
        DeviceCatalog::discover()
            .ok()
            .and_then(|catalog| catalog.open_backend(BackendName::Cpu).ok())
            .map(Rc::new)
    }

    fn bank_view(store: &StateStore, bank: usize) -> Tensor {
        store.recurrent_arenas()[0]
            .slice_leading(bank as u64, bank as u64 + 1)
            .unwrap()
    }

    fn bank_bytes(store: &StateStore, bank: usize) -> Vec<u8> {
        bank_view(store, bank).read_to_host().unwrap()
    }

    fn write_bank(store: &StateStore, bank: usize, bytes: &[u8]) {
        bank_view(store, bank).write_from_host(bytes).unwrap();
    }

    fn dense_component(width: usize) -> ComponentDescriptor {
        ComponentDescriptor::new(
            LayerRef::Target(0),
            CodecSpec::dense(DType::F32, width, width),
            1,
        )
        .unwrap()
    }

    #[test]
    fn component_specs_reject_non_float_empty_and_zero_shapes() {
        for spec in [
            ComponentSpec {
                shape: vec![],
                dtype: DType::F32,
            },
            ComponentSpec {
                shape: vec![4, 0],
                dtype: DType::F32,
            },
            ComponentSpec {
                shape: vec![4],
                dtype: DType::I32,
            },
        ] {
            assert_eq!(
                spec.bytes().unwrap_err(),
                "state components require nonempty floating tensors"
            );
        }
    }

    /// Claim exactly `[start, start + count)`, which must lie in one hole.
    fn claim_rows(arena: &Rc<RefCell<Arena>>, start: usize, count: usize) -> Claims {
        let mut inner = arena.borrow_mut();
        let hole = inner
            .free
            .iter()
            .position(|(hole, size)| *hole <= start && start + count <= hole + size)
            .expect("rows lie in one hole");
        let offset = start - inner.free[hole].0;
        inner.take(hole, offset, count);
        drop(inner);
        Claims::new(arena, vec![(start, count)])
    }

    /// One history of the given rows, each claimed as in [`claim_rows`].
    fn history(arena: &Rc<RefCell<Arena>>, ranges: &[(usize, usize)]) -> Claims {
        let mut claims = Claims::new(arena, vec![]);
        for &(start, count) in ranges {
            claims.append(claim_rows(arena, start, count));
        }
        claims
    }

    fn arena(rows: usize) -> Rc<RefCell<Arena>> {
        Rc::new(RefCell::new(Arena::new(rows)))
    }

    #[test]
    fn dropping_claims_coalesces_adjacent_arena_ranges() {
        let arena = arena(16);
        let left = claim_rows(&arena, 2, 3);
        let right = claim_rows(&arena, 5, 4);
        assert_eq!(arena.borrow().free, vec![(0, 2), (9, 7)]);
        drop(right);
        drop(left);
        assert_eq!(arena.borrow().free, vec![(0, 16)]);
        assert_eq!(arena.borrow().referenced, 0);
        assert!(arena.borrow().runs.is_empty());
    }

    #[test]
    fn rows_are_shared_at_row_granularity_and_freed_by_their_last_claim() {
        let arena = arena(32);
        let path = claim_rows(&arena, 0, 12);
        // Two branches share the first 8 and first 5 rows of the path.
        let mut eight = path.clone();
        drop(eight.split_off(8));
        let mut five = path.clone();
        drop(five.split_off(5));
        assert_eq!(arena.borrow().referenced, 12);
        // Pricing is by set: the path alone owns rows 8..12; the path and
        // the 8-row branch own 5..12; all three own everything.
        assert_eq!(arena.borrow().exclusive_rows(&[(0, 12)]), 4);
        assert_eq!(arena.borrow().exclusive_rows(&[(0, 12), (0, 8)]), 7);
        assert_eq!(
            arena.borrow().exclusive_rows(&[(0, 12), (0, 8), (0, 5)]),
            12
        );
        assert_eq!(arena.borrow().exclusive_rows(&[(0, 5)]), 0);
        drop(path);
        assert_eq!(arena.borrow().referenced, 8);
        assert_eq!(arena.borrow().free, vec![(8, 24)]);
        drop(eight);
        assert_eq!(arena.borrow().free, vec![(5, 27)]);
        // A history joins its physical successor without touching references.
        let tail = claim_rows(&arena, 5, 3);
        five.append(tail);
        assert_eq!(five.ranges(), [(0, 8)]);
        assert_eq!(arena.borrow().runs.len(), 1);
        drop(five);
        assert_eq!(arena.borrow().free, vec![(0, 32)]);
        assert!(arena.borrow().runs.is_empty());
    }

    #[test]
    fn arena_grows_histories_in_place_and_starts_new_ones_mid_hole() {
        let mut arena = Arena::new(100);
        // Row 0 has no preceding history: fill from its start.
        assert_eq!(arena.claim(None, 10), [(0, 10)]);
        // A new history leaves the rows after [0, 10) as that history's room.
        assert_eq!(arena.claim(None, 10), [(50, 10)]);
        assert_eq!(arena.free, [(10, 40), (60, 40)]);
        // Both grow in place, one row at a time, whatever the interleaving.
        for step in 0..5 {
            assert_eq!(arena.claim(Some(10 + step), 1), [(10 + step, 1)]);
            assert_eq!(arena.claim(Some(60 + step), 1), [(60 + step, 1)]);
        }
        // A third history starts mid-way through the largest hole (ties go to
        // the lower address); a history without room in place follows it.
        assert_eq!(arena.claim(None, 5), [(30, 5)]);
        assert_eq!(arena.claim(Some(35), 20), [(35, 15), (80, 5)]);
        assert_eq!(arena.free, [(15, 15), (65, 15), (85, 15)]);
        // A request larger than every hole takes whole holes, largest first.
        assert_eq!(arena.claim(Some(3), 40), [(15, 15), (65, 15), (87, 10)]);
        assert_eq!(arena.free, [(85, 2), (97, 3)]);
        assert_eq!(arena.available(), 5);
    }

    #[test]
    fn planes_allocate_lazily_as_one_stable_set() {
        let Some(device) = cpu_device() else {
            // A backend may be present yet fail Seismic's runtime calibration
            // under a noisy test host. Never substitute an accelerator here.
            return;
        };
        let store = StateStore::new(
            device,
            4,
            8,
            vec![dense_component(4)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        assert!(!store.history_allocated());
        assert!(store.history_planes().unwrap().is_empty());
        assert_eq!(store.total_history_row_bytes(), 32);
        assert_eq!(store.total_history_bytes(), 256);

        store
            .provision(
                &[RowDemand {
                    after: None,
                    rows: 1,
                }],
                0,
            )
            .unwrap();
        let first = store.history_planes().unwrap();
        assert!(store.history_allocated());
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].plane_index, 0);
        assert_eq!(first[1].plane_index, 1);
        assert_eq!(first[0].vector, VectorKind::Key);
        assert_eq!(first[1].vector, VectorKind::Value);
        assert!(first.iter().all(|plane| {
            plane.component_index == 0
                && plane.layer == LayerRef::Target(0)
                && plane.name == PlaneName::Dense
                && plane.row_bytes == 16
                && plane.base_row == 0
                && plane.buffer.extents() == [8, 1, 4]
        }));

        let second = store.history_planes().unwrap();
        assert!(first
            .iter()
            .zip(second.iter())
            .all(|(left, right)| left.buffer.shares_allocation(&right.buffer)));
    }

    #[test]
    fn capacity_error_uses_all_plane_bytes() {
        let row_bytes = validate_history_layout(&[dense_component(4)], 8).unwrap();
        let total_bytes = row_bytes * 8;
        assert_eq!(capacity_charge(3, total_bytes, 8, true).unwrap(), 96);
        assert_eq!(capacity_charge(2, total_bytes, 8, false).unwrap(), 64);
    }

    #[test]
    fn store_rejects_total_capacity_overflow_before_allocation() {
        let component = dense_component(usize::MAX / 4);
        assert!(matches!(
            validate_history_layout(&[component], 2),
            Err(LayoutError::ArithmeticOverflow(_))
        ));
    }

    #[test]
    fn duplicate_layer_descriptors_are_rejected() {
        let component = dense_component(4);
        assert!(matches!(
            validate_history_layout(&[component.clone(), component], 2),
            Err(LayoutError::DuplicateLayer(LayerRef::Target(0)))
        ));
    }

    #[test]
    fn prefix_split_handles_zero_interior_full_and_fragmented_claims() {
        for (accepted, expected_kept, expected_released) in [
            (0, vec![], vec![(0, 3), (8, 4)]),
            (5, vec![(0, 3), (8, 2)], vec![(10, 2)]),
            (7, vec![(0, 3), (8, 4)], vec![]),
        ] {
            let arena = arena(12);
            let mut kept = history(&arena, &[(0, 3), (8, 4)]);
            let released = kept.split_off(accepted);
            assert_eq!(kept.ranges(), expected_kept);
            assert_eq!(released.ranges(), expected_released);
        }
    }

    #[test]
    fn rejected_tail_is_not_recycled_while_a_claim_is_pinned() {
        let arena = arena(11);
        let mut kept = claim_rows(&arena, 0, 2);
        let tail = claim_rows(&arena, 8, 3);
        // A shared history still splits: the pin keeps exactly its rows.
        let tail_pin = tail.clone();
        kept.append(tail);
        let released = kept.split_off(3);
        assert_eq!(kept.ranges(), [(0, 2), (8, 1)]);
        drop(released);
        assert_eq!(arena.borrow().free, [(2, 6)]);
        drop(tail_pin);
        assert_eq!(arena.borrow().free, [(2, 6), (9, 2)]);
        drop(kept);
        assert_eq!(arena.borrow().free, [(0, 11)]);
    }

    #[test]
    fn prefix_commit_is_transactional_and_publishes_a_tape_version() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            8,
            8,
            vec![dense_component(1)],
            vec![ComponentSpec {
                shape: vec![1],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 2,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let state = store.create().unwrap();
        let original = state.bank_index();
        assert_eq!(original, ZERO_SEED_BANK);
        let checkpoint = state.checkpoint();

        let advance = OwnedStateAdvance::begin_speculative(state, 3, 1)
            .ok()
            .unwrap();
        assert_eq!(advance.bindings().previous_bank, original);
        assert_ne!(advance.bindings().following_bank, original);
        assert!(advance.bindings().recurrent[0].shares_allocation(&store.recurrent_arenas()[0]));
        assert_eq!(store.occupied_rows(), 3);
        let state = advance.abort();
        assert_eq!(state.position(), 0);
        assert_eq!(state.bank_index(), original);
        assert_eq!(store.occupied_rows(), 0);

        let advance = OwnedStateAdvance::begin_speculative(state, 3, 1)
            .ok()
            .unwrap();
        let successor = advance.bindings().following_bank;
        let OwnedAdvanceResolution::Committed(state) = advance.commit(2).ok().unwrap() else {
            panic!("an accepted prefix commits as a tape version");
        };
        assert_eq!(state.position(), 2);
        assert_eq!((state.bank_index(), state.tape_rows()), (successor, 1));
        assert_eq!(store.occupied_rows(), 2);
        assert_eq!(checkpoint.position(), 0);
        let checkpoint_fork = checkpoint.fork();
        assert_eq!(checkpoint_fork.position(), 0);
        assert_eq!(checkpoint_fork.bank_index(), original);
        assert!(checkpoint_fork.history_ranges().is_empty());
        drop(checkpoint_fork);

        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let OwnedAdvanceResolution::Aborted(state) = advance.commit(0).ok().unwrap() else {
            panic!("zero prefix must abort");
        };
        assert_eq!(state.position(), 2);
        assert_eq!(store.occupied_rows(), 2);
        // A plain advance has no interior recurrent version.
        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let (state, _) = advance.commit(1).err().unwrap();
        assert_eq!((state.position(), state.tape_rows()), (2, 1));
        assert_eq!(store.occupied_rows(), 2);
        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        assert_eq!(state.position(), 4);
        assert_eq!(store.occupied_rows(), 4);
    }

    /// Shared system prompt, two divergent requests and retained checkpoints
    /// on one path: every holder sees the same physical prefix rows, the
    /// store holds them once, and pricing a set charges them once.
    #[test]
    fn shared_prefix_is_the_same_rows_and_is_charged_once() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            64,
            256,
            vec![dense_component(1)],
            vec![ComponentSpec {
                shape: vec![1],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 4,
                in_flight: 4,
                retained: 4,
            },
        )
        .unwrap();
        let row = store.total_history_row_bytes();
        let bank = store.allocation_trace().unwrap().recurrent_bank_bytes;
        let commit = |state: SequenceState, rows: usize| {
            let advance = OwnedStateAdvance::begin(state, rows).ok().unwrap();
            let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap()
            else {
                panic!("full prefix must commit");
            };
            state
        };
        // The system prompt is one prefill run.
        let prompt = commit(store.create().unwrap(), 24);
        let system = prompt.checkpoint();
        assert_eq!(system.history_ranges(), [(0, 24)]);
        // Request A continues the path; requests B and C branch at the prompt.
        let a = commit(prompt, 8);
        let b = commit(system.fork(), 6);
        let c = commit(system.fork(), 3);
        let a_turn = a.checkpoint();
        assert_eq!(a.history_ranges(), [(0, 32)]);
        for branch in [&b, &c] {
            assert_eq!(branch.history_ranges()[0], (0, 24));
            assert_eq!(branch.history_ranges().len(), 2);
        }
        assert_eq!(store.occupied_rows(), 24 + 8 + 6 + 3);
        let census = store
            .holding_census(
                &[Holder::State(&a), Holder::State(&b), Holder::State(&c)],
                &[Holder::Checkpoint(&system), Holder::Checkpoint(&a_turn)],
                &[],
            )
            .unwrap();
        assert_eq!(census.retained, bank);
        assert_eq!(census.live, (24 + 8 + 6 + 3) * row + 3 * bank);
        assert_eq!(census.total(), store.committed_bytes());
        assert_eq!(census.total(), device.memory_usage().charged);
        let submitted = store
            .holding_census(
                &[Holder::State(&b), Holder::State(&c)],
                &[Holder::Checkpoint(&system), Holder::Checkpoint(&a_turn)],
                &[Holder::State(&a)],
            )
            .unwrap();
        assert_eq!(submitted.in_flight, 32 * row + bank);
        assert_eq!(submitted.live, (6 + 3) * row + 2 * bank);
        assert_eq!(submitted.retained, bank);
        assert_eq!(submitted.total(), store.committed_bytes());
        assert!(store
            .holding_census(
                &[Holder::State(&a), Holder::State(&b)],
                &[Holder::Checkpoint(&system), Holder::Checkpoint(&a_turn)],
                &[],
            )
            .is_err());
        // Alone, each live request owns only its private tail and its bank.
        assert_eq!(
            store.exclusive_bytes(&[Holder::State(&b)]).unwrap(),
            6 * row + bank
        );
        // The retained set (system prompt + A's turn) owns the prefix only
        // once every live request is gone, and A's tail and bank once A is.
        let retained = [Holder::Checkpoint(&system), Holder::Checkpoint(&a_turn)];
        assert_eq!(store.exclusive_bytes(&retained).unwrap(), bank);
        drop(a);
        let after_a = store
            .holding_census(
                &[Holder::State(&b), Holder::State(&c)],
                &[Holder::Checkpoint(&system), Holder::Checkpoint(&a_turn)],
                &[],
            )
            .unwrap();
        assert_eq!(after_a.retained, 8 * row + 2 * bank);
        assert_eq!(after_a.total(), store.committed_bytes());
        assert_eq!(
            store.exclusive_bytes(&retained).unwrap(),
            8 * row + 2 * bank
        );
        drop((b, c));
        assert_eq!(
            store.exclusive_bytes(&retained).unwrap(),
            32 * row + 2 * bank
        );
        // Repeated holders count once.
        let repeated = [
            Holder::Checkpoint(&system),
            Holder::Checkpoint(&system),
            Holder::Checkpoint(&a_turn),
        ];
        assert_eq!(
            store.exclusive_bytes(&repeated).unwrap(),
            32 * row + 2 * bank
        );
        drop((system, a_turn));
        assert_eq!(store.occupied_rows(), 0);
    }

    /// A history without room in place is relaid out, not split: the
    /// backing is full (no growth), yet the history continues in one run and
    /// an interior prefix commits without repair.
    #[test]
    fn full_backing_relays_out_a_history_without_room_instead_of_splitting_it() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            8,
            8,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active: 3,
                in_flight: 3,
                retained: 0,
            },
        )
        .unwrap();
        let states = (0..3)
            .map(|_| {
                let advance = OwnedStateAdvance::begin(store.create().unwrap(), 2)
                    .ok()
                    .unwrap();
                let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap()
                else {
                    panic!("full prefix must commit");
                };
                state
            })
            .collect::<Vec<_>>();
        let mut states = states.into_iter();
        let first = states.next().unwrap();
        let middle = states.next().unwrap();
        let last = states.next().unwrap();
        // Placement: first [0, 2), middle mid-hole at [4, 6), last [2, 4).
        let checkpoint = first.checkpoint();
        drop(last);
        assert_eq!(middle.history_ranges(), [(4, 2)]);
        // Two free rows follow `first`, it needs three: the relayout moves
        // `middle` past `first`'s room instead of splitting `first`.
        let advance = OwnedStateAdvance::begin(first, 3).ok().unwrap();
        assert_eq!(store.relayouts().count, 1);
        assert_eq!(store.committed().0, 8);
        assert_eq!(advance.bindings().destinations, [2, 3, 4]);
        assert_eq!(middle.history_ranges(), [(5, 2)]);
        let OwnedAdvanceResolution::Committed(first) = advance.commit(2).ok().unwrap() else {
            panic!("attention prefix must commit without repair");
        };
        assert_eq!(first.position(), 4);
        assert_eq!(first.history_ranges(), [(0, 4)]);
        assert_eq!(checkpoint.position(), 2);
        assert_eq!(checkpoint.fork().history_ranges(), [(0, 2)]);
        assert_eq!(store.occupied_rows(), 6);
        assert_eq!(middle.position(), 2);
    }

    #[test]
    fn pooled_banks_are_reused_and_checkpoint_claims_force_cow() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            8,
            8,
            vec![],
            vec![ComponentSpec {
                shape: vec![4],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        assert_eq!(store.available_banks(), 2);
        let state = store.create().unwrap();
        assert_eq!(store.available_banks(), 2);
        let zero_seed_bank = state.bank_index();

        let advance = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        let following_bank = advance.bindings().following_bank;
        assert_ne!(following_bank, zero_seed_bank);
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        assert_eq!(state.bank_index(), following_bank);
        assert_eq!(store.available_banks(), 1);
        let checkpoint = state.checkpoint();
        assert_eq!(checkpoint.bank_index(), following_bank);

        let advance = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        let second_bank = advance.bindings().following_bank;
        assert_ne!(second_bank, following_bank);
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        assert_eq!(store.available_banks(), 0);
        let Err((state, Error::BanksExhausted { capacity: 2 })) =
            OwnedStateAdvance::begin(state, 1)
        else {
            panic!("checkpoint must pin the old bank");
        };
        drop(checkpoint);
        assert_eq!(store.available_banks(), 1);
        let next = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        assert_eq!(next.bindings().following_bank, following_bank);
        let state = next.abort();
        assert_eq!(store.available_banks(), 1);
        let retry = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        assert_eq!(retry.bindings().following_bank, following_bank);
    }

    /// Every successor bank is disjoint from bank 0 and from every bank a live
    /// state, checkpoint, fork, or other in-flight advance can read, through
    /// forks, commits, aborts, and prefix commits.
    #[test]
    fn successor_banks_never_alias_a_readable_bank_or_the_zero_seed() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            16,
            16,
            vec![dense_component(1)],
            vec![
                ComponentSpec {
                    shape: vec![3, 8],
                    dtype: DType::BF16,
                },
                ComponentSpec {
                    shape: vec![2, 4, 4],
                    dtype: DType::F32,
                },
            ],
            BankCapacity {
                active: 3,
                in_flight: 3,
                retained: 2,
            },
        )
        .unwrap();
        assert_eq!(store.recurrent_arenas().len(), 2);
        assert_eq!(store.recurrent_arenas()[0].extents(), [9, 3, 8]);
        assert_eq!(store.recurrent_arenas()[1].extents(), [9, 2, 4, 4]);
        let readable = |states: &[&SequenceState], checkpoints: &[&StateCheckpoint]| {
            states
                .iter()
                .map(|state| state.bank_index())
                .chain(checkpoints.iter().map(|checkpoint| checkpoint.bank_index()))
                .collect::<Vec<_>>()
        };
        let parent = store.create().unwrap();
        let root = parent.checkpoint();
        let left = root.fork();
        let right = root.fork();
        // A launch provisions every advance's rows and successor bank before
        // the first of them begins.
        store
            .provision(
                &[2, 1, 3].map(|rows| RowDemand {
                    after: Some(0),
                    rows,
                }),
                3,
            )
            .unwrap();
        let left = OwnedStateAdvance::begin(left, 2).ok().unwrap();
        let right = OwnedStateAdvance::begin(right, 1).ok().unwrap();
        let parent = OwnedStateAdvance::begin_speculative(parent, 3, 1)
            .ok()
            .unwrap();
        let successors = [&left, &right, &parent].map(|advance| advance.bindings().following_bank);
        for (index, advance) in [&left, &right, &parent].into_iter().enumerate() {
            let bindings = advance.bindings();
            assert_eq!(bindings.previous_bank, ZERO_SEED_BANK);
            assert_ne!(bindings.following_bank, ZERO_SEED_BANK);
            assert!(!readable(&[], &[&root]).contains(&bindings.following_bank));
            assert!(successors
                .iter()
                .enumerate()
                .all(|(other, bank)| other == index || *bank != bindings.following_bank));
        }
        let OwnedAdvanceResolution::Committed(left) = left.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        let right = right.abort();
        let OwnedAdvanceResolution::Committed(parent) = parent.commit(2).ok().unwrap() else {
            panic!("an interior recurrent prefix commits as a tape version");
        };
        let branch = left.checkpoint();
        let fork = branch.fork();
        assert_eq!(fork.bank_index(), left.bank_index());
        assert_eq!(parent.bank_index(), successors[2]);
        assert_eq!(parent.tape_rows(), 1);
        assert!(
            !readable(&[&left, &right, &fork], &[&root, &branch]).contains(&parent.bank_index())
        );
        let advance = OwnedStateAdvance::begin(fork, 1).ok().unwrap();
        let following = advance.bindings().following_bank;
        assert_ne!(following, ZERO_SEED_BANK);
        assert!(!readable(&[&left, &right, &parent], &[&root, &branch]).contains(&following));
        drop(advance);
        drop((left, right, parent, root, branch));
        assert_eq!(store.available_banks(), store.committed().1 - 1);
    }

    #[test]
    fn fresh_sequences_share_a_pristine_seed_after_dirty_successor_reuse() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            8,
            8,
            vec![],
            vec![ComponentSpec {
                shape: vec![4],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let trace = store.allocation_trace().unwrap();
        assert_eq!(trace.recurrent_bank_bytes, 16);
        assert_eq!(trace.zero_seed_bytes, 16);
        assert_eq!(trace.recurrent_pool_bytes, 48);
        assert_eq!(store.available_banks(), 2);

        let first = store.create().unwrap();
        let seed_bank = first.bank_index();
        assert_eq!(seed_bank, ZERO_SEED_BANK);
        assert_eq!(bank_bytes(&store, seed_bank), vec![0; 16]);
        let advance = OwnedStateAdvance::begin(first, 1).ok().unwrap();
        let dirty = advance.bindings().following_bank;
        assert_ne!(dirty, seed_bank);
        write_bank(&store, dirty, &7.5f32.to_le_bytes().repeat(4));
        let OwnedAdvanceResolution::Committed(first) = advance.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        assert_eq!(first.bank_index(), dirty);
        assert_eq!(bank_bytes(&store, dirty), 7.5f32.to_le_bytes().repeat(4));
        drop(first);

        let second = store.create().unwrap();
        assert_eq!(second.bank_index(), seed_bank);
        assert_eq!(bank_bytes(&store, seed_bank), vec![0; 16]);
        assert_eq!(store.available_banks(), 2);
    }

    /// A sequence whose 17 visible rows are the even rows 0..34, with every
    /// row outside `free` and the sequence held by filler claims.
    fn fragmented(store: &Rc<StateStore>, free: &[(usize, usize)]) -> (SequenceState, Vec<Claims>) {
        let rows = store.history_capacity();
        store
            .provision(&[RowDemand { after: None, rows }], 0)
            .unwrap();
        assert_eq!(store.committed().0, rows);
        let mut state = store.create().unwrap();
        let even = (0..17).map(|row| (row * 2, 1)).collect::<Vec<_>>();
        state.claims.append(history(&store.arena, &even));
        state.position = 17;
        let held = |row: usize| {
            (row < 34 && row % 2 == 0)
                || free
                    .iter()
                    .any(|(start, count)| *start <= row && row < start + count)
        };
        let fillers = (0..store.history_capacity())
            .filter(|row| !held(*row))
            .map(|row| claim_rows(&store.arena, row, 1))
            .collect();
        assert_eq!(store.available_rows(), free.iter().map(|(_, n)| n).sum());
        (state, fillers)
    }

    #[test]
    fn compaction_is_bit_exact_and_publishes_only_after_success() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            64,
            64,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let (state, _fillers) = fragmented(&store, &[(40, 17)]);

        let planes = store.history_planes().unwrap();
        for plane in &planes {
            let mut bytes = vec![0u8; plane.buffer.byte_len() as usize];
            for row in 0..64 {
                let start = row * plane.row_bytes;
                bytes[start..start + plane.row_bytes].fill(row as u8);
            }
            let mut buffer = plane.buffer.clone();
            buffer.write_from_host(&bytes).unwrap();
        }

        assert!(state.compaction_needed());
        let old_ranges = state.history_ranges();
        let OwnedCompactionPreparation::Ready(failed) =
            OwnedCompaction::prepare(state, 64).ok().unwrap()
        else {
            panic!("contiguous destination must prepare compaction")
        };
        // A failed copy submission aborts its owned destination.
        let state = failed.abort();
        assert_eq!(state.history_ranges(), old_ranges);

        let OwnedCompactionPreparation::Ready(compaction) =
            OwnedCompaction::prepare(state, 64).ok().unwrap()
        else {
            panic!("released destination must be reusable")
        };
        assert_eq!(compaction.copies().len(), 2);
        for copy in compaction.copies() {
            assert_eq!(copy.from, (0..17).map(|row| row * 2).collect::<Vec<_>>());
            assert_eq!(copy.to, (40..57).collect::<Vec<_>>());
        }
        let bindings = compaction.bindings();
        for copy in bindings.copies {
            let plane = &bindings.history[copy.plane_index];
            let mut bytes = plane.buffer.read_to_host().unwrap();
            for (&from, &to) in copy.from.iter().zip(&copy.to) {
                let source = bytes[from * plane.row_bytes..(from + 1) * plane.row_bytes].to_vec();
                bytes[to * plane.row_bytes..(to + 1) * plane.row_bytes].copy_from_slice(&source);
            }
            let mut buffer = plane.buffer.clone();
            buffer.write_from_host(&bytes).unwrap();
        }
        let state = compaction.commit();
        assert_eq!(state.history_ranges(), [(40, 17)]);
        assert!(!state.compaction_needed());
        for plane in store.history_planes().unwrap() {
            let bytes = plane.buffer.read_to_host().unwrap();
            for (logical, row) in (40..57).enumerate() {
                assert_eq!(
                    &bytes[row * plane.row_bytes..(row + 1) * plane.row_bytes],
                    vec![(logical * 2) as u8; plane.row_bytes]
                );
            }
        }
    }

    #[test]
    fn compaction_defers_when_no_contiguous_run_is_free() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            64,
            64,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let (state, _fillers) = fragmented(&store, &[(40, 8), (50, 9)]);
        assert!(state.compaction_needed());
        let OwnedCompactionPreparation::Deferred {
            state,
            segments,
            visible_rows,
        } = OwnedCompaction::prepare(state, 64).ok().unwrap()
        else {
            panic!("fragmented capacity must defer compaction");
        };
        assert_eq!((segments, visible_rows), (17, 17));
        assert_eq!(state.history_ranges().len(), 17);
    }

    #[test]
    fn minimum_growth_claim_accounts_for_fragmented_history_relayout() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            64,
            64,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let (state, _fillers) = fragmented(&store, &[(40, 8), (50, 9)]);
        let demand = [state.demand(9)];
        let claim = store.growth_claim(&demand, 0).unwrap();
        assert!(claim.minimum_bytes > 0);
        let charged = device.memory_usage().charged;
        device.set_memory_limit(Some(charged + claim.minimum_bytes));
        store
            .provision_with_growth(&demand, 0, GrowthChoice::Minimum)
            .unwrap();
        assert_eq!(store.relayouts().count, 1);
        assert!(OwnedStateAdvance::begin(state, 9).is_ok());
    }

    /// Repacking restores adjacency with one bounded copy: a long shared
    /// prefix stays in place and only the recent decode runs move, joining
    /// into one run; a checkpoint on the prefix keeps seeing the same rows.
    #[test]
    fn repacking_joins_the_recent_runs_and_keeps_a_shared_prefix_in_place() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            512,
            1024,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 1,
            },
        )
        .unwrap();
        store
            .provision(
                &[RowDemand {
                    after: None,
                    rows: 1024,
                }],
                0,
            )
            .unwrap();
        // A 200-row prefix (retained by a checkpoint), then 15 one-row runs
        // separated by rows other histories hold.
        let mut state = store.create().unwrap();
        state.claims.append(claim_rows(&store.arena, 0, 200));
        state.position = 200;
        let prefix = state.checkpoint();
        let mut fillers = Vec::new();
        for run in 0..15 {
            let row = 300 + 2 * run;
            fillers.push(claim_rows(&store.arena, row + 1, 1));
            state.claims.append(claim_rows(&store.arena, row, 1));
            state.position += 1;
        }
        assert_eq!(state.history_ranges().len(), 16);
        assert!(state.compaction_needed());
        let OwnedCompactionPreparation::Ready(compaction) =
            OwnedCompaction::prepare(state, 64).ok().unwrap()
        else {
            panic!("the recent runs fit one bounded copy")
        };
        assert_eq!(compaction.rows(), 15);
        let copy = &compaction.copies()[0];
        assert_eq!(
            copy.from,
            (0..15).map(|run| 300 + 2 * run).collect::<Vec<_>>()
        );
        // The first free run that fits is the one right after the prefix.
        assert_eq!(copy.to, (200..215).collect::<Vec<_>>());
        let state = compaction.commit();
        assert_eq!(state.history_ranges(), [(0, 215)]);
        assert_eq!(prefix.history_ranges(), [(0, 200)]);
        // The moved runs' rows are free again; the prefix is held once.
        assert_eq!(store.occupied_rows(), 200 + 15 + fillers.len());
    }

    /// Lock-step serving of `active` requests for `steps` steps: chunked
    /// prefills interleaved with speculative decode (1 + 0..=3 draft rows,
    /// a random accepted prefix), every advance of a step in flight at once,
    /// requests finishing and new ones admitted into their slots. The arena
    /// holds `contexts` full contexts plus one batch. Returns the largest
    /// segment count any accepted history reached and the number of decode
    /// steps the longest request ran.
    fn serve_interleaved(active: usize, contexts: usize, steps: usize) -> Interleaved {
        const CONTEXT: usize = 512;
        const BATCH_ROWS: usize = 64;
        let device = cpu_device().expect("the CPU backend is available");
        let store = StateStore::new(
            device,
            CONTEXT,
            contexts * CONTEXT + BATCH_ROWS,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active,
                in_flight: active,
                retained: 0,
            },
        )
        .unwrap();
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut random = |bound: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % bound as u64) as usize
        };
        struct Request {
            serial: u32,
            state: SequenceState,
            prompt: usize,
            length: usize,
            decode_steps: usize,
        }
        // Every accepted row holds a tag of its request and position; a
        // finished request reads its whole history back through its ranges,
        // so every relayout copy is checked.
        let tag = |serial: u32, position: usize| (serial * 1024 + position as u32) as f32;
        let verify = |request: &Request| {
            let plane = store.history_planes().unwrap()[0].buffer.clone();
            let rows = request
                .state
                .history_ranges()
                .into_iter()
                .flat_map(|(start, count)| start..start + count)
                .map(|row| {
                    let bytes = plane
                        .slice_leading(row as u64, row as u64 + 1)
                        .unwrap()
                        .read_to_host()
                        .unwrap();
                    f32::from_le_bytes(bytes[..4].try_into().unwrap())
                })
                .collect::<Vec<_>>();
            let expected = (0..request.state.position())
                .map(|position| tag(request.serial, position))
                .collect::<Vec<_>>();
            assert_eq!(rows, expected, "request {} history", request.serial);
        };
        let mut slots: Vec<Option<Request>> = (0..active).map(|_| None).collect();
        let (mut max_segments, mut max_decode_steps, mut peak_committed) = (0, 0, 0);
        let (mut serials, mut written) = (0u32, 0usize);
        for step in 0..steps {
            for slot in &mut slots {
                if slot.is_none() {
                    let prompt = 16 + random(160);
                    serials += 1;
                    *slot = Some(Request {
                        serial: serials,
                        state: store.create().unwrap(),
                        prompt,
                        length: prompt + 200 + random(CONTEXT - prompt - 200),
                        decode_steps: 0,
                    });
                }
            }
            // Every advance of the step reserves before any resolves; the
            // scheduling order rotates so no request always reserves first.
            let order = (0..active)
                .map(|index| (index + step) % active)
                .collect::<Vec<_>>();
            let planned = order
                .iter()
                .map(|&index| {
                    let request = slots[index].take().unwrap();
                    let position = request.state.position();
                    let rows = if position < request.prompt {
                        (request.prompt - position).min(BATCH_ROWS / 2)
                    } else {
                        (1 + random(4)).min(request.length - position)
                    };
                    (index, request, rows)
                })
                .collect::<Vec<_>>();
            // The launch provisions the backing for all of its advances.
            let demands = planned
                .iter()
                .map(|(_, request, rows)| RowDemand {
                    after: request.state.history_end(),
                    rows: *rows,
                })
                .collect::<Vec<_>>();
            store.provision(&demands, 0).unwrap();
            peak_committed = peak_committed.max(store.committed().0);
            let mut advances = Vec::with_capacity(active);
            for (index, request, rows) in planned {
                let advance = match OwnedStateAdvance::begin(request.state, rows) {
                    Ok(advance) => advance,
                    Err((_, error)) => panic!("step {step}: {error}"),
                };
                let plane = &advance.bindings().history[0].buffer;
                for (offset, &row) in advance.bindings().destinations.iter().enumerate() {
                    let value = tag(request.serial, advance.position() + offset);
                    plane
                        .slice_leading(row as u64, row as u64 + 1)
                        .unwrap()
                        .write_from_host(&value.to_le_bytes())
                        .unwrap();
                }
                advances.push((
                    index,
                    rows,
                    request.serial,
                    request.prompt,
                    request.length,
                    request.decode_steps,
                    advance,
                ));
            }
            advances.reverse();
            for (index, rows, serial, prompt, length, decode_steps, advance) in advances {
                let decode = advance.position() >= prompt;
                let accepted = if decode { 1 + random(rows) } else { rows };
                let OwnedAdvanceResolution::Committed(state) =
                    advance.commit(accepted).ok().unwrap()
                else {
                    panic!("attention-only prefixes commit without repair");
                };
                written += accepted;
                let decode_steps = decode_steps + usize::from(decode);
                max_segments = max_segments.max(state.history_ranges().len());
                max_decode_steps = max_decode_steps.max(decode_steps);
                let request = Request {
                    serial,
                    state,
                    prompt,
                    length,
                    decode_steps,
                };
                if request.state.position() < length {
                    slots[index] = Some(request);
                } else {
                    verify(&request);
                }
            }
            // Worst case for thrash: the engine idles between every batch.
            store.shrink(ShrinkPolicy::Idle).unwrap();
            let mut ranges = slots
                .iter()
                .flatten()
                .flat_map(|request| request.state.history_ranges())
                .collect::<Vec<_>>();
            ranges.sort_unstable();
            assert!(
                ranges
                    .windows(2)
                    .all(|pair| pair[0].0 + pair[0].1 <= pair[1].0),
                "live histories overlap"
            );
            let committed = store.committed().0;
            peak_committed = peak_committed.max(committed);
            assert!(
                store.occupied_rows() <= committed,
                "claims lie in committed rows"
            );
        }
        for request in slots.iter().flatten() {
            verify(request);
        }
        drop(slots);
        store.shrink(ShrinkPolicy::Idle).unwrap();
        Interleaved {
            segments: max_segments,
            decode_steps: max_decode_steps,
            peak_committed,
            released_to: store.committed().0,
            relayouts: store.relayouts(),
            written,
        }
    }

    /// The backing commits rows and banks with demand, keeps every row's
    /// contents across growth, and returns bytes to the device ledger when
    /// the tail is unreferenced; nothing is recommitted under a transaction.
    #[test]
    fn backing_grows_and_shrinks_and_returns_device_memory() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            4096,
            16384,
            vec![dense_component(4)],
            vec![ComponentSpec {
                shape: vec![64],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 16,
                in_flight: 16,
                retained: 16,
            },
        )
        .unwrap();
        let charged = || device.memory_usage().charged;
        let base = charged();
        assert_eq!(store.committed(), (0, 3));
        // A 1,000-row prefill commits about what it uses, not the reservation.
        let advance = OwnedStateAdvance::begin(store.create().unwrap(), 1000)
            .ok()
            .unwrap();
        let rows = store.committed().0;
        assert!((1000..=1280).contains(&rows), "committed {rows} rows");
        // Growth is refused while a transaction holds the tensors.
        store
            .provision(
                &[RowDemand {
                    after: None,
                    rows: 5000,
                }],
                8,
            )
            .unwrap();
        assert_eq!(store.committed().0, rows);
        let written = (0..1000u32)
            .flat_map(|row| (row as f32).to_le_bytes().repeat(4))
            .collect::<Vec<_>>();
        advance.bindings().history[0]
            .buffer
            .slice_leading(0, 1000)
            .unwrap()
            .write_from_host(&written)
            .unwrap();
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        let grown = charged();
        assert!(grown > base);
        // Without transactions, growth commits more rows and banks and keeps
        // the accepted rows' contents.
        let demand = [state.demand(5000)];
        let claim = store.growth_claim(&demand, 8).unwrap();
        assert!(claim.preferred_bytes >= claim.minimum_bytes);
        assert!(claim.minimum_bytes > 0);
        let previous_limit = device.memory_usage().limit;
        device.set_memory_limit(Some(charged() + claim.minimum_bytes - 1));
        assert!(matches!(
            store.provision_with_growth(&demand, 8, GrowthChoice::Minimum),
            Err(Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity { .. }
            )))
        ));
        assert_eq!(store.committed().0, rows);
        assert_eq!(state.history_ranges(), [(0, 1000)]);
        device.set_memory_limit(previous_limit);
        store.provision(&demand, 8).unwrap();
        let (rows, banks) = store.committed();
        assert!(rows >= 6000 && rows < 16384, "committed {rows} rows");
        assert!(banks >= 9, "committed {banks} banks");
        assert!(charged() > grown);
        let plane = store.history_planes().unwrap()[0].buffer.clone();
        assert_eq!(
            plane
                .slice_leading(0, 1000)
                .unwrap()
                .read_to_host()
                .unwrap(),
            written
        );
        assert_eq!(state.history_ranges(), [(0, 1000)]);
        // A caller can still pin the old history tensor while the store
        // shrinks. Credit only the ledger decrease, which excludes its bytes.
        let before = charged();
        let released = store.shrink(ShrinkPolicy::Idle).unwrap();
        assert_eq!(released, before.saturating_sub(charged()));
        assert!(store.committed().0 < rows);
        assert!(store.committed().0 >= 1000);
        assert_eq!(
            store.external_pinned_bytes().unwrap(),
            plane.storage_bytes()
        );
        let pinned_charge = charged();
        drop(plane);
        assert!(charged() < pinned_charge);
        assert_eq!(store.external_pinned_bytes().unwrap(), 0);
        drop(state);
        // Idle hysteresis keeps a small backing; pressure releases it all.
        store.shrink(ShrinkPolicy::Pressure).unwrap();
        assert_eq!(store.committed(), (HISTORY_GRANULE, 3));
        assert!(store.release_idle().unwrap() > 0);
        assert_eq!(charged(), base);
    }

    /// A failed second CUDA history-plane growth must leave the published
    /// StateStore backing and both original planes usable at their old size.
    #[test]
    fn failed_second_cuda_plane_growth_keeps_published_history() {
        let Some(device) = DeviceCatalog::discover()
            .ok()
            .and_then(|catalog| catalog.open_backend(BackendName::Cuda).ok())
            .map(Rc::new)
        else {
            return;
        };
        eprintln!("StateStore rollback backend: {}", device.backend().as_str());
        let store = StateStore::new(
            device.clone(),
            512,
            512,
            vec![dense_component(4096)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        store.recommit_history(128).unwrap();
        let encoded = (0..4096u32)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        {
            let backing = store.backing.borrow();
            for plane in &backing.history {
                plane
                    .slice_leading(0, 1)
                    .unwrap()
                    .write_from_host(&encoded)
                    .unwrap();
            }
        }
        let baseline = device.memory_usage().charged;
        let tentative = store.backing.borrow().history[0].recommitted(256).unwrap();
        let first_delta = device.memory_usage().charged - baseline;
        assert!(first_delta > 0, "the first CUDA plane must grow physically");
        drop(tentative);
        assert_eq!(device.memory_usage().charged, baseline);
        device.set_memory_limit(Some(baseline + first_delta));
        assert!(matches!(
            store.recommit_history(256),
            Err(Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity { .. }
            )))
        ));
        device.set_memory_limit(None);
        assert_eq!(store.committed().0, 128);
        assert_eq!(device.memory_usage().charged, baseline);
        let backing = store.backing.borrow();
        for plane in &backing.history {
            assert_eq!(plane.committed_rows(), 128);
            assert_eq!(
                plane.slice_leading(0, 1).unwrap().read_to_host().unwrap(),
                encoded
            );
        }
    }

    /// Regression (chost 15:36): shrinking after every step released the
    /// successor banks the next step regrew, reallocating and copying the
    /// bank arenas every other decode step. With hysteresis a request that
    /// decodes 300 steps beside a retained prompt checkpoint recommits the
    /// backing only while it grows, even when the store idles between steps.
    #[test]
    fn decode_never_alternates_growing_and_shrinking_the_backing() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            4096,
            4 * 4096,
            vec![dense_component(4)],
            vec![ComponentSpec {
                shape: vec![256],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 4,
                in_flight: 4,
                retained: 4,
            },
        )
        .unwrap();
        let step = |state: SequenceState, rows: usize| {
            store.provision(&[state.demand(rows)], 1).unwrap();
            let advance = OwnedStateAdvance::begin(state, rows).ok().unwrap();
            let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap()
            else {
                panic!("full prefix must commit");
            };
            store.shrink(ShrinkPolicy::Idle).unwrap();
            state
        };
        let mut state = step(store.create().unwrap(), 500);
        let _prompt = state.checkpoint();
        let settled = store.recommits() + store.relayouts().count;
        let mut committed = store.committed();
        for _ in 0..300 {
            state = step(state, 1);
            let now = store.committed();
            assert!(
                now.0 >= committed.0 && now.1 >= committed.1,
                "decode shrank the backing from {committed:?} to {now:?}"
            );
            committed = now;
        }
        // Only geometric growth: the rows grow 500 -> 800, the banks once.
        let changes = store.recommits() + store.relayouts().count - settled;
        assert!(
            changes <= 3,
            "300 decode steps changed the backing {changes} times"
        );
    }

    /// Shrinking relays out live rows stranded high in the backing: the
    /// shared prefix stays one set of rows (a checkpoint and two branches
    /// see the same rows, charged once), the longer branch stays one run,
    /// every row keeps its contents, and the committed bytes fall.
    #[test]
    fn shrink_moves_live_tail_rows_down_and_keeps_shared_prefixes_shared() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            4096,
            8192,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active: 4,
                in_flight: 4,
                retained: 4,
            },
        )
        .unwrap();
        let tag = |row: usize| (row as f32).to_le_bytes();
        let commit = |state: SequenceState, rows: usize| {
            let advance = OwnedStateAdvance::begin(state, rows).ok().unwrap();
            let plane = &advance.bindings().history[0].buffer;
            for (offset, &row) in advance.bindings().destinations.iter().enumerate() {
                plane
                    .slice_leading(row as u64, row as u64 + 1)
                    .unwrap()
                    .write_from_host(&tag(advance.position() + offset))
                    .unwrap();
            }
            let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap()
            else {
                panic!("attention-only advances commit");
            };
            state
        };
        let read = |ranges: Vec<(usize, usize)>| {
            let plane = store.history_planes().unwrap()[0].buffer.clone();
            ranges
                .into_iter()
                .flat_map(|(start, count)| start..start + count)
                .map(|row| {
                    plane
                        .slice_leading(row as u64, row as u64 + 1)
                        .unwrap()
                        .read_to_host()
                        .unwrap()[..4]
                        .to_vec()
                })
                .collect::<Vec<_>>()
        };
        let expected = |rows: usize| (0..rows).map(|row| tag(row).to_vec()).collect::<Vec<_>>();
        // A large request fills the low rows; a prompt and two branches sit
        // above it.
        let large = commit(store.create().unwrap(), 3000);
        let prompt = commit(store.create().unwrap(), 100);
        let system = prompt.checkpoint();
        let long = commit(prompt, 50);
        let short = commit(system.fork(), 20);
        assert!(long.history_ranges()[0].0 >= 3000);
        drop(large);
        let before = (store.committed().0, device.memory_usage().charged);
        let occupied = store.occupied_rows();
        assert_eq!(occupied, 100 + 50 + 20);
        let shape = store.pressure_shrink_shape().unwrap();
        assert_eq!(shape.referenced_rows, occupied);
        assert!(shape.highest_referenced_row > shape.target_rows);
        assert!(shape.history_peak_bytes > 0);
        assert_eq!(shape.active_transactions, 0);
        let relayouts = store.relayouts().count;
        // The live rows sit above the low hole. Releasing this surplus
        // requires a fresh relayout backing; a tight peak charge cannot
        // perform that copy, even though the final backing would be smaller.
        device.set_memory_limit(Some(before.1));
        assert_eq!(store.shrink(ShrinkPolicy::Pressure).unwrap(), 0);
        assert_eq!(store.committed().0, before.0);
        assert_eq!(store.relayouts().count, relayouts);
        device.set_memory_limit(None);
        let released = store.shrink(ShrinkPolicy::Idle).unwrap();
        assert_eq!(store.relayouts().count, relayouts + 1);
        assert!(released > 0);
        assert!(store.committed().0 < before.0);
        assert!(device.memory_usage().charged < before.1);
        assert_eq!(store.occupied_rows(), occupied);
        assert_eq!(long.history_ranges(), [(0, 150)]);
        assert_eq!(system.history_ranges(), [(0, 100)]);
        assert_eq!(short.history_ranges()[0], (0, 100));
        assert_eq!(read(long.history_ranges()), expected(150));
        assert_eq!(read(short.history_ranges()), expected(120));
        assert_eq!(read(system.history_ranges()), expected(100));
    }

    #[test]
    fn pressure_compacts_claimed_banks_before_releasing_the_tail() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            64,
            256,
            vec![],
            vec![
                ComponentSpec {
                    shape: vec![4],
                    dtype: DType::F32,
                },
                ComponentSpec {
                    shape: vec![8],
                    dtype: DType::F32,
                },
            ],
            BankCapacity {
                active: 8,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        store.recommit_banks(10).unwrap();
        let mut claims = (0..9)
            .map(|_| Some(store.banks.acquire().unwrap()))
            .collect::<Vec<_>>();
        let kept = [2usize, 5, 9];
        for &index in &kept {
            for (plane_index, plane) in store.recurrent_arenas().iter().enumerate() {
                let bytes = vec![index as u8 + plane_index as u8; (4 + 4 * plane_index) * 4];
                plane
                    .slice_leading(index as u64, index as u64 + 1)
                    .unwrap()
                    .write_from_host(&bytes)
                    .unwrap();
            }
        }
        for index in 1..=9 {
            if !kept.contains(&index) {
                drop(claims[index - 1].take());
            }
        }
        let charged = device.memory_usage().charged;
        device.set_memory_limit(Some(charged));
        assert_eq!(store.shrink(ShrinkPolicy::Pressure).unwrap(), 0);
        assert_eq!(store.committed().1, 10);
        let remapped = kept.map(|index| claims[index - 1].as_ref().unwrap().index());
        assert_eq!(
            remapped, kept,
            "failed peak allocation leaves placement intact"
        );
        // The first destination plane fits, but the second does not. Its
        // failure must discard the first copy without publishing placement.
        device.set_memory_limit(Some(charged + 6 * 16));
        assert_eq!(store.shrink(ShrinkPolicy::Pressure).unwrap(), 0);
        assert_eq!(store.committed().1, 10);
        assert_eq!(device.memory_usage().charged, charged);
        assert_eq!(
            kept.map(|index| claims[index - 1].as_ref().unwrap().index()),
            kept
        );
        for old in kept {
            for (plane_index, plane) in store.recurrent_arenas().iter().enumerate() {
                let bytes = plane
                    .slice_leading(old as u64, old as u64 + 1)
                    .unwrap()
                    .read_to_host()
                    .unwrap();
                assert_eq!(bytes, vec![old as u8 + plane_index as u8; bytes.len()]);
            }
        }
        device.set_memory_limit(None);
        assert!(store.shrink(ShrinkPolicy::Pressure).unwrap() > 0);
        assert_eq!(store.committed().1, 6);
        let remapped = kept.map(|index| claims[index - 1].as_ref().unwrap().index());
        assert_eq!(remapped, [1, 2, 3]);
        for (old, new) in kept.into_iter().zip(remapped) {
            for (plane_index, plane) in store.recurrent_arenas().iter().enumerate() {
                let bytes = plane
                    .slice_leading(new as u64, new as u64 + 1)
                    .unwrap()
                    .read_to_host()
                    .unwrap();
                assert_eq!(bytes, vec![old as u8 + plane_index as u8; bytes.len()]);
            }
        }
        assert!(device.memory_usage().charged < charged);
    }

    struct Interleaved {
        segments: usize,
        decode_steps: usize,
        peak_committed: usize,
        released_to: usize,
        relayouts: Relayouts,
        written: usize,
    }

    #[test]
    fn interleaved_decode_keeps_every_request_in_one_run_with_elastic_backing() {
        // Reservations of 8 to 17 contexts, down to exactly one context per
        // request; the backing commits what the interleaved histories use
        // plus growth headroom. A history that outgrows its room is relaid
        // out, never split, so every request stays one run.
        for (active, contexts, steps) in [
            (8, 16, 800),
            (4, 5, 3000),
            (8, 9, 3000),
            (8, 8, 3000),
            (16, 17, 3000),
        ] {
            let run = serve_interleaved(active, contexts, steps);
            let reserved = contexts * 512 + 64;
            eprintln!(
                "interleaved active={active} contexts={contexts}: segments={} relayouts={} \
                 copied={} written={} peak_committed={} of {reserved}",
                run.segments,
                run.relayouts.count,
                run.relayouts.rows,
                run.written,
                run.peak_committed
            );
            assert!(
                run.decode_steps > 100,
                "requests ran {} decode steps",
                run.decode_steps
            );
            assert_eq!(
                run.segments, 1,
                "{active} requests in {contexts} contexts reached {} segments",
                run.segments
            );
            // At most active x context rows are ever referenced; the backing
            // stays proportional to use, not to the reservation.
            assert!(
                run.peak_committed <= reserved.min(active * 512 * 2),
                "{active} requests committed {} of {reserved} rows",
                run.peak_committed
            );
            // Unreferenced backing returns once the requests end.
            assert_eq!(run.released_to, HISTORY_GRANULE);
        }
    }
}
