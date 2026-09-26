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
pub use layout::{
    banks_per_slab, history_rows_per_slab, max_visible_spans, ModelStateLayout, SLAB_BYTE_TARGET,
    SLAB_ROW_TILE,
};

use placement::{Generation, LogicalId, Placement, PlacementError};
use seismic::{DType, Device, Element, SlabRegion, SlabTensor, Tensor, TensorStorageObserver};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

/// Failures owned by the state store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Request(String),
    Tensor(seismic::TensorError),
    Layout(LayoutError),
    Capacity { required: u64, available_bytes: u64 },
    BanksExhausted { capacity: usize },
    Placement(PlacementError),
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
            Self::Placement(error) => write!(f, "recurrent bank placement: {error:?}"),
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

impl From<PlacementError> for Error {
    fn from(error: PlacementError) -> Self {
        Self::Placement(error)
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

/// One logical attention-history component view over the store's slab table.
/// `base_row` is stable and currently always zero because all components
/// share the arena-global row domain.
#[derive(Clone)]
pub struct PlaneBuffer {
    pub plane_index: usize,
    pub component_index: usize,
    pub layer: LayerRef,
    pub vector: VectorKind,
    pub name: PlaneName,
    pub row_bytes: usize,
    pub slab_rows: u32,
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
/// history drops it. Every history is registered, so compaction can move
/// referenced rows and rewrite every affected history.
#[derive(Clone)]
struct Arena {
    slab_rows: usize,
    backed: BTreeSet<usize>,
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
    fn new(rows: usize, slab_rows: usize) -> Self {
        assert!(slab_rows > 0);
        let mut free = Vec::new();
        append_ranges(&mut free, [(0, rows)], slab_rows);
        Self {
            slab_rows,
            backed: (0..rows.div_ceil(slab_rows)).collect(),
            free,
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
        append_ranges(&mut joined, ranges, self.slab_rows);
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

    /// Empty one backed slab into free rows in other backed slabs. The
    /// placement is planned on a clone by the caller and published only
    /// after all row copies succeed.
    #[cfg(test)]
    fn compact_slab(&mut self, rows: usize, slab: usize) -> Option<Vec<(usize, usize, usize)>> {
        if !self.backed.contains(&slab) {
            return None;
        }
        let start = slab * self.slab_rows;
        let end = (start + self.slab_rows).min(rows);
        let mut spans = self
            .runs
            .range(start..end)
            .map(|(&row, run)| (row, run.count))
            .collect::<Vec<_>>();
        let mut holes = self
            .free
            .iter()
            .copied()
            .filter(|&(row, _)| row < start || row >= end)
            .collect::<Vec<_>>();
        let occupied = spans.iter().map(|(_, count)| count).sum::<usize>();
        if occupied == 0 || holes.iter().map(|(_, count)| count).sum::<usize>() < occupied {
            return None;
        }
        spans.sort_unstable_by_key(|&(row, count)| (std::cmp::Reverse(count), row));
        let mut placed = BTreeMap::new();
        for (span, count) in spans {
            let fitting = holes
                .iter()
                .enumerate()
                .filter(|(_, (_, size))| *size >= count)
                .min_by_key(|(_, &(row, size))| (size, row))
                .map(|(index, _)| index);
            let (mut row, mut remaining) = (span, count);
            while remaining > 0 {
                let hole = fitting.unwrap_or_else(|| {
                    (0..holes.len())
                        .max_by_key(|&index| (holes[index].1, std::cmp::Reverse(holes[index].0)))
                        .expect("other slabs have enough free rows")
                });
                let (to, size) = holes[hole];
                let taken = size.min(remaining);
                placed.insert(row, (taken, to));
                holes[hole] = (to + taken, size - taken);
                if holes[hole].1 == 0 {
                    holes.remove(hole);
                }
                row += taken;
                remaining -= taken;
            }
        }
        Some(self.remap(rows, placed))
    }

    #[cfg(test)]
    fn least_occupied_slab_compaction(
        &self,
        rows: usize,
    ) -> Option<(usize, Self, Vec<(usize, usize, usize)>)> {
        let mut candidates = self
            .backed
            .iter()
            .filter_map(|&index| {
                let start = index * self.slab_rows;
                let end = (start + self.slab_rows).min(rows);
                let occupied = self
                    .runs
                    .range(start..end)
                    .map(|(_, run)| run.count)
                    .sum::<usize>();
                (occupied > 0).then_some((occupied, std::cmp::Reverse(index), index))
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        candidates.into_iter().find_map(|(_, _, index)| {
            let mut planned = self.clone();
            planned
                .compact_slab(rows, index)
                .map(|moves| (index, planned, moves))
        })
    }

    /// Plan every row move for one shrink against the published placement.
    /// Destinations are free in that placement, even when several slabs are
    /// emptied together, so all copies may finish before anything publishes.
    fn compact_into_slabs(
        &self,
        rows: usize,
        keep: &BTreeSet<usize>,
    ) -> (Self, Vec<(usize, usize, usize)>) {
        let mut planned = self.clone();
        let mut spans = self
            .runs
            .iter()
            .filter(|(row, _)| !keep.contains(&(*row / self.slab_rows)))
            .map(|(&row, run)| (row, run.count))
            .collect::<Vec<_>>();
        let mut holes = self
            .free
            .iter()
            .copied()
            .filter(|(row, _)| keep.contains(&(row / self.slab_rows)))
            .collect::<Vec<_>>();
        debug_assert!(
            holes.iter().map(|(_, count)| count).sum::<usize>()
                >= spans.iter().map(|(_, count)| count).sum::<usize>()
        );
        spans.sort_unstable_by_key(|&(row, count)| (std::cmp::Reverse(count), row));
        let mut placed = BTreeMap::new();
        for (span, count) in spans {
            let fitting = holes
                .iter()
                .enumerate()
                .filter(|(_, (_, size))| *size >= count)
                .min_by_key(|(_, &(row, size))| (size, row))
                .map(|(index, _)| index);
            let (mut row, mut remaining) = (span, count);
            while remaining > 0 {
                let hole = fitting.unwrap_or_else(|| {
                    (0..holes.len())
                        .max_by_key(|&index| (holes[index].1, std::cmp::Reverse(holes[index].0)))
                        .expect("kept slabs have room for every moved row")
                });
                let (to, size) = holes[hole];
                let taken = size.min(remaining);
                placed.insert(row, (taken, to));
                holes[hole] = (to + taken, size - taken);
                if holes[hole].1 == 0 {
                    holes.remove(hole);
                }
                row += taken;
                remaining -= taken;
            }
        }
        let moves = if placed.is_empty() {
            Vec::new()
        } else {
            planned.remap(rows, placed)
        };
        for slab in self.backed.difference(keep) {
            planned.unback_empty_slab(*slab);
        }
        (planned, moves)
    }

    /// Rewrite every history through `placed` (old start -> (count, new
    /// start)); rows outside it keep their address. Rebuilds the runs from
    /// the rewritten histories and the free holes within `rows` committed
    /// rows, and returns the placed moves joined where adjacent.
    fn remap(
        &mut self,
        rows: usize,
        placed: BTreeMap<usize, (usize, usize)>,
    ) -> Vec<(usize, usize, usize)> {
        let slab_rows = self.slab_rows;
        let translate = |start: usize, count: usize| {
            let mut out = Vec::new();
            let (mut row, end) = (start, start + count);
            while row < end {
                match placed.range(..=row).next_back() {
                    Some((&from, &(moved, to))) if from + moved > row => {
                        let taken = (from + moved).min(end) - row;
                        append_ranges(&mut out, [(to + row - from, taken)], slab_rows);
                        row += taken;
                    }
                    _ => {
                        let stop = placed.range(row..end).next().map_or(end, |(&from, _)| from);
                        append_ranges(&mut out, [(row, stop - row)], slab_rows);
                        row = stop;
                    }
                }
            }
            out
        };
        for entry in self.entries.values_mut() {
            let mut ranges = Vec::with_capacity(entry.ranges.len());
            for &(start, count) in &entry.ranges {
                append_ranges(&mut ranges, translate(start, count), slab_rows);
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
                    (start + run.count == from
                        && start / slab_rows == from / slab_rows
                        && run.references == references)
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
        self.free.clear();
        for &slab in &self.backed {
            let start = slab * slab_rows;
            let end = ((slab + 1) * slab_rows).min(rows);
            let mut cursor = start;
            for (&run_start, run) in self.runs.range(start..end) {
                if run_start > cursor {
                    append_ranges(&mut self.free, [(cursor, run_start - cursor)], slab_rows);
                }
                cursor = run_start + run.count;
            }
            if end > cursor {
                append_ranges(&mut self.free, [(cursor, end - cursor)], slab_rows);
            }
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

    fn back_slab(&mut self, slab: usize, logical_rows: usize) {
        assert!(self.backed.insert(slab));
        let start = slab * self.slab_rows;
        let end = (start + self.slab_rows).min(logical_rows);
        let mut free = std::mem::take(&mut self.free);
        free.push((start, end - start));
        free.sort_unstable_by_key(|&(start, _)| start);
        append_ranges(&mut self.free, free, self.slab_rows);
    }

    fn unback_empty_slab(&mut self, slab: usize) {
        let start = slab * self.slab_rows;
        let end = start + self.slab_rows;
        assert!(self.runs.range(start..end).next().is_none());
        assert!(self.backed.remove(&slab));
        self.free.retain(|&(row, _)| row < start || row >= end);
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
                    Some((s, n))
                        if *s + *n == start && *s / self.slab_rows == start / self.slab_rows =>
                    {
                        *n += count
                    }
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
                if previous + before.count == row
                    && previous / self.slab_rows == row / self.slab_rows
                    && before.references == run.references
                {
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
fn append_ranges(
    into: &mut Vec<(usize, usize)>,
    ranges: impl IntoIterator<Item = (usize, usize)>,
    slab_rows: usize,
) {
    for (start, count) in ranges {
        let mut at = start;
        let mut remaining = count;
        while remaining > 0 {
            let taken = remaining.min(slab_rows - at % slab_rows);
            match into.last_mut() {
                Some((last, rows))
                    if *last + *rows == at && *last / slab_rows == at / slab_rows =>
                {
                    *rows += taken
                }
                _ => into.push((at, taken)),
            }
            at += taken;
            remaining -= taken;
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

    fn slab_rows(&self) -> usize {
        self.arena.borrow().slab_rows
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
        let slab_rows = arena.slab_rows;
        let entry = arena.entries.get_mut(&self.id).expect("registered history");
        append_ranges(&mut entry.ranges, ranges, slab_rows);
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
/// The zero seed's permanent logical claim. Compaction never moves it, since
/// it occupies the lowest slot.
const ZERO_SEED_CLAIM: LogicalId = LogicalId(0);

struct BankPoolInner {
    capacity: usize,
    placement: RefCell<Placement>,
    unavailable: RefCell<BTreeSet<usize>>,
    next_logical: Cell<u64>,
}

/// A stable logical claim on one row of every recurrent component. The
/// published placement alone decides which physical row holds it.
struct BankClaim {
    pool: Rc<BankPoolInner>,
    id: LogicalId,
}

impl Drop for BankClaim {
    fn drop(&mut self) {
        if self.id != ZERO_SEED_CLAIM {
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
    fn id(&self) -> LogicalId {
        self.0.id
    }

    fn index(&self) -> usize {
        self.0
            .pool
            .placement
            .borrow()
            .resolve(self.0.id)
            .expect("live bank has a physical placement")
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
            committed,
            BTreeMap::from([(ZERO_SEED_CLAIM, ZERO_SEED_BANK)]),
        )?;
        let inner = Rc::new(BankPoolInner {
            capacity: capacity.total()?,
            placement: RefCell::new(placement),
            unavailable: RefCell::new(BTreeSet::new()),
            next_logical: Cell::new(1),
        });
        let seed = BankHandle(Rc::new(BankClaim {
            pool: inner.clone(),
            id: ZERO_SEED_CLAIM,
        }));
        Ok((Self { inner }, seed))
    }

    fn acquire(&self) -> Result<BankHandle, Error> {
        let published = self.inner.placement.borrow().clone();
        let occupied = published
            .resources()
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        let unavailable = self.inner.unavailable.borrow();
        let index = (1..self.committed())
            .find(|index| !occupied.contains(index) && !unavailable.contains(index))
            .ok_or(Error::BanksExhausted {
                capacity: self.inner.capacity,
            })?;
        let id = LogicalId(self.inner.next_logical.get());
        let mut resources = published.resources().clone();
        resources.insert(id, index);
        *self.inner.placement.borrow_mut() = published.with_resources(resources)?;
        self.inner
            .next_logical
            .set(id.0.checked_add(1).expect("bank logical id exhausted"));
        Ok(BankHandle(Rc::new(BankClaim {
            pool: self.inner.clone(),
            id,
        })))
    }

    fn committed(&self) -> usize {
        self.inner.placement.borrow().capacity()
    }

    fn available(&self) -> usize {
        self.committed()
            - self.inner.placement.borrow().resources().len()
            - self.inner.unavailable.borrow().len()
    }

    fn resize(&self, from: usize, to: usize) {
        assert_eq!(self.committed(), from);
        let published = self.inner.placement.borrow().clone();
        *self.inner.placement.borrow_mut() = published
            .with_capacity(to)
            .expect("bank resize preserves every live placement");
        self.inner
            .unavailable
            .borrow_mut()
            .retain(|&index| index < to);
    }

    fn mark_slab_unavailable(&self, start: usize, end: usize) {
        let placement = self.inner.placement.borrow().clone();
        assert!(placement
            .resources()
            .values()
            .all(|&slot| slot < start || slot >= end));
        *self.inner.placement.borrow_mut() = placement
            .with_resources(placement.resources().clone())
            .expect("bank slab release advances placement generation");
        self.inner.unavailable.borrow_mut().extend(start..end);
    }

    fn mark_slab_available(&self, start: usize, end: usize) {
        let mut unavailable = self.inner.unavailable.borrow_mut();
        for slot in start..end {
            unavailable.remove(&slot);
        }
        drop(unavailable);
        let placement = self.inner.placement.borrow().clone();
        *self.inner.placement.borrow_mut() = placement
            .with_resources(placement.resources().clone())
            .expect("bank slab addition advances placement generation");
    }

    fn generation(&self) -> Generation {
        self.inner.placement.borrow().generation()
    }
}

/// Slab-backed history rows and recurrent banks. Graphs are sealed over the
/// logical shapes; only backed rows and banks are handed out.
struct StoreSlabs {
    history: Option<SlabTensor>,
    rows: usize,
    recurrent: Option<SlabTensor>,
    banks: usize,
}

/// Counts transactions that captured this store's tensors and write them
/// later (advances, compactions, conversions). The backing may be
/// changed only while none exists; a captured binding has a fixed slab
/// placement for its submission.
#[derive(Default)]
struct TransactionClaims {
    active: usize,
    histories: BTreeMap<u64, usize>,
    banks: BTreeMap<LogicalId, usize>,
}

#[derive(Clone)]
struct Transactions(Rc<RefCell<TransactionClaims>>);

struct Transaction {
    claims: Rc<RefCell<TransactionClaims>>,
    histories: Vec<u64>,
    banks: Vec<LogicalId>,
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
        self.banks.push(bank.id());
        *self.claims.borrow_mut().banks.entry(bank.id()).or_default() += 1;
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
        for id in &self.banks {
            let count = tracked.banks.get_mut(id).expect("tracked bank claim");
            *count -= 1;
            if *count == 0 {
                tracked.banks.remove(id);
            }
        }
    }
}
/// History rows are a shared arena; recurrent components are arenas of banks,
/// and each accepted version is one immutable bank. A checkpoint retains both
/// without copying tensor contents.
///
/// `history_capacity` rows and the bank capacity are reservations sealed into
/// graphs. Backing slabs are added as demand grows and released when empty
/// ([`StateStore::provision`], [`StateStore::shrink_with`]).
pub struct StateStore {
    device: Rc<Device>,
    context_capacity: usize,
    history_capacity: usize,
    components: Vec<ComponentDescriptor>,
    total_history_row_bytes: u64,
    history_slab_rows: usize,
    bank_capacity: BankCapacity,
    recurrent_bank_bytes: u64,
    bank_slab_banks: usize,
    component_specs: Vec<ComponentSpec>,
    backing: RefCell<StoreSlabs>,
    history_views: RefCell<Option<Vec<PlaneBuffer>>>,
    recurrent_views: RefCell<Option<Rc<[Tensor]>>>,
    retired_storage: RefCell<Vec<TensorStorageObserver>>,
    arena: Rc<RefCell<Arena>>,
    banks: BankPool,
    zero_seed: BankHandle,
    /// Live sequences and checkpoints: the store is idle only without both.
    owners: Cell<usize>,
    /// Live sequences alone: each may reserve a successor bank, while a
    /// checkpoint is frozen until a fork of it becomes a sequence.
    sequences: Cell<usize>,
    transactions: Transactions,
    compactions: Cell<Compactions>,
}

/// How [`StateStore::shrink_with`] releases slab backing. Idle releases empty
/// slabs beyond one spare per store. Reclaim also compacts occupied slabs and
/// releases every slab it can empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShrinkPolicy {
    Idle,
    Reclaim,
}

/// Shrink compactions a store performed: relocations of referenced history
/// rows and claimed recurrent banks into free slots of retained slabs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Compactions {
    pub count: usize,
    pub history_rows: usize,
    pub banks: usize,
}

/// One store's demand for history rows: a sequence whose history ends at
/// `after` (none for a fresh one) appending `rows`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowDemand {
    pub after: Option<usize>,
    pub rows: usize,
}

/// Additional Seismic charge for fixed slab additions during state growth.
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
    additional_slabs: usize,
    total: usize,
}
impl StateStore {
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
        let history_slab_rows = if components.is_empty() {
            0
        } else {
            history_rows_per_slab(total_history_row_bytes).map_err(Error::Request)?
        };
        let bank_slab_banks = if component_specs.is_empty() {
            0
        } else {
            banks_per_slab(recurrent_bank_bytes).map_err(Error::Request)?
        };
        let reserved_banks = bank_capacity.storage_total()?;
        // Banks of a store without recurrent components have no storage.
        let committed_banks = if component_specs.is_empty() {
            reserved_banks
        } else {
            bank_slab_banks.min(reserved_banks)
        };
        let (banks, zero_seed) = BankPool::new(bank_capacity, committed_banks)?;
        let mut recurrent = if component_specs.is_empty() {
            None
        } else {
            Some(SlabTensor::new(
                &device,
                bank_slab_banks as u64,
                reserved_banks as u64,
                component_specs
                    .iter()
                    .map(|spec| SlabRegion {
                        element: element(spec.dtype),
                        row_shape: spec.shape.iter().map(|&size| size as u64).collect(),
                    })
                    .collect(),
            )?)
        };
        if let Some(recurrent) = &mut recurrent {
            recurrent.add_slab()?;
        }
        let mut history = if components.is_empty() {
            None
        } else {
            Some(SlabTensor::new(
                &device,
                history_slab_rows as u64,
                history_capacity as u64,
                components
                    .iter()
                    .flat_map(ComponentDescriptor::planes)
                    .map(|plane| SlabRegion {
                        element: element(plane.dtype),
                        row_shape: plane.row_extents.iter().map(|&size| size as u64).collect(),
                    })
                    .collect(),
            )?)
        };
        let committed_rows = if let Some(history) = &mut history {
            history.add_slab()?;
            history_slab_rows.min(history_capacity)
        } else {
            0
        };
        Ok(Rc::new(Self {
            device,
            context_capacity,
            history_capacity,
            components,
            total_history_row_bytes,
            history_slab_rows,
            bank_capacity,
            recurrent_bank_bytes,
            bank_slab_banks,
            component_specs,
            backing: RefCell::new(StoreSlabs {
                history,
                rows: committed_rows,
                recurrent,
                banks: committed_banks,
            }),
            history_views: RefCell::new(None),
            recurrent_views: RefCell::new(None),
            retired_storage: RefCell::new(Vec::new()),
            arena: Rc::new(RefCell::new(Arena::new(
                committed_rows,
                history_slab_rows.max(1),
            ))),
            banks,
            zero_seed,
            owners: Cell::new(0),
            sequences: Cell::new(0),
            transactions: Transactions(Rc::new(RefCell::new(TransactionClaims::default()))),
            compactions: Cell::new(Compactions::default()),
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
        if let Some(views) = self.recurrent_views.borrow().as_ref() {
            return views.clone();
        }
        let views = self
            .backing
            .borrow()
            .recurrent
            .as_ref()
            .map(|slabs| {
                (0..self.component_specs.len())
                    .map(|index| slabs.logical_region(index).expect("backed recurrent slab"))
                    .collect::<Rc<[Tensor]>>()
            })
            .unwrap_or_else(|| Rc::from([]));
        *self.recurrent_views.borrow_mut() = Some(views.clone());
        views
    }
    /// Banks reserved in each recurrent arena, including the zero seed.
    pub fn recurrent_bank_count(&self) -> Result<usize, Error> {
        self.bank_capacity.storage_total()
    }
    /// One past the highest backed history row and recurrent bank. These
    /// extents may include unbacked slots after an interior slab is freed;
    /// [`StateStore::committed_bytes`] measures the actual backing.
    pub fn committed(&self) -> (usize, usize) {
        let backing = self.backing.borrow();
        (backing.rows, backing.banks)
    }
    /// Physical bytes of the committed backing.
    pub fn committed_bytes(&self) -> u64 {
        let backing = self.backing.borrow();
        backing
            .history
            .as_ref()
            .map_or(0, SlabTensor::storage_bytes)
            + backing
                .recurrent
                .as_ref()
                .map_or(0, SlabTensor::storage_bytes)
    }

    /// Whether a released slab might still be charged through an outside view.
    pub fn has_retired_storage(&self) -> bool {
        !self.retired_storage.borrow().is_empty()
    }

    /// Released slabs still charged because a caller holds a tensor view.
    /// The weak observers neither pin the allocations nor invent a charge:
    /// each live byte count comes from Seismic's physical allocation.
    pub fn external_pinned_bytes(&self) -> Result<u64, Error> {
        if !self.has_retired_storage() {
            return Ok(0);
        }
        let backing = self.backing.borrow();
        let current = backing
            .history
            .iter()
            .flat_map(SlabTensor::slab_storage_observers)
            .chain(
                backing
                    .recurrent
                    .iter()
                    .flat_map(SlabTensor::slab_storage_observers),
            )
            .map(|storage| storage.identity())
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

    fn record_retired_storage(&self, storage: TensorStorageObserver) {
        let mut retired = self.retired_storage.borrow_mut();
        retired.push(storage);
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
            banks.insert(holder.bank().id());
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
        let rows = self.committed().0;
        let placement = self.banks.inner.placement.borrow();
        let claimed_banks = placement
            .resources()
            .keys()
            .filter(|&&id| id != ZERO_SEED_CLAIM)
            .collect::<Vec<_>>();
        if claimed_banks
            .iter()
            .any(|id| !banks.contains(id) && !tracked.banks.contains_key(id))
        {
            return Err(Error::Request(
                "state census omits a recurrent bank claim".into(),
            ));
        }
        let occupied_rows = u64::try_from(arena.referenced)
            .map_err(|_| Error::Request("occupied row count exceeds u64".into()))?;
        let used_banks = u64::try_from(claimed_banks.len())
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
    pub fn history_slab_rows(&self) -> usize {
        self.history_slab_rows
    }
    pub fn max_visible_spans(&self) -> usize {
        max_visible_spans(self.context_capacity, self.history_slab_rows.max(1))
            .expect("validated context and slab rows")
    }
    pub fn bank_slab_banks(&self) -> usize {
        self.bank_slab_banks
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
        self.backing.borrow().rows != 0
    }

    /// Slab bindings cannot change while an accepted advance holds them.
    pub fn growth_unblocked(&self) -> bool {
        self.transactions.idle()
    }

    /// Add backing slabs for a launch before any transaction begins.
    /// A refused allocation returns a capacity
    /// error without publishing tentative backing or changing accepted state.
    /// Nothing changes while a transaction holds this store's tensors.
    pub fn provision(&self, demands: &[RowDemand], banks: usize) -> Result<(), Error> {
        self.provision_with_growth(demands, banks, GrowthChoice::Preferred)
    }

    /// A read-only claim for the slabs a launch must add. Minimum and
    /// preferred are equal while growth follows exact slab demand.
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
        let bank_slabs = self.bank_growth_slabs(banks)?;
        let history_slabs = history.additional_slabs;
        let history_bytes = backing
            .history
            .as_ref()
            .map_or(0, SlabTensor::slab_bytes)
            .checked_mul(history_slabs as u64)
            .ok_or_else(|| Error::Request("history slab claim overflows".into()))?;
        let bank_bytes = backing
            .recurrent
            .as_ref()
            .map_or(0, SlabTensor::slab_bytes)
            .checked_mul(bank_slabs as u64)
            .ok_or_else(|| Error::Request("bank slab claim overflows".into()))?;
        history_bytes
            .checked_add(bank_bytes)
            .ok_or_else(|| Error::Request("state slab claim overflows".into()))
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
        let bank_slabs = self.bank_growth_slabs(banks)?;
        let history_before = (bank_slabs != 0 && !self.components.is_empty())
            .then(|| self.arena.borrow().backed.clone());
        if !self.components.is_empty() {
            self.provision_history(demands, choice)?;
        }
        if bank_slabs != 0 {
            if let Err(error) = self.add_bank_slabs(bank_slabs) {
                if let Some(before) = &history_before {
                    self.rollback_history_growth(before)?;
                }
                return Err(error);
            }
        }
        Ok(())
    }

    /// Restore the published history backing when the bank half of one
    /// aggregate growth fails. No advance has claimed the newly added rows.
    fn rollback_history_growth(&self, before: &BTreeSet<usize>) -> Result<(), Error> {
        let added = self
            .arena
            .borrow()
            .backed
            .difference(before)
            .copied()
            .collect::<Vec<_>>();
        if added.is_empty() {
            return Ok(());
        }
        self.history_views.borrow_mut().take();
        let mut backing = self.backing.borrow_mut();
        let slabs = backing.history.as_mut().expect("history layout has slabs");
        for index in added {
            slabs.free_slab(index)?;
            self.arena.borrow_mut().unback_empty_slab(index);
        }
        backing.rows = slabs
            .slabs()
            .map(|(index, _)| ((index + 1) * self.history_slab_rows).min(self.history_capacity))
            .max()
            .unwrap_or(0);
        Ok(())
    }

    fn bank_growth_slabs(&self, banks: usize) -> Result<usize, Error> {
        let shortage = banks.saturating_sub(self.banks.available());
        if shortage == 0 || !self.has_recurrent_components() {
            return Ok(0);
        }
        let reserved = self.bank_capacity.storage_total()?;
        let backing = self.backing.borrow();
        let slabs = backing
            .recurrent
            .as_ref()
            .expect("recurrent layout has slabs");
        let backed = slabs
            .slabs()
            .map(|(index, _)| index)
            .collect::<BTreeSet<_>>();
        let mut additional = 0;
        let mut added_banks = 0;
        for index in 0..reserved.div_ceil(self.bank_slab_banks) {
            if !backed.contains(&index) && added_banks < shortage {
                let start = index * self.bank_slab_banks;
                added_banks += (start + self.bank_slab_banks).min(reserved) - start;
                additional += 1;
            }
        }
        if added_banks < shortage {
            return Err(Error::BanksExhausted {
                capacity: self.bank_capacity.total()?,
            });
        }
        Ok(additional)
    }

    /// Plan whole slab additions for the rows a launch needs beyond the
    /// backing's free rows. A history may continue in another span.
    fn history_growth_plan(
        &self,
        demands: &[RowDemand],
        _choice: GrowthChoice,
    ) -> Result<HistoryGrowthPlan, Error> {
        let total = demands.iter().map(|demand| demand.rows).sum::<usize>();
        if total == 0 || self.components.is_empty() {
            return Ok(HistoryGrowthPlan {
                additional_slabs: 0,
                total: 0,
            });
        }
        let arena = self.arena.borrow();
        let available = arena.available();
        let shortage = total.saturating_sub(available);
        let mut additional_slabs = 0;
        let mut added_rows = 0;
        for slab in 0..self.history_capacity.div_ceil(self.history_slab_rows) {
            if !arena.backed.contains(&slab) && added_rows < shortage {
                let start = slab * self.history_slab_rows;
                added_rows += (start + self.history_slab_rows).min(self.history_capacity) - start;
                additional_slabs += 1;
            }
        }
        if added_rows < shortage {
            return Err(Error::Capacity {
                required: total as u64 * self.total_history_row_bytes,
                available_bytes: (added_rows + available) as u64 * self.total_history_row_bytes,
            });
        }
        Ok(HistoryGrowthPlan {
            additional_slabs,
            total,
        })
    }

    fn provision_history(&self, demands: &[RowDemand], choice: GrowthChoice) -> Result<(), Error> {
        let plan = self.history_growth_plan(demands, choice)?;
        if plan.total == 0 {
            return Ok(());
        }
        if plan.additional_slabs > 0 {
            self.add_history_slabs(plan.additional_slabs)?;
        }
        Ok(())
    }

    /// Release empty history and bank slabs, retaining one spare of each at
    /// idle. Reclaim also moves referenced rows and claimed banks from sparse
    /// slabs into free slots of the retained slabs before releasing them.
    /// Returns the physical bytes released, measured by the device ledger:
    /// an earlier backing a caller still views stays charged. Nothing changes
    /// while a transaction exists.
    pub fn shrink_with<E, F>(self: &Rc<Self>, policy: ShrinkPolicy, mut copy: F) -> Result<u64, E>
    where
        E: From<Error>,
        F: FnMut(&SlabTensor, StoreCopy) -> Result<(), E>,
    {
        if !self.transactions.idle() {
            return Ok(0);
        }
        let before = self.device.memory_usage().charged;
        let backing = self.backing.borrow();
        let arena = self.arena.borrow();
        let mut history_slabs = arena
            .backed
            .iter()
            .map(|&index| {
                let start = index * self.history_slab_rows;
                let end = (start + self.history_slab_rows).min(self.history_capacity);
                let occupied = arena
                    .runs
                    .range(start..end)
                    .map(|(_, run)| run.count)
                    .sum::<usize>();
                (index, end - start, occupied)
            })
            .collect::<Vec<_>>();
        history_slabs
            .sort_unstable_by_key(|&(index, _, occupied)| (std::cmp::Reverse(occupied), index));
        let mut history_keep = BTreeSet::new();
        match policy {
            ShrinkPolicy::Idle => {
                history_keep.extend(
                    history_slabs
                        .iter()
                        .filter(|(_, _, occupied)| *occupied > 0)
                        .map(|(index, _, _)| *index),
                );
                if let Some(&(index, _, _)) =
                    history_slabs.iter().find(|(_, _, occupied)| *occupied == 0)
                {
                    history_keep.insert(index);
                }
            }
            ShrinkPolicy::Reclaim => {
                let mut capacity = 0;
                for &(index, rows, _) in &history_slabs {
                    if capacity >= arena.referenced {
                        break;
                    }
                    history_keep.insert(index);
                    capacity += rows;
                }
            }
        }
        let (planned_arena, history_moves) = arena.compact_into_slabs(backing.rows, &history_keep);
        drop(arena);

        let published_banks = self.banks.inner.placement.borrow().clone();
        let reserved = self.bank_capacity.storage_total().map_err(E::from)?;
        let mut bank_slabs = backing
            .recurrent
            .as_ref()
            .map(|slabs| {
                slabs
                    .slabs()
                    .map(|(index, _)| {
                        let start = index * self.bank_slab_banks;
                        let end = (start + self.bank_slab_banks).min(reserved);
                        let occupied = published_banks
                            .resources()
                            .values()
                            .filter(|&&slot| (start..end).contains(&slot))
                            .count();
                        (index, end - start, occupied)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        bank_slabs
            .sort_unstable_by_key(|&(index, _, occupied)| (std::cmp::Reverse(occupied), index));
        let mut bank_keep = BTreeSet::new();
        match policy {
            ShrinkPolicy::Idle => {
                bank_keep.extend(
                    bank_slabs
                        .iter()
                        .filter(|(_, _, occupied)| *occupied > 0)
                        .map(|(index, _, _)| *index),
                );
                if let Some(&(index, _, _)) =
                    bank_slabs.iter().find(|(_, _, occupied)| *occupied == 0)
                {
                    bank_keep.insert(index);
                }
            }
            ShrinkPolicy::Reclaim => {
                let mut capacity = 0;
                if !bank_slabs.is_empty() {
                    bank_keep.insert(0);
                    capacity = self.bank_slab_banks.min(reserved);
                }
                for &(index, banks, _) in &bank_slabs {
                    if capacity >= published_banks.resources().len() {
                        break;
                    }
                    if bank_keep.insert(index) {
                        capacity += banks;
                    }
                }
            }
        }
        let occupied = published_banks
            .resources()
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        let mut free_banks = bank_keep
            .iter()
            .flat_map(|&slab| {
                let start = slab * self.bank_slab_banks;
                start..(start + self.bank_slab_banks).min(reserved)
            })
            .filter(|index| !occupied.contains(index))
            .collect::<Vec<_>>()
            .into_iter();
        let mut planned_resources = published_banks.resources().clone();
        let mut bank_moves = Vec::new();
        for (&resource, &from) in published_banks.resources() {
            if bank_slabs.is_empty() {
                break;
            }
            if !bank_keep.contains(&(from / self.bank_slab_banks)) {
                let to = free_banks
                    .next()
                    .expect("kept bank slabs hold every claimed bank");
                planned_resources.insert(resource, to);
                bank_moves.push((from, to));
            }
        }
        let planned_banks = if bank_moves.is_empty() {
            published_banks.clone()
        } else {
            published_banks
                .with_resources(planned_resources)
                .map_err(Error::from)
                .map_err(E::from)?
        };

        // Both plans read the same published generation. Every destination is
        // free there, so even a later failed copy leaves both stores readable
        // and no physical slab has been released.
        if !history_moves.is_empty() {
            let slabs = backing.history.as_ref().expect("history layout has slabs");
            copy(
                slabs,
                self.history_copy_plan(slabs, &history_moves)
                    .map_err(E::from)?,
            )?;
        }
        if !bank_moves.is_empty() {
            let slabs = backing
                .recurrent
                .as_ref()
                .expect("recurrent layout has slabs");
            copy(
                slabs,
                self.bank_copy_plan(slabs, &bank_moves).map_err(E::from)?,
            )?;
        }
        drop(backing);

        *self.arena.borrow_mut() = planned_arena;
        *self.banks.inner.placement.borrow_mut() = planned_banks;
        let mut backing = self.backing.borrow_mut();
        if let Some(slabs) = backing.history.as_mut() {
            self.history_views.borrow_mut().take();
            let victims = slabs
                .slabs()
                .map(|(index, _)| index)
                .filter(|index| !history_keep.contains(index))
                .collect::<Vec<_>>();
            for index in victims {
                if let Some(observer) = slabs
                    .free_slab(index)
                    .map_err(Error::from)
                    .map_err(E::from)?
                {
                    self.record_retired_storage(observer);
                }
            }
            backing.rows = slabs
                .slabs()
                .map(|(index, _)| ((index + 1) * self.history_slab_rows).min(self.history_capacity))
                .max()
                .unwrap_or(0);
        }
        if let Some(slabs) = backing.recurrent.as_mut() {
            self.recurrent_views.borrow_mut().take();
            let victims = slabs
                .slabs()
                .map(|(index, _)| index)
                .filter(|index| !bank_keep.contains(index))
                .collect::<Vec<_>>();
            for index in victims {
                if let Some(observer) = slabs
                    .free_slab(index)
                    .map_err(Error::from)
                    .map_err(E::from)?
                {
                    let start = index * self.bank_slab_banks;
                    let end = (start + self.bank_slab_banks).min(reserved);
                    self.banks.mark_slab_unavailable(start, end);
                    self.record_retired_storage(observer);
                }
            }
            let extent = slabs
                .slabs()
                .map(|(index, _)| ((index + 1) * self.bank_slab_banks).min(reserved))
                .max()
                .expect("zero seed slab stays backed");
            if extent < backing.banks {
                self.banks.resize(backing.banks, extent);
                backing.banks = extent;
            }
        }
        drop(backing);
        if !history_moves.is_empty() || !bank_moves.is_empty() {
            let stats = self.compactions.get();
            self.compactions.set(Compactions {
                count: stats.count + 1,
                history_rows: stats.history_rows
                    + history_moves
                        .iter()
                        .map(|(_, _, count)| count)
                        .sum::<usize>(),
                banks: stats.banks + bank_moves.len(),
            });
        }
        Ok(before.saturating_sub(self.device.memory_usage().charged))
    }

    #[cfg(test)]
    fn shrink(self: &Rc<Self>, policy: ShrinkPolicy) -> Result<u64, Error> {
        self.shrink_with(policy, |slabs, plan| {
            for copy in plan.copies() {
                for (&from, &to) in copy.from.iter().zip(&copy.to) {
                    let source = slabs.region_rows(copy.plane_index, from as u64, 1)?;
                    let mut destination = slabs.region_rows(copy.plane_index, to as u64, 1)?;
                    let bytes = source
                        .read_to_host()
                        .map_err(|error| Error::Request(error.to_string()))?;
                    destination
                        .write_from_host(&bytes)
                        .map_err(|error| Error::Request(error.to_string()))?;
                }
            }
            Ok(())
        })
    }

    fn history_copy_plan(
        self: &Rc<Self>,
        slabs: &SlabTensor,
        moves: &[(usize, usize, usize)],
    ) -> Result<StoreCopy, Error> {
        let from = moves
            .iter()
            .flat_map(|&(start, _, count)| start..start + count)
            .collect();
        let to = moves
            .iter()
            .flat_map(|&(_, start, count)| start..start + count)
            .collect();
        StoreCopy::new(self.clone(), slabs, from, to)
    }

    fn bank_copy_plan(
        self: &Rc<Self>,
        slabs: &SlabTensor,
        moves: &[(usize, usize)],
    ) -> Result<StoreCopy, Error> {
        let from = moves.iter().map(|&(from, _)| from).collect();
        let to = moves.iter().map(|&(_, to)| to).collect();
        StoreCopy::new(self.clone(), slabs, from, to)
    }

    pub fn compactions(&self) -> Compactions {
        self.compactions.get()
    }

    /// The published recurrent bank placement generation. It advances with
    /// every acquisition, release, relocation and resize of the bank arena.
    pub fn bank_placement_generation(&self) -> Generation {
        self.banks.generation()
    }

    /// Add whole history slabs into the lowest unbacked slots.
    fn add_history_slabs(&self, count: usize) -> Result<(), Error> {
        let mut backing = self.backing.borrow_mut();
        if count == 0 {
            return Ok(());
        }
        let slab = backing.history.as_mut().expect("history layout has slabs");
        self.history_views.borrow_mut().take();
        let mut added = Vec::new();
        for _ in 0..count {
            match slab.add_slab() {
                Ok(index) => added.push(index),
                Err(error) => {
                    for index in added {
                        slab.free_slab(index)?;
                    }
                    return Err(error.into());
                }
            }
        }
        let mut arena = self.arena.borrow_mut();
        for index in added {
            arena.back_slab(index, self.history_capacity);
            backing.rows = backing
                .rows
                .max(((index + 1) * self.history_slab_rows).min(self.history_capacity));
        }
        Ok(())
    }

    /// Add whole bank slabs into the lowest unbacked slots.
    fn add_bank_slabs(&self, count: usize) -> Result<(), Error> {
        let mut backing = self.backing.borrow_mut();
        if count == 0 {
            return Ok(());
        }
        let slab = backing
            .recurrent
            .as_mut()
            .expect("recurrent layout has slabs");
        self.recurrent_views.borrow_mut().take();
        let mut added = Vec::new();
        for _ in 0..count {
            match slab.add_slab() {
                Ok(index) => added.push(index),
                Err(error) => {
                    for index in added {
                        slab.free_slab(index)?;
                    }
                    return Err(error.into());
                }
            }
        }
        let reserved = self.bank_capacity.storage_total()?;
        for index in added {
            let start = index * self.bank_slab_banks;
            let end = (start + self.bank_slab_banks).min(reserved);
            if end > backing.banks {
                self.banks.resize(backing.banks, end);
                backing.banks = end;
            }
            self.banks.mark_slab_available(start, end);
        }
        Ok(())
    }

    pub fn history_planes(&self) -> Result<Vec<PlaneBuffer>, Error> {
        if let Some(views) = self.history_views.borrow().as_ref() {
            return Ok(views.clone());
        }
        let backing = self.backing.borrow();
        if backing.rows == 0 {
            return Ok(Vec::new());
        }
        let slabs = backing.history.as_ref().expect("history layout has slabs");
        let views = self
            .components
            .iter()
            .enumerate()
            .flat_map(|(component_index, component)| {
                component
                    .planes()
                    .iter()
                    .map(move |plane| (component_index, component.layer, plane))
            })
            .enumerate()
            .map(|(plane_index, (component_index, layer, plane))| {
                Ok(PlaneBuffer {
                    plane_index,
                    component_index,
                    layer,
                    vector: plane.vector,
                    name: plane.name,
                    row_bytes: plane.row_bytes,
                    slab_rows: u32::try_from(self.history_slab_rows).expect("slab rows fit u32"),
                    base_row: 0,
                    buffer: slabs.logical_region(plane_index)?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        *self.history_views.borrow_mut() = Some(views.clone());
        Ok(views)
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
        let before = self.device.memory_usage().charged;
        let mut backing = self.backing.borrow_mut();
        let StoreSlabs { rows, history, .. } = &mut *backing;
        if let Some(slabs) = history.as_mut() {
            self.history_views.borrow_mut().take();
            let held = slabs.slabs().map(|(index, _)| index).collect::<Vec<_>>();
            for index in held {
                if let Some(observer) = slabs.free_slab(index)? {
                    self.arena.borrow_mut().unback_empty_slab(index);
                    self.record_retired_storage(observer);
                }
                *rows = slabs
                    .slabs()
                    .map(|(index, _)| {
                        ((index + 1) * self.history_slab_rows).min(self.history_capacity)
                    })
                    .max()
                    .unwrap_or(0);
            }
        }
        drop(backing);
        *self.arena.borrow_mut() = Arena::new(0, self.history_slab_rows.max(1));
        usize::try_from(before.saturating_sub(self.device.memory_usage().charged))
            .map_err(|_| Error::Request("reclaimable history bytes exceed host range".into()))
    }
    pub fn create(self: &Rc<Self>) -> Result<SequenceState, Error> {
        let bank = self.zero_seed.clone();
        let claims = Claims::new(&self.arena, vec![]);
        claims.set_live(true);
        self.owners.set(self.owners.get() + 1);
        self.sequences.set(self.sequences.get() + 1);
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
        let needs_preparation = self.arena.borrow().available() < count;
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
        self.store.sequences.set(self.store.sequences.get() - 1);
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
        self.history_ranges().len() >= self.store.max_visible_spans()
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

/// Pinned slab views and row maps for one unpublished store compaction.
/// The caller must finish every physical copy before returning success.
pub struct StoreCopy {
    store: Rc<StateStore>,
    planes: Vec<Tensor>,
    copies: Vec<PlaneCopy>,
    slab_rows: u32,
    row_capacity: usize,
}

impl StoreCopy {
    fn new(
        store: Rc<StateStore>,
        slabs: &SlabTensor,
        from: Vec<usize>,
        to: Vec<usize>,
    ) -> Result<Self, Error> {
        let planes = (0..slabs.region_count())
            .map(|index| slabs.logical_region(index))
            .collect::<Result<Vec<_>, _>>()?;
        let copies = (0..planes.len())
            .map(|plane_index| PlaneCopy {
                plane_index,
                from: from.clone(),
                to: to.clone(),
            })
            .collect();
        Ok(Self {
            store,
            planes,
            copies,
            slab_rows: u32::try_from(slabs.rows_per_slab()).expect("slab rows fit u32"),
            row_capacity: usize::try_from(slabs.logical_rows()).expect("logical rows fit usize"),
        })
    }

    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        Rc::ptr_eq(&self.store, store)
    }
    pub fn planes(&self) -> &[Tensor] {
        &self.planes
    }
    pub fn copies(&self) -> &[PlaneCopy] {
        &self.copies
    }
    pub fn slab_rows(&self) -> u32 {
        self.slab_rows
    }
    pub fn row_capacity(&self) -> usize {
        self.row_capacity
    }
    pub fn rows(&self) -> usize {
        self.copies.first().map_or(0, |copy| copy.from.len())
    }
    pub fn chunk(&self, start: usize, end: usize) -> Self {
        Self {
            store: self.store.clone(),
            planes: self.planes.clone(),
            copies: self
                .copies
                .iter()
                .map(|copy| PlaneCopy {
                    plane_index: copy.plane_index,
                    from: copy.from[start..end].to_vec(),
                    to: copy.to[start..end].to_vec(),
                })
                .collect(),
            slab_rows: self.slab_rows,
            row_capacity: self.row_capacity,
        }
    }
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
        self.store.sequences.set(self.store.sequences.get() + 1);
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
        Rc::new(RefCell::new(Arena::new(rows, rows.max(1))))
    }

    #[test]
    fn history_spans_and_free_holes_stop_at_slab_boundaries() {
        let arena = Rc::new(RefCell::new(Arena::new(0, 4)));
        for index in 0..3 {
            arena.borrow_mut().back_slab(index, 12);
        }
        assert_eq!(arena.borrow().free, vec![(0, 4), (4, 4), (8, 4)]);
        let ranges = arena.borrow_mut().claim(None, 6);
        assert!(ranges
            .iter()
            .all(|(start, count)| start / 4 == (start + count - 1) / 4));
        let claims = Claims::new(&arena, ranges);
        assert!(claims
            .ranges()
            .iter()
            .all(|(start, count)| start / 4 == (start + count - 1) / 4));
        drop(claims);
        assert_eq!(arena.borrow().free, vec![(0, 4), (4, 4), (8, 4)]);
    }

    #[test]
    fn idle_shrink_does_not_compact_occupied_history_or_bank_slabs() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            8192,
            8192,
            vec![dense_component(4096)],
            vec![ComponentSpec {
                shape: vec![4 * 1024 * 1024],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 7,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        store.add_history_slabs(1).unwrap();
        store.add_bank_slabs(1).unwrap();
        let rows = store.history_slab_rows();
        let _low = claim_rows(&store.arena, 0, 1);
        let high = claim_rows(&store.arena, rows, 1);
        let mut claims = (0..4)
            .map(|_| Some(store.banks.acquire().unwrap()))
            .collect::<Vec<_>>();
        drop(claims[0].take());
        drop(claims[1].take());
        let bank = claims[3].as_ref().unwrap();
        let before = (
            device.memory_usage().charged,
            store.committed(),
            high.ranges(),
            bank.index(),
            store.bank_placement_generation(),
            store.compactions(),
        );
        assert_eq!(
            store
                .shrink_with(ShrinkPolicy::Idle, |_, _| -> Result<(), Error> {
                    panic!("idle shrink must not copy occupied slabs")
                })
                .unwrap(),
            0
        );
        assert_eq!(
            (
                device.memory_usage().charged,
                store.committed(),
                high.ranges(),
                bank.index(),
                store.bank_placement_generation(),
                store.compactions(),
            ),
            before
        );
    }

    #[test]
    fn idle_shrink_keeps_one_empty_history_slab() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            8192,
            8192,
            vec![dense_component(4096)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        store.add_history_slabs(2).unwrap();
        let occupied = claim_rows(&store.arena, 0, 1);
        let before = device.memory_usage().charged;
        assert!(
            store
                .shrink_with(ShrinkPolicy::Idle, |_, _| -> Result<(), Error> {
                    panic!("idle shrink must not copy")
                })
                .unwrap()
                > 0
        );
        assert_eq!(
            store
                .backing
                .borrow()
                .history
                .as_ref()
                .unwrap()
                .slabs()
                .map(|(index, _)| index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(device.memory_usage().charged < before);
        drop(occupied);
        store.shrink(ShrinkPolicy::Idle).unwrap();
        assert_eq!(
            store
                .backing
                .borrow()
                .history
                .as_ref()
                .unwrap()
                .slabs()
                .map(|(index, _)| index)
                .collect::<Vec<_>>(),
            vec![0]
        );
        assert_eq!(store.compactions(), Compactions::default());
    }

    #[test]
    fn cpu_reclaim_compacts_partial_final_history_and_bank_slabs() {
        let Some(device) = cpu_device() else {
            return;
        };
        reclaim_compacts_partial_final_history_and_bank_slabs_on(device);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_reclaim_compacts_partial_final_history_and_bank_slabs() {
        let device = Rc::new(
            DeviceCatalog::discover()
                .unwrap()
                .open_backend(BackendName::Metal)
                .unwrap(),
        );
        reclaim_compacts_partial_final_history_and_bank_slabs_on(device);
    }

    fn reclaim_compacts_partial_final_history_and_bank_slabs_on(device: Rc<Device>) {
        let store = StateStore::new(
            device.clone(),
            5000,
            5000,
            vec![dense_component(4096)],
            vec![ComponentSpec {
                shape: vec![4 * 1024 * 1024],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 4,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        store.add_history_slabs(1).unwrap();
        store.add_bank_slabs(1).unwrap();
        let rows = store.history_slab_rows();
        assert!(rows < 5000);
        assert_eq!(store.bank_slab_banks(), 4);
        let _low = claim_rows(&store.arena, 0, 2);
        let high = claim_rows(&store.arena, rows, 1);
        let history_value = vec![29u8; 4096 * 4];
        store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .region_rows(0, rows as u64, 1)
            .unwrap()
            .write_from_host(&history_value)
            .unwrap();
        let mut claims = (0..4)
            .map(|_| Some(store.banks.acquire().unwrap()))
            .collect::<Vec<_>>();
        drop(claims[0].take());
        drop(claims[1].take());
        let bank = claims[3].as_ref().unwrap();
        let bank_value = vec![31u8; 16 * 1024 * 1024];
        write_bank(&store, bank.index(), &bank_value);
        let before = device.memory_usage().charged;
        assert!(store.shrink(ShrinkPolicy::Reclaim).unwrap() > 0);
        assert_eq!(store.committed(), (rows, 4));
        assert!(high.ranges()[0].0 < rows);
        assert!(bank.index() < 4);
        assert_eq!(
            store
                .backing
                .borrow()
                .history
                .as_ref()
                .unwrap()
                .region_rows(0, high.ranges()[0].0 as u64, 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            history_value
        );
        assert_eq!(bank_bytes(&store, bank.index()), bank_value);
        assert_eq!(store.compactions().history_rows, 1);
        assert_eq!(store.compactions().banks, 1);
        assert!(device.memory_usage().charged < before);
    }

    #[test]
    fn reclaim_selects_the_least_occupied_history_slab() {
        let arena = Rc::new(RefCell::new(Arena::new(0, 4)));
        for index in 0..3 {
            arena.borrow_mut().back_slab(index, 12);
        }
        let _full = claim_rows(&arena, 0, 4);
        let sparse = claim_rows(&arena, 4, 1);
        let _high = claim_rows(&arena, 8, 3);
        let (index, compacted, moves) = arena
            .borrow()
            .least_occupied_slab_compaction(12)
            .expect("one row fits in the high slab");
        assert_eq!(index, 1);
        assert_eq!(moves, vec![(4, 11, 1)]);
        assert_eq!(compacted.referenced, 8);
        assert_eq!(compacted.runs.range(4..8).count(), 0);
        assert_eq!(sparse.ranges(), vec![(4, 1)]);
    }

    #[test]
    fn reclaim_frees_the_sparsest_history_slab_without_new_charge() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            8192,
            8192,
            vec![dense_component(4096)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        store.add_history_slabs(2).unwrap();
        let rows = store.history_slab_rows();
        let _low = claim_rows(&store.arena, 0, rows);
        let sparse = claim_rows(&store.arena, rows, 1);
        let _high = claim_rows(&store.arena, 2 * rows, rows - 1);
        let value = vec![43u8; 4096 * 4];
        store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .region_rows(0, rows as u64, 1)
            .unwrap()
            .write_from_host(&value)
            .unwrap();
        let slab_bytes = store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .slab_bytes();
        let charged = device.memory_usage().charged;
        device.set_memory_limit(Some(charged));
        assert_eq!(store.shrink(ShrinkPolicy::Reclaim).unwrap(), slab_bytes);
        device.set_memory_limit(None);
        assert_eq!(store.compactions().history_rows, 1);
        assert_eq!(sparse.ranges(), vec![(3 * rows - 1, 1)]);
        assert!(store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .slab(1)
            .is_none());
        assert_eq!(
            store
                .backing
                .borrow()
                .history
                .as_ref()
                .unwrap()
                .region_rows(0, (3 * rows - 1) as u64, 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            value
        );
    }

    #[test]
    fn failed_store_copy_preserves_published_history_and_charge() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            8192,
            8192,
            vec![dense_component(4096)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        store.add_history_slabs(2).unwrap();
        let rows = store.history_slab_rows();
        let _low = claim_rows(&store.arena, 0, rows);
        let sparse = claim_rows(&store.arena, rows, 1);
        let _high = claim_rows(&store.arena, 2 * rows, rows - 1);
        let before_charge = device.memory_usage().charged;
        let before_ranges = sparse.ranges();
        let before_stats = store.compactions();
        let failure = store.shrink_with(ShrinkPolicy::Reclaim, |_, plan| {
            assert_eq!(plan.rows(), 1);
            Err(Error::Request("injected copy failure".into()))
        });
        assert!(
            matches!(failure, Err(Error::Request(message)) if message == "injected copy failure")
        );
        assert_eq!(sparse.ranges(), before_ranges);
        assert_eq!(store.compactions(), before_stats);
        assert_eq!(device.memory_usage().charged, before_charge);
        assert!(store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .slab(1)
            .is_some());
    }

    #[test]
    fn released_history_slab_stays_charged_while_a_view_pins_it() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            4096,
            4096,
            vec![dense_component(4)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let slab_bytes = store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .slab_bytes();
        let binding = store.history_planes().unwrap()[0].buffer.clone();
        let charged = device.memory_usage().charged;
        assert!(store.shrink(ShrinkPolicy::Reclaim).is_err());
        assert!(store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .slab(0)
            .is_some());
        assert_eq!(device.memory_usage().charged, charged);
        drop(binding);
        let pinned = store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .region_rows(0, 0, 1)
            .unwrap();
        assert_eq!(store.shrink(ShrinkPolicy::Reclaim).unwrap(), 0);
        assert_eq!(store.committed().0, 0);
        assert_eq!(device.memory_usage().charged, charged);
        assert_eq!(store.external_pinned_bytes().unwrap(), slab_bytes);
        assert_eq!(
            store.holding_census(&[], &[], &[]).unwrap().total()
                + store.external_pinned_bytes().unwrap(),
            device.memory_usage().charged
        );
        drop(pinned);
        assert_eq!(store.external_pinned_bytes().unwrap(), 0);
        assert_eq!(device.memory_usage().charged, charged - slab_bytes);
        assert_eq!(
            store.holding_census(&[], &[], &[]).unwrap().total(),
            device.memory_usage().charged
        );
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
        let mut arena = Arena::new(100, 100);
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
    fn planes_start_with_one_slab_as_one_stable_set() {
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
        assert!(store.history_allocated());
        assert_eq!(store.committed().0, 8);
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
    fn full_backing_continues_a_history_in_another_span() {
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
        // Two free rows follow `first`, so its third new row uses another span.
        let advance = OwnedStateAdvance::begin(first, 3).ok().unwrap();
        assert_eq!(store.committed().0, 8);
        assert_eq!(advance.bindings().destinations, [2, 3, 6]);
        assert_eq!(middle.history_ranges(), [(4, 2)]);
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
    fn fragmented_history_uses_free_committed_rows_without_growth_claim() {
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
        assert_eq!(claim.minimum_bytes, 0);
        let charged = device.memory_usage().charged;
        device.set_memory_limit(Some(charged + claim.minimum_bytes));
        store
            .provision_with_growth(&demand, 0, GrowthChoice::Minimum)
            .unwrap();
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
        // A 200-row prefix (retained by a checkpoint), then 16 one-row runs
        // separated by rows other histories hold.
        let mut state = store.create().unwrap();
        state.claims.append(claim_rows(&store.arena, 0, 200));
        state.position = 200;
        let prefix = state.checkpoint();
        let mut fillers = Vec::new();
        for run in 0..16 {
            let row = 300 + 2 * run;
            fillers.push(claim_rows(&store.arena, row + 1, 1));
            state.claims.append(claim_rows(&store.arena, row, 1));
            state.position += 1;
        }
        assert_eq!(state.history_ranges().len(), store.max_visible_spans());
        assert!(state.compaction_needed());
        let OwnedCompactionPreparation::Ready(compaction) =
            OwnedCompaction::prepare(state, 64).ok().unwrap()
        else {
            panic!("the recent runs fit one bounded copy")
        };
        assert_eq!(compaction.rows(), 16);
        let copy = &compaction.copies()[0];
        assert_eq!(
            copy.from,
            (0..16).map(|run| 300 + 2 * run).collect::<Vec<_>>()
        );
        // The first free run that fits is the one right after the prefix.
        assert_eq!(copy.to, (200..216).collect::<Vec<_>>());
        let state = compaction.commit();
        assert_eq!(state.history_ranges(), [(0, 216)]);
        assert_eq!(prefix.history_ranges(), [(0, 200)]);
        // The moved runs' rows are free again; the prefix is held once.
        assert_eq!(store.occupied_rows(), 200 + 16 + fillers.len());
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
        // finished request reads its whole history back through its ranges.
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
            written,
        }
    }

    /// The backing commits rows and banks with demand, keeps every row's
    /// contents across growth, and returns bytes to the device ledger when
    /// a slab is unreferenced; placement is fixed during a transaction.
    #[test]
    fn backing_grows_and_shrinks_and_returns_device_memory() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            16384,
            16384,
            vec![dense_component(4096)],
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
        assert_eq!(
            store.committed(),
            (store.history_slab_rows(), store.bank_slab_banks().min(49))
        );
        // A 1,000-row prefill commits exactly one slab.
        let advance = OwnedStateAdvance::begin(store.create().unwrap(), 1000)
            .ok()
            .unwrap();
        let rows = store.committed().0;
        assert_eq!(rows, store.history_slab_rows());
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
            .flat_map(|row| (row as f32).to_le_bytes().repeat(4096))
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
        assert_eq!(grown, base);
        // Without transactions, growth commits more rows and banks and keeps
        // the accepted rows' contents.
        let demand = [state.demand(10000)];
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
        assert!(rows >= 11000 && rows <= 16384, "committed {rows} rows");
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
        // A launch binding must be released before placement changes.
        drop(plane);
        let before = charged();
        let released = store.shrink(ShrinkPolicy::Reclaim).unwrap();
        assert_eq!(released, before.saturating_sub(charged()));
        assert!(store.committed().0 < rows);
        assert!(store.committed().0 >= 1000);
        assert_eq!(store.external_pinned_bytes().unwrap(), 0);
        drop(state);
        // Idle hysteresis keeps a small backing; reclaim releases it all.
        store.shrink(ShrinkPolicy::Reclaim).unwrap();
        assert_eq!(store.committed().0, 0);
        store.release_idle().unwrap();
        assert!(charged() < base);
    }

    /// Refused slab growth keeps the published backing and existing rows.
    #[test]
    fn failed_slab_growth_keeps_published_history() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            32768,
            32768,
            vec![dense_component(1024)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let encoded = (0..1024u32)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        {
            let backing = store.backing.borrow();
            backing
                .history
                .as_ref()
                .unwrap()
                .region_rows(0, 0, 1)
                .unwrap()
                .write_from_host(&encoded)
                .unwrap();
        }
        let baseline = device.memory_usage().charged;
        device.set_memory_limit(Some(baseline));
        assert!(matches!(
            store.add_history_slabs(1),
            Err(Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity { .. }
            )))
        ));
        device.set_memory_limit(None);
        assert_eq!(store.committed().0, store.history_slab_rows());
        assert_eq!(device.memory_usage().charged, baseline);
        let backing = store.backing.borrow();
        assert_eq!(
            backing
                .history
                .as_ref()
                .unwrap()
                .region_rows(0, 0, 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            encoded
        );
    }

    #[test]
    fn failed_bank_growth_rolls_back_history_growth() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            8192,
            8192,
            vec![dense_component(4096)],
            vec![ComponentSpec {
                shape: vec![4 * 1024 * 1024],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 8,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let demand = [RowDemand {
            after: None,
            rows: store.history_slab_rows() + 1,
        }];
        let baseline = device.memory_usage().charged;
        let committed = store.committed();
        let history_bytes = store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .slab_bytes();
        let bank_bytes = store
            .backing
            .borrow()
            .recurrent
            .as_ref()
            .unwrap()
            .slab_bytes();
        assert_eq!(
            store.growth_claim(&demand, 4).unwrap().minimum_bytes,
            history_bytes + bank_bytes
        );
        device.set_memory_limit(Some(baseline + history_bytes + bank_bytes - 1));
        assert!(matches!(
            store.provision_with_growth(&demand, 4, GrowthChoice::Minimum),
            Err(Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity { .. }
            )))
        ));
        device.set_memory_limit(None);
        assert_eq!(store.committed(), committed);
        assert_eq!(device.memory_usage().charged, baseline);
        assert_eq!(store.arena.borrow().backed, BTreeSet::from([0]));

        store
            .provision_with_growth(&demand, 4, GrowthChoice::Minimum)
            .unwrap();
        assert_eq!(store.arena.borrow().backed, BTreeSet::from([0, 1]));
        assert_eq!(store.committed().1, committed.1 + store.bank_slab_banks());
        assert_eq!(
            device.memory_usage().charged,
            baseline + history_bytes + bank_bytes
        );
    }

    /// Regression (chost 15:36): shrinking after every step released the
    /// successor banks the next step regrew, reallocating and copying the
    /// bank arenas every other decode step. With hysteresis a request that
    /// decodes 300 steps beside a retained prompt checkpoint changes the
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
    }

    /// Reclaim frees an empty interior history slab without moving a shared
    /// prefix, and growth reuses the lowest vacant slab index.
    #[test]
    fn reclaim_frees_empty_interior_slab_and_reuses_lowest_index() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            4096,
            8192,
            vec![dense_component(4096)],
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
                    .write_from_host(&tag(advance.position() + offset).repeat(4096))
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
        let long_ranges = long.history_ranges();
        let system_ranges = system.history_ranges();
        let short_ranges = short.history_ranges();
        assert!(long.history_ranges()[0].0 >= store.history_slab_rows());
        drop(large);
        let before = (store.committed().0, device.memory_usage().charged);
        let occupied = store.occupied_rows();
        assert_eq!(occupied, 100 + 50 + 20);
        // Reclaim copies only into space held by the store and succeeds when
        // the device refuses all new allocations.
        device.set_memory_limit(Some(before.1));
        let released = store.shrink(ShrinkPolicy::Reclaim).unwrap();
        device.set_memory_limit(None);
        assert!(released > 0);
        assert!(store.committed().0 <= before.0);
        assert!(device.memory_usage().charged < before.1);
        assert_eq!(store.compactions(), Compactions::default());
        assert_eq!(read(long.history_ranges()), expected(150));
        assert_eq!(store.compactions().count, 0);
        assert!(device.memory_usage().charged < before.1);
        assert_eq!(store.occupied_rows(), occupied);
        assert_eq!(long.history_ranges(), long_ranges);
        assert_eq!(system.history_ranges(), system_ranges);
        assert_eq!(short.history_ranges(), short_ranges);
        assert_eq!(read(long.history_ranges()), expected(150));
        assert_eq!(read(short.history_ranges()), expected(120));
        assert_eq!(read(system.history_ranges()), expected(100));
        let demand = [RowDemand {
            after: None,
            rows: store.history_slab_rows(),
        }];
        let claim = store.growth_claim(&demand, 0).unwrap();
        let slab_bytes = store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .slab_bytes();
        assert_eq!(claim.minimum_bytes, slab_bytes);
        assert_eq!(claim.preferred_bytes, slab_bytes);
        store.provision(&demand, 0).unwrap();
        assert!(store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .slab(0)
            .is_some());
    }

    #[test]
    fn reclaim_compacts_across_a_sparse_history_store() {
        let catalog = DeviceCatalog::discover().unwrap();
        for backend in [
            BackendName::Cpu,
            BackendName::Metal,
            BackendName::Cuda,
            BackendName::Vulkan,
        ] {
            let Ok(device) = catalog.open_backend(backend) else {
                continue;
            };
            let device = Rc::new(device);
            let store = StateStore::new(
                device.clone(),
                8192,
                8192,
                vec![dense_component(4096)],
                vec![],
                BankCapacity {
                    active: 1,
                    in_flight: 1,
                    retained: 0,
                },
            )
            .unwrap();
            let slab_rows = store.history_slab_rows();
            store.add_history_slabs(2).unwrap();
            let _low = claim_rows(&store.arena, 0, 1);
            let high = claim_rows(&store.arena, 2 * slab_rows + 1, 1);
            let value = (19f32).to_le_bytes().repeat(4096);
            store
                .backing
                .borrow()
                .history
                .as_ref()
                .unwrap()
                .region_rows(0, (2 * slab_rows + 1) as u64, 1)
                .unwrap()
                .write_from_host(&value)
                .unwrap();
            let slab_bytes = store
                .backing
                .borrow()
                .history
                .as_ref()
                .unwrap()
                .slab_bytes();
            let charged = device.memory_usage().charged;
            device.set_memory_limit(Some(charged));
            assert_eq!(store.shrink(ShrinkPolicy::Reclaim).unwrap(), 2 * slab_bytes);
            device.set_memory_limit(None);
            assert_eq!(store.committed().0, slab_rows);
            assert_eq!(store.compactions().history_rows, 1);
            let moved = high.ranges()[0].0;
            assert!(moved < slab_rows);
            assert_eq!(
                store
                    .backing
                    .borrow()
                    .history
                    .as_ref()
                    .unwrap()
                    .region_rows(0, moved as u64, 1)
                    .unwrap()
                    .read_to_host()
                    .unwrap(),
                value
            );
        }
    }

    #[test]
    fn reclaim_compacts_claimed_banks_before_releasing_slabs() {
        let catalog = DeviceCatalog::discover().unwrap();
        for backend in [
            BackendName::Cpu,
            BackendName::Metal,
            BackendName::Cuda,
            BackendName::Vulkan,
        ] {
            let Ok(device) = catalog.open_backend(backend) else {
                continue;
            };
            let device = Rc::new(device);
            let store = StateStore::new(
                device.clone(),
                64,
                256,
                vec![],
                vec![
                    ComponentSpec {
                        shape: vec![2 * 1024 * 1024],
                        dtype: DType::F32,
                    },
                    ComponentSpec {
                        shape: vec![2 * 1024 * 1024],
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
            store.add_bank_slabs(2).unwrap();
            let mut claims = (0..9)
                .map(|_| Some(store.banks.acquire().unwrap()))
                .collect::<Vec<_>>();
            let kept = [2usize, 5, 9];
            for &index in &kept {
                for (plane_index, plane) in store.recurrent_arenas().iter().enumerate() {
                    let bytes = vec![
                        index as u8 + plane_index as u8;
                        plane
                            .slice_leading(index as u64, index as u64 + 1)
                            .unwrap()
                            .byte_len() as usize
                    ];
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
            let placed = || kept.map(|index| claims[index - 1].as_ref().unwrap().index());
            let holds = |slots: [usize; 3]| {
                for (old, slot) in kept.into_iter().zip(slots) {
                    for (plane_index, plane) in store.recurrent_arenas().iter().enumerate() {
                        let bytes = plane
                            .slice_leading(slot as u64, slot as u64 + 1)
                            .unwrap()
                            .read_to_host()
                            .unwrap();
                        assert_eq!(bytes, vec![old as u8 + plane_index as u8; bytes.len()]);
                    }
                }
            };
            let charged = device.memory_usage().charged;
            // Bank 9 lies beyond the four kept banks; lower banks are free.
            let generation = store.bank_placement_generation();
            // Compaction and release use only already held storage.
            device.set_memory_limit(Some(charged));
            let released = store.shrink(ShrinkPolicy::Reclaim).unwrap();
            device.set_memory_limit(None);
            assert!(released > 0);
            assert_eq!(store.committed().1, 4);
            let after = placed();
            assert!(after.iter().all(|&slot| slot < 4));
            assert!(store.bank_placement_generation() > generation);
            assert_eq!(store.compactions().banks, 2);
            holds(after);
            assert_eq!(store.shrink(ShrinkPolicy::Reclaim).unwrap(), 0);
            assert_eq!(store.compactions().count, 1);
            assert_eq!(placed(), after);
            holds(after);
            assert!(device.memory_usage().charged < charged);
        }
    }

    #[test]
    fn reclaim_frees_empty_interior_bank_slab_and_reuses_its_index() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            64,
            64,
            vec![],
            vec![ComponentSpec {
                shape: vec![4 * 1024 * 1024],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 10,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        assert_eq!(store.bank_slab_banks(), 4);
        store.add_bank_slabs(2).unwrap();
        let mut claims = (0..8)
            .map(|_| Some(store.banks.acquire().unwrap()))
            .collect::<Vec<_>>();
        for index in 4..=7 {
            drop(claims[index - 1].take());
        }
        let high = claims[7].as_ref().unwrap();
        let value = vec![23u8; 16 * 1024 * 1024];
        store.recurrent_arenas()[0]
            .slice_leading(high.index() as u64, high.index() as u64 + 1)
            .unwrap()
            .write_from_host(&value)
            .unwrap();
        let slab_bytes = store
            .backing
            .borrow()
            .recurrent
            .as_ref()
            .unwrap()
            .slab_bytes();
        let charged = device.memory_usage().charged;
        device.set_memory_limit(Some(charged));
        assert_eq!(store.shrink(ShrinkPolicy::Reclaim).unwrap(), slab_bytes);
        device.set_memory_limit(None);
        assert_eq!(high.index(), 8);
        assert_eq!(store.available_banks(), 3);
        assert_eq!(
            store.recurrent_arenas()[0]
                .slice_leading(high.index() as u64, high.index() as u64 + 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            value
        );
        let claim = store.growth_claim(&[], 4).unwrap();
        assert_eq!(claim.minimum_bytes, slab_bytes);
        store.provision(&[], 4).unwrap();
        assert!(store
            .backing
            .borrow()
            .recurrent
            .as_ref()
            .unwrap()
            .slab(1)
            .is_some());
        assert_eq!(store.available_banks(), 7);
    }

    #[test]
    fn reclaim_compacts_banks_past_an_unbacked_interior_slab() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            64,
            64,
            vec![],
            vec![ComponentSpec {
                shape: vec![4 * 1024 * 1024],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 10,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        store.add_bank_slabs(2).unwrap();
        let mut claims = (0..8)
            .map(|_| Some(store.banks.acquire().unwrap()))
            .collect::<Vec<_>>();
        let high = claims[7].take().unwrap();
        for claim in claims {
            drop(claim);
        }
        let value = vec![37u8; 16 * 1024 * 1024];
        store.recurrent_arenas()[0]
            .slice_leading(high.index() as u64, high.index() as u64 + 1)
            .unwrap()
            .write_from_host(&value)
            .unwrap();
        let slab_bytes = store
            .backing
            .borrow()
            .recurrent
            .as_ref()
            .unwrap()
            .slab_bytes();
        let charged = device.memory_usage().charged;
        device.set_memory_limit(Some(charged));
        assert_eq!(store.shrink(ShrinkPolicy::Reclaim).unwrap(), 2 * slab_bytes);
        device.set_memory_limit(None);
        assert_eq!(high.index(), 1);
        assert_eq!(store.committed().1, 4);
        assert_eq!(store.compactions().banks, 1);
        assert_eq!(
            store.recurrent_arenas()[0]
                .slice_leading(high.index() as u64, high.index() as u64 + 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            value
        );
    }

    #[test]
    fn reclaim_frees_the_sparsest_bank_slab_without_new_charge() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            64,
            64,
            vec![],
            vec![ComponentSpec {
                shape: vec![4 * 1024 * 1024],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 11,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        store.add_bank_slabs(2).unwrap();
        let mut claims = (0..10)
            .map(|_| Some(store.banks.acquire().unwrap()))
            .collect::<Vec<_>>();
        for index in 5..=7 {
            drop(claims[index - 1].take());
        }
        let sparse = claims[3].as_ref().unwrap();
        let value = vec![47u8; 16 * 1024 * 1024];
        store.recurrent_arenas()[0]
            .slice_leading(sparse.index() as u64, sparse.index() as u64 + 1)
            .unwrap()
            .write_from_host(&value)
            .unwrap();
        let slab_bytes = store
            .backing
            .borrow()
            .recurrent
            .as_ref()
            .unwrap()
            .slab_bytes();
        let charged = device.memory_usage().charged;
        device.set_memory_limit(Some(charged));
        assert_eq!(store.shrink(ShrinkPolicy::Reclaim).unwrap(), slab_bytes);
        device.set_memory_limit(None);
        assert_eq!(sparse.index(), 11);
        assert_eq!(store.compactions().banks, 1);
        assert!(store
            .backing
            .borrow()
            .recurrent
            .as_ref()
            .unwrap()
            .slab(1)
            .is_none());
        assert_eq!(
            store.recurrent_arenas()[0]
                .slice_leading(sparse.index() as u64, sparse.index() as u64 + 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            value
        );
    }

    #[test]
    fn failed_store_copy_preserves_published_bank_and_charge() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device.clone(),
            64,
            64,
            vec![],
            vec![ComponentSpec {
                shape: vec![4 * 1024 * 1024],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 11,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        store.add_bank_slabs(2).unwrap();
        let mut claims = (0..10)
            .map(|_| Some(store.banks.acquire().unwrap()))
            .collect::<Vec<_>>();
        for index in 5..=7 {
            drop(claims[index - 1].take());
        }
        let sparse = claims[3].as_ref().unwrap();
        let before_slot = sparse.index();
        let before_generation = store.bank_placement_generation();
        let before_charge = device.memory_usage().charged;
        let before_stats = store.compactions();
        let failure = store.shrink_with(ShrinkPolicy::Reclaim, |_, plan| {
            assert_eq!(plan.rows(), 1);
            Err(Error::Request("injected bank copy failure".into()))
        });
        assert!(
            matches!(failure, Err(Error::Request(message)) if message == "injected bank copy failure")
        );
        assert_eq!(sparse.index(), before_slot);
        assert_eq!(store.bank_placement_generation(), before_generation);
        assert_eq!(store.compactions(), before_stats);
        assert_eq!(device.memory_usage().charged, before_charge);
        assert!(store
            .backing
            .borrow()
            .recurrent
            .as_ref()
            .unwrap()
            .slab(1)
            .is_some());
    }

    #[test]
    fn later_copy_failure_preserves_both_placements_values_and_charge() {
        let Some(device) = cpu_device() else { return };
        let store = StateStore::new(
            device.clone(),
            8192,
            8192,
            vec![dense_component(4096)],
            vec![ComponentSpec {
                shape: vec![4 * 1024 * 1024],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 11,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        store.add_history_slabs(2).unwrap();
        store.add_bank_slabs(2).unwrap();
        let rows = store.history_slab_rows();
        let _low = claim_rows(&store.arena, 0, rows);
        let sparse = claim_rows(&store.arena, rows, 1);
        let _high = claim_rows(&store.arena, 2 * rows, rows - 1);
        let history_value = vec![41u8; 4096 * 4];
        store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .region_rows(0, rows as u64, 1)
            .unwrap()
            .write_from_host(&history_value)
            .unwrap();
        let mut claims = (0..10)
            .map(|_| Some(store.banks.acquire().unwrap()))
            .collect::<Vec<_>>();
        for index in 5..=7 {
            drop(claims[index - 1].take());
        }
        let bank = claims[3].as_ref().unwrap();
        let bank_value = vec![47u8; 16 * 1024 * 1024];
        write_bank(&store, bank.index(), &bank_value);
        let before_charge = device.memory_usage().charged;
        let before_ranges = sparse.ranges();
        let before_slot = bank.index();
        let before_generation = store.bank_placement_generation();
        let before_stats = store.compactions();
        let before_history_slabs = store
            .backing
            .borrow()
            .history
            .as_ref()
            .unwrap()
            .slabs()
            .count();
        let before_bank_slabs = store
            .backing
            .borrow()
            .recurrent
            .as_ref()
            .unwrap()
            .slabs()
            .count();
        let mut copies = 0;
        let failure = store.shrink_with(ShrinkPolicy::Reclaim, |slabs, plan| {
            copies += 1;
            if copies == 2 {
                return Err(Error::Request("injected later copy failure".into()));
            }
            for part in plan.copies() {
                for (&from, &to) in part.from.iter().zip(&part.to) {
                    let source = slabs.region_rows(part.plane_index, from as u64, 1)?;
                    let mut destination = slabs.region_rows(part.plane_index, to as u64, 1)?;
                    destination.write_from_host(
                        &source
                            .read_to_host()
                            .map_err(|error| Error::Request(error.to_string()))?,
                    )?;
                }
            }
            Ok(())
        });
        assert!(
            matches!(failure, Err(Error::Request(message)) if message == "injected later copy failure")
        );
        assert_eq!(copies, 2);
        assert_eq!(sparse.ranges(), before_ranges);
        assert_eq!(bank.index(), before_slot);
        assert_eq!(store.bank_placement_generation(), before_generation);
        assert_eq!(store.compactions(), before_stats);
        assert_eq!(
            store
                .backing
                .borrow()
                .history
                .as_ref()
                .unwrap()
                .slabs()
                .count(),
            before_history_slabs
        );
        assert_eq!(
            store
                .backing
                .borrow()
                .recurrent
                .as_ref()
                .unwrap()
                .slabs()
                .count(),
            before_bank_slabs
        );
        assert_eq!(device.memory_usage().charged, before_charge);
        assert_eq!(
            store
                .backing
                .borrow()
                .history
                .as_ref()
                .unwrap()
                .region_rows(0, rows as u64, 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            history_value
        );
        assert_eq!(bank_bytes(&store, before_slot), bank_value);
    }

    struct Interleaved {
        segments: usize,
        decode_steps: usize,
        peak_committed: usize,
        released_to: usize,
        written: usize,
    }

    #[test]
    fn interleaved_decode_respects_span_bound_with_slab_backing() {
        // Reservations of 8 to 17 contexts, down to exactly one context per
        // request. Histories may continue in another span when their slab
        // has no adjacent free rows.
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
                "interleaved active={active} contexts={contexts}: segments={} \
                 written={} peak_committed={} of {reserved}",
                run.segments, run.written, run.peak_committed
            );
            assert!(
                run.decode_steps > 100,
                "requests ran {} decode steps",
                run.decode_steps
            );
            assert!(
                run.segments <= 17,
                "{active} requests in {contexts} contexts reached {} segments",
                run.segments
            );
            // A small logical reservation fits within one physical slab.
            assert!(
                run.peak_committed <= reserved,
                "{active} requests committed {} of {reserved} rows",
                run.peak_committed
            );
            assert!(run.released_to <= reserved);
        }
    }
}
