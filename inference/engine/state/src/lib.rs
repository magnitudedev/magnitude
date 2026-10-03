//! Family-neutral accepted sequence state and transactional history ownership.
//! Owned transactions can cross a submission boundary. Their callers reconcile
//! only after physical completion has been observed.
mod advance;
mod bank;
mod codec;
mod domain;
mod layout;
mod pipeline;
mod stage;
pub mod placement;

pub use advance::{
    CodecConversionStep, OwnedAdvanceBindings, OwnedAdvanceResolution, OwnedCodecAdvance,
    OwnedCodecBindings, OwnedStateAdvance, OwnedSuccessorAdvance, OwnedTailRelocation,
    TentativeAdvance,
};
pub use pipeline::{PipelineSequenceState, PipelineStateAdvance, PipelineStateRefusal};
pub use stage::{GlobalLayerId, StageLayerOrdinal, StageModelView};
pub use bank::{recurrent_bank_bytes, BankComponent};
pub use domain::{
    HistoryDomainId, HistoryDomainKind, HistoryDomainLayout, HistoryDomainPlan, HistorySource,
};

pub use codec::{
    Codec, CodecIdentity, CodecSpec, ComponentDescriptor, KvCodec, LayerRef, LayoutError,
    PlaneDescriptor, PlaneName, VectorKind, AFFINE_GROUP,
};
pub use layout::{
    banks_per_slab, history_geometry, HistoryGeometry, ModelStateLayout, MAX_HISTORY_SPANS,
    SLAB_BYTE_TARGET, SLAB_ROW_TILE,
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
    Capacity {
        required: u64,
        available_bytes: u64,
    },
    BanksExhausted {
        capacity: usize,
    },
    Placement(PlacementError),
    UnsupportedHistoryDomain(HistoryDomainKind),
    UnsupportedBankComponent(BankComponent),
    /// A caller whose launch format carries one history domain met a store
    /// with another number of them.
    HistoryDomains {
        count: usize,
    },
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
            Self::UnsupportedHistoryDomain(kind) => {
                write!(f, "history domain {kind:?} is not supported")
            }
            Self::UnsupportedBankComponent(component) => {
                write!(f, "recurrent bank component {component:?} is not supported")
            }
            Self::HistoryDomains { count } => {
                write!(
                    f,
                    "the store has {count} history domains where one is required"
                )
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

/// One logical attention-history component view over its domain's slab
/// table. `plane_index` numbers the planes of every stored domain of the
/// store in order (the index into [`StateStore::history_planes`]);
/// `component_index` is the component within its domain. `base_row` is
/// always zero: the plane's rows are its domain's rows.
#[derive(Clone)]
pub struct PlaneBuffer {
    pub plane_index: usize,
    pub domain: HistoryDomainId,
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
/// Rows are placed in pages of `page_rows`, a whole number per slab and per
/// domain capacity, so every page is complete. A history takes rows
/// only in place after its end within its last page, or as whole free pages
/// (pages without a referenced row), so every page it references is
/// complete except its first and last, and its spans stay within its
/// domain's span limit however requests interleave, fork or are reclaimed
/// (see [`history_geometry`]). The rows after a history's end in its last
/// page are never given to another history: its page holds a referenced
/// row. A history whose last page is partial but whose next row another
/// history took (a sibling fork) relocates that page's rows before it grows
/// ([`OwnedTailRelocation`]). A history takes the free page that begins at
/// its end when there is one; otherwise, as a fresh history does, the middle
/// page of the largest run of free pages, leaving the pages before it for
/// the history that ends there (a run at row 0 has none and is taken from
/// its start).
///
/// A row is referenced once by every history ([`Claims`]) covering it, so a
/// prefix is shared by any number of sequences and checkpoints at row
/// granularity, and a row returns to the free holes only when its last
/// history drops it. Every history is registered, so compaction can move
/// referenced rows and rewrite every affected history.
#[derive(Clone)]
struct Arena {
    slab_rows: usize,
    page_rows: usize,
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
    /// An arena whose first `rows` (whole pages) are backed.
    fn new(rows: usize, slab_rows: usize, page_rows: usize) -> Self {
        assert!(page_rows > 0 && slab_rows % page_rows == 0 && rows % page_rows == 0);
        let mut free = Vec::new();
        append_ranges(&mut free, [(0, rows)], slab_rows);
        Self {
            slab_rows,
            page_rows,
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

    /// One past the last row of the page holding `row`.
    fn page_end(&self, row: usize) -> usize {
        row / self.page_rows * self.page_rows + self.page_rows
    }

    /// The start of every free page, in address order: pages without a
    /// referenced row. A page lies within one slab, so within one hole.
    fn free_pages(&self) -> Vec<usize> {
        let mut pages = Vec::new();
        for &(start, count) in &self.free {
            let end = start + count;
            let mut page = start.next_multiple_of(self.page_rows);
            while page < end && self.page_end(page) <= end {
                pages.push(page);
                page += self.page_rows;
            }
        }
        pages
    }

    /// Rows a fresh history can take: the rows of every free page.
    fn available(&self) -> usize {
        self.free_pages().len() * self.page_rows
    }

    /// Rows a history ending at `end` can take in place: the free rows that
    /// begin at its end, up to the end of its last page.
    fn in_place(&self, end: Option<usize>) -> usize {
        let Some(end) = end.filter(|end| end % self.page_rows != 0) else {
            return 0;
        };
        self.free
            .iter()
            .find(|(start, _)| *start == end)
            .map_or(0, |&(start, count)| count.min(self.page_end(start) - start))
    }

    /// Whether a history ending at `end` must relocate its last page before
    /// it grows: the page has rows after its end, but another history took
    /// the next one.
    fn tail_blocked(&self, end: Option<usize>) -> bool {
        end.is_some_and(|end| {
            end % self.page_rows != 0
                && end < self.page_end(end - 1)
                && self.in_place(Some(end)) == 0
        })
    }

    /// Rows a history ending at `end` can take (see [`Arena::claim`]).
    fn claimable(&self, end: Option<usize>) -> usize {
        self.in_place(end) + self.available()
    }

    /// The free page a history ending at `end` takes next, given the free
    /// pages (see the placement rule above).
    fn next_page(&self, end: Option<usize>, pages: &[usize]) -> usize {
        if let Some(end) = end {
            if pages.binary_search(&end).is_ok() {
                return end;
            }
        }
        let mut largest: Option<(usize, usize)> = None;
        let mut first = 0;
        while first < pages.len() {
            let mut next = first + 1;
            while next < pages.len()
                && pages[next] == pages[next - 1] + self.page_rows
                && pages[next] / self.slab_rows == pages[first] / self.slab_rows
            {
                next += 1;
            }
            if largest.is_none_or(|(_, count)| next - first > count) {
                largest = Some((first, next - first));
            }
            first = next;
        }
        let (first, count) = largest.expect("claimed rows never exceed the available rows");
        if pages[first] == 0 {
            pages[first]
        } else {
            pages[first + (count - 1) / 2]
        }
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
        // Whole occupied pages move to free pages of kept slabs at the same
        // offsets, so every history keeps its page structure (and its span
        // limit), including the free rows after a history's end.
        let mut destinations = self
            .free_pages()
            .into_iter()
            .filter(|page| keep.contains(&(page / self.slab_rows)));
        let mut moved_pages = BTreeMap::new();
        let mut placed = BTreeMap::new();
        for (&start, run) in &self.runs {
            if keep.contains(&(start / self.slab_rows)) {
                continue;
            }
            let (mut row, end) = (start, start + run.count);
            while row < end {
                let page = row / self.page_rows * self.page_rows;
                let to = *moved_pages.entry(page).or_insert_with(|| {
                    destinations
                        .next()
                        .expect("kept slabs have a free page for every moved page")
                });
                let piece = end.min(self.page_end(row)) - row;
                placed.insert(row, (piece, to + row - page));
                row += piece;
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

    /// Claim `count` rows, in logical order, for a history whose end is row
    /// `after`: in place after its end within its last page, then whole free
    /// pages. The caller has checked that its tail is not blocked and that
    /// `count <= self.claimable(after)`.
    fn claim(&mut self, after: Option<usize>, count: usize) -> Vec<(usize, usize)> {
        debug_assert!(!self.tail_blocked(after));
        let mut claimed = Vec::new();
        let mut remaining = count;
        let mut end = after;
        while remaining > 0 {
            let in_place = self.in_place(end);
            let (start, taken) = if in_place > 0 {
                (end.expect("in-place rows follow a history end"), in_place)
            } else {
                (self.next_page(end, &self.free_pages()), self.page_rows)
            };
            let hole = self
                .free
                .iter()
                .position(|&(hole, size)| hole <= start && start < hole + size)
                .expect("the rows taken are free");
            let piece = self.take(hole, start - self.free[hole].0, taken.min(remaining));
            claimed.push(piece);
            end = Some(piece.0 + piece.1);
            remaining -= piece.1;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateAllocationTrace {
    pub context_capacity: usize,
    pub history: Vec<HistoryDomainTrace>,
    pub bank_capacity: BankCapacity,
    pub recurrent_bank_bytes: u64,
    pub zero_seed_bytes: u64,
    pub recurrent_pool_bytes: u64,
}

/// One stored history domain as allocated: `capacity` reserved rows of
/// `row_bytes`, its slab and page rows, and its span limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryDomainTrace {
    pub kind: HistoryDomainKind,
    pub capacity: usize,
    pub row_bytes: u64,
    pub bytes: u64,
    pub slab_rows: usize,
    pub page_rows: usize,
    pub span_limit: usize,
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

/// Slab-backed history rows (one slab tensor per stored domain) and
/// recurrent banks. Graphs are sealed over the logical shapes; only backed
/// rows and banks are handed out.
struct StoreSlabs {
    history: Vec<HistorySlabs>,
    recurrent: Option<SlabTensor>,
    banks: usize,
}

/// One stored domain's history slab tensor and one past its highest backed
/// row.
struct HistorySlabs {
    slabs: SlabTensor,
    rows: usize,
}

/// A registered history claim: its domain and its entry in that domain's
/// arena.
type HistoryKey = (usize, u64);

/// The history and bank claims that transactions (advances, relocations,
/// conversions) hold on this store's rows and banks: shared ownership, not a
/// right to change slab bindings ([`StoreBindings`]).
#[derive(Default)]
struct TransactionClaims {
    histories: BTreeMap<HistoryKey, usize>,
    banks: BTreeMap<LogicalId, usize>,
}

#[derive(Clone)]
struct Transactions(Rc<RefCell<TransactionClaims>>);

struct Transaction {
    claims: Rc<RefCell<TransactionClaims>>,
    histories: Vec<HistoryKey>,
    banks: Vec<LogicalId>,
}

impl Transactions {
    fn begin(&self) -> Transaction {
        Transaction {
            claims: self.0.clone(),
            histories: Vec::new(),
            banks: Vec::new(),
        }
    }
}

impl Transaction {
    /// Track one claim per stored domain, in domain order, and a bank.
    fn track(&mut self, claims: &[Claims], bank: &BankHandle) {
        for (domain, claims) in claims.iter().enumerate() {
            self.track_history(HistoryDomainId(domain), claims);
        }
        self.track_bank(bank);
    }
    fn track_history(&mut self, domain: HistoryDomainId, claims: &Claims) {
        let key = (domain.0, claims.id);
        self.histories.push(key);
        *self.claims.borrow_mut().histories.entry(key).or_default() += 1;
    }
    fn track_bank(&mut self, bank: &BankHandle) {
        self.banks.push(bank.id());
        *self.claims.borrow_mut().banks.entry(bank.id()).or_default() += 1;
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        let mut tracked = self.claims.borrow_mut();
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
/// One stored history domain of a store: its layout, placement geometry and
/// the arena of its rows (free space and per-row references).
struct HistoryDomain {
    kind: HistoryDomainKind,
    components: Vec<ComponentDescriptor>,
    row_bytes: u64,
    slab_rows: usize,
    page_rows: usize,
    /// Row addresses reserved for the domain and sealed into graphs.
    capacity: usize,
    span_limit: usize,
    arena: Rc<RefCell<Arena>>,
}

impl HistoryDomain {
    /// Whole slabs to add, lowest unbacked indices first, so that the free
    /// pages hold a demand of `rows` page-rounded rows (see
    /// [`StateStore::page_demand`]).
    fn growth_slabs(&self, rows: usize) -> Result<usize, Error> {
        if rows == 0 {
            return Ok(0);
        }
        let arena = self.arena.borrow();
        let pages = rows.div_ceil(self.page_rows);
        let free = arena.free_pages().len();
        let shortage = pages.saturating_sub(free);
        let mut additional_slabs = 0;
        let mut added_pages = 0;
        for slab in 0..self.capacity.div_ceil(self.slab_rows) {
            if !arena.backed.contains(&slab) && added_pages < shortage {
                let start = slab * self.slab_rows;
                added_pages += ((start + self.slab_rows).min(self.capacity) - start) / self.page_rows;
                additional_slabs += 1;
            }
        }
        if added_pages < shortage {
            return Err(Error::Capacity {
                required: rows as u64 * self.row_bytes,
                available_bytes: ((added_pages + free) * self.page_rows) as u64 * self.row_bytes,
            });
        }
        Ok(additional_slabs)
    }

    /// One past the highest row of the backed slabs.
    fn backed_extent(&self, slabs: &SlabTensor) -> usize {
        slabs
            .slabs()
            .map(|(index, _)| ((index + 1) * self.slab_rows).min(self.capacity))
            .max()
            .unwrap_or(0)
    }

    /// The first position a history at `position`, trimmed at `floor`,
    /// references.
    fn history_start(&self, position: usize, floor: usize) -> usize {
        self.kind.retained_from(position).max(floor)
    }

    /// Plan one shrink of this domain against its published arena: the
    /// slabs kept (idle: every occupied slab and one spare; reclaim: the
    /// most occupied slabs whose free whole pages hold every occupied page
    /// of the others), the page moves out of the others, and the arena to
    /// publish once they are copied.
    fn shrink_plan(&self, policy: ShrinkPolicy, rows: usize) -> ShrinkPlan {
        let arena = self.arena.borrow();
        let occupied_pages = arena
            .runs
            .iter()
            .flat_map(|(&start, run)| {
                (start / self.page_rows..=(start + run.count - 1) / self.page_rows)
                    .map(|page| page * self.page_rows)
            })
            .collect::<BTreeSet<_>>();
        let free_pages = arena.free_pages();
        // (slab, occupied pages, free whole pages)
        let mut slabs = arena
            .backed
            .iter()
            .map(|&index| {
                let in_slab = |page: &&usize| **page / self.slab_rows == index;
                let free = free_pages.iter().filter(in_slab).count();
                (index, occupied_pages.iter().filter(in_slab).count(), free)
            })
            .collect::<Vec<_>>();
        slabs.sort_unstable_by_key(|&(index, occupied, _)| (std::cmp::Reverse(occupied), index));
        let mut keep = BTreeSet::new();
        match policy {
            ShrinkPolicy::Idle => {
                keep.extend(
                    slabs
                        .iter()
                        .filter(|(_, occupied, _)| *occupied > 0)
                        .map(|(index, _, _)| *index),
                );
                if let Some(&(index, _, _)) = slabs.iter().find(|(_, occupied, _)| *occupied == 0) {
                    keep.insert(index);
                }
            }
            ShrinkPolicy::Reclaim => {
                let mut moved = occupied_pages.len();
                let mut room = 0;
                for &(index, occupied, free) in &slabs {
                    if room >= moved {
                        break;
                    }
                    keep.insert(index);
                    moved -= occupied;
                    room += free;
                }
            }
        }
        let (arena, moves) = arena.compact_into_slabs(rows, &keep);
        ShrinkPlan { arena, moves, keep }
    }
}

/// One domain's planned shrink (see [`HistoryDomain::shrink_plan`]).
struct ShrinkPlan {
    arena: Arena,
    moves: Vec<(usize, usize, usize)>,
    keep: BTreeSet<usize>,
}

/// Each stored history domain's rows are a shared arena; recurrent
/// components are arenas of banks, and each accepted version is one
/// immutable bank. A checkpoint retains both without copying tensor
/// contents.
///
/// Each domain's reserved rows and the bank capacity are reservations
/// sealed into graphs. Backing slabs are added as demand grows and released
/// when empty, only through the store's [`StoreBindings`].
pub struct StateStore {
    device: Rc<Device>,
    context_capacity: usize,
    domains: Vec<HistoryDomain>,
    /// Layers of Shared domains and the stored layers they read.
    shared: Vec<(LayerRef, HistorySource)>,
    bank_capacity: BankCapacity,
    recurrent_bank_bytes: u64,
    bank_slab_banks: usize,
    component_specs: Vec<ComponentSpec>,
    backing: RefCell<StoreSlabs>,
    history_views: RefCell<Option<Vec<PlaneBuffer>>>,
    recurrent_views: RefCell<Option<Rc<[Tensor]>>>,
    retired_storage: RefCell<Vec<TensorStorageObserver>>,
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

/// How [`StoreBindings::shrink_with`] releases slab backing. Idle releases empty
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

/// One store's demand for `rows` free rows in one stored history domain.
/// An advance of `n` rows demands `n` rows in every domain of its store
/// ([`SequenceState::demands`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowDemand {
    pub domain: HistoryDomainId,
    pub rows: usize,
}

/// The one right to change a store's slab bindings: adding slabs, moving
/// rows and banks, and releasing slabs. [`StateStore::new`] hands out exactly
/// one; it is never cloned. Row and bank claims (advances, relocations,
/// checkpoints) need no binding right.
pub struct StoreBindings {
    store: Rc<StateStore>,
}

impl std::ops::Deref for StoreBindings {
    type Target = Rc<StateStore>;
    fn deref(&self) -> &Rc<StateStore> {
        &self.store
    }
}

impl StoreBindings {
    /// Add backing slabs so the free rows hold `demands` and `banks` banks
    /// are free. A refused allocation returns a capacity error without
    /// publishing tentative backing or changing accepted state.
    pub fn provision(&mut self, demands: &[RowDemand], banks: usize) -> Result<(), Error> {
        self.store
            .provision_with_growth(demands, banks, GrowthChoice::Preferred)
    }

    pub fn provision_with_growth(
        &mut self,
        demands: &[RowDemand],
        banks: usize,
        choice: GrowthChoice,
    ) -> Result<(), Error> {
        self.store.provision_with_growth(demands, banks, choice)
    }

    /// Release empty history and bank slabs, retaining one spare of each at
    /// idle. Reclaim first moves referenced rows and claimed banks from
    /// sparse slabs into free slots of the retained ones through `copy`.
    /// Returns the physical bytes released, measured by the device ledger.
    pub fn shrink_with<E, F>(&mut self, policy: ShrinkPolicy, copy: F) -> Result<u64, E>
    where
        E: From<Error>,
        F: FnMut(&SlabTensor, StoreCopy) -> Result<(), E>,
    {
        self.store.shrink_with(policy, copy)
    }

    /// Drop every slab once no sequence or checkpoint owns the store.
    pub fn release_idle(&mut self) -> Result<usize, Error> {
        self.store.release_idle()
    }

    #[cfg(test)]
    fn shrink(&mut self, policy: ShrinkPolicy) -> Result<u64, Error> {
        self.store.shrink_on_host(policy)
    }
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

impl StateStore {
    /// The store backing belongs to this exact opened execution device.
    pub fn belongs_to_device(&self, device: &Device) -> bool {
        self.device.same_device(device)
    }

    /// A store of the given history domains and recurrent bank components.
    /// Every domain's reserved rows cover its row limit: the context for
    /// Token, `n` plus one advance of at most `max_advance` rows for
    /// Window(n). Each stored domain starts with one backed slab. The store's
    /// one [`StoreBindings`] comes with it.
    pub fn new(
        device: Rc<Device>,
        context_capacity: usize,
        max_advance: usize,
        domains: Vec<HistoryDomainPlan>,
        component_specs: Vec<ComponentSpec>,
        bank_capacity: BankCapacity,
    ) -> Result<(Rc<Self>, StoreBindings), Error> {
        if context_capacity == 0 {
            return Err(Error::Request(
                "history capacity must fit a positive sequence context".into(),
            ));
        }
        let (stored, shared) = domain::validate_domains(domains, context_capacity, max_advance)?;
        let recurrent_bank_bytes = recurrent_bank_bytes(&component_specs)?;
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
        let mut history = Vec::with_capacity(stored.len());
        let mut domains = Vec::with_capacity(stored.len());
        for domain in stored {
            let geometry = domain.geometry;
            let mut slabs = SlabTensor::new(
                &device,
                geometry.slab_rows as u64,
                domain.logical_rows as u64,
                domain
                    .components
                    .iter()
                    .flat_map(ComponentDescriptor::planes)
                    .map(|plane| SlabRegion {
                        element: element(plane.dtype),
                        row_shape: plane.row_extents.iter().map(|&size| size as u64).collect(),
                    })
                    .collect(),
            )?;
            slabs.add_slab()?;
            let rows = geometry.slab_rows.min(domain.logical_rows);
            history.push(HistorySlabs { slabs, rows });
            domains.push(HistoryDomain {
                kind: domain.kind,
                components: domain.components,
                row_bytes: domain.row_bytes,
                slab_rows: geometry.slab_rows,
                page_rows: geometry.page_rows,
                capacity: domain.logical_rows,
                span_limit: geometry.span_limit,
                arena: Rc::new(RefCell::new(Arena::new(
                    rows,
                    geometry.slab_rows,
                    geometry.page_rows,
                ))),
            });
        }
        let store = Rc::new(Self {
            device,
            context_capacity,
            domains,
            shared,
            bank_capacity,
            recurrent_bank_bytes,
            bank_slab_banks,
            component_specs,
            backing: RefCell::new(StoreSlabs {
                history,
                recurrent,
                banks: committed_banks,
            }),
            history_views: RefCell::new(None),
            recurrent_views: RefCell::new(None),
            retired_storage: RefCell::new(Vec::new()),
            banks,
            zero_seed,
            owners: Cell::new(0),
            sequences: Cell::new(0),
            transactions: Transactions(Rc::new(RefCell::new(TransactionClaims::default()))),
            compactions: Cell::new(Compactions::default()),
        });
        Ok((
            store.clone(),
            StoreBindings { store },
        ))
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
    /// One past the highest backed row of a domain. The extent may include
    /// unbacked rows after an interior slab is freed;
    /// [`StateStore::committed_bytes`] measures the actual backing.
    pub fn committed_rows(&self, domain: HistoryDomainId) -> usize {
        self.backing.borrow().history[domain.0].rows
    }
    /// One past the highest backed recurrent bank.
    pub fn committed_banks(&self) -> usize {
        self.backing.borrow().banks
    }
    /// Physical bytes of the committed backing.
    pub fn committed_bytes(&self) -> u64 {
        let backing = self.backing.borrow();
        backing
            .history
            .iter()
            .map(|history| history.slabs.storage_bytes())
            .sum::<u64>()
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
            .flat_map(|history| history.slabs.slab_storage_observers())
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
        let keys = |holder: &Holder<'_>| {
            holder
                .claims()
                .iter()
                .enumerate()
                .map(|(domain, claims)| (domain, claims.id))
                .collect::<Vec<HistoryKey>>()
        };
        let mut supplied = BTreeSet::new();
        let mut banks = BTreeSet::new();
        for holder in &holders {
            if !Rc::ptr_eq(self, holder.store()) {
                return Err(Error::Request(
                    "state census holder belongs to another store".into(),
                ));
            }
            for key in keys(holder) {
                if !supplied.insert(key) {
                    return Err(Error::Request(
                        "state census holder appears more than once".into(),
                    ));
                }
            }
            banks.insert(holder.bank().id());
        }
        let tracked = self.transactions.0.borrow();
        if live
            .iter()
            .chain(retained)
            .flat_map(keys)
            .any(|key| tracked.histories.contains_key(&key))
        {
            return Err(Error::Request(
                "state census marks a submitted history as live or retained".into(),
            ));
        }
        let mut occupied_history = 0u64;
        for (index, domain) in self.domains.iter().enumerate() {
            let arena = domain.arena.borrow();
            if arena.entries.keys().any(|&id| {
                !supplied.contains(&(index, id)) && !tracked.histories.contains_key(&(index, id))
            }) {
                return Err(Error::Request(
                    "state census omits a registered history claim".into(),
                ));
            }
            if self.backing.borrow().history[index].rows < arena.referenced {
                return Err(Error::Request(
                    "state census does not reconcile to committed backing".into(),
                ));
            }
            occupied_history = u64::try_from(arena.referenced)
                .ok()
                .and_then(|rows| rows.checked_mul(domain.row_bytes))
                .and_then(|bytes| occupied_history.checked_add(bytes))
                .ok_or_else(|| Error::Request("state census byte count overflow".into()))?;
        }
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
        let used_banks = u64::try_from(claimed_banks.len())
            .map_err(|_| Error::Request("occupied bank count exceeds u64".into()))?;
        let occupied = used_banks
            .checked_mul(self.recurrent_bank_bytes)
            .and_then(|bank| occupied_history.checked_add(bank))
            .ok_or_else(|| Error::Request("state census byte count overflow".into()))?;
        let model_seed = self.recurrent_bank_bytes;
        let committed = self.committed_bytes();
        let surplus = committed
            .checked_sub(occupied)
            .and_then(|bytes| bytes.checked_sub(model_seed))
            .ok_or_else(|| Error::Request("state census exceeds Seismic charged backing".into()))?;
        drop(placement);
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
        if census.total() != committed {
            return Err(Error::Request(
                "state census does not reconcile to committed backing".into(),
            ));
        }
        Ok(census)
    }
    /// The store's stored history domains, in plan order.
    pub fn history_domains(&self) -> impl Iterator<Item = HistoryDomainId> {
        (0..self.domains.len()).map(HistoryDomainId)
    }
    /// The store's one stored history domain, for a caller whose launch
    /// format carries exactly one.
    pub fn sole_history_domain(&self) -> Result<HistoryDomainId, Error> {
        match self.domains.len() {
            1 => Ok(HistoryDomainId(0)),
            count => Err(Error::HistoryDomains { count }),
        }
    }
    pub fn history_domain_kind(&self, domain: HistoryDomainId) -> HistoryDomainKind {
        self.domains[domain.0].kind
    }
    /// Where `layer`'s history lives: its own stored domain, or the source
    /// layer and domain a Shared domain binds. `None` for a layer without
    /// history in this store.
    pub fn history_source(&self, layer: LayerRef) -> Option<HistorySource> {
        self.domains
            .iter()
            .position(|domain| {
                domain
                    .components
                    .iter()
                    .any(|component| component.layer == layer)
            })
            .map(|domain| HistorySource {
                domain: HistoryDomainId(domain),
                layer,
            })
            .or_else(|| {
                self.shared
                    .iter()
                    .find(|(shared, _)| *shared == layer)
                    .map(|(_, source)| *source)
            })
    }
    /// The stored domains Shared layers read, ascending. A Shared layer
    /// appends nothing, so it reads its source domain's accepted rows and
    /// the rows its source appended for the advance: one read beyond the
    /// stored domains per such source domain.
    pub fn shared_source_domains(&self) -> Vec<HistoryDomainId> {
        let mut domains = self
            .shared
            .iter()
            .map(|(_, source)| source.domain)
            .collect::<Vec<_>>();
        domains.sort_unstable_by_key(|domain| domain.0);
        domains.dedup();
        domains
    }
    /// The history reads of a launch row, in order: every stored domain,
    /// then every Shared source domain (`shared_source_domains`).
    pub fn history_reads(&self) -> usize {
        self.domains.len() + self.shared_source_domains().len()
    }
    /// The history read `layer`'s attention takes (an index into
    /// `history_reads`): its own stored domain's, or its Shared source
    /// domain's. `None` for a layer without history in this store.
    pub fn history_read(&self, layer: LayerRef) -> Option<usize> {
        match self.shared.iter().find(|(shared, _)| *shared == layer) {
            Some((_, source)) => self
                .shared_source_domains()
                .iter()
                .position(|domain| *domain == source.domain)
                .map(|position| self.domains.len() + position),
            None => self.history_source(layer).map(|source| source.domain.0),
        }
    }
    /// Row addresses reserved for a domain: an exclusive bound on its rows.
    pub fn history_capacity(&self, domain: HistoryDomainId) -> usize {
        self.domains[domain.0].capacity
    }
    pub fn history_components(&self, domain: HistoryDomainId) -> &[ComponentDescriptor] {
        &self.domains[domain.0].components
    }
    pub fn history_row_bytes(&self, domain: HistoryDomainId) -> u64 {
        self.domains[domain.0].row_bytes
    }
    pub fn history_slab_rows(&self, domain: HistoryDomainId) -> usize {
        self.domains[domain.0].slab_rows
    }
    pub fn history_page_rows(&self, domain: HistoryDomainId) -> usize {
        self.domains[domain.0].page_rows
    }
    /// The most spans a history of the domain (or a Shared reader of it)
    /// presents to one launch; placement guarantees it (see
    /// [`history_geometry`]).
    pub fn span_limit(&self, domain: HistoryDomainId) -> usize {
        self.domains[domain.0].span_limit
    }
    /// The most spans any history of the store presents to one launch: the
    /// span class bound graphs are sealed to (1 without history).
    pub fn max_span_limit(&self) -> usize {
        self.domains
            .iter()
            .map(|domain| domain.span_limit)
            .max()
            .unwrap_or(1)
    }
    pub fn bank_slab_banks(&self) -> usize {
        self.bank_slab_banks
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
            history: self
                .domains
                .iter()
                .map(|domain| HistoryDomainTrace {
                    kind: domain.kind,
                    capacity: domain.capacity,
                    row_bytes: domain.row_bytes,
                    // Validated not to overflow at construction.
                    bytes: domain.row_bytes * domain.capacity as u64,
                    slab_rows: domain.slab_rows,
                    page_rows: domain.page_rows,
                    span_limit: domain.span_limit,
                })
                .collect(),
            bank_capacity: self.bank_capacity,
            recurrent_bank_bytes: self.recurrent_bank_bytes,
            zero_seed_bytes,
            recurrent_pool_bytes,
        })
    }
    pub fn history_allocated(&self) -> bool {
        self.backing
            .borrow()
            .history
            .iter()
            .any(|history| history.rows != 0)
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
        let plans = self.history_growth_plans(demands, choice)?;
        let backing = self.backing.borrow();
        let bank_slabs = self.bank_growth_slabs(banks)?;
        let history_bytes = plans
            .iter()
            .zip(&backing.history)
            .try_fold(0u64, |total, (&slabs, history)| {
                history
                    .slabs
                    .slab_bytes()
                    .checked_mul(slabs as u64)
                    .and_then(|bytes| total.checked_add(bytes))
            })
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

    fn provision_with_growth(
        &self,
        demands: &[RowDemand],
        banks: usize,
        choice: GrowthChoice,
    ) -> Result<(), Error> {
        let bank_slabs = self.bank_growth_slabs(banks)?;
        let plans = self.history_growth_plans(demands, choice)?;
        // Growth of every domain and the banks is one fallible operation.
        let before = self
            .domains
            .iter()
            .map(|domain| domain.arena.borrow().backed.clone())
            .collect::<Vec<_>>();
        for (index, &slabs) in plans.iter().enumerate() {
            if let Err(error) = self.add_history_slabs(HistoryDomainId(index), slabs) {
                self.rollback_history_growth(&before)?;
                return Err(error);
            }
        }
        if bank_slabs != 0 {
            if let Err(error) = self.add_bank_slabs(bank_slabs) {
                self.rollback_history_growth(&before)?;
                return Err(error);
            }
        }
        Ok(())
    }

    /// Restore every domain's published history backing when a later part
    /// of one aggregate growth fails. No advance has claimed the newly added
    /// rows.
    fn rollback_history_growth(&self, before: &[BTreeSet<usize>]) -> Result<(), Error> {
        for (index, (domain, before)) in self.domains.iter().zip(before).enumerate() {
            let added = domain
                .arena
                .borrow()
                .backed
                .difference(before)
                .copied()
                .collect::<Vec<_>>();
            if added.is_empty() {
                continue;
            }
            self.history_views.borrow_mut().take();
            let mut backing = self.backing.borrow_mut();
            let history = &mut backing.history[index];
            for slab in added {
                history.slabs.free_slab(slab)?;
                domain.arena.borrow_mut().unback_empty_slab(slab);
            }
            history.rows = domain.backed_extent(&history.slabs);
        }
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

    /// Plan whole slab additions, per stored domain, for the rows a launch
    /// needs beyond each domain's free rows. A history may continue in
    /// another span.
    fn history_growth_plans(
        &self,
        demands: &[RowDemand],
        _choice: GrowthChoice,
    ) -> Result<Vec<usize>, Error> {
        let mut plans = Vec::with_capacity(self.domains.len());
        for (index, domain) in self.domains.iter().enumerate() {
            let total = demands
                .iter()
                .filter(|demand| demand.domain.0 == index)
                .map(|demand| demand.rows)
                .sum::<usize>();
            plans.push(domain.growth_slabs(total)?);
        }
        if let Some(demand) = demands
            .iter()
            .find(|demand| demand.domain.0 >= self.domains.len())
        {
            return Err(Error::Request(format!(
                "row demand names absent history domain {:?}",
                demand.domain
            )));
        }
        Ok(plans)
    }

    /// Release empty history and bank slabs, retaining one spare of each at
    /// idle. Reclaim also moves referenced rows and claimed banks from sparse
    /// slabs into free slots of the retained slabs before releasing them.
    /// Returns the physical bytes released, measured by the device ledger:
    /// an earlier backing a caller still views stays charged.
    fn shrink_with<E, F>(self: &Rc<Self>, policy: ShrinkPolicy, mut copy: F) -> Result<u64, E>
    where
        E: From<Error>,
        F: FnMut(&SlabTensor, StoreCopy) -> Result<(), E>,
    {
        let before = self.device.memory_usage().charged;
        let backing = self.backing.borrow();
        // Each stored domain plans against its own published arena.
        let history_plans = self
            .domains
            .iter()
            .zip(&backing.history)
            .map(|(domain, history)| domain.shrink_plan(policy, history.rows))
            .collect::<Vec<_>>();

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

        // Every plan reads the same published generation. Every destination
        // is free there, so even a later failed copy leaves every domain and
        // the banks readable and no physical slab has been released.
        for (plan, history) in history_plans.iter().zip(&backing.history) {
            if !plan.moves.is_empty() {
                copy(
                    &history.slabs,
                    self.history_copy_plan(&history.slabs, &plan.moves)
                        .map_err(E::from)?,
                )?;
            }
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

        let history_rows = history_plans
            .iter()
            .flat_map(|plan| plan.moves.iter())
            .map(|(_, _, count)| count)
            .sum::<usize>();
        *self.banks.inner.placement.borrow_mut() = planned_banks;
        let mut backing = self.backing.borrow_mut();
        for ((domain, plan), history) in self
            .domains
            .iter()
            .zip(history_plans)
            .zip(&mut backing.history)
        {
            *domain.arena.borrow_mut() = plan.arena;
            self.history_views.borrow_mut().take();
            let victims = history
                .slabs
                .slabs()
                .map(|(index, _)| index)
                .filter(|index| !plan.keep.contains(index))
                .collect::<Vec<_>>();
            for index in victims {
                if let Some(observer) = history
                    .slabs
                    .free_slab(index)
                    .map_err(Error::from)
                    .map_err(E::from)?
                {
                    self.record_retired_storage(observer);
                }
            }
            history.rows = domain.backed_extent(&history.slabs);
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
        if history_rows != 0 || !bank_moves.is_empty() {
            let stats = self.compactions.get();
            self.compactions.set(Compactions {
                count: stats.count + 1,
                history_rows: stats.history_rows + history_rows,
                banks: stats.banks + bank_moves.len(),
            });
        }
        Ok(before.saturating_sub(self.device.memory_usage().charged))
    }

    #[cfg(test)]
    fn shrink_on_host(self: &Rc<Self>, policy: ShrinkPolicy) -> Result<u64, Error> {
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

    /// A copy of rows `from` to rows `to` in one history domain's slabs.
    fn history_row_copy(
        self: &Rc<Self>,
        domain: HistoryDomainId,
        from: Vec<usize>,
        to: Vec<usize>,
    ) -> Result<StoreCopy, Error> {
        let backing = self.backing.borrow();
        StoreCopy::new(self.clone(), &backing.history[domain.0].slabs, from, to)
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

    /// Add whole history slabs to a domain, into its lowest unbacked slots.
    fn add_history_slabs(&self, domain: HistoryDomainId, count: usize) -> Result<(), Error> {
        let mut backing = self.backing.borrow_mut();
        if count == 0 {
            return Ok(());
        }
        let history = &mut backing.history[domain.0];
        let slab = &mut history.slabs;
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
        let domain = &self.domains[domain.0];
        let mut arena = domain.arena.borrow_mut();
        for index in added {
            arena.back_slab(index, domain.capacity);
            history.rows = history
                .rows
                .max(((index + 1) * domain.slab_rows).min(domain.capacity));
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
        if backing.history.iter().all(|history| history.rows == 0) {
            return Ok(Vec::new());
        }
        let mut views = Vec::new();
        for (index, (domain, history)) in self.domains.iter().zip(&backing.history).enumerate() {
            let planes =
                domain
                    .components
                    .iter()
                    .enumerate()
                    .flat_map(|(component_index, component)| {
                        component
                            .planes()
                            .iter()
                            .map(move |plane| (component_index, component.layer, plane))
                    });
            for (region, (component_index, layer, plane)) in planes.enumerate() {
                views.push(PlaneBuffer {
                    plane_index: views.len(),
                    domain: HistoryDomainId(index),
                    component_index,
                    layer,
                    vector: plane.vector,
                    name: plane.name,
                    row_bytes: plane.row_bytes,
                    slab_rows: u32::try_from(domain.slab_rows).expect("slab rows fit u32"),
                    base_row: 0,
                    buffer: history.slabs.logical_region(region)?,
                });
            }
        }
        *self.history_views.borrow_mut() = Some(views.clone());
        Ok(views)
    }
    /// A domain's rows referenced by at least one claim; a shared row counts
    /// once.
    pub fn occupied_rows(&self, domain: HistoryDomainId) -> usize {
        self.domains[domain.0].arena.borrow().referenced
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
        let mut history = 0u64;
        for (index, domain) in self.domains.iter().enumerate() {
            let ranges = distinct
                .iter()
                .flat_map(|holder| holder.claims()[index].ranges())
                .collect::<Vec<_>>();
            let rows = domain.arena.borrow().exclusive_rows(&ranges);
            history = (rows as u64)
                .checked_mul(domain.row_bytes)
                .and_then(|bytes| history.checked_add(bytes))
                .ok_or_else(|| Error::Request("exclusive state byte count overflow".into()))?;
        }
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
        self.recurrent_bank_bytes
            .checked_mul(banks)
            .and_then(|recurrent| history.checked_add(recurrent))
            .ok_or_else(|| Error::Request("exclusive state byte count overflow".into()))
    }
    pub fn idle(&self) -> bool {
        self.owners.get() == 0
    }

    pub fn available_banks(&self) -> usize {
        self.banks.available()
    }
    /// Rows of the free backed pages of one domain: what advances beyond
    /// their histories' last pages can take without growth (see
    /// [`SequenceState::demands`]).
    pub fn free_rows(&self, domain: HistoryDomainId) -> usize {
        self.domains[domain.0].arena.borrow().available()
    }
    /// Drop the store's arena allocations when no sequence/checkpoint owns them.
    /// External completion/buffer pins may still retain physical storage.
    fn release_idle(&self) -> Result<usize, Error> {
        if !self.idle() {
            return Ok(0);
        }
        let before = self.device.memory_usage().charged;
        let mut backing = self.backing.borrow_mut();
        for (domain, history) in self.domains.iter().zip(&mut backing.history) {
            self.history_views.borrow_mut().take();
            let held = history
                .slabs
                .slabs()
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            for index in held {
                if let Some(observer) = history.slabs.free_slab(index)? {
                    domain.arena.borrow_mut().unback_empty_slab(index);
                    self.record_retired_storage(observer);
                }
                history.rows = domain.backed_extent(&history.slabs);
            }
            *domain.arena.borrow_mut() =
                Arena::new(0, domain.slab_rows, domain.page_rows);
        }
        drop(backing);
        usize::try_from(before.saturating_sub(self.device.memory_usage().charged))
            .map_err(|_| Error::Request("reclaimable history bytes exceed host range".into()))
    }
    pub fn create(self: &Rc<Self>) -> Result<SequenceState, Error> {
        let bank = self.zero_seed.clone();
        let claims = self
            .domains
            .iter()
            .map(|domain| {
                let claims = Claims::new(&domain.arena, vec![]);
                claims.set_live(true);
                claims
            })
            .collect::<Vec<_>>();
        self.owners.set(self.owners.get() + 1);
        self.sequences.set(self.sequences.get() + 1);
        Ok(SequenceState {
            store: self.clone(),
            position: 0,
            expected_end: 0,
            history_start: 0,
            starts: vec![0; claims.len()],
            claims,
            bank,
            tape: 0,
        })
    }
    /// Reserve `count` rows in every stored domain, in logical order, each
    /// following that domain's `histories` claim (none for rows of a new
    /// history; see [`Arena`] for placement). One domain's refusal releases
    /// the rows already reserved in the others. Reserving never adds slabs.
    fn reserve(&self, histories: Option<&[Claims]>, count: usize) -> Result<Vec<Claims>, Error> {
        self.reserve_rows(histories, &vec![count; self.domains.len()])
    }

    /// The free-page rows a history of `domain` ending at `end` needs to take
    /// `count` more rows: whatever its last page cannot hold in place,
    /// rounded up to whole pages.
    fn page_demand(&self, domain: HistoryDomainId, end: Option<usize>, count: usize) -> usize {
        let domain = &self.domains[domain.0];
        count
            .saturating_sub(domain.arena.borrow().in_place(end))
            .div_ceil(domain.page_rows)
            * domain.page_rows
    }

    /// Reserve `counts[domain]` rows in every stored domain (see
    /// [`StateStore::reserve`]).
    fn reserve_rows(
        &self,
        histories: Option<&[Claims]>,
        counts: &[usize],
    ) -> Result<Vec<Claims>, Error> {
        let ends = (0..self.domains.len())
            .map(|index| histories.and_then(|histories| histories[index].end()))
            .collect::<Vec<_>>();
        if self
            .domains
            .iter()
            .zip(&ends)
            .any(|(domain, &end)| domain.arena.borrow().tail_blocked(end))
        {
            return Err(Error::Request(
                "a history whose last page another history continued must relocate it before it grows"
                    .into(),
            ));
        }
        let mut reserved = Vec::with_capacity(self.domains.len());
        for ((domain, &count), &after) in self.domains.iter().zip(counts).zip(&ends) {
            let mut arena = domain.arena.borrow_mut();
            let claimable = arena.claimable(after);
            if claimable < count {
                return Err(Error::Capacity {
                    required: count as u64 * domain.row_bytes,
                    available_bytes: claimable as u64 * domain.row_bytes,
                });
            }
            let ranges = arena.claim(after, count);
            drop(arena);
            reserved.push(Claims::new(&domain.arena, ranges));
        }
        Ok(reserved)
    }

    /// A free successor bank. Claims never grow the store: its binding
    /// right's owner provisions banks first.
    fn successor_bank(&self) -> Result<BankHandle, Error> {
        self.banks.acquire()
    }

    fn begin_transaction(&self) -> Transaction {
        self.transactions.begin()
    }

    /// The first `rows` rows (at most a page) of a free page, as a fresh
    /// history takes them: the destination of a relocated last page. `None`
    /// without a free page.
    fn reserve_page(&self, domain: HistoryDomainId, rows: usize) -> Option<Claims> {
        let arena = &self.domains[domain.0].arena;
        if rows == 0 || rows > self.domains[domain.0].page_rows || arena.borrow().available() == 0 {
            return None;
        }
        let ranges = arena.borrow_mut().claim(None, rows);
        Some(Claims::new(arena, ranges))
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
    fn claims(&self) -> &[Claims] {
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

/// The rows of `ranges` (a history whose first row is at logical position
/// `start`) at positions `from` and later.
fn ranges_from(ranges: &[(usize, usize)], start: usize, from: usize) -> Vec<(usize, usize)> {
    split_ranges(ranges, from.saturating_sub(start)).1
}

pub struct SequenceState {
    store: Rc<StateStore>,
    position: usize,
    expected_end: usize,
    /// The trim floor: no domain references rows before it.
    history_start: usize,
    /// Per stored domain, the first position its history references: the
    /// trim floor, or for Window(n) at least `position - n`.
    starts: Vec<usize>,
    /// Per stored domain, claims on exactly the rows `[start, position)`, in
    /// logical order; a live history.
    claims: Vec<Claims>,
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
    /// This sequence's demand for appending `rows`: in every stored domain,
    /// the whole free pages beyond what its last page holds in place, for
    /// provisioning a launch's backing before its advances begin.
    pub fn demands(&self, rows: usize) -> Vec<RowDemand> {
        self.store
            .history_domains()
            .map(|domain| RowDemand {
                domain,
                rows: self
                    .store
                    .page_demand(domain, self.claims[domain.0].end(), rows),
            })
            .collect()
    }
    /// The rows this history references in one domain, in logical order:
    /// positions `[history_start(domain), position)`.
    pub fn history_ranges(&self, domain: HistoryDomainId) -> Vec<(usize, usize)> {
        self.claims[domain.0].ranges()
    }
    /// Every domain's ranges, in domain order.
    pub fn domain_ranges(&self) -> Vec<Vec<(usize, usize)>> {
        self.claims.iter().map(Claims::ranges).collect()
    }
    /// The logical position of the first row this history references in a
    /// domain.
    pub fn history_start(&self, domain: HistoryDomainId) -> usize {
        self.starts[domain.0]
    }
    /// The referenced rows of a domain at positions `from` and later: the
    /// visible rows of a query whose window begins at `from`.
    pub fn visible_ranges(&self, domain: HistoryDomainId, from: usize) -> Vec<(usize, usize)> {
        ranges_from(&self.history_ranges(domain), self.starts[domain.0], from)
    }
    /// The most spans any domain's history has.
    pub fn span_count(&self) -> usize {
        self.claims
            .iter()
            .map(|claims| claims.entry(|entry| entry.ranges.len()))
            .max()
            .unwrap_or(0)
    }
    /// Stop seeing rows before logical position `before` in every domain.
    /// Exactly the trimmed rows lose this sequence's reference.
    pub fn trim_history(&mut self, before: usize) -> Result<(), String> {
        if before < self.history_start || before > self.position {
            return Err("history trim must lie within accepted logical positions".into());
        }
        self.history_start = before;
        self.trim();
        Ok(())
    }

    /// Release each domain's references before its first retained position:
    /// the trim floor, and for Window(n) `position - n`.
    fn trim(&mut self) {
        for ((domain, claims), start) in self
            .store
            .domains
            .iter()
            .zip(&mut self.claims)
            .zip(&mut self.starts)
        {
            let retained = domain.history_start(self.position, self.history_start);
            if retained > *start {
                claims.drop_front(retained - *start);
                *start = retained;
            }
        }
    }

    /// The domains whose partial last page this history must relocate
    /// before it grows, because another history (a sibling fork) continued
    /// that page ([`OwnedTailRelocation`]).
    pub fn blocked_tails(&self) -> Vec<HistoryDomainId> {
        self.store
            .history_domains()
            .filter(|&domain| {
                self.store.domains[domain.0]
                    .arena
                    .borrow()
                    .tail_blocked(self.claims[domain.0].end())
            })
            .collect()
    }

    /// The growth relocating [`SequenceState::blocked_tails`] needs: one
    /// free page per domain.
    pub fn relocation_demands(&self) -> Vec<RowDemand> {
        self.blocked_tails()
            .into_iter()
            .map(|domain| RowDemand {
                domain,
                rows: self.store.history_page_rows(domain),
            })
            .collect()
    }

    /// A checkpoint at this position: it references the same rows (for
    /// Window(n), rows `[position - n, position)`), which forks and resumed
    /// requests share without copying.
    pub fn checkpoint(&self) -> StateCheckpoint {
        self.store.owners.set(self.store.owners.get() + 1);
        StateCheckpoint {
            store: self.store.clone(),
            position: self.position,
            history_start: self.history_start,
            starts: self.starts.clone(),
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
    starts: Vec<usize>,
    claims: Vec<Claims>,
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
    pub fn history_ranges(&self, domain: HistoryDomainId) -> Vec<(usize, usize)> {
        self.claims[domain.0].ranges()
    }
    pub fn history_start(&self, domain: HistoryDomainId) -> usize {
        self.starts[domain.0]
    }
    /// A live sequence sharing every row this checkpoint references.
    pub fn fork(&self) -> SequenceState {
        let claims = self.claims.clone();
        for claims in &claims {
            claims.set_live(true);
        }
        self.store.owners.set(self.store.owners.get() + 1);
        self.store.sequences.set(self.store.sequences.get() + 1);
        SequenceState {
            store: self.store.clone(),
            position: self.position,
            expected_end: self.position,
            history_start: self.history_start,
            starts: self.starts.clone(),
            claims,
            bank: self.bank.clone(),
            tape: self.tape,
        }
    }
}
/// Publish `count` rows, one per stored domain in `claims`, whose recurrent
/// version is (`following`, `tape`), then release the rows each window
/// domain no longer references.
fn install_commit(
    state: &mut SequenceState,
    claims: Vec<Claims>,
    following: &mut BankHandle,
    tape: usize,
    count: usize,
) {
    // Appending moves references: what a checkpoint or fork sharing this
    // history's rows sees never changes.
    for (history, claims) in state.claims.iter_mut().zip(claims) {
        history.append(claims);
    }
    std::mem::swap(&mut state.bank, following);
    state.tape = tape;
    state.position += count;
    state.trim();
}

#[cfg(test)]
mod domain_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, DeviceCatalog};
    use std::cell::Ref;

    /// The one domain of a Qwen-shaped store.
    const TOKEN: HistoryDomainId = HistoryDomainId(0);

    /// A store of one Token domain (none without components), as Qwen's
    /// target and head stores are.
    fn token_store(
        device: Rc<Device>,
        context: usize,
        rows: usize,
        components: Vec<ComponentDescriptor>,
        specs: Vec<ComponentSpec>,
        banks: BankCapacity,
    ) -> Result<StoreBindings, Error> {
        let domains = if components.is_empty() {
            vec![]
        } else {
            vec![HistoryDomainPlan {
                layout: HistoryDomainLayout::Token { components },
                logical_rows: rows,
            }]
        };
        StateStore::new(device, context, context, domains, specs, banks).map(|(_, bindings)| bindings)
    }

    impl StateStore {
        fn arena(&self) -> &Rc<RefCell<Arena>> {
            &self.domains[TOKEN.0].arena
        }

        fn history_slabs(&self) -> Ref<'_, SlabTensor> {
            Ref::map(self.backing.borrow(), |backing| {
                &backing.history[TOKEN.0].slabs
            })
        }
    }

    /// One past the highest backed row and bank, as the one-domain store
    /// reported them.
    fn committed_extent(store: &StateStore) -> (usize, usize) {
        (
            store
                .history_domains()
                .next()
                .map_or(0, |domain| store.committed_rows(domain)),
            store.committed_banks(),
        )
    }

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

    /// A backed arena of one slab that is one page.
    fn arena(rows: usize) -> Rc<RefCell<Arena>> {
        let rows = rows.max(1);
        Rc::new(RefCell::new(Arena::new(rows, rows, rows)))
    }

    #[test]
    fn history_spans_and_free_holes_stop_at_slab_boundaries() {
        let arena = Rc::new(RefCell::new(Arena::new(0, 4, 4)));
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
    fn growth_across_a_slab_boundary_stays_in_place() {
        let arena = Rc::new(RefCell::new(Arena::new(0, 8, 8)));
        for index in 0..4 {
            arena.borrow_mut().back_slab(index, 32);
        }
        // Chunked growth of one sequence: every chunk continues at the
        // history end, including a chunk that fills one slab's last rows and
        // spills into the next.
        let mut history = Vec::new();
        let mut end = None;
        for _ in 0..5 {
            let ranges = arena.borrow_mut().claim(end, 6);
            end = ranges.last().map(|(start, count)| start + count);
            append_ranges(&mut history, ranges, 8);
        }
        assert_eq!(history, vec![(0, 8), (8, 8), (16, 8), (24, 6)]);
    }

    #[test]
    fn idle_shrink_does_not_compact_occupied_history_or_bank_slabs() {
        let Some(device) = cpu_device() else {
            return;
        };
        let mut store = token_store(
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
        store.add_history_slabs(TOKEN, 1).unwrap();
        store.add_bank_slabs(1).unwrap();
        let rows = store.history_slab_rows(TOKEN);
        let _low = claim_rows(&store.arena(), 0, 1);
        let high = claim_rows(&store.arena(), rows, 1);
        let mut claims = (0..4)
            .map(|_| Some(store.banks.acquire().unwrap()))
            .collect::<Vec<_>>();
        drop(claims[0].take());
        drop(claims[1].take());
        let bank = claims[3].as_ref().unwrap();
        let before = (
            device.memory_usage().charged,
            committed_extent(&store),
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
                committed_extent(&store),
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
        let mut store = token_store(
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
        store.add_history_slabs(TOKEN, 2).unwrap();
        let occupied = claim_rows(&store.arena(), 0, 1);
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
                .history_slabs()
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
                .history_slabs()
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
        let mut store = token_store(
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
        store.add_history_slabs(TOKEN, 1).unwrap();
        store.add_bank_slabs(1).unwrap();
        let rows = store.history_slab_rows(TOKEN);
        assert!(rows < 5000);
        assert_eq!(store.bank_slab_banks(), 4);
        let _low = claim_rows(&store.arena(), 0, 2);
        let high = claim_rows(&store.arena(), rows, 1);
        let history_value = vec![29u8; 4096 * 4];
        store
            .history_slabs()
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
        assert_eq!(committed_extent(&store), (rows, 4));
        assert!(high.ranges()[0].0 < rows);
        assert!(bank.index() < 4);
        assert_eq!(
            store
                .history_slabs()
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

    /// Reclaim moves whole occupied pages to free pages of kept slabs at the
    /// same offsets, so a history keeps its page structure.
    #[test]
    fn reclaim_moves_whole_pages_at_their_offsets() {
        let arena = Rc::new(RefCell::new(Arena::new(0, 4, 2)));
        for index in 0..3 {
            arena.borrow_mut().back_slab(index, 12);
        }
        let _full = claim_rows(&arena, 0, 4);
        // Row 5: the second row of page [4, 6).
        let sparse = claim_rows(&arena, 5, 1);
        let _high = claim_rows(&arena, 8, 2);
        let (planned, moves) = arena
            .borrow()
            .compact_into_slabs(12, &BTreeSet::from([0, 2]));
        // Page [4, 6) moves to the free page [10, 12) at the same offset.
        assert_eq!(moves, vec![(5, 11, 1)]);
        assert_eq!(planned.referenced, 7);
        assert!(!planned.backed.contains(&1));
        assert_eq!(planned.entries[&sparse.id].ranges, vec![(11, 1)]);
    }

    #[test]
    fn reclaim_frees_the_sparsest_history_slab_without_new_charge() {
        let Some(device) = cpu_device() else {
            return;
        };
        let mut store = token_store(
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
        store.add_history_slabs(TOKEN, 2).unwrap();
        let rows = store.history_slab_rows(TOKEN);
        let page = store.history_page_rows(TOKEN);
        let _low = claim_rows(&store.arena(), 0, rows);
        let sparse = claim_rows(&store.arena(), rows, 1);
        // The high slab's last page is free: the sparse page moves there.
        let _high = claim_rows(&store.arena(), 2 * rows, rows - page);
        let moved = 3 * rows - page;
        let value = vec![43u8; 4096 * 4];
        store
            .history_slabs()
            .region_rows(0, rows as u64, 1)
            .unwrap()
            .write_from_host(&value)
            .unwrap();
        let slab_bytes = store
            .history_slabs()
            .slab_bytes();
        let charged = device.memory_usage().charged;
        device.set_memory_limit(Some(charged));
        assert_eq!(store.shrink(ShrinkPolicy::Reclaim).unwrap(), slab_bytes);
        device.set_memory_limit(None);
        assert_eq!(store.compactions().history_rows, 1);
        assert_eq!(sparse.ranges(), vec![(moved, 1)]);
        assert!(store
            .history_slabs()
            .slab(1)
            .is_none());
        assert_eq!(
            store
                .history_slabs()
                .region_rows(0, moved as u64, 1)
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
        let mut store = token_store(
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
        store.add_history_slabs(TOKEN, 2).unwrap();
        let rows = store.history_slab_rows(TOKEN);
        let page = store.history_page_rows(TOKEN);
        let _low = claim_rows(&store.arena(), 0, rows);
        let sparse = claim_rows(&store.arena(), rows, 1);
        // One whole page of the high slab stays free for the sparse page.
        let _high = claim_rows(&store.arena(), 2 * rows, rows - page);
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
            .history_slabs()
            .slab(1)
            .is_some());
    }

    #[test]
    fn released_history_slab_stays_charged_while_a_view_pins_it() {
        let Some(device) = cpu_device() else {
            return;
        };
        let mut store = token_store(
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
            .history_slabs()
            .slab_bytes();
        let binding = store.history_planes().unwrap()[0].buffer.clone();
        let charged = device.memory_usage().charged;
        assert!(store.shrink(ShrinkPolicy::Reclaim).is_err());
        assert!(store
            .history_slabs()
            .slab(0)
            .is_some());
        assert_eq!(device.memory_usage().charged, charged);
        drop(binding);
        let pinned = store
            .history_slabs()
            .region_rows(0, 0, 1)
            .unwrap();
        assert_eq!(store.shrink(ShrinkPolicy::Reclaim).unwrap(), 0);
        assert_eq!(committed_extent(&store).0, 0);
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
    fn arena_grows_histories_in_place_and_takes_whole_pages() {
        // One slab of ten 10-row pages.
        let mut arena = Arena::new(100, 100, 10);
        // Row 0 has no preceding history: fill from its start.
        assert_eq!(arena.claim(None, 10), [(0, 10)]);
        // A new history takes the middle page of the free pages, leaving the
        // pages after [0, 10) as that history's room.
        assert_eq!(arena.claim(None, 10), [(50, 10)]);
        assert_eq!(arena.free, [(10, 40), (60, 40)]);
        // Both grow into the page at their end, then in place, one row at a
        // time, whatever the interleaving.
        for step in 0..5 {
            assert_eq!(arena.claim(Some(10 + step), 1), [(10 + step, 1)]);
            assert_eq!(arena.claim(Some(60 + step), 1), [(60 + step, 1)]);
        }
        // Rows after a history's end in its page are never another's: a
        // third history takes the middle page of the largest free run (ties
        // go to the lower address).
        assert_eq!(arena.available(), 60);
        assert_eq!(arena.claim(None, 5), [(30, 5)]);
        // A history fills its page in place, then takes the page at its end,
        // then (that page taken) the middle page of the largest free run.
        assert_eq!(arena.claimable(Some(35)), 5 + 50);
        assert_eq!(arena.claim(Some(35), 20), [(35, 5), (40, 10), (80, 5)]);
        assert_eq!(arena.free, [(15, 15), (65, 15), (85, 15)]);
        // Only whole free pages serve a fresh history: pages 20, 70 and 90.
        assert_eq!(arena.available(), 30);
        assert_eq!(arena.claim(None, 25), [(20, 10), (70, 10), (90, 5)]);
        assert_eq!(arena.free, [(15, 5), (65, 5), (85, 5), (95, 5)]);
        assert_eq!(arena.available(), 0);
        assert_eq!(arena.claimable(Some(95)), 5);
        // A history whose next row another took must relocate its last page.
        assert!(arena.tail_blocked(Some(12)));
        assert!(!arena.tail_blocked(Some(15)));
        assert!(!arena.tail_blocked(Some(20)));
    }

    #[test]
    fn planes_start_with_one_slab_as_one_stable_set() {
        let Some(device) = cpu_device() else {
            // A backend may be present yet fail Seismic's runtime calibration
            // under a noisy test host. Never substitute an accelerator here.
            return;
        };
        let mut store = token_store(
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
        // Eight rows requested reserve one whole 256-row page.
        assert_eq!(committed_extent(&store).0, 256);
        assert_eq!(store.history_row_bytes(TOKEN), 32);
        assert_eq!(store.allocation_trace().unwrap().history[0].bytes, 32 * 256);

        store
            .provision(
                &[RowDemand {
                    domain: TOKEN,
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
                && plane.buffer.extents() == [256, 1, 4]
        }));

        let second = store.history_planes().unwrap();
        assert!(first
            .iter()
            .zip(second.iter())
            .all(|(left, right)| left.buffer.shares_allocation(&right.buffer)));
    }

    fn validate(plans: Vec<HistoryDomainPlan>) -> Result<(), Error> {
        domain::validate_domains(plans, 2, 1).map(|_| ())
    }

    fn token(components: Vec<ComponentDescriptor>, logical_rows: usize) -> HistoryDomainPlan {
        HistoryDomainPlan {
            layout: HistoryDomainLayout::Token { components },
            logical_rows,
        }
    }

    #[test]
    fn store_rejects_total_capacity_overflow_before_allocation() {
        let component = dense_component(usize::MAX / 4);
        assert!(matches!(
            validate(vec![token(vec![component], 2)]),
            Err(Error::Layout(LayoutError::ArithmeticOverflow(_)))
        ));
    }

    #[test]
    fn duplicate_layer_descriptors_are_rejected() {
        let component = dense_component(4);
        assert!(matches!(
            validate(vec![token(vec![component.clone(), component], 2)]),
            Err(Error::Layout(LayoutError::DuplicateLayer(LayerRef::Target(0))))
        ));
        // A layer belongs to one domain, stored or shared.
        assert!(matches!(
            validate(vec![
                token(vec![dense_component(4)], 2),
                HistoryDomainPlan {
                    layout: HistoryDomainLayout::Shared {
                        source: LayerRef::Target(0),
                        layers: vec![LayerRef::Target(0)],
                    },
                    logical_rows: 0,
                },
            ]),
            Err(Error::Layout(LayoutError::DuplicateLayer(LayerRef::Target(0))))
        ));
    }

    #[test]
    fn domain_layouts_are_validated() {
        let shared = |source, logical_rows| HistoryDomainPlan {
            layout: HistoryDomainLayout::Shared {
                source,
                layers: vec![LayerRef::Target(5)],
            },
            logical_rows,
        };
        assert!(matches!(
            validate(vec![token(vec![dense_component(4)], 2), shared(LayerRef::Target(3), 0)]),
            Err(Error::Layout(LayoutError::UnknownSharedSource(LayerRef::Target(3))))
        ));
        assert!(matches!(
            validate(vec![token(vec![dense_component(4)], 2), shared(LayerRef::Target(0), 2)]),
            Err(Error::Layout(LayoutError::SharedDomainRows(LayerRef::Target(0))))
        ));
        assert!(validate(vec![token(vec![dense_component(4)], 2), shared(LayerRef::Target(0), 0)]).is_ok());
        assert!(matches!(
            validate(vec![HistoryDomainPlan {
                layout: HistoryDomainLayout::Window {
                    rows: 0,
                    components: vec![dense_component(4)],
                },
                logical_rows: 2,
            }]),
            Err(Error::Layout(LayoutError::ZeroWindow))
        ));
        assert!(matches!(
            validate(vec![token(vec![], 2)]),
            Err(Error::Layout(LayoutError::EmptyHistoryDomain))
        ));
        // A Token domain reserves at least a context of rows.
        assert!(matches!(
            validate(vec![token(vec![dense_component(4)], 1)]),
            Err(Error::Request(_))
        ));
        assert!(matches!(
            validate(vec![HistoryDomainPlan {
                layout: HistoryDomainLayout::Block {
                    rate: 4,
                    components: vec![dense_component(4)],
                },
                logical_rows: 2,
            }]),
            Err(Error::UnsupportedHistoryDomain(HistoryDomainKind::Block { rate: 4 }))
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
        let store = token_store(
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
        assert_eq!(store.occupied_rows(TOKEN), 3);
        let state = advance.abort();
        assert_eq!(state.position(), 0);
        assert_eq!(state.bank_index(), original);
        assert_eq!(store.occupied_rows(TOKEN), 0);

        let advance = OwnedStateAdvance::begin_speculative(state, 3, 1)
            .ok()
            .unwrap();
        let successor = advance.bindings().following_bank;
        let OwnedAdvanceResolution::Committed(state) = advance.commit(2).ok().unwrap() else {
            panic!("an accepted prefix commits as a tape version");
        };
        assert_eq!(state.position(), 2);
        assert_eq!((state.bank_index(), state.tape_rows()), (successor, 1));
        assert_eq!(store.occupied_rows(TOKEN), 2);
        assert_eq!(checkpoint.position(), 0);
        let checkpoint_fork = checkpoint.fork();
        assert_eq!(checkpoint_fork.position(), 0);
        assert_eq!(checkpoint_fork.bank_index(), original);
        assert!(checkpoint_fork.history_ranges(TOKEN).is_empty());
        drop(checkpoint_fork);

        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let OwnedAdvanceResolution::Aborted(state) = advance.commit(0).ok().unwrap() else {
            panic!("zero prefix must abort");
        };
        assert_eq!(state.position(), 2);
        assert_eq!(store.occupied_rows(TOKEN), 2);
        // A plain advance has no interior recurrent version.
        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let (state, _) = advance.commit(1).err().unwrap();
        assert_eq!((state.position(), state.tape_rows()), (2, 1));
        assert_eq!(store.occupied_rows(TOKEN), 2);
        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        assert_eq!(state.position(), 4);
        assert_eq!(store.occupied_rows(TOKEN), 4);
    }

    /// Shared system prompt, two divergent requests and retained checkpoints
    /// on one path: every holder sees the same physical prefix rows, the
    /// store holds them once, and pricing a set charges them once.
    #[test]
    fn shared_prefix_is_the_same_rows_and_is_charged_once() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = token_store(
            device.clone(),
            1024,
            1024,
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
        let row = store.history_row_bytes(TOKEN);
        let bank = store.allocation_trace().unwrap().recurrent_bank_bytes;
        let commit = |state: SequenceState, rows: usize| {
            let advance = OwnedStateAdvance::begin(state, rows).ok().unwrap();
            let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap()
            else {
                panic!("full prefix must commit");
            };
            state
        };
        // The system prompt is one prefill run of one whole page, so branches
        // share it and each continues in a page of its own (a branch from a
        // partial page first relocates it; see the relocation tests).
        let page = store.history_page_rows(TOKEN);
        assert_eq!(page, 256);
        let prompt = commit(store.create().unwrap(), page);
        let system = prompt.checkpoint();
        assert_eq!(system.history_ranges(TOKEN), [(0, page)]);
        // Request A continues the path; requests B and C branch at the prompt.
        let a = commit(prompt, 8);
        let b = commit(system.fork(), 6);
        let c = commit(system.fork(), 3);
        let a_turn = a.checkpoint();
        assert_eq!(a.history_ranges(TOKEN), [(0, page + 8)]);
        for branch in [&b, &c] {
            assert_eq!(branch.history_ranges(TOKEN)[0], (0, page));
            assert_eq!(branch.history_ranges(TOKEN).len(), 2);
        }
        assert_eq!(store.occupied_rows(TOKEN), page + 8 + 6 + 3);
        let census = store
            .holding_census(
                &[Holder::State(&a), Holder::State(&b), Holder::State(&c)],
                &[Holder::Checkpoint(&system), Holder::Checkpoint(&a_turn)],
                &[],
            )
            .unwrap();
        assert_eq!(census.retained, bank);
        assert_eq!(census.live, (page as u64 + 8 + 6 + 3) * row + 3 * bank);
        assert_eq!(census.total(), store.committed_bytes());
        assert_eq!(census.total(), device.memory_usage().charged);
        let submitted = store
            .holding_census(
                &[Holder::State(&b), Holder::State(&c)],
                &[Holder::Checkpoint(&system), Holder::Checkpoint(&a_turn)],
                &[Holder::State(&a)],
            )
            .unwrap();
        assert_eq!(submitted.in_flight, (page as u64 + 8) * row + bank);
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
            (page as u64 + 8) * row + 2 * bank
        );
        // Repeated holders count once.
        let repeated = [
            Holder::Checkpoint(&system),
            Holder::Checkpoint(&system),
            Holder::Checkpoint(&a_turn),
        ];
        assert_eq!(
            store.exclusive_bytes(&repeated).unwrap(),
            (page as u64 + 8) * row + 2 * bank
        );
        drop((system, a_turn));
        assert_eq!(store.occupied_rows(TOKEN), 0);
    }

    /// A history that fills its page continues in the page at its end; a
    /// speculative advance across that boundary commits a prefix without
    /// repair, and the rejected page is free again.
    #[test]
    fn growth_continues_into_the_next_page_and_rolls_back_whole_pages() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = paged_store(device);
        let first = advance_all(store.create().unwrap(), 254);
        // A second history takes the middle page of the free pages.
        let middle = advance_all(store.create().unwrap(), 2);
        assert_eq!(first.history_ranges(TOKEN), [(0, 254)]);
        assert_eq!(middle.history_ranges(TOKEN), [(512, 2)]);
        let checkpoint = first.checkpoint();
        // Two rows fit in place; the third takes the page at the end.
        let advance = OwnedStateAdvance::begin(first, 3).ok().unwrap();
        assert_eq!(advance.bindings().destinations[0], [254, 255, 256]);
        let OwnedAdvanceResolution::Committed(first) = advance.commit(2).ok().unwrap() else {
            panic!("attention prefix must commit without repair");
        };
        assert_eq!(first.position(), 256);
        assert_eq!(first.history_ranges(TOKEN), [(0, 256)]);
        assert_eq!(checkpoint.fork().history_ranges(TOKEN), [(0, 254)]);
        assert_eq!(store.occupied_rows(TOKEN), 256 + 2);
        // The rejected row's page is whole and free again.
        assert_eq!(store.free_rows(TOKEN), 512);
        assert_eq!(middle.position(), 2);
    }

    #[test]
    fn pooled_banks_are_reused_and_checkpoint_claims_force_cow() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = token_store(
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
        // Three branches, each in a page of its own.
        let mut store = token_store(
            device,
            16,
            3 * 256,
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
                    domain: TOKEN,
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
        assert_eq!(store.available_banks(), committed_extent(&store).1 - 1);
    }

    #[test]
    fn fresh_sequences_share_a_pristine_seed_after_dirty_successor_reuse() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = token_store(
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

    /// A store of 1,024 rows (four 256-row pages) of one dense row for a
    /// 1,024-row context, every page backed.
    fn paged_store(device: Rc<Device>) -> Rc<StateStore> {
        let mut store = token_store(
            device,
            1024,
            1024,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active: 3,
                in_flight: 3,
                retained: 1,
            },
        )
        .unwrap();
        assert_eq!(store.history_page_rows(TOKEN), 256);
        store
            .provision(
                &[RowDemand {
                    domain: TOKEN,
                    rows: 1024,
                }],
                0,
            )
            .unwrap();
        Rc::clone(&store)
    }

    fn advance_all(state: SequenceState, rows: usize) -> SequenceState {
        let advance = OwnedStateAdvance::begin(state, rows).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("an attention advance commits");
        };
        state
    }

    /// Two forks of one checkpoint on a partial page: the first continues
    /// that page in place; the second must relocate the page before it
    /// grows. The relocation copies exactly the page's rows, publishes only
    /// after its copy, and an aborted relocation leaves the state unchanged.
    #[test]
    fn relocation_is_bit_exact_and_publishes_only_after_success() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = paged_store(device);
        let state = advance_all(store.create().unwrap(), 10);
        let prefix = state.history_ranges(TOKEN);
        assert_eq!(prefix.len(), 1);
        let (page, _) = prefix[0];
        // Tag every row of the store with its row number.
        for plane in store.history_planes().unwrap() {
            let mut bytes = vec![0u8; plane.buffer.byte_len() as usize];
            for row in 0..1024 {
                let start = row * plane.row_bytes;
                bytes[start..start + plane.row_bytes].fill(row as u8);
            }
            plane.buffer.clone().write_from_host(&bytes).unwrap();
        }
        let checkpoint = state.checkpoint();
        drop(state);
        let first = advance_all(checkpoint.fork(), 3);
        assert_eq!(first.history_ranges(TOKEN), [(page, 13)]);
        let second = checkpoint.fork();
        assert_eq!(second.blocked_tails(), [TOKEN]);
        assert_eq!(
            second.relocation_demands(),
            [RowDemand {
                domain: TOKEN,
                rows: 256,
            }]
        );
        // Growth without relocation is refused, not placed in another span.
        let Err((second, _)) = OwnedStateAdvance::begin(second, 1) else {
            panic!("a blocked tail grows only after relocation");
        };
        let failed = OwnedTailRelocation::prepare(second, TOKEN).ok().unwrap();
        let second = failed.abort();
        assert_eq!(second.history_ranges(TOKEN), [(page, 10)]);
        let relocation = OwnedTailRelocation::prepare(second, TOKEN).ok().unwrap();
        let copy = relocation.copy();
        assert_eq!(copy.rows(), 10);
        let destination = copy.copies()[0].to[0];
        assert_eq!(destination % 256, 0);
        for copy in copy.copies() {
            assert_eq!(copy.from, (page..page + 10).collect::<Vec<_>>());
            assert_eq!(copy.to, (destination..destination + 10).collect::<Vec<_>>());
        }
        // Run the copy on the host, then publish.
        for plane in store.history_planes().unwrap() {
            let mut bytes = plane.buffer.read_to_host().unwrap();
            for offset in 0..10 {
                let (from, to) = (page + offset, destination + offset);
                let source = bytes[from * plane.row_bytes..(from + 1) * plane.row_bytes].to_vec();
                bytes[to * plane.row_bytes..(to + 1) * plane.row_bytes].copy_from_slice(&source);
            }
            plane.buffer.clone().write_from_host(&bytes).unwrap();
        }
        let second = relocation.commit();
        assert_eq!(second.history_ranges(TOKEN), [(destination, 10)]);
        assert!(second.blocked_tails().is_empty());
        for plane in store.history_planes().unwrap() {
            let bytes = plane.buffer.read_to_host().unwrap();
            for offset in 0..10 {
                let row = destination + offset;
                assert_eq!(
                    &bytes[row * plane.row_bytes..(row + 1) * plane.row_bytes],
                    vec![(page + offset) as u8; plane.row_bytes]
                );
            }
        }
        // It now grows in place; the first fork and the checkpoint keep theirs.
        let second = advance_all(second, 3);
        assert_eq!(second.history_ranges(TOKEN), [(destination, 13)]);
        assert_eq!(first.history_ranges(TOKEN), [(page, 13)]);
        assert_eq!(checkpoint.history_ranges(TOKEN), [(page, 10)]);
    }

    /// Without a free page a relocation is a capacity error before any row
    /// moves, and the sequence is returned unchanged.
    #[test]
    fn relocation_without_a_free_page_is_a_capacity_error() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = paged_store(device);
        let checkpoint = advance_all(store.create().unwrap(), 10).checkpoint();
        let first = advance_all(checkpoint.fork(), 1);
        // Every other page is taken.
        let _pages = (0..3)
            .map(|_| advance_all(store.create().unwrap(), 256))
            .collect::<Vec<_>>();
        assert_eq!(store.free_rows(TOKEN), 0);
        let second = checkpoint.fork();
        let Err((second, error)) = OwnedTailRelocation::prepare(second, TOKEN) else {
            panic!("no free page holds the relocated rows");
        };
        assert!(matches!(error, Error::Capacity { .. }));
        assert_eq!(
            second.history_ranges(TOKEN),
            checkpoint.history_ranges(TOKEN)
        );
        assert_eq!(first.history_ranges(TOKEN).len(), 1);
    }

    /// Growth demand is what the last page cannot hold in place, rounded up
    /// to whole pages; rows inside another history's page never count.
    #[test]
    fn page_demand_counts_in_place_rows_then_whole_pages() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = paged_store(device);
        let state = advance_all(store.create().unwrap(), 10);
        let rows = |demands: Vec<RowDemand>| demands[0].rows;
        assert_eq!(rows(state.demands(246)), 0);
        assert_eq!(rows(state.demands(247)), 256);
        assert_eq!(rows(state.demands(246 + 513)), 768);
        assert_eq!(rows(store.create().unwrap().demands(1)), 256);
        // Three whole pages remain free for a fresh history.
        assert_eq!(store.free_rows(TOKEN), 768);
    }

    /// Fill `arena` as one history's chunked prompt, backing the lowest
    /// unbacked slab whenever it runs short, as provisioning does.
    /// Back the lowest unbacked slabs of an arena of `capacity` rows until
    /// a history ending at `end` can take `rows` more.
    fn back_for(arena: &Rc<RefCell<Arena>>, capacity: usize, end: Option<usize>, rows: usize) {
        let mut inner = arena.borrow_mut();
        while inner.claimable(end) < rows {
            let Some(slab) =
                (0..capacity.div_ceil(inner.slab_rows)).find(|slab| !inner.backed.contains(slab))
            else {
                return;
            };
            inner.back_slab(slab, capacity);
        }
    }

    /// The incident: Qwen3.8-27B (28,672-byte K8/V4 rows) at a 65,792-row
    /// context, a 65,536-token prompt in 512-row chunks. The committed
    /// allocator split every chunk that crossed a slab and reached 98-109
    /// spans against 64 sealed; paging holds one span per slab it touches.
    /// Four concurrent 16,384-token prompts sharing the row budget (84
    /// spans before) stay within the limit too.
    #[test]
    fn long_prompts_stay_within_the_span_limit_at_the_incident_geometry() {
        let geometry = history_geometry(28_672, 65_792, 0).unwrap();
        assert!(geometry.span_limit <= MAX_HISTORY_SPANS);
        for concurrent in [1usize, 2, 4, 8] {
            let capacity = (65_792 * concurrent).next_multiple_of(geometry.page_rows);
            let arena = Rc::new(RefCell::new(Arena::new(
                0,
                geometry.slab_rows,
                geometry.page_rows,
            )));
            let prompt = 65_536 / concurrent;
            let mut histories = (0..concurrent)
                .map(|_| Claims::new(&arena, vec![]))
                .collect::<Vec<_>>();
            let mut written = 0;
            while written < prompt {
                // The scheduler shares one 512-row budget across requests.
                let rows = (512 / concurrent).min(prompt - written);
                for history in &mut histories {
                    let end = history.end();
                    back_for(&arena, capacity, end, rows);
                    let ranges = arena.borrow_mut().claim(end, rows);
                    history.append(Claims::new(&arena, ranges));
                    let spans = history.ranges().len();
                    assert!(
                        spans <= geometry.span_limit,
                        "{concurrent} prompts: {spans} spans at {written} rows"
                    );
                }
                written += rows;
            }
            let peak = histories
                .iter()
                .map(|history| history.ranges().len())
                .max()
                .unwrap();
            eprintln!(
                "incident geometry, {concurrent} concurrent prompt(s) of {prompt}: {peak} spans \
                 (limit {}, page {} rows, slab {} rows)",
                geometry.span_limit, geometry.page_rows, geometry.slab_rows
            );
            if concurrent == 1 {
                // One span per slab touched.
                assert_eq!(peak, prompt.div_ceil(geometry.slab_rows));
            }
        }
    }

    /// Randomized placement over a sweep of geometries (rows of 256 B to
    /// 1 MiB; row limits of 2,048 to 1,048,576 rows, with and without rows a
    /// Shared reader sees appended): up to eight live histories grow in
    /// chunks of up to a sixteenth of the row limit, advance speculatively
    /// and roll back a random suffix, fork from checkpoints with both
    /// branches continuing (relocating a blocked last page), release a
    /// window's front rows, finish, and are reclaimed (the sparsest slab
    /// emptied into free pages of the others). After every operation every
    /// history presents at most `ceil(rows / page) + 1` spans, within its
    /// domain's span limit; and whenever the free pages hold a growth's page
    /// demand (the admission check), the claim succeeds without growth.
    #[test]
    fn every_placement_path_stays_within_the_span_limit() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut random = |bound: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % bound.max(1) as u64) as usize
        };
        for row_bytes in [256u64, 1_024, 8_960, 28_672, 131_072, 1 << 20] {
            for (row_limit, appended) in [
                (2_048, 0),
                (65_792, 0),
                (65_792, 512),
                (262_144, 0),
                (1_048_576, 1_024),
            ] {
                let geometry = history_geometry(row_bytes, row_limit, appended).unwrap();
                assert!(geometry.span_limit <= MAX_HISTORY_SPANS);
                let page = geometry.page_rows;
                let slab = geometry.slab_rows;
                let capacity = (row_limit * 4).next_multiple_of(page);
                let arena = Rc::new(RefCell::new(Arena::new(0, slab, page)));
                let mut live: Vec<(Claims, usize)> = Vec::new();
                let mut checkpoints: Vec<Claims> = Vec::new();
                for _ in 0..1_500 {
                    match random(12) {
                        // Admit a history (or fork one from a checkpoint).
                        0 | 1 if live.len() < 8 => {
                            if !checkpoints.is_empty() && random(2) == 0 {
                                let checkpoint = &checkpoints[random(checkpoints.len())];
                                let rows = checkpoint.rows();
                                live.push((checkpoint.clone(), rows));
                            } else {
                                live.push((Claims::new(&arena, vec![]), 0));
                            }
                        }
                        // Finish a history.
                        2 if !live.is_empty() => {
                            live.swap_remove(random(live.len()));
                        }
                        // Keep a checkpoint of a history.
                        3 if !live.is_empty() && checkpoints.len() < 6 => {
                            let index = random(live.len());
                            checkpoints.push(live[index].0.clone());
                        }
                        4 if !checkpoints.is_empty() => {
                            checkpoints.swap_remove(random(checkpoints.len()));
                        }
                        // Release a window's front rows.
                        5 if !live.is_empty() => {
                            let index = random(live.len());
                            let (history, _) = &mut live[index];
                            let rows = history.rows();
                            if rows > 1 {
                                history.drop_front(random(rows));
                            }
                        }
                        // Reclaim: empty the sparsest backed slab into free
                        // pages of the others when they hold its pages.
                        6 => {
                            let planned = {
                                let inner = arena.borrow();
                                let occupied = |index: usize| {
                                    inner
                                        .runs
                                        .iter()
                                        .flat_map(|(&start, run)| {
                                            start / page..=(start + run.count - 1) / page
                                        })
                                        .filter(|&page_index| page_index * page / slab == index)
                                        .collect::<BTreeSet<_>>()
                                        .len()
                                };
                                inner
                                    .backed
                                    .iter()
                                    .copied()
                                    .min_by_key(|&index| (occupied(index), index))
                                    .and_then(|victim| {
                                        let keep = inner
                                            .backed
                                            .iter()
                                            .copied()
                                            .filter(|&index| index != victim)
                                            .collect::<BTreeSet<_>>();
                                        let room = inner
                                            .free_pages()
                                            .into_iter()
                                            .filter(|free| keep.contains(&(free / slab)))
                                            .count();
                                        (room >= occupied(victim))
                                            .then(|| inner.compact_into_slabs(capacity, &keep).0)
                                    })
                            };
                            if let Some(planned) = planned {
                                *arena.borrow_mut() = planned;
                            }
                        }
                        // Grow, possibly speculatively with a rolled-back
                        // suffix.
                        _ if !live.is_empty() => {
                            let index = random(live.len());
                            let (history, position) = &mut live[index];
                            let rows = (1 + random(row_limit / 16)).min(row_limit - *position);
                            if rows == 0 {
                                continue;
                            }
                            if arena.borrow().tail_blocked(history.end()) {
                                // Relocate the last page (the copy itself is
                                // the executor's): its rows move to a fresh
                                // page's start.
                                let end = history.end().unwrap();
                                let first = history
                                    .ranges()
                                    .last()
                                    .unwrap()
                                    .0
                                    .max((end - 1) / page * page);
                                back_for(&arena, capacity, None, end - first);
                                if arena.borrow().available() == 0 {
                                    continue;
                                }
                                let moved = arena.borrow_mut().claim(None, end - first);
                                let kept = history.rows() - (end - first);
                                drop(history.split_off(kept));
                                history.append(Claims::new(&arena, moved));
                            }
                            let end = history.end();
                            // Admission: free pages holding the page demand
                            // guarantee the claim.
                            {
                                let inner = arena.borrow();
                                let demand = rows
                                    .saturating_sub(inner.in_place(end))
                                    .div_ceil(page)
                                    * page;
                                if inner.available() >= demand {
                                    assert!(inner.claimable(end) >= rows);
                                }
                            }
                            back_for(&arena, capacity, end, rows);
                            if arena.borrow().claimable(end) < rows {
                                continue;
                            }
                            let ranges = arena.borrow_mut().claim(end, rows);
                            let mut tentative = Claims::new(&arena, ranges);
                            let accepted = if random(3) == 0 {
                                1 + random(rows)
                            } else {
                                rows
                            };
                            drop(tentative.split_off(accepted));
                            history.append(tentative);
                            *position += accepted;
                        }
                        _ => {}
                    }
                    for (history, _) in &live {
                        let (spans, rows) = (history.ranges().len(), history.rows());
                        assert!(
                            spans <= rows.div_ceil(page) + 1 && spans <= geometry.span_limit,
                            "{spans} spans for {rows} rows (row bytes {row_bytes}, page {page}, \
                             limit {})",
                            geometry.span_limit
                        );
                    }
                }
            }
        }
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
        let mut bindings = token_store(
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
        let store = Rc::clone(&bindings);
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
                .history_ranges(TOKEN)
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
                .map(|(_, _, rows)| RowDemand {
                    domain: TOKEN,
                    rows: *rows,
                })
                .collect::<Vec<_>>();
            bindings.provision(&demands, 0).unwrap();
            peak_committed = peak_committed.max(committed_extent(&store).0);
            let mut advances = Vec::with_capacity(active);
            for (index, request, rows) in planned {
                let advance = match OwnedStateAdvance::begin(request.state, rows) {
                    Ok(advance) => advance,
                    Err((_, error)) => panic!("step {step}: {error}"),
                };
                let plane = &advance.bindings().history[0].buffer;
                for (offset, &row) in advance.bindings().destinations[0].iter().enumerate() {
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
                max_segments = max_segments.max(state.history_ranges(TOKEN).len());
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
            bindings.shrink(ShrinkPolicy::Idle).unwrap();
            let mut ranges = slots
                .iter()
                .flatten()
                .flat_map(|request| request.state.history_ranges(TOKEN))
                .collect::<Vec<_>>();
            ranges.sort_unstable();
            assert!(
                ranges
                    .windows(2)
                    .all(|pair| pair[0].0 + pair[0].1 <= pair[1].0),
                "live histories overlap"
            );
            let committed = committed_extent(&store).0;
            peak_committed = peak_committed.max(committed);
            assert!(
                store.occupied_rows(TOKEN) <= committed,
                "claims lie in committed rows"
            );
        }
        for request in slots.iter().flatten() {
            verify(request);
        }
        drop(slots);
        bindings.shrink(ShrinkPolicy::Idle).unwrap();
        Interleaved {
            segments: max_segments,
            span_limit: store.span_limit(TOKEN),
            reserved: store.allocation_trace().unwrap().history[0].capacity,
            decode_steps: max_decode_steps,
            peak_committed,
            released_to: committed_extent(&store).0,
            written,
        }
    }

    /// The backing commits rows and banks with demand, keeps every row's
    /// contents across growth, and returns bytes to the device ledger when
    /// a slab is unreferenced; a launch's captured binding pins placement.
    #[test]
    fn backing_grows_and_shrinks_and_returns_device_memory() {
        let Some(device) = cpu_device() else {
            return;
        };
        let mut store = token_store(
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
            committed_extent(&store),
            (store.history_slab_rows(TOKEN), store.bank_slab_banks().min(49))
        );
        // A 1,000-row prefill commits exactly one slab.
        let advance = OwnedStateAdvance::begin(store.create().unwrap(), 1000)
            .ok()
            .unwrap();
        let rows = committed_extent(&store).0;
        assert_eq!(rows, store.history_slab_rows(TOKEN));
        // A launch's captured binding pins slab placement: growth fails
        // rather than doing nothing, and the backing is unchanged.
        assert!(matches!(
            store.provision(
                &[RowDemand {
                    domain: TOKEN,
                    rows: 5000,
                }],
                8,
            ),
            Err(Error::Tensor(seismic::TensorError::SlabLayout(_)))
        ));
        assert_eq!(committed_extent(&store).0, rows);
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
        // Without a captured binding, growth commits more rows and banks and keeps
        // the accepted rows' contents.
        let demand = state.demands(10000);
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
        assert_eq!(committed_extent(&store).0, rows);
        assert_eq!(state.history_ranges(TOKEN), [(0, 1000)]);
        device.set_memory_limit(previous_limit);
        store.provision(&demand, 8).unwrap();
        let (rows, banks) = committed_extent(&store);
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
        assert_eq!(state.history_ranges(TOKEN), [(0, 1000)]);
        // A launch binding must be released before placement changes.
        drop(plane);
        let before = charged();
        let released = store.shrink(ShrinkPolicy::Reclaim).unwrap();
        assert_eq!(released, before.saturating_sub(charged()));
        assert!(committed_extent(&store).0 < rows);
        assert!(committed_extent(&store).0 >= 1000);
        assert_eq!(store.external_pinned_bytes().unwrap(), 0);
        drop(state);
        // Idle hysteresis keeps a small backing; reclaim releases it all.
        store.shrink(ShrinkPolicy::Reclaim).unwrap();
        assert_eq!(committed_extent(&store).0, 0);
        store.release_idle().unwrap();
        assert!(charged() < base);
    }

    /// Refused slab growth keeps the published backing and existing rows.
    #[test]
    fn failed_slab_growth_keeps_published_history() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = token_store(
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
        store
            .history_slabs()
            .region_rows(0, 0, 1)
            .unwrap()
            .write_from_host(&encoded)
            .unwrap();
        let baseline = device.memory_usage().charged;
        device.set_memory_limit(Some(baseline));
        assert!(matches!(
            store.add_history_slabs(TOKEN, 1),
            Err(Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity { .. }
            )))
        ));
        device.set_memory_limit(None);
        assert_eq!(committed_extent(&store).0, store.history_slab_rows(TOKEN));
        assert_eq!(device.memory_usage().charged, baseline);
        assert_eq!(
            store
                .history_slabs()
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
        let mut store = token_store(
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
            domain: TOKEN,
            rows: store.history_slab_rows(TOKEN) + 1,
        }];
        let baseline = device.memory_usage().charged;
        let committed = committed_extent(&store);
        let history_bytes = store
            .history_slabs()
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
        assert_eq!(committed_extent(&store), committed);
        assert_eq!(device.memory_usage().charged, baseline);
        assert_eq!(store.arena().borrow().backed, BTreeSet::from([0]));

        store
            .provision_with_growth(&demand, 4, GrowthChoice::Minimum)
            .unwrap();
        assert_eq!(store.arena().borrow().backed, BTreeSet::from([0, 1]));
        assert_eq!(committed_extent(&store).1, committed.1 + store.bank_slab_banks());
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
        let mut bindings = token_store(
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
        let store = Rc::clone(&bindings);
        let mut step = |state: SequenceState, rows: usize| {
            bindings.provision(&state.demands(rows), 1).unwrap();
            let advance = OwnedStateAdvance::begin(state, rows).ok().unwrap();
            let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap()
            else {
                panic!("full prefix must commit");
            };
            bindings.shrink(ShrinkPolicy::Idle).unwrap();
            state
        };
        let mut state = step(store.create().unwrap(), 500);
        let _prompt = state.checkpoint();
        let mut committed = committed_extent(&store);
        for _ in 0..300 {
            state = step(state, 1);
            let now = committed_extent(&store);
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
        let mut bindings = token_store(
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
        let store = Rc::clone(&bindings);
        let tag = |row: usize| (row as f32).to_le_bytes();
        let mut commit = |state: SequenceState, rows: usize| {
            bindings.provision(&state.demands(rows), 0).unwrap();
            let advance = OwnedStateAdvance::begin(state, rows).ok().unwrap();
            let plane = &advance.bindings().history[0].buffer;
            for (offset, &row) in advance.bindings().destinations[0].iter().enumerate() {
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
        // A large request fills the low rows; a one-page prompt and two
        // branches (each continuing in a page of its own) sit above it.
        assert_eq!(store.history_page_rows(TOKEN), 256);
        let large = commit(store.create().unwrap(), 3000);
        let prompt = commit(store.create().unwrap(), 256);
        let system = prompt.checkpoint();
        let long = commit(prompt, 50);
        let short = commit(system.fork(), 20);
        let long_ranges = long.history_ranges(TOKEN);
        let system_ranges = system.history_ranges(TOKEN);
        let short_ranges = short.history_ranges(TOKEN);
        assert!(long.history_ranges(TOKEN)[0].0 >= store.history_slab_rows(TOKEN));
        drop(large);
        let before = (committed_extent(&store).0, device.memory_usage().charged);
        let occupied = store.occupied_rows(TOKEN);
        assert_eq!(occupied, 256 + 50 + 20);
        // Reclaim copies only into space held by the store and succeeds when
        // the device refuses all new allocations.
        device.set_memory_limit(Some(before.1));
        let released = bindings.shrink(ShrinkPolicy::Reclaim).unwrap();
        device.set_memory_limit(None);
        assert!(released > 0);
        assert!(committed_extent(&store).0 <= before.0);
        assert!(device.memory_usage().charged < before.1);
        assert_eq!(store.compactions(), Compactions::default());
        assert_eq!(read(long.history_ranges(TOKEN)), expected(306));
        assert_eq!(store.compactions().count, 0);
        assert!(device.memory_usage().charged < before.1);
        assert_eq!(store.occupied_rows(TOKEN), occupied);
        assert_eq!(long.history_ranges(TOKEN), long_ranges);
        assert_eq!(system.history_ranges(TOKEN), system_ranges);
        assert_eq!(short.history_ranges(TOKEN), short_ranges);
        assert_eq!(read(long.history_ranges(TOKEN)), expected(306));
        assert_eq!(read(short.history_ranges(TOKEN)), expected(276));
        assert_eq!(read(system.history_ranges(TOKEN)), expected(256));
        let demand = [RowDemand {
            domain: TOKEN,
            rows: store.history_slab_rows(TOKEN),
        }];
        let claim = store.growth_claim(&demand, 0).unwrap();
        let slab_bytes = store
            .history_slabs()
            .slab_bytes();
        assert_eq!(claim.minimum_bytes, slab_bytes);
        assert_eq!(claim.preferred_bytes, slab_bytes);
        bindings.provision(&demand, 0).unwrap();
        assert!(store
            .history_slabs()
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
            let mut store = token_store(
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
            let slab_rows = store.history_slab_rows(TOKEN);
            store.add_history_slabs(TOKEN, 2).unwrap();
            let _low = claim_rows(&store.arena(), 0, 1);
            let high = claim_rows(&store.arena(), 2 * slab_rows + 1, 1);
            let value = (19f32).to_le_bytes().repeat(4096);
            store
                .history_slabs()
                .region_rows(0, (2 * slab_rows + 1) as u64, 1)
                .unwrap()
                .write_from_host(&value)
                .unwrap();
            let slab_bytes = store
                .history_slabs()
                .slab_bytes();
            let charged = device.memory_usage().charged;
            device.set_memory_limit(Some(charged));
            assert_eq!(store.shrink(ShrinkPolicy::Reclaim).unwrap(), 2 * slab_bytes);
            device.set_memory_limit(None);
            assert_eq!(committed_extent(&store).0, slab_rows);
            assert_eq!(store.compactions().history_rows, 1);
            let moved = high.ranges()[0].0;
            assert!(moved < slab_rows);
            assert_eq!(
                store
                    .history_slabs()
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
            let mut bindings = token_store(
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
            let store = Rc::clone(&bindings);
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
            let released = bindings.shrink(ShrinkPolicy::Reclaim).unwrap();
            device.set_memory_limit(None);
            assert!(released > 0);
            assert_eq!(committed_extent(&store).1, 4);
            let after = placed();
            assert!(after.iter().all(|&slot| slot < 4));
            assert!(store.bank_placement_generation() > generation);
            assert_eq!(store.compactions().banks, 2);
            holds(after);
            assert_eq!(bindings.shrink(ShrinkPolicy::Reclaim).unwrap(), 0);
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
        let mut store = token_store(
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
        let mut store = token_store(
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
        assert_eq!(committed_extent(&store).1, 4);
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
        let mut store = token_store(
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
        let mut store = token_store(
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
        let mut store = token_store(
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
        store.add_history_slabs(TOKEN, 2).unwrap();
        store.add_bank_slabs(2).unwrap();
        let rows = store.history_slab_rows(TOKEN);
        let page = store.history_page_rows(TOKEN);
        let _low = claim_rows(&store.arena(), 0, rows);
        let sparse = claim_rows(&store.arena(), rows, 1);
        let _high = claim_rows(&store.arena(), 2 * rows, rows - page);
        let history_value = vec![41u8; 4096 * 4];
        store
            .history_slabs()
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
            .history_slabs()
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
                .history_slabs()
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
                .history_slabs()
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
        span_limit: usize,
        /// The store's reserved rows (whole pages).
        reserved: usize,
        decode_steps: usize,
        peak_committed: usize,
        released_to: usize,
        written: usize,
    }

    #[test]
    fn interleaved_decode_respects_span_limit_with_slab_backing() {
        // Reservations of 8 to 17 contexts, down to exactly one context per
        // request. Histories continue in another page when theirs is full,
        // and never exceed the store's span limit.
        for (active, contexts, steps) in [
            (8, 16, 800),
            (4, 5, 3000),
            (8, 9, 3000),
            (8, 8, 3000),
            (16, 17, 3000),
        ] {
            let run = serve_interleaved(active, contexts, steps);
            let reserved = run.reserved;
            assert_eq!(reserved, (contexts * 512 + 64).next_multiple_of(256));
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
                run.segments <= run.span_limit,
                "{active} requests in {contexts} contexts reached {} segments of {}",
                run.segments,
                run.span_limit
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
