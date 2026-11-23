//! Family-neutral accepted sequence state and its history ownership. A row
//! or bank is held exactly as long as a claim of it lives: the store's arenas
//! and bank pool are the one record of what its state occupies. Owned
//! advances can cross a submission boundary; their callers reconcile only
//! after physical completion has been observed.
mod advance;
mod bank;
mod codec;
mod domain;
mod layout;

pub use advance::{
    CodecConversionStep, OwnedAdvanceBindings, OwnedAdvanceResolution, OwnedChainAdvance,
    OwnedChainedAdvance, OwnedCodecAdvance, OwnedCodecBindings, OwnedStateAdvance,
    OwnedSuccessorAdvance, OwnedTailRelocation, TentativeAdvance,
};
pub use bank::{BankComponent, recurrent_bank_bytes};
pub use domain::{
    HistoryDomainId, HistoryDomainKind, HistoryDomainLayout, HistoryDomainPlan, HistoryReads,
    HistorySource, produced_domain, slab_regions, state_bytes,
};

pub use codec::{
    AFFINE_GROUP, AffineGeometry, Codec, CodecIdentity, CodecSpec, ComponentDescriptor,
    ComponentRole, HistoryEncoding, HistoryValue, KvCodec, LayerRef, LayoutError, PlaneDescriptor, PlaneName,
    VectorKind,
};
pub use layout::{
    HistoryGeometry, HistoryPages, MAX_HISTORY_SPANS, ModelStateLayout, SLAB_ROW_TILE, bank_regions, bank_slab_bytes, banks_per_slab, history_codec,
    history_geometry, history_pages, history_smallest_slabs, kv_codec_policy, recurrent_smallest_slab, smallest_slabs, window_pages,
};

use seismic::{DType, Device, Pool, SlabFill, SlabTensor, Tensor};
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
    /// A step's rows need a slab its provisioning did not back: the step
    /// fits once it provisions.
    Unbacked,
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
            Self::Unbacked => f.write_str("a step's rows need a slab its provisioning did not back"),
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

impl Error {
    /// Whether this refuses a step only until it provisions as usual: its
    /// rows' slabs are not backed, or every recurrent bank is claimed. Any
    /// other error is a fault.
    pub fn until_provisioned(&self) -> bool {
        matches!(self, Self::Unbacked | Self::BanksExhausted { .. })
    }
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
}
impl ComponentSpec {
    /// Physical bytes in one component of a recurrent state bank: a
    /// nonempty numeric tensor (floating state, or integer words such as the
    /// n-gram token window), whose zero bits are its zero seed.
    pub fn bytes(&self) -> Result<usize, String> {
        if self.shape.is_empty() || self.shape.contains(&0) || !self.dtype.is_numeric() {
            return Err("state components require nonempty numeric tensors".into());
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
/// history took, or where another live history ends at the same row and may
/// take it first (sibling forks), relocates that page's rows before it grows
/// ([`OwnedTailRelocation`]). A history takes the free page that begins at
/// its end when there is one; otherwise, as a fresh history does, the middle
/// page of the largest run of free pages, leaving the pages before it for
/// the history that ends there (a run at row 0 has none and is taken from
/// its start).
///
/// Placement sees only admitted slabs. A slab backed for steps queued behind
/// the one provisioned waits in `reserve`, empty and outside the free rows,
/// until a step's own demand admits it ([`HistoryDomain::admission`]): every
/// step then places its rows exactly as it would had it been provisioned
/// alone, whether it runs chained or not.
///
/// A row is referenced once by every history ([`Claims`]) covering it, so a
/// prefix is shared by any number of sequences and checkpoints at row
/// granularity, and a row returns to the free holes only when its last
/// history drops it. Every history is registered, so placement sees every
/// live history's end ([`Arena::tail_blocked`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlabState {
    Admitted,
    Reserved,
}

struct Arena {
    slab_rows: usize,
    page_rows: usize,
    /// Backed slabs and whether each is admitted to placement or held in reserve.
    slabs: BTreeMap<usize, SlabState>,
    free: Vec<(usize, usize)>,
    runs: BTreeMap<usize, Run>,
    referenced: usize,
    entries: BTreeMap<u64, Entry>,
    /// Reference counts of live entries ending at each row address, so
    /// [`Self::tail_blocked`] runs in O(log N) without scanning every entry.
    live_ends: BTreeMap<usize, usize>,
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
            slabs: (0..rows.div_ceil(slab_rows))
                .map(|slab| (slab, SlabState::Admitted))
                .collect(),
            free,
            runs: BTreeMap::new(),
            referenced: 0,
            entries: BTreeMap::new(),
            live_ends: BTreeMap::new(),
            next_entry: 0,
        }
    }

    fn is_backed(&self, slab: usize) -> bool {
        self.slabs.contains_key(&slab)
    }

    fn is_admitted(&self, slab: usize) -> bool {
        self.slabs.get(&slab) == Some(&SlabState::Admitted)
    }

    fn backed_slabs(&self) -> BTreeSet<usize> {
        self.slabs.keys().copied().collect()
    }

    fn reserved_slabs(&self) -> impl Iterator<Item = usize> + '_ {
        self.slabs
            .iter()
            .filter_map(|(&slab, &state)| (state == SlabState::Reserved).then_some(slab))
    }

    fn admit_all_reserved(&mut self, logical_rows: usize) {
        let reserved: Vec<usize> = self.reserved_slabs().collect();
        for slab in reserved {
            self.admit_slab(slab, logical_rows);
        }
    }

    fn add_live_end(&mut self, end: Option<usize>) {
        if let Some(end) = end {
            *self.live_ends.entry(end).or_insert(0) += 1;
        }
    }

    fn remove_live_end(&mut self, end: Option<usize>) {
        if let Some(end) = end {
            let count = self
                .live_ends
                .get_mut(&end)
                .expect("invariant violated: live history end missing from live_ends");
            *count -= 1;
            if *count == 0 {
                self.live_ends.remove(&end);
            }
        }
    }

    fn register(&mut self, ranges: Vec<(usize, usize)>, live: bool) -> u64 {
        let id = self.next_entry;
        self.next_entry += 1;
        let mut joined = Vec::with_capacity(ranges.len());
        append_ranges(&mut joined, ranges, self.slab_rows);
        let entry = Entry {
            ranges: joined,
            live,
        };
        if live {
            self.add_live_end(entry.end());
        }
        self.entries.insert(id, entry);
        id
    }

    fn set_entry_live(&mut self, id: u64, live: bool) {
        let entry = self.entries.get_mut(&id).expect("registered history");
        if entry.live == live {
            return;
        }
        let end = entry.end();
        entry.live = live;
        if live {
            self.add_live_end(end);
        } else {
            self.remove_live_end(end);
        }
    }

    fn remove_entry(&mut self, id: u64) -> Option<Entry> {
        let entry = self.entries.remove(&id)?;
        if entry.live {
            self.remove_live_end(entry.end());
        }
        Some(entry)
    }

    fn mutate_entry_ranges<T>(
        &mut self,
        id: u64,
        mutate: impl FnOnce(&mut Vec<(usize, usize)>, usize) -> T,
    ) -> T {
        let slab_rows = self.slab_rows;
        let entry = self.entries.get_mut(&id).expect("registered history");
        let live = entry.live;
        let old_end = entry.end();
        let result = mutate(&mut entry.ranges, slab_rows);
        let new_end = entry.end();
        if live && old_end != new_end {
            self.remove_live_end(old_end);
            self.add_live_end(new_end);
        }
        result
    }

    /// One past the last row of the page holding `row`.
    fn page_end(&self, row: usize) -> usize {
        row / self.page_rows * self.page_rows + self.page_rows
    }

    /// Number of free pages (pages without a referenced row). A page lies
    /// within one slab, so within one hole.
    fn free_page_count(&self) -> usize {
        let mut count = 0;
        for &(start, len) in &self.free {
            let end = start + len;
            let first = start.next_multiple_of(self.page_rows);
            if first < end {
                count += (end - first) / self.page_rows;
            }
        }
        count
    }

    /// Number of free pages lying in `slab`.
    fn free_pages_in_slab(&self, slab: usize) -> usize {
        let slab_start = slab * self.slab_rows;
        let slab_end = slab_start + self.slab_rows;
        let first_hole = self.free.partition_point(|&(start, _)| start < slab_start);
        let mut count = 0;
        for &(start, len) in &self.free[first_hole..] {
            if start >= slab_end {
                break;
            }
            let end = start + len;
            let first = start.next_multiple_of(self.page_rows);
            if first < end {
                count += (end - first) / self.page_rows;
            }
        }
        count
    }

    /// Rows a fresh history can take: the rows of every free page.
    fn available(&self) -> usize {
        self.free_page_count() * self.page_rows
    }

    /// `(hole, start, count)` of rows a history ending at `end` can take in
    /// place within its last page, or `None` if it has no partial last page or
    /// those rows are not free.
    fn in_place_hole(&self, end: Option<usize>) -> Option<(usize, usize, usize)> {
        let end = end.filter(|end| end % self.page_rows != 0)?;
        let hole = self
            .free
            .binary_search_by_key(&end, |&(start, _)| start)
            .ok()?;
        let (start, count) = self.free[hole];
        let rest = self.page_end(start) - start;
        (count >= rest).then_some((hole, start, rest))
    }

    /// Rows a history ending at `end` can take in place: the free rows that
    /// begin at its end, up to the end of its last page.
    fn in_place(&self, end: Option<usize>) -> usize {
        self.in_place_hole(end).map_or(0, |(_, _, count)| count)
    }

    /// Whether a history ending at `end` has a partial last page whose next
    /// row another history took.
    fn tail_taken(&self, end: Option<usize>) -> bool {
        end.is_some_and(|end| {
            end % self.page_rows != 0
                && end < self.page_end(end - 1)
                && self.in_place(Some(end)) == 0
        })
    }

    /// Whether rows growing at `tail` must first relocate their history's
    /// last page: the page has rows after its end, but another history took
    /// the next one, or another live history ends at the same row and may
    /// take it first (sibling forks of one state, which a launch can grow
    /// together).
    fn tail_blocked(&self, tail: Tail) -> bool {
        let Tail::Extending { end, history } = tail else {
            return false;
        };
        if end % self.page_rows == 0 {
            return false;
        }
        if self.tail_taken(Some(end)) {
            return true;
        }
        let total = self.live_ends.get(&end).copied().unwrap_or(0);
        let self_counts = self
            .entries
            .get(&history)
            .is_some_and(|entry| entry.live && entry.end() == Some(end));
        total > usize::from(self_counts)
    }

    /// Rows a history ending at `end` can take (see [`Arena::claim`]).
    fn claimable(&self, end: Option<usize>) -> usize {
        self.in_place(end) + self.available()
    }

    /// The `(hole, start)` of the free page a history ending at `end` takes
    /// next (see the placement rule above). Every maximal contiguous run of
    /// free pages within one slab lies inside a single coalesced hole in
    /// `self.free`.
    fn next_page(&self, end: Option<usize>) -> (usize, usize) {
        if let Some(end) = end.filter(|end| end % self.page_rows == 0) {
            let idx = self.free.partition_point(|&(start, _)| start <= end);
            if idx > 0 {
                let hole = idx - 1;
                let (start, count) = self.free[hole];
                if end + self.page_rows <= start + count {
                    return (hole, end);
                }
            }
        }
        let mut largest: Option<(usize, usize, usize)> = None;
        for (hole, &(start, len)) in self.free.iter().enumerate() {
            let end = start + len;
            let first = start.next_multiple_of(self.page_rows);
            if first >= end {
                continue;
            }
            let pages = (end - first) / self.page_rows;
            if pages > 0 && largest.is_none_or(|(_, _, count)| pages > count) {
                largest = Some((hole, first, pages));
            }
        }
        let (hole, first, count) = largest.expect("claimed rows never exceed the available rows");
        let start = if first == 0 {
            0
        } else {
            first + ((count - 1) / 2) * self.page_rows
        };
        (hole, start)
    }

    /// Claim `count` rows, in logical order, for a history whose end is row
    /// `after`: in place after its end within its last page, then whole free
    /// pages. The caller has checked that its tail is not blocked and that
    /// `count <= self.claimable(after)`.
    fn claim(&mut self, after: Option<usize>, count: usize) -> Vec<(usize, usize)> {
        debug_assert!(!self.tail_taken(after));
        let mut claimed = Vec::new();
        let mut remaining = count;
        let mut end = after;
        while remaining > 0 {
            let (hole, start, taken) = self.in_place_hole(end).unwrap_or_else(|| {
                let (hole, start) = self.next_page(end);
                (hole, start, self.page_rows)
            });
            debug_assert!(
                self.free[hole].0 <= start && start < self.free[hole].0 + self.free[hole].1
            );
            let piece = self.take(hole, start - self.free[hole].0, taken.min(remaining));
            claimed.push(piece);
            end = Some(piece.0 + piece.1);
            remaining -= piece.1;
        }
        claimed
    }

    /// Insert a free hole lying within a single slab into `self.free`, merging
    /// in place with adjacent holes in the same slab.
    fn insert_free(&mut self, start: usize, count: usize) {
        if count == 0 {
            return;
        }
        let pos = self.free.partition_point(|&(hole, _)| hole < start);
        let slab = start / self.slab_rows;
        let merge_prev = pos > 0 && {
            let (prev_start, prev_count) = self.free[pos - 1];
            prev_start + prev_count == start && prev_start / self.slab_rows == slab
        };
        let merge_next = pos < self.free.len() && {
            let (next_start, _) = self.free[pos];
            start + count == next_start && next_start / self.slab_rows == slab
        };
        match (merge_prev, merge_next) {
            (true, true) => {
                let next_count = self.free.remove(pos).1;
                self.free[pos - 1].1 += count + next_count;
            }
            (true, false) => {
                self.free[pos - 1].1 += count;
            }
            (false, true) => {
                let (_, next_count) = self.free[pos];
                self.free[pos] = (start, count + next_count);
            }
            (false, false) => {
                self.free.insert(pos, (start, count));
            }
        }
    }

    /// Back `slab`, held in reserve until a demand admits it.
    fn back_slab(&mut self, slab: usize) {
        assert!(self.slabs.insert(slab, SlabState::Reserved).is_none());
    }

    /// Back `slab` and admit it at once, as a step provisioned alone does.
    fn back_admitted(&mut self, slab: usize, logical_rows: usize) {
        self.back_slab(slab);
        self.admit_slab(slab, logical_rows);
    }

    /// Admit a reserved slab to placement: its rows (within `logical_rows`)
    /// join the free rows.
    fn admit_slab(&mut self, slab: usize, logical_rows: usize) {
        let state = self.slabs.get_mut(&slab).expect("admitted slab must be backed");
        assert_eq!(*state, SlabState::Reserved);
        *state = SlabState::Admitted;
        let start = slab * self.slab_rows;
        let end = (start + self.slab_rows).min(logical_rows);
        self.insert_free(start, end - start);
    }

    /// Return an admitted slab without referenced rows to reserve. A slab
    /// another history took rows of since stays admitted.
    fn withdraw_slab(&mut self, slab: usize) {
        if !self.empty_slab(slab) {
            return;
        }
        let start = slab * self.slab_rows;
        let end = start + self.slab_rows;
        self.free.retain(|&(row, _)| row < start || row >= end);
        let state = self.slabs.get_mut(&slab).expect("withdrawn slab must be backed");
        assert_eq!(*state, SlabState::Admitted);
        *state = SlabState::Reserved;
    }

    fn empty_slab(&self, slab: usize) -> bool {
        let start = slab * self.slab_rows;
        self.runs.range(start..start + self.slab_rows).next().is_none()
    }

    fn unback_empty_slab(&mut self, slab: usize) {
        let start = slab * self.slab_rows;
        let end = start + self.slab_rows;
        assert!(self.empty_slab(slab));
        assert!(self.slabs.remove(&slab).is_some());
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
        for row in starts {
            let run = self.runs.get_mut(&row).expect("run listed above");
            run.references -= 1;
            if run.references == 0 {
                let count = run.count;
                self.runs.remove(&row);
                self.referenced -= count;
                self.insert_free(row, count);
            }
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

}

/// One registered history: address ranges in logical order (physically
/// adjacent ranges joined). `live` marks the history of a sequence, which can
/// still grow; checkpoints and tentative rows are frozen.
#[derive(Debug, PartialEq, Eq)]
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

/// The rows `histories` fresh histories of `rows` rows each hold in an empty
/// domain of `geometry`, each filled whole before the next begins, as a
/// store fills requests that arrive together: per history, its `(start,
/// count)` ranges in logical order. Slabs are backed as the store backs
/// them, the lowest unbacked while the free pages are short, and every row
/// is placed by the store's own rule ([`Arena::claim`]): a history keeps
/// to itself, away from the others.
pub fn fresh_history_ranges(
    geometry: HistoryGeometry,
    histories: usize,
    rows: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut arena = Arena::new(0, geometry.slab_rows, geometry.page_rows);
    let pages = rows.div_ceil(geometry.page_rows);
    (0..histories)
        .map(|_| {
            while arena.free_page_count() < pages {
                arena.back_admitted(arena.slabs.len(), usize::MAX);
            }
            let mut ranges = Vec::new();
            append_ranges(&mut ranges, arena.claim(None, rows), geometry.slab_rows);
            ranges
        })
        .collect()
}

/// Where rows grow in one domain: either a new history with no predecessor
/// row, or extending a live history after its last row `end` (which never
/// contends with itself for its next row in [`Arena::tail_blocked`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tail {
    /// The rows of a new history (or an empty history with no rows yet).
    New,
    /// Growing after `end`, belonging to live history `history`.
    Extending { end: usize, history: u64 },
}

impl Tail {
    fn end(self) -> Option<usize> {
        match self {
            Self::New => None,
            Self::Extending { end, .. } => Some(end),
        }
    }
}

/// A history's claims: one reference on each row of its ranges, registered in
/// the arena, so the arena knows every history's ranges and end.
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

    /// Where this history grows: after its last row, or as a new history if empty.
    fn tail(&self) -> Tail {
        match self.end() {
            Some(end) => Tail::Extending {
                end,
                history: self.id,
            },
            None => Tail::New,
        }
    }

    fn set_live(&self, live: bool) {
        self.arena.borrow_mut().set_entry_live(self.id, live);
    }

    /// Unregister without releasing: the caller takes the references.
    fn into_ranges(self) -> Vec<(usize, usize)> {
        let entry = self
            .arena
            .borrow_mut()
            .remove_entry(self.id)
            .expect("registered history");
        entry.ranges
    }

    /// Move `next`'s rows (logically following this history's) onto its end.
    fn append(&mut self, next: Claims) {
        debug_assert!(Rc::ptr_eq(&self.arena, &next.arena));
        let ranges = next.into_ranges();
        self.arena
            .borrow_mut()
            .mutate_entry_ranges(self.id, |entry_ranges, slab_rows| {
                append_ranges(entry_ranges, ranges, slab_rows);
            });
    }

    /// Keep the first `keep` rows; return the remainder as a frozen history.
    fn split_off(&mut self, keep: usize) -> Claims {
        let tail = self
            .arena
            .borrow_mut()
            .mutate_entry_ranges(self.id, |entry_ranges, _| {
                let (head, tail) = split_ranges(entry_ranges, keep);
                *entry_ranges = head;
                tail
            });
        Claims::new(&self.arena, tail)
    }

    /// Release the first `rows` rows.
    fn drop_front(&mut self, rows: usize) {
        let head = self
            .arena
            .borrow_mut()
            .mutate_entry_ranges(self.id, |entry_ranges, _| {
                let (head, tail) = split_ranges(entry_ranges, rows);
                *entry_ranges = tail;
                head
            });
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
        if let Some(entry) = arena.remove_entry(self.id) {
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
    /// This capacity, refused where it has no active or in-flight request.
    pub fn validated(self) -> Result<Self, Error> {
        if self.active == 0 || self.in_flight == 0 {
            return Err(Error::Request(
                "bank pool requires positive active and in-flight capacity".into(),
            ));
        }
        Ok(self)
    }

    pub fn total(self) -> Result<usize, Error> {
        let this = self.validated()?;
        this.active
            .checked_add(this.in_flight)
            .and_then(|value| value.checked_add(this.retained))
            .ok_or_else(|| Error::Request("bank pool capacity overflow".into()))
    }

    /// The banks a store commits at construction: the pristine seed a first
    /// request starts from plus the successor banks one request holds in
    /// flight (`in_flight` is the requests' successors in all), so its first
    /// step of any kind grows no slab and the load's startup charge holds
    /// everything its warm-up forward and first prefill allocate. Further
    /// requests' banks grow the store by slabs the in-process guarantee covers.
    pub fn startup_banks(self) -> Result<usize, Error> {
        let this = self.validated()?;
        Ok(1 + this.in_flight.div_ceil(this.active))
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

struct BankPoolInner {
    /// The writable banks the store holds at most.
    capacity: usize,
    /// One past the highest committed slot.
    committed: Cell<usize>,
    /// The claimed slots, the zero seed's among them for the store's life.
    claimed: RefCell<BTreeSet<usize>>,
    /// Committed slots whose slab is released.
    unavailable: RefCell<BTreeSet<usize>>,
}

/// A claim on one row of every recurrent component: its slot, fixed for the
/// claim's life, since no bank moves.
struct BankClaim {
    pool: Rc<BankPoolInner>,
    slot: usize,
}

impl Drop for BankClaim {
    fn drop(&mut self) {
        if self.slot != ZERO_SEED_BANK {
            assert!(self.pool.claimed.borrow_mut().remove(&self.slot), "a live bank's slot is claimed");
        }
    }
}

#[derive(Clone)]
struct BankHandle(Rc<BankClaim>);

impl BankHandle {
    fn index(&self) -> usize {
        self.0.slot
    }
}

/// The store's bank slots and their claims. The zero seed is permanently
/// claimed; writable slots are acquired lowest first.
struct BankPool {
    inner: Rc<BankPoolInner>,
}

impl BankPool {
    fn new(capacity: BankCapacity, committed: usize) -> Result<(Self, BankHandle), Error> {
        assert!(committed > ZERO_SEED_BANK, "a store commits its zero seed's slot");
        let inner = Rc::new(BankPoolInner {
            capacity: capacity.total()?,
            committed: Cell::new(committed),
            claimed: RefCell::new(BTreeSet::from([ZERO_SEED_BANK])),
            unavailable: RefCell::new(BTreeSet::new()),
        });
        let seed = BankHandle(Rc::new(BankClaim {
            pool: inner.clone(),
            slot: ZERO_SEED_BANK,
        }));
        Ok((Self { inner }, seed))
    }

    fn acquire(&self) -> Result<BankHandle, Error> {
        let slot = {
            let claimed = self.inner.claimed.borrow();
            let unavailable = self.inner.unavailable.borrow();
            (1..self.committed())
                .find(|slot| !claimed.contains(slot) && !unavailable.contains(slot))
                .ok_or(Error::BanksExhausted {
                    capacity: self.inner.capacity,
                })?
        };
        self.inner.claimed.borrow_mut().insert(slot);
        Ok(BankHandle(Rc::new(BankClaim {
            pool: self.inner.clone(),
            slot,
        })))
    }

    fn committed(&self) -> usize {
        self.inner.committed.get()
    }

    /// The claimed slots, the zero seed's among them.
    fn claimed(&self) -> BTreeSet<usize> {
        self.inner.claimed.borrow().clone()
    }

    fn available(&self) -> usize {
        self.committed() - self.inner.claimed.borrow().len() - self.inner.unavailable.borrow().len()
    }

    fn resize(&self, from: usize, to: usize) {
        assert_eq!(self.committed(), from);
        assert!(
            self.inner.claimed.borrow().iter().all(|&slot| slot < to),
            "a resize keeps every claimed slot"
        );
        self.inner.committed.set(to);
        self.inner
            .unavailable
            .borrow_mut()
            .retain(|&slot| slot < to);
    }

    fn mark_slab_unavailable(&self, start: usize, end: usize) {
        assert!(
            self.inner
                .claimed
                .borrow()
                .iter()
                .all(|&slot| slot < start || slot >= end),
            "a released bank slab holds no claimed bank"
        );
        self.inner.unavailable.borrow_mut().extend(start..end);
    }

    fn mark_slab_available(&self, start: usize, end: usize) {
        let mut unavailable = self.inner.unavailable.borrow_mut();
        for slot in start..end {
            unavailable.remove(&slot);
        }
    }
}

mod backing {
    use super::{Error, PlaneBuffer};
    use seismic::{SlabTensor, Tensor};
    use std::rc::Rc;

    /// One stored domain's history slab tensor and one past its highest backed
    /// row.
    pub(super) struct HistorySlabs {
        pub(super) slabs: SlabTensor,
        pub(super) rows: usize,
    }

    /// All stored history domains' slab tensors paired with their cached
    /// logical plane views. Every slab mutation goes through [`Self::mutate_domain`],
    /// invalidating cached views by construction.
    pub(super) struct HistoryBacking {
        slabs: Vec<HistorySlabs>,
        views: Option<Vec<PlaneBuffer>>,
    }

    impl HistoryBacking {
        pub(super) fn slabs(&self) -> &[HistorySlabs] {
            &self.slabs
        }

        pub(super) fn mutate_domain<T>(
            &mut self,
            index: usize,
            f: impl FnOnce(&mut HistorySlabs) -> T,
        ) -> T {
            self.views = None;
            f(&mut self.slabs[index])
        }

        pub(super) fn views_or_build(
            &mut self,
            build: impl FnOnce(&[HistorySlabs]) -> Result<Vec<PlaneBuffer>, Error>,
        ) -> Result<Vec<PlaneBuffer>, Error> {
            if let Some(views) = &self.views {
                return Ok(views.clone());
            }
            if self.slabs.iter().all(|history| history.rows == 0) {
                return Ok(Vec::new());
            }
            let views = build(&self.slabs)?;
            self.views = Some(views.clone());
            Ok(views)
        }
    }

    /// Recurrent bank slab tensor, its committed bank extent, and its cached
    /// logical arena views. Exists only when the store has recurrent
    /// components, so `banks` cannot exist without `slabs`, and every slab
    /// mutation goes through [`Self::mutate_slabs`], invalidating cached views
    /// by construction.
    pub(super) struct RecurrentBacking {
        slabs: SlabTensor,
        banks: usize,
        views: Option<Rc<[Tensor]>>,
    }

    impl RecurrentBacking {
        pub(super) fn new(slabs: SlabTensor, banks: usize) -> Self {
            Self {
                slabs,
                banks,
                views: None,
            }
        }

        pub(super) fn slabs(&self) -> &SlabTensor {
            &self.slabs
        }

        pub(super) fn banks(&self) -> usize {
            self.banks
        }

        pub(super) fn mutate_slabs<T>(
            &mut self,
            f: impl FnOnce(&mut SlabTensor, &mut usize) -> T,
        ) -> T {
            self.views = None;
            f(&mut self.slabs, &mut self.banks)
        }

        pub(super) fn views(&mut self, component_count: usize) -> Rc<[Tensor]> {
            if let Some(views) = &self.views {
                return views.clone();
            }
            let views: Rc<[Tensor]> = (0..component_count)
                .map(|index| self.slabs.logical_region(index).expect("backed recurrent slab"))
                .collect();
            self.views = Some(views.clone());
            views
        }
    }

    /// Slab-backed history rows (one slab tensor per stored domain) and
    /// recurrent banks, unified with their cached logical views. Graphs are
    /// sealed over the logical shapes; only backed rows and banks are handed
    /// out.
    pub(super) struct StoreBacking {
        pub(super) history: HistoryBacking,
        pub(super) recurrent: Option<RecurrentBacking>,
    }

    impl StoreBacking {
        pub(super) fn new(
            history: Vec<HistorySlabs>,
            recurrent: Option<RecurrentBacking>,
        ) -> Self {
            Self {
                history: HistoryBacking {
                    slabs: history,
                    views: None,
                },
                recurrent,
            }
        }

        pub(super) fn all_slabs(&self) -> impl Iterator<Item = &SlabTensor> {
            self.history
                .slabs
                .iter()
                .map(|history| &history.slabs)
                .chain(self.recurrent.iter().map(RecurrentBacking::slabs))
        }

        /// Drop cached views (which hold slab slots), count owned history and
        /// recurrent slabs separately, and drop every slab tensor before
        /// returning granules to their pools.
        pub(super) fn drop_and_count_owned_slabs(
            &mut self,
            count_owned: impl Fn(&SlabTensor) -> u64,
        ) -> (u64, u64) {
            self.history.views = None;
            if let Some(recurrent) = &mut self.recurrent {
                recurrent.views = None;
            }
            let history = self.history.slabs.iter().map(|h| count_owned(&h.slabs)).sum();
            let recurrent = self.recurrent.as_ref().map_or(0, |r| count_owned(r.slabs()));
            self.history.slabs.clear();
            self.recurrent = None;
            (history, recurrent)
        }
    }
}

use backing::{HistorySlabs, RecurrentBacking, StoreBacking};

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
    /// The slabs to admit so that the free pages hold a demand of `rows`
    /// page-rounded rows (see [`StateStore::page_demand`]): the lowest of
    /// those outside placement, reserved or unbacked, as a step provisioned
    /// alone would back them. Steps provisioned ahead back the same slabs
    /// in the same order, so each step admits what it would have grown.
    fn admission(&self, rows: usize) -> Result<Vec<usize>, Error> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        let arena = self.arena.borrow();
        let pages = rows.div_ceil(self.page_rows);
        let free = arena.free_page_count();
        let shortage = pages.saturating_sub(free);
        let mut slabs = Vec::new();
        let mut added_pages = 0;
        for slab in 0..self.capacity.div_ceil(self.slab_rows) {
            if added_pages >= shortage {
                break;
            }
            if !arena.is_admitted(slab) {
                let start = slab * self.slab_rows;
                added_pages +=
                    ((start + self.slab_rows).min(self.capacity) - start) / self.page_rows;
                slabs.push(slab);
            }
        }
        if added_pages < shortage {
            let required = (rows as u64)
                .checked_mul(self.row_bytes)
                .ok_or(LayoutError::ArithmeticOverflow("required history bytes"))?;
            let available_bytes = (added_pages + free)
                .checked_mul(self.page_rows)
                .and_then(|rows| (rows as u64).checked_mul(self.row_bytes))
                .ok_or(LayoutError::ArithmeticOverflow("available history bytes"))?;
            return Err(Error::Capacity {
                required,
                available_bytes,
            });
        }
        Ok(slabs)
    }

    /// Unbacked slabs among `admission(rows)`: the slabs to add.
    fn growth_slabs(&self, rows: usize) -> Result<usize, Error> {
        let arena = self.arena.borrow();
        Ok(self
            .admission(rows)?
            .into_iter()
            .filter(|&slab| !arena.is_backed(slab))
            .count())
    }

    /// One past the highest row of the backed slabs.
    fn backed_extent(&self, slabs: &SlabTensor) -> usize {
        slabs
            .backed()
            .into_iter()
            .map(|index| ((index + 1) * self.slab_rows).min(self.capacity))
            .max()
            .unwrap_or(0)
    }

    /// The first row a history at `position`, trimmed at `floor`,
    /// references, in the domain's rows: a position, or for Block the first
    /// block with no trimmed position, clamped to the completed blocks
    /// (`position / rate`).
    fn history_start(&self, position: usize, floor: usize) -> usize {
        match self.kind {
            HistoryDomainKind::Block { rate } => floor.div_ceil(rate).min(position / rate),
            kind => kind.retained_from(position).max(floor),
        }
    }

    /// The slabs a shrink keeping room for `rows` rows keeps, every
    /// reserved slab first admitted (a shrink runs with no step queued, its
    /// caller orphaning the queue, so none waits for a step's admission):
    /// each slab with a referenced row and, while the free pages of those
    /// fall short of the rows' pages, the lowest empty slabs, as
    /// [`Self::admission`] would back them again.
    fn retained_slabs(&self, rows: usize) -> BTreeSet<usize> {
        let mut arena = self.arena.borrow_mut();
        arena.admit_all_reserved(self.capacity);
        let mut keep = arena
            .slabs
            .keys()
            .copied()
            .filter(|&slab| !arena.empty_slab(slab))
            .collect::<BTreeSet<_>>();
        let mut room = keep
            .iter()
            .map(|&slab| arena.free_pages_in_slab(slab))
            .sum::<usize>();
        let pages = rows.div_ceil(self.page_rows);
        for &slab in arena.slabs.keys() {
            if room >= pages {
                break;
            }
            if keep.insert(slab) {
                room += arena.free_pages_in_slab(slab);
            }
        }
        keep
    }
}

/// Each stored history domain's rows are a shared arena; recurrent
/// components are arenas of banks, and each accepted version is one
/// immutable bank. A checkpoint retains both without copying tensor
/// contents.
///
/// Granule pools backing a [`StateStore`]: one pool for KV history slabs and
/// an optional dedicated pool for recurrent-state bank slabs (so large
/// recurrent banks on hybrid models never inflate KV history slabs' granule).
#[derive(Clone)]
pub struct StatePools {
    pub history: Pool,
    pub recurrent: Option<Pool>,
}

impl From<Pool> for StatePools {
    fn from(history: Pool) -> Self {
        Self { history, recurrent: None }
    }
}

/// Bytes freed from each state pool by [`StoreBindings::shrink_by_pool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShrinkRelease {
    pub history: u64,
    pub recurrent: u64,
}

impl ShrinkRelease {
    pub fn total(self) -> u64 {
        self.history + self.recurrent
    }
}

/// Each domain's reserved rows and the bank capacity are reservations
/// sealed into graphs. Backing slabs are added as demand grows and released
/// when empty, only through the store's [`StoreBindings`].
pub struct StateStore {
    device: Device,
    /// The granule pool for history slabs: each backed history slab is one of
    /// its granules.
    pool: Pool,
    /// The granule pool for recurrent bank slabs (when `component_specs` is
    /// non-empty); may be distinct from `pool` so large recurrent banks do not
    /// inflate history granules.
    recurrent_pool: Option<Pool>,
    context_capacity: usize,
    domains: Vec<HistoryDomain>,
    /// Layers of Shared domains and the stored layers they read.
    shared: Vec<(LayerRef, HistorySource)>,
    /// The history reads of a launch row.
    reads: HistoryReads,
    bank_capacity: BankCapacity,
    recurrent_bank_bytes: u64,
    bank_slab_banks: usize,
    component_specs: Vec<ComponentSpec>,
    backing: RefCell<StoreBacking>,
    banks: BankPool,
    zero_seed: BankHandle,
}

/// One store's demand for `rows` free rows in one stored history domain.
/// An advance of `n` rows demands `n` rows in every domain of its store
/// ([`SequenceState::demands`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowDemand {
    pub domain: HistoryDomainId,
    pub rows: usize,
}

/// What the admitted requests' next steps claim of a store
/// ([`StoreBindings::shrink`]): each domain's page-rounded rows
/// (`SequenceState::demands`) and successor banks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NextSteps {
    pub rows: Vec<RowDemand>,
    pub banks: usize,
}

/// The one right to change a store's slab bindings: adding and releasing
/// slabs. [`StateStore::new`] hands out exactly one; it is never cloned.
/// Row and bank claims (advances, relocations, checkpoints) need no binding
/// right.
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
    /// Add backing slabs so the free rows hold one step's `demands` and
    /// `banks` banks are free, with nothing queued behind it. A refused
    /// allocation returns a capacity error without publishing tentative
    /// backing or changing accepted state.
    pub fn provision(&mut self, demands: &[RowDemand], banks: usize) -> Result<(), Error> {
        self.store
            .provision_with_growth(demands, &[], banks, GrowthChoice::Preferred)
    }

    /// Back slabs for one step's `demands` and, held in reserve, for the
    /// rows `queued` of the steps queued behind it, which their own
    /// successors admit ([`TentativeAdvance::successors`]); admit the
    /// step's.
    pub fn provision_with_growth(
        &mut self,
        demands: &[RowDemand],
        queued: &[RowDemand],
        banks: usize,
        choice: GrowthChoice,
    ) -> Result<(), Error> {
        self.store
            .provision_with_growth(demands, queued, banks, choice)
    }

    /// Release every slab beyond the room `next` claims: what the admitted
    /// requests' next steps claim is not surplus, and nothing else is kept.
    /// Moves no row or bank ([`StateStore::shrink`]). Returns the bytes of
    /// the granules freed, which the caller returns to the OS through
    /// [`Pool::return_freed`], the one mechanism that empties sparse blocks.
    pub fn shrink(&mut self, next: &NextSteps) -> Result<u64, Error> {
        self.store.shrink(next)
    }

    /// Like [`Self::shrink`], returning history and recurrent bytes separately
    /// so callers return each to its own pool.
    pub fn shrink_by_pool(&mut self, next: &NextSteps) -> Result<ShrinkRelease, Error> {
        self.store.shrink_by_pool(next)
    }
}

/// The bytes of the slabs a state growth adds: each slab's bytes, which a
/// granule of the store's pool holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateGrowthClaim {
    pub minimum_bytes: u64,
    pub preferred_bytes: u64,
}

impl Drop for StateStore {
    /// The store's slabs leave its pool(s) with it, and the bytes their
    /// granules held return through each pool's one return path
    /// ([`Pool::return_freed`]): dropping a store returns its memory as
    /// releasing its slabs does, granules a flight still reads returning
    /// as it completes. A view of a slab's rows still holds its granule.
    ///
    /// # Panics
    /// Where the pool cannot return them: a device fault, which no caller
    /// of a drop can take. Dropped while its thread already panics, the
    /// fault is logged instead, the bytes stay owed to the OS in the pool,
    /// and the panic under way is the report: a second would abort the
    /// process before it reports.
    fn drop(&mut self) {
        // The cached logical views hold the slab stores' slots.
        let (history_slabs, recurrent_slabs) = self
            .backing
            .get_mut()
            .drop_and_count_owned_slabs(owned_slabs);
        let return_pool = |pool: &Pool, slabs: u64| {
            let freed = slabs * pool.granule();
            if freed == 0 {
                return;
            }
            if let Err(error) = pool.return_freed(freed) {
                if std::thread::panicking() {
                    tracing::error!(freed, %error, "a state store dropped while panicking could not return its bytes to the OS: they stay owed");
                    return;
                }
                panic!("a dropped state store's {freed} bytes could not return to the OS: {error}");
            }
        };
        return_pool(&self.pool, history_slabs);
        if let Some(recurrent_pool) = &self.recurrent_pool {
            return_pool(recurrent_pool, recurrent_slabs);
        }
    }
}

/// The slabs of `slabs` that are granules of its pool: every backed slot
/// past the lent ones, which view their owner's storage
/// ([`SlabTensor::lent`]).
fn owned_slabs(slabs: &SlabTensor) -> u64 {
    let lent = slabs.lent();
    slabs.backed().into_iter().filter(|slot| *slot >= lent).count() as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrowthChoice {
    Minimum,
    Preferred,
}

impl StateStore {
    /// A store of the given history domains and recurrent bank components.
    /// Every domain's reserved rows cover its row limit: the context for
    /// Token, `n` plus one advance of at most `max_advance` rows for
    /// Window(n). Each stored domain starts with one backed slab. Every slab
    /// is a granule of `pool`, the pool of the store on `device` the state
    /// lies in, and holds as many whole pages or banks as fit one
    /// ([`HistoryDomainLayout::geometry`], [`banks_per_slab`]): a granule
    /// below a page or a bank is refused. The store's one
    /// [`StoreBindings`] comes with it.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        charges: &seismic::Charges,
        pools: impl Into<StatePools>,
        context_capacity: usize,
        max_advance: usize,
        domains: Vec<HistoryDomainPlan>,
        component_specs: Vec<ComponentSpec>,
        bank_capacity: BankCapacity,
    ) -> Result<(Rc<Self>, StoreBindings), Error> {
        let StatePools { history: pool, recurrent: recurrent_pool_opt } = pools.into();
        if context_capacity == 0 {
            return Err(Error::Request(
                "history capacity must fit a positive sequence context".into(),
            ));
        }
        let (stored, shared) = domain::validate_domains(domains, context_capacity, max_advance)?;
        let reads = HistoryReads::new(&stored, &shared);
        let recurrent_bank_bytes = recurrent_bank_bytes(&component_specs)?;
        let regions = bank_regions(&component_specs);
        let recurrent_pool = (!component_specs.is_empty())
            .then(|| recurrent_pool_opt.unwrap_or_else(|| pool.clone()));
        let bank_slab_banks = match &recurrent_pool {
            None => 0,
            Some(rpool) => banks_per_slab(&regions, rpool.granule()).map_err(Error::Request)?,
        };
        let reserved_banks = bank_capacity.storage_total()?;
        // Banks of a store without recurrent components have no storage.
        // When `bank_slab_banks == 1`, slab 0 holds `ZERO_SEED_BANK` alone:
        // kernels never write it, and an unbacked slab slot already maps to
        // the pool's zeroed `absent` granule, so slab 0 stays unbacked.
        let (committed_banks, startup_slabs, bank_start_slab) = if component_specs.is_empty() {
            (reserved_banks, 0, 0)
        } else if bank_slab_banks == 1 {
            let writable = bank_capacity.startup_banks()?.saturating_sub(1);
            ((1 + writable).min(reserved_banks), writable, 1)
        } else {
            let slabs = bank_capacity.startup_banks()?.div_ceil(bank_slab_banks);
            ((slabs * bank_slab_banks).min(reserved_banks), slabs, 0)
        };
        let (banks, zero_seed) = BankPool::new(bank_capacity, committed_banks)?;
        let recurrent = match &recurrent_pool {
            None => None,
            Some(rpool) => {
                let slabs = SlabTensor::in_pool(
                    charges,
                    rpool,
                    bank_slab_banks as u64,
                    reserved_banks as u64,
                    &regions,
                )?;
                // Kernels read a bank before any writes it.
                if startup_slabs > 0 {
                    slabs.add_slabs_from(bank_start_slab, startup_slabs, SlabFill::Zeroed)?;
                }
                Some(RecurrentBacking::new(slabs, committed_banks))
            }
        };
        let mut history = Vec::with_capacity(stored.len());
        let mut domains = Vec::with_capacity(stored.len());
        for domain in stored {
            let geometry = domain.pages.in_granule(domain.row_bytes, pool.granule()).map_err(Error::Request)?;
            let slabs = SlabTensor::in_pool(
                charges,
                &pool,
                geometry.slab_rows as u64,
                domain.logical_rows as u64,
                &slab_regions(&domain.components),
            )?;
            slabs.add_slab(SlabFill::Zeroed)?;
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
            device: charges.device(),
            pool,
            recurrent_pool,
            context_capacity,
            domains,
            shared,
            reads,
            bank_capacity,
            recurrent_bank_bytes,
            bank_slab_banks,
            component_specs,
            backing: RefCell::new(StoreBacking::new(history, recurrent)),
            banks,
            zero_seed,
        });
        Ok((store.clone(), StoreBindings { store }))
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
        self.backing
            .borrow_mut()
            .recurrent
            .as_mut()
            .map_or_else(|| Rc::from([]), |recurrent| recurrent.views(self.component_specs.len()))
    }
    /// Banks reserved in each recurrent arena, including the zero seed.
    pub fn recurrent_bank_count(&self) -> Result<usize, Error> {
        self.bank_capacity.storage_total()
    }
    /// One past the highest backed row of a domain. The extent may include
    /// unbacked rows after an interior slab is freed;
    /// [`StateStore::committed_bytes`] measures the actual backing.
    pub fn committed_rows(&self, domain: HistoryDomainId) -> usize {
        self.backing.borrow().history.slabs()[domain.0].rows
    }
    /// One past the highest backed recurrent bank.
    pub fn committed_banks(&self) -> usize {
        self.backing
            .borrow()
            .recurrent
            .as_ref()
            .map_or_else(|| self.banks.committed(), RecurrentBacking::banks)
    }
    /// The granule pool its history slabs are drawn from.
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// The granule pool its recurrent bank slabs are drawn from, when the
    /// store has recurrent components.
    pub fn recurrent_pool(&self) -> Option<&Pool> {
        self.recurrent_pool.as_ref()
    }

    fn bank_start_slab(&self) -> usize {
        usize::from(self.bank_slab_banks == 1)
    }

    /// Physical bytes of the committed backing: each slab store's address
    /// table and its backed slabs' bytes. The slabs are granules of the
    /// store's pool(s), which are charged for whole blocks of them.
    pub fn committed_bytes(&self) -> u64 {
        let backing = self.backing.borrow();
        let tables = backing
            .all_slabs()
            .map(SlabTensor::address_table_bytes)
            .sum::<u64>();
        tables + self.backed_slab_bytes()
    }

    /// Bytes of the backed slabs alone, granules of the store's pool(s): the
    /// committed backing less its address tables, which their own claim
    /// commits.
    pub fn backed_slab_bytes(&self) -> u64 {
        let backing = self.backing.borrow();
        backing
            .all_slabs()
            .map(|slabs| slabs.backed_count() as u64 * slabs.pool().granule())
            .sum()
    }

    /// Bytes of the backed slabs that still hold at least one referenced
    /// history row or claimed recurrent bank (including the zero seed slab
    /// when backed, matching [`Self::shrink`]), in each pool's granule units.
    /// Used by `StateSlabs::settle` so `backed`, `referenced`, and `written`
    /// share the same slab-granule units.
    pub fn referenced_slab_bytes(&self) -> u64 {
        let backing = self.backing.borrow();
        let history_slabs: u64 = self
            .domains
            .iter()
            .zip(backing.history.slabs())
            .map(|(domain, history)| {
                let arena = domain.arena.borrow();
                history.slabs.backed().into_iter().filter(|&slab| !arena.empty_slab(slab)).count() as u64
            })
            .sum();
        let recurrent_bytes: u64 = backing.recurrent.as_ref().map_or(0, |recurrent| {
            let occupied = self.banks.claimed();
            let reserved = self.bank_capacity.storage_total().unwrap_or(usize::MAX);
            let slabs = recurrent
                .slabs()
                .backed()
                .into_iter()
                .filter(|&slab| {
                    let start = slab * self.bank_slab_banks;
                    let end = (start + self.bank_slab_banks).min(reserved);
                    (start..end).any(|bank| occupied.contains(&bank))
                })
                .count() as u64;
            slabs * recurrent.slabs().pool().granule()
        });
        history_slabs * self.pool.granule() + recurrent_bytes
    }

    /// The state its holders reference: every row some history claims and
    /// every recurrent bank some holder claims (the model's seed bank
    /// apart), at their bytes.
    pub fn referenced_bytes(&self) -> u64 {
        let history = self
            .domains
            .iter()
            .map(|domain| domain.arena.borrow().referenced as u64 * domain.row_bytes)
            .sum::<u64>();
        let banks = self.banks.claimed().len() as u64 - 1;
        history + banks * self.recurrent_bank_bytes
    }

    /// The state the store holds for its holders: what they reference and
    /// the model's seed bank. The rest of [`Self::committed_bytes`] is
    /// backing no claim uses.
    pub fn occupied_bytes(&self) -> u64 {
        self.referenced_bytes() + self.recurrent_bank_bytes
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
    /// Where `layer`'s attended history lives: its own stored domain (a
    /// pooled source's Block rows), or the source layer and domain a Shared
    /// domain binds. `None` for a layer without attended history in this
    /// store.
    pub fn history_source(&self, layer: LayerRef) -> Option<HistorySource> {
        self.domains
            .iter()
            .position(|domain| {
                domain
                    .components
                    .iter()
                    .any(|component| component.layer == layer && component.attended())
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
    /// The stored domains of the produced rows of `layer`
    /// (`LayerRef::IndexKeys`, `Pooled` or their head forms): the Window of
    /// its raw projections or incomplete group (none for keys of a pooled
    /// source's rows, nor at rate 1), and the Block of its rows or keys, each
    /// one history read of a launch row (`history_reads`). `None` for a layer
    /// the store keeps none of.
    pub fn produced_domains(
        &self,
        layer: LayerRef,
    ) -> Option<(Option<HistoryDomainId>, HistoryDomainId)> {
        let find = |block: bool| {
            self.domains
                .iter()
                .position(|domain| {
                    matches!(domain.kind, HistoryDomainKind::Block { .. }) == block
                        && domain.components.iter().any(|component| component.layer == layer)
                })
                .map(HistoryDomainId)
        };
        Some((find(false), find(true)?))
    }
    /// Whether `domain` holds produced rows ([`produced_domain`]), which a
    /// launch's rows read as stored once written.
    pub fn is_produced_domain(&self, domain: HistoryDomainId) -> bool {
        let domain = &self.domains[domain.0];
        produced_domain(domain.kind, &domain.components)
    }
    /// The history reads of a launch row.
    pub fn history_reads(&self) -> &HistoryReads {
        &self.reads
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
            .slabs()
            .iter()
            .any(|history| history.rows != 0)
    }

    /// A read-only claim for the slabs a launch must add for its `demands`
    /// and the rows `queued` behind it. Minimum and preferred are equal
    /// while growth follows exact slab demand.
    pub fn growth_claim(
        &self,
        demands: &[RowDemand],
        queued: &[RowDemand],
        banks: usize,
    ) -> Result<StateGrowthClaim, Error> {
        let demands = [demands, queued].concat();
        Ok(StateGrowthClaim {
            minimum_bytes: self.growth_bytes(&demands, banks, GrowthChoice::Minimum)?,
            preferred_bytes: self.growth_bytes(&demands, banks, GrowthChoice::Preferred)?,
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
            .zip(backing.history.slabs())
            .try_fold(0u64, |total, (&slabs, history)| {
                history
                    .slabs
                    .pool()
                    .granule()
                    .checked_mul(slabs as u64)
                    .and_then(|bytes| total.checked_add(bytes))
            })
            .ok_or_else(|| Error::Request("history slab claim overflows".into()))?;
        let bank_bytes = backing
            .recurrent
            .as_ref()
            .map_or(0, |recurrent| recurrent.slabs().pool().granule())
            .checked_mul(bank_slabs as u64)
            .ok_or_else(|| Error::Request("bank slab claim overflows".into()))?;
        history_bytes
            .checked_add(bank_bytes)
            .ok_or_else(|| Error::Request("state slab claim overflows".into()))
    }

    fn provision_with_growth(
        &self,
        demands: &[RowDemand],
        queued: &[RowDemand],
        banks: usize,
        choice: GrowthChoice,
    ) -> Result<(), Error> {
        let bank_slabs = self.bank_growth_slabs(banks)?;
        let plans = self.history_growth_plans(&[demands, queued].concat(), choice)?;
        // Growth of every domain and the banks is one fallible operation.
        let before = self
            .domains
            .iter()
            .map(|domain| domain.arena.borrow().backed_slabs())
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
        // Every slab the step's demands admit lies in the prefix backed for
        // them and the queued rows.
        self.admit(demands).map(drop)
    }

    /// Restore every domain's published history backing when a later part
    /// of one aggregate growth fails. No advance has claimed the newly added
    /// rows.
    fn rollback_history_growth(&self, before: &[BTreeSet<usize>]) -> Result<(), Error> {
        let mut freed = 0u64;
        for (index, (domain, before)) in self.domains.iter().zip(before).enumerate() {
            let added = domain
                .arena
                .borrow()
                .slabs
                .keys()
                .copied()
                .filter(|slab| !before.contains(slab))
                .collect::<Vec<_>>();
            if added.is_empty() {
                continue;
            }
            self.backing.borrow_mut().history.mutate_domain(
                index,
                |history| -> Result<(), Error> {
                    for slab in &added {
                        history.slabs.free_slab(*slab)?;
                        domain.arena.borrow_mut().unback_empty_slab(*slab);
                    }
                    history.rows = domain.backed_extent(&history.slabs);
                    Ok(())
                },
            )?;
            freed += added.len() as u64;
        }
        self.pool.return_freed(freed * self.pool.granule())?;
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
            .expect("recurrent layout has slabs")
            .slabs();
        let backed = slabs.backed().into_iter().collect::<BTreeSet<_>>();
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
        self.domains
            .iter()
            .zip(self.domain_demands(demands)?)
            .map(|(domain, rows)| domain.growth_slabs(rows))
            .collect()
    }

    /// Each stored domain's total of `demands`.
    fn domain_demands(&self, demands: &[RowDemand]) -> Result<Vec<usize>, Error> {
        let mut totals = vec![0usize; self.domains.len()];
        for demand in demands {
            let Some(total) = totals.get_mut(demand.domain.0) else {
                return Err(Error::Request(format!(
                    "row demand names absent history domain {:?}",
                    demand.domain
                )));
            };
            *total = total
                .checked_add(demand.rows)
                .ok_or(LayoutError::ArithmeticOverflow("domain row demand"))?;
        }
        Ok(totals)
    }

    /// Admit the slabs one step's `demands` need to placement
    /// ([`HistoryDomain::admission`]), before the step claims its rows.
    /// Admits nothing unless every one of them is backed: admitting never
    /// grows the store. Returns what it admitted.
    fn admit(&self, demands: &[RowDemand]) -> Result<Vec<(HistoryDomainId, usize)>, Error> {
        let mut admitted = Vec::new();
        for (index, (domain, rows)) in self
            .domains
            .iter()
            .zip(self.domain_demands(demands)?)
            .enumerate()
        {
            for slab in domain.admission(rows)? {
                if !domain.arena.borrow().is_backed(slab) {
                    return Err(Error::Unbacked);
                }
                admitted.push((HistoryDomainId(index), slab));
            }
        }
        for &(domain, slab) in &admitted {
            let domain = &self.domains[domain.0];
            domain.arena.borrow_mut().admit_slab(slab, domain.capacity);
        }
        Ok(admitted)
    }

    /// Return slabs [`StateStore::admit`] admitted for a step that never ran
    /// to reserve, those still without referenced rows.
    fn withdraw(&self, admitted: &[(HistoryDomainId, usize)]) {
        for &(domain, slab) in admitted {
            self.domains[domain.0]
                .arena
                .borrow_mut()
                .withdraw_slab(slab);
        }
    }

    /// Release every slab beyond the room `next` claims, moving no row or
    /// bank: each domain keeps its slabs with a referenced row and, while
    /// their free pages fall short of its rows in `next`, its lowest empty
    /// slabs ([`HistoryDomain::retained_slabs`]); the banks keep their slabs
    /// with a claimed bank (the zero seed's among them) and, while their
    /// free banks fall short of `next.banks`, the lowest empty ones. Returns
    /// the bytes of the granules the released slabs held, freed in the pool
    /// for outside demand: the pool returns them ([`Pool::return_freed`]).
    /// A released slab a caller still views keeps its granule until the view
    /// drops.
    fn shrink(&self, next: &NextSteps) -> Result<u64, Error> {
        self.shrink_by_pool(next).map(ShrinkRelease::total)
    }

    fn shrink_by_pool(&self, next: &NextSteps) -> Result<ShrinkRelease, Error> {
        let mut history_released = 0u64;
        let mut recurrent_released = 0u64;
        let mut backing = self.backing.borrow_mut();
        for index in 0..self.domains.len() {
            let domain = &self.domains[index];
            let id = HistoryDomainId(index);
            let rows = next
                .rows
                .iter()
                .filter(|demand| demand.domain == id)
                .try_fold(0usize, |total, demand| total.checked_add(demand.rows))
                .ok_or(LayoutError::ArithmeticOverflow("shrink row demand"))?;
            let keep = domain.retained_slabs(rows);
            let victims = backing
                .history
                .slabs()[index]
                .slabs
                .backed()
                .into_iter()
                .filter(|slab| !keep.contains(slab))
                .collect::<Vec<_>>();
            if victims.is_empty() {
                continue;
            }
            backing
                .history
                .mutate_domain(index, |history| -> Result<(), Error> {
                    for slab in victims {
                        assert!(history.slabs.free_slab(slab)?, "a backed history slab frees");
                        domain.arena.borrow_mut().unback_empty_slab(slab);
                        history_released += 1;
                    }
                    history.rows = domain.backed_extent(&history.slabs);
                    Ok(())
                })?;
        }
        let mut recurrent_granule = 0u64;
        if let Some(recurrent) = backing.recurrent.as_mut() {
            recurrent_granule = recurrent.slabs().pool().granule();
            let reserved = self.bank_capacity.storage_total()?;
            let span = |slab: usize| {
                let start = slab * self.bank_slab_banks;
                start..(start + self.bank_slab_banks).min(reserved)
            };
            let occupied = self.banks.claimed();
            let backed = recurrent.slabs().backed().into_iter().collect::<BTreeSet<_>>();
            let mut keep = backed
                .iter()
                .copied()
                .filter(|&slab| span(slab).any(|bank| occupied.contains(&bank)))
                .collect::<BTreeSet<_>>();
            let mut room = keep
                .iter()
                .map(|&slab| span(slab).filter(|bank| !occupied.contains(bank)).count())
                .sum::<usize>();
            for &slab in &backed {
                if room >= next.banks {
                    break;
                }
                if keep.insert(slab) {
                    room += span(slab).len();
                }
            }
            let victims = backed.difference(&keep).copied().collect::<Vec<_>>();
            if !victims.is_empty() {
                recurrent.mutate_slabs(|slabs, banks| -> Result<(), Error> {
                    for slab in victims {
                        assert!(slabs.free_slab(slab)?, "a backed bank slab frees");
                        recurrent_released += 1;
                        let range = span(slab);
                        self.banks.mark_slab_unavailable(range.start, range.end);
                    }
                    let extent = slabs
                        .backed()
                        .into_iter()
                        .map(|slab| span(slab).end)
                        .max()
                        .unwrap_or(1);
                    if extent < *banks {
                        self.banks.resize(*banks, extent);
                        *banks = extent;
                    }
                    Ok(())
                })?;
            }
        }
        Ok(ShrinkRelease {
            history: history_released * self.pool.granule(),
            recurrent: recurrent_released * recurrent_granule,
        })
    }

    /// A copy of rows `from` to rows `to` in one history domain's slabs.
    fn history_row_copy(
        self: &Rc<Self>,
        domain: HistoryDomainId,
        from: Vec<usize>,
        to: Vec<usize>,
    ) -> Result<StoreCopy, Error> {
        let backing = self.backing.borrow();
        StoreCopy::new(
            self.clone(),
            &backing.history.slabs()[domain.0].slabs,
            from,
            to,
        )
    }

    /// Add whole history slabs to a domain, into its lowest unbacked slots,
    /// held in reserve until a step's demand admits them.
    fn add_history_slabs(&self, domain: HistoryDomainId, count: usize) -> Result<(), Error> {
        if count == 0 {
            return Ok(());
        }
        let mut backing = self.backing.borrow_mut();
        backing
            .history
            .mutate_domain(domain.0, |history| -> Result<(), Error> {
                let slab = &mut history.slabs;
                let mut added = Vec::new();
                for _ in 0..count {
                    match slab.add_slab(SlabFill::Zeroed) {
                        Ok(index) => added.push(index),
                        Err(error) => {
                            for index in &added {
                                slab.free_slab(*index)?;
                            }
                            self.pool
                                .return_freed(added.len() as u64 * self.pool.granule())?;
                            return Err(error.into());
                        }
                    }
                }
                let domain = &self.domains[domain.0];
                let mut arena = domain.arena.borrow_mut();
                for index in added {
                    arena.back_slab(index);
                    history.rows = history
                        .rows
                        .max(((index + 1) * domain.slab_rows).min(domain.capacity));
                }
                Ok(())
            })
    }

    /// Add whole bank slabs into the lowest unbacked writable slots.
    fn add_bank_slabs(&self, count: usize) -> Result<(), Error> {
        if count == 0 {
            return Ok(());
        }
        let start_slab = self.bank_start_slab();
        let mut backing = self.backing.borrow_mut();
        let recurrent = backing
            .recurrent
            .as_mut()
            .expect("recurrent layout has slabs");
        let rpool = self.recurrent_pool.as_ref().unwrap_or(&self.pool);
        recurrent.mutate_slabs(|slab, banks| -> Result<(), Error> {
            let mut added = Vec::new();
            for _ in 0..count {
                match slab.add_slab_from(start_slab, SlabFill::Zeroed) {
                    Ok(index) => added.push(index),
                    Err(error) => {
                        for index in &added {
                            slab.free_slab(*index)?;
                        }
                        rpool
                            .return_freed(added.len() as u64 * rpool.granule())?;
                        return Err(error.into());
                    }
                }
            }
            let reserved = self.bank_capacity.storage_total()?;
            for index in added {
                let start = index * self.bank_slab_banks;
                let end = (start + self.bank_slab_banks).min(reserved);
                if end > *banks {
                    self.banks.resize(*banks, end);
                    *banks = end;
                }
                self.banks.mark_slab_available(start, end);
            }
            Ok(())
        })
    }

    pub fn history_planes(&self) -> Result<Vec<PlaneBuffer>, Error> {
        self.backing.borrow_mut().history.views_or_build(|slabs| {
            let mut views = Vec::new();
            for (index, (domain, history)) in self.domains.iter().zip(slabs).enumerate() {
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
            Ok(views)
        })
    }
    /// A domain's rows referenced by at least one claim; a shared row counts
    /// once.
    pub fn occupied_rows(&self, domain: HistoryDomainId) -> usize {
        self.domains[domain.0].arena.borrow().referenced
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
    pub fn create(self: &Rc<Self>) -> Result<SequenceState, Error> {
        let bank = self.zero_seed.clone();
        let domains = self
            .domains
            .iter()
            .map(|domain| {
                let claims = Claims::new(&domain.arena, vec![]);
                claims.set_live(true);
                DomainHistory { start: 0, claims }
            })
            .collect();
        Ok(SequenceState {
            snapshot: HistorySnapshot {
                store: self.clone(),
                position: 0,
                history_start: 0,
                domains,
                bank,
                tape: 0,
            },
            expected_end: 0,
        })
    }
    /// Reserve the rows an advance of `count` rows from `position` adds to
    /// every stored domain (`HistoryDomainKind::advance_rows`), in logical
    /// order, each growing at that domain's `tails` tail (see [`Arena`] for
    /// placement). One domain's refusal releases the rows already reserved
    /// in the others. Reserving never adds slabs.
    fn reserve(&self, tails: &[Tail], position: usize, count: usize) -> Result<Vec<Claims>, Error> {
        let counts = self
            .domains
            .iter()
            .map(|domain| domain.kind.advance_rows(position, count))
            .collect::<Vec<_>>();
        self.reserve_rows(tails, &counts)
    }

    /// The demand of a history growing at each domain's `tails` tail by
    /// `counts[domain]` rows: in every stored domain, the page demand of
    /// those rows.
    fn demands(&self, tails: &[Tail], counts: &[usize]) -> Vec<RowDemand> {
        self.history_domains()
            .map(|domain| RowDemand {
                domain,
                rows: self.page_demand(domain, tails[domain.0].end(), counts[domain.0]),
            })
            .collect()
    }

    /// The free-page rows a history of `domain` ending at `end` needs to take
    /// `count` more rows: whatever its last page cannot hold in place,
    /// rounded up to whole pages.
    fn page_demand(&self, domain: HistoryDomainId, end: Option<usize>, count: usize) -> usize {
        let domain = &self.domains[domain.0];
        layout::page_demand_rows(domain.page_rows, domain.arena.borrow().in_place(end), count)
            .unwrap_or(usize::MAX)
    }

    /// Reserve `counts[domain]` rows in every stored domain (see
    /// [`StateStore::reserve`]). A domain taking no rows does not grow, so
    /// its tail may be blocked.
    fn reserve_rows(&self, tails: &[Tail], counts: &[usize]) -> Result<Vec<Claims>, Error> {
        if self
            .domains
            .iter()
            .zip(tails)
            .zip(counts)
            .any(|((domain, &tail), &count)| count > 0 && domain.arena.borrow().tail_blocked(tail))
        {
            return Err(Error::Request(
                "a history whose last page another history continued, or may continue first, must relocate it before it grows"
                    .into(),
            ));
        }
        let mut reserved = Vec::with_capacity(self.domains.len());
        for ((domain, &count), &tail) in self.domains.iter().zip(counts).zip(tails) {
            if count == 0 {
                reserved.push(Claims::new(&domain.arena, Vec::new()));
                continue;
            }
            let after = tail.end();
            let mut arena = domain.arena.borrow_mut();
            let claimable = arena.claimable(after);
            if claimable < count {
                let required = (count as u64)
                    .checked_mul(domain.row_bytes)
                    .ok_or(LayoutError::ArithmeticOverflow("required history bytes"))?;
                let available_bytes = (claimable as u64)
                    .checked_mul(domain.row_bytes)
                    .ok_or(LayoutError::ArithmeticOverflow("available history bytes"))?;
                return Err(Error::Capacity {
                    required,
                    available_bytes,
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

/// The rows of `ranges` (a history whose first row is at logical position
/// `start`) at positions `from` and later.
fn ranges_from(ranges: &[(usize, usize)], start: usize, from: usize) -> Vec<(usize, usize)> {
    split_ranges(ranges, from.saturating_sub(start)).1
}

/// One stored domain's logical start position paired with its row claims.
#[derive(Clone)]
struct DomainHistory {
    /// The first position this domain's history references: the trim floor,
    /// or for Window(n) at least `position - n`.
    start: usize,
    /// Claims on exactly the rows `[start, position)`, in logical order.
    claims: Claims,
}

/// An accepted sequence position, its per-domain history claims, and its
/// recurrent bank version — shared by both live [`SequenceState`] and frozen
/// [`StateCheckpoint`].
#[derive(Clone)]
struct HistorySnapshot {
    store: Rc<StateStore>,
    position: usize,
    /// The trim floor: no domain references rows before it.
    history_start: usize,
    /// Per stored domain, its logical start and row claims.
    domains: Vec<DomainHistory>,
    bank: BankHandle,
    /// Rows of `bank`'s tape that complete the accepted recurrent state: the
    /// bank holds the state `tape` rows before `position`, and the next
    /// advance replays those tape rows before its own (see
    /// [`OwnedStateAdvance::begin_rows`]). 0 after a plain advance.
    tape: usize,
}

impl HistorySnapshot {
    fn position(&self) -> usize {
        self.position
    }
    fn bank_index(&self) -> usize {
        self.bank.index()
    }
    fn tape_rows(&self) -> usize {
        self.tape
    }
    fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        Rc::ptr_eq(&self.store, store)
    }
    fn history_ranges(&self, domain: HistoryDomainId) -> Vec<(usize, usize)> {
        self.domains[domain.0].claims.ranges()
    }
    fn history_start(&self, domain: HistoryDomainId) -> usize {
        self.domains[domain.0].start
    }
}

pub struct SequenceState {
    snapshot: HistorySnapshot,
    expected_end: usize,
}
impl SequenceState {
    pub fn position(&self) -> usize {
        self.snapshot.position()
    }
    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.snapshot.belongs_to(store)
    }

    pub fn expected_end(&self) -> usize {
        self.expected_end
    }
    /// The accepted recurrent bank: the row of every recurrent arena that
    /// holds this sequence's state.
    pub fn bank_index(&self) -> usize {
        self.snapshot.bank_index()
    }
    /// The tape rows of the accepted bank that complete this sequence's
    /// recurrent state; the next advance reads version (bank, tape).
    pub fn tape_rows(&self) -> usize {
        self.snapshot.tape_rows()
    }
    pub fn anticipate(&mut self, position: usize) -> Result<(), String> {
        if position > self.snapshot.store.context_capacity {
            return Err("anticipated position exceeds context capacity".into());
        }
        self.expected_end = self.expected_end.max(position);
        Ok(())
    }
    /// This sequence's demand for appending `rows`: in every stored domain,
    /// the whole free pages beyond what its last page holds in place of the
    /// domain's rows of that advance, for provisioning a launch's backing
    /// before its advances begin.
    pub fn demands(&self, rows: usize) -> Vec<RowDemand> {
        let counts = self
            .snapshot
            .store
            .domains
            .iter()
            .map(|domain| domain.kind.advance_rows(self.snapshot.position, rows))
            .collect::<Vec<_>>();
        self.snapshot.store.demands(&self.tails(), &counts)
    }
    /// Where this history grows, per domain.
    fn tails(&self) -> Vec<Tail> {
        self.snapshot
            .domains
            .iter()
            .map(|domain| domain.claims.tail())
            .collect()
    }
    /// The demand of `queued` rows appended behind an advance of `rows`, as
    /// steps queued behind it: what appending both needs beyond
    /// [`Self::demands`] of `rows`.
    pub fn queued_demands(&self, rows: usize, queued: usize) -> Vec<RowDemand> {
        self.demands(rows.saturating_add(queued))
            .into_iter()
            .zip(self.demands(rows))
            .map(|(both, step)| RowDemand {
                domain: both.domain,
                rows: both.rows.saturating_sub(step.rows),
            })
            .collect()
    }
    /// The rows this history references in one domain, in logical order:
    /// positions `[history_start(domain), position)` (for Block, blocks
    /// `[history_start(domain), position / rate)`).
    pub fn history_ranges(&self, domain: HistoryDomainId) -> Vec<(usize, usize)> {
        self.snapshot.history_ranges(domain)
    }
    /// Every domain's ranges, in domain order.
    pub fn domain_ranges(&self) -> Vec<Vec<(usize, usize)>> {
        self.snapshot
            .domains
            .iter()
            .map(|domain| domain.claims.ranges())
            .collect()
    }
    /// The logical position of the first row this history references in a
    /// domain (for Block, the first block).
    pub fn history_start(&self, domain: HistoryDomainId) -> usize {
        self.snapshot.history_start(domain)
    }
    /// The referenced rows of a domain at positions `from` and later: the
    /// visible rows of a query whose window begins at `from`.
    pub fn visible_ranges(&self, domain: HistoryDomainId, from: usize) -> Vec<(usize, usize)> {
        ranges_from(
            &self.history_ranges(domain),
            self.snapshot.domains[domain.0].start,
            from,
        )
    }
    /// The spans of `domain`'s accepted history.
    pub fn domain_span_count(&self, domain: HistoryDomainId) -> usize {
        self.snapshot.domains[domain.0]
            .claims
            .entry(|entry| entry.ranges.len())
    }
    /// The most spans any domain's history has.
    pub fn span_count(&self) -> usize {
        self.snapshot
            .domains
            .iter()
            .map(|domain| domain.claims.entry(|entry| entry.ranges.len()))
            .max()
            .unwrap_or(0)
    }
    /// Stop seeing rows before logical position `before` in every domain.
    /// Exactly the trimmed rows lose this sequence's reference.
    pub fn trim_history(&mut self, before: usize) -> Result<(), String> {
        if before < self.snapshot.history_start || before > self.snapshot.position {
            return Err("history trim must lie within accepted logical positions".into());
        }
        self.snapshot.history_start = before;
        self.trim();
        Ok(())
    }

    /// Release each domain's references before its first retained position:
    /// the trim floor, and for Window(n) `position - n`.
    fn trim(&mut self) {
        for (domain, history) in self
            .snapshot
            .store
            .domains
            .iter()
            .zip(&mut self.snapshot.domains)
        {
            let retained =
                domain.history_start(self.snapshot.position, self.snapshot.history_start);
            if retained > history.start {
                history.claims.drop_front(retained - history.start);
                history.start = retained;
            }
        }
    }

    /// The domains whose partial last page this history must relocate
    /// before it grows, because another history (a sibling fork) continued
    /// that page or, live, ends there too and may continue it first
    /// ([`OwnedTailRelocation`]). Siblings relocate one at a time, so the
    /// last one left keeps growing in place.
    pub fn blocked_tails(&self) -> Vec<HistoryDomainId> {
        self.snapshot
            .store
            .history_domains()
            .filter(|&domain| {
                self.snapshot.store.domains[domain.0]
                    .arena
                    .borrow()
                    .tail_blocked(self.snapshot.domains[domain.0].claims.tail())
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
                rows: self.snapshot.store.history_page_rows(domain),
            })
            .collect()
    }

    /// A checkpoint at this position: it references the same rows (for
    /// Window(n), rows `[position - n, position)`), which forks and resumed
    /// requests share without copying.
    pub fn checkpoint(&self) -> StateCheckpoint {
        StateCheckpoint {
            snapshot: self.snapshot.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneCopy {
    pub plane_index: usize,
    pub from: Vec<usize>,
    pub to: Vec<usize>,
}

/// Pinned slab views and row maps for one unpublished copy of a history's
/// rows ([`OwnedTailRelocation::copy`]). The caller finishes every physical
/// copy before it commits.
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
    snapshot: HistorySnapshot,
}

impl StateCheckpoint {
    pub fn position(&self) -> usize {
        self.snapshot.position()
    }
    pub fn bank_index(&self) -> usize {
        self.snapshot.bank_index()
    }
    pub fn tape_rows(&self) -> usize {
        self.snapshot.tape_rows()
    }
    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.snapshot.belongs_to(store)
    }
    pub fn store(&self) -> &Rc<StateStore> {
        &self.snapshot.store
    }
    pub fn history_ranges(&self, domain: HistoryDomainId) -> Vec<(usize, usize)> {
        self.snapshot.history_ranges(domain)
    }
    pub fn history_start(&self, domain: HistoryDomainId) -> usize {
        self.snapshot.history_start(domain)
    }
    /// A live sequence sharing every row this checkpoint references.
    pub fn fork(&self) -> SequenceState {
        let snapshot = self.snapshot.clone();
        for domain in &snapshot.domains {
            domain.claims.set_live(true);
        }
        let expected_end = snapshot.position;
        SequenceState {
            snapshot,
            expected_end,
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
    for (history, claims) in state.snapshot.domains.iter_mut().zip(claims) {
        history.claims.append(claims);
    }
    std::mem::swap(&mut state.snapshot.bank, following);
    state.snapshot.tape = tape;
    state.snapshot.position += count;
    state.trim();
}

#[cfg(test)]
mod domain_tests;

/// The granule the tests' own pools open at, and the one their geometries
/// are figured in: 64 MiB, which every slab of their banks and history
/// pages fits, the widest a page of 256 rows of 256 KiB. No load's granule
/// follows from it: a load's is its state's smallest slab
/// (`planning::state_pool_granule`).
#[cfg(test)]
pub(crate) const TEST_GRANULE: u64 = 64 << 20;

/// A test's own device, the tally of its device store, and that store's
/// granule pool charged to the tally, as a load opens one: tests beside each
/// other in one process read and bound only their own.
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct OwnDevice {
    pub(crate) device: Rc<Device>,
    pub(crate) tally: seismic::memory::Tally,
    pub(crate) pool: Pool,
}

#[cfg(test)]
impl OwnDevice {
    /// `backend`'s device of `catalog`, its pool of `granule`-byte granules,
    /// one a block: each slab a test backs is a block of its own, charged to
    /// the tally from when it is added until it is freed and the pool returns
    /// its empty blocks.
    pub(crate) fn open(
        catalog: &seismic::Devices,
        backend: seismic::BackendName,
        granule: u64,
    ) -> Result<Self, seismic::OpenError> {
        Self::open_in_blocks(catalog, backend, granule, 1)
    }

    /// [`Self::open`], each of the pool's blocks `granules` granules: a
    /// block can keep a live slab beside a free granule, which the pool
    /// empties into another block's ([`Pool::return_freed`]).
    pub(crate) fn open_in_blocks(
        catalog: &seismic::Devices,
        backend: seismic::BackendName,
        granule: u64,
        granules: u64,
    ) -> Result<Self, seismic::OpenError> {
        let (device, tally) =
            seismic::testing::open_backend_tallied(catalog, backend, Default::default())?;
        let entry: std::sync::Arc<dyn seismic::memory::Entry> = std::sync::Arc::new(tally.clone());
        let charges = seismic::PoolCharges::one(&device.charges_in(seismic::Contract::DeviceOnly, entry));
        let pool = Pool::open(&charges, seismic::Contract::DeviceOnly, granule, granules * granule)
            .expect("a test's own device opens its pool");
        Ok(Self {
            device: Rc::new(device),
            tally,
            pool,
        })
    }

    /// The CPU's, of `granule`-byte granules.
    pub(crate) fn cpu_of(granule: u64) -> Self {
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        Self::open(&catalog, seismic::BackendName::Cpu, granule).expect("the CPU backend opens")
    }

    /// The CPU's, of [`TEST_GRANULE`]-byte granules.
    pub(crate) fn cpu() -> Self {
        Self::cpu_of(TEST_GRANULE)
    }

    /// What the device store holds with `store` the one slab store in the
    /// pool, of a granule a block ([`Self::open`]): the pool a block for each
    /// backed slab and its absent granule, each at what the backend charges
    /// a block; beside the pool, the store's address tables, at what the
    /// backend charges them. Both charges are measured, never derived: the
    /// same allocations made again through the same path, in the same order,
    /// charged to a scratch tally of the same store and dropped.
    pub(crate) fn assert_holds(&self, store: &StateStore) {
        assert_eq!(self.pool.block_bytes(), self.pool.granule(), "a fixture's pool of a granule a block");
        let identity = seismic::memory::Store::identity(&self.tally).clone();
        let scratch = || {
            let tally = seismic::memory::Tally::unbounded(identity.clone());
            let entry: std::sync::Arc<dyn seismic::memory::Entry> = std::sync::Arc::new(tally.clone());
            (tally, entry)
        };
        // A block's charge: a pool opening holds its absent granule alone,
        // one granule, a block's bytes.
        let (tally, entry) = scratch();
        let block = {
            let charges = seismic::PoolCharges::one(&self.device.charges_in(seismic::Contract::DeviceOnly, entry));
            let opened = Pool::open(&charges, seismic::Contract::DeviceOnly, self.pool.granule(), self.pool.block_bytes())
                .expect("a scratch pool opens beside the fixture's");
            let block = opened.held_bytes();
            assert_eq!(tally.held(), block);
            block
        };
        let backing = store.backing.borrow();
        // In the order the store made them: its recurrent slabs, then each
        // history domain's.
        let stores = backing
            .recurrent
            .iter()
            .map(RecurrentBacking::slabs)
            .chain(backing.history.slabs().iter().map(|history| &history.slabs))
            .collect::<Vec<_>>();
        let slabs = stores.iter().map(|slabs| slabs.backed().len() as u64).sum::<u64>();
        // The tables' charge: slab stores of the same slots and regions, in
        // the same pool, so tables of the same bytes, whatever the regions'
        // rows.
        let (tally, entry) = scratch();
        // The fixture's store charged to the scratch tally; any other store
        // the device reaches to an unbounded tally of its own.
        let charges = self.device.charges(|store| match *store == identity {
            true => entry.clone(),
            false => std::sync::Arc::new(seismic::memory::Tally::unbounded(store.clone())),
        });
        let tables = {
            let made = stores
                .iter()
                .map(|slabs| {
                    let region = seismic::SlabRegion { element: seismic::Element::dense(DType::F32), row_shape: vec![1] };
                    SlabTensor::in_pool(&charges, &self.pool, slabs.rows_per_slab(), slabs.logical_rows(), &vec![region; slabs.region_count()])
                        .expect("a scratch slab store opens in the fixture's pool")
                })
                .collect::<Vec<_>>();
            assert!(made.iter().zip(&stores).all(|(made, slabs)| made.capacity() == slabs.capacity()));
            let tables = tally.held();
            drop(made);
            tables
        };
        assert_eq!(tally.held(), 0, "the scratch tables return their charge as they drop");
        let pooled = self.pool.held_bytes();
        assert_eq!(pooled, (slabs + 1) * block, "a block for each of {slabs} backed slabs and the absent granule");
        assert_eq!(self.tally.held() - pooled, tables, "the store's tables beside the pool");
    }

    /// A store of `domains` and `specs` on this device, in its pool.
    pub(crate) fn store(
        &self,
        context_capacity: usize,
        max_advance: usize,
        domains: Vec<HistoryDomainPlan>,
        specs: Vec<ComponentSpec>,
        banks: BankCapacity,
    ) -> Result<(Rc<StateStore>, StoreBindings), Error> {
        StateStore::new(
            &seismic::Charges::tally(&self.device),
            self.pool.clone(),
            context_capacity,
            max_advance,
            domains,
            specs,
            banks,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::BackendName;
    use std::cell::Ref;

    /// The one domain of a Qwen-shaped store.
    const TOKEN: HistoryDomainId = HistoryDomainId(0);

    /// A store of one Token domain (none without components), as Qwen's
    /// target and head stores are.
    fn token_store(
        device: OwnDevice,
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
        device
            .store(context, context, domains, specs, banks)
            .map(|(_, bindings)| bindings)
    }

    impl StateStore {
        fn arena(&self) -> &Rc<RefCell<Arena>> {
            &self.domains[TOKEN.0].arena
        }

        /// Add `count` history slabs and admit them, as a step's
        /// provisioning does.
        fn add_admitted_history_slabs(&self, count: usize) -> Result<(), Error> {
            self.add_history_slabs(TOKEN, count)?;
            self.arena()
                .borrow_mut()
                .admit_all_reserved(self.domains[TOKEN.0].capacity);
            Ok(())
        }

        fn history_slabs(&self) -> Ref<'_, SlabTensor> {
            Ref::map(self.backing.borrow(), |backing| {
                &backing.history.slabs()[TOKEN.0].slabs
            })
        }
    }

    /// The pool's return of `returned` of the `charged` bytes the device
    /// held, as `Pool::return_freed` documents it: a block it emptied stays
    /// charged until the copies out of it complete, then is released from
    /// their completion off this thread. `complete` reads back what moved,
    /// the device work production completes by reading; the tally reaches
    /// `charged - returned` within the pool's own test's deadline
    /// (`a_vacate_enqueues_its_moves_beside_work_in_flight_and_releases_the_block_behind_them`).
    fn returns_once_moved(device: &OwnDevice, charged: u64, returned: u64, complete: impl FnOnce()) {
        complete();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while device.tally.held() != charged - returned {
            assert!(std::time::Instant::now() < deadline, "the emptied block is released once the copies out of it complete");
            std::thread::sleep(std::time::Duration::from_millis(1));
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


    fn cpu_device() -> OwnDevice {
        OwnDevice::cpu()
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

    /// A bank of more than a slab's target holds one bank per slab: the
    /// store commits the seed's slab and the slab of the writable bank a
    /// sequence's first step writes, so that step grows nothing.
    #[test]
    fn a_first_step_grows_no_bank_slab() {
        let (store, _bindings) = OwnDevice::cpu_of(WIDE_BANK_BYTES)
            .store(
                64,
                64,
                vec![],
                vec![ComponentSpec { shape: vec![20 * 1024 * 1024], dtype: DType::F32 }],
                BankCapacity { active: 2, in_flight: 1, retained: 0 },
            )
            .unwrap();
        assert_eq!(store.bank_slab_banks(), 1);
        assert_eq!(store.available_banks(), BankCapacity { active: 2, in_flight: 1, retained: 0 }.startup_banks().unwrap() - 1);
        assert_eq!(store.growth_claim(&[], &[], 1).unwrap().minimum_bytes, 0);
    }

    /// The bytes of a bank of 20 Mi F32 words, more than a slab's target:
    /// one bank a slab, and a granule.
    const WIDE_BANK_BYTES: u64 = 20 * 1024 * 1024 * 4;

    /// A growth claims the bytes of each slab it adds, the slab layout's
    /// own figure, never a sum of its own: a slab holds them of the one
    /// granule of the store's pool it takes, and a pool with no free
    /// granule grows by a block for it.
    #[test]
    fn a_slab_growth_claims_the_granules_of_its_slabs() {
        let device = OwnDevice::cpu_of(WIDE_BANK_BYTES);
        let (store, mut bindings) = device
            .store(
                64,
                64,
                vec![],
                vec![ComponentSpec { shape: vec![20 * 1024 * 1024], dtype: DType::F32 }],
                BankCapacity { active: 2, in_flight: 1, retained: 0 },
            )
            .unwrap();
        let slab_bytes = store
            .backing
            .borrow()
            .recurrent
            .as_ref()
            .expect("a store of components holds bank slabs")
            .slabs()
            .slab_bytes();
        assert!(slab_bytes <= device.pool.granule());
        // Two banks are wanted, one is committed past the seed: one slab,
        // which holds a whole granule.
        assert_eq!(store.growth_claim(&[], &[], 2).unwrap().minimum_bytes, device.pool.granule());
        let held = device.tally.held();
        bindings.provision(&[], 2).unwrap();
        assert_eq!(device.tally.held(), held + device.pool.block_bytes());
    }

    #[test]
    fn component_specs_reject_non_numeric_empty_and_zero_shapes() {
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
                dtype: DType::Bool,
            },
        ] {
            assert_eq!(
                spec.bytes().unwrap_err(),
                "state components require nonempty numeric tensors"
            );
        }
        let words = ComponentSpec {
            shape: vec![4, 1],
            dtype: DType::U32,
        };
        assert_eq!(words.bytes().unwrap(), 16);
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
            arena.borrow_mut().back_admitted(index, 12);
        }
        assert_eq!(arena.borrow().free, vec![(0, 4), (4, 4), (8, 4)]);
        let ranges = arena.borrow_mut().claim(None, 6);
        assert!(
            ranges
                .iter()
                .all(|(start, count)| start / 4 == (start + count - 1) / 4)
        );
        let claims = Claims::new(&arena, ranges);
        assert!(
            claims
                .ranges()
                .iter()
                .all(|(start, count)| start / 4 == (start + count - 1) / 4)
        );
        drop(claims);
        assert_eq!(arena.borrow().free, vec![(0, 4), (4, 4), (8, 4)]);
    }

    #[test]
    fn growth_across_a_slab_boundary_stays_in_place() {
        let arena = Rc::new(RefCell::new(Arena::new(0, 8, 8)));
        for index in 0..4 {
            arena.borrow_mut().back_admitted(index, 32);
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

    /// A shrink keeping no room releases no slab that holds a referenced row
    /// or a claimed bank, however sparse: no row or bank moves, and nothing
    /// is freed.
    #[test]
    fn a_shrink_releases_no_occupied_history_or_bank_slab() {
        let device = cpu_device();
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
        store.add_admitted_history_slabs(1).unwrap();
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
            device.tally.held(),
            committed_extent(&store),
            high.ranges(),
            bank.index(),
        );
        assert_eq!(store.shrink(&NextSteps::default()).unwrap(), 0);
        assert_eq!(
            (
                device.tally.held(),
                committed_extent(&store),
                high.ranges(),
                bank.index(),
            ),
            before
        );
    }

    /// A shrink keeps the room the next steps claim and no spare beyond it:
    /// the lowest empty slab while a next step's page needs it, none once
    /// no step does.
    #[test]
    fn a_shrink_keeps_only_the_next_steps_room() {
        let device = cpu_device();
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
        store.add_admitted_history_slabs(2).unwrap();
        let granule = device.pool.granule();
        let occupied = claim_rows(&store.arena(), 0, 1);
        let before = device.tally.held();
        assert_eq!(store.shrink(&NextSteps::default()).unwrap(), 2 * granule);
        assert_eq!(store.history_slabs().backed(), vec![0]);
        assert_eq!(device.pool.return_freed(2 * granule).unwrap(), 2 * device.pool.block_bytes());
        assert!(device.tally.held() < before);
        drop(occupied);
        let next = NextSteps {
            rows: vec![RowDemand { domain: TOKEN, rows: 1 }],
            banks: 0,
        };
        assert_eq!(store.shrink(&next).unwrap(), 0, "the next step's page is not surplus");
        assert_eq!(store.history_slabs().backed(), vec![0]);
        assert_eq!(store.shrink(&NextSteps::default()).unwrap(), granule);
        assert!(store.history_slabs().backed().is_empty());
    }

    #[test]
    fn cpu_sparse_final_history_and_bank_slabs_stay_and_their_block_returns() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let device = OwnDevice::open_in_blocks(&catalog, BackendName::Cpu, TEST_GRANULE, 2).unwrap();
        sparse_final_history_and_bank_slabs_stay_and_their_block_returns_on(device);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_sparse_final_history_and_bank_slabs_stay_and_their_block_returns() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let device = OwnDevice::open_in_blocks(&catalog, BackendName::Metal, TEST_GRANULE, 2).unwrap();
        sparse_final_history_and_bank_slabs_stay_and_their_block_returns_on(device);
    }

    /// A partial final history slab and a final bank slab, each holding
    /// one referenced row or claimed bank, stay where they are: the shrink
    /// releases only the empty slab between, moving nothing. The pool then
    /// returns what it freed, a whole block, emptying the emptiest block
    /// into free granules it already holds where no block is empty
    /// (`Pool::return_freed`): with no new charge, every row and bank at
    /// its address, and its bytes intact. Five slabs fill three blocks of
    /// two granules; four live slabs leave one block to return.
    fn sparse_final_history_and_bank_slabs_stay_and_their_block_returns_on(device: OwnDevice) {
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
        store.add_admitted_history_slabs(2).unwrap();
        store.add_bank_slabs(1).unwrap();
        let rows = store.history_slab_rows(TOKEN);
        assert!(2 * rows < 5000 && 3 * rows > 5000, "slab 2 is partial and final");
        assert_eq!(store.bank_slab_banks(), 4);
        let _low = claim_rows(&store.arena(), 0, 2);
        let high = claim_rows(&store.arena(), 2 * rows, 1);
        let history_value = vec![29u8; 4096 * 4];
        store
            .history_slabs()
            .region_rows(0, (2 * rows) as u64, 1)
            .unwrap()
            .write_from_host(&history_value)
            .unwrap();
        let mut claims = (0..4)
            .map(|_| Some(store.banks.acquire().unwrap()))
            .collect::<Vec<_>>();
        drop(claims[0].take());
        drop(claims[1].take());
        let bank = claims[3].as_ref().unwrap();
        assert_eq!(bank.index(), 4, "the final bank slab's");
        let bank_value = vec![31u8; 16 * 1024 * 1024];
        write_bank(&store, bank.index(), &bank_value);
        let (extent, ranges, slot) = (committed_extent(&store), high.ranges(), bank.index());
        let charged = device.tally.held();
        device.tally.bound(charged);
        let freed = store.shrink(&NextSteps::default()).unwrap();
        assert_eq!(freed, device.pool.granule(), "the empty middle history slab");
        assert_eq!(store.history_slabs().backed(), vec![0, 2]);
        let returned = device.pool.return_freed(freed).unwrap();
        device.tally.bound(u64::MAX);
        assert_eq!(returned, device.pool.block_bytes());
        assert_eq!(device.tally.held(), charged - returned);
        assert_eq!((committed_extent(&store), high.ranges(), bank.index()), (extent, ranges, slot));
        assert_eq!(
            store
                .history_slabs()
                .region_rows(0, (2 * rows) as u64, 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            history_value
        );
        assert_eq!(bank_bytes(&store, bank.index()), bank_value);
    }

    /// A sparse history slab between full ones stays, its row at its
    /// address; the empty slab the shrink releases frees a granule, and the
    /// pool returns a whole block of what it holds without a new charge,
    /// copying the sparse slab's granule where its block has no empty one.
    #[test]
    fn a_sparse_history_slab_returns_its_block_through_the_pool_without_new_charge() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let device = OwnDevice::open_in_blocks(&catalog, BackendName::Cpu, TEST_GRANULE, 2).unwrap();
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
        store.add_admitted_history_slabs(3).unwrap();
        let rows = store.history_slab_rows(TOKEN);
        assert!(4 * rows <= 8192);
        let _low = claim_rows(&store.arena(), 0, rows);
        let sparse = claim_rows(&store.arena(), 2 * rows + 1, 1);
        let value = vec![43u8; 4096 * 4];
        store
            .history_slabs()
            .region_rows(0, (2 * rows + 1) as u64, 1)
            .unwrap()
            .write_from_host(&value)
            .unwrap();
        let charged = device.tally.held();
        device.tally.bound(charged);
        // Two empty slabs free two granules: one block's worth.
        let freed = store.shrink(&NextSteps::default()).unwrap();
        assert_eq!(freed, 2 * device.pool.granule());
        assert_eq!(store.history_slabs().backed(), vec![0, 2]);
        let returned = device.pool.return_freed(freed).unwrap();
        device.tally.bound(u64::MAX);
        assert_eq!(returned, device.pool.block_bytes());
        assert_eq!(device.tally.held(), charged - returned);
        assert_eq!(sparse.ranges(), vec![(2 * rows + 1, 1)]);
        assert_eq!(
            store
                .history_slabs()
                .region_rows(0, (2 * rows + 1) as u64, 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            value
        );
    }

    /// A pool whose granule is no whole number of host pages holds each
    /// block, and its absent granule, at the backend's rounding of it: the
    /// store's charge is its blocks at what each is charged, and its tables
    /// beside them.
    #[test]
    fn a_store_in_blocks_the_backend_rounds_holds_them_at_their_charge() {
        let device = OwnDevice::cpu_of(TEST_GRANULE + 256);
        let store = token_store(
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
        assert!(device.pool.held_bytes() > 2 * device.pool.block_bytes(), "the backend rounds the blocks up");
        device.assert_holds(&store);
    }

    /// A launch's binding of a history plane reads through its slab store's
    /// address table, which stays valid as slabs are released: it holds no
    /// slab. A view of a slab's rows holds the slab's granule, and so its
    /// block, until the view drops; the pool returns the block then.
    #[test]
    fn released_history_slab_stays_charged_while_a_view_of_its_rows_holds_it() {
        let device = cpu_device();
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
        let block = device.pool.block_bytes();
        let binding = store.history_planes().unwrap()[0].buffer.clone();
        let pinned = store.history_slabs().region_rows(0, 0, 1).unwrap();
        device.assert_holds(&store);
        let charged = device.tally.held();
        // The slab is released, its granule freed once the view drops.
        assert_eq!(store.shrink(&NextSteps::default()).unwrap(), device.pool.granule());
        assert_eq!(device.pool.return_freed(device.pool.granule()).unwrap(), 0);
        assert_eq!(committed_extent(&store).0, 0);
        assert!(store.history_slabs().backed().is_empty());
        assert_eq!(device.tally.held(), charged);
        // The pool holds the viewed slab's block beside its absent granule.
        assert_eq!(device.pool.held_bytes(), 2 * block);
        drop(binding);
        assert_eq!(device.pool.return_freed(0).unwrap(), 0);
        drop(pinned);
        assert_eq!(device.pool.return_freed(0).unwrap(), block, "the granule owed behind the view returns once it drops");
        assert_eq!(device.tally.held(), charged - block);
        device.assert_holds(&store);
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
        assert!(arena.tail_taken(Some(12)));
        assert!(!arena.tail_taken(Some(15)));
        assert!(!arena.tail_taken(Some(20)));
    }

    #[test]
    fn planes_start_with_one_slab_as_one_stable_set() {
        let device = cpu_device();
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
        assert!(
            first
                .iter()
                .zip(second.iter())
                .all(|(left, right)| left.buffer.shares_allocation(&right.buffer))
        );
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
            Err(Error::Layout(LayoutError::DuplicateLayer(
                LayerRef::Target(0)
            )))
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
            Err(Error::Layout(LayoutError::DuplicateLayer(
                LayerRef::Target(0)
            )))
        ));
    }

    /// A store's history reads are those of its validated domains: a Shared
    /// layer reads its source's domain after the stored ones, and a layout
    /// the store refuses has none.
    #[test]
    fn history_reads_follow_the_validated_domains() {
        let shared = |layer| HistoryDomainPlan {
            layout: HistoryDomainLayout::Shared {
                source: LayerRef::Target(0),
                layers: vec![layer],
            },
            logical_rows: 0,
        };
        let reads = HistoryReads::of(
            &[token(vec![dense_component(4)], 2), shared(LayerRef::Target(5))],
            2,
            1,
        )
        .unwrap();
        assert_eq!(reads.len(), 2);
        assert_eq!(reads.shared_sources(), [HistoryDomainId(0)]);
        assert_eq!(
            (reads.read(LayerRef::Target(0)), reads.read(LayerRef::Target(5))),
            (Some(0), Some(1))
        );
        assert!(matches!(
            HistoryReads::of(
                &[token(vec![dense_component(4)], 2), shared(LayerRef::Target(0))],
                2,
                1
            ),
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
            validate(vec![
                token(vec![dense_component(4)], 2),
                shared(LayerRef::Target(3), 0)
            ]),
            Err(Error::Layout(LayoutError::UnknownSharedSource(
                LayerRef::Target(3)
            )))
        ));
        assert!(matches!(
            validate(vec![
                token(vec![dense_component(4)], 2),
                shared(LayerRef::Target(0), 2)
            ]),
            Err(Error::Layout(LayoutError::SharedDomainRows(
                LayerRef::Target(0)
            )))
        ));
        assert!(
            validate(vec![
                token(vec![dense_component(4)], 2),
                shared(LayerRef::Target(0), 0)
            ])
            .is_ok()
        );
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
            Err(Error::Layout(LayoutError::WrongRepresentation(_)))
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
        let device = cpu_device();
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

        let advance = OwnedStateAdvance::begin_rows(state, 1, 2, 0)
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

        let advance = OwnedStateAdvance::begin_rows(state, 1, 2, 0)
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
    /// on one path: every holder sees the same physical prefix rows, and the
    /// store holds them once, until their last holder drops.
    #[test]
    fn shared_prefix_is_the_same_rows_and_is_charged_once() {
        let device = cpu_device();
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
        // A, B and C each hold a bank of their own; the system prompt holds
        // the prompt's, and A's turn shares A's.
        assert_eq!(store.referenced_bytes(), (page as u64 + 8 + 6 + 3) * row + 4 * bank);
        device.assert_holds(&store);
        drop(a);
        assert_eq!(store.referenced_bytes(), (page as u64 + 8 + 6 + 3) * row + 4 * bank, "A's turn keeps A's rows and bank");
        drop((b, c));
        assert_eq!(store.referenced_bytes(), (page as u64 + 8) * row + 2 * bank);
        drop((system, a_turn));
        assert_eq!(store.occupied_rows(TOKEN), 0);
        assert_eq!(store.referenced_bytes(), 0);
    }

    /// A history that fills its page continues in the page at its end; a
    /// speculative advance across that boundary commits a prefix without
    /// repair, and the rejected page is free again.
    #[test]
    fn growth_continues_into_the_next_page_and_rolls_back_whole_pages() {
        let device = cpu_device();
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
        let device = cpu_device();
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
        let device = cpu_device();
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
        let parent = OwnedStateAdvance::begin_rows(parent, 1, 2, 0)
            .ok()
            .unwrap();
        let successors = [&left, &right, &parent].map(|advance| advance.bindings().following_bank);
        for (index, advance) in [&left, &right, &parent].into_iter().enumerate() {
            let bindings = advance.bindings();
            assert_eq!(bindings.previous_bank, ZERO_SEED_BANK);
            assert_ne!(bindings.following_bank, ZERO_SEED_BANK);
            assert!(!readable(&[], &[&root]).contains(&bindings.following_bank));
            assert!(
                successors
                    .iter()
                    .enumerate()
                    .all(|(other, bank)| other == index || *bank != bindings.following_bank)
            );
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
        // A live sibling ending where the fork does would contend for the
        // rows it grows into; `branch` keeps `left`'s bank readable.
        drop(left);
        let advance = OwnedStateAdvance::begin(fork, 1).ok().unwrap();
        let following = advance.bindings().following_bank;
        assert_ne!(following, ZERO_SEED_BANK);
        assert!(!readable(&[&right, &parent], &[&root, &branch]).contains(&following));
        drop(advance);
        drop((right, parent, root, branch));
        assert_eq!(store.available_banks(), committed_extent(&store).1 - 1);
    }

    #[test]
    fn fresh_sequences_share_a_pristine_seed_after_dirty_successor_reuse() {
        let device = cpu_device();
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
    fn paged_store(device: OwnDevice) -> Rc<StateStore> {
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

    /// The third of three prompts places its chunks on the same pages
    /// whether it runs one provisioned step at a time or as a chain whose
    /// first step provisions every chunk queued behind it: the slab backed
    /// ahead joins placement only when the chunk that would have grown it
    /// claims its rows.
    #[test]
    fn a_chain_claims_the_pages_its_chunks_claim_one_step_at_a_time() {
        let device = cpu_device();
        const CHUNK: usize = 256;
        let place = |chained: bool| {
            let mut store = token_store(
                device.clone(),
                4096,
                4096,
                vec![dense_component(8192)],
                vec![],
                BankCapacity {
                    active: 3,
                    in_flight: 3,
                    retained: 0,
                },
            )
            .unwrap();
            // Four pages a slab: the first two prompts leave one free page
            // in the first slab, and the third prompt's chunks need more.
            assert_eq!(
                (store.history_slab_rows(TOKEN), store.history_page_rows(TOKEN)),
                (1024, 256)
            );
            let mut held = Vec::new();
            for rows in [300, 100] {
                let state = store.create().unwrap();
                store.provision(&state.demands(rows), 0).unwrap();
                held.push(advance_all(state, rows));
            }
            let mut state = store.create().unwrap();
            if chained {
                store
                    .provision_with_growth(
                        &state.demands(CHUNK),
                        &state.queued_demands(CHUNK, 2 * CHUNK),
                        0,
                        GrowthChoice::Preferred,
                    )
                    .unwrap();
                // The chain backed every chunk's slab up front.
                assert_eq!(store.arena().borrow().backed_slabs(), BTreeSet::from([0, 1]));
                let first = OwnedStateAdvance::begin(state, CHUNK).ok().unwrap();
                // A successor dropped unattached never ran: the slab it
                // admitted returns to reserve.
                let free = store.free_rows(TOKEN);
                drop(first.successor(2 * CHUNK).unwrap());
                assert_eq!(store.free_rows(TOKEN), free);
                let second = first.successor(CHUNK).unwrap();
                let third = second.successor(CHUNK).unwrap();
                let OwnedAdvanceResolution::Committed(committed) =
                    first.commit_all().ok().unwrap()
                else {
                    panic!("an attention advance commits");
                };
                state = committed;
                for successor in [second, third] {
                    let OwnedAdvanceResolution::Committed(committed) =
                        successor.attach(state).ok().unwrap().commit_all().ok().unwrap()
                    else {
                        panic!("an attention advance commits");
                    };
                    state = committed;
                }
            } else {
                for _ in 0..3 {
                    store.provision(&state.demands(CHUNK), 0).unwrap();
                    state = advance_all(state, CHUNK);
                }
            }
            held.push(state);
            held.iter()
                .map(|state| state.history_ranges(TOKEN))
                .collect::<Vec<_>>()
        };
        assert_eq!(place(true), place(false));
    }

    /// Two forks of one checkpoint on a partial page: the first continues
    /// that page in place; the second must relocate the page before it
    /// grows. The relocation copies exactly the page's rows, publishes only
    /// after its copy, and an aborted relocation leaves the state unchanged.
    #[test]
    fn relocation_is_bit_exact_and_publishes_only_after_success() {
        let device = cpu_device();
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

    /// Requests that arrive together lie apart, each by the store's rule: a
    /// fresh history takes the middle of the largest free run, and backs a
    /// slab only when the free pages are short.
    #[test]
    fn fresh_histories_lie_where_the_store_places_them() {
        let geometry = HistoryGeometry {
            page_rows: 256,
            slab_rows: 29 * 256,
            span_limit: 18,
        };
        let ranges = fresh_history_ranges(geometry, 8, 2_621);
        assert_eq!(ranges[0], [(0, 2_621)]);
        // The second begins at the middle of the 18 free pages after the
        // first, runs to the slab's end, and takes its last rows from the
        // middle of what is left.
        assert_eq!(ranges[1], [(4_864, 2_560), (3_584, 61)]);
        let mut rows = ranges
            .iter()
            .flat_map(|history| history.iter().copied())
            .collect::<Vec<_>>();
        assert!(ranges.iter().all(|history| {
            history.iter().map(|&(_, count)| count).sum::<usize>() == 2_621
                && history.len() <= geometry.span_limit
        }));
        rows.sort_unstable();
        assert!(rows.windows(2).all(|pair| pair[0].0 + pair[0].1 <= pair[1].0));
        assert_eq!(fresh_history_ranges(geometry, 2, 0), [vec![], vec![]]);
    }

    /// Without a free page a relocation is a capacity error before any row
    /// moves, and the sequence is returned unchanged.
    #[test]
    fn relocation_without_a_free_page_is_a_capacity_error() {
        let device = cpu_device();
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
        let device = cpu_device();
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
                (0..capacity.div_ceil(inner.slab_rows)).find(|&slab| !inner.is_backed(slab))
            else {
                return;
            };
            inner.back_admitted(slab, capacity);
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
        let geometry = history_geometry(28_672, 65_792, 0, TEST_GRANULE).unwrap();
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
    /// window's front rows, finish, and release their empty slabs. After
    /// every operation every history presents at most `ceil(rows / page) + 1` spans, within its
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
                // The granule a store of this domain alone takes: its
                // smallest slab, one page.
                let pages = history_pages(row_limit, appended);
                let granule = pages.page_bytes(row_bytes).unwrap();
                let geometry = pages.in_granule(row_bytes, granule).unwrap();
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
                        // Shrink: release every empty backed slab, as a
                        // shrink keeping no room does; no row moves.
                        6 => {
                            let mut inner = arena.borrow_mut();
                            let empty = inner
                                .slabs
                                .keys()
                                .copied()
                                .filter(|&index| inner.empty_slab(index))
                                .collect::<Vec<_>>();
                            for index in empty {
                                inner.unback_empty_slab(index);
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
                            if arena.borrow().tail_taken(history.end()) {
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
                                let demand =
                                    rows.saturating_sub(inner.in_place(end)).div_ceil(page) * page;
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
        let device = cpu_device();
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
            bindings.shrink(&NextSteps::default()).unwrap();
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
        bindings.shrink(&NextSteps::default()).unwrap();
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
    /// contents across growth, and returns its blocks to the device when a
    /// slab is unreferenced. A launch's captured binding reads through the
    /// slab store's address table, which stays valid as slabs are added and
    /// released: it pins no placement.
    #[test]
    fn backing_grows_and_shrinks_and_returns_device_memory() {
        let device = cpu_device();
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
        let charged = || device.tally.held();
        let base = charged();
        assert_eq!(
            committed_extent(&store),
            (
                store.history_slab_rows(TOKEN),
                store.bank_slab_banks().min(49)
            )
        );
        // A 1,000-row prefill commits exactly one slab.
        let advance = OwnedStateAdvance::begin(store.create().unwrap(), 1000)
            .ok()
            .unwrap();
        let rows = committed_extent(&store).0;
        assert_eq!(rows, store.history_slab_rows(TOKEN));
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
        let claim = store.growth_claim(&demand, &[], 8).unwrap();
        assert!(claim.preferred_bytes >= claim.minimum_bytes);
        assert!(claim.minimum_bytes > 0);
        device.tally.bound(charged() + claim.minimum_bytes - 1);
        assert!(matches!(
            store.provision_with_growth(&demand, &[], 8, GrowthChoice::Minimum),
            Err(Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity { .. }
            )))
        ));
        assert_eq!(committed_extent(&store).0, rows);
        assert_eq!(state.history_ranges(TOKEN), [(0, 1000)]);
        device.tally.bound(u64::MAX);
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
        drop(plane);
        let before = charged();
        let released = store.shrink(&NextSteps::default()).unwrap();
        assert!(released > 0);
        // Each released slab's granule was its block's one: every block
        // returns whole.
        assert_eq!(device.pool.return_freed(released).unwrap(), released);
        assert_eq!(released, before - charged());
        assert!(committed_extent(&store).0 < rows);
        assert!(committed_extent(&store).0 >= 1000);
        assert_eq!(device.pool.return_freed(0).unwrap(), 0);
        drop(state);
        // With no sequence left, no slab is kept.
        let released = store.shrink(&NextSteps::default()).unwrap();
        assert_eq!(device.pool.return_freed(released).unwrap(), released);
        assert_eq!(committed_extent(&store).0, 0);
        assert_eq!(store.shrink(&NextSteps::default()).unwrap(), 0, "an idle store holds nothing more to release");
        assert!(charged() < base);
    }

    /// Refused slab growth keeps the published backing and existing rows.
    #[test]
    fn failed_slab_growth_keeps_published_history() {
        let device = cpu_device();
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
        let baseline = device.tally.held();
        device.tally.bound(baseline);
        assert!(matches!(
            store.add_history_slabs(TOKEN, 1),
            Err(Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity { .. }
            )))
        ));
        device.tally.bound(u64::MAX);
        assert_eq!(committed_extent(&store).0, store.history_slab_rows(TOKEN));
        assert_eq!(device.tally.held(), baseline);
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
        let device = cpu_device();
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
        let baseline = device.tally.held();
        let committed = committed_extent(&store);
        let history_bytes = store.history_slabs().slab_bytes();
        let bank_bytes = store
            .backing
            .borrow()
            .recurrent
            .as_ref()
            .unwrap()
            .slabs()
            .slab_bytes();
        assert_eq!(
            store.growth_claim(&demand, &[], 4).unwrap().minimum_bytes,
            history_bytes + bank_bytes
        );
        // Each slab grows the pool by a block: the history slab's is
        // admitted, the bank slab's refused.
        let block = device.pool.block_bytes();
        device.tally.bound(baseline + 2 * block - 1);
        assert!(matches!(
            store.provision_with_growth(&demand, &[], 4, GrowthChoice::Minimum),
            Err(Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity { .. }
            )))
        ));
        device.tally.bound(u64::MAX);
        assert_eq!(committed_extent(&store), committed);
        assert_eq!(device.tally.held(), baseline);
        assert_eq!(store.arena().borrow().backed_slabs(), BTreeSet::from([0]));

        store
            .provision_with_growth(&demand, &[], 4, GrowthChoice::Minimum)
            .unwrap();
        assert_eq!(store.arena().borrow().backed_slabs(), BTreeSet::from([0, 1]));
        assert_eq!(
            committed_extent(&store).1,
            committed.1 + store.bank_slab_banks()
        );
        assert_eq!(device.tally.held(), baseline + 2 * block);
    }

    /// Regression (chost 15:36): shrinking after every step released the
    /// successor banks the next step regrew, reallocating and copying the
    /// bank arenas every other decode step. A shrink keeps the room the
    /// next step claims, its row and successor bank, so a request that
    /// decodes 300 steps beside a retained prompt checkpoint changes the
    /// backing only while it grows, even when the store shrinks between
    /// steps.
    #[test]
    fn decode_never_alternates_growing_and_shrinking_the_backing() {
        let device = cpu_device();
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
            let next = NextSteps {
                rows: state.demands(1),
                banks: 1,
            };
            bindings.shrink(&next).unwrap();
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

    /// A shrink releases surplus but not a live sequence's next step: one
    /// whose last page is full still advances a row without growing.
    #[test]
    fn a_shrink_keeps_room_for_a_live_sequences_next_step() {
        let device = cpu_device();
        let mut bindings = token_store(
            device,
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
        let mut commit = |state: SequenceState, rows: usize| {
            bindings.provision(&state.demands(rows), 0).unwrap();
            let OwnedAdvanceResolution::Committed(state) =
                OwnedStateAdvance::begin(state, rows).ok().unwrap().commit_all().ok().unwrap()
            else {
                panic!("attention-only advances commit");
            };
            state
        };
        let large = commit(store.create().unwrap(), 3000);
        let live = commit(store.create().unwrap(), 256);
        drop(large);
        let next = NextSteps {
            rows: live.demands(2),
            banks: 0,
        };
        assert!(bindings.shrink(&next).unwrap() > 0, "the freed rows are surplus");
        let claim = bindings.growth_claim(&live.demands(2), &[], 0).unwrap();
        assert_eq!(claim.minimum_bytes, 0, "the next steps' rows open pages the shrink kept");
    }

    /// One step's chunks chained behind two slots are admitted together, as
    /// its successors are: the slab their rows need stays admitted while
    /// either chunk holds the step, and returns to reserve once neither ran.
    #[test]
    fn a_steps_chunks_across_slots_are_admitted_together() {
        let device = cpu_device();
        let mut bindings = token_store(
            device,
            4096,
            4096,
            vec![dense_component(8192)],
            vec![],
            BankCapacity {
                active: 2,
                in_flight: 4,
                retained: 0,
            },
        )
        .unwrap();
        let store = Rc::clone(&bindings);
        assert_eq!(
            (store.history_slab_rows(TOKEN), store.history_page_rows(TOKEN)),
            (1024, 256)
        );
        // Two prompts' first chunks take a page each of the first slab; the
        // chunks chained behind them need four pages more, two of them a
        // reserved slab's.
        let states = [store.create().unwrap(), store.create().unwrap()];
        let demands = states.iter().flat_map(|state| state.demands(256)).collect::<Vec<_>>();
        let queued = states
            .iter()
            .flat_map(|state| state.queued_demands(256, 512))
            .collect::<Vec<_>>();
        bindings
            .provision_with_growth(&demands, &queued, 0, GrowthChoice::Minimum)
            .unwrap();
        assert_eq!(
            store.arena().borrow().reserved_slabs().collect::<BTreeSet<_>>(),
            BTreeSet::from([1])
        );
        let [first, second] = states.map(|state| {
            TentativeAdvance::Accepted(OwnedStateAdvance::begin(state, 256).ok().unwrap())
        });
        let mut chunks = TentativeAdvance::chained([(&first, 512), (&second, 512)]).unwrap();
        assert!(store.arena().borrow().reserved_slabs().next().is_none());
        drop(chunks.pop());
        assert!(store.arena().borrow().reserved_slabs().next().is_none());
        drop(chunks);
        assert_eq!(
            store.arena().borrow().reserved_slabs().collect::<BTreeSet<_>>(),
            BTreeSet::from([1])
        );
    }

    /// A slab left in reserve when its queue was orphaned returns to
    /// placement at a shrink, its pages the room the shrink keeps: no
    /// further slab stays charged beside it.
    #[test]
    fn a_shrink_counts_a_reserved_slab_as_the_room_it_returns() {
        let device = cpu_device();
        let mut bindings = token_store(
            device,
            4096,
            4096,
            vec![dense_component(8192)],
            vec![],
            BankCapacity {
                active: 2,
                in_flight: 2,
                retained: 0,
            },
        )
        .unwrap();
        let store = Rc::clone(&bindings);
        assert_eq!(
            (store.history_slab_rows(TOKEN), store.history_page_rows(TOKEN)),
            (1024, 256)
        );
        // A live history fills the first slab; three empty slabs follow, the
        // first of them in reserve, as an orphaned queue left its backing.
        let state = store.create().unwrap();
        bindings.provision(&state.demands(1024), 0).unwrap();
        let live = advance_all(state, 1024);
        store.add_admitted_history_slabs(3).unwrap();
        store.arena().borrow_mut().withdraw_slab(1);
        let next = NextSteps {
            rows: live.demands(1),
            banks: 0,
        };
        bindings.shrink(&next).unwrap();
        let arena = store.arena().borrow();
        assert_eq!(arena.backed_slabs(), BTreeSet::from([0, 1]));
        assert!(arena.reserved_slabs().next().is_none());
    }

    /// A shrink frees an empty interior history slab without moving a shared
    /// prefix, and growth reuses the lowest vacant slab index.
    #[test]
    fn a_shrink_frees_an_empty_interior_slab_and_growth_reuses_the_lowest_index() {
        let device = cpu_device();
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
        let before = (
            committed_extent(&store).0,
            device.tally.held(),
        );
        let occupied = store.occupied_rows(TOKEN);
        assert_eq!(occupied, 256 + 50 + 20);
        // The shrink and the pool's return allocate nothing: they succeed
        // when the device refuses every new allocation.
        device.tally.bound(before.1);
        let released = bindings.shrink(&NextSteps::default()).unwrap();
        assert!(released > 0);
        assert_eq!(device.pool.return_freed(released).unwrap(), released);
        device.tally.bound(u64::MAX);
        assert!(committed_extent(&store).0 <= before.0);
        assert_eq!(device.tally.held(), before.1 - released);
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
        let claim = store.growth_claim(&demand, &[], 0).unwrap();
        let slab_bytes = store.history_slabs().slab_bytes();
        assert_eq!(claim.minimum_bytes, slab_bytes);
        assert_eq!(claim.preferred_bytes, slab_bytes);
        bindings.provision(&demand, 0).unwrap();
        assert!(store.history_slabs().backed().contains(&0));
    }

    /// On every backend, a shrink of a sparse history store releases its
    /// empty slab and moves no row; the pool returns a whole block of its
    /// two-granule blocks, emptying the emptiest into a free granule it
    /// holds where none is empty, without a new charge, the sparse row
    /// readable at its address.
    #[test]
    fn a_sparse_history_store_returns_a_block_without_moving_a_row_on_every_backend() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        for backend in [
            BackendName::Cpu,
            BackendName::Metal,
            BackendName::Cuda,
            BackendName::Vulkan,
        ] {
            let Ok(device) = OwnDevice::open_in_blocks(&catalog, backend, TEST_GRANULE, 2) else {
                continue;
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
            let slab_rows = store.history_slab_rows(TOKEN);
            store.add_admitted_history_slabs(2).unwrap();
            let _low = claim_rows(&store.arena(), 0, 1);
            let high = claim_rows(&store.arena(), 2 * slab_rows + 1, 1);
            let value = (19f32).to_le_bytes().repeat(4096);
            store
                .history_slabs()
                .region_rows(0, (2 * slab_rows + 1) as u64, 1)
                .unwrap()
                .write_from_host(&value)
                .unwrap();
            let extent = committed_extent(&store);
            let charged = device.tally.held();
            device.tally.bound(charged);
            let freed = store.shrink(&NextSteps::default()).unwrap();
            assert_eq!(freed, device.pool.granule());
            let returned = device.pool.return_freed(freed).unwrap();
            device.tally.bound(u64::MAX);
            assert_eq!(returned, device.pool.block_bytes());
            assert_eq!(committed_extent(&store), extent);
            assert_eq!(high.ranges(), vec![(2 * slab_rows + 1, 1)]);
            returns_once_moved(&device, charged, returned, || {
                assert_eq!(
                    store
                        .history_slabs()
                        .region_rows(0, (2 * slab_rows + 1) as u64, 1)
                        .unwrap()
                        .read_to_host()
                        .unwrap(),
                    value
                );
            });
        }
    }

    /// On every backend, claimed banks keep their slots while a shrink
    /// releases the empty bank slab between them; the pool returns a whole
    /// block of its two-granule blocks without a new charge, and every
    /// claimed bank reads back its bytes.
    #[test]
    fn claimed_banks_keep_their_slots_while_their_empty_slab_returns_on_every_backend() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        for backend in [
            BackendName::Cpu,
            BackendName::Metal,
            BackendName::Cuda,
            BackendName::Vulkan,
        ] {
            let Ok(device) = OwnDevice::open_in_blocks(&catalog, backend, TEST_GRANULE, 2) else {
                continue;
            };
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
            assert_eq!(store.bank_slab_banks(), 4);
            store.add_bank_slabs(2).unwrap();
            let mut claims = (0..9)
                .map(|_| Some(store.banks.acquire().unwrap()))
                .collect::<Vec<_>>();
            // Bank 2 in the first slab, bank 9 in the last; the middle slab
            // empties.
            let kept = [2usize, 9];
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
            let holds = || {
                for index in kept {
                    for (plane_index, plane) in store.recurrent_arenas().iter().enumerate() {
                        let bytes = plane
                            .slice_leading(index as u64, index as u64 + 1)
                            .unwrap()
                            .read_to_host()
                            .unwrap();
                        assert_eq!(bytes, vec![index as u8 + plane_index as u8; bytes.len()]);
                    }
                }
            };
            assert_eq!(placed(), kept);
            let extent = committed_extent(&store);
            let charged = device.tally.held();
            let available = store.available_banks();
            device.tally.bound(charged);
            let freed = bindings.shrink(&NextSteps::default()).unwrap();
            assert_eq!(freed, device.pool.granule());
            let returned = device.pool.return_freed(freed).unwrap();
            device.tally.bound(u64::MAX);
            assert_eq!(returned, device.pool.block_bytes());
            assert_eq!(committed_extent(&store), extent);
            assert_eq!(store.available_banks(), available - 4, "the released slab's banks left the free slots");
            assert_eq!(placed(), kept);
            returns_once_moved(&device, charged, returned, holds);
            assert_eq!(bindings.shrink(&NextSteps::default()).unwrap(), 0);
            assert_eq!(placed(), kept);
            holds();
        }
    }

    /// A shrink frees an empty interior bank slab, moving no bank, and
    /// growth reuses its index.
    #[test]
    fn a_shrink_frees_an_empty_interior_bank_slab_and_growth_reuses_its_index() {
        let device = cpu_device();
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
            .slabs()
            .slab_bytes();
        let charged = device.tally.held();
        device.tally.bound(charged);
        let freed = store.shrink(&NextSteps::default()).unwrap();
        assert_eq!(freed, device.pool.granule());
        assert_eq!(device.pool.return_freed(freed).unwrap(), device.pool.block_bytes());
        device.tally.bound(u64::MAX);
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
        let claim = store.growth_claim(&[], &[], 4).unwrap();
        assert_eq!(claim.minimum_bytes, slab_bytes);
        store.provision(&[], 4).unwrap();
        assert!(
            store
                .backing
                .borrow()
                .recurrent
                .as_ref()
                .unwrap()
                .slabs()
                .backed()
                .contains(&1)
        );
        assert_eq!(store.available_banks(), 7);
    }

    /// A bank past an unbacked interior slab keeps its slot and bytes, and
    /// the committed banks reach it; once it is released, the next shrink
    /// releases its slab and the committed banks end at the seed's slab.
    #[test]
    fn a_bank_past_an_unbacked_interior_slab_stays_until_released() {
        let device = cpu_device();
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
        let high = claims[7].take().unwrap();
        for claim in claims {
            drop(claim);
        }
        assert_eq!(high.index(), 8);
        let value = vec![37u8; 16 * 1024 * 1024];
        store.recurrent_arenas()[0]
            .slice_leading(high.index() as u64, high.index() as u64 + 1)
            .unwrap()
            .write_from_host(&value)
            .unwrap();
        let extent = committed_extent(&store).1;
        let charged = device.tally.held();
        device.tally.bound(charged);
        let freed = store.shrink(&NextSteps::default()).unwrap();
        assert_eq!(freed, device.pool.granule(), "the interior slab");
        assert_eq!(device.pool.return_freed(freed).unwrap(), device.pool.block_bytes());
        device.tally.bound(u64::MAX);
        assert_eq!(high.index(), 8);
        assert_eq!(committed_extent(&store).1, extent);
        assert_eq!(
            store.recurrent_arenas()[0]
                .slice_leading(high.index() as u64, high.index() as u64 + 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            value
        );
        drop(high);
        assert_eq!(store.shrink(&NextSteps::default()).unwrap(), device.pool.granule());
        assert_eq!(committed_extent(&store).1, 4);
    }

    /// A sparse bank slab stays, its bank at its slot; the empty slabs the
    /// shrink releases free two granules, and the pool returns a whole block
    /// of its two-granule blocks without a new charge, copying the sparse
    /// slab's granule where its block has no empty one.
    #[test]
    fn a_sparse_bank_slab_returns_its_block_through_the_pool_without_new_charge() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let device = OwnDevice::open_in_blocks(&catalog, BackendName::Cpu, TEST_GRANULE, 2).unwrap();
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
        assert_eq!(store.bank_slab_banks(), 4);
        // Slabs of banks 0..4, 4..8, 8..12 and 12..13.
        store.add_bank_slabs(3).unwrap();
        let mut claims = (0..10)
            .map(|_| Some(store.banks.acquire().unwrap()))
            .collect::<Vec<_>>();
        for index in 5..=10 {
            drop(claims[index - 1].take());
        }
        let sparse = claims[3].as_ref().unwrap();
        assert_eq!(sparse.index(), 4);
        let value = vec![47u8; 16 * 1024 * 1024];
        write_bank(&store, sparse.index(), &value);
        let charged = device.tally.held();
        device.tally.bound(charged);
        let freed = store.shrink(&NextSteps::default()).unwrap();
        assert_eq!(freed, 2 * device.pool.granule());
        let returned = device.pool.return_freed(freed).unwrap();
        device.tally.bound(u64::MAX);
        assert_eq!(returned, device.pool.block_bytes());
        assert_eq!(device.tally.held(), charged - returned);
        assert_eq!(sparse.index(), 4);
        assert_eq!(
            store.backing.borrow().recurrent.as_ref().unwrap().slabs().backed(),
            vec![0, 1]
        );
        assert_eq!(bank_bytes(&store, sparse.index()), value);
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

    #[test]
    fn block_domain_trim_history_preserves_claim_count_across_partial_block() {
        const BLOCKS: HistoryDomainId = HistoryDomainId(0);
        let device = OwnDevice::cpu();
        let component = ComponentDescriptor::produced(
            LayerRef::IndexKeys { block: 0, part: 0 },
            DType::F32,
            4,
            false,
        )
        .unwrap();
        let (store, mut bindings) = device
            .store(
                128,
                8,
                vec![HistoryDomainPlan {
                    layout: HistoryDomainLayout::Block {
                        rate: 4,
                        components: vec![component],
                    },
                    logical_rows: 512,
                }],
                vec![],
                BankCapacity {
                    active: 1,
                    in_flight: 2,
                    retained: 0,
                },
            )
            .unwrap();
        bindings
            .provision_with_growth(
                &[RowDemand { domain: BLOCKS, rows: 16 }],
                &[RowDemand { domain: BLOCKS, rows: 256 }],
                3,
                GrowthChoice::Preferred,
            )
            .unwrap();
        let state = store.create().unwrap();
        let advance = OwnedStateAdvance::begin(state, 6).ok().unwrap();
        let OwnedAdvanceResolution::Committed(mut state) = advance.commit_all().ok().unwrap() else {
            panic!("advance to position 6 must commit");
        };
        assert_eq!(state.position(), 6);
        assert_eq!(state.history_start(BLOCKS), 0);
        assert_eq!(state.snapshot.domains[BLOCKS.0].claims.rows(), 6 / 4 - state.history_start(BLOCKS));

        // Trimming at position 5 lies inside incomplete block 1 ([4, 8)):
        // history_start must clamp to 6 / 4 = 1 rather than 5.div_ceil(4) = 2.
        state.trim_history(5).unwrap();
        assert_eq!(state.history_start(BLOCKS), 1);
        assert_eq!(state.snapshot.domains[BLOCKS.0].claims.rows(), 0);
        assert_eq!(state.snapshot.domains[BLOCKS.0].claims.rows(), 6 / 4 - state.history_start(BLOCKS));
        assert!(state.history_ranges(BLOCKS).is_empty());

        // Advancing by 2 completes block 1 (position 8), which contains
        // trimmed position 4 < 5 and must be dropped so starts advances to 2
        // and claims stays empty. Both following_history (via successor) and
        // SequenceState::trim must agree.
        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let successor = advance.successor(4).unwrap();
        assert_eq!(successor.history_start(BLOCKS), 2);
        assert!(successor.history_ranges(BLOCKS).is_empty());

        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("advance to position 8 must commit");
        };
        assert_eq!(state.position(), 8);
        assert_eq!(state.history_start(BLOCKS), 2);
        assert_eq!(state.snapshot.domains[BLOCKS.0].claims.rows(), 0);
        assert_eq!(state.snapshot.domains[BLOCKS.0].claims.rows(), 8 / 4 - state.history_start(BLOCKS));

        // Attaching the successor advances to position 12 (completing block 2,
        // tokens 8..12, all >= 5), which is retained.
        let second = successor.attach(state).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = second.commit_all().ok().unwrap() else {
            panic!("advance to position 12 must commit");
        };
        assert_eq!(state.position(), 12);
        assert_eq!(state.history_start(BLOCKS), 2);
        assert_eq!(state.snapshot.domains[BLOCKS.0].claims.rows(), 1);
        assert_eq!(state.snapshot.domains[BLOCKS.0].claims.rows(), 12 / 4 - state.history_start(BLOCKS));
    }

    #[test]
    fn checked_arithmetic_rejects_overflow_without_panicking() {
        let rate = 4usize;
        let position = usize::MAX - 2;
        let count = 10usize;
        let expected = ((position as u128 + count as u128) / rate as u128
            - position as u128 / rate as u128) as usize;
        assert_eq!(
            HistoryDomainKind::Block { rate }.advance_rows(position, count),
            expected
        );

        let pages = history_pages(usize::MAX, usize::MAX);
        assert!(pages.span_limit <= MAX_HISTORY_SPANS);
        assert_eq!(layout::page_demand_rows(256, 0, usize::MAX), None);

        let device = OwnDevice::cpu();
        let store = token_store(
            device,
            8,
            256,
            vec![ComponentDescriptor::new(
                LayerRef::Target(0),
                CodecSpec::dense(DType::F32, 1, 1),
                1,
            )
            .unwrap()],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let state = store.create().unwrap();
        let (_, error) = OwnedStateAdvance::begin_rows(state, usize::MAX, 1, 0)
            .err()
            .unwrap();
        assert!(matches!(error, Error::Layout(LayoutError::ArithmeticOverflow(_))));

        assert!(matches!(
            store.domain_demands(&[
                RowDemand {
                    domain: TOKEN,
                    rows: usize::MAX,
                },
                RowDemand {
                    domain: TOKEN,
                    rows: 1,
                },
            ]),
            Err(Error::Layout(LayoutError::ArithmeticOverflow(_)))
        ));
    }
}
